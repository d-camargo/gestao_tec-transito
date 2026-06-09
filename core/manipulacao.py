"""Leitura e processamento dos arquivos .xls (Mapa de Turma).

Versão desacoplada do Google Colab: as funções aceitam tanto caminhos de
arquivo quanto objetos file-like (como os uploads do Streamlit), e devolvem
DataFrames em memória em vez de gravar CSVs em disco.

Também extrai metadados do cabeçalho do XLS (Curso, Etapa/Bimestre, Período
Letivo, Turma) e valida que o arquivo cobre **um único bimestre**, condição
necessária para que as estatísticas do relatório façam sentido.
"""
import re
import unicodedata

import pandas as pd

from .disciplinas import (
    catalogo_nomes_conhecidos,
    detectar_serie,
    eh_disciplina_ensino_medio,
)


# Etapa esperada: exatamente "Xº Bimestre" (X em 1..4).
_RE_BIMESTRE_UNICO = re.compile(r"^\s*([1-4])\s*[ºo°]\s*Bimestre\s*$", re.IGNORECASE)


class ArquivoInvalidoError(ValueError):
    """Erro de validação amigável para problemas estruturais no XLS."""


def remover_acentos(txt):
    """Remove acentos de uma string."""
    if not isinstance(txt, str):
        txt = str(txt)
    return ''.join(
        c for c in unicodedata.normalize('NFD', txt)
        if unicodedata.category(c) != 'Mn'
    )


def _ler_xls_bruto(arquivo_xls):
    """Lê o XLS inteiro como DataFrame de strings (sem cabeçalho).

    Centraliza o tratamento de erro de leitura para mensagens amigáveis.
    """
    try:
        return pd.read_excel(arquivo_xls, header=None, dtype=str, engine='xlrd')
    except Exception as e:
        raise ArquivoInvalidoError(
            f"Não foi possível abrir o arquivo Excel (.xls). Detalhe: {e}"
        )


def _valor_apos_rotulo(df_bruto, rotulo, max_linha=10):
    """Retorna o primeiro valor não-nulo encontrado à direita do rótulo
    informado nas primeiras `max_linha` linhas. Compara sem acento e em
    minúsculas. Devolve string ou None.
    """
    rotulo_norm = remover_acentos(rotulo).lower().rstrip(':').strip()
    limite = min(max_linha, len(df_bruto))
    for i in range(limite):
        for j, cel in enumerate(df_bruto.iloc[i].tolist()):
            if cel is None:
                continue
            texto = remover_acentos(str(cel)).lower().strip().rstrip(':').strip()
            if texto == rotulo_norm:
                for k in range(j + 1, df_bruto.shape[1]):
                    val = df_bruto.iat[i, k]
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()
                return None
    return None


def extrair_metadados(arquivo_xls):
    """Extrai metadados do cabeçalho do XLS (linhas 0–4 do mapa de turma).

    Retorna dict com chaves: ``curso``, ``etapa``, ``bimestre_num`` (int 1..4),
    ``periodo_letivo``, ``turma``. Levanta ``ArquivoInvalidoError`` se a etapa
    não corresponder exatamente a um bimestre único (ex.: arquivos agregando
    vários bimestres).
    """
    if isinstance(arquivo_xls, pd.DataFrame):
        df_bruto = arquivo_xls
    else:
        df_bruto = _ler_xls_bruto(arquivo_xls)

    curso = _valor_apos_rotulo(df_bruto, 'Curso')
    etapa = _valor_apos_rotulo(df_bruto, 'Etapa')
    periodo = _valor_apos_rotulo(df_bruto, 'Período Letivo')
    turma = _valor_apos_rotulo(df_bruto, 'Turma')

    if not etapa:
        raise ArquivoInvalidoError(
            "O arquivo não informa a etapa (linha 'Etapa:' do cabeçalho). "
            "Reexporte o Mapa de Turma do SIGAA."
        )

    match = _RE_BIMESTRE_UNICO.match(etapa)
    if not match:
        raise ArquivoInvalidoError(
            "Este relatório processa apenas **um bimestre por vez**, mas o "
            f"arquivo cobre: \"{etapa}\". Exporte o Mapa de Turma do SIGAA "
            "selecionando um único bimestre (1º, 2º, 3º ou 4º) e tente "
            "novamente."
        )
    bimestre_num = int(match.group(1))

    return {
        'curso': curso,
        'etapa': etapa,
        'bimestre_num': bimestre_num,
        'periodo_letivo': periodo,
        'turma': turma,
    }


def extrair_legenda(arquivo_xls):
    """Lê a seção "LEGENDA" do XLS e devolve um dict {código: nome completo}.

    A legenda no Mapa de Turma fica abaixo da lista de alunos, marcada por uma
    linha com o texto "LEGENDA" e seguida por pares ``código | nome``. Como o
    SIGAA não exporta os mesmos códigos com nomes idênticos entre cursos, essa
    legenda é a fonte autoritativa de nomes para o relatório.

    Retorna dict vazio se não houver legenda detectável (mantemos comportamento
    tolerante para não derrubar arquivos antigos).
    """
    if isinstance(arquivo_xls, pd.DataFrame):
        df = arquivo_xls
    else:
        df = _ler_xls_bruto(arquivo_xls)

    if df.shape[1] < 3:
        return {}

    inicio = -1
    for i in range(len(df)):
        cel = df.iat[i, 1]
        if pd.notna(cel) and remover_acentos(str(cel)).strip().lower() == 'legenda':
            inicio = i + 1
            break
    if inicio == -1:
        return {}

    legenda = {}
    for i in range(inicio, len(df)):
        codigo = df.iat[i, 1]
        nome = df.iat[i, 2]
        if pd.isna(codigo) or not str(codigo).strip():
            continue
        codigo = str(codigo).strip()
        if remover_acentos(codigo).lower() == 'legenda':
            continue
        if pd.isna(nome):
            continue
        nome = str(nome).strip()
        if not nome:
            continue
        legenda[codigo] = nome
    return legenda


def extrair_dataframes(arquivo_xls):
    """Lê o arquivo XLS, identifica o cabeçalho dinamicamente e monta os
    DataFrames de Notas e Faltas.

    `arquivo_xls` pode ser um caminho, um objeto file-like (upload) ou um
    DataFrame já lido por ``_ler_xls_bruto`` (evita releitura).
    Robusto contra variações na formatação do cabeçalho e células mescladas.
    """
    if isinstance(arquivo_xls, pd.DataFrame):
        df_full = arquivo_xls
    else:
        df_full = _ler_xls_bruto(arquivo_xls)

    # Encontra a linha onde o cabeçalho começa procurando por "Matrícula"
    header_start_row = -1
    mat_col_idx = -1
    nome_col_idx = -1

    for i, row in df_full.iterrows():
        search_row = [remover_acentos(str(cell)).lower() for cell in row.tolist()]
        if 'matricula' in search_row:
            header_start_row = i
            mat_col_idx = search_row.index('matricula')
            if 'nome do aluno' in search_row:
                nome_col_idx = search_row.index('nome do aluno')
            elif 'nome' in search_row:
                nome_col_idx = search_row.index('nome')
            break

    if header_start_row == -1 or mat_col_idx == -1 or nome_col_idx == -1:
        raise ArquivoInvalidoError(
            "Não foi possível localizar as colunas 'Matrícula' e 'Nome' no arquivo."
        )

    header_disciplinas = df_full.iloc[header_start_row].ffill()
    header_tipo_dado = df_full.iloc[header_start_row + 1]
    df_data = df_full.iloc[header_start_row + 2:].copy()

    df_identificacao = df_data[[mat_col_idx, nome_col_idx]].copy()
    df_identificacao.columns = ['matricula', 'nome']

    matriculas_limpas = df_identificacao['matricula'].str.replace(r'\D', '', regex=True)
    mask_validos = matriculas_limpas.str.match(r'^20\d{9}$', na=False)

    df_alunos_validos = df_identificacao[mask_validos].reset_index(drop=True)
    df_data_validos = df_data[mask_validos].reset_index(drop=True)

    df_notas = df_alunos_validos.copy()
    df_faltas = df_alunos_validos.copy()

    rotulos_nao_disciplina = {
        'matricula', 'nome', 'nome do aluno', 'situacao', 'total faltas', 'nan',
    }

    for i in range(len(header_disciplinas)):
        tipo_raw = header_tipo_dado.iloc[i]
        if pd.isna(tipo_raw):
            continue
        tipo_dado = str(tipo_raw).strip().upper()
        disciplina = str(header_disciplinas.iloc[i]).strip()

        if i in [mat_col_idx, nome_col_idx] or not disciplina:
            continue
        if remover_acentos(disciplina).lower() in rotulos_nao_disciplina:
            continue

        if tipo_dado == 'N':
            df_notas[disciplina] = pd.to_numeric(df_data_validos.iloc[:, i], errors='coerce')
        elif tipo_dado == 'F':
            df_faltas[disciplina] = pd.to_numeric(df_data_validos.iloc[:, i], errors='coerce')

    return df_notas, df_faltas


def ajustar_dataframe(df, disciplinas_dict):
    """Garante que o DataFrame contenha todas as colunas de disciplina esperadas
    e na ordem correta, descartando colunas indesejadas."""
    colunas_finais = ['matricula', 'nome'] + list(disciplinas_dict.keys())
    for col in colunas_finais:
        if col not in df.columns:
            df[col] = pd.NA
    return df[colunas_finais]


def preencher_dados_faltantes(df_principal, df_fonte_para_preencher, cols_ensino_medio):
    """Preenche dados (notas ou faltas) de ensino médio no DF principal usando
    um DF fonte já processado.

    ``cols_ensino_medio`` é a lista de **códigos** das disciplinas de ensino
    médio a herdar (identificadas dinamicamente pelo nome na legenda, e não por
    uma lista fixa — os códigos mudam a cada série/estrutura curricular)."""
    cols_existentes = [c for c in cols_ensino_medio if c in df_fonte_para_preencher.columns]
    df_fonte_subset = df_fonte_para_preencher[['matricula'] + cols_existentes]
    return pd.merge(df_principal, df_fonte_subset, on='matricula', how='left')


def disciplinas_dict_de_df(df, legenda=None):
    """Constrói o dicionário {código: nome} a partir das colunas de um DataFrame
    já processado.

    Ordem de prioridade dos nomes:
    1. ``legenda`` extraída do próprio XLS (fonte autoritativa do SIGAA);
    2. catálogo estático em ``disciplinas.py`` (nomes amigáveis conhecidos);
    3. o próprio código (fallback).
    """
    catalogo = catalogo_nomes_conhecidos()
    legenda = legenda or {}
    cols = [c for c in df.columns if c not in ('matricula', 'nome')]
    nomes = {}
    for c in cols:
        chave = str(c)
        nomes[c] = legenda.get(chave) or catalogo.get(chave) or chave
    return nomes


def _curso_amigavel(curso_bruto):
    """A partir do nome do curso que vem no XLS (ex.: 'TÉCNICO EM EDIFICAÇÕES
    - BH-1EDIF (INTEGRADO - MTN)') extrai um rótulo curto e legível para o
    relatório (ex.: 'Edificações'). Devolve string ou None.
    """
    if not curso_bruto:
        return None
    # Pega o trecho antes do primeiro hífen, removendo o prefixo "TÉCNICO EM".
    parte = curso_bruto.split('-')[0].strip()
    parte = re.sub(r'(?i)^t[ée]cnico\s+em\s+', '', parte).strip()
    if not parte:
        return None
    return parte.title()


def processar_curso_generico(arquivo_xls):
    """Fluxo padrão (1 arquivo): extrai notas/faltas de qualquer mapa de turma.

    Serve para qualquer curso do EPTNM cujo mapa já contenha todas as
    disciplinas (técnicas + ensino médio).

    Retorna: (df_notas, df_faltas, disciplinas_dict, metadados)
    """
    df_bruto = _ler_xls_bruto(arquivo_xls)
    metadados = extrair_metadados(df_bruto)
    df_notas, df_faltas = extrair_dataframes(df_bruto)
    legenda = extrair_legenda(df_bruto)
    disciplinas_dict = disciplinas_dict_de_df(df_notas, legenda)
    metadados['curso_amigavel'] = _curso_amigavel(metadados.get('curso'))
    metadados['serie'] = detectar_serie(disciplinas_dict)
    return df_notas, df_faltas, disciplinas_dict, metadados


def processar_transito_estradas(arquivo_transito_xls, arquivo_estradas_xls):
    """Fluxo especial 1ª série: processa **Trânsito + Estradas** juntos e
    devolve dois conjuntos independentes (TT e EST), prontos para virarem
    dois relatórios separados.

    O mapa de Trânsito não traz as notas de ensino médio; elas são puxadas
    do mapa de Estradas (que já carrega EM). Os alunos do mapa de Trânsito
    são removidos do conjunto de Estradas para evitar dupla contagem.

    Retorna lista com dois itens, cada um no formato:
        (df_notas, df_faltas, disciplinas_dict, metadados)
    """
    df_bruto_tt = _ler_xls_bruto(arquivo_transito_xls)
    df_bruto_est = _ler_xls_bruto(arquivo_estradas_xls)

    meta_tt = extrair_metadados(df_bruto_tt)
    meta_est = extrair_metadados(df_bruto_est)

    if meta_tt['bimestre_num'] != meta_est['bimestre_num']:
        raise ArquivoInvalidoError(
            "Os dois arquivos cobrem bimestres diferentes "
            f"(Trânsito: {meta_tt['etapa']}, Estradas: {meta_est['etapa']}). "
            "Selecione o mesmo bimestre em ambos os mapas."
        )

    df_notas_tt, df_faltas_tt = extrair_dataframes(df_bruto_tt)
    df_notas_est, df_faltas_est = extrair_dataframes(df_bruto_est)

    legenda_tt_arq = extrair_legenda(df_bruto_tt)
    legenda_est_arq = extrair_legenda(df_bruto_est)

    # O mapa de Trânsito traz apenas as disciplinas técnicas; as do ensino médio
    # vêm do mapa de Estradas. Identificamos quais colunas de Estradas são de
    # ensino médio pelo **nome** (legenda), pois os códigos variam entre séries
    # — uma lista fixa de códigos quebrava na 1ª/3ª série (ver disciplinas.py).
    disc_est_completo = disciplinas_dict_de_df(df_notas_est, legenda_est_arq)
    cols_ensino_medio = [c for c, nome in disc_est_completo.items()
                         if eh_disciplina_ensino_medio(nome)]

    # Trânsito herda colunas de ensino médio a partir do mapa de Estradas.
    df_notas_tt = preencher_dados_faltantes(df_notas_tt, df_notas_est, cols_ensino_medio)
    df_faltas_tt = preencher_dados_faltantes(df_faltas_tt, df_faltas_est, cols_ensino_medio)

    # Estradas perde os alunos que já estão em Trânsito (não há sobreposição
    # real, mas garantimos a separação).
    matriculas_tt = set(df_notas_tt['matricula'].dropna())
    df_notas_est = df_notas_est[~df_notas_est['matricula'].isin(matriculas_tt)].reset_index(drop=True)
    df_faltas_est = df_faltas_est[~df_faltas_est['matricula'].isin(matriculas_tt)].reset_index(drop=True)

    # Mescla legendas: o curso destino tem prioridade sobre a herança.
    legenda_tt = {**legenda_est_arq, **legenda_tt_arq}
    legenda_est = legenda_est_arq

    disciplinas_tt = disciplinas_dict_de_df(df_notas_tt, legenda_tt)
    disciplinas_est = disciplinas_dict_de_df(df_notas_est, legenda_est)

    meta_tt['curso_amigavel'] = _curso_amigavel(meta_tt.get('curso')) or 'Trânsito'
    meta_est['curso_amigavel'] = _curso_amigavel(meta_est.get('curso')) or 'Estradas'
    meta_tt['serie'] = detectar_serie(disciplinas_tt)
    meta_est['serie'] = detectar_serie(disciplinas_est)

    return [
        (df_notas_tt, df_faltas_tt, disciplinas_tt, meta_tt),
        (df_notas_est, df_faltas_est, disciplinas_est, meta_est),
    ]


# --- Compatibilidade retroativa ---------------------------------------------
# Antes da abertura para qualquer curso, o ponto de entrada do Trânsito era
# `processar_transito(transito_xls, completo_xls)` e devolvia um único par
# (df_notas, df_faltas, disciplinas). Manter um alias evita quebrar callers
# externos que ainda usem essa assinatura, mas a nova UI usa
# `processar_transito_estradas`.
def processar_transito(arquivo_transito_xls, arquivo_completo_xls):
    resultado = processar_transito_estradas(arquivo_transito_xls, arquivo_completo_xls)
    df_notas, df_faltas, disciplinas_dict, _meta = resultado[0]
    return df_notas, df_faltas, disciplinas_dict

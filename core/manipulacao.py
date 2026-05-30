"""Leitura e processamento dos arquivos .xls (Mapa de Turma).

Versão desacoplada do Google Colab: as funções aceitam tanto caminhos de
arquivo quanto objetos file-like (como os uploads do Streamlit), e devolvem
DataFrames em memória em vez de gravar CSVs em disco.
"""
import unicodedata

import pandas as pd

from .disciplinas import (
    disciplinas_ensino_medio,
    disciplinas_tecnicas_transito,
    catalogo_nomes_conhecidos,
)


def remover_acentos(txt):
    """Remove acentos de uma string."""
    if not isinstance(txt, str):
        txt = str(txt)
    return ''.join(
        c for c in unicodedata.normalize('NFD', txt)
        if unicodedata.category(c) != 'Mn'
    )


def extrair_dataframes(arquivo_xls):
    """Lê o arquivo XLS, identifica o cabeçalho dinamicamente e monta os
    DataFrames de Notas e Faltas.

    `arquivo_xls` pode ser um caminho ou um objeto file-like (upload).
    Robusto contra variações na formatação do cabeçalho e células mescladas.
    """
    try:
        # Lê a planilha inteira como texto para evitar erros de tipo
        df_full = pd.read_excel(arquivo_xls, header=None, dtype=str, engine='xlrd')
    except Exception as e:
        raise ValueError(f"Não foi possível ler o arquivo Excel. Erro: {e}")

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
        raise ValueError(
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

    for i in range(len(header_disciplinas)):
        tipo_dado = str(header_tipo_dado.iloc[i]).strip().upper()
        disciplina = str(header_disciplinas.iloc[i]).strip()

        if i in [mat_col_idx, nome_col_idx] or not disciplina:
            continue
        if disciplina.isdigit():
            continue

        if 'N' in tipo_dado:
            df_notas[disciplina] = pd.to_numeric(df_data_validos.iloc[:, i], errors='coerce')
        elif 'F' in tipo_dado:
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


def preencher_dados_faltantes(df_principal, df_fonte_para_preencher):
    """Preenche dados (notas ou faltas) de ensino médio no DF principal usando
    um DF fonte já processado."""
    cols_para_adicionar = ['matricula'] + list(disciplinas_ensino_medio.keys())
    cols_existentes = [c for c in cols_para_adicionar if c in df_fonte_para_preencher.columns]
    df_fonte_subset = df_fonte_para_preencher[cols_existentes]
    return pd.merge(df_principal, df_fonte_subset, on='matricula', how='left')


def disciplinas_dict_de_df(df):
    """Constrói o dicionário {código: nome} a partir das colunas de um DataFrame
    já processado, usando nomes amigáveis conhecidos quando disponíveis e, caso
    contrário, o próprio cabeçalho da coluna. Usado para cursos genéricos."""
    catalogo = catalogo_nomes_conhecidos()
    cols = [c for c in df.columns if c not in ('matricula', 'nome')]
    return {c: catalogo.get(str(c), str(c)) for c in cols}


def processar_curso_generico(arquivo_xls):
    """Fluxo padrão (1 arquivo): extrai notas/faltas de qualquer mapa de turma.

    Serve para qualquer curso do EPTNM cujo mapa já contenha todas as
    disciplinas (técnicas + ensino médio).

    Retorna: (df_notas, df_faltas, disciplinas_dict)
    """
    df_notas, df_faltas = extrair_dataframes(arquivo_xls)
    disciplinas_dict = disciplinas_dict_de_df(df_notas)
    return df_notas, df_faltas, disciplinas_dict


def processar_transito(arquivo_transito_xls, arquivo_completo_xls):
    """Fluxo especial do Curso Técnico em Trânsito (2 arquivos).

    O mapa de Trânsito não traz as notas de ensino médio; elas são puxadas do
    mapa completo da escola. Retorna apenas a turma de Trânsito.

    Retorna: (df_notas, df_faltas, disciplinas_dict)
    """
    disciplinas_dict = {**disciplinas_tecnicas_transito, **disciplinas_ensino_medio}

    df_notas_transito, df_faltas_transito = extrair_dataframes(arquivo_transito_xls)
    df_notas_completo, df_faltas_completo = extrair_dataframes(arquivo_completo_xls)

    # Completa o ensino médio a partir do mapa completo
    df_notas_transito = preencher_dados_faltantes(df_notas_transito, df_notas_completo)
    df_faltas_transito = preencher_dados_faltantes(df_faltas_transito, df_faltas_completo)

    df_notas_transito = ajustar_dataframe(df_notas_transito, disciplinas_dict)
    df_faltas_transito = ajustar_dataframe(df_faltas_transito, disciplinas_dict)

    return df_notas_transito, df_faltas_transito, disciplinas_dict

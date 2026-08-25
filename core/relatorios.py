"""Cálculo de estatísticas, geração de gráficos, comentário por IA e relatório PDF.

Versão desacoplada do Google Colab:
- Os gráficos são devolvidos como figuras matplotlib (exibíveis no Streamlit) e
  convertidos para PNG em memória quando montados no PDF.
- O PDF é gerado em um buffer (BytesIO), pronto para download.
- A chave da OpenAI é recebida por parâmetro; se ausente, o comentário por IA
  é simplesmente pulado.

Particularidades do CEFET-MG:
- Cada bimestre tem pontuação máxima diferente (1º e 3º: 20 pts; 2º e 4º: 30
  pts), o que muda o limiar de aprovação parcial (60% dos pontos). O limiar é
  calculado dinamicamente a partir dos metadados do arquivo.

Faltas:
- Análise por **sinal estatístico** (P90 e média+2σ por disciplina; alunos com
  faltas totais acima da média da turma). Sem dependência de carga horária ou
  calendário.
"""
import io
import math
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # backend sem display, adequado para servidor
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from babel.dates import format_date
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from .disciplinas import disciplinas_stem


# --------------------------------
# Pontuação por bimestre (CEFET-MG)
# --------------------------------
MAX_PONTOS_BIMESTRE = {1: 20, 2: 30, 3: 20, 4: 30}
PERCENTUAL_APROVACAO = 0.60  # 60% dos pontos do bimestre


def _limiar_aprovacao(metadados):
    """Calcula o limiar de aprovação parcial a partir dos metadados.

    Devolve (limiar_float, max_pts_int, bim_num_int). Se metadados estiverem
    ausentes/desconhecidos, usa o padrão de 12.0 (60% de 20) para não quebrar
    fluxos legados.
    """
    bim = (metadados or {}).get('bimestre_num')
    if bim not in MAX_PONTOS_BIMESTRE:
        return 12.0, 20, None
    max_pts = MAX_PONTOS_BIMESTRE[bim]
    return max_pts * PERCENTUAL_APROVACAO, max_pts, bim


# --------------------------------
# Utilidades de nome de disciplina
# --------------------------------
def get_simplified_name(code, disciplinas_dict):
    """Devolve o nome legível da disciplina, usando o catálogo/legenda
    quando disponível. CAIXA ALTA do SIGAA é convertida para Title Case para
    ficar mais legível em tabelas e gráficos.
    """
    full_name = disciplinas_dict.get(str(code), str(code))
    return full_name.strip().title()


# --------------------------------
# Estatísticas
# --------------------------------
def calcular_estatisticas(df_notas, disciplinas_dict, df_faltas=None, metadados=None):
    """Calcula as estatísticas básicas (notas + faltas) para um curso/bimestre."""
    df_notas = df_notas.copy()
    estatisticas = {}
    limiar, max_pts, bim_num = _limiar_aprovacao(metadados)
    estatisticas['limiar_aprovacao'] = limiar
    estatisticas['max_pontos_bimestre'] = max_pts
    estatisticas['bimestre_num'] = bim_num
    estatisticas['metadados'] = metadados or {}

    # Classifica as disciplinas presentes em três grupos:
    # - sem_dados: nenhuma nota lançada OU todas as notas iguais a zero;
    # - incompletas: têm notas, mas algum sinal de suspeita de lançamento
    #   incompleto disparou (ver sinais A-D logo abaixo);
    # - com_notas: as demais (entram nas estatísticas e gráficos).
    disciplinas_presentes = [c for c in disciplinas_dict if c in df_notas.columns]
    disciplinas_com_notas = []
    disciplinas_sem_dados = []
    total_alunos_turma = len(df_notas)

    # 1ª passada: separa sem_dados de com_notas e coleta, por disciplina
    # com_notas, a nota máxima observada e a cobertura (fração de alunos com
    # nota lançada) — a mediana do sinal C (abaixo) só pode ser calculada
    # depois que todos os máximos estiverem coletados.
    info_disciplinas = {}
    for c in disciplinas_presentes:
        serie_col = pd.to_numeric(df_notas[c], errors='coerce')
        serie_validas = serie_col.dropna()
        if serie_validas.empty or serie_validas.max() == 0:
            disciplinas_sem_dados.append(c)
            continue
        disciplinas_com_notas.append(c)
        info_disciplinas[c] = {
            'max_observado': float(serie_validas.max()),
            'cobertura': (len(serie_validas) / total_alunos_turma) if total_alunos_turma else 0.0,
        }

    # 2ª passada: quatro sinais de suspeita de lançamento incompleto — B e C
    # se cobrem mutuamente (B pega o mapa inteiro parcial, que deixaria a
    # mediana de C cega; C pega a disciplina destoante num bimestre difícil,
    # que B marcaria injustamente na turma toda), e D é independente dos
    # outros três (cobertura baixa, mesmo com quem lançou tirando nota alta).
    maximos_por_disciplina = {c: info['max_observado'] for c, info in info_disciplinas.items()}
    incompletas = []
    for c in disciplinas_com_notas:
        max_observado = info_disciplinas[c]['max_observado']
        cobertura = info_disciplinas[c]['cobertura']
        motivos = []

        if max_observado <= 0.50 * max_pts:
            motivos.append('A')  # forte: provavelmente não lançada
        if max_observado < 0.90 * max_pts:
            motivos.append('B')  # absoluto: conferir
        outros_maximos = [v for cod, v in maximos_por_disciplina.items() if cod != c]
        if outros_maximos:
            mediana_outros = float(pd.Series(outros_maximos).median())
            if max_observado < 0.80 * mediana_outros:
                motivos.append('C')  # relativo: conferir
        if cobertura < 0.90:
            motivos.append('D')  # cobertura: conferir

        if motivos:
            incompletas.append({
                'codigo': c,
                'nome': get_simplified_name(c, disciplinas_dict),
                'max_observado': max_observado,
                'cobertura': cobertura,
                'motivos': motivos,
            })

    estatisticas['disciplinas_sem_dados'] = [
        (c, get_simplified_name(c, disciplinas_dict)) for c in disciplinas_sem_dados]
    estatisticas['disciplinas_incompletas'] = incompletas
    codigos_incompletas = {d['codigo'] for d in incompletas}

    df_apenas_notas = df_notas[disciplinas_com_notas]

    media_por_aluno = df_apenas_notas.mean(axis=1)
    estatisticas['total_alunos'] = len(df_notas)
    estatisticas['media_geral_turma'] = media_por_aluno.mean()
    estatisticas['desvio_padrao_medias'] = media_por_aluno.std()

    taxa_aprovacao = (df_apenas_notas >= limiar).all(axis=1).mean() * 100
    estatisticas['taxa_aprovacao_geral'] = f"{taxa_aprovacao:.2f}%"

    media_por_disciplina = df_apenas_notas.mean().sort_values()
    if not media_por_disciplina.empty:
        disciplina_menor_code = media_por_disciplina.index[0]
        estatisticas['disciplina_menor_media_code'] = disciplina_menor_code
        estatisticas['disciplina_menor_media_nome'] = get_simplified_name(
            disciplina_menor_code, disciplinas_dict)
        estatisticas['menor_media'] = media_por_disciplina.iloc[0]
        estatisticas['desvio_padrao_disciplina_critica'] = df_apenas_notas[disciplina_menor_code].std()
        estatisticas['alunos_abaixo_limiar_disciplina_critica'] = int(
            (df_apenas_notas[disciplina_menor_code] < limiar).sum())

        disciplina_maior_code = media_por_disciplina.index[-1]
        estatisticas['disciplina_maior_media_nome'] = get_simplified_name(
            disciplina_maior_code, disciplinas_dict)
        estatisticas['maior_media'] = media_por_disciplina.iloc[-1]
    else:
        for key in ['disciplina_menor_media_nome', 'disciplina_maior_media_nome', 'disciplina_menor_media_code']:
            estatisticas[key] = "N/A"
        for key in ['menor_media', 'maior_media', 'desvio_padrao_disciplina_critica', 'alunos_abaixo_limiar_disciplina_critica']:
            estatisticas[key] = 0

    df_notas['disciplinas_abaixo_limiar'] = (df_apenas_notas < limiar).sum(axis=1)

    # Lista por critério absoluto de entrada: todos os alunos cuja média
    # própria seja menor que o limiar de aprovação parcial (60% da pontuação
    # do bimestre) — corte absoluto, não mais relativo à média da turma.
    # Aluno com média EXATAMENTE igual ao limiar não entra. Ordenada por
    # média ascendente, desempate por nome.
    df_medias = pd.DataFrame({
        'nome': df_notas['nome'],
        '_media': media_por_aluno,
        'disciplinas_abaixo_limiar': df_notas['disciplinas_abaixo_limiar'],
    })
    abaixo_limiar = df_medias[df_medias['_media'] < limiar]
    abaixo_limiar = abaixo_limiar.sort_values(by=['_media', 'nome'])
    estatisticas['alunos_abaixo_limiar'] = pd.DataFrame({
        'nome': abaixo_limiar['nome'].tolist(),
        'Média': [f"{v:.2f}".replace('.', ',') for v in abaixo_limiar['_media']],
        'disciplinas_abaixo_limiar': abaixo_limiar['disciplinas_abaixo_limiar'].tolist(),
    })
    estatisticas['n_abaixo_limiar'] = int(len(abaixo_limiar))
    # Buraco conhecido do corte por limiar: alunos com média ACIMA/IGUAL ao
    # limiar podem ainda ter disciplina abaixo do limiar. Eles não entram na
    # tabela, mas são contados na legenda para o coordenador saber que existem.
    acima_limiar = df_medias[df_medias['_media'] >= limiar]
    estatisticas['n_acima_limiar_com_disciplina_abaixo'] = int(
        (acima_limiar['disciplinas_abaixo_limiar'] >= 1).sum())

    summary_list = []
    for col in disciplinas_com_notas:
        stats = df_apenas_notas[col].describe()
        nome_disc = get_simplified_name(col, disciplinas_dict)
        if col in codigos_incompletas:
            nome_disc += ' *'
        summary_list.append({
            'Disciplina': nome_disc,
            'Média': f"{stats.get('mean', 0):.2f}".replace('.', ','),
            'Mediana': f"{stats.get('50%', 0):.2f}".replace('.', ','),
            'Desv. Padrão': f"{stats.get('std', 0):.2f}".replace('.', ','),
            'Mínimo': f"{stats.get('min', 0):.2f}".replace('.', ','),
            'Máximo': f"{stats.get('max', 0):.2f}".replace('.', ','),
        })
    estatisticas['boxplot_summary_df'] = pd.DataFrame(summary_list)
    estatisticas['disciplinas_com_notas'] = disciplinas_com_notas

    # ----- Faltas (sinal estatístico) -----
    if df_faltas is not None and not df_faltas.empty:
        estatisticas.update(_calcular_estatisticas_faltas(df_faltas, disciplinas_dict))

    # ----- STEM (Matemática, Física e Química) -----
    estatisticas.update(
        _calcular_estatisticas_stem(df_notas, df_faltas, disciplinas_dict, estatisticas, limiar))

    # ----- Tabela completa de desempenho por aluno (notas + faltas) -----
    estatisticas.update(
        tabela_desempenho_por_aluno(df_notas, df_faltas, disciplinas_dict, estatisticas))

    return estatisticas


def _calcular_estatisticas_stem(df_notas, df_faltas, disciplinas_dict, estatisticas, limiar):
    """Resumo estatístico das disciplinas STEM (Matemática, Física e Química).

    Usa ``disciplinas_stem`` (classificação pelo nome da legenda), restrita às
    disciplinas que estão em ``estatisticas['disciplinas_com_notas']``. Turma
    sem nenhuma STEM devolve ``stem_disponivel = False`` e DataFrame vazio,
    sem levantar erro.
    """
    stem_dict = disciplinas_stem(disciplinas_dict)
    com_notas = estatisticas.get('disciplinas_com_notas', [])
    stem_codigos = [c for c in stem_dict if c in com_notas]

    if not stem_codigos:
        return {
            'stem_disponivel': False,
            'stem_summary_df': pd.DataFrame(),
            'stem_codigos': [],
        }

    tem_faltas = df_faltas is not None and not df_faltas.empty \
        and estatisticas.get('faltas_disponiveis', False)

    linhas = []
    for cod in stem_codigos:
        s = pd.to_numeric(df_notas[cod], errors='coerce').dropna()
        abaixo = int((s < limiar).sum())
        pct_abaixo = (abaixo / len(s) * 100) if len(s) else 0.0
        media_faltas = ''
        if tem_faltas and cod in df_faltas.columns:
            fs = pd.to_numeric(df_faltas[cod], errors='coerce')
            if fs.notna().any():
                media_faltas = f"{fs.mean():.1f}".replace('.', ',')
        linhas.append({
            'Disciplina': get_simplified_name(cod, disciplinas_dict),
            'Média': f"{s.mean():.2f}".replace('.', ','),
            'Mediana': f"{s.median():.2f}".replace('.', ','),
            'Desv. Padrão': f"{s.std():.2f}".replace('.', ','),
            'Abaixo do Limiar': abaixo,
            '% Abaixo': f"{pct_abaixo:.1f}%".replace('.', ','),
            'Média de Faltas': media_faltas,
        })

    return {
        'stem_disponivel': True,
        'stem_summary_df': pd.DataFrame(linhas),
        'stem_codigos': stem_codigos,
    }


def _calcular_estatisticas_faltas(df_faltas, disciplinas_dict):
    """Resumo estatístico das faltas: por disciplina (média, mediana, P90, σ,
    quantidade de alunos acima de P90 e acima de média+2σ) + alunos com faltas
    totais acima da média da turma.
    """
    cols = [c for c in disciplinas_dict if c in df_faltas.columns]
    cols = [c for c in cols if df_faltas[c].notna().any()]
    if not cols:
        return {
            'faltas_disponiveis': False,
            'faltas_summary_df': pd.DataFrame(),
            'alunos_acima_media_faltas': pd.DataFrame(),
        }

    df_f = df_faltas[cols].apply(pd.to_numeric, errors='coerce')
    summary = []
    for col in cols:
        s = df_f[col].dropna()
        if s.empty:
            continue
        media = s.mean()
        mediana = s.median()
        p90 = s.quantile(0.90)
        std = s.std()
        cutoff_sigma = media + 2 * std
        n_acima_p90 = int((s > p90).sum())
        n_acima_sigma = int((s > cutoff_sigma).sum())
        summary.append({
            'Disciplina': get_simplified_name(col, disciplinas_dict),
            'Média': f"{media:.1f}",
            'Mediana': f"{mediana:.1f}",
            'P90': f"{p90:.1f}",
            'Desv. Padrão': f"{std:.1f}",
            'Alunos > P90': n_acima_p90,
            'Alunos > μ+2σ': n_acima_sigma,
        })
    summary_df = pd.DataFrame(summary)

    df_faltas_aluno = df_faltas[['nome'] + cols].copy()
    df_faltas_aluno['Total Faltas'] = df_f.sum(axis=1)

    # Lista por critério estatístico de entrada: todos os alunos cujo total de
    # faltas seja maior que a média de faltas totais da turma (em faltas, "pior
    # que a média" é estar acima dela). O P90 e o μ+2σ deixam de ser porta de
    # entrada e viram destaque (coluna Sinal) dentro da lista.
    media_total = df_faltas_aluno['Total Faltas'].mean()
    mediana_total = df_faltas_aluno['Total Faltas'].median()
    p90_total = df_faltas_aluno['Total Faltas'].quantile(0.90)
    sigma2_total = media_total + 2 * df_faltas_aluno['Total Faltas'].std()

    def _sinal(total):
        if total > p90_total and total > sigma2_total:
            return '> P90 e > μ+2σ'
        if total > p90_total:
            return '> P90'
        return ''

    acima = df_faltas_aluno[df_faltas_aluno['Total Faltas'] > media_total]
    acima = acima.sort_values(by=['Total Faltas', 'nome'], ascending=[False, True])
    lista_faltas = pd.DataFrame({
        'nome': acima['nome'].tolist(),
        'Total Faltas': acima['Total Faltas'].astype(int).tolist(),
        'Sinal': [_sinal(t) for t in acima['Total Faltas']],
    })

    return {
        'faltas_disponiveis': True,
        'faltas_summary_df': summary_df,
        'alunos_acima_media_faltas': lista_faltas,
        'media_faltas_total': media_total,
        'mediana_faltas_total': mediana_total,
        'p90_faltas_total': p90_total,
        'sigma2_faltas_total': sigma2_total,
        'total_faltas_turma': int(df_faltas_aluno['Total Faltas'].sum()),
        '_faltas_cols': cols,
    }


def tabela_desempenho_por_aluno(df_notas, df_faltas, disciplinas_dict, estatisticas):
    """Monta a tabela completa de desempenho por aluno (notas + faltas), uma
    linha por aluno e uma coluna por disciplina com notas.

    Célula de cada disciplina: ``"nota / faltas"`` (ex.: ``14,5 / 3``); sem
    faltas lançadas para a disciplina, mostra só a nota; nota ausente vira
    ``"—"``. Cabeçalhos abreviados ``D1..Dn``, com uma legenda à parte
    mapeando cada abreviação ao nome completo da disciplina — garante que a
    tabela caiba independentemente do comprimento dos nomes do SIGAA.

    Devolve um dicionário com ``tabela_completa_df`` (ordenado por nome do
    aluno, colunas finais ``Média`` e ``Faltas``) e ``legenda_disciplinas``
    (lista de tuplas ``(abreviação, nome_completo)``).
    """
    disciplinas_com_notas = estatisticas.get('disciplinas_com_notas', [])

    legenda_disciplinas = [
        (f"D{i + 1}", get_simplified_name(c, disciplinas_dict))
        for i, c in enumerate(disciplinas_com_notas)
    ]
    abreviacoes = {c: abrev for c, (abrev, _) in zip(disciplinas_com_notas, legenda_disciplinas)}

    tem_faltas = (
        df_faltas is not None and not df_faltas.empty
        and estatisticas.get('faltas_disponiveis', False)
    )
    faltas_por_disciplina = {}
    faltas_totais_por_nome = {}
    if tem_faltas:
        cols_faltas = estatisticas.get('_faltas_cols', [])
        df_f = df_faltas[cols_faltas].apply(pd.to_numeric, errors='coerce')
        for col in cols_faltas:
            faltas_por_disciplina[col] = pd.Series(df_f[col].values, index=df_faltas['nome'])
        faltas_totais_por_nome = pd.Series(
            df_f.sum(axis=1).values, index=df_faltas['nome']).to_dict()

    linhas = []
    for _, row in df_notas.iterrows():
        nome = row['nome']
        linha = {'Aluno': nome}
        notas_aluno = []
        for c in disciplinas_com_notas:
            abrev = abreviacoes[c]
            nota = pd.to_numeric(pd.Series([row.get(c)]), errors='coerce').iloc[0]
            if pd.isna(nota):
                linha[abrev] = '—'
                continue
            notas_aluno.append(nota)

            falta = None
            if c in faltas_por_disciplina and nome in faltas_por_disciplina[c].index:
                valor_falta = faltas_por_disciplina[c].loc[nome]
                if isinstance(valor_falta, pd.Series):
                    valor_falta = valor_falta.iloc[0]
                if pd.notna(valor_falta):
                    falta = int(valor_falta)

            nota_fmt = f"{nota:.1f}".replace('.', ',')
            linha[abrev] = f"{nota_fmt} / {falta}" if falta is not None else nota_fmt

        linha['Média'] = (
            f"{(sum(notas_aluno) / len(notas_aluno)):.2f}".replace('.', ',')
            if notas_aluno else '—'
        )

        total_faltas = faltas_totais_por_nome.get(nome)
        linha['Faltas'] = (
            str(int(total_faltas)) if total_faltas is not None and pd.notna(total_faltas) else '—'
        )

        linhas.append(linha)

    colunas = ['Aluno'] + [abrev for abrev, _ in legenda_disciplinas] + ['Média', 'Faltas']
    tabela_completa_df = pd.DataFrame(linhas, columns=colunas)
    tabela_completa_df = tabela_completa_df.sort_values(by='Aluno').reset_index(drop=True)

    return {
        'tabela_completa_df': tabela_completa_df,
        'legenda_disciplinas': legenda_disciplinas,
    }


def _formatar_ida(valor):
    """Formata o IDA com 2 casas, vírgula decimal e sinal explícito
    (ex.: ``+0,42`` / ``-0,31``)."""
    if pd.isna(valor):
        return ''
    return f"{float(valor):+.2f}".replace('.', ',')


def _faixa_ida(valor):
    """Faixa de leitura do IDA: Confortável / Regular / Atenção / Crítico."""
    if pd.isna(valor):
        return ''
    if valor >= 0.30:
        return 'Confortável'
    if valor >= 0.0:
        return 'Regular'
    if valor >= -0.30:
        return 'Atenção'
    return 'Crítico'


def calcular_ida(df_notas, df_faltas, estatisticas, limiar, max_pts):
    """Calcula o IDA — Índice de Desempenho do Aluno (-1 a +1) por aluno.

    O zero é o limiar de aprovação parcial (60% da pontuação do bimestre):
    sinal negativo significa, literalmente, estar abaixo do necessário para
    aprovação. Fórmula:

    - ``C_nota``: linear por partes com quebra no limiar —
      ``(media - limiar) / (max_pts - limiar)`` para ``media >= limiar``
      (0..+1) e ``(media - limiar) / limiar`` para ``media < limiar``
      (-1..0). Cada ramo é normalizado pela própria amplitude, então os
      extremos batem exatamente em -1 (nota 0) e +1 (nota máxima).
    - ``C_falta``: ``clip(1 - faltas/P90_faltas_turma, -1, +1)``. Zero
      faltas → +1; faltas iguais ao P90 → 0; o dobro do P90 ou mais → -1.
    - ``IDA = 0,7·C_nota + 0,3·C_falta`` (pesos do risco de repetência);
      sem faltas disponíveis ou com P90 = 0, vale só ``C_nota``.

    Devolve um DataFrame com ``nome``, ``ida``, ``c_nota``, ``c_falta`` e
    ``faixa``, ordenado por ``ida`` ascendente (desempate por ``nome``).
    """
    disciplinas = estatisticas.get('disciplinas_com_notas', [])
    if disciplinas:
        media_aluno = df_notas[disciplinas].apply(
            pd.to_numeric, errors='coerce').mean(axis=1)
    else:
        media_aluno = pd.Series(float('nan'), index=df_notas.index)

    c_nota = pd.Series(float('nan'), index=df_notas.index, dtype='float64')
    abaixo = media_aluno < limiar
    c_nota[~abaixo] = (media_aluno[~abaixo] - limiar) / (max_pts - limiar)
    c_nota[abaixo] = (media_aluno[abaixo] - limiar) / limiar

    c_falta = pd.Series(float('nan'), index=df_notas.index, dtype='float64')
    tem_faltas = (
        df_faltas is not None
        and not df_faltas.empty
        and estatisticas.get('faltas_disponiveis', False)
    )
    if tem_faltas:
        cols_faltas = estatisticas.get('_faltas_cols', [])
        df_f = df_faltas[cols_faltas].apply(pd.to_numeric, errors='coerce')
        total_faltas = df_f.sum(axis=1)
        p90 = total_faltas.quantile(0.90)
        if p90 > 0:
            # Casa faltas com notas pelo nome do aluno (mesmo mapeamento do
            # risco de repetência).
            faltas_map = pd.Series(total_faltas.values, index=df_faltas['nome'])
            faltas_aluno = df_notas['nome'].map(faltas_map).fillna(0.0)
            c_falta = (1.0 - faltas_aluno / p90).clip(lower=-1.0, upper=1.0)

    usa_faltas = c_falta.notna().any()
    ida = 0.7 * c_nota + 0.3 * c_falta if usa_faltas else c_nota

    res = pd.DataFrame({
        'nome': df_notas['nome'],
        'ida': ida,
        'c_nota': c_nota,
        'c_falta': c_falta,
    })
    res['faixa'] = res['ida'].apply(_faixa_ida)
    return res.sort_values(by=['ida', 'nome']).reset_index(drop=True)


def calcular_estatisticas_multibimestre(lista_conjuntos):
    """Calcula estatísticas agregadas a partir de múltiplos bimestres.

    Retorna um dicionário com os DataFrames prontos para exibição ou None se
    tivermos menos de 2 conjuntos de bimestres.
    """
    if not lista_conjuntos or len(lista_conjuntos) < 2:
        return None

    # Ordena para garantir ordem crescente de bimestre
    lista_conjuntos = sorted(lista_conjuntos, key=lambda c: c[3].get('bimestre_num', 0))
    sorted_bimestres = [c[3].get('bimestre_num') for c in lista_conjuntos]

    # Estruturas para acumular dados por aluno (chave = matricula)
    student_data = {}
    all_estatisticas = []

    for df_notas, df_faltas, disciplinas_dict, metadados in lista_conjuntos:
        estatisticas = calcular_estatisticas(df_notas, disciplinas_dict, df_faltas, metadados)
        all_estatisticas.append(estatisticas)

        bim_num = estatisticas.get('bimestre_num')
        if bim_num not in MAX_PONTOS_BIMESTRE:
            continue

        disc_com_notas = estatisticas.get('disciplinas_com_notas', [])
        if disc_com_notas:
            df_notas_num = df_notas[disc_com_notas].apply(pd.to_numeric, errors='coerce')
            medias_serie = df_notas_num.mean(axis=1)
        else:
            medias_serie = pd.Series(pd.NA, index=df_notas.index)

        media_map = pd.Series(medias_serie.values, index=df_notas['matricula'])
        nome_map = pd.Series(df_notas['nome'].values, index=df_notas['matricula'])

        has_faltas = (
            df_faltas is not None
            and not df_faltas.empty
            and estatisticas.get('faltas_disponiveis', False)
        )
        if has_faltas:
            cols_faltas = estatisticas.get('_faltas_cols', [])
            df_f = df_faltas[cols_faltas].apply(pd.to_numeric, errors='coerce')
            total_faltas_serie = df_f.sum(axis=1)
            faltas_map = pd.Series(total_faltas_serie.values, index=df_faltas['matricula'])
        else:
            faltas_map = pd.Series(dtype='float64')

        all_mats = set(df_notas['matricula'].dropna())
        if df_faltas is not None and 'matricula' in df_faltas.columns:
            all_mats = all_mats.union(set(df_faltas['matricula'].dropna()))

        for mat in all_mats:
            if pd.isna(mat):
                continue

            nome = None
            if mat in nome_map.index:
                nome = nome_map.loc[mat]
                if isinstance(nome, pd.Series):
                    nome = nome.iloc[0]
            if (nome is None or pd.isna(nome)) and df_faltas is not None and 'nome' in df_faltas.columns:
                nome_faltas_map = pd.Series(df_faltas['nome'].values, index=df_faltas['matricula'])
                if mat in nome_faltas_map.index:
                    nome = nome_faltas_map.loc[mat]
                    if isinstance(nome, pd.Series):
                        nome = nome.iloc[0]
            if nome is None or pd.isna(nome):
                nome = "Desconhecido"

            if mat not in student_data:
                student_data[mat] = {
                    'nome': nome,
                    'medias': {},
                    'faltas': {}
                }
            else:
                student_data[mat]['nome'] = nome

            if mat in media_map.index:
                val_media = media_map.loc[mat]
                if isinstance(val_media, pd.Series):
                    val_media = val_media.iloc[0]
                if pd.notna(val_media):
                    student_data[mat]['medias'][bim_num] = float(val_media)

            if has_faltas and mat in faltas_map.index:
                val_faltas = faltas_map.loc[mat]
                if isinstance(val_faltas, pd.Series):
                    val_faltas = val_faltas.iloc[0]
                if pd.notna(val_faltas):
                    student_data[mat]['faltas'][bim_num] = int(val_faltas)

    # Agora que temos os dados consolidados, vamos construir as tabelas
    linhas_anual = []
    lista_queda = []

    # 2 últimos bimestres enviados
    bim_penultimo = sorted_bimestres[-2]
    bim_ultimo = sorted_bimestres[-1]

    max_prev = MAX_PONTOS_BIMESTRE[bim_penultimo]
    max_last = MAX_PONTOS_BIMESTRE[bim_ultimo]

    # Verifica se há qualquer dado de faltas disponível nos bimestres
    qualquer_falta_disponivel = any(est.get('faltas_disponiveis', False) for est in all_estatisticas)

    for mat, data in student_data.items():
        nome = data['nome']
        medias = data['medias']
        faltas = data['faltas']

        # 1. Média Anual
        obtained_sum = 0.0
        weight_sum = 0.0
        for b, val_med in medias.items():
            obtained_sum += val_med
            weight_sum += MAX_PONTOS_BIMESTRE[b]

        if weight_sum > 0:
            media_anual = (obtained_sum / weight_sum) * 100
            n_bimestres = len(medias)
            status = "Completa" if n_bimestres == 4 else f"Parcial (baseada em {n_bimestres} de 4 bimestres)"
            media_anual_str = f"{media_anual:.1f}%"
        else:
            media_anual_str = "N/A"
            status = "N/A"

        # Faltas acumuladas
        if qualquer_falta_disponivel:
            if faltas:
                faltas_acumuladas = sum(faltas.values())
                faltas_str = str(faltas_acumuladas)
            else:
                faltas_str = "0"
        else:
            faltas_str = "N/A"

        linha_anual = {
            'Aluno': nome,
            'Média Anual': media_anual_str,
            'Faltas Acumuladas': faltas_str,
            'Status': status
        }
        linhas_anual.append(linha_anual)

        # 2. Tendência
        val_prev = medias.get(bim_penultimo)
        val_last = medias.get(bim_ultimo)

        if val_prev is not None and val_last is not None:
            pct_prev = (val_prev / max_prev) * 100
            pct_last = (val_last / max_last) * 100
            diff = pct_last - pct_prev

            if diff < -0.001:
                # Vírgula como separador decimal (PT-BR), como o resto do
                # relatório — só nos números, "p.p." mantém o ponto literal.
                val_prev_txt = f"{val_prev:.1f}".replace('.', ',')
                pct_prev_txt = f"{pct_prev:.1f}".replace('.', ',')
                val_last_txt = f"{val_last:.1f}".replace('.', ',')
                pct_last_txt = f"{pct_last:.1f}".replace('.', ',')
                diff_txt = f"{diff:+.1f}".replace('.', ',')
                lista_queda.append({
                    'Aluno': nome,
                    'Bimestre Anterior': f"{val_prev_txt}/{max_prev} ({pct_prev_txt}%)",
                    'Bimestre Atual': f"{val_last_txt}/{max_last} ({pct_last_txt}%)",
                    'Diferença (p.p.)': f"{diff_txt} p.p.",
                    '_diff_raw': diff
                })

    df_media_anual = pd.DataFrame(linhas_anual)
    if not df_media_anual.empty:
        df_media_anual = df_media_anual.sort_values(by='Aluno').reset_index(drop=True)

    if lista_queda:
        # Todos os alunos com queda, sem limitar a um Top N — ordenados pela
        # maior queda (diferença mais negativa) primeiro.
        df_queda = pd.DataFrame(lista_queda)
        df_queda = df_queda.sort_values(by='_diff_raw').drop(columns=['_diff_raw']).reset_index(drop=True)
    else:
        df_queda = pd.DataFrame(columns=['Aluno', 'Bimestre Anterior', 'Bimestre Atual', 'Diferença (p.p.)'])

    # ----- IDA (Índice de Desempenho do Aluno), acumulado nos bimestres enviados -----
    ida_df = calcular_ida_multibimestre(student_data)
    ida_medio_turma = ida_df['ida'].mean()

    # ----- Probabilidade de reprovação anual (só notas) -----
    df_probabilidade = calcular_probabilidade_reprovacao(student_data)

    return {
        'df_media_anual': df_media_anual,
        'df_queda': df_queda,
        'ida_df': ida_df,
        'ida_medio_turma': ida_medio_turma,
        'df_probabilidade': df_probabilidade,
        'bimestres_estatisticas': all_estatisticas,
        'sorted_bimestres': sorted_bimestres,
        'bim_penultimo': bim_penultimo,
        'bim_ultimo': bim_ultimo
    }


def calcular_ida_multibimestre(student_data):
    """Calcula o IDA — Índice de Desempenho do Aluno (-1 a +1) no acumulado
    dos bimestres enviados, aplicando ao acumulado a mesma forma da fórmula
    do bimestre único (``calcular_ida``):

    - ``S`` = soma das médias do aluno nos bimestres em que ele tem nota.
    - ``P`` = soma dos ``MAX_PONTOS_BIMESTRE`` desses mesmos bimestres.
    - ``L = 0,60 · P`` (limiar acumulado).
    - ``C_nota = (S-L)/(P-L)`` se ``S >= L``; ``(S-L)/L`` se ``S < L`` —
      sempre em [-1, +1].
    - ``C_falta = clip(1 - faltas_acumuladas/P90_acumulado_da_turma, -1, +1)``.
    - ``IDA = 0,7·C_nota + 0,3·C_falta``; sem faltas disponíveis ou com
      P90 = 0, vale só ``C_nota``.

    Aluno sem nenhuma média lançada em nenhum bimestre enviado tem
    ``ida = NaN`` e fica fora da tabela/gráfico (quem consome filtra pelo
    NaN).

    Devolve um DataFrame com ``nome``, ``ida``, ``c_nota``, ``c_falta`` e
    ``faixa``, ordenado por ``ida`` ascendente (desempate por ``nome``).
    """
    nomes = []
    c_notas = []
    faltas_acumuladas = []
    tem_alguma_falta = any(data.get('faltas') for data in student_data.values())

    for data in student_data.values():
        nomes.append(data['nome'])
        medias = data.get('medias', {})
        if not medias:
            c_notas.append(float('nan'))
            faltas_acumuladas.append(float('nan'))
            continue

        s = sum(medias.values())
        p = sum(MAX_PONTOS_BIMESTRE[b] for b in medias if b in MAX_PONTOS_BIMESTRE)
        if p <= 0:
            c_notas.append(float('nan'))
        else:
            limiar_acumulado = 0.60 * p
            if s >= limiar_acumulado:
                c_notas.append((s - limiar_acumulado) / (p - limiar_acumulado))
            else:
                c_notas.append((s - limiar_acumulado) / limiar_acumulado)

        faltas = data.get('faltas', {})
        faltas_acumuladas.append(float(sum(faltas.values())) if faltas else 0.0)

    c_nota = pd.Series(c_notas, dtype='float64')
    faltas_serie = pd.Series(faltas_acumuladas, dtype='float64')

    c_falta = pd.Series(float('nan'), index=c_nota.index, dtype='float64')
    if tem_alguma_falta:
        p90 = faltas_serie.quantile(0.90)
        if pd.notna(p90) and p90 > 0:
            c_falta = (1.0 - faltas_serie / p90).clip(lower=-1.0, upper=1.0)

    usa_faltas = c_falta.notna().any()
    ida = 0.7 * c_nota + 0.3 * c_falta if usa_faltas else c_nota

    res = pd.DataFrame({'nome': nomes, 'ida': ida, 'c_nota': c_nota, 'c_falta': c_falta})
    res['faixa'] = res['ida'].apply(_faixa_ida)
    return res.sort_values(by=['ida', 'nome']).reset_index(drop=True)


def calcular_probabilidade_reprovacao(student_data):
    """Calcula, por aluno, a probabilidade de reprovação anual no CEFET-MG
    (aprovação = 60 de 100 pontos: 20/30/20/30). Só considera notas — a
    exigência de 75% de frequência anual precisa de carga horária, que o
    Mapa de Turma não traz.

    Parte determinística (exata):
    - ``S`` = pontos obtidos = soma das médias do aluno nos bimestres em que
      tem nota lançada.
    - ``R`` = 100 − soma dos ``MAX_PONTOS_BIMESTRE`` **dos bimestres em que
      esse aluno tem nota** (⚠️ **por aluno, não global** — correção de
      2026-08-24: com ``R`` global, um aluno sem nota num bimestre que a
      turma já enviou perdia esses pontos duas vezes, pois eles não entram
      em ``S`` [ele não tem nota] nem em ``R`` [o bimestre "já foi
      enviado"], e simplesmente somem do total anual — pessimismo espúrio
      que dispara "reprovação matemática" para quem só está com lançamento
      pendente). Com ``R`` por aluno, ``S`` e ``R`` são sempre
      complementares: ``pontos_lançados_do_aluno + R = 100``.
    - ``N`` = 60 − ``S`` = pontos que ainda faltam para a aprovação.
    - ``N <= 0`` → "Aprovado por nota" (0%). ``N > R`` → "Não alcança 60
      pontos nas notas regulares" (probabilidade ``NaN``, exibida como "—" —
      o EPTNM tem recuperação/exame final, então essa linha **não** é 100%
      de reprovação; o modelo só não enxerga essa via de aprovação). Caso
      contrário, "Depende" — segue para a parte probabilística.

    Parte probabilística (só no caso "Depende"): aproveitamento do aluno por
    bimestre ``p_b = média_b / max_b``; ``μ_i``/``s_i`` (média/desvio
    amostral) desses ``p_b``; encolhimento ``σ_i² = w·s_i² + (1−w)·σ_pool²``
    com ``w = n_i/(n_i+1)`` e ``σ_pool²`` = média dos ``s_i²`` dos alunos com
    ≥2 bimestres (sem isso, desvio entre alunos dos ``p`` médios); piso
    ``σ = 0,05``. Bimestres restantes do aluno = os que não estão em
    ``medias`` dele. Pontos restantes ``X = Σ max_b·p_b`` desses bimestres,
    com ``E[X] = μ_i·R`` e ``Var[X] = σ_i²·Σ max_b²``.
    ``P(reprovar) = Φ((N − μ_i·R) / (σ_i·√(Σ max_b²)))``, sempre limitada a
    [1%, 99%] — o modelo não tem lastro para afirmar certeza. ``Φ`` sem
    SciPy: ``0,5·(1+erf(z/√2))``.

    Devolve um DataFrame com ``Aluno``, ``S``, ``Pontos_Lancados`` (Σmax dos
    bimestres em que o próprio aluno tem nota), ``R``, ``N``, ``Situacao`` e
    ``Prob_Reprovacao`` (float 0-1; ``NaN`` na situação "Não alcança 60
    pontos nas notas regulares" — sem lastro para um número, já que a
    recuperação/exame final não entra na conta — e também se, por algum
    motivo, não sobrar nenhum bimestre para modelar). Ordenado por
    gravidade: primeiro "Não alcança 60 pontos nas notas regulares" (por
    ``N`` decrescente), depois "Depende" (por probabilidade decrescente),
    por último "Aprovado por nota".
    """
    APROVACAO_ANUAL = 60.0
    TOTAL_ANUAL = 100.0
    SIGMA_PISO = 0.05

    # ----- μ_i e s_i² por aluno, e o pool de encolhimento -----
    mu_por_aluno = {}
    s_i_quadrado_por_aluno = {}
    s_i_quadrado_validos = []
    for mat, data in student_data.items():
        medias = data.get('medias', {})
        p_bs = [medias[b] / MAX_PONTOS_BIMESTRE[b] for b in medias if b in MAX_PONTOS_BIMESTRE]
        if not p_bs:
            continue
        mu = sum(p_bs) / len(p_bs)
        mu_por_aluno[mat] = mu
        if len(p_bs) >= 2:
            variancia = sum((p - mu) ** 2 for p in p_bs) / (len(p_bs) - 1)
            s_i_quadrado_por_aluno[mat] = variancia
            s_i_quadrado_validos.append(variancia)

    if s_i_quadrado_validos:
        sigma_pool_quadrado = sum(s_i_quadrado_validos) / len(s_i_quadrado_validos)
    elif len(mu_por_aluno) >= 2:
        media_mus = sum(mu_por_aluno.values()) / len(mu_por_aluno)
        sigma_pool_quadrado = sum(
            (m - media_mus) ** 2 for m in mu_por_aluno.values()) / (len(mu_por_aluno) - 1)
    else:
        sigma_pool_quadrado = SIGMA_PISO ** 2

    linhas = []
    for mat, data in student_data.items():
        nome = data['nome']
        medias = data.get('medias', {})
        bimestres_do_aluno = [b for b in medias if b in MAX_PONTOS_BIMESTRE]
        pontos_lancados = sum(MAX_PONTOS_BIMESTRE[b] for b in bimestres_do_aluno)
        S = sum(medias.values())
        R = TOTAL_ANUAL - pontos_lancados  # por aluno: pontos_lancados + R = 100 sempre
        N = APROVACAO_ANUAL - S

        if N <= 0:
            situacao = 'Aprovado por nota'
            prob = 0.0
        elif N > R:
            # Não é 100% de reprovação: o EPTNM tem recuperação/exame final,
            # via de aprovação que este modelo (só notas regulares) não
            # enxerga. Sem lastro para um número aqui — fica NaN, exibido
            # como "—" no PDF. Manter 1.0 seria armadilha: alguém lê a
            # coluna depois e reporta "100% de reprovação".
            situacao = 'Não alcança 60 pontos nas notas regulares'
            prob = float('nan')
        else:
            situacao = 'Depende'
            bimestres_restantes = [b for b in MAX_PONTOS_BIMESTRE if b not in bimestres_do_aluno]
            soma_max_quadrado_restantes = sum(MAX_PONTOS_BIMESTRE[b] ** 2 for b in bimestres_restantes)
            if soma_max_quadrado_restantes > 0:
                n_i = len(medias)
                mu_i = mu_por_aluno.get(mat, 0.0)
                s_i_quadrado = s_i_quadrado_por_aluno.get(mat, sigma_pool_quadrado)
                w = n_i / (n_i + 1)
                sigma_i_quadrado = w * s_i_quadrado + (1 - w) * sigma_pool_quadrado
                sigma_i = max(math.sqrt(sigma_i_quadrado), SIGMA_PISO)

                z = (N - mu_i * R) / (sigma_i * math.sqrt(soma_max_quadrado_restantes))
                prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                prob = min(max(prob, 0.01), 0.99)
            else:
                prob = float('nan')

        linhas.append({
            'Aluno': nome,
            'S': S,
            'Pontos_Lancados': pontos_lancados,
            'R': R,
            'N': N,
            'Situacao': situacao,
            'Prob_Reprovacao': prob,
        })

    df = pd.DataFrame(linhas, columns=[
        'Aluno', 'S', 'Pontos_Lancados', 'R', 'N', 'Situacao', 'Prob_Reprovacao'])
    if df.empty:
        return df
    # Ordenação por gravidade, não por probabilidade crua — com a
    # probabilidade virando NaN em "Não alcança 60 pontos nas notas
    # regulares", um sort simples por Prob_Reprovacao (na_position='last')
    # jogaria justo os alunos mais críticos para o fim da tabela. A ordem
    # de grupo é fixa (mais grave primeiro); dentro de cada grupo, "Não
    # alcança..." por N decrescente e "Depende" por probabilidade
    # decrescente — "Aprovado por nota" fecha a tabela, também por N
    # decrescente (todos com N ≤ 0, ordem sem efeito prático).
    ordem_situacao = {
        'Não alcança 60 pontos nas notas regulares': 0,
        'Depende': 1,
        'Aprovado por nota': 2,
    }
    df['_ordem_grupo'] = df['Situacao'].map(ordem_situacao)
    df['_chave_secundaria'] = df.apply(
        lambda r: r['Prob_Reprovacao'] if r['Situacao'] == 'Depende' else r['N'], axis=1)
    # Dentro do grupo "Depende" muitos alunos empatam em 1%/99% (a
    # probabilidade é limitada a [1%,99%]) e a ordem entre eles saía
    # arbitrária — desempata por N decrescente (quem precisa de mais pontos
    # primeiro), para a tabela ficar legível de cima para baixo.
    df['_chave_terciaria'] = df['N']
    return df.sort_values(
        by=['_ordem_grupo', '_chave_secundaria', '_chave_terciaria'],
        ascending=[True, False, False]
    ).drop(
        columns=['_ordem_grupo', '_chave_secundaria', '_chave_terciaria']
    ).reset_index(drop=True)


# --------------------------------
# Comentário por IA (opcional)
# --------------------------------
def gerar_comentario_ia(estatisticas, nome_curso, api_key, modelo="gpt-4o-mini"):
    """Gera um comentário analítico via API da OpenAI. Se `api_key` for vazio,
    devolve uma mensagem informando que a análise foi pulada."""
    if not api_key:
        return ("A análise por IA não foi gerada (chave da OpenAI não configurada). "
                "Configure-a para habilitar este comentário.")

    summary_markdown = estatisticas['boxplot_summary_df'].to_markdown(index=False)
    limiar = estatisticas.get('limiar_aprovacao', 12.0)
    max_pts = estatisticas.get('max_pontos_bimestre', 20)
    bim = estatisticas.get('bimestre_num')
    prompt = f"""
    Você é um especialista em análise de dados educacionais. Com base nos dados a seguir, gere uma análise em português. Não faça em formato MarkDown, ou seja, não use * ou #.

    **Contexto:**
    - **Curso:** {nome_curso}
    - **Bimestre:** {bim if bim else 'n/d'} (pontuação máxima {max_pts}, aprovação parcial ≥ {limiar:.1f})
    - **Total de Alunos:** {estatisticas['total_alunos']}

    **Análise Geral da Turma:**
    - **Média Geral (0-{max_pts}):** {estatisticas['media_geral_turma']:.2f}
    - **Dispersão das Médias (Desvio Padrão):** {estatisticas['desvio_padrao_medias']:.2f}
    - **Taxa de Aprovação Geral (Nota >= {limiar:.1f} em tudo):** {estatisticas['taxa_aprovacao_geral']}

    **Resumo Estatístico por Disciplina:**
    {summary_markdown}

    **Instruções:**
    1. Primeiro Parágrafo: desempenho geral da turma (média satisfatória? turma homogênea ou heterogênea? taxa de aprovação preocupante?).
    2. Segundo Parágrafo: analise a disciplina com menor média ({estatisticas['disciplina_menor_media_nome']}) e o número de alunos com nota baixa ({estatisticas.get('alunos_abaixo_limiar_disciplina_critica', 0)}).
    3. Terceiro Parágrafo: com base na tabela, aponte disciplinas com desempenho ruim, compare indicadores e sugira melhorias.

    O tom deve ser profissional e objetivo.
    """

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": modelo, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content')
        if content:
            return content.replace('\n', '<br/>')
        return "Análise da IA indisponível (resposta inesperada da API)."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            return "A análise por IA não pôde ser gerada: a chave da OpenAI é inválida ou expirou."
        return f"Erro HTTP ao chamar a API da IA: {e}"
    except requests.exceptions.RequestException as e:
        return f"Erro de comunicação com a IA: {e}"
    except Exception as e:
        return f"Erro ao processar o comentário da IA: {e}"


# --------------------------------
# Gráficos (retornam figuras matplotlib)
# --------------------------------
def _colunas_com_notas(df_notas, disciplinas_dict):
    """Disciplinas com notas reais (descarta as sem dados ou totalmente zeradas),
    mantendo a coerência com as tabelas do relatório."""
    cols = []
    for c in disciplinas_dict:
        if c in df_notas.columns:
            s = pd.to_numeric(df_notas[c], errors='coerce')
            if s.notna().any() and s.max() > 0:
                cols.append(c)
    return cols


def grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict, max_pts=20):
    presentes = _colunas_com_notas(df_notas, disciplinas_dict)
    if not presentes:
        return None
    media_aluno = df_notas[presentes].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(media_aluno, kde=True, bins=15, ax=ax)
    ax.set_title(f'Distribuição das Médias Finais - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Média Final do Aluno (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict, max_pts=20):
    presentes = _colunas_com_notas(df_notas, disciplinas_dict)
    if not presentes:
        return None
    media = df_notas[presentes].mean().sort_values(ascending=False)
    labels = [get_simplified_name(c, disciplinas_dict) for c in media.index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=media.values, y=labels, ax=ax)
    ax.set_title(f'Média por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Média da Turma (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, max_pts)
    fig.tight_layout()
    return fig


def grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict, max_pts=20):
    presentes = _colunas_com_notas(df_notas, disciplinas_dict)
    if not presentes:
        return None
    df_melted = df_notas.melt(value_vars=presentes, var_name='disciplina_code', value_name='nota')
    df_melted['disciplina_nome'] = df_melted['disciplina_code'].apply(
        lambda c: get_simplified_name(c, disciplinas_dict))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='nota', y='disciplina_nome', data=df_melted, orient='h', ax=ax)
    ax.set_title(f'Dispersão de Notas por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Nota (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, max_pts)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    fig.tight_layout()
    return fig


def grafico_faltas_total_por_aluno(df_faltas, nome_curso, cols_disciplinas):
    cols = [c for c in cols_disciplinas if c in df_faltas.columns]
    if not cols:
        return None
    totais = df_faltas[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).dropna()
    if totais.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(totais, kde=True, bins=15, ax=ax, color='steelblue')
    ax.set_title(f'Distribuição de Faltas Totais por Aluno - {nome_curso}', fontsize=16)
    ax.set_xlabel('Total de Faltas no Bimestre', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def grafico_faltas_boxplot_disciplina(df_faltas, nome_curso, disciplinas_dict, cols_disciplinas):
    cols = [c for c in cols_disciplinas if c in df_faltas.columns]
    if not cols:
        return None
    df_melted = df_faltas.melt(value_vars=cols, var_name='disciplina_code', value_name='faltas')
    df_melted['faltas'] = pd.to_numeric(df_melted['faltas'], errors='coerce')
    df_melted = df_melted.dropna(subset=['faltas'])
    if df_melted.empty:
        return None
    df_melted['disciplina_nome'] = df_melted['disciplina_code'].apply(
        lambda c: get_simplified_name(c, disciplinas_dict))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='faltas', y='disciplina_nome', data=df_melted, orient='h', ax=ax, color='lightcoral')
    ax.set_title(f'Dispersão de Faltas por Disciplina - {nome_curso}', fontsize=16)
    ax.set_xlabel('Faltas no Bimestre', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    fig.tight_layout()
    return fig


def grafico_dispersao_notas_faltas(df_notas, df_faltas, nome_curso, estatisticas, limiar):
    """Dispersão notas × faltas com quadrantes de risco — síntese das listas
    2.2 (abaixo do limiar) e 2.3 (acima da média de faltas) num só gráfico.

    Eixo X = total de faltas do aluno; eixo Y = média do aluno (pontos do
    bimestre). Um ponto por aluno; alunos sem faltas lançadas ficam de fora.

    Os cortes que definem os quadrantes são os mesmos das tabelas 2.2/2.3: o
    **limiar de aprovação** (horizontal) e a **média de faltas da turma**
    (vertical — corte relativo, não o P90, que por construção jogaria só 10%
    da turma para a direita e deixaria os quadrantes da direita vazios). O
    P90 entra só como linha tracejada de referência.

    Quatro quadrantes pintados por baixo dos pontos (``axvspan`` com
    ``alpha≈0,12`` e ``zorder`` abaixo do scatter): **Em dia** (verde),
    **Risco de frequência** (âmbar), **Risco de desempenho** (laranja) e
    **Risco duplo** (vermelho) — rótulos novos, de propósito, para não
    colidir com as faixas do IDA (Confortável/Regular/Atenção/Crítico), que
    são outro eixo de leitura. Só os alunos do quadrante "Risco duplo"
    recebem numeração no ponto (por gravidade: menor média primeiro,
    desempate por mais faltas), com a lista número → nome logo abaixo da
    legenda — nome anotado no próprio ponto não escala para aglomerados.

    Devolve ``None`` sem faltas disponíveis na turma — mesmo contrato dos
    demais gráficos de faltas.
    """
    if not estatisticas.get('faltas_disponiveis', False) or df_faltas is None or df_faltas.empty:
        return None

    disciplinas_com_notas = estatisticas.get('disciplinas_com_notas', [])
    cols_faltas = estatisticas.get('_faltas_cols', [])
    if not disciplinas_com_notas or not cols_faltas:
        return None

    media_faltas_turma = estatisticas.get('media_faltas_total')
    p90_faltas_turma = estatisticas.get('p90_faltas_total')
    if media_faltas_turma is None or pd.isna(media_faltas_turma):
        return None

    media_por_aluno = df_notas[disciplinas_com_notas].apply(
        pd.to_numeric, errors='coerce').mean(axis=1)
    media_map = pd.Series(media_por_aluno.values, index=df_notas['nome'])

    df_f = df_faltas[cols_faltas].apply(pd.to_numeric, errors='coerce')
    total_faltas = df_f.sum(axis=1)
    # só entram alunos com faltas de fato lançadas em pelo menos uma disciplina
    tem_faltas_lancadas = df_f.notna().any(axis=1)
    faltas_map = pd.Series(
        total_faltas[tem_faltas_lancadas].values, index=df_faltas.loc[tem_faltas_lancadas, 'nome'])

    nomes_comuns = [n for n in faltas_map.index if n in media_map.index]
    if not nomes_comuns:
        return None

    pontos = pd.DataFrame({
        'nome': nomes_comuns,
        'faltas': [faltas_map[n] for n in nomes_comuns],
        'media': [media_map[n] for n in nomes_comuns],
    }).dropna(subset=['faltas', 'media'])
    if pontos.empty:
        return None

    def _quadrante(row):
        acima_limiar = row['media'] >= limiar
        acima_faltas = row['faltas'] > media_faltas_turma
        if acima_limiar and not acima_faltas:
            return 'Em dia'
        if acima_limiar and acima_faltas:
            return 'Risco de frequência'
        if not acima_limiar and not acima_faltas:
            return 'Risco de desempenho'
        return 'Risco duplo'

    pontos['quadrante'] = pontos.apply(_quadrante, axis=1)

    cores_quadrante = {
        'Em dia': 'seagreen',
        'Risco de frequência': '#e6a817',
        'Risco de desempenho': '#ff7f0e',
        'Risco duplo': '#d62728',
    }
    cores_pontos = pontos['quadrante'].map(cores_quadrante)

    x_max = max(float(pontos['faltas'].max()), float(media_faltas_turma)) * 1.15 + 1
    y_max = max(float(pontos['media'].max()), float(limiar)) * 1.1 + 1

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # ----- quadrantes pintados por baixo dos pontos (axvspan; ymin/ymax em
    # fração do eixo y, calculados a partir do limiar já em unidades de
    # dados, já que o ylim é 0..y_max) -----
    frac_limiar = limiar / y_max
    for (x0, x1), quadrante in [
        ((0, media_faltas_turma), 'Em dia'),
        ((media_faltas_turma, x_max), 'Risco de frequência'),
    ]:
        ax.axvspan(x0, x1, ymin=frac_limiar, ymax=1.0,
                   color=cores_quadrante[quadrante], alpha=0.12, zorder=0)
    for (x0, x1), quadrante in [
        ((0, media_faltas_turma), 'Risco de desempenho'),
        ((media_faltas_turma, x_max), 'Risco duplo'),
    ]:
        ax.axvspan(x0, x1, ymin=0.0, ymax=frac_limiar,
                   color=cores_quadrante[quadrante], alpha=0.12, zorder=0)

    # ----- cortes de referência (cores distintas: sem isso as duas linhas
    # sólidas pretas ficam indistinguíveis na legenda) -----
    ax.axhline(limiar, color='darkred', linewidth=1.2, zorder=2,
               label=f'Limiar de aprovação — horizontal ({limiar:.1f})')
    ax.axvline(media_faltas_turma, color='darkblue', linewidth=1.2, zorder=2,
               label=f'Média de faltas da turma — vertical ({media_faltas_turma:.1f})')
    if p90_faltas_turma is not None and pd.notna(p90_faltas_turma):
        ax.axvline(p90_faltas_turma, color='grey', linestyle=':', linewidth=1.2, zorder=2,
                   label=f'P90 de faltas (referência, {p90_faltas_turma:.1f})')

    # ----- pontos -----
    ax.scatter(pontos['faltas'], pontos['media'], c=cores_pontos, s=60,
               edgecolor='black', linewidth=0.5, zorder=3)

    # Só o quadrante "Risco duplo" recebe numeração (turma inteira rotulada
    # fica ilegível, e é esse quadrante que exige ação). Nome por ponto não
    # escala: dois alunos com faltas/média quase iguais sempre colidem, e
    # deslocar o texto só empurra o problema para outro lugar do gráfico.
    # Em vez de nome, cada ponto ganha um número pequeno — a numeração é por
    # gravidade (menor média primeiro, desempate por mais faltas), então o
    # número já é ordem de prioridade — e o nome de cada número vai na lista
    # compacta abaixo da legenda, não no gráfico.
    risco_duplo = pontos[pontos['quadrante'] == 'Risco duplo'].sort_values(
        by=['media', 'faltas'], ascending=[True, False]).reset_index(drop=True)
    risco_duplo['numero'] = risco_duplo.index + 1

    # deslocamento cicla entre 4 posições ao redor do ponto: mesmo com dois
    # alunos praticamente no mesmo (faltas, média), pontos consecutivos na
    # ordenação acima nunca recebem a mesma posição, então os números não se
    # sobrepõem. Perto da borda direita do eixo, força o lado esquerdo para
    # o número não vazar da área do gráfico.
    deslocamentos = [(7, 7), (7, -10), (-10, 7), (-10, -10)]
    for i, row in risco_duplo.iterrows():
        dx, dy = deslocamentos[i % len(deslocamentos)]
        if row['faltas'] > x_max * 0.9 and dx > 0:
            dx = -dx
        ha = 'left' if dx > 0 else 'right'
        va = 'bottom' if dy > 0 else 'top'
        ax.annotate(str(row['numero']), (row['faltas'], row['media']),
                    textcoords='offset points', xytext=(dx, dy),
                    fontsize=8, fontweight='bold', ha=ha, va=va, zorder=4)

    # rótulo de cada quadrante no canto correspondente
    rotulos_canto = [
        (0.02, 0.98, 'Em dia', cores_quadrante['Em dia']),
        (0.98, 0.98, 'Risco de frequência', cores_quadrante['Risco de frequência']),
        (0.02, 0.02, 'Risco de desempenho', cores_quadrante['Risco de desempenho']),
        (0.98, 0.02, 'Risco duplo', cores_quadrante['Risco duplo']),
    ]
    for x_rel, y_rel, texto, cor in rotulos_canto:
        ha = 'left' if x_rel < 0.5 else 'right'
        va = 'top' if y_rel > 0.5 else 'bottom'
        ax.text(x_rel, y_rel, texto, transform=ax.transAxes, fontsize=10,
                fontweight='bold', color=cor, ha=ha, va=va, zorder=4)

    ax.set_title(f'Dispersão de Notas × Faltas - {nome_curso}', fontsize=16)
    ax.set_xlabel('Total de Faltas no Bimestre', fontsize=12)
    ax.set_ylabel('Média do Aluno', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
    legenda = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=9)
    fig.tight_layout()

    # lista número → nome do quadrante "Risco duplo", logo abaixo da legenda
    # das linhas de corte. Empacota os itens "N Nome" em linhas sem estourar
    # a largura da figura (quebra por número de caracteres, não por item, já
    # que o nome do aluno tem tamanho variável). A posição vertical é lida da
    # própria legenda já desenhada (bbox em coordenadas de figura), então
    # funciona com 2 ou 3 linhas de legenda sem precisar de conta manual.
    if not risco_duplo.empty:
        itens = [f'{int(n)} {nome}' for n, nome in
                 zip(risco_duplo['numero'], risco_duplo['nome'])]
        separador = ' · '
        largura_max_chars = 100
        linhas, linha_atual, comprimento_atual = [], [], 0
        for item in itens:
            acrescimo = len(item) + (len(separador) if linha_atual else 0)
            if linha_atual and comprimento_atual + acrescimo > largura_max_chars:
                linhas.append(separador.join(linha_atual))
                linha_atual, comprimento_atual = [item], len(item)
            else:
                linha_atual.append(item)
                comprimento_atual += acrescimo
        if linha_atual:
            linhas.append(separador.join(linha_atual))
        texto_lista = '\n'.join(linhas)

        fig.canvas.draw()
        bbox_legenda = legenda.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox_legenda_fig = bbox_legenda.transformed(fig.transFigure.inverted())
        fig.text(0.5, bbox_legenda_fig.y0 - 0.015, texto_lista,
                  transform=fig.transFigure, fontsize=8, ha='center', va='top', zorder=4)

    return fig


def grafico_stem_boxplot(df_notas, nome_curso, disciplinas_dict, cols_stem, max_pts, limiar):
    """Boxplot horizontal das notas nas disciplinas STEM, com linha vertical
    tracejada no limiar de aprovação. Devolve None sem coluna STEM com notas."""
    stem_dict = {c: disciplinas_dict[c] for c in (cols_stem or [])
                 if c in (disciplinas_dict or {})}
    presentes = _colunas_com_notas(df_notas, stem_dict)
    if not presentes:
        return None
    df_melted = df_notas.melt(value_vars=presentes, var_name='disciplina_code', value_name='nota')
    df_melted['disciplina_nome'] = df_melted['disciplina_code'].apply(
        lambda c: get_simplified_name(c, disciplinas_dict))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='nota', y='disciplina_nome', data=df_melted, orient='h', ax=ax)
    ax.axvline(limiar, linestyle='--', color='darkred',
               label=f'Limiar de aprovação ({limiar:.1f})')
    ax.set_title(f'Dispersão de Notas nas Disciplinas STEM - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Nota (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, max_pts)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def grafico_stem_faixas(df_notas, nome_curso, disciplinas_dict, cols_stem, max_pts):
    """Barras horizontais empilhadas com o percentual de alunos por faixa de
    aproveitamento (``< 40%``, ``40–60%``, ``60–80%``, ``≥ 80%`` da
    pontuação do bimestre), uma barra por disciplina STEM — responde onde
    está a massa crítica de cada disciplina, o que o boxplot não responde.
    Paleta semáforo (vermelho → verde). Devolve None sem coluna STEM com
    notas."""
    stem_dict = {c: disciplinas_dict[c] for c in (cols_stem or [])
                 if c in (disciplinas_dict or {})}
    presentes = _colunas_com_notas(df_notas, stem_dict)
    if not presentes:
        return None

    faixas = ['< 40%', '40–60%', '60–80%', '≥ 80%']
    cores = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c']

    labels = []
    distribuicoes = []
    for c in presentes:
        s = pd.to_numeric(df_notas[c], errors='coerce').dropna()
        if s.empty:
            continue
        pct = s / max_pts * 100
        n = len(pct)
        distribuicoes.append([
            (pct < 40).sum() / n * 100,
            ((pct >= 40) & (pct < 60)).sum() / n * 100,
            ((pct >= 60) & (pct < 80)).sum() / n * 100,
            (pct >= 80).sum() / n * 100,
        ])
        labels.append(get_simplified_name(c, disciplinas_dict))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.6 * len(labels) + 1.5)))
    esquerda = [0.0] * len(labels)
    for i, faixa in enumerate(faixas):
        valores = [d[i] for d in distribuicoes]
        ax.barh(labels, valores, left=esquerda, color=cores[i], label=faixa)
        esquerda = [e + v for e, v in zip(esquerda, valores)]

    ax.set_title(f'Faixas de Aproveitamento nas Disciplinas STEM - {nome_curso}', fontsize=16)
    ax.set_xlabel('Percentual de Alunos', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    fig.tight_layout()
    return fig


def grafico_stem_sobreposicao(df_notas, nome_curso, cols_stem, limiar):
    """Barras verticais com a contagem de alunos abaixo do limiar em 0, 1, 2
    ou 3 das disciplinas STEM — identifica o núcleo duro (quem está mal nas
    três ao mesmo tempo precisa de intervenção diferente de quem está mal em
    uma só). Devolve None sem coluna STEM com notas."""
    presentes = [c for c in (cols_stem or []) if c in df_notas.columns]
    if not presentes:
        return None
    df_num = df_notas[presentes].apply(pd.to_numeric, errors='coerce')
    if df_num.dropna(how='all').empty:
        return None

    n_abaixo = (df_num < limiar).sum(axis=1)
    contagem = n_abaixo.value_counts().reindex(range(len(presentes) + 1), fill_value=0)

    paleta = ['seagreen', '#ffdd57', '#ff7f0e', '#d62728']
    cores = [paleta[min(i, len(paleta) - 1)] for i in contagem.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(contagem.index.astype(str), contagem.values, color=cores)
    ax.set_title(f'Sobreposição de Baixo Desempenho nas STEM - {nome_curso}', fontsize=16)
    ax.set_xlabel('Nº de disciplinas STEM abaixo do limiar', fontsize=12)
    ax.set_ylabel('Número de Alunos', fontsize=12)
    for i, v in enumerate(contagem.values):
        ax.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    return fig


def grafico_stem_media_comparativa(df_notas, nome_curso, disciplinas_dict, cols_stem,
                                    media_geral, max_pts, limiar):
    """Barras da média de cada disciplina STEM, com linhas de referência na
    média geral da turma e no limiar. Devolve None sem coluna STEM com notas."""
    stem_dict = {c: disciplinas_dict[c] for c in (cols_stem or [])
                 if c in (disciplinas_dict or {})}
    presentes = _colunas_com_notas(df_notas, stem_dict)
    if not presentes:
        return None
    media = df_notas[presentes].apply(pd.to_numeric, errors='coerce').mean()
    labels = [get_simplified_name(c, disciplinas_dict) for c in media.index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=media.values, y=labels, ax=ax)
    ax.axvline(limiar, linestyle='--', color='darkred',
               label=f'Limiar de aprovação ({limiar:.1f})')
    ax.axvline(media_geral, linestyle=':', color='darkblue',
               label=f'Média geral da turma ({media_geral:.1f})')
    ax.set_title(f'Média por Disciplina STEM - {nome_curso}', fontsize=16)
    ax.set_xlabel(f'Média da Turma (0 a {max_pts})', fontsize=12)
    ax.set_ylabel('Disciplina', fontsize=12)
    ax.set_xlim(0, max_pts)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def grafico_ida_alunos(ida_df, ida_medio_turma, nome_curso):
    """Barras horizontais do IDA apenas dos alunos abaixo da média de IDA da
    turma (a turma inteira num só gráfico fica ilegível), ordenadas do menor
    para o maior: vermelhas para IDA negativo, verdes para positivo. Eixo x
    fixo em [-1, +1], linha cheia no zero (limiar de aprovação) e tracejada na
    média de IDA da turma. Devolve None com ``ida_df`` vazio ou sem ninguém
    abaixo da média."""
    if ida_df is None or ida_df.empty or 'ida' not in ida_df.columns:
        return None
    if ida_medio_turma is None or pd.isna(ida_medio_turma):
        return None
    df = ida_df.dropna(subset=['ida'])
    df = df[df['ida'] < ida_medio_turma].sort_values(by=['ida', 'nome'])
    if df.empty:
        return None
    cores = ['darkred' if v < 0 else 'seagreen' for v in df['ida']]
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.4 * len(df) + 1.5)))
    ax.barh(df['nome'].tolist(), df['ida'].tolist(), color=cores)
    ax.axvline(0.0, color='black', linewidth=1.2,
               label='Limiar de aprovação (IDA = 0)')
    ax.axvline(ida_medio_turma, color='grey', linestyle='--', linewidth=1.2,
               label=f'Média de IDA da turma ({ida_medio_turma:+.2f})'.replace('.', ','))
    ax.set_xlim(-1, 1)
    ax.set_title(f'IDA — Alunos Abaixo da Média da Turma ({nome_curso})', fontsize=14)
    ax.set_xlabel('IDA (-1 a +1)', fontsize=12)
    ax.set_ylabel('Aluno', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig


def gerar_todos_graficos(df_notas, nome_curso, disciplinas_dict, estatisticas, df_faltas=None):
    """Gera todas as figuras e devolve um dicionário {chave: Figure|None}."""
    max_pts = estatisticas.get('max_pontos_bimestre', 20)
    limiar = estatisticas.get('limiar_aprovacao', 12.0)
    figuras = {
        'distribuicao_geral': grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict, max_pts),
        'media_disciplina': grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict, max_pts),
        'boxplot_disciplinas': grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict, max_pts),
    }
    if estatisticas.get('stem_disponivel'):
        cols_stem = estatisticas.get('stem_codigos', [])
        figuras['stem_boxplot'] = grafico_stem_boxplot(
            df_notas, nome_curso, disciplinas_dict, cols_stem, max_pts, limiar)
        figuras['stem_faixas'] = grafico_stem_faixas(
            df_notas, nome_curso, disciplinas_dict, cols_stem, max_pts)
        figuras['stem_sobreposicao'] = grafico_stem_sobreposicao(
            df_notas, nome_curso, cols_stem, limiar)
        figuras['stem_media_comparativa'] = grafico_stem_media_comparativa(
            df_notas, nome_curso, disciplinas_dict, cols_stem,
            estatisticas.get('media_geral_turma', 0.0), max_pts, limiar)
    if df_faltas is not None and estatisticas.get('faltas_disponiveis'):
        cols = estatisticas.get('_faltas_cols', [])
        figuras['faltas_total_aluno'] = grafico_faltas_total_por_aluno(df_faltas, nome_curso, cols)
        figuras['faltas_boxplot_disciplina'] = grafico_faltas_boxplot_disciplina(
            df_faltas, nome_curso, disciplinas_dict, cols)
        figuras['dispersao_notas_faltas'] = grafico_dispersao_notas_faltas(
            df_notas, df_faltas, nome_curso, estatisticas, limiar)
    return figuras


def _fig_para_imagem(fig):
    """Converte uma figura matplotlib em BytesIO PNG para uso no reportlab."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf


# --------------------------------
# Relatório PDF (em memória) com sumário (TOC)
# --------------------------------
def _cabecalho_factory(logo_path=None):
    def adicionar_cabecalho(canvas, doc):
        canvas.saveState()
        # `doc.pagesize` fica travado no tamanho passado ao BaseDocTemplate
        # (retrato) — o ReportLab só atualiza o tamanho físico do canvas
        # (`canvas._pagesize`) por PageTemplate, em `checkPageSize`, chamado
        # antes de `onPage`. Ler `doc.pagesize` aqui fazia o cabeçalho inteiro
        # (logo e texto institucional) sair desenhado fora da página em
        # paisagem, onde o alto de doc.pagesize (29,7cm) excede a própria
        # altura física da página (21cm). `canvas._pagesize` é o tamanho real
        # da página corrente nas duas orientações.
        largura, altura = canvas._pagesize

        # A logo institucional aparece em TODAS as páginas (retrato e
        # paisagem — `canvas._pagesize` já reflete o template ativo, então a
        # centralização abaixo se adapta sozinha às duas orientações),
        # centrada verticalmente na faixa entre o topo da página e o topo do
        # frame do corpo. Nos dois templates o frame do corpo abre a mesma
        # folga de 4,1 cm no topo (retrato: y=2,5cm + altura-6,6cm; paisagem:
        # y=2cm + altura-6,1cm), por isso a constante abaixo vale para os dois.
        if logo_path:
            try:
                logo = ImageReader(logo_path)
                logo_w, logo_h = logo.getSize()
                aspect = logo_h / float(logo_w)
                display_h = 2.6 * cm
                display_w = display_h / aspect
                topo_frame = altura - 4.1 * cm
                centro_y = (altura + topo_frame) / 2.0
                y_logo = centro_y - display_h / 2.0
                canvas.drawImage(logo, 1.5 * cm, y_logo,
                                 width=display_w, height=display_h,
                                 preserveAspectRatio=True, mask='auto')
            except Exception:
                pass

        canvas.setFont('Times-Bold', 10)
        canvas.setFillColor(colors.HexColor('#002060'))
        y = altura - 2.3 * cm
        canvas.drawCentredString(largura / 2.0, y, "Serviço Público Federal")
        canvas.drawCentredString(largura / 2.0, y - 0.5 * cm, "Ministério da Educação")
        canvas.drawCentredString(largura / 2.0, y - 1.0 * cm,
                                 "Centro Federal de Educação Tecnológica de Minas Gerais")

        canvas.setFont('Times-Italic', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(
            largura / 2.0, 1.5 * cm,
            "Desenvolvido pelo Professor Diego Camargo (diegocamargo@cefetmg.br).")
        canvas.restoreState()
    return adicionar_cabecalho


class _DocComSumario(BaseDocTemplate):
    """BaseDocTemplate que captura os títulos H1/H2 do story e popula o TOC.

    O ReportLab dispara `afterFlowable` após cada flowable; identificamos os
    Paragraphs estilizados como ``H1Sumario`` ou ``H2Sumario`` e emitimos um
    ``TOCEntry`` com o nível, texto e número de página correspondente.
    """

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        text = flowable.getPlainText()
        if style_name == 'H1Sumario':
            self.notify('TOCEntry', (0, text, self.page))
        elif style_name == 'H2Sumario':
            self.notify('TOCEntry', (1, text, self.page))


def criar_relatorio_pdf(nome_curso, estatisticas, figuras, logo_path=None, estatisticas_multibimestre=None):
    """Cria o relatório em PDF e devolve um BytesIO pronto para download."""
    buffer = io.BytesIO()
    doc = _DocComSumario(buffer, pagesize=A4)
    largura, altura = A4
    # Frame retrato: abre 0,6 cm a mais no topo (altura-6cm → altura-6,6cm) para
    # dar espaço à logo de 2,6 cm sem furar a primeira linha do corpo.
    frame = Frame(2.5 * cm, 2.5 * cm, largura - 5 * cm, altura - 6.6 * cm, id='normal')
    # Frame paisagem (tabela completa de alunos, item 2.1): nasce já com a
    # mesma folga de topo do retrato (4,1 cm), com margens laterais/inferior
    # próprias para maximizar a largura útil da tabela.
    largura_pais, altura_pais = landscape(A4)
    frame_pais = Frame(2 * cm, 2 * cm, largura_pais - 4 * cm, altura_pais - 6.1 * cm, id='fpais')
    doc.addPageTemplates([
        PageTemplate(id='principal', frames=[frame],
                     onPage=_cabecalho_factory(logo_path)),
        PageTemplate(id='paisagem', frames=[frame_pais],
                     onPage=_cabecalho_factory(logo_path), pagesize=landscape(A4)),
    ])

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', parent=styles['BodyText'],
                              alignment=TA_JUSTIFY, fontName='Times-Roman'))
    style_titulo = ParagraphStyle(name='TituloCapa', fontSize=22, alignment=TA_CENTER,
                                  leading=26, spaceAfter=1.5 * cm,
                                  textColor=colors.HexColor('#002060'), fontName='Times-Bold')
    style_subtitulo = ParagraphStyle(name='SubTituloCapa', fontSize=18, alignment=TA_CENTER,
                                     spaceAfter=2 * cm, textColor=colors.HexColor('#002060'),
                                     fontName='Times-Roman')
    style_capa_info = ParagraphStyle(name='CapaInfo', fontSize=13, alignment=TA_CENTER,
                                     spaceBefore=0.3 * cm, fontName='Times-Bold',
                                     textColor=colors.HexColor('#002060'))
    style_data = ParagraphStyle(name='DataCapa', fontSize=12, alignment=TA_CENTER,
                                spaceBefore=9 * cm, fontName='Times-Roman')
    # Estilos H1/H2 que o `_DocComSumario` registra no TOC.
    style_h1 = ParagraphStyle(name='H1Sumario', parent=styles['h1'],
                              fontName='Times-Bold', textColor=colors.HexColor('#002060'),
                              spaceBefore=12, spaceAfter=8)
    style_h2 = ParagraphStyle(name='H2Sumario', parent=styles['h2'],
                              fontName='Times-Bold')
    style_corpo = styles['Justify']
    style_caption = ParagraphStyle(name='Caption', parent=styles['BodyText'],
                                   alignment=TA_LEFT, fontName='Times-Italic',
                                   fontSize=9, textColor=colors.grey)
    style_toc_titulo = ParagraphStyle(name='TocTitulo', fontSize=16, alignment=TA_CENTER,
                                      fontName='Times-Bold',
                                      textColor=colors.HexColor('#002060'),
                                      spaceAfter=0.8 * cm)
    style_nota = ParagraphStyle(name='NotaRodape', parent=styles['BodyText'],
                                alignment=TA_LEFT, fontName='Times-Italic',
                                fontSize=8, textColor=colors.HexColor('#7a3030'),
                                spaceBefore=4)
    style_celula = ParagraphStyle(name='Celula', parent=styles['BodyText'],
                                   fontName='Times-Roman', fontSize=9, leading=11)
    style_celula_b = ParagraphStyle(name='CelulaB', parent=styles['BodyText'],
                                     fontName='Times-Bold', fontSize=9, leading=11)
    # Cabeçalho de tabela com fundo escuro e célula centrada — usados pelas
    # tabelas de alunos (média, IDA e faltas).
    style_cab_branco = ParagraphStyle(name='CabBranco', parent=style_celula_b,
                                       textColor=colors.whitesmoke, alignment=TA_CENTER)
    style_cel_centro = ParagraphStyle(name='CelCentro', parent=style_celula,
                                      alignment=TA_CENTER)

    story = []

    # Numeração automática de capítulos/seções (item: sumário efetivo) e quebras
    # de página que não deixam "fantasmas" — qualquer Spacer pendurado antes da
    # quebra é descartado para não gerar páginas em branco.
    contadores = {'cap': 0, 'sub': 0}

    def h1(texto):
        contadores['cap'] += 1
        contadores['sub'] = 0
        story.append(Paragraph(f"{contadores['cap']}. {texto}", style_h1))

    def h2(texto):
        contadores['sub'] += 1
        story.append(Paragraph(
            f"{contadores['cap']}.{contadores['sub']} {texto}", style_h2))

    def quebra_pagina():
        while story and isinstance(story[-1], Spacer):
            story.pop()
        story.append(PageBreak())

    def quebra_para(template_id):
        """Quebra de página trocando o PageTemplate ativo a partir da
        próxima página (usado para entrar/sair da página em paisagem)."""
        story.append(NextPageTemplate(template_id))
        quebra_pagina()

    # --- Metadados usados em todo o relatório ---
    limiar = estatisticas.get('limiar_aprovacao', 12.0)
    max_pts = estatisticas.get('max_pontos_bimestre', 20)
    bim = estatisticas.get('bimestre_num')
    meta = estatisticas.get('metadados', {}) or {}
    serie = meta.get('serie')
    serie_txt = {1: '1ª Série', 2: '2ª Série', 3: '3ª Série'}.get(serie)

    # --- Capa ---
    story.append(Spacer(1, 5 * cm))
    story.append(Paragraph("RELATÓRIO DE ACOMPANHAMENTO ACADÊMICO", style_titulo))
    subtitulo = f"Curso Técnico em {nome_curso.title()}"
    if serie_txt:
        subtitulo += f" — {serie_txt}"
    story.append(Paragraph(subtitulo, style_subtitulo))
    partes_capa = []
    if bim:
        partes_capa.append(f"{bim}º Bimestre")
    if meta.get('periodo_letivo'):
        partes_capa.append(f"Período Letivo {meta['periodo_letivo']}")
    if partes_capa:
        story.append(Paragraph(" · ".join(partes_capa), style_capa_info))
    data_pt = format_date(datetime.now(), format="d 'de' MMMM 'de' y", locale='pt_BR')
    story.append(Paragraph(data_pt, style_data))
    quebra_pagina()

    # --- Sumário ---
    story.append(Paragraph("Sumário", style_toc_titulo))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontName='Times-Bold', fontSize=12, name='TOCH1',
                       leftIndent=0, firstLineIndent=-20, spaceBefore=10, leading=16),
        ParagraphStyle(fontName='Times-Roman', fontSize=10, name='TOCH2',
                       leftIndent=20, firstLineIndent=-20, spaceBefore=4, leading=14),
    ]
    story.append(toc)
    quebra_pagina()

    # --- Glossário (termos usados no relatório) ---
    h1("Glossário")
    story.append(Paragraph(
        "Os termos abaixo aparecem ao longo deste relatório.", style_caption))
    story.append(Spacer(1, 0.3 * cm))
    termos = [
        ("Média", "Soma das notas dividida pela quantidade (de alunos ou de "
                  "disciplinas). Indica o desempenho típico."),
        ("Mediana", "Valor central quando as notas são ordenadas; metade da turma "
                    "fica acima e metade abaixo. É menos sensível a casos extremos "
                    "que a média."),
        ("Desvio Padrão (σ)", "Mede o quanto as notas se afastam da média. Quanto "
                              "maior, mais heterogênea é a turma."),
        ("Pontuação do Bimestre", f"Total de pontos distribuíveis no bimestre "
                                  f"(20 no 1º e 3º; 30 no 2º e 4º). Neste relatório: {max_pts}."),
        ("Limiar de Aprovação Parcial", "60% da pontuação do bimestre — referência "
                                        f"de acompanhamento (aqui, ≥ {limiar:.1f})."),
        ("Taxa de Aprovação", "Percentual de alunos com nota igual ou acima do "
                              "limiar em todas as disciplinas."),
        ("P90 (Percentil 90)", "Valor abaixo do qual estão 90% dos alunos. Em "
                               "faltas, quem ultrapassa o P90 destoa do grupo."),
        ("μ + 2σ", "Média mais dois desvios padrão. Limite estatístico que sinaliza "
                   "valores atípicos (usado na análise de faltas)."),
        ("Disciplina sem dados", "Disciplina sem nenhuma nota lançada ou com todas "
                                 "as notas zeradas. Não entra nas estatísticas."),
        ("Asterisco (*)",
         "Sinaliza disciplinas com suspeita de lançamento incompleto (seção 3), por "
         "um ou mais dos seguintes motivos: a nota máxima da turma está bem abaixo "
         "da pontuação do bimestre; a disciplina destoa das demais do mapa; ou só "
         "uma parte dos alunos tem nota lançada. Quando a nota máxima observada é "
         "igual ou inferior à metade da pontuação do bimestre, o sinal é tratado à "
         "parte, com texto mais forte, por ser o caso em que a disciplina "
         "provavelmente não foi lançada. Em qualquer caso, convém confirmar com "
         "o(a) professor(a) responsável."),
        ("Probabilidade de Reprovação",
         "Estimativa (seção 5.4, só na análise multibimestral) da chance de "
         "reprovação anual por nota no CEFET-MG (aprovação = 60 de 100 pontos: "
         "20/30/20/30), considerando apenas notas dos bimestres regulares — nem "
         "a exigência de 75% de frequência anual (precisa de carga horária, que "
         "o Mapa de Turma não traz) nem a recuperação/exame final entram na "
         "conta. Três situações: pontos que faltam (N) ≤ 0 → \"Aprovado por "
         "nota\" (0%); N maior que os pontos ainda em disputa (R) → \"Não "
         "alcança 60 pontos nas notas regulares\" — probabilidade não estimada "
         "(exibida como \"—\"), porque a recuperação é uma via de aprovação que "
         "este modelo não enxerga, e não é 100% de reprovação; caso contrário, "
         "\"Depende\" — a probabilidade é estimada por um modelo normal a "
         "partir do aproveitamento do próprio aluno nos bimestres já lançados, "
         "com encolhimento estatístico (dados de poucos bimestres puxados em "
         "direção à variância da turma, para não superestimar a confiança) e "
         "piso de desvio-padrão. O resultado é sempre limitado entre 1% e 99% "
         "— o modelo nunca afirma certeza. Assume que o desempenho futuro do "
         "aluno se parece estatisticamente com o seu passado; não é decisão "
         "oficial de retenção."),
        ("Faixa de aproveitamento",
         "As quatro faixas do gráfico de faixas de aproveitamento das disciplinas "
         "STEM (seção 4): menos de 40%, de 40% a 60%, de 60% a 80% e 80% ou mais da "
         "pontuação do bimestre. Mostram onde está concentrada a massa de alunos "
         "em cada disciplina — o boxplot mostra a dispersão, a faixa mostra o "
         "volume."),
        ("Quadrantes de Notas × Faltas",
         "Os quatro rótulos do gráfico de dispersão notas × faltas (seção 2.5): "
         "Em dia (nota ≥ limiar e faltas ≤ média da turma), Risco de frequência "
         "(nota ≥ limiar, faltas > média), Risco de desempenho (nota < limiar, "
         "faltas ≤ média) e Risco duplo (nota < limiar e faltas > média — os "
         "únicos numerados no gráfico, por gravidade). Os cortes são o limiar de "
         "aprovação (horizontal) e a média de faltas da turma (vertical) — a "
         "mesma média usada em 2.3, corte relativo à turma, não o limite legal de "
         "25% da carga horária. Não confundir estes rótulos com as faixas do IDA "
         "(Confortável/Regular/Atenção/Crítico): são eixos de leitura diferentes."),
        ("Média Anual Parcial/Completa",
         "Estimativa da média anual ponderada obtida pelos pesos oficiais do CEFET-MG (20/30/20/30 "
         "pontos para os 1º, 2º, 3º e 4º bimestres, respectivamente). É rotulada como Completa "
         "se os dados de todos os 4 bimestres forem fornecidos, ou Parcial se baseada em "
         "apenas 1, 2 ou 3 bimestres."),
        ("Tendência",
         "Direção do desempenho do aluno (melhora, estável ou queda) comparando o percentual de "
         "nota obtida em relação à nota máxima no bimestre atual versus o bimestre anterior."),
        ("Faltas Acumuladas",
         "Soma simples da quantidade de faltas do aluno ao longo de todos os bimestres "
         "enviados. É rotulada como parcial se não incluir dados dos 4 bimestres."),
        ("IDA — Índice de Desempenho do Aluno",
         "Índice de -1 a +1 que resume a situação do aluno no <b>acumulado dos "
         "bimestres enviados</b> (seção 5.2 — exclusivo da análise multibimestral, "
         "com ao menos 2 bimestres), somando nota e falta acumuladas. O zero é o "
         "limiar de aprovação parcial acumulado (60% da soma dos pontos dos "
         "bimestres enviados): IDA negativo significa estar abaixo do necessário "
         "para aprovação no acumulado até aqui. Fórmula: S = soma das médias do "
         "aluno nos bimestres em que tem nota; P = soma da pontuação desses "
         "bimestres; L = 0,60·P; a componente de notas é (S-L)/(P-L) para S ≥ L, e "
         "(S-L)/L para S &lt; L — sempre entre -1 e +1. A componente de faltas é "
         "1 - (faltas acumuladas ÷ P90 acumulado de faltas da turma), limitada "
         "entre -1 e +1. O IDA combina as duas pelos pesos 0,7 e 0,3; sem faltas "
         "lançadas (ou com P90 igual a zero), usa apenas as notas (peso 1,0). "
         "Faixas de leitura: IDA ≥ +0,30 Confortável · 0 a +0,30 Regular · -0,30 a 0 "
         "Atenção · abaixo de -0,30 Crítico. É indicador de acompanhamento, não "
         "regra oficial de retenção do CEFET-MG."),
        ("Disciplinas STEM",
         "Matemática, Física e Química, identificadas pelo nome na legenda do mapa "
         "— os códigos do SIGAA mudam a cada série, o nome não. Ganham seção "
         "própria porque são as disciplinas que mais reprovam no EPTNM e, no boxplot "
         "geral, ficam diluídas entre as técnicas e as humanidades. Educação Física "
         "não entra no grupo."),
        ("Abaixo do Limiar",
         "Critério de entrada da lista de notas (seção 2.2): entra todo aluno cuja "
         "média própria é menor que o limiar de aprovação parcial (60% da "
         "pontuação do bimestre). É um corte <b>absoluto</b>, ligado à aprovação — "
         "diferente de um corte relativo à turma, o tamanho da lista não depende do "
         "desempenho dos colegas. Aluno exatamente no limiar não entra."),
        ("Acima da média de faltas",
         "O espelho do critério de notas para faltas (seção 2.3): entra todo aluno "
         "cujo total de faltas supera a média de faltas totais da turma. Ao "
         "contrário do corte de notas (absoluto, no limiar de aprovação), este "
         "continua sendo um corte <b>relativo</b> à turma — muda de tamanho "
         "conforme a turma, e estar acima da média não significa faltar em excesso "
         "pela regra oficial. A coluna Sinal destaca, dentro da lista, quem também "
         "ultrapassa o P90 e o μ+2σ."),
    ]
    linhas_gloss = [[Paragraph(f"<b>{t}</b>", style_celula_b), Paragraph(d, style_celula)]
                    for t, d in termos]
    tabela_gloss = Table(linhas_gloss, colWidths=[4.5 * cm, 11.5 * cm])
    tabela_gloss.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef1f7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(tabela_gloss)
    quebra_pagina()

    # --- Estatísticas gerais ---
    h1("Estatísticas Gerais da Turma")
    if bim:
        story.append(Paragraph(
            f"Bimestre: <b>{bim}º</b> · Pontuação máxima: <b>{max_pts}</b> · "
            f"Aprovação parcial (≥ 60%): <b>{limiar:.1f}</b>",
            style_caption,
        ))
    story.append(Spacer(1, 0.4 * cm))
    dados_gerais = [
        ['Total de Alunos:', estatisticas['total_alunos']],
        ['Média Geral da Turma:', f"{estatisticas['media_geral_turma']:.2f}"],
        ['Desvio Padrão das Médias:', f"{estatisticas['desvio_padrao_medias']:.2f}"],
        [f'Taxa de Aprovação (≥ {limiar:.1f} em tudo):', estatisticas['taxa_aprovacao_geral']],
        ['Disciplina com Maior Média:', f"{estatisticas['disciplina_maior_media_nome']} ({estatisticas['maior_media']:.2f})"],
        ['Disciplina com Menor Média:', f"{estatisticas['disciplina_menor_media_nome']} ({estatisticas['menor_media']:.2f})"],
    ]
    if estatisticas.get('faltas_disponiveis'):
        dados_gerais.extend([
            ['Média de Faltas por Aluno:', f"{estatisticas.get('media_faltas_total', 0.0):.1f}"],
            ['Mediana de Faltas:', f"{estatisticas.get('mediana_faltas_total', 0.0):.1f}"],
            ['P90 de Faltas:', f"{estatisticas.get('p90_faltas_total', 0.0):.1f}"],
        ])
        if 'total_faltas_turma' in estatisticas:
            dados_gerais.append(
                ['Total de Faltas da Turma:', str(estatisticas['total_faltas_turma'])])
    if meta.get('turma'):
        dados_gerais.insert(0, ['Turma:', meta['turma']])
    tabela_geral = Table(dados_gerais, colWidths=[7 * cm, 9 * cm])
    tabela_geral.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
    ]))
    story.append(tabela_geral)
    story.append(Spacer(1, 1 * cm))

    # --- 2.1 Desempenho e Frequência por Aluno (paisagem) ---
    quebra_para('paisagem')
    h2("Desempenho e Frequência por Aluno")
    story.append(Spacer(1, 0.3 * cm))
    style_pais_cab = ParagraphStyle(name='PaisCab', parent=style_cab_branco, fontSize=6, leading=7)
    style_pais_cel = ParagraphStyle(name='PaisCel', parent=style_celula, fontSize=6, leading=7)
    style_pais_cel_centro = ParagraphStyle(name='PaisCelCentro', parent=style_cel_centro, fontSize=6, leading=7)
    tabela_completa_df = estatisticas.get('tabela_completa_df', pd.DataFrame())
    legenda_disc = estatisticas.get('legenda_disciplinas', [])
    if not tabela_completa_df.empty:
        colunas_tab = tabela_completa_df.columns.tolist()
        cabecalho_pais = [Paragraph(f"<b>{c}</b>", style_pais_cab) for c in colunas_tab]
        dados_pais = [cabecalho_pais]
        for _, row in tabela_completa_df.iterrows():
            linha = [Paragraph(str(row['Aluno']), style_pais_cel)]
            for c in colunas_tab[1:]:
                linha.append(Paragraph(str(row[c]), style_pais_cel_centro))
            dados_pais.append(linha)

        largura_pagina_util = 25 * cm
        largura_aluno_col = 3.2 * cm
        largura_media_col = 1.8 * cm
        largura_faltas_col = 1.8 * cm
        n_disc = len(legenda_disc)
        resto = largura_pagina_util - largura_aluno_col - largura_media_col - largura_faltas_col
        largura_disc_col = (resto / n_disc) if n_disc else resto
        col_widths_pais = [largura_aluno_col] + [largura_disc_col] * n_disc + \
            [largura_media_col, largura_faltas_col]

        tabela_pais = Table(dados_pais, colWidths=col_widths_pais, repeatRows=1)
        tabela_pais.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002060')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(tabela_pais)

        if legenda_disc:
            pares_por_linha = 3
            linhas_legenda = []
            for i in range(0, len(legenda_disc), pares_por_linha):
                grupo = legenda_disc[i:i + pares_por_linha]
                linha = []
                for abrev, nome in grupo:
                    linha.append(Paragraph(f"<b>{abrev}</b>", style_celula_b))
                    linha.append(Paragraph(nome, style_celula))
                while len(linha) < pares_por_linha * 2:
                    linha.append(Paragraph('', style_celula))
                linhas_legenda.append(linha)
            col_widths_legenda = [1.3 * cm, 6.9 * cm] * pares_por_linha
            tabela_legenda = Table(linhas_legenda, colWidths=col_widths_legenda)
            tabela_legenda.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef1f7')),
                ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#eef1f7')),
                ('BACKGROUND', (4, 0), (4, -1), colors.HexColor('#eef1f7')),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ]))
            # KeepTogether: legenda + tabela como bloco atômico — sem isso a
            # tabela D1…Dn (só 2-3 linhas) caía inteira numa segunda página em
            # paisagem, desperdiçando a folha, mesmo havendo espaço de sobra
            # na primeira página. Espaçamento enxuto (spacers e padding
            # menores que os do resto do relatório) para o bloco caber no
            # resto de página que sobra depois da tabela de alunos.
            story.append(KeepTogether([
                Spacer(1, 0.15 * cm),
                Paragraph(
                    "Legenda das disciplinas — os cabeçalhos abreviados (D1…Dn) garantem "
                    "que a tabela caiba independentemente do comprimento dos nomes do "
                    "SIGAA. Célula \"nota / faltas\"; sem faltas lançadas para a "
                    "disciplina, mostra só a nota.", style_caption),
                Spacer(1, 0.1 * cm),
                tabela_legenda,
            ]))
    else:
        story.append(Paragraph("Sem dados suficientes para montar a tabela de desempenho por aluno.", style_corpo))
    quebra_para('principal')

    # --- 2.2 Alunos com Média Abaixo do Limiar ---
    h2("Alunos com Média Abaixo do Limiar")
    story.append(Spacer(1, 0.3 * cm))
    abaixo_limiar = estatisticas.get('alunos_abaixo_limiar', pd.DataFrame())
    if not abaixo_limiar.empty:
        dados_abaixo = [[
            Paragraph("<b>Aluno</b>", style_cab_branco),
            Paragraph("<b>Média</b>", style_cab_branco),
            Paragraph(f"<b>Disciplinas &lt; {limiar:.1f}</b>", style_cab_branco),
        ]]
        for _, row in abaixo_limiar.iterrows():
            dados_abaixo.append([
                Paragraph(str(row['nome']), style_celula),
                Paragraph(str(row['Média']), style_cel_centro),
                Paragraph(str(row['disciplinas_abaixo_limiar']), style_cel_centro),
            ])
        tabela_abaixo = Table(
            dados_abaixo, colWidths=[8.5 * cm, 3.5 * cm, 4 * cm], repeatRows=1)
        tabela_abaixo.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(tabela_abaixo)
        story.append(Spacer(1, 0.2 * cm))
        n_abaixo = estatisticas.get('n_abaixo_limiar', len(abaixo_limiar))
        legenda_abaixo = (
            f"Entram os {n_abaixo} alunos (de {estatisticas['total_alunos']}) cuja "
            f"média própria é inferior ao limiar de aprovação parcial ({limiar:.1f} "
            f"— 60% da pontuação do bimestre). É um corte absoluto: aluno "
            f"exatamente no limiar não entra."
        )
        n_acima = estatisticas.get('n_acima_limiar_com_disciplina_abaixo', 0)
        if n_acima:
            legenda_abaixo += (
                f" Outros {n_acima} alunos com média igual ou acima do limiar ainda "
                f"têm ao menos uma disciplina abaixo do limiar."
            )
        story.append(Paragraph(legenda_abaixo, style_caption))
    else:
        story.append(Paragraph(
            "Nenhum aluno ficou com média abaixo do limiar de aprovação parcial.", style_corpo))
    story.append(Spacer(1, 0.8 * cm))

    # --- 2.3 Alunos com Faltas Acima da Média / 2.4 Resumo de Faltas por Disciplina ---
    if estatisticas.get('faltas_disponiveis'):
        h2("Alunos com Faltas Acima da Média")
        story.append(Paragraph(
            "Os limites estatísticos (P90 e μ+2σ) usados nesta análise <b>não "
            "substituem</b> o limite legal de 25% da carga horária — eles apenas "
            "sinalizam, sem depender do calendário, quem se afasta do "
            "comportamento típico da turma e merece um olhar atento.",
            style_corpo,
        ))
        story.append(Spacer(1, 0.4 * cm))
        acima_faltas = estatisticas.get('alunos_acima_media_faltas', pd.DataFrame())
        if not acima_faltas.empty:
            dados_falt = [[
                Paragraph("<b>Aluno</b>", style_cab_branco),
                Paragraph("<b>Total de Faltas</b>", style_cab_branco),
                Paragraph("<b>Sinal</b>", style_cab_branco),
            ]]
            for _, row in acima_faltas.iterrows():
                dados_falt.append([
                    Paragraph(str(row['nome']), style_celula),
                    Paragraph(str(row['Total Faltas']), style_cel_centro),
                    Paragraph(str(row['Sinal']), style_cel_centro),
                ])
            tabela_falt = Table(
                dados_falt, colWidths=[8 * cm, 4 * cm, 4 * cm], repeatRows=1)
            tabela_falt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7a3030')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(tabela_falt)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                f"Corte de entrada: média de faltas totais da turma = "
                f"{estatisticas.get('media_faltas_total', 0.0):.1f}. A coluna Sinal "
                f"destaca quem, dentro da lista, também ultrapassa o P90 "
                f"({estatisticas.get('p90_faltas_total', 0.0):.1f}) e o μ+2σ "
                f"({estatisticas.get('sigma2_faltas_total', 0.0):.1f}).", style_caption))
            story.append(Spacer(1, 0.8 * cm))
        else:
            story.append(Paragraph(
                "Nenhum aluno passou da média de faltas totais da turma.", style_corpo))
            story.append(Spacer(1, 0.8 * cm))

        h2("Resumo de Faltas por Disciplina")
        story.append(Spacer(1, 0.3 * cm))
        df_faltas_summary = estatisticas.get('faltas_summary_df', pd.DataFrame())
        if not df_faltas_summary.empty:
            style_header_faltas_disc = ParagraphStyle(
                name='HeaderFaltasDisc',
                parent=style_celula_b,
                fontSize=8,
                leading=10,
                textColor=colors.whitesmoke,
                alignment=TA_CENTER
            )
            style_centro_faltas_disc = ParagraphStyle(
                name='CentroFaltasDisc',
                parent=style_celula,
                fontSize=8,
                leading=10,
                alignment=TA_CENTER
            )
            style_esquerda_faltas_disc = ParagraphStyle(
                name='EsquerdaFaltasDisc',
                parent=style_celula,
                fontSize=8,
                leading=10,
                alignment=TA_LEFT
            )
            headers_faltas_disc = [
                Paragraph(f"<b>{col}</b>", style_header_faltas_disc)
                for col in df_faltas_summary.columns
            ]
            dados_falt_disc = [headers_faltas_disc]
            for _, row in df_faltas_summary.iterrows():
                linha = []
                for i, val in enumerate(row):
                    st = style_esquerda_faltas_disc if i == 0 else style_centro_faltas_disc
                    linha.append(Paragraph(str(val), st))
                dados_falt_disc.append(linha)

            cols_widths = [4.5 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 2 * cm, 2 * cm, 2.4 * cm]
            tabela_falt_disc = Table(dados_falt_disc, colWidths=cols_widths)
            tabela_falt_disc.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7a3030')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(tabela_falt_disc)
            story.append(Spacer(1, 0.8 * cm))

    # --- 2.5 Visualizações Gráficas ---
    h2("Visualizações Gráficas")
    story.append(Spacer(1, 0.4 * cm))
    for chave in ['distribuicao_geral', 'media_disciplina', 'boxplot_disciplinas',
                  'faltas_total_aluno', 'faltas_boxplot_disciplina', 'dispersao_notas_faltas']:
        fig = figuras.get(chave)
        if fig is not None:
            story.append(Image(_fig_para_imagem(fig), width=16 * cm, height=11 * cm, kind='proportional'))
            story.append(Spacer(1, 0.8 * cm))

    # --- 3. Resumo estatístico por disciplina ---
    quebra_pagina()
    h1("Resumo Estatístico por Disciplina")
    story.append(Spacer(1, 0.4 * cm))
    df_summary = estatisticas['boxplot_summary_df']
    # Células como Paragraph (não string crua): nome de disciplina comprido
    # (com asterisco de suspeita) invade a coluna vizinha se for texto puro.
    dados_summary = [[
        Paragraph(f"<b>{col}</b>", style_cab_branco) for col in df_summary.columns
    ]]
    for _, row in df_summary.iterrows():
        dados_summary.append([
            Paragraph(str(row[df_summary.columns[0]]), style_celula),
            *[Paragraph(str(v), style_cel_centro) for v in row[1:]],
        ])
    tabela_summary = Table(dados_summary, colWidths=[4.5 * cm, 2 * cm, 2 * cm, 2.5 * cm, 2 * cm, 2 * cm], repeatRows=1)
    tabela_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.cadetblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(tabela_summary)

    # Nota(s) de rodapé (disciplinas com suspeita de lançamento incompleto) —
    # asterisco. `disciplinas_incompletas` é lista de dicionários (codigo, nome,
    # max_observado, cobertura, motivos em A/B/C/D); A tem texto próprio, mais
    # forte, e nunca aparece na nota "a conferir" junto com B/C/D.
    incompletas = estatisticas.get('disciplinas_incompletas') or []
    if incompletas:
        fortes = [d for d in incompletas if 'A' in d['motivos']]
        conferir = [d for d in incompletas if 'A' not in d['motivos']]

        if fortes:
            nomes_fortes = '; '.join(
                f"{d['nome']} (nota máxima observada {d['max_observado']:.1f})"
                for d in fortes)
            story.append(Paragraph(
                f"<b>* (sinal forte)</b> Disciplina(s) provavelmente <b>não "
                f"lançada(s)</b> — nota máxima observada igual ou inferior à "
                f"metade da pontuação do bimestre ({max_pts / 2:.0f}): "
                f"{nomes_fortes}. Confirme com o(a) professor(a) antes de "
                f"considerar estes dados.",
                style_nota,
            ))

        if conferir:
            partes_conferir = []
            for d in conferir:
                motivos_txt = []
                if 'B' in d['motivos']:
                    motivos_txt.append(
                        f"nota máxima da turma ({d['max_observado']:.1f}) bem "
                        f"abaixo da pontuação do bimestre")
                if 'C' in d['motivos']:
                    motivos_txt.append("destoa das demais disciplinas do mapa")
                if 'D' in d['motivos']:
                    motivos_txt.append(
                        f"só {d['cobertura'] * 100:.0f}% dos alunos têm nota lançada")
                partes_conferir.append(f"{d['nome']} ({'; '.join(motivos_txt)})")
            story.append(Paragraph(
                f"<b>*</b> Disciplina(s) com sinal de lançamento possivelmente "
                f"incompleto, a conferir com o(a) professor(a): "
                f"{'; '.join(partes_conferir)}.",
                style_nota,
            ))
    story.append(Spacer(1, 0.8 * cm))

    # --- 3.1 Disciplinas sem Notas Lançadas ---
    sem_dados = estatisticas.get('disciplinas_sem_dados') or []
    if sem_dados:
        h2("Disciplinas sem Notas Lançadas")
        story.append(Paragraph(
            "As disciplinas abaixo não tinham notas lançadas (ou estavam todas "
            "zeradas) no momento do processamento e, por isso, <b>não foram "
            "consideradas</b> nas estatísticas e nos gráficos. Recomenda-se "
            "verificar o lançamento com o(a) professor(a) responsável.",
            style_corpo,
        ))
        story.append(Spacer(1, 0.3 * cm))
        dados_sem = [['Disciplina', 'Observação']]
        for _cod, nome in sem_dados:
            dados_sem.append([
                Paragraph(nome, style_celula),
                Paragraph("Não havia dados lançados para esta disciplina no período.",
                          style_celula),
            ])
        tabela_sem = Table(dados_sem, colWidths=[6 * cm, 10 * cm])
        tabela_sem.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8a6d00')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(tabela_sem)
        story.append(Spacer(1, 0.8 * cm))

    # --- 4. Análise de Matemática, Física e Química (STEM) ---
    if estatisticas.get('stem_disponivel'):
        quebra_pagina()
        h1("Análise de Matemática, Física e Química")
        story.append(Paragraph(
            "Recorte de Matemática, Física e Química, identificadas pelo nome na "
            "legenda do mapa — os códigos do SIGAA mudam a cada série, o nome "
            "não. São as disciplinas que mais reprovam no EPTNM e, no boxplot "
            "geral, ficam diluídas entre as técnicas e as humanidades.", style_corpo))
        story.append(Spacer(1, 0.4 * cm))
        df_stem = estatisticas.get('stem_summary_df', pd.DataFrame())
        if not df_stem.empty:
            dados_stem = [df_stem.columns.tolist()] + df_stem.values.tolist()
            tabela_stem = Table(
                dados_stem,
                colWidths=[4.2 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 2.2 * cm, 1.8 * cm, 2.2 * cm],
                repeatRows=1)
            tabela_stem.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.cadetblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            story.append(tabela_stem)
            story.append(Spacer(1, 0.5 * cm))
        for chave in ('stem_boxplot', 'stem_faixas', 'stem_sobreposicao', 'stem_media_comparativa'):
            fig = figuras.get(chave)
            if fig is not None:
                story.append(Image(_fig_para_imagem(fig),
                                   width=16 * cm, height=11 * cm, kind='proportional'))
                story.append(Spacer(1, 0.5 * cm))

    # --- 5. Análise Multibimestral ---
    if estatisticas_multibimestre is not None:
        quebra_pagina()
        h1("Análise Multibimestral")
        
        # Aviso de limitação
        style_aviso = ParagraphStyle(
            name='AvisoMultibimestre',
            parent=style_celula,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#8a6d00')
        )
        tabela_aviso = Table([[Paragraph(
            "<b>Aviso sobre limitações:</b> Esta análise consolida informações de múltiplos bimestres "
            "com finalidade de acompanhamento pedagógico. A estimativa de Média Anual baseia-se nos "
            "pesos oficiais do CEFET-MG (20/30/20/30). Contudo, a aprovação final depende também da "
            "frequência global anual mínima de 75% da carga horária, dado que não está disponível nesta "
            "análise. As informações apresentadas são indicativas de acompanhamento e não constituem "
            "uma garantia ou decisão oficial de retenção ou aprovação.",
            style_aviso
        )]], colWidths=[16 * cm])
        tabela_aviso.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fcf8e3')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#faf2cc')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(tabela_aviso)
        story.append(Spacer(1, 0.5 * cm))

        # --- 5.1 Estimativa de Média Anual ---
        is_completa = len(estatisticas_multibimestre.get('sorted_bimestres', [])) == 4
        label_status = "COMPLETA" if is_completa else "PARCIAL"
        h2(f"Estimativa de Média Anual ({label_status})")
        story.append(Spacer(1, 0.3 * cm))
        df_anual = estatisticas_multibimestre['df_media_anual']
        if not df_anual.empty:
            style_celula_centro = ParagraphStyle(
                name='CelulaCentroMultibimestreAnual', parent=style_celula, alignment=TA_CENTER)

            headers_anual = [Paragraph(f"<b>{col}</b>", style_celula_b) for col in df_anual.columns]
            dados_anual = [headers_anual]
            for _, row in df_anual.iterrows():
                linha = [
                    Paragraph(str(row['Aluno']), style_celula),
                    Paragraph(str(row['Média Anual']), style_celula_centro),
                    Paragraph(str(row['Faltas Acumuladas']), style_celula_centro),
                    Paragraph(str(row['Status']), style_celula),
                ]
                dados_anual.append(linha)

            tabela_anual = Table(dados_anual, colWidths=[6 * cm, 2.5 * cm, 3 * cm, 4.5 * cm], repeatRows=1)
            tabela_anual.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002060')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(tabela_anual)
            story.append(Spacer(1, 0.8 * cm))

        # --- 5.2 IDA — Índice de Desempenho do Aluno (acumulado) ---
        h2("IDA — Índice de Desempenho do Aluno")
        story.append(Paragraph(
            f"O IDA resume a situação do aluno no acumulado dos bimestres "
            f"enviados, numa escala de -1 a +1 em que o zero é o limiar de "
            f"aprovação parcial acumulado (60% da soma dos pontos dos bimestres "
            f"enviados): valores negativos indicam desempenho abaixo do "
            f"necessário para aprovação no acumulado. A fórmula, os pesos e as "
            f"faixas de leitura estão explicados no glossário.", style_corpo))
        story.append(Spacer(1, 0.4 * cm))
        ida_df_multi = estatisticas_multibimestre.get('ida_df', pd.DataFrame())
        ida_medio_multi = estatisticas_multibimestre.get('ida_medio_turma')
        fig_ida_multi = grafico_ida_alunos(ida_df_multi, ida_medio_multi, nome_curso)
        if fig_ida_multi is not None:
            story.append(Image(_fig_para_imagem(fig_ida_multi),
                               width=16 * cm, height=11 * cm, kind='proportional'))
            plt.close(fig_ida_multi)
            story.append(Spacer(1, 0.4 * cm))
        negativos_multi = (
            ida_df_multi[ida_df_multi['ida'] < 0] if not ida_df_multi.empty else ida_df_multi)
        if not negativos_multi.empty:
            dados_ida_multi = [[
                Paragraph("<b>Aluno</b>", style_cab_branco),
                Paragraph("<b>IDA</b>", style_cab_branco),
                Paragraph("<b>Faixa</b>", style_cab_branco),
            ]]
            for _, row in negativos_multi.iterrows():
                dados_ida_multi.append([
                    Paragraph(str(row['nome']), style_celula),
                    Paragraph(_formatar_ida(row['ida']), style_cel_centro),
                    Paragraph(str(row['faixa']), style_cel_centro),
                ])
            tabela_ida_multi = Table(dados_ida_multi, colWidths=[8 * cm, 4 * cm, 4 * cm], repeatRows=1)
            tabela_ida_multi.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002060')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(tabela_ida_multi)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                "O corte desta tabela é o zero (o limiar de aprovação acumulado), "
                "não a média da turma. O gráfico acima mostra apenas os alunos "
                "abaixo da média de IDA da turma, para manter a leitura legível.",
                style_caption))
        else:
            story.append(Paragraph(
                "Nenhum aluno tem IDA negativo no acumulado — na componente de "
                "notas, a turma inteira está no ou acima do limiar de aprovação "
                "parcial.", style_corpo))
        story.append(Spacer(1, 0.8 * cm))

        # --- 5.3 Alunos com Maior Queda de Desempenho ---
        h2("Alunos com Maior Queda de Desempenho")
        story.append(Spacer(1, 0.3 * cm))
        df_q = estatisticas_multibimestre['df_queda']
        if not df_q.empty:
            style_celula_centro = ParagraphStyle(
                name='CelulaCentroMultibimestreQueda', parent=style_celula, alignment=TA_CENTER)

            headers_q = [Paragraph(f"<b>{col}</b>", style_celula_b) for col in df_q.columns]
            dados_q = [headers_q]
            for _, row in df_q.iterrows():
                linha = [
                    Paragraph(str(row['Aluno']), style_celula),
                    Paragraph(str(row['Bimestre Anterior']), style_celula_centro),
                    Paragraph(str(row['Bimestre Atual']), style_celula_centro),
                    Paragraph(str(row['Diferença (p.p.)']), style_celula_centro),
                ]
                dados_q.append(linha)

            tabela_q = Table(dados_q, colWidths=[6 * cm, 3.5 * cm, 3.5 * cm, 3 * cm], repeatRows=1)
            tabela_q.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7a3030')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(tabela_q)
            story.append(Spacer(1, 0.8 * cm))
        else:
            story.append(Paragraph("Nenhum aluno apresentou queda de desempenho entre os dois últimos bimestres.", style_corpo))
            story.append(Spacer(1, 0.8 * cm))

        # --- 5.4 Probabilidade de Reprovação ---
        h2("Probabilidade de Reprovação")
        story.append(Spacer(1, 0.3 * cm))
        texto_aviso_prob = (
            "<b>Aviso importante:</b> Esta é uma estimativa de acompanhamento que "
            "considera <b>apenas notas</b> — ignora a exigência de frequência "
            "mínima de 75% da carga horária anual <b>e a recuperação/exame "
            "final</b> (a estimativa cobre só os pontos dos bimestres regulares) "
            "— e assume que o desempenho futuro do aluno se parece "
            "estatisticamente com o seu próprio passado. <b>Não é</b> a decisão "
            "oficial de retenção do CEFET-MG."
        )
        tabela_aviso_prob = Table([[Paragraph(texto_aviso_prob, style_corpo)]], colWidths=[16 * cm])
        tabela_aviso_prob.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fcf8e3')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#faf2cc')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(tabela_aviso_prob)
        story.append(Spacer(1, 0.5 * cm))

        df_prob = estatisticas_multibimestre.get('df_probabilidade', pd.DataFrame())
        if not df_prob.empty:
            dados_prob = [[
                Paragraph("<b>Aluno</b>", style_cab_branco),
                Paragraph("<b>Pontos Obtidos</b>", style_cab_branco),
                Paragraph("<b>Pontos Restantes</b>", style_cab_branco),
                Paragraph("<b>Precisa</b>", style_cab_branco),
                Paragraph("<b>Situação</b>", style_cab_branco),
                Paragraph("<b>Prob. de Reprovação</b>", style_cab_branco),
            ]]
            for _, row in df_prob.iterrows():
                prob = row['Prob_Reprovacao']
                prob_txt = '—' if pd.isna(prob) else f"{prob * 100:.0f}%"
                precisa_txt = f"{max(row['N'], 0.0):.1f}".replace('.', ',')
                obtidos_txt = f"{row['S']:.1f}".replace('.', ',') + f" / {row['Pontos_Lancados']:.0f}"
                dados_prob.append([
                    Paragraph(str(row['Aluno']), style_celula),
                    Paragraph(obtidos_txt, style_cel_centro),
                    Paragraph(f"{row['R']:.0f}", style_cel_centro),
                    Paragraph(precisa_txt, style_cel_centro),
                    Paragraph(str(row['Situacao']), style_cel_centro),
                    Paragraph(prob_txt, style_cel_centro),
                ])
            tabela_prob = Table(
                dados_prob, colWidths=[4.5 * cm, 2.6 * cm, 2 * cm, 1.6 * cm, 2.8 * cm, 2.5 * cm],
                repeatRows=1)
            tabela_prob.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002060')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            story.append(tabela_prob)
        else:
            story.append(Paragraph(
                "Sem dados suficientes para estimar a probabilidade de reprovação.",
                style_corpo))
        story.append(Spacer(1, 0.8 * cm))

    # --- 6. Comentário da IA ---
    if estatisticas.get('comentario_ia'):
        quebra_pagina()
        h1("Análise e Comentários (Gerado por Inteligência Artificial)")
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(estatisticas['comentario_ia'], style_corpo))

    # Remove qualquer Spacer pendurado no fim do documento — evita uma página em
    # branco extra quando o último conteúdo termina perto do rodapé (item 6).
    while story and isinstance(story[-1], Spacer):
        story.pop()

    # `multiBuild` faz duas passadas: a primeira coleta as entradas do TOC, a
    # segunda monta o documento final já com os números de página corretos.
    doc.multiBuild(story)
    # Libera as figuras matplotlib para não acumular memória entre relatórios.
    for fig in figuras.values():
        if fig is not None:
            plt.close(fig)
    buffer.seek(0)
    return buffer

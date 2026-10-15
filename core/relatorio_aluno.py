"""Módulo de comparação e relatório individual: aluno × turma.

Contratos e regras:
- D1: Não altera core/relatorios.py; importa o que for necessário de lá.
- D2: Aluno identificado por matrícula, nunca por nome. Matrícula ausente lança ValueError.
- D3: Resumo com média do aluno x média da turma, posição com rank(method='min', ascending=False),
  disciplinas abaixo do limiar, faltas opcionais; tabela por disciplina com apenas
  disciplinas_com_notas (incompletas com ' *'); evolução multibimestral em % do MAX_PONTOS_BIMESTRE.
- D5: listar_alunos, comparar_aluno_turma, evolucao_aluno_turma, criar_relatorio_aluno_pdf.
- E1-E8: consolidar_bimestres, comparar_stem, QUADRANTES, _quadrante,
  grafico_dispersao_stem_aluno.
"""

from datetime import datetime
import io
from typing import Any

from babel.dates import format_date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle
)

from core.disciplinas import _normalizar, disciplinas_stem
from core.relatorios import (
    COR_CABECALHO_TABELA,
    COR_TEXTO_CABECALHO_TABELA,
    MAX_PONTOS_BIMESTRE,
    PERCENTUAL_APROVACAO,
    _cabecalho_factory,
    _fig_para_imagem,
    calcular_estatisticas,
    get_simplified_name,
)

QUADRANTES = {
    'Em dia': 'seagreen',
    'Risco de frequência': '#e6a817',
    'Risco de desempenho': '#ff7f0e',
    'Risco duplo': '#d62728',
}


def listar_alunos(grupos: Any) -> list[tuple[str, str]]:
    """Deduplica alunos por matrícula entre bimestres e grupos e ordena por nome.

    `grupos` é a lista de grupos que o app.py monta para processar (cada grupo,
    uma lista de conjuntos `(df_notas, df_faltas, disciplinas_dict, metadados)`).
    Retorna uma lista de tuplas [(matricula, nome)] ordenada por nome.casefold().
    """
    if not grupos:
        return []

    alunos_map: dict[str, str] = {}  # matricula -> nome

    for grupo in grupos:
        if not grupo:
            continue
        for conjunto in grupo:
            df_notas = conjunto[0]
            if df_notas is None or df_notas.empty or 'matricula' not in df_notas.columns:
                continue
            tem_nome = 'nome' in df_notas.columns
            for _, row in df_notas.iterrows():
                mat = row.get('matricula')
                if pd.isna(mat):
                    continue
                mat_str = str(mat).strip()
                if not mat_str:
                    continue
                nome_val = row.get('nome') if tem_nome else ''
                nome_str = str(nome_val).strip() if pd.notna(nome_val) else ''
                if mat_str not in alunos_map or (not alunos_map[mat_str] and nome_str):
                    alunos_map[mat_str] = nome_str

    ordenados = sorted(alunos_map.items(), key=lambda item: (item[1].casefold(), item[0]))
    return [(mat, nome) for mat, nome in ordenados]


def _quadrante(nota: Any, faltas: Any, limiar: float, media_faltas: float) -> str | None:
    """Classifica o aluno em um dos 4 quadrantes de notas x faltas por disciplina (E6).

    Cortes:
    - nota >= limiar conta como acima do limiar;
    - faltas > media_faltas (estrito) conta como acima da média de faltas (faltas == média conta como abaixo).

    Quadrantes:
    - nota >= limiar e faltas <= media_faltas -> 'Em dia'
    - nota >= limiar e faltas > media_faltas  -> 'Risco de frequência'
    - nota < limiar e faltas <= media_faltas  -> 'Risco de desempenho'
    - nota < limiar e faltas > media_faltas   -> 'Risco duplo'
    """
    if nota is None or pd.isna(nota) or faltas is None or pd.isna(faltas):
        return None
    acima_limiar = float(nota) >= float(limiar)
    acima_faltas = float(faltas) > float(media_faltas)
    if acima_limiar and not acima_faltas:
        return 'Em dia'
    if acima_limiar and acima_faltas:
        return 'Risco de frequência'
    if not acima_limiar and not acima_faltas:
        return 'Risco de desempenho'
    return 'Risco duplo'


def comparar_stem(
    df_notas: pd.DataFrame,
    df_faltas: pd.DataFrame | None,
    disciplinas_dict: dict,
    estat: dict,
    matricula: Any,
) -> pd.DataFrame:
    """Compara o desempenho do aluno nas disciplinas STEM (Matemática, Física, Química).

    Retorna DataFrame com colunas:
        'codigo', 'disciplina', 'nota_aluno', 'media_turma', 'posicao', 'total_com_nota'
        e — se há faltas — 'faltas_aluno', 'media_faltas_turma', 'quadrante'.

    Regras (E5, E6):
    - Disciplinas = disciplinas_stem(disciplinas_dict) ∩ estat['disciplinas_com_notas'],
      ordenadas Matemática, Física, Química.
    - Posição na disciplina via rank(method='min', ascending=False), NaN se aluno sem nota;
    - DataFrame vazio se não houver disciplinas STEM com notas;
    - Lança ValueError se a matrícula não estiver presente em df_notas.
    """
    if df_notas is None or df_notas.empty or 'matricula' not in df_notas.columns:
        raise ValueError(f"Matrícula '{matricula}' não encontrada na turma.")

    str_matricula = str(matricula).strip()
    mats_serie = df_notas['matricula'].astype(str).str.strip()
    idx_aluno = df_notas.index[mats_serie == str_matricula]
    if len(idx_aluno) == 0:
        raise ValueError(f"Matrícula '{matricula}' não encontrada na turma.")

    aluno_idx = idx_aluno[0]
    aluno_row = df_notas.loc[aluno_idx]

    stem_dict = disciplinas_stem(disciplinas_dict)
    disciplinas_com_notas = estat.get('disciplinas_com_notas', [])
    stem_com_notas = [c for c in stem_dict if c in disciplinas_com_notas]
    stem_validas = [c for c in stem_com_notas if c in df_notas.columns]

    def _ordem_stem_chave(cod: str) -> tuple[int, int]:
        nome = (disciplinas_dict or {}).get(cod, '')
        norm = _normalizar(nome)
        if 'matematica' in norm:
            prio = 0
        elif 'fisica' in norm:
            prio = 1
        elif 'quimica' in norm:
            prio = 2
        else:
            prio = 3
        return (prio, 0)

    stem_ordenadas = sorted(stem_validas, key=_ordem_stem_chave)

    tem_faltas = (
        df_faltas is not None
        and not df_faltas.empty
        and 'matricula' in df_faltas.columns
        and estat.get('faltas_disponiveis', True)
    )

    colunas = ['codigo', 'disciplina', 'nota_aluno', 'media_turma', 'posicao', 'total_com_nota']
    if tem_faltas:
        colunas.extend(['faltas_aluno', 'media_faltas_turma', 'quadrante'])

    if not stem_ordenadas:
        return pd.DataFrame(columns=colunas)

    limiar = float(estat.get('limiar_aprovacao', 12.0))

    incompletas_raw = estat.get('disciplinas_incompletas', [])
    codigos_incompletas = set()
    for item in incompletas_raw:
        if isinstance(item, dict) and 'codigo' in item:
            codigos_incompletas.add(item['codigo'])
        elif isinstance(item, str):
            codigos_incompletas.add(item)

    if tem_faltas:
        mats_faltas = df_faltas['matricula'].astype(str).str.strip()
        idx_faltas = df_faltas.index[mats_faltas == str_matricula]
    else:
        idx_faltas = []

    linhas = []
    for c in stem_ordenadas:
        nome_disc = get_simplified_name(c, disciplinas_dict)
        if c in codigos_incompletas:
            nome_disc += ' *'

        serie_c = pd.to_numeric(df_notas[c], errors='coerce')
        serie_com_nota = serie_c.dropna()
        total_com_nota = int(len(serie_com_nota))

        media_disc_turma = float(serie_com_nota.mean()) if not serie_com_nota.empty else np.nan

        val_aluno = pd.to_numeric(pd.Series([aluno_row.get(c)]), errors='coerce').iloc[0]
        nota_aluno_disc = float(val_aluno) if pd.notna(val_aluno) else np.nan

        if pd.notna(nota_aluno_disc):
            ranks = serie_c.rank(method='min', ascending=False)
            pos_val = ranks.loc[aluno_idx]
            posicao = int(pos_val) if pd.notna(pos_val) else np.nan
        else:
            posicao = np.nan

        linha = {
            'codigo': c,
            'disciplina': nome_disc,
            'nota_aluno': nota_aluno_disc,
            'media_turma': media_disc_turma,
            'posicao': posicao,
            'total_com_nota': total_com_nota,
        }

        if tem_faltas:
            if c in df_faltas.columns:
                serie_f_c = pd.to_numeric(df_faltas[c], errors='coerce').dropna()
                media_f_turma = float(serie_f_c.mean()) if not serie_f_c.empty else 0.0
                if len(idx_faltas) > 0:
                    val_f_aluno = pd.to_numeric(pd.Series([df_faltas.loc[idx_faltas[0]].get(c)]), errors='coerce').iloc[0]
                    faltas_aluno_disc = int(val_f_aluno) if pd.notna(val_f_aluno) else 0
                else:
                    faltas_aluno_disc = 0
            else:
                media_f_turma = 0.0
                faltas_aluno_disc = 0

            quadrante = _quadrante(nota_aluno_disc, faltas_aluno_disc, limiar, media_f_turma)

            linha['faltas_aluno'] = faltas_aluno_disc
            linha['media_faltas_turma'] = media_f_turma
            linha['quadrante'] = quadrante

        linhas.append(linha)

    return pd.DataFrame(linhas, columns=colunas)


def comparar_aluno_turma(
    df_notas: pd.DataFrame,
    df_faltas: pd.DataFrame | None,
    disciplinas_dict: dict,
    estat: dict,
    matricula: Any,
) -> dict:
    """Compara o desempenho de um aluno específico com a turma.

    Valores numéricos crus (formatação fica para o PDF).
    Retorna:
        {
            'aluno': {'matricula': str, 'nome': str},
            'resumo': {
                'media_aluno': float,
                'media_turma': float,
                'posicao': int,
                'total_alunos': int,
                'disciplinas_abaixo_limiar': int,
                ['faltas_aluno': int,]
                ['media_faltas_turma': float,]
            },
            'por_disciplina': DataFrame com colunas:
                'codigo', 'disciplina', 'nota_aluno', 'media_turma', 'mediana_turma', 'diferenca'
                e opcionalmente 'faltas_aluno', 'media_faltas_turma' se houver faltas,
            'stem': DataFrame com as disciplinas STEM (E5),
            '_dados': (df_notas, df_faltas) referenciados (E8).
        }
    """
    if df_notas is None or df_notas.empty or 'matricula' not in df_notas.columns:
        raise ValueError(f"Matrícula '{matricula}' não encontrada na turma.")

    str_matricula = str(matricula).strip()
    mats_serie = df_notas['matricula'].astype(str).str.strip()
    idx_aluno = df_notas.index[mats_serie == str_matricula]
    if len(idx_aluno) == 0:
        raise ValueError(f"Matrícula '{matricula}' não encontrada na turma.")

    aluno_idx = idx_aluno[0]
    aluno_row = df_notas.loc[aluno_idx]
    nome_aluno = str(aluno_row.get('nome', '')).strip() if pd.notna(aluno_row.get('nome')) else ''

    disciplinas_com_notas = estat.get('disciplinas_com_notas', [])
    disciplinas_validas = [c for c in disciplinas_com_notas if c in df_notas.columns]

    incompletas_raw = estat.get('disciplinas_incompletas', [])
    codigos_incompletas = set()
    for item in incompletas_raw:
        if isinstance(item, dict) and 'codigo' in item:
            codigos_incompletas.add(item['codigo'])
        elif isinstance(item, str):
            codigos_incompletas.add(item)

    if disciplinas_validas:
        df_num = df_notas[disciplinas_validas].apply(pd.to_numeric, errors='coerce')
        media_por_aluno = df_num.mean(axis=1)
        ranks = media_por_aluno.rank(method='min', ascending=False)
        posicao = int(ranks.loc[aluno_idx]) if pd.notna(ranks.loc[aluno_idx]) else 1
        val_media_aluno = media_por_aluno.loc[aluno_idx]
        media_aluno = float(val_media_aluno) if pd.notna(val_media_aluno) else np.nan
    else:
        media_por_aluno = pd.Series(dtype=float)
        posicao = 1
        media_aluno = np.nan

    total_alunos = len(df_notas)
    media_turma = float(estat.get('media_geral_turma', media_por_aluno.mean() if not media_por_aluno.empty else 0.0))

    limiar = float(estat.get('limiar_aprovacao', 12.0))
    if disciplinas_validas:
        notas_aluno_serie = pd.to_numeric(aluno_row[disciplinas_validas], errors='coerce')
        disciplinas_abaixo_limiar = int((notas_aluno_serie < limiar).sum())
    else:
        disciplinas_abaixo_limiar = 0

    tem_faltas = (
        df_faltas is not None
        and not df_faltas.empty
        and 'matricula' in df_faltas.columns
    )

    if tem_faltas:
        cols_faltas = estat.get('_faltas_cols', [c for c in disciplinas_dict if c in df_faltas.columns])
        cols_faltas_validas = [c for c in cols_faltas if c in df_faltas.columns]
        mats_faltas = df_faltas['matricula'].astype(str).str.strip()
        idx_faltas = df_faltas.index[mats_faltas == str_matricula]
        if len(idx_faltas) > 0 and cols_faltas_validas:
            row_f = df_faltas.loc[idx_faltas[0]]
            faltas_aluno = int(pd.to_numeric(row_f[cols_faltas_validas], errors='coerce').fillna(0).sum())
        else:
            faltas_aluno = 0

        if cols_faltas_validas:
            df_f_num = df_faltas[cols_faltas_validas].apply(pd.to_numeric, errors='coerce').fillna(0)
            media_faltas_turma = float(estat.get('media_faltas_total', df_f_num.sum(axis=1).mean()))
        else:
            media_faltas_turma = float(estat.get('media_faltas_total', 0.0))

    resumo = {
        'media_aluno': media_aluno,
        'media_turma': media_turma,
        'posicao': posicao,
        'total_alunos': total_alunos,
        'disciplinas_abaixo_limiar': disciplinas_abaixo_limiar,
    }
    if tem_faltas:
        resumo['faltas_aluno'] = faltas_aluno
        resumo['media_faltas_turma'] = media_faltas_turma

    linhas_disc = []
    for c in disciplinas_validas:
        nome_disc = get_simplified_name(c, disciplinas_dict)
        if c in codigos_incompletas:
            nome_disc += ' *'

        serie_notas_c = pd.to_numeric(df_notas[c], errors='coerce').dropna()
        media_disc_turma = float(serie_notas_c.mean()) if not serie_notas_c.empty else np.nan
        mediana_disc_turma = float(serie_notas_c.median()) if not serie_notas_c.empty else np.nan

        val_aluno = pd.to_numeric(pd.Series([aluno_row.get(c)]), errors='coerce').iloc[0]
        nota_aluno_disc = float(val_aluno) if pd.notna(val_aluno) else np.nan

        diferenca = (nota_aluno_disc - media_disc_turma) if pd.notna(nota_aluno_disc) and pd.notna(media_disc_turma) else np.nan

        linha = {
            'codigo': c,
            'disciplina': nome_disc,
            'nota_aluno': nota_aluno_disc,
            'media_turma': media_disc_turma,
            'mediana_turma': mediana_disc_turma,
            'diferenca': diferenca,
        }

        if tem_faltas:
            if c in df_faltas.columns:
                serie_f_c = pd.to_numeric(df_faltas[c], errors='coerce').dropna()
                media_f_turma = float(serie_f_c.mean()) if not serie_f_c.empty else 0.0
                if len(idx_faltas) > 0:
                    val_f_aluno = pd.to_numeric(pd.Series([df_faltas.loc[idx_faltas[0]].get(c)]), errors='coerce').iloc[0]
                    faltas_aluno_disc = int(val_f_aluno) if pd.notna(val_f_aluno) else 0
                else:
                    faltas_aluno_disc = 0
            else:
                media_f_turma = 0.0
                faltas_aluno_disc = 0

            linha['faltas_aluno'] = faltas_aluno_disc
            linha['media_faltas_turma'] = media_f_turma

        linhas_disc.append(linha)

    colunas = ['codigo', 'disciplina', 'nota_aluno', 'media_turma', 'mediana_turma', 'diferenca']
    if tem_faltas:
        colunas.extend(['faltas_aluno', 'media_faltas_turma'])

    por_disciplina = pd.DataFrame(linhas_disc, columns=colunas)

    df_stem = comparar_stem(df_notas, df_faltas, disciplinas_dict, estat, matricula)

    return {
        'aluno': {'matricula': str_matricula, 'nome': nome_aluno},
        'resumo': resumo,
        'por_disciplina': por_disciplina,
        'stem': df_stem,
        '_dados': (df_notas, df_faltas),
    }


def evolucao_aluno_turma(grupo: list, matricula: Any) -> pd.DataFrame | None:
    """Calcula a evolução do aluno e da turma ao longo dos bimestres.

    Retorna None se o grupo tiver menos de 2 bimestres.
    Caso contrário, devolve DataFrame com colunas ['bimestre', 'pct_aluno', 'pct_turma'],
    onde as porcentagens são calculadas em relação ao MAX_PONTOS_BIMESTRE de cada bimestre.
    Se o aluno não aparecer em um bimestre, pct_aluno será np.nan.
    """
    if not grupo or len(grupo) < 2:
        return None

    grupo_ordenado = sorted(
        grupo,
        key=lambda c: (c[3] or {}).get('bimestre_num') if (c[3] and (c[3] or {}).get('bimestre_num') is not None) else 0,
    )

    str_matricula = str(matricula).strip()
    linhas = []

    for conjunto in grupo_ordenado:
        df_notas, df_faltas, disc, meta = conjunto
        meta = meta or {}
        bim = meta.get('bimestre_num')
        max_pts = MAX_PONTOS_BIMESTRE.get(bim, 20)

        estat_bim = calcular_estatisticas(df_notas, disc, df_faltas=df_faltas, metadados=meta)
        disc_com_notas = estat_bim.get('disciplinas_com_notas', [])

        media_turma = estat_bim.get('media_geral_turma')
        if media_turma is not None and pd.notna(media_turma) and max_pts > 0:
            pct_turma = (float(media_turma) / max_pts) * 100.0
        else:
            pct_turma = np.nan

        pct_aluno = np.nan
        if df_notas is not None and not df_notas.empty and 'matricula' in df_notas.columns:
            mats = df_notas['matricula'].astype(str).str.strip()
            aluno_match = df_notas[mats == str_matricula]
            if not aluno_match.empty and disc_com_notas:
                row_aluno = aluno_match.iloc[0]
                notas_serie = pd.to_numeric(row_aluno[disc_com_notas], errors='coerce')
                media_aluno = notas_serie.mean()
                if pd.notna(media_aluno) and max_pts > 0:
                    pct_aluno = (float(media_aluno) / max_pts) * 100.0

        linhas.append({
            'bimestre': bim,
            'pct_aluno': pct_aluno,
            'pct_turma': pct_turma,
        })

    return pd.DataFrame(linhas, columns=['bimestre', 'pct_aluno', 'pct_turma'])


def consolidar_bimestres(
    grupo: list,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict, dict]:
    """Consolida múltiplos bimestres em aproveitamento acumulado em % (E2–E4).

    `grupo` é uma lista de conjuntos (df_notas, df_faltas, disc_dict, metadados).
    Retorna tupla:
        (df_notas_pct, df_faltas_acum, disciplinas_dict, estat_acum)

    Regras principais:
    - Aproveitamento em % (0 a 100) por aluno e por disciplina;
    - Bimestre sem nota do aluno na disciplina fica fora do numerador e do denominador;
    - Disciplina semestral entra apenas nos bimestres em que tem nota na turma;
    - Faltas somadas com min_count=1 (tudo NaN continua NaN);
    - Turma do acumulado = alunos presentes no bimestre mais recente;
    - estat_acum mínimo com acumulado=True, limiar_aprovacao=60.0;
    - Grupo com menos de 2 bimestres lança ValueError.
    """
    if not grupo or len(grupo) < 2:
        raise ValueError("A consolidação de bimestres requer pelo menos 2 bimestres.")

    grupo_ordenado = sorted(
        grupo,
        key=lambda c: (c[3] or {}).get('bimestre_num') if (c[3] and (c[3] or {}).get('bimestre_num') is not None) else 0,
    )

    bimestres_acumulados = [
        int((c[3] or {}).get('bimestre_num'))
        for c in grupo_ordenado
        if (c[3] or {}).get('bimestre_num') is not None
    ]

    dados_bimestres = []
    disciplinas_dict = {}

    for conjunto in grupo_ordenado:
        df_notas, df_faltas, disc_dict, meta = conjunto
        meta = meta or {}
        disc_dict = disc_dict or {}
        estat_b = calcular_estatisticas(df_notas, disc_dict, df_faltas=df_faltas, metadados=meta)
        bim_num = meta.get('bimestre_num')
        max_pts = MAX_PONTOS_BIMESTRE.get(bim_num, 20)

        # Atualiza dicionário de disciplinas (o mais recente vence em conflito)
        disciplinas_dict.update(disc_dict)

        map_notas = {}
        if df_notas is not None and not df_notas.empty and 'matricula' in df_notas.columns:
            for _, row in df_notas.iterrows():
                mat = row.get('matricula')
                if pd.notna(mat):
                    m_str = str(mat).strip()
                    if m_str:
                        map_notas[m_str] = row

        map_faltas = {}
        if df_faltas is not None and not df_faltas.empty and 'matricula' in df_faltas.columns:
            for _, row in df_faltas.iterrows():
                mat = row.get('matricula')
                if pd.notna(mat):
                    m_str = str(mat).strip()
                    if m_str:
                        map_faltas[m_str] = row

        dados_bimestres.append({
            'df_notas': df_notas,
            'df_faltas': df_faltas,
            'disc_dict': disc_dict,
            'meta': meta,
            'estat': estat_b,
            'bim_num': bim_num,
            'max_pts': max_pts,
            'map_notas': map_notas,
            'map_faltas': map_faltas,
        })

    # Alunos presentes no bimestre mais recente (E3)
    mais_recente = dados_bimestres[-1]
    df_notas_recente = mais_recente['df_notas']
    if df_notas_recente is None or df_notas_recente.empty or 'matricula' not in df_notas_recente.columns:
        raise ValueError("O bimestre mais recente não possui dados de notas válidos ou coluna 'matricula'.")

    alunos_turma = []
    for _, row in df_notas_recente.iterrows():
        mat = row.get('matricula')
        if pd.isna(mat):
            continue
        m_str = str(mat).strip()
        if not m_str:
            continue
        nome = row.get('nome', '')
        nome_str = str(nome).strip() if pd.notna(nome) else ''
        alunos_turma.append({'matricula': mat, 'mat_str': m_str, 'nome': nome_str})

    # União de disciplinas com notas (na ordem de disciplinas_dict)
    todas_com_notas = set()
    for item in dados_bimestres:
        todas_com_notas.update(item['estat'].get('disciplinas_com_notas', []))

    disciplinas_com_notas = [c for c in disciplinas_dict if c in todas_com_notas]

    # União de disciplinas incompletas (incompleta em qualquer bimestre -> ' *')
    codigos_incompletas = set()
    for item in dados_bimestres:
        for inc in item['estat'].get('disciplinas_incompletas', []):
            if isinstance(inc, dict) and 'codigo' in inc:
                codigos_incompletas.add(inc['codigo'])
            elif isinstance(inc, str):
                codigos_incompletas.add(inc)

    disciplinas_incompletas = [
        {
            'codigo': c,
            'nome': get_simplified_name(c, disciplinas_dict),
        }
        for c in disciplinas_dict
        if c in codigos_incompletas
    ]

    # Monta df_notas_pct (E2)
    linhas_pct = []
    for al in alunos_turma:
        linha = {'matricula': al['matricula'], 'nome': al['nome']}
        for c in disciplinas_com_notas:
            soma_notas = 0.0
            soma_max = 0.0
            tem_nota = False
            for item in dados_bimestres:
                if c in item['estat'].get('disciplinas_com_notas', []):
                    row_b = item['map_notas'].get(al['mat_str'])
                    if row_b is not None and c in row_b:
                        val = row_b[c]
                        if pd.notna(val):
                            try:
                                n_float = float(val)
                                soma_notas += n_float
                                soma_max += item['max_pts']
                                tem_nota = True
                            except (ValueError, TypeError):
                                pass
            if tem_nota and soma_max > 0:
                linha[c] = (soma_notas / soma_max) * 100.0
            else:
                linha[c] = np.nan
        linhas_pct.append(linha)

    cols_notas = ['matricula', 'nome'] + disciplinas_com_notas
    df_notas_pct = pd.DataFrame(linhas_pct, columns=cols_notas)

    # Média geral da turma (média das médias por aluno em %)
    if disciplinas_com_notas and not df_notas_pct.empty:
        df_notas_num = df_notas_pct[disciplinas_com_notas].apply(pd.to_numeric, errors='coerce')
        medias_alunos = df_notas_num.mean(axis=1)
        media_geral_turma = float(medias_alunos.mean()) if not medias_alunos.dropna().empty else 0.0
    else:
        media_geral_turma = 0.0

    # Monta df_faltas_acum (E2)
    qualquer_falta_disp = any(
        item['estat'].get('faltas_disponiveis', False)
        for item in dados_bimestres
    )

    uniao_cols_faltas = set()
    for item in dados_bimestres:
        if item['estat'].get('faltas_disponiveis', False):
            uniao_cols_faltas.update(item['estat'].get('_faltas_cols', []))
        elif item['df_faltas'] is not None and not item['df_faltas'].empty:
            cols = [c for c in disciplinas_dict if c in item['df_faltas'].columns]
            uniao_cols_faltas.update(cols)

    _faltas_cols = [c for c in disciplinas_dict if c in uniao_cols_faltas]

    if qualquer_falta_disp and _faltas_cols:
        linhas_faltas = []
        for al in alunos_turma:
            linha_f = {'matricula': al['matricula'], 'nome': al['nome']}
            for c in _faltas_cols:
                vals_c = []
                for item in dados_bimestres:
                    df_f = item['df_faltas']
                    if df_f is not None and not df_f.empty and c in df_f.columns:
                        row_f = item['map_faltas'].get(al['mat_str'])
                        if row_f is not None and c in row_f:
                            v = row_f[c]
                            if pd.notna(v):
                                try:
                                    vals_c.append(float(v))
                                except (ValueError, TypeError):
                                    vals_c.append(np.nan)
                            else:
                                vals_c.append(np.nan)
                        else:
                            vals_c.append(np.nan)
                    else:
                        vals_c.append(np.nan)
                soma_f = pd.Series(vals_c).sum(min_count=1)
                linha_f[c] = soma_f
            linhas_faltas.append(linha_f)

        cols_f = ['matricula', 'nome'] + _faltas_cols
        df_faltas_acum = pd.DataFrame(linhas_faltas, columns=cols_f)
        df_f_num = df_faltas_acum[_faltas_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        media_faltas_total = float(df_f_num.sum(axis=1).mean()) if len(df_faltas_acum) > 0 else 0.0
        faltas_disponiveis = True
    else:
        df_faltas_acum = None
        faltas_disponiveis = False
        _faltas_cols = []
        media_faltas_total = 0.0

    meta_recente = mais_recente['meta']
    bimestre_recente = meta_recente.get('bimestre_num')

    estat_acum = {
        'limiar_aprovacao': PERCENTUAL_APROVACAO * 100.0,
        'max_pontos_bimestre': 100,
        'bimestre_num': bimestre_recente,
        'metadados': meta_recente,
        'disciplinas_com_notas': disciplinas_com_notas,
        'disciplinas_incompletas': disciplinas_incompletas,
        'media_geral_turma': media_geral_turma,
        'faltas_disponiveis': faltas_disponiveis,
        '_faltas_cols': _faltas_cols,
        'media_faltas_total': media_faltas_total,
        'acumulado': True,
        'bimestres_acumulados': bimestres_acumulados,
    }

    return df_notas_pct, df_faltas_acum, disciplinas_dict, estat_acum


def _fmt_num(val: Any, casas: int = 2, sinal: bool = False) -> str:
    """Formata valor numérico com vírgula decimal ou retorna '—' se nulo/ausente."""
    if val is None or pd.isna(val):
        return "—"
    try:
        fval = float(val)
        if sinal:
            if abs(fval) < 1e-9:
                s = f"{0.0:.{casas}f}"
            else:
                s = f"{fval:+.{casas}f}"
        else:
            s = f"{fval:.{casas}f}"
        return s.replace('.', ',')
    except (ValueError, TypeError):
        return "—"


def grafico_dispersao_stem_aluno(
    df_notas: pd.DataFrame,
    df_faltas: pd.DataFrame | None,
    estat: dict,
    codigo: str,
    nome_disciplina: str,
    matricula: Any,
) -> Any:
    """Gráfico de dispersão notas × faltas por disciplina STEM com quadrantes (E7).

    - Eixo X = faltas do aluno na disciplina;
    - Eixo Y = nota na disciplina ('Nota (0 a {max_pts})' ou 'Aproveitamento acumulado (%)');
    - Quadrantes pintados por baixo (axvspan, alpha 0.12, cores de QUADRANTES);
    - Linhas de corte: limiar de aprovação (horizontal) e média de faltas da turma na disciplina (vertical);
    - Rótulos dos quadrantes nos cantos correspondentes;
    - Colegas: pontos cinza-claro pequenos, sem nenhum texto;
    - Aluno: marcador estrela grande (marker='*', s≈350, borda preta, cor do quadrante) com o nome na legenda;
    - Devolve None sem faltas disponíveis, sem nota/falta do aluno na disciplina, ou sem a coluna.
    """
    if df_notas is None or df_notas.empty:
        return None
    if df_faltas is None or df_faltas.empty:
        return None
    if not estat.get('faltas_disponiveis', True):
        return None
    if 'matricula' not in df_notas.columns or 'matricula' not in df_faltas.columns:
        return None
    if codigo not in df_notas.columns or codigo not in df_faltas.columns:
        return None

    str_mat = str(matricula).strip()
    mats_notas = df_notas['matricula'].astype(str).str.strip()
    idx_aluno_n = df_notas.index[mats_notas == str_mat]
    if len(idx_aluno_n) == 0:
        return None

    mats_faltas = df_faltas['matricula'].astype(str).str.strip()
    idx_aluno_f = df_faltas.index[mats_faltas == str_mat]
    if len(idx_aluno_f) == 0:
        return None

    aluno_row_n = df_notas.loc[idx_aluno_n[0]]
    aluno_row_f = df_faltas.loc[idx_aluno_f[0]]

    val_nota_a = pd.to_numeric(pd.Series([aluno_row_n.get(codigo)]), errors='coerce').iloc[0]
    if pd.isna(val_nota_a):
        return None

    val_falta_a = pd.to_numeric(pd.Series([aluno_row_f.get(codigo)]), errors='coerce').iloc[0]
    if pd.isna(val_falta_a):
        return None

    nota_aluno = float(val_nota_a)
    faltas_aluno = float(val_falta_a)
    nome_aluno = str(aluno_row_n.get('nome', '')).strip() if pd.notna(aluno_row_n.get('nome')) else str_mat
    if not nome_aluno:
        nome_aluno = str_mat

    # Monta pontos válidos da turma (alunos com nota e faltas não-nulas na disciplina)
    df_n_subset = pd.DataFrame({
        'mat_str': mats_notas,
        'nota': pd.to_numeric(df_notas[codigo], errors='coerce'),
    })
    df_f_subset = pd.DataFrame({
        'mat_str': mats_faltas,
        'faltas': pd.to_numeric(df_faltas[codigo], errors='coerce'),
    })
    merged = pd.merge(df_n_subset, df_f_subset, on='mat_str', how='inner').dropna(subset=['nota', 'faltas'])
    if merged.empty or str_mat not in merged['mat_str'].values:
        return None

    limiar = float(estat.get('limiar_aprovacao', 12.0))
    serie_f_disc = pd.to_numeric(df_faltas[codigo], errors='coerce').dropna()
    media_faltas = float(serie_f_disc.mean()) if not serie_f_disc.empty else 0.0

    quadrante_aluno = _quadrante(nota_aluno, faltas_aluno, limiar, media_faltas)
    cor_aluno = QUADRANTES.get(quadrante_aluno, 'seagreen')

    # Rótulo e limites do eixo Y
    is_acumulado = bool(estat.get('acumulado'))
    if is_acumulado:
        max_pts = 100
        ylabel = "Aproveitamento acumulado (%)"
    else:
        bim = estat.get('bimestre_num')
        max_pts = int(estat.get('max_pontos_bimestre') or MAX_PONTOS_BIMESTRE.get(bim, 20))
        ylabel = f"Nota (0 a {max_pts})"

    max_f = float(merged['faltas'].max())
    x_max = max(max_f, media_faltas) * 1.15 + 1.0
    if x_max <= 0:
        x_max = 5.0

    max_nota = float(merged['nota'].max())
    y_max = max(max_nota, limiar, float(max_pts)) * 1.05

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Quadrantes pintados por baixo
    frac_limiar = limiar / y_max
    for (x0, x1), q_nome in [
        ((0, media_faltas), 'Em dia'),
        ((media_faltas, x_max), 'Risco de frequência'),
    ]:
        ax.axvspan(x0, x1, ymin=frac_limiar, ymax=1.0,
                   color=QUADRANTES[q_nome], alpha=0.12, zorder=0)

    for (x0, x1), q_nome in [
        ((0, media_faltas), 'Risco de desempenho'),
        ((media_faltas, x_max), 'Risco duplo'),
    ]:
        ax.axvspan(x0, x1, ymin=0.0, ymax=frac_limiar,
                   color=QUADRANTES[q_nome], alpha=0.12, zorder=0)

    # Linhas de corte
    ax.axhline(limiar, color='darkred', linewidth=1.2, zorder=2,
               label=f'Limiar de aprovação ({limiar:.1f})'.replace('.', ','))
    ax.axvline(media_faltas, color='darkblue', linewidth=1.2, zorder=2,
               label=f'Média de faltas da turma ({media_faltas:.1f})'.replace('.', ','))

    # Rótulos dos quadrantes nos cantos
    rotulos_canto = [
        (0.02, 0.98, 'Em dia', QUADRANTES['Em dia']),
        (0.98, 0.98, 'Risco de frequência', QUADRANTES['Risco de frequência']),
        (0.02, 0.02, 'Risco de desempenho', QUADRANTES['Risco de desempenho']),
        (0.98, 0.02, 'Risco duplo', QUADRANTES['Risco duplo']),
    ]
    for x_rel, y_rel, texto, cor in rotulos_canto:
        ha = 'left' if x_rel < 0.5 else 'right'
        va = 'top' if y_rel > 0.5 else 'bottom'
        ax.text(x_rel, y_rel, texto, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', color=cor, ha=ha, va=va, zorder=4)

    # Colegas em cinza sem texto
    colegas = merged[merged['mat_str'] != str_mat]
    if not colegas.empty:
        ax.scatter(
            colegas['faltas'],
            colegas['nota'],
            c='#a0a0a0',
            s=35,
            alpha=0.7,
            edgecolors='none',
            zorder=3,
            label='Colegas de turma',
        )

    # Aluno em estrela com cor do quadrante e nome só na legenda
    ax.scatter(
        [faltas_aluno],
        [nota_aluno],
        marker='*',
        s=350,
        c=cor_aluno,
        edgecolors='black',
        linewidths=1.0,
        zorder=5,
        label=nome_aluno,
    )

    ax.set_title(f'Dispersão Notas × Faltas — {nome_disciplina}', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Faltas na Disciplina', fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def _gerar_grafico_barras_disciplinas(
    df_disc: pd.DataFrame,
    limiar: float,
    max_pts: int,
    is_acumulado: bool = False,
) -> Any:
    """Gera gráfico de barras agrupadas: aluno × média da turma por disciplina."""
    if df_disc is None or df_disc.empty:
        return None

    labels = df_disc['disciplina'].tolist()
    notas_a = [float(v) if pd.notna(v) else np.nan for v in df_disc['nota_aluno']]
    medias_t = [float(v) if pd.notna(v) else np.nan for v in df_disc['media_turma']]

    n = len(labels)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, max(4.5, min(7.0, 0.3 * n + 2.0))))
    rotulo_aluno = 'Aproveitamento do Aluno' if is_acumulado else 'Nota do Aluno'
    ax.bar(x - width / 2, notas_a, width, label=rotulo_aluno, color='#002060')
    ax.bar(x + width / 2, medias_t, width, label='Média da Turma', color='#7092be')

    if is_acumulado:
        rotulo_limiar = f'Limiar de aprovação ({limiar:.0f}%)'
    else:
        rotulo_limiar = f'Limiar de aprovação ({limiar:.1f})'.replace('.', ',')
    ax.axhline(limiar, color='darkred', linestyle='--', linewidth=1.2, label=rotulo_limiar)

    ax.set_title('Desempenho por Disciplina — Aluno × Turma', fontsize=13, fontweight='bold', pad=10)
    if is_acumulado:
        ax.set_ylabel('Aproveitamento acumulado (%)', fontsize=10)
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel(f'Nota (0 a {max_pts})', fontsize=10)
        ax.set_ylim(0, max_pts * 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(loc='upper right', fontsize=8.5)
    fig.tight_layout()
    return fig


def _gerar_grafico_evolucao(evolucao: pd.DataFrame) -> Any:
    """Gera gráfico de linha da evolução do aproveitamento: aluno × turma."""
    if evolucao is None or evolucao.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.2))
    bimestres = evolucao['bimestre'].tolist()
    pct_aluno = [float(v) if pd.notna(v) else np.nan for v in evolucao['pct_aluno']]
    pct_turma = [float(v) if pd.notna(v) else np.nan for v in evolucao['pct_turma']]

    x_indices = list(range(len(bimestres)))
    x_labels = [f"{int(b)}º Bim" if pd.notna(b) else "" for b in bimestres]

    ax.plot(x_indices, pct_aluno, marker='o', linewidth=2.2, markersize=7,
            label='Aproveitamento do Aluno', color='#002060')
    ax.plot(x_indices, pct_turma, marker='s', linewidth=2.0, markersize=6,
            linestyle='--', label='Média da Turma', color='#7092be')

    limiar_pct = PERCENTUAL_APROVACAO * 100.0  # 60%
    ax.axhline(limiar_pct, color='darkred', linestyle=':', linewidth=1.2,
               label=f'Limiar de aprovação ({limiar_pct:.0f}%)')

    ax.set_title('Evolução do Aproveitamento — Aluno × Turma', fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Aproveitamento (% do bimestre)', fontsize=10)
    ax.set_xlabel('Bimestre', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8.5)
    fig.tight_layout()
    return fig


def criar_relatorio_aluno_pdf(
    nome_curso: str,
    comparacao: dict,
    estat: dict,
    evolucao: pd.DataFrame | None = None,
    logo_path: str | None = None,
) -> io.BytesIO:
    """Gera o relatório individual em PDF comparando o aluno com a turma.

    Estrutura segundo D3, D5 e E8:
    1. Capa enxuta / cabeçalho de texto com metadados do aluno e curso.
    2. Resumo com média aluno x média turma, posição na turma e disciplinas < limiar.
    3. Tabela por disciplina com notas, médias, medianas, diferenças e faltas opcionais.
    4. Gráfico de barras agrupadas aluno x turma com linha do limiar.
    5. Seção de Matemática, Física e Química (STEM) com tabela e dispersões (E8).
    6. Seção de evolução multibimestral (se evolucao não for None).
    7. Nota de rodapé de privacidade.
    """
    buffer = io.BytesIO()
    largura, altura = A4
    frame = Frame(2.5 * cm, 2.5 * cm, largura - 5 * cm, altura - 6.6 * cm, id='normal')
    doc = BaseDocTemplate(buffer, pagesize=A4)
    doc.addPageTemplates([
        PageTemplate(id='principal', frames=[frame], onPage=_cabecalho_factory(logo_path)),
    ])

    styles = getSampleStyleSheet()
    style_titulo = ParagraphStyle(
        name='TituloAluno',
        fontName='Times-Bold',
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#002060'),
        spaceAfter=0.3 * cm,
    )
    style_h2 = ParagraphStyle(
        name='H2Aluno',
        fontName='Times-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#002060'),
        spaceBefore=0.35 * cm,
        spaceAfter=0.15 * cm,
    )
    style_corpo = ParagraphStyle(
        name='CorpoAluno',
        fontName='Times-Roman',
        fontSize=9,
        leading=12,
    )
    style_celula = ParagraphStyle(
        name='CelulaAluno',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
    )
    style_celula_centro = ParagraphStyle(
        name='CelulaCentroAluno',
        parent=style_celula,
        alignment=TA_CENTER,
    )
    style_celula_bold = ParagraphStyle(
        name='CelulaBoldAluno',
        parent=style_celula,
        fontName='Times-Bold',
    )
    style_cab_branco = ParagraphStyle(
        name='CabBrancoAluno',
        parent=style_celula_bold,
        textColor=COR_TEXTO_CABECALHO_TABELA,
        alignment=TA_CENTER,
    )
    style_nota = ParagraphStyle(
        name='NotaAluno',
        fontName='Times-Italic',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#555555'),
        spaceBefore=0.2 * cm,
    )

    meta = estat.get('metadados', {}) or {}
    limiar = float(estat.get('limiar_aprovacao', 12.0))
    is_acumulado = bool(estat.get('acumulado'))
    bim = estat.get('bimestre_num') or meta.get('bimestre_num')
    max_pts = int(estat.get('max_pontos_bimestre') or (100 if is_acumulado else MAX_PONTOS_BIMESTRE.get(bim, 20)))
    casas_dec = 1 if is_acumulado else 2
    sufixo_pct = "%" if is_acumulado else ""

    serie = meta.get('serie')
    serie_txt = {1: '1ª Série', 2: '2ª Série', 3: '3ª Série'}.get(serie)
    if not nome_curso.lower().startswith('curso'):
        curso_txt = f"Curso Técnico em {nome_curso.title()}"
    else:
        curso_txt = nome_curso.title()
    if serie_txt and serie_txt not in curso_txt:
        curso_txt += f" — {serie_txt}"

    aluno_info = comparacao.get('aluno', {})
    nome_aluno = str(aluno_info.get('nome', '')).strip()
    mat_aluno = str(aluno_info.get('matricula', '')).strip()

    try:
        data_pt = format_date(datetime.now(), format="d 'de' MMMM 'de' y", locale='pt_BR')
    except Exception:
        data_pt = datetime.now().strftime("%d/%m/%Y")

    story = []

    # 1. Capa enxuta / cabeçalho de texto
    story.append(Paragraph("RELATÓRIO INDIVIDUAL — ALUNO × TURMA", style_titulo))

    linha_aluno = f"<b>Aluno:</b> {nome_aluno} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Matrícula:</b> {mat_aluno}"
    linha_curso = f"<b>Curso:</b> {curso_txt}"
    partes_extras = []
    if is_acumulado:
        bims = estat.get('bimestres_acumulados', [])
        if bims:
            partes_extras.append(f"Acumulado do {min(bims)}º ao {max(bims)}º Bimestre")
        else:
            partes_extras.append("Acumulado")
    elif bim:
        partes_extras.append(f"{bim}º Bimestre")
    if meta.get('periodo_letivo'):
        partes_extras.append(f"Período Letivo: {meta['periodo_letivo']}")
    if meta.get('turma'):
        partes_extras.append(f"Turma: {meta['turma']}")
    partes_extras.append(data_pt)
    linha_detalhes = " &nbsp;·&nbsp; ".join(partes_extras)

    story.append(Paragraph(linha_aluno, style_corpo))
    story.append(Paragraph(linha_curso, style_corpo))
    story.append(Paragraph(linha_detalhes, style_corpo))
    if is_acumulado:
        frase_e2 = "Aproveitamento acumulado: soma dos pontos obtidos ÷ soma dos pontos distribuídos nos bimestres enviados; limiar de 60%."
        story.append(Spacer(1, 0.1 * cm))
        story.append(Paragraph(frase_e2, style_nota))
    story.append(Spacer(1, 0.25 * cm))

    # 2. Resumo
    story.append(Paragraph("Resumo do Desempenho", style_h2))

    resumo = comparacao.get('resumo', {})
    media_a = resumo.get('media_aluno')
    media_t = resumo.get('media_turma')
    posicao = resumo.get('posicao', 1)
    total_alunos = resumo.get('total_alunos', 1)
    disc_abaixo = resumo.get('disciplinas_abaixo_limiar', 0)
    tem_faltas = 'faltas_aluno' in resumo

    if is_acumulado:
        media_a_txt = (_fmt_num(media_a, 1) + "%") if pd.notna(media_a) else "—"
        media_t_txt = (_fmt_num(media_t, 1) + "%") if pd.notna(media_t) else "—"
    else:
        media_a_txt = _fmt_num(media_a, 2)
        media_t_txt = _fmt_num(media_t, 2)

    if pd.notna(media_a) and float(media_a) < limiar:
        p_media_a = Paragraph(f'<font color="#c00000"><b>{media_a_txt}</b></font>', style_celula_centro)
    else:
        p_media_a = Paragraph(f"<b>{media_a_txt}</b>", style_celula_centro)

    p_media_t = Paragraph(media_t_txt, style_celula_centro)

    pos_txt = f"{posicao}º de {total_alunos}"
    p_pos = Paragraph(pos_txt, style_celula_centro)

    p_disc_abaixo = Paragraph(str(disc_abaixo), style_celula_centro)

    rotulo_nota_resumo = "<b>Aproveitamento acumulado (%)</b>" if is_acumulado else "<b>Média do Aluno</b>"
    rotulo_media_t_resumo = "<b>Média da Turma (%)</b>" if is_acumulado else "<b>Média da Turma</b>"
    rotulo_limiar_resumo = f"<b>Disciplinas &lt; {limiar:.0f}%</b>" if is_acumulado else f"<b>Disciplinas &lt; {limiar:.1f}</b>".replace('.', ',')
    rotulo_faltas_resumo = "<b>Faltas acumuladas</b>" if is_acumulado else "<b>Faltas do Aluno</b>"

    if tem_faltas:
        faltas_a = resumo.get('faltas_aluno', 0)
        media_f_t = resumo.get('media_faltas_turma', 0.0)
        p_faltas_a = Paragraph(str(int(round(faltas_a))), style_celula_centro)
        p_media_f_t = Paragraph(_fmt_num(media_f_t, 1 if is_acumulado else 2), style_celula_centro)

        cab_resumo = [
            Paragraph(rotulo_nota_resumo, style_cab_branco),
            Paragraph(rotulo_media_t_resumo, style_cab_branco),
            Paragraph("<b>Posição na Turma</b>", style_cab_branco),
            Paragraph(rotulo_limiar_resumo, style_cab_branco),
            Paragraph(rotulo_faltas_resumo, style_cab_branco),
            Paragraph("<b>Média Faltas Turma</b>", style_cab_branco),
        ]
        linha_resumo = [p_media_a, p_media_t, p_pos, p_disc_abaixo, p_faltas_a, p_media_f_t]
        col_w_resumo = [3.2 * cm, 2.6 * cm, 2.5 * cm, 2.5 * cm, 2.6 * cm, 2.6 * cm]
    else:
        cab_resumo = [
            Paragraph(rotulo_nota_resumo, style_cab_branco),
            Paragraph(rotulo_media_t_resumo, style_cab_branco),
            Paragraph("<b>Posição na Turma</b>", style_cab_branco),
            Paragraph(rotulo_limiar_resumo, style_cab_branco),
        ]
        linha_resumo = [p_media_a, p_media_t, p_pos, p_disc_abaixo]
        col_w_resumo = [4.6 * cm, 3.8 * cm, 3.8 * cm, 3.8 * cm]

    tabela_resumo = Table([cab_resumo, linha_resumo], colWidths=col_w_resumo, repeatRows=1)
    tabela_resumo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COR_CABECALHO_TABELA),
        ('TEXTCOLOR', (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(tabela_resumo)
    story.append(Spacer(1, 0.25 * cm))

    # 3. Tabela por disciplina
    story.append(Paragraph("Desempenho por Disciplina", style_h2))

    df_disc = comparacao.get('por_disciplina', pd.DataFrame())
    rotulo_nota_disc = "<b>Aproveitamento acumulado (%)</b>" if is_acumulado else "<b>Nota Aluno</b>"
    rotulo_media_t_disc = "<b>Média Turma (%)</b>" if is_acumulado else "<b>Média Turma</b>"
    rotulo_mediana_t_disc = "<b>Mediana Turma (%)</b>" if is_acumulado else "<b>Mediana Turma</b>"
    rotulo_dif_disc = "<b>Diferença (%)</b>" if is_acumulado else "<b>Diferença</b>"
    rotulo_faltas_disc = "<b>Faltas acumuladas</b>" if is_acumulado else "<b>Faltas Aluno</b>"

    if df_disc is not None and not df_disc.empty:
        if tem_faltas:
            cab_disc = [
                Paragraph("<b>Disciplina</b>", style_cab_branco),
                Paragraph(rotulo_nota_disc, style_cab_branco),
                Paragraph(rotulo_media_t_disc, style_cab_branco),
                Paragraph(rotulo_mediana_t_disc, style_cab_branco),
                Paragraph(rotulo_dif_disc, style_cab_branco),
                Paragraph(rotulo_faltas_disc, style_cab_branco),
                Paragraph("<b>Média Faltas</b>", style_cab_branco),
            ]
            col_w_disc = [4.4 * cm, 2.2 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm]
        else:
            cab_disc = [
                Paragraph("<b>Disciplina</b>", style_cab_branco),
                Paragraph(rotulo_nota_disc, style_cab_branco),
                Paragraph(rotulo_media_t_disc, style_cab_branco),
                Paragraph(rotulo_mediana_t_disc, style_cab_branco),
                Paragraph(rotulo_dif_disc, style_cab_branco),
            ]
            col_w_disc = [5.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]

        linhas_tabela_disc = [cab_disc]
        for _, row in df_disc.iterrows():
            p_disc = Paragraph(str(row['disciplina']), style_celula)

            val_nota = row['nota_aluno']
            if pd.isna(val_nota):
                p_nota = Paragraph("—", style_celula_centro)
            else:
                fmt_n = _fmt_num(val_nota, casas_dec) + sufixo_pct
                if float(val_nota) < limiar:
                    p_nota = Paragraph(f'<font color="#c00000"><b>{fmt_n}</b></font>', style_celula_centro)
                else:
                    p_nota = Paragraph(fmt_n, style_celula_centro)

            p_media = Paragraph(
                (_fmt_num(row['media_turma'], casas_dec) + sufixo_pct) if pd.notna(row['media_turma']) else "—",
                style_celula_centro,
            )
            p_mediana = Paragraph(
                (_fmt_num(row['mediana_turma'], casas_dec) + sufixo_pct) if pd.notna(row['mediana_turma']) else "—",
                style_celula_centro,
            )
            p_dif = Paragraph(
                (_fmt_num(row['diferenca'], casas_dec, sinal=True) + sufixo_pct) if pd.notna(row['diferenca']) else "—",
                style_celula_centro,
            )

            if tem_faltas:
                val_f = row.get('faltas_aluno')
                f_txt = str(int(round(val_f))) if pd.notna(val_f) else "0"
                p_f = Paragraph(f_txt, style_celula_centro)
                p_mf = Paragraph(_fmt_num(row.get('media_faltas_turma'), casas_dec), style_celula_centro)
                linhas_tabela_disc.append([p_disc, p_nota, p_media, p_mediana, p_dif, p_f, p_mf])
            else:
                linhas_tabela_disc.append([p_disc, p_nota, p_media, p_mediana, p_dif])

        tabela_disc = Table(linhas_tabela_disc, colWidths=col_w_disc, repeatRows=1)
        tabela_disc.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COR_CABECALHO_TABELA),
            ('TEXTCOLOR', (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(tabela_disc)

        if any(str(row['disciplina']).endswith(' *') for _, row in df_disc.iterrows()):
            story.append(Spacer(1, 0.1 * cm))
            story.append(Paragraph("* Disciplina com dados parciais ou em andamento.", style_nota))
    else:
        story.append(Paragraph("Sem notas registradas para as disciplinas neste bimestre.", style_corpo))

    # 4. Gráfico de barras agrupadas
    fig_barras = _gerar_grafico_barras_disciplinas(df_disc, limiar, max_pts, is_acumulado=is_acumulado)
    if fig_barras is not None:
        img_barras = _fig_para_imagem(fig_barras)
        plt.close(fig_barras)
        story.append(KeepTogether([
            Paragraph("Comparação Visual por Disciplina", style_h2),
            Spacer(1, 0.15 * cm),
            Image(img_barras, width=16 * cm, height=8.0 * cm, kind='proportional'),
        ]))

    # Seção Matemática, Física e Química (E8)
    df_stem = comparacao.get('stem')
    if df_stem is not None and not df_stem.empty:
        story.append(Spacer(1, 0.2 * cm))
        rotulo_nota_stem = "<b>Aproveitamento acumulado (%)</b>" if is_acumulado else "<b>Nota do Aluno</b>"
        rotulo_media_t_stem = "<b>Média Turma (%)</b>" if is_acumulado else "<b>Média Turma</b>"
        rotulo_faltas_stem = "<b>Faltas acumuladas</b>" if is_acumulado else "<b>Faltas Aluno</b>"

        tem_faltas_stem = tem_faltas and 'faltas_aluno' in df_stem.columns

        if tem_faltas_stem:
            cab_stem = [
                Paragraph("<b>Disciplina</b>", style_cab_branco),
                Paragraph(rotulo_nota_stem, style_cab_branco),
                Paragraph(rotulo_media_t_stem, style_cab_branco),
                Paragraph("<b>Posição na Disciplina</b>", style_cab_branco),
                Paragraph(rotulo_faltas_stem, style_cab_branco),
                Paragraph("<b>Média Faltas</b>", style_cab_branco),
                Paragraph("<b>Quadrante</b>", style_cab_branco),
            ]
            col_w_stem = [3.8 * cm, 2.2 * cm, 1.9 * cm, 2.3 * cm, 1.9 * cm, 1.7 * cm, 2.2 * cm]
        else:
            cab_stem = [
                Paragraph("<b>Disciplina</b>", style_cab_branco),
                Paragraph(rotulo_nota_stem, style_cab_branco),
                Paragraph(rotulo_media_t_stem, style_cab_branco),
                Paragraph("<b>Posição na Disciplina</b>", style_cab_branco),
            ]
            col_w_stem = [5.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]

        linhas_tabela_stem = [cab_stem]
        for _, row in df_stem.iterrows():
            p_disc = Paragraph(str(row['disciplina']), style_celula)

            val_nota = row.get('nota_aluno')
            if val_nota is None or pd.isna(val_nota):
                p_nota = Paragraph("—", style_celula_centro)
            else:
                fmt_n = _fmt_num(val_nota, casas_dec) + sufixo_pct
                if float(val_nota) < limiar:
                    p_nota = Paragraph(f'<font color="#c00000"><b>{fmt_n}</b></font>', style_celula_centro)
                else:
                    p_nota = Paragraph(fmt_n, style_celula_centro)

            val_media_t = row.get('media_turma')
            p_media = Paragraph(
                (_fmt_num(val_media_t, casas_dec) + sufixo_pct) if pd.notna(val_media_t) else "—",
                style_celula_centro,
            )

            pos = row.get('posicao')
            tot = row.get('total_com_nota')
            if pd.notna(pos) and pd.notna(tot):
                pos_txt = f"{int(pos)}º de {int(tot)}"
            else:
                pos_txt = "—"
            p_pos = Paragraph(pos_txt, style_celula_centro)

            if tem_faltas_stem:
                val_f = row.get('faltas_aluno')
                f_txt = str(int(round(val_f))) if pd.notna(val_f) else "0"
                p_f = Paragraph(f_txt, style_celula_centro)
                p_mf = Paragraph(_fmt_num(row.get('media_faltas_turma'), casas_dec), style_celula_centro)
                quad = row.get('quadrante')
                quad_txt = str(quad) if pd.notna(quad) and quad else "—"
                cor_quad = QUADRANTES.get(quad_txt)
                if cor_quad:
                    p_quad = Paragraph(f'<font color="{cor_quad}"><b>{quad_txt}</b></font>', style_celula_centro)
                else:
                    p_quad = Paragraph(quad_txt, style_celula_centro)
                linhas_tabela_stem.append([p_disc, p_nota, p_media, p_pos, p_f, p_mf, p_quad])
            else:
                linhas_tabela_stem.append([p_disc, p_nota, p_media, p_pos])

        tabela_stem = Table(linhas_tabela_stem, colWidths=col_w_stem, repeatRows=1)
        tabela_stem.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COR_CABECALHO_TABELA),
            ('TEXTCOLOR', (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))

        bloco_cab_stem = [
            Paragraph("Matemática, Física e Química", style_h2),
            Spacer(1, 0.15 * cm),
            tabela_stem,
        ]
        if not tem_faltas_stem:
            bloco_cab_stem.extend([
                Spacer(1, 0.15 * cm),
                Paragraph("Sem faltas lançadas: os gráficos de dispersão notas × faltas não são gerados.", style_nota),
            ])
        story.append(KeepTogether(bloco_cab_stem))

        if tem_faltas_stem:
            dados = comparacao.get('_dados')
            if dados:
                df_notas_dados, df_faltas_dados = dados
            else:
                df_notas_dados, df_faltas_dados = None, None

            for _, row in df_stem.iterrows():
                cod = str(row['codigo'])
                nome_disc = str(row['disciplina'])
                fig_disp = grafico_dispersao_stem_aluno(
                    df_notas_dados,
                    df_faltas_dados,
                    estat,
                    cod,
                    nome_disc,
                    mat_aluno,
                )
                if fig_disp is not None:
                    img_disp = _fig_para_imagem(fig_disp)
                    plt.close(fig_disp)
                    story.append(KeepTogether([
                        Paragraph(f"Dispersão Notas × Faltas — {nome_disc}", style_h2),
                        Spacer(1, 0.15 * cm),
                        Image(img_disp, width=15 * cm, height=10.3 * cm, kind='proportional'),
                    ]))

    # 5. Evolução (apenas se fornecido)
    if evolucao is not None and not evolucao.empty:
        cab_evo = [
            Paragraph("<b>Bimestre</b>", style_cab_branco),
            Paragraph("<b>Aproveitamento do Aluno (%)</b>", style_cab_branco),
            Paragraph("<b>Média da Turma (%)</b>", style_cab_branco),
        ]
        linhas_evo = [cab_evo]
        for _, row in evolucao.iterrows():
            b_num = row['bimestre']
            b_txt = f"{int(b_num)}º Bimestre" if pd.notna(b_num) else "—"
            p_b = Paragraph(b_txt, style_celula_centro)
            pct_a_val = row['pct_aluno']
            pct_t_val = row['pct_turma']
            p_pct_a = Paragraph(_fmt_num(pct_a_val, 1) + ("%" if pd.notna(pct_a_val) else ""), style_celula_centro)
            p_pct_t = Paragraph(_fmt_num(pct_t_val, 1) + ("%" if pd.notna(pct_t_val) else ""), style_celula_centro)
            linhas_evo.append([p_b, p_pct_a, p_pct_t])

        tabela_evo = Table(linhas_evo, colWidths=[4.0 * cm, 6.0 * cm, 6.0 * cm], repeatRows=1)
        tabela_evo.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COR_CABECALHO_TABELA),
            ('TEXTCOLOR', (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))

        fig_evo = _gerar_grafico_evolucao(evolucao)
        itens_evo = [
            Paragraph("Evolução ao Longo dos Bimestres", style_h2),
            Spacer(1, 0.15 * cm),
            tabela_evo,
        ]
        if fig_evo is not None:
            img_evo = _fig_para_imagem(fig_evo)
            plt.close(fig_evo)
            itens_evo.extend([
                Spacer(1, 0.2 * cm),
                Image(img_evo, width=16 * cm, height=6.2 * cm, kind='proportional'),
            ])
        story.append(KeepTogether(itens_evo))

    # 6. Nota de privacidade
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Nota de privacidade:</b> Os dados da turma são agregados; nenhum outro aluno é identificado neste relatório.",
        style_nota,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


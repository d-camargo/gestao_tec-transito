"""Módulo de comparação e relatório individual: aluno × turma.

Contratos e regras:
- D1: Não altera core/relatorios.py; importa o que for necessário de lá.
- D2: Aluno identificado por matrícula, nunca por nome. Matrícula ausente lança ValueError.
- D3: Resumo com média do aluno x média da turma, posição com rank(method='min', ascending=False),
  disciplinas abaixo do limiar, faltas opcionais; tabela por disciplina com apenas
  disciplinas_com_notas (incompletas com ' *'); evolução multibimestral em % do MAX_PONTOS_BIMESTRE.
- D5: listar_alunos, comparar_aluno_turma, evolucao_aluno_turma, criar_relatorio_aluno_pdf.
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
                e opcionalmente 'faltas_aluno', 'media_faltas_turma' se houver faltas.
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

    return {
        'aluno': {'matricula': str_matricula, 'nome': nome_aluno},
        'resumo': resumo,
        'por_disciplina': por_disciplina,
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


def _gerar_grafico_barras_disciplinas(
    df_disc: pd.DataFrame,
    limiar: float,
    max_pts: int,
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
    ax.bar(x - width / 2, notas_a, width, label='Nota do Aluno', color='#002060')
    ax.bar(x + width / 2, medias_t, width, label='Média da Turma', color='#7092be')
    ax.axhline(limiar, color='darkred', linestyle='--', linewidth=1.2,
               label=f'Limiar de aprovação ({limiar:.1f})'.replace('.', ','))

    ax.set_title('Desempenho por Disciplina — Aluno × Turma', fontsize=13, fontweight='bold', pad=10)
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

    Estrutura segundo D3 e D5:
    1. Capa enxuta / cabeçalho de texto com metadados do aluno e curso.
    2. Resumo com média aluno x média turma, posição na turma e disciplinas < limiar.
    3. Tabela por disciplina com notas, médias, medianas, diferenças e faltas opcionais.
    4. Gráfico de barras agrupadas aluno x turma com linha do limiar.
    5. Seção de evolução multibimestral (se evolucao não for None).
    6. Nota de rodapé de privacidade.
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
    bim = estat.get('bimestre_num') or meta.get('bimestre_num')
    max_pts = int(estat.get('max_pontos_bimestre') or MAX_PONTOS_BIMESTRE.get(bim, 20))

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
    if bim:
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

    media_a_txt = _fmt_num(media_a, 2)
    if pd.notna(media_a) and float(media_a) < limiar:
        p_media_a = Paragraph(f'<font color="#c00000"><b>{media_a_txt}</b></font>', style_celula_centro)
    else:
        p_media_a = Paragraph(f"<b>{media_a_txt}</b>", style_celula_centro)

    media_t_txt = _fmt_num(media_t, 2)
    p_media_t = Paragraph(media_t_txt, style_celula_centro)

    pos_txt = f"{posicao}º de {total_alunos}"
    p_pos = Paragraph(pos_txt, style_celula_centro)

    p_disc_abaixo = Paragraph(str(disc_abaixo), style_celula_centro)

    if tem_faltas:
        faltas_a = resumo.get('faltas_aluno', 0)
        media_f_t = resumo.get('media_faltas_turma', 0.0)
        p_faltas_a = Paragraph(str(int(faltas_a)), style_celula_centro)
        p_media_f_t = Paragraph(_fmt_num(media_f_t, 2), style_celula_centro)

        cab_resumo = [
            Paragraph("<b>Média do Aluno</b>", style_cab_branco),
            Paragraph("<b>Média da Turma</b>", style_cab_branco),
            Paragraph("<b>Posição na Turma</b>", style_cab_branco),
            Paragraph(f"<b>Disciplinas &lt; {limiar:.1f}</b>".replace('.', ','), style_cab_branco),
            Paragraph("<b>Faltas do Aluno</b>", style_cab_branco),
            Paragraph("<b>Média Faltas Turma</b>", style_cab_branco),
        ]
        linha_resumo = [p_media_a, p_media_t, p_pos, p_disc_abaixo, p_faltas_a, p_media_f_t]
        col_w_resumo = [2.7 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 2.6 * cm, 2.6 * cm]
    else:
        cab_resumo = [
            Paragraph("<b>Média do Aluno</b>", style_cab_branco),
            Paragraph("<b>Média da Turma</b>", style_cab_branco),
            Paragraph("<b>Posição na Turma</b>", style_cab_branco),
            Paragraph(f"<b>Disciplinas &lt; {limiar:.1f}</b>".replace('.', ','), style_cab_branco),
        ]
        linha_resumo = [p_media_a, p_media_t, p_pos, p_disc_abaixo]
        col_w_resumo = [4.0 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm]

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
    if df_disc is not None and not df_disc.empty:
        if tem_faltas:
            cab_disc = [
                Paragraph("<b>Disciplina</b>", style_cab_branco),
                Paragraph("<b>Nota Aluno</b>", style_cab_branco),
                Paragraph("<b>Média Turma</b>", style_cab_branco),
                Paragraph("<b>Mediana Turma</b>", style_cab_branco),
                Paragraph("<b>Diferença</b>", style_cab_branco),
                Paragraph("<b>Faltas Aluno</b>", style_cab_branco),
                Paragraph("<b>Média Faltas</b>", style_cab_branco),
            ]
            col_w_disc = [4.6 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm]
        else:
            cab_disc = [
                Paragraph("<b>Disciplina</b>", style_cab_branco),
                Paragraph("<b>Nota Aluno</b>", style_cab_branco),
                Paragraph("<b>Média Turma</b>", style_cab_branco),
                Paragraph("<b>Mediana Turma</b>", style_cab_branco),
                Paragraph("<b>Diferença</b>", style_cab_branco),
            ]
            col_w_disc = [6.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]

        linhas_tabela_disc = [cab_disc]
        for _, row in df_disc.iterrows():
            p_disc = Paragraph(str(row['disciplina']), style_celula)

            val_nota = row['nota_aluno']
            if pd.isna(val_nota):
                p_nota = Paragraph("—", style_celula_centro)
            else:
                fmt_n = _fmt_num(val_nota, 2)
                if float(val_nota) < limiar:
                    p_nota = Paragraph(f'<font color="#c00000"><b>{fmt_n}</b></font>', style_celula_centro)
                else:
                    p_nota = Paragraph(fmt_n, style_celula_centro)

            p_media = Paragraph(_fmt_num(row['media_turma'], 2), style_celula_centro)
            p_mediana = Paragraph(_fmt_num(row['mediana_turma'], 2), style_celula_centro)
            p_dif = Paragraph(_fmt_num(row['diferenca'], 2, sinal=True), style_celula_centro)

            if tem_faltas:
                val_f = row.get('faltas_aluno')
                f_txt = str(int(val_f)) if pd.notna(val_f) else "0"
                p_f = Paragraph(f_txt, style_celula_centro)
                p_mf = Paragraph(_fmt_num(row.get('media_faltas_turma'), 2), style_celula_centro)
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
    fig_barras = _gerar_grafico_barras_disciplinas(df_disc, limiar, max_pts)
    if fig_barras is not None:
        img_barras = _fig_para_imagem(fig_barras)
        plt.close(fig_barras)
        story.append(KeepTogether([
            Paragraph("Comparação Visual por Disciplina", style_h2),
            Spacer(1, 0.15 * cm),
            Image(img_barras, width=16 * cm, height=8.0 * cm, kind='proportional'),
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


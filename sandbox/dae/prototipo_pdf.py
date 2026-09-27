"""Geração do protótipo de relatório PDF com destaques da Assistência Estudantil (DAE).

Implementa as decisões de arquitetura:
- D6 / D10-iii: Bloco 'Período de apuração da frequência' com mês de referência e
  situação por bimestre (meses de calendário, meses lançados e selo 'parcial').
- D10-i: Rodapé com aviso de documento de uso interno em todas as páginas
  (via onFirstPage e onLaterPages).
- D10-ii: Quadro destacado de 'Tratamento de dados pessoais (LGPD)' com a nota NOTA_LGPD.
- D10-iv / D7: Seção 'Estudantes acompanhados pela Assistência Estudantil' com
  frequências por bimestre, acumulada ponderada, alerta '< 75%' destacado,
  legenda de siglas e nota sobre o pressuposto 'N/C' (Nada consta).
- D11: Organização mensal das saídas em saida/<AAAA-MM>/ (ou saida/sintetico/ para dados sintéticos).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Assegura que o diretório sandbox/dae e a raiz do projeto estejam no sys.path
_DIR_DAE = Path(__file__).resolve().parent
_DIR_RAIZ = _DIR_DAE.parent.parent
for _p in (_DIR_DAE, _DIR_RAIZ):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.relatorios import COR_CABECALHO_TABELA, COR_TEXTO_CABECALHO_TABELA

try:
    from .carregar import carregar_dae
    from .frequencia import (
        MESES_POR_BIMESTRE,
        mes_referencia,
        periodo_apuracao,
        tabela_frequencia,
    )
except ImportError:
    from carregar import carregar_dae
    from frequencia import (
        MESES_POR_BIMESTRE,
        mes_referencia,
        periodo_apuracao,
        tabela_frequencia,
    )


# D10-i: Aviso em todas as páginas
TEXTO_RODAPE_USO_INTERNO = (
    "Documento de uso interno — contém dados pessoais de estudantes (LGPD)"
)

# RASCUNHO — texto sujeito à validação do Diego (D10)
NOTA_LGPD = (
    "Tratamento de dados pessoais (LGPD): este relatório destina-se exclusivamente ao "
    "acompanhamento acadêmico e da assistência estudantil pela coordenação e demais órgãos "
    "internos do CEFET-MG (execução de políticas públicas pelo poder público — Lei nº "
    "13.709/2018, art. 7º, III, e art. 23). São exibidos apenas nome, matrícula, frequência, "
    "desempenho e vínculo com programa da Assistência Estudantil; CPF, e-mail e renda não são "
    "exibidos. Quem recebe este documento deve: não repassá-lo fora da instituição; não "
    "publicá-lo; armazená-lo em local de acesso restrito; e descartá-lo ao fim do uso."
)

# D7: Pressuposto N/C na planilha da DAE
NOTA_PE_DE_MEIA = (
    "Pé-de-Meia: N/C na planilha da DAE foi lido como ‘Nada consta’ e não marca PdM."
)

LEGENDA_PROGRAMAS = (
    "<b>Legenda das siglas:</b> <b>PdM</b>: Programa Pé-de-Meia; "
    "<b>BA/BP</b>: Bolsa Alimentação / Bolsa Permanência; "
    "<b>BCE</b>: Bolsa Complementação Educacional."
)

_MAPA_MES_NUMERO = {
    "janeiro": "01",
    "fevereiro": "02",
    "marco": "03",
    "março": "03",
    "abril": "04",
    "maio": "05",
    "junho": "06",
    "julho": "07",
    "agosto": "08",
    "setembro": "09",
    "outubro": "10",
    "novembro": "11",
    "dezembro": "12",
}


def obter_dados_sinteticos() -> pd.DataFrame:
    """Retorna DataFrame sintético representativo de estudantes da DAE."""
    dados = [
        {
            "matricula": "20261010001",
            "nome": "Ana Silva",
            "unidade": "Campus I",
            "curso": "TÉCNICO EM TRÂNSITO",
            "pe_de_meia": "elegivel",
            "bolsa_ba_bp": True,
            "bolsa_bce": False,
            "programas": ["PdM", "BA/BP"],
            "ha_ofertadas_fevereiro": 20.0, "ha_presenciadas_fevereiro": 18.0,
            "ha_ofertadas_marco": 150.0, "ha_presenciadas_marco": 120.0,
            "ha_ofertadas_abril": 140.0, "ha_presenciadas_abril": 110.0,
            "ha_ofertadas_maio": 160.0, "ha_presenciadas_maio": 130.0,
            "ha_ofertadas_junho": 130.0, "ha_presenciadas_junho": 100.0,
            "ha_ofertadas_julho": 80.0, "ha_presenciadas_julho": 70.0,
            "ha_ofertadas_agosto": 150.0, "ha_presenciadas_agosto": 140.0,
            "ha_ofertadas_setembro": np.nan, "ha_presenciadas_setembro": np.nan,
            "ha_ofertadas_outubro": np.nan, "ha_presenciadas_outubro": np.nan,
            "ha_ofertadas_novembro": np.nan, "ha_presenciadas_novembro": np.nan,
            "acumulado_dae": 0.85,
        },
        {
            "matricula": "20261010002",
            "nome": "Bruno Souza",
            "unidade": "Campus I",
            "curso": "TÉCNICO EM TRÂNSITO",
            "pe_de_meia": "nao_elegivel",
            "bolsa_ba_bp": False,
            "bolsa_bce": True,
            "programas": ["BCE"],
            "ha_ofertadas_fevereiro": 20.0, "ha_presenciadas_fevereiro": 15.0,
            "ha_ofertadas_marco": 150.0, "ha_presenciadas_marco": 100.0,
            "ha_ofertadas_abril": 140.0, "ha_presenciadas_abril": 95.0,
            "ha_ofertadas_maio": 160.0, "ha_presenciadas_maio": 110.0,
            "ha_ofertadas_junho": 130.0, "ha_presenciadas_junho": 90.0,
            "ha_ofertadas_julho": 80.0, "ha_presenciadas_julho": 55.0,
            "ha_ofertadas_agosto": 150.0, "ha_presenciadas_agosto": 105.0,
            "ha_ofertadas_setembro": np.nan, "ha_presenciadas_setembro": np.nan,
            "ha_ofertadas_outubro": np.nan, "ha_presenciadas_outubro": np.nan,
            "ha_ofertadas_novembro": np.nan, "ha_presenciadas_novembro": np.nan,
            "acumulado_dae": 0.70,
        },
        {
            "matricula": "20261010003",
            "nome": "Carlos Lima",
            "unidade": "Campus I",
            "curso": "TÉCNICO EM TRÂNSITO",
            "pe_de_meia": "nada_consta",
            "bolsa_ba_bp": False,
            "bolsa_bce": False,
            "programas": [],
            "ha_ofertadas_fevereiro": 20.0, "ha_presenciadas_fevereiro": 20.0,
            "ha_ofertadas_marco": 150.0, "ha_presenciadas_marco": 145.0,
            "ha_ofertadas_abril": 140.0, "ha_presenciadas_abril": 135.0,
            "ha_ofertadas_maio": 160.0, "ha_presenciadas_maio": 150.0,
            "ha_ofertadas_junho": 130.0, "ha_presenciadas_junho": 125.0,
            "ha_ofertadas_julho": 80.0, "ha_presenciadas_julho": 75.0,
            "ha_ofertadas_agosto": 150.0, "ha_presenciadas_agosto": 140.0,
            "ha_ofertadas_setembro": np.nan, "ha_presenciadas_setembro": np.nan,
            "ha_ofertadas_outubro": np.nan, "ha_presenciadas_outubro": np.nan,
            "ha_ofertadas_novembro": np.nan, "ha_presenciadas_novembro": np.nan,
            "acumulado_dae": 0.95,
        },
        {
            "matricula": "20261010004",
            "nome": "Daniela Rocha",
            "unidade": "Campus I",
            "curso": "TÉCNICO EM TRÂNSITO",
            "pe_de_meia": "indefinida",
            "bolsa_ba_bp": True,
            "bolsa_bce": True,
            "programas": ["BA/BP", "BCE"],
            "ha_ofertadas_fevereiro": 20.0, "ha_presenciadas_fevereiro": 12.0,
            "ha_ofertadas_marco": 150.0, "ha_presenciadas_marco": 90.0,
            "ha_ofertadas_abril": 140.0, "ha_presenciadas_abril": 84.0,
            "ha_ofertadas_maio": 160.0, "ha_presenciadas_maio": 96.0,
            "ha_ofertadas_junho": 130.0, "ha_presenciadas_junho": 78.0,
            "ha_ofertadas_julho": 80.0, "ha_presenciadas_julho": 48.0,
            "ha_ofertadas_agosto": 150.0, "ha_presenciadas_agosto": 90.0,
            "ha_ofertadas_setembro": np.nan, "ha_presenciadas_setembro": np.nan,
            "ha_ofertadas_outubro": np.nan, "ha_presenciadas_outubro": np.nan,
            "ha_ofertadas_novembro": np.nan, "ha_presenciadas_novembro": np.nan,
            "acumulado_dae": 0.60,
        },
    ]
    return pd.DataFrame(dados)


def determinar_caminho_saida(
    mes_ref: str | None,
    usando_dados_sinteticos: bool,
    caminho_saida: str | Path | None = None,
) -> Path:
    """Determina o caminho de saída para prototipo_destaque.pdf conforme Decisão D11."""
    if caminho_saida is not None:
        p = Path(caminho_saida)
        if p.suffix.lower() == ".pdf":
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        p.mkdir(parents=True, exist_ok=True)
        return p / "prototipo_destaque.pdf"

    base_saida = _DIR_DAE / "saida"
    if usando_dados_sinteticos:
        pasta = base_saida / "sintetico"
    else:
        if not mes_ref:
            raise ValueError(
                "Não foi possível determinar o mês de referência (nenhum mês com HA "
                "ofertadas > 0 na base); não é possível definir a pasta de saída <AAAA-MM>."
            )
        mes_norm = str(mes_ref).lower()
        mm = _MAPA_MES_NUMERO[mes_norm]
        pasta = base_saida / f"2026-{mm}"

    pasta.mkdir(parents=True, exist_ok=True)
    return pasta / "prototipo_destaque.pdf"


def desenhar_rodape(canvas, doc) -> None:
    """Desenha o rodapé institucional de uso interno em todas as páginas (D10-i)."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(1.75 * cm, 1.2 * cm, TEXTO_RODAPE_USO_INTERNO)
    canvas.drawRightString(
        doc.pagesize[0] - 1.75 * cm, 1.2 * cm, f"Página {canvas._pageNumber}"
    )
    canvas.restoreState()


def montar_flowables(
    df: pd.DataFrame,
    bimestres: Sequence[int] = (1, 2, 3),
    curso: str = "TÉCNICO EM TRÂNSITO",
) -> list:
    """Monta a lista de flowables do relatório PDF na ordem especificada.

    Ordem:
        (i) Bloco 'Período de apuração da frequência' (D6 / D10-iii)
        (ii) Trecho da tabela 2.1 com coluna extra 'Prog.' (siglas de programas)
        (iii) Quadro 'Tratamento de dados pessoais (LGPD)' com NOTA_LGPD (D10-ii)
        (iv) Seção 'Estudantes acompanhados pela Assistência Estudantil' com
             frequências, alerta < 75%, legenda e nota Pé-de-Meia (D10-iv).
    """
    lista_bimestres = [int(b) for b in bimestres]
    ref_mes = mes_referencia(df)
    ref_mes_nome = ref_mes.capitalize() if ref_mes else "Não identificado"
    info_periodo = periodo_apuracao(df, bimestres=lista_bimestres)
    df_freq = tabela_frequencia(df, bimestres=lista_bimestres)

    styles = getSampleStyleSheet()

    style_titulo = ParagraphStyle(
        name="DocTitulo",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=COR_CABECALHO_TABELA,
        alignment=1,  # Centro
        spaceAfter=3,
    )
    style_subtitulo = ParagraphStyle(
        name="DocSubTitulo",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#444444"),
        alignment=1,  # Centro
        spaceAfter=12,
    )
    style_h2 = ParagraphStyle(
        name="SecH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=COR_CABECALHO_TABELA,
        spaceBefore=8,
        spaceAfter=4,
    )
    style_corpo = ParagraphStyle(
        name="SecCorpo",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )
    style_caption = ParagraphStyle(
        name="SecCaption",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#444444"),
    )
    style_cab = ParagraphStyle(
        name="TabCab",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=COR_TEXTO_CABECALHO_TABELA,
        alignment=1,  # Centro
    )
    style_cel = ParagraphStyle(
        name="TabCel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.black,
    )
    style_cel_centro = ParagraphStyle(
        name="TabCelCentro",
        parent=style_cel,
        alignment=1,  # Centro
    )
    style_lgpd = ParagraphStyle(
        name="LGPDTexto",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1a202c"),
        alignment=4,  # Justificado
    )

    story: list = []

    # Cabeçalho do documento
    story.append(
        Paragraph("Acompanhamento Discente e Assistência Estudantil (DAE)", style_titulo)
    )
    story.append(
        Paragraph(
            f"Relatório Integrado — Curso: <b>{curso}</b> | Ano Letivo: 2026",
            style_subtitulo,
        )
    )

    # -------------------------------------------------------------------------
    # (i) Bloco "Período de apuração da frequência" (D6/D10-iii)
    # -------------------------------------------------------------------------
    story.append(
        Paragraph("<b>1. Período de apuração da frequência</b>", style_h2)
    )
    story.append(
        Paragraph(
            f"Mês de referência do snapshot: <b>{ref_mes_nome}/2026</b>",
            style_corpo,
        )
    )

    linhas_periodo = [[
        Paragraph("<b>Bimestre</b>", style_cab),
        Paragraph("<b>Meses do Calendário</b>", style_cab),
        Paragraph("<b>Meses Lançados</b>", style_cab),
        Paragraph("<b>Situação</b>", style_cab),
    ]]

    for b in lista_bimestres:
        dados_b = info_periodo.get(b, {})
        meses_cal = dados_b.get("meses_calendario", [])
        meses_lanc = dados_b.get("meses_lancados", [])
        parcial = dados_b.get("parcial", False)

        cal_str = ", ".join(m.capitalize() for m in meses_cal) if meses_cal else "—"
        lanc_str = ", ".join(m.capitalize() for m in meses_lanc) if meses_lanc else "Nenhum"

        if parcial:
            sit_p = Paragraph(
                "<b><font color='#c00000'>Parcial</font></b>", style_cel_centro
            )
        else:
            sit_p = Paragraph("Completo", style_cel_centro)

        linhas_periodo.append([
            Paragraph(f"<b>{b}º Bimestre</b>", style_cel_centro),
            Paragraph(cal_str, style_cel),
            Paragraph(lanc_str, style_cel),
            sit_p,
        ])

    tab_periodo = Table(
        linhas_periodo,
        colWidths=[2.8 * cm, 6.0 * cm, 5.7 * cm, 3.0 * cm],
    )
    tab_periodo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
        ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tab_periodo)
    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # (ii) Trecho da tabela 2.1 com a coluna extra “Prog.” (siglas de programas)
    # -------------------------------------------------------------------------
    story.append(
        Paragraph("<b>2. Desempenho e Frequência por Aluno (Trecho Tabela 2.1 com Programas)</b>", style_h2)
    )
    story.append(
        Paragraph(
            "Demonstração da inclusão da coluna <b>Prog.</b> (siglas dos programas DAE) "
            "ao lado do nome do estudante na tabela consolidada de desempenho e faltas por disciplina.",
            style_caption,
        )
    )
    story.append(Spacer(1, 0.1 * cm))

    linhas_21 = [[
        Paragraph("<b>Aluno</b>", style_cab),
        Paragraph("<b>Prog.</b>", style_cab),
        Paragraph("<b>D1 (Português)</b>", style_cab),
        Paragraph("<b>D2 (Matemática)</b>", style_cab),
        Paragraph("<b>D3 (Trânsito)</b>", style_cab),
        Paragraph("<b>Média</b>", style_cab),
        Paragraph("<b>Faltas</b>", style_cab),
    ]]

    # Valores ilustrativos realistas de desempenho para cada estudante
    amostras_desempenho = [
        ("16,0 / 2", "18,5 / 0", "15,0 / 4", "16,5", "6"),
        ("11,0 / 8", "12,0 / 6", "10,5 / 10", "11,2", "24"),
        ("19,0 / 0", "18,0 / 1", "19,5 / 0", "18,8", "1"),
        ("9,5 / 12", "10,0 / 14", "11,0 / 8", "10,2", "34"),
    ]

    for idx, (_, row) in enumerate(df.iterrows()):
        nome = str(row["nome"])
        progs_list = row.get("programas", [])
        if isinstance(progs_list, list) and progs_list:
            progs_str = ", ".join(progs_list)
        else:
            progs_str = "—"

        d1, d2, d3, med, fal = amostras_desempenho[idx % len(amostras_desempenho)]

        linhas_21.append([
            Paragraph(nome, style_cel),
            Paragraph(f"<b>{progs_str}</b>", style_cel_centro),
            Paragraph(d1, style_cel_centro),
            Paragraph(d2, style_cel_centro),
            Paragraph(d3, style_cel_centro),
            Paragraph(med, style_cel_centro),
            Paragraph(fal, style_cel_centro),
        ])

    tab_21 = Table(
        linhas_21,
        colWidths=[4.6 * cm, 2.5 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.8 * cm, 1.7 * cm],
    )
    tab_21.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
        ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tab_21)
    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # (iii) Quadro “Tratamento de dados pessoais (LGPD)” com NOTA_LGPD (D10-ii)
    # -------------------------------------------------------------------------
    story.append(
        Paragraph("<b>3. Tratamento de dados pessoais (LGPD)</b>", style_h2)
    )

    quadro_lgpd_conteudo = [
        Paragraph(
            f"<b>Quadro de Proteção de Dados:</b><br/>{NOTA_LGPD}",
            style_lgpd,
        )
    ]
    tab_quadro_lgpd = Table([[quadro_lgpd_conteudo]], colWidths=[17.5 * cm])
    tab_quadro_lgpd.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
        ("BOX", (0, 0), (-1, -1), 0.75, COR_CABECALHO_TABELA),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(tab_quadro_lgpd)
    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # (iv) Seção “Estudantes acompanhados pela Assistência Estudantil” (D10-iv)
    # -------------------------------------------------------------------------
    story.append(
        Paragraph("<b>4. Estudantes acompanhados pela Assistência Estudantil</b>", style_h2)
    )

    cab_estudantes = [
        Paragraph("<b>Nome do Estudante</b>", style_cab),
        Paragraph("<b>Programas</b>", style_cab),
    ]
    for b in lista_bimestres:
        cab_estudantes.append(Paragraph(f"<b>Freq. {b}º Bim.</b>", style_cab))
    cab_estudantes.append(Paragraph("<b>Freq. Acum.</b>", style_cab))
    cab_estudantes.append(Paragraph("<b>Alerta (&lt; 75%)</b>", style_cab))

    linhas_estudantes = [cab_estudantes]
    estilos_tabela_est = [
        ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
        ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]

    # D8: listar somente estudantes com programa da Assistência Estudantil
    df_freq = df_freq[df_freq["programas"].apply(lambda p: isinstance(p, list) and len(p) > 0)]

    for idx, (_, row) in enumerate(df_freq.iterrows(), start=1):
        nome = str(row["nome"])
        progs_list = row.get("programas", [])
        if isinstance(progs_list, list) and progs_list:
            progs_str = ", ".join(progs_list)
        else:
            progs_str = "—"

        linha_est = [
            Paragraph(nome, style_cel),
            Paragraph(f"<b>{progs_str}</b>", style_cel_centro),
        ]

        for b in lista_bimestres:
            val_b = row.get(f"freq_bim_{b}")
            if pd.isna(val_b):
                linha_est.append(Paragraph("—", style_cel_centro))
            else:
                linha_est.append(Paragraph(f"{val_b:.1%}", style_cel_centro))

        freq_acum = row.get("freq_acumulada")
        abaixo_75 = bool(row.get("abaixo_75", False))

        if pd.isna(freq_acum):
            linha_est.append(Paragraph("—", style_cel_centro))
            linha_est.append(Paragraph("—", style_cel_centro))
        elif abaixo_75:
            linha_est.append(
                Paragraph(f"<b><font color='#c00000'>{freq_acum:.1%}</font></b>", style_cel_centro)
            )
            linha_est.append(
                Paragraph("<b><font color='#c00000'>&lt; 75%</font></b>", style_cel_centro)
            )
            # Destaque de alerta na linha do aluno com presença abaixo do limiar
            estilos_tabela_est.append(
                ("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#fff2f2"))
            )
        else:
            linha_est.append(Paragraph(f"{freq_acum:.1%}", style_cel_centro))
            linha_est.append(Paragraph("Regular", style_cel_centro))

        linhas_estudantes.append(linha_est)

    # Cálculo dinâmico das larguras de coluna
    n_bims = len(lista_bimestres)
    larg_nome = 4.6 * cm
    larg_progs = 2.5 * cm
    larg_acum = 2.2 * cm
    larg_alerta = 2.2 * cm
    larg_restante = 17.5 * cm - (larg_nome + larg_progs + larg_acum + larg_alerta)
    larg_bim = (larg_restante / n_bims) if n_bims > 0 else 2.0 * cm

    col_widths_est = [larg_nome, larg_progs] + [larg_bim] * n_bims + [larg_acum, larg_alerta]

    tab_estudantes = Table(linhas_estudantes, colWidths=col_widths_est)
    tab_estudantes.setStyle(TableStyle(estilos_tabela_est))

    bloco_iv = [
        tab_estudantes,
        Spacer(1, 0.2 * cm),
        Paragraph(LEGENDA_PROGRAMAS, style_caption),
        Spacer(1, 0.1 * cm),
        Paragraph(f"<b>Observação:</b> {NOTA_PE_DE_MEIA}", style_caption),
    ]
    story.append(KeepTogether(bloco_iv))

    return story


def extrair_texto_flowables(flowables: list, incluir_rodape: bool = True) -> str:
    """Extrai texto dos flowables e rodapé para testes e validação das strings."""
    partes: list[str] = []
    if incluir_rodape:
        partes.append(TEXTO_RODAPE_USO_INTERNO)

    for item in flowables:
        if isinstance(item, Paragraph):
            partes.append(item.text)
        elif isinstance(item, Table):
            for row in item._cellvalues:
                for cell in row:
                    if isinstance(cell, Paragraph):
                        partes.append(cell.text)
                    elif isinstance(cell, (list, tuple)):
                        for sub in cell:
                            if isinstance(sub, Paragraph):
                                partes.append(sub.text)
                            elif isinstance(sub, str):
                                partes.append(sub)
                    elif isinstance(cell, str):
                        partes.append(cell)
        elif isinstance(item, KeepTogether):
            for sub_elem in item._content:
                if isinstance(sub_elem, Paragraph):
                    partes.append(sub_elem.text)
                elif isinstance(sub_elem, Table):
                    for row in sub_elem._cellvalues:
                        for cell in row:
                            if isinstance(cell, Paragraph):
                                partes.append(cell.text)
                            elif isinstance(cell, str):
                                partes.append(cell)

    return "\n".join(partes)


def gerar_prototipo_pdf(
    caminho_dae: str | Path | None = None,
    curso: str = "TÉCNICO EM TRÂNSITO",
    bimestres: Sequence[int] | str = (1, 2, 3),
    caminho_saida: str | Path | None = None,
) -> Path:
    """Gera o protótipo de destaque em PDF com os dados da DAE (ou sintéticos)."""
    # 1. Obtenção dos dados
    usando_sintetico = caminho_dae is None
    if usando_sintetico:
        df = obter_dados_sinteticos()
    else:
        df = carregar_dae(caminho_dae)

    # 2. Filtragem pelo curso se a coluna existir e houver dados correspondentes
    if curso and "curso" in df.columns:
        mask_curso = df["curso"].str.upper() == curso.upper()
        if mask_curso.any():
            df = df[mask_curso].reset_index(drop=True)

    # 3. Normalização dos bimestres
    if isinstance(bimestres, str):
        lista_bimestres = [int(b.strip()) for b in bimestres.split(",") if b.strip()]
    else:
        lista_bimestres = [int(b) for b in bimestres]

    # 4. Determinação do caminho de saída (D11)
    ref = mes_referencia(df)
    caminho_pdf = determinar_caminho_saida(
        mes_ref=ref,
        usando_dados_sinteticos=usando_sintetico,
        caminho_saida=caminho_saida,
    )

    # 5. Montagem dos flowables
    story = montar_flowables(df, bimestres=lista_bimestres, curso=curso)

    # 6. Geração do PDF com SimpleDocTemplate e rodapé em todas as páginas (D10-i)
    doc = SimpleDocTemplate(
        str(caminho_pdf),
        pagesize=A4,
        leftMargin=1.75 * cm,
        rightMargin=1.75 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2.0 * cm,
    )
    doc.build(story, onFirstPage=desenhar_rodape, onLaterPages=desenhar_rodape)

    return caminho_pdf


def _criar_parser() -> argparse.ArgumentParser:
    """Cria e configura o parser de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera protótipo de relatório PDF com destaques da Assistência Estudantil (DAE)."
    )
    parser.add_argument(
        "--dae",
        dest="dae",
        type=str,
        default=None,
        help="Caminho para o arquivo da DAE (.xlsx ou .csv). Padrão: dados sintéticos.",
    )
    parser.add_argument(
        "--curso",
        dest="curso",
        type=str,
        default="TÉCNICO EM TRÂNSITO",
        help="Curso para filtragem (padrão: 'TÉCNICO EM TRÂNSITO').",
    )
    parser.add_argument(
        "--bimestres",
        dest="bimestres",
        type=str,
        default="1,2,3",
        help="Bimestres a analisar separados por vírgula (padrão: '1,2,3').",
    )
    parser.add_argument(
        "--saida",
        dest="saida",
        type=str,
        default=None,
        help="Caminho de saída personalizado para o arquivo PDF.",
    )
    return parser


def _executar_cli(argv: list[str] | None = None) -> int:
    """Função de entrada para execução via CLI."""
    parser = _criar_parser()
    args = parser.parse_args(argv)

    try:
        lista_bimestres = [int(b.strip()) for b in args.bimestres.split(",") if b.strip()]
    except ValueError:
        print(
            f"Erro: lista de bimestres inválida: '{args.bimestres}'. Use formato ex.: '1,2,3'",
            file=sys.stderr,
        )
        return 1

    pdf_gerado = gerar_prototipo_pdf(
        caminho_dae=args.dae,
        curso=args.curso,
        bimestres=lista_bimestres,
        caminho_saida=args.saida,
    )
    print(f"Protótipo PDF gerado com sucesso em: {pdf_gerado}")
    return 0


if __name__ == "__main__":
    sys.exit(_executar_cli())

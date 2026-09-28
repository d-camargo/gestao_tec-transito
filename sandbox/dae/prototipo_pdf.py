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
- C14: Bloco 'Cruzamento com o mapa de turma' com métricas do mapa (resumo_mapa),
  conciliação por bimestre (cruzar), recorte Pé-de-Meia (contagem_pe_de_meia) ou
  aviso de pendência quando DAE for sintética.
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

PASTA_DADOS: Path = _DIR_DAE / "dados"

from core.manipulacao import processar_multiplos_bimestres
from core.relatorios import COR_CABECALHO_TABELA, COR_TEXTO_CABECALHO_TABELA

try:
    from .calendario import (
        ANO_PADRAO,
        CAMINHO_MD_PADRAO,
        Calendario,
        carregar_calendario,
        ch_nominal,
        faixa_ch,
        sabados_do_responsavel,
    )
    from .carregar import _normalizar_matricula, carregar_dae
    from .ch_efetiva import (
        carregar_ch_efetiva,
        casar_disciplinas,
        ch_da_turma,
        divergencias_calendario,
        normalizar_disciplina,
        nota_sabados,
        remover_acentos,
        resumo_por_carga,
    )
    from .cruzamento import contagem_pe_de_meia, cruzar, resumo_mapa
    from .det import (
        CURSOS_DET,
        ROTULO_DET,
        carregar_det,
        classificar_mapas,
        conjuntos_det,
        eh_det,
        resumo_det,
        resumo_frequencia_det,
    )
    from .frequencia import (
        MESES_POR_BIMESTRE,
        limite_faltas,
        mes_referencia,
        periodo_apuracao,
        resumo_frequencia_por_disciplina,
        tabela_frequencia,
    )
except ImportError:
    from calendario import (
        ANO_PADRAO,
        CAMINHO_MD_PADRAO,
        Calendario,
        carregar_calendario,
        ch_nominal,
        faixa_ch,
        sabados_do_responsavel,
    )
    from carregar import _normalizar_matricula, carregar_dae
    from ch_efetiva import (
        carregar_ch_efetiva,
        casar_disciplinas,
        ch_da_turma,
        divergencias_calendario,
        normalizar_disciplina,
        nota_sabados,
        remover_acentos,
        resumo_por_carga,
    )
    from cruzamento import contagem_pe_de_meia, cruzar, resumo_mapa
    from det import (
        CURSOS_DET,
        ROTULO_DET,
        carregar_det,
        classificar_mapas,
        conjuntos_det,
        eh_det,
        resumo_det,
        resumo_frequencia_det,
    )
    from frequencia import (
        MESES_POR_BIMESTRE,
        limite_faltas,
        mes_referencia,
        periodo_apuracao,
        resumo_frequencia_por_disciplina,
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

# D8: Quadro de demonstração quando os dados da DAE são sintéticos na prévia
TEXTO_DEMONSTRACAO = (
    "DEMONSTRAÇÃO — os dados individuais da Assistência Estudantil desta seção são "
    "FICTÍCIOS (export da DAE ainda não recebido). Os agregados do mapa de turma, do "
    "calendário e da CH efetiva são reais."
)


def _largura(cm_prototipo: float | Sequence[float], layout: str = "app") -> Any:
    """Escala largura(s) de colunas do protótipo para 16 cm úteis quando layout='app' (D6)."""
    fator = (16.0 / 17.5) if layout == "app" else 1.0
    if isinstance(cm_prototipo, (list, tuple)):
        return [w * fator for w in cm_prototipo]
    return cm_prototipo * fator


_escalar_largura = _largura


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
    df = pd.DataFrame(dados)
    df.attrs["sintetico"] = True
    return df


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


def _resolver_cruzamento_det(cruzamento: Any) -> tuple[bool, Any]:
    """Identifica e normaliza cruzamento para o DET, se aplicável.

    Retorna (True, conjuntos_det) se for DET, ou (False, None) caso contrário.
    """
    if cruzamento is None:
        return False, None
    if isinstance(cruzamento, conjuntos_det):
        return True, cruzamento
    if isinstance(cruzamento, dict):
        chaves_norm = {remover_acentos(str(k)).lower().strip(): v for k, v in cruzamento.items()}
        if "estradas" in chaves_norm and "transito" in chaves_norm:
            val_est = chaves_norm["estradas"]
            val_tt = chaves_norm["transito"]
            if (
                val_est and isinstance(val_est, (list, tuple))
                and isinstance(val_est[0], (list, tuple)) and len(val_est[0]) >= 4
            ) or (
                val_tt and isinstance(val_tt, (list, tuple))
                and isinstance(val_tt[0], (list, tuple)) and len(val_tt[0]) >= 4
            ):
                return True, conjuntos_det(val_tt, val_est)
            return True, carregar_det(cruzamento)
    if isinstance(cruzamento, (list, tuple)):
        if (
            len(cruzamento) == 2
            and isinstance(cruzamento[0], (list, tuple))
            and isinstance(cruzamento[1], (list, tuple))
            and (
                (cruzamento[0] and isinstance(cruzamento[0][0], (list, tuple)) and len(cruzamento[0][0]) >= 4)
                or (cruzamento[1] and isinstance(cruzamento[1][0], (list, tuple)) and len(cruzamento[1][0]) >= 4)
            )
        ):
            return True, conjuntos_det(cruzamento[0], cruzamento[1])
        try:
            classif = classificar_mapas(cruzamento)
            if eh_det(classif):
                return True, carregar_det(classif)
        except Exception:
            pass
    return False, None


def montar_flowables(
    df: pd.DataFrame,
    bimestres: Sequence[int] = (1, 2, 3),
    curso: str = "TÉCNICO EM TRÂNSITO",
    cenario: str = "A",
    calendario: Calendario | Path | str | None = None,
    sabado_reproduz: str | None = None,
    cruzamento: Sequence[str | Path] | Sequence[object] | dict | None = None,
    ch_efetiva: str | Path | pd.DataFrame | None = None,
    layout: str = "prototipo",
) -> list:
    """Monta a lista de flowables do relatório PDF na ordem especificada.

    Ordem:
        (i) Bloco 'Período de apuração da frequência' (D6 / D10-iii / C7)
        (ii) Bloco 'Calendário acadêmico e carga horária efetiva' (C7)
        (iii) Trecho da tabela 2.1 com coluna extra 'Prog.' (siglas de programas)
        (iv) Quadro 'Tratamento de dados pessoais (LGPD)' com NOTA_LGPD (D10-ii)
        (v) Seção 'Estudantes acompanhados pela Assistência Estudantil' com
             frequências, alerta < 75%, legenda e nota Pé-de-Meia (D10-iv).
        (vi) Seção 'Cruzamento com o mapa de turma' (C14) quando cruzamento for fornecido.
        (vii) Bloco 'CH efetiva por disciplina (horário real)' (C18).
    """
    layout_norm = (layout or "prototipo").lower().strip()
    if layout_norm not in ("prototipo", "app"):
        raise ValueError(f"Layout inválido: '{layout}'. Escolha entre 'prototipo' e 'app'.")

    # Helper único para escalar colWidths para 16 cm úteis no frame do app (D6)
    def _largura(w: float | Sequence[float]) -> Any:
        return _escalar_largura(w, layout=layout_norm)

    if isinstance(calendario, Calendario):
        cal = calendario
    else:
        cal = carregar_calendario(caminho=calendario)

    cenario_norm = (cenario or "A").upper().strip()
    if cenario_norm not in ("A", "REAL"):
        raise ValueError(f"Cenário inválido: '{cenario}'. Use 'A' ou 'REAL'.")

    sab_rep_norm = sabado_reproduz.upper().strip() if sabado_reproduz else None
    if sab_rep_norm is not None and sab_rep_norm not in ("SEG", "TER", "QUA", "QUI", "SEX"):
        raise ValueError(
            f"sabado_reproduz inválido: '{sabado_reproduz}'. Escolha entre SEG, TER, QUA, QUI, SEX."
        )

    lista_bimestres = [int(b) for b in bimestres]
    ref_mes = mes_referencia(df)
    ref_mes_nome = ref_mes.capitalize() if ref_mes else "Não identificado"
    info_periodo = periodo_apuracao(
        df,
        bimestres=lista_bimestres,
        calendario=cal,
        cenario=cenario_norm,
        curso=curso,
    )
    df_freq = tabela_frequencia(df, bimestres=lista_bimestres)

    styles = getSampleStyleSheet()
    if layout_norm == "app":
        styles["Normal"].fontName = "Times-Roman"

    font_bold = "Times-Bold" if layout_norm == "app" else "Helvetica-Bold"
    font_regular = "Times-Roman" if layout_norm == "app" else "Helvetica"
    font_italic = "Times-Italic" if layout_norm == "app" else "Helvetica"

    style_titulo = ParagraphStyle(
        name="DocTitulo",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=15,
        leading=18,
        textColor=COR_CABECALHO_TABELA,
        alignment=1,  # Centro
        spaceAfter=3,
    )
    style_subtitulo = ParagraphStyle(
        name="DocSubTitulo",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#444444"),
        alignment=1,  # Centro
        spaceAfter=12,
    )
    style_h2 = ParagraphStyle(
        name="H2Sumario" if layout_norm == "app" else "SecH2",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=10.5,
        leading=13,
        textColor=COR_CABECALHO_TABELA,
        spaceBefore=8,
        spaceAfter=4,
    )
    style_corpo = ParagraphStyle(
        name="SecCorpo",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )
    style_caption = ParagraphStyle(
        name="SecCaption",
        parent=styles["Normal"],
        fontName=font_italic,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#444444"),
    )
    style_cab = ParagraphStyle(
        name="TabCab",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=8,
        leading=10,
        textColor=COR_TEXTO_CABECALHO_TABELA,
        alignment=1,  # Centro
    )
    style_cel = ParagraphStyle(
        name="TabCel",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.black,
    )
    style_cel_centro = ParagraphStyle(
        name="TabCelCentro",
        parent=style_cel,
        fontName=font_regular,
        alignment=1,  # Centro
    )
    style_lgpd = ParagraphStyle(
        name="LGPDTexto",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1a202c"),
        alignment=4,  # Justificado
    )

    story: list = []

    # Cabeçalho do documento (somente no layout protótipo)
    if layout_norm == "prototipo":
        story.append(
            Paragraph("Acompanhamento Discente e Assistência Estudantil (DAE)", style_titulo)
        )
        story.append(
            Paragraph(
                f"Relatório Integrado — Curso: <b>{curso}</b> | Ano Letivo: {cal.ano} | Cenário: <b>{cenario_norm}</b>",
                style_subtitulo,
            )
        )

    dae_sintetica = bool(df.attrs.get("sintetico", False))

    # D8: Com DAE sintética e layout="app", o primeiro flowable é o quadro "DEMONSTRAÇÃO"
    if layout_norm == "app" and dae_sintetica:
        quadro_dem = [
            Paragraph(TEXTO_DEMONSTRACAO, style_corpo)
        ]
        tab_dem = Table([[quadro_dem]], colWidths=_largura([17.5 * cm]))
        tab_dem.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9e6")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d9822b")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(tab_dem)
        story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # (i) Bloco "Período de apuração da frequência" (D6/D10-iii/C7)
    # -------------------------------------------------------------------------
    titulo_1 = (
        "Período de apuração da frequência"
        if layout_norm == "app"
        else "<b>1. Período de apuração da frequência</b>"
    )
    story.append(Paragraph(titulo_1, style_h2))
    story.append(
        Paragraph(
            f"Mês de referência do snapshot: <b>{ref_mes_nome}/{cal.ano}</b>",
            style_corpo,
        )
    )

    linhas_periodo = [[
        Paragraph("<b>Bimestre</b>", style_cab),
        Paragraph("<b>Meses do Calendário</b>", style_cab),
        Paragraph("<b>Meses Lançados</b>", style_cab),
        Paragraph(f"<b>Dias letivos (cenário {cenario_norm})</b>", style_cab),
        Paragraph("<b>Diários até</b>", style_cab),
        Paragraph("<b>Situação</b>", style_cab),
    ]]

    for b in lista_bimestres:
        dados_b = info_periodo.get(b, {})
        meses_cal = dados_b.get("meses_calendario", [])
        meses_lanc = dados_b.get("meses_lancados", [])
        parcial = dados_b.get("parcial", False)
        dias_let = dados_b.get("dias_letivos", "—")
        limite_diarios = dados_b.get("limite_diarios", "—")
        dias_m = dados_b.get("dias_por_mes", {})

        if dias_m:
            cal_partes = [f"{m.capitalize()} ({qtd} d)" for m, qtd in dias_m.items()]
            cal_str = ", ".join(cal_partes)
        else:
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
            Paragraph(str(dias_let), style_cel_centro),
            Paragraph(str(limite_diarios), style_cel_centro),
            sit_p,
        ])

    tab_periodo = Table(
        linhas_periodo,
        colWidths=_largura([2.0 * cm, 5.2 * cm, 3.4 * cm, 2.7 * cm, 2.0 * cm, 2.2 * cm]),
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
    # (ii) Bloco "Calendário acadêmico e carga horária efetiva" (C7)
    # -------------------------------------------------------------------------
    titulo_2 = (
        "Calendário acadêmico e carga horária efetiva"
        if layout_norm == "app"
        else "<b>2. Calendário acadêmico e carga horária efetiva</b>"
    )
    story.append(Paragraph(titulo_2, style_h2))

    cal_arquivo_nome = (
        Path(calendario).name
        if (calendario is not None and not isinstance(calendario, Calendario))
        else CAMINHO_MD_PADRAO.name
    )
    story.append(
        Paragraph(
            f"<b>Fonte:</b> {cal.titulo} (arquivo <code>{cal_arquivo_nome}</code>), "
            f"homologado pela <b>{cal.deliberacao}</b>.",
            style_corpo,
        )
    )

    sabs_curso_info = []
    if curso:
        for b_num in sorted(cal.bimestres.keys()):
            for s_data in sabados_do_responsavel(cal, responsavel=curso, bimestres=[b_num]):
                sabs_curso_info.append((s_data, b_num))

    if cenario_norm == "REAL":
        if sabs_curso_info:
            sabs_str = ", ".join(
                f"{d.strftime('%d/%m')} ({b}º BI)" for d, b in sabs_curso_info
            )
            txt_sabs = f" Sábados letivos atribuídos ao curso ({curso}): <b>{sabs_str}</b>."
        else:
            txt_sabs = f" Nenhum sábado letivo específico atribuído ao curso ({curso})."

        if sab_rep_norm:
            txt_rep = f" O sábado letivo reproduz a grade horária de <b>{sab_rep_norm}</b>."
        else:
            txt_rep = " Horário de dia útil reproduzido pelos sábados apurado em faixa."

        texto_cenario = (
            f"Apuração sob o <b>cenário REAL</b>: considera as aulas regulares em dias úteis "
            f"somadas aos sábados letivos sob responsabilidade da coordenação do curso.{txt_sabs}{txt_rep} "
            f"Sábados de áreas acadêmicas específicas não entram no REAL por ausência de "
            f"mapeamento determinístico entre disciplina e área."
        )
    else:
        texto_cenario = (
            "Apuração sob o <b>cenário A</b> (padrão institucional / piso): considera apenas as aulas "
            "em dias úteis regulares (segunda a sexta-feira), desconsiderando sábados letivos. É o "
            "cenário conservador oficial para monitoramento de frequência e alertas precoces (&lt; 75%). "
            "O <b>cenário REAL</b> incorpora os sábados letivos temáticos atribuídos à coordenação do curso."
        )
    story.append(Paragraph(texto_cenario, style_corpo))

    # Tabela dias letivos bimestre × dia útil do cenário ativo
    incluir_col_sab = (cenario_norm == "REAL" and sab_rep_norm is None)
    if incluir_col_sab:
        cab_sem = [
            Paragraph("<b>Bimestre</b>", style_cab),
            Paragraph("<b>SEG</b>", style_cab),
            Paragraph("<b>TER</b>", style_cab),
            Paragraph("<b>QUA</b>", style_cab),
            Paragraph("<b>QUI</b>", style_cab),
            Paragraph("<b>SEX</b>", style_cab),
            Paragraph("<b>SÁB (Curso)</b>", style_cab),
            Paragraph("<b>Total</b>", style_cab),
        ]
        col_w_sem = [2.8 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm]
    else:
        cab_sem = [
            Paragraph("<b>Bimestre</b>", style_cab),
            Paragraph("<b>SEG</b>", style_cab),
            Paragraph("<b>TER</b>", style_cab),
            Paragraph("<b>QUA</b>", style_cab),
            Paragraph("<b>QUI</b>", style_cab),
            Paragraph("<b>SEX</b>", style_cab),
            Paragraph("<b>Total</b>", style_cab),
        ]
        col_w_sem = [3.5 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm]

    linhas_sem = [cab_sem]
    somas_col = {"SEG": 0, "TER": 0, "QUA": 0, "QUI": 0, "SEX": 0, "SAB": 0, "Total": 0}

    for b_num in (1, 2, 3, 4):
        bim_obj = cal.bimestres.get(b_num)
        if not bim_obj:
            continue
        d_sem = {d: bim_obj.dias_semana.get(d, 0) for d in ("SEG", "TER", "QUA", "QUI", "SEX")}
        n_sab_curso = len(sabados_do_responsavel(cal, responsavel=curso, bimestres=[b_num])) if (curso and cenario_norm == "REAL") else 0

        if cenario_norm == "REAL" and sab_rep_norm:
            d_sem[sab_rep_norm] += n_sab_curso

        tot_b = sum(d_sem.values()) + (n_sab_curso if incluir_col_sab else 0)

        for d in ("SEG", "TER", "QUA", "QUI", "SEX"):
            somas_col[d] += d_sem[d]
        somas_col["SAB"] += n_sab_curso
        somas_col["Total"] += tot_b

        row_b = [Paragraph(f"<b>{b_num}º Bimestre</b>", style_cel_centro)]
        for d in ("SEG", "TER", "QUA", "QUI", "SEX"):
            row_b.append(Paragraph(str(d_sem[d]), style_cel_centro))
        if incluir_col_sab:
            row_b.append(Paragraph(str(n_sab_curso), style_cel_centro))
        row_b.append(Paragraph(f"<b>{tot_b}</b>", style_cel_centro))
        linhas_sem.append(row_b)

    row_soma = [Paragraph("<b>Soma</b>", style_cel_centro)]
    for d in ("SEG", "TER", "QUA", "QUI", "SEX"):
        row_soma.append(Paragraph(f"<b>{somas_col[d]}</b>", style_cel_centro))
    if incluir_col_sab:
        row_soma.append(Paragraph(f"<b>{somas_col['SAB']}</b>", style_cel_centro))
    row_soma.append(Paragraph(f"<b>{somas_col['Total']}</b>", style_cel_centro))
    linhas_sem.append(row_soma)

    tab_sem = Table(linhas_sem, colWidths=_largura(col_w_sem))
    tab_sem.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
        ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f4f8")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tab_sem)
    story.append(Spacer(1, 0.2 * cm))

    # Nota do resíduo de fronteira (maio/outubro/dezembro)
    story.append(
        Paragraph(
            "<b>Nota sobre divisão de fronteiras entre bimestres:</b> Meses de transição têm dias letivos "
            "distribuídos oficialmente: <b>maio</b> divide-se em 5 dias (1º BI) e 17 dias (2º BI), "
            "totalizando 22 dias; <b>outubro</b> divide-se em 3 dias (3º BI) e 19 dias (4º BI), "
            "totalizando 22 dias; e <b>dezembro</b> possui 4 dias letivos (alocados no 4º BI).",
            style_caption,
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    # Tabela de faixas de carga horária efetiva (1, 2, 3, 4 CH)
    linhas_faixas = [[
        Paragraph("<b>Aulas/sem</b>", style_cab),
        Paragraph("<b>Nominal</b>", style_cab),
        Paragraph("<b>Cenário A (Mín–Máx)</b>", style_cab),
        Paragraph("<b>Cenário REAL (Mín–Máx)</b>", style_cab),
        Paragraph("<b>Limite Faltas (75%)*</b>", style_cab),
        Paragraph("<b>Pior Arranjo</b>", style_cab),
    ]]

    for ch in (1, 2, 3, 4):
        nom = ch_nominal(ch)
        min_a, dist_a, max_a, _ = faixa_ch(cal, ch, cenario="A")
        min_r, dist_r, max_r, _ = faixa_ch(
            cal, ch, cenario="REAL", curso=curso if curso else "TÉCNICO EM ESTRADAS"
        )

        if cenario_norm == "REAL":
            min_usado = min_r
            dist_usado = dist_r
        else:
            min_usado = min_a
            dist_usado = dist_a

        lf = limite_faltas(min_usado)
        pior_arr = ", ".join(f"{d} ({qtd})" for d, qtd in sorted(dist_usado.items()))

        linhas_faixas.append([
            Paragraph(f"{ch} h/a", style_cel_centro),
            Paragraph(f"{nom} h/a", style_cel_centro),
            Paragraph(f"{min_a} a {max_a} h/a", style_cel_centro),
            Paragraph(f"{min_r} a {max_r} h/a", style_cel_centro),
            Paragraph(f"{lf} faltas", style_cel_centro),
            Paragraph(pior_arr, style_cel_centro),
        ])

    tab_faixas = Table(
        linhas_faixas,
        colWidths=_largura([2.6 * cm, 2.0 * cm, 3.2 * cm, 3.2 * cm, 3.1 * cm, 3.4 * cm]),
    )
    tab_faixas.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
        ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tab_faixas)
    story.append(Spacer(1, 0.1 * cm))
    story.append(
        Paragraph(
            f"<b>* Limite legal de faltas (Art. 24, LDB):</b> Máximo de faltas para manter frequência "
            f"&ge; 75,0%, calculado sobre o piso da carga horária efetiva no cenário ativo ({cenario_norm}).",
            style_caption,
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # (iii) Trecho da tabela 2.1 com a coluna extra “Prog.” (siglas de programas)
    # -------------------------------------------------------------------------
    titulo_3 = (
        "Desempenho e Frequência por Aluno (Trecho Tabela 2.1 com Programas)"
        if layout_norm == "app"
        else "<b>3. Desempenho e Frequência por Aluno (Trecho Tabela 2.1 com Programas)</b>"
    )
    story.append(Paragraph(titulo_3, style_h2))
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
        colWidths=_largura([4.6 * cm, 2.5 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.8 * cm, 1.7 * cm]),
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
    # (iv) Quadro “Tratamento de dados pessoais (LGPD)” com NOTA_LGPD (D10-ii)
    # -------------------------------------------------------------------------
    titulo_4 = (
        "Tratamento de dados pessoais (LGPD)"
        if layout_norm == "app"
        else "<b>4. Tratamento de dados pessoais (LGPD)</b>"
    )
    story.append(Paragraph(titulo_4, style_h2))

    quadro_lgpd_conteudo = [
        Paragraph(
            f"<b>Quadro de Proteção de Dados:</b><br/>{NOTA_LGPD}",
            style_lgpd,
        )
    ]
    tab_quadro_lgpd = Table([[quadro_lgpd_conteudo]], colWidths=_largura([17.5 * cm]))
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
    # (v) Seção “Estudantes acompanhados pela Assistência Estudantil” (D10-iv)
    # -------------------------------------------------------------------------
    titulo_5 = (
        "Estudantes acompanhados pela Assistência Estudantil"
        if layout_norm == "app"
        else "<b>5. Estudantes acompanhados pela Assistência Estudantil</b>"
    )
    story.append(Paragraph(titulo_5, style_h2))

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

    tab_estudantes = Table(linhas_estudantes, colWidths=_largura(col_widths_est))
    tab_estudantes.setStyle(TableStyle(estilos_tabela_est))

    bloco_iv = [
        tab_estudantes,
        Spacer(1, 0.2 * cm),
        Paragraph(LEGENDA_PROGRAMAS, style_caption),
        Spacer(1, 0.1 * cm),
        Paragraph(f"<b>Observação:</b> {NOTA_PE_DE_MEIA}", style_caption),
    ]
    story.append(KeepTogether(bloco_iv))

    # -------------------------------------------------------------------------
    # (vi) Seção “Cruzamento com o mapa de turma” (C14)
    # -------------------------------------------------------------------------
    conjuntos: list = []
    eh_det_cruz, det_obj = _resolver_cruzamento_det(cruzamento)
    if cruzamento:
        story.append(Spacer(1, 0.35 * cm))
        titulo_6 = (
            "Cruzamento com o mapa de turma"
            if layout_norm == "app"
            else "<b>6. Cruzamento com o mapa de turma</b>"
        )
        story.append(Paragraph(titulo_6, style_h2))

        if eh_det_cruz:
            r_det = resumo_det(det_obj)
            est_n = r_det.alunos_por_curso.get("Estradas", 0)
            tt_n = r_det.alunos_por_curso.get("Trânsito", 0)

            linhas_mapa = [
                [
                    Paragraph("<b>Alunos DET (Estradas + Trânsito)</b>", style_cab),
                    Paragraph("<b>Estradas</b>", style_cab),
                    Paragraph("<b>Trânsito</b>", style_cab),
                    Paragraph("<b>Interseção</b>", style_cab),
                ],
                [
                    Paragraph(f"{r_det.alunos_total} alunos", style_cel_centro),
                    Paragraph(f"{est_n} alunos", style_cel_centro),
                    Paragraph(f"{tt_n} alunos", style_cel_centro),
                    Paragraph(str(r_det.intersecao), style_cel_centro),
                ],
            ]
            tab_mapa = Table(
                linhas_mapa,
                colWidths=_largura([6.5 * cm, 3.5 * cm, 3.5 * cm, 4.0 * cm]),
            )
            tab_mapa.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story.append(tab_mapa)
            story.append(Spacer(1, 0.2 * cm))

            story.append(
                Paragraph(
                    f"<b>{ROTULO_DET}:</b> {r_det.alunos_total} estudantes no departamento "
                    f"(Estradas: {est_n}, Trânsito: {tt_n}, interseção: {r_det.intersecao}).",
                    style_caption,
                )
            )
            story.append(Spacer(1, 0.2 * cm))

            # Recorte Pé-de-Meia com a tupla
            curso_tupla = ("estradas", "transito")
            contagem_curso = contagem_pe_de_meia(df, curso_contem=curso_tupla)

            mats_mapa = set()
            for conj in det_obj:
                for item in conj:
                    df_f = item[1]
                    if "matricula" in df_f.columns:
                        mats_mapa.update(df_f["matricula"].apply(_normalizar_matricula).dropna().unique())

            contagem_mapa = contagem_pe_de_meia(df, matriculas=mats_mapa)

            linhas_pdm = [
                [
                    Paragraph("<b>Situação Pé-de-Meia</b>", style_cab),
                    Paragraph("<b>Estudantes no Mapa</b>", style_cab),
                    Paragraph("<b>Total no Curso (Estradas + Trânsito)</b>", style_cab),
                ],
                [
                    Paragraph("Elegível", style_cel),
                    Paragraph(str(contagem_mapa["elegivel"]), style_cel_centro),
                    Paragraph(str(contagem_curso["elegivel"]), style_cel_centro),
                ],
                [
                    Paragraph("Não elegível", style_cel),
                    Paragraph(str(contagem_mapa["nao_elegivel"]), style_cel_centro),
                    Paragraph(str(contagem_curso["nao_elegivel"]), style_cel_centro),
                ],
                [
                    Paragraph("N/C (Nada consta)", style_cel),
                    Paragraph(str(contagem_mapa["nada_consta"]), style_cel_centro),
                    Paragraph(str(contagem_curso["nada_consta"]), style_cel_centro),
                ],
            ]
            if contagem_mapa["indefinida"] > 0 or contagem_curso["indefinida"] > 0:
                linhas_pdm.append([
                    Paragraph("Elegibilidade indefinida", style_cel),
                    Paragraph(str(contagem_mapa["indefinida"]), style_cel_centro),
                    Paragraph(str(contagem_curso["indefinida"]), style_cel_centro),
                ])
            linhas_pdm.append([
                Paragraph("<b>Total</b>", style_cel),
                Paragraph(f"<b>{contagem_mapa['total']}</b>", style_cel_centro),
                Paragraph(f"<b>{contagem_curso['total']}</b>", style_cel_centro),
            ])

            tab_pdm = Table(linhas_pdm, colWidths=_largura([7.5 * cm, 5.0 * cm, 5.0 * cm]))
            tab_pdm.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f4f8")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story.append(tab_pdm)

            dae_sintetica = bool(df.attrs.get("sintetico", False))
            if dae_sintetica:
                story.append(Spacer(1, 0.2 * cm))
                quadro_pendencia = [
                    Paragraph(
                        "<b>Cruzamento pendente:</b> Planilha oficial de acompanhamento discente da DAE "
                        "não fornecida (base sintética em uso). A conciliação individual de faltas e o "
                        "cruzamento com os registros da Assistência Estudantil estão pendentes até a "
                        "disponibilização do arquivo oficial da DAE.",
                        style_corpo,
                    )
                ]
                tab_pendencia = Table([[quadro_pendencia]], colWidths=_largura([17.5 * cm]))
                tab_pendencia.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9e6")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d9822b")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(tab_pendencia)
            else:
                df_unido, resumo = cruzar(df, det_obj, bimestres=lista_bimestres)
                tot_cruz = resumo["total"]
                pct_dois = f"{resumo['nos_dois'] / tot_cruz:.1%}" if tot_cruz > 0 else "—"
                pct_app = f"{resumo['so_app'] / tot_cruz:.1%}" if tot_cruz > 0 else "—"
                pct_dae = f"{resumo['so_dae'] / tot_cruz:.1%}" if tot_cruz > 0 else "—"

                linhas_cob = [
                    [
                        Paragraph("<b>Situação da Cobertura</b>", style_cab),
                        Paragraph("<b>Estudantes</b>", style_cab),
                        Paragraph("<b>Percentual</b>", style_cab),
                    ],
                    [
                        Paragraph("Nos dois (conciliados)", style_cel),
                        Paragraph(str(resumo["nos_dois"]), style_cel_centro),
                        Paragraph(pct_dois, style_cel_centro),
                    ],
                    [
                        Paragraph("Só no Mapa de Turma", style_cel),
                        Paragraph(str(resumo["so_app"]), style_cel_centro),
                        Paragraph(pct_app, style_cel_centro),
                    ],
                    [
                        Paragraph("Só na DAE", style_cel),
                        Paragraph(str(resumo["so_dae"]), style_cel_centro),
                        Paragraph(pct_dae, style_cel_centro),
                    ],
                    [
                        Paragraph("<b>Total</b>", style_cel),
                        Paragraph(f"<b>{tot_cruz}</b>", style_cel_centro),
                        Paragraph("100,0%", style_cel_centro),
                    ],
                ]
                tab_cob = Table(linhas_cob, colWidths=_largura([8.5 * cm, 4.5 * cm, 4.5 * cm]))
                tab_cob.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f4f8")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]))
                story.append(Spacer(1, 0.25 * cm))
                story.append(tab_cob)

                diff_msgs = []
                for b in lista_bimestres:
                    col_diff = f"diff_faltas_bim_{b}"
                    if col_diff in df_unido.columns:
                        diff_abs = df_unido.loc[df_unido["_merge"] == "both", col_diff].dropna().abs()
                        if len(diff_abs) > 0:
                            med = float(diff_abs.median())
                            p90 = float(diff_abs.quantile(0.90))
                            diff_msgs.append(f"{b}º BI: mediana {med:.1f}, P90 {p90:.1f}")
                if diff_msgs:
                    story.append(Spacer(1, 0.15 * cm))
                    story.append(
                        Paragraph("<b>Discrepância de faltas (|diff_faltas|):</b> " + " | ".join(diff_msgs), style_caption)
                    )
        else:
            if isinstance(cruzamento, (str, Path)):
                conjuntos = processar_multiplos_bimestres([str(cruzamento)])
            elif isinstance(cruzamento, (list, tuple)) and cruzamento:
                primeiro = cruzamento[0]
                if isinstance(primeiro, (str, Path)):
                    conjuntos = processar_multiplos_bimestres([str(p) for p in cruzamento])
                elif isinstance(primeiro, (tuple, list)) and len(primeiro) >= 4:
                    conjuntos = list(cruzamento)
                else:
                    conjuntos = processar_multiplos_bimestres([str(p) for p in cruzamento])

            if conjuntos:
                res_mapa = resumo_mapa(conjuntos)[0]

            linhas_mapa = [
                [
                    Paragraph("<b>Alunos (Mapa)</b>", style_cab),
                    Paragraph("<b>Disciplinas</b>", style_cab),
                    Paragraph("<b>Total Faltas</b>", style_cab),
                    Paragraph("<b>Bimestre</b>", style_cab),
                ],
                [
                    Paragraph(str(res_mapa["n_alunos"]), style_cel_centro),
                    Paragraph(str(res_mapa["n_disciplinas"]), style_cel_centro),
                    Paragraph(str(res_mapa["faltas_total"]), style_cel_centro),
                    Paragraph(f"{res_mapa['bimestre_num']}º", style_cel_centro),
                ],
            ]
            tab_mapa = Table(
                linhas_mapa,
                colWidths=_largura([4.5 * cm, 4.0 * cm, 4.5 * cm, 4.5 * cm]),
            )
            tab_mapa.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story.append(tab_mapa)
            story.append(Spacer(1, 0.2 * cm))

            invalidas = res_mapa["n_alunos"] - res_mapa["n_matriculas_validas"]
            if res_mapa["n_matriculas_duplicadas"] > 0 or invalidas > 0:
                story.append(
                    Paragraph(
                        f"<b>Validações de consistência:</b> Matrículas duplicadas: {res_mapa['n_matriculas_duplicadas']} | "
                        f"Matrículas fora do padrão (inválidas): {invalidas}",
                        style_caption,
                    )
                )
                story.append(Spacer(1, 0.15 * cm))

            dae_sintetica = bool(df.attrs.get("sintetico", False))

            if dae_sintetica:
                quadro_pendencia = [
                    Paragraph(
                        "<b>Cruzamento pendente:</b> Planilha oficial de acompanhamento discente da DAE "
                        "não fornecida (base sintética em uso). A conciliação individual de faltas e o "
                        "cruzamento com os registros da Assistência Estudantil estão pendentes até a "
                        "disponibilização do arquivo oficial da DAE.",
                        style_corpo,
                    )
                ]
                tab_pendencia = Table([[quadro_pendencia]], colWidths=_largura([17.5 * cm]))
                tab_pendencia.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9e6")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d9822b")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(tab_pendencia)
            else:
                df_unido, resumo = cruzar(df, conjuntos, bimestres=lista_bimestres)

                tot_cruz = resumo["total"]
                pct_dois = f"{resumo['nos_dois'] / tot_cruz:.1%}" if tot_cruz > 0 else "—"
                pct_app = f"{resumo['so_app'] / tot_cruz:.1%}" if tot_cruz > 0 else "—"
                pct_dae = f"{resumo['so_dae'] / tot_cruz:.1%}" if tot_cruz > 0 else "—"

                linhas_cob = [
                    [
                        Paragraph("<b>Situação da Cobertura</b>", style_cab),
                        Paragraph("<b>Estudantes</b>", style_cab),
                        Paragraph("<b>Percentual</b>", style_cab),
                    ],
                    [
                        Paragraph("Nos dois (conciliados)", style_cel),
                        Paragraph(str(resumo["nos_dois"]), style_cel_centro),
                        Paragraph(pct_dois, style_cel_centro),
                    ],
                    [
                        Paragraph("Só no Mapa de Turma", style_cel),
                        Paragraph(str(resumo["so_app"]), style_cel_centro),
                        Paragraph(pct_app, style_cel_centro),
                    ],
                    [
                        Paragraph("Só na DAE", style_cel),
                        Paragraph(str(resumo["so_dae"]), style_cel_centro),
                        Paragraph(pct_dae, style_cel_centro),
                    ],
                    [
                        Paragraph("<b>Total</b>", style_cel),
                        Paragraph(f"<b>{tot_cruz}</b>", style_cel_centro),
                        Paragraph("100,0%", style_cel_centro),
                    ],
                ]
                tab_cob = Table(linhas_cob, colWidths=_largura([8.5 * cm, 4.5 * cm, 4.5 * cm]))
                tab_cob.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f4f8")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]))
                story.append(tab_cob)
                story.append(Spacer(1, 0.25 * cm))

                curso_alvo = None
                for item in conjuntos:
                    meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
                    curso_alvo = meta.get("curso_amigavel") or meta.get("curso")
                    if curso_alvo:
                        break
                if not curso_alvo:
                    curso_alvo = curso

                contagem_curso = contagem_pe_de_meia(df, curso_contem=curso_alvo)

                mats_mapa = set()
                for item in conjuntos:
                    df_f = item[1]
                    if "matricula" in df_f.columns:
                        mats_mapa.update(df_f["matricula"].apply(_normalizar_matricula).dropna().unique())

                contagem_mapa = contagem_pe_de_meia(df, matriculas=mats_mapa)

                linhas_pdm = [
                    [
                        Paragraph("<b>Situação Pé-de-Meia</b>", style_cab),
                        Paragraph("<b>Estudantes no Mapa</b>", style_cab),
                        Paragraph(f"<b>Total no Curso ({curso_alvo or 'DAE'})</b>", style_cab),
                    ],
                    [
                        Paragraph("Elegível", style_cel),
                        Paragraph(str(contagem_mapa["elegivel"]), style_cel_centro),
                        Paragraph(str(contagem_curso["elegivel"]), style_cel_centro),
                    ],
                    [
                        Paragraph("Não elegível", style_cel),
                        Paragraph(str(contagem_mapa["nao_elegivel"]), style_cel_centro),
                        Paragraph(str(contagem_curso["nao_elegivel"]), style_cel_centro),
                    ],
                    [
                        Paragraph("N/C (Nada consta)", style_cel),
                        Paragraph(str(contagem_mapa["nada_consta"]), style_cel_centro),
                        Paragraph(str(contagem_curso["nada_consta"]), style_cel_centro),
                    ],
                ]
                if contagem_mapa["indefinida"] > 0 or contagem_curso["indefinida"] > 0:
                    linhas_pdm.append([
                        Paragraph("Elegibilidade indefinida", style_cel),
                        Paragraph(str(contagem_mapa["indefinida"]), style_cel_centro),
                        Paragraph(str(contagem_curso["indefinida"]), style_cel_centro),
                    ])
                linhas_pdm.append([
                    Paragraph("<b>Total</b>", style_cel),
                    Paragraph(f"<b>{contagem_mapa['total']}</b>", style_cel_centro),
                    Paragraph(f"<b>{contagem_curso['total']}</b>", style_cel_centro),
                ])

                tab_pdm = Table(linhas_pdm, colWidths=_largura([7.5 * cm, 5.0 * cm, 5.0 * cm]))
                tab_pdm.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f4f8")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]))
                story.append(tab_pdm)

                diff_msgs = []
                for b in lista_bimestres:
                    col_diff = f"diff_faltas_bim_{b}"
                    if col_diff in df_unido.columns:
                        diff_abs = df_unido.loc[df_unido["_merge"] == "both", col_diff].dropna().abs()
                        if len(diff_abs) > 0:
                            med = float(diff_abs.median())
                            p90 = float(diff_abs.quantile(0.90))
                            diff_msgs.append(f"{b}º BI: mediana {med:.1f}, P90 {p90:.1f}")
                if diff_msgs:
                    story.append(Spacer(1, 0.15 * cm))
                    story.append(
                        Paragraph("<b>Discrepância de faltas (|diff_faltas|):</b> " + " | ".join(diff_msgs), style_caption)
                    )

    # -------------------------------------------------------------------------
    # (vii) Bloco “CH efetiva por disciplina (horário real)” (C18)
    # -------------------------------------------------------------------------
    num_sec = "7." if cruzamento else "6."

    # C18: default = dados/CH_Efetiva_Disciplinas_Integrado_<ANO_PADRAO>.xlsx,
    # se existir; planilha inválida propaga o erro da carga (não é "ausente").
    if isinstance(ch_efetiva, pd.DataFrame):
        df_ch = ch_efetiva
        nome_planilha = "planilha fornecida em memória"
    else:
        caminho_ch_resolvido = (
            Path(ch_efetiva)
            if ch_efetiva is not None
            else PASTA_DADOS / f"CH_Efetiva_Disciplinas_Integrado_{ANO_PADRAO}.xlsx"
        )
        nome_planilha = caminho_ch_resolvido.name
        df_ch = carregar_ch_efetiva(caminho_ch_resolvido) if caminho_ch_resolvido.exists() else None

    if df_ch is None:
        story.append(Spacer(1, 0.35 * cm))
        titulo_ch = (
            "CH efetiva por disciplina (horário real)"
            if layout_norm == "app"
            else f"<b>{num_sec} CH efetiva por disciplina (horário real)</b>"
        )
        story.append(Paragraph(titulo_ch, style_h2))
        quadro_ausente = [
            Paragraph(
                "<b>Planilha de CH efetiva ausente:</b> frequência por disciplina usa a faixa do calendário. "
                "O detalhamento das horas-aula lecionadas por disciplina depende da disponibilização "
                "da planilha de horários em sandbox/dae/dados/.",
                style_corpo,
            )
        ]
        tab_ausente = Table([[quadro_ausente]], colWidths=_largura([17.5 * cm]))
        tab_ausente.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9e6")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d9822b")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(tab_ausente)
    else:
        divs = divergencias_calendario(df_ch, cal)
        if eh_det_cruz:
            sabs_det = sabados_do_responsavel(cal, "DET")
            if not sabs_det:
                sabs_det = sorted(list(set(sabados_do_responsavel(cal, "Estradas") + sabados_do_responsavel(cal, "Trânsito"))))
            detalhes_sabs = []
            for dt in sabs_det:
                b_num = None
                for num, b in cal.bimestres.items():
                    if b.inicio <= dt <= b.fim:
                        b_num = num
                        break
                fmt_dt = dt.strftime("%d/%m")
                if b_num is not None:
                    detalhes_sabs.append(f"{fmt_dt} ({b_num}º bimestre)")
                else:
                    detalhes_sabs.append(fmt_dt)
            sabs_str = ", ".join(detalhes_sabs) if detalhes_sabs else "23/05 (2º bimestre)"
            nota_sab = (
                f"Sábado(s) letivo(s) atribuído(s) aos cursos de Estradas e Trânsito (DET): {sabs_str}. "
                f"Os sábados letivos não entram no cômputo da carga horária efetiva das disciplinas (cenário A)."
            )
        else:
            nota_sab = nota_sabados(cal, curso)

        story.append(Spacer(1, 0.35 * cm))
        titulo_ch = (
            "CH efetiva por disciplina (horário real)"
            if layout_norm == "app"
            else f"<b>{num_sec} CH efetiva por disciplina (horário real)</b>"
        )
        story.append(Paragraph(titulo_ch, style_h2))
        story.append(
            Paragraph(
                f"<b>Fonte:</b> planilha <code>{nome_planilha}</code> (cenário A — sem sábados; "
                f"origem: {df_ch.attrs.get('ch_origem', 'planilha')}). "
                f"Divergências com o calendário oficial: {len(divs)}.",
                style_corpo,
            )
        )
        for d in divs:
            story.append(Paragraph(f"• {d}", style_caption))

        if eh_det_cruz:
            bim_det = int(r_det.bimestres[0] if r_det.bimestres else (lista_bimestres[0] if lista_bimestres else 1))
            res_freq_det = resumo_frequencia_det(det_obj, df_ch, bimestre=bim_det, calendario=cal)
            df_det_res = res_freq_det.df
            sem_horario_det = res_freq_det.sem_horario

            story.append(
                Paragraph(
                    f"<b>Resumo da oferta DET (Estradas + Trânsito):</b> {len(df_det_res)} disciplinas da planilha.",
                    style_corpo,
                )
            )
            story.append(Spacer(1, 0.15 * cm))

            linhas_tab_ch = [
                [
                    Paragraph("<b>Disciplina</b>", style_cab),
                    Paragraph("<b>Escopo</b>", style_cab),
                    Paragraph("<b>Aulas/sem</b>", style_cab),
                    Paragraph(f"<b>CH {bim_det}º BI</b>", style_cab),
                    Paragraph("<b>CH efetiva no ano</b>", style_cab),
                    Paragraph("<b>% do nominal</b>", style_cab),
                    Paragraph("<b>Limite de faltas p/ 75%</b>", style_cab),
                    Paragraph("<b>&lt; 75% freq.</b>", style_cab),
                    Paragraph("<b>Fonte</b>", style_cab),
                ]
            ]

            for _, row in df_det_res.iterrows():
                ch_b = row.get("ch_bim")
                lim_f = row.get("limite_faltas_bim")
                n_abaixo = row.get("n_abaixo_75")
                pct_nom = row.get("%_nominal")
                ch_ano = row.get("ch_efetiva_ano")

                ch_str = f"{int(ch_b)} h/a" if pd.notna(ch_b) else "—"
                ch_ano_str = f"{int(ch_ano)} h/a" if pd.notna(ch_ano) else "—"
                pct_str = f"{pct_nom:.1%}" if pd.notna(pct_nom) else "—"
                lim_str = str(int(lim_f)) if pd.notna(lim_f) else "—"

                if pd.notna(n_abaixo):
                    if int(n_abaixo) > 0:
                        abaixo_p = Paragraph(f"<b><font color='#c00000'>{int(n_abaixo)}</font></b>", style_cel_centro)
                    else:
                        abaixo_p = Paragraph("0", style_cel_centro)
                else:
                    abaixo_p = Paragraph("—", style_cel_centro)

                aulas_sem = row.get("aulas_sem")
                aulas_str = str(int(aulas_sem)) if pd.notna(aulas_sem) else "—"
                fonte_str = "Planilha" if row.get("fonte") == "planilha" else "Sem horário na planilha"

                linhas_tab_ch.append([
                    Paragraph(str(row["disciplina"]), style_cel),
                    Paragraph(str(row["escopo"]), style_cel_centro),
                    Paragraph(aulas_str, style_cel_centro),
                    Paragraph(ch_str, style_cel_centro),
                    Paragraph(ch_ano_str, style_cel_centro),
                    Paragraph(pct_str, style_cel_centro),
                    Paragraph(lim_str, style_cel_centro),
                    abaixo_p,
                    Paragraph(fonte_str, style_cel_centro),
                ])

            tab_ch = Table(
                linhas_tab_ch,
                colWidths=_largura([4.2 * cm, 2.3 * cm, 1.3 * cm, 1.5 * cm, 1.7 * cm, 1.5 * cm, 1.8 * cm, 1.4 * cm, 1.8 * cm]),
            )
            tab_ch.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story.append(tab_ch)

            if sem_horario_det:
                partes_sem = []
                for c_nome in sorted(sem_horario_det.keys()):
                    discs = sem_horario_det[c_nome]
                    if discs:
                        partes_sem.append(f"<b>{c_nome}:</b> {', '.join(sorted(discs))}")
                if partes_sem:
                    story.append(Spacer(1, 0.15 * cm))
                    story.append(
                        Paragraph(
                            f"<b>Sem horário na planilha (por curso):</b> {' | '.join(partes_sem)} "
                            f"(aulas lecionadas em laboratórios, oficinas ou instalações fora das salas 305–437).",
                            style_caption,
                        )
                    )

            story.append(Spacer(1, 0.15 * cm))
            story.append(Paragraph(f"<b>Sábados letivos:</b> {nota_sab}", style_caption))

        elif not conjuntos:
            story.append(Spacer(1, 0.15 * cm))

            # C18: sem mapas, só o resumo_por_carga das linhas do --curso
            alvo_curso = normalizar_disciplina(str(curso)).lower().strip()
            df_carga = df_ch[
                df_ch["curso"].apply(lambda c: alvo_curso in normalizar_disciplina(c).lower())
            ]
            res_carga = resumo_por_carga(df_carga)

            tot_disc = res_carga["total"]
            pct_abaixo = f"{res_carga['< 90%'] / tot_disc:.1%}" if tot_disc > 0 else "—"
            pct_90_95 = f"{res_carga['90–95%'] / tot_disc:.1%}" if tot_disc > 0 else "—"
            pct_95_mais = f"{res_carga['≥ 95%'] / tot_disc:.1%}" if tot_disc > 0 else "—"

            linhas_dist = [
                [
                    Paragraph("<b>Faixa de Carga Horária (% da nominal)</b>", style_cab),
                    Paragraph("<b>Disciplinas na Grade</b>", style_cab),
                    Paragraph("<b>Percentual</b>", style_cab),
                ],
                [
                    Paragraph("< 90%", style_cel),
                    Paragraph(str(res_carga["< 90%"]), style_cel_centro),
                    Paragraph(pct_abaixo, style_cel_centro),
                ],
                [
                    Paragraph("90–95%", style_cel),
                    Paragraph(str(res_carga["90–95%"]), style_cel_centro),
                    Paragraph(pct_90_95, style_cel_centro),
                ],
                [
                    Paragraph("≥ 95%", style_cel),
                    Paragraph(str(res_carga["≥ 95%"]), style_cel_centro),
                    Paragraph(pct_95_mais, style_cel_centro),
                ],
                [
                    Paragraph("<b>Total</b>", style_cel),
                    Paragraph(f"<b>{tot_disc}</b>", style_cel_centro),
                    Paragraph("100,0%", style_cel_centro),
                ],
            ]
            tab_dist = Table(linhas_dist, colWidths=_largura([8.5 * cm, 4.5 * cm, 4.5 * cm]))
            tab_dist.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f4f8")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story.append(tab_dist)
            story.append(Spacer(1, 0.2 * cm))

            quadro_aviso_mapa = [
                Paragraph(
                    "<b>Mapa da turma não informado:</b> para CH por disciplina, informe o mapa da turma.",
                    style_corpo,
                )
            ]
            tab_aviso = Table([[quadro_aviso_mapa]], colWidths=_largura([17.5 * cm]))
            tab_aviso.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9e6")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d9822b")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(tab_aviso)
            story.append(Spacer(1, 0.15 * cm))
            story.append(Paragraph(f"<b>Sábados letivos:</b> {nota_sab}", style_caption))
        else:
            for df_notas_cj, df_faltas_cj, legenda_cj, meta_cj in conjuntos:
                curso_alvo = meta_cj.get("curso_amigavel") or meta_cj.get("curso") or curso
                serie_val = meta_cj.get("serie")
                if serie_val is None:
                    turma_str = str(meta_cj.get("turma", ""))
                    m_s = re.search(r"-(\d)[A-Z]", turma_str) or re.search(r"(\d)[ªa]\s*s[ée]rie", turma_str, re.I)
                    serie_alvo = int(m_s.group(1)) if m_s else 2
                else:
                    serie_alvo = int(serie_val)
                turma_alvo = meta_cj.get("turma") or "A"
                bim_cj = int(meta_cj.get("bimestre_num") or (lista_bimestres[0] if lista_bimestres else 1))

                df_turma = ch_da_turma(df_ch, curso_alvo, serie_alvo, turma_alvo)
                ch_casada, sem_linha = casar_disciplinas(legenda_cj, df_turma)
                resumo_freq = resumo_frequencia_por_disciplina(
                    df_faltas_cj,
                    legenda_cj,
                    ch_casada,
                    bimestre=bim_cj,
                    cal=cal,
                )

                story.append(
                    Paragraph(
                        f"<b>Resumo da oferta:</b> {len(ch_casada)} disciplinas da planilha, "
                        f"{len(sem_linha)} sem horário.",
                        style_corpo,
                    )
                )
                story.append(Spacer(1, 0.15 * cm))

                linhas_tab_ch = [
                    [
                        Paragraph("<b>Disciplina</b>", style_cab),
                        Paragraph("<b>Aulas/sem</b>", style_cab),
                        Paragraph(f"<b>CH {bim_cj}º BI</b>", style_cab),
                        Paragraph("<b>CH efetiva no ano</b>", style_cab),
                        Paragraph("<b>% do nominal</b>", style_cab),
                        Paragraph("<b>Limite de faltas p/ 75%</b>", style_cab),
                        Paragraph("<b>&lt; 75% freq.</b>", style_cab),
                        Paragraph("<b>Fonte</b>", style_cab),
                    ]
                ]

                itens_casados = [r for r in resumo_freq if r["fonte"] == "planilha"]
                itens_sem_horario = [r for r in resumo_freq if r["fonte"] != "planilha"]
                itens_casados.sort(key=lambda r: str(r["disciplina"]))
                itens_sem_horario.sort(key=lambda r: str(r["disciplina"]))

                for res_d in itens_casados + itens_sem_horario:
                    ch_b = res_d["ch_bim"]
                    lim_f = res_d["limite_faltas_bim"]
                    n_abaixo = res_d["n_abaixo_75"]
                    pct_nom = res_d["%_nominal"]
                    ch_ano = res_d["ch_efetiva_ano"]

                    ch_str = f"{ch_b} h/a" if ch_b is not None else "—"
                    ch_ano_str = f"{ch_ano} h/a" if ch_ano is not None else "—"
                    pct_str = f"{pct_nom:.1%}" if pct_nom is not None else "—"
                    lim_str = str(lim_f) if lim_f is not None else "—"

                    if n_abaixo is not None:
                        if n_abaixo > 0:
                            abaixo_p = Paragraph(f"<b><font color='#c00000'>{n_abaixo}</font></b>", style_cel_centro)
                        else:
                            abaixo_p = Paragraph("0", style_cel_centro)
                    else:
                        abaixo_p = Paragraph("—", style_cel_centro)

                    aulas_str = str(res_d["aulas_sem"]) if res_d["aulas_sem"] is not None else "—"
                    fonte_str = "Planilha" if res_d["fonte"] == "planilha" else "Sem horário na planilha"

                    linhas_tab_ch.append([
                        Paragraph(str(res_d["disciplina"]), style_cel),
                        Paragraph(aulas_str, style_cel_centro),
                        Paragraph(ch_str, style_cel_centro),
                        Paragraph(ch_ano_str, style_cel_centro),
                        Paragraph(pct_str, style_cel_centro),
                        Paragraph(lim_str, style_cel_centro),
                        abaixo_p,
                        Paragraph(fonte_str, style_cel_centro),
                    ])

                tab_ch = Table(
                    linhas_tab_ch,
                    colWidths=_largura([4.6 * cm, 1.6 * cm, 1.7 * cm, 1.9 * cm, 1.7 * cm, 2.1 * cm, 1.7 * cm, 2.2 * cm]),
                )
                tab_ch.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COR_CABECALHO_TABELA),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COR_TEXTO_CABECALHO_TABELA),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]))
                story.append(tab_ch)

                if sem_linha:
                    nomes_sem = ", ".join(sorted(sem_linha))
                    story.append(Spacer(1, 0.15 * cm))
                    story.append(
                        Paragraph(
                            f"<b>Sem horário na planilha:</b> {nomes_sem} "
                            f"(aulas lecionadas em laboratórios, oficinas ou instalações fora das salas 305–437).",
                            style_caption,
                        )
                    )

                story.append(Spacer(1, 0.15 * cm))
                story.append(Paragraph(f"<b>Sábados letivos:</b> {nota_sab}", style_caption))

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
    cenario: str = "A",
    calendario: Calendario | Path | str | None = None,
    sabado_reproduz: str | None = None,
    caminhos_mapas: Sequence[str | Path] | str | Path | None = None,
    caminho_ch: str | Path | None = None,
    layout: str = "prototipo",
) -> Path:
    """Gera o protótipo de destaque em PDF com os dados da DAE (ou sintéticos)."""
    # 1. Obtenção dos dados
    usando_sintetico = caminho_dae is None
    if usando_sintetico:
        df = obter_dados_sinteticos()
    else:
        df = carregar_dae(caminho_dae)
        df.attrs["sintetico"] = False

    # 5. Normalização dos mapas para cruzamento (C14)
    lista_mapas: list[str | Path] | None = None
    if caminhos_mapas:
        if isinstance(caminhos_mapas, (str, Path)):
            lista_mapas = [caminhos_mapas]
        else:
            lista_mapas = list(caminhos_mapas)

    cruzamento_arg: Any = lista_mapas
    eh_det_mapas = False
    if lista_mapas:
        try:
            classif = classificar_mapas(lista_mapas)
            if eh_det(classif):
                eh_det_mapas = True
                cruzamento_arg = carregar_det(classif)
        except Exception:
            cruzamento_arg = lista_mapas

    # 2. Filtragem pelo curso se a coluna existir e houver dados correspondentes (não filtra se for DET)
    if not eh_det_mapas and curso and "curso" in df.columns:
        mask_curso = df["curso"].str.upper() == curso.upper()
        if mask_curso.any():
            attrs_salvas = df.attrs.copy()
            df = df[mask_curso].reset_index(drop=True)
            df.attrs.update(attrs_salvas)

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

    # 6. Montagem dos flowables
    story = montar_flowables(
        df,
        bimestres=lista_bimestres,
        curso=curso,
        cenario=cenario,
        calendario=calendario,
        sabado_reproduz=sabado_reproduz,
        cruzamento=cruzamento_arg,
        ch_efetiva=caminho_ch,
        layout=layout,
    )

    # 7. Geração do PDF com SimpleDocTemplate e rodapé em todas as páginas (D10-i)
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
        "--mapas",
        dest="mapas",
        nargs="*",
        default=None,
        help="Caminhos para arquivos de Mapa de Turma (.xls).",
    )
    parser.add_argument(
        "--ch-efetiva",
        dest="ch_efetiva",
        type=str,
        default=None,
        help="Caminho para o arquivo da planilha de CH efetiva (.xlsx). Padrão: arquivo oficial em dados/.",
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
        "--cenario",
        dest="cenario",
        choices=["A", "REAL"],
        default="A",
        help="Cenário de apuração do calendário ('A' ou 'REAL', padrão: 'A').",
    )
    parser.add_argument(
        "--calendario",
        dest="calendario",
        type=str,
        default=None,
        help="Caminho para o arquivo Markdown do Calendário Escolar (padrão: oficial 2026).",
    )
    parser.add_argument(
        "--sabado-reproduz",
        dest="sabado_reproduz",
        choices=["SEG", "TER", "QUA", "QUI", "SEX"],
        default=None,
        help="Dia útil que o sábado letivo reproduz no cenário REAL (choices: SEG..SEX).",
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
        cenario=args.cenario,
        calendario=args.calendario,
        sabado_reproduz=args.sabado_reproduz,
        caminhos_mapas=args.mapas,
        caminho_ch=args.ch_efetiva,
    )
    print(f"Protótipo PDF gerado com sucesso em: {pdf_gerado}")
    return 0


if __name__ == "__main__":
    sys.exit(_executar_cli())

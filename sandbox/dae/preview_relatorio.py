"""Módulo para injeção da seção DAE no fluxo de PDF do app original.

Este módulo resolve a lacuna de extensibilidade do app (que não expõe um
ponto de extensão para inserir novas seções no PDF) através de um monkeypatch
temporário em `core.relatorios._DocComSumario.multiBuild`.

O patch é seguro porque o escopo da interceptação é restrito estritamente ao
bloco do context manager `injetar_secao_dae`, revertendo a classe ao estado
original imediatamente após o uso. No entanto, injetar flowables via monkeypatch
carrega o risco de acoplar a extensão a detalhes de implementação da API privada
(`_DocComSumario`). No futuro, se a arquitetura D5 de `secoes_extras` for
implementada nativamente no app, este módulo será descontinuado (essa integração
nativa está fora do escopo atual).

Na prévia gerada por este módulo, a análise de frequência da DAE por carga
horária efetiva substitui a análise estatística de faltas do app original (uma
análise de faltas só, evitando redundância). A remoção ocorre na entrada dos dados
via `estatisticas_sem_analise_faltas` e `figuras_sem_analise_faltas` (D2),
preservando dados brutos e indicadores compostos como a coluna de faltas em 2.1
e o IDA (D1). No wrapper de `multiBuild` (D3), os verbetes de faltas retirados
são filtrados do glossário, um apontamento para o capítulo DAE é inserido antes
de Visualizações Gráficas e uma nota explicativa é adicionada após o H1 da DAE (D4).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock
import copy
import re

import matplotlib.pyplot as plt
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import NextPageTemplate, PageBreak, Paragraph, Table, TableStyle

# Assegura o path do projeto e do sandbox
_DIR_DAE = Path(__file__).resolve().parent
_DIR_RAIZ = _DIR_DAE.parent.parent
if str(_DIR_DAE) not in sys.path:
    sys.path.insert(0, str(_DIR_DAE))
if str(_DIR_RAIZ) not in sys.path:
    sys.path.insert(0, str(_DIR_RAIZ))

from core.relatorios import _DocComSumario, criar_relatorio_pdf
import core.relatorios
import core.manipulacao as manipulacao
from sandbox.dae import prototipo_pdf
from sandbox.dae.det import classificar_mapas, eh_det, carregar_det, PASTA_DADOS

# Faixa de prévia desenhada no rodapé de todas as páginas (D5, item 4)
TEXTO_FAIXA_PREVIA = (
    "PRÉVIA gerada no sandbox/dae — não é relatório oficial. A seção DAE usa dados "
    "fictícios de demonstração (export da DAE pendente)."
)

_LOGO_PADRAO = _DIR_RAIZ / "assets" / "logo_cefet.png" 

# Termos do glossário referentes à análise de faltas por sinal estatístico do app (D3).
TERMOS_GLOSSARIO_FALTAS = (
    "μ + 2σ",
    "Quadrantes de Notas × Faltas",
    "Acima da média de faltas",
)

# Textos literais do plano para substituição da análise de faltas (D4).
APONTAMENTO_FREQUENCIA = (
    "A análise de frequência desta turma está no capítulo {n} "
    "(Acompanhamento Discente — DAE), calculada sobre a carga horária efetiva de cada "
    "disciplina."
)

NOTA_SUBSTITUICAO_FALTAS = (
    "Nesta prévia, este capítulo é a única análise de frequência do relatório: "
    "ele substitui a análise de faltas por sinal estatístico (média, P90 e μ+2σ da turma) "
    "do relatório atual, que foi retirada para não haver duas análises de faltas."
)

# Chaves das figuras geradas por gerar_todos_graficos referentes à análise
# estatística de faltas do app original (D2).
CHAVES_FIGURAS_FALTAS = (
    "faltas_total_aluno",
    "faltas_boxplot_disciplina",
    "dispersao_notas_faltas",
)


def estatisticas_sem_analise_faltas(estat: dict[str, Any]) -> dict[str, Any]:
    """Retorna cópia das estatísticas com faltas_disponiveis=False (D2), sem mutar o original."""
    copia = estat.copy()
    copia["faltas_disponiveis"] = False
    return copia


def figuras_sem_analise_faltas(figuras: dict[str, Any]) -> dict[str, Any]:
    """Retorna cópia do dicionário sem as 3 figuras de faltas (D2), fechando-as com plt.close."""
    copia = {}
    for chave, fig in figuras.items():
        if chave in CHAVES_FIGURAS_FALTAS:
            if fig is not None:
                plt.close(fig)
        else:
            copia[chave] = fig
    return copia


def _slug(texto: str | None) -> str:
    """Gera slug mantendo letras acentuadas, replicando a lógica de app.py."""
    return re.sub(r"\W+", "_", (texto or "curso").strip().lower()).strip("_") or "curso"


def _fmt_bimestres(bimestres: list[Any]) -> str:
    """Formata a lista de bimestres enviados para nome de arquivo/uso (ex.: '1-3')."""
    validos = sorted({b for b in bimestres if b is not None})
    if not validos:
        return "X"
    if len(validos) == 1:
        return str(validos[0])
    if validos == list(range(validos[0], validos[-1] + 1)):
        return f"{validos[0]}-{validos[-1]}"
    return ",".join(str(b) for b in validos)


@contextmanager
def injetar_secao_dae(
    flowables_dae: list[Any],
    registro: list[Any] | None = None,
    registro_story: list[Any] | None = None,
):
    """Context manager que injeta flowables da DAE ao final do relatório PDF."""
    original_multiBuild = _DocComSumario.multiBuild

    def patched_multiBuild(self, story, **kwargs):
        # 1. Filtro do glossário (D3)
        idx_h1_glossario = None
        for i, f in enumerate(story):
            if (
                isinstance(f, Paragraph)
                and getattr(f, "style", None)
                and getattr(f.style, "name", None) == "H1Sumario"
                and f.getPlainText().strip().endswith("Glossário")
            ):
                idx_h1_glossario = i
                break

        if idx_h1_glossario is None:
            raise RuntimeError("O app mudou: H1Sumario de Glossário não foi encontrado no story.")

        idx_tabela_gloss = None
        for i in range(idx_h1_glossario + 1, len(story)):
            if isinstance(story[i], Table):
                idx_tabela_gloss = i
                break

        if idx_tabela_gloss is None:
            raise RuntimeError("O app mudou: Tabela do glossário não foi encontrada após o H1Sumario de Glossário.")

        tabela_orig = story[idx_tabela_gloss]
        novas_linhas = []
        termos_encontrados = set()
        for row in getattr(tabela_orig, "_cellvalues", []):
            cell = row[0]
            while isinstance(cell, (list, tuple)) and len(cell) > 0:
                cell = cell[0]
            termo = cell.getPlainText().strip() if hasattr(cell, "getPlainText") else str(cell).strip()
            if termo in TERMOS_GLOSSARIO_FALTAS:
                termos_encontrados.add(termo)
            else:
                novas_linhas.append(row)

        if len(termos_encontrados) != len(TERMOS_GLOSSARIO_FALTAS):
            faltantes = set(TERMOS_GLOSSARIO_FALTAS) - termos_encontrados
            raise RuntimeError(f"O app mudou: termos do glossário não encontrados: {faltantes}")

        nova_tabela = Table(novas_linhas, colWidths=getattr(tabela_orig, "_colWidths", None))
        # TableStyle replicado literalmente de core/relatorios.py:1858-1864
        nova_tabela.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef1f7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story[idx_tabela_gloss] = nova_tabela

        # 2. Encontra a numeração do último H1
        max_h1 = 0
        for f in story:
            if isinstance(f, Paragraph) and getattr(f, "style", None) and getattr(f.style, "name", None) == 'H1Sumario':
                texto = f.getPlainText()
                partes = texto.split(".")
                if partes and partes[0].isdigit():
                    max_h1 = max(max_h1, int(partes[0]))

        n_secao = max_h1 + 1
        titulo_secao = f"{n_secao}. Acompanhamento Discente — DAE (Estradas + Trânsito)"

        # 3. Apontamento de frequência antes do H2 "Visualizações Gráficas" (D4)
        idx_h2_graficos = None
        for i, f in enumerate(story):
            if (
                isinstance(f, Paragraph)
                and getattr(f, "style", None)
                and getattr(f.style, "name", None) == "H2Sumario"
                and f.getPlainText().strip().endswith("Visualizações Gráficas")
            ):
                idx_h2_graficos = i
                break

        if idx_h2_graficos is None:
            raise RuntimeError("O app mudou: H2Sumario de Visualizações Gráficas não foi encontrado no story.")

        style_corpo = ParagraphStyle(
            "CorpoPrevia",
            parent=getSampleStyleSheet()["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=14,
            spaceBefore=4,
            spaceAfter=6,
        )
        p_apontamento = Paragraph(APONTAMENTO_FREQUENCIA.format(n=n_secao), style_corpo)
        story.insert(idx_h2_graficos, p_apontamento)

        # 4. Registra e adiciona H1 e nota explicativa da DAE ao story (D4)
        style_h1 = ParagraphStyle(
            "H1Sumario",
            parent=getSampleStyleSheet()["h1"],
            fontName="Times-Bold",
            textColor=colors.HexColor("#002060"),
            spaceBefore=12,
            spaceAfter=8,
        )
        novo_h1 = Paragraph(titulo_secao, style_h1)
        p_nota = Paragraph(NOTA_SUBSTITUICAO_FALTAS, style_corpo)
        if registro is not None:
            registro.append(novo_h1)
            registro.append(p_nota)

        story.append(NextPageTemplate("principal"))
        story.append(PageBreak())
        story.append(novo_h1)
        story.append(p_nota)

        # 5. Renumera os H2 e adiciona os flowables da DAE
        h2_count = 1
        for f in flowables_dae:
            if isinstance(f, Paragraph) and getattr(f, "style", None) and getattr(f.style, "name", None) == 'H2Sumario':
                novo_p = Paragraph(f"{n_secao}.{h2_count} {f.getPlainText()}", f.style)
                story.append(novo_p)
                if registro is not None:
                    registro.append(novo_p)
                h2_count += 1
            else:
                f_copy = copy.deepcopy(f)
                story.append(f_copy)
                if registro is not None:
                    registro.append(f_copy)

        # 6. Faixa de prévia no rodapé de todas as páginas (D5), além do
        # cabeçalho institucional original do app
        for pt in self.pageTemplates:
            orig_onPage = pt.onPage

            def novo_onPage(canvas, doc, _orig=orig_onPage):
                if _orig:
                    _orig(canvas, doc)
                canvas.saveState()
                largura, _altura = canvas._pagesize
                canvas.setFillColor(colors.HexColor("#fdecea"))
                canvas.rect(0, 0, largura, 0.95 * cm, fill=1, stroke=0)
                canvas.setFillColor(colors.HexColor("#7a3030"))
                canvas.setFont("Times-Roman", 6.5)
                canvas.drawCentredString(largura / 2.0, 0.32 * cm, TEXTO_FAIXA_PREVIA)
                canvas.restoreState()

            pt.onPage = novo_onPage

        # 7. Captura opcional do story completo para testes (D5)
        if registro_story is not None:
            registro_story.extend(story)

        return original_multiBuild(self, story, **kwargs)

    with mock.patch.object(_DocComSumario, "multiBuild", new=patched_multiBuild):
        yield


def gerar_previa(
    caminhos_mapas=None,
    caminho_dae=None,
    caminho_ch=None,
    pasta_saida=None,
    cenario="A",
    sabado_reproduz=None,
    registro=None,
    registro_story=None,
) -> list[Path]:
    if caminhos_mapas is None:
        caminhos_mapas = [
            p for p in sorted(PASTA_DADOS.glob("*.xls"))
            if p.is_file() and not p.name.startswith("~$")
        ]
        
    classif = classificar_mapas(caminhos_mapas)
    
    # Exige DET ou um curso só
    eh_det_mapas = eh_det(classif)
    cursos_encontrados = [k for k, v in classif.items() if v]
    
    if not eh_det_mapas and len(cursos_encontrados) != 1:
        raise ValueError("gerar_previa exige mapas do DET (Estradas e Trânsito) ou de apenas um curso.")

    cruzamento_arg = None
    if eh_det_mapas:
        cruzamento_arg = carregar_det(classif)
    else:
        cruzamento_arg = classif[cursos_encontrados[0]]
        
    df_dae = prototipo_pdf.obter_dados_sinteticos() if caminho_dae is None else prototipo_pdf.carregar_dae(caminho_dae)
    
    # Descobre bimestres a partir dos mapas
    todos_mapas = []
    for lista in classif.values():
        todos_mapas.extend(lista)
    
    bimestres_set = set()
    for mapa in todos_mapas:
        df_mapa = manipulacao._ler_xls_bruto(mapa)
        meta = manipulacao.extrair_metadados(df_mapa)
        if meta and meta.get("bimestre_num"):
            bimestres_set.add(meta["bimestre_num"])
            
    bimestres = sorted(list(bimestres_set)) if bimestres_set else [1, 2, 3]

    flowables_dae = prototipo_pdf.montar_flowables(
        df=df_dae,
        bimestres=bimestres,
        curso="TÉCNICO EM TRÂNSITO",
        cenario=cenario,
        sabado_reproduz=sabado_reproduz,
        cruzamento=cruzamento_arg,
        ch_efetiva=caminho_ch,
        layout="app"
    )

    saida_dir = Path(pasta_saida) if pasta_saida else _DIR_DAE / "saida" / "preview"
    saida_dir.mkdir(parents=True, exist_ok=True)
    
    pdfs_gerados = []
    
    # Pipeline do app replicado (sem IA e sem e-mail): ver _gerar_pdf_para_conjunto
    # em app.py:211-231 e o laço de processar_e_enviar em app.py:291-310 (D7).
    grupos_processar = []
    if eh_det_mapas:
        conjuntos_tt_validos = [c for c in cruzamento_arg.transito if not c[0].empty]
        conjuntos_est_validos = [c for c in cruzamento_arg.estradas if not c[0].empty]
        if conjuntos_tt_validos:
            grupos_processar.append(("Trânsito", conjuntos_tt_validos))
        if conjuntos_est_validos:
            grupos_processar.append(("Estradas", conjuntos_est_validos))
    else:
        curso = cursos_encontrados[0]
        conjuntos = manipulacao.processar_multiplos_bimestres(classif[curso])
        conjuntos_validos = [c for c in conjuntos if not c[0].empty]
        if conjuntos_validos:
            grupos_processar.append((curso, conjuntos_validos))
            
    for curso, grupo in grupos_processar:
        conjunto_recente = grupo[-1]
        df_notas, df_faltas, disciplinas_dict, metadados = conjunto_recente
        nome_curso_meta = metadados.get('curso_amigavel') or metadados.get('curso') or curso
        
        estat = core.relatorios.calcular_estatisticas(
            df_notas, disciplinas_dict, df_faltas=df_faltas, metadados=metadados)
            
        figuras = core.relatorios.gerar_todos_graficos(
            df_notas, nome_curso_meta, disciplinas_dict, estat, df_faltas=df_faltas)
            
        estatisticas_multibimestre = core.relatorios.calcular_estatisticas_multibimestre(grupo)
        
        bimestres_grupo = [c[3].get('bimestre_num') for c in grupo]
        bim = _fmt_bimestres(bimestres_grupo) if bimestres_grupo else (metadados.get('bimestre_num') or 'X')
        serie = metadados.get('serie')
        serie_tag = f"_{serie}aserie" if serie else ""
        nome_arquivo = f"relatorio_{_slug(nome_curso_meta)}{serie_tag}_bim{bim}_previa_dae.pdf"
        caminho_pdf = saida_dir / nome_arquivo
        
        logo = str(_LOGO_PADRAO) if _LOGO_PADRAO.exists() else None
        primeiro_curso = curso == grupos_processar[0][0]
        with injetar_secao_dae(
            flowables_dae,
            registro=registro if primeiro_curso else None,
            registro_story=registro_story if primeiro_curso else None,
        ):
            buffer = criar_relatorio_pdf(
                nome_curso=nome_curso_meta,
                estatisticas=estatisticas_sem_analise_faltas(estat),
                figuras=figuras_sem_analise_faltas(figuras),
                logo_path=logo,
                estatisticas_multibimestre=estatisticas_multibimestre,
            )
            
        caminho_pdf.write_bytes(buffer.getvalue())
        pdfs_gerados.append(caminho_pdf)
        
    return pdfs_gerados

def _criar_parser():
    parser = argparse.ArgumentParser(description="Gera prévia DAE acoplada ao PDF principal do app.")
    parser.add_argument("--mapas", nargs="*", default=None)
    parser.add_argument("--dae", default=None)
    parser.add_argument("--ch-efetiva", default=None)
    parser.add_argument("--saida", default=None)
    parser.add_argument("--cenario", default="A")
    parser.add_argument("--sabado-reproduz", default=None)
    return parser

if __name__ == "__main__":
    parser = _criar_parser()
    args = parser.parse_args()
    try:
        gerar_previa(
            caminhos_mapas=args.mapas,
            caminho_dae=args.dae,
            caminho_ch=args.ch_efetiva,
            pasta_saida=args.saida,
            cenario=args.cenario,
            sabado_reproduz=args.sabado_reproduz
        )
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

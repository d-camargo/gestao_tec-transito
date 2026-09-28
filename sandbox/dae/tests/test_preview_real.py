"""Testes de validação da prévia DAE acoplada ao relatório com dados reais.

Executa gerar_previa() sem argumentos com pasta_saida=tmp_path e confere:
- 2 PDFs gerados com os nomes oficiais do app:
  * relatorio_trânsito_2aserie_bim1_previa_dae.pdf
  * relatorio_estradas_2aserie_bim1_previa_dae.pdf
- Cada PDF com tamanho > 100 KB;
- No registro, a seção DAE traz:
  * "45"
  * "Estradas 24"/"Trânsito 21" (ou formato "Estradas: 24" / "Trânsito: 21")
  * "divergências com o calendário oficial: 0"
  * "DEMONSTRAÇÃO"
  * Nenhuma matrícula real (nenhuma sequência de 11 dígitos que esteja nos mapas reais);
- No registro_story:
  * Sem 2.3/2.4 de faltas e sem "P90 de Faltas";
  * Glossário sem os 3 termos de faltas;
  * Apontamento para o capítulo DAE presente antes de "Visualizações Gráficas";
  * "Visualizações Gráficas" numerado como 2.3;
- Mensagens de assert contêm exclusivamente contagens/rótulos agregados (LGPD).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import pytest
from reportlab.platypus import Paragraph, Table

import core.manipulacao as manipulacao
from det import (
    CAMINHO_CH_EFETIVA_PADRAO,
    CAMINHO_ESTRADAS_PADRAO,
    CAMINHO_TRANSITO_PADRAO,
)
from preview_relatorio import (
    NOTA_SUBSTITUICAO_FALTAS,
    TERMOS_GLOSSARIO_FALTAS,
    gerar_previa,
)
from prototipo_pdf import extrair_texto_flowables

pytestmark = pytest.mark.skipif(
    not (
        CAMINHO_ESTRADAS_PADRAO.exists()
        and CAMINHO_TRANSITO_PADRAO.exists()
        and CAMINHO_CH_EFETIVA_PADRAO.exists()
    ),
    reason="Mapas reais Estradas/Trânsito ou CH efetiva não encontrados em sandbox/dae/dados/.",
)


def _texto_celula(cell: Any) -> str:
    while isinstance(cell, (list, tuple)) and len(cell) > 0:
        cell = cell[0]
    return cell.getPlainText().strip() if hasattr(cell, "getPlainText") else str(cell).strip()


def test_gerar_previa_dados_reais(tmp_path: Path) -> None:
    """Verifica geração da prévia DAE acoplada usando dados reais."""
    registro: list = []
    registro_story: list = []

    # Executa gerar_previa() sem argumentos de arquivos (usa dados reais padrão)
    pdfs = gerar_previa(
        pasta_saida=tmp_path,
        registro=registro,
        registro_story=registro_story,
    )

    # 1. Confere 2 PDFs gerados com os nomes especificados
    assert len(pdfs) == 2, f"Esperado 2 PDFs gerados, obtido {len(pdfs)}"

    nomes_esperados = {
        "relatorio_trânsito_2aserie_bim1_previa_dae.pdf",
        "relatorio_estradas_2aserie_bim1_previa_dae.pdf",
    }
    nomes_obtidos = {p.name for p in pdfs}
    assert nomes_obtidos == nomes_esperados, (
        f"Esperados {len(nomes_esperados)} arquivos específicos, obtidos {len(nomes_obtidos)}"
    )

    # 2. Confere que cada PDF tem tamanho > 100 KB
    for p in pdfs:
        tamanho = p.stat().st_size
        assert tamanho > 100 * 1024, (
            f"Arquivo com tamanho {tamanho} bytes não excede 100 KB"
        )

    # 3. Confere flowables e conteúdo textual da seção DAE no registro
    assert len(registro) > 0, f"Esperado flowables no registro, obtido {len(registro)}"
    texto_dae = extrair_texto_flowables(registro, incluir_rodape=False)

    # Traz "45"
    assert "45" in texto_dae, "Esperado número '45' no registro da seção DAE"

    # Traz "Estradas 24"/"Trânsito 21" ou "Estradas: 24"/"Trânsito: 21"
    tem_estradas_24 = "Estradas: 24" in texto_dae or "Estradas 24" in texto_dae
    tem_transito_21 = "Trânsito: 21" in texto_dae or "Trânsito 21" in texto_dae
    assert tem_estradas_24, "Esperada contagem de 24 alunos para Estradas na seção DAE"
    assert tem_transito_21, "Esperada contagem de 21 alunos para Trânsito na seção DAE"

    # Traz "divergências com o calendário oficial: 0"
    assert "divergências com o calendário oficial: 0" in texto_dae.lower(), (
        "Esperada menção a 'divergências com o calendário oficial: 0' na seção DAE"
    )

    # Traz "DEMONSTRAÇÃO"
    assert "DEMONSTRAÇÃO" in texto_dae.upper(), (
        "Esperada menção a 'DEMONSTRAÇÃO' na seção DAE"
    )

    # 4. Confere que nenhuma matrícula real de 11 dígitos dos mapas aparece no registro
    df_est = manipulacao._ler_xls_bruto(CAMINHO_ESTRADAS_PADRAO)
    df_tra = manipulacao._ler_xls_bruto(CAMINHO_TRANSITO_PADRAO)
    mats_est = set(re.findall(r"\b\d{11}\b", df_est.to_string()))
    mats_tra = set(re.findall(r"\b\d{11}\b", df_tra.to_string()))
    todas_mats_reais = mats_est | mats_tra
    assert len(todas_mats_reais) > 0, f"Esperadas matrículas nos mapas reais, obtido {len(todas_mats_reais)}"

    mats_no_registro = set(re.findall(r"\b\d{11}\b", texto_dae))
    vazadas = mats_no_registro & todas_mats_reais
    assert len(vazadas) == 0, f"Encontradas {len(vazadas)} matrículas reais vazadas na seção DAE"

    # 5. Confere story real capturado via registro_story (D3, D4, D5)
    assert len(registro_story) > 0, (
        f"Esperado flowables no registro_story, obtido {len(registro_story)}"
    )

    # 5.1. Glossário: não tem nenhum dos 3 termos e preserva termos fundamentais
    idx_h1_glossario = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph)
        and getattr(f, "style", None)
        and getattr(f.style, "name", None) == "H1Sumario"
        and f.getPlainText().strip().endswith("Glossário")
    )
    tabela_gloss = next(
        f for f in registro_story[idx_h1_glossario + 1:]
        if isinstance(f, Table)
    )
    termos_glossario = [
        _texto_celula(row[0])
        for row in tabela_gloss._cellvalues
    ]
    for termo in TERMOS_GLOSSARIO_FALTAS:
        assert termo not in termos_glossario, (
            f"Termo proibido do glossário presente: '{termo}'"
        )
    assert "Média" in termos_glossario, "Termo 'Média' ausente no glossário"
    assert "P90 (Percentil 90)" in termos_glossario, (
        "Termo 'P90 (Percentil 90)' ausente no glossário"
    )
    assert "Faltas Acumuladas" in termos_glossario, (
        "Termo 'Faltas Acumuladas' ausente no glossário"
    )

    # 5.2. Apontamento para o capítulo DAE presente antes de Visualizações Gráficas
    idx_h1_dae = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph)
        and getattr(f, "style", None)
        and getattr(f.style, "name", None) == "H1Sumario"
        and "Acompanhamento Discente — DAE" in f.getPlainText()
    )
    n_secao = int(registro_story[idx_h1_dae].getPlainText().split(".")[0])
    texto_apontamento_esperado = f"está no capítulo {n_secao} (Acompanhamento Discente — DAE)"

    idx_h2_graficos = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph)
        and getattr(f, "style", None)
        and getattr(f.style, "name", None) == "H2Sumario"
        and f.getPlainText().strip().endswith("Visualizações Gráficas")
    )
    idx_apontamento = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph) and texto_apontamento_esperado in f.getPlainText()
    )
    assert idx_apontamento < idx_h2_graficos, (
        "Apontamento deve anteceder o H2 de Visualizações Gráficas"
    )
    assert idx_apontamento == idx_h2_graficos - 1, (
        "Apontamento deve preceder imediatamente o H2 de Visualizações Gráficas"
    )

    # 5.3. Visualizações Gráficas como 2.3
    h2_graficos = [
        f.getPlainText()
        for f in registro_story
        if isinstance(f, Paragraph)
        and getattr(f, "style", None)
        and f.style.name == "H2Sumario"
        and "Visualizações Gráficas" in f.getPlainText()
    ]
    assert len(h2_graficos) == 1, (
        f"Esperado 1 H2 de Visualizações Gráficas, obtido {len(h2_graficos)}"
    )
    assert h2_graficos[0] == "2.3 Visualizações Gráficas", (
        f"Esperado título '2.3 Visualizações Gráficas', obtido '{h2_graficos[0]}'"
    )

    # 5.4. Sem seções 2.3/2.4 de faltas
    h2_faltas_cap2 = [
        f.getPlainText()
        for f in registro_story
        if isinstance(f, Paragraph)
        and getattr(f, "style", None)
        and getattr(f.style, "name", None) == "H2Sumario"
        and (f.getPlainText().startswith("2.3") or f.getPlainText().startswith("2.4"))
        and "faltas" in f.getPlainText().lower()
    ]
    assert len(h2_faltas_cap2) == 0, (
        f"Seções de faltas encontradas no capítulo 2: {len(h2_faltas_cap2)}"
    )

    # 5.5. Sem termos proibidos em nenhum flowable do story (sem P90 de Faltas, etc.)
    termos_proibidos = [
        "Alunos com Faltas Acima da Média",
        "Resumo de Faltas por Disciplina",
        "P90 de Faltas",
        "Quadrantes",
    ]
    for f in registro_story:
        if isinstance(f, Paragraph):
            txt = f.getPlainText()
            for termo in termos_proibidos:
                assert termo not in txt, f"Termo proibido '{termo}' encontrado no story"
        elif isinstance(f, Table):
            for row in getattr(f, "_cellvalues", []):
                for cell in row:
                    txt = _texto_celula(cell)
                    for termo in termos_proibidos:
                        assert termo not in txt, (
                            f"Termo proibido '{termo}' encontrado no story"
                        )

    # 5.6. NOTA_SUBSTITUICAO_FALTAS logo após o H1 DAE
    assert registro_story[idx_h1_dae + 1].getPlainText() == NOTA_SUBSTITUICAO_FALTAS, (
        "Nota explicativa de substituição de faltas ausente logo após H1 DAE"
    )

    # 5.7. Coluna "Faltas" na tabela 2.1 continua presente (D1)
    idx_h2_2_1 = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph)
        and getattr(f, "style", None)
        and getattr(f.style, "name", None) == "H2Sumario"
        and "Desempenho e Frequência por Aluno" in f.getPlainText()
    )
    tabela_2_1 = next(
        f for f in registro_story[idx_h2_2_1 + 1:]
        if isinstance(f, Table)
    )
    cabecalho_2_1 = [
        _texto_celula(cell)
        for cell in tabela_2_1._cellvalues[0]
    ]
    assert "Faltas" in cabecalho_2_1, (
        "Coluna 'Faltas' ausente no cabeçalho da tabela 2.1"
    )

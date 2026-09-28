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
- Mensagens de assert contêm exclusivamente contagens agregadas (LGPD).
"""

from __future__ import annotations

from pathlib import Path
import re
import pytest

import core.manipulacao as manipulacao
from det import (
    CAMINHO_CH_EFETIVA_PADRAO,
    CAMINHO_ESTRADAS_PADRAO,
    CAMINHO_TRANSITO_PADRAO,
)
from preview_relatorio import gerar_previa
from prototipo_pdf import extrair_texto_flowables

pytestmark = pytest.mark.skipif(
    not (
        CAMINHO_ESTRADAS_PADRAO.exists()
        and CAMINHO_TRANSITO_PADRAO.exists()
        and CAMINHO_CH_EFETIVA_PADRAO.exists()
    ),
    reason="Mapas reais Estradas/Trânsito ou CH efetiva não encontrados em sandbox/dae/dados/.",
)


def test_gerar_previa_dados_reais(tmp_path: Path) -> None:
    """Verifica geração da prévia DAE acoplada usando dados reais."""
    registro: list = []

    # Executa gerar_previa() sem argumentos de arquivos (usa dados reais padrão)
    pdfs = gerar_previa(pasta_saida=tmp_path, registro=registro)

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

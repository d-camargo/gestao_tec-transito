import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import Mock
from tests.conftest import mapa_sintetico

from core.relatorios import _DocComSumario, criar_relatorio_pdf
from reportlab.platypus import Paragraph
from sandbox.dae.preview_relatorio import gerar_previa, injetar_secao_dae

def test_gerar_previa_com_dados_sinteticos(tmp_path, monkeypatch, planilha_ch_sintetica, caminho_xlsx):
    import core.manipulacao as manipulacao
    from core.relatorios import _DocComSumario
    
    # 1. Mapas sintéticos EST+TT (monkeypatch do passo 2)
    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST.2A",
        disciplinas={"TOP": "TOPOGRAFIA"},
        alunos=[
            ("20261010991", "Ana Falsa", {"TOP": 15.0}, {"TOP": 0}),
            ("20261010993", "Carlos Falso", {"TOP": 18.0}, {"TOP": 0}),
        ],
    )
    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TRA.2A",
        disciplinas={"TRA": "TRANSPORTES"},
        alunos=[
            ("20261010991", "Ana Falsa", {"TRA": 16.0}, {"TRA": 0}),
            ("20261010992", "Bruno Falso", {"TRA": 17.0}, {"TRA": 0}),
        ],
    )

    original_ler = manipulacao._ler_xls_bruto

    def _mock_ler(arquivo_xls):
        if isinstance(arquivo_xls, pd.DataFrame):
            return arquivo_xls
        nome = Path(str(arquivo_xls)).name.lower()
        if "est" in nome:
            return df_est
        if "tra" in nome or "tt" in nome:
            return df_tt
        return original_ler(arquivo_xls)

    monkeypatch.setattr(manipulacao, "_ler_xls_bruto", _mock_ler)

    mapa_est = tmp_path / "Estradas_2026.xls"
    mapa_tra = tmp_path / "Transito_2026.xls"
    mapa_est.touch()
    mapa_tra.touch()
    
    monkeypatch.setattr("sandbox.dae.preview_relatorio.PASTA_DADOS", tmp_path)

    registro = []
    original_multiBuild = _DocComSumario.multiBuild

    # Act
    pdfs_gerados = gerar_previa(
        caminhos_mapas=[mapa_est, mapa_tra],
        caminho_dae=caminho_xlsx,
        caminho_ch=planilha_ch_sintetica,
        pasta_saida=tmp_path,
        registro=registro
    )
    
    # Assert
    assert len(pdfs_gerados) == 2
    
    pdf_nomes = [p.name for p in pdfs_gerados]
    assert any(n.endswith("_previa_dae.pdf") and "trânsito" in n.lower() for n in pdf_nomes)
    assert any(n.endswith("_previa_dae.pdf") and "estradas" in n.lower() for n in pdf_nomes)
    
    for p in pdfs_gerados:
        with open(p, "rb") as f:
            assert f.read().startswith(b"%PDF")
            
    # "no `registro`, exatamente um H1Sumario com "Acompanhamento Discente — DAE (Estradas + Trânsito)"
    # "e número = nº de H1Sumario do app + 1"
    h1s = [f for f in registro if getattr(f, 'style', None) and f.style.name == 'H1Sumario']
    assert len(h1s) == 1
    titulo_h1 = h1s[0].getPlainText()
    assert "Acompanhamento Discente — DAE (Estradas + Trânsito)" in titulo_h1
    
    # O número está no começo: "N. Acompanhamento..."
    n_secao = int(titulo_h1.split(".")[0])
    
    h2s = [f for f in registro if getattr(f, 'style', None) and f.style.name == 'H2Sumario']
    for h2 in h2s:
        assert h2.getPlainText().startswith(f"{n_secao}.")
        
    from sandbox.dae.prototipo_pdf import extrair_texto_flowables
    texto_total_secao = extrair_texto_flowables(registro, incluir_rodape=False)
    assert "DEMONSTRAÇÃO" in texto_total_secao.upper()
    assert "N/C (Nada consta)" in texto_total_secao
    assert "LGPD" in texto_total_secao
    assert "Pé-de-Meia" in texto_total_secao
    
    # Um mês do período (D6)
    meses_possiveis = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
    assert any(m in texto_total_secao for m in meses_possiveis)
    
    # E não contém nenhum nome/matrícula dos mapas sintéticos
    assert "20261010991" not in texto_total_secao
    assert "Ana Falsa" not in texto_total_secao
    assert "20261010992" not in texto_total_secao
    assert "Bruno Falso" not in texto_total_secao
    assert "20261010993" not in texto_total_secao
    assert "Carlos Falso" not in texto_total_secao

    # Depois do with, core.relatorios._DocComSumario.multiBuild é o objeto original
    assert _DocComSumario.multiBuild is original_multiBuild


def test_injetar_secao_dae_restaura_multibuild():
    """O patch vive só dentro do with; fora dele criar_relatorio_pdf não injeta nada."""
    original = _DocComSumario.multiBuild
    with injetar_secao_dae([]):
        assert _DocComSumario.multiBuild is not original
    assert _DocComSumario.multiBuild is original

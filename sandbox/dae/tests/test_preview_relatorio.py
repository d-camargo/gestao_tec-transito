import io
from unittest import mock
from pathlib import Path
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, Table

import core.relatorios
from core.relatorios import _DocComSumario, criar_relatorio_pdf
from tests.conftest import mapa_sintetico
from tests.test_pdf_ponta_a_ponta import _turma
from sandbox.dae.preview_relatorio import (
    APONTAMENTO_FREQUENCIA,
    CHAVES_FIGURAS_FALTAS,
    NOTA_SUBSTITUICAO_FALTAS,
    TERMOS_GLOSSARIO_FALTAS,
    estatisticas_sem_analise_faltas,
    figuras_sem_analise_faltas,
    gerar_previa,
    injetar_secao_dae,
)

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
    registro_story = []
    original_multiBuild = _DocComSumario.multiBuild

    # Act
    pdfs_gerados = gerar_previa(
        caminhos_mapas=[mapa_est, mapa_tra],
        caminho_dae=caminho_xlsx,
        caminho_ch=planilha_ch_sintetica,
        pasta_saida=tmp_path,
        registro=registro,
        registro_story=registro_story,
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

    def _texto_celula(cell):
        while isinstance(cell, (list, tuple)) and len(cell) > 0:
            cell = cell[0]
        return cell.getPlainText().strip() if hasattr(cell, "getPlainText") else str(cell).strip()

    # Assertivas sobre registro_story (D3, D4, D5)
    assert len(registro_story) > 0

    # 1. Glossário: não tem nenhum dos 3 termos e ainda tem "Média", "P90 (Percentil 90)" e "Faltas Acumuladas"
    idx_h1_glossario = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph) and getattr(f, "style", None) and getattr(f.style, "name", None) == "H1Sumario" and f.getPlainText().strip().endswith("Glossário")
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
        assert termo not in termos_glossario
    assert "Média" in termos_glossario
    assert "P90 (Percentil 90)" in termos_glossario
    assert "Faltas Acumuladas" in termos_glossario

    # 2. Apontamento: o texto "está no capítulo N (Acompanhamento Discente — DAE)" aparece com N = número do H1 DAE e antes do H2 "Visualizações Gráficas"
    idx_h2_graficos = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph) and getattr(f, "style", None) and getattr(f.style, "name", None) == "H2Sumario" and f.getPlainText().strip().endswith("Visualizações Gráficas")
    )
    texto_apontamento_esperado = f"está no capítulo {n_secao} (Acompanhamento Discente — DAE)"
    idx_apontamento = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph) and texto_apontamento_esperado in f.getPlainText()
    )
    assert idx_apontamento < idx_h2_graficos
    assert idx_apontamento == idx_h2_graficos - 1

    # 3. NOTA_SUBSTITUICAO_FALTAS aparece logo após o H1 DAE
    idx_h1_dae = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph) and getattr(f, "style", None) and getattr(f.style, "name", None) == "H1Sumario" and "Acompanhamento Discente — DAE" in f.getPlainText()
    )
    assert registro_story[idx_h1_dae + 1].getPlainText() == NOTA_SUBSTITUICAO_FALTAS

    # 4. Nenhum texto do story contém os termos proibidos
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
                assert termo not in txt, f"Termo proibido '{termo}' encontrado no Paragraph: {txt}"
        elif isinstance(f, Table):
            for row in getattr(f, "_cellvalues", []):
                for cell in row:
                    txt = _texto_celula(cell)
                    for termo in termos_proibidos:
                        assert termo not in txt, f"Termo proibido '{termo}' encontrado na Table: {txt}"

    # 5. A coluna "Faltas" de 2.1 continua presente (D1)
    idx_h2_2_1 = next(
        i for i, f in enumerate(registro_story)
        if isinstance(f, Paragraph) and getattr(f, "style", None) and getattr(f.style, "name", None) == "H2Sumario" and "Desempenho e Frequência por Aluno" in f.getPlainText()
    )
    tabela_2_1 = next(
        f for f in registro_story[idx_h2_2_1 + 1:]
        if isinstance(f, Table)
    )
    cabecalho_2_1 = [
        _texto_celula(cell)
        for cell in tabela_2_1._cellvalues[0]
    ]
    assert "Faltas" in cabecalho_2_1

    # Depois do with, core.relatorios._DocComSumario.multiBuild é o objeto original
    assert _DocComSumario.multiBuild is original_multiBuild


def test_injetar_secao_dae_restaura_multibuild():
    """O patch vive só dentro do with; fora dele criar_relatorio_pdf não injeta nada."""
    original = _DocComSumario.multiBuild
    with injetar_secao_dae([]):
        assert _DocComSumario.multiBuild is not original
    assert _DocComSumario.multiBuild is original


def test_estatisticas_sem_analise_faltas():
    """Com faltas_disponiveis=True, devolve cópia com False mantendo original True e demais chaves idênticas (is)."""
    orig = {
        "faltas_disponiveis": True,
        "media_geral": 14.5,
        "disciplinas": ["MAT", "FIS"],
        "metadados": {"turma": "TRA.1A"},
    }
    resultado = estatisticas_sem_analise_faltas(orig)

    assert resultado is not orig
    assert resultado["faltas_disponiveis"] is False
    assert orig["faltas_disponiveis"] is True

    for k in orig:
        if k != "faltas_disponiveis":
            assert resultado[k] is orig[k]


def test_figuras_sem_analise_faltas():
    """figuras_sem_analise_faltas tira exatamente as 3 chaves e mantém as outras, fechando as figuras removidas."""
    fig_dist = plt.figure()
    fig_med = plt.figure()
    fig_f1 = plt.figure()
    fig_f2 = plt.figure()
    fig_f3 = plt.figure()

    figuras = {
        "distribuicao_geral": fig_dist,
        "media_disciplina": fig_med,
        "faltas_total_aluno": fig_f1,
        "faltas_boxplot_disciplina": fig_f2,
        "dispersao_notas_faltas": fig_f3,
    }

    with mock.patch("matplotlib.pyplot.close") as mock_close:
        resultado = figuras_sem_analise_faltas(figuras)

        # Original permanece inalterado
        assert len(figuras) == 5
        for k in CHAVES_FIGURAS_FALTAS:
            assert k in figuras

        # Exatamente as 3 chaves foram removidas e as outras mantidas com o mesmo objeto
        assert set(resultado.keys()) == {"distribuicao_geral", "media_disciplina"}
        assert resultado["distribuicao_geral"] is fig_dist
        assert resultado["media_disciplina"] is fig_med

        # plt.close chamado para cada figura removida
        chamadas = [call.args[0] for call in mock_close.call_args_list]
        assert fig_f1 in chamadas
        assert fig_f2 in chamadas
        assert fig_f3 in chamadas
        assert fig_dist not in chamadas
        assert fig_med not in chamadas

    plt.close(fig_dist)
    plt.close(fig_med)
    plt.close(fig_f1)
    plt.close(fig_f2)
    plt.close(fig_f3)


def test_criar_relatorio_pdf_sem_analise_faltas():
    """criar_relatorio_pdf com versões filtradas omite faltas e numera Visualizações Gráficas como 2.3."""
    df_notas, df_faltas, disciplinas, metadados = _turma(1)
    estat = core.relatorios.calcular_estatisticas(
        df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados
    )
    assert estat.get("faltas_disponiveis") is True

    figuras = core.relatorios.gerar_todos_graficos(
        df_notas, "Trânsito", disciplinas, estat, df_faltas=df_faltas
    )
    assert all(k in figuras for k in CHAVES_FIGURAS_FALTAS)

    estat_filtrado = estatisticas_sem_analise_faltas(estat)
    figuras_filtradas = figuras_sem_analise_faltas(figuras)

    with mock.patch.object(core.relatorios._DocComSumario, "multiBuild") as mock_mb:
        core.relatorios.criar_relatorio_pdf(
            nome_curso="Trânsito",
            estatisticas=estat_filtrado,
            figuras=figuras_filtradas,
        )
        story = mock_mb.call_args.args[0]

    termos_proibidos = [
        "Alunos com Faltas Acima da Média",
        "Resumo de Faltas por Disciplina",
        "P90 de Faltas",
    ]

    for f in story:
        if isinstance(f, Paragraph):
            txt = f.getPlainText()
            for termo in termos_proibidos:
                assert termo not in txt, f"Termo proibido '{termo}' encontrado no Paragraph: {txt}"
        elif isinstance(f, Table):
            for row in getattr(f, "_cellvalues", []):
                for cell in row:
                    txt = cell.getPlainText() if isinstance(cell, Paragraph) else str(cell)
                    for termo in termos_proibidos:
                        assert termo not in txt, f"Termo proibido '{termo}' encontrado na Table: {txt}"

    h2_graficos = [
        f.getPlainText()
        for f in story
        if isinstance(f, Paragraph)
        and getattr(f, "style", None)
        and f.style.name == "H2Sumario"
        and "Visualizações Gráficas" in f.getPlainText()
    ]
    assert len(h2_graficos) == 1
    assert h2_graficos[0] == "2.3 Visualizações Gráficas"


def test_injetar_secao_dae_sem_glossario_gera_erro():
    """Um story sem glossário passado ao wrapper deve lançar RuntimeError."""
    with pytest.raises(RuntimeError, match="(?i)glossário"):
        with injetar_secao_dae([]):
            _DocComSumario(io.BytesIO()).multiBuild([Paragraph("Teste", getSampleStyleSheet()["Normal"])])


def test_injetar_secao_dae_sem_termos_glossario_gera_erro():
    """Story com glossário sem os 3 termos esperados deve lançar RuntimeError."""
    style_h1 = ParagraphStyle("H1Sumario", fontName="Times-Bold")
    story = [
        Paragraph("1. Glossário", style_h1),
        Table([[Paragraph("Média", getSampleStyleSheet()["Normal"]), Paragraph("desc", getSampleStyleSheet()["Normal"])]]),
    ]
    with pytest.raises(RuntimeError, match="termos do glossário não encontrados"):
        with injetar_secao_dae([]):
            _DocComSumario(io.BytesIO()).multiBuild(story)


def test_injetar_secao_dae_sem_h2_graficos_gera_erro():
    """Story sem H2 Visualizações Gráficas deve lançar RuntimeError."""
    style_h1 = ParagraphStyle("H1Sumario", fontName="Times-Bold")
    linhas = [
        [Paragraph("<b>Média</b>", getSampleStyleSheet()["Normal"]), Paragraph("desc", getSampleStyleSheet()["Normal"])]
    ] + [
        [Paragraph(f"<b>{t}</b>", getSampleStyleSheet()["Normal"]), Paragraph("desc", getSampleStyleSheet()["Normal"])]
        for t in TERMOS_GLOSSARIO_FALTAS
    ]
    story = [
        Paragraph("1. Glossário", style_h1),
        Table(linhas),
    ]
    with pytest.raises(RuntimeError, match="Visualizações Gráficas"):
        with injetar_secao_dae([]):
            _DocComSumario(io.BytesIO()).multiBuild(story)

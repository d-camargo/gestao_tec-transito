"""Testes do gerador de protótipo PDF com destaques da DAE."""

from __future__ import annotations

from pathlib import Path
import re
import sys

from unittest.mock import patch

import pandas as pd
import pytest

from reportlab.platypus import KeepTogether, Paragraph, Table

_DIR_TESTS = Path(__file__).resolve().parent
_DIR_DAE = _DIR_TESTS.parent
_DIR_RAIZ = _DIR_DAE.parent.parent
for _p in (_DIR_DAE, _DIR_RAIZ):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from carregar import carregar_dae
from core.relatorios import COR_CABECALHO_TABELA, COR_TEXTO_CABECALHO_TABELA
from det import carregar_det, conjuntos_det
from prototipo_pdf import (
    LEGENDA_PROGRAMAS,
    NOTA_LGPD,
    NOTA_PE_DE_MEIA,
    TEXTO_DEMONSTRACAO,
    TEXTO_RODAPE_USO_INTERNO,
    _executar_cli,
    _largura,
    determinar_caminho_saida,
    extrair_texto_flowables,
    gerar_prototipo_pdf,
    montar_flowables,
    obter_dados_sinteticos,
)
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, Table
from tests.conftest import mapa_sintetico


def _obter_texto_teste(caminho_pdf: Path, flowables: list) -> str:
    """Extrai texto do PDF via pypdf se instalado; caso contrário, extrai dos flowables."""
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(caminho_pdf))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        return extrair_texto_flowables(flowables, incluir_rodape=True)


def test_gerar_prototipo_pdf_em_tmp_path(tmp_path: Path):
    """Gera em tmp_path, verifica início %PDF e palavras-chave obrigatórias no texto."""
    saida = tmp_path / "prototipo_destaque.pdf"
    resultado = gerar_prototipo_pdf(caminho_saida=saida)

    assert resultado.exists()
    assert resultado == saida

    conteudo_bytes = saida.read_bytes()
    assert conteudo_bytes.startswith(b"%PDF"), "O arquivo gerado deve começar com '%PDF'"

    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, bimestres=[1, 2, 3])
    texto = _obter_texto_teste(saida, flowables)

    # Exigências do passo 5:
    # Contém "uso interno", "LGPD", "Nada consta" e o nome de ao menos um mês do período
    assert "uso interno" in texto.lower()
    assert "LGPD" in texto
    assert "Nada consta" in texto

    meses_esperados = [
        "fevereiro",
        "marco",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
    ]
    assert any(m in texto.lower() for m in meses_esperados), (
        "O texto deve conter o nome de ao menos um mês do período de apuração."
    )


def test_ordem_e_conteudo_dos_flowables():
    """Valida a ordem estrita dos blocos especificados no passo 5 e C7:

    (i) bloco 'Período de apuração da frequência'
    (ii) bloco 'Calendário acadêmico e carga horária efetiva'
    (iii) trecho da tabela 2.1 com a coluna extra 'Prog.'
    (iv) quadro 'Tratamento de dados pessoais (LGPD)'
    (v) seção 'Estudantes acompanhados pela Assistência Estudantil'
    """
    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, bimestres=[1, 2, 3])

    # Coleta os textos dos títulos H2 para checar a ordem
    titulos = [
        f.text for f in flowables if isinstance(f, Paragraph) and "<b>" in f.text
    ]

    # Encontra os índices dos blocos
    idx_bloco_i = next(
        i for i, t in enumerate(titulos) if "Período de apuração" in t
    )
    idx_bloco_cal = next(
        i for i, t in enumerate(titulos) if "Calendário acadêmico" in t
    )
    idx_bloco_ii = next(
        i for i, t in enumerate(titulos) if "Tabela 2.1" in t or "Desempenho e Frequência" in t
    )
    idx_bloco_iii = next(
        i for i, t in enumerate(titulos) if "Tratamento de dados pessoais" in t
    )
    idx_bloco_iv = next(
        i for i, t in enumerate(titulos) if "Estudantes acompanhados" in t
    )

    assert idx_bloco_i < idx_bloco_cal < idx_bloco_ii < idx_bloco_iii < idx_bloco_iv, (
        f"A ordem dos blocos deve ser (i) < cal < (ii) < (iii) < (iv), obtido: "
        f"i={idx_bloco_i}, cal={idx_bloco_cal}, ii={idx_bloco_ii}, iii={idx_bloco_iii}, iv={idx_bloco_iv}"
    )


def test_nota_lgpd_e_comentario_rascunho():
    """Valida que NOTA_LGPD contém o comentário obrigatório de rascunho (D10)."""
    arquivo_codigo = _DIR_DAE / "prototipo_pdf.py"
    conteudo_codigo = arquivo_codigo.read_text(encoding="utf-8")

    padrao = r"#\s*RASCUNHO\s*—\s*texto sujeito à validação do Diego \(D10\)\s*\n\s*NOTA_LGPD\s*="
    assert re.search(padrao, conteudo_codigo), (
        "A constante NOTA_LGPD deve ser precedida pelo comentário: "
        "'# RASCUNHO — texto sujeito à validação do Diego (D10)'"
    )
    assert "LGPD" in NOTA_LGPD


def test_tabela_21_coluna_prog():
    """Verifica que o trecho da tabela 2.1 possui a coluna extra 'Prog.'."""
    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, bimestres=[1, 2, 3])

    tabelas = [f for f in flowables if isinstance(f, Table)]
    assert len(tabelas) >= 2

    tab_21 = next(
        t for t in tabelas if any("Prog" in getattr(cell, "text", str(cell)) for cell in t._cellvalues[0])
    )
    cabecalho = [cell.text for cell in tab_21._cellvalues[0]]
    assert "<b>Prog.</b>" in cabecalho or "Prog." in cabecalho

    # Verifica siglas de programas presentes
    texto_tabela = " ".join(
        cell.text for row in tab_21._cellvalues[1:] for cell in row
    )
    assert "PdM, BA/BP" in texto_tabela
    assert "BCE" in texto_tabela


def test_secao_estudantes_alerta_e_notas():
    """Verifica seção iv: frequências, destaque < 75%, legenda e nota Pé-de-Meia."""
    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, bimestres=[1, 2, 3])

    texto_geral = extrair_texto_flowables(flowables, incluir_rodape=False)

    # Verifica alerta < 75%
    assert "< 75%" in texto_geral or "&lt; 75%" in texto_geral

    # Verifica legenda com as três siglas
    assert "PdM" in LEGENDA_PROGRAMAS
    assert "BA/BP" in LEGENDA_PROGRAMAS
    assert "BCE" in LEGENDA_PROGRAMAS

    # Verifica nota exata sobre Pé-de-Meia e 'Nada consta'
    assert "Pé-de-Meia: N/C na planilha da DAE foi lido como ‘Nada consta’ e não marca PdM" in NOTA_PE_DE_MEIA
    assert NOTA_PE_DE_MEIA in texto_geral

    # D8: a seção (iv) lista somente alunos com programa — Carlos Lima (programas == [])
    # não pode aparecer; os demais sim.
    blocos_iv = [f for f in flowables if isinstance(f, KeepTogether)]
    assert len(blocos_iv) == 1, "Deve haver exatamente um bloco KeepTogether (seção iv)."
    texto_iv = extrair_texto_flowables(blocos_iv[0]._content, incluir_rodape=False)

    assert "Carlos Lima" not in texto_iv
    for nome_com_programa in ("Ana Silva", "Bruno Souza", "Daniela Rocha"):
        assert nome_com_programa in texto_iv

    # Na tabela 2.1 da seção (ii), todos os alunos devem continuar aparecendo
    tabelas = [f for f in flowables if isinstance(f, Table)]
    assert len(tabelas) >= 2
    tab_21 = next(
        t for t in tabelas if any("Prog" in getattr(cell, "text", str(cell)) for cell in t._cellvalues[0])
    )
    texto_21 = " ".join(
        cell.text for row in tab_21._cellvalues[1:] for cell in row
    )
    assert "Carlos Lima" in texto_21


def test_cores_cabecalho_importadas():
    """Verifica que as cores de cabeçalho vêm de core.relatorios."""
    from core.relatorios import (
        COR_CABECALHO_TABELA as COR_REF,
        COR_TEXTO_CABECALHO_TABELA as COR_TXT_REF,
    )
    assert COR_CABECALHO_TABELA == COR_REF
    assert COR_TEXTO_CABECALHO_TABELA == COR_TXT_REF


def test_determinar_caminho_saida_d11(tmp_path: Path):
    """Valida regra D11: saida/sintetico/ para dados sintéticos e saida/<AAAA-MM>/ para real."""
    # Sintético
    caminho_sint = determinar_caminho_saida(mes_ref="agosto", usando_dados_sinteticos=True)
    assert caminho_sint.parent.name == "sintetico"
    assert caminho_sint.name == "prototipo_destaque.pdf"

    # Real com mês de referência agosto -> 2026-08
    caminho_real = determinar_caminho_saida(mes_ref="agosto", usando_dados_sinteticos=False)
    assert caminho_real.parent.name == "2026-08"
    assert caminho_real.name == "prototipo_destaque.pdf"

    # Customizado via tmp_path
    custom = determinar_caminho_saida(
        mes_ref="agosto", usando_dados_sinteticos=True, caminho_saida=tmp_path / "meu_dir"
    )
    assert custom.parent == tmp_path / "meu_dir"
    assert custom.name == "prototipo_destaque.pdf"


def test_determinar_caminho_saida_sem_mes_ref_levanta_erro():
    """D11: sem mês de referência em dados reais, deve levantar ValueError em vez de inventar mês."""
    with pytest.raises(ValueError):
        determinar_caminho_saida(mes_ref=None, usando_dados_sinteticos=False)


def test_gerar_prototipo_com_arquivo_real(caminho_xlsx: Path, tmp_path: Path):
    """Testa geração a partir de arquivo .xlsx sintético usando caminho_dae."""
    saida = tmp_path / "saida_xlsx" / "destaque.pdf"
    resultado = gerar_prototipo_pdf(
        caminho_dae=caminho_xlsx,
        curso="TÉCNICO EM TRÂNSITO",
        bimestres=[1, 2, 3],
        caminho_saida=saida,
    )
    assert resultado.exists()
    assert resultado.read_bytes().startswith(b"%PDF")


def test_executar_cli(tmp_path: Path):
    """Testa interface CLI com flags --bimestres e --saida."""
    saida = tmp_path / "cli_out.pdf"
    codigo = _executar_cli(["--bimestres", "1,2,3", "--saida", str(saida)])
    assert codigo == 0
    assert saida.exists()
    assert saida.read_bytes().startswith(b"%PDF")

    # Bimestres inválidos devem retornar código de erro 1
    codigo_err = _executar_cli(["--bimestres", "abc,def"])
    assert codigo_err == 1


def test_calendario_e_cenario_default_a():
    """Verifica reflexo do calendário no protótipo com cenário A (default)."""
    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, bimestres=[1, 2, 3])
    texto = extrair_texto_flowables(flowables, incluir_rodape=False)

    assert "Calendário acadêmico" in texto
    assert "Deliberação CEPE" in texto
    assert "cenário A" in texto
    assert "105" in texto
    assert "114" in texto
    assert "140" in texto
    assert "152" in texto
    assert "22/05" in texto

    meses_esperados = [
        "fevereiro",
        "marco",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    ]
    assert any(m in texto.lower() for m in meses_esperados), (
        "Deve conter o nome de ao menos um mês do período (D6)"
    )


def test_calendario_e_cenario_real_estradas():
    """Verifica reflexo do calendário no protótipo com cenário REAL e curso Estradas."""
    df = obter_dados_sinteticos()
    flowables = montar_flowables(
        df,
        bimestres=[1, 2, 3],
        curso="TÉCNICO EM ESTRADAS",
        cenario="REAL",
    )
    texto = extrair_texto_flowables(flowables, incluir_rodape=False)

    assert "Calendário acadêmico" in texto
    assert "Deliberação CEPE" in texto
    assert "cenário REAL" in texto
    assert "23/05" in texto
    assert "70" in texto
    assert "78" in texto
    assert "22/05" in texto

    meses_esperados = [
        "fevereiro",
        "marco",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    ]
    assert any(m in texto.lower() for m in meses_esperados), (
        "Deve conter o nome de ao menos um mês do período (D6)"
    )


def test_executar_cli_cenarios(tmp_path: Path):
    """Testa CLI com --cenario REAL --curso 'TÉCNICO EM ESTRADAS' e erro com --cenario B."""
    saida = tmp_path / "cli_real_estradas.pdf"
    codigo = _executar_cli([
        "--cenario", "REAL",
        "--curso", "TÉCNICO EM ESTRADAS",
        "--saida", str(saida),
    ])
    assert codigo == 0
    assert saida.exists()
    assert saida.read_bytes().startswith(b"%PDF")

    with pytest.raises(SystemExit):
        _executar_cli(["--cenario", "B"])


def test_cruzamento_sem_mapas_sem_bloco_cruzamento():
    """Sem mapas, não deve incluir a seção 'Cruzamento com o mapa de turma' (C14)."""
    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, bimestres=[1, 2, 3])
    texto = extrair_texto_flowables(flowables, incluir_rodape=False)
    assert "Cruzamento com o mapa" not in texto


def test_cruzamento_mapas_dae_sintetica_pendente():
    """Com mapas e DAE sintética, deve exibir 'Cruzamento pendente' (C14)."""
    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 18.0}])
    df_faltas = pd.DataFrame([{"matricula": "20261010001", "nome": "Mock Aluno", "MAT": 10}])
    meta = {"bimestre_num": 1, "curso": "Técnico em Estradas"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    df = obter_dados_sinteticos()
    with patch("prototipo_pdf.processar_multiplos_bimestres", return_value=conjuntos_mock):
        flowables = montar_flowables(
            df,
            bimestres=[1],
            curso="TÉCNICO EM ESTRADAS",
            cruzamento=["mapa_mock.xls"],
        )
    texto = extrair_texto_flowables(flowables, incluir_rodape=False)
    assert "Cruzamento com o mapa" in texto
    assert "Cruzamento pendente" in texto


def test_cruzamento_mapas_dae_real_xlsx_cobertura_e_pdm(caminho_xlsx: Path):
    """Com mapas e .xlsx de teste: 'Nos dois', 'N/C (Nada consta)' e nenhuma matrícula/nome do mock (C14)."""
    mock_nome = "Estudante Mock Especial"
    mock_mat = "20261010999"
    df_notas = pd.DataFrame([
        {"matricula": "20261010003", "MAT": 18.0},
        {"matricula": mock_mat, "MAT": 15.0},
    ])
    df_faltas = pd.DataFrame(
        [
            # Aluno casado (Carlos Lima na DAE com N/C - Nada consta)
            {"matricula": "20261010003", "nome": "Carlos Lima", "MAT": 62},
            # Aluno só no app com dados únicos do mock
            {"matricula": mock_mat, "nome": mock_nome, "MAT": 15},
        ]
    )
    meta = {"bimestre_num": 1, "curso": "Técnico em Estradas"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    df_real = carregar_dae(caminho_xlsx)
    with patch("prototipo_pdf.processar_multiplos_bimestres", return_value=conjuntos_mock):
        flowables = montar_flowables(
            df_real,
            bimestres=[1],
            curso="TÉCNICO EM ESTRADAS",
            cruzamento=["mapa_mock.xls"],
        )
    texto = extrair_texto_flowables(flowables, incluir_rodape=False)

    assert "Cruzamento com o mapa" in texto
    assert "Nos dois" in texto
    assert "N/C (Nada consta)" in texto

    # LGPD: Nenhuma matrícula ou nome do mock deve vazar no relatório
    assert mock_nome not in texto
    assert mock_mat not in texto


def test_executar_cli_com_mapas_e_dae_sintetica(tmp_path: Path):
    """Testa CLI com flag --mapas e DAE sintética gerando PDF com aviso de pendência."""
    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 18.0}])
    df_faltas = pd.DataFrame([{"matricula": "20261010001", "nome": "Mock", "MAT": 10}])
    meta = {"bimestre_num": 1, "curso": "Técnico em Estradas"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    saida = tmp_path / "cli_mapas.pdf"
    with patch("prototipo_pdf.processar_multiplos_bimestres", return_value=conjuntos_mock):
        codigo = _executar_cli([
            "--curso", "TÉCNICO EM ESTRADAS",
            "--mapas", "mock.xls",
            "--saida", str(saida),
        ])
    assert codigo == 0
    assert saida.exists()
    assert saida.read_bytes().startswith(b"%PDF")


def test_ch_efetiva_com_planilha_e_mapas(planilha_ch_sintetica: Path):
    """Com planilha sintética e mapa mockado: valida bloco de CH efetiva real (C18)."""
    mock_prof = "Prof. Fictício Topografia"
    mock_aluno = "Aluno Ficticio Especial"
    mock_mat = "20261010999"

    df_notas = pd.DataFrame([
        {"matricula": mock_mat, "TOP": 18.0, "QUI": 15.0}
    ])
    df_faltas = pd.DataFrame([
        {"matricula": mock_mat, "nome": mock_aluno, "TOP": 2, "QUI": 0}
    ])
    legenda = {
        "TOP": "TOPOGRAFIA",
        "QUI": "QUÍMICA",  # sem linha na planilha_ch_sintetica
    }
    meta = {
        "bimestre_num": 1,
        "curso": "TÉCNICO EM ESTRADAS",
        "curso_amigavel": "Estradas",
        "turma": "EST-2A",
        "serie": 2,
    }
    conjuntos_mock = [(df_notas, df_faltas, legenda, meta)]

    df = obter_dados_sinteticos()
    with patch("prototipo_pdf.processar_multiplos_bimestres", return_value=conjuntos_mock):
        flowables = montar_flowables(
            df,
            bimestres=[1],
            curso="TÉCNICO EM ESTRADAS",
            cruzamento=["mapa_mock.xls"],
            ch_efetiva=planilha_ch_sintetica,
        )

    texto = extrair_texto_flowables(flowables, incluir_rodape=False)

    assert "CH efetiva por disciplina" in texto
    assert "cenário A" in texto
    assert "cenário A — sem sábados" in texto
    assert "ch_sintetica.xlsx" in texto
    assert "origem: planilha" in texto
    assert "Divergências com o calendário oficial: 0" in texto
    assert "TOPOGRAFIA" in texto
    assert "20 h/a" in texto or "20" in texto
    assert "CH efetiva no ano" in texto
    assert "% do nominal" in texto
    assert "Limite de faltas" in texto
    assert "Sem horário na planilha" in texto
    assert "QUÍMICA" in texto

    # Minimização de dados / LGPD: nenhum nome de professor fictício nem matrícula/nome de aluno
    assert mock_prof not in texto
    assert "Prof. Fictício" not in texto
    assert mock_aluno not in texto
    assert mock_mat not in texto


def test_ch_efetiva_planilha_sem_mapas(planilha_ch_sintetica: Path):
    """Com planilha sintética e sem mapas: valida distribuição por carga e aviso (C18)."""
    df = obter_dados_sinteticos()
    flowables = montar_flowables(
        df,
        bimestres=[1],
        curso="Trânsito",
        ch_efetiva=planilha_ch_sintetica,
        cruzamento=None,
    )

    texto = extrair_texto_flowables(flowables, incluir_rodape=False)

    assert "< 90%" in texto
    assert "90–95%" in texto
    assert "≥ 95%" in texto
    assert "informe o mapa da turma" in texto
    # distribuição restrita às linhas do curso "Trânsito" (TT-2A + EST/TT-2A = 5 linhas:
    # 3 abaixo de 90%, 1 entre 90–95%, 1 >= 95% — e não as 7 da planilha inteira)
    linhas = texto.splitlines()
    assert linhas[linhas.index("< 90%") + 1] == "3"
    assert linhas[linhas.index("90–95%") + 1] == "1"
    assert linhas[linhas.index("≥ 95%") + 1] == "1"


def test_ch_efetiva_sem_planilha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sem planilha (caminho inexistente e default ausente via monkeypatch): aviso de ausência (C18)."""
    monkeypatch.setattr("prototipo_pdf.PASTA_DADOS", tmp_path)
    df = obter_dados_sinteticos()

    # Caminho inexistente
    flowables_inex = montar_flowables(
        df,
        bimestres=[1],
        ch_efetiva=tmp_path / "inexistente.xlsx",
    )
    texto_inex = extrair_texto_flowables(flowables_inex, incluir_rodape=False)
    assert "Planilha de CH efetiva ausente" in texto_inex

    # Default ausente (ch_efetiva=None com PASTA_DADOS vazia)
    flowables_default = montar_flowables(
        df,
        bimestres=[1],
        ch_efetiva=None,
    )
    texto_def = extrair_texto_flowables(flowables_default, incluir_rodape=False)
    assert "Planilha de CH efetiva ausente" in texto_def


def test_executar_cli_com_ch_efetiva(planilha_ch_sintetica: Path, tmp_path: Path):
    """Testa interface CLI com flag --ch-efetiva (C18)."""
    saida = tmp_path / "cli_ch_efetiva.pdf"
    codigo = _executar_cli([
        "--curso", "TÉCNICO EM ESTRADAS",
        "--ch-efetiva", str(planilha_ch_sintetica),
        "--saida", str(saida),
    ])
    assert codigo == 0
    assert saida.exists()
    assert saida.read_bytes().startswith(b"%PDF")


def test_prototipo_pdf_det_com_mapas_sinteticos_e_ch_efetiva(
    planilha_ch_sintetica: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valida protótipo PDF com mapas sintéticos EST+TT e planilha de CH efetiva.

    - Bloco 'Cruzamento com o mapa de turma' mostra o resumo DET (total, por curso, interseção)
      e Pé-de-Meia com a tupla, contendo 'Estradas + Trânsito'.
    - Bloco 'CH efetiva por disciplina' usa resumo_frequencia_det com colunas de escopo:
      'Núcleo comum', 'Trânsito' e 'Estradas'.
    - Nota de sábados cita '23/05' para Estradas e Trânsito.
    - Minimização de dados / LGPD: nenhuma matrícula ou nome discente dos mapas,
      nem nome de professor fictício no texto extraído.
    """
    import core.manipulacao as manipulacao

    original_ler = manipulacao._ler_xls_bruto

    def _mock_ler(arquivo_xls):
        if isinstance(arquivo_xls, pd.DataFrame):
            return arquivo_xls
        return original_ler(arquivo_xls)

    monkeypatch.setattr(manipulacao, "_ler_xls_bruto", _mock_ler)

    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST-2A",
        disciplinas={
            "ING": "LÍNGUA ESTRANGEIRA: INGLÊS - 2ª SÉRIE",
            "TOP": "TOPOGRAFIA",
            "SOL": "LABORATÓRIO DE SOLOS",
        },
        alunos=[
            ("20260000001", "Aluno Compartilhado 1", {"ING": 15.0, "TOP": 12.0, "SOL": 12.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
            ("20260000002", "Aluno Compartilhado 2", {"ING": 14.0, "TOP": 13.0, "SOL": 13.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
            ("20260000003", "Aluno Estradas 3", {"ING": 16.0, "TOP": 14.0, "SOL": 14.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
            ("20260000004", "Aluno Estradas 4", {"ING": 17.0, "TOP": 15.0, "SOL": 15.0}, {"ING": 0, "TOP": 0, "SOL": 0}),
        ],
    )

    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TT-2A",
        disciplinas={
            "OPT": "OPERAÇÃO DE TRANSPORTES",
            "TRA": "PLANEJAMENTO DE TRANSPORTES",
        },
        alunos=[
            ("20260000001", "Aluno Compartilhado 1", {"OPT": 18.0, "TRA": 18.0}, {"OPT": 0, "TRA": 0}),
            ("20260000002", "Aluno Compartilhado 2", {"OPT": 19.0, "TRA": 19.0}, {"OPT": 0, "TRA": 0}),
        ],
    )

    det = carregar_det([df_est, df_tt])
    df = obter_dados_sinteticos()

    # 1. Teste passando conjuntos_det diretamente
    flowables = montar_flowables(
        df,
        bimestres=[1],
        curso="TÉCNICO EM ESTRADAS",
        cruzamento=det,
        ch_efetiva=planilha_ch_sintetica,
    )
    texto = extrair_texto_flowables(flowables, incluir_rodape=False)

    assert "Estradas + Trânsito" in texto
    assert "Núcleo comum" in texto
    assert "Trânsito" in texto
    assert "Estradas" in texto
    assert "23/05" in texto

    # Nenhuma matrícula dos mapas
    for mat in ("20260000001", "20260000002", "20260000003", "20260000004"):
        assert mat not in texto

    # Nenhum nome discente dos mapas
    for nome in ("Aluno Compartilhado", "Aluno Estradas 3", "Aluno Estradas 4"):
        assert nome not in texto

    # Nenhum nome de professor fictício
    for prof in (
        "Prof. Fictício Topografia",
        "Prof. Fictício Ingles",
        "Prof. Fictício Transportes",
        "Prof. Fictício",
    ):
        assert prof not in texto

    # 2. Teste passando dict de carregar_det
    dict_det = {"Estradas": det.estradas, "Trânsito": det.transito}
    flowables_dict = montar_flowables(
        df,
        bimestres=[1],
        curso="TÉCNICO EM ESTRADAS",
        cruzamento=dict_det,
        ch_efetiva=planilha_ch_sintetica,
    )
    texto_dict = extrair_texto_flowables(flowables_dict, incluir_rodape=False)

    assert "Estradas + Trânsito" in texto_dict
    assert "Núcleo comum" in texto_dict
    assert "Trânsito" in texto_dict
    assert "Estradas" in texto_dict
    assert "23/05" in texto_dict


def test_executar_cli_det_dois_mapas(tmp_path: Path, planilha_ch_sintetica: Path, monkeypatch: pytest.MonkeyPatch):
    """Testa CLI passando mapas dos dois cursos via --mapas montando o DET."""
    import core.manipulacao as manipulacao

    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST-2A",
        disciplinas={"ING": "INGLÊS", "TOP": "TOPOGRAFIA"},
        alunos=[("20260000001", "Aluno 1", {"ING": 15.0, "TOP": 12.0}, {"ING": 0, "TOP": 0})],
    )
    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TT-2A",
        disciplinas={"OPT": "OPERAÇÃO DE TRANSPORTES"},
        alunos=[("20260000001", "Aluno 1", {"OPT": 18.0}, {"OPT": 0})],
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

    mapa_est = tmp_path / "mock_est.xls"
    mapa_tt = tmp_path / "mock_tt.xls"
    mapa_est.touch()
    mapa_tt.touch()

    saida = tmp_path / "cli_det_out.pdf"
    codigo = _executar_cli([
        "--curso", "TÉCNICO EM ESTRADAS",
        "--mapas", str(mapa_est), str(mapa_tt),
        "--ch-efetiva", str(planilha_ch_sintetica),
        "--saida", str(saida),
    ])
    assert codigo == 0
    assert saida.exists()
    assert saida.read_bytes().startswith(b"%PDF")


def _extrair_todas_tabelas(flowables: list) -> list[Table]:
    """Extrai recursivamente todos os objetos Table dos flowables."""
    tabelas: list[Table] = []
    for item in flowables:
        if isinstance(item, Table):
            tabelas.append(item)
        elif isinstance(item, KeepTogether):
            for sub in item._content:
                if isinstance(sub, Table):
                    tabelas.append(sub)
    return tabelas


def _extrair_todos_paragrafos(flowables: list) -> list[Paragraph]:
    """Extrai recursivamente todos os objetos Paragraph dos flowables."""
    paras: list[Paragraph] = []
    for item in flowables:
        if isinstance(item, Paragraph):
            paras.append(item)
        elif isinstance(item, Table):
            for row in item._cellvalues:
                for cell in row:
                    if isinstance(cell, Paragraph):
                        paras.append(cell)
                    elif isinstance(cell, (list, tuple)):
                        for sub in cell:
                            if isinstance(sub, Paragraph):
                                paras.append(sub)
        elif isinstance(item, KeepTogether):
            for sub in item._content:
                if isinstance(sub, Paragraph):
                    paras.append(sub)
                elif isinstance(sub, Table):
                    for row in sub._cellvalues:
                        for cell in row:
                            if isinstance(cell, Paragraph):
                                paras.append(cell)
                            elif isinstance(cell, (list, tuple)):
                                for sub2 in cell:
                                    if isinstance(sub2, Paragraph):
                                        paras.append(sub2)
    return paras


def test_layout_invalido_gera_value_error():
    """Valida que layout inválido levanta ValueError."""
    df = obter_dados_sinteticos()
    with pytest.raises(ValueError, match="Layout inválido"):
        montar_flowables(df, layout="invalido")


def test_layout_app_tabelas_largura_maxima_16cm(
    planilha_ch_sintetica: Path, monkeypatch: pytest.MonkeyPatch
):
    """Com layout='app', todo Table tem soma de larguras ≤ 16 cm (D6)."""
    import core.manipulacao as manipulacao

    original_ler = manipulacao._ler_xls_bruto

    def _mock_ler(arquivo_xls):
        if isinstance(arquivo_xls, pd.DataFrame):
            return arquivo_xls
        return original_ler(arquivo_xls)

    monkeypatch.setattr(manipulacao, "_ler_xls_bruto", _mock_ler)

    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST-2A",
        disciplinas={"ING": "INGLÊS", "TOP": "TOPOGRAFIA"},
        alunos=[("20260000001", "Aluno 1", {"ING": 15.0, "TOP": 12.0}, {"ING": 0, "TOP": 0})],
    )
    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TT-2A",
        disciplinas={"OPT": "OPERAÇÃO DE TRANSPORTES"},
        alunos=[("20260000001", "Aluno 1", {"OPT": 18.0}, {"OPT": 0})],
    )
    det = carregar_det([df_est, df_tt])
    df = obter_dados_sinteticos()

    # 1. Caso com DET e planilha CH (gera todas as tabelas possíveis)
    flowables = montar_flowables(
        df,
        bimestres=[1, 2],
        curso="TÉCNICO EM ESTRADAS",
        cruzamento=det,
        ch_efetiva=planilha_ch_sintetica,
        layout="app",
    )
    tabelas = _extrair_todas_tabelas(flowables)
    assert len(tabelas) >= 5, "Deve conter múltiplas tabelas"
    for t in tabelas:
        soma_cm = sum(t._colWidths) / cm
        assert soma_cm <= 16.0 + 1e-4, f"Tabela excede 16 cm: {soma_cm:.4f} cm"

    # 2. Caso sem cruzamento e sem CH
    flowables_simples = montar_flowables(
        df,
        bimestres=[1],
        layout="app",
    )
    tabelas_simples = _extrair_todas_tabelas(flowables_simples)
    assert len(tabelas_simples) >= 3
    for t in tabelas_simples:
        soma_cm = sum(t._colWidths) / cm
        assert soma_cm <= 16.0 + 1e-4, f"Tabela excede 16 cm: {soma_cm:.4f} cm"


def test_layout_app_titulos_h2_sumario_sem_numero_proprio():
    """Com layout='app', existe ao menos um Paragraph com style.name == 'H2Sumario'
    e nenhum texto de título começa com dígito + ponto (D6).
    """
    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, layout="app")

    h2_paras = [
        p for p in flowables
        if isinstance(p, Paragraph) and getattr(p.style, "name", None) == "H2Sumario"
    ]
    assert len(h2_paras) >= 1, "Deve existir ao menos um Paragraph com style.name == 'H2Sumario'"

    for p in h2_paras:
        texto = p.getPlainText().strip()
        assert not re.match(r"^\d+\.", texto), f"Título começa com dígito + ponto: '{texto}'"
        assert not re.match(r"^<b>\d+\.", p.text.strip()), f"Texto bruto começa com dígito + ponto: '{p.text}'"


def test_layout_app_nenhum_estilo_com_fonte_helvetica():
    """Com layout='app', nenhum estilo usa fonte Helvetica* (D6: Times)."""
    df = obter_dados_sinteticos()
    flowables = montar_flowables(df, layout="app")

    paras = _extrair_todos_paragrafos(flowables)
    assert len(paras) > 0, "Deve haver parágrafos gerados"

    for p in paras:
        font = p.style.fontName
        assert not font.lower().startswith("helvetica"), (
            f"Estilo '{p.style.name}' usa fonte '{font}', que começa com Helvetica"
        )


def test_demonstracao_e_ficticios_com_dae_sintetica_e_ausentes_com_dae_real(
    caminho_xlsx: Path,
):
    """'DEMONSTRAÇÃO' e 'FICTÍCIOS' presentes com DAE sintética e ausentes com
    o .xlsx sintético de teste carregado como DAE real (D8).
    Com DAE sintética e layout='app', o primeiro flowable é o quadro 'DEMONSTRAÇÃO'.
    """
    # 1. Com DAE sintética
    df_sint = obter_dados_sinteticos()
    flowables_sint = montar_flowables(df_sint, layout="app")
    texto_sint = extrair_texto_flowables(flowables_sint, incluir_rodape=False)

    assert "DEMONSTRAÇÃO" in texto_sint
    assert "FICTÍCIOS" in texto_sint

    # O primeiro flowable deve ser o quadro "DEMONSTRAÇÃO" de D8 (texto literal de D8)
    primeiro = flowables_sint[0]
    assert isinstance(primeiro, Table), f"Primeiro flowable deve ser Table, mas é {type(primeiro)}"
    texto_primeiro = extrair_texto_flowables([primeiro], incluir_rodape=False)
    assert "DEMONSTRAÇÃO" in texto_primeiro
    assert "FICTÍCIOS" in texto_primeiro
    assert TEXTO_DEMONSTRACAO in texto_primeiro

    # 2. Com o .xlsx sintético de teste carregado como DAE real
    df_real = carregar_dae(caminho_xlsx)
    flowables_real = montar_flowables(df_real, layout="app")
    texto_real = extrair_texto_flowables(flowables_real, incluir_rodape=False)

    assert "DEMONSTRAÇÃO" not in texto_real
    assert "FICTÍCIOS" not in texto_real

    # O primeiro flowable NÃO deve ser o quadro DEMONSTRAÇÃO
    primeiro_real = flowables_real[0]
    assert isinstance(primeiro_real, Paragraph)
    assert getattr(primeiro_real.style, "name", None) == "H2Sumario"


def test_helper_largura_unitario():
    """Testa o helper _largura diretamente."""
    # Escala para 16 cm quando layout='app'
    assert _largura(17.5 * cm, layout="app") == pytest.approx(16.0 * cm)
    assert _largura(17.5 * cm, layout="prototipo") == pytest.approx(17.5 * cm)

    # Lista de larguras somando 17.5 cm escala para somar 16.0 cm
    cols = [7.5 * cm, 5.0 * cm, 5.0 * cm]
    cols_app = _largura(cols, layout="app")
    assert sum(cols_app) == pytest.approx(16.0 * cm)

    cols_proto = _largura(cols, layout="prototipo")
    assert sum(cols_proto) == pytest.approx(17.5 * cm)





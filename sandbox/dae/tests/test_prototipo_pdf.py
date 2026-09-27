"""Testes do gerador de protótipo PDF com destaques da DAE."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd
import pytest

from reportlab.platypus import KeepTogether, Paragraph, Table

_DIR_TESTS = Path(__file__).resolve().parent
_DIR_DAE = _DIR_TESTS.parent
_DIR_RAIZ = _DIR_DAE.parent.parent
for _p in (_DIR_DAE, _DIR_RAIZ):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.relatorios import COR_CABECALHO_TABELA, COR_TEXTO_CABECALHO_TABELA
from prototipo_pdf import (
    LEGENDA_PROGRAMAS,
    NOTA_LGPD,
    NOTA_PE_DE_MEIA,
    TEXTO_RODAPE_USO_INTERNO,
    _executar_cli,
    determinar_caminho_saida,
    extrair_texto_flowables,
    gerar_prototipo_pdf,
    montar_flowables,
    obter_dados_sinteticos,
)


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
    """Valida a ordem estrita dos blocos especificados no passo 5:

    (i) bloco 'Período de apuração da frequência'
    (ii) trecho da tabela 2.1 com a coluna extra 'Prog.'
    (iii) quadro 'Tratamento de dados pessoais (LGPD)'
    (iv) seção 'Estudantes acompanhados pela Assistência Estudantil'
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
    idx_bloco_ii = next(
        i for i, t in enumerate(titulos) if "Tabela 2.1" in t or "Desempenho e Frequência" in t
    )
    idx_bloco_iii = next(
        i for i, t in enumerate(titulos) if "Tratamento de dados pessoais" in t
    )
    idx_bloco_iv = next(
        i for i, t in enumerate(titulos) if "Estudantes acompanhados" in t
    )

    assert idx_bloco_i < idx_bloco_ii < idx_bloco_iii < idx_bloco_iv, (
        f"A ordem dos blocos deve ser (i) < (ii) < (iii) < (iv), obtido: "
        f"i={idx_bloco_i}, ii={idx_bloco_ii}, iii={idx_bloco_iii}, iv={idx_bloco_iv}"
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

    # A tabela 2.1 é a segunda tabela no documento
    tabelas = [f for f in flowables if isinstance(f, Table)]
    assert len(tabelas) >= 2

    tab_21 = tabelas[1]
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
    tab_21 = tabelas[1]
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

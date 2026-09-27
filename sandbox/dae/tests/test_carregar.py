"""Testes unitários para o carregador da planilha da DAE (sandbox/dae/carregar.py)."""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from carregar import carregar_dae


def test_xlsx_e_csv_dao_mesmo_dataframe(caminho_xlsx: Path, caminho_csv: Path) -> None:
    """Verifica se carregar o .xlsx e o .csv sintéticos produz exatamente o mesmo DataFrame."""
    df_xlsx = carregar_dae(caminho_xlsx)
    df_csv = carregar_dae(caminho_csv)

    pd.testing.assert_frame_equal(df_xlsx, df_csv)


def test_colunas(caminho_xlsx: Path) -> None:
    """Verifica a presença e ordem das colunas esperadas no DataFrame de saída."""
    df = carregar_dae(caminho_xlsx)

    colunas_fixas = [
        "matricula",
        "nome",
        "unidade",
        "curso",
        "pe_de_meia",
        "bolsa_ba_bp",
        "bolsa_bce",
        "programas",
    ]
    meses = [
        "fevereiro",
        "marco",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
    ]
    colunas_meses = []
    for m in meses:
        colunas_meses.extend([f"ha_ofertadas_{m}", f"ha_presenciadas_{m}"])

    esperadas = colunas_fixas + colunas_meses + ["acumulado_dae"]

    assert list(df.columns) == esperadas
    assert len(df) == 4


def test_ausencia_de_cpf(caminho_xlsx: Path, caminho_csv: Path) -> None:
    """Verifica que o CPF foi descartado na entrada e não está no DataFrame (D4)."""
    df_xlsx = carregar_dae(caminho_xlsx)
    df_csv = carregar_dae(caminho_csv)

    for df in (df_xlsx, df_csv):
        assert "cpf" not in df.columns
        assert "CPF" not in df.columns
        for col in df.columns:
            assert "cpf" not in col.lower()


def test_nc_nada_consta_sem_pdm(caminho_xlsx: Path) -> None:
    """Verifica que 'N/C' normaliza para 'nada_consta' e não marca PdM em programas (D7)."""
    df = carregar_dae(caminho_xlsx)

    aluno_nc = df[df["matricula"] == "20261010003"].iloc[0]
    assert aluno_nc["nome"] == "Carlos Lima"
    assert aluno_nc["pe_de_meia"] == "nada_consta"
    assert "PdM" not in aluno_nc["programas"]
    assert aluno_nc["programas"] == []


def test_cada_valor_de_dominio_pe_de_meia_e_programas(tmp_path: Path) -> None:
    """Verifica cada valor do domínio de pe_de_meia e a composição correta de programas."""
    caminho = tmp_path / "teste_dominios.csv"

    # Monta CSV com alunos cobrindo todas as combinações de pe_de_meia e bolsas
    row0 = [""] * 9 + ["Fevereiro", "", ""]
    row1 = [
        "Nome do estudante",
        "Matrícula",
        "Acumulado",
        "CPF",
        "Unidade",
        "Curso",
        "Pé de meia",
        "Bolsas DAE - BA/BP",
        "BOLSAS DAE - BCE",
        "HA ofertadas",
        "HA presenciadas",
        "Frequência",
    ]
    alunos_teste = [
        # Elegível com todas as bolsas -> ['PdM', 'BA/BP', 'BCE']
        ["Aluno 1", "20260000001", "0.80", "001", "BH", "Transito", "Elegível", "Sim", "Sim", "20", "16", "0.8"],
        # Elegível apenas com PdM -> ['PdM']
        ["Aluno 2", "20260000002", "0.80", "002", "BH", "Transito", "elegivel", "Não", "Não", "20", "16", "0.8"],
        # Não elegível com BCE -> ['BCE']
        ["Aluno 3", "20260000003", "0.80", "003", "BH", "Transito", "Não elegível", "Não", "Sim", "20", "16", "0.8"],
        # N/C com BA/BP -> ['BA/BP']
        ["Aluno 4", "20260000004", "0.80", "004", "BH", "Transito", "N/C", "Sim", "Não", "20", "16", "0.8"],
        # Elegibilidade indefinida sem bolsa -> []
        ["Aluno 5", "20260000005", "0.80", "005", "BH", "Transito", "Elegibilidade indefinida", "Não", "Não", "20", "16", "0.8"],
    ]

    with open(caminho, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row0)
        writer.writerow(row1)
        for al in alunos_teste:
            writer.writerow(al)

    df = carregar_dae(caminho)

    r1 = df[df["matricula"] == "20260000001"].iloc[0]
    assert r1["pe_de_meia"] == "elegivel"
    assert r1["programas"] == ["PdM", "BA/BP", "BCE"]

    r2 = df[df["matricula"] == "20260000002"].iloc[0]
    assert r2["pe_de_meia"] == "elegivel"
    assert r2["programas"] == ["PdM"]

    r3 = df[df["matricula"] == "20260000003"].iloc[0]
    assert r3["pe_de_meia"] == "nao_elegivel"
    assert r3["programas"] == ["BCE"]

    r4 = df[df["matricula"] == "20260000004"].iloc[0]
    assert r4["pe_de_meia"] == "nada_consta"
    assert r4["programas"] == ["BA/BP"]

    r5 = df[df["matricula"] == "20260000005"].iloc[0]
    assert r5["pe_de_meia"] == "indefinida"
    assert r5["programas"] == []


def test_valor_desconhecido_vira_indefinida(tmp_path: Path) -> None:
    """Verifica que qualquer valor não catalogado ou nulo em pe_de_meia normaliza para 'indefinida'."""
    caminho = tmp_path / "teste_desconhecido.csv"

    row0 = [""] * 9 + ["Fevereiro", "", ""]
    row1 = [
        "Nome do estudante",
        "Matrícula",
        "Acumulado",
        "CPF",
        "Unidade",
        "Curso",
        "Pé de meia",
        "Bolsas DAE - BA/BP",
        "BOLSAS DAE - BCE",
        "HA ofertadas",
        "HA presenciadas",
        "Frequência",
    ]
    alunos_teste = [
        ["Aluno Desconhecido", "20260000010", "0.80", "010", "BH", "Transito", "Qualquer Coisa", "Não", "Não", "20", "16", "0.8"],
        ["Aluno Vazio", "20260000011", "0.80", "011", "BH", "Transito", "", "Não", "Não", "20", "16", "0.8"],
    ]

    with open(caminho, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row0)
        writer.writerow(row1)
        for al in alunos_teste:
            writer.writerow(al)

    df = carregar_dae(caminho)

    r_desc = df[df["matricula"] == "20260000010"].iloc[0]
    assert r_desc["pe_de_meia"] == "indefinida"
    assert "PdM" not in r_desc["programas"]

    r_vazio = df[df["matricula"] == "20260000011"].iloc[0]
    assert r_vazio["pe_de_meia"] == "indefinida"
    assert "PdM" not in r_vazio["programas"]


def test_meses_vazios_geram_nan(caminho_xlsx: Path) -> None:
    """Verifica que meses sem lançamento (setembro a novembro) ficam como NaN."""
    df = carregar_dae(caminho_xlsx)

    # Meses lançados (fev-ago) devem ter valores numéricos não-nulos
    assert df["ha_ofertadas_fevereiro"].notna().all()
    assert df["ha_presenciadas_fevereiro"].notna().all()
    assert df["ha_ofertadas_agosto"].notna().all()
    assert df["ha_presenciadas_agosto"].notna().all()

    # Meses vazios (set-nov) devem ser exclusivamente NaN
    for m in ("setembro", "outubro", "novembro"):
        assert df[f"ha_ofertadas_{m}"].isna().all(), f"ha_ofertadas_{m} deveria ser NaN"
        assert df[f"ha_presenciadas_{m}"].isna().all(), f"ha_presenciadas_{m} deveria ser NaN"


def test_csv_sem_colunas_obrigatorias_erro(tmp_path: Path) -> None:
    """Verifica que CSV sem as colunas obrigatórias levanta ValueError informando a aba '2026 - geral'."""
    caminho = tmp_path / "invalido.csv"
    with open(caminho, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Coluna1", "Coluna2"])
        writer.writerow(["A", "B"])

    with pytest.raises(ValueError, match="2026 - geral"):
        carregar_dae(caminho)


def test_extensao_diferente_de_xlsx_ou_csv_erro(tmp_path: Path) -> None:
    """Verifica que extensões diferentes de .xlsx e .csv levantam ValueError."""
    caminho_txt = tmp_path / "arquivo.txt"
    caminho_txt.write_text("conteudo")

    with pytest.raises(ValueError, match=r"\.xlsx ou \.csv"):
        carregar_dae(caminho_txt)

    caminho_xls = tmp_path / "arquivo.xls"
    caminho_xls.write_text("conteudo")

    with pytest.raises(ValueError, match=r"\.xlsx ou \.csv"):
        carregar_dae(caminho_xls)


def test_xlsx_sem_aba_2026_geral_erro(tmp_path: Path) -> None:
    """Verifica que .xlsx sem a aba '2026 - geral' levanta ValueError."""
    caminho = tmp_path / "sem_aba_geral.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Emails"
    wb.save(caminho)

    with pytest.raises(ValueError, match="2026 - geral"):
        carregar_dae(caminho)


def test_aba_emails_nao_e_lida(caminho_xlsx: Path) -> None:
    """Verifica que os dados da aba 'Emails' não vazam para o DataFrame retornado (D4)."""
    df = carregar_dae(caminho_xlsx)

    # Verifica que não há colunas de e-mail e nenhum dado de e-mail nos valores
    assert "email" not in [c.lower() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            assert not df[col].astype(str).str.contains("@aluno.cefetmg.br").any()


def test_matricula_apenas_digitos(tmp_path: Path) -> None:
    """Verifica que a matrícula é limpa para conter exclusivamente dígitos."""
    caminho = tmp_path / "matriculas.csv"
    row0 = [""] * 9
    row1 = [
        "Nome do estudante",
        "Matrícula",
        "Acumulado",
        "CPF",
        "Unidade",
        "Curso",
        "Pé de meia",
        "Bolsas DAE - BA/BP",
        "BOLSAS DAE - BCE",
    ]
    alunos = [
        ["Aluno Formatado", "2026.101.001-A", "0.80", "001", "BH", "Transito", "Elegível", "Não", "Não"],
        ["Aluno com Espaço", " 2026101002 ", "0.80", "002", "BH", "Transito", "Elegível", "Não", "Não"],
    ]
    with open(caminho, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row0)
        writer.writerow(row1)
        for al in alunos:
            writer.writerow(al)

    df = carregar_dae(caminho)
    assert list(df["matricula"]) == ["2026101001", "2026101002"]

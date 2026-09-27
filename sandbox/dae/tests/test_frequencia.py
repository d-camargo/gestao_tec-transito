"""Testes unitários para o módulo de cálculo de frequência (sandbox/dae/frequencia.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carregar import carregar_dae
from frequencia import (
    MESES_CALENDARIO,
    MESES_POR_BIMESTRE,
    frequencia_ponderada,
    mes_referencia,
    meses_lancados,
    periodo_apuracao,
    tabela_frequencia,
)


def test_media_simples_vs_ponderada_divergem() -> None:
    """Verifica que a média simples e a ponderada divergem.

    Caso do teste especificado:
        - Fev: 20 presenciadas / 20 ofertadas (100%)
        - Mar: 75 presenciadas / 150 ofertadas (50%)
        - Média simples: (100% + 50%) / 2 = 75,0%
        - Ponderada: (20 + 75) / (20 + 150) = 95 / 170 ≈ 55,88% (≈ 55,9%)
    """
    df = pd.DataFrame(
        [
            {
                "matricula": "20261010999",
                "nome": "Estudante Exemplo",
                "ha_ofertadas_fevereiro": 20.0,
                "ha_presenciadas_fevereiro": 20.0,
                "ha_ofertadas_marco": 150.0,
                "ha_presenciadas_marco": 75.0,
                "acumulado_dae": 0.75,
            }
        ]
    )

    # Cálculo isolado via frequencia_ponderada
    freq_pond = frequencia_ponderada(df, ["fevereiro", "marco"])
    val_pond = freq_pond.iloc[0]

    media_simples = (20.0 / 20.0 + 75.0 / 150.0) / 2.0
    assert media_simples == 0.75
    assert pytest.approx(val_pond, abs=0.001) == 0.559
    assert val_pond == 95.0 / 170.0
    assert val_pond != media_simples

    # Na tabela_frequencia consolidada
    mapa = {1: ["fevereiro", "marco"]}
    tab = tabela_frequencia(df, meses_por_bimestre=mapa, bimestres=[1])
    assert pytest.approx(tab["freq_bim_1"].iloc[0], abs=0.001) == 0.559
    assert pytest.approx(tab["freq_acumulada"].iloc[0], abs=0.001) == 0.559
    assert tab["abaixo_75"].iloc[0] is True or tab["abaixo_75"].iloc[0] == True

    # diff_vs_dae_pp: ponderada - acumulado_dae em p.p. (aproximadamente -19.1 p.p.)
    diff_esperada = (val_pond - 0.75) * 100.0
    assert pytest.approx(tab["diff_vs_dae_pp"].iloc[0], abs=0.01) == diff_esperada
    assert pytest.approx(tab["diff_vs_dae_pp"].iloc[0], abs=0.1) == -19.1


def test_aluno_sem_mes_lancado_nan_e_nao_marcado() -> None:
    """Verifica que aluno sem horas lançadas gera NaN na frequência e não é marcado em abaixo_75."""
    df = pd.DataFrame(
        [
            {
                "matricula": "20261010001",
                "nome": "Sem Lancamento",
                "ha_ofertadas_fevereiro": np.nan,
                "ha_presenciadas_fevereiro": np.nan,
                "ha_ofertadas_marco": np.nan,
                "ha_presenciadas_marco": np.nan,
                "acumulado_dae": np.nan,
            },
            {
                "matricula": "20261010002",
                "nome": "Com Zeros",
                "ha_ofertadas_fevereiro": 0.0,
                "ha_presenciadas_fevereiro": 0.0,
                "ha_ofertadas_marco": 0.0,
                "ha_presenciadas_marco": 0.0,
                "acumulado_dae": np.nan,
            },
        ]
    )

    freq_pond = frequencia_ponderada(df, ["fevereiro", "marco"])
    assert pd.isna(freq_pond.iloc[0])
    assert pd.isna(freq_pond.iloc[1])

    tab = tabela_frequencia(df, meses_por_bimestre={1: ["fevereiro", "marco"]}, bimestres=[1])
    assert pd.isna(tab["freq_bim_1"].iloc[0])
    assert pd.isna(tab["freq_bim_1"].iloc[1])
    assert pd.isna(tab["freq_acumulada"].iloc[0])
    assert pd.isna(tab["freq_acumulada"].iloc[1])

    # Não marcado como abaixo de 75%
    assert tab["abaixo_75"].iloc[0] is False or tab["abaixo_75"].iloc[0] == False
    assert tab["abaixo_75"].iloc[1] is False or tab["abaixo_75"].iloc[1] == False
    assert tab["abaixo_75"].dtype == bool

    # diff_vs_dae_pp é NaN
    assert pd.isna(tab["diff_vs_dae_pp"].iloc[0])
    assert pd.isna(tab["diff_vs_dae_pp"].iloc[1])


def test_fronteira_exata_75_por_cento() -> None:
    """Verifica o comportamento no limiar de 75%: estritamente < 0.75 marca True."""
    df = pd.DataFrame(
        [
            {
                "matricula": "1",
                "nome": "Exatamente 75%",
                "ha_ofertadas_fevereiro": 100.0,
                "ha_presenciadas_fevereiro": 75.0,
            },
            {
                "matricula": "2",
                "nome": "Abaixo 75% (74.9%)",
                "ha_ofertadas_fevereiro": 1000.0,
                "ha_presenciadas_fevereiro": 749.0,
            },
            {
                "matricula": "3",
                "nome": "Acima 75% (75.1%)",
                "ha_ofertadas_fevereiro": 1000.0,
                "ha_presenciadas_fevereiro": 751.0,
            },
        ]
    )

    tab = tabela_frequencia(df, meses_por_bimestre={1: ["fevereiro"]}, bimestres=[1])

    # Aluno 1: 0.75 -> NÃO está abaixo de 75%
    assert tab.loc[0, "freq_acumulada"] == 0.75
    assert tab.loc[0, "abaixo_75"] is False or tab.loc[0, "abaixo_75"] == False

    # Aluno 2: 0.749 -> abaixo de 75%
    assert tab.loc[1, "freq_acumulada"] == 0.749
    assert tab.loc[1, "abaixo_75"] is True or tab.loc[1, "abaixo_75"] == True

    # Aluno 3: 0.751 -> acima de 75%
    assert tab.loc[2, "freq_acumulada"] == 0.751
    assert tab.loc[2, "abaixo_75"] is False or tab.loc[2, "abaixo_75"] == False


def test_mapeamento_de_bimestre_customizado() -> None:
    """Verifica que um mapeamento alternativo mês -> bimestre é respeitado."""
    custom_map = {
        1: ["fevereiro"],
        2: ["marco", "abril"],
    }
    df = pd.DataFrame(
        [
            {
                "matricula": "1",
                "nome": "Aluno Custom",
                "ha_ofertadas_fevereiro": 20.0,
                "ha_presenciadas_fevereiro": 16.0,
                "ha_ofertadas_marco": 100.0,
                "ha_presenciadas_marco": 80.0,
                "ha_ofertadas_abril": 50.0,
                "ha_presenciadas_abril": 40.0,
            }
        ]
    )

    periodo = periodo_apuracao(df, meses_por_bimestre=custom_map, bimestres=[1, 2])
    assert periodo[1]["meses_calendario"] == ["fevereiro"]
    assert periodo[1]["meses_lancados"] == ["fevereiro"]
    assert periodo[1]["parcial"] is False

    assert periodo[2]["meses_calendario"] == ["marco", "abril"]
    assert periodo[2]["meses_lancados"] == ["marco", "abril"]
    assert periodo[2]["parcial"] is False

    tab = tabela_frequencia(df, meses_por_bimestre=custom_map, bimestres=[1, 2])
    assert tab["freq_bim_1"].iloc[0] == 16.0 / 20.0  # 0.80
    assert tab["freq_bim_2"].iloc[0] == (80.0 + 40.0) / (100.0 + 50.0)  # 120/150 = 0.80
    assert tab["freq_acumulada"].iloc[0] == (16.0 + 120.0) / (20.0 + 150.0)  # 136/170 = 0.80


def test_bimestre_com_so_um_mes_lancado_parcial_true_e_meses_lancados_correto(
    caminho_xlsx: Path,
) -> None:
    """Verifica que bimestre com mês pendente tem parcial=True e meses_lancados exato.

    Na fixture, fev a ago estão lançados; set a nov estão vazios.
    No 3º bimestre default (agosto e setembro):
        - agosto está lançado
        - setembro não está lançado
        - parcial deve ser True
        - meses_lancados deve conter apenas ['agosto']
    """
    df = carregar_dae(caminho_xlsx)

    periodo = periodo_apuracao(df, MESES_POR_BIMESTRE)

    # 1º Bimestre: fev, mar, abr (todos lançados)
    assert periodo[1]["meses_calendario"] == ["fevereiro", "marco", "abril"]
    assert periodo[1]["meses_lancados"] == ["fevereiro", "marco", "abril"]
    assert periodo[1]["parcial"] is False

    # 2º Bimestre: mai, jun, jul (todos lançados)
    assert periodo[2]["meses_calendario"] == ["maio", "junho", "julho"]
    assert periodo[2]["meses_lancados"] == ["maio", "junho", "julho"]
    assert periodo[2]["parcial"] is False

    # 3º Bimestre: ago, set (apenas agosto lançado)
    assert periodo[3]["meses_calendario"] == ["agosto", "setembro"]
    assert periodo[3]["meses_lancados"] == ["agosto"]
    assert periodo[3]["parcial"] is True

    # 4º Bimestre: out, nov (nenhum lançado)
    assert periodo[4]["meses_calendario"] == ["outubro", "novembro"]
    assert periodo[4]["meses_lancados"] == []
    assert periodo[4]["parcial"] is True


def test_bimestres_1_2_ignora_meses_do_terceiro(caminho_xlsx: Path) -> None:
    """Verifica que filtrar bimestres=[1, 2] não contabiliza meses do 3º na acumulada."""
    df = carregar_dae(caminho_xlsx)

    tab_12 = tabela_frequencia(df, bimestres=[1, 2])

    # Colunas de saída
    assert "freq_bim_1" in tab_12.columns
    assert "freq_bim_2" in tab_12.columns
    assert "freq_bim_3" not in tab_12.columns
    assert "freq_bim_4" not in tab_12.columns
    assert "freq_acumulada" in tab_12.columns

    # Frequência acumulada deve ser idêntica ao cálculo exclusivo sobre fev-jul
    meses_12 = ["fevereiro", "marco", "abril", "maio", "junho", "julho"]
    esperada_12 = frequencia_ponderada(df, meses_12)
    pd.testing.assert_series_equal(tab_12["freq_acumulada"], esperada_12, check_names=False)

    # E deve divergir do acumulado incluindo agosto (3º bimestre)
    meses_com_ago = meses_12 + ["agosto"]
    esperada_com_ago = frequencia_ponderada(df, meses_com_ago)
    assert not tab_12["freq_acumulada"].equals(esperada_com_ago)


def test_mes_referencia(caminho_xlsx: Path) -> None:
    """Verifica identificação do mês de referência (D11): 'agosto' na fixture e None em base vazia."""
    df_fixture = carregar_dae(caminho_xlsx)
    assert mes_referencia(df_fixture) == "agosto"

    # Base vazia (sem linhas)
    df_vazio = pd.DataFrame()
    assert mes_referencia(df_vazio) is None

    # Base com colunas mas sem nenhum lançamento > 0
    df_sem_horas = pd.DataFrame(
        {
            "ha_ofertadas_fevereiro": [0.0, np.nan],
            "ha_ofertadas_marco": [np.nan, 0.0],
            "ha_ofertadas_abril": [np.nan, np.nan],
        }
    )
    assert mes_referencia(df_sem_horas) is None


def test_nenhuma_coluna_de_saida_por_mes(caminho_xlsx: Path) -> None:
    """Verifica a regra de D5: nenhuma coluna mensal permanece na tabela de frequência."""
    df = carregar_dae(caminho_xlsx)
    tab = tabela_frequencia(df, bimestres=[1, 2, 3])

    for col in tab.columns:
        assert not col.startswith("ha_ofertadas_"), f"Coluna mensal encontrada: {col}"
        assert not col.startswith("ha_presenciadas_"), f"Coluna mensal encontrada: {col}"
        for m in MESES_CALENDARIO:
            assert col != f"freq_{m}", f"Coluna mensal encontrada: {col}"

    colunas_obrigatorias = [
        "matricula",
        "nome",
        "freq_bim_1",
        "freq_bim_2",
        "freq_bim_3",
        "freq_acumulada",
        "abaixo_75",
        "diff_vs_dae_pp",
    ]
    for c in colunas_obrigatorias:
        assert c in tab.columns, f"Coluna esperada ausente: {c}"

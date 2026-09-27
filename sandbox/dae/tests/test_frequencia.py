"""Testes unitários para o módulo de cálculo de frequência (sandbox/dae/frequencia.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calendario import carregar_calendario, ch_lecionada
from carregar import carregar_dae
from frequencia import (
    MESES_CALENDARIO,
    MESES_POR_BIMESTRE,
    faixa_frequencia_por_faltas,
    frequencia_ponderada,
    frequencia_por_disciplina,
    frequencia_por_faltas,
    limite_faltas,
    mes_referencia,
    meses_lancados,
    meses_por_bimestre_do_calendario,
    periodo_apuracao,
    resumo_frequencia_por_disciplina,
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


def test_limite_faltas() -> None:
    """Verifica limite_faltas para diferentes cargas horárias (C6)."""
    assert limite_faltas(70) == 17
    assert limite_faltas(76) == 19
    assert limite_faltas(78) == 19
    assert limite_faltas(80) == 20
    assert limite_faltas(18) == 4
    assert limite_faltas(31) == 7


def test_frequencia_por_faltas_e_calendario() -> None:
    """Verifica frequencia_por_faltas com cargas horárias do calendário nos cenários A e REAL."""
    cal = carregar_calendario()

    # 2 aulas na TER, anual, A (CH 76): 19 faltas → exatamente 0,75, 20 → < 0,75
    assert ch_lecionada(cal, {"TER": 2}, cenario="A") == 76
    assert frequencia_por_faltas(19, {"TER": 2}, cal) == 0.75
    assert frequencia_por_faltas(20, {"TER": 2}, cal) < 0.75

    # 2 aulas na SEG (CH 70): 17 faltas → ≥ 0,75, 18 → < 0,75
    assert ch_lecionada(cal, {"SEG": 2}, cenario="A") == 70
    assert frequencia_por_faltas(17, {"SEG": 2}, cal) >= 0.75
    assert frequencia_por_faltas(18, {"SEG": 2}, cal) < 0.75

    # REAL Estradas SEX/SEX: 18 faltas → 1 − 18/72
    assert (
        ch_lecionada(cal, {"SEX": 2}, cenario="REAL", curso="TÉCNICO EM ESTRADAS", sabado_reproduz="SEX")
        == 72
    )
    res = frequencia_por_faltas(
        18, {"SEX": 2}, cal, "REAL", curso="TÉCNICO EM ESTRADAS", sabado_reproduz="SEX"
    )
    assert res == 1.0 - 18.0 / 72.0


def test_frequencia_por_faltas_series_e_ch_zero() -> None:
    """Verifica suporte a pd.Series e caso de CH 0 (arranjo vazio) retornando NaN."""
    # pd.Series devolve pd.Series com o mesmo índice
    s_faltas = pd.Series([19, 20], index=["aluno_1", "aluno_2"])
    res = frequencia_por_faltas(s_faltas, {"TER": 2}, cal=carregar_calendario())
    assert isinstance(res, pd.Series)
    assert list(res.index) == ["aluno_1", "aluno_2"]
    assert res.loc["aluno_1"] == 0.75
    assert res.loc["aluno_2"] < 0.75

    # arranjo vazio → CH lecionada 0 → NaN (escalar e Series)
    cal = carregar_calendario()
    assert np.isnan(frequencia_por_faltas(10, {}, cal))
    res_zero = frequencia_por_faltas(s_faltas, {}, cal)
    assert isinstance(res_zero, pd.Series)
    assert res_zero.isna().all()


def test_faixa_frequencia_por_faltas() -> None:
    """Verifica cálculo da faixa de frequência para 4 aulas semanais no cenário A."""
    cal = carregar_calendario()
    faixa = faixa_frequencia_por_faltas(20, 4, cal, "A")
    assert faixa == (1.0 - 20.0 / 140.0, 1.0 - 20.0 / 152.0)


def test_meses_por_bimestre_do_calendario() -> None:
    """Verifica meses_por_bimestre_do_calendario igual a MESES_POR_BIMESTRE com dezembro no 4º."""
    cal = carregar_calendario()
    mapa_cal = meses_por_bimestre_do_calendario(cal)
    esperado = dict(MESES_POR_BIMESTRE)
    esperado[4] = list(esperado[4]) + ["dezembro"]
    assert mapa_cal == esperado
    assert mapa_cal[4] == ["outubro", "novembro", "dezembro"]


def test_periodo_apuracao_com_calendario(caminho_xlsx: Path) -> None:
    """Verifica periodo_apuracao integrado ao calendário nos cenários A e REAL."""
    df = carregar_dae(caminho_xlsx)
    cal = carregar_calendario()

    p_sem = periodo_apuracao(df, bimestres=[1, 2])
    p_cal = periodo_apuracao(df, calendario=cal, bimestres=[1, 2])

    # meses_calendario/meses_lancados/parcial iguais aos da chamada sem calendário
    for b in [1, 2]:
        assert p_cal[b]["meses_calendario"] == p_sem[b]["meses_calendario"]
        assert p_cal[b]["meses_lancados"] == p_sem[b]["meses_lancados"]
        assert p_cal[b]["parcial"] == p_sem[b]["parcial"]

    # dias_letivos 50 e 48
    assert p_cal[1]["dias_letivos"] == 50
    assert p_cal[2]["dias_letivos"] == 48

    # dias_letivos_total 53 e 54
    assert p_cal[1]["dias_letivos_total"] == 53
    assert p_cal[2]["dias_letivos_total"] == 54

    # dias_por_mes do 1º = {"fevereiro":5,"marco":23,"abril":20,"maio":5}
    assert p_cal[1]["dias_por_mes"] == {
        "fevereiro": 5,
        "marco": 23,
        "abril": 20,
        "maio": 5,
    }

    # limite_diarios 22/05 e 14/08
    assert p_cal[1]["limite_diarios"] == "22/05"
    assert p_cal[2]["limite_diarios"] == "14/08"

    # com cenario="REAL", curso="TÉCNICO EM ESTRADAS" → dias_letivos 50 e 49
    p_real = periodo_apuracao(
        df,
        calendario=cal,
        bimestres=[1, 2],
        cenario="REAL",
        curso="TÉCNICO EM ESTRADAS",
    )
    assert p_real[1]["dias_letivos"] == 50
    assert p_real[2]["dias_letivos"] == 49


def test_frequencia_por_disciplina_e_resumo_memoria() -> None:
    """Verifica frequencia_por_disciplina e resumo_frequencia_por_disciplina com dados em memória (C17)."""
    # 1. Dados fictícios em memória: disciplina com ch_bim_1 = 20 e faltas [0, 5, 6]
    df_faltas = pd.DataFrame(
        {
            "matricula": ["20261010001", "20261010002", "20261010003"],
            "nome": ["Aluno A", "Aluno B", "Aluno C"],
            "MAT": [0, 5, 6],
            "SEM_CH": [0, 5, 6],
        }
    )
    legenda = {"MAT": "Matemática", "SEM_CH": "Disciplina Sem CH"}

    # Disciplina casada da planilha (MAT): ch_bim_1 = 20, ch_bim_2 = 18
    ch_casada = {
        "MAT": {
            "disciplina": "Matemática",
            "aulas_sem": 2,
            "ch_nominal": 80,
            "ch_bim_1": 20,
            "ch_bim_2": 18,
            "ch_efetiva": 76,
        }
    }

    # Apuração 1º BI
    freq_b1 = frequencia_por_disciplina(df_faltas, legenda, ch_casada, bimestre=1)
    resumo_b1 = resumo_frequencia_por_disciplina(df_faltas, legenda, ch_casada, bimestre=1)

    # MAT: ch_bim_1 = 20 e faltas [0, 5, 6] → frequências [1,0; 0,75; 0,70], n_abaixo_75 = 1, limite_faltas_bim = 5
    np.testing.assert_allclose(freq_b1["MAT"].tolist(), [1.0, 0.75, 0.70])
    mat_b1 = next(r for r in resumo_b1 if r["disciplina"] == "Matemática")
    assert mat_b1["fonte"] == "planilha"
    assert mat_b1["aulas_sem"] == 2
    assert mat_b1["ch_bim"] == 20
    assert mat_b1["ch_efetiva_ano"] == 76
    assert mat_b1["ch_nominal"] == 80
    assert mat_b1["%_nominal"] == 76 / 80
    assert mat_b1["limite_faltas_bim"] == 5
    assert mat_b1["n_abaixo_75"] == 1
    assert mat_b1["n_alunos"] == 3

    # Disciplina sem CH → coluna NaN e fonte "sem horário" com n_abaixo_75 None
    assert freq_b1["SEM_CH"].isna().all()
    sem_ch_b1 = next(r for r in resumo_b1 if r["disciplina"] == "Disciplina Sem CH")
    assert sem_ch_b1["fonte"] == "sem horário"
    assert sem_ch_b1["ch_bim"] is None
    assert sem_ch_b1["limite_faltas_bim"] is None
    assert sem_ch_b1["n_abaixo_75"] is None

    # A mesma disciplina sem CH com aulas_sem_estimadas={código: 2} no 1º BI
    # → fonte "estimada", ch_bim "18–22" e n_abaixo_75 contado sobre 18
    freq_est = frequencia_por_disciplina(
        df_faltas,
        legenda,
        ch_casada,
        bimestre=1,
    )
    resumo_est = resumo_frequencia_por_disciplina(
        df_faltas,
        legenda,
        ch_casada,
        bimestre=1,
        aulas_sem_estimadas={"SEM_CH": 2},
    )
    # C17: frequencia_por_disciplina só usa CH da planilha — SEM_CH segue NaN mesmo com estimativa
    assert freq_est["SEM_CH"].isna().all()

    sem_ch_est = next(r for r in resumo_est if r["disciplina"] == "Disciplina Sem CH")
    assert sem_ch_est["fonte"] == "estimada"
    assert sem_ch_est["ch_bim"] == "18–22"
    # n_abaixo_75 contado sobre 18: [1 - 0/18, 1 - 5/18, 1 - 6/18] -> 5 e 6 < 0.75 -> 2
    assert sem_ch_est["n_abaixo_75"] == 2
    assert sem_ch_est["limite_faltas_bim"] == 4

    # Bimestre 2 usa ch_bim_2 (para MAT, ch_bim_2 = 18)
    freq_b2 = frequencia_por_disciplina(df_faltas, legenda, ch_casada, bimestre=2)
    resumo_b2 = resumo_frequencia_por_disciplina(df_faltas, legenda, ch_casada, bimestre=2)
    mat_b2 = next(r for r in resumo_b2 if r["disciplina"] == "Matemática")

    assert mat_b2["ch_bim"] == 18
    assert mat_b2["limite_faltas_bim"] == 4
    assert mat_b2["n_abaixo_75"] == 2
    np.testing.assert_allclose(
        freq_b2["MAT"].tolist(),
        [1.0, 1.0 - 5 / 18, 1.0 - 6 / 18],
    )

    # O resumo não contém matrícula nem nome (só chaves de C17)
    assert all("matricula" not in r and "nome" not in r for r in resumo_b1)
    assert all("matricula" not in r and "nome" not in r for r in resumo_est)
    assert len(resumo_b1) == 2



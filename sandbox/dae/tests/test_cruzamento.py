"""Testes unitários para o cruzamento de dados DAE x App (sandbox/dae/cruzamento.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from carregar import carregar_dae
from cruzamento import _executar_cli, cruzar


def test_cruzamento_casa_e_sobra_de_cada_lado() -> None:
    """Verifica que o outer join identifica quem casa, quem sobra na DAE e quem sobra no app."""
    # DAE com 3 estudantes: 001, 002, 003
    df_dae = pd.DataFrame(
        [
            {
                "matricula": "20261010001",
                "nome": "Estudante Ambos 1",
                "ha_ofertadas_fevereiro": 20.0,
                "ha_presenciadas_fevereiro": 18.0,
            },
            {
                "matricula": "20261010002",
                "nome": "Estudante Ambos 2",
                "ha_ofertadas_fevereiro": 20.0,
                "ha_presenciadas_fevereiro": 15.0,
            },
            {
                "matricula": "20261010003",
                "nome": "Estudante Só DAE",
                "ha_ofertadas_fevereiro": 20.0,
                "ha_presenciadas_fevereiro": 20.0,
            },
        ]
    )

    # App com 3 estudantes: 001, 002, 004 (004 não está na DAE)
    df_faltas_app = pd.DataFrame(
        [
            {
                "matricula": "20261010001",
                "nome": "Estudante Ambos 1",
                "matematica": 2,
                "portugues": 0,
            },
            {
                "matricula": "20261010002",
                "nome": "Estudante Ambos 2",
                "matematica": 4,
                "portugues": 3,
            },
            {
                "matricula": "20261010004",
                "nome": "Estudante Só App",
                "matematica": 1,
                "portugues": 1,
            },
        ]
    )

    df_unido, resumo = cruzar(df_dae, df_faltas_app, bimestres=[1])

    # 1. Validação do Resumo de Cobertura
    assert resumo["so_app"] == 1
    assert resumo["so_dae"] == 1
    assert resumo["nos_dois"] == 2
    assert resumo["total"] == 4

    # 2. Validação do indicador _merge
    df_unido_indexed = df_unido.set_index("matricula")
    assert df_unido_indexed.loc["20261010001", "_merge"] == "both"
    assert df_unido_indexed.loc["20261010002", "_merge"] == "both"
    assert df_unido_indexed.loc["20261010003", "_merge"] == "left_only"
    assert df_unido_indexed.loc["20261010004", "_merge"] == "right_only"

    # 3. diff_faltas_bim_1 calculado somente para quem casa
    # Aluno 001: app faltas = 2+0=2; dae faltas = 20-18=2 => diff = 0
    assert df_unido_indexed.loc["20261010001", "diff_faltas_bim_1"] == 0.0
    # Aluno 002: app faltas = 4+3=7; dae faltas = 20-15=5 => diff = 7-5=2
    assert df_unido_indexed.loc["20261010002", "diff_faltas_bim_1"] == 2.0
    # Aluno 003 (só DAE): NaN
    assert np.isnan(df_unido_indexed.loc["20261010003", "diff_faltas_bim_1"])
    # Aluno 004 (só App): NaN
    assert np.isnan(df_unido_indexed.loc["20261010004", "diff_faltas_bim_1"])


def test_diff_faltas_bimestre_calculo_correto(caminho_xlsx: Path) -> None:
    """Verifica o cálculo exato de diff_faltas_bim_<n> para múltiplos meses do bimestre."""
    df_dae = carregar_dae(caminho_xlsx)
    # Alunos presentes na fixture caminho_xlsx:
    # 20261010001 (Ana Silva):
    #   fev: 20 ofer, 18 pres (diff 2)
    #   mar: 150 ofer, 120 pres (diff 30)
    #   abr: 140 ofer, 110 pres (diff 30)
    #   Total faltas DAE bim 1 = 62
    # 20261010002 (Bruno Souza):
    #   fev: 20 ofer, 15 pres (diff 5)
    #   mar: 150 ofer, 100 pres (diff 50)
    #   abr: 140 ofer, 95 pres (diff 45)
    #   Total faltas DAE bim 1 = 100

    df_faltas_app = pd.DataFrame(
        [
            {
                "matricula": "20261010001",
                "ART": 10,
                "BIO": 20,
                "MAT": 32,  # Soma = 62 -> diff esperada = 62 - 62 = 0
            },
            {
                "matricula": "20261010002",
                "ART": 20,
                "BIO": 30,
                "MAT": 45,  # Soma = 95 -> diff esperada = 95 - 100 = -5
            },
            {
                "matricula": "20261010099",  # Estudante que não existe na DAE
                "ART": 5,
                "BIO": 5,
                "MAT": 5,
            },
        ]
    )

    df_unido, resumo = cruzar(df_dae, df_faltas_app, bimestres=[1])

    assert resumo["nos_dois"] == 2
    assert resumo["so_app"] == 1
    assert resumo["so_dae"] == 2  # Estudantes 3 e 4 da DAE

    idx = df_unido.set_index("matricula")
    assert idx.loc["20261010001", "diff_faltas_bim_1"] == 0.0
    assert idx.loc["20261010002", "diff_faltas_bim_1"] == -5.0
    assert np.isnan(idx.loc["20261010099", "diff_faltas_bim_1"])


def test_cruzamento_multiplos_bimestres() -> None:
    """Verifica a comparação independente para múltiplos bimestres (ex.: bim 1 e bim 2)."""
    df_dae = pd.DataFrame(
        [
            {
                "matricula": "20261010001",
                # Bimestre 1: fev, mar, abr
                "ha_ofertadas_fevereiro": 20.0,
                "ha_presenciadas_fevereiro": 18.0,  # 2
                "ha_ofertadas_marco": 100.0,
                "ha_presenciadas_marco": 90.0,      # 10
                "ha_ofertadas_abril": 80.0,
                "ha_presenciadas_abril": 75.0,      # 5 => faltas bim 1 DAE = 17
                # Bimestre 2: mai, jun, jul
                "ha_ofertadas_maio": 100.0,
                "ha_presenciadas_maio": 90.0,       # 10
                "ha_ofertadas_junho": 100.0,
                "ha_presenciadas_junho": 85.0,      # 15
                "ha_ofertadas_julho": 50.0,
                "ha_presenciadas_julho": 45.0,      # 5 => faltas bim 2 DAE = 30
            }
        ]
    )

    # df_faltas_app em formato de dicionário por bimestre
    df_faltas_app = {
        1: pd.DataFrame(
            [{"matricula": "20261010001", "MAT": 12, "POR": 8}]  # Soma = 20 => diff = 20 - 17 = 3
        ),
        2: pd.DataFrame(
            [{"matricula": "20261010001", "MAT": 15, "POR": 15}]  # Soma = 30 => diff = 30 - 30 = 0
        ),
    }

    df_unido, resumo = cruzar(df_dae, df_faltas_app, bimestres=[1, 2])

    assert resumo["nos_dois"] == 1
    assert "diff_faltas_bim_1" in df_unido.columns
    assert "diff_faltas_bim_2" in df_unido.columns

    row = df_unido.iloc[0]
    assert row["diff_faltas_bim_1"] == 3.0
    assert row["diff_faltas_bim_2"] == 0.0


def test_cruzamento_com_conjuntos_processar_multiplos_bimestres() -> None:
    """Verifica que a função aceita a lista de tuplas retornada por processar_multiplos_bimestres."""
    df_dae = pd.DataFrame(
        [
            {
                "matricula": "20261010001",
                "ha_ofertadas_fevereiro": 20.0,
                "ha_presenciadas_fevereiro": 16.0,  # 4 faltas DAE
            }
        ]
    )

    # Estrutura sintética imitando o retorno de processar_multiplos_bimestres:
    # (df_notas, df_faltas, disc_dict, meta)
    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 15.0}])
    df_faltas = pd.DataFrame([{"matricula": "20261010001", "MAT": 6}])  # 6 faltas App
    meta = {"bimestre_num": 1, "curso": "Trânsito"}
    conjuntos = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    df_unido, resumo = cruzar(df_dae, conjuntos)

    assert resumo["nos_dois"] == 1
    # diff = 6 (app) - 4 (dae) = 2.0
    assert df_unido.iloc[0]["diff_faltas_bim_1"] == 2.0


def test_cli_apenas_agregados_sem_vazar_nomes_ou_matriculas(
    caminho_xlsx: Path, capsys: pytest.CaptureFixture
) -> None:
    """Verifica que a execução CLI imprime agregados, mês de referência e nunca nomes/matrículas."""
    # Mock do processar_multiplos_bimestres para retornar dados de 2 estudantes
    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 18.0}])
    df_faltas = pd.DataFrame(
        [
            {"matricula": "20261010001", "nome": "Ana Silva", "MAT": 62},
            {"matricula": "20261010002", "nome": "Bruno Souza", "MAT": 95},
        ]
    )
    meta = {"bimestre_num": 1, "curso": "Técnico em Trânsito"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    with patch("cruzamento.processar_multiplos_bimestres", return_value=conjuntos_mock):
        codigo_retorno = _executar_cli(
            ["--dae", str(caminho_xlsx), "--mapas", "fake_mapa_bim1.xls"]
        )

    assert codigo_retorno == 0
    saida = capsys.readouterr().out

    # 1. Deve imprimir mês de referência da DAE
    assert "Mês de referência DAE:" in saida

    # 2. Deve imprimir os meses usados por bimestre
    assert "Bimestre 1:" in saida
    assert "meses usados =" in saida

    # 3. Deve imprimir contagens agregadas
    assert "Resumo de Cobertura:" in saida
    assert "Só no App:" in saida
    assert "Só na DAE:" in saida
    assert "Nos dois:" in saida

    # 4. Deve imprimir estatísticas agregadas (mediana e p90 de |diff_faltas|)
    assert "Discrepância de Faltas (|diff_faltas|) por Bimestre:" in saida
    assert "Mediana (|diff_faltas|):" in saida
    assert "Percentil 90 (|diff_faltas|):" in saida

    # 5. LGPD (D1, D4): NUNCA deve imprimir nomes ou números de matrícula
    assert "Ana Silva" not in saida
    assert "Bruno Souza" not in saida
    assert "Carlos Lima" not in saida
    assert "Daniela Rocha" not in saida
    assert "20261010001" not in saida
    assert "20261010002" not in saida
    assert "20261010003" not in saida
    assert "20261010004" not in saida

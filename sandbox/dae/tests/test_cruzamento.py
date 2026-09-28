"""Testes unitários para o cruzamento de dados DAE x App (sandbox/dae/cruzamento.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from carregar import carregar_dae
from cruzamento import _eh_matricula_valida, _executar_cli, contagem_pe_de_meia, cruzar, resumo_mapa
from tests.conftest import mapa_sintetico


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


def test_resumo_mapa_metricas_e_validacoes() -> None:
    """Verifica que resumo_mapa conta alunos, duplicadas, matrícula de 10 dígitos como inválida, disciplinas e faltas."""
    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 10.0}])
    df_faltas = pd.DataFrame(
        [
            {"matricula": "20261010001", "MAT": 10, "POR": 5},  # válida (15 faltas)
            {"matricula": "2026101000", "MAT": 2, "POR": 3},    # 10 dígitos (inválida, 5 faltas)
            {"matricula": "2026101000", "MAT": 4, "POR": 1},    # duplicada (5 faltas)
            {"matricula": "20261010002", "MAT": 6, "POR": 9},   # válida (15 faltas)
        ]
    )
    meta = {"bimestre_num": 1, "curso": "Técnico em Estradas"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática", "POR": "Português"}, meta)]

    res = resumo_mapa(conjuntos_mock)
    r0 = res[0]

    assert r0["n_alunos"] == 4
    assert r0["n_matriculas_duplicadas"] == 1
    assert r0["n_matriculas_validas"] == 2
    assert r0["n_disciplinas"] == 2
    assert r0["faltas_total"] == 40
    assert r0["bimestre_num"] == 1
    assert r0["curso_amigavel"] == "Técnico em Estradas"
    # Medianas das somas por aluno [15, 5, 5, 15] = 10.0
    assert r0["faltas_mediana_aluno"] == 10.0


def test_contagem_pe_de_meia_filtros_e_zeros() -> None:
    """Verifica contagem_pe_de_meia com curso_contem='estradas', restrição por matriculas e presença de zeros."""
    df_dae = pd.DataFrame(
        [
            {"matricula": "20261010001", "curso": "TÉCNICO EM ESTRADAS", "pe_de_meia": "elegivel"},
            {"matricula": "20261010002", "curso": "Tecnico em Estradas", "pe_de_meia": "nada_consta"},
            {"matricula": "20261010003", "curso": "TÉCNICO EM TRÂNSITO", "pe_de_meia": "elegivel"},
            {"matricula": "20261010004", "curso": "Tecnico em Estradas", "pe_de_meia": "elegivel"},
        ]
    )

    # 1. curso_contem="estradas" casa "TÉCNICO EM ESTRADAS" e "Tecnico em Estradas"
    contagem_estradas = contagem_pe_de_meia(df_dae, curso_contem="estradas")
    assert contagem_estradas["elegivel"] == 2
    assert contagem_estradas["nada_consta"] == 1
    assert contagem_estradas["nao_elegivel"] == 0
    assert contagem_estradas["indefinida"] == 0
    assert contagem_estradas["total"] == 3
    # Zeros e chaves presentes
    assert "nao_elegivel" in contagem_estradas
    assert "indefinida" in contagem_estradas
    assert "nada_consta" in contagem_estradas

    # Matrícula de 11 dígitos sem prefixo "20" continua válida
    assert _eh_matricula_valida("12345678901")

    # 2. matriculas restringe
    contagem_restrita = contagem_pe_de_meia(
        df_dae,
        curso_contem="estradas",
        matriculas=["20261010001", "20261010003"],
    )
    assert contagem_restrita["elegivel"] == 1
    assert contagem_restrita["nada_consta"] == 0
    assert contagem_restrita["nao_elegivel"] == 0
    assert contagem_restrita["indefinida"] == 0
    assert contagem_restrita["total"] == 1


def test_cli_so_com_mapas_retorno_zero_e_pendente(capsys: pytest.CaptureFixture) -> None:
    """Verifica que CLI chamado apenas com --mapas retorna 0, exibe PENDENTE: e preserva LGPD."""
    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 15.0}])
    df_faltas = pd.DataFrame(
        [
            {"matricula": "20261010001", "nome": "Ana Silva", "MAT": 10},
            {"matricula": "20261010002", "nome": "Bruno Souza", "MAT": 20},
        ]
    )
    meta = {"bimestre_num": 1, "curso": "Técnico em Estradas"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    with patch("cruzamento.processar_multiplos_bimestres", return_value=conjuntos_mock):
        codigo = _executar_cli(["--mapas", "fake_mapa.xls"])

    assert codigo == 0
    saida = capsys.readouterr().out
    assert "PENDENTE:" in saida
    assert "20261010001" not in saida
    assert "20261010002" not in saida
    assert "Ana Silva" not in saida
    assert "Bruno Souza" not in saida


def test_cli_sem_nada_com_pasta_dados_vazia_retorno_um(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que o CLI sem argumentos retorna 1 quando PASTA_DADOS está vazia."""
    monkeypatch.setattr("cruzamento.PASTA_DADOS", tmp_path)
    codigo = _executar_cli([])
    assert codigo == 1


def test_cli_descoberta_com_mapa_e_apenas_planilha_ch_modo_so_mapa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verifica que tmp_path com .xls + CH_Efetiva... entra em modo só-mapa com PENDENTE: (CH não é tomada por DAE)."""
    (tmp_path / "Estradas_ficticio.xls").touch()
    (tmp_path / "CH_Efetiva_Disciplinas_Integrado_2026.xlsx").touch()
    monkeypatch.setattr("cruzamento.PASTA_DADOS", tmp_path)

    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 15.0}])
    df_faltas = pd.DataFrame([{"matricula": "20261010001", "nome": "Ana Silva", "MAT": 5}])
    meta = {"bimestre_num": 1, "curso": "Técnico em Estradas"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    with patch("cruzamento.processar_multiplos_bimestres", return_value=conjuntos_mock):
        codigo = _executar_cli([])

    assert codigo == 0
    saida = capsys.readouterr().out
    assert "PENDENTE:" in saida
    assert "20261010001" not in saida
    assert "Ana Silva" not in saida


def test_cli_descoberta_com_mapa_dae_e_planilha_ch_cruzamento_completo(
    tmp_path: Path,
    caminho_xlsx: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Verifica descoberta com .xls fictício + DAE sintético + planilha CH: cruzamento completo com N/C (Nada consta)."""
    import shutil

    (tmp_path / "Estradas_ficticio.xls").touch()
    (tmp_path / "CH_Efetiva_Disciplinas_Integrado_2026.xlsx").touch()
    shutil.copy(caminho_xlsx, tmp_path / "dae_sintetico_2026.xlsx")
    monkeypatch.setattr("cruzamento.PASTA_DADOS", tmp_path)

    df_notas = pd.DataFrame([{"matricula": "20261010001", "MAT": 18.0}])
    df_faltas = pd.DataFrame(
        [
            {"matricula": "20261010001", "nome": "Ana Silva", "MAT": 62},
            {"matricula": "20261010002", "nome": "Bruno Souza", "MAT": 95},
        ]
    )
    meta = {"bimestre_num": 1, "curso": "Técnico em Estradas"}
    conjuntos_mock = [(df_notas, df_faltas, {"MAT": "Matemática"}, meta)]

    with patch("cruzamento.processar_multiplos_bimestres", return_value=conjuntos_mock):
        codigo = _executar_cli([])

    assert codigo == 0
    saida = capsys.readouterr().out
    assert "Resumo de Cobertura:" in saida
    assert "N/C (Nada consta)" in saida
    assert "PENDENTE:" not in saida
    assert "20261010001" not in saida
    assert "Ana Silva" not in saida


def test_contagem_pe_de_meia_tupla_cursos_uniao() -> None:
    """Verifica que contagem_pe_de_meia com tupla casa 'TÉCNICO EM ESTRADAS' e 'Técnico em Trânsito' e soma."""
    df_dae = pd.DataFrame(
        [
            {"matricula": "20261010001", "curso": "TÉCNICO EM ESTRADAS", "pe_de_meia": "elegivel"},
            {"matricula": "20261010002", "curso": "Técnico em Trânsito", "pe_de_meia": "elegivel"},
            {"matricula": "20261010003", "curso": "TÉCNICO EM EDIFICAÇÕES", "pe_de_meia": "elegivel"},
            {"matricula": "20261010004", "curso": "Técnico em Estradas", "pe_de_meia": "nada_consta"},
            {"matricula": "20261010005", "curso": "TÉCNICO EM TRÂNSITO", "pe_de_meia": "Não elegível"},
        ]
    )
    contagem = contagem_pe_de_meia(df_dae, curso_contem=("estradas", "transito"))
    assert contagem["elegivel"] == 2
    assert contagem["nada_consta"] == 1
    assert contagem["nao_elegivel"] == 1
    assert contagem["indefinida"] == 0
    assert contagem["total"] == 4


def test_cli_det_dois_mapas_sinteticos_sem_dae(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """CLI com dois mapas sintéticos EST+TT (via monkeypatch como no passo 2) sem DAE → retorno 0, 'Estradas + Trânsito', PENDENTE:, nenhuma matrícula/nome."""
    import core.manipulacao as manipulacao

    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST.2A",
        disciplinas={"TOP": "TOPOGRAFIA"},
        alunos=[
            ("20261010001", "Ana Silva", {"TOP": 15.0}, {"TOP": 0}),
            ("20261010003", "Carlos Lima", {"TOP": 18.0}, {"TOP": 0}),
        ],
    )
    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TRA.2A",
        disciplinas={"TRA": "TRANSPORTES"},
        alunos=[
            ("20261010001", "Ana Silva", {"TRA": 16.0}, {"TRA": 0}),
            ("20261010002", "Bruno Souza", {"TRA": 17.0}, {"TRA": 0}),
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

    (tmp_path / "Estradas_2026.xls").touch()
    (tmp_path / "Transito_2026.xls").touch()
    monkeypatch.setattr("cruzamento.PASTA_DADOS", tmp_path)

    codigo = _executar_cli([])

    assert codigo == 0
    saida = capsys.readouterr().out
    assert "Estradas + Trânsito" in saida
    assert "PENDENTE:" in saida
    assert "DET" in saida
    assert "interseção 0" in saida

    # LGPD: nenhuma matrícula ou nome discente pode vazar
    for mat in ("20261010001", "20261010002", "20261010003"):
        assert mat not in saida
    for nome in ("Ana", "Silva", "Bruno", "Souza", "Carlos", "Lima"):
        assert nome not in saida


def test_cli_det_dois_mapas_sinteticos_com_dae(
    tmp_path: Path,
    caminho_xlsx: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """CLI com dois mapas sintéticos EST+TT e .xlsx sintético da DAE → cobertura sobre o conjunto DET e 'N/C (Nada consta)'."""
    import shutil
    import core.manipulacao as manipulacao

    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST.2A",
        disciplinas={"TOP": "TOPOGRAFIA"},
        alunos=[
            ("20261010001", "Ana Silva", {"TOP": 15.0}, {"TOP": 0}),
            ("20261010003", "Carlos Lima", {"TOP": 18.0}, {"TOP": 0}),
        ],
    )
    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TRA.2A",
        disciplinas={"TRA": "TRANSPORTES"},
        alunos=[
            ("20261010001", "Ana Silva", {"TRA": 16.0}, {"TRA": 0}),
            ("20261010002", "Bruno Souza", {"TRA": 17.0}, {"TRA": 0}),
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

    (tmp_path / "Estradas_2026.xls").touch()
    (tmp_path / "Transito_2026.xls").touch()
    shutil.copy(caminho_xlsx, tmp_path / "dae_sintetico_2026.xlsx")
    monkeypatch.setattr("cruzamento.PASTA_DADOS", tmp_path)

    codigo = _executar_cli([])

    assert codigo == 0
    saida = capsys.readouterr().out
    assert "Resumo de Cobertura:" in saida
    assert "N/C (Nada consta)" in saida
    assert "Estradas + Trânsito" in saida
    assert "PENDENTE:" not in saida

    # Cobertura sobre o conjunto DET (3 discentes no mapa, 4 na DAE: 3 nos dois, 1 só DAE, 0 só app)
    assert "Nos dois: 3" in saida
    assert "Só no App: 0" in saida
    assert "Só na DAE: 1" in saida
    assert "Total: 4" in saida

    # LGPD: nenhuma matrícula ou nome discente pode vazar
    for mat in ("20261010001", "20261010002", "20261010003", "20261010004"):
        assert mat not in saida
    for nome in ("Ana", "Silva", "Bruno", "Souza", "Carlos", "Lima", "Daniela", "Rocha"):
        assert nome not in saida


def test_cruzamento_direto_com_conjuntos_det(
    caminho_xlsx: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que cruzar aceita diretamente a estrutura conjuntos_det."""
    from det import carregar_det, conjuntos_det
    import core.manipulacao as manipulacao

    df_est = mapa_sintetico(
        curso="TÉCNICO EM ESTRADAS",
        bimestre=1,
        turma="EST.2A",
        disciplinas={"TOP": "TOPOGRAFIA"},
        alunos=[
            ("20261010001", "Ana Silva", {"TOP": 15.0}, {"TOP": 2}),
            ("20261010003", "Carlos Lima", {"TOP": 18.0}, {"TOP": 0}),
        ],
    )
    df_tt = mapa_sintetico(
        curso="TÉCNICO EM TRÂNSITO",
        bimestre=1,
        turma="TRA.2A",
        disciplinas={"TRA": "TRANSPORTES"},
        alunos=[
            ("20261010001", "Ana Silva", {"TRA": 16.0}, {"TRA": 0}),
            ("20261010002", "Bruno Souza", {"TRA": 17.0}, {"TRA": 5}),
        ],
    )

    original_ler = manipulacao._ler_xls_bruto

    def _mock_ler(arquivo_xls):
        if isinstance(arquivo_xls, pd.DataFrame):
            return arquivo_xls
        return original_ler(arquivo_xls)

    monkeypatch.setattr(manipulacao, "_ler_xls_bruto", _mock_ler)

    cd = carregar_det([df_est, df_tt])
    assert isinstance(cd, conjuntos_det)

    df_dae = carregar_dae(caminho_xlsx)
    df_unido, resumo = cruzar(df_dae, cd)

    assert resumo["nos_dois"] == 3
    assert resumo["so_app"] == 0
    assert resumo["so_dae"] == 1
    assert resumo["total"] == 4



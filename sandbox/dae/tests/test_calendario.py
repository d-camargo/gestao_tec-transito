"""Testes unitários para o módulo de calendário acadêmico (sandbox/dae/calendario.py).

Testa a carga e validações C1/C3/C4 a partir do Markdown oficial versionado
e através de arquivos sintéticos mutados em tmp_path.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import pytest

from calendario import (
    ANO_PADRAO,
    BLOCOS_POR_CH,
    CAMINHO_MD_PADRAO,
    CENARIOS,
    DIAS_SEMANA,
    DIAS_UTEIS,
    SEMANAS_NOMINAIS,
    Calendario,
    carregar_calendario,
    ch_lecionada,
    ch_nominal,
    dias_letivos,
    dias_por_mes_bimestre,
    divergencias,
    faixa_ch,
    sabados_do_responsavel,
)


def test_constantes_exportadas() -> None:
    """Verifica as constantes essenciais definidas no módulo (C1, C4)."""
    assert BLOCOS_POR_CH == {1: (1,), 2: (2,), 3: (2, 1), 4: (2, 2)}
    assert DIAS_UTEIS == ("SEG", "TER", "QUA", "QUI", "SEX")
    assert DIAS_SEMANA == ("SEG", "TER", "QUA", "QUI", "SEX", "SAB")
    assert CENARIOS == ("A", "REAL")
    assert ANO_PADRAO == 2026
    assert SEMANAS_NOMINAIS == 40


def test_ano_calendario() -> None:
    """Verifica se o ano do calendário oficial versionado é 2026."""
    cal = carregar_calendario()
    assert cal.ano == 2026


def test_periodos_bimestres() -> None:
    """Verifica os períodos oficiais de cada bimestre (23/02–08/05, 11/05–17/07, 03/08–03/10, 05/10–04/12)."""
    cal = carregar_calendario()
    assert cal.bimestres[1].inicio == date(2026, 2, 23)
    assert cal.bimestres[1].fim == date(2026, 5, 8)
    assert cal.bimestres[2].inicio == date(2026, 5, 11)
    assert cal.bimestres[2].fim == date(2026, 7, 17)
    assert cal.bimestres[3].inicio == date(2026, 8, 3)
    assert cal.bimestres[3].fim == date(2026, 10, 3)
    assert cal.bimestres[4].inicio == date(2026, 10, 5)
    assert cal.bimestres[4].fim == date(2026, 12, 4)


def test_dias_letivos_e_acumulado() -> None:
    """Verifica os dias letivos por bimestre (53, 54, 49, 44) e o acumulado (53, 107, 156, 200)."""
    cal = carregar_calendario()
    assert [cal.bimestres[i].dias_letivos for i in (1, 2, 3, 4)] == [53, 54, 49, 44]
    assert [cal.bimestres[i].acumulado for i in (1, 2, 3, 4)] == [53, 107, 156, 200]


def test_limite_diarios() -> None:
    """Verifica as datas-limite para fechamento de diários (22/05, 14/08, 09/10, 07/12)."""
    cal = carregar_calendario()
    assert cal.bimestres[1].limite_diarios == date(2026, 5, 22)
    assert cal.bimestres[2].limite_diarios == date(2026, 8, 14)
    assert cal.bimestres[3].limite_diarios == date(2026, 10, 9)
    assert cal.bimestres[4].limite_diarios == date(2026, 12, 7)

    assert cal.limite_diarios == {
        1: date(2026, 5, 22),
        2: date(2026, 8, 14),
        3: date(2026, 10, 9),
        4: date(2026, 12, 7),
    }


def test_dias_semana_tabela_oficial() -> None:
    """Verifica se dias_semana espelha exatamente a tabela oficial do Objetivo (linhas e Soma)."""
    cal = carregar_calendario()

    # Linhas dos bimestres
    assert cal.dias_semana[1]["SEG"] == 10
    assert cal.dias_semana[1]["TER"] == 10
    assert cal.dias_semana[1]["QUA"] == 11
    assert cal.dias_semana[1]["QUI"] == 10
    assert cal.dias_semana[1]["SEX"] == 9
    assert cal.dias_semana[1]["SAB"] == 3
    assert cal.dias_semana[1]["Total"] == 53

    assert cal.dias_semana[2]["SEG"] == 10
    assert cal.dias_semana[2]["TER"] == 10
    assert cal.dias_semana[2]["QUA"] == 10
    assert cal.dias_semana[2]["QUI"] == 9
    assert cal.dias_semana[2]["SEX"] == 9
    assert cal.dias_semana[2]["SAB"] == 6
    assert cal.dias_semana[2]["Total"] == 54

    assert cal.dias_semana[3]["SEG"] == 8
    assert cal.dias_semana[3]["TER"] == 9
    assert cal.dias_semana[3]["QUA"] == 9
    assert cal.dias_semana[3]["QUI"] == 9
    assert cal.dias_semana[3]["SEX"] == 9
    assert cal.dias_semana[3]["SAB"] == 5
    assert cal.dias_semana[3]["Total"] == 49

    assert cal.dias_semana[4]["SEG"] == 7
    assert cal.dias_semana[4]["TER"] == 9
    assert cal.dias_semana[4]["QUA"] == 8
    assert cal.dias_semana[4]["QUI"] == 9
    assert cal.dias_semana[4]["SEX"] == 8
    assert cal.dias_semana[4]["SAB"] == 3
    assert cal.dias_semana[4]["Total"] == 44

    # Linha Soma: SEG 35, TER 38, QUA 38, QUI 37, SEX 35, SAB 17 (Total 200)
    soma = cal.dias_semana["Soma"]
    assert soma["SEG"] == 35
    assert soma["TER"] == 38
    assert soma["QUA"] == 38
    assert soma["QUI"] == 37
    assert soma["SEX"] == 35
    assert soma["SAB"] == 17
    assert soma["Total"] == 200

    # Acesso direto aos totais anuais por dia útil
    assert cal.dias_semana["SEG"] == 35
    assert cal.dias_semana["TER"] == 38
    assert cal.dias_semana["QUA"] == 38
    assert cal.dias_semana["QUI"] == 37
    assert cal.dias_semana["SEX"] == 35
    assert cal.dias_semana["SAB"] == 17


def test_dias_mes() -> None:
    """Verifica a contagem mensal de dias letivos ({2:5, ..., 12:4}) e a soma anual de 200 dias."""
    cal = carregar_calendario()
    esperado = {2: 5, 3: 23, 4: 20, 5: 22, 6: 23, 7: 14, 8: 23, 9: 23, 10: 22, 11: 21, 12: 4}
    assert cal.dias_mes == esperado
    assert sum(cal.dias_mes.values()) == 200


def test_sabados_letivos() -> None:
    """Verifica os 17 sábados letivos (3, 6, 5, 3) e o sábado de Estradas e Trânsito em 23/05."""
    cal = carregar_calendario()
    assert len(cal.sabados) == 17
    assert [len(cal.bimestres[i].sabados) for i in (1, 2, 3, 4)] == [3, 6, 5, 3]

    sab_23_05 = [s for s in cal.bimestres[2].sabados if s.data == date(2026, 5, 23)]
    assert len(sab_23_05) == 1
    resp = sab_23_05[0].responsavel
    assert "Estradas" in resp
    assert "Trânsito" in resp


def test_sem_aula_4o_bimestre() -> None:
    """Verifica se sem_aula do 4º BI contém 12/10, 28/10, 02/11 e 20/11."""
    cal = carregar_calendario()
    sem_aula_4 = cal.bimestres[4].sem_aula
    assert date(2026, 10, 12) in sem_aula_4
    assert date(2026, 10, 28) in sem_aula_4
    assert date(2026, 11, 2) in sem_aula_4
    assert date(2026, 11, 20) in sem_aula_4


# ==============================================================================
# Casos de erro com arquivos Markdown alterados em tmp_path (C3)
# ==============================================================================


def test_erro_total_linha_errado(tmp_path: Path) -> None:
    """Verifica que erro na soma da linha da tabela de dias por semana gera ValueError com a seção."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    mod = texto.replace(
        "| 1º BI | 10 | 10 | 11 | 10 | 9 | 3 | 53 |",
        "| 1º BI | 10 | 10 | 11 | 10 | 9 | 3 | 54 |",
    )
    p = tmp_path / "cal_total_errado.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="Dias letivos por dia da semana"):
        carregar_calendario(caminho=p)


def test_erro_valor_negativo(tmp_path: Path) -> None:
    """Verifica que valor negativo gera ValueError citando a seção correspondente."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    mod = texto.replace(
        "| 1º BI | 10 | 10 | 11 | 10 | 9 | 3 | 53 |",
        "| 1º BI | -1 | 10 | 11 | 10 | 9 | 3 | 42 |",
    )
    p = tmp_path / "cal_negativo.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="Dias letivos por dia da semana"):
        carregar_calendario(caminho=p)


def test_erro_acumulado_errado(tmp_path: Path) -> None:
    """Verifica que inconsistência no Acumulado gera ValueError citando 'Visão geral'."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    mod = texto.replace(
        "| 2º | 11/05 (seg) | 17/07 (sex) | 54 | 107 | 14/08 (sex) |",
        "| 2º | 11/05 (seg) | 17/07 (sex) | 54 | 108 | 14/08 (sex) |",
    )
    p = tmp_path / "cal_acum_errado.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="Visão geral"):
        carregar_calendario(caminho=p)


def test_erro_tabela_por_mes_somando_199(tmp_path: Path) -> None:
    """Verifica que tabela por mês com soma 199 gera ValueError citando 'Dias letivos por mês'."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    mod = texto.replace(
        "| 5 | 23 | 20 | 22 | 23 | 14 | 23 | 23 | 22 | 21 | 4 | 200 |",
        "| 4 | 23 | 20 | 22 | 23 | 14 | 23 | 23 | 22 | 21 | 4 | 199 |",
    )
    p = tmp_path / "cal_mes_199.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="Dias letivos por mês"):
        carregar_calendario(caminho=p)


def test_erro_um_sabado_a_menos_no_1o_bi(tmp_path: Path) -> None:
    """Verifica que divergência entre sábados listados e previstos gera ValueError citando a seção."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    mod = texto.replace("| 25/04 | Área de Inglês (DELTEC) |\n", "")
    p = tmp_path / "cal_menos_sabado.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="1º Bimestre"):
        carregar_calendario(caminho=p)


def test_erro_sabado_com_data_de_sexta_feira(tmp_path: Path) -> None:
    """Verifica que sábado letivo com data de outro dia da semana gera ValueError citando a seção."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    # 24/04/2026 cai em uma sexta-feira
    mod = texto.replace("| 25/04 | Área de Inglês (DELTEC) |", "| 24/04 | Área de Inglês (DELTEC) |")
    p = tmp_path / "cal_sabado_sexta.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="1º Bimestre"):
        carregar_calendario(caminho=p)


def test_erro_secao_dias_letivos_por_mes_removida(tmp_path: Path) -> None:
    """Verifica que a ausência da seção obrigatória 'Dias letivos por mês' gera ValueError citando-a."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    mod = re.sub(r"### Dias letivos por mês.*?(?=---)", "", texto, flags=re.DOTALL)
    p = tmp_path / "cal_sem_secao_mes.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="Dias letivos por mês"):
        carregar_calendario(caminho=p)


def test_copia_valida_movendo_dia_altera_dias_semana(tmp_path: Path) -> None:
    """Verifica que uma cópia válida com 1 dia movido de SEG para TER altera dias_semana dinamicamente."""
    cal_orig = carregar_calendario()
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")

    # Move 1 dia de SEG (10 -> 9) para TER (10 -> 11) no 1º BI, mantendo Total 53.
    # Na Soma, move de SEG (35 -> 34) para TER (38 -> 39), mantendo Total 200.
    mod = texto.replace(
        "| 1º BI | 10 | 10 | 11 | 10 | 9 | 3 | 53 |",
        "| 1º BI | 9 | 11 | 11 | 10 | 9 | 3 | 53 |",
    ).replace(
        "| **Soma** | **35** | **38** | **38** | **37** | **35** | **17** | **200** |",
        "| **Soma** | **34** | **39** | **38** | **37** | **35** | **17** | **200** |",
    )
    p = tmp_path / "cal_valido_movido.md"
    p.write_text(mod, encoding="utf-8")

    cal_mod = carregar_calendario(caminho=p)

    # 1º Bimestre teve os valores alterados
    assert cal_mod.dias_semana[1]["SEG"] == 9
    assert cal_mod.dias_semana[1]["TER"] == 11
    assert cal_orig.dias_semana[1]["SEG"] == 10
    assert cal_orig.dias_semana[1]["TER"] == 10

    # Linha Soma refletiu a mudança
    assert cal_mod.dias_semana["Soma"]["SEG"] == 34
    assert cal_mod.dias_semana["Soma"]["TER"] == 39
    assert cal_mod.dias_semana["SEG"] == 34
    assert cal_mod.dias_semana["TER"] == 39

    # Os outros bimestres continuam iguais
    assert cal_mod.dias_semana[2] == cal_orig.dias_semana[2]


# ==============================================================================
# Testes de cálculo de dias letivos, sábados do responsável e divergências (Passo 2)
# ==============================================================================


def test_sabados_do_responsavel() -> None:
    """Verifica a identificação dos sábados letivos por responsável e bimestre (C4, C5)."""
    cal = carregar_calendario()
    ano = cal.ano

    # sabados_do_responsavel — "TÉCNICO EM ESTRADAS" → [23/05]
    assert sabados_do_responsavel(cal, "TÉCNICO EM ESTRADAS") == [date(ano, 5, 23)]
    # "Trânsito" → [23/05]
    assert sabados_do_responsavel(cal, "Trânsito") == [date(ano, 5, 23)]
    # "Mecatrônica" → [30/05, 20/06]
    assert sabados_do_responsavel(cal, "Mecatrônica") == [date(ano, 5, 30), date(ano, 6, 20)]
    # "Matemática" → [21/03]
    assert sabados_do_responsavel(cal, "Matemática") == [date(ano, 3, 21)]
    # "Meio Ambiente" → [13/06]
    assert sabados_do_responsavel(cal, "Meio Ambiente") == [date(ano, 6, 13)]
    # "Meio Ambiente Inexistente" → [] (todas as palavras significativas são exigidas)
    assert sabados_do_responsavel(cal, "Meio Ambiente Inexistente") == []
    # "Estradas" com bimestres=[1] → []
    assert sabados_do_responsavel(cal, "Estradas", bimestres=[1]) == []


def test_dias_letivos() -> None:
    """Verifica a contagem de dias letivos nos cenários A e REAL e validação de erros."""
    cal = carregar_calendario()

    # dias_letivos(cal) = {SEG 35, TER 38, QUA 38, QUI 37, SEX 35} (soma 183)
    esperado_anual = {"SEG": 35, "TER": 38, "QUA": 38, "QUI": 37, "SEX": 35}
    dl_anual = dias_letivos(cal)
    assert dl_anual == esperado_anual
    assert sum(dl_anual.values()) == 183

    # (cal, "A", [1]) = {10,10,11,10,9}
    assert dias_letivos(cal, "A", [1]) == {"SEG": 10, "TER": 10, "QUA": 11, "QUI": 10, "SEX": 9}
    # [2] = {10,10,10,9,9}
    assert dias_letivos(cal, "A", [2]) == {"SEG": 10, "TER": 10, "QUA": 10, "QUI": 9, "SEX": 9}
    # [3] = {8,9,9,9,9}
    assert dias_letivos(cal, "A", [3]) == {"SEG": 8, "TER": 9, "QUA": 9, "QUI": 9, "SEX": 9}
    # [4] = {7,9,8,9,8}
    assert dias_letivos(cal, "A", [4]) == {"SEG": 7, "TER": 9, "QUA": 8, "QUI": 9, "SEX": 8}

    # (cal, "REAL", [2], curso="Estradas", sabado_reproduz="QUI") = QUI 10 e o resto igual ao A
    res_real_b2 = dias_letivos(cal, "REAL", [2], curso="Estradas", sabado_reproduz="QUI")
    res_a_b2 = dias_letivos(cal, "A", [2])
    assert res_real_b2["QUI"] == 10
    assert {k: v for k, v in res_real_b2.items() if k != "QUI"} == {k: v for k, v in res_a_b2.items() if k != "QUI"}

    # REAL anual Estradas/QUI → QUI 38
    res_real_anual = dias_letivos(cal, "REAL", curso="Estradas", sabado_reproduz="QUI")
    assert res_real_anual["QUI"] == 38
    assert res_real_anual["SEG"] == 35
    assert res_real_anual["TER"] == 38
    assert res_real_anual["QUA"] == 38
    assert res_real_anual["SEX"] == 35

    # REAL com bimestres=[1] = A
    assert dias_letivos(cal, "REAL", [1], curso="Estradas", sabado_reproduz="QUI") == dias_letivos(cal, "A", [1])

    # REAL sem sabado_reproduz ou sem curso → ValueError
    with pytest.raises(ValueError, match="sabado_reproduz"):
        dias_letivos(cal, "REAL", curso="Estradas")

    with pytest.raises(ValueError, match="curso"):
        dias_letivos(cal, "REAL", sabado_reproduz="QUI")

    # cenário "B" → ValueError
    with pytest.raises(ValueError, match="Cenário inválido"):
        dias_letivos(cal, "B")


def test_dias_por_mes_bimestre() -> None:
    """Verifica a distribuição mensal dos dias letivos por bimestre (53/54/49/44)."""
    cal = carregar_calendario()

    # dias_por_mes_bimestre(cal) = {1:{2:5,3:23,4:20,5:5}, 2:{5:17,6:23,7:14}, 3:{8:23,9:23,10:3}, 4:{10:19,11:21,12:4}}
    esperado = {
        1: {2: 5, 3: 23, 4: 20, 5: 5},
        2: {5: 17, 6: 23, 7: 14},
        3: {8: 23, 9: 23, 10: 3},
        4: {10: 19, 11: 21, 12: 4},
    }
    resultado = dias_por_mes_bimestre(cal)
    assert resultado == esperado

    # cada bimestre somando 53/54/49/44
    assert sum(resultado[1].values()) == 53
    assert sum(resultado[2].values()) == 54
    assert sum(resultado[3].values()) == 49
    assert sum(resultado[4].values()) == 44


def test_divergencias() -> None:
    """Verifica que a reconstrução dia a dia confere com as tabelas-resumo (C4).

    O levantamento do plano (rev. 4) previa 2 divergências no 1º BI (QUI 11 × 10
    e abril 21 × 20) por uma quinta-feira de abril "não listada" — mas no .md
    oficial em uso (versão de maio/2026) a quinta 02/04 ESTÁ listada em "Dias
    sem aula" e a reconstrução fecha exatamente com as tabelas-resumo
    (QUI: 11 quintas − 02/04 = 10; abril: 22 úteis − 4 sem aula + 2 sábados = 20).
    Se uma revisão futura do calendário reintroduzir uma lacuna, esta função
    passa a expô-la aqui.
    """
    cal = carregar_calendario()
    divs = divergencias(cal)
    assert divs == []


def test_erro_total_bimestre_diferente_da_visao_geral(tmp_path: Path) -> None:
    """Total do bimestre na tabela por dia da semana ≠ Dias letivos da Visão geral → ValueError."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    mod = texto.replace(
        "| 1º BI | 10 | 10 | 11 | 10 | 9 | 3 | 53 |",
        "| 1º BI | 11 | 10 | 11 | 10 | 9 | 3 | 54 |",
    ).replace(
        "| **Soma** | **35** | **38** | **38** | **37** | **35** | **17** | **200** |",
        "| **Soma** | **36** | **38** | **38** | **37** | **35** | **17** | **201** |",
    )
    p = tmp_path / "cal_total_bim_divergente.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="Dias letivos por dia da semana"):
        carregar_calendario(caminho=p)


def test_erro_sabado_fora_do_periodo_do_bimestre(tmp_path: Path) -> None:
    """Sábado letivo listado com data fora do período do bimestre → ValueError."""
    texto = CAMINHO_MD_PADRAO.read_text(encoding="utf-8")
    # 25/07/2026 é sábado, mas está fora do 1º BI (23/02–08/05) — e do 2º BI (até 17/07)
    mod = texto.replace("| 25/04 | Área de Inglês (DELTEC) |", "| 25/07 | Área de Inglês (DELTEC) |")
    p = tmp_path / "cal_sabado_fora_periodo.md"
    p.write_text(mod, encoding="utf-8")

    with pytest.raises(ValueError, match="fora do período do bimestre"):
        carregar_calendario(caminho=p)


# ==============================================================================
# Testes de CH nominal, lecionada e faixas de variação (Passo 3 / C4)
# ==============================================================================


def test_ch_nominal() -> None:
    """Verifica o cálculo de CH nominal para 1, 2, 3 e 4 aulas semanais (40 semanas nominais)."""
    assert [ch_nominal(i) for i in (1, 2, 3, 4)] == [40, 80, 120, 160]
    assert ch_nominal(2) == 80


def test_ch_lecionada_mesmo_dia_e_razoes() -> None:
    """Verifica 2 aulas no mesmo dia (A anual) e as razões sobre ch_nominal(2) = 80."""
    cal = carregar_calendario()

    # 2 aulas no mesmo dia, anual, A — SEG 70, TER 76, QUA 76, QUI 74, SEX 70
    assert ch_lecionada(cal, {"SEG": 2}) == 70
    assert ch_lecionada(cal, {"TER": 2}) == 76
    assert ch_lecionada(cal, {"QUA": 2}) == 76
    assert ch_lecionada(cal, {"QUI": 2}) == 74
    assert ch_lecionada(cal, {"SEX": 2}) == 70

    # Razão sobre ch_nominal(2) = 80: 0,875 / 0,95 / 0,95 / 0,925 / 0,875
    nom = ch_nominal(2)
    assert nom == 80
    assert ch_lecionada(cal, {"SEG": 2}) / nom == 0.875
    assert ch_lecionada(cal, {"TER": 2}) / nom == 0.95
    assert ch_lecionada(cal, {"QUA": 2}) / nom == 0.95
    assert ch_lecionada(cal, {"QUI": 2}) / nom == 0.925
    assert ch_lecionada(cal, {"SEX": 2}) / nom == 0.875


def test_ch_lecionada_dias_distintos_1_mais_1() -> None:
    """Verifica 1+1 em dias distintos (as 10 combinações) entre 70 e 76 com extremos atingidos."""
    import itertools

    cal = carregar_calendario()
    combinacoes = list(itertools.combinations(DIAS_UTEIS, 2))
    assert len(combinacoes) == 10

    chs = [ch_lecionada(cal, {d1: 1, d2: 1}) for d1, d2 in combinacoes]
    assert all(70 <= val <= 76 for val in chs)
    assert min(chs) == 70
    assert max(chs) == 76


def test_ch_lecionada_especificos_e_cenarios() -> None:
    """Verifica casos específicos de apuração bimestral, arranjos com zeros e cenário REAL."""
    cal = carregar_calendario()

    # bimestral A ch_lecionada(cal, {"SEG":2}, "A", [1]) = 20
    assert ch_lecionada(cal, {"SEG": 2}, "A", [1]) == 20

    # arranjo com zeros ({"SEG":0,"TER":2,"QUA":0,"QUI":0,"SEX":0}) = 76
    assert ch_lecionada(cal, {"SEG": 0, "TER": 2, "QUA": 0, "QUI": 0, "SEX": 0}) == 76

    # ch_lecionada(cal, {"QUA":4}) = 152 (dentro da faixa de 4 CH)
    assert ch_lecionada(cal, {"QUA": 4}) == 152

    # ch_lecionada(cal, {"SEX":2}, "REAL", curso="Estradas", sabado_reproduz="SEX")
    # = 72 no ano e 20 com bimestres=[2] (A: 70 e 18), e com sabado_reproduz="SEG" = 70
    assert ch_lecionada(cal, {"SEX": 2}, "REAL", curso="Estradas", sabado_reproduz="SEX") == 72
    assert ch_lecionada(cal, {"SEX": 2}, "REAL", [2], curso="Estradas", sabado_reproduz="SEX") == 20
    assert ch_lecionada(cal, {"SEX": 2}, "A") == 70
    assert ch_lecionada(cal, {"SEX": 2}, "A", [2]) == 18
    assert ch_lecionada(cal, {"SEX": 2}, "REAL", curso="Estradas", sabado_reproduz="SEG") == 70


def test_faixa_ch_cenario_a() -> None:
    """Verifica faixas no cenário A para 1, 2, 3 e 4 CH com arranjos canônicos."""
    cal = carregar_calendario()

    # 1 CH: (35, {"SEG":1}, 38, {"TER":1})
    assert faixa_ch(cal, 1, "A") == (35, {"SEG": 1}, 38, {"TER": 1})

    # 2 CH: (70, {"SEG":2}, 76, {"TER":2})
    assert faixa_ch(cal, 2, "A") == (70, {"SEG": 2}, 76, {"TER": 2})

    # 3 CH: (105, {"SEG":2,"SEX":1}, 114, {"TER":2,"QUA":1})
    assert faixa_ch(cal, 3, "A") == (105, {"SEG": 2, "SEX": 1}, 114, {"TER": 2, "QUA": 1})

    # 4 CH: (140, {"SEG":2,"SEX":2}, 152, {"TER":2,"QUA":2})
    assert faixa_ch(cal, 4, "A") == (140, {"SEG": 2, "SEX": 2}, 152, {"TER": 2, "QUA": 2})


def test_faixa_ch_cenario_real() -> None:
    """Verifica faixas no cenário REAL com sábados de curso e curso sem sábado casado."""
    cal = carregar_calendario()

    # faixa_ch no REAL com curso="Estradas" — 2 CH 70–78, 3 CH 105–116, 4 CH 140–154
    f2 = faixa_ch(cal, 2, "REAL", curso="Estradas")
    assert f2[0] == 70 and f2[2] == 78

    f3 = faixa_ch(cal, 3, "REAL", curso="Estradas")
    assert f3[0] == 105 and f3[2] == 116

    f4 = faixa_ch(cal, 4, "REAL", curso="Estradas")
    assert f4[0] == 140 and f4[2] == 154

    # REAL com curso sem sábado casado (ex. "Inexistente") = A
    assert faixa_ch(cal, 1, "REAL", curso="Inexistente") == faixa_ch(cal, 1, "A")
    assert faixa_ch(cal, 2, "REAL", curso="Inexistente") == faixa_ch(cal, 2, "A")
    assert faixa_ch(cal, 3, "REAL", curso="Inexistente") == faixa_ch(cal, 3, "A")
    assert faixa_ch(cal, 4, "REAL", curso="Inexistente") == faixa_ch(cal, 4, "A")


def test_faixa_ch_erros() -> None:
    """Verifica que faixa_ch levanta ValueError para CH não suportada ou parâmetros inválidos."""
    cal = carregar_calendario()

    with pytest.raises(ValueError, match="Carga horária semanal não suportada"):
        faixa_ch(cal, 5)

    with pytest.raises(ValueError, match="Carga horária semanal não suportada"):
        faixa_ch(cal, 0)

    with pytest.raises(ValueError, match="curso"):
        faixa_ch(cal, 2, "REAL")



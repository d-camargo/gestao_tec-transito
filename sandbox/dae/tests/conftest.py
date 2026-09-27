"""Fixtures e configuração de testes para o sandbox DAE.

D9: Este conftest garante que o diretório sandbox/dae e a raiz do repositório
estejam no sys.path, permitindo importar tanto 'carregar' quanto módulos de 'core'.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import openpyxl
import pytest

# D9: Assegura sandbox/dae e raiz no sys.path
_DIR_DAE = Path(__file__).resolve().parent.parent
_DIR_RAIZ = _DIR_DAE.parent.parent

for _caminho in (_DIR_DAE, _DIR_RAIZ):
    if str(_caminho) not in sys.path:
        sys.path.insert(0, str(_caminho))

MESES_DAE = [
    "Fevereiro",
    "Março",
    "Abril",
    "Maio",
    "Junho",
    "Julho",
    "Agosto",
    "Setembro",
    "Outubro",
    "Novembro",
]


@pytest.fixture
def dados_sinteticos_dae() -> dict[str, list]:
    """Retorna estrutura de linhas para gerar o arquivo sintético da DAE.

    Contém:
        - Linha 1 (meses na 1ª linha com preenchimento para rotular o trio)
        - Linha 2 (cabeçalho de colunas)
        - 4 estudantes inventados, com meses fev-ago lançados e set-nov vazios:
          * Aluno 1: Elegível (PdM), BA/BP Sim, BCE Não
          * Aluno 2: Não elegível, BA/BP Não, BCE Sim
          * Aluno 3: N/C (pressuposto 'nada_consta', sem PdM), sem bolsas
          * Aluno 4: Elegibilidade indefinida, BA/BP Sim, BCE Sim
    """
    # Linha 0: 9 colunas fixas vazias + 3 colunas por mês (primeira é o nome do mês)
    row0: list[object] = [""] * 9
    for mes in MESES_DAE:
        row0.extend([mes, "", ""])

    # Linha 1: Cabeçalhos das 9 colunas fixas + trios mensais
    row1: list[object] = [
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
    for _ in MESES_DAE:
        row1.extend(["HA ofertadas", "HA presenciadas", "Frequência"])

    # Dados de 4 alunos inventados
    alunos = [
        # Aluno 1: Elegível, BA/BP
        {
            "fixo": [
                "Ana Silva",
                "20261010001",
                0.85,
                "111.111.111-11",
                "Campus I",
                "TÉCNICO EM TRÂNSITO",
                "Elegível",
                "Sim",
                "Não",
            ],
            "lancados": [
                (20, 18, 0.90),      # fev
                (150, 120, 0.80),    # mar
                (140, 110, 0.7857),  # abr
                (160, 130, 0.8125),  # mai
                (130, 100, 0.7692),  # jun
                (80, 70, 0.8750),    # jul
                (150, 140, 0.9333),  # ago
            ],
        },
        # Aluno 2: Não elegível, BCE
        {
            "fixo": [
                "Bruno Souza",
                "20261010002",
                0.70,
                "222.222.222-22",
                "Campus I",
                "TÉCNICO EM TRÂNSITO",
                "Não elegível",
                "Não",
                "Sim",
            ],
            "lancados": [
                (20, 15, 0.75),
                (150, 100, 0.6667),
                (140, 95, 0.6786),
                (160, 110, 0.6875),
                (130, 90, 0.6923),
                (80, 55, 0.6875),
                (150, 105, 0.7000),
            ],
        },
        # Aluno 3: N/C (pressuposto 'nada_consta', sem PdM)
        {
            "fixo": [
                "Carlos Lima",
                "20261010003",
                0.95,
                "333.333.333-33",
                "Campus I",
                "TÉCNICO EM ESTRADAS",
                "N/C",
                "Não",
                "Não",
            ],
            "lancados": [
                (20, 20, 1.00),
                (150, 145, 0.9667),
                (140, 135, 0.9643),
                (160, 150, 0.9375),
                (130, 125, 0.9615),
                (80, 75, 0.9375),
                (150, 140, 0.9333),
            ],
        },
        # Aluno 4: Elegibilidade indefinida, BA/BP + BCE
        {
            "fixo": [
                "Daniela Rocha",
                "20261010004",
                0.60,
                "444.444.444-44",
                "Campus I",
                "TÉCNICO EM TRÂNSITO",
                "Elegibilidade indefinida",
                "Sim",
                "Sim",
            ],
            "lancados": [
                (20, 12, 0.60),
                (150, 90, 0.60),
                (140, 84, 0.60),
                (160, 96, 0.60),
                (130, 78, 0.60),
                (80, 48, 0.60),
                (150, 90, 0.60),
            ],
        },
    ]

    rows_alunos: list[list[object]] = []
    for al in alunos:
        row: list[object] = list(al["fixo"])
        # 7 meses lançados (fev-ago)
        for tri in al["lancados"]:
            row.extend(list(tri))
        # 3 meses vazios (set-nov)
        for _ in range(3):
            row.extend([None, None, None])
        rows_alunos.append(row)

    return {
        "row0": row0,
        "row1": row1,
        "rows_alunos": rows_alunos,
    }


@pytest.fixture
def caminho_xlsx(tmp_path: Path, dados_sinteticos_dae: dict[str, list]) -> Path:
    """Gera um arquivo .xlsx sintético via openpyxl.

    Inclui a aba '2026 - geral' e uma aba adicional 'Emails' que NÃO deve ser lida.
    """
    caminho = tmp_path / "dae_sintetico.xlsx"
    wb = openpyxl.Workbook()

    # Aba útil
    ws_geral = wb.active
    ws_geral.title = "2026 - geral"
    ws_geral.append(dados_sinteticos_dae["row0"])
    ws_geral.append(dados_sinteticos_dae["row1"])
    for row in dados_sinteticos_dae["rows_alunos"]:
        ws_geral.append(row)

    # Aba extra que não pode ser lida (D4)
    ws_emails = wb.create_sheet("Emails")
    ws_emails.append(["Matrícula", "Email"])
    ws_emails.append(["20261010001", "ana@aluno.cefetmg.br"])
    ws_emails.append(["20261010002", "bruno@aluno.cefetmg.br"])
    ws_emails.append(["20261010003", "carlos@aluno.cefetmg.br"])
    ws_emails.append(["20261010004", "daniela@aluno.cefetmg.br"])

    wb.save(caminho)
    return caminho


@pytest.fixture
def caminho_csv(tmp_path: Path, dados_sinteticos_dae: dict[str, list]) -> Path:
    """Gera um arquivo .csv equivalente à aba '2026 - geral'."""
    caminho = tmp_path / "dae_sintetico.csv"

    with open(caminho, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(dados_sinteticos_dae["row0"])
        writer.writerow(dados_sinteticos_dae["row1"])
        for row in dados_sinteticos_dae["rows_alunos"]:
            # None é convertido para string vazia no CSV
            writer.writerow([c if c is not None else "" for c in row])

    return caminho


def gerar_planilha_ch_sintetica(caminho: Path, com_formulas: bool = False) -> Path:
    """Gera arquivo .xlsx sintético para testes de CH efetiva.

    Contém as 4 abas oficiais com o mesmo cabeçalho e 7 linhas fictícias:
        - Cursos: "Estradas", "Estradas / Trânsito (núcleo comum)", "Trânsito"
        - Turmas: "EST-2A", "EST/TT-2A", "TT-2A"
        - Um par T1/T2 em "EST/TT-2A" (REDAÇÃO)
        - Uma linha de 1 aula em "EST/TT-2A" (FILOSOFIA)
        - Professores fictícios
        - Valores numéricos calculados com a tabela oficial (quando com_formulas=False)
        - Fórmulas do Excel sem cache (quando com_formulas=True)
    """
    wb = openpyxl.Workbook()

    # 1. Aba 'Leia-me'
    ws_leia = wb.active
    ws_leia.title = "Leia-me"
    ws_leia.append(["CH efetivamente lecionada por disciplina — Integrado, CEFET-MG BH, 2026"])
    ws_leia.append([None])
    ws_leia.append(["Fontes"])
    ws_leia.append(["• Horários sintéticos fictícios para testes unitários."])
    ws_leia.append([None])
    ws_leia.append(["Como a CH é calculada"])
    ws_leia.append(["• CH do bimestre = Σ (aulas naquele dia × dias letivos daquele dia)."])
    ws_leia.append(["• CH nominal = aulas semanais × 40 semanas."])
    ws_leia.append(["• Os sábados letivos NÃO entram (cenário A)."])
    ws_leia.append([None])
    ws_leia.append(["Limitações"])
    ws_leia.append(["• Cobre apenas as salas do bloco 305–437."])

    # 2. Aba 'Calendário'
    ws_cal = wb.create_sheet("Calendário")
    ws_cal.append(["Dias letivos por dia da semana — 2026 (sem sábados)", None, None, None, None, None, None, None, None])
    ws_cal.append([None] * 9)
    ws_cal.append(["Bimestre", "SEG", "TER", "QUA", "QUI", "SEX", "Dias úteis", "Sábados letivos", "Total"])
    ws_cal.append(["1º BI", 10, 10, 11, 10, 9, 50, 3, 53])
    ws_cal.append(["2º BI", 10, 10, 10, 9, 9, 48, 6, 54])
    ws_cal.append(["3º BI", 8, 9, 9, 9, 9, 44, 5, 49])
    ws_cal.append(["4º BI", 7, 9, 8, 9, 8, 41, 3, 44])
    ws_cal.append(["Ano", 35, 38, 38, 37, 35, 183, 17, 200])

    # 3. Aba 'CH por disciplina'
    ws_ch = wb.create_sheet("CH por disciplina")
    header_ch = [
        "Curso", "Turma", "Subgrupo", "Sigla", "Disciplina", "Professor(a)",
        "SEG", "TER", "QUA", "QUI", "SEX", "Aulas/sem", "CH nominal",
        "1º BI", "2º BI", "3º BI", "4º BI", "CH efetiva", "Diferença", "% do nominal"
    ]
    ws_ch.append(header_ch)

    linhas_ficticias = [
        # Linha 2: EST-2A, Topografia, 2 aulas na TER
        ["Estradas", "EST-2A", "—", "TOP", "TOPOGRAFIA", "Prof. Fictício Topografia", 0, 2, 0, 0, 0, 2, 80, 20, 20, 18, 18, 76, -4, 0.95],
        # Linha 3: EST-2A, Máquinas e Equipamentos, 2 aulas na SEX
        ["Estradas", "EST-2A", "—", "ME", "MÁQUINAS E EQUIPAMENTOS", "Prof. Fictício Maquinas", 0, 0, 0, 0, 2, 2, 80, 18, 18, 18, 16, 70, -10, 0.875],
        # Linha 4: EST/TT-2A, Inglês, 2 aulas na SEX
        ["Estradas / Trânsito (núcleo comum)", "EST/TT-2A", "—", "ING", "INGLÊS", "Prof. Fictício Ingles", 0, 0, 0, 0, 2, 2, 80, 18, 18, 18, 16, 70, -10, 0.875],
        # Linha 5: EST/TT-2A, Filosofia, 1 aula na QUA (linha de 1 aula)
        ["Estradas / Trânsito (núcleo comum)", "EST/TT-2A", "—", "FIL", "FILOSOFIA", "Prof. Fictício Filosofia", 0, 0, 1, 0, 0, 1, 40, 11, 10, 9, 8, 38, -2, 0.95],
        # Linha 6: EST/TT-2A, Redação, T1 (par T1/T2), 2 aulas na SEG
        ["Estradas / Trânsito (núcleo comum)", "EST/TT-2A", "T1", "RED", "REDAÇÃO", "Prof. Fictício Redacao T1", 2, 0, 0, 0, 0, 2, 80, 20, 20, 16, 14, 70, -10, 0.875],
        # Linha 7: EST/TT-2A, Redação, T2 (par T1/T2), 2 aulas na SEG
        ["Estradas / Trânsito (núcleo comum)", "EST/TT-2A", "T2", "RED", "REDAÇÃO", "Prof. Fictício Redacao T2", 2, 0, 0, 0, 0, 2, 80, 20, 20, 16, 14, 70, -10, 0.875],
        # Linha 8: TT-2A, Operação de Transportes, 2 aulas na QUI
        ["Trânsito", "TT-2A", "—", "OPT", "OPERAÇÃO DE TRANSPORTES", "Prof. Fictício Transportes", 0, 0, 0, 2, 0, 2, 80, 20, 18, 18, 18, 74, -6, 0.925],
    ]

    for idx, r_data in enumerate(linhas_ficticias, start=2):
        row = list(r_data)
        if com_formulas:
            row[11] = f"=SUM(G{idx}:K{idx})"
            row[12] = f"=L{idx}*40"
            row[13] = f"=SUMPRODUCT($G{idx}:$K{idx},Calendário!$B$4:$F$4)"
            row[14] = f"=SUMPRODUCT($G{idx}:$K{idx},Calendário!$B$5:$F$5)"
            row[15] = f"=SUMPRODUCT($G{idx}:$K{idx},Calendário!$B$6:$F$6)"
            row[16] = f"=SUMPRODUCT($G{idx}:$K{idx},Calendário!$B$7:$F$7)"
            row[17] = f"=SUM(N{idx}:Q{idx})"
            row[18] = f"=R{idx}-M{idx}"
            row[19] = f'=IF(M{idx}=0,"",R{idx}/M{idx})'
        ws_ch.append(row)

    # 4. Aba 'Resumo por carga'
    ws_res = wb.create_sheet("Resumo por carga")
    ws_res.append(["Referência: CH efetiva anual por aula semanal em cada dia (sem sábados)", None, None])
    ws_res.append([None, None, None])
    ws_res.append(["Dia", "h/a por aula semanal", "% de 40 semanas"])
    ws_res.append(["SEG", 35, 0.875])
    ws_res.append(["TER", 38, 0.95])
    ws_res.append(["QUA", 38, 0.95])
    ws_res.append(["QUI", 37, 0.925])
    ws_res.append(["SEX", 35, 0.875])
    ws_res.append([None, None, None])
    ws_res.append(["Distribuição das disciplinas por % do nominal", None, None])
    ws_res.append(["abaixo de 90%", 4, None])
    ws_res.append(["90% a 95%", 1, None])
    ws_res.append(["95% ou mais", 2, None])
    ws_res.append(["Total", 7, None])

    wb.save(caminho)
    return caminho


@pytest.fixture
def planilha_ch_sintetica(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Fixture que gera arquivo .xlsx sintético para testes de CH efetiva.

    Por padrão gera a versão numérica. Caso parametrizada com indirect=True ou
    request.param='formulas' (ou True), gera a variante com fórmulas.
    """
    param = getattr(request, "param", False)
    com_formulas = bool(param and param not in ("numerica", False))
    nome = "ch_sintetica_formulas.xlsx" if com_formulas else "ch_sintetica.xlsx"
    return gerar_planilha_ch_sintetica(tmp_path / nome, com_formulas=com_formulas)


@pytest.fixture
def planilha_ch_sintetica_formulas(tmp_path: Path) -> Path:
    """Fixture que gera a variante com fórmulas (sem cache) da planilha sintética de CH."""
    return gerar_planilha_ch_sintetica(tmp_path / "ch_sintetica_formulas.xlsx", com_formulas=True)

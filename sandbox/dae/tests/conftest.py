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

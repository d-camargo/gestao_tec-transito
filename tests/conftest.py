"""Helpers e fixtures reutilizáveis para os testes da aplicação.

Todo dado de aluno aqui é sintético e gerado em memória — o repositório é
público e mapas de turma reais contêm dados pessoais (LGPD).
"""

import pandas as pd
import pytest

import core.manipulacao as manipulacao


def mapa_sintetico(curso="TÉCNICO EM TRÂNSITO", bimestre=1, turma="TRA.1A",
                   disciplinas=None, alunos=None):
    """Devolve o DataFrame bruto de um Mapa de Turma do SIGAA (.xls).

    Monta o cabeçalho (Curso/Etapa/Período Letivo/Turma), a tabela de notas
    e faltas (uma coluna "N" e uma "F" por disciplina) e o bloco LEGENDA,
    imitando o layout lido por `core.manipulacao`.

    - `disciplinas`: dict {codigo: nome_da_legenda}.
    - `alunos`: lista de tuplas (matricula, nome, {cod: nota}, {cod: faltas}).
    """
    if isinstance(bimestre, int):
        bimestre = f"{bimestre}º Bimestre"
    if disciplinas is None:
        disciplinas = {
            "MAT": "Matemática I",
            "POR": "Língua Portuguesa I",
            "TRA01": "Legislação de Trânsito",
        }

    codigos = list(disciplinas)
    linhas = [
        ["Curso:", curso],
        ["Etapa:", bimestre],
        ["Período Letivo:", "2026/1"],
        ["Turma:", turma],
        [],
    ]

    cabecalho = ["Matrícula", "Nome do Aluno"]
    tipos = ["", ""]
    for codigo in codigos:
        cabecalho += [codigo, codigo]
        tipos += ["N", "F"]
    linhas.append(cabecalho)
    linhas.append(tipos)

    if alunos is None:
        alunos = [
            ("20260000001", "Aluno 1",
             {c: 15.0 for c in codigos}, {c: 2 for c in codigos}),
            ("20260000002", "Aluno 2",
             {c: 12.0 for c in codigos}, {c: 0 for c in codigos}),
        ]
    for matricula, nome, notas, faltas in alunos:
        linha = [matricula, nome]
        for codigo in codigos:
            linha += [str(notas.get(codigo, 0.0)), str(faltas.get(codigo, 0))]
        linhas.append(linha)

    linhas.append([])
    linhas.append(["", "LEGENDA", ""])
    for codigo, nome in disciplinas.items():
        linhas.append(["", codigo, nome])

    n_colunas = max(len(linha) for linha in linhas)
    return pd.DataFrame(
        [linha + [None] * (n_colunas - len(linha)) for linha in linhas])


@pytest.fixture(autouse=True)
def xls_sintetico(monkeypatch):
    """Faz `_ler_xls_bruto` aceitar os DataFrames sintéticos como se fossem XLS.

    Com isso, os testes entregam o DataFrame do helper no lugar do arquivo,
    sem versionar `.xls` reais nem depender de arquivo fora do repositório.
    """
    original = manipulacao._ler_xls_bruto

    def _ler(arquivo_xls):
        if isinstance(arquivo_xls, pd.DataFrame):
            return arquivo_xls
        return original(arquivo_xls)

    monkeypatch.setattr(manipulacao, "_ler_xls_bruto", _ler)

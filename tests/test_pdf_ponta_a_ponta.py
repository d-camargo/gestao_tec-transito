"""Teste ponta a ponta do PDF (Passo 11): 1 bimestre (sem 5.x) e 3 bimestres
(com 5.1-5.4), conferindo que o buffer não é vazio e que as seções removidas
da reestruturação não aparecem no relatório.

A checagem de texto usa PyMuPDF (`fitz`), que **não está instalado neste
venv** — por isso vem atrás de `pytest.importorskip('fitz')` e é pulada aqui.
O que roda sempre (sem depender de lib de extração de texto) é a checagem de
tamanho do buffer nos dois cenários.
"""

import unittest

import pandas as pd
import pytest

from core.relatorios import (
    calcular_estatisticas,
    calcular_estatisticas_multibimestre,
    criar_relatorio_pdf,
    gerar_todos_graficos,
)

DISCIPLINAS = {
    'MAT': 'MATEMÁTICA - 2ª SÉRIE',
    'FIS': 'FÍSICA - 2ª SÉRIE',
    'QUI': 'QUÍMICA - 2ª SÉRIE',
}

# Seções que a reestruturação de 2026-08-24/25 removeu do relatório —
# nenhuma delas pode aparecer no texto extraído do PDF.
SECOES_REMOVIDAS = [
    'Risco de Repetência',
    'Situação Geral do Aluno',
    'Menor Desempenho',
    'Análise de Frequência',
    'Evolução das Médias',
    'Top 10',
    'Reprovação matemática',
]


def _turma(bim_num, seed=0):
    """5 alunos com notas/faltas plausíveis para o bimestre `bim_num`,
    variando um pouco por `seed` para simular evolução entre bimestres."""
    max_pts = {1: 20, 2: 30, 3: 20, 4: 30}[bim_num]
    nomes = ['Ana', 'Beto', 'Caio', 'Duda', 'Elis']
    matriculas = [f'2026000000{i}' for i in range(1, 6)]
    base = [0.90, 0.75, 0.60, 0.45, 0.30]
    df_notas = pd.DataFrame({
        'matricula': matriculas,
        'nome': nomes,
        'MAT': [round(max_pts * max(0.05, min(0.98, b + 0.03 * seed)), 1) for b in base],
        'FIS': [round(max_pts * max(0.05, min(0.98, b + 0.05 * seed)), 1) for b in base],
        'QUI': [round(max_pts * max(0.05, min(0.98, b - 0.02 * seed)), 1) for b in base],
    })
    df_faltas = pd.DataFrame({
        'matricula': matriculas,
        'nome': nomes,
        'MAT': [0, 1, 2, 3, 4],
        'FIS': [1, 0, 2, 4, 5],
        'QUI': [0, 0, 1, 2, 3],
    })
    metadados = {'bimestre_num': bim_num, 'turma': 'TRA.2A', 'periodo_letivo': '2026/1'}
    return df_notas, df_faltas, DISCIPLINAS, metadados


class TestPdfUmBimestre(unittest.TestCase):
    """1 bimestre: sem `estatisticas_multibimestre` — os itens 5.x não entram."""

    def test_buffer_nao_vazio(self):
        df_notas, df_faltas, disciplinas, metadados = _turma(1)
        stats = calcular_estatisticas(
            df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados)
        figuras = gerar_todos_graficos(df_notas, 'Trânsito', disciplinas, stats, df_faltas)
        buf = criar_relatorio_pdf('Trânsito', stats, figuras)
        conteudo = buf.getvalue()
        self.assertIsNotNone(conteudo)
        self.assertGreater(len(conteudo), 1000)


class TestPdfTresBimestres(unittest.TestCase):
    """3 bimestres: com `estatisticas_multibimestre` — itens 5.1 a 5.4 entram."""

    def _gerar(self):
        conjuntos = [_turma(1, seed=0), _turma(2, seed=1), _turma(3, seed=2)]
        estatisticas_multibimestre = calcular_estatisticas_multibimestre(conjuntos)
        df_notas, df_faltas, disciplinas, metadados = conjuntos[-1]  # bimestre mais recente
        stats = calcular_estatisticas(
            df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados)
        figuras = gerar_todos_graficos(df_notas, 'Trânsito', disciplinas, stats, df_faltas)
        buf = criar_relatorio_pdf(
            'Trânsito', stats, figuras,
            estatisticas_multibimestre=estatisticas_multibimestre)
        return buf

    def test_buffer_nao_vazio(self):
        buf = self._gerar()
        conteudo = buf.getvalue()
        self.assertIsNotNone(conteudo)
        self.assertGreater(len(conteudo), 1000)

    def test_buffer_maior_que_o_de_um_bimestre(self):
        """Com as seções 5.1-5.4 a mais, o PDF de 3 bimestres tem que ser
        sensivelmente maior que o de 1 bimestre só."""
        df_notas, df_faltas, disciplinas, metadados = _turma(1)
        stats = calcular_estatisticas(
            df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados)
        figuras = gerar_todos_graficos(df_notas, 'Trânsito', disciplinas, stats, df_faltas)
        buf_1 = criar_relatorio_pdf('Trânsito', stats, figuras)

        buf_3 = self._gerar()
        self.assertGreater(len(buf_3.getvalue()), len(buf_1.getvalue()))


class TestSecoesRemovidasNaoAparecem(unittest.TestCase):
    """Checagem de texto — só roda se `fitz` (PyMuPDF) estiver instalado."""

    def _extrair_texto(self, buf):
        fitz = pytest.importorskip('fitz')
        doc = fitz.open(stream=buf.getvalue(), filetype='pdf')
        try:
            return '\n'.join(pagina.get_text() for pagina in doc)
        finally:
            doc.close()

    def test_secoes_removidas_ausentes_em_um_bimestre(self):
        df_notas, df_faltas, disciplinas, metadados = _turma(1)
        stats = calcular_estatisticas(
            df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados)
        figuras = gerar_todos_graficos(df_notas, 'Trânsito', disciplinas, stats, df_faltas)
        buf = criar_relatorio_pdf('Trânsito', stats, figuras)
        texto = self._extrair_texto(buf)
        for secao in SECOES_REMOVIDAS:
            self.assertNotIn(secao, texto)
        # Sanidade: seções que DEVEM existir num bimestre único.
        self.assertIn('Alunos com Média Abaixo do Limiar', texto)
        self.assertIn('Desempenho e Frequência por Aluno', texto)
        self.assertNotIn('Análise Multibimestral', texto)

    def test_secoes_removidas_ausentes_em_tres_bimestres(self):
        conjuntos = [_turma(1, seed=0), _turma(2, seed=1), _turma(3, seed=2)]
        estatisticas_multibimestre = calcular_estatisticas_multibimestre(conjuntos)
        df_notas, df_faltas, disciplinas, metadados = conjuntos[-1]
        stats = calcular_estatisticas(
            df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados)
        figuras = gerar_todos_graficos(df_notas, 'Trânsito', disciplinas, stats, df_faltas)
        buf = criar_relatorio_pdf(
            'Trânsito', stats, figuras,
            estatisticas_multibimestre=estatisticas_multibimestre)
        texto = self._extrair_texto(buf)
        for secao in SECOES_REMOVIDAS:
            self.assertNotIn(secao, texto)
        # Sanidade: seções que DEVEM existir com 3 bimestres.
        self.assertIn('Análise Multibimestral', texto)
        self.assertIn('Probabilidade de Reprovação', texto)
        self.assertIn('Índice de Desempenho do Aluno', texto)


if __name__ == "__main__":
    unittest.main()

"""Testes da tabela completa de desempenho por aluno (Passo 3, `core.relatorios`).

`tabela_desempenho_por_aluno` monta uma linha por aluno, com uma coluna
abreviada (D1..Dn) por disciplina com notas, célula ``"nota / faltas"`` (só a
nota quando a disciplina não tem faltas lançadas, ``—`` quando a nota está
ausente), e uma legenda mapeando cada abreviação ao nome completo.
"""

import unittest

import pandas as pd

from core.relatorios import calcular_estatisticas, get_simplified_name

DISCIPLINAS = {
    'MAT': 'MATEMÁTICA - 2ª SÉRIE',
    'FIS': 'FÍSICA - 2ª SÉRIE',
}
METADADOS = {'bimestre_num': 1}  # 20 pts, limiar 12.0


def _turma():
    """Beto vem antes de Ana na entrada, para provar que a saída é
    ordenada por nome (não pela ordem de chegada do mapa).

    - Ana: MAT = 15,0 (falta 3), FIS ausente (nota não lançada) → "—".
    - Beto: MAT = 10,0 (falta 5), FIS = 12,0 sem falta lançada para a
      disciplina (só a nota, sem "/").
    """
    df_notas = pd.DataFrame({
        'nome': ['Beto', 'Ana'],
        'MAT': [10.0, 15.0],
        'FIS': [12.0, None],
    })
    # Só MAT tem faltas lançadas no mapa — FIS fica de fora de `_faltas_cols`.
    df_faltas = pd.DataFrame({
        'nome': ['Beto', 'Ana'],
        'MAT': [5, 3],
    })
    return df_notas, df_faltas


class TestTabelaDesempenhoPorAluno(unittest.TestCase):
    """Passo 3: uma linha por aluno, células "nota / faltas", legenda D1..Dn."""

    def test_uma_linha_por_aluno_ordenada_por_nome(self):
        df_notas, df_faltas = _turma()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        tabela = stats['tabela_completa_df']
        self.assertEqual(len(tabela), 2)
        self.assertEqual(tabela['Aluno'].tolist(), ['Ana', 'Beto'])

    def test_legenda_consistente_com_as_colunas(self):
        df_notas, df_faltas = _turma()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        tabela = stats['tabela_completa_df']
        legenda = stats['legenda_disciplinas']
        esperado = [
            ('D1', get_simplified_name('MAT', DISCIPLINAS)),
            ('D2', get_simplified_name('FIS', DISCIPLINAS)),
        ]
        self.assertEqual(legenda, esperado)
        # As abreviações da legenda são exatamente as colunas de disciplina
        # da tabela (entre "Aluno" e "Média"/"Faltas").
        abrevs_legenda = [abrev for abrev, _ in legenda]
        colunas_disciplina = tabela.columns.tolist()[1:-2]
        self.assertEqual(colunas_disciplina, abrevs_legenda)

    def test_celula_nota_barra_faltas(self):
        df_notas, df_faltas = _turma()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        tabela = stats['tabela_completa_df']
        beto = tabela[tabela['Aluno'] == 'Beto'].iloc[0]
        self.assertEqual(beto['D1'], '10,0 / 5')  # MAT: nota e falta

    def test_disciplina_sem_faltas_mostra_so_a_nota(self):
        """FIS não tem faltas lançadas no mapa — a célula de Beto em FIS
        mostra só a nota, sem "/ falta"."""
        df_notas, df_faltas = _turma()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        tabela = stats['tabela_completa_df']
        beto = tabela[tabela['Aluno'] == 'Beto'].iloc[0]
        self.assertEqual(beto['D2'], '12,0')
        self.assertNotIn('/', beto['D2'])

    def test_nota_ausente_vira_travessao(self):
        df_notas, df_faltas = _turma()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        tabela = stats['tabela_completa_df']
        ana = tabela[tabela['Aluno'] == 'Ana'].iloc[0]
        self.assertEqual(ana['D2'], '—')  # FIS ausente para Ana

    def test_colunas_finais_media_e_faltas(self):
        df_notas, df_faltas = _turma()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        tabela = stats['tabela_completa_df']
        self.assertEqual(tabela.columns.tolist()[-2:], ['Média', 'Faltas'])

        ana = tabela[tabela['Aluno'] == 'Ana'].iloc[0]
        # Ana só tem nota em MAT (15,0) — média é a própria nota; faltas
        # totais somam só o que está em `_faltas_cols` (MAT) = 3.
        self.assertEqual(ana['Média'], '15,00')
        self.assertEqual(ana['Faltas'], '3')

        beto = tabela[tabela['Aluno'] == 'Beto'].iloc[0]
        # Beto: média de MAT (10,0) e FIS (12,0) = 11,00; faltas = só MAT = 5.
        self.assertEqual(beto['Média'], '11,00')
        self.assertEqual(beto['Faltas'], '5')

    def test_sem_faltas_disponiveis_celulas_so_com_a_nota(self):
        """Sem `df_faltas`, nenhuma disciplina tem faltas lançadas — todas
        as células viram só a nota, sem "/"."""
        df_notas, _ = _turma()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        tabela = stats['tabela_completa_df']
        beto = tabela[tabela['Aluno'] == 'Beto'].iloc[0]
        self.assertEqual(beto['D1'], '10,0')
        self.assertEqual(beto['Faltas'], '—')


if __name__ == "__main__":
    unittest.main()

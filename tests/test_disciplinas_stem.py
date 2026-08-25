"""Testes da identificação de disciplinas STEM (Matemática, Física e Química).

Cobre a armadilha registrada no plano: 'EDUCAÇÃO FÍSICA' normaliza para
'educacao fisica', que contém a substring 'fisica' — sem exclusão explícita,
entraria na seção STEM.
"""

import unittest

from core.disciplinas import disciplinas_stem


class TestDisciplinasStem(unittest.TestCase):
    """Testes para a função disciplinas_stem."""

    def test_inclui_matematica_fisica_quimica(self):
        """(a) Matemática, Física e Química entram no grupo STEM."""
        disciplinas = {
            '1MAT.006': 'MATEMÁTICA - 2ª SÉRIE',
            '1CIE.011': 'FÍSICA - 2ª SÉRIE',
            '1QUI.003': 'QUÍMICA - 2ª SÉRIE',
        }
        self.assertEqual(disciplinas_stem(disciplinas), disciplinas)

    def test_exclui_educacao_fisica(self):
        """(b) Educação Física NÃO entra, apesar de conter 'fisica'."""
        disciplinas = {
            '1DEFISD.006': 'EDUCAÇÃO FÍSICA - 2ª SÉRIE',
            '1MAT.006': 'MATEMÁTICA - 2ª SÉRIE',
        }
        stem = disciplinas_stem(disciplinas)
        self.assertNotIn('1DEFISD.006', stem)
        self.assertIn('1MAT.006', stem)

    def test_exclui_tecnicas_e_humanas(self):
        """(c) Laboratório de Solos (técnica) e História (humana) não entram."""
        disciplinas = {
            '1TT.43': 'LABORATÓRIO DE SOLOS',
            'HIST.2': 'HISTÓRIA - 2ª SÉRIE',
        }
        self.assertEqual(disciplinas_stem(disciplinas), {})

    def test_dicionario_vazio_devolve_vazio(self):
        """(d) Dicionário vazio devolve vazio."""
        self.assertEqual(disciplinas_stem({}), {})


if __name__ == "__main__":
    unittest.main()

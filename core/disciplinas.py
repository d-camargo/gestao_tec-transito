"""Catálogo de disciplinas dos cursos técnicos (Trânsito e Estradas).

Centralizado aqui para ser reutilizado tanto na manipulação dos dados
quanto na geração dos relatórios.
"""

disciplinas_ensino_medio = {
    '1DEFISD.006': 'EDUCAÇÃO FÍSICA - 2ª SÉRIE',
    '1LIN.003': 'LÍNGUA ESTRANGEIRA: INGLÊS - 2ª SÉRIE',
    '1MAT.006': 'MATEMÁTICA - 2ª SÉRIE',
    '1QUI.003': 'QUÍMICA - 2ª SÉRIE',
    '1TFIL2.1': 'FILOSOFIA - 2ª SÉRIE',
    '1TLP2.1': 'LÍNGUA PORTUGUESA - 2ª SÉRIE',
    '1TRED2.01': 'REDAÇÃO - 2ª SÉRIE',
    'GEO.2': 'GEOGRAFIA - 2ª SÉRIE',
    'HIST.2': 'HISTÓRIA - 2ª SÉRIE',
    '1CIE.010': 'BIOLOGIA - 2ª SÉRIE',
    '1CIE.011': 'FÍSICA - 2ª SÉRIE',
    'SOC.2': 'SOCIOLOGIA - 2ª SÉRIE',
}

disciplinas_tecnicas_transito = {
    '1TT.009': 'PLANEJAMENTO DE TRANSPORTES',
    '1TT.35': 'LABORATÓRIO DE PESQUISA DE TRANSPORTES E TRÂNSITO',
    '1TT.37': 'LABORATÓRIO DE TOPOGRAFIA URBANA',
    '1TT.62': 'LABORATÓRIO DE SEGURANÇA VIÁRIA',
    '4929': 'INTRODUÇÃO À ENGENHARIA DE TRÁFEGO',
}

disciplinas_tecnicas_estradas = {
    '1TT.43': 'LABORATÓRIO DE SOLOS',
    '1TT.44': 'LABORATÓRIO DE DESENHO TOPOGRÁFICO',
    '1TT.45': 'LABORATÓRIO DE TOPOGRAFIA',
    '5801': 'SOLOS',
    '5802': 'TOPOGRAFIA',
    '5811': 'MÁQUINAS E EQUIPAMENTOS',
}


def catalogo_nomes_conhecidos() -> dict:
    """Reúne todos os nomes amigáveis de disciplinas conhecidos (ensino médio +
    técnicas de Trânsito e Estradas). Usado para rotular cursos genéricos quando
    o código da disciplina coincide com algum já catalogado."""
    return {
        **disciplinas_ensino_medio,
        **disciplinas_tecnicas_transito,
        **disciplinas_tecnicas_estradas,
    }

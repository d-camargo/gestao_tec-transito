"""Catálogo de disciplinas dos cursos técnicos (Trânsito e Estradas).

Centralizado aqui para ser reutilizado tanto na manipulação dos dados
quanto na geração dos relatórios.
"""
import re
import unicodedata
from collections import Counter

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


# --------------------------------
# Classificação de disciplinas por nome (independente do código)
# --------------------------------
# Os códigos das disciplinas mudam a cada série/estrutura curricular do SIGAA
# (ex.: 'MATEMÁTICA - 2ª SÉRIE' = '1MAT.006', mas '...3ª SÉRIE' = 'MAT1.001').
# Por isso, a identificação do ensino médio e da série é feita pelo **nome**
# (legenda do XLS), não por uma lista fixa de códigos.
_RE_SERIE = re.compile(r'(\d)\D{0,4}serie')

# Palavras-chave que caracterizam disciplinas do ensino médio (núcleo comum).
_EM_KEYWORDS = (
    'matematica', 'portugues', 'ingles', 'lingua estrangeira', 'espanhol',
    'quimica', 'fisica', 'biologia', 'historia', 'geografia', 'sociologia',
    'filosofia', 'redacao', 'arte', 'ciencias', 'literatura',
)


def _normalizar(txt) -> str:
    """Minúsculas e sem acentos, para comparação tolerante de nomes."""
    txt = '' if txt is None else str(txt)
    txt = ''.join(
        c for c in unicodedata.normalize('NFD', txt)
        if unicodedata.category(c) != 'Mn'
    )
    return txt.lower().strip()


def eh_disciplina_ensino_medio(nome) -> bool:
    """Indica se o nome de uma disciplina corresponde ao ensino médio (núcleo
    comum), seja pelo sufixo "Nª SÉRIE" ou por uma palavra-chave conhecida."""
    n = _normalizar(nome)
    if not n:
        return False
    if _RE_SERIE.search(n):
        return True
    return any(k in n for k in _EM_KEYWORDS)


def detectar_serie(disciplinas_dict):
    """Deduz a série (1, 2 ou 3) a partir dos nomes das disciplinas do ensino
    médio (ex.: "... - 3ª SÉRIE"). Devolve int 1..3 ou ``None`` se não for
    possível inferir (ex.: mapa só com disciplinas técnicas)."""
    contagem = Counter()
    for nome in (disciplinas_dict or {}).values():
        m = _RE_SERIE.search(_normalizar(nome))
        if m:
            contagem[int(m.group(1))] += 1
    return contagem.most_common(1)[0][0] if contagem else None

"""Gestão Acadêmica EPTNM — CEFET-MG.

App web (Streamlit) para coordenadores dos cursos técnicos de nível médio (EPTNM).
O coordenador informa seu e-mail institucional, faz o upload do Mapa de Turma
(.xls) e recebe o relatório de acompanhamento acadêmico (PDF) por e-mail.

Caso especial: o Curso integrado **Trânsito + Estradas (1ª série)** exige 2
arquivos (mapa de Trânsito + mapa de Estradas) e produz **2 PDFs**, um para
cada curso.
"""
import os
import re

import streamlit as st

from core import relatorios
from core.email_sender import DOMINIO_INSTITUCIONAL, email_valido, enviar_relatorio
from core.manipulacao import (
    ArquivoInvalidoError,
    processar_curso_generico,
    processar_transito_estradas,
)
from core.usage_tracker import registrar_uso

st.set_page_config(
    page_title="Gestão Acadêmica EPTNM — CEFET-MG",
    page_icon="🎓",
    layout="centered",
)

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo_cefet.png")


def _secret(nome, default=""):
    """Lê um segredo de st.secrets e, em fallback, da variável de ambiente."""
    try:
        if nome in st.secrets:
            return st.secrets[nome]
    except Exception:
        pass
    return os.environ.get(nome, default)


def _slug(texto):
    return re.sub(r'\W+', '_', (texto or 'curso').strip().lower()).strip('_') or 'curso'


# --------------------------------
# Cabeçalho
# --------------------------------
st.title("🎓 Gestão Acadêmica EPTNM — CEFET-MG")
st.caption("Educação Profissional Técnica de Nível Médio · Centro Federal de "
           "Educação Tecnológica de Minas Gerais")
st.markdown(
    "Informe seu e-mail institucional e envie o **Mapa de Turma** (`.xls`). "
    "O relatório de acompanhamento acadêmico será gerado e **enviado para o seu "
    f"e-mail `@{DOMINIO_INSTITUCIONAL}`**."
)

# --------------------------------
# Barra lateral — configurações do curso
# --------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")
    eh_transito_estradas = st.checkbox(
        "Curso integrado Trânsito + Estradas (1ª série)",
        value=False,
        help="Marque se você é coordenador(a) da 1ª série de Trânsito ou "
             "Estradas. Esse modo requer os dois mapas e gera dois relatórios "
             "(um para cada curso) no mesmo e-mail.",
    )
    usar_ia = st.toggle(
        "Incluir análise por IA (OpenAI)",
        value=False,
        help="Adiciona um comentário analítico ao relatório. Requer a chave "
             "OPENAI_API_KEY configurada.",
    )
    if usar_ia:
        st.caption("🔒 Os **nomes e as notas individuais dos alunos não são "
                   "enviados à IA** — apenas estatísticas agregadas por disciplina.")
    st.divider()
    if eh_transito_estradas:
        st.info("Modo **Trânsito + Estradas**: envie o mapa de Trânsito **e** o "
                "mapa de Estradas (o de Estradas traz as notas do ensino médio). "
                "Você receberá dois relatórios.")
    else:
        st.info("Envie **1 arquivo**: o mapa de turma completo do seu curso. "
                "Funciona para qualquer curso técnico do EPTNM.")

    st.divider()
    st.subheader("🔒 Privacidade e LGPD")
    st.warning(
        "Os **nomes dos alunos não são entregues à IA**. Quando a análise por IA "
        "está ativa, apenas **estatísticas agregadas por disciplina** (médias, "
        "medianas, desvios, mín./máx.) são enviadas à OpenAI — **sem nomes e sem "
        "notas individuais**."
    )
    with st.expander("Saiba mais sobre o tratamento dos dados"):
        st.markdown(
            "- **Nomes e notas** são *dados pessoais* protegidos pela **LGPD**. "
            "Notas escolares **não** são *dados sensíveis* no sentido legal "
            "(art. 5º, II), mas seguem protegidas como dados pessoais.\n"
            "- Esses dados ficam apenas no **processamento interno** e no **PDF** "
            "enviado à sua caixa institucional `@cefetmg.br`.\n"
            "- O mapa de turma e o PDF são processados **em memória** e **não são "
            "armazenados** no servidor.\n"
            "- Pela política da **API da OpenAI**, os dados enviados **não são "
            "usados para treinar** os modelos — ainda assim, por isso, **nenhum "
            "nome ou nota individual** é compartilhado."
        )

# --------------------------------
# Formulário
# --------------------------------
email = st.text_input(
    f"📧 Seu e-mail institucional (@{DOMINIO_INSTITUCIONAL})",
    placeholder=f"seu.nome@{DOMINIO_INSTITUCIONAL}",
)

if eh_transito_estradas:
    c1, c2 = st.columns(2)
    with c1:
        arquivo_transito = st.file_uploader(
            "📄 Mapa — Trânsito (.xls)", type=["xls"], key="up_transito")
    with c2:
        arquivo_estradas = st.file_uploader(
            "📄 Mapa — Estradas (.xls)", type=["xls"], key="up_estradas")
    arquivos_ok = bool(arquivo_transito and arquivo_estradas)
else:
    arquivo_unico = st.file_uploader(
        "📄 Mapa de Turma (.xls)", type=["xls"], key="up_unico")
    arquivos_ok = bool(arquivo_unico)

st.caption("ℹ️ O nome do curso e o bimestre são lidos automaticamente do cabeçalho "
           "do arquivo. **Envie o mapa de um único bimestre por vez.**")

enviar = st.button("📨 Gerar e enviar relatório por e-mail", type="primary")


# --------------------------------
# Processamento
# --------------------------------
def _gerar_pdf_para_conjunto(conjunto, usar_ia, api_key):
    """Gera (nome_arquivo, pdf_buffer, nome_curso) para um conjunto (df, df, disc, meta)."""
    df_notas, df_faltas, disciplinas_dict, metadados = conjunto
    nome_curso = metadados.get('curso_amigavel') or metadados.get('curso') or 'Curso'

    estat = relatorios.calcular_estatisticas(
        df_notas, disciplinas_dict, df_faltas=df_faltas, metadados=metadados)
    if usar_ia:
        estat['comentario_ia'] = relatorios.gerar_comentario_ia(
            estat, nome_curso, api_key)
    figuras = relatorios.gerar_todos_graficos(
        df_notas, nome_curso, disciplinas_dict, estat, df_faltas=df_faltas)
    logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
    pdf_buffer = relatorios.criar_relatorio_pdf(
        nome_curso, estat, figuras, logo_path=logo)

    bim = metadados.get('bimestre_num') or 'X'
    nome_arquivo = f"relatorio_{_slug(nome_curso)}_bim{bim}.pdf"
    return nome_arquivo, pdf_buffer, nome_curso


def processar_e_enviar():
    if not email_valido(email):
        st.error(f"Informe um e-mail válido terminado em `@{DOMINIO_INSTITUCIONAL}`. "
                 "Só professores do CEFET-MG podem usar este serviço.")
        return
    if not arquivos_ok:
        st.error("Envie o(s) arquivo(s) `.xls` necessário(s).")
        return

    remetente = _secret("GMAIL_USER")
    senha_app = _secret("GMAIL_APP_PASSWORD")
    if not remetente or not senha_app:
        st.error("O envio de e-mail ainda não foi configurado pelo administrador "
                 "(GMAIL_USER / GMAIL_APP_PASSWORD).")
        return

    api_key = _secret("OPENAI_API_KEY") if usar_ia else ""

    # 1. Processa dados
    try:
        with st.spinner("Processando o(s) mapa(s) de turma..."):
            if eh_transito_estradas:
                conjuntos = processar_transito_estradas(arquivo_transito, arquivo_estradas)
            else:
                conjuntos = [processar_curso_generico(arquivo_unico)]
    except ArquivoInvalidoError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        return

    conjuntos_validos = [c for c in conjuntos if not c[0].empty]
    if not conjuntos_validos:
        st.error("Nenhum aluno válido foi encontrado no arquivo. Verifique o mapa de turma.")
        return

    # 2. Gera PDFs
    try:
        with st.spinner("Gerando o(s) relatório(s)..."):
            anexos = []
            cursos = []
            for conjunto in conjuntos_validos:
                nome_arquivo, pdf_buffer, nome_curso = _gerar_pdf_para_conjunto(
                    conjunto, usar_ia, api_key)
                anexos.append((nome_arquivo, pdf_buffer))
                cursos.append(nome_curso)
    except Exception as e:
        st.error(f"Erro ao gerar o(s) relatório(s): {e}")
        return

    # 3. Envia por e-mail
    try:
        with st.spinner(f"Enviando para {email}..."):
            enviar_relatorio(
                destinatario=email.strip(),
                remetente=remetente,
                senha_app=senha_app,
                anexos=anexos,
                cursos=cursos,
            )
    except Exception as e:
        st.error(f"Não foi possível enviar o e-mail: {e}")
        return

    bim = conjuntos_validos[0][3].get('bimestre_num') if conjuntos_validos else None
    registrar_uso(cursos, bim, email.strip(), st.secrets)

    cursos_fmt = " e ".join(f"**{c}**" for c in cursos)
    st.success(f"✅ Relatório(s) — {cursos_fmt} — enviado(s) para **{email.strip()}**.")
    st.info("Verifique sua caixa de entrada (e a pasta de spam). "
            "Nenhum arquivo fica armazenado nesta página.")


if enviar:
    processar_e_enviar()

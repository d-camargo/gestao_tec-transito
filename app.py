"""Gestão Acadêmica EPTNM — CEFET-MG.

App web (Streamlit) para coordenadores dos cursos técnicos de nível médio (EPTNM).
O coordenador informa seu e-mail institucional, faz o upload do Mapa de Turma
(.xls) e recebe o relatório de acompanhamento acadêmico (PDF) por e-mail.

Caso especial: o Curso Técnico em Trânsito exige 2 arquivos (mapa de Trânsito +
mapa completo, de onde são puxadas as notas de ensino médio).
"""
import os

import streamlit as st

from core import relatorios
from core.email_sender import DOMINIO_INSTITUCIONAL, email_valido, enviar_relatorio
from core.manipulacao import processar_curso_generico, processar_transito

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
    eh_transito = st.checkbox(
        "Curso Técnico em Trânsito",
        value=False,
        help="Marque apenas se você é coordenador(a) do Curso Técnico em Trânsito. "
             "Este curso exige 2 arquivos (mapa de Trânsito + mapa completo).",
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
    if eh_transito:
        st.info("Modo **Trânsito**: envie o mapa de Trânsito **e** o mapa completo "
                "da escola (necessário para as notas de ensino médio).")
    else:
        st.info("Envie **1 arquivo**: o mapa de turma completo do seu curso.")

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

if eh_transito:
    nome_curso = "Trânsito"
    st.text_input("Curso", value="Técnico em Trânsito", disabled=True)
else:
    nome_curso = st.text_input(
        "Nome do curso",
        placeholder="Ex.: Edificações, Eletrônica, Informática...",
    )

if eh_transito:
    c1, c2 = st.columns(2)
    with c1:
        arquivo_transito = st.file_uploader(
            "📄 Mapa de Turma — Trânsito (.xls)", type=["xls"], key="up_transito")
    with c2:
        arquivo_completo = st.file_uploader(
            "📄 Mapa de Turma — Completo (.xls)", type=["xls"], key="up_completo")
    arquivos_ok = bool(arquivo_transito and arquivo_completo)
else:
    arquivo_unico = st.file_uploader(
        "📄 Mapa de Turma (.xls)", type=["xls"], key="up_unico")
    arquivos_ok = bool(arquivo_unico)

nome_ok = bool(nome_curso and nome_curso.strip())
enviar = st.button("📨 Gerar e enviar relatório por e-mail", type="primary")


# --------------------------------
# Processamento
# --------------------------------
def processar_e_enviar():
    # Validações
    if not email_valido(email):
        st.error(f"Informe um e-mail válido terminado em `@{DOMINIO_INSTITUCIONAL}`. "
                 "Só professores do CEFET-MG podem usar este serviço.")
        return
    if not nome_ok:
        st.error("Informe o nome do curso.")
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

    # 1. Processa os dados
    try:
        with st.spinner("Processando o mapa de turma..."):
            if eh_transito:
                df_notas, _df_faltas, disciplinas_dict = processar_transito(
                    arquivo_transito, arquivo_completo)
            else:
                df_notas, _df_faltas, disciplinas_dict = processar_curso_generico(
                    arquivo_unico)
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        return

    if df_notas.empty:
        st.error("Nenhum aluno válido foi encontrado no arquivo. Verifique o mapa de turma.")
        return

    # 2. Gera relatório (estatísticas, gráficos, IA opcional, PDF)
    try:
        with st.spinner("Gerando o relatório..."):
            estatisticas = relatorios.calcular_estatisticas(df_notas, disciplinas_dict)
            if usar_ia:
                estatisticas['comentario_ia'] = relatorios.gerar_comentario_ia(
                    estatisticas, nome_curso, api_key)
            figuras = relatorios.gerar_todos_graficos(
                df_notas, nome_curso, disciplinas_dict, estatisticas)
            logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
            pdf_buffer = relatorios.criar_relatorio_pdf(
                nome_curso, estatisticas, figuras, logo_path=logo)
    except Exception as e:
        st.error(f"Erro ao gerar o relatório: {e}")
        return

    # 3. Envia por e-mail
    try:
        with st.spinner(f"Enviando para {email}..."):
            nome_arquivo = f"relatorio_{nome_curso.strip().lower().replace(' ', '_')}.pdf"
            enviar_relatorio(
                destinatario=email.strip(),
                pdf_buffer=pdf_buffer,
                nome_arquivo=nome_arquivo,
                nome_curso=nome_curso.strip(),
                remetente=remetente,
                senha_app=senha_app,
            )
    except Exception as e:
        st.error(f"Não foi possível enviar o e-mail: {e}")
        return

    st.success(f"✅ Relatório do curso **{nome_curso.strip()}** enviado para **{email.strip()}**.")
    st.info("Verifique sua caixa de entrada (e a pasta de spam). "
            "O arquivo não fica armazenado nesta página.")


if enviar:
    processar_e_enviar()

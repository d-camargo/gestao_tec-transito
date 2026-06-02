"""Envio do relatório PDF por e-mail (SMTP — Gmail com senha de app).

As credenciais do remetente vêm de st.secrets / variáveis de ambiente e nunca
ficam no código. O destinatário é informado pelo professor no app e deve
pertencer ao domínio institucional.
"""
import re
import smtplib
import ssl
from email.message import EmailMessage

DOMINIO_INSTITUCIONAL = "cefetmg.br"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def email_valido(email: str, dominio: str = DOMINIO_INSTITUCIONAL) -> bool:
    """Valida o formato e exige o domínio institucional (ex.: @cefetmg.br)."""
    if not email:
        return False
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        return False
    return email.endswith("@" + dominio.lower())


def enviar_relatorio(
    destinatario,
    remetente,
    senha_app,
    anexos=None,
    cursos=None,
    *,
    pdf_buffer=None,
    nome_arquivo=None,
    nome_curso=None,
    host="smtp.gmail.com",
    port=465,
):
    """Envia um ou mais PDFs como anexos. Lança exceção em caso de falha.

    Forma preferida (vários PDFs):
        ``enviar_relatorio(dest, rem, senha, anexos=[(nome, buffer), ...],
                            cursos=['Trânsito', 'Estradas'])``

    Forma legada (1 PDF) continua suportada via ``pdf_buffer``,
    ``nome_arquivo`` e ``nome_curso``.
    """
    if not remetente or not senha_app:
        raise RuntimeError(
            "Credenciais de e-mail não configuradas. Defina GMAIL_USER e "
            "GMAIL_APP_PASSWORD em st.secrets ou nas variáveis de ambiente."
        )

    # Compatibilidade com a chamada antiga.
    if anexos is None and pdf_buffer is not None and nome_arquivo is not None:
        anexos = [(nome_arquivo, pdf_buffer)]
        if cursos is None and nome_curso:
            cursos = [nome_curso]

    if not anexos:
        raise RuntimeError("Nenhum anexo informado para envio.")

    cursos = cursos or []
    if len(cursos) == 1:
        assunto = f"Relatório de Acompanhamento Acadêmico — {cursos[0]}"
        descricao = f"do curso {cursos[0]}"
    elif len(cursos) > 1:
        nomes = " e ".join(cursos)
        assunto = f"Relatórios de Acompanhamento Acadêmico — {nomes}"
        descricao = f"dos cursos {nomes}"
    else:
        assunto = "Relatório de Acompanhamento Acadêmico"
        descricao = ""

    msg = EmailMessage()
    msg["From"] = remetente
    msg["To"] = destinatario
    msg["Subject"] = assunto
    msg.set_content(
        f"Olá,\n\n"
        f"Segue em anexo o Relatório de Acompanhamento Acadêmico "
        f"{descricao}, gerado pela plataforma de Gestão Acadêmica EPTNM do CEFET-MG.\n\n"
        f"Este é um e-mail automático, não responda.\n"
    )

    for arquivo, buffer in anexos:
        pdf_bytes = buffer.getvalue() if hasattr(buffer, "getvalue") else buffer
        msg.add_attachment(
            pdf_bytes, maintype="application", subtype="pdf", filename=arquivo
        )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as server:
        server.login(remetente, senha_app)
        server.send_message(msg)

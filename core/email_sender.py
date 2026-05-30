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
    pdf_buffer,
    nome_arquivo,
    nome_curso,
    remetente,
    senha_app,
    host="smtp.gmail.com",
    port=465,
):
    """Envia o PDF como anexo. Lança exceção em caso de falha de envio."""
    if not remetente or not senha_app:
        raise RuntimeError(
            "Credenciais de e-mail não configuradas. Defina GMAIL_USER e "
            "GMAIL_APP_PASSWORD em st.secrets ou nas variáveis de ambiente."
        )

    msg = EmailMessage()
    msg["From"] = remetente
    msg["To"] = destinatario
    msg["Subject"] = f"Relatório de Acompanhamento Acadêmico — {nome_curso}"
    msg.set_content(
        f"Olá,\n\n"
        f"Segue em anexo o Relatório de Acompanhamento Acadêmico do curso "
        f"{nome_curso}, gerado pela plataforma de Gestão Acadêmica EPTNM do CEFET-MG.\n\n"
        f"Este é um e-mail automático, não responda.\n"
    )

    pdf_bytes = pdf_buffer.getvalue() if hasattr(pdf_buffer, "getvalue") else pdf_buffer
    msg.add_attachment(
        pdf_bytes, maintype="application", subtype="pdf", filename=nome_arquivo
    )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as server:
        server.login(remetente, senha_app)
        server.send_message(msg)

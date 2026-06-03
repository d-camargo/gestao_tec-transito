"""Registro de uso em Google Sheets.

Cada relatório gerado com sucesso grava uma linha na planilha configurada.
Falhas de logging são silenciosas — nunca interrompem o fluxo principal.

Configuração necessária em st.secrets (ou secrets.toml):

    GOOGLE_SHEETS_ID = "1ABC..."   # ID da planilha (da URL)

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN RSA PRIVATE KEY-----\\n..."
    client_email = "nome@projeto.iam.gserviceaccount.com"
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "..."
"""
import datetime

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

_HEADER = ["Data/Hora", "Curso(s)", "Bimestre", "E-mail do Coordenador"]


def registrar_uso(cursos, bimestre, email_coordenador, secrets):
    """Grava uma linha de uso na planilha Google Sheets.

    Parameters
    ----------
    cursos : list[str] | str
        Nome(s) do(s) curso(s) processados.
    bimestre : int | str | None
        Número do bimestre (1–4) ou None se desconhecido.
    email_coordenador : str
        E-mail institucional do coordenador que usou o app.
    secrets : mapping
        Objeto st.secrets (ou dict equivalente). Deve conter
        ``GOOGLE_SHEETS_ID`` e a seção ``[gcp_service_account]``.
    """
    try:
        _gravar(cursos, bimestre, email_coordenador, secrets)
    except Exception:
        pass  # Logging nunca deve interromper o app principal


def _gravar(cursos, bimestre, email_coordenador, secrets):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return

    sheet_id = secrets.get("GOOGLE_SHEETS_ID", "")
    if not sheet_id:
        return

    try:
        creds_info = dict(secrets["gcp_service_account"])
    except (KeyError, TypeError):
        return

    creds = Credentials.from_service_account_info(creds_info, scopes=_SCOPES)
    client = gspread.authorize(creds)
    planilha = client.open_by_key(sheet_id)
    aba = planilha.sheet1

    # Garante cabeçalho na primeira linha se a aba estiver vazia
    if aba.row_count == 0 or not aba.row_values(1):
        aba.append_row(_HEADER)

    cursos_str = ", ".join(cursos) if isinstance(cursos, list) else str(cursos)
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    aba.append_row([agora, cursos_str, str(bimestre or "—"), email_coordenador])

import os
import smtplib
from email.message import EmailMessage

def send_email(to_addresses: list[str], subject: str, body: str) -> str:
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password or sender_email == "your_email@gmail.com":
        return "Error: SENDER_EMAIL or SENDER_PASSWORD not configured properly in .env"

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(to_addresses)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return f"Email successfully sent to {len(to_addresses)} recipients."
    except Exception as e:
        return f"Failed to send email: {e}"

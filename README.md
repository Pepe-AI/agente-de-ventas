# WhatsApp Echo Agent — Walking Skeleton

First increment of a conversational WhatsApp agent. For now it does exactly one
thing: receives a message from the Twilio WhatsApp Sandbox, validates the
request signature, and replies `Echo: <text>`.

The architecture leaves the right seams for later increments: a `Channel` port
(swap Twilio for another provider by writing one adapter), a neutral domain
model, and a `handle_message` orchestration seam where conversational logic
will plug in.

## Requirements

- Python 3.12+
- A Twilio account with the WhatsApp Sandbox enabled
- [ngrok](https://ngrok.com/) (or any tunnel) to expose your local server

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt

copy .env.example .env              # cp on macOS/Linux
# then fill in your Twilio credentials in .env
```

## Run

```powershell
uvicorn app.main:app --reload --port 8000
```

In another terminal, open the tunnel:

```powershell
ngrok http 8000
```

Copy the public HTTPS URL ngrok prints (e.g. `https://abc123.ngrok-free.app`).

## Point the Twilio Sandbox at the webhook

1. Twilio Console → **Messaging → Try it out → Send a WhatsApp message → Sandbox settings**.
2. Set **"When a message comes in"** to:
   `https://<your-ngrok-subdomain>/webhook/whatsapp` with method **POST**.
3. Save.

## Try it

Send a WhatsApp message to the sandbox number. You should receive `Echo: <your message>`.

Signature validation works behind the tunnel because the webhook reconstructs
the public URL from the `X-Forwarded-Proto` and `Host` headers rather than the
internal request URL. A POST with an invalid/missing signature returns `403`.

## Test

```powershell
pytest
ruff check .
```

## Project layout

```
app/
  main.py            # HTTP layer + composition root
  config.py          # Settings (pydantic-settings) + header constants
  channels/
    base.py          # Channel port (Protocol)
    twilio.py        # TwilioChannel adapter (parse, send, verify signature)
  domain/
    models.py        # IncomingMessage (neutral domain model)
    orchestrator.py  # handle_message — the conversational-logic seam
tests/
```

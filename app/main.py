"""HTTP layer and composition root.

The endpoint is intentionally thin: it only deals with HTTP concerns (form
parsing, signature verification, status codes) and delegates all behavior to
the domain (:func:`handle_message`) and the transport (:class:`Channel`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Request, Response
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.channels.base import Channel
from app.channels.twilio import InvalidPayloadError, TwilioChannel
from app.config import HttpHeader, Settings, get_settings
from app.domain.orchestrator import handle_message

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger()

WEBHOOK_PATH = "/webhook/whatsapp"
EMPTY_TWIML = "<Response></Response>"
TWIML_MEDIA_TYPE = "application/xml"

app = FastAPI(title="WhatsApp Echo Agent")


@lru_cache
def get_channel() -> Channel:
    """Build the transport adapter once (composition root).

    Wired here so handlers never instantiate clients themselves and the whole
    app depends only on the :class:`Channel` abstraction.
    """
    settings: Settings = get_settings()
    token = settings.twilio_auth_token.get_secret_value()
    return TwilioChannel(
        validator=RequestValidator(token),
        client=Client(settings.twilio_account_sid, token),
        from_=settings.twilio_whatsapp_from,
    )


def build_public_url(request: Request) -> str:
    """Reconstruct the public URL Twilio used to sign the request.

    Behind a tunnel (ngrok) ``request.url`` is the internal address, which
    would make signature validation fail. We rebuild it from the forwarded
    headers, falling back to ``request.url`` for local development.
    """
    proto = request.headers.get(HttpHeader.X_FORWARDED_PROTO)
    host = request.headers.get(HttpHeader.HOST)
    if proto and host:
        url = f"{proto}://{host}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        return url
    return str(request.url)


@app.post(WEBHOOK_PATH)
async def whatsapp_webhook(
    request: Request,
    channel: Annotated[Channel, Depends(get_channel)],
) -> Response:
    """Receive an inbound WhatsApp message and reply with its echo."""
    # Twilio posts urlencoded fields; keep only str values (drop any uploads).
    form = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    signature = request.headers.get(HttpHeader.X_TWILIO_SIGNATURE, "")
    url = build_public_url(request)

    if not channel.verify_signature(url, form, signature):
        log.warning("invalid_signature", url=url)
        return Response(status_code=403)

    try:
        msg = channel.parse_incoming(form)
    except InvalidPayloadError as exc:
        log.warning("malformed_payload", error=str(exc))
        return Response(status_code=400)

    log.info(
        "incoming_message",
        sender=msg.sender,
        message_id=msg.message_id,
        text=msg.text,
    )

    reply = handle_message(msg)
    await channel.send(msg.sender, reply)

    return Response(content=EMPTY_TWIML, media_type=TWIML_MEDIA_TYPE)

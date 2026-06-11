# WhatsApp Echo Agent

A conversational WhatsApp agent built incrementally.

- **Increment 1** — echo skeleton: receive a Twilio WhatsApp message, validate
  the signature, reply `Echo: <text>`.
- **Increment 2** — concurrency layer on Redis: fast-ack + idempotency +
  debounce/buffering + per-conversation lock + flood protection. `handle_message`
  still echoes, but now receives the **combined** text of the buffered messages.

The architecture keeps the right seams for later increments: a `Channel` port
(swap Twilio for another provider by writing one adapter), a neutral domain
model, and a `handle_message` orchestration seam where conversational logic
will plug in.

## How it works (increment 2)

The webhook validates the signature, runs a sequence of fast Redis-backed checks,
and returns `200` immediately. The orchestrator call and the reply happen in a
background flush — never in the request.

Endpoint checks, in order:

1. Validate Twilio signature → `403` if invalid.
2. `blocked:{sender}` present → discard silently (`200`, no buffering, no reply).
3. Duplicate `MessageSid` (idempotency, `SET NX` + TTL) → discard (`200`).
4. Rate-limit hit (`INCR` fixed window); over threshold → block, discard (`200`).
5. `RPUSH` to the sender buffer; over `BUFFER_MAX` → block, discard (`200`).
6. Register debounce token + schedule background flush.
7. Fast-ack `200`.

Background flush (after the debounce window):

1. Still the latest debounce token? If not, abort (a newer message will flush).
2. Acquire the per-conversation lock (`SET NX EX`); if not, abort.
3. Drain the whole buffer (`LRANGE` + `DEL`) and join with `\n`.
4. If the sender got blocked meanwhile, abort without replying.
5. `handle_message(combined)` → `Channel.send`.
6. Release the lock (owner-safe, via `WATCH`/`MULTI`).

## Deployment

The service is deployed on **Render** as a Python web service, with Redis via
Render's internal `REDIS_URL` (configured in the dashboard). Render deploys
native Python — there is no Docker or local Redis setup.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TWILIO_ACCOUNT_SID` | yes | — | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | yes | — | Twilio auth token (used for signature validation) |
| `TWILIO_WHATSAPP_FROM` | yes | — | WhatsApp sender, e.g. `whatsapp:+14155238886` |
| `REDIS_URL` | yes | — | Redis connection URL (Render internal URL) |
| `GEMINI_API_KEY` | yes | — | Google Gemini API key (LLM) |
| `LLM_MODEL` | no | `gemini-3.5-flash` | Gemini model id |
| `DEBOUNCE_WINDOW_S` | no | `3` | Debounce window before a flush fires |
| `DEDUP_TTL_S` | no | `3600` | Idempotency key TTL |
| `LOCK_TTL_S` | no | `30` | Per-conversation lock TTL |
| `RATE_WINDOW_S` | no | `10` | Rate-limit fixed window |
| `RATE_THRESHOLD` | no | `15` | Max messages per window before blocking |
| `BLOCK_COOLDOWN_S` | no | `600` | Block duration |
| `BUFFER_MAX` | no | `10` | Max buffered messages before treating as flood |

See [.env.example](.env.example) for the full list.

## Local development (tests only)

Code and unit tests run locally; the unit tests use **fakeredis**, so no local
Redis is needed. The full server runs on Render (it needs `REDIS_URL`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt

pytest                # unit tests (fakeredis)
ruff check .          # lint
ruff format --check . # formatting
pyright               # strict type check (app/)
```

## End-to-end test (on Render)

E2E runs against the **deployed** service — no ngrok.

1. Run the unit tests locally (`pytest`) and make sure they pass.
2. Commit and push to the branch Render deploys from. Render redeploys automatically.
3. Point the Twilio Sandbox webhook at the Render URL:
   Twilio Console → **Messaging → Try it out → Send a WhatsApp message → Sandbox settings**
   → set **"When a message comes in"** to
   `https://<your-app>.onrender.com/webhook/whatsapp` (method **POST**) → Save.
4. From your phone, send WhatsApp messages to the sandbox number and verify:
   - **Debounce:** send 3 messages quickly → you get **one** combined echo.
   - **Idempotency:** a Twilio retry of the same message is processed once.
   - **Flood:** spam past the threshold → that number is blocked and ignored for
     the cooldown.
5. Inspect **Render Logs** for the structured events (`incoming_buffered`,
   `flush_sent`, `duplicate_discard`, `flood_blocked`, ...).

Signature validation works behind Render's proxy because the webhook
reconstructs the public URL from the `X-Forwarded-Proto` and `Host` headers
rather than the internal request URL.

## Project layout

```
app/
  main.py              # HTTP layer + composition root (thin endpoint, fast-ack)
  config.py            # Settings (pydantic-settings) + header constants
  channels/
    base.py            # Channel port (Protocol)
    twilio.py          # TwilioChannel adapter (parse, send, verify signature)
  concurrency/
    keys.py            # Redis key namespace (no magic strings)
    dedup.py           # idempotency (SET NX)
    rate_limit.py      # flood: rate-limit + block
    buffer.py          # per-sender buffer (RPUSH / LRANGE+DEL)
    debounce.py        # debounce token bookkeeping
    lock.py            # per-conversation lock (acquire / owner-safe release)
    config.py          # ConcurrencyConfig
    flush.py           # background flush + scheduling
  crm/
    relay.py           # relay-to-human seam (stub; Kommo Chats API in inc 8)
  llm/
    base.py            # LLM port (Protocol, structured output)
    gemini.py          # GeminiLLM adapter (google-genai native structured output)
  understanding/
    schemas.py         # response schemas (dummy placeholder for inc 3)
    engine.py          # understand_turn — filled/missing slots + question
  domain/
    models.py          # IncomingMessage + Referral (neutral domain model)
    orchestrator.py    # handle_message — runs the understanding engine
tests/
```

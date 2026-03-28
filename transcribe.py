"""
Voice note handling: download audio from Meta's CDN, transcribe with Whisper.

Called by main.py when a WhatsApp webhook delivers a message with type "audio".
"""
import logging
from io import BytesIO

import httpx
from openai import AsyncOpenAI

from config import WHATSAPP_TOKEN, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Meta Graph API version for media download.
# Kept separate from the messaging version in whatsapp.py so each
# can be bumped independently if Meta deprecates one.
_GRAPH_VERSION = "v21.0"

# Whisper API hard limit is 25 MB. WhatsApp's own limit for voice
# notes is 16 MB, so we'll never hit 25 MB — but we reject anything
# above 20 MB as a safety margin to avoid a confusing Whisper error.
_MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB

_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Map WhatsApp / Meta MIME types to the file extensions Whisper accepts.
_MIME_TO_EXT: dict[str, str] = {
    "audio/ogg": "ogg",
    "audio/ogg; codecs=opus": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp4": "mp4",
    "audio/aac": "aac",
    "audio/amr": "amr",
    "audio/wav": "wav",
    "audio/webm": "webm",
}


def _ext(mime_type: str) -> str:
    """Return a Whisper-compatible extension for a given MIME type."""
    base = mime_type.split(";")[0].strip().lower()
    return _MIME_TO_EXT.get(base, "ogg")  # OGG/Opus is the WhatsApp default


async def download_media(media_id: str) -> tuple[bytes, str]:
    """
    Download a WhatsApp voice note from Meta's CDN.

    Two-step process:
      1. GET /v21.0/{media_id} → returns the CDN download URL and MIME type.
      2. GET {cdn_url}         → returns the raw audio bytes.

    Args:
        media_id: The media ID from the WhatsApp webhook payload.

    Returns:
        (audio_bytes, mime_type) — raw audio and its MIME type string.

    Raises:
        ValueError:  If the file is too large for Whisper.
        httpx.*:     If Meta's API returns an error.
    """
    auth_headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        # Step 1 — resolve the media ID to a CDN URL
        meta_resp = await client.get(
            f"https://graph.facebook.com/{_GRAPH_VERSION}/{media_id}",
            headers=auth_headers,
            timeout=10.0,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        cdn_url   = meta.get("url")
        mime_type = meta.get("mime_type", "audio/ogg")
        file_size = meta.get("file_size", 0)

        if not cdn_url:
            raise ValueError(f"No CDN URL returned for media_id={media_id}")

        # Reject before downloading if Meta already tells us it's too big
        if file_size and file_size > _MAX_AUDIO_BYTES:
            raise ValueError(
                f"Voice note too large: {file_size:,} bytes "
                f"(limit {_MAX_AUDIO_BYTES:,} bytes)"
            )

        # Step 2 — download the audio bytes
        audio_resp = await client.get(
            cdn_url,
            headers=auth_headers,
            timeout=30.0,
            follow_redirects=True,
        )
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content

    # Double-check after download (Meta's file_size can be absent or wrong)
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise ValueError(
            f"Voice note too large after download: {len(audio_bytes):,} bytes"
        )

    logger.info(
        "Downloaded media %s — %d bytes, MIME: %s",
        media_id, len(audio_bytes), mime_type,
    )
    return audio_bytes, mime_type


async def transcribe(audio_bytes: bytes, mime_type: str) -> str:
    """
    Transcribe audio bytes using OpenAI Whisper.

    Whisper automatically detects the language (Arabic, English, etc.) —
    no language hint needed.

    Args:
        audio_bytes: Raw audio data.
        mime_type:   MIME type of the audio (used to name the virtual file
                     so Whisper knows the format).

    Returns:
        Stripped transcription string. Empty string if nothing was heard.

    Raises:
        openai.APIError: On Whisper API failures.
    """
    ext = _ext(mime_type)

    # Whisper expects a file-like object with a .name attribute that
    # ends in a recognised audio extension.
    audio_file = BytesIO(audio_bytes)
    audio_file.name = f"voice.{ext}"

    response = await _openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )

    text = (response.text or "").strip()
    logger.info("Whisper result (%d bytes audio): %r", len(audio_bytes), text[:120])
    return text

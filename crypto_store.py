"""
At-rest encryption helpers for sensitive values (e.g. user-supplied LLM API keys).

The encryption key is resolved in this order:

1. The ``AGENT_LASSO_ENC_KEY`` environment variable (a urlsafe-base64 Fernet key).
2. A key file persisted OUTSIDE the database at ``~/.agent-lasso/enc.key``
   (created with ``0600`` permissions). If it does not exist it is generated
   once via :func:`cryptography.fernet.Fernet.generate_key` and reused on every
   subsequent run.

The key file location can be overridden with ``AGENT_LASSO_ENC_KEY_FILE``.

Design goals:

* Never crash on legacy plaintext data. :func:`decrypt` returns the value
  unchanged when it is not a valid Fernet token, so rows written before this
  module existed keep working. They are transparently re-encrypted the next
  time the owning record is saved (writes always go through :func:`encrypt`).
"""

import os
import stat
import logging
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ENV_KEY_VAR = "AGENT_LASSO_ENC_KEY"
_ENV_KEY_FILE_VAR = "AGENT_LASSO_ENC_KEY_FILE"
_DEFAULT_KEY_DIR = Path.home() / ".agent-lasso"
_DEFAULT_KEY_FILE = _DEFAULT_KEY_DIR / "enc.key"

# Cached Fernet instance so we don't re-read the key file on every call.
_fernet: Optional[Fernet] = None


def _key_file_path() -> Path:
    override = os.getenv(_ENV_KEY_FILE_VAR)
    return Path(override) if override else _DEFAULT_KEY_FILE


def _load_or_create_key() -> bytes:
    """Return a Fernet key from the environment or the on-disk key file.

    When neither exists, generate a new key, persist it to the key file with
    ``0600`` permissions, and return it.
    """
    env_key = os.getenv(_ENV_KEY_VAR)
    if env_key:
        return env_key.encode() if isinstance(env_key, str) else env_key

    key_file = _key_file_path()
    if key_file.exists():
        return key_file.read_bytes().strip()

    # Generate and persist a fresh key outside the database.
    key = Fernet.generate_key()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    # Create the file with restrictive permissions from the start.
    fd = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    try:
        os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        # Best effort on platforms without full POSIX permission support.
        pass
    logger.warning(
        "Generated a new encryption key at %s. Back this file up securely; "
        "losing it makes encrypted API keys unrecoverable. Set the %s "
        "environment variable to manage the key explicitly.",
        key_file,
        _ENV_KEY_VAR,
    )
    return key


def get_fernet() -> Fernet:
    """Return a cached :class:`~cryptography.fernet.Fernet` instance."""
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_load_or_create_key())
    return _fernet


def encrypt(value: Optional[str]) -> Optional[str]:
    """Encrypt a plaintext string, returning a urlsafe token string.

    ``None`` passes through unchanged so callers can store NULLs.
    """
    if value is None:
        return None
    token = get_fernet().encrypt(value.encode("utf-8"))
    return token.decode("utf-8")


def decrypt(value: Optional[str]) -> Optional[str]:
    """Decrypt a Fernet token string back to plaintext.

    If ``value`` is not a valid Fernet token (e.g. legacy plaintext written
    before encryption was introduced) it is returned unchanged so the caller
    never crashes on pre-existing data. Such rows are transparently
    re-encrypted the next time the owning record is saved.
    """
    if value is None:
        return None
    try:
        return get_fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError, TypeError):
        # Legacy plaintext (or otherwise undecryptable) — return as-is.
        return value

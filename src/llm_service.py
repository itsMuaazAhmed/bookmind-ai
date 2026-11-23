"""
LLMService – thin wrapper around the Google Gemini API.

The class loads the API key from the environment (or .env) at import time,
configures the `google.generativeai` client, and lazily creates a
`GenerativeModel` instance.  The model name and API key are configurable
via environment variables:

    GEMINI_API_KEY   – required
    GEMINI_MODEL     – optional, defaults to "gemini-1.5-flash"

If the primary model is not available (e.g., a 404 error), the service
automatically falls back to a known‑good model ("gemini-1.0-pro").
This prevents the UI from crashing when the free tier does not expose the
requested model.
"""

import os
from typing import Optional

import google.generativeai as genai

# ----------------------------------------------------------------------
# Configuration – read once at import time
# ----------------------------------------------------------------------
API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "❌ GEMINI_API_KEY not found. Create a .env file (or set the variable in the shell) with:\n"
        "GEMINI_API_KEY=your-gemini-api-key"
    )

# Primary model – can be overridden via env var.
PRIMARY_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
# Fallback model known to work with the v1beta endpoint.
FALLBACK_MODEL: str = "gemini-1.0-pro"

# Configure the client – idempotent.
genai.configure(api_key=API_KEY)


class LLMService:
    """Facade for the Gemini GenerativeModel with graceful fallback.

    The service tries to instantiate ``PRIMARY_MODEL`` first.  If the model
    is not found (the API returns a 404), it switches to ``FALLBACK_MODEL``
    and logs a warning.  The active model name can be inspected via the
    ``model_name`` property.
    """

    def __init__(self) -> None:
        self._model: Optional[genai.GenerativeModel] = None
        self._active_model_name: str = PRIMARY_MODEL

    # ------------------------------------------------------------------
    # Internal helper to lazily create the model with fallback handling
    # ------------------------------------------------------------------
    def _ensure_model(self) -> genai.GenerativeModel:
        if self._model is not None:
            return self._model
        try:
            self._model = genai.GenerativeModel(self._active_model_name)
        except Exception as e:
            # Most likely a 404 indicating the model is unavailable.
            print(f"⚠️ Model {self._active_model_name} not available ({e}); falling back to {FALLBACK_MODEL}")
            self._active_model_name = FALLBACK_MODEL
            self._model = genai.GenerativeModel(self._active_model_name)
        return self._model

    @property
    def model(self) -> genai.GenerativeModel:
        """Return the lazy‑loaded model instance (with fallback)."""
        return self._ensure_model()

    @property
    def model_name(self) -> str:
        """Name of the model currently in use (primary or fallback)."""
        return self._active_model_name

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
        """Generate a text response from Gemini, with fallback if the model does not support `generateContent`.

        Parameters
        ----------
        prompt: str
            The full prompt (system + user question).
        temperature: float, default 0.2
            Controls randomness; lower = more deterministic.

        Returns
        -------
        str
            The stripped text response.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            return response.text.strip()
        except Exception as e:
            # If generation fails (e.g., model does not support the method),
            # fall back to the known‑good model and retry once.
            print(f"⚠️ Generation with model {self._active_model_name} failed ({e}); trying fallback {FALLBACK_MODEL}")
            self._active_model_name = FALLBACK_MODEL
            self._model = None  # force recreation of fallback model
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            return response.text.strip()

    @staticmethod
    def list_models() -> list[str]:
        """Return a list of model names the current account can access.

        Useful for troubleshooting model‑availability issues.
        """
        try:
            models = genai.list_models()
            return [m.name for m in models]
        except Exception as e:
            print(f"⚠️ Unable to list models: {e}")
            return []
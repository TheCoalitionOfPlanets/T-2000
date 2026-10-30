import re

from .errors import T2000Error


def split_phrases(text: str, max_chars: int = 180) -> list[str]:
    """Bound phrase latency while preferring punctuation and preserving every word.

    This segments complete TTS requests; it is not neural chunk streaming.
    Decimal numbers and initials are kept intact where possible.
    """
    text = " ".join(text.split())
    if not text:
        raise T2000Error("Enter text to speak.")
    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    sentences = re.split(r"(?<=[!?;])\s+|(?<=[.])\s+(?=[A-Z\"'“])", text)
    phrases = []
    for sentence in sentences:
        words = sentence.split()
        current = ""
        for word in words:
            if len(word) > max_chars:
                raise T2000Error("A word is too long to synthesize. Add spaces or shorten it.")
            if current and len(current) + 1 + len(word) > max_chars:
                phrases.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            phrases.append(current)
    return phrases

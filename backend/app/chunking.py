"""Bounded paragraph/sentence splitting without dropping punctuation or words."""
import re
from .errors import OpenVoiceError


def chunks(text: str, limit: int = 350) -> list[str]:
    if not isinstance(text, str) or not text.strip() or len(text) > 100000:
        raise OpenVoiceError('Enter between 1 and 100,000 characters.')
    result = []
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        current = ''
        for sentence in re.split(r'(?<=[.!?。！？])\s+', paragraph.strip()):
            sentence = ' '.join(sentence.split())
            # Keep a complete sentence together whenever it fits by itself.
            # Previously the word loop could split it just to fill the last chunk.
            if current and len(sentence) <= limit and len(current) + 1 + len(sentence) > limit:
                result.append(current)
                current = ''
            for word in sentence.split():
                if len(word) > limit:
                    raise OpenVoiceError('A word exceeds the chunk limit. Add punctuation or spaces.')
                if len(current) + len(word) + 1 > limit:
                    result.append(current)
                    current = ''
                current = (current + ' ' + word).strip()
            if len(current) >= limit // 2:
                result.append(current)
                current = ''
        if current:
            result.append(current)
    return result

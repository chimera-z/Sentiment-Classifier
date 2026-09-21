import re
import string


def clean_text(text: str) -> str:
    text = text.lower()

    # Preserve HTML line breaks as a special token.
    text = re.sub(r"<br\s*/?>", " br ", text)

    # Remove URLs.
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers.
    text = re.sub(r"\d+", "", text)

    # Remove punctuation.
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove non-word characters.
    text = re.sub(r"\W+", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text

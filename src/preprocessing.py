"""
Week 1 - Data Loading & Cleaning
---------------------------------
Everything here is implemented from scratch using only Python's standard
library (re, string). No NLTK / spaCy / sklearn text utilities are used,
per the "from scratch" requirement of the project brief.
"""

import re
import string

# A hand-written stopword list (English). Kept as a plain set so it's easy
# to inspect/extend -- this replaces nltk.corpus.stopwords.
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what what's
when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself
yourselves
""".split())

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML, remove punctuation & digits, collapse
    whitespace. Pure regex/str ops -- no library text-cleaners."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # urls
    text = re.sub(r"<.*?>", " ", text)                  # html tags
    text = text.translate(_PUNCT_TABLE)                 # punctuation
    text = re.sub(r"\d+", " ", text)                    # digits
    text = re.sub(r"\s+", " ", text).strip()            # whitespace
    return text


def tokenize(text: str) -> list:
    """Manual whitespace tokenizer (replaces nltk.word_tokenize)."""
    return text.split(" ") if text else []


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]


def simple_stem(word: str) -> str:
    """A tiny rule-based stemmer (Porter-lite) so we avoid importing
    nltk's PorterStemmer. Handles the most common English suffixes."""
    for suf in ("ational", "tional", "ization", "iveness", "fulness",
                "ousness", "ing", "edly", "ies", "ied", "ed", "ly",
                "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def preprocess(text: str) -> list:
    """Full Week-1 pipeline: clean -> tokenize -> remove stopwords -> stem."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = [simple_stem(t) for t in tokens]
    return tokens


if __name__ == "__main__":
    sample = "BREAKING: Scientists Discover the Moon is Actually Made of Cheese!!! Visit http://fakeurl.com"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
    print("Tokens   :", preprocess(sample))

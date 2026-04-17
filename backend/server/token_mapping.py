HF_SPACE_MARKER = "▁"

TRIE_START = "A"
TRIE_STOP = "Z"

TRIE_SPACE = "S"

TRIE_NUMPAD = "N"
TRIE_SHIFT = "U"
TRIE_SPECIAL_SHIFT = "Q"

# intentionally does not include TRIE_START and TRIE_STOP
TRIE_CONTROL_CHARS = TRIE_NUMPAD + TRIE_SHIFT + TRIE_SPECIAL_SHIFT

numbers = "0123456789"
letters = "abcdefghijklmnopqrstuvwxyz"
uppercase_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
normal_punctuation = "',.?"
other_special_chars = r"""!"$%&()*+-/:;<=>@[\]^_`{|}~#"""

mappable_hf_chars = (numbers +
    letters +
    uppercase_letters +
    normal_punctuation +
    other_special_chars +
    HF_SPACE_MARKER
)

# Uppercase letters are normally encoded via TRIE_SHIFT + lowercase; standalone `A` is TRIE_START
# (root token) and must appear in this set for round-trip from lex tokens.
legal_trie_chars = (numbers +
    letters + # intentionally omit uppercase letters
    normal_punctuation +
    other_special_chars +
    TRIE_CONTROL_CHARS +
    TRIE_START +
    TRIE_SPACE
)

def has_trie_mapping(hf_token: str) -> bool:
    return all(char in mappable_hf_chars for char in hf_token)

def hf_token_to_trie_token(hf_token: str) -> str:
    assert has_trie_mapping(hf_token)
    chars = list(hf_token)
    out = ""
    while chars:
        c = chars.pop(0)
        if c in numbers:
            out += TRIE_NUMPAD
            out += c
        elif c in uppercase_letters:
            out += TRIE_SHIFT
            out += c.lower()
        elif c in other_special_chars:
            out += TRIE_SPECIAL_SHIFT
            out += c
        elif c == HF_SPACE_MARKER:
            out += TRIE_SPACE
        else:
            out += c
    return out

def trie_token_to_hf_token(trie_token: str) -> str:
    assert all(char in legal_trie_chars for char in trie_token), f"invalid trie token: {trie_token!r}"
    out = ""
    chars = list(trie_token)
    try:
        while chars:
            c = chars.pop(0)
            if c in TRIE_NUMPAD:
                next = chars.pop(0)
                assert next in numbers
                out += next
            elif c in TRIE_SHIFT:
                next = chars.pop(0)
                assert next in letters
                out += next.upper()
            elif c in TRIE_SPECIAL_SHIFT:
                next = chars.pop(0)
                assert next in other_special_chars
                out += next
            elif c == TRIE_SPACE:
                out += HF_SPACE_MARKER
            else:
                out += c
        return out
    except IndexError as e:
        raise ValueError(f"invalid trie token (failed to pop): {trie_token!r}") from e

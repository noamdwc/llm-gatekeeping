
# Common zero-width / invisible characters
ZERO_WIDTH_CHARS = {
    "\u200b",  # ZWSP
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u2060",  # Word joiner
    "\ufeff",  # BOM / ZWNBSP
}

CONTROL_CATS = {"Cc", "Cf", "Co", "Cs"}

BIDI_CHARS = {
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
}

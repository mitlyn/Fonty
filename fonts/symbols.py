from itertools import chain

SYMBOLS = {
    "ua": [*"АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя"],
    "en": [*"AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"],
    "puncts": [*",.;:'\"()/[]{}\\/!@#$%^&*?-+=*<>"],
    "digits": [*"0123456789"],
}


def get_symbols(name: str = None):
    """Returns specified symbol set."""

    if name in SYMBOLS:
        return SYMBOLS[name]

    if name in {"alpha", "a", "letters", "l"}:
        return SYMBOLS["ua"] + SYMBOLS["en"]

    if name in {"symbolic", "s", "non-letters", "n"}:
        return SYMBOLS["puncts"] + SYMBOLS["digits"]

    else:
        return list(chain.from_iterable(SYMBOLS.values()))

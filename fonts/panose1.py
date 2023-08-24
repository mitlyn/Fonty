PANOSE_FEATURES = (
    'family', 'serif_style', 'weight', 'proportion', 'contrast',
    'stroke_variation', 'arm_style', 'letterform', 'midline', 'xheight'
)


def digits_to_features(digits: tuple) -> dict:
    return dict(zip(PANOSE_FEATURES, digits))
BASE_NAME= "chinese"

fs = (15, 30)
LAYOUT = dict(
    IMAGE_WIDTH= (600, 1000),
    IMAGE_HEIGHT= (200, 1000),
    FONT_SIZE= fs,
    CHAR_SPACING = [0, 0, 0, 0, 0, 1, 2, 5, 10],
    LINE_SPACING = (int(1 / 5 * fs[0]), int(15 / 25 * fs[0])),
)

CONTEXT = dict(
    HANZI_PROPORTION = -1,
    LETTER_PROPORTION = (0, 0.1),
    NUMBER_PROPORTION = (0, 0.1),
    PUNCTUATION_PROPORTION = (0, 0.1),
    SPACE_PROPORTION = (0, 0.1),
)
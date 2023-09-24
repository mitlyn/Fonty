import torch.nn as nn
from torch.nn.functional import one_hot
from torch import tensor, zeros, arange, cat, int64, float32

# TODO: Panose Batches
# TODO: CLS Token

# *----------------------------------------------------------------------------* Panose Constants

# Number of categories for each panose feature.
sizes = tensor((6, 12, 4, 14, 16, 10, 11, 12, 16, 14, 8, 10, 7, 11, 14, 14, 7, 13, 10, 8, 9, 16, 6, 13, 10, 10, 10, 10, 10), dtype=int64, device="cuda")

# Mappings of family kinds into unified vocabulary positions.
text = tensor(( 0,  4,  1,  5,  3,  6,  7,  8,  9, 10), dtype=int64, device="cuda")
hand = tensor(( 0, 11,  1,  2, 12,  3, 13, 14, 15, 16), dtype=int64, device="cuda")
deco = tensor(( 0, 17,  1, 18,  3,  4, 19, 20, 21, 22), dtype=int64, device="cuda")
pict = tensor(( 0, 23,  1,  2,  5, 24, 25, 26, 27, 28), dtype=int64, device="cuda")

# *----------------------------------------------------------------------------* Linear Embedder

class LinearEmbedder(nn.Module):
    def __init__(self, filters=64):
        super(LinearEmbedder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(253, filters * 4), # { filters**3 // 4 | filters * 4}
            nn.GELU(),
            # nn.Unflatten(0, (1, filters * 4, filters // 4, filters // 4))
        )

    def forward(self, panose):
        unified = zeros(29, dtype=int64, device="cuda")

        if panose[0] == 3:      # Handwritten
            unified[hand] = panose
        elif panose[0] == 4:    # Decorative
            unified[deco] = panose
        elif panose[0] == 5:    # Pictorial
            unified[pict] = panose
        else:                   # Text or Undefined
            unified[text] = panose

        encoded = cat([
            one_hot(unified[i], num_classes=sizes[i])[2:]
            for i in arange(29)
        ]).to(float32)

        return self.linear(encoded)

# *----------------------------------------------------------------------------* Embedding Notes

""" Unified Vocabulary: unique features, merged number of classes
    0      Family Kind [6]
    1      Weight [12]
    2      Spacing [4]
    3      Contrast [14]
    4      Serif Style & Serif Variant [16]
    5      2.4 Proportion [10]
    6      2.6 Stroke Variation [11]
    7      2.7 Arm Style [12]
    8      2.8 Letterform [16]
    9      2.9 Midline [14]
    10      2.10 X-height [8]
    11      3.2 Tool Kind [10]
    12      3.5 Aspect Ratio [7]
    13      3.7 Topology [11]
    14      3.8 Form [14]
    15      3.9 Finials [14]
    16      3.10 X-ascent [7]
    17      4.2 Class [13]
    18      4.4 Aspect [10]
    19      4.7 Treatment [8]
    20      4.8 Lining [9]
    21      4.9 Topology [16]
    22      4.10 Range of Characters [6]
    23      5.2 Kind [13]
    24      5.6 Aspect Ratio of Character 94 [10]
    25      5.7 Aspect Ratio of Character 119 [10]
    26      5.8 Aspect Ratio of Character 157 [10]
    27      5.9 Aspect Ratio of Character 163 [10]
    28      5.10 Aspect Ratio of Character 211 [10]
"""

""" Text Kind -> Index
    1. Family Kind [0]
    2. Serif Style [4]
    3. Weight   [1]
    4. Proportion [5]
    5. Contrast [3]
    6. Stroke Variation [6]
    7. Arm Style [7]
    8. Letterform [8]
    9. Midline [9]
    10. X-height [10]
"""

""" Handwritten Kind -> Index
    1. Family Kind [0]
    2. Tool Kind [11]
    3. Weight [1]
    4. Spacing [2]
    5. Aspect Ratio [12]
    6. Contrast [3]
    7. Topology [13]
    8. Form [14]
    9. Finials [15]
    10. X-ascent [16]
"""

""" Decorative Kind -> Index
    1. Family Kind [0]
    2. Class [17]
    3. Weight [1]
    4. Aspect [18]
    5. Contrast [3]
    6. Serif Variant [4]
    7. Treatment [19]
    8. Lining [20]
    9. Topology [21]
    10. Range of Characters [22]
"""

""" Pictorial (Symbol) Kind -> Index
    1. Family Kind [0]
    2. Kind [23]
    3. Weight [1] -X-
    4. Spacing [2]
    5. Aspect Ratio & Contrast [5] -X-
    6. Aspect Ratio of Character 94 [24]
    7. Aspect Ratio of Character 119 [25]
    8. Aspect Ratio of Character 157 [26]
    9. Aspect Ratio of Character 163 [27]
    10. Aspect Ratio of Character 211 [28]
"""
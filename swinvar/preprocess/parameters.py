#           0    1    2    3    4    5    6    7    8    9   10   11  12  13    14  15   16    17
# CHANNEL = ['A', 'C', 'G', 'T', 'I', 'D', '*', 'a', 'c', 'g','t', 'i','d','#']
#           0    1    2    3    4     5     6     7     8    9    10   11  12   13     14   15   16    17
CHANNEL = ['A', 'C', 'G', 'T', 'I', 'M_I', 'D', 'M_D', '*', 'a', 'c', 'g','t', 'i', 'm_i','d', 'm_d','#']
CHANNEL_SIZE = len(CHANNEL)
BASE2INDEX = dict(zip(CHANNEL, range(CHANNEL_SIZE)))

Q_MAX = 60
Q0 = 20

VARIANT = [
    "A",
    "C",
    "G",
    "T",
    "Insert",
    "Deletion"
]
VARIANT_SIZE = len(VARIANT)
VARIANT_LABELS = dict(zip(VARIANT, range(VARIANT_SIZE)))

GENOTYPE_LABELS = {
    "hom_ref": 0, # 0/0
    "het": 1, # 0/1, 1/0
    "hom_alt": 2, # 1/1
    "het_alt": 3, # 1/2, 2/1
}
GENOTYPE_SIZE = len(GENOTYPE_LABELS)

min_mapping_quality = 0
min_base_quality = 0
min_freq = 0.01
flank_size = 21
windows_size = 2 * flank_size + 1

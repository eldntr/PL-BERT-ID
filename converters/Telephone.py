
from singleton_decorator import singleton

import re

@singleton
class Telephone:
    """
    Steps:
    - 1 Convert to lowercase and replace parentheses with dashes
    - 2 Convert each character in the token
    - 3 Remove multiple "sil"'s in a row. Also remove "sil" at the start.
    - 4 Replace subsequent "o"s with "hundred" or "thousand" where applicable

    Note:
    Telephone contains 0-9, "-", a-z, A-Z, " ", "(", ")"
    1 case with dots too: 527-28479 U.S. -> five two seven sil two eight four seven nine
    2 cases with commas too: 116-20, RCA, -> one one six sil two o sil r c a
                             2 1943-1990, -> two sil one nine four three sil one nine nine o
    Data is not 100% accurate: 15-16 OCTOBER 1987 -> one five sil one six sil october sil one nine eight seven

    Missed cases:
    Difference between abbreviations and words:
    "53-8 FNB MATIES" -> "five three sil eight sil f n b sil maties"
              instead of "five three sil eight sil f n b sil m a t i e s"
    """
    def __init__(self):
        super().__init__()
        # Translation dict
        self.trans_dict = {
            " ": "jeda",
            "-": "jeda",

            "x": "sambungan",

            "0": "nol",
            "1": "satu",
            "2": "dua",
            "3": "tiga",
            "4": "empat",
            "5": "lima",
            "6": "enam",
            "7": "tujuh",
            "8": "delapan",
            "9": "sembilan",
        }
        # Regex to filter out parentheses
        self.filter_regex = re.compile(r"[()]")

    def convert(self, token: str) -> str:
        # 1 Convert to lowercase and replace parentheses with dashes
        token = self.filter_regex.sub("-", token.lower())

        # 2 Convert list of characters
        result_list = [self.trans_dict[c] if c in self.trans_dict else c for c in token]

        # 3 Remove multiple "jeda"'s in a row. Also remove "jeda" at the start.
        result_list = [section for i, section in enumerate(result_list) if section != "jeda" or (i - 1 >= 0 and result_list[i - 1] != "jeda")]

        # 4 Iterate over result_list and replace multiple "o"s in a row with "hundred" or "thousand", 
        # but only if preceded with something other than "o" or "jeda", and if succeeded with "jeda" or the end of the list.
        i = 0
        while i < len(result_list):
            offset = 0
            while i + offset < len(result_list) and result_list[i + offset] == "nol":
                offset += 1

            if (i + offset >= len(result_list) or result_list[i + offset] == "jeda") and (i - 1 < 0 or result_list[i - 1] not in ("nol", "jeda")) and offset in (2, 3):
                result_list[i : offset + i] = ["ratus"] if offset == 2 else ["ribu"]

            i += 1

        return " ".join(result_list)

    def convert(self, token: str) -> str:
        # 1 Convert to lowercase and replace parentheses with dashes
        token = self.filter_regex.sub("-", token.lower())

        # 2 Convert list of characters
        result_list = [self.trans_dict[c] if c in self.trans_dict else c for c in token]

        # 3 Remove multiple "jeda"'s in a row. Also remove "jeda" at the start.
        result_list = [section for i, section in enumerate(result_list) if section != "jeda" or (i - 1 >= 0 and result_list[i - 1] != "jeda")]

        # 4 Iterate over result_list and replace multiple "o"s in a row with "hundred" or "thousand", 
        # but only if preceded with something other than "o" or "jeda", and if succeeded with "jeda" or the end of the list.
        i = 0
        while i < len(result_list):
            offset = 0
            while i + offset < len(result_list) and result_list[i + offset] == "nol":
                offset += 1

            if (i + offset >= len(result_list) or result_list[i + offset] == "jeda") and (i - 1 < 0 or result_list[i - 1] not in ("nol", "jeda")) and offset in (2, 3):
                result_list[i : offset + i] = ["ratus"] if offset == 2 else ["ribu"]

            i += 1

        return " ".join(result_list)

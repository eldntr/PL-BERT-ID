
from singleton_decorator import singleton

import re

@singleton
class Digit:
    """
    Steps:
    - 1 Filter out anything that isn't a digit
    - 2 Check for special case
    - 3 Convert each digit to text
    - 4 Space out the text

    Special Cases:
    007 -> double o 7
    while 003 -> o o 3
    """
    def __init__(self):
        super().__init__()
        # Regex used to filter out non digits
        self.filter_regex = re.compile("[^0-9]")
        # Translation dict to convert digits to text
        self.trans_dict = {
            "0": "nol",
            "1": "satu",
            "2": "dua",
            "3": "tiga",
            "4": "empat",
            "5": "lima",
            "6": "enam",
            "7": "tujuh",
            "8": "delapan",
            "9": "sembilan"
        }

    def convert(self, token: str) -> str:
        # 1 Filter out anything that isn't a digit
        token = self.filter_regex.sub("", token)
        # 2 Check for special case
        if token == "007":
            return "nol nol tujuh"
        # 3 & 4 Convert each digit to text and space out the text
        token = " ".join([self.trans_dict[c] for c in token])
        return token


from singleton_decorator import singleton

import re

from .Cardinal import Cardinal

@singleton
class Fraction:
    """
    Steps:
    - 1 Filter out commas
    - 2 Check whether we the input consists for example ½
    - 3 If it does, get the text representing the unicode character
    - 4 If it does, check whether there are remaining values (eg in the case of 8 ½)
    - 5 If it does have remaining values, convert those remaining values to Cardinal style and prepend them
    - 6 Try to match values of the format .../...
    - 7 Get the numerator and denominator of the match.
    - 8 Strip the numerator and denominator of spaces
    - 9 Turn the numerator into cardinal style
    - 10 Test the denominator for edge cases such as "1", "2" and "4"
    - 11 If no such edge cases apply, convert the denominator to cardinal style, 
         and replace the last word with a word in the -th or -ths style.
    - 12 Get the potential remaining values (eg in 2 1/2)
    - 13 Turn these values to cardinal, prepend them to the result, and potentially changing "one" to "a".

    Special Cases:
    ½, ¼, ¾, ⅔, ⅛, ⅞, ⅝, etc.
    1½ -> one and a half
    ½  -> one half
    8 1/2 -> 8 and a half
    1/4 -> one quarter
    4/1 -> four over one
    100 000/24 -> one hundred thousand twenty fourths

    Note:
    Always has either a "x y/z", "x/y", "½", or "8 ½"
    """
    def __init__(self):
        super().__init__()
        # Regex to filter out commas
        self.filter_regex = re.compile(",")
        # Regex to filter out spaces
        self.space_filter_regex = re.compile(" ")
        # Translation dict for special cases
        self.trans_dict = {
            "½": {
                "prepended": "",
                "single": "setengah",
                "text": ""
                },
            "⅓": {
                "prepended": "se",
                "single": "satu",
                "text": "pertiga"
                },
            "⅔": {
                "prepended": "dua",
                "single": "dua",
                "text": "pertiga"
                },
            "¼": {
                "prepended": "se",
                "single": "satu",
                "text": "perempat"
                },
            "¾": {
                "prepended": "tiga",
                "single": "tiga",
                "text": "perempat"
                },
            "⅕": {
                "prepended": "se",
                "single": "satu",
                "text": "perlima"
                },
            "⅖": {
                "prepended": "dua",
                "single": "dua",
                "text": "perlima"
                },
            "⅗": {
                "prepended": "tiga",
                "single": "tiga",
                "text": "perlima"
                },
            "⅘": {
                "prepended": "empat",
                "single": "empat",
                "text": "perlima"
                },
            "⅙": {
                "prepended": "se",
                "single": "satu",
                "text": "perenam"
                },
            "⅚": {
                "prepended": "lima",
                "single": "lima",
                "text": "perenam"
                },
            "⅐": {
                "prepended": "se",
                "single": "satu",
                "text": "pertujuh"
                },
            "⅛": {
                "prepended": "se",
                "single": "satu",
                "text": "perdelapan"
                },
            "⅜": {
                "prepended": "tiga",
                "single": "tiga",
                "text": "perdelapan"
                },
            "⅝": {
                "prepended": "lima",
                "single": "lima",
                "text": "perdelapan"
                },
            "⅞": {
                "prepended": "tujuh",
                "single": "tujuh",
                "text": "perdelapan"
                },
            "⅑": {
                "prepended": "se",
                "single": "satu",
                "text": "persembilan"
                },
            "⅒": {
                "prepended": "se",
                "single": "satu",
                "text": "persepuluh"
                }
        }
        # Regex to check for special case
        self.special_regex = re.compile(f"({'|'.join(self.trans_dict.keys())})")
        self.cardinal = Cardinal()

        # Regex for .../...
        # The simpler version of this regex does not allow for "100 000/24" to be seen as "100000/24"
        #self.slash_regex = re.compile(r"(-?\d+) */ *(-?\d+)")
        self.slash_regex = re.compile(r"(-?\d{1,3}( \d{3})+|-?\d+) *\/ *(-?\d{1,3}( \d{3})+|-?\d+)")

        # Translation from Cardinal style to Ordinal style
        self.trans_denominator = {
            "nol": "nol",
            "satu": "satu",
            "dua": "dua",
            "tiga": "tiga",
            "empat": "empat",
            "lima": "lima",
            "enam": "enam",
            "tujuh": "tujuh",
            "delapan": "delapan",
            "sembilan": "sembilan",

            "sepuluh": "sepuluh",
            "dua puluh": "dua puluh",
            "tiga puluh": "tiga puluh",
            "empat puluh": "empat puluh",
            "lima puluh": "lima puluh",
            "enam puluh": "enam puluh",
            "tujuh puluh": "tujuh puluh",
            "delapan puluh": "delapan puluh",
            "sembilan puluh": "sembilan puluh",

            "sebelas": "sebelas",
            "dua belas": "dua belas",
            "tiga belas": "tiga belas",
            "empat belas": "empat belas",
            "lima belas": "lima belas",
            "enam belas": "enam belas",
            "tujuh belas": "tujuh belas",
            "delapan belas": "delapan belas",
            "sembilan belas": "sembilan belas",

            "ratus": "ratus",
            "ribu": "ribu",
            "juta": "juta",
            "miliar": "miliar",
            "triliun": "triliun",
            "kuadriliun": "kuadriliun",
            "kuintiliun": "kuintiliun",
            "sekstiliun": "sekstiliun",
            "septiliun": "septiliun",
            "oktiliun": "oktiliun",
            "undesiliun": "undesiliun",
            "tredesiliun": "tredesiliun",
            "kuatuordesiliun": "kuatuordesiliun",
            "kuindesiliun": "kuindesiliun",
            "seksdesiliun": "seksdesiliun",
            "septendesiliun": "septendesiliun",
            "oktodesiliun": "oktodesiliun",
            "novemdesiliun": "novemdesiliun",
            "vigintiliun": "vigintiliun"
        }

        # Translation dict for edge cases
        self.edge_dict = {
            "1": {
                "singular": "per satu",
                "plural": "per satu"
                },
            "2": {
                "singular": "setengah",
                "plural": "setengah"
                },
            "4":{
                "singular": "seperempat",
                "plural": "perempat"
                }
        }

    def convert(self, token: str) -> str:
        # 1 Filter commas and dots, but keep spaces
        token = self.filter_regex.sub("", token)
        # 2 Check for special unicode case
        match = self.special_regex.search(token)
        if match:
            # 3 Get fraction match from first group
            frac = match.group(1)
            frac_dict = self.trans_dict[frac]

            # 4 Check whether remainder contains a number, eg in "1½"
            remainder = self.special_regex.sub("", token)
            # 5 If it does, convert it using the Cardinal convertion: "1202" -> "one thousand two hundred two"
            if remainder:
                prefix = self.cardinal.convert(remainder)
                result = f"{prefix} {frac_dict['prepended']} {frac_dict['text']}"
            else:
                result = f"{frac_dict['single']} {frac_dict['text']}"
        
        else:
            # 6 Match .../... as two groups
            match = self.slash_regex.search(token)
            if match:
                # 7 Get the numerator and denominators from the groups
                numerator = match.group(1)
                denominator = match.group(3)
                
                # 8 Strip the numerator and denominator of spaces
                numerator = self.space_filter_regex.sub("", numerator)
                denominator = self.space_filter_regex.sub("", denominator)

                # 9 The numerator is a number in cardinal style
                numerator_text = self.cardinal.convert(numerator)

                # 10 We have some edge cases to deal with
                if denominator in self.edge_dict:
                    # Apply edge cases
                    result = f"{numerator_text} {self.edge_dict[denominator][('singular' if abs(int(numerator)) == 1 else 'plural')]}"
                
                else:
                    # 11 Convert the denominator to cardinal style, and convert the last word to
                    # the denominator style using self.trans_denominator.
                    denominator_text_list = self.cardinal.convert(denominator).split(" ")
                    denominator_text_list[-1] = self.trans_denominator[denominator_text_list[-1]]
                    # Potentially add "s" if the numerator is larger than 1.
                    # eg ninth -> ninths 
                    if abs(int(numerator)) != 1:
                        denominator_text_list[-1] += ""
                    denominator_text = " ".join(denominator_text_list)
                    result = f"{numerator_text} per {denominator_text}"
                
                # 12 Get remaining values
                remainder = self.slash_regex.sub("", token)
                if remainder:
                    # 13 Transform remaining values to cardinal
                    remainder_text = self.cardinal.convert(remainder)
                    # Potentially transform "one" to "a" if possible
                    result_list = result.split()
                    if result_list[0] == "satu":
                        result_list[0] = "se"
                    # Prepend the remaining values in cardinal style
                    result = f"{remainder_text} {' '.join(result_list)}"
            
            else:
                # Unhandled case. Should not occur if token really is of the FRACTION class
                result = token

        return result

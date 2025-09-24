
from singleton_decorator import singleton

import re

from .Roman import Roman
from .Cardinal import Cardinal

@singleton
class Ordinal:
    """
    Steps:
    - 1 Filter out commas, spaces and ordinal indicators
    - 2 Check for Roman Numeral, and convert to string integer if so.
    - 3 If so, set prefix to "the", and suffix to "'s" if the roman numeral ends with "s"
    - 4 If not, potentially remove ordinal suffixes ("th", "nd", "st" or "rd", with a potential "s" at the end)
    - 5 Convert the remaining stringed integer to Cardinal, and replace the final word with a word in the ordinal style.
    - 6 Apply pre- and/or suffixes

    Edge Cases:
    II -> (sometimes) second
    II -> (sometimes) the second
    II's -> (the) second's
    
    Note:
    Values are always:
    - Roman numerals (including dots or suffixed with 's)
      - Potentially suffixed by these: "th", "nd", "st" or "rd", with a potential "s" at the end, and potentially capitalized.
    - Numbers (potentially commas and/or spaces)
      - Potentially suffixed by these: "th", "nd", "st" or "rd", with a potential "s" at the end, and potentially capitalized.
    - Numbers + ª or º (Ordinal indicators)

    Missed Cases:
    When input is not of the aforementioned forms
    Difference between edge case 1 and edge case 2. The prefix "the" is always prepended when there is a roman numeral.
    """
    def __init__(self):
        super().__init__()
        # Regex to filter out commas, spaces and ordinal indicators
        self.filter_regex = re.compile(r"[, ºª]")
        # Regex to detect the standard cases
        self.standard_case_regex = re.compile(r"(?i)(\d+)(th|nd|st|rd)(s?)")
        # Roman conversion and detection of roman numeral cases
        self.roman = Roman()
        # Cardinal conversion
        self.cardinal = Cardinal()

        # Translation from Cardinal style to Ordinal style
        self.trans_denominator = {
            "nol": "kenol",
            "satu": "pertama",
            "dua": "kedua",
            "tiga": "ketiga",
            "empat": "keempat",
            "lima": "kelima",
            "enam": "keenam",
            "tujuh": "ketujuh",
            "delapan": "kedelapan",
            "sembilan": "kesembilan",

            "sepuluh": "kesepuluh",
            "dua puluh": "kedua puluh",
            "tiga puluh": "ketiga puluh",
            "empat puluh": "keempat puluh",
            "lima puluh": "kelima puluh",
            "enam puluh": "keenam puluh",
            "tujuh puluh": "ketujuh puluh",
            "delapan puluh": "kedelapan puluh",
            "sembilan puluh": "kesembilan puluh",

            "sebelas": "kesebelas",
            "dua belas": "kedua belas",
            "tiga belas": "ketiga belas",
            "empat belas": "keempat belas",
            "lima belas": "kelima belas",
            "enam belas": "keenam belas",
            "tujuh belas": "ketujuh belas",
            "delapan belas": "kedelapan belas",
            "sembilan belas": "kesembilan belas",

            "seratus": "keseratus",
            "seribu": "keseribu",
            "sejuta": "kesejuta",
            "miliar": "kemiliar",
            "triliun": "ketriliun",
            "kuadriliun": "kekuadriliun",
            "kuintiliun": "kekuintiliun",
            "sekstiliun": "kesekstiliun",
            "septiliun": "keseptiliun",
            "oktiliun": "keoktiliun",
            "undesiliun": "keundesiliun",
            "tredesiliun": "ketredesiliun",
            "kuatuordesiliun": "kekuatuordesiliun",
            "kuindesiliun": "kekuindesiliun",
            "seksdesiliun": "keseksdesiliun",
            "septendesiliun": "keseptendesiliun",
            "oktodesiliun": "keoktodesiliun",
            "novemdesiliun": "kenovemdesiliun",
            "vigintiliun": "kevigintiliun"
        }
    
    def convert(self, token: str) -> str:
        
        # 1 Filter out commas, spaces and ordinal indicators
        token = self.filter_regex.sub("", token)

        prefix = ""
        suffix = ""
        # 2 Check if the token is a roman numeral. 
        # If it is, convert token to a string of the integer the roman numeral represents.
        # Furthermore, update the suffix with 's if applicable
        if self.roman.check_if_roman(token):
            # 3 Update token, and set suffix and prefix
            if not token.endswith(("th", "nd", "st", "rd")):
                prefix = "the"
            token, suffix = self.roman.convert(token)
        
        else:
            # 4 Otherwise, we should be dealing with Num + "th", "nd", "st" or "rd".
            match = self.standard_case_regex.fullmatch(token)
            if match:
                # Set token to the number to convert, and add suffix "s" if applicable
                token = match.group(1)
                suffix = match.group(3)
        
        # 5 Token should now be a string representing some integer
        # Convert the number to cardinal style, and convert the last word to
        # the ordinal style using self.trans_denominator.
        number_text_list = self.cardinal.convert(token).split(" ")
        number_text_list[-1] = self.trans_denominator[number_text_list[-1]]
        result = " ".join(number_text_list)

        # 6 Apply pre- and suffixes, if applicable
        if prefix:
            result = f"{prefix} {result}"
        if suffix:
            result = f"{result}{suffix}"

        return result

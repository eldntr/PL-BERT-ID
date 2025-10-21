from singleton_decorator import singleton
import re
from .Cardinal import Cardinal

@singleton
class Fraction:
    """
    Class ini mengkonversi pecahan ke dalam bentuk terucap Bahasa Indonesia.
    
    Logika utama:
    - 1/2 -> "setengah"
    - 1/4 -> "seperempat"
    - 1/N -> "seper" + cardinal(N) (misal: "sepertiga", "seperdelapan")
    - M/N -> cardinal(M) + " per" + cardinal(N) (misal: "dua pertiga")
    - X Y/Z -> cardinal(X) + (hasil konversi Y/Z) (misal: "delapan setengah")
    """
    def __init__(self):
        super().__init__()
        # Regex
        self.filter_regex = re.compile(",")
        self.space_filter_regex = re.compile(" ")
        
        # Dict untuk karakter unicode pecahan
        self.trans_dict = {
            "½": "setengah",
            "⅓": "sepertiga",
            "⅔": "dua pertiga",
            "¼": "seperempat",
            "¾": "tiga perempat",
            "⅕": "seperlima",
            "⅖": "dua perlima",
            "⅗": "tiga perlima",
            "⅘": "empat perlima",
            "⅙": "seperenam",
            "⅚": "lima perenam",
            "⅛": "seperdelapan",
            "⅜": "tiga perdelapan",
            "⅝": "lima perdelapan",
            "⅞": "tujuh perdelapan",
        }
        
        # Regex untuk .../...
        self.slash_regex = re.compile(r"(-?\d{1,3}( \d{3})+|-?\d+) *\/ *(-?\d{1,3}( \d{3})+|-?\d+)")
        
        # Regex untuk karakter unicode (di-build dari dict di atas)
        self.special_regex = re.compile(f"({'|'.join(self.trans_dict.keys())})")
        
        # Konverter Cardinal
        self.cardinal = Cardinal()

        # Dict 'trans_denominator' dan 'edge_dict' (versi Inggris)
        # tidak digunakan untuk logika Bahasa Indonesia.

    def convert(self, token: str) -> str:
        # 1 Filter koma
        token = self.filter_regex.sub("", token)
        
        # 2 Cek kasus spesial unicode (misal: ½)
        match = self.special_regex.search(token)
        if match:
            # 3 Ambil teks pecahan (misal: "setengah")
            frac = match.group(1)
            frac_text = self.trans_dict[frac]

            # 4 Cek sisa angka (misal: "8" dari "8 ½")
            remainder = self.special_regex.sub("", token).strip()
            
            # 5 Jika ada sisa, konversi ke cardinal dan gabungkan
            if remainder:
                prefix = self.cardinal.convert(remainder)
                # "delapan" + "setengah" -> "delapan setengah"
                result = f"{prefix} {frac_text}"
            else:
                # Jika tidak, kembalikan teks pecahan
                result = frac_text
        
        else:
            # 6 Cek format .../...
            match = self.slash_regex.search(token)
            if match:
                # 7 Ambil pembilang (numerator) dan penyebut (denominator)
                numerator = match.group(1)
                denominator = match.group(3)
                
                # 8 Bersihkan spasi
                numerator = self.space_filter_regex.sub("", numerator)
                denominator = self.space_filter_regex.sub("", denominator)

                # 9 Konversi pembilang ke cardinal
                numerator_text = self.cardinal.convert(numerator)
                
                # 10.A Kasus spesial "1/2" (setengah)
                if numerator == "1" and denominator == "2":
                    result = "setengah"
                
                # 10.B Kasus spesial "1/4" (seperempat)
                elif numerator == "1" and denominator == "4":
                    result = "seperempat"
                    
                # 11 Logika umum "per"
                else:
                    denominator_cardinal = self.cardinal.convert(denominator)
                    
                    # Jika pembilang adalah 1 (misal: 1/3, 1/8)
                    if numerator == "1":
                        # Gunakan awalan "seper-"
                        # "seper" + "tiga" -> "sepertiga"
                        # "seper" + "delapan" -> "seperdelapan"
                        # "seper" + "seratus" -> "seperseratus"
                        result = f"seper{denominator_cardinal}"
                    
                    # Jika pembilang > 1 (misal: 2/3, 5/8)
                    else:
                        # Gunakan "per"
                        # "dua" + "per" + "tiga" -> "dua pertiga"
                        result = f"{numerator_text} per{denominator_cardinal}"
                
                # 12 Cek sisa angka (misal: "8" dari "8 1/2")
                remainder = self.slash_regex.sub("", token).strip()
                if remainder:
                    # 13 Konversi sisa ke cardinal
                    remainder_text = self.cardinal.convert(remainder)
                    
                    # Gabungkan (tanpa "and", misal: "delapan setengah")
                    result = f"{remainder_text} {result}"
            
            else:
                # Kasus tidak terduga
                result = token

        return result
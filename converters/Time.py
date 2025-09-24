
from singleton_decorator import singleton

import re

from .Cardinal import Cardinal

@singleton
class Time:
    """
    Steps:
    - 1 Strip the token to remove extra spaces
    - 2 Try to match "hh.mm (pm)"
      - 2.1 Add the hour, potentially roughly modulo 12
      - 2.2 Add the minute if it exists and is not 00
      - 2.3 Add "hundred" or "o'clock" if minute is not added
      - 2.4 Prepend the suffix, eg pm
    - 3 Otherwise, try to match "(hh:)mm:ss(.ms) (pm)"
      - 3.1 If hour, add it as cardinal and add "hour" with proper plurality
      - 3.2 If minute, add it as cardinal and add "minute" with proper plurality
      - 3.3 If seconds, add "and" if seconds is the last number, add seconds as cardinal, and "second" with proper plurality
      - 3.4 If milliseconds, add "and", milliseconds as cardinal, and "millisecond" with proper plurality
      - 3.5 If suffix, prepend the suffix with padded spaces
    - 4 Otherwise, try to match "xxH", eg. "PM3"
      - 2.1 Add the hour, potentially roughly modulo 12
    
    Edge case:
    "PM2" -> "two p m"
    """
    def __init__(self):
        super().__init__()

        # Regex to filter out dots
        self.filter_regex = re.compile(r"[. ]")
        # Regex to detect time in the form xx:yy (pm)
        self.time_regex = re.compile(r"^(?P<hour>\d{1,2}) *((?::|.) *(?P<minute>\d{1,2}))? *(?P<suffix>[a-zA-Z\. ]*)$", flags=re.I)
        # Regex to detect time in the form xx:yy:zz
        self.full_time_regex = re.compile(r"^(?:(?P<hour>\d{1,2}) *:)? *(?P<minute>\d{1,2})(?: *: *(?P<seconds>\d{1,2})(?: *. *(?P<milliseconds>\d{1,2}))?)? *(?P<suffix>[a-zA-Z\. ]*)$", flags=re.I)
        # Regex to detect time in the form AM5 and PM7
        self.ampm_time_regex = re.compile(r"^(?P<suffix>[a-zA-Z\. ]*)(?P<hour>\d{1,2})", flags=re.I)

        # Cardinal conversion
        self.cardinal = Cardinal()

    def convert(self, token: str) -> str:

        # 1 Strip the token to remove extra spaces
        token = token.strip()

        result_list = []

        # 2 Try to match "hh.mm (pm)"
        match = self.time_regex.match(token)
        if match:
            # Extract hour, minute and suffix from the match
            hour, minute, suffix = match.group("hour"), match.group("minute"), match.group("suffix")

            # 2.1 Add the hour
            result_list.append(self.cardinal.convert(hour))

            # 2.2 Add the minute if it exists and is not just zeros
            if minute and minute != "00":
                result_list.append(self.cardinal.convert(minute))
            # 2.3 If there is no minute, add "tepat"
            elif minute == "00":
                result_list.append("tepat")

            # 2.4 Add the suffix, eg pm
            if suffix:
                result_list += [c for c in suffix.lower() if c not in (" ", ".")]
            
            return " ".join(result_list)

        # 3 Try to match "(hh:)mm:ss(.ms) (pm)"
        match = self.full_time_regex.match(token)
        if match:
            # Extract values from match
            hour, minute, seconds, milliseconds, suffix = match.group("hour"), match.group("minute"), match.group("seconds"), match.group("milliseconds"), match.group("suffix")

            # 3.1 If hour, add it as cardinal and add "jam"
            if hour:
                result_list.append(self.cardinal.convert(hour))
                result_list.append("jam")
            # 3.2 If minute, add it as cardinal and add "menit"
            if minute:
                result_list.append(self.cardinal.convert(minute))
                result_list.append("menit")
            # 3.3 If seconds, add "lewat", add seconds as cardinal, and "detik"
            if seconds:
                result_list.append("lewat")
                result_list.append(self.cardinal.convert(seconds))
                result_list.append("detik")
            # 3.4 If milliseconds, add "lewat", milliseconds as cardinal, and "milidetik"
            if milliseconds:
                if not seconds:
                    result_list.append("lewat")
                result_list.append(self.cardinal.convert(milliseconds))
                result_list.append("milidetik")
            # 3.5 If suffix, add the suffix
            if suffix:
                result_list += [c for c in suffix.lower() if c not in (" ", ".")]
            
            return " ".join(result_list)
        
        # 4 Try to match "xxH", eg. "PM3"
        match = self.ampm_time_regex.match(token)
        if match:
            # Extract values from match
            hour, suffix = match.group("hour"), match.group("suffix")

            result_list.append(self.cardinal.convert(hour))
            result_list += [c for c in suffix.lower() if c not in (" ", ".")]
            return " ".join(result_list)

        return token

    def modulo_hour(self, hour: str) -> str:
        # This function is not used in the Indonesian localization but kept for compatibility.
        # "12pm" -> "twelve p m", while "1pm" -> "one p m"
        if hour == "12":
            return hour
        return str(int(hour) % 12)

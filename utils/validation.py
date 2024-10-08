import re

regex_pattern_price_parsing = r"^.+-\d{1,5}$"

# matches '<=800'
regex_pattern_price_parsing_max_value = r"^<=\d{1,5}$"

regex_match_int = r"^\d{1,5}$"

# Test cases FIXME make real tesst
test_strings = [
    "0-500",
    "12-34",
    "1234-5678",
    "12345-67890",
    "123-4567",
    "1234-5",
    "0-12345",
    "max-800",
    "max: 1200",
    "maximum 1200",
    "0-1200",
]


def extract_numbers(number_string):
    numbers = re.findall(r"\d+", number_string)
    return numbers


def validate_price_filter(filter_values: list[str]) -> list[str]:
    validated_filter_values = []

    for filter in filter_values:
        numbers = extract_numbers(filter)

        validated_filter_values = [*validated_filter_values, *numbers]
    return validated_filter_values

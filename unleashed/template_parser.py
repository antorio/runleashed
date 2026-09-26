import re
from datetime import datetime

template_functions = {
    "timestamp": lambda data: str(int(datetime.now().timestamp())),
    "i": lambda data: data.get("index", False),
    "file": lambda data: data.get("file", False),
    "date": lambda data: datetime.now().strftime("%Y-%m-%d"),
    "time": lambda data: datetime.now().strftime("%H-%M-%S"),
}


PLACEHOLDERS = '{file} {i} {date} {time} {timestamp}'


def unknown_placeholders(text: str):
    """Placeholders in an output template that parse() does not know."""
    return [m for m in re.findall(r"\{([^}]+)\}", text or '') if m not in template_functions]


def parse(text: str, data: dict):
    pattern = r"\{([^}]+)\}"

    matches = re.findall(pattern, text)

    for match in matches:
        fn = template_functions.get(match)
        if fn is None:
            continue            # unknown placeholder: kept as written (it raised KeyError after the render)
        replacement = fn(data)
        if replacement is not False:
            text = text.replace(f"{{{match}}}", replacement)

    return text

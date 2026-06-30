EXPECTED = {
    "name":     "Vishnu S",
    "email":    "senseicoder09@gmail.com",
    "linkedin": "https://www.linkedin.com/in/vishnu-s-42757a310/",
    "github":   "https://github.com/Blackmoon390"
}

# ============================================================
# Author  : Vishnu S
# Email   : senseicoder09@gmail.com
# LinkedIn: https://www.linkedin.com/in/vishnu-s-42757a310/
# GitHub  : https://github.com/Blackmoon390
# ============================================================

# Field name mapping (as they appear in the txt file)
FIELD_MAP = {
    "name":     "Name",
    "email":    "Email",
    "linkedin": "LinkedIn",
    "github":   "GitHub"
}
# data='''# ============================================================
# # Author  : Vishnu S
# # Email   : senseicoder09@gmail.com
# # LinkedIn: https://www.linkedin.com/in/vishnu-s-42757a310/
# # GitHub  : https://github.com/Blackmoon390
# # ============================================================'''


def verify(filepath: str = "ced.txt") -> bool:
    parsed = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line and not line.strip().startswith("="):
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()

                for field, label in FIELD_MAP.items():
                    if key == label.lower():
                        parsed[field] = value
                        break

    return all(
        parsed.get(field, "").lower() == expected.lower()
        for field, expected in EXPECTED.items()
    )


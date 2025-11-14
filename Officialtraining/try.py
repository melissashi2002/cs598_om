import string
import re
def normalize_token(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

def main():
    print(normalize_token("Hello,world!"))

if __name__ == "__main__":
    main()
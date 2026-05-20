# ANSI COLOR CODES
BOLD_RED = "\033[1;31m"
BOLD_TEAL = "\033[1;36m"
RESET = "\033[0m"

def printf(msg, type="info"):
    if type != "info":
        print(f"\n{BOLD_RED}[ERR] {msg}{RESET}")
    else:
        print(f"\n{BOLD_TEAL}{msg}...{RESET}")

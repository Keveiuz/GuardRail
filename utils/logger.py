class Logger():
    def __init__(self):
        # ========== 颜色定义 ==========
        self.GREEN = "\033[92m"    # INFO
        self.YELLOW = "\033[93m"   # WARNING
        self.RED = "\033[91m"      # ERROR
        self.BLUE = "\033[94m"     # DEBUG
        self.MAGENTA = "\033[95m"  # STAT
        self.CYAN = "\033[96m"     # SUCCESS
        self.GRAY = "\033[90m"     # NOTE
        self.CRITICAL = "\033[41;97m"  # 红底白字
        self.RESET = "\033[0m"     # 重置颜色

    def info(self, msg): print(f"{self.GREEN}[INFO]{self.RESET} {msg}")
    def warn(self, msg): print(f"{self.YELLOW}[WARNING]{self.RESET} {msg}")
    def error(self, msg): print(f"{self.RED}[ERROR]{self.RESET} {msg}")
    def debug(self, msg): print(f"{self.BLUE}[DEBUG]{self.RESET} {msg}")
    def stat(self, msg): print(f"{self.MAGENTA}[STAT]{self.RESET} {msg}")
    def success(self, msg): print(f"{self.CYAN}[SUCCESS]{self.RESET} {msg}")
    def note(self, msg): print(f"{self.GRAY}[NOTE]{self.RESET} {msg}")
    def critical(self, msg): print(f"{self.CRITICAL}[CRITICAL]{self.RESET} {msg}")
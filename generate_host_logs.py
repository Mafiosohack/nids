"""
generate_host_logs.py — Linux auth.log Attack Simulator  (Phase 2)

Writes a realistic /var/log/auth.log-style file that tells the host side of an
intrusion story. Drop the output into host_logs/ and host_log_sensor.py picks
it up automatically, posting alerts into the same unified kill chain as the
network and cloud sensors.

Usage:
    python generate_host_logs.py            # full breach → root → persistence chain
    python generate_host_logs.py brute      # failed-login burst only
    python generate_host_logs.py root       # login + privilege escalation
    python generate_host_logs.py persist    # account creation / sudo group

Output: host_logs/<scenario>.log
"""

import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path("host_logs")

HOST     = "web-prod-01"          # the compromised server
ATTACKER = "203.0.113.66"         # attacker source IP (unifies with network side)
VICTIM_USER = "deploy"            # the account whose password was cracked

# Rolling timestamp so every line is a little later than the previous one.
_clock = datetime.now() - timedelta(seconds=90)


def stamp() -> str:
    """Next syslog-format timestamp: 'Jul 19 07:51:14'."""
    global _clock
    _clock += timedelta(seconds=random.randint(1, 4))
    # %d is zero-padded; real syslog space-pads, but the sensor accepts both.
    return _clock.strftime("%b %d %H:%M:%S")


def line(msg: str) -> str:
    return f"{stamp()} {HOST} {msg}"


# ─────────────────────────────────────────────
#  SCENARIO BUILDERS
# ─────────────────────────────────────────────
def brute_lines(n: int = 8):
    """A burst of failed SSH logins — credential access."""
    out = []
    users = [VICTIM_USER, "admin", "root", "test", "oracle", "postgres"]
    for _ in range(n):
        u = random.choice(users)
        port = random.randint(40000, 60000)
        out.append(line(
            f"sshd[{random.randint(1000,9999)}]: Failed password for "
            f"{'invalid user ' if u in ('test','oracle') else ''}{u} "
            f"from {ATTACKER} port {port} ssh2"))
    return out


def login_lines(user: str = VICTIM_USER):
    """The brute force finally works — confirmed breach."""
    pid = random.randint(1000, 9999)
    return [
        line(f"sshd[{pid}]: Accepted password for {user} from {ATTACKER} "
             f"port {random.randint(40000,60000)} ssh2"),
        line(f"sshd[{pid}]: pam_unix(sshd:session): session opened for user "
             f"{user} by (uid=0)"),
    ]


def privesc_lines(user: str = VICTIM_USER):
    """Escalation to root + reading sensitive files — root access & collection."""
    pid = random.randint(1000, 9999)
    return [
        line(f"sudo:   {user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; "
             f"COMMAND=/bin/bash"),
        line(f"sudo:   {user} : TTY=pts/0 ; PWD=/root ; USER=root ; "
             f"COMMAND=/bin/cat /etc/shadow"),
        line(f"su[{pid}]: pam_unix(su:session): session opened for user root "
             f"by {user}(uid=1001)"),
        line(f"sudo:   {user} : TTY=pts/0 ; PWD=/root ; USER=root ; "
             f"COMMAND=/usr/bin/cat /home/{user}/.ssh/id_rsa"),
    ]


def persist_lines():
    """New backdoor account + added to sudo — persistence."""
    return [
        line("useradd[24001]: new user: name=svc-backup, UID=0, GID=0, "
             "home=/home/svc-backup, shell=/bin/bash"),
        line("usermod[24010]: add 'svc-backup' to group 'sudo'"),
        line("passwd[24012]: password changed for svc-backup"),
    ]


SCENARIOS = {
    "brute":   lambda: brute_lines(9),
    "root":    lambda: login_lines() + privesc_lines(),
    "persist": lambda: login_lines() + persist_lines(),
    "fullchain": lambda: (
        brute_lines(8) + login_lines() + privesc_lines() + persist_lines()
    ),
}


def write_scenario(name: str):
    builder = SCENARIOS.get(name)
    if builder is None:
        print(f"Unknown scenario '{name}'. Choose from: {', '.join(SCENARIOS)}")
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.log"
    lines = builder()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] Wrote {len(lines)} auth-log lines → {path}")
    print(f"    Attacker {ATTACKER} vs host '{HOST}'. "
          f"host_log_sensor.py will detect these within a few seconds.")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "fullchain"
    write_scenario(scenario)

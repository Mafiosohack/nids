"""Response playbook — what to do about an alert, per rule, mapped to MITRE.

The dashboard used to recommend "Block Source IP" for every alert. For a third
of the catalogue that is wrong, and for some it is harmful:

  * BEHAVIOR_UNEXPECTED_OUTBOUND's source is YOUR compromised host calling
    out. Blocking it takes your own server offline and leaves the C2 endpoint
    — the thing to block — reachable from every other host.
  * MITM_ARP_SPOOF's source IP is forged. Blocking it blocks the victim whose
    address was stolen, not the impostor.
  * COLLECTION_STAGING files the alert under the collecting host, i.e. the
    victim.

So the immediate step is written per rule, from the rule's `src_semantics`, and
the longer-term hardening comes from MITRE ATT&CK's own mitigation list for the
rule's technique.

Every M-number below was checked against the technique's page on
attack.mitre.org (2026-10-14). The `action` text is ours: what that mitigation
means concretely for the thing this sensor saw. Techniques MITRE says cannot be
mitigated with preventive controls (T1074) say so instead of inventing one.
"""

from typing import Dict, List, Optional

from .rules import EXTERNAL_SENSOR_STAGES, RULES

ATTACK_BASE = "https://attack.mitre.org"


def technique_url(tid: str) -> str:
    return f"{ATTACK_BASE}/techniques/{tid.replace('.', '/')}/"


def mitigation_url(mid: str) -> str:
    return f"{ATTACK_BASE}/mitigations/{mid}/"


MITIGATION_NAMES: Dict[str, str] = {
    "M1013": "Application Developer Guidance",
    "M1016": "Vulnerability Scanning",
    "M1018": "User Account Management",
    "M1022": "Restrict File and Directory Permissions",
    "M1026": "Privileged Account Management",
    "M1027": "Password Policies",
    "M1028": "Operating System Configuration",
    "M1030": "Network Segmentation",
    "M1031": "Network Intrusion Prevention",
    "M1032": "Multi-factor Authentication",
    "M1034": "Limit Hardware Installation",
    "M1035": "Limit Access to Resource Over Network",
    "M1036": "Account Use Policies",
    "M1037": "Filter Network Traffic",
    "M1038": "Execution Prevention",
    "M1040": "Behavior Prevention on Endpoint",
    "M1041": "Encrypt Sensitive Information",
    "M1042": "Disable or Remove Feature or Program",
    "M1047": "Audit",
    "M1048": "Application Isolation and Sandboxing",
    "M1050": "Exploit Protection",
    "M1051": "Update Software",
    "M1057": "Data Loss Prevention",
}

# technique -> name + the MITRE mitigations most relevant to what this sensor
# detects, each with a concrete action. Not every mitigation MITRE lists is
# repeated (T1059's code signing is not actionable from a network alert); none
# is listed that MITRE does not.
TECHNIQUES: Dict[str, dict] = {
    "T1046": {"name": "Network Service Discovery", "mitigations": [
        ("M1030", "Segment so a host can only reach the services it needs; a scan then maps one segment, not the network."),
        ("M1042", "Turn off services nobody uses. Every open port in the scan is one to justify or close."),
        ("M1031", "If the scanner is not authorized, rate-limit or drop it at the perimeter IPS."),
    ]},
    "T1110": {"name": "Brute Force", "mitigations": [
        ("M1036", "Enforce lockout or exponential back-off on the targeted service (e.g. fail2ban / sshd MaxAuthTries)."),
        ("M1032", "Require MFA on the targeted service so a guessed password is not enough."),
        ("M1027", "Enforce password length and ban known-breached passwords."),
        ("M1018", "Remove or disable default and unused accounts on the target."),
    ]},
    "T1190": {"name": "Exploit Public-Facing Application", "mitigations": [
        ("M1051", "Patch or replace the vulnerable service named in the alert. These are known, public exploits."),
        ("M1042", "If the service is not needed, disable it. Most lab exploits target services nobody should expose."),
        ("M1030", "Move the service behind a segment boundary so only clients that need it can reach it."),
        ("M1048", "Run the service sandboxed or containerised so a successful exploit does not reach the host."),
        ("M1016", "Scan exposed services regularly; this one should have been found before an attacker did."),
    ]},
    "T1059": {"name": "Command and Scripting Interpreter", "mitigations": [
        ("M1038", "Restrict which interpreters can run on the target, and from which parent processes."),
        ("M1042", "Remove interpreters and shells the server does not need."),
        ("M1026", "Make sure the exploited service does not run as root, so a shell it spawns is unprivileged."),
        ("M1040", "Deploy endpoint behaviour blocking to stop a service spawning a shell."),
    ]},
    "T1571": {"name": "Non-Standard Port", "mitigations": [
        ("M1030", "Default-deny egress: servers should only reach the destinations and ports they need."),
        ("M1031", "Alert or block at the IPS on protocol/port mismatch (e.g. no TLS on 443)."),
    ]},
    "T1071": {"name": "Application Layer Protocol", "mitigations": [
        ("M1037", "Block the beacon destination at egress; route outbound web traffic through a proxy that logs it."),
        ("M1031", "Add an IPS signature for the beacon's interval and size pattern."),
    ]},
    "T1499.001": {"name": "Endpoint Denial of Service: OS Exhaustion Flood", "mitigations": [
        ("M1037", "Enable SYN cookies on the target and rate-limit SYNs per source upstream."),
    ]},
    "T1498": {"name": "Network Denial of Service", "mitigations": [
        ("M1037", "Filter upstream (ISP / scrubbing service); a flood that reaches your link has already won."),
    ]},
    "T1498.002": {"name": "Network Denial of Service: Reflection Amplification", "mitigations": [
        ("M1037", "Do not expose open resolvers / NTP / SSDP / memcached to the Internet; filter spoofed sources (BCP 38)."),
    ]},
    "T1078": {"name": "Valid Accounts", "mitigations": [
        ("M1032", "Require MFA, so a stolen or guessed credential is not a login."),
        ("M1027", "Rotate the credential that was used, and any shared with it."),
        ("M1026", "Remove standing admin rights from the account; elevate only when needed."),
        ("M1018", "Disable accounts that are not tied to a current person or service."),
    ]},
    "T1021": {"name": "Remote Services", "mitigations": [
        ("M1035", "Allow SSH/SMB/RDP only from a management subnet or jump host, not host-to-host."),
        ("M1032", "Require MFA for remote administration."),
        ("M1042", "Disable remote services on hosts that do not need them."),
        ("M1047", "Audit which accounts can log in remotely to which hosts."),
    ]},
    "T1041": {"name": "Exfiltration Over C2 Channel", "mitigations": [
        ("M1057", "Apply DLP on egress for the data classes this host holds."),
        ("M1031", "Alert on sustained outbound volume from servers at the IPS."),
    ]},
    "T1074": {"name": "Data Staged", "mitigations": [],
              "no_preventive_control": ("MITRE: this technique cannot be easily mitigated with "
                                         "preventive controls, because it abuses normal system "
                                         "features. Detection and the response below are the control.")},
    "T1552.005": {"name": "Unsecured Credentials: Cloud Instance Metadata API", "mitigations": [
        ("M1042", "Disable the metadata service on instances that do not use it."),
        ("M1037", "Require IMDSv2 (session tokens) and block 169.254.169.254 from containers and proxies."),
        ("M1035", "Limit which processes / workloads can reach the metadata endpoint."),
    ]},
    "T1048": {"name": "Exfiltration Over Alternative Protocol", "mitigations": [
        ("M1037", "Force DNS through your resolvers; block direct outbound DNS (udp/tcp 53) from hosts."),
        ("M1031", "Alert on long, high-entropy query names at the IPS / resolver."),
        ("M1057", "Apply DLP to the data this host holds."),
        ("M1030", "Segment so servers cannot resolve arbitrary external names."),
    ]},
    "T1557.002": {"name": "Adversary-in-the-Middle: ARP Cache Poisoning", "mitigations": [
        ("M1037", "Enable Dynamic ARP Inspection / DHCP snooping on the switch."),
        ("M1035", "Pin static ARP entries for the gateway and critical servers."),
        ("M1041", "Make sure traffic on this segment is encrypted, so an interception yields nothing usable."),
        ("M1042", "Disable gratuitous ARP acceptance on hosts where the OS allows it."),
    ]},
    "T1200": {"name": "Hardware Additions", "mitigations": [
        ("M1035", "Use 802.1X / NAC so an unknown device gets no network access."),
        ("M1034", "Lock down physical ports and unused switch ports."),
    ]},
    "T1548": {"name": "Abuse Elevation Control Mechanism", "mitigations": [
        ("M1026", "Review sudoers / admin group membership on the host."),
        ("M1022", "Audit setuid binaries and world-writable paths on the host."),
        ("M1051", "Patch local privilege-escalation vulnerabilities."),
        ("M1047", "Audit elevation events centrally so the next one is seen in seconds."),
    ]},
    "T1136": {"name": "Create Account", "mitigations": [
        ("M1026", "Restrict who can create accounts; the new account should be deleted if not authorized."),
        ("M1032", "Require MFA so a created account cannot log in on a password alone."),
        ("M1028", "Alert on account creation via OS audit policy."),
    ]},
    "T1005": {"name": "Data from Local System", "mitigations": [
        ("M1057", "Apply DLP to the sensitive files this host holds."),
    ]},
}

# ── Immediate response, per rule ──────────────────────────────────────────────
# {src} and {dst} are filled in by the dashboard. "contain" is the first move;
# "verify" is the check that decides true vs false positive.
_SCAN = {
    "contain": "If {src} is not an authorized scanner, rate-limit or block it at the perimeter. If it is, add it to authorized_scanners in assets.json so its scans are recorded at INFO.",
    "verify": "Is {src} one of yours (vulnerability scanner, monitoring)? A scan alone is not a breach — check whether anything from {src} reached a later stage.",
}
_BRUTE = {
    "contain": "Block {src} at the service ({dst}), and lock or rate-limit the targeted account.",
    "verify": "Check auth logs on {dst} for a SUCCESSFUL login from {src} after the failures. One success turns this into an incident.",
}
_EXPLOIT_IN = {
    "contain": "Block {src}, and isolate {dst} from the network until it is checked: assume the exploit worked.",
    "verify": "On {dst}: new processes, shells or listeners spawned by the exploited service around this time. A shell or outbound-connection alert on {dst} confirms it.",
}
_HOST_CALLING_OUT = {
    "contain": "Isolate {src} — it is YOUR host, and it is the one that is compromised. Block the destination {dst} at egress for every host. Do not block {src} at the perimeter and stop there.",
    "verify": "On {src}: which process owns the connection to {dst} (ss -tnp / netstat -ano)? A shell or unknown binary is a compromise; a recognised updater is a baseline gap.",
}
_SHELL = {
    "contain": "Find which end is yours: if {dst} is your server, isolate it and kill the session. Block {src} if it is not an administrator.",
    "verify": "Is this an admin's SSH session? A matching login in {dst}'s auth log means false positive — mark it so, and deploy host_log_sensor.py so it is suppressed automatically.",
}
_FLOOD = {
    "contain": "Rate-limit {src} upstream. If the source is spoofed or distributed, blocking single IPs will not hold — escalate to the ISP / scrubbing provider.",
    "verify": "Is {dst} actually degraded? A busy but healthy service at this rate may be legitimate load (backups, sync).",
}

RULE_RESPONSE: Dict[str, dict] = {
    **{r: _SCAN for r in ("TCP_NULL_SCAN", "TCP_FIN_SCAN", "TCP_XMAS_SCAN",
                          "TCP_SYNFIN_SCAN", "TCP_SYN_SCAN", "TCP_CONNECT_SCAN",
                          "SCAN_SLOW_RATE", "SCAN_FRAGMENTED")},
    "SCAN_DISTRIBUTED": {
        "contain": "Do not block the listed source alone — a distributed scan uses many. Tighten the perimeter for the scanned ports instead.",
        "verify": "Which ports were swept? Close any that should not be reachable from outside.",
    },
    "BRUTEFORCE_STANDARD_RATE": _BRUTE,
    "BRUTEFORCE_LOW_RATE": {
        **_BRUTE,
        "verify": "Low-and-slow guessing is built to sit under lockout thresholds. Check auth logs on {dst} for the whole window, not just the last few minutes.",
    },
    **{r: _EXPLOIT_IN for r in ("EXPLOIT_VSFTPD_BACKDOOR", "EXPLOIT_SAMBA_USERMAP",
                                "EXPLOIT_DISTCC_CMDEXEC", "EXPLOIT_UNREALIRCD_BACKDOOR",
                                "EXPLOIT_JAVA_RMI_DESERIALIZATION")},
    "EXPLOIT_VSFTPD_BACKDOOR_SHELL": {
        "contain": "The backdoor shell is OPEN. Isolate {dst} now and block {src}. Then take vsftpd 2.3.4 out of service — it is backdoored, not merely vulnerable.",
        "verify": "Port 6200 listening on {dst} is the backdoor itself; there is no benign explanation.",
    },
    "BEHAVIOR_UNEXPECTED_OUTBOUND": _HOST_CALLING_OUT,
    "BEHAVIOR_UNCOMMON_EGRESS_PORT": _HOST_CALLING_OUT,
    "BEHAVIOR_C2_BEACON": _HOST_CALLING_OUT,
    "BEHAVIOR_CLEARTEXT_ON_TLS_PORT": _HOST_CALLING_OUT,
    "BEHAVIOR_NEW_LISTENING_PORT": {
        "contain": "Isolate {src} — this alert is filed under the host that is now listening. Block inbound to the new port.",
        "verify": "On {src}: which process owns the new listener (ss -ltnp)? If it is a service you just deployed, add the port to its listening_ports baseline.",
    },
    "BEHAVIOR_INTERACTIVE_SHELL": _SHELL,
    "FLOOD_SYN": _FLOOD,
    "FLOOD_PACKET_RATE": _FLOOD,
    "FLOOD_UDP_AMPLIFICATION": {
        "contain": "{src} is being used as (or is) a reflector. Close the amplifying service to the Internet; the victim is whoever the source addresses claim to be.",
        "verify": "Is the UDP service on {dst} meant to be reachable from outside at all?",
    },
    "BREACH_SESSION": {
        "contain": "Treat the session as a successful login by an attacker: block {src}, reset the credentials used on {dst}.",
        "verify": "Auth log on {dst}: which account logged in from {src}, and was it expected?",
    },
    "LATERAL_ADMIN_SWEEP": {
        "contain": "Isolate {src} — an internal host touching many hosts' admin ports is usually already compromised. Disable the account it is using.",
        "verify": "Is {src} a management / patching server doing its job? If so, scope its access and record it; if not, it is lateral movement.",
    },
    "EXFIL_BULK_OUTBOUND": {
        "contain": "Cut {src}'s egress to {dst} now; volume already sent cannot be recalled, volume still to come can.",
        "verify": "What is {dst}? A known backup or sync target is a baseline gap; an unknown host is exfiltration.",
    },
    "COLLECTION_STAGING": {
        "contain": "Isolate {src} — the alert is filed under the host collecting the data, which is where the attacker is.",
        "verify": "What did {src} pull, and from where? Large internal pulls by a backup server are normal; by a workstation, not.",
    },
    "CREDS_CLOUD_METADATA": {
        "contain": "Rotate the instance role credentials that {src} could have read, then require IMDSv2.",
        "verify": "Which process on {src} queried the metadata endpoint? SDKs do this; a web app reached via SSRF should not.",
    },
    "EXFIL_DNS_TUNNEL": {
        "contain": "Block {src}'s direct outbound DNS, and sinkhole the domain being queried.",
        "verify": "Long, random-looking subdomains of one parent domain are tunnelling; check whether the domain belongs to a known security / CDN vendor.",
    },
    "MITM_ARP_SPOOF": {
        "contain": "Do NOT block {src}: that address is forged — blocking it cuts off the real host. Find the impostor's MAC (in the details) on the switch port table and shut that port.",
        "verify": "Compare the MAC in the details with assets.json trusted_arp. A NIC swap or failover also changes the MAC.",
    },
    "ASSET_ROGUE_HOST": {
        "contain": "Locate the device by MAC on the switch and quarantine its port until identified.",
        "verify": "New laptop, phone or VM? If authorized, add it to known_hosts in assets.json.",
    },
    "ANOMALY_TRAFFIC_BASELINE": {
        "contain": "No containment on this alone; it is a deviation, not a verdict.",
        "verify": "What changed for {src}? Look for a specific alert from the same host in the same window.",
    },
    "ML_FLOW_ANOMALY": {
        "contain": "No containment on the model's word alone — corroborate first.",
        "verify": "Does a rule-based alert from {src} exist around the same time? The model's false-positive rate is in the Model Comparison panel.",
    },
    "CORRELATED_INCIDENT": {
        "contain": "Contain at the stage the incident reached: the furthest-stage alert from {src} says what to do.",
        "verify": "Open the incident row: one source advancing through stages is rarely benign, but check each member alert.",
    },
    "ACTIVE_INTRUSION": {
        "contain": "Isolate the hosts {src} touched and block {src}. This source got in and then acted inside within one correlation window.",
        "verify": "Work the member alerts from the initial-access one forward; confirm the first one, and the rest follow.",
    },
    # Host / cloud log sensor alert types (their rule_id is the type name).
    "Host Brute Force": _BRUTE,
    "Cloud Brute Force": _BRUTE,
    "SSH Login": {
        "contain": "Nothing yet, if expected. If not: kill the session on {dst} and disable the account.",
        "verify": "Was this login expected — right person, right time, right source?",
    },
    "Root Access": {
        "contain": "Disable the account that elevated and kill its sessions on {dst}.",
        "verify": "Was this sudo / su by an administrator doing planned work?",
    },
    "Persistence": {
        "contain": "Remove the account / cron / service that was added on {dst}, after capturing it for evidence.",
        "verify": "Is there a change ticket for it?",
    },
    "Sensitive File Access": {
        "contain": "Rotate anything in the file that was read (keys, password hashes).",
        "verify": "Which account read it, and does it ever need to?",
    },
}

DEFAULT_RESPONSE = {
    "contain": "Decide whose address {src} is before acting: an outside attacker is blocked; one of your hosts is isolated.",
    "verify": "Check what else {src} did in the same window — a lone alert with no follow-up is weaker evidence.",
}


def _technique_entry(tid: Optional[str]) -> Optional[dict]:
    if not tid:
        return None
    t = TECHNIQUES.get(tid)
    if t is None:
        return {"id": tid, "name": None, "url": technique_url(tid),
                "mitigations": [], "unmapped": True}
    out = {
        "id": tid, "name": t["name"], "url": technique_url(tid),
        "mitigations": [
            {"id": m, "name": MITIGATION_NAMES[m], "url": mitigation_url(m),
             "action": action}
            for m, action in t["mitigations"]
        ],
    }
    if t.get("no_preventive_control"):
        out["no_preventive_control"] = t["no_preventive_control"]
    return out


def playbook() -> dict:
    """Everything the dashboard needs to render advice, loaded once per session.

    Keyed by rule_id. Host/cloud sensor alerts carry their type name as rule_id,
    so those are included under that name.
    """
    rules: Dict[str, dict] = {}
    for r in RULES.values():
        rules[r.rule_id] = {
            "title": r.title, "stage": r.stage, "tactic": r.tactic,
            "src_semantics": r.src_semantics,
            "technique": _technique_entry(r.technique),
            "response": RULE_RESPONSE.get(r.rule_id, DEFAULT_RESPONSE),
        }
    for name, (stage, tactic, tid) in EXTERNAL_SENSOR_STAGES.items():
        rules[name] = {
            "title": name, "stage": stage, "tactic": tactic,
            "src_semantics": "actor",
            "technique": _technique_entry(tid),
            "response": RULE_RESPONSE.get(name, DEFAULT_RESPONSE),
        }
    return {"rules": rules, "default_response": DEFAULT_RESPONSE,
            "source": "MITRE ATT&CK mitigations, attack.mitre.org; actions are this project's."}


def unmapped_techniques() -> List[str]:
    """Techniques referenced by a rule but missing here. Tests assert empty."""
    used = {r.technique for r in RULES.values() if r.technique}
    used |= {t for (_, _, t) in EXTERNAL_SENSOR_STAGES.values()}
    return sorted(used - set(TECHNIQUES))

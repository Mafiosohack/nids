"""Rule catalogue — one stable, unique ID per detection.

Every alert the NIDS emits names a `rule_id` from this file. The ID is the join
key for correlation, for the efficacy harness, and for MITRE ATT&CK mapping, so
it must stay stable once shipped: rename the *title*, never the ID.

Two things here that the old generic labels could not express:

  * `discriminators` — the detail keys an alert for this rule MUST carry. This is
    what turns "bruteforce" into "BRUTEFORCE_LOW_RATE at 0.03 attempts/sec on
    tcp/22". `validate_alert_details()` enforces it and the harness asserts it,
    so a rule cannot silently regress into a generic bucket again.
  * `generalizes` — whether the rule catches techniques beyond the exact ones it
    was written for. Tier 1 payload signatures do not (False); Tier 2 behavioural
    rules do (True). `coverage_report()` renders this distinction for the
    dashboard so nobody reads five exploit signatures as "exploit coverage".

SCOPE LIMIT: this is a network sensor. Every rule below is inferred from packets
on the wire. It has no host, process, or EDR visibility — a detection that needs
to know which process opened a socket, or whether a login actually succeeded,
cannot be made here and is not claimed here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Kill chain ────────────────────────────────────────────────────────────────
# Ordered. The correlation engine uses index order to decide whether a source is
# ADVANCING through the chain (recon -> exploit -> C2) rather than just being
# noisy at one stage. main.py imports this so there is a single definition.
KILL_CHAIN_ORDER: List[str] = [
    "reconnaissance",
    "credential_access",
    "initial_access",
    "privilege_escalation",
    "persistence",
    "command_and_control",
    "lateral_movement",
    "collection",
    "exfiltration",
]

STAGE_LABELS: Dict[str, str] = {
    "reconnaissance":       "Reconnaissance",
    "credential_access":    "Credential Access",
    "initial_access":       "Initial Access (BREACH)",
    "privilege_escalation": "Privilege Escalation (ROOT)",
    "persistence":          "Persistence",
    "command_and_control":  "Command & Control",
    "lateral_movement":     "Lateral Movement",
    "collection":           "Collection / Staging",
    "exfiltration":         "Exfiltration",
}

STAGE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(KILL_CHAIN_ORDER)}

# ── Tiers ─────────────────────────────────────────────────────────────────────
TIER_SIGNATURE   = "signature"       # exact protocol/flag match, single packet
TIER_BEHAVIORAL  = "behavioral"      # rate/threshold over a window
TIER_EXPLOIT_1   = "exploit_tier1"   # payload signature for one specific exploit
TIER_EXPLOIT_2   = "exploit_tier2"   # post-exploitation behaviour, exploit-agnostic
TIER_CORRELATION = "correlation"     # emitted by the correlation engine
TIER_ML          = "ml"              # forwarded by live_ids_v2.py

TIER_LABELS = {
    TIER_SIGNATURE:   "Signature (packet-level)",
    TIER_BEHAVIORAL:  "Behavioural (windowed)",
    TIER_EXPLOIT_1:   "Tier 1 — exploit payload signature",
    TIER_EXPLOIT_2:   "Tier 2 — post-exploitation behaviour",
    TIER_CORRELATION: "Correlation",
    TIER_ML:          "Machine learning",
}

SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def severity_rank(sev: str) -> int:
    return SEVERITY_ORDER.get((sev or "").lower(), 0)


def max_severity(a: str, b: str) -> str:
    return a if severity_rank(a) >= severity_rank(b) else b


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str                 # human label; also the legacy alert "type"
    severity: str
    tier: str
    stage: Optional[str]       # kill-chain stage, or None (not chain-relevant)
    tactic: Optional[str]
    technique: Optional[str]   # MITRE ATT&CK technique id
    description: str
    # Detail keys every alert for this rule must carry. These are the
    # discriminating features — what makes the alert specific rather than generic.
    discriminators: Tuple[str, ...] = ()
    # True  = catches a class of behaviour beyond the specific case it was written for
    # False = only fires on exactly the enumerated case
    generalizes: bool = False
    # Whose IP is the alert's `src`? "actor" = the host doing it (correlate on it).
    # "target" = the victim. "unknown" = spoofable, do not attribute.
    src_semantics: str = "actor"
    # Exploits/CVEs this rule covers, for the coverage report. Tier 1 only.
    covers: Tuple[str, ...] = ()


# ──────────────────────────────────────────────────────────────────────────────
#  THE CATALOGUE
# ──────────────────────────────────────────────────────────────────────────────
_RULES: List[Rule] = [

    # ── Reconnaissance: TCP scan signatures ──────────────────────────────────
    # Each flag combination gets its own ID. Previously every one of these
    # collapsed into a single "Stealth Scan" alert with no way to tell them apart.
    Rule("TCP_NULL_SCAN", "TCP NULL Scan", "medium", TIER_SIGNATURE,
         "reconnaissance", "Discovery", "T1046",
         "TCP packet with no flags set (nmap -sN). Closed ports RST, open ports "
         "stay silent — the attacker maps the host from what does NOT answer.",
         discriminators=("tcp_flags", "tcp_flags_hex", "dst_port", "scan_technique"),
         covers=("nmap -sN",)),

    Rule("TCP_FIN_SCAN", "TCP FIN Scan", "medium", TIER_SIGNATURE,
         "reconnaissance", "Discovery", "T1046",
         "Bare FIN with no established session (nmap -sF). Slips past stateless "
         "filters that only inspect SYN.",
         discriminators=("tcp_flags", "tcp_flags_hex", "dst_port", "scan_technique"),
         covers=("nmap -sF",)),

    Rule("TCP_XMAS_SCAN", "TCP XMAS Scan", "medium", TIER_SIGNATURE,
         "reconnaissance", "Discovery", "T1046",
         "FIN+PSH+URG set together (nmap -sX) — 'lit up like a Christmas tree'. "
         "An illegal combination no legitimate stack emits.",
         discriminators=("tcp_flags", "tcp_flags_hex", "dst_port", "scan_technique"),
         covers=("nmap -sX",)),

    Rule("TCP_SYNFIN_SCAN", "TCP SYN+FIN Scan", "medium", TIER_SIGNATURE,
         "reconnaissance", "Discovery", "T1046",
         "SYN and FIN set simultaneously — contradictory (open and close at once), "
         "used to evade naive rule sets.",
         discriminators=("tcp_flags", "tcp_flags_hex", "dst_port", "scan_technique")),

    Rule("TCP_SYN_SCAN", "TCP SYN (half-open) Scan", "high", TIER_BEHAVIORAL,
         "reconnaissance", "Discovery", "T1046",
         "Many distinct ports probed with bare SYNs that are never completed "
         "(nmap -sS). The default nmap scan.",
         discriminators=("distinct_ports", "window_seconds", "ports_per_sec",
                         "sample_ports", "scan_technique"),
         covers=("nmap -sS",)),

    Rule("TCP_CONNECT_SCAN", "TCP Connect Scan", "high", TIER_BEHAVIORAL,
         "reconnaissance", "Discovery", "T1046",
         "Many distinct ports where the handshake COMPLETED and was then torn "
         "down immediately (nmap -sT). Distinguished from SYN scan by the "
         "presence of SYN-ACKs — this is a full connect(), not half-open.",
         discriminators=("distinct_ports", "window_seconds", "ports_per_sec",
                         "sample_ports", "scan_technique", "handshakes_completed"),
         covers=("nmap -sT",)),

    Rule("SCAN_SLOW_RATE", "Low-and-Slow Scan", "medium", TIER_BEHAVIORAL,
         "reconnaissance", "Discovery", "T1046",
         "Distinct ports probed by one source over a long horizon, each burst "
         "staying under the short-window threshold.",
         discriminators=("distinct_ports", "window_seconds", "ports_per_sec"),
         generalizes=True),

    Rule("SCAN_DISTRIBUTED", "Distributed Scan", "high", TIER_BEHAVIORAL,
         "reconnaissance", "Discovery", "T1046",
         "Many sources each probing one destination, every source individually "
         "below the per-host threshold.",
         discriminators=("source_count", "window_seconds", "target"),
         generalizes=True, src_semantics="unknown"),

    Rule("SCAN_FRAGMENTED", "Fragmented Scan", "high", TIER_SIGNATURE,
         "reconnaissance", "Discovery", "T1046",
         "Scan probe delivered in IP fragments (nmap -f) to split the TCP header "
         "across packets. Reassembled before rule matching, so it normalises to "
         "the same signature as the unfragmented scan.",
         discriminators=("fragment_count", "reassembled_rule_id", "scan_technique"),
         generalizes=True, covers=("nmap -f", "nmap --mtu")),

    # ── Credential access ────────────────────────────────────────────────────
    # Split by MEASURED rate. Both carry attempts_per_sec so an analyst can see
    # why the classification landed where it did.
    Rule("BRUTEFORCE_STANDARD_RATE", "Brute Force (standard rate)", "high",
         TIER_BEHAVIORAL, "credential_access", "Credential Access", "T1110",
         "Repeated authentication attempts against one service at or above the "
         "standard-rate threshold — an unthrottled password-guessing tool.",
         discriminators=("attempts", "attempts_per_sec", "rate_class",
                         "window_seconds", "service", "dst_port"),
         covers=("hydra", "medusa", "ncrack", "msf auxiliary/scanner/*_login")),

    Rule("BRUTEFORCE_LOW_RATE", "Brute Force (low and slow)", "high",
         TIER_BEHAVIORAL, "credential_access", "Credential Access", "T1110",
         "Repeated authentication attempts deliberately paced below the "
         "standard-rate threshold to stay under lockout/alerting policy. Needs a "
         "long observation window; the give-away is persistence, not volume.",
         discriminators=("attempts", "attempts_per_sec", "rate_class",
                         "window_seconds", "service", "dst_port"),
         generalizes=True,
         covers=("hydra -t 1 -W", "manual credential stuffing")),

    # ── Tier 1: exploit payload signatures ───────────────────────────────────
    # NARROW BY CONSTRUCTION. Each fires on exactly one exploit's wire format.
    # A variant that changes the trigger string evades these — that is what Tier 2
    # is for.
    Rule("EXPLOIT_VSFTPD_BACKDOOR", "vsftpd 2.3.4 Backdoor Trigger", "high",
         TIER_EXPLOIT_1, "initial_access", "Initial Access", "T1190",
         "FTP USER command whose username contains the ':)' smiley — the trigger "
         "for the backdoor shipped in the compromised vsftpd 2.3.4 tarball.",
         discriminators=("matched_field", "matched_value", "dst_port", "exploit"),
         covers=("vsftpd 2.3.4 backdoor (OSVDB-73573)",)),

    Rule("EXPLOIT_VSFTPD_BACKDOOR_SHELL", "vsftpd Backdoor Shell Opened", "critical",
         TIER_EXPLOIT_1, "command_and_control", "Command and Control", "T1059",
         "Connection to tcp/6200 shortly after the vsftpd ':)' trigger — the "
         "backdoor root shell is now listening and the attacker has reached it. "
         "This is the confirmation half of EXPLOIT_VSFTPD_BACKDOOR.",
         discriminators=("dst_port", "seconds_after_trigger", "exploit"),
         covers=("vsftpd 2.3.4 backdoor (OSVDB-73573)",)),

    Rule("EXPLOIT_SAMBA_USERMAP", "Samba usermap_script Command Injection", "critical",
         TIER_EXPLOIT_1, "initial_access", "Initial Access", "T1190",
         "Shell metacharacters (; | ` $()) inside the SMB session-setup username. "
         "Samba with 'username map script' passes the field to /bin/sh unsanitised.",
         discriminators=("matched_field", "matched_value", "metachars", "dst_port", "exploit"),
         covers=("Samba usermap_script (CVE-2007-2447)",)),

    Rule("EXPLOIT_DISTCC_CMDEXEC", "distccd Remote Command Execution", "critical",
         TIER_EXPLOIT_1, "initial_access", "Initial Access", "T1190",
         "Shell command strings visible in the cleartext distcc DIST/ARGV request. "
         "distccd with no access control compiles whatever it is handed.",
         discriminators=("matched_value", "dst_port", "exploit"),
         covers=("distccd cmd exec (CVE-2004-2687)",)),

    Rule("EXPLOIT_UNREALIRCD_BACKDOOR", "UnrealIRCd 3.2.8.1 Backdoor", "critical",
         TIER_EXPLOIT_1, "initial_access", "Initial Access", "T1190",
         "IRC traffic containing the 'AB;' prefix — the backdoor command marker in "
         "the trojaned UnrealIRCd 3.2.8.1 archive. Everything after it runs as a "
         "shell command.",
         discriminators=("matched_value", "injected_command", "dst_port", "exploit"),
         covers=("UnrealIRCd 3.2.8.1 backdoor (CVE-2010-2075)",)),

    Rule("EXPLOIT_JAVA_RMI_DESERIALIZATION", "Java RMI Deserialization Payload",
         "critical", TIER_EXPLOIT_1, "initial_access", "Initial Access", "T1190",
         "Java serialized-object stream header (AC ED 00 05) delivered to an RMI "
         "endpoint — the transport for a deserialization gadget chain.",
         discriminators=("magic_bytes", "dst_port", "exploit"),
         covers=("Java RMI registry deserialization",)),

    # ── Tier 2: post-exploitation behaviour ──────────────────────────────────
    # HIGHER PRIORITY THAN TIER 1. These describe what a successful exploit
    # *causes*, not how it was delivered, so they fire on exploits nobody has
    # written a signature for — including ones that did not exist when this shipped.
    Rule("BEHAVIOR_UNEXPECTED_OUTBOUND", "Unexpected Outbound Connection", "critical",
         TIER_EXPLOIT_2, "command_and_control", "Command and Control", "T1571",
         "A host whose baseline outbound behaviour is near-zero (a server: it "
         "accepts connections, it does not make them) initiated an outbound SYN. "
         "This is the reverse-shell indicator, and it holds regardless of which "
         "exploit produced the shell.",
         discriminators=("dst_port", "destination", "baseline_outbound_conns",
                         "baseline_window_seconds", "indicator"),
         generalizes=True),

    Rule("BEHAVIOR_NEW_LISTENING_PORT", "New Listening Port on Target", "critical",
         TIER_EXPLOIT_2, "persistence", "Persistence", "T1571",
         "A port answering SYN-ACK on a monitored host that is absent from the "
         "known-good baseline snapshot. This is the bind-shell indicator, and it "
         "holds regardless of which exploit opened the port.",
         discriminators=("listening_port", "host", "baseline_ports", "indicator"),
         generalizes=True, src_semantics="target"),

    Rule("BEHAVIOR_INTERACTIVE_SHELL", "Interactive Shell Session", "critical",
         TIER_EXPLOIT_2, "command_and_control", "Command and Control", "T1059",
         "A TCP session with the SHAPE of someone typing commands: keystroke-"
         "sized packets one way, larger output bursts the other, turn-taking, "
         "long-lived and low-volume. Independent of port and of payload, so it "
         "catches the reverse shell on tcp/443 that the baseline and port-"
         "heuristic rules both miss. Suppressed when host_log_sensor.py confirms "
         "a real login for the same host pair.",
         discriminators=("shell_port", "small_packet_ratio", "output_input_ratio",
                         "turn_ratio", "packets", "duration_sec", "indicator"),
         generalizes=True),

    Rule("BEHAVIOR_UNCOMMON_EGRESS_PORT", "Uncommon Egress Port", "high",
         TIER_EXPLOIT_2, "command_and_control", "Command and Control", "T1571",
         "An internal host opened an outbound session to the Internet on a port "
         "that is not normal client traffic. Weaker than "
         "BEHAVIOR_UNEXPECTED_OUTBOUND (a port heuristic rather than a per-host "
         "baseline), but it covers hosts with no baseline in the asset inventory.",
         discriminators=("dst_port", "destination", "indicator"),
         generalizes=True),

    Rule("BEHAVIOR_C2_BEACON", "C2 Beaconing", "high",
         TIER_EXPLOIT_2, "command_and_control", "Command and Control", "T1071",
         "Repeated connections to one destination at a regular interval "
         "(low coefficient of variation on inter-arrival gaps) — a check-in "
         "heartbeat. Independent of payload, protocol and encryption. When the "
         "per-check-in byte counts are also observed, their variation is a "
         "second, independent feature: near-identical request sizes on a fixed "
         "timer promote the finding, widely varying ones demote it to a "
         "scheduled poller.",
         discriminators=("beacon_interval_sec", "interval_cv", "connections",
                         "destination", "size_evidence", "indicator"),
         generalizes=True),

    Rule("BEHAVIOR_CLEARTEXT_ON_TLS_PORT", "No TLS Handshake on TLS Port",
         "critical",
         TIER_EXPLOIT_2, "command_and_control", "Command and Control", "T1571",
         "A connection to a port where TLS is the convention (443, 993, 8443, "
         "…) whose first client data packet does not begin with a TLS handshake "
         "record. Attackers pick 443 because egress filters allow it and often "
         "do not wrap the channel in real TLS. Costs three bytes at a fixed "
         "offset and needs no baseline, so it renders a verdict on the first "
         "data packet — far earlier than the traffic-shape rule can. Only flows "
         "whose SYN was observed are judged, so a sensor starting mid-session "
         "cannot produce a false positive.",
         discriminators=("dst_port", "destination", "payload_class",
                         "first_bytes_hex", "indicator"),
         generalizes=True),

    # ── Other network behaviour (pre-existing detectors, now ID'd) ───────────
    Rule("FLOOD_SYN", "SYN Flood", "critical", TIER_BEHAVIORAL, None,
         "Impact", "T1499.001",
         "SYN packets far in excess of returning ACKs — half-open connection "
         "exhaustion.",
         discriminators=("syn_count", "ack_count", "syn_ack_ratio", "window_seconds"),
         generalizes=True),

    Rule("FLOOD_PACKET_RATE", "Packet Flood / DDoS", "critical", TIER_BEHAVIORAL,
         None, "Impact", "T1498",
         "Raw packet rate from one source above the volumetric threshold.",
         discriminators=("packet_count", "window_seconds", "packets_per_sec"),
         generalizes=True),

    Rule("FLOOD_UDP_AMPLIFICATION", "UDP Amplification", "high", TIER_BEHAVIORAL,
         None, "Impact", "T1498.002",
         "Burst of requests to a known reflector service (DNS/NTP/SSDP/memcached).",
         discriminators=("dst_port", "request_count", "window_seconds"),
         generalizes=True),

    Rule("BREACH_SESSION", "Suspected Breach", "critical", TIER_BEHAVIORAL,
         "initial_access", "Initial Access", "T1078",
         "Data-heavy session with a service this source was brute-forcing moments "
         "earlier. Network-inferred only — confirm against host auth logs.",
         discriminators=("target_port", "session_bytes", "confidence")),

    Rule("LATERAL_ADMIN_SWEEP", "Lateral Movement", "critical", TIER_BEHAVIORAL,
         "lateral_movement", "Lateral Movement", "T1021",
         "One internal host establishing sessions to many internal hosts on "
         "admin/remote-management ports.",
         discriminators=("internal_targets", "window_seconds"), generalizes=True),

    Rule("EXFIL_BULK_OUTBOUND", "Data Exfiltration", "critical", TIER_BEHAVIORAL,
         "exfiltration", "Exfiltration", "T1041",
         "Sustained outbound byte volume from an internal host to an external IP.",
         discriminators=("bytes_transferred", "window_seconds", "destination"),
         generalizes=True),

    Rule("COLLECTION_STAGING", "Data Staging", "high", TIER_BEHAVIORAL,
         "collection", "Collection", "T1074",
         "One internal host aggregating large volumes from many internal hosts — "
         "pre-exfiltration collection.",
         discriminators=("source_hosts", "window_seconds"),
         generalizes=True, src_semantics="target"),

    Rule("CREDS_CLOUD_METADATA", "Cloud Metadata Access", "high", TIER_BEHAVIORAL,
         "credential_access", "Credential Access", "T1552.005",
         "Burst of requests to a cloud instance-metadata endpoint — SSRF stealing "
         "instance IAM credentials.",
         discriminators=("metadata_ip", "hits", "window_seconds"), generalizes=True),

    Rule("EXFIL_DNS_TUNNEL", "DNS Tunneling", "high", TIER_BEHAVIORAL,
         "exfiltration", "Exfiltration", "T1048",
         "Sustained long / high-entropy DNS queries to one parent domain.",
         discriminators=("parent_domain", "suspicious_queries", "window_seconds"),
         generalizes=True),

    Rule("MITM_ARP_SPOOF", "ARP Spoofing", "critical", TIER_SIGNATURE, None,
         "Collection", "T1557.002",
         "An IP claimed by an unexpected MAC. Not attributed to a kill chain: at "
         "layer 2 the attacker forges the source, so per-actor attribution would "
         "be unreliable.",
         discriminators=("ip", "verified"), src_semantics="unknown"),

    Rule("ASSET_ROGUE_HOST", "Rogue Host", "medium", TIER_BEHAVIORAL, None,
         "Discovery", "T1200",
         "An internal IP announced on the LAN that is absent from the asset "
         "inventory.",
         discriminators=("ip", "mac"), src_semantics="unknown"),

    Rule("ANOMALY_TRAFFIC_BASELINE", "Traffic Anomaly", "medium", TIER_BEHAVIORAL,
         None, "Discovery", None,
         "A source's connection rate spiked far above its OWN learned normal.",
         discriminators=("burst", "baseline_mean", "sigma"), generalizes=True),

    # ── Correlation output ───────────────────────────────────────────────────
    Rule("CORRELATED_INCIDENT", "Correlated Incident", "high", TIER_CORRELATION,
         None, None, None,
         "Multiple alerts from one source advancing FORWARD through the kill "
         "chain inside the correlation window. Replaces the old Attack Chain / "
         "Multi-Vector Attack / Coordinated Attack labels. The severity shown is "
         "the incident's own: the worst member alert, escalated to critical once "
         "three kill-chain stages are reached. 'high' here is only the floor.",
         discriminators=("incident_id", "stages", "window_seconds", "event_count")),

    Rule("ACTIVE_INTRUSION", "Active Intrusion", "critical", TIER_CORRELATION,
         None, None, None,
         "A host has reached initial access AND at least one post-access stage. "
         "Distinct from CORRELATED_INCIDENT: that one groups alerts, this one is "
         "the verdict that the attacker is inside and operating.",
         discriminators=("kill_chain", "stage_count")),

    # ── ML sensor ────────────────────────────────────────────────────────────
    Rule("ML_FLOW_ANOMALY", "ML Anomaly", "medium", TIER_ML, None, None, None,
         "Flow classified as attack by the CIC-IDS2017 model in live_ids_v2.py.",
         discriminators=("predicted_class",), generalizes=True),
]

RULES: Dict[str, Rule] = {r.rule_id: r for r in _RULES}

# Legacy alert "type" -> rule_id, so alerts posted to /alert by the host and
# cloud sensors (which predate rule IDs) still land on a catalogue entry.
LEGACY_TYPE_MAP: Dict[str, str] = {
    "Port Scan":            "TCP_SYN_SCAN",
    "Stealth Scan":         "SCAN_SLOW_RATE",
    "Slow Scan":            "SCAN_SLOW_RATE",
    "Distributed Scan":     "SCAN_DISTRIBUTED",
    "Brute Force Attempt":  "BRUTEFORCE_STANDARD_RATE",
    "SYN Flood":            "FLOOD_SYN",
    "DDoS":                 "FLOOD_PACKET_RATE",
    "UDP Amplification":    "FLOOD_UDP_AMPLIFICATION",
    "Breach":               "BREACH_SESSION",
    "Lateral Movement":     "LATERAL_ADMIN_SWEEP",
    "Data Exfiltration":    "EXFIL_BULK_OUTBOUND",
    "Data Staging":         "COLLECTION_STAGING",
    "C2 Beaconing":         "BEHAVIOR_C2_BEACON",
    "Reverse Shell":        "BEHAVIOR_UNCOMMON_EGRESS_PORT",
    "Cloud Metadata Access": "CREDS_CLOUD_METADATA",
    "DNS Tunneling":        "EXFIL_DNS_TUNNEL",
    "ARP Spoofing":         "MITM_ARP_SPOOF",
    "Rogue Host":           "ASSET_ROGUE_HOST",
    "Traffic Anomaly":      "ANOMALY_TRAFFIC_BASELINE",
    "ML Anomaly":           "ML_FLOW_ANOMALY",
}

# Alert types from the host / cloud log sensors. They are not packet rules, so
# they are not in the catalogue, but they still carry a kill-chain stage.
# (stage, tactic, technique)
EXTERNAL_SENSOR_STAGES: Dict[str, Tuple[str, str, str]] = {
    "Host Brute Force":      ("credential_access",    "Credential Access",    "T1110"),
    "SSH Login":             ("initial_access",       "Valid Accounts",       "T1078"),
    "Root Access":           ("privilege_escalation", "Privilege Escalation", "T1548"),
    "Persistence":           ("persistence",          "Persistence",          "T1136"),
    "Sensitive File Access": ("collection",           "Collection",           "T1005"),
    "Cloud Port Scan":       ("reconnaissance",       "Discovery",            "T1046"),
    "Cloud Recon":           ("reconnaissance",       "Discovery",            "T1046"),
    "Cloud Brute Force":     ("credential_access",    "Credential Access",    "T1110"),
    "Impossible Travel":     ("initial_access",       "Initial Access",       "T1078"),
}


def get_rule(rule_id: str) -> Optional[Rule]:
    return RULES.get(rule_id)


def resolve_rule(rule_id: Optional[str], alert_type: Optional[str]) -> Optional[Rule]:
    """Find a catalogue entry from an explicit rule_id, else a legacy type name."""
    if rule_id and rule_id in RULES:
        return RULES[rule_id]
    if alert_type:
        mapped = LEGACY_TYPE_MAP.get(alert_type)
        if mapped:
            return RULES.get(mapped)
    return None


def validate_alert_details(rule_id: str, details: Dict) -> List[str]:
    """Return the discriminator keys this rule requires but the alert is missing.

    Empty list = the alert carries its discriminating features. The efficacy
    harness asserts this for every alert it observes, which is what stops a rule
    from decaying back into a generic bucket label.
    """
    rule = RULES.get(rule_id)
    if rule is None:
        return [f"<unknown rule_id {rule_id!r}>"]
    return [k for k in rule.discriminators if k not in details or details[k] is None]


# ──────────────────────────────────────────────────────────────────────────────
#  COVERAGE REPORT
# ──────────────────────────────────────────────────────────────────────────────
def coverage_report() -> Dict:
    """Machine-readable rule coverage, for `GET /rules/coverage` and the dashboard.

    The point of this structure is the honesty note per tier. Tier 1 lists five
    exploits and catches those five; Tier 2 lists a handful of behaviours and
    catches an open-ended class. Presenting them as one number would overstate
    coverage. Counts are computed, never written down, so adding a rule cannot
    leave a stale figure behind.
    """
    tiers: Dict[str, Dict] = {}
    for rule in RULES.values():
        t = tiers.setdefault(rule.tier, {
            "tier": rule.tier,
            "label": TIER_LABELS.get(rule.tier, rule.tier),
            "rules": [],
            "generalizing_rules": 0,
            "specific_rules": 0,
        })
        t["rules"].append({
            "rule_id":        rule.rule_id,
            "title":          rule.title,
            "severity":       rule.severity,
            "stage":          rule.stage,
            "stage_label":    STAGE_LABELS.get(rule.stage) if rule.stage else None,
            "mitre_tactic":   rule.tactic,
            "mitre_technique": rule.technique,
            "generalizes":    rule.generalizes,
            "covers":         list(rule.covers),
            "discriminators": list(rule.discriminators),
            "description":    rule.description,
        })
        if rule.generalizes:
            t["generalizing_rules"] += 1
        else:
            t["specific_rules"] += 1

    for t in tiers.values():
        t["rules"].sort(key=lambda r: r["rule_id"])
        t["rule_count"] = len(t["rules"])

    if TIER_EXPLOIT_1 in tiers:
        enumerated = sorted({c for r in RULES.values()
                             if r.tier == TIER_EXPLOIT_1 for c in r.covers})
        tiers[TIER_EXPLOIT_1]["coverage_note"] = (
            "NARROW: these signatures fire on the exact wire format of the "
            f"{len(enumerated)} enumerated exploits and nothing else. An exploit "
            "variant that changes the trigger string, or any exploit not in this "
            "list, will NOT match. Do not read this tier as 'exploit coverage'."
        )
        tiers[TIER_EXPLOIT_1]["enumerated_exploits"] = enumerated

    if TIER_EXPLOIT_2 in tiers:
        tiers[TIER_EXPLOIT_2]["coverage_note"] = (
            "BROAD: these detect what a successful exploit CAUSES (a shell calling "
            "out, a new port listening, a C2 heartbeat, a session on 443 that "
            "never negotiated TLS) rather than how it was "
            "delivered. They therefore fire on exploitation techniques beyond the "
            "five enumerated in Tier 1, including ones with no signature written "
            "for them. Ranked above Tier 1 for this reason."
        )

    return {
        "tiers": [tiers[t] for t in (TIER_EXPLOIT_2, TIER_EXPLOIT_1, TIER_SIGNATURE,
                                     TIER_BEHAVIORAL, TIER_CORRELATION, TIER_ML)
                  if t in tiers],
        "total_rules": len(RULES),
        "scope_limit": (
            "Network-only detection. This sensor sees packets on the monitored "
            "segment. It has no host, process, or EDR visibility: it cannot see "
            "in-memory execution, an exploit delivered over an encrypted channel "
            "it cannot decrypt, local privilege escalation, or activity on a host "
            "whose traffic never crosses this segment. Host-level evidence comes "
            "from host_log_sensor.py, which is a separate and independent source."
        ),
    }

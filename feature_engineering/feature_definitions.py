"""
Feature Definitions and Security Relevance
Defines all 41 features used in intrusion detection with
explanations of why each matters for security.
"""

from typing import Dict, List

# Complete feature definitions with security context
FEATURE_DEFINITIONS: Dict[str, Dict] = {
    "duration": {
        "description": "Length of the connection in seconds",
        "importance": "HIGH",
        "security_relevance": (
            "DoS attacks have very short durations. "
            "Data exfiltration shows unusually long connections."
        )
    },
    "protocol_type": {
        "description": "Protocol type (TCP, UDP, ICMP)",
        "importance": "HIGH",
        "security_relevance": (
            "TCP used in SYN floods. UDP used in amplification attacks. "
            "ICMP used in ping floods and reconnaissance."
        )
    },
    "service": {
        "description": "Network service on destination (http, ftp, smtp...)",
        "importance": "MEDIUM",
        "security_relevance": (
            "Identifies target service. Unusual services may indicate backdoors."
        )
    },
    "flag": {
        "description": "Status of the TCP connection",
        "importance": "HIGH",
        "security_relevance": (
            "S0 flag = half-open connections indicating SYN flood. "
            "REJ flag = rejected connections indicating scanning."
        )
    },
    "src_bytes": {
        "description": "Bytes sent from source to destination",
        "importance": "HIGH",
        "security_relevance": (
            "Large outbound transfers may indicate data exfiltration."
        )
    },
    "dst_bytes": {
        "description": "Bytes sent from destination to source",
        "importance": "HIGH",
        "security_relevance": (
            "Large inbound transfers may indicate malware download."
        )
    },
    "land": {
        "description": "1 if source and destination IP/port are same",
        "importance": "CRITICAL",
        "security_relevance": (
            "LAND attack indicator. Legitimate value is always 0."
        )
    },
    "wrong_fragment": {
        "description": "Number of wrong fragments",
        "importance": "HIGH",
        "security_relevance": (
            "Fragmentation attacks and IDS evasion techniques."
        )
    },
    "urgent": {
        "description": "Number of urgent packets",
        "importance": "MEDIUM",
        "security_relevance": (
            "Rarely used in normal traffic. Non-zero value is suspicious."
        )
    },
    "hot": {
        "description": "Number of hot indicators (sensitive file access)",
        "importance": "CRITICAL",
        "security_relevance": (
            "Access to /etc/passwd, privileged commands. High = compromise."
        )
    },
    "num_failed_logins": {
        "description": "Number of failed login attempts",
        "importance": "CRITICAL",
        "security_relevance": (
            "Brute force attacks, credential stuffing, password spraying."
        )
    },
    "logged_in": {
        "description": "1 if successfully logged in",
        "importance": "HIGH",
        "security_relevance": (
            "Successful authentication. Post-auth attacks if logged_in=1."
        )
    },
    "num_compromised": {
        "description": "Number of compromised conditions",
        "importance": "CRITICAL",
        "security_relevance": (
            "Direct indicator of successful attack on the system."
        )
    },
    "root_shell": {
        "description": "1 if root shell was obtained",
        "importance": "CRITICAL",
        "security_relevance": (
            "Privilege escalation success. Most severe attack indicator."
        )
    },
    "su_attempted": {
        "description": "1 if su root command attempted",
        "importance": "HIGH",
        "security_relevance": (
            "Privilege escalation attempt. May indicate compromised account."
        )
    },
    "num_root": {
        "description": "Number of root accesses",
        "importance": "HIGH",
        "security_relevance": (
            "Root-level operations. High frequency indicates compromise."
        )
    },
    "num_file_creations": {
        "description": "Number of file creation operations",
        "importance": "MEDIUM",
        "security_relevance": (
            "Malware installation, backdoor creation, web shell upload."
        )
    },
    "num_shells": {
        "description": "Number of shell prompts",
        "importance": "HIGH",
        "security_relevance": (
            "Interactive shell access. Reverse shell connections."
        )
    },
    "num_access_files": {
        "description": "Number of operations on access control files",
        "importance": "HIGH",
        "security_relevance": (
            "/etc/passwd access, ACL modifications, permission changes."
        )
    },
    "num_outbound_cmds": {
        "description": "Number of outbound commands in FTP session",
        "importance": "MEDIUM",
        "security_relevance": "Data exfiltration via FTP."
    },
    "is_host_login": {
        "description": "1 if login belongs to host list",
        "importance": "LOW",
        "security_relevance": "Distinguishes network vs host logins."
    },
    "is_guest_login": {
        "description": "1 if guest login",
        "importance": "MEDIUM",
        "security_relevance": (
            "Anonymous access. Guest account abuse vector."
        )
    },
    "count": {
        "description": "Connections to same host in past 2 seconds",
        "importance": "HIGH",
        "security_relevance": (
            "High connection rate indicates port scan or DoS attack."
        )
    },
    "srv_count": {
        "description": "Connections to same service in past 2 seconds",
        "importance": "HIGH",
        "security_relevance": "Service-specific attacks, application-layer DoS."
    },
    "serror_rate": {
        "description": "% connections with SYN errors",
        "importance": "HIGH",
        "security_relevance": (
            "SYN flood: high error rate. Normal rate is very low (<1%)."
        )
    },
    "srv_serror_rate": {
        "description": "% connections with SYN errors to same service",
        "importance": "MEDIUM",
        "security_relevance": "Service-targeted SYN flood attacks."
    },
    "rerror_rate": {
        "description": "% connections with REJ errors",
        "importance": "HIGH",
        "security_relevance": (
            "Connection rejections. High rate indicates port scanning."
        )
    },
    "srv_rerror_rate": {
        "description": "% connections with REJ errors to same service",
        "importance": "MEDIUM",
        "security_relevance": "Service rejection patterns."
    },
    "same_srv_rate": {
        "description": "% connections to same service",
        "importance": "MEDIUM",
        "security_relevance": "Low rate = scanning multiple services."
    },
    "diff_srv_rate": {
        "description": "% connections to different services",
        "importance": "MEDIUM",
        "security_relevance": "High diversity = service scanning reconnaissance."
    },
    "srv_diff_host_rate": {
        "description": "% connections to different hosts same service",
        "importance": "MEDIUM",
        "security_relevance": "Horizontal scanning, worm propagation."
    },
    "dst_host_count": {
        "description": "Connections to same destination host",
        "importance": "LOW",
        "security_relevance": "Activity baseline for destination host."
    },
    "dst_host_srv_count": {
        "description": "Connections to same service on destination",
        "importance": "LOW",
        "security_relevance": "Service usage patterns on destination."
    },
    "dst_host_same_srv_rate": {
        "description": "% connections to same service on destination",
        "importance": "MEDIUM",
        "security_relevance": "Service concentration patterns."
    },
    "dst_host_diff_srv_rate": {
        "description": "% connections to different services on destination",
        "importance": "MEDIUM",
        "security_relevance": "Multi-service attacks on destination host."
    },
    "dst_host_same_src_port_rate": {
        "description": "% connections using same source port",
        "importance": "LOW",
        "security_relevance": "Source port reuse and NAT detection."
    },
    "dst_host_srv_diff_host_rate": {
        "description": "% connections from different hosts to same service",
        "importance": "MEDIUM",
        "security_relevance": "Distributed attacks (DDoS) detection."
    },
    "dst_host_serror_rate": {
        "description": "% SYN errors for connections to destination",
        "importance": "HIGH",
        "security_relevance": "Destination under SYN flood attack."
    },
    "dst_host_srv_serror_rate": {
        "description": "% SYN errors for connections to destination service",
        "importance": "MEDIUM",
        "security_relevance": "Service-specific attack on destination."
    },
    "dst_host_rerror_rate": {
        "description": "% REJ errors for connections to destination",
        "importance": "MEDIUM",
        "security_relevance": "Destination rejecting connections."
    },
    "dst_host_srv_rerror_rate": {
        "description": "% REJ errors for connections to destination service",
        "importance": "MEDIUM",
        "security_relevance": "Service rejection and authentication failures."
    },
}


def get_all_feature_names() -> List[str]:
    """Get list of all feature names."""
    return list(FEATURE_DEFINITIONS.keys())


def get_high_importance_features() -> List[str]:
    """Get features with HIGH or CRITICAL importance."""
    return [
        name for name, info in FEATURE_DEFINITIONS.items()
        if info['importance'] in ('HIGH', 'CRITICAL')
    ]


def print_feature_summary() -> None:
    """Print a summary of all features."""
    print("\n" + "="*60)
    print(f"TOTAL FEATURES: {len(FEATURE_DEFINITIONS)}")
    print("="*60)
    for name, info in FEATURE_DEFINITIONS.items():
        print(f"\n[{info['importance']}] {name}")
        print(f"  {info['description']}")
        print(f"  Security: {info['security_relevance']}")


if __name__ == "__main__":
    print_feature_summary()
"""
src/utils/url_analyzer.py
--------------------------
Phishing URL detection module.

Features:
  - URL structural analysis (length, subdomains, special chars)
  - TLD-based risk scoring
  - Domain age heuristics
  - Brand impersonation detection (typosquatting)
  - Lexical URL features for ML

This module can be used standalone or integrated into the main pipeline.
"""

import re
import math
import socket
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from loguru import logger

try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False


# ─────────────────────────────────────────────
# Known legitimate brands (for impersonation detection)
# ─────────────────────────────────────────────
BRAND_DOMAINS = {
    "paypal": "paypal.com",
    "amazon": "amazon.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "netflix": "netflix.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "twitter": "twitter.com",
    "linkedin": "linkedin.com",
    "bank": "chase.com",
    "wellsfargo": "wellsfargo.com",
    "irs": "irs.gov",
    "ebay": "ebay.com",
    "dropbox": "dropbox.com",
    "zoom": "zoom.us",
}

# Suspicious patterns in URLs
SUSPICIOUS_URL_PATTERNS = [
    r"@",              # Credentials in URL: http://attacker.com@victim.com
    r"//.*//",         # Double slashes
    r"\.{3,}",         # Multiple consecutive dots
    r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",  # IP address host
    r"bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly|is\.gd",  # URL shorteners
    r"login.*verify|verify.*login",
    r"secure.*update|update.*secure",
    r"account.*confirm|confirm.*account",
]

HIGH_RISK_TLDS = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".cc",
    ".top", ".click", ".link", ".info", ".online", ".site",
    ".website", ".space", ".fun", ".win", ".men", ".loan",
    ".download", ".racing", ".date", ".party", ".review",
}

LEGITIMATE_TLDS = {".com", ".org", ".net", ".gov", ".edu", ".io", ".co"}


class URLAnalyzer:
    """
    Analyse URLs found in emails for phishing indicators.

    Example:
        analyzer = URLAnalyzer()
        result = analyzer.analyze("http://secure-paypal.xyz/login?verify=1")
        print(result["risk_score"], result["indicators"])
    """

    def analyze(self, url: str) -> Dict:
        """
        Full URL risk analysis.

        Returns:
            risk_score: float [0, 1]
            risk_level: "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
            indicators: list of risk indicators found
            features: dict of numerical features for ML
        """
        url = url.strip()
        if not url.startswith(("http://", "https://", "www.")):
            url = "http://" + url

        parsed = self._safe_parse(url)
        features = self._extract_features(url, parsed)
        indicators = self._detect_indicators(url, parsed, features)

        # Score: weighted sum of features
        score = self._compute_score(features, indicators)

        return {
            "url": url,
            "risk_score": round(score, 4),
            "risk_level": self._risk_level(score),
            "indicators": indicators,
            "features": features,
            "domain": parsed.get("domain", ""),
            "uses_https": features["uses_https"],
        }

    def analyze_batch(self, urls: List[str]) -> List[Dict]:
        return [self.analyze(url) for url in urls]

    # ──────────────────────────────────────────
    # Feature extraction
    # ──────────────────────────────────────────

    def _extract_features(self, url: str, parsed: Dict) -> Dict:
        hostname = parsed.get("hostname", "")
        path = parsed.get("path", "")
        full_url = url.lower()

        # Domain decomposition
        if TLDEXTRACT_AVAILABLE:
            ext = tldextract.extract(url)
            domain = ext.domain
            suffix = "." + ext.suffix if ext.suffix else ""
            subdomains = ext.subdomain.split(".") if ext.subdomain else []
        else:
            domain = hostname.split(".")[-2] if "." in hostname else hostname
            suffix = "." + hostname.split(".")[-1] if "." in hostname else ""
            subdomains = hostname.split(".")[:-2]

        features = {
            # Length features
            "url_length":     len(url),
            "hostname_length": len(hostname),
            "path_length":    len(path),

            # Domain structure
            "subdomain_count":   len(subdomains),
            "has_ip_host":       int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", hostname))),
            "dot_count":         hostname.count("."),
            "hyphen_count":      domain.count("-"),
            "digit_in_domain":   int(bool(re.search(r"\d", domain))),
            "domain_length":     len(domain),

            # TLD risk
            "high_risk_tld":     int(suffix in HIGH_RISK_TLDS),
            "legitimate_tld":    int(suffix in LEGITIMATE_TLDS),

            # Path/query features
            "path_depth":        path.count("/"),
            "has_query":         int("?" in url),
            "query_length":      len(parsed.get("query", "")),
            "has_fragment":      int("#" in url),
            "special_char_count": sum(url.count(c) for c in ["@", "//", "%", "~"]),

            # Protocol
            "uses_https":        int(url.startswith("https://")),
            "uses_http":         int(url.startswith("http://") and not url.startswith("https://")),

            # Entropy (random-looking domains often used in phishing)
            "domain_entropy":    self._entropy(domain),

            # Brand impersonation
            "brand_in_domain":   int(any(b in domain for b in BRAND_DOMAINS.keys())),
            "brand_impersonated": int(self._detect_brand_impersonation(domain, subdomains, suffix)),

            # URL shortener
            "is_shortened":      int(bool(re.search(r"bit\.ly|tinyurl|goo\.gl|t\.co", full_url))),
        }

        return features

    def _detect_indicators(self, url: str, parsed: Dict, features: Dict) -> List[str]:
        indicators = []
        url_lower = url.lower()

        if features["has_ip_host"]:
            indicators.append("IP address used as host (avoids domain blacklists)")
        if features["high_risk_tld"]:
            indicators.append(f"High-risk TLD detected")
        if features["brand_impersonated"]:
            indicators.append("Possible brand impersonation (typosquatting)")
        if features["domain_entropy"] > 3.5:
            indicators.append(f"High domain entropy ({features['domain_entropy']:.2f}) — possibly auto-generated")
        if features["subdomain_count"] > 3:
            indicators.append(f"Excessive subdomains ({features['subdomain_count']}) — common in phishing")
        if features["hyphen_count"] > 1:
            indicators.append(f"Multiple hyphens in domain — common in phishing")
        if features["is_shortened"]:
            indicators.append("URL shortener detected — hides actual destination")
        if features["uses_http"] and not features["uses_https"]:
            indicators.append("Non-HTTPS URL — data transmitted in plaintext")
        if features["url_length"] > 200:
            indicators.append(f"Very long URL ({features['url_length']} chars) — obfuscation")
        if "@" in url:
            indicators.append("@ symbol in URL — credential-stuffing technique")

        for pattern in SUSPICIOUS_URL_PATTERNS:
            if re.search(pattern, url_lower):
                pass  # Already covered above

        for brand, legit_domain in BRAND_DOMAINS.items():
            if brand in url_lower and legit_domain not in url_lower:
                if brand in parsed.get("hostname", "").lower():
                    indicators.append(f"Brand '{brand}' in hostname but not on {legit_domain}")

        return indicators

    def _compute_score(self, features: Dict, indicators: List[str]) -> float:
        score = 0.0

        # Hard indicators
        if features["has_ip_host"]:          score += 0.30
        if features["brand_impersonated"]:   score += 0.35
        if features["high_risk_tld"]:        score += 0.25
        if features["is_shortened"]:         score += 0.20
        if "@" in str(features):             score += 0.20

        # Soft indicators
        if features["domain_entropy"] > 3.5: score += 0.15
        if features["subdomain_count"] > 2:  score += features["subdomain_count"] * 0.03
        if features["hyphen_count"] > 1:     score += features["hyphen_count"] * 0.05
        if features["uses_http"]:            score += 0.10
        if features["url_length"] > 100:     score += 0.05
        if features["digit_in_domain"]:      score += 0.08

        # Penalty: legitimate signals
        if features["uses_https"]:           score -= 0.05
        if features["legitimate_tld"]:       score -= 0.05

        # Bonus from extra indicators
        score += len(indicators) * 0.02

        return min(max(score, 0.0), 1.0)

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    def _safe_parse(self, url: str) -> Dict:
        try:
            parsed = urlparse(url)
            return {
                "scheme":   parsed.scheme,
                "hostname": parsed.hostname or "",
                "path":     parsed.path or "",
                "query":    parsed.query or "",
                "fragment": parsed.fragment or "",
                "domain":   (parsed.hostname or "").split(".")[-2] if "." in (parsed.hostname or "") else "",
            }
        except Exception:
            return {"hostname": "", "path": "", "query": "", "domain": ""}

    def _entropy(self, s: str) -> float:
        """Shannon entropy of a string — random strings have high entropy."""
        if not s:
            return 0.0
        freq = {c: s.count(c) / len(s) for c in set(s)}
        return -sum(p * math.log2(p) for p in freq.values())

    def _detect_brand_impersonation(
        self, domain: str, subdomains: List[str], tld: str
    ) -> bool:
        """Check if domain uses a brand name but isn't the real brand."""
        all_parts = [domain] + subdomains
        for brand, legit_domain in BRAND_DOMAINS.items():
            for part in all_parts:
                if brand in part:
                    legit_tld = "." + legit_domain.split(".")[-1]
                    if tld != legit_tld:
                        return True
                    if domain != brand:
                        return True
        return False

    def _risk_level(self, score: float) -> str:
        if score >= 0.7:  return "CRITICAL"
        if score >= 0.5:  return "HIGH"
        if score >= 0.3:  return "MEDIUM"
        if score >= 0.15: return "LOW"
        return "SAFE"


class EmailHeaderAnalyzer:
    """
    Forensic analysis of email headers.

    Checks SPF, DKIM, DMARC alignment and sender authenticity.
    """

    def analyze(self, headers: Dict[str, str]) -> Dict:
        """
        Analyse parsed email headers for spoofing indicators.

        Args:
            headers: dict from EmailPreprocessor.extract_header_features()
        """
        indicators = []
        score = 0.0

        sender = headers.get("from", "")
        reply_to = headers.get("reply_to", "")
        return_path = headers.get("return_path", "")
        spf = headers.get("received_spf", "").lower()
        auth = headers.get("authentication_results", "").lower()

        # SPF check
        if "fail" in spf:
            indicators.append("SPF FAIL — sender not authorised to send from this domain")
            score += 0.35
        elif "softfail" in spf:
            indicators.append("SPF SOFTFAIL — possible spoofing")
            score += 0.20
        elif "pass" in spf:
            score -= 0.10

        # DKIM check
        if "dkim=fail" in auth:
            indicators.append("DKIM signature invalid — email may be modified or spoofed")
            score += 0.30
        elif "dkim=pass" in auth:
            score -= 0.10

        # DMARC check
        if "dmarc=fail" in auth:
            indicators.append("DMARC policy failure — domain impersonation likely")
            score += 0.30
        elif "dmarc=pass" in auth:
            score -= 0.15

        # Reply-to mismatch (common in phishing)
        if reply_to and sender:
            sender_domain = self._extract_domain(sender)
            reply_domain = self._extract_domain(reply_to)
            if sender_domain and reply_domain and sender_domain != reply_domain:
                indicators.append(
                    f"Reply-To domain ({reply_domain}) differs from sender ({sender_domain})"
                )
                score += 0.25

        # Return-path mismatch
        if return_path and sender:
            rp_domain = self._extract_domain(return_path)
            from_domain = self._extract_domain(sender)
            if rp_domain and from_domain and rp_domain != from_domain:
                indicators.append("Return-Path domain differs from From domain")
                score += 0.15

        return {
            "header_risk_score": round(min(score, 1.0), 4),
            "indicators": indicators,
            "spf_result": "pass" if "pass" in spf else "fail" if "fail" in spf else "none",
            "dkim_result": "pass" if "dkim=pass" in auth else "fail" if "dkim=fail" in auth else "none",
            "dmarc_result": "pass" if "dmarc=pass" in auth else "fail" if "dmarc=fail" in auth else "none",
        }

    def _extract_domain(self, email_str: str) -> str:
        match = re.search(r"@([\w.-]+)", email_str)
        return match.group(1).lower() if match else ""

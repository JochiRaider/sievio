# Security Policy

This project ingests content from local repositories and remote sources (e.g., GitHub repositories and web-hosted PDFs). As a result, it includes explicit defenses against common ingestion risks such as SSRF, unsafe redirects, path traversal, symlink attacks, and archive “zip bomb” patterns. Please report any suspected vulnerabilities responsibly.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Preferred reporting path:
1. **GitHub Security Advisories** (Private Vulnerability Reporting), if available in this repository:
   - Use the repository’s **“Report a vulnerability”** / **Security** tab flow.
2. If private reporting is not available, contact the maintainers through a **private channel listed in the repository metadata** (e.g., a maintainer email on the GitHub profile), and reference this repository.

When reporting, include as much of the following as possible:
- A clear description of the impact (e.g., SSRF, credential leakage, arbitrary file read/write, RCE).
- A minimal reproduction (sample config/TOML and command invocation if relevant).
- Affected component(s) and inputs (URLs, file paths, archive structure, etc.).
- Any constraints you observed (e.g., only under certain config flags).
- Logs and stack traces with **secrets removed**.

We will acknowledge reports as soon as practical and coordinate a fix and disclosure plan with the reporter.

## Supported Versions

Security fixes are applied to the latest release line. If you are using an older version, we recommend upgrading to the newest published version before validating whether an issue is still present.

## Security-Relevant Design Notes

This section documents existing security controls to help reporters and users reason about expected behavior.

### Network fetching (SSRF / redirect controls)

Remote fetches are expected to be routed through the project’s safe HTTP client. The default policy is designed to:
- Block non-global (private, loopback, link-local, multicast, unspecified) IPs during DNS resolution.
- Restrict redirects to related hosts or an explicit allowlist, and block HTTPS → HTTP downgrade redirects.
- Drop sensitive headers (e.g., `Authorization`, cookies) on cross-host redirects.

If you find any input path that performs remote HTTP(S) requests without these controls—or bypasses host/IP/redirect checks—please report it.  

### GitHub ingestion

GitHub API calls and zipball downloads are performed via the safe HTTP client and include additional safeguards such as:
- Streamed zipball download with size caps.
- Archive extraction defenses, including member-count limits, total-uncompressed limits, per-file caps, compression-ratio checks, and rejection of unsafe paths and symlinks inside archives.

### Web PDF ingestion

Web PDF sources validate and normalize URLs and reject embedded credentials in URLs. Downloads are streamed with hard caps and (optionally) content sniffing to ensure the payload looks like a PDF.

### Local filesystem traversal

Local repository traversal is designed to avoid common filesystem attacks:
- Enforces root containment and rejects traversal segments.
- Avoids following symlinks by default, and includes checks to prevent opening files outside the configured root.
- Honors `.gitignore` (when enabled) and prunes ignored directories from traversal.

## Coordinated Disclosure

If you report a vulnerability, please keep details confidential until a fix is available. We will work with you to:
- Validate the report.
- Develop and test a fix.
- Publish a release and a security advisory / changelog entry describing the impact and remediation.

## Security Hardening Guidance for Users

If you are operating in a high-risk environment (untrusted inputs), consider:
- Running ingestion in a restricted environment (container / VM) with least-privilege filesystem access.
- Using strict allowlists for remote domains when possible.
- Setting conservative caps for remote downloads and archive processing.
- Avoiding execution of untrusted code; treat ingested artifacts as data.

## Acknowledgements

We appreciate responsible disclosure and will credit reporters in release notes or advisories when requested (subject to reporter preference).
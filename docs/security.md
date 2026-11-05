# Security

research-forest handles a lot of untrusted input: paper metadata, abstracts, citation contexts,
snippets, downloaded PDFs and web pages. Anyone can publish a paper, so all of it is treated as
hostile. Whatever an LLM produces from that input is treated as untrusted as well.

| Threat | Main defences |
|---|---|
| Prompt injection via papers, snippets or web pages | Data fencing, sanitisation, bounded schema-validated outputs, tool-less agents |
| Cross-site / DNS-rebinding attacks on the local server | Token auth (`SameSite=Strict` cookie), same-origin writes |
| Stored XSS from hostile metadata | HTML escaping, http(s)-only links, strict CSP |
| SSRF via open-access PDF URLs | Public-address-only guard on every redirect hop |
| Malicious PDFs or HTML | Isolated, resource-limited worker process (+ bubblewrap) |
| Cost or resource exhaustion | Input limits, output caps, refinement fan-out caps, job limit |

## Prompt injection

- **Fencing.** Third-party and LLM-derived text only ever appears inside
  `<untrusted_data source="…">` blocks. The system prompt declares those blocks to be inert data
  that can never change the task, the ratings, which items get selected or removed, or the
  output format. The user's own words go in separate `<user_request>`, `<user_command>` and
  `<user_question>` blocks.
- **Sanitisation** ([`safety.py`](../research_forest/safety.py)). Before interpolation, text has control characters, zero-width
  characters, bidi-override and Unicode "tag" characters stripped. These can hide instructions
  from humans while the model still sees them. Anything resembling our delimiter tags is
  removed too, so data can never close its own block. Every field is length-clipped.
- **Bounded outputs** ([`models.py`](../research_forest/models.py)). Every LLM response is validated against a strict schema.
  Strings are sanitised and length-capped, lists are count-capped, numbers are clamped (for
  example, a refinement can add at most 50 papers per op and runs at most 8 ops), and enums
  are enforced. The same rules apply when LLM output is fed back into later prompts, which
  covers second-order injection.
- **Reference-only selection.** The LLM refers to items only through the integer refs it was
  given. Unknown or out-of-range refs and node handles are dropped.
- **Damage control.** Every refinement takes a snapshot first (so it can be undone), and the
  log warns when a refinement removes more than half the graph. Instruction-like text in a
  paper is logged as a warning but kept as data, because papers *about* prompt injection
  legitimately contain such phrases.
- **Agent backends.** Claude Code runs with `--tools ""`, `--strict-mcp-config` (no MCP
  servers) and `--disable-slash-commands`, in an empty temporary directory. Injected text
  therefore has no tools to act through. Codex runs in its `read-only` sandbox (no writes,
  no network). Note that Codex can still *read* local files, so its output is only ever used
  as validated, length-capped data. Prefer `claude-code` or `openai` when you process
  untrusted corpora on a machine holding secrets.

## Web server

- **Token auth.** Every `/api` call needs an access token. `research-forest serve` prints a URL
  containing it. Opening that URL swaps the token for an `HttpOnly`, `SameSite=Strict` cookie
  and strips it from the address bar. Scripts can send `Authorization: Bearer <token>`
  instead. Set `RF_API_TOKEN` to use a fixed token. This blocks other local users,
  cross-site request forgery (CSRF) and DNS-rebinding attacks from websites you visit.
- **Same-origin writes.** State-changing requests with a foreign `Origin` header are rejected.
- **Security headers.** A strict CSP allows no inline or third-party scripts (so even an
  injected `onerror=` handler cannot run). Framing is denied, along with `nosniff`, `no-referrer`
  and `no-store` for API responses. The OpenAPI and docs endpoints are disabled.
- **Output encoding.** All data is HTML-escaped in the UI. Only `http(s)` links are rendered,
  both server-side and client-side, so `javascript:` URLs never become links. A browser test
  ([`tests/test_ui_xss.py`](../tests/test_ui_xss.py)) seeds hostile content straight into the database and checks that
  nothing executes.
- **Limits.** Request sizes are bounded, at most `RF_MAX_JOBS` (default 2) builds or refinements
  run at once, node ids are validated, and the full-text search query syntax is neutralised.
- Binding to a non-loopback address prints a warning. The traffic is plain HTTP, so put a
  TLS reverse proxy in front if you expose the server.

## Sandboxed full-text processing

Downloads and PDF parsing run in [`sources/fulltext_worker.py`](../research_forest/sources/fulltext_worker.py), a separate process with these
properties:

- **Isolated Python.** It is started with `python -I` (no `PYTHONPATH`, no user site-packages,
  no cwd on `sys.path`) and imports nothing from this package.
- **Minimal environment.** It runs in a throwaway temp directory with only `PATH`, locale, proxy
  and CA variables set. API keys are *not* passed in.
- **Resource limits.** `RLIMIT_AS` (memory), `RLIMIT_CPU`, `RLIMIT_FSIZE` (almost no disk writes)
  and no core dumps, plus a wall-clock timeout. Extracted text is capped, and the output is
  re-validated by the parent.
- **SSRF protection.** Only `http(s)` on the default ports is allowed, with no credentials in
  the URL. Every host must resolve *only* to public addresses: loopback, private,
  link-local/cloud-metadata, CGNAT, ULA and multicast addresses are all refused. This is checked
  again on **every redirect hop** (at most 5). If an egress proxy is configured and the host
  can't be resolved locally, resolution is left to the proxy.
- **Content limits.** Only PDF, HTML or plain-text responses are accepted. The size limit
  (`RF_FULLTEXT_MAX_MB`) is enforced on *decompressed* bytes, which defeats gzip bombs, and
  documents are kept in memory only.
- **bubblewrap** (`bwrap`), when installed: read-only system and Python paths, your home
  directory hidden, a private `/tmp`, and every namespace unshared except the network.

## Residual risks

- DNS answers can change between the SSRF check and the actual connection (rebinding races).
  bubblewrap does not close this, because it shares the network namespace.
- LLM-based defences reduce prompt-injection risk but cannot eliminate it. The deterministic
  bounds above limit what a successful injection can do.


## Reporting a vulnerability

Please open a private security advisory on GitHub rather than a public issue.

# Local MCP

Stdio only. This process does not open a port and does not use the network. ChatGPT Desktop starts it.

The filled-in copy sheet is [COPY.txt](COPY.txt). Use that file. Do not paste a path into the ChatGPT website form "Add custom MCP server". That form wants a public URL, and this server has none.

From the repo root, once, and again after `git pull`. Python 3.12 matches the Dockerfile. The system `python3` does not have this repo's packages, which is why Codex stops on `No module named 'numpy'`.

```bash
git pull
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
chmod +x mcp/run.sh
```

`mcp/run.sh` then starts `.venv/bin/python` by absolute path. It does not look at `PATH`.

In ChatGPT Desktop: Settings, MCP servers, Add server.

| Field | Value |
| --- | --- |
| Name | tqqq-book |
| Transport | STDIO |
| Command | the absolute path to `mcp/run.sh` |
| Arguments | leave blank |

Do not enter a URL. Do not point the command at `src/tqqq_book/mcp_server.py`. Save, then Restart. In a chat, `/mcp` shows whether it connected.

The same process can be written in `~/.codex/config.toml`, which the desktop app shares with Codex:

```toml
[mcp_servers.tqqq-book]
command = "/absolute/path/to/montecarlo-risk-engine/mcp/run.sh"
```

After a newer `main`, pull and reinstall into the same venv, then restart Codex:

```bash
git pull
.venv/bin/python -m pip install -r requirements.txt
```

`tqqq_book` prices the opening puts and calls. Put DTE and horizon days are separate knobs. `tqqq_monte_carlo` returns the P&L distribution and a day-by-day 5th/50th/95th ribbon for spot and P&L. `tqqq_report` writes `runs/tqqq-desk.html` with those bands and the premium on every leg. Defaults are 100 lots, put DTE 180, horizon 100 days. A premium is the analytic Black-Scholes price, not a broker quote. Nothing is ordered. The skill text is [skill/tqqq-book/SKILL.md](skill/tqqq-book/SKILL.md).

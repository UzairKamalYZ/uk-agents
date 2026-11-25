# Deployment

`deploy.sh` runs uk-agents on a machine and keeps it on the latest `main`. One script, no extra tooling: it needs only `git` and `uv`.

It starts four services:

| Service | Command it runs | Reachable at |
|---|---|---|
| REST API | `uv run python -m clients.rest.main` | `http://<host>:8000` |
| Telegram bot | `uv run python -m clients.telegram.main` | your bot, in Telegram |
| Streamlit UI | `uv run python -m clients.streamlit.main` | `http://<host>:8501` |
| Property crawler | `uv run python -m property_crawler.main` | — (fills Pinecone every hour; see [property_crawler/README.md](../property_crawler/README.md)) |

Then, every minute, it:

1. restarts any service that has died;
2. asks GitHub whether `main` has moved. If it has, it pulls the new commits, builds (`uv sync --locked`), runs the test suite, and restarts the services on the new code. With the crawler among the services, every build also runs `playwright install chromium`, so the browser always matches the Playwright version (it downloads only when that changed).

A commit that fails to build or fails its tests is **not deployed**: the checkout is rolled back and the services keep running the last good version. The script tries again when `main` moves on.

## One-time setup

On the machine that will run the services:

1. Install the prerequisites:
   - `git`
   - [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.13 itself)
   - Node.js with `npx` (the MCP servers in `mcp.json` are started with it)
   - Only when `ANTHROPIC_API=false`: the `claude` CLI, logged in
   - For the property crawler, the system libraries Chromium needs (Ubuntu 22.04 / 24.04). After step 2, from the clone:

     ```bash
     uv sync && sudo .venv/bin/playwright install-deps chromium
     ```

2. Clone the repository into a folder used only for deployment, and leave it on `main`:

   ```bash
   git clone https://github.com/UzairKamalYZ/uk-agents.git ~/uk-agents
   cd ~/uk-agents
   ```

   The clone must be able to `git fetch` without asking for a password. For a private repository, use an SSH URL with a [deploy key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys), or run `gh auth login` once.

3. Create the configuration:

   ```bash
   cp .env-template .env
   ```

   Fill in at least `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, and the Pinecone settings. If the machine is reachable from the internet, set `API_KEY` too — the REST API is open without it, and the Streamlit UI sends it automatically.

## Run it

```bash
./deployment/deploy.sh
```

It stays in the foreground and prints what it does. Ctrl-C stops it and the services.

```
2026-11-16 22:19:45 [deploy] building (uv sync)
2026-11-16 22:19:45 [deploy] started rest (pid 9522)
2026-11-16 22:19:45 [deploy] started telegram (pid 9523)
2026-11-16 22:19:45 [deploy] started streamlit (pid 9524)
2026-11-16 22:19:45 [deploy] running 6616842; checking origin/main every 60s
2026-11-16 22:24:54 [deploy] new commit on main: 2f5b7ae improve finance prompt
2026-11-16 22:24:54 [deploy] building (uv sync)
2026-11-16 22:24:54 [deploy] running tests
2026-11-16 22:25:10 [deploy] deploying 2f5b7ae
```

If the checkout is already behind `main` when the script starts, it updates first and starts on the newest commit (the tests are not run for that first start).

## Keep it running after logout and reboot (Linux)

`uk-agents.service` is a systemd user unit that runs `deploy.sh` and restarts it if it ever exits. It assumes the clone is at `~/uk-agents`; edit the two paths in the file if yours is elsewhere.

```bash
mkdir -p ~/.config/systemd/user
cp deployment/uk-agents.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now uk-agents
sudo loginctl enable-linger "$USER"     # start at boot, without anyone logging in
```

| To | Run |
|---|---|
| See whether it is running | `systemctl --user status uk-agents` |
| Follow the deploy log | `journalctl --user -u uk-agents -f` |
| Stop everything | `systemctl --user stop uk-agents` |
| Restart everything | `systemctl --user restart uk-agents` |

On macOS, or anywhere without systemd, run it in `tmux`/`screen`, or in the background:

```bash
nohup ./deployment/deploy.sh >> logs/deploy.log 2>&1 &
```

## Stop it

Each of these stops the script and all the services.

| Started with | Stop with |
|---|---|
| systemd | `systemctl --user stop uk-agents` |
| `./deployment/deploy.sh` in a terminal | Ctrl-C in that terminal |
| the `nohup` command | `pkill -TERM -f deployment/deploy.sh` |

Allow up to a minute: each service gets `DEPLOY_STOP_TIMEOUT` seconds (default 20) to shut down before it is killed. Don't use `kill -9` on the script — that leaves the services running without it.

With systemd, `stop` lasts until the next reboot. To keep it from starting again at boot:

```bash
systemctl --user disable uk-agents
```

## Logs

| File | What is in it |
|---|---|
| `logs/agent.log` | The application log of the REST API, Telegram bot and Streamlit UI (rotating) |
| `logs/crawler.log` | The property crawler's log: every pass, every site (rotating) |
| `logs/deploy-rest.log`, `logs/deploy-telegram.log`, `logs/deploy-streamlit.log`, `logs/deploy-crawler.log` | Raw output of each service since its last start — look here when a service won't start |
| `logs/deploy-tests.log` | Output of the last test run — look here when a commit was not deployed |
| `logs/deploy-playwright.log` | Output of the last `playwright install chromium` |

## Settings

Environment variables for `deploy.sh`; all optional.

| Variable | Default | Meaning |
|---|---|---|
| `DEPLOY_SERVICES` | `rest telegram streamlit crawler` | Which services to run. `crawler` is the property crawler; every other name is a folder in `clients/`, so a new client is deployed by adding its name here (e.g. `"rest telegram streamlit crawler cron"`) |
| `DEPLOY_BRANCH` | `main` | Branch to follow; the checkout must be on it |
| `DEPLOY_REMOTE` | `origin` | Git remote to fetch from |
| `DEPLOY_POLL_SECONDS` | `60` | How often to check GitHub and the services |
| `DEPLOY_RUN_TESTS` | `true` | `false` deploys without running the tests |
| `DEPLOY_STOP_TIMEOUT` | `20` | Seconds a service gets to shut down before it is killed |

With systemd, set them in the unit file, for example `Environment=DEPLOY_SERVICES="rest telegram"`.

## Good to know

- **Only merges to `main` are deployed.** The script fast-forwards and never touches local changes; if the checkout has commits of its own that `main` doesn't, it stops updating and keeps running what it has. Don't develop in the deployment clone.
- **A deploy restarts the services**, so they are unavailable for a few seconds, and a request in flight at that moment is cut off. Conversation history, expenses and paused confirmations are in SQLite files and survive.
- **Changes to `.env` are not picked up automatically** — it is not in git. Restart after editing it (`systemctl --user restart uk-agents`, or Ctrl-C and run the script again).
- **REST and Telegram each load their own copy of the agents** (embedding model, MCP servers; Telegram also loads Whisper). The crawler loads the embedding model too, and runs Chromium during a pass. Allow roughly 3–4 GB of memory for the four services.
- **The FBR tax agent drives Chrome on the same machine**, so it works only where a desktop Chrome with the Claude extension is available — not on a headless server. The other agents are unaffected.
- **The WhatsApp listener and mail monitor are not started.** They are separate services (`python -m agents.whatsapp.main`, `python -m agents.mails.main`), not clients; run them alongside if you use them.
- **A deploy restarts the crawler**, which then starts a pass at once. That is cheap: it only embeds listings it hasn't seen. `DEPLOY_SERVICES="rest telegram streamlit"` leaves it out.

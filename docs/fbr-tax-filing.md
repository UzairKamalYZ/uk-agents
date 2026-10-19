# FBR-PK Tax Filing agent

Files your Pakistan income tax return (Form 114(1)) and wealth statement (Form 116) on FBR IRIS from a Telegram chat. It does this by running your own `fbr-iris-filing` skill (in `~/work/fbr-tax-filing`) in a Claude Code session that controls Chrome on your Mac.

## What you need

- The Mac awake, Chrome open, and the Claude in Chrome extension connected.
- `claude` logged in on the Mac. The agent always uses that login, not `ANTHROPIC_API_KEY`.
- `TELEGRAM_OWNER_ID` set in `.env` to your Telegram user id (from @userinfobot). The agent is owner-only and refuses everyone while this is empty.
- Optional: set `FBR_TAX_DIR` if the filing project lives somewhere else.

## A filing, message by message

```
you:  file my FBR return
bot:  🤖 FBR-PK Tax Filing is working on your query…
bot:  Opened IRIS in Chrome. Please log in, then tell me when you're on the home screen.
you:  logged in
bot:  Read-only (capture last year's return) or full filing? Tax year 2027?
you:  full, 2027
bot:  … asks for 30 June bank balances / foreign remittance, one group at a time …
you:  HBL Rs 450,000
bot:  Checkpoint: last year vs this year … Approve saving this as a draft in IRIS?
you:  yes
bot:  Draft saved in IRIS. Review it and submit it yourself in IRIS,
      or send "submit the return" if you want me to open the declaration.
```

- **Each message is one step.** The agent resumes the same Claude Code session, so it remembers where the filing is. After 12 hours without a message, or after an error, the next message starts a new session. The skill's checkpoint file (`snapshots/<year>.draft.json`) lets it resume from the last completed step.
- **You log in.** It never types your CNIC, password, OTP or PIN.
- **Submission.** It stops at Save Draft. It clicks Submit only when your own message says so, and even then you type the 4-digit e-filing PIN yourself.
- **One step at a time.** There is one browser. A message sent while a step is running gets "Still working…".
- **A step can take minutes.** Reading a whole return is many page loads.

## Safety rails

- **Enforced in code (`agents/fbr_tax/runner.py`):**
  - no shell and no web tools;
  - files can be written only in `snapshots/` and the skill's `iris-notes.md`;
  - the only browser tools are Claude in Chrome's;
  - everything else is denied, not asked.
- **Enforced by the skill:** no invented amounts, stop on CAPTCHA, payment or unexpected screens, and the step-8 approval before saving.
- **Where tax data goes:**
  - Snapshots stay in the filing project.
  - The agent's replies, which can include amounts, go through Telegram and are stored in `sessions.db`.
  - With `LANGCHAIN_TRACING_V2=true`, replies are also sent to LangSmith.

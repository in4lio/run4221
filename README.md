# 42+21 (Full’n’Half)

42+21 is a Telegram bot [@run4221bot](https://t.me/run4221bot) and background tracker
for marathon and half marathon registration openings.

The project is still in active development, so features, commands, and data may change
quickly while the service is being shaped.

Updates are posted in the channel [@run4221](https://t.me/run4221) and on
[run4221.com](https://run4221.com).

![42+21 welcome](images/42+21_welcome.png)

## Contribution Notice

This repository is developing rapidly in close collaboration with Codex. Unfortunately,
until the project reaches a stable state, external pull requests are disabled and will not
be reviewed or accepted.

## Quick Start

Use Python 3.12 or newer.

This project uses `uv` for Python environment, dependency management, and command execution.

Install `uv` if needed:

```bash
brew install uv
```

Create/update the project environment:

```bash
uv sync --extra dev
```

Create local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set `TELEGRAM_BOT_TOKEN` to the token from BotFather. Do not commit `.env`.
The default local database is `data/run4221.sqlite3`; it is created on startup.

To enable OpenAI extraction for `/add_event`, set:

```env
AI_EXTRACTOR_PROVIDER=openai
RUN4221_PROMPT_SOURCE=file
RUN4221_PROMPT_DIR=private/prompts
OPENAI_API_KEY=...
OPENAI_EXTRACT_MODEL=gpt-5.4-mini
```

Run the bot locally:

```bash
uv run python -m run4221
```

Public channel delivery is disabled by default. After the bot is an administrator of the
configured channel, enable the production-light publisher explicitly:

```env
TELEGRAM_CHANNEL_ID=@run4221
TELEGRAM_CHANNEL_POSTING_ENABLED=true
TELEGRAM_CHANNEL_POLL_SECONDS=60
```

Then open Telegram and send `/start` to [@run4221bot](https://t.me/run4221bot).

## Commands Available Now

Public:

- `/start` - welcome message
- `/help` - short command list
- `/list_events` - list all tracked events
- `/list_open` - list events with currently open registration
- `/search_events` - search tracked events
- `/show_event` - show event details
- `/suggest` - suggest a new event for tracking

Moderator:

- `/add_event` - add an event from URL
- `/archive_event` - archive an active event
- `/delete_event` - permanently delete an event
- `/edit_event` - edit event fields
- `/list_archive` - list archived events
- `/restore_event` - restore an archived event
- `/todo` - show pending updates, suggestions, and channel actions
- `/channel_drafts` - review public news drafts and delivery failures
- `/channel_correction <event-id>` - prepare an explicit correction draft for review
- `/update_event` - manually scan event registration status

- `/apply_update` - apply a pending update
- `/list_updates` - list pending updates
- `/next_update` - show the oldest pending update
- `/reject_update` - reject a pending update
- `/show_update` - show proposed update details

- `/apply_suggestion` - start event review from a suggestion
- `/list_suggestions` - list pending suggestions
- `/next_suggestion` - show the oldest pending suggestion
- `/reject_suggestion` - reject a pending suggestion
- `/show_suggestion` - show one pending suggestion


## General Workflow

The idea is simple: runners should not have to refresh marathon websites every few days
just to catch a registration opening. 42+21 keeps a watchlist of running events, checks
official pages, and turns important changes into useful updates.

Users can follow the public stream in [@run4221](https://t.me/run4221) or on
[run4221.com](https://run4221.com). The bot will also support personal event
notifications, so a runner can subscribe to one specific race and get a direct message
when something important changes.

If an event is missing, a user can suggest it with `/suggest`. The suggestion goes into a
review queue, where a moderator or AI-assisted workflow checks the official source,
confirms the event details, and adds it to tracking when it looks right.

The long-term workflow should be handled mostly by AI agents. One agent discovers event
profiles from official URLs, another keeps checking registration pages, and another can
prepare update drafts for moderator review. Humans stay in the loop for uncertain cases,
new event approval, and anything that could affect public posts.

Once an event is tracked, the bot monitors it in the background. When registration opens,
a date changes, a race sells out, or useful new information appears, the system prepares
an update and publishes it to the channel and website after the required review.

The Telegram channel is a news feed, not an administrative log. First announcements and
detected changes require a separate public-preview action. Only reminders derived from
approved registration dates can publish automatically; edits, queue activity, archive,
restore, and delete actions stay silent.

Every public post starts with the event name, distance, date, and location, followed by a
short emoji-labelled update. Registration-date changes include the new value; unchanged
registration data produces no news item. Approved opening and closing dates can also
produce deterministic `tomorrow` reminders.

If Telegram returns an unknown delivery result, the message stays in the moderator queue
and is never retried automatically. A moderator must first inspect the channel, then record
either `Already published` or `Confirmed absent — retry`. Explicit corrections can be
prepared with `/channel_correction <event-id>` and follow the same preview requirement.

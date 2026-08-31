# Telegram Message Style Guidance (Draft)

This document defines the first shared style rules for Telegram bot messages.

## Principles

- Start major standalone title and detail messages with a short bold title prefixed by `✨`, for example `<b>✨ Tracked events</b>` or `<b>✨ Update #7</b>`.
- Put one blank line after major detail titles before field rows.
- Start smaller operational prompts with a short bold title that names the message intent.
- Keep short one-line notifications plain, for example `Cancelled.` or `Nothing to cancel.`
- Prefix warning messages with `⚠️ <b>Warning</b>:` so they are easy to notice.
- Use `<code>` only for values the user can or should copy exactly, especially event IDs in lists and details.
- For `Field: Value` pairs, format the field label in bold. Format ID values as `<code>`, tag values as `<u>`, and other values as plain escaped text.
- Use `<i>` for secondary hints, temporary notes, and all examples.
- Use `<blockquote>` only for source checks, evidence, or compact diagnostics.
- Use `<pre><code class="language-python">` only for literal code or structured diagnostics.
- Use Telegram links for the channel and bot handles when they are shown in message text.
- Use reply-keyboard buttons for standard dialog inputs such as `Cancel`, `ok`, and `-` while the bot waits for typed input.
- Use inline buttons for actions attached to a specific message, such as `Show`, `Apply`, `Reject`, `Confirm`, `Archive`, and `Restore`.
- Do not print workflow commands as action hints when a button can perform the action.
- Remove reply-keyboard buttons when a guided flow finishes or is cancelled.
- In draft confirmation prompts, underline the proposed value under review; keep public IDs in `<code>`.
- Keep `/help` focused on command documentation, not long explanations.

## Allowed HTML Styles

- `<b>Title or important label</b>`
- `<i>Secondary note or example</i>`
- `<u>Needs review</u>`
- `<s>Old value</s>`
- `<code>copyable_event_id</code>`
- `<pre><code class="language-python">...</code></pre>`
- `<span class="tg-spoiler">hidden detail</span>`
- `<blockquote>Source check or diagnostics</blockquote>`
- `<a href="https://t.me/run4221">@run4221</a>`

## Standard Patterns

### Help Message

Start `/help` with a plain sentence that says what the bot can do, following the
BotFather-style pattern: intro, "You can control me...", then grouped command sections.
Keep the public channel as a Telegram link in the intro.

```html
I can help you track marathon and half marathon registration openings. Public updates are posted in <a href="https://t.me/run4221">@run4221</a>.

<b>Service is currently in implementation stage.</b>

You can control me by sending these commands:

/start - show this message
/help - show this help

<b>Events</b>
/list_events - list tracked events
```

### Status Card

Use a bold title, then compact field lines. Field labels are bold. ID values use
`<code>`, tag values use `<u>`, and ordinary values stay as plain text.

```html
<b>✨ Todo</b>
<b>Updates</b>: 2
<b>Suggestion</b>: 5
```

### List Caption

Send list captions as a separate bold message, then send each card as its own message
with its own buttons. The visual separation comes from Telegram message spacing, not an
extra blank line inside one message.

```text
message 1: <b>✨ Tracked events</b>
message 2: <b>EDP Lisbon Half Marathon</b>
           Lisbon, Portugal | Half marathon | 2026-03-08
           <b>Tags</b>: <u>pt, eu, 21</u>
           <b>ID</b>: <code>lisbon.21</code>
```

### Queue Item

Use stable record handles for update items, keep event IDs in code, and attach action
buttons under the item. Avoid visible list positions for updates because queue order can
change after another update is applied or rejected. Keep status-like metadata as normal
text unless it is meant to be copied.

```html
<b>Update #17</b>
<b>Event ID</b>: <code>berlin.42</code>
<b>Type</b>: registration_window
<b>Confidence</b>: 0.91
<b>Fields</b>: registration_status, registration_url
```

Update detail panels should show diffs as stacked old/new values, not as one long
inline arrow. Use strikethrough only for the old value.

```html
<b>What's changed</b>
- <b>registration_status</b>
  <s>unknown</s>
  open
- <b>registration_url</b>
  <s>https://example.com/old</s>
  https://example.com/new
```

### Button Placement

Use two different button styles intentionally.

Reply-keyboard buttons appear near the Telegram input field. Use them only when the bot
is waiting for the next typed value and the button is a standard input shortcut.

```text
Cancel
ok | Cancel
ok | - | Cancel
```

Inline buttons appear under a specific bot message. Use them when the action belongs to
that message or queue item.

```text
Show
Apply | Reject
Confirm | Cancel
Archive | Cancel
Restore | Delete
```

Do not mix the two roles: reply-keyboard `Cancel` belongs near the message input during
input flows; message-specific `Cancel` belongs inline only for confirmation/action
messages. The typed `/cancel` command should still work.

### Editable Management Panel

For queue-style management screens, prefer replacing the existing bot message instead of
sending a new message for every action. This keeps old action buttons out of the chat.

Moderator management panels follow this pattern:

```text
Queue/list
  title message: <b>✨ Pending updates</b>
  item messages: one card per item, each with its own Show button

Show -> replace with detail
  title: <b>✨ Update #7</b>
  blank line after title
  buttons: Apply | Reject
           Apply | Reject
           Restore | Delete
           Back

Action -> replace with confirmation dialog
  buttons: Confirm | Cancel

Confirm -> perform action, then replace with result message
  buttons: none

Back -> return to the previous panel, card, or list for that workflow
```

Use this for update, suggestion, archive, restore, and delete management. Deletion is
permanent, so it should use a preview plus a final confirmation before performing the
action.

Pending updates, pending suggestions, and archived events should visually match
`/list_events`: a separate bold title message, then one compact card per record with the
action button directly below that card.

Use this pattern for moderator queues when one editable panel can represent the whole
workflow. Use new messages only when the result is meant to remain as a separate audit
note or when Telegram cannot edit the original message.

### Confirmation

Use a direct title, show exactly what will change, and attach `Confirm` plus `Cancel`.
Avoid asking users to type confirmation text unless the action is destructive.

```html
<b>✨ Confirm apply #7</b>

<b>Event</b>: <code>berlin.42</code>
<b>Action</b>: apply
```

### Source Check

Use this block when evidence matters. Keep local filesystem paths out of Telegram
messages; show snapshot filenames only. Researcher-created queue records use the same
compact block in suggestion/update details and again in apply, partial-apply, reject,
and final new-event confirmations. Show only the bounded evidence summary, HTTPS source,
capture time, run ID, artifact basename, and short hash prefix; never show raw queue
markers, absolute paths, or raw audit files.

```html
<blockquote><b>Source check</b>
<b>Source</b>: https://example.com/race
<b>Captured</b>: 2026-08-31T14:00:00+00:00
<b>Evidence</b>: Registration is open.
<b>Trust</b>: stored approved event source
<b>Run ID</b>: <code>2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc</code>
<b>Artifact</b>: 20260831T140000Z-page.json
<b>Hash</b>: <code>bbbbbbbbbbbb</code></blockquote>
```

System-authored suggestions display `<b>From</b>: Researcher worker`. Suggestions from
Telegram subscribers keep their existing submitter and note formatting.

### Input Prompt

Use a bold caption with the input marker when the bot waits for the next reply. Keep the
instruction as a simple sentence, and put examples in italic, not code. Standard dialog
inputs such as `Cancel`, `ok`, and `-` should be reply-keyboard buttons under the
message input field, not command text printed in the message body.

```html
<b>💬 Update event</b>
Send an event ID.
<b>Example</b>: <i>berlin.42</i>
```

Reply keyboard:

```text
ok | - | Cancel
```

Common mappings:

- Missing command parameter, for example `/show_event`: reply keyboard `Cancel`.
- Draft confirmation, for example `/add_event` field review: reply keyboard `ok | Cancel`.
- Distance input or distance draft confirmation: reply keyboard `42 | 21 | 42,21`, plus `ok` when confirming a draft value.
- Optional or unknown value, for example event date or registration URL: reply keyboard `ok | - | Cancel`.
- Optional note in `/suggest`: reply keyboard `- | Cancel`.
- Finished, cancelled, or failed flow: remove the reply keyboard.

Current input marker: `💬`. It marks messages that are waiting for a typed reply or
a reply-keyboard button.

Draft extraction starts with a major title, then each field is confirmed in a separate
input prompt:

```html
<b>✨ Draft extracted from URL</b>

<b>💬 Event name</b>
<b>Draft</b>: <u>TCS Amsterdam Marathon</u>
Reply ok to keep it, or send the corrected value.
```

### Open questions

1. Develop styles
1. Confirm button is active even after Cancel selected.
   General confirmation conception is wrong.
1. When button? when just \command?
1. Waiting input and button vs button without waiting?
1. Emoji for command title + paragraph!?
1. IDs / tags / Headers style

"""Canonical task specs for the formal study (one spec per task class).

Each spec describes the same underlying task; ``clean_prompts.py`` derives
intent / workflow / action *suffixes* from these templates. The AndroidWorld
``task.goal`` (built from the official task template + sampled params) is always
the intent baseline — we only *append* workflow/action guidance so the total
information remains consistent with the canonical task definition.

Rules (see README):
- Three granularities are information-equivalent (no extra facts in workflow
  vs intent beyond structured decomposition).
- Action steps are UI-anchored; no pixel coordinates.
- Workflow lists sub-goals, not click-by-click detail.
- Perturbations are maintained separately (``perturbations.py``), not edited ad
  hoc in these strings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalTaskSpec:
    """Single source of truth for prompts derived for one registry task."""

    task_class: str
    goal_state: str
    param_summary: str
    allowed_ui_anchors: tuple[str, ...] = ()
    forbidden_shortcuts: tuple[str, ...] = ()
    workflow_template: str = ""
    action_template: str = ""
    notes: str = ""


# ── Main 6 ───────────────────────────────────────────────────────────────────

CLOCK_TIMER_ENTRY = CanonicalTaskSpec(
    task_class="ClockTimerEntry",
    goal_state="Timer shows the target duration; timer is not running.",
    param_summary="hours, minutes, seconds from task.goal",
    allowed_ui_anchors=("Clock app", "Timer tab", "numeric keypad"),
    forbidden_shortcuts=("Starting the timer",),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open the Clock app.\n"
        "2. Switch to the Timer tab.\n"
        "3. Enter the specified duration using the on-screen keypad.\n"
        "4. Do not start the timer; it must remain set but not running."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open the Clock app from the launcher or app drawer.\n"
        "2. Tap the 'Timer' tab at the bottom of the screen.\n"
        "3. Use the numeric keypad to enter hours, minutes, and seconds so the "
        "display matches the duration in the task instructions.\n"
        "4. Verify the displayed duration before finishing.\n"
        "5. Do not tap 'Start'. The timer must remain idle."
    ),
)

SIMPLE_CALENDAR_ADD_ONE_EVENT = CanonicalTaskSpec(
    task_class="SimpleCalendarAddOneEvent",
    goal_state="One new event exists with correct date, time, title, description, duration.",
    param_summary="year, month, day, hour, event_title, event_description, duration_mins",
    allowed_ui_anchors=(
        "Simple Calendar Pro",
        "month view / day selection",
        "new event / FAB",
        "title, time, description fields",
    ),
    forbidden_shortcuts=("Shortcuts that skip required fields",),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open Simple Calendar Pro.\n"
        "2. Navigate to the correct calendar date.\n"
        "3. Create a new event.\n"
        "4. Set title, description, start time, and duration as specified.\n"
        "5. Save the event."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open Simple Calendar Pro from the launcher or app drawer.\n"
        "2. From the calendar view, move to the month that contains the target date; "
        "tap the correct day.\n"
        "3. Use the new-event control (often a '+' button) to create an event.\n"
        "4. Enter the event title and description from the instructions.\n"
        "5. Set the start time and end time or duration to match the requested length.\n"
        "6. Save using the checkmark or 'Save' control."
    ),
)

EXPENSE_ADD_SINGLE = CanonicalTaskSpec(
    task_class="ExpenseAddSingle",
    goal_state="Exactly one new expense row matches the target in Pro Expense.",
    param_summary="Full expense text is embedded in task.goal (ROW_OBJECTS).",
    allowed_ui_anchors=("Pro Expense", "+ / new expense", "amount, name, category, notes"),
    forbidden_shortcuts=("Importing from files unless task requires it",),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open Pro Expense.\n"
        "2. Start creating a new expense entry.\n"
        "3. Enter every field required by the task text (amount, name, category, notes).\n"
        "4. Save the entry."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open Pro Expense from the launcher or app drawer.\n"
        "2. Tap the '+' or 'add' control for a new expense.\n"
        "3. Fill the amount field to match the task.\n"
        "4. Fill the name/title field.\n"
        "5. Choose the category that matches the task.\n"
        "6. Fill the notes field with the specified text.\n"
        "7. Save with the checkmark or Save button."
    ),
    notes="Guidance is generic; all concrete values come from task.goal.",
)

FILES_MOVE_FILE = CanonicalTaskSpec(
    task_class="FilesMoveFile",
    goal_state="File appears under destination folder; gone from source.",
    param_summary="file_name, source_folder, destination_folder",
    allowed_ui_anchors=("Files app", "folder list", "Move to / context menu"),
    forbidden_shortcuts=(
        "Hard-coded emulator volume names as a trick (use navigation the UI exposes)",
    ),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open the Files app.\n"
        "2. Browse to '{source_folder}' and locate '{file_name}'.\n"
        "3. Start a move operation for that file.\n"
        "4. Choose '{destination_folder}' as the destination.\n"
        "5. Confirm the move."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open the Files app.\n"
        "2. Open the navigation drawer or storage browser and select the internal/"
        "device storage volume described in the task instructions (name may vary by device).\n"
        "3. Open the '{source_folder}' folder.\n"
        "4. Long-press '{file_name}' to select it, then choose 'Move' or equivalent.\n"
        "5. Navigate up and into '{destination_folder}'.\n"
        "6. Confirm 'Move here' or OK to finish."
    ),
)

RECIPE_ADD_SINGLE_RECIPE = CanonicalTaskSpec(
    task_class="RecipeAddSingleRecipe",
    goal_state="One new recipe in Broccoli matches all fields in task.goal.",
    param_summary="Full recipe text is embedded in task.goal (ROW_OBJECTS).",
    allowed_ui_anchors=("Broccoli recipe app", "New recipe", "title, times, ingredients"),
    forbidden_shortcuts=("Skipping required fields",),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open the Broccoli recipe app.\n"
        "2. Start a new recipe.\n"
        "3. Fill title, description, servings, times, ingredients, and directions per the task.\n"
        "4. Save the recipe."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open Broccoli from the launcher or app drawer.\n"
        "2. Tap New Recipe or the '+' control.\n"
        "3. Enter the title and description from the task text.\n"
        "4. Set servings and preparation time.\n"
        "5. Enter ingredients exactly as specified.\n"
        "6. Enter directions.\n"
        "7. Save with the checkmark or Save."
    ),
    notes="Guidance is generic; all concrete values come from task.goal.",
)

SIMPLE_SMS_SEND_RECEIVED_ADDRESS = CanonicalTaskSpec(
    task_class="SimpleSmsSendReceivedAddress",
    goal_state="SMS to name1 contains the address that name2 sent.",
    param_summary="name1, number, name2, message (address); see task.goal",
    allowed_ui_anchors=("Simple SMS Messenger", "conversation list", "compose field"),
    forbidden_shortcuts=("Fabricating an address not in the thread",),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open Simple SMS Messenger.\n"
        "2. Find the recent sender who just sent the address. The sender may appear "
        "as the contact name '{name2}' or as a raw phone number. Open that "
        "conversation and read the address.\n"
        "3. Start a new message to {name1}.\n"
        "4. Send that exact address as the message body."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open Simple SMS Messenger.\n"
        "2. From the conversation list, identify the sender who just sent the "
        "address. Match either the contact name '{name2}' or the corresponding "
        "raw phone-number thread if the name is not shown, then open it.\n"
        "3. Read the latest inbound message and note the full address text.\n"
        "4. Return to the conversation list.\n"
        "5. Start a new message to '{name1}'. If the picker shows phone numbers "
        "instead of names, select the recipient entry that matches {name1}.\n"
        "6. Type the address exactly as received.\n"
        "7. Send the message."
    ),
)

# ── Backup ───────────────────────────────────────────────────────────────────

MARKOR_MOVE_NOTE = CanonicalTaskSpec(
    task_class="MarkorMoveNote",
    goal_state="Note file lives under destination folder in Markor.",
    param_summary="file_name, source_folder, destination_folder",
    allowed_ui_anchors=("Markor", "folder browser", "move / cut-paste"),
    forbidden_shortcuts=("Deleting and recreating instead of move",),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open Markor.\n"
        "2. Go to '{source_folder}'.\n"
        "3. Select '{file_name}'.\n"
        "4. Move it to '{destination_folder}'.\n"
        "5. Confirm."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open Markor.\n"
        "2. Open the '{source_folder}' folder.\n"
        "3. Long-press '{file_name}' to select it.\n"
        "4. Choose Move from the toolbar or overflow menu.\n"
        "5. Navigate to '{destination_folder}'.\n"
        "6. Confirm move."
    ),
)

# ── Sanity / calibration only ────────────────────────────────────────────────

CONTACTS_ADD_CONTACT = CanonicalTaskSpec(
    task_class="ContactsAddContact",
    goal_state="New contact exists with requested name and number.",
    param_summary="name, number",
    allowed_ui_anchors=("Contacts", "Create contact", "name and phone fields"),
    forbidden_shortcuts=(),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open Contacts.\n"
        "2. Create a new contact.\n"
        "3. Enter the name '{name}'.\n"
        "4. Enter the phone '{number}'.\n"
        "5. Save."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open the Contacts app.\n"
        "2. Tap add / create new contact.\n"
        "3. Fill first/last or full name fields to match '{name}'.\n"
        "4. Enter phone '{number}'.\n"
        "5. Save."
    ),
)

SIMPLE_SMS_SEND = CanonicalTaskSpec(
    task_class="SimpleSmsSend",
    goal_state="Outbound SMS matches recipient and body in task.goal.",
    param_summary="number, message",
    allowed_ui_anchors=("Simple SMS Messenger", "recipient field", "send"),
    forbidden_shortcuts=(),
    workflow_template=(
        "Follow these sub-goals:\n"
        "1. Open Simple SMS Messenger.\n"
        "2. Start a new message to {number}.\n"
        "3. Type {message}.\n"
        "4. Send."
    ),
    action_template=(
        "Follow these UI-anchored steps:\n"
        "1. Open Simple SMS Messenger.\n"
        "2. Tap compose / new message.\n"
        "3. Enter recipient {number}.\n"
        "4. Enter body: {message}.\n"
        "5. Tap send."
    ),
)


CANONICAL_SPECS: dict[str, CanonicalTaskSpec] = {
    s.task_class: s
    for s in (
        CLOCK_TIMER_ENTRY,
        SIMPLE_CALENDAR_ADD_ONE_EVENT,
        EXPENSE_ADD_SINGLE,
        FILES_MOVE_FILE,
        RECIPE_ADD_SINGLE_RECIPE,
        SIMPLE_SMS_SEND_RECEIVED_ADDRESS,
        MARKOR_MOVE_NOTE,
        CONTACTS_ADD_CONTACT,
        SIMPLE_SMS_SEND,
    )
}

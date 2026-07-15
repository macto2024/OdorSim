"""Extra keyboard listener for the ternary e-nose token (teleop 4b).

robosuite's :class:`Keyboard` device owns the arm/gripper keys
(``arrows``, ``.;``, ``e/r``, ``y/h``, ``o/p``, ``space``, ``b``, ``s``,
``=``, ``Ctrl+q``). We layer a second ``pynput`` listener for the e-nose:

    ``3`` sample (one-shot -> auto sample then filter)   ``4`` idle    ``5`` filter

``sample`` is a momentary trigger that starts the auto sequence
(sample hold -> filter hold -> idle). ``idle`` / ``filter`` are latched and
also emit a one-shot *force* flag so teleop can cancel an in-flight sequence
immediately.
"""

from __future__ import annotations

SAMPLE, IDLE, FILTER = 1, 0, -1

# Digit keys away from robosuite's usual bindings; 0/1/2 are often contested.
KEY_SAMPLE = "3"
KEY_IDLE = "4"
KEY_FILTER = "5"

_TOKEN_NAME = {SAMPLE: "SAMPLE", IDLE: "idle", FILTER: "FILTER"}


class EnoseKeyState:
    def __init__(self):
        self._latched = IDLE
        self._pending_sample = False
        self._force: int | None = None
        self._listener = None
        self._last_key_event: str | None = None

    def reset(self) -> None:
        self._latched = IDLE
        self._pending_sample = False
        self._force = None
        self._last_key_event = None

    def attach_to_keyboard(self, device) -> None:
        """Start an independent pynput listener for the e-nose keys."""
        from pynput.keyboard import Listener

        self._listener = Listener(on_press=self._on_press)
        self._listener.start()
        print(
            f"[enose] listening on keys: "
            f"{KEY_SAMPLE}=sample  {KEY_IDLE}=idle  {KEY_FILTER}=filter",
            flush=True,
        )

    def _on_press(self, key) -> None:
        try:
            char = key.char
        except AttributeError:
            return
        if char is None:
            return
        char = char.lower()
        if char == KEY_SAMPLE:
            self._pending_sample = True
            self._force = None
            # Sequence always ends in idle; clear any prior filter latch.
            self._latched = IDLE
            self._notify(
                f"key={KEY_SAMPLE} -> SAMPLE "
                f"(auto: sample hold -> filter hold -> idle)"
            )
        elif char == KEY_IDLE:
            self._pending_sample = False
            self._latched = IDLE
            self._force = IDLE
            self._notify(f"key={KEY_IDLE} -> idle (cancel sequence / valve closed)")
        elif char == KEY_FILTER:
            self._pending_sample = False
            self._latched = FILTER
            self._force = FILTER
            self._notify(f"key={KEY_FILTER} -> FILTER (cancel sequence / purge now)")

    def _notify(self, msg: str) -> None:
        self._last_key_event = msg
        # Leading newline so we don't overwrite the \r HUD line.
        print(f"\n[enose] {msg}", flush=True)

    def consume(self) -> tuple[int, bool]:
        """Return ``(token, forced)`` for this control step.

        ``forced`` is True only when the operator just pressed idle/filter,
        so teleop can cancel an auto sample→filter sequence immediately.
        A pending sample is a one-shot trigger (``forced=False``).
        """
        if self._force is not None:
            token = self._force
            self._force = None
            self._pending_sample = False
            return token, True
        if self._pending_sample:
            self._pending_sample = False
            return SAMPLE, False
        return self._latched, False

    def describe(self, token: int | None = None) -> str:
        if token is None:
            if self._force is not None:
                token = self._force
            elif self._pending_sample:
                token = SAMPLE
            else:
                token = self._latched
        return _TOKEN_NAME.get(int(token), "?")

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

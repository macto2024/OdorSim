"""Extra keyboard listener for the ternary e-nose token (teleop 4b).

robosuite's :class:`Keyboard` device owns the arm/gripper keys. We layer a
second ``pynput`` listener on top for the e-nose:

    ``1`` sample (one-shot trigger -> auto-hold)   ``0`` idle    ``2`` filter

``sample`` is a momentary trigger (consumed once, then the latched state
resumes) so a single press starts one auto-hold window; ``idle`` / ``filter``
are latched states that persist until another key is pressed.
"""

from __future__ import annotations

SAMPLE, IDLE, FILTER = 1, 0, -1


class EnoseKeyState:
    def __init__(self):
        self._latched = IDLE
        self._pending_sample = False
        self._listener = None

    def reset(self) -> None:
        self._latched = IDLE
        self._pending_sample = False

    def attach_to_keyboard(self, device) -> None:
        """Start an independent pynput listener for the e-nose keys."""
        from pynput.keyboard import Listener

        self._listener = Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key) -> None:
        try:
            char = key.char
        except AttributeError:
            return
        if char == "1":
            self._pending_sample = True
        elif char == "0":
            self._latched = IDLE
        elif char == "2":
            self._latched = FILTER

    def consume(self) -> int:
        """Return the token for this control step (one-shot sample honored)."""
        if self._pending_sample:
            self._pending_sample = False
            return SAMPLE
        return self._latched

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

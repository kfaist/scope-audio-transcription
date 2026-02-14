"""Scope Audio Plugin — Active Prompt Monitor
Tiny always-on-top overlay showing the current active prompt.
Tails the Scope log for AUDIO-PLUGIN lines.

Shows:
  🎤 VOICE  — green flash when voice noun prompt injected
  📝 UI PROMPT — yellow flash when UI prompt submitted
  🔶 FALLBACK — orange flash when voice times out
  🔇 skipped transcriptions (no nouns)
  🔍 extracted nouns
  📊 amplitude levels

Double-click to launch. Close with X or Alt+F4.
"""
import tkinter as tk
import re
import os
import time
import threading

LOG_PATH = os.path.expandvars(
    r"%APPDATA%\Daydream Scope\logs\main.log"
)


class PromptMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Scope Voice")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("500x120+20+20")
        self.root.resizable(True, True)
        self.root.overrideredirect(False)

        # Status label (voice / UI / info)
        self.status_var = tk.StringVar(value="waiting for voice...")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#888888",
            bg="#1a1a2e",
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=10, pady=(6, 0))

        # Main prompt display
        self.prompt_var = tk.StringVar(value="(no prompt yet)")
        self.prompt_label = tk.Label(
            self.root,
            textvariable=self.prompt_var,
            font=("Segoe UI Semibold", 14),
            fg="#00d4ff",
            bg="#1a1a2e",
            anchor="w",
            wraplength=480,
        )
        self.prompt_label.pack(fill="x", padx=10, pady=(0, 2))

        # Detail line (nouns, amplitude, transcription)
        self.detail_var = tk.StringVar(value="")
        self.detail_label = tk.Label(
            self.root,
            textvariable=self.detail_var,
            font=("Segoe UI", 8),
            fg="#666666",
            bg="#1a1a2e",
            anchor="w",
            wraplength=480,
        )
        self.detail_label.pack(fill="x", padx=10, pady=(0, 6))

        # Tail thread
        self._running = True
        self._thread = threading.Thread(target=self._tail_log, daemon=True)
        self._thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._running = False
        self.root.destroy()

    def _tail_log(self):
        """Tail the log file for prompt changes."""
        # Patterns — matches BOTH old and new log formats
        # New format: AUDIO-PLUGIN: >>> NEW PROMPT (voice): 'freddy krueger'
        # Old format: AUDIO-PLUGIN: >>> NEW PROMPT: 'freddy krueger'
        re_new_prompt = re.compile(
            r"AUDIO-PLUGIN: >>> NEW PROMPT\s*\(?(\w[\w\s]*?)?\)?:\s*'(.+?)'"
        )
        re_expired = re.compile(r"AUDIO-PLUGIN: voice prompt expired")
        re_nouns = re.compile(r"AUDIO-PLUGIN: nouns extracted: (.+)")
        re_no_nouns = re.compile(r"AUDIO-PLUGIN: no nouns found in '(.+?)'")
        re_amplitude = re.compile(r"AUDIO-PLUGIN: audio amplitude=(\d+\.\d+)")
        re_ui_changed = re.compile(r"AUDIO-PLUGIN: UI prompt changed to '(.+?)'")
        re_transcription = re.compile(r"AUDIO-PLUGIN: result='(.+?)'")
        re_transcribing = re.compile(r"AUDIO-PLUGIN: transcribing")

        # Seek to end of file
        try:
            f = open(LOG_PATH, "r", encoding="utf-8", errors="replace")
            f.seek(0, 2)  # end of file
        except FileNotFoundError:
            self._update_status("log file not found")
            self._update_prompt(LOG_PATH)
            return

        while self._running:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue

            # Only process AUDIO-PLUGIN lines
            if "AUDIO-PLUGIN" not in line:
                continue

            # New prompt injected (voice, UI, or fallback)
            m = re_new_prompt.search(line)
            if m:
                source = m.group(1) or "unknown"
                prompt = m.group(2)
                source_lower = source.strip().lower()
                if "voice" in source_lower:
                    self._update_status("VOICE")
                    self._flash("#00ff88")
                elif "ui fallback" in source_lower:
                    self._update_status("FALLBACK (voice timed out)")
                    self._flash("#ff8800")
                elif "ui" in source_lower:
                    self._update_status("UI PROMPT")
                    self._flash("#ffdd00")
                else:
                    self._update_status(f"PROMPT ({source})")
                    self._flash("#00d4ff")
                self._update_prompt(prompt)
                continue

            # UI prompt changed (debounced acceptance)
            m = re_ui_changed.search(line)
            if m:
                self._update_status("UI PROMPT")
                self._update_prompt(m.group(1))
                self._flash("#ffdd00")
                continue

            # Voice expired
            if re_expired.search(line):
                self._update_status("FALLBACK (voice timed out)")
                self._update_prompt("(reverted to prompt box)")
                self._flash("#ff8800")
                continue

            # Transcribing indicator
            if re_transcribing.search(line):
                self._update_detail("transcribing...")
                continue

            # Transcription result
            m = re_transcription.search(line)
            if m:
                text = m.group(1)
                if text:
                    self._update_detail(f"heard: \"{text}\"")
                continue

            # No nouns found
            m = re_no_nouns.search(line)
            if m:
                self._update_status(f"skipped: \"{m.group(1)}\" (no nouns)")
                continue

            # Nouns extracted
            m = re_nouns.search(line)
            if m:
                self._update_detail(f"nouns: {m.group(1)}")
                continue

            # Amplitude
            m = re_amplitude.search(line)
            if m:
                amp = float(m.group(1))
                bar_len = min(int(amp * 50), 20)
                bar = "|" * bar_len
                self._update_detail(f"amp: {amp:.4f} {bar}")
                continue

        f.close()

    def _update_status(self, text):
        try:
            self.root.after(0, lambda: self.status_var.set(text))
        except Exception:
            pass

    def _update_prompt(self, text):
        try:
            self.root.after(0, lambda: self.prompt_var.set(text))
        except Exception:
            pass

    def _update_detail(self, text):
        try:
            self.root.after(0, lambda: self.detail_var.set(text))
        except Exception:
            pass

    def _flash(self, color):
        """Brief color flash on new prompt."""
        def do_flash():
            self.prompt_label.configure(fg=color)
            self.root.after(1200, lambda: self.prompt_label.configure(fg="#00d4ff"))
        try:
            self.root.after(0, do_flash)
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    PromptMonitor().run()

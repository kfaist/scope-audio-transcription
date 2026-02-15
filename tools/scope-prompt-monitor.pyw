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

# Patterns — matches BOTH old and new log formats
RE_NEW_PROMPT = re.compile(
    r"AUDIO-PLUGIN: >>> (?:FLUSH \+ INJECT|NEW PROMPT)\s*\(?(\w[\w\s]*?)?\)?:\s*'(.+?)'"
)
RE_EXPIRED = re.compile(r"AUDIO-PLUGIN: voice prompt expired")
RE_NOUNS = re.compile(r"AUDIO-PLUGIN: nouns extracted: (.+)")
RE_NO_NOUNS = re.compile(r"AUDIO-PLUGIN: no nouns found in '(.+?)'")
RE_AMPLITUDE = re.compile(r"AUDIO-PLUGIN: audio amplitude=(\d+\.\d+)")
RE_UI_CHANGED = re.compile(r"AUDIO-PLUGIN: UI prompt changed to '(.+?)'")
RE_UI_INITIAL = re.compile(r"AUDIO-PLUGIN: initial UI prompt recorded: '(.+?)'")
RE_TRANSCRIPTION = re.compile(r"AUDIO-PLUGIN: result='(.+?)'")
RE_TRANSCRIBING = re.compile(r"AUDIO-PLUGIN: transcribing")


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
        self.status_var = tk.StringVar(value="starting up...")
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
        self.prompt_var = tk.StringVar(value="(scanning log...)")
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

    def _scan_last_prompt(self, filepath):
        """Read the log file backwards to find the most recent prompt state."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                # Scan entire log backwards — queue spam can bury prompt lines thousands of lines back
                tail = lines
        except (FileNotFoundError, PermissionError):
            return

        # Walk backwards to find the most recent prompt-setting line
        for line in reversed(tail):
            if "AUDIO-PLUGIN" not in line:
                continue

            m = RE_NEW_PROMPT.search(line)
            if m:
                source = (m.group(1) or "").strip().lower()
                prompt = m.group(2)
                if "voice" in source:
                    self._update_status("🎤 VOICE (recovered)")
                    self.prompt_label.configure(fg="#00ff88")
                elif "ui" in source:
                    self._update_status("📝 UI PROMPT (recovered)")
                    self.prompt_label.configure(fg="#ffdd00")
                else:
                    self._update_status(f"PROMPT (recovered)")
                    self.prompt_label.configure(fg="#00d4ff")
                self._update_prompt(prompt)
                return

            m = RE_UI_CHANGED.search(line)
            if m:
                self._update_status("📝 UI PROMPT (recovered)")
                self._update_prompt(m.group(1))
                self.prompt_label.configure(fg="#ffdd00")
                return

            m = RE_UI_INITIAL.search(line)
            if m:
                self._update_status("📝 UI PROMPT (startup)")
                self._update_prompt(m.group(1))
                self.prompt_label.configure(fg="#ffdd00")
                return

    def _tail_log(self):
        """Scan for last known prompt, then tail the log for changes."""
        # First: recover current state from existing log
        self._scan_last_prompt(LOG_PATH)

        # Now open and seek to end for live tailing
        try:
            f = open(LOG_PATH, "r", encoding="utf-8", errors="replace")
            f.seek(0, 2)  # end of file
        except FileNotFoundError:
            self._update_status("⚠ log file not found")
            self._update_prompt(LOG_PATH)
            return

        while self._running:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue

            if "AUDIO-PLUGIN" not in line:
                continue

            # New prompt injected (voice, UI, or fallback)
            m = RE_NEW_PROMPT.search(line)
            if m:
                source = m.group(1) or "unknown"
                prompt = m.group(2)
                source_lower = source.strip().lower()
                if "voice" in source_lower:
                    self._update_status("🎤 VOICE")
                    self._flash("#00ff88")
                elif "ui fallback" in source_lower:
                    self._update_status("🔶 FALLBACK (voice timed out)")
                    self._flash("#ff8800")
                elif "ui" in source_lower:
                    self._update_status("📝 UI PROMPT")
                    self._flash("#ffdd00")
                else:
                    self._update_status(f"PROMPT ({source})")
                    self._flash("#00d4ff")
                self._update_prompt(prompt)
                continue

            # UI prompt changed
            m = RE_UI_CHANGED.search(line)
            if m:
                self._update_status("📝 UI PROMPT")
                self._update_prompt(m.group(1))
                self._flash("#ffdd00")
                continue

            # Initial UI prompt on startup
            m = RE_UI_INITIAL.search(line)
            if m:
                self._update_status("📝 UI PROMPT (startup)")
                self._update_prompt(m.group(1))
                self.prompt_label.configure(fg="#ffdd00")
                continue

            # Voice expired
            if RE_EXPIRED.search(line):
                self._update_status("🔶 FALLBACK (voice timed out)")
                self._update_prompt("(reverted to prompt box)")
                self._flash("#ff8800")
                continue

            # Transcribing indicator
            if RE_TRANSCRIBING.search(line):
                self._update_detail("🔄 transcribing...")
                continue

            # Transcription result
            m = RE_TRANSCRIPTION.search(line)
            if m:
                text = m.group(1)
                if text:
                    self._update_detail(f'heard: "{text}"')
                continue

            # No nouns found
            m = RE_NO_NOUNS.search(line)
            if m:
                self._update_detail(f'🔇 skipped: "{m.group(1)}" (no nouns)')
                continue

            # Nouns extracted
            m = RE_NOUNS.search(line)
            if m:
                self._update_detail(f"🔍 nouns: {m.group(1)}")
                continue

            # Amplitude
            m = RE_AMPLITUDE.search(line)
            if m:
                amp = float(m.group(1))
                bar_len = min(int(amp * 50), 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                self._update_detail(f"📊 amp: {amp:.4f} {bar}")
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

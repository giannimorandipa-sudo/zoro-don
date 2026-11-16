from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfilt


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG = "#111214"
PANEL = "#1a1b1e"
PANEL_2 = "#202125"
LINE = "#34353a"
TEXT = "#f4f2ed"
MUTED = "#92939a"
LIME = "#c7fb79"
PURPLE = "#b4a0ff"

DEFAULTS = {
    "input": 0.0, "low": 0.0, "presence": 0.0, "glue": 22.0,
    "body": 0.0, "limiter": 35.0, "saturation": 8.0, "warmth": 0.0,
    "clean": 0.0, "width": 100.0, "ceiling": -1.0, "calm": 0.0,
    "output": 0.0,
}


def analyze_audio(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {"lufs": -96.0, "peak": -96.0, "crest": 0.0, "corr": 0.0}
    stride = max(1, len(audio) // 150_000)
    sample = audio[::stride].astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(sample * sample)))
    peak = float(np.max(np.abs(sample)))
    left, right = sample[:, 0], sample[:, 1]
    denom = float(np.sqrt(np.mean(left * left) * np.mean(right * right)))
    corr = float(np.mean(left * right) / denom) if denom > 1e-12 else 0.0
    rms_db = 20 * math.log10(max(rms, 1e-8))
    peak_db = 20 * math.log10(max(peak, 1e-8))
    return {"lufs": rms_db - 0.7, "peak": peak_db, "crest": peak_db - rms_db, "corr": corr}


def process_audio(audio: np.ndarray, sample_rate: int, p: dict[str, float], active: dict[str, bool], bypass: bool = False) -> np.ndarray:
    """Lightweight mastering chain; loudness analysis is an estimate, not a standards meter."""
    x = np.asarray(audio, dtype=np.float32).copy()
    if x.ndim == 1:
        x = np.column_stack((x, x))
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if bypass:
        return x

    x *= np.float32(10 ** (p["input"] / 20))
    if active.get("tone", True):
        nyq = sample_rate / 2
        if nyq > 180:
            lp = butter(2, min(120, nyq * .8), fs=sample_rate, btype="lowpass", output="sos")
            low_band = sosfilt(lp, x, axis=0)
            x += low_band * np.float32(10 ** (p["low"] / 20) - 1)
        if nyq > 7000:
            bp = butter(2, [min(2800, nyq * .3), min(6500, nyq * .9)], fs=sample_rate, btype="bandpass", output="sos")
            mid_band = sosfilt(bp, x, axis=0)
            x += mid_band * np.float32(10 ** (p["presence"] / 20) - 1)
    if active.get("calm", False) and p["calm"] > 0 and sample_rate > 12_000:
        harsh = butter(2, [3200, min(7800, sample_rate * .46)], fs=sample_rate, btype="bandpass", output="sos")
        x -= sosfilt(harsh, x, axis=0) * np.float32(p["calm"] / 100 * .18)
    if active.get("dynamics", True) and (p["glue"] > 0 or p["limiter"] > 0):
        detector_size = max(1, int(sample_rate * .004))
        envelope = maximum_filter1d(np.max(np.abs(x), axis=1), size=detector_size, mode="nearest")
        env_db = 20 * np.log10(np.maximum(envelope, 1e-7))
        threshold = -8.0 - p["glue"] * .12
        ratio = 1.0 + p["glue"] * .045
        reduction = np.maximum(0.0, (env_db - threshold) * (1 - 1 / ratio))
        if p["limiter"] > 0:
            reduction += np.maximum(0.0, env_db - p["ceiling"]) * (p["limiter"] / 100)
        gain = np.power(10.0, -reduction / 20).astype(np.float32)
        x *= gain[:, None]
    if active.get("dynamics", True) and p["body"] > 0:
        floor = np.float32(10 ** (-42 / 20))
        x += np.sign(x) * np.maximum(floor - np.abs(x), 0) * np.float32(p["body"] / 100 * .75)
    if active.get("color", True):
        if p["warmth"] > 0 and sample_rate > 300:
            lp = butter(1, min(180, sample_rate * .4), fs=sample_rate, btype="lowpass", output="sos")
            x += sosfilt(lp, x, axis=0) * np.float32(p["warmth"] / 100 * .10)
        if p["clean"] > 0 and sample_rate > 100:
            hp = butter(2, min(28 + p["clean"] * .45, sample_rate * .1), fs=sample_rate, btype="highpass", output="sos")
            x = sosfilt(hp, x, axis=0)
        drive = p["saturation"] / 100 * .65
        if drive > 0:
            norm = math.tanh(1 + drive)
            x = np.tanh(x * np.float32(1 + drive)) / np.float32(norm)
    if active.get("space", True):
        mid = (x[:, 0] + x[:, 1]) * .5
        side = (x[:, 0] - x[:, 1]) * (.5 * p["width"] / 100)
        x[:, 0], x[:, 1] = mid + side, mid - side
    x *= np.float32(10 ** (p["output"] / 20))
    if active.get("dynamics", True):
        ceiling = 10 ** (p["ceiling"] / 20)
        x = np.tanh(x / np.float32(ceiling)) * np.float32(ceiling)
    return np.nan_to_num(x, copy=False).astype(np.float32, copy=False)


class SonoraApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sonora — Mastering Desk")
        self.geometry("1260x850")
        self.minsize(1000, 720)
        self.configure(fg_color=BG)
        self.audio: np.ndarray | None = None
        self.reference: np.ndarray | None = None
        self.processed: np.ndarray | None = None
        self.reference_path: Path | None = None
        self.reference_rate = 44100
        self.audio_path: Path | None = None
        self.sample_rate = 44100
        self.playing_reference = False
        self.bypass = False
        self.active = {"tone": True, "dynamics": True, "color": True, "space": True, "calm": False}
        self.params = DEFAULTS.copy()
        self.variables: dict[str, ctk.DoubleVar] = {}
        self.value_labels: dict[str, ctk.CTkLabel] = {}
        self.toggles: dict[str, ctk.CTkSwitch] = {}
        self.snapshots: dict[str, tuple[dict, dict]] = {}
        self.stream = None
        self.cursor = 0
        self.render_timer = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.render_generation = 0
        self.wave_points: np.ndarray | None = None
        self._build_ui()
        self._set_initial_meter()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(16, 10))
        header.grid_columnconfigure(1, weight=1)
        logo = ctk.CTkLabel(header, text="S", width=38, height=38, corner_radius=12, fg_color=LIME,
                            text_color="#151611", font=ctk.CTkFont(size=20, weight="bold"))
        logo.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ctk.CTkLabel(header, text="SONORA", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT).grid(row=0, column=1, sticky="sw")
        ctk.CTkLabel(header, text="MASTERING DESK  ·  WINDOWS", font=ctk.CTkFont(size=9), text_color=MUTED).grid(row=1, column=1, sticky="nw")
        self.project_label = ctk.CTkLabel(header, text="PROGETTO LOCALE", fg_color=PANEL, corner_radius=14, text_color="#b9bac0", padx=12, pady=7, font=ctk.CTkFont(size=9))
        self.project_label.grid(row=0, column=2, rowspan=2, sticky="e")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 20))
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_columnconfigure(1, minsize=290)
        self.body.grid_rowconfigure(0, weight=1)
        self.left = ctk.CTkFrame(self.body, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.left.grid_columnconfigure(0, weight=1)
        self.left.grid_rowconfigure(2, weight=1)
        self._build_track_card()
        self._build_chain()
        self._build_footer()
        self._build_meter()

    def _card(self, parent, **kwargs):
        return ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=LINE, **kwargs)

    def _build_track_card(self):
        card = self._card(self.left)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(15, 6))
        head.grid_columnconfigure(0, weight=1)
        info = ctk.CTkFrame(head, fg_color="transparent")
        info.grid(row=0, column=0, sticky="w")
        self._label(info, "TRACCIA MASTER", 9, MUTED, weight="bold").pack(anchor="w")
        self.track_name = self._label(info, "Nessun audio caricato", 15, TEXT, weight="bold")
        self.track_name.pack(anchor="w", pady=(3, 0))
        self.track_meta = self._label(info, "Apri un file WAV, AIFF, FLAC oppure OGG", 9, MUTED)
        self.track_meta.pack(anchor="w", pady=(2, 0))
        actions = ctk.CTkFrame(head, fg_color="transparent")
        actions.grid(row=0, column=1, sticky="e")
        self._button(actions, "＋  Apri audio", self.load_master, primary=True).pack(side="left", padx=3)
        self._button(actions, "Riferimento", self.load_reference).pack(side="left", padx=3)
        self._button(actions, "Esporta WAV", self.export, primary=False).pack(side="left", padx=(3, 0))
        self.wave_canvas = ctk.CTkCanvas(card, height=105, bg="#141517", highlightthickness=1, highlightbackground="#292a2e")
        self.wave_canvas.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 8))
        self.wave_canvas.bind("<Configure>", lambda _e: self.draw_wave())
        transport = ctk.CTkFrame(card, fg_color="transparent")
        transport.grid(row=2, column=0, sticky="ew", padx=18, pady=(2, 14))
        transport.grid_columnconfigure(2, weight=1)
        self.play_button = self._button(transport, "▶", self.toggle_play, primary=True, width=38)
        self.play_button.grid(row=0, column=0, padx=(0, 10))
        self.time_label = self._label(transport, "00:00 / 00:00", 10, MUTED, mono=True)
        self.time_label.grid(row=0, column=1, padx=(0, 12))
        self.progress = ctk.CTkProgressBar(transport, height=5, progress_color=LIME, fg_color="#3b3c40")
        self.progress.set(0)
        self.progress.grid(row=0, column=2, sticky="ew", padx=(0, 10))
        self.progress.bind("<Button-1>", self.seek)
        self.bypass_switch = ctk.CTkSwitch(transport, text="Bypass", width=80, command=self.toggle_bypass,
                                           progress_color=LIME, button_color="#f4f2ed", button_hover_color="#fff")
        self.bypass_switch.grid(row=0, column=3, padx=(4, 0))
        self.bypass_switch.deselect()

    def _build_chain(self):
        top = ctk.CTkFrame(self.left, fg_color="transparent")
        top.grid(row=1, column=0, sticky="ew", pady=(3, 8))
        self._label(top, "CATENA DI MASTERING", 12, TEXT, weight="bold").pack(side="left")
        presets = ctk.CTkFrame(top, fg_color="transparent")
        presets.pack(side="right")
        for name, key in (("BALANCED", "balanced"), ("WARM", "warm"), ("LOUD", "loud"), ("RESET", "reset")):
            self._button(presets, name, lambda k=key: self.apply_preset(k), width=72, height=26, small=True).pack(side="left", padx=2)
        grid = ctk.CTkScrollableFrame(self.left, fg_color="transparent", corner_radius=0,
                                      orientation="vertical", scrollbar_button_color="#414247")
        grid.grid(row=2, column=0, sticky="nsew")
        for col in range(3):
            grid.grid_columnconfigure(col, weight=1, uniform="modules")
        self._module_tone(grid, 0, 0)
        self._module_dynamics(grid, 0, 1)
        self._module_color(grid, 0, 2)
        self._module_stereo(grid, 1, 0)
        self._module_calm(grid, 1, 1)
        self._module_gain(grid, 1, 2)

    def _module(self, parent, title, enabled_key, row, col, columnspan=1):
        card = self._card(parent)
        card.grid(row=row, column=col, columnspan=columnspan, sticky="nsew", padx=(0, 8), pady=(0, 9))
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(12, 7))
        self._label(head, title, 10, TEXT, weight="bold").pack(side="left")
        if enabled_key:
            sw = ctk.CTkSwitch(head, text="", width=38, command=lambda k=enabled_key: self.toggle_module(k),
                               progress_color="#688546", button_color=LIME, button_hover_color="#daffa5")
            sw.pack(side="right")
            sw.select()
            self.toggles[enabled_key] = sw
        return card

    def _slider(self, parent, key, label, lo, hi, unit="", decimals=0):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(4, 6))
        line = ctk.CTkFrame(row, fg_color="transparent")
        line.pack(fill="x", pady=(0, 4))
        self._label(line, label, 10, "#c5c5ca").pack(side="left")
        value = self._label(line, self._format_value(key, self.params[key]), 10, LIME, mono=True)
        value.pack(side="right")
        self.value_labels[key] = value
        var = ctk.DoubleVar(value=self.params[key])
        self.variables[key] = var
        slider = ctk.CTkSlider(row, from_=lo, to=hi, variable=var, number_of_steps=max(1, int((hi-lo)*10)),
                               progress_color=LIME, button_color=LIME, button_hover_color="#dfffac",
                               fg_color="#3b3c40", height=14, command=lambda v, k=key: self._value_changed(k, v))
        slider.pack(fill="x")

    def _module_tone(self, parent, row, col):
        card = self._module(parent, "01  ·  TONE & EQ", "tone", row, col)
        self.eq_canvas = ctk.CTkCanvas(card, height=50, bg=PANEL, highlightthickness=0)
        self.eq_canvas.pack(fill="x", padx=14, pady=(0, 2))
        self.eq_canvas.bind("<Configure>", lambda _e: self.draw_eq())
        self._slider(card, "low", "Low shelf", -6, 6, " dB", 1)
        self._slider(card, "presence", "Presence", -5, 5, " dB", 1)

    def _module_dynamics(self, parent, row, col):
        card = self._module(parent, "02  ·  DYNAMICS", "dynamics", row, col)
        self._slider(card, "glue", "Glue compression", 0, 100, "%")
        self._slider(card, "body", "Body", 0, 100, "%")
        self._slider(card, "limiter", "Limiter drive", 0, 100, "%")

    def _module_color(self, parent, row, col):
        card = self._module(parent, "03  ·  COLORE", "color", row, col)
        self._slider(card, "saturation", "Saturazione", 0, 100, "%")
        self._slider(card, "warmth", "Warmth", 0, 100, "%")
        self._slider(card, "clean", "Clean / low cut", 0, 100, "%")

    def _module_stereo(self, parent, row, col):
        card = self._module(parent, "04  ·  STEREO", "space", row, col)
        self._slider(card, "width", "Stereo width", 0, 150, "%")
        self._slider(card, "ceiling", "Output ceiling", -6, 0, " dB", 1)

    def _module_calm(self, parent, row, col):
        card = self._module(parent, "05  ·  CALM", "calm", row, col)
        self._slider(card, "calm", "Resonance reduction", 0, 100, "%")
        self._label(card, "Attenua con delicatezza le frequenze aspre.", 9, MUTED).pack(anchor="w", padx=14, pady=(4, 12))

    def _module_gain(self, parent, row, col):
        card = self._module(parent, "06  ·  GAIN", None, row, col)
        self._slider(card, "input", "Input gain", -12, 12, " dB", 1)
        self._slider(card, "output", "Output gain", -12, 12, " dB", 1)

    def _build_footer(self):
        card = self._card(self.left)
        card.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        footer = ctk.CTkFrame(card, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=10)
        self._label(footer, "SNAPSHOT", 9, MUTED, weight="bold").pack(side="left", padx=(0, 8))
        for name in "ABC":
            self._button(footer, name, lambda n=name: self.snapshot(n), width=37, height=28).pack(side="left", padx=3)
        self._label(footer, "Analisi loudness indicativa · elaborazione locale", 9, MUTED).pack(side="right")

    def _build_meter(self):
        self.meter = self._card(self.body)
        self.meter.grid(row=0, column=1, sticky="nsew")
        self.meter.grid_columnconfigure(0, weight=1)
        self._label(self.meter, "LOUDNESS METER", 12, TEXT, weight="bold").grid(row=0, column=0, sticky="w", padx=16, pady=(16, 13))
        score = ctk.CTkFrame(self.meter, fg_color="#141517", corner_radius=10, border_width=1, border_color=LINE)
        score.grid(row=1, column=0, sticky="ew", padx=13)
        self._label(score, "INTEGRATED LOUDNESS · EST.", 9, MUTED).pack(anchor="w", padx=13, pady=(12, 0))
        row = ctk.CTkFrame(score, fg_color="transparent")
        row.pack(fill="x", padx=13, pady=(3, 0))
        self.lufs_label = self._label(row, "−∞", 32, TEXT, weight="bold", mono=True)
        self.lufs_label.pack(side="left")
        self._label(row, "LUFS", 10, MUTED, mono=True).pack(side="left", padx=7, pady=(13, 0))
        self.lufs_bar = ctk.CTkProgressBar(score, height=6, progress_color=LIME, fg_color="#34353a")
        self.lufs_bar.pack(fill="x", padx=13, pady=(12, 3))
        self.lufs_bar.set(0)
        scale = ctk.CTkFrame(score, fg_color="transparent")
        scale.pack(fill="x", padx=13, pady=(0, 9))
        for text in ("−24", "−18", "−14", "−8", "−4"):
            self._label(scale, text, 8, MUTED, mono=True).pack(side="left", expand=True)
        self.loud_status = self._label(score, "Apri un file audio per iniziare.", 9, "#c6c6cb", wraplength=245)
        self.loud_status.pack(anchor="w", padx=13, pady=(0, 12))

        stats = ctk.CTkFrame(self.meter, fg_color="transparent")
        stats.grid(row=2, column=0, sticky="ew", padx=8, pady=10)
        stats.grid_columnconfigure((0, 1), weight=1)
        self.stat_labels = {}
        for i, (key, title, value) in enumerate((("peak", "TRUE PEAK · EST.", "−∞ dBTP"), ("crest", "CREST FACTOR", "— dB"),
                                                  ("corr", "STEREO CORRELATION", "—"), ("duration", "DURATION", "00:00"))):
            box = ctk.CTkFrame(stats, fg_color=PANEL_2, corner_radius=9)
            box.grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=4)
            self._label(box, title, 8, MUTED, mono=True).pack(anchor="w", padx=9, pady=(9, 4))
            lab = self._label(box, value, 13, TEXT, mono=True, weight="bold")
            lab.pack(anchor="w", padx=9, pady=(0, 9))
            self.stat_labels[key] = lab

        ref = ctk.CTkFrame(self.meter, fg_color="transparent")
        ref.grid(row=3, column=0, sticky="ew", padx=14, pady=(8, 0))
        self._label(ref, "REFERENCE CHECK", 9, MUTED, weight="bold").pack(anchor="w")
        self.reference_label = self._label(ref, "Nessun riferimento caricato", 9, "#c6c6cb", wraplength=240)
        self.reference_label.pack(anchor="w", pady=(6, 8))
        self.reference_button = self._button(ref, "Carica riferimento", self.load_reference, width=150)
        self.reference_button.pack(anchor="w")
        compare = ctk.CTkFrame(ref, fg_color="transparent")
        compare.pack(anchor="w", pady=(9, 0))
        self._button(compare, "Master", lambda: self.play_source(False), width=72).pack(side="left", padx=(0, 5))
        self._button(compare, "Riferimento", lambda: self.play_source(True), width=94).pack(side="left")
        self._label(self.meter, "I file restano sul tuo computer.\nIl livello LUFS è una stima RMS.", 9, MUTED).grid(row=4, column=0, sticky="sw", padx=16, pady=(22, 14))

    def _label(self, parent, text, size=10, color=TEXT, weight="normal", mono=False, **kwargs):
        family = "Consolas" if mono else "Segoe UI"
        return ctk.CTkLabel(parent, text=text, text_color=color, font=ctk.CTkFont(family=family, size=size, weight=weight), **kwargs)

    def _button(self, parent, text, command, primary=False, width=None, height=32, small=False):
        kwargs = dict(text=text, command=command, height=height, corner_radius=8,
                      font=ctk.CTkFont(size=9 if small else 10, weight="bold"),
                      fg_color=LIME if primary else PANEL_2,
                      hover_color="#d9ffa5" if primary else "#2c2d31",
                      text_color="#151611" if primary else TEXT,
                      border_width=0 if primary else 1,
                      border_color=LINE)
        if width is not None:
            kwargs["width"] = width
        return ctk.CTkButton(parent, **kwargs)

    def _format_value(self, key, value):
        if key in ("low", "presence", "input", "output", "ceiling"):
            return f"{value:.1f} dB"
        return f"{value:.0f}%"

    def _set_initial_meter(self):
        for k in self.params:
            self.params[k] = float(self.params[k])

    def load_master(self):
        path = filedialog.askopenfilename(title="Apri traccia master", filetypes=[("Audio supportato", "*.wav *.aiff *.aif *.flac *.ogg"), ("Tutti i file", "*.*")])
        if not path:
            return
        self._load_audio(Path(path), reference=False)

    def load_reference(self):
        path = filedialog.askopenfilename(title="Apri traccia di riferimento", filetypes=[("Audio supportato", "*.wav *.aiff *.aif *.flac *.ogg"), ("Tutti i file", "*.*")])
        if path:
            self._load_audio(Path(path), reference=True)

    def _load_audio(self, path: Path, reference=False):
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] > 2:
                data = data[:, :2]
        except Exception as exc:
            messagebox.showerror("File non leggibile", f"Non riesco a decodificare il file audio.\n\n{exc}")
            return
        if reference:
            self.reference_rate = sr
            self.reference, self.reference_path = data, path
            self.reference_label.configure(text=f"{path.name} · {self._duration(data, sr)}")
        else:
            self._stop()
            self.audio, self.audio_path, self.sample_rate = data, path, sr
            self.cursor = 0
            self.track_name.configure(text=path.name)
            self.track_meta.configure(text=f"Stereo · {sr} Hz · {self._duration(data, sr)} · {len(data):,} campioni")
            self.wave_points = self._reduce_wave(data)
            self.render_processed()
            self._set_stats(analyze_audio(data))
            self.draw_wave()
        self.project_label.configure(text="RIFERIMENTO PRONTO" if reference else "PROGETTO LOCALE")

    @staticmethod
    def _duration(data, sr):
        seconds = int(len(data) / sr)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    @staticmethod
    def _reduce_wave(data, n=1200):
        hop = max(1, len(data) // n)
        mono = np.max(np.abs(data[:, :2]), axis=1)
        usable = len(mono) // hop * hop
        return mono[:usable].reshape(-1, hop).max(axis=1) if usable else mono

    def _set_stats(self, stats):
        self.lufs_label.configure(text=f"{stats['lufs']:.1f}")
        self.lufs_bar.set(max(0.0, min(1.0, (stats["lufs"] + 24) / 22)))
        self.stat_labels["peak"].configure(text=f"{stats['peak']:.1f} dBTP")
        self.stat_labels["crest"].configure(text=f"{stats['crest']:.1f} dB")
        self.stat_labels["corr"].configure(text=f"{stats['corr']:.2f}")
        if stats["lufs"] > -9:
            status = "Livello elevato: ascolta eventuali artefatti."
        elif stats["lufs"] > -16:
            status = "Livello equilibrato per un ascolto di prova."
        else:
            status = "Master dinamico: puoi aumentare il livello se vuoi."
        self.loud_status.configure(text=status)

    def _value_changed(self, key, raw_value):
        self.params[key] = float(raw_value)
        self.value_labels[key].configure(text=self._format_value(key, self.params[key]))
        if key in ("low", "presence"):
            self.draw_eq()
        if self.render_timer:
            self.after_cancel(self.render_timer)
        self.render_timer = self.after(220, self.render_processed)

    def toggle_module(self, key):
        self.active[key] = bool(self.toggles[key].get())
        self.render_processed()

    def toggle_bypass(self):
        self.bypass = bool(self.bypass_switch.get())
        self.render_processed()

    def render_processed(self):
        if self.audio is None:
            return
        self.render_generation += 1
        generation = self.render_generation
        audio, sr = self.audio, self.sample_rate
        p, active, bypass = self.params.copy(), self.active.copy(), self.bypass
        self.loud_status.configure(text="Aggiornamento elaborazione…")
        future = self.executor.submit(process_audio, audio, sr, p, active, bypass)
        def finish(fut):
            try:
                result = fut.result()
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Elaborazione audio", str(exc)))
                return
            def install():
                if generation != self.render_generation:
                    return
                self.processed = result
                self._set_stats(analyze_audio(result))
            self.after(0, install)
        future.add_done_callback(finish)

    def apply_preset(self, name):
        presets = {
            "balanced": DEFAULTS,
            "warm": {**DEFAULTS, "low": 1.2, "presence": -.5, "glue": 30, "body": 15, "limiter": 45, "saturation": 18, "warmth": 20, "width": 95},
            "loud": {**DEFAULTS, "presence": .8, "glue": 55, "body": 18, "limiter": 72, "saturation": 12, "clean": 8, "input": 1},
            "reset": {**DEFAULTS, "glue": 0, "limiter": 0, "saturation": 0},
        }
        self.params.update(presets[name])
        for key, var in self.variables.items():
            var.set(self.params[key])
            self.value_labels[key].configure(text=self._format_value(key, self.params[key]))
        self.render_processed()

    def snapshot(self, name):
        if name in self.snapshots:
            self.params, self.active = self.snapshots[name][0].copy(), self.snapshots[name][1].copy()
            for key, var in self.variables.items():
                var.set(self.params[key])
                self.value_labels[key].configure(text=self._format_value(key, self.params[key]))
            for key, switch in self.toggles.items():
                switch.select() if self.active[key] else switch.deselect()
            self.render_processed()
            return
        self.snapshots[name] = (self.params.copy(), self.active.copy())
        self.project_label.configure(text=f"SNAPSHOT {name} SALVATO")

    def toggle_play(self):
        if self.audio is None:
            messagebox.showinfo("Sonora", "Apri una traccia audio prima di avviare la riproduzione.")
            return
        if self.stream and self.stream.active:
            self._stop()
        else:
            self.play_source(self.playing_reference)

    def play_source(self, reference=False):
        data = self.reference if reference else (self.audio if self.bypass else self.processed)
        if data is None:
            return
        self._stop()
        self.playing_reference = reference
        start = 0 if reference else self.cursor
        self.play_cursor = min(start, max(0, len(data) - 1))
        def callback(outdata, frames, _time, _status):
            end = min(self.play_cursor + frames, len(data))
            count = end - self.play_cursor
            outdata[:] = 0
            if count:
                outdata[:count] = data[self.play_cursor:end]
            self.play_cursor = end
            if not reference:
                self.cursor = end
            if count < frames:
                raise sd.CallbackStop
        try:
            sr = self._reference_sample_rate() if reference else self.sample_rate
            self.stream = sd.OutputStream(samplerate=sr, channels=2, dtype="float32", callback=callback, blocksize=1024)
            self.stream.start()
            self.play_button.configure(text="Ⅱ")
            self._update_transport()
        except Exception as exc:
            self.stream = None
            messagebox.showerror("Uscita audio", f"Non riesco ad avviare la riproduzione.\n\n{exc}")

    def _reference_sample_rate(self):
        return self.reference_rate

    def _stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.play_button.configure(text="▶")

    def seek(self, event):
        if self.audio is None:
            return
        frac = max(0.0, min(1.0, event.x / max(self.progress.winfo_width(), 1)))
        self.cursor = int(frac * len(self.audio))
        if self.stream:
            self.play_source(self.playing_reference)
        else:
            self._update_transport()

    def _update_transport(self):
        if self.audio is not None:
            total = len(self.audio)
            frac = min(1.0, self.cursor / max(1, total))
            self.progress.set(frac)
            self.time_label.configure(text=f"{self._duration(self.audio[:self.cursor], self.sample_rate)} / {self._duration(self.audio, self.sample_rate)}")
            self.stat_labels["duration"].configure(text=self._duration(self.audio, self.sample_rate))
            self.draw_wave(frac)
            if self.stream and self.stream.active:
                self.after(90, self._update_transport)
            elif self.cursor >= total:
                self.cursor = 0
                self.play_button.configure(text="▶")
                self.progress.set(0)

    def export(self):
        if self.audio is None:
            messagebox.showinfo("Esporta WAV", "Apri prima una traccia audio.")
            return
        default = (self.audio_path.stem if self.audio_path else "master") + "_sonora.wav"
        path = filedialog.asksaveasfilename(title="Esporta master", defaultextension=".wav", initialfile=default,
                                           filetypes=[("WAV 24 bit", "*.wav")])
        if not path:
            return
        self.project_label.configure(text="RENDER WAV…")
        audio, sr, p, active, bypass = self.audio, self.sample_rate, self.params.copy(), self.active.copy(), self.bypass
        def render():
            result = process_audio(audio, sr, p, active, bypass)
            sf.write(path, result, sr, subtype="PCM_24", format="WAV")
            self.after(0, lambda: (self.project_label.configure(text="WAV ESPORTATO"), messagebox.showinfo("Esportazione completata", f"File salvato in:\n{path}")))
        self.executor.submit(render)

    def draw_wave(self, progress=0):
        if not hasattr(self, "wave_canvas"):
            return
        c = self.wave_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        mid = h / 2
        c.create_line(0, mid, w, mid, fill="#303136")
        if self.wave_points is None or len(self.wave_points) == 0:
            c.create_text(w/2, mid, text="IMPORTA UNA TRACCIA AUDIO", fill="#707178", font=("Segoe UI", 9, "bold"))
            return
        indices = np.linspace(0, len(self.wave_points)-1, min(w, len(self.wave_points))).astype(int)
        amp = self.wave_points[indices] * (h * .43)
        for x, a in enumerate(amp):
            col = LIME if x < w * progress else "#899c64"
            c.create_line(x, mid-a, x, mid+a, fill=col)

    def draw_eq(self):
        if not hasattr(self, "eq_canvas"):
            return
        c = self.eq_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        for i in range(1, 4):
            c.create_line(0, h*i/4, w, h*i/4, fill="#333439")
        pts = []
        for i in range(w):
            t = i / max(w-1, 1)
            db = self.params["low"] * math.exp(-(t/.18)**2) + self.params["presence"] * math.exp(-((t-.68)/.16)**2)
            pts.extend((i, h/2-db*3))
        if len(pts) > 3:
            c.create_line(*pts, fill=LIME, width=2, smooth=True)

    def _close(self):
        self._stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


if __name__ == "__main__":
    SonoraApp().mainloop()

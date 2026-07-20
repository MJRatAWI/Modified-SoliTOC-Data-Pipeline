import os
import sys
import tempfile
import tkinter as tk
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from my_pipeline.nodes.nodes import (  # noqa: E402
    DEFAULT_AUTO_WINDOW,
    DEFAULT_BASELINE_END,
    DEFAULT_BASELINE_START,
    DEFAULT_FACTOR,
    DEFAULT_FIXED_BASELINE,
    DEFAULT_FIXED_VAL,
    DEFAULT_FLOW_COL_IDX,
    DEFAULT_INTERP_EVERY,
    DEFAULT_MAX_ENERGY_TEMP,
    DEFAULT_PREVIEW,
    DEFAULT_SHIFT,
    DEFAULT_ZONE_FLOW_RELATIVE_INCREASE,
    OUT_SUBDIRS,
    compute_baseline_and_preview_data,
    detect_zone_boundaries,
    format_batch_summary_row,
    parse_program_metadata,
    process_file,
    write_batch_exports,
)


class ToolTip:
    """Simple hover tooltip for Tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)
        self.widget.bind("<ButtonPress>", self.hide)

    def show(self, event=None) -> None:
        if self.tip_window or not self.text:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 3

        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tip_window,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=260,
        )
        label.pack(ipadx=4, ipady=2)

    def hide(self, event=None) -> None:
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def show_preview_modal(  # noqa: PLR0913, PLR0915
    root: tk.Tk,
    time: np.ndarray,
    co2_raw: np.ndarray,
    temp_raw: np.ndarray,
    baseline_val: float,
    bi0: int,
    bi1: int,
    filename: str,
    omit_below: float = 100.0,
    preview_shift_seconds: float = 0.0,
    zone_timepoints: dict[str, float] | None = None,
) -> bool:
    """Display a preview window with a baseline plot before processing."""
    win = tk.Toplevel(root)
    win.title("Baseline preview — Confirm or Skip")
    win.geometry("480x140")
    win.transient(root)
    win.grab_set()
    win.lift()
    win.focus_force()
    root.update_idletasks()
    win.update_idletasks()

    lbl = ttk.Label(
        win,
        text=f"Baseline preview — {os.path.basename(filename)} (CO2 < {omit_below} omitted)",
    )
    lbl.pack(anchor="w", padx=8, pady=(8, 2))

    info_label = ttk.Label(
        win, text="Use the Continue or Skip buttons in the browser window to proceed."
    )
    info_label.pack(padx=8, pady=(4, 8))

    co2_arr = np.array(co2_raw, dtype=float)
    mask_ok = (~np.isnan(co2_arr)) & (co2_arr >= omit_below)

    if np.sum(mask_ok) == 0:
        msg = ttk.Label(
            win, text=f"No CO2 values >= {omit_below} present — nothing to preview."
        )
        msg.pack(padx=8, pady=8)
        result = {"choice": None}

        def _wait_for_browser() -> None:
            result["choice"] = True
            win.destroy()

        root.after(100, _wait_for_browser)
        root.wait_window(win)
        return bool(result["choice"])

    time_plot, co2_plot = time[mask_ok], co2_arr[mask_ok]
    minpos = (
        np.nanmin(co2_plot[np.isfinite(co2_plot)])
        if np.any(np.isfinite(co2_plot))
        else None
    )
    offset = 0.0

    if minpos is None or minpos <= 0:
        sd = (
            float(np.nanstd(co2_plot[np.isfinite(co2_plot)]))
            if np.any(np.isfinite(co2_plot))
            else 1.0
        )
        offset = max(1e-6, 0.01 * sd)

    co2_for_log = co2_plot + offset
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_plot,
            y=co2_for_log,
            mode="lines",
            name=f"CO2 (>= {omit_below}) + offset",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    temp_arr = np.asarray(temp_raw, dtype=float)
    temp_plot = temp_arr[mask_ok]
    temp_time_plot = time_plot + float(preview_shift_seconds)
    fig.add_trace(
        go.Scatter(
            x=temp_time_plot,
            y=temp_plot,
            mode="lines",
            name="Temperature (raw, visual shift)",
            line=dict(color="#d62728", width=1.2),
            yaxis="y2",
        )
    )

    try:
        t0, t1 = time[bi0], time[bi1]
        if not (t1 < time_plot.min() or t0 > time_plot.max()):
            x0 = max(t0, time_plot.min())
            x1 = min(t1, time_plot.max())
            fig.add_vrect(
                x0=x0,
                x1=x1,
                annotation_text="baseline window",
                annotation_position="inside top left",
                fillcolor="orange",
                opacity=0.25,
                line_width=0,
            )
            fig.add_trace(
                go.Scatter(
                    x=[t0, t1],
                    y=[baseline_val + offset, baseline_val + offset],
                    mode="lines",
                    name=f"baseline={baseline_val:.3g}",
                    line=dict(color="red", dash="dash", width=1.5),
                )
            )
    except Exception:
        pass

    if zone_timepoints:
        zone_colors = {
            "Start-Run": "#2ca02c",
            "Start-Ramp": "#ff7f0e",
            "Start-Plateau": "#9467bd",
            "Start-Oxidation": "#d62728",
            "End-Run": "#8c564b",
        }
        for name, xval in zone_timepoints.items():
            if xval < time_plot.min() or xval > time_plot.max():
                continue
            fig.add_vline(
                x=xval,
                line_color=zone_colors.get(name, "#444444"),
                line_dash="dot",
                line_width=1.5,
                annotation_text=name,
                annotation_position="top",
            )

    fig.update_layout(
        title=f"Baseline preview — {os.path.basename(filename)}",
        xaxis_title="Time (s)",
        yaxis_title="CO2 (log scale, arb. units)",
        yaxis_type="log",
        yaxis2=dict(
            title="Temperature (°C)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=30),
    )

    preview_path = os.path.join(
        tempfile.gettempdir(), f"solitoc_preview_{os.getpid()}.html"
    )
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    result = {"choice": None}

    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/continue"):
                result["choice"] = True
            elif self.path.startswith("/skip"):
                result["choice"] = False
            else:
                result["choice"] = None

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body><p>Selection recorded. You can close this tab.</p></body></html>"
            )
            try:
                root.after(0, win.destroy)
                self.server.shutdown()
            except Exception:
                pass

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), PreviewHandler)
    port = server.server_address[1]
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    browser_html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Baseline preview</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          .actions {{ margin-top: 16px; }}
          button {{ margin-right: 10px; padding: 8px 12px; cursor: pointer; }}
        </style>
      </head>
      <body>
        <h3>Baseline preview</h3>
        <p>{os.path.basename(filename)}</p>
        <div>{plot_html}</div>
        <div class="actions">
          <button onclick="window.location.href='http://127.0.0.1:{port}/continue'">Continue</button>
          <button onclick="window.location.href='http://127.0.0.1:{port}/skip'">Skip File</button>
        </div>
      </body>
    </html>
    """
    with open(preview_path, "w", encoding="utf-8") as fh:
        fh.write(browser_html)

    root.after(0, lambda: webbrowser.open_new_tab(preview_path))
    root.wait_window(win)
    try:
        server.shutdown()
        server.server_close()
    except Exception:
        pass
    return bool(result["choice"])


class MergedApp:
    """Tkinter application for batch processing SoliTOC files."""

    def __init__(self, master: tk.Tk):  # noqa: PLR0915
        self.master = master
        master.title("SoliTOC Processor")
        master.geometry("1000x800")

        frm = ttk.Frame(master, padding=12)
        frm.pack(fill="both", expand=True)
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(frm, text="Input / Output", padding=(10, 10))
        input_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 8))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Input TXT files or folder:").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ToolTip(
            self.input_entry,
            "Enter one or more TXT files separated by semicolons, or choose a folder containing TXT files.",
        )
        self.browse_files_btn = ttk.Button(
            input_frame, text="Browse files", command=self.select_files
        )
        self.browse_files_btn.grid(row=0, column=2, padx=(4, 2))
        ToolTip(
            self.browse_files_btn,
            "Open a file picker to select one or more TXT input files.",
        )
        self.browse_folder_btn = ttk.Button(
            input_frame, text="Browse folder", command=self.select_folder
        )
        self.browse_folder_btn.grid(row=0, column=3)
        ToolTip(
            self.browse_folder_btn,
            "Choose a folder that contains the TXT files to process.",
        )

        ttk.Label(input_frame, text="Output folder:").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.out_var = tk.StringVar()
        self.out_entry = ttk.Entry(input_frame, textvariable=self.out_var)
        self.out_entry.grid(row=1, column=1, sticky="ew", padx=4)
        ToolTip(
            self.out_entry,
            "Choose the folder where the processed CSV and summary files will be written.",
        )
        self.browse_out_btn = ttk.Button(
            input_frame, text="Browse...", command=self.select_out
        )
        self.browse_out_btn.grid(row=1, column=2, columnspan=2, sticky="w")
        ToolTip(
            self.browse_out_btn, "Select the output directory for the generated files."
        )

        ttk.Label(input_frame, text="Batch summary base name:").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.batch_summary_name_var = tk.StringVar()
        self.batch_summary_name_entry = ttk.Entry(
            input_frame, textvariable=self.batch_summary_name_var
        )
        self.batch_summary_name_entry.grid(row=2, column=1, sticky="ew", padx=4)
        ToolTip(
            self.batch_summary_name_entry,
            "Base name for the generated batch summary file.",
        )

        baseline_frame = ttk.LabelFrame(frm, text="Baseline settings", padding=(10, 10))
        baseline_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 8))
        baseline_frame.columnconfigure(1, weight=1)
        baseline_frame.columnconfigure(3, weight=1)

        ttk.Label(baseline_frame, text="Mode:").grid(row=0, column=0, sticky="w")
        self.baseline_mode = tk.StringVar(value="manual")
        self.baseline_mode.trace_add(
            "write", lambda *a: self._on_baseline_mode_change()
        )
        self.manual_radio = ttk.Radiobutton(
            baseline_frame, text="Manual", variable=self.baseline_mode, value="manual"
        )
        self.manual_radio.grid(row=0, column=1, sticky="w", padx=(0, 6))
        self.fixed_radio = ttk.Radiobutton(
            baseline_frame,
            text="Fixed time interval",
            variable=self.baseline_mode,
            value="fixed",
        )
        self.fixed_radio.grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.auto_radio = ttk.Radiobutton(
            baseline_frame,
            text="Auto-detect",
            variable=self.baseline_mode,
            value="auto",
        )
        self.auto_radio.grid(row=0, column=3, sticky="w", padx=(0, 6))
        self.fixed_value_radio = ttk.Radiobutton(
            baseline_frame,
            text="Fixed Value",
            variable=self.baseline_mode,
            value="fixed_val",
        )
        self.fixed_value_radio.grid(row=0, column=4, sticky="w")

        ttk.Label(baseline_frame, text="Manual baseline start:").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.man_start = ttk.Entry(baseline_frame, width=8)
        self.man_start.insert(0, str(DEFAULT_BASELINE_START))
        self.man_start.grid(row=1, column=1, sticky="w")

        ttk.Label(baseline_frame, text="end:").grid(row=1, column=2, sticky="w")
        self.man_end = ttk.Entry(baseline_frame, width=8)
        self.man_end.insert(0, str(DEFAULT_BASELINE_END))
        self.man_end.grid(row=1, column=3, sticky="w")

        ttk.Label(baseline_frame, text="Fixed time interval min:").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.fixed_time_start = ttk.Entry(baseline_frame, width=8)
        self.fixed_time_start.insert(0, str(DEFAULT_FIXED_BASELINE[0]))
        self.fixed_time_start.grid(row=2, column=1, sticky="w")

        ttk.Label(baseline_frame, text="max:").grid(row=2, column=2, sticky="w")
        self.fixed_time_end = ttk.Entry(baseline_frame, width=8)
        self.fixed_time_end.insert(0, str(DEFAULT_FIXED_BASELINE[1]))
        self.fixed_time_end.grid(row=2, column=3, sticky="w")

        ttk.Label(baseline_frame, text="Fixed baseline val:").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.fixed_val_entry = ttk.Entry(baseline_frame, width=8)
        self.fixed_val_entry.insert(0, str(DEFAULT_FIXED_VAL))
        self.fixed_val_entry.grid(row=3, column=1, sticky="w")

        ttk.Label(baseline_frame, text="Auto search start:").grid(
            row=4, column=0, sticky="w", pady=4
        )
        self.auto_start = ttk.Entry(baseline_frame, width=8)
        self.auto_start.grid(row=4, column=1, sticky="w")

        ttk.Label(baseline_frame, text="end:").grid(row=4, column=2, sticky="w")
        self.auto_end = ttk.Entry(baseline_frame, width=8)
        self.auto_end.grid(row=4, column=3, sticky="w")

        ttk.Label(baseline_frame, text="Auto window (s):").grid(
            row=3, column=2, sticky="w"
        )
        self.auto_win = ttk.Entry(baseline_frame, width=8)
        self.auto_win.insert(0, str(DEFAULT_AUTO_WINDOW))
        self.auto_win.grid(row=3, column=3, sticky="w")

        self.preview_var = tk.BooleanVar(value=DEFAULT_PREVIEW)
        self.preview_check = ttk.Checkbutton(
            baseline_frame,
            text="Show baseline preview (skipped for fixed value)",
            variable=self.preview_var,
        )
        self.preview_check.grid(row=5, column=0, columnspan=5, sticky="w", pady=(8, 0))

        process_frame = ttk.LabelFrame(frm, text="Processing options", padding=(10, 10))
        process_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 8))
        process_frame.columnconfigure(1, weight=1)
        process_frame.columnconfigure(3, weight=1)

        ttk.Label(process_frame, text="Shift points (CO2 earlier):").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.shift = ttk.Entry(process_frame, width=8)
        self.shift.insert(0, str(DEFAULT_SHIFT))
        self.shift.grid(row=0, column=1, sticky="w")

        ttk.Label(process_frame, text="Factor (area→µgC):").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        self.factor = ttk.Entry(process_frame, width=12)
        self.factor.insert(0, str(DEFAULT_FACTOR))
        self.factor.grid(row=0, column=3, sticky="w")

        ttk.Label(process_frame, text="Temp snapshot every (s):").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.interp_every = ttk.Entry(process_frame, width=8)
        self.interp_every.insert(0, str(DEFAULT_INTERP_EVERY))
        self.interp_every.grid(row=1, column=1, sticky="w")

        ttk.Label(process_frame, text="Max Temp (°C):").grid(
            row=1, column=2, sticky="w", padx=(12, 0)
        )
        self.max_temp = ttk.Entry(process_frame, width=8)
        self.max_temp.insert(0, str(DEFAULT_MAX_ENERGY_TEMP))
        self.max_temp.grid(row=1, column=3, sticky="w")

        ttk.Label(
            process_frame, text="Optional integration T-interval Tmin,Tmax:"
        ).grid(row=2, column=0, sticky="w", pady=4)
        self.tinterval = ttk.Entry(process_frame, width=22)
        self.tinterval.grid(row=2, column=1, sticky="w")

        action_frame = ttk.Frame(frm, padding=(0, 6))
        action_frame.grid(row=3, column=0, sticky="ew", padx=4)
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(
            action_frame, text="Start Processing", command=self.start_processing
        )
        self.start_btn.grid(row=0, column=0, sticky="w", pady=4)
        self.quit_btn = ttk.Button(action_frame, text="Quit", command=master.quit)
        self.quit_btn.grid(row=0, column=1, sticky="e", pady=4)

        log_frame = ttk.LabelFrame(frm, text="Log / Status", padding=(10, 10))
        log_frame.grid(row=4, column=0, sticky="nsew", padx=4, pady=(0, 4))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = tk.Text(log_frame, height=18, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        self.clear_log_btn = ttk.Button(
            log_frame, text="Clear Log", command=self.clear_log
        )
        self.clear_log_btn.grid(row=1, column=0, sticky="e", pady=(8, 0))

        frm.rowconfigure(4, weight=1)
        self._setup_tooltips()
        self._on_baseline_mode_change()

    def _setup_tooltips(self) -> None:
        ToolTip(
            self.input_entry,
            "Enter one or more TXT files separated by semicolons, or choose a folder containing TXT files.",
        )
        ToolTip(
            self.out_entry,
            "Choose the folder where the processed CSV and summary files will be written.",
        )
        ToolTip(
            self.batch_summary_name_entry,
            "Base name for the generated batch summary file.",
        )
        ToolTip(self.manual_radio, "Use a manually chosen baseline time range.")
        ToolTip(self.fixed_radio, "Use a fixed baseline time interval.")
        ToolTip(
            self.auto_radio,
            "Let the app automatically detect the best baseline window.",
        )
        ToolTip(
            self.fixed_value_radio,
            "Use a fixed baseline value instead of a time window.",
        )
        ToolTip(self.man_start, "Start index for the manual baseline window.")
        ToolTip(self.man_end, "End index for the manual baseline window.")
        ToolTip(self.fixed_time_start, "Start of the fixed baseline time interval.")
        ToolTip(self.fixed_time_end, "End of the fixed baseline time interval.")
        ToolTip(
            self.fixed_val_entry,
            "Baseline value to use when the fixed value mode is selected.",
        )
        ToolTip(self.auto_start, "Optional start limit for automatic baseline search.")
        ToolTip(self.auto_end, "Optional end limit for automatic baseline search.")
        ToolTip(self.auto_win, "Window size used when auto-detecting the baseline.")
        ToolTip(
            self.preview_check, "Show the baseline preview before processing each file."
        )
        ToolTip(
            self.shift,
            "Number of points to shift the CO2 signal earlier before integration.",
        )
        ToolTip(
            self.factor,
            "Conversion factor from integrated area to micrograms of carbon.",
        )
        ToolTip(
            self.interp_every,
            "Snapshot spacing in seconds for temperature smoothing between Start-Ramp and Start-Plateau.",
        )
        ToolTip(
            self.max_temp,
            "Target maximum temperature at Start-Oxidation for Temperature_Energy.",
        )
        ToolTip(
            self.tinterval, "Optional integration temperature interval as Tmin,Tmax."
        )
        ToolTip(
            self.start_btn, "Start processing the selected files and write outputs."
        )
        ToolTip(self.quit_btn, "Close the application.")

    def clear_log(self) -> None:
        self.log.delete("1.0", tk.END)

    def _on_baseline_mode_change(self) -> None:
        mode = self.baseline_mode.get()
        state_manual = "normal" if mode == "manual" else "disabled"
        state_fixed_time = "normal" if mode == "fixed" else "disabled"
        state_fixed_val = "normal" if mode == "fixed_val" else "disabled"
        state_auto = "normal" if mode == "auto" else "disabled"

        self.man_start.config(state=state_manual)
        self.man_end.config(state=state_manual)
        self.fixed_time_start.config(state=state_fixed_time)
        self.fixed_time_end.config(state=state_fixed_time)
        self.fixed_val_entry.config(state=state_fixed_val)
        self.auto_start.config(state=state_auto)
        self.auto_end.config(state=state_auto)
        self.auto_win.config(state=state_auto)

    def select_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select TXT files", filetypes=[("TXT files", "*.txt")]
        )
        if paths:
            self.input_var.set(";".join(paths))

    def select_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder with TXT files")
        if folder:
            self.input_var.set(folder)

    def select_out(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.out_var.set(folder)

    def log_msg(self, txt: str) -> None:
        self.log.insert(tk.END, txt + "\n")
        self.log.see(tk.END)
        self.master.update_idletasks()

    def start_processing(self) -> None:  # noqa: PLR0912, PLR0915
        in_text = self.input_var.get().strip()
        out_dir = self.out_var.get().strip()
        if not in_text:
            messagebox.showerror("Error", "Select input files or folder.")
            return
        if not out_dir:
            messagebox.showerror("Error", "Select output folder.")
            return

        if os.path.isdir(in_text):
            files = [
                os.path.join(in_text, f)
                for f in os.listdir(in_text)
                if f.lower().endswith(".txt")
            ]
        else:
            files = [f for f in in_text.split(";") if f.strip()]
        files = [f for f in files if os.path.isfile(f)]

        if not files:
            messagebox.showerror("Error", "No files found.")
            return

        try:
            man_start = int(self.man_start.get())
            man_end = int(self.man_end.get())
            baseline_mode = self.baseline_mode.get()
            fixed_val_input = float(self.fixed_val_entry.get())
            fixed_time_start = float(self.fixed_time_start.get())
            fixed_time_end = float(self.fixed_time_end.get())
            auto_s = self.auto_start.get().strip()
            auto_e = self.auto_end.get().strip()
            auto_s = None if auto_s == "" else float(auto_s)
            auto_e = None if auto_e == "" else float(auto_e)
            auto_win = float(self.auto_win.get())
            preview_on = bool(self.preview_var.get())
            shift_pts = int(self.shift.get())
            factor = float(self.factor.get())
            interp_every = float(self.interp_every.get())
            max_temp = float(self.max_temp.get())

            tinterval_txt = self.tinterval.get().strip()
            temp_interval = None
            if tinterval_txt:
                tmin, tmax = [float(x.strip()) for x in tinterval_txt.split(",")]
                temp_interval = (tmin, tmax)
        except Exception as exc:
            messagebox.showerror("Parameter error", str(exc))
            return

        for sub in OUT_SUBDIRS.values():
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

        summary_rows = []
        thermogram_exports = []
        for f in files:
            self.log_msg(f"Processing: {f}")
            try:
                header_line, time_arr, co2_arr, bval, bi0, bi1, df_preview = (
                    compute_baseline_and_preview_data(
                        f,
                        0,
                        1,
                        -1,
                        baseline_mode,
                        manual_baseline=(man_start, man_end),
                        fixed_baseline=(fixed_time_start, fixed_time_end),
                        auto_search_range=(auto_s, auto_e),
                        auto_window_size=auto_win,
                        fixed_val=fixed_val_input,
                    )
                )

                zone_timepoints = None
                try:
                    resolved_temp_col = 1
                    resolved_flow_col = DEFAULT_FLOW_COL_IDX
                    temp_arr = df_preview.iloc[:, resolved_temp_col].to_numpy()
                    flow_arr = df_preview.iloc[:, resolved_flow_col].to_numpy()
                    program_meta = parse_program_metadata(f, header_line=header_line)
                    zone_boundaries = detect_zone_boundaries(
                        time_arr,
                        temp_arr,
                        flow_arr,
                        flow_relative_increase=DEFAULT_ZONE_FLOW_RELATIVE_INCREASE,
                        ramp_reference_time=program_meta.get("b_d3_time_s"),
                        c4_seconds=program_meta.get("c4_s"),
                    )
                    # Baseline preview is shown in CO2 time space, so shift only internal phase markers.
                    # Keep run start/end anchored to the original file boundaries.
                    zone_timepoints = {
                        name: float(
                            ts if name in {"Start-Run", "End-Run"} else ts + shift_pts
                        )
                        for name, ts in zone_boundaries["times"].items()
                    }
                except Exception as exc:
                    self.log_msg(f"Zone preview markers unavailable for {f}: {exc}")

                if preview_on and baseline_mode != "fixed_val":
                    decision = show_preview_modal(
                        self.master,
                        time_arr,
                        co2_arr,
                        temp_arr,
                        bval,
                        bi0,
                        bi1,
                        filename=f,
                        omit_below=100.0,
                        preview_shift_seconds=shift_pts,
                        zone_timepoints=zone_timepoints,
                    )
                    if not decision:
                        self.log_msg(f"User skipped file: {f}")
                        continue
            except Exception as exc:
                self.log_msg(f"Preview/baseline detection failed for {f}: {exc}")
                continue

            try:
                res = process_file(
                    f,
                    out_dir,
                    time_col_index=0,
                    temp_col_index=1,
                    co2_col_index=-1,
                    flow_col_index=DEFAULT_FLOW_COL_IDX,
                    baseline_mode=baseline_mode,
                    manual_baseline=(man_start, man_end),
                    fixed_baseline=(fixed_time_start, fixed_time_end),
                    auto_search_range=(auto_s, auto_e),
                    auto_window_size=auto_win,
                    fixed_val=fixed_val_input,
                    shift=shift_pts,
                    factor=factor,
                    interp_every=interp_every,
                    max_energy_temp=max_temp,
                    temp_interval=temp_interval,
                    zone_flow_relative_increase=DEFAULT_ZONE_FLOW_RELATIVE_INCREASE,
                )
                base_name = res.get("basefn", os.path.splitext(os.path.basename(f))[0])
                summary_rows.append(format_batch_summary_row(res, base_name))
                thermogram_df = res.get("thermogram_df")
                if isinstance(thermogram_df, pd.DataFrame):
                    thermogram_exports.append((base_name, thermogram_df))
            except Exception as exc:
                self.log_msg(f"ERROR processing {f}: {exc}")

        if summary_rows:
            batch_base_name = (
                self.batch_summary_name_var.get() or "batch_summary"
            ).strip()
            paths = write_batch_exports(
                Path(out_dir), batch_base_name, summary_rows, thermogram_exports
            )
            self.log_msg(f"Batch summary: {paths['batch_csv_path']}")
        else:
            self.log_msg("No files processed.")
        self.log_msg("Done.")


def main() -> None:
    root = tk.Tk()
    MergedApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        if root.winfo_exists():
            root.destroy()


if __name__ == "__main__":
    main()

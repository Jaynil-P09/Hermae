"""Native playback of images and outputs from real training forward passes."""

import atexit
from collections import deque
from math import ceil
import multiprocessing
from queue import Empty, Full
from time import monotonic

import numpy as np


PREVIEW_MODES = ("first-pass", "all")
_viewer = None


def add_preview_argument(parser):
    parser.add_argument("--live-preview", nargs="?", const="first-pass", choices=PREVIEW_MODES,
                        help="open a popup for every image in epoch 1 or every image in all epochs")


def preview_for(args):
    global _viewer
    if not getattr(args, "live_preview", None):
        return None
    if _viewer is None:
        _viewer = TrainingPreview(args.live_preview)
        atexit.register(_viewer.finish)
    return _viewer


class TrainingPreview:
    def __init__(self, mode):
        context = multiprocessing.get_context("spawn")
        self.frames = context.Queue(maxsize=3)
        self.closed = context.Event()
        self.mode = mode
        self.last_preview = float("-inf")
        self.process = context.Process(target=run_window, args=(self.frames, self.closed), daemon=True)
        self.process.start()

    def active(self, epoch):
        if self.closed.is_set() or not self.process.is_alive():
            return False
        return self.mode == "all" or epoch == 1 or monotonic() - self.last_preview >= .35

    def show(self, model, epoch, epochs, panels, details):
        """Queue a batch of detached visual planes. Later sampled epochs never block."""
        if not self.active(epoch):
            return
        every = self.mode == "all" or epoch == 1
        frames = []
        for title, tensor, style in panels:
            values = (tensor if every else tensor[:1]).detach().to(device="cpu").float().numpy()
            frames.append([(title, display_plane(plane, style)) for plane in values])
        packet = (model, epoch, epochs, frames, details if every else details[:1])
        while self.active(epoch):
            try:
                self.frames.put(packet, block=every, timeout=.12 if every else None)
                self.last_preview = monotonic()
                return
            except Full:
                if not every:
                    return

    def finish(self):
        while not self.closed.is_set() and self.process.is_alive():
            try:
                self.frames.put(None, timeout=.12)
                self.process.join()
                break
            except Full:
                continue
        self.frames.cancel_join_thread()
        self.frames.close()


def display_plane(plane, style):
    """Stretch display pixels without changing the training tensors."""
    if style == "mask":
        return (255 * np.nan_to_num(plane).clip(0, 1)).astype(np.uint8)
    if style == "heat":
        signal = np.nan_to_num(plane).clip(0, 1)
        return (np.stack((signal, signal**2, .45 * signal**4), axis=-1) * 255).astype(np.uint8)
    if style == "signed":
        warm, cool = np.array([235, 154, 76]), np.array([86, 155, 228])
        scale = max(float(np.percentile(np.abs(plane), 98)), 1e-6)
        signal = np.clip(np.nan_to_num(plane) / scale, -1, 1)
        return (19 + np.abs(signal)[..., None] * np.where(signal[..., None] >= 0, warm, cool)).clip(0, 255).astype(np.uint8)
    low, high = np.percentile(np.nan_to_num(plane), (3, 99))
    scaled = ((np.nan_to_num(plane) - low) / max(high - low, 1e-6)).clip(0, 1)
    return (np.sqrt(scaled) * 255).astype(np.uint8)


def captured_forward(model, layer, *inputs):
    """Return the real training logits and the first spatial feature map."""
    features = []

    def capture(_, __, output):
        if not features:
            features.append(output.detach().abs().mean(dim=1))

    hook = layer.register_forward_hook(capture)
    try:
        logits = model(*inputs)
    finally:
        hook.remove()
    return logits, features[0]


def panel_layout(count, width, height, caption_height, gap):
    """Choose the grid that fits the largest square images."""
    options = []
    for columns in range(1, count + 1):
        rows = ceil(count / columns)
        side = min(width // columns - 2 * gap, height // rows - caption_height - 2 * gap)
        options.append((side, -(rows * columns - count), columns, rows))
    side, _, columns, rows = max(options)
    return columns, rows, max(1, side)


def run_window(frames, closed):
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as exc:
        print(f"Training preview unavailable: {exc}", flush=True)
        closed.set()
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Unable to open training preview: {exc}", flush=True)
        closed.set()
        return
    root.title("Hermae | Live training images")
    root.configure(background="#101513")
    root.minsize(360, 330)
    root.geometry(f"{min(1500, round(root.winfo_screenwidth() * .85))}x"
                  f"{min(950, round(root.winfo_screenheight() * .8))}")
    text_options = dict(bg="#101513", fg="#dce8de", font=("Consolas", 11))
    heading = tk.Label(root, text="Waiting for training images", font=("Consolas", 17, "bold"),
                       bg="#101513", fg="#a4dfb1", justify="left")
    heading.pack(anchor="w", padx=15, pady=(14, 4))
    status = tk.Label(root, text="The first live batch will appear here.", justify="left", **text_options)
    status.pack(anchor="w", padx=15)
    gallery = tk.Frame(root, bg="#101513")
    gallery.pack(fill="both", expand=True, padx=12, pady=8)
    gallery.grid_propagate(False)
    cells, images, captions = [], [], []
    for _ in range(5):
        cell = tk.Frame(gallery, bg="#101513")
        caption = tk.Label(cell, **text_options)
        caption.pack(fill="x", pady=(0, 4))
        image = tk.Label(cell, bg="#101513", borderwidth=0, highlightthickness=0)
        image.pack()
        cells.append(cell)
        images.append(image)
        captions.append(caption)
    controls = tk.Frame(root, bg="#101513")
    controls.pack(fill="x", padx=15, pady=(0, 12))
    paused = tk.BooleanVar(value=False)
    delay = tk.IntVar(value=25)
    toggle = tk.Checkbutton(controls, text="Pause playback", variable=paused, selectcolor="#26302a",
                            activebackground="#101513", activeforeground="#a4dfb1", **text_options)
    toggle.pack(side="left")
    tk.Label(controls, text="Delay (ms)", **text_options).pack(side="left", padx=(8, 4))
    tk.Scale(controls, from_=15, to=500, variable=delay, orient="horizontal", length=100,
             highlightthickness=0, troughcolor="#344238", **text_options).pack(side="left", fill="x", expand=True)
    pending = deque()
    shown = 0
    current = ()
    layout = None
    resize_job = None
    gap = 6

    def draw_frame():
        nonlocal layout, resize_job
        if resize_job is not None:
            root.after_cancel(resize_job)
            resize_job = None
        if not current:
            return
        width, height = gallery.winfo_width(), gallery.winfo_height()
        caption_height = (20 if width >= 620 else 38) + 4
        columns, rows, side = panel_layout(len(current), width, height, caption_height, gap)
        arrangement = (len(current), columns, rows)
        if arrangement != layout:
            layout = arrangement
            for i in range(10):
                active = i < 2 * columns
                gallery.grid_columnconfigure(i, weight=int(active), uniform="panel" if active else "")
            for i in range(5):
                active = i < rows
                gallery.grid_rowconfigure(i, weight=int(active), uniform="panel" if active else "")
            for i, cell in enumerate(cells):
                if i >= len(current):
                    cell.grid_remove()
                    continue
                row, column = divmod(i, columns)
                row_count = min(columns, len(current) - row * columns)
                cell.grid(row=row, column=2 * column + columns - row_count, columnspan=2, padx=gap, pady=gap)
        for i, (title, values) in enumerate(current):
            image = images[i]
            photo = Image.fromarray(values).resize((side, side), Image.Resampling.NEAREST)
            image.photo = ImageTk.PhotoImage(photo)
            image.configure(image=image.photo)
            captions[i].configure(text=title, wraplength=max(1, width // columns - 2 * gap))

    def on_resize(event):
        nonlocal resize_job
        heading.configure(wraplength=event.width)
        status.configure(wraplength=event.width)
        if resize_job is None:
            resize_job = root.after_idle(draw_frame)

    gallery.bind("<Configure>", on_resize)

    def close():
        closed.set()
        root.destroy()

    def update():
        nonlocal shown, current
        if not paused.get():
            if not pending:
                try:
                    packet = frames.get_nowait()
                    if packet is None:
                        close()
                        return
                    model, epoch, epochs, panels, details = packet
                    pending.extend((model, epoch, epochs, pairs, detail) for pairs, detail in
                                   zip(zip(*panels), details))
                except Empty:
                    pass
            if pending:
                model, epoch, epochs, current, detail = pending.popleft()
                shown += 1
                heading.configure(text=f"{model}   |   epoch {epoch}/{epochs}")
                status.configure(text=f"Image {shown:,}  ·  {detail}")
                draw_frame()
        root.after(delay.get(), update)

    root.protocol("WM_DELETE_WINDOW", close)

    def callback_error(kind, error, traceback):
        print(f"Training preview stopped: {kind.__name__}: {error}", flush=True)
        close()

    root.report_callback_exception = callback_error
    root.after(0, update)
    try:
        root.mainloop()
    finally:
        closed.set()

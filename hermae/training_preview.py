"""Native playback of images and outputs from real training forward passes."""

import atexit
from collections import deque
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
    root.minsize(780, 360)
    root.geometry("1080x415")
    text_options = dict(bg="#101513", fg="#dce8de", font=("Consolas", 11))
    heading = tk.Label(root, text="Waiting for training images", font=("Consolas", 17, "bold"),
                       bg="#101513", fg="#a4dfb1")
    heading.pack(anchor="w", padx=15, pady=(14, 4))
    status = tk.Label(root, text="The first live batch will appear here.", **text_options)
    status.pack(anchor="w", padx=15)
    gallery = tk.Frame(root, bg="#101513")
    gallery.pack(fill="both", expand=True, padx=12, pady=8)
    cells, images, captions = [], [], []
    for index in range(5):
        cell = tk.Frame(gallery, bg="#101513")
        cell.grid(row=0, column=index, sticky="nsew", padx=4)
        gallery.grid_columnconfigure(index, weight=1, uniform="panel")
        caption = tk.Label(cell, **text_options)
        caption.pack(fill="x", pady=(0, 4))
        image = tk.Label(cell, bg="#101513")
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
    tk.Label(controls, text="Milliseconds per image", **text_options).pack(side="left", padx=(15, 4))
    tk.Scale(controls, from_=15, to=500, variable=delay, orient="horizontal", length=150,
             highlightthickness=0, troughcolor="#344238", **text_options).pack(side="left")
    pending = deque()
    shown = 0

    def close():
        closed.set()
        root.destroy()

    def update():
        nonlocal shown
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
                model, epoch, epochs, pairs, detail = pending.popleft()
                shown += 1
                heading.configure(text=f"{model}   |   epoch {epoch}/{epochs}")
                status.configure(text=f"Image {shown:,}  ·  {detail}")
                size = max(80, min(225, (root.winfo_width() - 70) // len(pairs), gallery.winfo_height() - 32))
                for index, (cell, caption, image) in enumerate(zip(cells, captions, images)):
                    if index >= len(pairs):
                        cell.grid_remove()
                        continue
                    cell.grid()
                    title, values = pairs[index]
                    photo = Image.fromarray(values).resize((size, size), Image.Resampling.NEAREST)
                    image.photo = ImageTk.PhotoImage(photo)
                    image.configure(image=image.photo)
                    caption.configure(text=title)
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

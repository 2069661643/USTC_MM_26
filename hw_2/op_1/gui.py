import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk

from svdimage import SVDImage


def log(module_name: str, info: str) -> None:
    """按要求打印关键活动日志。"""
    print(f"[{datetime.now():%H:%M:%S}] <{module_name}> : {info}")


class ZoomableImageView(ttk.Frame):
    """提供图像等比缩放与平移的显示控件，不修改底层图像对象。"""

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#1f1f1f", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._pil_image: Image.Image | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._need_refit = False

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

    def set_image(self, image: Image.Image, reset_view: bool = True) -> None:
        self._pil_image = image
        if reset_view:
            # 首次布局阶段画布尺寸可能尚未稳定，延后一次自适应可避免图像过小。
            self._need_refit = True
            self._fit_to_canvas()
        self._render()

    def _fit_to_canvas(self) -> None:
        if self._pil_image is None:
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        if canvas_w <= 2 or canvas_h <= 2:
            self._need_refit = True
            return
        img_w, img_h = self._pil_image.size
        fit_scale = min(canvas_w / max(1, img_w), canvas_h / max(1, img_h))

        self._zoom = max(0.05, min(20.0, fit_scale))
        self._offset_x = 0.0
        self._offset_y = 0.0

    def _on_configure(self, _event: tk.Event) -> None:
        if self._pil_image is not None:
            if self._need_refit:
                self._fit_to_canvas()
                self._need_refit = False
            self._render()

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_move(self, event: tk.Event) -> None:
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self._drag_start_x = event.x
        self._drag_start_y = event.y

        self._offset_x += dx
        self._offset_y += dy
        self._render()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self._pil_image is None:
            return

        if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4:
            factor = 1.1
        else:
            factor = 1.0 / 1.1

        self._zoom = max(0.05, min(20.0, self._zoom * factor))
        self._render()

    def _render(self) -> None:
        self.canvas.delete("all")
        if self._pil_image is None:
            return

        img_w, img_h = self._pil_image.size
        target_w = max(1, int(round(img_w * self._zoom)))
        target_h = max(1, int(round(img_h * self._zoom)))
        resized = self._pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        center_x = canvas_w // 2 + int(self._offset_x)
        center_y = canvas_h // 2 + int(self._offset_y)

        self.canvas.create_image(center_x, center_y, image=self._tk_image, anchor=tk.CENTER)


class EmbeddedWindow:
    """主窗口中的内嵌活动窗体，支持拖拽、收起、放大、关闭。"""

    def __init__(
        self,
        desktop: tk.Frame,
        title: str,
        module_name: str,
        x: int,
        y: int,
        width: int,
        height: int,
        on_close,
    ) -> None:
        self.desktop = desktop
        self.module_name = module_name
        self.on_close_callback = on_close

        self.frame = tk.Frame(desktop, bd=1, relief=tk.RAISED, bg="#f6f7fb")
        self.frame.place(x=x, y=y, width=width, height=height)

        self.title_bar = tk.Frame(self.frame, bg="#2f3d52", height=28)
        self.title_bar.pack(fill=tk.X, side=tk.TOP)

        self.title_var = tk.StringVar(value=title)
        self.title_label = tk.Label(
            self.title_bar,
            textvariable=self.title_var,
            fg="#f0f4ff",
            bg="#2f3d52",
            anchor=tk.W,
            padx=8,
        )
        self.title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_max = tk.Button(self.title_bar, text="□", width=2, command=self.toggle_maximize)
        self.btn_min = tk.Button(self.title_bar, text="_", width=2, command=self.toggle_minimize)
        self.btn_close = tk.Button(self.title_bar, text="X", width=2, command=self.close)
        self.btn_close.pack(side=tk.RIGHT, padx=(0, 2), pady=2)
        self.btn_min.pack(side=tk.RIGHT, padx=(0, 2), pady=2)
        self.btn_max.pack(side=tk.RIGHT, padx=(0, 2), pady=2)

        self.body = ttk.Frame(self.frame)
        self.body.pack(fill=tk.BOTH, expand=True)

        self.size_grip = tk.Label(self.frame, text="◢", bg="#f6f7fb", fg="#666")
        self.size_grip.place(relx=1.0, rely=1.0, anchor=tk.SE)

        self._drag_x = 0
        self._drag_y = 0
        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self._resize_w = 0
        self._resize_h = 0
        self._resize_start_x = 0
        self._resize_start_y = 0
        self._normal_geometry: tuple[int, int, int, int] | None = None
        self._minimized = False
        self._maximized = False

        for widget in (self.frame, self.title_bar, self.title_label):
            widget.bind("<Button-1>", self._on_focus)

        self.title_bar.bind("<ButtonPress-1>", self._on_drag_start)
        self.title_bar.bind("<B1-Motion>", self._on_drag_move)
        self.title_label.bind("<ButtonPress-1>", self._on_drag_start)
        self.title_label.bind("<B1-Motion>", self._on_drag_move)

        self.size_grip.bind("<ButtonPress-1>", self._on_resize_start)
        self.size_grip.bind("<B1-Motion>", self._on_resize_move)

    def exists(self) -> bool:
        return bool(self.frame.winfo_exists())

    def set_title(self, title: str) -> None:
        self.title_var.set(title)

    def lift(self) -> None:
        self.frame.lift()

    def _on_focus(self, _event: tk.Event) -> None:
        self.lift()

    def _desktop_size(self) -> tuple[int, int]:
        self.desktop.update_idletasks()
        return max(1, self.desktop.winfo_width()), max(1, self.desktop.winfo_height())

    def _on_drag_start(self, event: tk.Event) -> None:
        self.lift()
        self._drag_x = event.x_root
        self._drag_y = event.y_root
        # 记录鼠标在窗体左上角坐标系中的偏移，保证拖动连续不跳变。
        self._drag_offset_x = event.x_root - self.frame.winfo_rootx()
        self._drag_offset_y = event.y_root - self.frame.winfo_rooty()

    def _on_drag_move(self, event: tk.Event) -> None:
        if self._maximized:
            return

        self._drag_x = event.x_root
        self._drag_y = event.y_root

        desktop_root_x = self.desktop.winfo_rootx()
        desktop_root_y = self.desktop.winfo_rooty()
        x = event.x_root - desktop_root_x - self._drag_offset_x
        y = event.y_root - desktop_root_y - self._drag_offset_y
        w = self.frame.winfo_width()
        h = self.frame.winfo_height()

        desktop_w, desktop_h = self._desktop_size()
        x = max(0, min(desktop_w - w, x))
        y = max(0, min(desktop_h - 28, y))
        self.frame.place(x=x, y=y)

    def _on_resize_start(self, event: tk.Event) -> None:
        if self._maximized:
            return
        self.lift()
        self._resize_start_x = event.x_root
        self._resize_start_y = event.y_root
        self._resize_w = self.frame.winfo_width()
        self._resize_h = self.frame.winfo_height()

    def _on_resize_move(self, event: tk.Event) -> None:
        if self._maximized:
            return

        dx = event.x_root - self._resize_start_x
        dy = event.y_root - self._resize_start_y
        new_w = max(320, self._resize_w + dx)
        new_h = max(180, self._resize_h + dy)

        desktop_w, desktop_h = self._desktop_size()
        x = self.frame.winfo_x()
        y = self.frame.winfo_y()
        new_w = min(new_w, desktop_w - x)
        new_h = min(new_h, desktop_h - y)

        self.frame.place(width=new_w, height=new_h)

    def toggle_minimize(self) -> None:
        if self._minimized:
            self.body.pack(fill=tk.BOTH, expand=True)
            self.size_grip.place(relx=1.0, rely=1.0, anchor=tk.SE)
            if self._normal_geometry is not None:
                _, _, w, h = self._normal_geometry
                self.frame.place(width=w, height=h)
            self._minimized = False
            log(self.module_name, f"restore {self.title_var.get()}")
            return

        self._normal_geometry = (
            self.frame.winfo_x(),
            self.frame.winfo_y(),
            self.frame.winfo_width(),
            self.frame.winfo_height(),
        )
        self.body.pack_forget()
        self.size_grip.place_forget()
        self.frame.place(height=28)
        self._minimized = True
        self._maximized = False
        log(self.module_name, f"minimize {self.title_var.get()}")

    def toggle_maximize(self) -> None:
        desktop_w, desktop_h = self._desktop_size()

        if self._maximized:
            if self._normal_geometry is not None:
                x, y, w, h = self._normal_geometry
                self.frame.place(x=x, y=y, width=w, height=h)
            self._maximized = False
            log(self.module_name, f"restore {self.title_var.get()}")
            return

        self._normal_geometry = (
            self.frame.winfo_x(),
            self.frame.winfo_y(),
            self.frame.winfo_width(),
            self.frame.winfo_height(),
        )
        if self._minimized:
            self.body.pack(fill=tk.BOTH, expand=True)
            self.size_grip.place(relx=1.0, rely=1.0, anchor=tk.SE)
            self._minimized = False

        self.frame.place(x=0, y=0, width=desktop_w, height=desktop_h)
        self._maximized = True
        log(self.module_name, f"maximize {self.title_var.get()}")

    def close(self) -> None:
        if not self.exists():
            return
        self.frame.destroy()
        self.on_close_callback()


class TargetSVDCWindow:
    """Target 的 SVDC 子窗口，负责压缩计算与保存。"""

    def __init__(
        self,
        parent: "TargetWindow",
        original_image: Image.Image,
        image_name: str,
        close_callback,
        x: int,
        y: int,
    ) -> None:
        self.parent = parent
        self.image_name = image_name
        self.close_callback = close_callback

        self.window = EmbeddedWindow(
            desktop=parent.app.desktop,
            title=f"{image_name}-SVDC",
            module_name="Target-SVDC",
            x=x,
            y=y,
            width=900,
            height=560,
            on_close=self.on_close,
        )
        log("Target-SVDC", f"create window {image_name}-SVDC")

        self.original_image = original_image.copy().convert("RGB")
        self.current_image = self.original_image.copy()
        self.image_w = self.original_image.width
        self.image_h = self.original_image.height
        self.min_wh = min(self.original_image.width, self.original_image.height)
        self.current_rank = self.min_wh

        self.svd_image = SVDImage(self.original_image)
        log("SVDImage", f"create for {self.image_name}")

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Panedwindow(self.window.body, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(container)
        right_frame = ttk.Frame(container, padding=12)
        container.add(left_frame, weight=4)
        container.add(right_frame, weight=2)

        self.viewer = ZoomableImageView(left_frame)
        self.viewer.pack(fill=tk.BOTH, expand=True)
        self.viewer.set_image(self.original_image, reset_view=True)

        action_row = ttk.Frame(right_frame)
        action_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action_row, text="Save", command=self.save_default).pack(side=tk.LEFT)
        ttk.Label(action_row, text=" or ").pack(side=tk.LEFT, padx=6)
        ttk.Button(action_row, text="Save as", command=self.save_as).pack(side=tk.LEFT)

        path_row = ttk.Frame(right_frame)
        path_row.pack(fill=tk.X, pady=(0, 12))

        self.save_path_var = tk.StringVar(value="./out/out")
        ttk.Entry(path_row, textvariable=self.save_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ext_var = tk.StringVar(value=".png")
        self.ext_combo = ttk.Combobox(
            path_row,
            textvariable=self.ext_var,
            state="readonly",
            values=[".png", ".jpg", ".bmp"],
            width=7,
        )
        self.ext_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.ext_combo.current(0)
        self.ext_combo.bind("<<ComboboxSelected>>", self._on_ext_changed)

        rank_row = ttk.Frame(right_frame)
        rank_row.pack(fill=tk.X)
        ttk.Label(rank_row, text="rank = ").pack(side=tk.LEFT)

        self.rank_var = tk.StringVar(value=str(self.min_wh))
        self.rank_entry = ttk.Entry(rank_row, textvariable=self.rank_var, width=10)
        self.rank_entry.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(rank_row, text="Compute", command=self.compute).pack(side=tk.LEFT)
        self.rank_entry.bind("<Return>", self._on_rank_entry_commit)
        self.rank_entry.bind("<FocusOut>", self._on_rank_entry_commit)

        self.rank_scale = tk.Scale(
            right_frame,
            from_=0,
            to=self.min_wh,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=False,
            command=self._on_rank_scale_change,
        )
        self.rank_scale.pack(fill=tk.X, pady=(8, 4))
        self.rank_scale.set(self.min_wh)

        self.ratio_var = tk.StringVar()
        ttk.Label(right_frame, textvariable=self.ratio_var).pack(fill=tk.X, pady=(0, 10))
        self._update_ratio_text(self.min_wh)

    def _clip_rank(self, rank: int) -> int:
        return max(0, min(self.min_wh, rank))

    def _on_rank_scale_change(self, value: str) -> None:
        rank = self._clip_rank(int(float(value)))
        self.rank_var.set(str(rank))
        self._update_ratio_text(rank)

    def _on_rank_entry_commit(self, _event: tk.Event) -> None:
        raw = self.rank_var.get().strip()
        try:
            rank = int(raw)
        except ValueError:
            return

        rank = self._clip_rank(rank)
        self.rank_var.set(str(rank))
        self.rank_scale.set(rank)
        self._update_ratio_text(rank)

    def _update_ratio_text(self, rank: int) -> None:
        total = self.image_h * self.image_w
        if total <= 0:
            ratio = 0.0
        else:
            ratio = 100.0 * ((self.image_h + self.image_w) * rank * 8) / total
        self.ratio_var.set(f"Expected compression ratio is {ratio:.2f}%")

    def _on_ext_changed(self, _event: tk.Event) -> None:
        self.window.set_title(f"{self.image_name}-SVDC ({self.ext_var.get()})")

    def save_default(self) -> None:
        out_dir = Path("./out")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = out_dir / f"out_r{self.current_rank}_{timestamp}.png"
        self.current_image.save(file_path)
        log("Target-SVDC", f"save image {file_path}")

    def save_as(self) -> None:
        base = self.save_path_var.get().strip()
        if not base:
            messagebox.showinfo("Info", "Save path is empty.")
            return

        target_path = Path(f"{base}{self.ext_var.get()}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_image.save(target_path)
        log("Target-SVDC", f"save as image {target_path}")

    def compute(self) -> None:
        raw_rank = self.rank_var.get().strip()
        try:
            rank = int(raw_rank)
        except ValueError:
            messagebox.showinfo("Info", "rank should be a number!")
            log("Target-SVDC", f"compute rejected invalid rank {raw_rank!r}")
            return

        rank = self._clip_rank(rank)
        self.rank_var.set(str(rank))
        self.rank_scale.set(rank)
        self._update_ratio_text(rank)

        try:
            self.current_image = self.svd_image.svdmatrix2image(rank=rank)
        except Exception as exc:
            messagebox.showinfo("Info", f"Compute failed: {exc}")
            log("Target-SVDC", f"compute failed rank={rank}, err={exc}")
            return

        self.current_rank = rank
        self.viewer.set_image(self.current_image, reset_view=False)
        log("Target-SVDC", f"compute rank={rank}")

    def on_close(self) -> None:
        log("SVDImage", f"release for {self.image_name}")
        self.svd_image = None
        log("Target-SVDC", f"close window {self.image_name}-SVDC")
        self.close_callback()


class TargetWindow:
    """由用户加载的图像活动窗口，内嵌于主窗口。"""

    def __init__(self, app: "SVDCWindow", image_path: Path, image: Image.Image, x: int, y: int) -> None:
        self.app = app
        self.image_path = image_path
        self.image = image.copy().convert("RGB")
        self.image_name = image_path.name

        self.window = EmbeddedWindow(
            desktop=app.desktop,
            title=self.image_name,
            module_name="Target",
            x=x,
            y=y,
            width=680,
            height=460,
            on_close=self.on_close,
        )
        log("Target", f"create window {self.image_name}")

        self.svdc_window: TargetSVDCWindow | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        action_bar = ttk.Frame(self.window.body, padding=(8, 8, 8, 4))
        action_bar.pack(fill=tk.X)

        ttk.Button(action_bar, text="SVDC", command=self.open_svdc).pack(side=tk.LEFT)
        ttk.Label(action_bar, text="Original Image Preview").pack(side=tk.LEFT, padx=(10, 0))

        preview_frame = ttk.Frame(self.window.body, padding=(8, 4, 8, 8))
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.viewer = ZoomableImageView(preview_frame)
        self.viewer.pack(fill=tk.BOTH, expand=True)
        self.viewer.set_image(self.image, reset_view=True)

    def open_svdc(self) -> None:
        if self.svdc_window is not None and self.svdc_window.window.exists():
            log("Target", f"SVDC already exists for {self.image_name}")
            self.svdc_window.window.lift()
            return

        self.svdc_window = TargetSVDCWindow(
            parent=self,
            original_image=self.image,
            image_name=self.image_name,
            close_callback=self._on_svdc_closed,
            x=self.window.frame.winfo_x() + 40,
            y=self.window.frame.winfo_y() + 40,
        )

    def _on_svdc_closed(self) -> None:
        self.svdc_window = None

    def on_close(self) -> None:
        if self.svdc_window is not None and self.svdc_window.window.exists():
            self.svdc_window.window.close()
            log("Target", f"close child SVDC for {self.image_name}")

        self.app.unregister_target(self)
        log("Target", f"close window {self.image_name}")


class SVDCWindow:
    """主窗口封装类。"""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("SVDC GUI")
        self.root.geometry("1200x760")
        self.root.minsize(960, 620)

        self.targets: list[TargetWindow] = []
        self._build_main_ui()

        log("SVDCWindow", "main window created")

    def _build_main_ui(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load image...", command=self.load_image)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        workspace = ttk.Frame(self.root, padding=12)
        workspace.pack(fill=tk.BOTH, expand=True)

        self.desktop = tk.Frame(workspace, bg="#d9dee7", bd=1, relief=tk.SUNKEN)
        self.desktop.pack(fill=tk.BOTH, expand=True)

        bg_label = tk.Label(
            self.desktop,
            text="Activity Workspace",
            bg="#d9dee7",
            fg="#26384d",
            font=("Segoe UI", 14, "bold"),
            anchor=tk.NW,
            justify=tk.LEFT,
        )
        bg_label.place(x=12, y=10)

    def load_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not selected:
            return

        path = Path(selected)
        try:
            with Image.open(path) as img:
                loaded_image = img.copy()
        except Exception:
            messagebox.showinfo("Info", f"Load image failed: {path}")
            log("SVDCWindow", f"load image failed {path}")
            return

        # 采用层叠偏移，避免多个窗口完全重叠。
        index = len(self.targets)
        x = 40 + (index % 6) * 28
        y = 56 + (index % 6) * 24

        target = TargetWindow(self, path, loaded_image, x=x, y=y)
        self.targets.append(target)
        log("SVDCWindow", f"load image success {path}")

    def unregister_target(self, target: TargetWindow) -> None:
        if target in self.targets:
            self.targets.remove(target)

    def run(self) -> None:
        log("SVDCWindow", "run main loop")
        self.root.mainloop()


if __name__ == "__main__":
    SVDCWindow().run()

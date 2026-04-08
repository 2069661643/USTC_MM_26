import PIL
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import time

from dft.dft import DFTMatCompress, DDFTMatCompress, ZipMat, ZipFig, DezipFig


class DFTImage:
    """基于 DFT + 游程编码的图像压缩封装。"""

    def __init__(self, image: PIL.Image.Image | str | Path):
        if isinstance(image, (str, Path)):
            self.image = Image.open(image).convert("RGB")
        elif isinstance(image, PIL.Image.Image):
            self.image = image.convert("RGB")
        else:
            raise TypeError("Parameter image must be str or PIL.Image.Image")

        self.last_bin_size = 0
        self.last_compression_ratio = 0.0

    def dftmatrix2image(self, mode: int | str = "max", err: float = 50) -> PIL.Image.Image:
        """执行 DFT 压缩并还原图像，返回还原图，同时记录中间 bin 文件大小。"""
        img_array = np.array(self.image, dtype=np.float64)
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        dr = DFTMatCompress(r, size=mode, show=False)
        dg = DFTMatCompress(g, size=mode, show=False)
        db = DFTMatCompress(b, size=mode, show=False)

        zr = ZipMat(dr, err=err)
        zg = ZipMat(dg, err=err)
        zb = ZipMat(db, err=err)

        out_dir = Path("./out")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bin_path = out_dir / f"dft_tmp_{stamp}.bin"

        try:
            ZipFig(zr, zg, zb, filepath=str(bin_path))
            self.last_bin_size = bin_path.stat().st_size

            rr, rg, rb = DezipFig(filepath=str(bin_path))
            ir = DDFTMatCompress(rr, size=mode, show=False)
            ig = DDFTMatCompress(rg, size=mode, show=False)
            ib = DDFTMatCompress(rb, size=mode, show=False)
        finally:
            if bin_path.exists():
                for _ in range(10):
                    try:
                        bin_path.unlink()
                        break
                    except PermissionError:
                        time.sleep(0.05)

        out = np.stack([
            np.clip(np.round(ir.real), 0, 255).astype(np.uint8),
            np.clip(np.round(ig.real), 0, 255).astype(np.uint8),
            np.clip(np.round(ib.real), 0, 255).astype(np.uint8),
        ], axis=2)

        h, w = out.shape[:2]
        denom = max(1, h * w * 3)
        self.last_compression_ratio = 100.0 * self.last_bin_size / denom

        return Image.fromarray(out, "RGB")

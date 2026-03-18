import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage import io

## read image
im = io.imread('../figs/original.png')
if im.ndim == 3 and im.shape[2] == 4:
    im = im[:, :, :3]

## draw 2 copies of the image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(bottom=0.22)
ax1.imshow(im)
ax1.set_title('Input image')
ax1.axis('off')
himg = ax2.imshow(np.zeros_like(im))
ax2.set_title('Resized Image\nAdjust sliders and click the button')
ax2.axis('off')

slider_col_ax = fig.add_axes([0.15, 0.10, 0.30, 0.03])
slider_row_ax = fig.add_axes([0.15, 0.05, 0.30, 0.03])
slider_col = Slider(slider_col_ax, 'Col scale', 0.5, 2.0, valinit=1.0)
slider_row = Slider(slider_row_ax, 'Row scale', 0.5, 2.0, valinit=1.0)

btn_ax = fig.add_axes([0.60, 0.06, 0.20, 0.06])
btn = Button(btn_ax, 'Seam Carving', color='lightblue', hovercolor='deepskyblue')

def on_click(event):
    h, w = im.shape[:2]
    target_w = max(1, int(w * slider_col.val))
    target_h = max(1, int(h * slider_row.val))
    result = seam_carve_image(im, (target_h, target_w))
    himg.set_data(result)
    himg.set_extent([0, result.shape[1], result.shape[0], 0])
    ax2.set_title(f'Resized Image ({result.shape[0]}x{result.shape[1]})')
    fig.canvas.draw_idle()

btn.on_clicked(on_click)


## TODO: implement function: seam_carve_image
def seam_carve_image(im, sz):
    """Seam carving to resize image to target size.

    Args:
        im: (h, w, 3) input RGB image (uint8)
        sz: (target_h, target_w) target size

    Returns:
        resized image of shape (target_h, target_w, 3)
    """
    raise NotImplementedError


plt.show()

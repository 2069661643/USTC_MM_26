import PIL
import numpy as np
from svd.svd import svd as svd
from PIL import Image
from pathlib import Path

class SVDImage:
    """
    A class for image compression using Singular Value Decomposition (SVD).
    
    This class decomposes an RGB image into three separate SVD matrices (one for each color channel)
    and allows reconstruction of the image with reduced rank for compression purposes.
    
    :param image: The input image or path to image file.
    :type image: PIL.Image.Image | str | Path
    
    :ivar image: The original image.
    :ivar rU, rS, rV: SVD matrices for the red channel.
    :ivar gU, gS, gV: SVD matrices for the green channel.
    :ivar bU, bS, bV: SVD matrices for the blue channel.
    
    Example usage:
        >>> from PIL import Image
        >>> svd_img = SVDImage('example.jpg')
        >>> compressed = svd_img.svdmatrix2image(rank=50)
        >>> compressed.save('compressed.jpg')
    """
    def __init__(self, image : PIL.Image.Image | str | Path):
        if isinstance(image, (str, Path)):  # 支持str和pathlib.Path
            self.image = Image.open(image)
        elif isinstance(image, PIL.Image.Image):
            self.image = image
        else:
            raise TypeError("Parameter image must be str or PIL.Image.Image")
        self.rU = None
        self.rS = None
        self.rV = None
        self.gU = None
        self.gS = None
        self.gV = None
        self.bU = None
        self.bS = None
        self.bV = None
        self.image2svdmatrix(image)

    def image2svdmatrix(self, image : PIL.Image.Image):
        img_array = np.array(image.convert('RGB'), dtype=np.float64)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        self.rU, self.rS, self.rV = svd(r)
        self.gU, self.gS, self.gV = svd(g)
        self.bU, self.bS, self.bV = svd(b)

    def svdmatrix2image(self, rank : None | int = None, err = 1e-8 ) -> PIL.Image.Image:
        """
        According to the rank you have given, generate the image

        :param rank: The rank of the svd you want to use. If `None`, rank will be min(*image.shape).
        :type rank: None | int
        :param err: The error tolerance you want to use.
        :type err: float
        :return: the image
        :rtype: PIL.Image.Image
        """
        if rank is None:
            rank = min(*self.rS.shape)
        # rank clip
        max_rank = min(
            self.rS.shape[0], self.rS.shape[1],
            self.gS.shape[0], self.gS.shape[1],
            self.bS.shape[0], self.bS.shape[1],
            self.rU.shape[1], self.gU.shape[1], self.bU.shape[1],
            self.rV.shape[1], self.gV.shape[1], self.bV.shape[1],
        )
        rank = max(0, min(int(rank), max_rank))

        r = np.zeros(self.rS.shape)
        g = np.zeros(self.gS.shape)
        b = np.zeros(self.bS.shape)

        for i in range(rank):
            sigma = self.rS[i, i]
            if sigma <= err:
                break
            r += sigma * (self.rU[:, i:i+1] @ self.rV[:, i:i+1].T)
        for i in range(rank):
            sigma = self.gS[i, i]
            if sigma <= err:
                break
            g += sigma * (self.gU[:, i:i+1] @ self.gV[:, i:i+1].T)
        for i in range(rank):
            sigma = self.bS[i, i]
            if sigma <= err:
                break
            b += sigma * (self.bU[:, i:i+1] @ self.bV[:, i:i+1].T)

        r = np.clip(np.round(r), 0, 255).astype(np.uint8)
        g = np.clip(np.round(g), 0, 255).astype(np.uint8)
        b = np.clip(np.round(b), 0, 255).astype(np.uint8)

        img_array = np.stack([r, g, b], axis=2)
        return Image.fromarray(img_array, 'RGB')


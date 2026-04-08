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

        r = np.zeros(self.rS.shape)
        g = np.zeros(self.gS.shape)
        b = np.zeros(self.bS.shape)

        for i in range(rank):
            if self.rS[i] <= err:
                break
            r += self.rS[i,i] * self.rU[:,i:i+1] @ self.rV[:,i:i+1]
        for i in range(rank):
            if self.gS[i] <= err:
                break
            g += self.gS[i,i] * self.gU[:,i:i+1] @ self.gV[:,i:i+1]
        for i in range(rank):
            if self.bS[i] <= err:
                break
            b += self.bS[i,i] * self.bU[:,i:i+1] @ self.bV[:,i:i+1]

        r = np.clip(np.round(r), 0, 255).astype(np.uint8)
        g = np.clip(np.round(g), 0, 255).astype(np.uint8)
        b = np.clip(np.round(b), 0, 255).astype(np.uint8)

        img_array = np.stack([r, g, b], axis=2)
        return Image.fromarray(img_array, 'RGB')


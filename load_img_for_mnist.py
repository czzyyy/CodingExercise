# 参考： https://niektemme.com/2016/02/21/tensorflow-handwriting/

# load our own img for test mnist recognition model
from PIL import Image, ImageFilter


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')

    im.save('read.png')
    im_flap = list(im.getdata())
    # print('img_data:', im_flap)
    if sum(list(im.getdata())[:10]) > 10 * 125:
        im_flap = [(255 - x) for x in list(im.getdata())]
    im.putdata(im_flap)
    im.save('flap.png')

    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), 0)  # creates black canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if nheight == 0:  # rare case but minimum is 1 pixel
            nheigth = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if nwidth == 0:  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    newImage.save('sample.png')

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    # tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    tva = [x * 1.0 / 255.0 for x in tv]
    # tva = tv  # 0~255
    return tva
    # print(tva)

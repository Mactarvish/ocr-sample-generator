from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


if __name__ == '__main__':
    canvas_np = np.ones((500, 500, 3), dtype=np.uint8) * 230
    canvas_pil = Image.fromarray(canvas_np)
    drawer = ImageDraw.Draw(canvas_pil)

    font1 = ImageFont.truetype("assets/fonts/en/Average Sans.ttf", size=50)
    font2 = ImageFont.truetype("assets/fonts/en/times.ttf", size=50)
    font3 = ImageFont.truetype("assets/fonts/cn/SimSun.ttf", size=50)
    drawer.text((10, 10), "abcdefghijklmn", fill=(0, 0, 0), font=font1)
    drawer.text((10, 200), "abcdefghijklmn", fill=(0, 0, 0), font=font2)
    drawer.text((10, 400), "发为发威尔法他", fill=(0, 0, 0), font=font3)

    canvas_np = np.array(canvas_pil)
    cv2.line(canvas_np, (10, 10), (1000, 10), (0,0))
    cv2.line(canvas_np, (10, 200), (1000, 200), (0,0))
    cv2.line(canvas_np, (10, 400), (1000, 400), (0,0))
    cv2.imwrite("pgs/ee.png", canvas_np)
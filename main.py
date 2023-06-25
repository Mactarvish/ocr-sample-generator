import os
import random
import argparse
from tqdm import trange
import glob

from PIL import Image
import numpy as np

from utils.parse_yaml import parse_yaml_config
from utils.parse_py import parse_py_config
from core.ocr_sample_generator import OCRSampleGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_config_path")
    parser.add_argument("save_dir")
    parser.add_argument("generation_num", type=int)
    args = parser.parse_args()

    config = parse_py_config(args.src_config_path)
    os.makedirs(args.save_dir, exist_ok=True)
    image_save_dir = os.path.join(args.save_dir, "images")
    label_save_dir = os.path.join(args.save_dir, "labels")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    for gn in trange(args.generation_num):
        W = np.random.randint(*config.IMAGE_WIDTH)
        H = np.random.randint(*config.IMAGE_HEIGHT)
        FONT_SIZE = np.random.randint(*config.FONT_SIZE)
        LINE_SPACING = max(3, np.random.randint(int(1 / 5 * FONT_SIZE), int(15 / 25 * FONT_SIZE)))
        CHAR_SPACING = random.sample(config.CHAR_SPACING, 1)[0]
        img = Image.new("RGB", (W, H), tuple(config.IMAGE_BASE_COLOR))

        cn_font_paths = glob.glob(os.path.join("assets/fonts/cn", "*.ttf"))
        en_font_paths = glob.glob(os.path.join("assets/fonts/en", "*.ttf"))
        
        cn_font_path = random.sample(cn_font_paths, 1)[0]
        en_font_path = random.sample(en_font_paths, 1)[0]

        g = OCRSampleGenerator([cn_font_path, en_font_path], img, font_size=FONT_SIZE, line_spacing=LINE_SPACING, char_spacing=CHAR_SPACING,
                                    save_dir=args.save_dir)

        tl_y_offset = np.random.randint(0, 40) - 20
        tl_y = tl_y_offset
        line_height = g.font_size
        for i in range(g.canvas_pil.height // (g.font_size + g.line_spacing) + 3):
            tl_y += (line_height + g.line_spacing)
            line_height = g.put_ptext_line(config.MAIN_LANGUAGE, tl_y)
        g.aug()

        if config.BASE_NAME == '':
            name = "%05d" % gn
        else:
            name = "%s_%05d" % (config.BASE_NAME, gn)
        g.save(name, save_visualize=True)


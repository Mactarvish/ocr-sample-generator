import os
import random
import argparse
from tqdm import trange
import glob

from PIL import Image
import numpy as np

from utils.parse_py import parse_py_config
from utils.parse_random import parse_random
from core.ocr_sample_generator import OCRSampleGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_config_path")
    parser.add_argument("save_dir")
    parser.add_argument("generation_num", type=int)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    config = parse_py_config(args.src_config_path)
    print(config)
    for gn in trange(args.generation_num):
        # 解析样本参数配置
        W = parse_random(config["LAYOUT"]["IMAGE_WIDTH"])
        H = parse_random(config["LAYOUT"]["IMAGE_HEIGHT"])
        FONT_SIZE = parse_random(config["LAYOUT"]["FONT_SIZE"])
        LINE_SPACING = max(3, parse_random(config["LAYOUT"]["LINE_SPACING"]))
        CHAR_SPACING = parse_random(config["LAYOUT"]["CHAR_SPACING"])
        SAVE_RECTIFIED_LINES_SEPARATELY = config["SAVE_RECTIFIED_LINES_SEPARATELY"] \
                                            if "SAVE_RECTIFIED_LINES_SEPARATELY" in config else False
        img = Image.new("RGB", (W, H), (255, 255, 255))
        
        cn_font_paths = glob.glob(os.path.join("assets/fonts/cn", "*.ttf"))
        en_font_paths = glob.glob(os.path.join("assets/fonts/en", "*.ttf"))
        
        cn_font_path = random.sample(cn_font_paths, 1)[0]
        en_font_path = random.sample(en_font_paths, 1)[0]
        # 实例化一个样本生成器
        g = OCRSampleGenerator([cn_font_path, en_font_path], img, font_size=FONT_SIZE, line_spacing=LINE_SPACING, char_spacing=CHAR_SPACING,
                                    save_dir=args.save_dir)

        tl_y_offset = np.random.randint(0, 40) - 20
        tl_y = tl_y_offset
        line_height = g.font_size
        for i in range(g.canvas_pil.height // (g.font_size + g.line_spacing)):
            # 绘制一行内容
            line_height = g.put_ptext_line(config["CONTEXT"], tl_y)
            tl_y += (line_height + g.line_spacing)
        g.aug()

        if config["BASE_NAME"] == '':
            name = "%05d" % gn
        else:
            name = "%s_%05d" % (config["BASE_NAME"], gn)
        g.save(name, save_visualize=True, save_rectified_lines_separately=SAVE_RECTIFIED_LINES_SEPARATELY)


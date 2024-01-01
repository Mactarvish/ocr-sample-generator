import os
import random
from itertools import chain
import math

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

import imgaug.augmenters as iaa
import imgaug as ia

from augs import AddBackground
from .instance import Instance
from .category import Category

from utils.parse_random import parse_random


from matplotlib.colors import hsv_to_rgb
COLORS = [(hsv_to_rgb([i / 10., 1.0, 1.0]) * 255).astype(int).tolist() for i in range(3)]


class OCRSampleGenerator(object):
    def __init__(self, ttf_paths, canvas_pil, font_size, line_spacing, save_dir, 
                 char_spacing=0):
        assert isinstance(canvas_pil, Image.Image), type(canvas_pil)
        assert isinstance(ttf_paths, list) and len(ttf_paths) == 2, ttf_paths
        self.canvas_pil = canvas_pil
        self.line_spacing = line_spacing
        self.font_size = font_size
        self.save_dir = save_dir
        self.char_spacing = char_spacing
        self.instance_image_paths = {}
        
        self.image_font = {"hanzi_font": ImageFont.truetype(ttf_paths[0], size=self.font_size), 
                            "no_hanzi_font": ImageFont.truetype(ttf_paths[1], size=self.font_size)}
        self.labels = []
        self.hanzis = []
        self.init_hanzis()
        self.en_punctuations = ", . ; : ' ( ) ( ) ? ! _ _".split()
        self.cn_punctuations = "， 。 、 ？ ； ： ” “ ！ … （ ）".split()
        self.punctuations = self.en_punctuations + self.cn_punctuations
        self.letters = [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)]
        self.nums = [chr(ord('0') + i) for i in range(10)]
        self.hanzi_font_chars = set(self.hanzis + self.cn_punctuations)

        self.hanzi_w, self.hanzi_h = self.image_font["hanzi_font"].getsize("我")
        self.letter_w, self.letter_h = self.image_font["no_hanzi_font"].getsize("g")
    
    def init_hanzis(self):
        with open("assets/zidian_new_5883.txt", 'r', encoding="utf-8") as f:
            text = f.read()
            self.hanzis = list(text)
    
    def put_instance(self, instance, tl_x, tl_y, center_y=None):
        w, h = instance.width, instance.height
        category = instance.category

        if center_y is not None:
            assert tl_y == -1, tl_y
            tl_y = int(center_y - h / 2)
        # 上下左右添加一点裕量
        x1, y1, x2, y2, category = tl_x - 1, tl_y - 1, tl_x + w + 1, tl_y + h + 1, category
        label = (x1, y1, x2, y2, category.value)

        # 越界框处理：
        # 不相交，无效
        if x1 >= self.canvas_pil.size[0] or y1 >= self.canvas_pil.size[1] or x2 <= 0 or y2 <= 0:
            return (x1, y1, x1, y1, -1)
        # 上下越界
        if any(y <= 0 or y > self.canvas_pil.size[1] for y in [y1, y2]):
            # 如果越界面积大于阈值，那么视为负样本；否则无效
            if 0 < y2 / (y2 - y1) < 0.5 or 0 < (self.canvas_pil.size[1] - y1) / (y2 - y1) < 0.5:
                drawer = ImageDraw.Draw(self.canvas_pil)
                self.paint_text(drawer, instance, tl_x, tl_y)
                label = (x1, y1, x2, y1, x2, y2, x1, y2, category.value)
                return label
            else:
                return (x1, y1, x1, y1, -1)
        # 左右越界，直接无效
        elif any(x <= 0 or x > self.canvas_pil.size[0] for x in [x1, x2]):
            print(f"坐标越界：{label} 图像大小：{self.canvas_pil.size}") 
            # 无效框，返回起始位置
            return (x1, y1, x1, y1, -1)

        # 无越界，正常绘制
        drawer = ImageDraw.Draw(self.canvas_pil)
        self.paint_text(drawer, instance, tl_x, tl_y)
        # 两点框转为四点框
        label = (x1, y1, x2, y1, x2, y2, x1, y2, category.value, instance.text)
        self.labels.append(label)
        return label
    
    def paint_text(self, drawer, instance, tl_x, tl_y):
        font = self.image_font["hanzi_font"]
        # font = self.image_font["no_hanzi_font"]
        begin_index = 0
        end_index = len(instance.text)

        before = tl_x
        if instance.char_spacing == 0:
            for i in range(len(instance.text) + 1):
                if i == len(instance.text):
                    end_index = i
                else:
                    if instance.text[i] in self.hanzi_font_chars:
                        if font != self.image_font["hanzi_font"]:
                            end_index = i
                    else:
                        if font == self.image_font["hanzi_font"]:
                            end_index = i
                # end_index被更新了，说明切换了字体，此时把上次的内容绘制出来
                if end_index == i:
                    text = instance.text[begin_index: end_index]
                    if len(text) != 0:
                        drawer.text(xy=(tl_x, tl_y), text=text, fill=(0, 0, 0), font=font)
                        w, h = font.getsize(text)
                        tl_x += w
                    # 切换字体
                    if font == self.image_font["hanzi_font"]:
                        font = self.image_font["no_hanzi_font"]
                    else:
                        font = self.image_font["hanzi_font"]
                    begin_index = i
        # 如果实例的字符间距不是0，那么只能一个字符一个字符地打印
        else:
            for k, c in enumerate(instance.text):
                if c in self.hanzi_font_chars:
                    font = self.image_font["hanzi_font"]
                else:
                    font = self.image_font["no_hanzi_font"]
                drawer.text(xy=(tl_x, tl_y), text=c, fill=(0, 0, 0), font=font)
                w, h = font.getsize(c)
                tl_x += w
                if k != len(instance.text) - 1:
                    tl_x += instance.char_spacing            

    def aug(self):
        image_np = np.array(self.canvas_pil)
        h, w, c = image_np.shape
        seq = iaa.Sequential([
            # 这个fit_output非常有用，作用是通过补边让原图内容不旋出图去
            iaa.Affine(translate_px={"x": (1, 5)}, rotate=(-6, 6), cval=255, fit_output=True),
            iaa.Sometimes(0.75,
                AddBackground("assets/backgrounds", (0.1, 0.3)),
            ),
            iaa.Sometimes(0.1,
                iaa.AdditiveGaussianNoise(scale=0.5*2),
            ),
            iaa.Sometimes(0.1,
                iaa.SaltAndPepper(),
            ),
            iaa.Sometimes(0.1,
            iaa.JpegCompression(compression=(0, 15))
            )
        ])
        polygons = [ia.Polygon([(l[0], l[1]), (l[2], l[3]), (l[4], l[5]), (l[6], l[7])], label=l[8]) for l in self.labels]
        images_aug, polygons_aug = seq(image=image_np, polygons=polygons)

        self.labels = [list(map(round, p.coords.reshape(-1).tolist())) + [p.label] + [l[-1]] for p, l in zip(polygons_aug, self.labels)]
        self.canvas_pil = Image.fromarray(images_aug)

    def save(self, save_name, save_visualize=True, save_rectified_lines_separately=False):
        def parse_p4c_to_txt_line(label):
            CLASSES = ['ptext']
            # 9,6,797,5,797,36,9,36,###,ptext
            label = label[:-2] + [CLASSES[0], label[-1]]
            label = list(map(str, label))
            label.insert(8, "###")
            return ','.join(label)

        if len(self.labels) == 0:
            print("无有效实例，不保存")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        global_canvas_save_dir = os.path.join(self.save_dir, "global_canvas", "images")
        global_label_save_dir = os.path.join(self.save_dir, "global_canvas", "labels")
        os.makedirs(global_canvas_save_dir, exist_ok=True)
        os.makedirs(global_label_save_dir, exist_ok=True)


        global_canvas_save_path = os.path.join(global_canvas_save_dir, save_name + ".jpg")
        global_label_save_path = os.path.join(global_label_save_dir, save_name + ".txt")

        self.canvas_pil.save(global_canvas_save_path)
        txt_lines = [parse_p4c_to_txt_line(label) for label in self.labels]
        with open(global_label_save_path, 'w') as f:
            for line in txt_lines:
                f.write(line + '\n')
        
        if save_visualize:
            self.visualize(save_name)

        # 逐行保存透视变换后的图像（将旋转后的每行拧回正行）
        if save_rectified_lines_separately:
            rectified_lines_save_dir = os.path.join(self.save_dir, "rectified_lines")
            for i, label in enumerate(self.labels):
                single_line_save_path = os.path.join(rectified_lines_save_dir, "images", save_name + f"_{i}.jpg")
                single_label_save_path = os.path.join(rectified_lines_save_dir, "labels", save_name + f"_{i}.txt")
                os.makedirs(os.path.dirname(single_line_save_path), exist_ok=True)
                os.makedirs(os.path.dirname(single_label_save_path), exist_ok=True)
                x1, y1, x2, y2, x3, y3, x4, y4, c, text = label
                canvas_np = np.array(self.canvas_pil).copy()
                # 使用变换将四点框转为矩形框
                src = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]).astype(np.float32)
                w = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                h = round(math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
                dst = np.array([(0, 0), (w, 0), (w, h), (0, h)]).astype(np.float32)
                M = cv2.getPerspectiveTransform(src, dst)
                rectified_line = cv2.warpPerspective(canvas_np, M, (w, h))
                rectified_line = Image.fromarray(rectified_line)
                rectified_line.save(single_line_save_path)
                label = [0, 0, w, 0, w, h, 0, h, c, text]
                text_line = parse_p4c_to_txt_line(label)
                with open(single_label_save_path, 'w') as f:
                    f.write(text_line + '\n')

    def visualize(self, save_name):
        os.makedirs(self.save_dir, exist_ok=True)
        visualization_save_dir = os.path.join(self.save_dir, "global_canvas", "visualization")
        os.makedirs(visualization_save_dir, exist_ok=True)

        canvas_np = np.array(self.canvas_pil).copy()
        for label in self.labels:
            x1, y1, x2, y2, x3, y3, x4, y4, c, text = label
            cv2.polylines(canvas_np, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], True, COLORS[c + 1])
        save_path = os.path.join(visualization_save_dir, save_name + ".jpg")
        cv2.imwrite(save_path, canvas_np)
        print(save_path)

    
    def sample_chars(self, chars, num):
        if num == 0:
            return []
        indexes = np.random.randint(0, len(chars), num)
        selected = [chars[i] for i in indexes]
        return selected


    def generate_ptext_instance(self, length, context_config, add_end_punctuations=False):
        selected_chars = []
        
        # 计算实例中不同类型字符的数量
        hanzi_proportion = parse_random(context_config["HANZI_PROPORTION"])
        letter_proportion = parse_random(context_config["LETTER_PROPORTION"])
        number_proportion = parse_random(context_config["NUMBER_PROPORTION"])
        punctuation_proportion = parse_random(context_config["PUNCTUATION_PROPORTION"])
        space_proportion = parse_random(context_config["SPACE_PROPORTION"])
        # 不同字符类型比例中，至多有一个是-1，-1的这个类型的实际比例等于 1-其它类型比例之和
        # 按照 汉字 字母 数字 标点符号 空格 的顺序依次累加比例，如果加到某一项时总比例超过了1，那么后续类型比例直接置0
        tps = [hanzi_proportion, letter_proportion, number_proportion, punctuation_proportion, space_proportion]
        m1_index = -1
        p_sum = 0
        for i in range(len(tps)):
            if tps[i] == -1:
                if m1_index != -1:
                    raise ValueError(f"不同字符类型比例中，至多有一个是-1 {tps}")
                m1_index = i
            else:
                if p_sum + tps[i] >= 1:
                    tps[i] = 1 - p_sum
                p_sum += tps[i]
        # 出现了-1，计算实际比例
        if m1_index != -1:
            tps[m1_index] = 1. - p_sum
        [hanzi_proportion, letter_proportion, number_proportion, punctuation_proportion, space_proportion] = tps
        [hanzi_num, letter_num, number_num, punctuation_num, space_num] = [round(e * length) for e in tps]

        # 汉字
        selected_chars.append(self.sample_chars(self.hanzis, hanzi_num))
        # 英文字母
        selected_chars.append(self.sample_chars(self.letters, letter_num))
        # 数字
        selected_chars.append(self.sample_chars(self.nums, number_num))
        # 标点符号
        selected_chars.append(self.sample_chars(self.punctuations, punctuation_num))
        # 空格
        selected_chars.append(self.sample_chars([' '], space_num))

        # 所有的字符撮到一块，然后打乱顺序
        selected_chars = list(chain(*selected_chars))
        random.shuffle(selected_chars)

        # 移除最前或最后的空格
        i = 0
        for i in range(len(selected_chars)):
            if selected_chars[i] == ' ':
                i += 1
            else:
                break
        j = len(selected_chars) - 1
        for j in range(len(selected_chars) - 1, -1, -1):
            if selected_chars[j] == ' ':
                j -= 1
            else:
                break    
        selected_chars = selected_chars[i: j + 1]

        end = ['']
        if add_end_punctuations:
            end = random.sample(self.punctuations, 1)
        text = ''.join(selected_chars + end)
        print(text)
        return Instance(self, text, Category.PTEXT, char_spacing=self.char_spacing)

    def put_ptext_line(self, context_config, tl_y, tl_x=1):
        # 一行中的最大实例数量
        max_instance_num_per_line = 1
        # 计算在给定的画布尺寸、字符大小和单行实例个数条件下，每个实例的最大长度
        max_ptext_length = self.canvas_pil.size[0] / max_instance_num_per_line / (self.letter_w + self.char_spacing)
        # 每个实例至少一个字符
        ptext_instances = [self.generate_ptext_instance(np.random.randint(1, max_ptext_length), context_config) for _ in range(max_instance_num_per_line)]

        # 绘制实例后剩余的空间宽度
        rest_width = self.canvas_pil.size[0] - sum(instance.width for instance in ptext_instances)
        # 如果剩余空间不足，那么强行指定一个正的起始位置
        if rest_width <= 1:
            tl_x = np.random.randint(1, 10)
        else:
            tl_x = np.random.randint(1, rest_width) + tl_x

        max_height = max(i.height for i in ptext_instances)

        # 逐实例绘制
        for instance in ptext_instances:
            label = self.put_instance(instance, tl_x,tl_y)
            # 同行内的不同实例的间隔
            tl_x = label[2] + np.random.randint(1, 20)
        
        return max_height

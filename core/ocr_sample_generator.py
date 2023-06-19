import os
import random
from itertools import chain

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

import imgaug.augmenters as iaa
import imgaug as ia

from augs import AddBackground
from .instance import Instance
from .category import Category


class OCRSampleGenerator(object):
    def __init__(self, ttf_paths, canvas_pil, font_size, line_spacing, save_dir, char_spacing=0):
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

    def save(self, save_name, save_visualize=True):
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
        image_save_dir = os.path.join(self.save_dir, "images")
        label_save_dir = os.path.join(self.save_dir, "labels")
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)
        image_save_path = os.path.join(image_save_dir, save_name + ".jpg")
        label_save_path = os.path.join(label_save_dir, save_name + ".txt")

        self.canvas_pil.save(image_save_path)
        txt_lines = [parse_p4c_to_txt_line(label) for label in self.labels]
        with open(label_save_path, 'w') as f:
            for line in txt_lines:
                f.write(line + '\n')
        
        if save_visualize:
            self.visualize(save_name)

    def visualize(self, save_name):
        os.makedirs(self.save_dir, exist_ok=True)
        visualization_save_dir = os.path.join(self.save_dir, "visualization")
        os.makedirs(visualization_save_dir, exist_ok=True)
        def _to_color(indx, base):
            """ return (b, r, g) tuple"""
            base2 = base * base
            b = 2 - indx / base2
            r = 2 - (indx % base2) / base
            g = 2 - (indx % base2) % base
            return b * 127, r * 127, g * 127
        base = int(np.ceil(pow(13, 1. / 3)))
        COLORS = [_to_color(x, base) for x in range(21)]

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


    def generate_ptext_instance(self, length, main_lang="chinese", add_end=False):
        assert main_lang in ["chinese", "english", "mix"], main_lang
        selected_chars = []
        if main_lang == "chinese":
            space_proportion = 0
            space_num = np.random.randint(0, 3)
            selected_chars.append(self.sample_chars([' '], space_num))

            # 标点符号个数不超过文本总长度的1/10
            punctuation_proportion = 1 / 10
            punctuations_num = int(np.random.power(3) * length * punctuation_proportion)
            # punctuations_num = np.random.randint(0, length * punctuation_proportion + 1)
            selected_chars.append(self.sample_chars(self.punctuations, punctuations_num))
            
            number_proportion = 1 / 10
            number_num = int(np.random.power(3) * length * number_proportion)
            # number_num = np.random.randint(0, length * number_proportion + 1) * 2
            selected_chars.append(self.sample_chars(self.nums, number_num))
            
            # 英文字母个数不超过文本总长度的xx
            letter_proportion = 1 / 10
            letter_num = int(np.random.power(3) * length * letter_proportion)
            # letter_num = np.random.randint(0, length * letter_proportion + 1) * 2
            selected_chars.append(self.sample_chars(self.letters, letter_num))

            hanzi_num = length - punctuations_num - number_num // 2 - letter_num // 2
            selected_chars.append(self.sample_chars(self.hanzis, hanzi_num))
        elif main_lang == "english":
            space_proportion = 1 / 5
            space_num = int(np.random.power(3) * length * space_proportion)
            selected_chars.append(self.sample_chars([' ', ' ', ' ', ', ', '. '], space_num))

            # 标点符号个数不超过文本总长度的xx
            punctuation_proportion = 1 / 20
            punctuations_num = int(np.random.power(3) * length * punctuation_proportion)
            selected_chars.append(self.sample_chars(self.punctuations, punctuations_num))

            number_proportion = 1 / 5
            number_num = int(np.random.power(3) * length * number_proportion)

            letter_num = length - punctuations_num - number_num // 2 - space_num // 2
            selected_chars.append(self.sample_chars(self.letters, letter_num))
        elif main_lang == "mix":
            space_proportion = 0
            space_num = np.random.randint(0, 3)
            selected_chars.append(self.sample_chars([' '], space_num))

            # 标点符号个数不超过文本总长度的1/10
            punctuation_proportion = 1 / 10
            punctuations_num = int(np.random.power(3) * length * punctuation_proportion)
            # punctuations_num = np.random.randint(0, length * punctuation_proportion + 1)
            selected_chars.append(self.sample_chars(self.punctuations, punctuations_num))
            
            number_proportion = 1 / 2
            number_num = int(np.random.power(3) * length * number_proportion)
            # number_num = np.random.randint(0, length * number_proportion + 1) * 2
            selected_chars.append(self.sample_chars(self.nums, number_num))
            
            # 英文字母个数不超过文本总长度的xx
            letter_proportion = 0.7
            letter_num = int(np.random.power(3) * length * letter_proportion)
            # letter_num = np.random.randint(0, length * letter_proportion + 1) * 2
            selected_chars.append(self.sample_chars(self.letters, letter_num))

            hanzi_num = length - punctuations_num - number_num // 2 - letter_num // 2
            selected_chars.append(self.sample_chars(self.hanzis, hanzi_num))
        else:
            raise NotImplementedError(main_lang)

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
        if add_end:
            end = random.sample(self.punctuations, 1)
        text = ''.join(selected_chars + end)
        print(text)
        return Instance(self, text, Category.PTEXT, char_spacing=self.char_spacing)

    def put_ptext_line(self, main_lang, tl_y, tl_x=1):
        rest_width = self.canvas_pil.size[0]
        text_num = 1
        max_ptext_length = rest_width / text_num / (self.letter_w + self.char_spacing)
        ptext_instances = [self.generate_ptext_instance(max(10, np.random.randint(0, max(4, max_ptext_length))), main_lang) for _ in range(text_num)]

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

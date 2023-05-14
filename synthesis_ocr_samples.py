import os
import sys
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tqdm import tqdm
from augs import AddBackground, AddTransparent, AddWatermark
import glob
from itertools import chain
import random
import imgaug.augmenters as iaa
import imgaug as ia
from enum import Enum
from tqdm import trange
import cv2
import numpy as np
import argparse


print("注意，latex图像必须是png格式，jpg不予考虑")

class Category(Enum):
    PTEXT         = 0
    HTEXT         = 1
    PFORMULA      = 2
    HFORMULA      = 3
    LD            = 4
    NLD           = 5
    OC            = 6
    MIXFORMULA    = 7
    GRAPH         = 8
    EXCEL         = 9
    P_FORMULA_SET = 10
    P_UP_DOWN     = 11
    H_FORMULA_SET = 12
    H_UP_DOWN     = 13
    SUBFIELD      = 14


PTEXT         = Category.PTEXT
HTEXT         = Category.HTEXT
PFORMULA      = Category.PFORMULA
HFORMULA      = Category.HFORMULA
LD            = Category.LD
NLD           = Category.NLD
OC            = Category.OC
MIXFORMULA    = Category.MIXFORMULA
GRAPH         = Category.GRAPH
EXCEL         = Category.EXCEL
P_FORMULA_SET = Category.P_FORMULA_SET
P_UP_DOWN     = Category.P_UP_DOWN
H_FORMULA_SET = Category.H_FORMULA_SET
H_UP_DOWN     = Category.H_UP_DOWN
SUBFIELD      = Category.SUBFIELD


ALL_CATEGORIES = [PTEXT, HTEXT, PFORMULA, HFORMULA, LD, NLD, OC, MIXFORMULA,
                    GRAPH, EXCEL, P_FORMULA_SET, P_UP_DOWN, H_FORMULA_SET,H_UP_DOWN, SUBFIELD]


song_ttf_path = "/data1/mchk/mmdetection_for_common_ocr/pgs/SimSun.ttf"
times_new_roman_ttf_path = "/data1/mchk/mmdetection_for_common_ocr/pgs/TimesNewRoman.ttf"

MAX_LENGTH = 1333
MIN_PFORMULA_HEIGHT = 15
MIN_HFORMULA_HEIGHT = 15

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch * 3, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

SHENG = \
'''
b p m f d t n l g k h j q x zh ch sh r z c s y w
'''.split()

YUN = \
'''
ǖn ǘn ǚn ǜn
āng áng ǎng àng
ou
iē ié iě iè
iū iú iǔ iù
ēi éi ěi èi
āi ái ǎi ài
oū oú oǔ où
īng íng ǐng ìng
ong
ūe úe ǔe ùe
ue
ēr ér ěr èr
āo áo ǎo ào
ēn én ěn èn
īn ín ǐn ìn
ū ú ǔ ù
uī uí uǐ uì
ūn ún ǔn ùn
un
en
ing
ui
a
ē é ě è
ǖ ǘ ǚ ǜ
ai
ie
ān án ǎn àn
In
ēng éng ěng èng
er
iu
in
ang
ā á ǎ à
ao
ü
U
eng
ō ó ǒ ò
an
ei
īí ǐ ì
e
ōng óng ǒng òng
i
'''.split()

class Instance(object):
    def __init__(self, parent, text, category, is_path=False, char_spacing=0) -> None:
        super().__init__()
        assert isinstance(parent, DetectionCaseGenerator), type(parent)
        assert isinstance(category, Category), type(category)
        self.parent = parent
        self.width = 0
        self.height = 0
        self.char_spacing = char_spacing
        self.init_instance_attribute(text, category, is_path)

    def init_instance_attribute(self, text, category, is_path):
        self.width = 0
        self.text = text
        self.category = category
        self.is_path = is_path
        self.image_np = None
        if is_path:
            assert os.path.isfile(self.text) and self.text.endswith(".png"), self.text
            print(self.text)
            self.image_np = cv2.imread(text)
            h, w, c = self.image_np.shape
            if category == PFORMULA:
                self.width = round(self.parent.hanzi_h / h * w)
                self.height = round(self.parent.hanzi_h)
            else:
                # scale = np.random.uniform(self.parent.hanzi_h / h, 1.)
                scale = self.parent.hanzi_h / h
                scale = scale * np.random.uniform(1., 2.)
                self.height = int(self.image_np.shape[0] * scale)
                self.width = int(self.image_np.shape[1] * scale)
                print("---", self.parent.hanzi_h)
                print("---", self.height)
            self.image_np = cv2.resize(self.image_np, (self.width, self.height))
        else:
            self.height = self.parent.letter_h
            # 一段文本中可能有多种不同的字体，遍历文本，根据内容确定字体，进而确定渲染尺寸
            if self.char_spacing == 0:
                begin = 0
                font = self.parent.image_font["hanzi_font"]
                for i in range(len(text)):
                    if text[i] in self.parent.hanzi_font_chars:
                        if font != self.parent.image_font["hanzi_font"]:
                            self.width += font.getsize(text[begin: i])[0]
                            begin = i
                            font = self.parent.image_font["hanzi_font"]
                    else:
                        if font != self.parent.image_font["no_hanzi_font"]:
                            self.width += font.getsize(text[begin: i])[0]
                            begin = i
                            font = self.parent.image_font["no_hanzi_font"]
                self.width += font.getsize(text[begin:])[0]
            else:
                # ！！！这里务必要注意，font.getsize("abc")[0] != sum(font.getsize(k) for k in "abc")
                # ！！！因此非0字符间距的字符串计算实例宽度时一定要一个字符一个字符地计算
                for c in text:
                    if c in self.parent.hanzi_font_chars:
                        self.width += self.parent.image_font["hanzi_font"].getsize(c)[0]
                    else:
                        self.width += self.parent.image_font["no_hanzi_font"].getsize(c)[0]
                # 加上字符间距
                self.width += (len(text) - 1) * self.char_spacing
                    

class DetectionCaseGenerator(object):
    def __init__(self, ttf_paths, canvas_pil, font_size, line_spacing, save_dir, char_spacing=0, instance_image_dir_dict=None) -> None:
        assert isinstance(canvas_pil, Image.Image), type(canvas_pil)
        assert isinstance(ttf_paths, list) and len(ttf_paths) == 2, ttf_paths
        self.canvas_pil = canvas_pil
        self.line_spacing = line_spacing
        self.font_size = font_size
        self.save_dir = save_dir
        self.char_spacing = char_spacing
        assert isinstance(instance_image_dir_dict, dict), instance_image_dir_dict
        self.instance_image_dir_dict = instance_image_dir_dict
        self.instance_image_paths = {}
        for key in self.instance_image_dir_dict:
            self.instance_image_paths[key] = glob.glob(os.path.join(self.instance_image_dir_dict[key], "*.png"))
        
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
        self.special_symbols = "▲ ■ △ ●".split()
        self.hanzi_font_chars = set(self.hanzis + self.cn_punctuations + self.special_symbols)

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
                if instance.is_path:
                    canvas_np = np.array(self.canvas_pil)
                    h = min(self.canvas_pil.size[1] - y1, y2 + 1, h)
                    w = min(self.canvas_pil.size[0] - x1, x2 + 1, w)
                    y1 = max(0, min(y1, self.canvas_pil.size[1] - 1))
                    x1 = max(0, min(x1, self.canvas_pil.size[0] - 1))
                    canvas_np[y1: y1 + h, x1: x1 + w, :] = instance.image_np[:h, :w, :]
                    self.canvas_pil = Image.fromarray(canvas_np)
                else:
                    drawer = ImageDraw.Draw(self.canvas_pil)
                    # drawer.text(xy=(tl_x, tl_y), text=instance.text, fill=(0, 0, 0), font=self.image_font)
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
        if instance.is_path:
            canvas_np = np.array(self.canvas_pil)
            canvas_np[tl_y: tl_y + h, tl_x: tl_x + w, :] = instance.image_np
            self.canvas_pil = Image.fromarray(canvas_np)
        else:
            drawer = ImageDraw.Draw(self.canvas_pil)
            self.paint_text(drawer, instance, tl_x, tl_y)
        # 两点框转为四点框
        label = (x1, y1, x2, y1, x2, y2, x1, y2, category.value)
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
            iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": (MAX_LENGTH if max(h, w) > MAX_LENGTH else max(h, w))}),
            iaa.Sometimes(0.75,
                AddBackground("/data1/mchk/dataset/common_ocr/base/random_background/neirongyun_cropped/", (0.5, 1)),
            ),
            # 添加透字
            iaa.Sometimes(0.05,
                AddCharsBehind("/data1/mchk/dataset/common_ocr/base/neirongyun_no_math_1333/images/"),
            ),
            # 添加水印
            iaa.Sometimes(0.2,
                AddWatermark("assets/fonts/cn"),
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
        self.labels = [tuple(map(round, p.coords.reshape(-1).tolist() + [p.label])) for p in polygons_aug]
        self.canvas_pil = Image.fromarray(images_aug)

    def save(self, save_name, save_visualize=True):
        def parse_p4c_to_txt_line(label):
            CLASSES = ['ptext', 'htext', 'pformula', 'hformula', 'ld', 'nld', 'oc', 'mixformula',
                    'graph', 'excel', 'p_formula_set', 'p_up_down', 'h_formula_set', 'h_up_down', "subfield"]
            # 9,6,797,5,797,36,9,36,###,subfield
            label = label[:-1] + (CLASSES[label[-1]],)
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
            x1, y1, x2, y2, x3, y3, x4, y4, c = label
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


    def generate_ptext(self, length, main_lang="chinese", add_end=False):
        assert main_lang in ["chinese", "english", "mix", "pinyin"], main_lang
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
        elif main_lang == "pinyin":
            selected_chars.append(self.generate_pinyin().split())
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
        return Instance(self, text, PTEXT, char_spacing=self.char_spacing)

    def generate_splited_indexes(self, text_length, index_num):
        if text_length == 0 or index_num == 0:
            return []
        # 循环采样，直到运算符号两两不挨着
        assert index_num <= text_length // 2, "%d %d" % (index_num, text_length)
        redo = True
        while redo:
            redo = False
            insert_positions = sorted(np.random.randint(1, text_length - 1, index_num))
            for i in range(len(insert_positions) - 1):
                if insert_positions[i] + 1 == insert_positions[i + 1]:
                    redo = True
                    break
        return insert_positions

    def generate_hformula(self, max_length=20):
        # 如果字高足够高，那么一定几率直接返回一个latex图片；否则公式图片会被缩放得太糊
        if self.hanzi_h > MIN_HFORMULA_HEIGHT and "hformula" in self.instance_image_dir_dict:
            instance_image_path = random.sample(self.instance_image_paths["hformula"], 1)[0]
            return Instance(self, instance_image_path, HFORMULA, is_path=True)
        return None

    def generate_htext(self, max_length=20):
        # 如果字高足够高，那么一定几率直接返回一个latex图片；否则公式图片会被缩放得太糊
        if self.hanzi_h > MIN_HFORMULA_HEIGHT and "htext" in self.instance_image_dir_dict:
            instance_image_path = random.sample(self.instance_image_paths["htext"], 1)[0]
            return Instance(self, instance_image_path, HTEXT, is_path=True)
        return None


    def generate_pformula(self, max_length=20):
        # 如果字高足够高，那么一定几率直接返回一个latex图片；否则公式图片会被缩放得太糊
        if self.hanzi_h > MIN_PFORMULA_HEIGHT and "pformula" in self.instance_image_dir_dict and np.random.randint(0, 2) == 0:
            instance_image_path = random.sample(self.instance_image_paths["pformula"], 1)[0]
            print(instance_image_path)
            return Instance(self, instance_image_path, PFORMULA, is_path=True)

        op_symbols = "+-=<>≥≤×÷÷" # ∥⊥⇒
        nl = self.nums
        length = np.random.randint(3, max_length)
        indexes = np.random.randint(0, len(nl), length)
        pformula = [nl[i] for i in indexes]

        op_nums = min(length // 2, np.random.randint(1, 8))
        op_indexes = np.random.randint(0, len(op_symbols), op_nums)
        selected_symbols = [op_symbols[i] for i in op_indexes]

        insert_positions = self.generate_splited_indexes(len(pformula), op_nums)
        for i,p in zip(insert_positions, selected_symbols):
            pformula[i] = p

        # 如果公式足够长，那么一定几率插入一个逗号
        if length > 10 and np.random.randint(0, 5) == 0:
            pformula[np.random.randint(length // 4, 3 * length // 4)] = random.sample([",", "，"], 1)[0]
        # 删除头部的逗号和句号
        i = 0
        for i in range(len(pformula)):
            if pformula[i] not in [',', '.', '，', '。']:
                break
        j = len(pformula) - 1
        for j in range(len(pformula) - 1, -1, -1):
            if pformula[j] not in [',', '.', '，', '。']:
                break
        pformula = pformula[i: j + 1]
                

        pformula = ''.join(pformula)

        return Instance(self, pformula, PFORMULA, char_spacing=self.char_spacing)


    def generate_pupdown(self):
        return

    # 参数里这个tl_x是一行的起始位置可能的最左端
    def put_formula_ptext_graph_line(self, tl_y, tl_x=1, main_lang="chinese", allow_formula=True):
        pformulas = []
        formula_num = np.random.randint(0, 4)
        # 纯英文，不含公式
        if main_lang == "english":
            formula_num = 0
        # 削减公式出现的几率
        if np.random.randint(0, 3) != 0:
            formula_num = 0
        
        if not allow_formula:
            formula_num = 0
            
        for i in range(formula_num):
            k = np.random.randint(0, 3)
            if k == 0:
                pformulas.append(self.generate_pformula())
            elif k == 1:
                pformulas.append(self.generate_hformula())
            else:
                pformulas.append(self.generate_htext())
        # todo 移除None的情况
        pformulas = [p for p in pformulas if p is not None]
        formula_num = len(pformulas)
        
        pformula_widths = [instance.width for instance in pformulas]

        rest_width = self.canvas_pil.size[0] - sum(pformula_widths)
        if formula_num == 0:
            text_num = 1
        else:
            text_num = formula_num + np.random.randint(0, 2)
        # min_ptext_length = 3 if main_lang == "chinese" else 10
        min_ptext_length = 10 if main_lang == "chinese" else 10
        max_ptext_length = rest_width / text_num / (self.hanzi_w + self.char_spacing)
        ptexts = [self.generate_ptext(max(min_ptext_length, np.random.randint(0, max(4, max_ptext_length))), main_lang=main_lang) for _ in range(text_num)]

        # ptext和pformula填上去之后剩余的空间宽度
        rest_width = self.canvas_pil.size[0] - sum(instance.width for instance in pformulas + ptexts)
        # 如果剩余空间不足，那么强行指定一个正的起始位置
        if rest_width <= 1:
            tl_x = np.random.randint(1, 10)
        else:
            tl_x = np.random.randint(1, rest_width) + tl_x

        # ptext和pformula交替出现
        instances = []
        i = 0
        j = 0
        while i < len(ptexts) and j < len(pformulas):
            instances.append(ptexts[i])
            instances.append(pformulas[j])
            i += 1
            j += 1
        if i < len(ptexts):
            instances.extend(ptexts[i:])
        elif j < len(pformulas):
            instances.extend(pformulas[j:])
        
        # 实例中随机插入一个特殊符号
        # if np.random.randint(0, 10) == 0:
        #     graph = Instance(self, random.sample(self.special_symbols, 1)[0], GRAPH)
        #     instances.insert(np.random.randint(0, len(instances) + 1), graph)

        max_height = max(i.height for i in instances)

        last_valid_label = -1
        for i, instance in enumerate(instances):
            # 由于每次放置实例可能出现越界导致放置失败的情况，同时两个同类的实例不能相邻，因此需要额外判定当前实例和上个有效实例类别是否相同
            if instance.category.value == last_valid_label:
                continue
            if i > 0 and instance.category == PTEXT:
                # 较大的概率给pformula后边（也就是ptext前边）接一个标点符号
                if np.random.randint(0, 2) == 0:
                    text = random.sample(self.punctuations + [',', '.'], 1)[0] + instance.text
                    instance.init_instance_attribute(text, PTEXT, is_path=False)
            # label = self.put_instance(instance, tl_x,tl_y)
            label = self.put_instance(instance, tl_x,-1, center_y=tl_y + max_height / 2 - self.hanzi_h / 2)
            if label[-1] != -1:
                last_valid_label = label[-1]
                
            tl_x = label[2] + np.random.randint(1, 20)
        
        return max_height

    def generate_pinyin(self):
        s = random.sample(SHENG, 1)[0]
        y = random.sample(YUN, 1)[0]
        pinyin = s + y
        return pinyin

    def put_pinyin(self, tl_y, tl_x=1):
        formula_num = np.random.randint(0, 4)
        # 纯英文，不含公式
        formula_num = 0
        
        # htexts = []
        # for _ in range(np.random.randint(5)):
        #     htexts.append(self.generate_htext())
        # # todo 移除None的情况
        # htexts = [p for p in htexts if p is not None]
        # formula_num = len(htexts)
        # htext_widths = [instance.width for instance in htexts]

        # rest_width = self.canvas_pil.size[0] - sum(htext_widths)
        if formula_num == 0:
            text_num = 1
        else:
            text_num = formula_num + np.random.randint(0, 2)
        ptexts = [self.generate_ptext(np.random.randint(2, 5), main_lang="pinyin") for _ in range(text_num)]

        # ptext和htext填上去之后剩余的空间宽度
        rest_width = self.canvas_pil.size[0] - sum(instance.width for instance in ptexts)
        # 如果剩余空间不足，那么强行指定一个正的起始位置
        if rest_width <= 1:
            tl_x = np.random.randint(1, 10)
        else:
            tl_x = np.random.randint(1, rest_width) + tl_x

        # ptext和htext交替出现
        instances = []
        instances = ptexts
        # i = 0
        # j = 0
        # while i < len(ptexts) and j < len(htexts):
        #     instances.append(ptexts[i])
        #     instances.append(htexts[j])
        #     i += 1
        #     j += 1
        # if i < len(ptexts):
        #     instances.extend(ptexts[i:])
        # elif j < len(htexts):
        #     instances.extend(htexts[j:])

        max_height = max(i.height for i in instances)

        last_valid_label = -1
        for i, instance in enumerate(instances):
            # 由于每次放置实例可能出现越界导致放置失败的情况，同时两个同类的实例不能相邻，因此需要额外判定当前实例和上个有效实例类别是否相同
            if instance.category.value == last_valid_label:
                continue
            if i > 0 and instance.category == PTEXT:
                # 较大的概率给htext后边（也就是ptext前边）接一个标点符号
                if np.random.randint(0, 2) == 0:
                    text = random.sample(self.punctuations + [',', '.'], 1)[0] + instance.text
                    instance.init_instance_attribute(text, PTEXT, is_path=False)
            # label = self.put_instance(instance, tl_x,tl_y)
            label = self.put_instance(instance, tl_x,-1, center_y=tl_y + max_height / 2 - self.hanzi_h / 2)
            if label[-1] != -1:
                last_valid_label = label[-1]
                
            tl_x = label[2] + np.random.randint(1, 20)
        
        return max_height

    def put_htext_graph_line(self, tl_y, tl_x=1, main_lang="chinese"):
        htexts = []
        formula_num = np.random.randint(0, 4)
        # 纯英文，不含公式
        if main_lang == "english":
            formula_num = 0
        # 削减公式出现的几率
        if np.random.randint(0, 3) != 0:
            formula_num = 0
        
        for _ in range(np.random.randint(5)):
            htexts.append(self.generate_htext())
        # todo 移除None的情况
        htexts = [p for p in htexts if p is not None]
        formula_num = len(htexts)
        
        htext_widths = [instance.width for instance in htexts]

        rest_width = self.canvas_pil.size[0] - sum(htext_widths)
        if formula_num == 0:
            text_num = 1
        else:
            text_num = formula_num + np.random.randint(0, 2)
        # min_ptext_length = 3 if main_lang == "chinese" else 10
        min_ptext_length = 10 if main_lang == "chinese" else 10
        max_ptext_length = rest_width / text_num / (self.hanzi_w + self.char_spacing)
        ptexts = [self.generate_ptext(max(min_ptext_length, np.random.randint(0, max(4, max_ptext_length))), main_lang=main_lang) for _ in range(text_num)]

        # ptext和htext填上去之后剩余的空间宽度
        rest_width = self.canvas_pil.size[0] - sum(instance.width for instance in htexts + ptexts)
        # 如果剩余空间不足，那么强行指定一个正的起始位置
        if rest_width <= 1:
            tl_x = np.random.randint(1, 10)
        else:
            tl_x = np.random.randint(1, rest_width) + tl_x

        # ptext和htext交替出现
        instances = []
        i = 0
        j = 0
        while i < len(ptexts) and j < len(htexts):
            instances.append(ptexts[i])
            instances.append(htexts[j])
            i += 1
            j += 1
        if i < len(ptexts):
            instances.extend(ptexts[i:])
        elif j < len(htexts):
            instances.extend(htexts[j:])
        
        # 实例中随机插入一个特殊符号
        # if np.random.randint(0, 10) == 0:
        #     graph = Instance(self, random.sample(self.special_symbols, 1)[0], GRAPH)
        #     instances.insert(np.random.randint(0, len(instances) + 1), graph)

        max_height = max(i.height for i in instances)

        last_valid_label = -1
        for i, instance in enumerate(instances):
            # 由于每次放置实例可能出现越界导致放置失败的情况，同时两个同类的实例不能相邻，因此需要额外判定当前实例和上个有效实例类别是否相同
            if instance.category.value == last_valid_label:
                continue
            if i > 0 and instance.category == PTEXT:
                # 较大的概率给htext后边（也就是ptext前边）接一个标点符号
                if np.random.randint(0, 2) == 0:
                    text = random.sample(self.punctuations + [',', '.'], 1)[0] + instance.text
                    instance.init_instance_attribute(text, PTEXT, is_path=False)
            # label = self.put_instance(instance, tl_x,tl_y)
            label = self.put_instance(instance, tl_x,-1, center_y=tl_y + max_height / 2 - self.hanzi_h / 2)
            if label[-1] != -1:
                last_valid_label = label[-1]
                
            tl_x = label[2] + np.random.randint(1, 20)
        
        return max_height



    # 参数里这个tl_x是一行的起始位置可能的最左端
    def _put_english_ptext_line(self, tl_y, tl_x=1):
        pformulas = []
        formula_num = 0
        for i in range(formula_num):
            pformulas.append(self.generate_pformula())
        
        pformula_widths = [instance.width for instance in pformulas]

        rest_width = self.canvas_pil.size[0] - sum(pformula_widths)
        if formula_num == 0:
            text_num = 1
        else:
            text_num = formula_num + np.random.randint(0, 2)
        max_ptext_length = rest_width / text_num / (self.letter_w + self.char_spacing)
        ptexts = [self.generate_ptext(max(10, np.random.randint(0, max(4, max_ptext_length))), main_lang="english") for _ in range(text_num)]

        # ptext和pformula填上去之后剩余的空间宽度
        rest_width = self.canvas_pil.size[0] - sum(instance.width for instance in pformulas + ptexts)
        # 如果剩余空间不足，那么强行指定一个正的起始位置
        if rest_width <= 1:
            tl_x = np.random.randint(1, 10)
        else:
            tl_x = np.random.randint(1, rest_width) + tl_x

        # ptext和pformula交替出现
        instances = []
        i = 0
        j = 0
        while i < len(ptexts) and j < len(pformulas):
            instances.append(ptexts[i])
            instances.append(pformulas[j])
            i += 1
            j += 1
        if i < len(ptexts):
            instances.extend(ptexts[i:])
        elif j < len(pformulas):
            instances.extend(pformulas[j:])


        max_height = max(i.height for i in instances)

        for i, instance in enumerate(instances):
            label = self.put_instance(instance, tl_x,tl_y)
            # label = self.put_instance(instance, tl_x,-1, center_y=tl_y + max_height / 2 - self.hanzi_h / 2)
            tl_x = label[2] + np.random.randint(1, 20)
        
        return max_height

def formula2img(str_latex, out_file, img_size=(5,3), font_size=16):
    fig = plt.figure(figsize=img_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    font_prop = font_manager.FontProperties(fname=song_ttf_path)

    plt.text(0.5, 0.5, str_latex,font_properties=font_prop, fontsize=font_size, verticalalignment='center', horizontalalignment='center')
    plt.savefig(out_file)


def generate_latex(text):
    return f"${' '.join(text)}$"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir")
    parser.add_argument("generation_num", type=int)
    parser.add_argument("--base-name", type=str, default='')
    parser.add_argument("--only_big", action="store_true")
    parser.add_argument("--only_small", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    image_save_dir = os.path.join(args.save_dir, "images")
    label_save_dir = os.path.join(args.save_dir, "labels")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    for gn in trange(args.generation_num):
        W = np.random.randint(600, 1000)
        H = np.random.randint(50, 2000)
        FONT_SIZE = np.random.randint(15,30)
        # LINE_SPACING = int(10 / 25 * FONT_SIZE)
        LINE_SPACING = max(3, np.random.randint(int(1 / 5 * FONT_SIZE), int(15 / 25 * FONT_SIZE)))
        CHAR_SPACING = random.sample([0,0,0,0,0,1,2,5, 10], 1)[0]
        CHAR_SPACING = random.sample([0,0,0,0,0,], 1)[0]
        img = Image.new("RGB", (W, H), (255, 255, 255))

        # 生成大条大字图
        if args.only_big:
            W = np.random.randint(2500, 3000)
            H = np.random.randint(200, 500)
            FONT_SIZE = np.random.randint(80,100)
            LINE_SPACING = np.random.randint(10, 20)
            CHAR_SPACING = np.random.randint(40, 60)
            img = Image.new("RGB", (W, H), (255, 255, 255))
        elif args.only_small:
            W = np.random.randint(150, 250)
            H = np.random.randint(100, 150)
            FONT_SIZE = np.random.randint(5, 15)
            LINE_SPACING = np.random.randint(3, 10)
            CHAR_SPACING = 0
            img = Image.new("RGB", (W, H), (255, 255, 255))

        cn_font_paths = glob.glob(os.path.join("assets/fonts/cn", "*.ttf"))
        en_font_paths = glob.glob(os.path.join("assets/fonts/en", "*.ttf"))
        
        cn_font_path = "assets/fonts/cn/SimSun.ttf"
        en_font_path = "assets/fonts/en/TimesNewRoman.ttf"

        # if np.random.randint(0, 10) < 3:
            # cn_font_path = random.sample(cn_font_paths, 1)[0]
        if np.random.randint(0, 10) < 5:
            cn_font_path = "assets/fonts/cn/楷体_GB2312.ttf"
        

        g = DetectionCaseGenerator([cn_font_path, en_font_path], img, font_size=FONT_SIZE, line_spacing=LINE_SPACING, char_spacing=CHAR_SPACING,
                                    save_dir=args.save_dir, 
                                    instance_image_dir_dict={"pformula":"/data1/mchk/dataset/common_ocr/base/pformula/images_cropped_h_lt_60/",
                                                             "htext":"/data1/mchk/dataset/ai_mark/base/htext_patch/htext/images/",
                                                             "hformula":"/data1/mchk/dataset/common_ocr/base/handwritting_patch/hformula/images"}
                                    )

        tl_y_offset = np.random.randint(0, 40) - 20
        tl_y = tl_y_offset
        line_height = g.font_size
        for i in range(g.canvas_pil.height // (g.font_size + g.line_spacing) + 3):
            # tl_y = (g.font_size + g.line_spacing) * i + tl_y_offset
            tl_y += (line_height + g.line_spacing)

            line_height = g.put_pinyin(tl_y)
            # 一定概率啥也不写
            # if np.random.randint(0, 8) != 0:
                # line_height = g.put_htext_graph_line(tl_y, main_lang="chinese")
                # k = np.random.randint(0, 5)
                # if k < 3:
                    # line_height = g.put_formula_ptext_graph_line(tl_y, main_lang="chinese")
                # elif k == 3:
                    # line_height = g.put_formula_ptext_graph_line(tl_y, main_lang="mix")
                # else:
                    # line_height = g.put_formula_ptext_graph_line(tl_y, main_lang="english")
                # else:
                    # line_height = g.put_english_ptext_line(tl_y)


        g.aug()
        if args.base_name == '':
            name = "%d" % gn
        else:
            name = "%s_%d" % (args.base_name, gn)
        g.save(name, save_visualize=True)

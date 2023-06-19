from .category import Category


class Instance(object):
    def __init__(self, parent, text, category, char_spacing=0) -> None:
        super().__init__()
        from .ocr_sample_generator import OCRSampleGenerator
        assert isinstance(parent, OCRSampleGenerator), type(parent)
        assert isinstance(category, Category), type(category)
        self.parent = parent
        self.width = 0
        self.height = 0
        self.char_spacing = char_spacing
        self.init_instance_attribute(text, category)

    def init_instance_attribute(self, text, category):
        self.width = 0
        self.text = text
        self.category = category
        self.image_np = None

        # 一段文本中可能有多种不同的字体，遍历文本，根据内容确定字体，进而确定渲染尺寸
        if self.char_spacing == 0:
            begin = 0
            font = self.parent.image_font["hanzi_font"]
            for i in range(len(text)):
                # 字体发生切换，计算使用上一字体渲染的字符串的尺寸
                if text[i] in self.parent.hanzi_font_chars and font != self.parent.image_font["hanzi_font"]:
                    w, h = font.getsize(text[begin: i])
                    self.width += w
                    self.height = max(self.height, h)
                    begin = i
                    font = self.parent.image_font["hanzi_font"]
                elif text[i] not in self.parent.hanzi_font_chars and font != self.parent.image_font["no_hanzi_font"]:
                    w, h = font.getsize(text[begin: i])
                    self.width += w
                    self.height = max(self.height, h)
                    begin = i
                    font = self.parent.image_font["no_hanzi_font"]
            w, h = font.getsize(text[begin:])
            self.width += w
            self.height = max(self.height, h)
        else:
            # ！！！这里务必要注意，font.getsize("abc")[0] != sum(font.getsize(k) for k in "abc")
            # ！！！因此非0字符间距的字符串计算实例宽度时一定要一个字符一个字符地计算
            for c in text:
                if c in self.parent.hanzi_font_chars:
                    w, h = self.parent.image_font["hanzi_font"].getsize(c)
                else:
                    w, h = self.parent.image_font["no_hanzi_font"].getsize(c)
                self.width += w
                self.height = max(self.height, h)
            # 加上字符间距
            self.width += (len(text) - 1) * self.char_spacing
                    
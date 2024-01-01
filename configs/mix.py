BASE_NAME= "mix" # 合成图像的前缀名称

fs = (15, 30)
LAYOUT = dict( # 生成图像的布局配置
    IMAGE_WIDTH= (600, 1000), # 图像的宽度范围，()表示从连续区间中取值，长度超出图像宽度的文本行会被忽略
    IMAGE_HEIGHT= (200, 1000), # 图像的高度范围，文本行从上到下对高度进行填充，填满为止
    FONT_SIZE= fs,
    CHAR_SPACING = [0, 0, 0, 0, 0, 0, 1, 2, 5, 10], # 字符间距，[]表示从集合中随机抽取一个
    LINE_SPACING = (int(1 / 5 * fs[0]), int(15 / 25 * fs[0])), # 行间距
)

CONTEXT = dict( # 生成图像的文本内容比例配置
    HANZI_PROPORTION = (0.2, 0.5), # 汉字比例
    LETTER_PROPORTION = (0.2, 0.5), # 英文字母比例
    NUMBER_PROPORTION = (0, 0.1), # 数字比例
    PUNCTUATION_PROPORTION = (0, 0.1), # 标点符号比例
    SPACE_PROPORTION = (0, 0.1), # 空格比例
)
SAVE_RECTIFIED_LINES_SEPARATELY = True # 是否逐行保存，即根据生成的样本把每个文本行裁出来拉正保存（通常用于训练识别模型），使能后会在保存目录下新建一个rectified_lines文件夹保存每张图上的每个文本行

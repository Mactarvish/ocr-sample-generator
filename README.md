# ocr-sample-generator

OCR样本生成器，可自动生成用于训练OCR检测和识别模型的图片样本和标注。

- 精简方便的配置模式，快速自定义需要的样本规格
- 支持随机生成文本行，同时生成精确的四点框坐标和文本内容
- 支持自定义增广
- 支持生成样本可视化
- 支持不同字体
- 支持自定义文本布局和文本类型



<div align="center">
  <img src="docs/images/1689003771417.jpg"/>
  <img src="docs/images/1689003895980.jpg"/>
</div>


## 立即使用

先安装依赖

```pip install -r requirements.txt```

然后执行

```python main.py configs/xxx.py dst_dir num_samples```

其中，`xxx.py`表示配置文件，`dst_dir`表示生成样本的保存目录，样本图片文件保存在`dst_dir/images`，标注文件保存在`dst_dir/labels`，可视化图像保存在`dst_dir/visualization`

例如

```python main.py configs/english.py pgs 20```

将按照配置文件`english.py`的规格生成20个样本，保存在`pgs`文件夹中。

## 欢迎issue和pr
import os
import urllib.request


def download(url, dst_dir, filename="auto"):
    os.makedirs(dst_dir, exist_ok=True)
    if filename == "auto":
        filename = os.path.basename(url)
    dst_file_path = os.path.join(dst_dir, filename)
    # 下载文件
    urllib.request.urlretrieve(url, dst_file_path)
    print('下载完成')


if __name__ == '__main__':
    download('http://d.xiazaiziti.com/en_fonts/fonts/t/Times-New-Roman.ttf', "fonts/en")
    download('http://d.xiazaiziti.com/en_fonts/fonts/s/SimSun.ttf', "fonts/cn")
    download('http://d.xiazaiziti.com/fonts/qita/%E6%A5%B7%E4%BD%93_GB2312.ttf', "fonts/cn", "楷体_GB2312.ttf")
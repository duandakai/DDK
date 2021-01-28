import os
from flask import Flask
from flask_cors import *
from flask_restful import Resource, Api
from flask_restful import reqparse

import time
import torch
import json
import re
import six
import base64
from PIL import Image

from detect_web001 import detect_main
from pathlib import Path
from threading import Lock
import requests
from io import BytesIO
from tempfile import TemporaryDirectory
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int)
    parser.add_argument('--weights', nargs='+', type=str, default='model_weights/gongfu_09_28_V3/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images_web', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output_web', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true',help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms',default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    return parser.parse_args()


i_args = init_args()


app = Flask(__name__)
CORS(app, supports_credentials=True)
api = Api(app)

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)

def base64_to_PIL(string):
    """
    base64 string to PIL
    """
    try:    
            base64_data = base64.b64decode(string)
            buf = six.BytesIO()
            buf.write(base64_data)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            return img
    except:
        return None
    
    
def PIL_to_base64(image):
    output = BytesIO()
    image.save(output,format='png')
    contents = output.getvalue()
    output.close()
    string = base64.b64encode(contents)
    return string

# 根据url获取图片，返回图片二进制数组
def get_image(url):
    # todo 上安全获取图片
    # 返回本地图片作为测试
    import logging
    import urllib
    url = urllib.parse.unquote(url)
    # print(url)
    logging.warning(url)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    #im = Image.open("./inference/images/004749.jpg")
    if img:
        logging.warning(img)
        return img
    else:
        logging.error(img)

def bs64_to_urls(data:bytes)->list:
    res = []
    print(data)
    for i, bstr in enumerate(data.encode('utf8').split(b' ')):
        res.append(base64.b64decode(bstr).decode('utf8'))
    return res


def predict():

    t = time.time()
    parser = reqparse.RequestParser()
    parser.add_argument('url',required=True)
    data = parser.parse_args()
    
    url = data['url']
    #url = url.decode("utf-8")
    img = get_image(url)
    res = []
    with TemporaryDirectory() as tmpdir:
        path = tmpdir + "/detect.jpg"
        if img.mode == "P":
            img = img.convert('RGB')
        for d in [0,90,180,270]:
            img_ = img.rotate(d)
            img_.save(path,"JPEG")
            res.extend(detect_main(path, i_args))
    print(res)
    if res == []:
        return '0'
    _res = '0'
    for r in res:
        if float(r.split(' ')[1]) > 0.1:
            _res = '1'
    print(res,_res)
    return _res


todos = {
            'favicon.ico':'',
            'predict':predict,
        }

class TodoSimple(Resource):
        def get(self, todo_id):
            print('todo --> {}'.format(todo_id))
            if todo_id != "favicon.ico":
                return {todo_id: todos[todo_id]()}

        def post(self,todo_id):
            print('todo --> {}'.format(todo_id))
            return {todo_id: todos[todo_id]()}

api.add_resource(TodoSimple, '/<string:todo_id>')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=i_args.port)

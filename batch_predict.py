import requests
import json
import cv2
import os,shutil
from skimage import io
# img_src = 'http://wx2.sinaimg.cn/mw690/ac38503ely1fesz8m0ov6j20qo140dix.jpg'
import numpy as np
import urllib
# import predictor 
from predictor import Predictor
import time

base_url = "http://172.18.39.210:8611"
down_load_img_last_pre_url = "/object-image/pull"
upload_img_last_pre_url = "/object-image/report-mark-result"
down_load_img_url = base_url + down_load_img_last_pre_url
upload_img_url = base_url + upload_img_last_pre_url
# upload_img_url = "F:/server_auto_mark"
predictor = Predictor()
class ImgModel:
    def __init__(self,img_id,img_class,tips,img_sn,img_url,img_base_url,is_Marked,file_name,created_time):
        self.img_id = img_id
        self.img_class = img_class
        self.tips = tips
        self.img_base_url = img_base_url
        self.img_url = img_url
        self.completed_img_url = self.img_base_url + self.img_url
        self.is_Marked = is_Marked
        self.file_name = file_name
        self.created_time = created_time
        self.img_sn = img_sn
        self.is_Auto_marked = False
        self.auto_marked_checked = False
        self.img_width = 0
        self.img_height = 0
        self.mark_result_dic = {}
        self.img_category_label = ""
        
        
    def get_label_str(self):
         return "sss"

def auto_mark_img(img):
    return [[10,"ss",100,100,200,200]]

def upload_img_mark_result(img_model):
    # cookies.clear
    headers = {
        "content-type": "application/json"
    }
    conn = requests.session()#设置一个回话
    data = json.dumps({'autoMarkChecked':0,'label':img_model.img_category_label,'markResult':json.dumps(img_model.mark_result_dic),'recordId':img_model.img_id})
    print("upload json......",data)
    resp = conn.post(upload_img_url,data,headers=headers)
    # 打印请求的头
    # print(resp.request.headers)
    up_img_reuslt = json.loads(resp.content)
    print("upload result......",up_img_reuslt)
    if up_img_reuslt["result"] == True:
        print("upload success")
    else:
        print("upload failed")
        
    

#返回左上角坐标
def get_left_top_position(bn_x_center,bn_y_center,bn_width,bn_height):
    return bn_x_center-bn_width/2,bn_y_center - bn_height/2
    
#返回四个顶点坐标,逆时针
def get_bbox_four_position(bn_x_center,bn_y_center,bn_width,bn_height):
    x_top_left,y_top_left = get_left_top_position(bn_x_center,bn_y_center,bn_width,bn_height)
    x_bottom_left = x_top_left
    y_bottom_left = y_top_left + bn_height
    
    x_bottom_right = x_bottom_left+bn_width
    y_bottom_right = y_bottom_left
    
    x_top_right = x_top_left + bn_width
    y_top_right = y_top_left
    
    return [x_top_left,y_top_left,x_bottom_left,y_bottom_left,x_bottom_right,y_bottom_right,x_top_right,y_top_right]
    
def get_bbox_area(bn_width,bn_height):
    return bn_width * bn_height
    


def batch_download():
	r = requests.get(down_load_img_url)
	content_str = str(r.text)
	json_data = json.loads(content_str)
	print("pull img data result======================")
	print("json_data:", json_data)
	images_jsons = json_data["result"]["objectImages"]
	print("images_jsons:", images_jsons)
	img_base_url = json_data["result"]["serverAddre"]
	

	for img_json in images_jsons[:]:
	    img_model = ImgModel(img_json["id"],img_json["type"],img_json["tips"],img_json["sn"],img_json["imgUrl"],img_base_url,img_json["isMarked"],img_json["fileName"],img_json["created"])
	    # img_model.completed_img_url = 'http://wx2.sinaimg.cn/mw690/ac38503ely1fesz8m0ov6j20qo140dix.jpg'
	    resp = urllib.request.urlopen(img_model.completed_img_url)
	    image = np.asarray(bytearray(resp.read()), dtype="uint8")
	    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	    if image is None:
	        print("imgurl is null",img_model.completed_img_url)
	    else:
	        width = image.shape[1]
	        height = image.shape[0]
	        if width < 10 or height < 10:
	            continue
	        up_result_json = {}
	        print("img_model.completed_img_url:", img_model.completed_img_url)
	        # mark_results = auto_mark_img(image)
	        mark_results = predictor.detect(image)
	        imags = [{"id":img_model.img_id,"file_name":img_model.file_name,"width":width,"height":height}]
	        up_result_json["images"] = imags
	        up_result_json["type"] = "instances"
	        annotations = []
	        categories = []
	        mark_result_count = 1
	        for mark_result in mark_results:
	            annation_item = {}
	            label_number = int(mark_result[0])
	            label = mark_result[1]
	            x_center = int(mark_result[2])
	            y_center = int(mark_result[3])
	            bbox_width = int(mark_result[4])
	            bbox_height = int(mark_result[5])
	            annation_item["segmentation"] = [get_bbox_four_position(x_center,y_center,bbox_width,bbox_height)]
	            annation_item["area"] = get_bbox_area(bbox_width,bbox_height)
	            annation_item["iscrowd"] = 0
	            annation_item["ignore"] = 0
	            annation_item["image_id"] = img_model.img_id
	            left_top_x,left_top_y = get_left_top_position(x_center,y_center,bbox_width,bbox_height)
	            annation_item["bbox"] = [left_top_x,left_top_y,bbox_width,bbox_height]
	            annation_item["category_id"] = label_number
	            annation_item["id"] = mark_result_count
	            annation_item["labelColor"] = "rgba(255,255,255,1)"
	            mark_result_count += 1
	            annotations.append(annation_item)
	    
	            category_item = {}
	            category_item["supercategory"] = "none"
	            category_item["id"] = label_number
	            category_item["name"] = label
	            categories.append(category_item)
	            
	            img_model.img_category_label = label
	        up_result_json["annotations"] = annotations
	        up_result_json["categories"] = categories
	        img_model.mark_result_dic = up_result_json
	        
	        print("begin upload mark result .......")
	        upload_img_mark_result(img_model)
        
if __name__ == '__main__':
	while True:
		time.sleep(10)
		batch_download()


        
# ss = "http://172.18.39.34:8181/ai/M00/00/3A/rBInIl6mi9OAVyiRAAAqn791pYg338.png"
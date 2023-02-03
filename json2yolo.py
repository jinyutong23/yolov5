import os
import json
import shutil

#-------------json文件的标签转yolo格式的标签--------------
json_dir = "/disk/home/jinyt/xuchuang2022/XuCh2/FR4_Set/annotations/"
txt_out = "./label/"

jsons = os.listdir(json_dir)
for i in jsons:
    json_file = json_dir + i
    json_name = i.split(".")[0]
    if not os.path.exists(txt_out):
        os.mkdir(txt_out)

    txt_file = txt_out + json_name+ ".txt"


    with open(json_file, 'r') as f:
        content = json.load(f)

    len_json = len(content)
    for i in range (len_json):
        w = content[i]["xmax"]-content[i]["xmin"]
        h = content[i]["ymax"]-content[i]["ymin"]
        xmin = content[i]["xmin"]
        ymin = content[i]["ymin"]
        centerx = xmin+w/2
        centery = ymin+h/2
        if content[i]["class"]=="溢胶":
            class_name = str(0)
        if content[i]["class"]=="脏污":
            class_name = str(1)
        if content[i]["class"]=="破损":
            class_name = str(2)
        
        yolo = class_name + " " + str(centerx)+ " "+ str(centery)+ " "+ str(w) +" "+ str(h) 
    
        
        with open(txt_file, 'a') as fp:
            fp.write(yolo+ "\n")


#------------按照标签将图片挑出来-----
json_dir = "/disk/home/jinyt/xuchuang2022/XuCh2/FR4_Set/annotations/"
image_dir = "/disk/home/jinyt/xuchuang2022/XuCh2/FR4_Set/images/"
image_out = "/disk/home/jinyt/project/yolov5/xuchuang_data/images/"

jsons = os.listdir(json_dir)
for i in jsons:
    json_path = json_dir + i
    json_name = i.split(".")[0]
    image_path = image_dir + json_name + ".jpg"

    shutil.copy(image_path,image_out+json_name+".jpg")



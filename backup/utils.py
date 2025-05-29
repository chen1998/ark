# %%
import os, glob
import sys
import subprocess
#
import numpy as np
import pandas as pd
import csv
#
from os import listdir
from os.path import isfile, isdir, join,getsize
from shutil import *
import shutil
from pathlib import Path

#
import cv2
import matplotlib.pyplot as plt
#
import torch
#
import warnings
warnings.filterwarnings("ignore")
#
from sqlalchemy import create_engine
import pandas as pd
import pymysql


# %%
import datetime, time
now_time = time.strftime("%Y%m%d", time.localtime())
now_month = now_time[0:6]
print(now_time)
#
from datetime import datetime

#%%

def sqldb_connect(type_to,table_name,database_name,savedf):
    host = "10.97.141.73"
    username = "ruby"
    password = "!!Ru542209056"
    #database_name = "mqc_data_form"
    conn = pymysql.connect(host=host, user=username, password=password, database=database_name)
    
    if type_to =="clean":
        cursor = conn.cursor()
        if savedf =='clean':
            sql_table_query = "SHOW TABLES LIKE '%clean%' "
        elif savedf == 'mpspec':
            sql_table_query = "SHOW TABLES LIKE '%mpspec%' "
        elif savedf == 'all':  
            sql_table_query = "SHOW TABLES"  
        cursor.execute(sql_table_query)
        clean_tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        conn.close()
        clean_tables = [v.upper() for v in clean_tables]

        return clean_tables
    
    elif type_to =="get_sql_table":
        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database_name}")
        sqltable = pd.read_sql_table(table_name, con=engine)
        conn.close()

        return sqltable
    
    elif type_to =="rename":
        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database_name}")
        sql = f"ALTER TABLE {table_name[0]} RENAME TO {table_name[1]};"

     
        with engine.connect() as conn:
            conn.execute(sql)
        return []
    elif type_to =="save":
        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database_name}")
        savedf.to_sql(table_name ,con = engine,if_exists = 'replace',index = False)
        return []
    
    elif type_to == 'del':
        cursor = conn.cursor()
        
        if isinstance(table_name, list) and len(table_name) > 0:
            for table in table_name:
                sql = f"DROP TABLE IF EXISTS {table};"
                cursor.execute(sql)
        
        elif table_name == '':
            sql = f"DROP DATABASE IF EXISTS {database_name};"
            cursor.execute(sql)

        conn.commit()
        cursor.close()
        conn.close()
        return []
    else:
        return []
    return []


#%%
import chardet
def encode_char_read_content(path):
    encoding = 0 
    try:
        with open(path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            if encoding !=0:
                with open(path, 'r', encoding=encoding) as file:
                    content = file.read()
                    return content
    except Exception as e:
        print(f"{path} :{e}")
        return e
#%%
def runchart_tab_clean_ppids(runchart_data):
    ppids = []
    for sheet_name, data in runchart_data.items():
        
        if '_' not in sheet_name :
            continue
        row0 = data[data.columns[0]].tolist()
        ppid_ixs = [ix for ix , v in zip(range(len(row0)),row0) if v =='Recipe']
        
        
        for ix in ppid_ixs :
            ppids = ppids + [v for v in data.loc[ix].tolist() if is_number(v) and not pd.isna(v)]
        ppids = list(set(ppids))
    return ppids


# %%
#

# %%
def is_number(s):
    try:  
        float(s)
        return True
    except ValueError:  
        return False

def rowdata_clean(path):
    try:
        with open(path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            df = pd.DataFrame(rows)
            info_df_columns = ['DeviceName', 'PROC TOOL', 'Lot ID','SLOT NO', 'Date (YYYY/MM/DD)', 'Start Time (hh:mm:ss)']
            info_df = df.iloc[:28,:2]
            info_df = info_df[info_df.apply(lambda row: all(row !='x'),axis=1)]
            values = [df[df.columns[1]][ix] for ix in info_df.index if df[df.columns[0]][ix] in info_df_columns]
            values.append(values[-2].replace('/','') + values[-1].replace(':',''))
            
            return [v.replace(" ","") for v in values if "/" not in v and ":" not in v]
    except Exception as e:
        issue = "issue(rowdata_clean): %s"%(e)
        return [issue]

def array_vstack(array1 , list1):
    if len(array1) ==0:
        array1 = np.array(list1,dtype = object)
    else:
        array1 = np.vstack([array1, list1])
    return array1

def shutil_file(old_folder,old_fnlist,new_folder):
    os.makedirs(new_folder , exist_ok=True)
    for fn in old_fnlist:
        old_path = os.path.join(old_folder,fn)
        new_path = os.path.join(new_folder,fn)
        if fn not in os.listdir(old_folder):
            shutil.copy(old_path , new_path )

# %%
def color_mark(img,color):
    if color =="red":
        lg,ug = np.array([0, 50, 50]),np.array([10, 255, 255])
    elif color =="green":
        lg,ug = np.array([35, 50, 50]),np.array([140, 255, 255])
    elif color =="blue":
        lg,ug = np.array([35, 1, 200]),np.array([140, 255, 255])
    elif color =="black":
        lg,ug = np.array([0,0,0]),np.array([1,1,1])

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lg,ug)
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask))
    hsv_img = cv2.bitwise_or(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),cv2.cvtColor(gray_mask,cv2.COLOR_GRAY2BGR))
   
    #hsv_img = img.copy()
    #hsv_img[np.where(mask==255)]=[255,255,255]
    return mask,hsv_img

def rotate_bound(image, angle):
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def colormark_pixel_tolimit(image):
    limit_xlist , limit_ylist = [],[]
    for color in ["blue","green"]: #"red",
        mask,maskimg = color_mark(image.copy(),color)
        color_pixel = np.where(mask)
        for axis in [1,0]: #1:x , 0:y
            try:
                if axis ==1:
                    limit_xlist = limit_xlist + [np.min(color_pixel[axis]),np.max(color_pixel[axis])]
                else:
                    limit_ylist = limit_ylist + [np.min(color_pixel[axis]),np.max(color_pixel[axis])]
            except Exception as e:
                print(e)

    if len(limit_xlist) !=0:
        min_x = min(limit_xlist)
        max_x = max(limit_xlist)
        if max_x < image.shape[1]/2 and min_x < image.shape[1]/2:
            x_limit1 = int(round((max_x + min_x)/2))
            x_limit2 = image.shape[1]
        elif max_x > image.shape[1]/2 and min_x > image.shape[1]/2:
            x_limit2 = int(round((max_x + min_x)/2))
            x_limit1 = 0
        elif max_x - min_x < 300:
            x_limit1 , x_limit2 = min_x+15 , max_x-15
        else:
            x_limit1 , x_limit2 = 0 , image.shape[1]
    else:
        x_limit1 , x_limit2 = 0 , image.shape[1]
    
    if len(limit_ylist) !=0:
        min_y = min(limit_ylist)
        max_y = max(limit_ylist)
        #if (max_y - min_y)<350 and (max_y - min_y)>200:
        if max_y > image.shape[0] / 2 and min_y < image.shape[0] / 2:
            y_limit1 , y_limit2 = min_y + 15 , max_y - 15
        elif max_y < image.shape[0]/2:
            y_limit1 = int(round((max_y + min_y)/2))
            y_limit2 = image.shape[0]
        elif min_y > image.shape[0]/2:
            y_limit1 = 0
            y_limit2 = int(round((max_y + min_y)/2))
        else:
            y_limit1 , y_limit2 = 0,image.shape[0]
    else:
        y_limit1 , y_limit2 = 0,image.shape[0]

    return x_limit1 , x_limit2 , y_limit1 , y_limit2

def pre_image_processing(fnlist,processing_folder):
    
    #fnlist = glob.glob(os.path.join(image_folder, "*"))
    resizes = []
    for imgpath in fnlist:
        #print(imgpath)
        fn = imgpath.split("\\")[-1]
        image = cv2.imread(imgpath)
        if image is None:
            continue 
        image_height, image_width = image.shape[:2]
        x_limit1 , x_limit2 , y_limit1 , y_limit2 = colormark_pixel_tolimit(image)
        #print(x_limit1 , x_limit2 , y_limit1 , y_limit2)
        if  abs(y_limit1 - y_limit2) > abs(x_limit1 - x_limit2):
            rotate = -90
            rotate_image = rotate_bound(image, rotate)
            x_limit1 , x_limit2 , y_limit1 , y_limit2 = colormark_pixel_tolimit(rotate_image)
            cut_image = rotate_image[y_limit1:y_limit2,x_limit1:x_limit2]
        else:
            rotate = 0
            cut_image = image[y_limit1:y_limit2,x_limit1:x_limit2]

        cut_image_height, cut_image_width = cut_image.shape[:2]
        processing_imgpath = os.path.join(processing_folder,fn)
        #print(cut_image.shape)
        resizes.append([fn,image_height, image_width,cut_image_height, cut_image_width,x_limit1,x_limit2,y_limit1,y_limit2,rotate])
        #print(processing_imgpath)
        cv2.imwrite(processing_imgpath, cut_image)

    processing_df = pd.DataFrame(resizes, columns=['Image name', 'image_height', 'image_width','cut_image_height', 'cut_image_width','x_limit1','x_limit2','y_limit1','y_limit2','rotate'])
    return processing_df
        

# %%
def obj_detect(image_folder,output_folder):
    model_path = 'D:/yolo/yolov5/runs/train/chipid_yolov5s3/weights/best.pt'
    #model_path = 'D:/Project/Chipid_identify/runcode/best.pt'
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"detect {image_folder}----")
    command = [
        'python', 'D:/yolo/yolov5/detect.py',
        '--weights', model_path,
        '--imgsz', '416',
        '--conf-thres', '0.4',
        '--source', image_folder,
        '--project', output_folder,
        '--name', 'exp', 
        '--save-txt' 
    ]
    result = subprocess.run(command, capture_output=True, text=True)   
    #print(f'{output_folder}---')
    return result
'''
def code_dot_label(code):
    show_image = code.copy()
    img_blur = cv2.GaussianBlur(code,(3,3),7)
    #edged = cv2.Canny(img_blur.copy(), 30,80)
    edged = cv2.Canny(img_blur.copy(), 80,120) 
    cnt,_ = cv2.findContours(edged, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    dot_count = 0
    for idx, ct in enumerate(cnt):
        x1,y1,w,h = cv2.boundingRect(ct)     
        x2, y2, ar = x1+w, y1+h, w*h
        area = cv2.contourArea(ct)
        perimeter = cv2.arcLength(ct,True)
        mid_xpoint = float( (x1 + x2)/2)
        mid_ypoint = float( (y1 + y2)/2)
        center_coordinates=( int(mid_xpoint) , int(mid_ypoint)  )
        radius = float((h+w)/4)
        dot_circle =radius**2*np.pi
        if radius >1 and radius <6 and area<50 and x1 > 10 and x1 > 10 and y1 > 10 :
            cv2.circle(show_image, center_coordinates, 2 , (253,253,19), -1)
            dot_count+=1
    
    return show_image,dot_count
'''  
#%%
def code_dot_label(code):
    show_image = code.copy()
    gray_image = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_image, (5, 5), 1)
    edged = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    code_height, code_width = code.shape[:2]
    max_xe , max_ye = code_width-14,code_height-14
    dot_count = 0
    '''
    c = 0
    d=4
    plt.figure(figsize = (10,3))
    c+=1;plt.subplot(1,d,c) ; plt.title('code',fontsize = 16) ; plt.imshow(code)
    c+=1;plt.subplot(1,d,c) ; plt.title('blur',fontsize = 16) ; plt.imshow(img_blur)
    c+=1;plt.subplot(1,d,c) ; plt.title('edged',fontsize = 16) ; plt.imshow(edged)
    '''
    for ct in contours:
        x1,y1,w,h = cv2.boundingRect(ct)     
        x2, y2, ar = x1+w, y1+h, w*h
        area = cv2.contourArea(ct)
        perimeter = cv2.arcLength(ct,True)
        mid_xpoint = float( (x1 + x2)/2)
        mid_ypoint = float( (y1 + y2)/2)
        center_coordinates=( int(mid_xpoint) , int(mid_ypoint)  )
        radius = float((h+w)/4)
        dot_circle =radius**2*np.pi
        if radius >1 and radius <25 and x1 > 10 and y1 > 10 and x2<max_xe and y2<max_ye :
            cv2.circle(show_image, center_coordinates, 2 , (253,253,19), -1)
            dot_count+=1
    #c+=1;plt.subplot(1,d,c) ; plt.title('show',fontsize = 16) ; plt.imshow(show_image)
    return show_image,dot_count
#%%
def plt_image_save_show(fnbox,imgbox,imgpath):
    c = 0  
    plt.figure(figsize = (10,4))
    if len(fnbox[0:len(imgbox)])>9:
        d = 9
        for img,name in zip(imgbox,fnbox[0:len(imgbox)]):
            c+=1;plt.subplot(1,d,c) ; plt.title(name,fontsize = 16) ; plt.imshow(img )
            if c==9:
                plt.show()
                plt.figure(figsize = (20,10))
                c=0
    else:
        d=len(fnbox[0:len(imgbox)])
        for img,name,ix in zip(imgbox,fnbox[0:len(imgbox)],range(len(imgbox))):
            c+=1;plt.subplot(1,d,c) ; plt.title(name,fontsize = 16) 
            plt.xticks(fontsize = 24)
            plt.imshow(img)
    #imgpath= os.path.join(fld,"%s.jpg"%(fn))
    #plt.savefig(imgpath,bbox_inches = 'tight')
    #plt.close()
    plt.show()

# %%
def judge_um_data(results):
    judges  = []
    all_alert = []
    for index , row in results.iterrows():
        alerts = []
        dot_count , top , buttom , xe , code_w , code_h = row[['2Dcode_dot count', 'Top(um)', 'Buttom(um)', 'Xedge(um)','2Dcode_width(um)', '2Dcode_height(um)']]
        if is_number(code_w) and is_number(code_h ) and int(dot_count )<10:
            alert = f'2Dcode dot count<10({dot_count })'
            alerts.append(alert)
            if float(code_w)<800 or float(code_h)<800:
                alert = f'2Dcode width or height alert({code_w}x{code_h})'
                alerts.append(alert)
        for v ,name,spec in zip([top , buttom , xe],['TOP','Buttom','XEdge'],[[150,690],[150,690],[500,1200]]):
            if is_number(v):
                if v<spec[0] or v>spec[-1]:
                    alert = f'{name} OVER SPEC ({v})'
                    alerts.append(alert)
            elif v !='Mid':
                alert = f'{name} ERROR'
                alerts.append(alert)

        if len(alerts) !=0:
            alert = "\n".join(alerts)
            all_alert.append(alert)
            judges.append("NG")
        else:
            all_alert.append("")
            judges.append("OK")

    results['Judge'] = judges
    results['Alert'] = all_alert
    return results
#results = judge_um_data(results)
#results

# %%

def output_obj_axis(output_root, show_folder, original_image_folder, processing_df, rls):
    #
    exp_folder = os.path.join(output_root, "exp")
    os.makedirs(show_folder, exist_ok=True)
    #
    dot_folder = os.path.join(show_folder,"dot")
    os.makedirs(dot_folder, exist_ok=True)
    #
    print("obj_axis-> %s"%str(len(processing_df)))
    labels_folder = Path(os.path.join(exp_folder, 'labels'))
    #
    csv_path = os.path.join(show_folder,'identify.csv')
    #
    results = []
    c=0
    
    code_width_um , code_height_um = 1000 , 1000
    height_um = 1840
    
    for index, row in processing_df.iterrows():
        dot_count ,dot_image_height , dot_image_width, code = "","","",""
        dot_um_per_px , cut_um_per_px = 0, 0 
        image_name , cut_height_px , cut_width_px = row[['Image name','cut_image_height','cut_image_width']]
        x_limit1, x_limit2, y_limit1, y_limit2, angle = row[['x_limit1', 'x_limit2', 'y_limit1', 'y_limit2','rotate']]
        
        cut_um_per_px = cut_height_px / height_um
        #print(image_name)
        original_image_path = os.path.join(original_image_folder, image_name)
        if not os.path.exists(original_image_path):
            print(f'fail: {original_image_path}')
            continue

        original_image = cv2.imread(original_image_path)
        original_image_height , original_image_width = original_image.shape[:2]
        h2 , w2 = int(original_image_height/2), int(original_image_width/2)
        #
        label_file = labels_folder / (Path(image_name).stem + '.txt')
        if not label_file.exists():
            continue
        edges =[]
        #print(cut_height_px , cut_width_px)
        #print(x_limit1, x_limit2, y_limit1, y_limit2)
        output_image_path = os.path.join(show_folder, image_name)
        with open(label_file, 'r') as file:
            label_count = 0
            for line in file:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                x_center *= cut_width_px
                y_center *= cut_height_px
                w *= cut_width_px
                h *= cut_height_px
                if w<20 or h<20:
                    continue
                x1 = int(x_center - w / 2) + x_limit1
                y1 = int(y_center - h / 2) + y_limit1
                x2 = int(x_center + w / 2) + x_limit1
                y2 = int(y_center + h / 2) + y_limit1
                list1 = [label_count , class_id]
                list1 = list1 + [abs(round(x1-x_limit1)), abs(round(y1-y_limit1)),abs(round( x2-x_limit1)), abs(round(y2-y_limit1)), round(w, 2), round(h, 2)]
                
                if row['rotate'] != 0:
                    x1, y1, x2, y2 = y1 , x1 , y2 , x2
                if class_id == 1 and abs(w-h)<10:
                    code = 1
                    code_image = original_image[y1:y2,x1:x2]

                    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    dot_image,dot_count = code_dot_label(code_image)
                    #cv2.imwrite(os.path.join(dot_folder,image_name), dot_image)
                    dot_image_height , dot_image_width = dot_image.shape[:2]
                    
                    #print(f'{dot_image_height} ,{ dot_image_width} : {dot_count}')
                    list1.append(dot_count)
                else:
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    list1.append("")
                    
                
                #print(label_count,(x1-x_limit1, y1-y_limit1), (x2-x_limit1, y2-y_limit1))
                edges.append(list1)
                cv2.putText(original_image, str(label_count), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                label_count+=1
        
        #cv2.imwrite(output_image_path, original_image)
        #
        
        #print(f'{edges}\n---------')
        c1 , c2 , c3 , c4 = 4,2,3,5
        max_x2 = max([int(v) for v in [v[c1] for v in edges] if is_number(v)])
        min_x1 = min([float(v) for v in [v[c2] for v in edges] if is_number(v)])
        max_y2 = max([float(v) for v in [v[c4] for v in edges] if is_number(v)])
        min_y1 = min([float(v) for v in [v[c3] for v in edges] if is_number(v)])
        #
        top_gap = int(min_y1)
        buttom_gap =int( abs(cut_height_px - max_y2))
        #
        if abs(min_x1)<50 and abs(cut_width_px-max_x2)<50:
            Xedge = 'Mid'
        elif abs(min_x1) > abs(cut_width_px-max_x2):
            Xedge = round(min_x1) 
           
        elif  abs(cut_width_px-min_x1) > abs(min_x1) :
            Xedge = cut_width_px -round(max_x2)
            

        #print(f'{image_name} , angle:{angle} -> \ntop_gap:{top_gap},buttom_gap:{buttom_gap} ,min_x1:{min_x1} , max_x2:{max_x2},Xedge:{Xedge}')
        

        if code ==1:
            new_width = int(dot_image_width * (original_image_height / dot_image_height))
            resized_image = cv2.resize(dot_image.copy(), (new_width, original_image_height))
            merged_image = np.hstack((original_image.copy(), resized_image))
            cv2.imwrite(os.path.join(dot_folder,image_name), merged_image)
            dot_um_per_px = dot_image_width / code_height_um
            show_img = merged_image.copy()
        else:
            show_img = original_image.copy()
        #
        cv2.imwrite(output_image_path, show_img)
        #
        if dot_um_per_px !=0:
            um_per_px = (cut_um_per_px + dot_um_per_px) *0.5
        else:
            um_per_px = cut_um_per_px
        #
        #print(f"cut:{cut_um_per_px } dot:{ dot_um_per_px} = {um_per_px}---\n")
        cut_height_um , cut_width_um = cut_height_px *  um_per_px , cut_width_px *  um_per_px
        result = [image_name ,  angle ,um_per_px , top_gap , buttom_gap , Xedge  , dot_image_width , dot_image_height,  dot_count ]
        results = array_vstack(results , result)
    

        

    results = pd.DataFrame(results, columns=['Image name', 'Image Angle', 'um_per_px', 'Top(px)', 'Buttom(px)', 'Xedge(px)','2Dcode_width(px)', '2Dcode_height(px)', '2Dcode_dot count' ])
    for cl in ['Top(px)', 'Buttom(px)', 'Xedge(px)', '2Dcode_width(px)', '2Dcode_height(px)']:
        values =[]
        for v in results[cl]:
            if is_number(v):
                v = round(float(v)/float(um_per_px))
                
            values.append(v)
        results[cl.replace("px","um")]  = values
    
    results = judge_um_data(results)
    results.to_csv(csv_path, index=False)
    #print(results.iloc[::,1::])
    #
    return results
#obj_df = output_obj_axis(mark_folder, processing_folder, preimg_folder, processing_df)




# %%
def rls_split_um(values,char):
    for value in values:
        if  value !='x':
            v1,v2 = [int(v) for v in value.split(char)]
            return v1 , v2
    return 'x','x'

# %%
def caculate_judge_data(boxes):
    if len(boxes) ==1:
        data  = boxes[0]
        max_YEdge = data[4]
        min_YEdge = data[2]
        min_XEdge = data[1]
        max_XEdge = data[3]

        if data[0] ==1:
            code_data = data
            data  = []
        else:
            code_data = []
            data = boxes[0]

    else:
        data = np.array(boxes,dtype=object)
        code_data = data[data[:,0] == 1 ]
        data = data[data[:,0] == 0 ]
        
        max_YEdge = np.max(data[1:,4]) if len(data) >1 else data[0][4]
        min_YEdge = np.min(data[1:,2]) if len(data) >1 else data[0][2]
        min_XEdge = np.min(data[1:,1]) if len(data) >1 else data[0][1]
        max_XEdge = np.max(data[1:,3]) if len(data) >1 else data[0][3]


    if len(code_data) !=0:
        x1, y1, x2, y2 = code_data[0][1:] if isinstance(code_data,np.ndarray) else code_data[1:]
        w = abs(x2-x1)
        h = abs(y2-y1)
        if w > 100 and h > 100 and abs(w-h)<15:
            
            code_data = ",".join([str(int(v)) for v in [x1, y1, x2, y2]])
            code_height = h
            code_x1 = x1
        else:
            code_data = ''
            code_height = ''
            code_x1 = ''
    else:
        code_data = ''
        code_height = ''
        code_x1 = ''

    return min_XEdge, min_YEdge, max_XEdge, max_YEdge, code_data, len(data), code_height, code_x1

#%%
def convert_coords_to_original(label_coords, processing_data):
    x_center, y_center, width, height = label_coords
    cut_img_width, cut_img_height = processing_data['cut_image_width'], processing_data['cut_image_height']
    
    x_center = x_center * cut_img_width
    y_center = y_center * cut_img_height
    width = width * cut_img_width
    height = height * cut_img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    
    return x1, y1, x2, y2

def draw_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
   
    for box in boxes:
        label, x1, y1, x2, y2 = box
        if label != 1 :
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness)
            
    return image

def process_labels_and_draw_boxes(processing_df, label_folder, original_images_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    code_folder = f'{output_folder}/code_output'
    os.makedirs(code_folder , exist_ok=True)
    edges = []
    code_edges = []
    for index, row in processing_df.iterrows():
        img_name = row['Image name']
        label_path = os.path.join(label_folder, f"{os.path.splitext(img_name)[0]}.txt")
        original_img_path = os.path.join(original_images_folder, img_name)

        if not os.path.exists(label_path) or not os.path.exists(original_img_path):
            continue
        
        original_image = cv2.imread(original_img_path)

        
        with open(label_path, 'r') as f:
            boxes = []
            for line in f.readlines():
                #print(line)
                label_data = list(map(float, line.strip().split()))  
                x1, y1, x2, y2 = convert_coords_to_original(label_data[1:], row)
                
                boxes.append((label_data[0], x1, y1, x2, y2))
        
        image_with_boxes = draw_boxes_on_image(original_image, boxes)

        output_img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_img_path, image_with_boxes)

        min_XEdge, min_YEdge, max_XEdge, max_YEdge, code_data, num_id, code_height, code_startX = caculate_judge_data(boxes)
        
        if code_data != '':
            x1, y1, x2, y2 = list(map(int,code_data.split(",")))
            code_image = original_image[y1:y2,x1:x2]
            show_image,dot_count = code_dot_label(code_image)
            #print(f'2Dcode dot:{dot_count}-----------')
            cv2.imwrite(f'{code_folder}/{img_name}', show_image)
            #dot_count
        else:
            dot_count = ''
        
        edges.append([img_name, min_XEdge, min_YEdge, max_XEdge, max_YEdge, code_data, num_id, code_height, code_startX,dot_count])
    edges = pd.DataFrame(edges , columns =['Image name','MinX', 'MinY', 'MaxX', 'MaxY', 'CodeData', 'Num_ID', 'code_height', 'code_startX','Num_Dot'])   
    #
    
    edges = pd.merge(processing_df,edges, on = 'Image name', how = 'left')

    #print(f"Processed images saved to {output_folder}")
    return edges#edge_df

#%%
def IDType_calculate_um_per_pixel(rls):
    um_dict = rls[4:12]
    
    chipid_size, code_size, edge_size = [rls[ix] for ix in [ 5, 6, 11 ]]
    #print(chipid_size, code_size, edge_size)
    chipid_size = list(map(int, chipid_size.split("*") ))
    code_size = list(map(int, code_size.split("*") ))
    top, buttom = [ v if '/' not in v else v.split("/") for v in edge_size.split("~")]

    if isinstance(buttom,list):
        buttom, only2dcode_buttom = buttom[0],[buttom[-1]]
    else:
        only2dcode_buttom = 0
    
    edge_size = [top, buttom]

    if int(rls[4]) == 2:
        only2dcode_ROI_size, only2dcode_code_size, onlychipid_size =  [ v.split("*") if '*' in v else 0 for v in rls[8:11] ]
    else:
        only2dcode_ROI_size, only2dcode_code_size, onlychipid_size = 0,0,0

    
    
    return chipid_size, code_size, edge_size, only2dcode_buttom, only2dcode_ROI_size, only2dcode_code_size, onlychipid_size
    
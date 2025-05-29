# %%
import os, glob
import sys
import subprocess
#
from collections import Counter
import numpy as np
import pandas as pd
import csv
import math
import chardet
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
import inspect
import utils  
from utils import *
module_members = dir(utils)
functions = [member for member in module_members if inspect.isfunction(getattr(utils, member))]
print("utils.py -->")
#for func_name in functions:
#    print(func_name)

# %%

# %%
#sqldb_connect('save','chipid_control_rls','chip_id_model_rule',model_rules)
model_rules = sqldb_connect('get_sql_table','chipid_control_rls','chip_id_model_rule','')
rls_models = model_rules['abbrename'].tolist()

# %%
import time
import datetime
from datetime import timedelta
current_date = datetime.datetime.now()
print(current_date)
strTime = current_date.strftime("%Y-%m-%d %H:%M:%S")
print(strTime)
mon = str(current_date.strftime("%m"))
years2_ago = current_date - timedelta(days=2*365)
now_time = time.strftime("%Y%m%d", time.localtime())
print(f'folder date:{now_time}')
#


# %%
#
project_root = "D:/Project/Chipid_identify"

#
aaimf001_root = "//10.97.36.14"
mov_root = f'{aaimf001_root}/aamov/{mon}'
print(f'download from {mov_root}')
#
objdetet_output_root = os.path.join(project_root ,"objdetect_output")
eda_df_root = f'{objdetet_output_root}/backup_edge_result'

#
today_output_root = os.path.join(objdetet_output_root,str(now_time))

#
#
dist3wServer= 'E:/wamp64/www'
chipWebRoot = f'{dist3wServer}/chipid2dcode'
detect_results_root = f'{chipWebRoot}/detect_results'
towebRCP_folder = f'{chipWebRoot}/UserFeedBack(CID)'
#
web_server = "http://10.97.141.73"
project_http = f'{web_server}/chipid2dcode'
#
his_marks = os.listdir(chipWebRoot)
#his_marks 
#
#okpath = f"{project_root}/control/ID_control_table.xlsx"
#control_rls = pd.read_excel(okpath, sheet_name="table")
ORC_recipe_root = 'D:/ORC_recipe'
#
process_folder = 'D:/Project/process_info'
runchart_path = f'{process_folder}/runchart.xlsx'
runchart_file = pd.ExcelFile(runchart_path)
#
models_df = sqldb_connect("get_sql_table","models_info_data",'mqc_data_form',"")

# %%
adi_paths = [folder for folder in glob.glob(os.path.join(mov_root, "*")) if 'P1 ADI' in folder]
chipid_folders = [[folder , [model for model in rls_models if model in folder][0]] for folder in adi_paths if len([model for model in rls_models if model in folder]) !=0]
if len(chipid_folders) ==0:
    print('MOV 0 Data---')
    sys.exit()
else:
    os.makedirs(today_output_root, exist_ok=True)
    print(chipid_folders )

# %%


# %%
edge_df = []
info_cols = ['Model','Mark','MQCTool','IEX','Lot','Slot','DateTime','Date']

for folder_data in chipid_folders :
    folder , model = folder_data
    print(f'-{folder}->')
    
    for f in [f for f in glob.glob(os.path.join(folder, "*")) if '.txt' in f ]:
        
        mark = f.split("\\")[-1].split(" ")[0] 
        info = rowdata_clean(f)
        
        file_time = info[-1]
        rls = model_rules.loc[(model_rules['abbrename'] ==model) & (model_rules['MC'].str.contains(info[0][2::])) ]
        #
        if len(rls) ==0:
            continue
        
        info = [model,mark] + info + [info[-1][4:8]]
        
        
        #
        start_no = int(rls['start_no'][rls.index[0]])
        
        
        imgfolder = f.replace(".txt",'_JPG')
        imglist = [v for v in os.listdir(imgfolder) if '.jpg' in v  and int(v.split("_")[0]) >= start_no ]
        #
        img_counts = len(imglist)
        if img_counts <4:
            continue
        elif len(info) <7 :
            info = info + ['']*(7-len(info))
        #
        mark_folder = os.path.join(today_output_root,mark)
        os.makedirs(mark_folder , exist_ok=True)
        shutil.copy(f,os.path.join(mark_folder,f.split("\\")[-1]))
        #
        oriimg_folder = os.path.join(mark_folder,"Image")
        os.makedirs(oriimg_folder , exist_ok=True)
        if len(os.listdir(oriimg_folder)) !=len(imglist):
            print(f'copy mov image data({img_counts}) to dist folder')
            shutil_file(imgfolder,imglist ,oriimg_folder)
        
        #
        print(f'{mark}(file_time ): {img_counts}')
        #
        processing_folder = f'{mark_folder}/processing'
        os.makedirs(processing_folder , exist_ok=True)
        processing_df = pre_image_processing(glob.glob(os.path.join(oriimg_folder , "*")),processing_folder)
        
        #
        detect_outputfolder = f'{mark_folder}/detect_output'
        label_folder = os.path.join(detect_outputfolder, 'exp/labels')  
        if not os.path.exists(label_folder):
            detect_result = obj_detect(processing_folder,detect_outputfolder)
        else:
            print(f'{mark} already detect')
        #
        relables_folder = f'{mark_folder}/backup_output'
        #
        
        edges = process_labels_and_draw_boxes(processing_df, label_folder, processing_folder, relables_folder)
        for cl,value in zip(info_cols,info ):
            edges[cl] = value
        edges.to_csv(f'{mark_folder}/{file_time}_{mark}_edges(px).csv')
    
        if len(edges) >0 and len(edge_df)>0:
            edge_df = pd.concat([edge_df, edges])
        elif len(edges) >0:
            edge_df = edges
        else:
            edge_df = pd.DataFrame([[None]*len(edges.columns) + info],columns = edges.columns.tolist() + info_cols)
        #
        webmark_folder = os.path.join(detect_results_root,mark)
        if os.path.exists(webmark_folder):
            continue
        shutil.copytree(relables_folder,webmark_folder)
    
    

# %%
if len(edge_df) ==0:
    sys.exit()
edge_df.reset_index(inplace = True,drop = True)   
rls_columns = ['ROI size','2dcode size', 'top/buttom', 'only2dcode_buttom', 'only2dcode ROI size', 'only2dcode size','onlyid ROI size']
for cl in ['px_per_um','model_key','chips']:
    edge_df[cl] = None

for model in edge_df['Model'].unique():
    if model is None:
        continue
    rls = model_rules[model_rules['abbrename'] .str.contains(model)]
    
    if len(rls) ==0:
        continue
    rls = rls.loc[rls.index[0]].tolist()
    key, chips = rls[-3], rls[-1]
    print(f'{model}-> {key}\n{chips}')
    chipid_size, code_size, edge_size, only2dcode_buttom, only2dcode_ROI_size, only2dcode_code_size, onlychipid_size = \
        IDType_calculate_um_per_pixel(rls)
    
    uni = edge_df[edge_df['Model'] == model]
    uni_idxs = uni.index.tolist()
    
    #uni_df['height_per_um'] = [min(w,h)/chipid_size[0] for w,h in zip(uni_df['cut_image_width'],uni_df['cut_image_height']) ]
    rls_data = [chipid_size, code_size, edge_size, only2dcode_buttom, only2dcode_ROI_size, only2dcode_code_size, onlychipid_size,]
    for lst,cl in zip(rls_data,rls_columns):
        #print(f'{cl}->{lst}')
        if isinstance(lst,list):
            edge_df.loc[ uni_idxs,cl] =  ",".join(list(map(str,lst)))
        else:
            edge_df.loc[ uni_idxs,cl] =  None
    data= []
    um_h , um_code_h = chipid_size[0], code_size[0]
    for ix, h, w, ch in zip(uni_idxs,uni['cut_image_height'],uni['cut_image_width'],uni['code_height']):
        #print(ix, h, w, ch)
        h = min([h,w])
        per_um = h/um_h
            
        if is_number(ch):
            code_per_um = ch/um_code_h
            #print(f'code:{code_per_um}')
            per_um = (per_um +  code_per_um)/2
        #print(f'{ix}->{per_um}')
        data.append(per_um)
        edge_df['px_per_um'][ix] = per_um
    edge_df.loc[uni.index,['model_key','chips']] = key, chips

# %%
per_cols = [ 'cut_image_height','cut_image_width', 'x_limit1', 'x_limit2', 'y_limit1', 'y_limit2', 'MinX', 'MinY', 'MaxX', 'MaxY', 'code_height', 'code_startX']

for cl in per_cols:
    edge_df[f'{cl}(px)'] = edge_df[cl].tolist()
    #edge_df[cl] = [float(v)/per if v is not None and is_number(v)  else '' for v,per in zip(edge_df[f'{cl}(px)'],edge_df['px_per_um'])]
    #print(cl)
    edge_df[cl] = [round(float(v)/per)  if v is not None and v !='' and is_number(v) else None for v,per in zip(edge_df[f'{cl}(px)'],edge_df['px_per_um'])]

for cl in ['Judge','Alert']:
    edge_df[cl] = None

# %%
edge_df['No'] = [v.split("_")[0] for v in edge_df['Image name']]
edge_df.sort_values(by = ['DateTime','Mark','No'],ascending=True,inplace = True)
edge_df.reset_index(inplace = True,drop = True)

# %%
alert_dict = []
only_code_df = edge_df.loc[(edge_df['Num_ID'] == 0) & (edge_df['CodeData'] != '') & (edge_df['cut_image_width'] < 1500) & (edge_df['cut_image_height'] <1500)]
only_code_df = only_code_df.loc[[ix for ix, v  in zip(only_code_df.index,only_code_df['only2dcode_buttom']) \
                                if not pd.isna(v) and v is not None ],only_code_df.columns]
if len(only_code_df)>0:
    alert_dict = alert_dict + [[ix,f'Dot < 10({v})', 'NG'] for ix , v in zip(only_code_df.index,only_code_df['Num_Dot']) if v <10]
    for buttom in only_code_df['only2dcode_buttom'].unique():
        buttom = float(buttom)
        for cl , edge in zip(['MinX', 'MinY', 'MaxX', 'MaxY'],['Left','Top','Right','Buttom']):
            if 'Min' in cl:
                alert_dict = alert_dict + [[ix,f'{edge} over({v})', 'NG'] for ix , v in zip(only_code_df.index,only_code_df[cl]) if v > buttom*1.5 or v < buttom*0.5]
            else:
                if 'X' in cl:  #-> 32*8 different w,h
                    eix = 1
                else:
                    eix = 0
                alert_dict = alert_dict + [[ix,f'{edge} over(%s)'%str(round(abs(int(roi.split(",")[eix])-v))), 'NG'] for ix , v,roi in zip(only_code_df.index,only_code_df[cl],only_code_df['only2dcode ROI size']) \
                                           if abs(int(roi.split(",")[eix])-v) > buttom*1.5 or abs(int(roi.split(",")[eix])-v) < buttom*0.5]
    #alert_dict = alert_dict + [[ix,'','OK'] for ix in only_code_df.index]
#
id_code_df = edge_df.loc[[ix for ix in edge_df.index if ix not in only_code_df.index],edge_df.columns]
#print(id_code_df .loc[id_code_df .index,['Num_ID','Num_Dot','CodeData']])
alert_dict = alert_dict + [[ix,f'ID < 2({v})', 'NG'] for ix , v in zip(id_code_df.index,id_code_df['Num_ID']) if v<2]
alert_dict = alert_dict + [[ix,f'Dot < 10({v})', 'NG'] for ix , v, code in zip(id_code_df.index,id_code_df['Num_Dot'],id_code_df['CodeData']) if code != '' and v <10]

for edge in id_code_df['top/buttom'].unique():
    
    top , buttom = list(map(float,edge.split(",")))
    alert_dict = alert_dict + [[ix,f'Top over({v})', 'NG'] for ix , v in zip(id_code_df.index,id_code_df['MinY']) if v<200 ]
    alert_dict = alert_dict + [[ix,'buttom over(%s)'%str(round(abs(int(roi.split(",")[0])-v))), 'NG'] for ix , v,roi in zip(id_code_df.index,id_code_df['MaxY'],id_code_df['ROI size']) if abs(int(roi.split(",")[0])-v)>buttom or abs(int(roi.split(",")[0])-v)<top ]
#
edge_df['code_edge'] = [max([abs(c-x1),abs(c-x2)]) if c is not None and not pd.isna(c) else ''  for c,x1,x2 in zip(edge_df['code_startX'],edge_df['x_limit1'],edge_df['x_limit2'])]
alert_dict = alert_dict + [[ix,'2DCode to edge over({c})'] for ix,c,num in zip(edge_df.index,edge_df['code_edge'],edge_df['Num_ID']) if is_number(c) and (int(c)<2000) and ((int(num)<4) and (int(num)>0))]

ng_idxs = list(set([v[0] for v in alert_dict]))
edge_df.loc[ng_idxs,'Judge'] = 'NG'
for ix in ng_idxs:
    alerts = "\n".join(list(set([v[1] for v in alert_dict if v[0]==ix])))
    print(f'{ix}:{alerts}')
    edge_df['Alert'][ix] = alerts
edge_df.loc[[ix for ix in edge_df.index if ix not in ng_idxs],'Judge'] = 'OK'

# %%
'''
exists_marks = []
markfolders = []
for date in ['20241004','20241007']:
    datefolder = f"{objdetet_output_root}/{date}"
    print(datefolder)
    for mark in [mark for mark in os.listdir(datefolder) if mark not in exists_marks]:
        print(f'{mark}-->')
        exists_marks.append(mark )
        mark_folder = f'{datefolder}/{mark}'
        mark_edge = [f for f in os.listdir(mark_folder) if '.csv' in f][0]
        if date != mark_edge[:8]:
            new_date = mark_edge[:8]
            newdate_folder = f'{objdetet_output_root}/{new_date}'
            if not os.path.exists(newdate_folder):
                shutil.move(mark_folder,newdate_folder)
            markfolders.append(newdate_folder)
        markfolders.append(mark_folder)
'''

# %%
#dist exsite data 
'''
edge_df = []
for date in [date for date in os.listdir(objdetet_output_root) if '202410' in date]:
    datefolder = f"{objdetet_output_root}/{date}"
    print(f'{date}----------------------')
    for mark in os.listdir(datefolder):
        mark_folder = f'{datefolder}/{mark}'
        print(f'{mark_folder}-->')
        model = mark.split("_")[-1]
        for f in [f for f in glob.glob(os.path.join(mark_folder, "*")) if '.txt' in f ]:
            print(f)
            info = rowdata_clean(f)
            file_time = info[-1]
            rls = model_rules.loc[(model_rules['abbrename'] ==model) & (model_rules['MC'].str.contains(info[0][2::])) ]
           
            #
            if len(rls) ==0:
                continue
        
            info = [model,mark] + info + [info[-1][4:8]]
            start_no = int(rls['start_no'][rls.index[0]])
            oriimg_folder = os.path.join(mark_folder,"Image")

            processing_folder = f'{mark_folder}/processing'
            os.makedirs(processing_folder , exist_ok=True)
            processing_df = pre_image_processing(glob.glob(os.path.join(oriimg_folder , "*")),processing_folder)
            detect_outputfolder = f'{mark_folder}/detect_output'
            label_folder = os.path.join(detect_outputfolder, 'exp/labels')  
            if not os.path.exists(label_folder):
                detect_result = obj_detect(processing_folder,detect_outputfolder)
            else:
                print(f'{mark} already detect')

            relables_folder = f'{mark_folder}/backup_output'
            edges = process_labels_and_draw_boxes(processing_df, label_folder, processing_folder, relables_folder)
            for cl,value in zip(info_cols,info ):
                edges[cl] = value
            edges.to_csv(f'{mark_folder}/{file_time}_{mark}_edges(px).csv')
    
            if len(edges) >0 and len(edge_df)>0:
                edge_df = pd.concat([edge_df, edges])
            elif len(edges) >0:
                edge_df = edges
            else:
                edge_df = pd.DataFrame([[None]*len(edges.columns) + info],columns = edges.columns.tolist() + info_cols)
            #
            webmark_folder = os.path.join(detect_results_root,mark)
            if os.path.exists(webmark_folder):
                continue
            shutil.copytree(relables_folder,webmark_folder)
'''


# %%
print(len(edge_df))
edge_df.drop_duplicates(subset=['DateTime','DateTime','Mark','No'],inplace = True)
print(len(edge_df))

# %%

del_charts = [ '(','-','.']
mask_title = ['Model','PEP', 'Mask Reticle Name','Mask Set ID','Code', 'Tool']
rcp_columns = ['MainRCP','RCPMarkingIndex','chip','MarkingFrameXPositioninGlass','MarkingFrameYPositioninGlass']
RN = 'RecipeName = '
for cl in rcp_columns:
    edge_df[cl] = None

for iex in edge_df['IEX'].unique():
    iex_df = edge_df[edge_df['IEX']==iex]
    iex_char = iex[5] if is_number(iex[5]) else iex[5].lower()
    orc_folder = f'{ORC_recipe_root}/aaiex{iex_char}20/Main'
    edg_lst = [f for f in os.listdir(orc_folder) if f[:3] =='RCP' ]
    iexdf = iex_df[iex_df['IEX']==iex]
    print(f'{iex}--->')
    for model in iexdf['Model'].unique():
        mdf = iexdf[iexdf['Model'] ==model]
        key = [v for v in mdf['model_key'].unique() if v is not None][0]
        print(f'{model} ->{key}')
        
        runchart_data = {sheet: runchart_file.parse(sheet) for sheet in runchart_file.sheet_names if model in sheet or key ==sheet}
        ppids = runchart_tab_clean_ppids(runchart_data)
        
        for ppid in ppids:
            rcp_fn = f'RCP{ppid}.dat'
            if rcp_fn not in os.listdir(orc_folder):
                rcp_fn = None
                continue
            path = f'{orc_folder}/{rcp_fn}'
            
            content = encode_char_read_content(path)
            fndata = [v.replace("\t"," ") for v in content.split("\n")]
            recipe_info = [[ix,v] for ix ,v  in zip(range(len(fndata)),fndata) if RN in v and 'P1' in v and key.split("_")[0][:3] in v.split(RN)[-1] ]
            if len(recipe_info) !=0:
                recipe_model = recipe_info[0][-1][16:-1]
                startix = recipe_info[0][0]
                print(f'match RCP recipe name : \n{recipe_model}')
                print(path)
                touser_path = f'{towebRCP_folder}/{rcp_fn}'
                shutil.copy2(path,touser_path)

                chipdatas = [[ix,v.split(" = ")[-1][1:-1] if v.split(" = ")[-1][1:-1] !='' else v.split(" = ")[-1] if is_number(v.split(" = ")[-1]) else '',\
                              fndata[ix+1][32:],fndata[ix+2][32:]] for ix,v in zip(range(len(fndata)),fndata) if 'MarkingFrameCutorCellNoIndex = ' in v ]
                chips = [v[1] for v in chipdatas]
                chiplst = []
                for chip in chips:
                    if chip not in chiplst or chip =='' :
                        chiplst.append(chip)

                '''
                if len([num for num in Counter(chiplst).values() if num>1]) > 3:
                    uni_code = 'T'
                    chiplst2 = []
                    for chip in chips:
                        if chip not in chiplst2 or chip =='' :
                            chiplst2.append(chip)
                else:
                    print(chiplst)
                    uni_code = 'F'
                '''
                break
            else:
                rcp_fn = None
        chip_dict = pd.DataFrame(chipdatas, columns = ['index','chip','XP','YP'])
        #num_chips = len(chiplst)
        num_chips = len(chiplst)
        #
        print(f'{model}:({ppid})->{chiplst}')
        
        uni_marks = mdf['Mark'].unique()
        for mark in uni_marks:
            mark_df = mdf[mdf['Mark'] == mark]
            print(f'{mark}',mark_df['DateTime'].unique())
            num_imgs = len(mark_df)
            nolst = edge_df[edge_df['Mark']==mark]['No'].tolist()
            #if uni_code == 'T':# and len([ix for ix in mark_df if ix in only_code_df.index]) !=0 :
                #chiplst = chiplst2
            

            for num in range(0,10,2):
                if (num_imgs-num)%num_chips ==0:
                    chip_count = int( (num_imgs-num)/num_chips)  
                    break
            print(f'{mark}:{num_imgs}->{chip_count}')
            for ix,no,num in zip(mark_df.index,mark_df['No'],range(len(mark_df))):
                #print(ix,no,num)
                chip_num = math.floor(num/chip_count)
                chipdata = chip_dict.loc[chip_num].tolist()
                #print(f'chip:{chip}->{chipdata}')
                #for cl in rcp_columns[1:]:

                edge_df.loc[ix,rcp_columns[1:]] = chipdata
                #print(edge_df.loc[ix,rcp_columns[1:]])
        edge_df.loc[mdf.index,'MainRCP'] = rcp_fn
        #print(edge_df.loc[mdf.index,rcp_columns])
        

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
edge_df.head(10)

# %%
#print(edge_df.columns)
recolumns = ['Model', 'Mark', 'MQCTool',
       'IEX', 'Lot', 'Slot', 'DateTime', 'Date','chip','No','Image name','Judge', 'Alert','cut_image_height',
       'cut_image_width', 'x_limit1', 'x_limit2', 'y_limit1', 'y_limit2',
       'rotate', 'MinX', 'MinY', 'MaxX', 'MaxY','code_edge', 'Num_ID',
       'code_height', 'Num_Dot',  'px_per_um','MainRCP','model_key','chips','MarkingFrameXPositioninGlass','MarkingFrameYPositioninGlass'] 
recolumns = recolumns + [cl for cl in edge_df.columns if cl not in recolumns]
edge_df = edge_df[recolumns]
edge_df['Date'] = edge_df['Date'].apply(lambda x: "%s/%s"%(x[:2],x[2:]))
edge_savepath = f'{objdetet_output_root}/backup_edge_result/{now_time}_edge_df.csv'
if os.path.exists(edge_savepath):
       edge_df.to_csv(edge_savepath, mode='a', index=False, header=False)
else:
       edge_df.to_csv(edge_savepath,index=False)

edge_df.head(1)

# %%

#sqldb_connect('save','edge_table','chip_id_detector_result',edge_df)

# %%
#df = sqldb_connect('get_sql_table','edge_table','chip_id_detector_result','')
#df['Date'] = df['Date'].apply(lambda x: "%s/%s"%(x[:2],x[2:]))
#df.head(3)

# %%
for cl in ['USER','UserUpdateTime','Note']:
    edge_df[cl] = None

# %%
his_edge = sqldb_connect('get_sql_table','edge_table','chip_id_detector_result','')
num_his = len(his_edge)
if len(edge_df) !=0:
    num_edges = len(edge_df)
    print(f'new table :{num_edges}')
    if len(his_edge) !=0 and len(edge_df) !=0:
        print(f'his data row:{num_his}')
        edge_df = pd.concat([his_edge,edge_df])
        
    edge_df.sort_values(by = ['DateTime','Mark','No'],ascending=False,inplace = True)
    edge_df.drop_duplicates(subset=['DateTime','Mark','No'],inplace = True)
    edge_df.reset_index(inplace = True,drop = True)
    edge_df['index'] = edge_df.index.tolist()
    num_edges = len(edge_df)
    print(f'clean-> {num_edges}')

# %%
sqldb_connect('save','edge_table','chip_id_detector_result',edge_df)
# %%
edge_df["date_sort"] = pd.to_datetime(edge_df['DateTime'], format='%Y%m%d%H%M%S', errors='coerce')
now_time = time.strftime("%Y%m%d", time.localtime())
now_time_dt = datetime.strptime(now_time, "%Y%m%d")

one_week_delta = timedelta(weeks=1)
filtered_df = edge_df[(edge_df["date_sort"] >= now_time_dt - one_week_delta) & (edge_df["date_sort"] <= now_time_dt + one_week_delta)]
print(f'one week data:{len(filtered_df)}')
filtered_df['hrel'] = [f'{project_http}/detect_results/{mark}/{fn}' for mark,fn in zip(filtered_df['Mark'],filtered_df['Image name'])]
filtered_df.to_csv(f'{eda_df_root}/chipidTOEDA.csv',index=False)

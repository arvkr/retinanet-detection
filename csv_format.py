import pandas as pd
import ast

def change_format(left_x,top_y,width,height):
    anno = {}
    anno['x1'] = int(left_x)
    anno['y1'] = int(top_y)
    anno['x2'] = int(left_x + width)
    anno['y2'] = int(top_y + height)

    return anno

orig_ann = pd.read_csv('./orig_anno.csv')
new_ann_list = []
for i in range(206):
    ann_dict = {}
    ann_dict['file_name'] = './dataset/clean_resized/' + orig_ann.iloc[i,0]
    lx = ast.literal_eval(orig_ann.iloc[i,5])['x']
    ty = ast.literal_eval(orig_ann.iloc[i,5])['y']
    w = ast.literal_eval(orig_ann.iloc[i,5])['width']
    h = ast.literal_eval(orig_ann.iloc[i,5])['height']
    anno = change_format(lx,ty,w,h)
    ann_dict['x1'] = anno['x1']
    ann_dict['y1'] = anno['y1']
    ann_dict['x2'] = anno['x2']
    ann_dict['y2'] = anno['y2']
    ann_dict['body_part'] = ast.literal_eval(orig_ann.iloc[i,6])['body_part']
    new_ann_list.append(ann_dict)

new_ann = pd.DataFrame(new_ann_list)
new_ann.to_csv('csv_annotations.csv', index = False)
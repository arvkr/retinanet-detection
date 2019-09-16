def change_format(centre_x,centre_y,width,height):
    anno = {}
    anno['x1'] = int(centre_x - width/2)
    anno['y1'] = int(centre_y - height/2)
    anno['x2'] = int(centre_x + width/2)
    anno['y2'] = int(centre_y + height/2)

    return anno

cx = 278
cy = 121
w = 61
h = 32
anno = change_format(cx,cy,w,h)
print(anno)

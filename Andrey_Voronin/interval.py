def get_interval(frames, fps):
    interval=[]
    temp=[] 
    for i, frame in enumerate (frames):
        if i==0:
            temp.append(frame)
        elif (frame-temp[-1])<(fps*2):
            temp.append(frame)
        else:
            if len(temp)>3:
                interval.append(interval_to_time(temp[0],temp[-1],fps))
            temp.clear()
            temp.append(frame)


    if len(temp)>3:
        interval.append(interval_to_time(temp[0],temp[-1],fps)) #доп условие
    return interval

def interval_to_time(start,stop,fps):
    return (f'{start/fps}сек.', f'{stop/fps}сек.')


# t=[11, 12, 13, 14, 17, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74, 82, 83, 93, 95, 96, 97, 111, 112, 113]
# print(get_interval(t, 25))
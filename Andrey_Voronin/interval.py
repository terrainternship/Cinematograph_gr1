def get_interval(frames, fps):
    interval=[]
    temp=[] 
    for i, frame in enumerate (frames):
        if i==0:
            temp.append(frame)
        elif frame-temp[-1]<(fps*2):
            temp.append(frame)
        else:
            if len(temp)>5:
                interval.append(interval_to_time(temp[0],temp[-1]),fps)
                temp.append(frame)

    if len(temp)>5:
        interval.append(interval_to_time(temp[0],temp[-1],fps)) #доп условие
    return interval

def interval_to_time(start,stop,fps):
    return (f'{start/fps}сек.', f'{stop/fps}сек.')
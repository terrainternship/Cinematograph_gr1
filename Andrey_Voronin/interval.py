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
    time_start = ['']*5
    time_start[0] = start/fps
    time_start[1] = str(int(time_start[0]//3600%60)).zfill(2)
    time_start[2] = str(int(time_start[0]//60%60)).zfill(2)
    time_start[3] = str(int(time_start[0]%60)).zfill(2)
    time_start[4] = str(int(time_start[0]%1*100)).zfill(2)

    time_stop = ['']*5
    time_stop[0] = stop/fps
    time_stop[1] = str(int(time_stop[0]//3600%60)).zfill(2)
    time_stop[2] = str(int(time_stop[0]//60%60)).zfill(2)
    time_stop[3] = str(int(time_stop[0]%60)).zfill(2)
    time_stop[4] = str(int(time_stop[0]%1*100)).zfill(2)

    return (f'{time_start[1]}:{time_start[2]}:{time_start[3]}:{time_start[4]}',
            f'{time_stop[1]}:{time_stop[2]}:{time_stop[3]}:{time_stop[4]}')


# t=[11, 12, 13, 14, 17, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74, 82, 83, 93, 95, 96, 97, 111, 112, 113,912001,912002,912003,912004,912005,912006,912007]
# print(get_interval(t, 25))
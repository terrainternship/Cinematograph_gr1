import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
from glob import glob 
import cv2
import datetime
from ultralytics import YOLO
from interval import *




def on_predict_postprocess_end(predictor):
    '''
    CallBack для предикта.
    Позволяет получить данные для отображения прогресса предикта
    '''
    inf=predictor.batch[3].split()
    file = inf[3]
    file = file[file.rfind('\\')+1:]
    inf=inf[2][1:-1].split('/')
    cur_frame=int(inf[0])
    all_fame=int(inf[1])
    progress_file['maximum'] = all_fame
    progress_file['value'] = cur_frame
    lbl_state['text']=f"Обработка: {file} текущий фрэйм {cur_frame}/{all_fame}, Общий прогресс {progress['value']}%"
    window.update()



def Track(file):
    '''
    Предикт модели
    out - список кадров, на которых обнаружена сигарета
    '''
    print(float(sca.get()))
    rez=model.predict(file, save=False, conf=float(sca.get()), stream=True,  verbose=False)
    out =[i+1 for i,r in enumerate (rez) if len(r.boxes.cls)>0]
    progress_file.stop()
    return out


def open_file():
    error=0
    txt_edit.delete('1.0', tk.END)
    start_dir = filedialog.askdirectory()
    files = []
    pattern   = "*.[ma][pvo][4iv]"
    lbl_state['text']='Поиск файлов...'
    window.update()
    for dir,_,_ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir,pattern)))
    
    all=len(files)
    all_size=0

    for i,file in enumerate(files):
        all_size += os.path.getsize(file)


    Lbl_info['text']=f'Найдено файлов: {all} \nОбщий объем {all_size/(2**30):.2f}Гб'
    window.update()

    for i,file in enumerate(files):
        try:
            video = cv2.VideoCapture(file) 
            fps = video.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video 
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the video 
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the video
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            size = os.path.getsize(file)/(2**20)
            txt_edit.insert(tk.END, f'{"="*80}\n')
            txt_edit.insert(tk.END, f'{file:15}\n')
            txt_edit.insert(tk.END, f'{"Разрешение":^16}{"Частота кадров":>15}{"Всего кадров":>15}{"Длительность":>17}{"Размер":>13}\n')
            txt_edit.insert(tk.END,f"{width:>8}×{height:<8}")
            txt_edit.insert(tk.END,f"{fps:>14.2f}")
            txt_edit.insert(tk.END,f"{frame_count:>15}")
            txt_edit.insert(tk.END,f"{frame_count/fps:>13.3f}сек.")
            txt_edit.insert(tk.END,f"{size:>11.2f}Мб\n")


            progress_file.start()
            result=Track(file) # Вызов функции Track
            txt_edit.insert(tk.END,f"Временные метки начала и окончания сцен курения с сигаретами:\n")
            txt_edit.insert(tk.END,f"{get_interval(result,fps,skip=int(spin.get()))}\n")
            progress_file.stop()
        except:
            error+=1
        
        

        txt_edit.yview_moveto('1.0')
        window.update()
        progress['value'] = round((i+1)*100/all)

    progress['value'] = 100
    lbl_state['text']=f"Обработка завершена. Обработано файлов - {all-error}"
    
    
    file_name = datetime.datetime.today().strftime("%Y-%m-%d-%H.%M.%S")

    with open(f'{start_dir}/{file_name}.txt', "w", encoding="utf-8") as output_file:
        text = txt_edit.get("1.0", tk.END)
        output_file.write(text)
    
    txt_edit.insert(tk.END,f"\nЛог сохранен в файл: {f'{start_dir}/{file_name}.txt'}\n")
    txt_edit.yview_moveto('1.0')

def save_file():
    """Сохраняем текущий файл как новый файл."""
    filepath = filedialog.asksaveasfilename(
        defaultextension="txt",
        filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
    )
    if not filepath:
        return
    with open(filepath, "w", encoding="utf-8") as output_file:
        text = txt_edit.get("1.0", tk.END)
        output_file.write(text)
    window.title(f"Smoke Detect - {filepath}")


DIR_PATH = os.path.dirname(os.path.realpath(__file__))




dir_path = f'{DIR_PATH}\\yolo_weights\\*.pt'
yolo_model = ','.join(glob(dir_path))


# Load the model.
model = YOLO(yolo_model)
# model.to('cuda')
model.add_callback("on_predict_postprocess_end", on_predict_postprocess_end) 

window = tk.Tk()

skip = tk.IntVar(value=5)
exact = tk.DoubleVar(value=0.25)

window.title("Детектор сцен курения на видео v0.4")
window.iconbitmap('ico/logo.ico')
window.rowconfigure(0, minsize=50, weight=0)
window.rowconfigure(1, minsize=100, weight=1)
window.columnconfigure(0, minsize=200, weight=1)

txt_edit = tk.Text(window)

open_btn_image = tk.PhotoImage(file='ico/open.png')
save_btn_image = tk.PhotoImage(file='ico/save.png')

fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=0)
btn_open = tk.Button(fr_buttons, image=open_btn_image, text="Открыть", bd=0, command=open_file)
btn_save = tk.Button(fr_buttons, image=save_btn_image, text="Сохранить как...", bd=0, command=save_file)
Lbl_info = tk.Label(fr_buttons,text="")
sca = tk.Scale(fr_buttons, label='Порог обнаружения:', orient="horizontal", length=300, from_=0, to=1, tickinterval=0.1, resolution=0.01, variable=exact)
Lbl_spin = tk.Label(fr_buttons,text="Интервал:")
spin = tk.Spinbox(fr_buttons,  from_=0, to=60, width=5, textvariable=skip)  




fr_buttons.columnconfigure(4, minsize=200, weight=1)
btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=0, column=1, sticky="ew", padx=5)
Lbl_info.grid(row=0, column=4, sticky="e", padx=5)
Lbl_spin.grid(row=0, column=2, sticky="en", padx=0)
spin.grid(row=0, column=2, sticky="ew", padx=5)
sca.grid(row=0, column=3, sticky="ew", padx=5)





fr_buttons.grid(row=0, column=0, sticky="ew")
txt_edit.grid(row=1, column=0, sticky="nsew")

scroll = tk.Scrollbar(command=txt_edit.yview)
scroll.grid(row=1, column=1, sticky="ns")
txt_edit.config(yscrollcommand=scroll.set)

lbl_state=tk.Label(text='Состояние')
lbl_state.grid(row=2,column=0,sticky="w", padx=10)



s = ttk.Style()
s.theme_use('alt')
s.configure("file.Horizontal.TProgressbar", background='lime')



progress_file = ttk.Progressbar(orient="horizontal",  length=100, value=0, style='file.Horizontal.TProgressbar')
progress_file.grid(row=3,column=0,sticky="ew")

progress = ttk.Progressbar(orient="horizontal", length=100, value=0)
progress.grid(row=4,column=0,sticky="ew")
window.mainloop()
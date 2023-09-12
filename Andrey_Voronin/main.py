import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
from glob import glob 
import cv2
import datetime 


from track import Track # Библиотека детектор
from interval import *

def open_file():
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
        lbl_state['text']=f"Обработка: {file} {progress['value']:20}%"
        window.update()
        
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

            result=Track(file) # Вызов функции Track
            print(result)
            print(get_interval(result,fps))
            txt_edit.insert(tk.END,f"{get_interval(result,fps)}\n")
        except:
            pass
        
        

        txt_edit.yview_moveto('1.0')
        window.update()
        progress['value'] = round((i+1)*100/all)

    progress['value'] = 100
    lbl_state['text']=f"Обработка завершена. Обработано файлов - {all}"
    
    # dir_path = os.path.dirname(os.path.realpath(__file__))
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
 



window = tk.Tk()
window.title("Smoke Detect")
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
 
fr_buttons.columnconfigure(2, minsize=200, weight=1)
btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=0, column=1, sticky="ew", padx=5)
Lbl_info.grid(row=0, column=2, sticky="e", padx=5)

fr_buttons.grid(row=0, column=0, sticky="ew")
txt_edit.grid(row=1, column=0, sticky="nsew")

scroll = tk.Scrollbar(command=txt_edit.yview)
scroll.grid(row=1, column=1, sticky="ns")
txt_edit.config(yscrollcommand=scroll.set)

lbl_state=tk.Label(text='Состояние')
lbl_state.grid(row=2,column=0,sticky="w", padx=10)


# progress_file = ttk.Progressbar(orient="horizontal",  mode="indeterminate", length=100, value=0)
# progress_file.grid(row=3,column=0,sticky="ew")

progress = ttk.Progressbar(orient="horizontal", length=100, value=0)
progress.grid(row=4,column=0,sticky="ew")

window.mainloop()
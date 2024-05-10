from tkinter import *
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import chardet
import numpy as np
import pandas as pd
import csv
import os
import math

def check_sample_type():
    sample_type = sample_var.get()  # Получаем выбранный тип образца
    if sample_type == "Цилиндр":
        diameter_label.grid(row=4, column=1)
        entry_diameter.grid(row=4, column=2)
    else:
        diameter_label.grid_forget()
        entry_diameter.grid_forget()   

def tens():
    tens_type = tens_var.get()
    if tens_type == 'Напряжения':
        num_tens1.grid(row=8, column=1)
        num_tens1_entry.grid(row=8, column=2)
        num_stress.grid_forget()
        num_stress_entry.grid_forget()
        rab_sech_label.grid_forget()
        rab_sech_entry.grid_forget()
        n_kn_label.grid_forget()
        n_kn_entry.grid_forget()
    else:
        num_stress.grid(row=8, column=1) 
        num_stress_entry.grid(row=8, column=2) 
        rab_sech_label.grid(row=9, column=1)
        rab_sech_entry.grid(row=9, column=2)
        n_kn_label.grid(row=10, column=1)
        n_kn_entry.grid(row=10, column=2)  
        num_tens1.grid_forget()
        num_tens1_entry.grid_forget()

def choose_elong():
    type_elong = elong.get()
    if type_elong == 'д':
        nach_rass_label.grid(row=13, column=1)
        nach_rass_entry.grid(row=13, column=2)
        koef_volt.grid_forget()
        koef_volt_entry.grid_forget()
    elif type_elong == 'в':
        koef_volt.grid(row=13, column=1)
        koef_volt_entry.grid(row=13, column=2)
        nach_rass_label.grid_forget()
        nach_rass_entry.grid_forget() 
    else:
        nach_rass_label.grid_forget()
        nach_rass_entry.grid_forget() 
        koef_volt.grid_forget()
        koef_volt_entry.grid_forget()   

# Функция для обработки нажатия кнопки "Выбрать файл"
def choose_file():
    file_path = filedialog.askopenfilename()
    entry_file_path.delete(0, END)  # Очистить поле ввода
    entry_file_path.insert(0, file_path)  # Вставить выбранный файл в поле ввода

# Функция для обработки нажатия кнопки "Запустить программу"
def run_program():
    global data_elong, data_tens, markirovka, type_sample, young_mod, length_work, num_tens, type_elong, num_elong, predel_teck, rab_sech, n_kn
    file_path = entry_file_path.get()
    with open(file_path, 'rb') as f:
        dt = f.read()
        encoding = chardet.detect(dt)['encoding'] 
    try:
        if '.csv' in file_path or '.TRA' in file_path or '.tra' in file_path: #or '.dat' in file_path:
            data = pd.read_csv(file_path, sep=None, encoding=encoding, engine='python')
            data = data.apply(pd.to_numeric, errors='coerce').dropna()       
        elif '.xls' in file_path:
            data = pd.read_excel(file_path, sheet_name=sheet_name.get()) 
            data = data.apply(pd.to_numeric, errors='coerce')#.dropna()  
        elif '.dat' in file_path:
            #data = pd.read_table(file_path, delimiter='\s+', encoding=encoding, engine='python') # delimiter='\s+'  
            #data = data.apply(pd.to_numeric, errors='coerce') 
            directory = os.path.dirname(file_path)
            new_file_path = os.path.join(directory, 'new_file.dat')
            with open(file_path) as f, open(new_file_path, 'w') as output_file:    
                for line in f:
                    if not any(char.isalpha() for char in line):
                        cleaned_line = ' '.join(line.split())
                        output_file.write(cleaned_line + '\n')
            data = pd.read_table(new_file_path, sep=None, encoding=encoding, engine='python', header=None)
            data = data.apply(pd.to_numeric, errors='coerce') 
            os.remove(new_file_path)     
        else:
            #data = pd.read_fwf(file_path, delimiter=None, encoding=encoding, engine='python')
            data = pd.read_table(file_path, sep=None, encoding=encoding, engine='python')
            data = data.apply(pd.to_numeric, errors='coerce')
    except:        
        messagebox.showinfo('Ошибка', 'Что-то пошло не так, проверьте входные данные из файла. Примечание: Программа работает только с корректными разделителями файлов формата .dat') 
          
    
    markirovka = str(entry_mark.get())
    type_sample = str(sample_var.get())
    young_mod = float(young_mod_entry.get())
    length_work = float(length_work_entry.get()) 
    num_tens = num_tens1_entry.get() or num_stress_entry.get()
    type_elong = elong.get()
    num_elong = int(num_elong_entry.get())
    predel_teck = float(predel_teck_entry.get())
    if '.xls' in file_path or '.dat' in file_path:
        num_tens = int(num_tens) - 1
    else:
        num_tens = int(num_tens) - 2
    n_kn = str(n_kn_entry.get())
    if n_kn == 'н' or n_kn == 'Н':
        rab_sech = float(rab_sech_entry.get())
        data.iloc[:, num_tens] = ((data.iloc[:, num_tens] / 1000) / rab_sech) * 1000
    elif n_kn == 'кн' or n_kn == 'кН':
        rab_sech = float(rab_sech_entry.get())
        data.iloc[:, num_tens] = ((data.iloc[:, num_tens] / rab_sech) * 1000)
    else:
        rab_sech = float(rab_sech_entry.get())
        data.iloc[:, num_tens] = ((data.iloc[:, num_tens] / rab_sech) * 1000) * float(n_kn)  
    
    if '.xls' in file_path or '.dat' in file_path:
        num_elong = int(num_elong) - 1
    else:
        num_elong = int(num_elong) - 2

    if type_elong == 'Удлинение':   
        data.iloc[:, num_elong] = (data.iloc[:, num_elong] / 100)
    elif type_elong == 'т':
        data.iloc[:, num_elong] = (data.iloc[:, num_elong] / length_work)#.abs()
    elif type_elong == 'в':
        data.iloc[:, num_elong] = (data.iloc[:, num_elong] / length_work) * float(koef_volt_entry.get())  
    else:
        nach_rass = float(nach_rass_entry.get())
        data.iloc[:, num_elong] = (data.iloc[:, num_elong] / nach_rass)

    data_elong = np.array(data.iloc[:, num_elong])
    if data_elong[0] > 0:
        delta = data_elong[0] - 0
    else:
        delta = 0 - data_elong[0]
    data_elong = np.add(data_elong, delta)
    data_tens = np.array(data.iloc[:, num_tens]) 
    mask = np.isreal(data_elong)
    data_elong = data_elong[mask]
    data_tens = data_tens[mask] 
    data_elong = data_elong[data_tens > 0]
    data_tens = data_tens[data_tens > 0] 
    plt.xlabel("Удлинение")
    plt.ylabel("Напряжение, МПа")
    plt.plot(data_elong, data_tens, 'b')
    #plt.plot(data.iloc[:, num_elong], data.iloc[:, num_tens], 'b')
    plt.title('График напряжения-удлинения по данным из файла')
    plt.show()

def smoothed(): 
    global data_elong, data_tens, res_x, res_y, young_mod, predel_teck, ln_res, exp_res, koef_dinam   
    x = data_elong
    y = data_tens
    if smooth.get() == 'д':
        window_size = 15
        smoothed_y = np.copy(y)
        for _ in range(10):
            for i in range(15, len(y)-1):
                smoothed_y[i] = np.mean(y[i-window_size:i+window_size])  
        plt.plot(x, smoothed_y, 'k')
        plt.show() 
        data_tens = smoothed_y
        y = data_tens 
    
    x = data_elong[:len(y) // 2]#[y <= predel_teck]
    y = data_tens[:len(y) // 2]

    y1 = 0.4 * predel_teck
    y2 = 0.7 * predel_teck
    indices = np.where((y >= y1) & (y <= y2))
    x1 = x[indices[0][0]]
    x2 = x[indices[0][-1]]

    dx1 = x2 - x1
    dy1 = y2 - y1

    a = (y2 - y1) / (x2 - x1)  # наклон прямой
    b = y1 - a * x1  # смещение прямой по вертикали

    x_pred = (y[-1] + b) / a  
    x_intersect = 0#-b / a
    y_intersect = 0  
    # Создаем массив точек для построения луча
    x_plt = [x1, x1 - 2*dx1, x2, x2 + 2*dx1]
    y_plt = [y1, y1 - 2*dy1, y2, y2 + 2*dy1]
    koef_dinam = float(koef_entry.get())
    data_tens = data_tens * koef_dinam
    #ВТОРАЯ ДИАГРАММА
    c = (x2 - (y2 / young_mod)) / y2
    arr_x = np.array(data_elong)
    arr_y = np.array(data_tens)
    index = np.where(arr_y >= y2)[0][0] #индекс точки 0.7 предела текучести
    recalculated =np.array(arr_x - (c * arr_y))
    x_kon = recalculated[len(recalculated) - 1] # конечное значение х второго графика
    recal_x = recalculated[index] #координата х 0.7предела текучести
    recal_y = arr_y[index] #координата у 0.7предела текучести
    recalculated_x = recalculated[index:] #координаты х от 0.7предела текучести и до конца
    recalculated_y = arr_y[index:] #координаты у 0.7 предела текучести и до конца
    index_max = np.argmax(recalculated_y)#indices[-1]#np.argmax(recalculated_y)
    xy = recalculated_x[index_max] #значение х при у макс
    x_after = recalculated_x[index_max:]# часть графика после максимума           
    y_after = recalculated_y[index_max:]
    elong_procent = float(elong_procent_entry.get()) / 100
    delta = elong_procent - x_kon
    x_calc = x_after + delta
    result_x = np.concatenate((recalculated_x[:index_max], x_calc))
    result_x = np.insert(result_x, 0, recal_x)

    result_x = np.insert(result_x, 0, x_intersect)

    result_y = np.concatenate((recalculated_y[:index_max], y_after))
    result_y = np.insert(result_y, 0, recal_y)

    result_y = np.insert(result_y, 0, y_intersect)
    res_x = result_x
    res_y = result_y
    #ТРЕТЬЯ ДИАГРАММА - ИСТИННАЯ
    ln_result_x = np.log(result_x + 1)
    exp_result_y = result_y * np.exp(ln_result_x)

    plt.plot(ln_result_x, exp_result_y, 'y') #- это 
    plt.xlabel('Удлинение')
    plt.ylabel('Напряжения, МПа')
    plt.title('Истинная диаграмма')
    plt.show()
    ln_res = ln_result_x
    exp_res = exp_result_y


def program():
    global res_x, res_y, ln_res, exp_res, A_entry, B, ind_max, after_max, exp_y, C
    A = float(A_entry.get())
    C = float(C_entry.get())
    ln_result_x = ln_res
    exp_result_y = exp_res
    #аппроксимация после максимума
    index_max2 = np.argmax(res_y) #np.argmax(exp_result_y)
    max_x2 = res_x[index_max2] #ln_result_x[index_max2] #коэф В, равный х в максимуме на 2 диаграмме
    after_max_x = ln_result_x[index_max2 - 300:]
    after_max_y = exp_result_y[index_max2 - 300:]   
    exp_y_recal = A * ((float(young_mod_entry.get()) * (after_max_x + C)) / A)**max_x2
    plt.plot(ln_result_x, exp_result_y, 'y', label='Истинная')
    plt.plot(after_max_x, exp_y_recal, 'k', label='Аппроксимация')
    plt.legend()
    plt.xlabel('Удлинение')
    plt.ylabel('Напряжения, МПа')
    plt.show()
    B = max_x2
    ind_max = index_max2
    after_max = after_max_x[300:]
    exp_y = exp_y_recal[300:]


def wrapper():
    global ln_res, exp_res, ind_max, after_max, exp_y
    ln_res_1 = np.concatenate((ln_res[:ind_max], after_max))
    exp_res_1 = np.concatenate((exp_res[:ind_max], exp_y))
    #ln_res, exp_res = ln_result_x, exp_result_y
    plt.plot(ln_res_1, exp_res_1, 'y', label='Истинная')
    plt.xlabel('Удлинение')
    plt.ylabel('Напряжение, МПа')
    plt.show()
    

def program_choose():
    global markirovka, ln_res, exp_res, B, A_entry, C, after_max, exp_y, approc_x, approc_y, ind_max
    ln_result_x, exp_result_y = ln_res[:ind_max + 1], exp_res[:ind_max + 1]
    A = float(A_entry.get())
    C = float(C_entry.get())

    plt.figure(figsize=(10, 8), dpi=100)
    plt.plot(ln_result_x, exp_result_y)
    plt.plot(after_max, exp_y, 'k')
    plt.xlabel("Удлинение")
    plt.ylabel("Напряжение, МПа")
    plt.title('Выберите точки на синем графике (точка касания с апроксимацией не включается).\nПравый клик для отмены предыдущей точки. \nКолесо мышки для масштабирования. \nEnter для завершения')

    selected_points = []
    def on_scroll(event):
        zoom_factor = 1.1
        if event.button == 'up':
            zoom_factor = 0.9
        
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    
        xdata = event.xdata
        ydata = event.ydata
    
        new_xlim = [xdata - (xdata - xlim[0]) * zoom_factor, xdata + (xlim[1] - xdata) * zoom_factor]
        new_ylim = [ydata - (ydata - ylim[0]) * zoom_factor, ydata + (ylim[1] - ydata) * zoom_factor]
    
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        plt.draw()

    def onclick(event):
        if event.button == 3 and selected_points:
            plt.plot(selected_points[-1][0], selected_points[-1][1], color='white', marker='o')
            plt.show()
            selected_points.pop()



    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.gcf().canvas.mpl_connect('scroll_event', on_scroll)


# Получаем координаты выбранных точек по клику
    selected_points = []#plt.ginput(n=12, show_clicks=True, timeout=0)
    all_points = [(x, y) for x, y in zip(ln_result_x, exp_result_y)]
    while True:
        point = plt.ginput(n=1, show_clicks=True, timeout=0)
        if not point: # если пользователь нажал клавишу для завершения
            break
        #distances = np.abs(ln_result_x - point[0][0])
        distances = [np.sqrt((point[0][0] - point_[0])**2 + (point[0][1] - point_[1])**2) for point_ in all_points]
        min_dist_idx = np.argmin(distances)
        closest_point = all_points[min_dist_idx]
        selected_points.append(closest_point)
        plt.plot(closest_point[0], closest_point[1], 'ro')
        #plt.plot(point[0][0], point[0][1], 'ro')
    plt.show()

    approc_x = np.linspace(after_max[-1], 3.0, 300)
    approc_y = A * ((float(young_mod_entry.get()) * (approc_x + C)) / A)**B
    approc_x = np.concatenate((after_max[:-1], approc_x))
    approc_y = np.concatenate((exp_y[:-1], approc_y))
    points1 = np.linspace(0, np.where(approc_x < 3.0*B)[0][-1], 20 - len(selected_points) - 4, dtype=int)
    points2 = np.linspace(np.where(approc_x >= 3.0*B)[0][0], len(approc_x) - 1, 4, dtype=int)
    points = np.append(points1, points2)
    for i in points:
        selected_points.append((approc_x[i], approc_y[i]))
    print("Выбранные точки:")
    for i, point in enumerate(selected_points, 1):
        print(f'x{i}={point[0]}, y{i}={point[1]}')
    data_before_headers = [[f'Маркировка - {markirovka}'], [f'Коэффициент А = {A}'], [f'Коэффициент В = {B}'], [f'Коэффициент C = {C}']]

    columns = ['Истинное х', 'Истинное у', 'Х', 'Y']
    folder_file = filedialog.askdirectory()
    name = f'{folder_file}/{name_entry.get()}.csv'
    with open(name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data_before_headers)
        writer.writerow(columns)
        def SpecialRound(val):
            if 0.001 <= val < 0.01:
                s = 6
            elif 0.01 <= val < 0.1:
                s = 5
            elif 0.1 <= val < 1:
                s = 4
            elif 1 <= val < 10:
                s = 3
            elif 10 <= val < 100:
                s = 2
            elif 100 <= val < 1000:
                s = 1
            else:
                s = 0                      
    # Возвращаем результат
            return round(val, s)
        y_ = SpecialRound(selected_points[0][1]) 
        selected_points[0] = (y_ / 200000, y_)
        for i in range(len(ln_res)):
            if i == 0:
                row = [ln_res[i], 
                   exp_res[i], 
                   selected_points[i][0], 
                   selected_points[i][1]]
                writer.writerow(row)
            else:
                row = [SpecialRound(ln_res[i]), 
                        SpecialRound(exp_res[i]), 
                        SpecialRound(selected_points[i][0]) if i < 20 else '', 
                        SpecialRound(selected_points[i][1]) if i < 20 else '']
                writer.writerow(row)


# Создание окна приложения
root = Tk()
root.title("Истинные диаграммы")
#root.geometry('1200x600')

frame = Frame(root, padx=10, pady=10)
frame.pack(expand=True)

markirovka = Label(frame, text='Введите маркировку образца ')
markirovka.grid(row=2, column=1)
entry_mark = Entry(frame, width=30)
entry_mark.grid(row=2, column=2)

sample_var = StringVar()

sample_label = Label(frame, text='Какой тип образца? ')
sample_label.grid(row=3, column=1)
sample_cylinder = Radiobutton(frame, text="Цилиндр", variable=sample_var, value="Цилиндр", command=check_sample_type)
sample_cylinder.grid(row=3, column=2)
sample_flat = Radiobutton(frame, text="Плоский", variable=sample_var, value="Плоский", command=check_sample_type)
sample_flat.grid(row=3, column=3)
diameter_label = Label(frame, text="Введите диаметр образца в мм")
entry_diameter = Entry(frame, width=30)

young_label = Label(frame, text='Введите модуль упругости в МПа ')
young_label.grid(row=5, column=1)
young_mod_entry = Entry(frame, width=30)
young_mod_entry.grid(row=5, column=2)

# Поле для ввода пути к файлу
file_label = Label(frame, text='Выберите файл')
file_label.grid(row=0, column=1)
entry_file_path = Entry(frame, width=30)
entry_file_path.grid(row=0, column=2)

# Кнопка "Выбрать файл"
button_choose_file = Button(frame, text="Выбрать файл", command=choose_file)
button_choose_file.grid(row=0, column=3)
sheet_name_label = Label(frame, text='Введите название листа с данными, если формат файла .xls')
sheet_name_label.grid(row=1, column=1)
sheet_name = Entry(frame, width=30)
sheet_name.grid(row=1, column=2)


length_label = Label(frame, text='Введите длину рабочей части в мм')
length_label.grid(row=6, column=1)
length_work_entry = Entry(frame, width=30)
length_work_entry.grid(row=6, column=2)

tens_var = StringVar()
tens_label = Label(frame, text='Задана нагрузка или напряжение?')
tens_label.grid(row=7, column=1)
stress = Radiobutton(frame, text="Нагрузка", variable=tens_var, value="Нагрузка")
stress.grid(row=7, column=2)
tens_ = Radiobutton(frame, text="Напряжения", variable=tens_var, value="Напряжения")
tens_.grid(row=7, column=3)
num_tens1 = Label(frame, text='Введите порядковый номер столбца с напряжениями из файла')
num_tens1_entry = Entry(frame, width=30)
num_stress = Label(frame, text='Введите порядковый номер столбца с нагрузкой из файла')
num_stress_entry = Entry(frame, width=30)

rab_sech_label = Label(frame, text='Введите площадь рабочего сечения образца в мм2')
rab_sech_entry = Entry(frame, width=30)

n_kn_label = Label(frame, text='В чем задана нагрузка? (Н/кН) или введите коэффициент пересчета из Вольтов')
n_kn_entry = Entry(frame, width=30)
button_tens = Button(frame, text="Подтвердить", command=tens)
button_tens.grid(row=7, column=4)

elong = StringVar()
num_elong_label = Label(frame, text='Задано удлинение или перемещение?')
num_elong_label.grid(row=11, column=1)
elong_1 = Radiobutton(frame, variable=elong, text='Удлинение', value='Удлинение')
elong_1.grid(row=11, column=2)
elong_2 = Radiobutton(frame, variable=elong, text='Перемещение траверсы', value='т')
elong_2.grid(row=11, column=3)
elong_3 = Radiobutton(frame, variable=elong, text='Перемещение по датчику', value='д')
elong_3.grid(row=11, column=4)
elong_4 = Radiobutton(frame, variable=elong, text='Перемещение/В', value='в')
elong_4.grid(row=11, column=5)
button_elong = Button(frame, text='Подтвердить', command=choose_elong)
button_elong.grid(row=11, column=6)
nach_rass_label = Label(frame, text='Введите начальное расстояние между ножками датчика в мм')
nach_rass_entry = Entry(frame, width=30)
koef_volt = Label(frame, text='Введите коэффициент для пересчета из Вольтов')
koef_volt_entry = Entry(frame, width=30)
elong_label = Label(frame, text='Введите порядковый номер столбца с удлинениями/перемещениями')
elong_label.grid(row=12, column=1)
num_elong_entry = Entry(frame, width=30)
num_elong_entry.grid(row=12, column=2)



predel_teck_label = Label(frame, text='Введите условный предел текучести в МПа')
predel_teck_label.grid(row=14, column=1)
predel_teck_entry = Entry(frame, width=30)
predel_teck_entry.grid(row=14, column=2)

elong_procent_label = Label(frame, text='Введите значение удлинения из отчета в процентах')
elong_procent_label.grid(row=15, column=1)
elong_procent_entry = Entry(frame, width=30)
elong_procent_entry.grid(row=15, column=2)

koef_label = Label(frame, text='Введите коэффициент динамичности')
koef_label.grid(row=16, column=1)
koef_entry = Entry(frame, width=30)
koef_entry.grid(row=16, column=2)
koef_dinam = 1

markirovka = None
type_sample = None
young_mod = None
length_work = None
num_tens = None
type_elong = None
num_elong = None
predel_teck = None
rab_sech = None
n_kn = None 
res_x = None
res_y = None
sp = []
ln_res = None
exp_res = None
B = None
ind_max = None
after_max = None
exp_y = None
C = None
approc_x = []
approc_y = []

data_elong = None
data_tens = None

# Кнопка "Запустить программу"
button_run_program = Button(frame, text="Запустить программу", command=run_program)
button_run_program.grid(row=17, column=2)

smooth = StringVar()
need_smooth_label = Label(frame, text='Необходимо ли сгладить график?')
need_smooth_y = Radiobutton(frame, variable=smooth, text='Да', value='д')
need_smooth_n = Radiobutton(frame, variable=smooth, text='Нет', value='н')
need_smooth_label.grid(row=18, column=1)
need_smooth_y.grid(row=18, column=2)
need_smooth_n.grid(row=18, column=3)
button_smooth = Button(frame, text='Подтвердить', command=smoothed)
button_smooth.grid(row=18, column=4)


A_label = Label(frame, text='Введите коэффициенты А и С')
A_label.grid(row=19, column=1)
A_entry = Entry(frame, width=30)
A_entry.grid(row=19, column=2)
C_entry = Entry(frame, width=30)
C_entry.grid(row=19, column=3)
update_button = Button(frame, text="Обновить параметр A и С", command=program)
update_button.grid(row=19, column=4)
button_wrap = Button(frame, text='Готово', command=wrapper)
button_wrap.grid(row=19, column=5)


name_label = Label(frame, text='Введите название будущего файла')
name_label.grid(row=20, column=1)
name_entry = Entry(frame, width=30)
name_entry.grid(row=20, column=2)
button_program_choose = Button(frame, text='Сформировать файл', command=program_choose)
button_program_choose.grid(row=21, column=2)



root.mainloop()

from tkinter import *
import tkinter as tk
from tkinter import filedialog as fd
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.io import imread_collection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model 
from sklearn import ensemble
# from skimage import data
from skimage.feature import match_template
from skimage.color import rgb2gray
import cv2
import heapq
import sys

class GuiMain:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('280x420')
        self.var_1 = tk.Entry(self.root)
        self.var_2 = tk.Entry(self.root)
        self.txt = tk.Text(self.root, height=15, width=20)
#         self.file = fd.askopenfilename()
        self.build_gui()

    def build_gui(self):
        tk.Label(self.root, text='Ilość wyszókiwanych hieroglifów').pack()
        self.var_1.pack()
        tk.Label(self.root, text='Tolreancja nachodzenia hieroglifów na siebie').pack()
        self.var_2.pack()
        tk.Label(self.root, text='Wyniki').pack()
        self.txt.pack()

        tk.Button(self.root, text="Wybierz i Analizuj Zdjęcie", 
                  fg="Black", command=self.select_glyph).pack()
        
    def select_glyph(self):
        self.txt.delete('1.0', END)
        filetypes = (
            ('All files', ('*.jpg', '*.png', '*.jpeg')),
        )
        path = fd.askopenfilename(filetypes=filetypes)
        img = mpimg.imread(path)
        img_gray = rgb2gray(img)
        unicodem, glyph_id, data, label = self.prepare_data()
        self.find_glyphs(int(self.var_1.get()), img_gray, int(self.var_2.get()),
                         unicodem, glyph_id, data, label) 
#         plt.imshow(img_gray, cmap=plt.cm.gray)
#         plt.show()  

    @staticmethod
    def popup_window(x):
        window = tk.Toplevel()
        window.geometry('250x80')
        tk.Label(window, text=f'Wynik = {x}').pack()
        button_close = tk.Button(window, text="Close", command=window.destroy)
        button_close.pack(fill='x')

    def run(self):
        self.root.mainloop()

    #rest of code
    
    def prepare_data(self):
        unicode = []
        glyph_id = []
        data = []
        label = []

        

        with open("unicode.txt") as file:
            lines = file.readlines()
        for i in range(len(lines)):
            glyph_id.append(lines[i][42] + lines[i][44:46].lstrip('0'))
            unicode.append(lines[i][0])
            
        col_dir = 'data/*.png'
        col = imread_collection(col_dir)

        for i in range(len(col)):
            name = col.files[i].split('_')[1].split('.')[0]   
            if name != 'UNKNOWN':
                label.append(name)
                if(col[i].ndim > 2):
                    data.append(self.fix_shape(col[i]))
                else:
                    data.append(col[i])
        return(unicode,glyph_id,data,label)
        
    def fix_shape(self, img):
        if img.shape[2] == 3:
            new_img = np.delete(img, np.arange(0, img.size, 3))
            new_img = np.delete(new_img, np.arange(0, new_img.size, 2))
            new_img.shape = (img.shape[0], img.shape[1])
        elif img.shape[2]  == 4:
            new_img = np.delete(img, np.arange(0, img.size, 2))
            new_img = np.delete(new_img, np.arange(1, new_img.size, 2))
            new_img.shape = (img.shape[0], img.shape[1])
        return new_img
    
    def check_conflict(self, x, y, x_list, y_list, overlap_tolerance):
        x_conflict = False
        y_conflict = False
        for xs in x_list:
            if x in range(xs-overlap_tolerance, xs+overlap_tolerance):
                x_conflict = True
        for ys in y_list:
            if y in range(ys-overlap_tolerance, ys+overlap_tolerance):
                y_conflict = True
        if((x_conflict == True) and
           (y_conflict == True)):
            return True
        else:
            return False
        
    def find_glyphs(self, number, img, overlap_tolerance, unicode, glyph_id, data, label):
        score = []
        xs = []
        ys = []
    
        for i in range(len(data)):
#                 if log == True:
            j = (i + 1) / len(data)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            if np.any(np.less(img.shape, data[i].shape)):
#                 if log == True:
                sys.stdout.write("size conflic skippping hieroglyph")
            else:
                result = match_template(img, data[i])
                ij = np.unravel_index(np.argmax(result), result.shape)
                x, y = ij[::-1]
                score.append(result[y,x])
                xs.append(x)
                ys.append(y)
                
        best_fit = heapq.nlargest(2000, range(len(score)), key=score.__getitem__)
        
        uniqe_best_fit = []
        loc_labels = []
        loc_x = []
        loc_y = []
        j = 0
        while len(loc_labels) < number:
            conflict = self.check_conflict(xs[best_fit[j]], ys[best_fit[j]],
                           loc_x, loc_y, overlap_tolerance)
            if(conflict == True):
                j = j + 1
            else:
                loc_labels.append(label[best_fit[j]])
                loc_x.append(xs[best_fit[j]])
                loc_y.append(ys[best_fit[j]])
                uniqe_best_fit.append(best_fit[j])
                j = j + 1      
         
        fig = plt.figure(figsize=(50, 20))  
        ax = plt.subplot(1, 3, 2)
        ax.imshow(img, cmap=plt.cm.gray)
        
        for i in uniqe_best_fit:
            rect = plt.Rectangle((xs[i], ys[i]), data[i].shape[1],
                                 data[i].shape[0], edgecolor='r', facecolor='none')
            name = plt.text(xs[i]+5, ys[i]+5, label[i], color='r', size = 'x-large')
            ax.add_patch(rect)
    
        plt.show()    
        
        print("Zanlieziono następujące hierogliphy: ")
        loc_unicodes = []
        for k in loc_labels:
            result = glyph_id.index(k)
            self.txt.insert(END, k + ' - ' + unicode[result] + '\n')
            print(k + ' - ' + unicode[result])
        self.root.mainloop()

    
if __name__ == '__main__':
    gui = GuiMain()
    gui.run()

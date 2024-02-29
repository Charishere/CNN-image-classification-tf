import tkinter as tk
import pandas as pd
from tkinter import messagebox
from tkinter import filedialog as fd
from tkinter import font
import os
import sys
from tkinter import scrolledtext
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
from PIL import ImageTk, Image

class CustomError(Exception):
    def __init__(self, message="Custom error occurred"):
        self.message = message
        super().__init__(self.message)
                
class CNN(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("CNN Model for image classification")
        self.geometry("483x500")
        self.resizable(width=False, height=False)


        self.main_frame = tk.Frame(self, border=5)
        self.main_frame.grid(row=0, column=0)
        self.frame_p = tk.Frame(self.main_frame)
        self.frame_t = tk.Frame(self.main_frame)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, sticky="w")
        label_font = font.Font(weight="bold")
        predict = tk.Button(self.button_frame, text="Predict", command=self.show_frame_p, fg="#008080",font=label_font)
        predict.grid(row=0,column=0,sticky="w", pady=8)
        train_but = tk.Button(self.button_frame, text="Train", command=self.show_frame_t, fg="#008080", font=label_font)
        train_but.grid(row=0,column=1,sticky="w", pady=8)



        #data frame
        self.train_executed = False
        self.model = None
        self.norm_test = None
        self.dict = None
        self.ph_d = None
        self.index = 0

        input = tk.Frame(self.frame_p)
        input.grid(row=0, column=0, sticky="w") 

        img_lbl = tk.Label(input, text='Image')
        img_lbl.grid(row=0,column=0,sticky="w", padx=0)
        self.entry_img = tk.Entry(input, width=31)
        self.entry_img.grid(row=0,column=1,sticky="w", padx=0)
        self.entry_img.config(state="readonly", readonlybackground='#FFFFFF')

        

        

        ###classify
        status_fr = tk.Frame(self.frame_p)
        status_fr.grid(row=1, column=0, sticky="w") 
        model_lbl = tk.Label(status_fr, text='Model status:')
        model_lbl.grid(row=0,column=0,sticky="w", padx=0)
        self.status = tk.StringVar()
        self.status.set("未识别")
        status_lbl = tk.Label(status_fr, textvariable=self.status, fg='#D2691E')
        status_lbl.grid(row=0,column=1,sticky="w", padx=0)

        #pre frame
        output = tk.Frame(self.frame_p, relief=tk.GROOVE, borderwidth=2)
        output.grid(row=4, column=0,sticky="w", ipadx=199)
        output_l = tk.Label(output, text='Output', fg='#4682B4')
        output_l.grid(row=0, column=0, sticky="w")
        tk.Label(output, text='', font=('AppleSystemUIFont',1)).grid(row=1, column=0)
        result = tk.Label(output, text='Result:')
        result.grid(row=2, column=0, sticky="w")
        acc = tk.Label(output, text='Accuracy:')
        acc.grid(row=3, column=0, sticky="w")

        self.result_txt = tk.StringVar()
        self.result_txt.set('')
        result_lbl = tk.Label(output, textvariable=self.result_txt)
        result_lbl.grid(row=2, column=1, sticky="w")

        self.accu_txt = tk.StringVar()
        self.accu_txt.set('')
        accu_lbl = tk.Label(output, textvariable=self.accu_txt)
        accu_lbl.grid(row=3, column=1, sticky="w")
            


        def previous_img():
            try:
                print(self.train_executed)
                if self.train_executed:
                    self.index -= 1
                    if self.index >= 0:                          
                        probability = self.model.predict(self.norm_test)
                        print('p',self.index)
                        self.show_probability(self.norm_test, probability, self.index)
                        self.change_text()
                        self.update_lbl()
                        
                else:
                    raise CustomError('Please train first')
            except ValueError:
                messagebox.showerror("Error", "Please train first")
            except CustomError as ce:
                messagebox.showerror("Error", str(ce))
            except FileNotFoundError:
                messagebox.showerror("Error", "Image not found")
            except Exception as e:
                messagebox.showerror("Error", "Failed to load the previous image")
        
        
        def load_img():
            try:
                print(self.train_executed)
                self.not_change_text()
                self.update_idletasks()
                if self.train_executed:
                    probability = self.model.predict(self.norm_test)
                    print('x',self.index)
                    self.show_probability(self.norm_test, probability, self.index)
                    self.change_text()
                    self.update_lbl() 
                    self.index += 1 
                else:
                    raise CustomError('Please train first') 
            except IndexError:
                messagebox.showinfo("Info", "No more images")
            except CustomError as ce:
                messagebox.showerror("Error", str(ce))
            except FileNotFoundError:
                messagebox.showerror("Error", "Image not found")
            except Exception as e:
                messagebox.showerror("Error", "Failed to load the image")

        


        prev_img = tk.Button(input, text="Prev", relief=tk.RAISED, command=previous_img)
        prev_img.grid(row=0,column=2, sticky="w", padx=4) 
        next_ig = tk.Button(input, text="Next", command=load_img)
        next_ig.grid(row=0,column=3,sticky="w", padx=0)

        file_lbl = tk.Label(input, text='File')
        file_lbl.grid(row=1,column=0,sticky="w", padx=0)
        self.entry_file = tk.Entry(input, width=31)
        self.entry_file.grid(row=1,column=1,sticky="w", padx=0)
        self.entry_file.config(state="readonly", readonlybackground='#FFFFFF')

        def load_data():
            try:
                print(self.train_executed)
                self.not_change_text()
                self.update_idletasks()
                if self.train_executed:        
                    probability = self.model.predict(self.norm_test)
                    self.entry_file.config(state="normal")
                    self.entry_file.delete(0, tk.END)
                    self.entry_file.insert(0, 'Images from the selected folder...')
                    self.entry_file.config(state="readonly")
                    self.update_idletasks()
                    self.change_text()
                    self.update_idletasks()
                    for self.index in range(len(probability)):
                        self.show_probability(self.norm_test, probability, self.index)                            
                        self.update_lbl()
                        self.update_idletasks() 
                    self.can_navigate = True
                else:
                    raise CustomError('Please train first') 
            except CustomError as ce:
                messagebox.showerror("Error", str(ce))                  
            except FileNotFoundError:
                messagebox.showerror("Error", "Folder not found")
            except Exception as e:
                messagebox.showerror("Error", "Failed to load data")
           

        load_but = tk.Button(input, text="Load", command=load_data)
        load_but.grid(row=1,column=2,sticky="w", padx=4)

    

        ###show panel      
        image_panel_frame = tk.Frame(self.frame_p)
        image_panel_frame.grid(row=3, column=0, pady=3, sticky="w")
        img = ImageTk.PhotoImage(Image.new("RGB", (467, 280), "white"))
        self.panel = tk.Label(image_panel_frame, image=img)
        self.panel.image = img
        self.panel.grid(row=0, column=0)
        self.panel.zoom_level = 1.0            


        



        #train frame
        train_fr = tk.Frame(self.frame_t)
        train_fr.grid(row=0, column=0, sticky="w") 

        label_fr = tk.Frame(self.frame_t)
        label_fr.grid(row=0, column=0, sticky="w")
        train_folder_lbl = tk.Entry(label_fr, width=30)
        train_folder_lbl.grid(row=0,column=0,sticky="w", padx=0)



        table_fr = tk.Frame(self.frame_t)
        table_fr.grid(row=1, column=0, sticky="w")        
        index = tk.StringVar()
        index.set("Index")
        ent_ind = tk.Entry(table_fr, textvariable=index, state="readonly", width=4, readonlybackground='#C0C0C0')
        ent_ind.grid(row=0,column=0,sticky="w", padx=0)
        one = tk.StringVar()
        one.set("1")
        ent_1 = tk.Entry(table_fr, textvariable=one, state="readonly", width=4, readonlybackground='#C0C0C0')
        ent_1.grid(row=1,column=0,sticky="w", padx=0)        
        two = tk.StringVar()
        two.set("2")
        ent_2 = tk.Entry(table_fr, textvariable=two, state="readonly", width=4, readonlybackground='#C0C0C0')
        ent_2.grid(row=2,column=0,sticky="w", padx=0)
        three = tk.StringVar()
        three.set("3")
        ent_3 = tk.Entry(table_fr, textvariable=three, state="readonly", width=4, readonlybackground='#C0C0C0')
        ent_3.grid(row=3,column=0,sticky="w", padx=0)
        four = tk.StringVar()
        four.set("4")
        ent_3 = tk.Entry(table_fr, textvariable=four, state="readonly", width=4, readonlybackground='#C0C0C0')
        ent_3.grid(row=4,column=0,sticky="w", padx=0)

        label = tk.StringVar()
        label.set("Label")
        ent_lbl = tk.Entry(table_fr, textvariable=label, state="readonly", width=10, readonlybackground='#696969')
        ent_lbl.grid(row=0,column=1,sticky="w", padx=0)
        ent_l1 = tk.Entry(table_fr, bg='#696969', width=10)
        ent_l1.grid(row=1,column=1,sticky="w", padx=0)
        ent_l2 = tk.Entry(table_fr, bg='#696969', width=10)
        ent_l2.grid(row=2,column=1,sticky="w", padx=0)
        ent_l3 = tk.Entry(table_fr, bg='#696969', width=10)
        ent_l3.grid(row=3,column=1,sticky="w", padx=0)
        ent_l4 = tk.Entry(table_fr, bg='#696969', width=10)
        ent_l4.grid(row=4,column=1,sticky="w", padx=0)

        data = tk.StringVar()
        data.set("Sample")
        ent_spl = tk.Entry(table_fr, textvariable=data, state="readonly", width=10, readonlybackground='#C0C0C0')
        ent_spl.grid(row=0,column=2,sticky="w", padx=0)
        ent_s1 = tk.Entry(table_fr, bg='#C0C0C0', width=10)
        ent_s1.grid(row=1,column=2,sticky="w", padx=0)
        ent_s2 = tk.Entry(table_fr, bg='#C0C0C0', width=10)
        ent_s2.grid(row=2,column=2,sticky="w", padx=0)
        ent_s3 = tk.Entry(table_fr, bg='#C0C0C0', width=10)
        ent_s3.grid(row=3,column=2,sticky="w", padx=0)
        ent_s4 = tk.Entry(table_fr, bg='#C0C0C0', width=10)
        ent_s4.grid(row=4,column=2,sticky="w", padx=0)





        def train_folder():
            folder_path = fd.askdirectory(initialdir='/')
            if folder_path:
                train_folder_lbl.delete(0, tk.END)
            train_folder_lbl.insert(0, folder_path)

        def load_train():
            train_img = train_folder_lbl.get()
            try:
                if not train_img:
                   raise ValueError("Please select a folder first")
                else:                    
                    B1_0 = []
                    B2_0 = []
                    C1_5 = []
                    C2_0 = []
                    #f_202 = []
                    #f_302 = []
                    #f_402 = []
                    #f_502 = []
                    for folder_name in os.listdir(train_img):
                        folder_path = os.path.join(train_img, folder_name)
                        if os.path.isdir(folder_path):
                            for img in os.listdir(folder_path):
                                if not img.lower().endswith('.ds_store'):
                                    img_path = os.path.join(folder_path, img)
                                    name = img.split(sep='-')
                                    if name[0] == 'B2.0':
                                        B2_0.append(img_path)
                                        #f_202.append(img_path)
                                    elif name[0] == 'B1.0':
                                        B1_0.append(img_path)
                                        #f_302.append(img_path)
                                    elif name[0] == 'C1.5':
                                        C1_5.append(img_path)
                                        #f_402.append(img_path)
                                    else:
                                        C2_0.append(img_path)
                                        #f_502.append(img_path)
                    if all(len(lst) != 0 for lst in (B1_0, B2_0, C1_5, C2_0)):
                        print('Train folder path:', train_img)
                        return B1_0, B2_0, C1_5, C2_0, train_img
                    else:
                        messagebox.showerror('Error', 'Invalid folder')

            except ValueError:
                messagebox.showerror('Error', 'Please select a folder first')
            except FileNotFoundError:
                messagebox.showerror('Error', 'Folder not found')
            except Exception as e:
                messagebox.showerror('Error', 'Failed to load folder')
                
        def ent_show(): 
            try:  
                B1_0, B2_0, C1_5, C2_0, train_img = load_train()
                s1 = len(B1_0)
                s2 = len(B2_0)
                s3 = len(C1_5)
                s4 = len(C2_0)
                ent_l1.delete(0, tk.END)
                ent_l1.insert(tk.END, 'B1.0')
                #ent_l1.insert(tk.END, '202')
                ent_l1.config(state="readonly")
                ent_l1.config(readonlybackground='#696969')
                ent_s1.delete(0, tk.END)
                ent_s1.insert(tk.END, s1)
                ent_s1.config(state="readonly")
                ent_s1.config(readonlybackground='#C0C0C0')
                ent_l2.delete(0, tk.END)
                ent_l2.insert(tk.END, 'B2.0')
                #ent_l2.insert(tk.END, '302')
                ent_l2.config(state="readonly")
                ent_l2.config(readonlybackground='#696969')
                ent_s2.delete(0, tk.END)
                ent_s2.insert(tk.END, s2)
                ent_s2.config(state="readonly")
                ent_s2.config(readonlybackground='#C0C0C0')
                ent_l3.delete(0, tk.END)
                ent_l3.insert(tk.END, 'C1.5')
                #ent_l3.insert(tk.END, '402')
                ent_l3.config(state="readonly")
                ent_l3.config(readonlybackground='#696969')
                ent_s3.delete(0, tk.END)
                ent_s3.insert(tk.END, s3)
                ent_s3.config(state="readonly")
                ent_s3.config(readonlybackground='#C0C0C0')
                ent_l4.delete(0, tk.END)
                ent_l4.insert(tk.END, 'C2.0')
                #ent_l4.insert(tk.END, '502')
                ent_l4.config(state="readonly")
                ent_l4.config(readonlybackground='#696969')
                ent_s4.delete(0, tk.END)
                ent_s4.insert(tk.END, s4)
                ent_s4.config(state="readonly")
                ent_s4.config(readonlybackground='#C0C0C0') 
            except Exception as e:
                pass

        but_sl = tk.Button(label_fr, text='Select', command=train_folder)
        but_ld = tk.Button(label_fr, text='Load', command=ent_show)
        but_sl.grid(row=0,column=1,sticky="w", padx=0)
        but_ld.grid(row=0,column=2,sticky="w", padx=0)



        train_fr = tk.Frame(self.frame_t)
        train_fr.grid(row=2, column=0, sticky="w")
                
        ratio_lbl = tk.Label(train_fr, text='Ratio:')
        ratio_lbl.grid(row=0,column=0,sticky="w")
        ratio_ent = tk.Entry(train_fr, width=7)
        ratio_ent.grid(row=0,column=0,sticky="w",padx=38)


        def get_ratio():
            ratio = float(ratio_ent.get())
            if ratio < 0 or ratio >= 0.5:
                messagebox.showerror("Error","Invalid number. Please enter a number less than 0.5 and bigger than 0.")
            else:
                return ratio
          

        import input_data
        import cnn_model
        def train():
            sys.stdout = text_redirector(text, "stdout")

            try:
                filename =  train_folder_lbl.get()
                if not filename:
                    messagebox.showerror('Error', 'Please select a folder first')
                else:
                    epochs = 20
                    try:
                        ratio = get_ratio()
                        print('ratio:',ratio)
                    except ValueError:
                        messagebox.showerror('Error', 'Please enter the ratio')                
                    norm_train, train_lbl, norm_test, test_lbl, dict, photo_dict = input_data.input_files(filename, ratio)
                    self.norm_test = norm_test
                    self.dict = dict
                    self.ph_d = photo_dict
                    print('')
                    print('')
                    print('Training...')
                    print('')
                    model = cnn_model.cnn_prediction()
                    model.summary()                    
                    print('')
                    model.compile(loss='categorical_crossentropy', 
                                optimizer='adam',
                                metrics=['accuracy']
                                )
                    sys.stdout = sys.__stdout__
                    train_history = model.fit(norm_train, 
                                            train_lbl,
                                            validation_split=0.0005,
                                            shuffle=True,
                                            epochs=epochs, 
                                            batch_size=256
                                            )
                    self.train_executed = True
                    self.model = model
                    sys.stdout = text_redirector(text, "stdout")
                    print('Training completed!')
                    print('')
                    print('')                     

            except Exception as e:
                messagebox.showerror('Error',"Training failed")
            finally:
                sys.stdout = sys.__stdout__

        train_but = tk.Button(train_fr, text='Train', command=train)
        train_but.grid(row=0,column=2,sticky="w", padx=0)
        
            

        output = tk.Label(train_fr, text='Output:')
        output.grid(row=1,column=0,sticky="w", padx=0)  

        output_fr = tk.Frame(self.frame_t)
        output_fr.grid(row=3, column=0, sticky="w")

        
        class text_redirector:
            def __init__(self, widget, tag="stdout"):
                self.widget = widget
                self.tag = tag

            def write(self, s):
                self.widget.configure(state="normal")
                self.widget.insert("end", s, (self.tag,))
                self.widget.configure(state="disabled")
                self.widget.update_idletasks()
            def flush(self):
                pass
        text = scrolledtext.ScrolledText(output_fr, wrap="word", height=17, width=66,padx=2)
        text.pack(expand=True, fill="both")
             



        self.show_frame_p()

    def show_frame_p(self):
        self.frame_t.grid_remove()   
        self.frame_p.grid(row=1, column=0, sticky="w")         

    def show_frame_t(self):
        self.frame_p.grid_remove()  
        self.frame_t.grid(row=1, column=0, sticky="w")  


    def generate_and_display(self):
        originals = []
        index = 0  
        print(self.images)      
        if len(self.images) != 0:
            for image in self.images:
                originals.append(image)
                img = Image.open(image)                       
                img_p = img.resize((467, 280), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_p)
                self.panel.configure(image=img_tk)
                self.panel.image = img_tk
                self.panel.original_image = img
                viewer = self.Viewer(self.images, originals)
                self.panel.bind("<Button-1>", lambda event: viewer.show())
        else:
            messagebox.showerror("Error","Please train first.")

    def show_probability(self, X_img, probability, i):
            self.comp = {}
            self.images = []
            print(self.dict)
            for j in range(1, 5):                
            #for j in range(0, 5):
                #if j != 1:
                    print("{} Probability: {:.9f}".format(self.dict[j], probability[i][j]))
                    self.path = self.ph_d[i]
                    p_path = os.path.basename(self.path)
                    if self.dict[j] in self.comp:
                        self.comp[self.dict[j]].append(probability[i][j])
                    else:
                        self.comp[self.dict[j]] = [probability[i][j]]
            self.images.append(self.path)
            self.entry_img.config(state="normal")
            self.entry_img.delete(0, tk.END)
            self.entry_img.insert(0, p_path)
            self.entry_img.config(state="readonly")
            self.generate_and_display()


    def change_text(self):
            self.status.set("识别完成")
    def not_change_text(self):
            self.status.set("未识别")

    def update_lbl(self):
        max_key = max(self.comp, key=lambda k: max(self.comp[k]))
        key = str(max_key)
        print(self.comp[max_key])
        val = str(max(self.comp[max_key]))
        print(val)
        self.result_txt.set(key)
        self.accu_txt.set(val)       
    
    class Viewer:
        def __init__(self, images, original_image):
            self.images = images
            self.index = 0
            self.viewer_window = None
            self.original_image = original_image
            self.can_navigate = False
        def show(self):
            if not self.viewer_window:
                self.viewer_window = tk.Toplevel()
                self.viewer_window.title("Image Viewer")           
                # Create Label to display image
                self.image_label = tk.Label(self.viewer_window)
                self.image_label.pack()
                # Create navigation buttons
                btn_prev = tk.Button(self.viewer_window, text="<", command=self.show_prev)
                btn_next = tk.Button(self.viewer_window, text=">", command=self.show_next)
                btn_prev.pack(side=tk.LEFT)
                btn_next.pack(side=tk.RIGHT)
                self.image_label.bind("<Button-1>", lambda event: self.open_default_viewer())           
            # Display the first image
                self.show_image()

        def show_image(self):
            if self.viewer_window and self.image_label.winfo_exists():
                img = self.images[self.index]
                title = f"Image {self.index + 1}"
                img = Image.open(img)
                img = img.resize((467, 280), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.configure(image=img_tk, text=title, compound='bottom')
                self.image_label.image = img_tk
                self.image_label.title = title
            else:
                messagebox.showerror("Error","Error occurs")
        def show_next(self):
            if self.index < len(self.images) - 1 and self.can_navigate:
                self.index += 1
                self.show_image()
        def show_prev(self):
            if self.index > 0 and self.can_navigate:
                self.index -= 1
                self.show_image()    
        def open_default_viewer(self):
            if self.index < len(self.original_image):
                original_image = self.original_image[self.index]
                original_image = Image.open(original_image)
                original_image.show()


if __name__ == "__main__":
    app = CNN()
    app.mainloop()

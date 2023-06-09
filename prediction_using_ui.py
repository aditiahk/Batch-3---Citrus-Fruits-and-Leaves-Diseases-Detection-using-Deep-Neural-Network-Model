import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import tkinter.messagebox

window = tk.Tk()

window.title("Citrus Fruit and leaves Detection")

window.geometry("350x350")
im=Image.open('./6.jpg')
im=im.resize((350,350))
ph_im=ImageTk.PhotoImage(im)
l=tk.Label(image=ph_im)
l.place(x=0,y=0)

title = tk.Label(text = "Click below to choose the image..", background = "lightblue", fg="Brown", font=("Lucida Grande", 15))
title.grid()

def Black_spot():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. Plant")

    window1.geometry("500x510")

    im=Image.open('./6.jpg')
    im=im.resize((500,510))
    ph_im=ImageTk.PhotoImage(im)
    l=tk.Label(image=ph_im)
    l.place(x=0,y=0)

    def exit():
        window1.destroy()

    rem = "The remedies for BlackSpot are:\n\n "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = "Use drip irrigation or water by hand at ground level to keep fruits dry \n and free from water that black spot and \n other diseases use to spread.\n"
    remedies1 = tk.Label(text=rem1, background="pink",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def canker():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. Plant")

    window1.geometry("650x510")

    im=Image.open('./6.jpg')
    im=im.resize((350,510))
    ph_im=ImageTk.PhotoImage(im)
    l=tk.Label(image=ph_im)
    l.place(x=0,y=0)

    def exit():
        window1.destroy()
        
    rem = "The remedies for citrus canker are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " No cure exists for citrus canker; disease management is the \n only way to control the disease.\n Citrus canker management involves \n the use of the timely applications of copper-containing \n products and windbreaks to hinder inoculum dispersal."
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def greening():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. Plant")

    window1.geometry("520x510")

    im=Image.open('./6.jpg')
    im=im.resize((520,510))
    ph_im=ImageTk.PhotoImage(im)
    l=tk.Label(image=ph_im)
    l.place(x=0,y=0)

    def exit():
        window1.destroy()
        
    rem = "The remedies for Citrus Greening are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = "Citrus greening is spread by a disease-infected insect,\n the Asian citrus psyllid (Diaphorina citri Kuwayama or ACP), \n and has put the future of America's citrus at risk.\n Infected trees produce fruits that are green, misshapen and \n bitter, unsuitable for sale as fresh fruit or for juice."
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def healthy():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. Plant")

    window1.geometry("500x510")

    im=Image.open('./6.jpg')
    im=im.resize((500,510))
    ph_im=ImageTk.PhotoImage(im)
    l=tk.Label(image=ph_im)
    l.place(x=0,y=0)

    def exit():
        window1.destroy()

    rem = "The Fruit images are healthy:\n\n "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Discard or destroy any affected friuts. \n  Do not compost them. \n  Rotate the citrus orchards plants yearly to prevent re-infection next year. \n Use copper fungicites."
    remedies1 = tk.Label(text=rem1, background="pink",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def melanose():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. Plant")

    window1.geometry("650x510")

    im=Image.open('./6.jpg')
    im=im.resize((650,510))
    ph_im=ImageTk.PhotoImage(im)
    l=tk.Label(image=ph_im)
    l.place(x=0,y=0)

    def exit():
        window1.destroy()
        
    rem = "The remedies for Melanose are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Obviously prevention is better than cure,\n but the cleaning out and removal of dead wood to remove inoculum \n of the melanose fungus is important, especially in older trees.\n Protectant copper sprays are the only product registered for melanose control"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

def Scab():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. Plant")

    window1.geometry("520x510")

    im=Image.open('./6.jpg')
    im=im.resize((520,510))
    ph_im=ImageTk.PhotoImage(im)
    l=tk.Label(image=ph_im)
    l.place(x=0,y=0)

    def exit():
        window1.destroy()
        
    rem = "The remedies for Scab are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " Protective copper sprays are the only products  registered to control scab in citrus.\n Since copper is a protectant fungicide the entire fruit surface needs to have a continuous \n coating of copper in order to be protected from infection \n by the fungal spores."
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()
def analysis():


    from tensorflow.keras.models import load_model
    from collections import deque
    import numpy as np
    import argparse
    import pickle
    import cv2
    button2.destroy()
    img_dims = 64
    batch_size = 16

    print("[INFO] loading model and label binarizer...")
    model = load_model('./new_model_leaves.h5')
    #model.summary()
    lb = pickle.loads(open('./new_model_leaves.pkl', "rb").read())
    img = cv2.imread("img\\2.jpeg")

    output = img.copy()
    output = cv2.resize(output, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32")
    preds = model.predict(np.expand_dims(img, axis=0))[0]


    i = np.argmax(preds)
    label = lb.classes_[i]
    text = "{}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 2)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    message = tk.Label(text='Status: '+label, background="lightgreen",
                       fg="Brown", font=("", 15))
    message.grid(column=0, row=3, padx=10, pady=10)
    if label == 'Black spot':
        diseasename = "bacterial"
        disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                           fg="Black", font=("", 15))
        disease.grid(column=0, row=4, padx=10, pady=10)
        r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
        r.grid(column=0, row=5, padx=10, pady=10)
        button3 = tk.Button(text="Remedies", command=Black_spot)
        button3.grid(column=0, row=6, padx=10, pady=10)
        
    elif label == 'canker':
        diseasename = "canker "
        disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                           fg="Black", font=("", 15))
        disease.grid(column=0, row=4, padx=10, pady=10)
        r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
        r.grid(column=0, row=5, padx=10, pady=10)
        button3 = tk.Button(text="Remedies", command=canker)
        button3.grid(column=0, row=6, padx=10, pady=10)
    elif label == 'greening':
        diseasename = "Lgreening "
        disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                           fg="Black", font=("", 15))
        disease.grid(column=0, row=4, padx=10, pady=10)
        r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
        r.grid(column=0, row=5, padx=10, pady=10)
        button3 = tk.Button(text="Remedies", command=greening)
        button3.grid(column=0, row=6, padx=10, pady=10)


    elif label == 'melanose':
        diseasename = "melanose "
        disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                           fg="Black", font=("", 15))
        disease.grid(column=0, row=4, padx=10, pady=10)
        r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
        r.grid(column=0, row=5, padx=10, pady=10)
        button3 = tk.Button(text="Remedies", command=melanose)
        button3.grid(column=0, row=6, padx=10, pady=10)

    elif label == 'Scab':
        diseasename = "Scab "
        disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                           fg="Black", font=("", 15))
        disease.grid(column=0, row=4, padx=10, pady=10)
        r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
        r.grid(column=0, row=5, padx=10, pady=10)
        button3 = tk.Button(text="Remedies", command=Scab)
        button3.grid(column=0, row=6, padx=10, pady=10)
        
    elif label == 'healthy':
        r = tk.Label(text='healthy', background="lightgreen", fg="Black",
                     font=("", 15))
        r.grid(column=0, row=4, padx=10, pady=10)
        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=7, padx=20, pady=20)
        button = tk.Button(text="Back to Home", command=healthy)
        button.grid(column=0, row=10, padx=20, pady=20)
    
    


def openphoto():
    global button2
    dirPath = "img"
    fileList = os.listdir(dirPath)
    for fileName in fileList:  
        os.remove(dirPath + "/" + fileName) 
    fileName = askopenfilename(initialdir='test dataset', title='Select image for analysis ',filetypes=[(('image    files','*.jpg','*.png'),('all files','*.*'))])
    dst = "img\\2.jpeg"
    shutil.copy(fileName, dst)
    title.destroy()
    button1.destroy()

    button2 = tk.Button(window,text="Analyse Image",bg='#0052cc', fg='#ffffff',command = analysis,height=5,width=25,font=('algerian',10,'bold'))
    button2.grid(column=0, row=500, padx=40, pady = 10)

button1 = tk.Button(text="Get image", command = openphoto,height=5,width=25,bg='#0052cc', fg='#ffffff',font=('algerian',10,'bold'))

button1.grid(column=0, row=500, padx=40, pady = 10)

window.mainloop()






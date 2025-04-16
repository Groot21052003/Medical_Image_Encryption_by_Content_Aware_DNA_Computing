from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import cv2
from hashlib import sha1
from dna import dna_decode,dna_encode, decompose_matrix
import random
from skimage.metrics import structural_similarity as ssim

main = tkinter.Tk()
main.title("Medical Image Encryption by Content-Aware DNpip install scikit-image==0.16.2A Computing for Secure Healthcare") 
main.geometry("1300x1200")

global filename
global plain_image, public_key
global encrypt_image, dna_encoding, blue_e,green_e,red_e
global h,w
global x0, random_value
global img

def upload(): 
    global filename, plain_image
    filename = filedialog.askopenfilename(initialdir="testImages")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    plain_image = cv2.imread(filename)
    plain_image = cv2.resize(plain_image, (300, 300))
    cv2.imwrite("test.png", plain_image)
    plain_image = cv2.imread("test.png")
    filename = "test.png"
    cv2.imshow("Image Loaded", plain_image)
    cv2.waitKey(0)

#get Random value for pixel encoding using SHA256
def generateRandomValue():
    global x0, random_value
    global plain_image
    global encrypt_image
    sha = sha1(plain_image).hexdigest()
    x0 = ord(sha[0]) ^ ord(sha[1]) ^ ord(sha[2]) ^ ord(sha[3]) ^ ord(sha[4]) ^ ord(sha[5]) ^ ord(sha[6]) ^ ord(sha[7]) ^ ord(sha[8]) ^ ord(sha[9]) ^ ord(sha[10]) ^ ord(sha[11]) ^ord(sha[12]) ^ ord(sha[13]) ^ord(sha[14]) ^ ord(sha[15])
    random_value = ord(sha[16]) ^ ord(sha[17]) ^ ord(sha[18]) ^ ord(sha[19]) ^ ord(sha[20]) ^ ord(sha[21]) ^ ord(sha[22]) ^ ord(sha[23]) ^ ord(sha[24]) ^ ord(sha[25]) ^ ord(sha[26]) ^ ord(sha[27]) ^ ord(sha[28]) ^ ord(sha[29]) ^ ord(sha[30]) ^ ord(sha[31])
    x0 = x0 / random.randrange(150, 255) #generate random encoding for each pixel
    random_value = int(random_value / 510)
    text.insert(END,"PWLCM Random Pixel Encoding Value : "+str(random_value + x0)+"\n\n")    

def dnaEncoding(): #function to apply dna encoding on image
    text.delete('1.0', END)
    global filename, plain_image, dna_encoding, blue_e,green_e,red_e
    blue,green,red = decompose_matrix(filename) #get al colour pixels from given file image
    blue_e,green_e,red_e = dna_encode(blue,green,red) #now apply DNA encoding on all pixels
    dna_encoding = np.dstack((red_e,green_e,blue_e))#convert DNA encoded pixels into image
    text.insert(END,"DNA Encoded pixels\n")
    text.insert(END,str(dna_encoding)+"\n\n")
    print(dna_encoding.shape)    
   
def correlation(original, encrypted):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    encrypted = cv2.cvtColor(encrypted, cv2.COLOR_BGR2GRAY)
    original = cv2.resize(original, (100, 100))
    encrypted = cv2.resize(encrypted, (100, 100))
    print(original.shape)
    print(encrypted.shape)
    corr = ssim(original, encrypted, data_range = encrypted.max() - encrypted.min())
    return corr

def runEncryption():#function to encrypt DNA encoded image
    text.delete('1.0', END)
    global dna_encoding, h, w, random_value, encrypt_image, plain_image, public_key
    generateRandomValue()
    h = dna_encoding.shape[0]
    w = dna_encoding.shape[1]
    public_key = random.randrange(29, 31)
    for y in range(0, h):
        for x in range(0, w):
            img1 = dna_encoding[y,x,0]
            img2 = dna_encoding[y,x,1]
            img3 = dna_encoding[y,x,2]
            dna_encoding[y,x,0] = ord(img1) ^ random_value #XOR operations
            dna_encoding[y,x,1] = ord(img2) ^ random_value
            dna_encoding[y,x,2] = ord(img3) ^ random_value
    encrypt_image = dna_encoding
    encrypt_image = encrypt_image.astype(int) * public_key
    print(encrypt_image)
    cv2.imwrite("test.png", encrypt_image)
    corr = correlation(plain_image, cv2.imread("test.png"))
    text.insert(END,"Propose Algorithm Image Correlation : "+str(corr)+"\n\n")
    figure, axis = plt.subplots(nrows=1, ncols=3,figsize=(10,10))
    axis[0].set_title("Original Image")
    axis[1].set_title("Encrypted Image")
    axis[2].set_title("Histogram")
    axis[0].imshow(cv2.cvtColor(plain_image, cv2.COLOR_BGR2RGB))
    axis[1].imshow(encrypt_image/255)
    axis[2].hist(encrypt_image.ravel(),256,[0,256])
    figure.tight_layout()
    plt.show()     

def runDecryption():
    global encrypt_image, public_key, random_value, dna_encoding
    global blue_e,green_e,red_e
    enc = dna_encoding
    h = enc.shape[0]
    w = enc.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            img1 = int(enc[y,x,0])#XOR operations
            img2 = int(enc[y,x,1])
            img3 = int(enc[y,x,2])
            e1 = img1 ^ random_value
            e2 = img2 ^ random_value
            e3 = img3 ^ random_value
            enc[y,x,0] = chr(e1)
            enc[y,x,1] = chr(e2)
            enc[y,x,2] = chr(e3)#reverse decryption process
    b,g,r = dna_decode(blue_e,green_e,red_e)#DNA decoding to get normal image 
    decrypt_img = np.dstack((r,g,b))            
    figure, axis = plt.subplots(nrows=1, ncols=3,figsize=(10,10))
    axis[0].set_title("Original Image")
    axis[1].set_title("Encrypted Image")
    axis[2].set_title("Decrypted Image")
    axis[0].imshow(cv2.cvtColor(plain_image, cv2.COLOR_BGR2RGB))
    axis[1].imshow(encrypt_image/255)
    axis[2].imshow(decrypt_img)
    figure.tight_layout()
    plt.show() 

def close():
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='Medical Image Encryption by Content-Aware DNA Computing for Secure Healthcare')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Medical Image", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

randomDNAButton = Button(main, text="Run Random DNA Encoding Module", command=dnaEncoding, bg='#ffb3fe')
randomDNAButton.place(x=350,y=550)
randomDNAButton.config(font=font1) 

encButton1 = Button(main, text="Run Permutation & Encryption", command=runEncryption, bg='#ffb3fe')
encButton1.place(x=690,y=550)
encButton1.config(font=font1) 

decButton = Button(main, text="Run Decryption Algorithm", command=runDecryption, bg='#ffb3fe')
decButton.place(x=50,y=600)
decButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
exitButton.place(x=350,y=600)
exitButton.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()

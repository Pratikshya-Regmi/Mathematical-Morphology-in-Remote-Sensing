import tkinter,tkinter.filedialog
from tkinter import *
import sys
import cv2
#import io

import numpy as np

import matplotlib.pyplot as plt

root = Tk()
w = Label(text = 'MATHEMATICAL MORPHOLOGY', font = ('times', 24, 'bold'))
w.pack()


##root.withdraw()
##input_image = 'input1.png'
##print(type(input_image))
##print(input_image)


def select_image():
    global panel1
    global input_img
    input_image = tkinter.filedialog.askopenfile(parent=root,mode='rb',title='Choose a file')
    #print(type(input_image))
    input_img=input_image.name

#label1 = Label( root, text="Kernel Size")
#E1 = Entry(root, bd =5)

#def getKernel():
 #   global kernel_size
  #  kernel_size=E1.get()
    
##print (type(kernel_size))
##submit_buton=Button(root, text= "Submit",command = getKernel)
##label1.pack()
##E1.pack()
##submit_buton.pack()
   
select_image()    
def print_input():
    #print (input_img)
    logo = PhotoImage(file=input_img)
    panel1=Label(image=logo)
    panel1.image=logo
    panel1.pack(side='bottom',padx=10,pady=10)
    morphbutton.pack()
    
b=Button(root,text='Print Original Image', command=print_input).pack()




        
def morph():
    dilationbut= Button(root,text= 'Dilation',command=dilation_fun).pack(side= LEFT,padx=50)
    erosionbut= Button(root,text= 'Erosion',command=erosion_fun).pack(side= LEFT,padx=50)
    openingbut= Button(root,text= 'Opening',command=opening_fun).pack(side= LEFT,padx=50)
    closingbut= Button(root,text= 'Closing',command=closing_fun).pack(side= LEFT,padx=50)

morphbutton= Button(root,text= 'Process Morphology',command=morph)

def dilation_fun():
    a_dilated_image = dilation(new_binary_image1,kernel,image_zeros)
    img_dilation = cv2.dilate(binary, kernel, iterations=1)

    fig = plt.figure(figsize=(20,10))
    
    a=fig.add_subplot(1,3,1)
    imgplot = plt.imshow(image,cmap=plt.cm.gray)
    a.set_title('Original Image')
    
    a=fig.add_subplot(1,3,2)
    imgplot = plt.imshow(img_dilation,cmap=plt.cm.gray)
    a.set_title('Using cv2.dilate()')

    a=fig.add_subplot(1,3,3)
    imgplot = plt.imshow(a_dilated_image,cmap=plt.cm.gray)
    a.set_title('Custom Dilated')
    plt.show()

def erosion_fun():
    a_eroded_image = erosion(new_binary_image2,kernel,image_zeros)
    img_eroded = cv2.erode(binary,kernel,iterations=1)

    fig = plt.figure(figsize=(20,10))

    a=fig.add_subplot(1,3,1)
    imgplot = plt.imshow(image,cmap=plt.cm.gray)
    a.set_title('Original Image')
    
    a=fig.add_subplot(1,3,2)
    imgplot = plt.imshow(img_eroded,cmap=plt.cm.gray)
    a.set_title('Using cv2.erode()')

    a=fig.add_subplot(1,3,3)
    imgplot = plt.imshow(a_eroded_image,cmap=plt.cm.gray)
    a.set_title('Custom Eroded')
    plt.show()

def opening_fun():
    img_eroded = cv2.erode(binary,kernel,iterations=1)
    img_opening=cv2.dilate(img_eroded, kernel, iterations=1)
    custom_erosion = erosion(new_binary_image3,kernel,image_zeros)
    opened_image = dilation(custom_erosion,kernel,image_zeros)
    fig = plt.figure(figsize=(20,10))

    a=fig.add_subplot(1,3,1)
    imgplot = plt.imshow(image,cmap=plt.cm.gray)
    a.set_title('Original Image')
    
    a=fig.add_subplot(1,3,2)
    imgplot = plt.imshow(img_opening,cmap=plt.cm.gray)
    a.set_title('Using cv2.dilation(erode)')

    a=fig.add_subplot(1,3,3)
    imgplot = plt.imshow(opened_image,cmap=plt.cm.gray)
    a.set_title('Custom Opened')
    plt.show()

def closing_fun():
    img_dilation = cv2.dilate(binary, kernel, iterations=1)
    img_closing=cv2.erode(img_dilation, kernel, iterations=1)
    custom_dilation = dilation(new_binary_image4,kernel,image_zeros)
    closed_image = erosion(custom_dilation,kernel,image_zeros)
    fig = plt.figure(figsize=(20,10))

    a=fig.add_subplot(1,3,1)
    imgplot = plt.imshow(image,cmap=plt.cm.gray)
    a.set_title('Original Image')
    
    a=fig.add_subplot(1,3,2)
    imgplot = plt.imshow(img_closing,cmap=plt.cm.gray)
    a.set_title('Using cv2.erode(dilate)')

    a=fig.add_subplot(1,3,3)
    imgplot = plt.imshow(closed_image,cmap=plt.cm.gray)
    a.set_title('Custom Closed')
    plt.show()
    
#print(getKernel())
#inp = str(input_image)
#print(inp)
#print(type(inp))
#inpu=io.BufferedReader.name
#print(inpu)

#logo = PhotoImage(file=input_img)
#select_image()
image = cv2.imread(input_img,0)
#print (type(image1))

#print(type(image))

#w3= Label(text='Input Image                                                                                                                                                           Output').pack()


#w4=Label(root , image = logo).pack(side= 'right')
#w1= Label(root, text='Input Image').pack()
#w2=Label(root ,justify=LEFT, image = logo).pack()


#plt.imshow(image, cmap=plt.cm.gray)
#plt.show()
#print (image)

print(image.shape)


        
# Convert the image into a binary image if it is present in color
ret,binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

binary = ((binary)/int(255))
#print(binary)
#print(binary.shape)

original_rows,original_columns = binary.shape

new_rows,new_columns = original_rows+4, original_columns+4
new_binary_image = np.zeros((new_rows,new_columns))
#print (new_binary_image)
#print(new_binary_image.shape)

new_binary_image1 = np.zeros((new_rows,new_columns))
new_binary_image2 = np.zeros((new_rows,new_columns))
new_binary_image3 = np.zeros((new_rows,new_columns))
new_binary_image4 = np.zeros((new_rows,new_columns))

image_zeros = new_binary_image
#temporary_image = new_binary_image
new_binary_image1[2:new_rows-2,2:new_columns-2]= binary
new_binary_image2[2:new_rows-2,2:new_columns-2]= binary
new_binary_image3[2:new_rows-2,2:new_columns-2]= binary
new_binary_image4[2:new_rows-2,2:new_columns-2]= binary
#print (new_binary_image)
#print(new_binary_image.shape)
#sys.exit()

#correct
print(binary.shape,new_binary_image.shape)

#img_kernel = np.ones((5,5),np.uint8)
#print(kernel.shape)
kernel_size = input('Enter the size of Matrix(Odd Squared matrix):')
#print(type(kernel_size))
#print (kernel_size)
#int(eval(str(s)))
#getKernel()
kernel_s=int(kernel_size)
#print(type(kernel_s))
kernel = np.ones((kernel_s,kernel_s), np.uint8)
dil_rows,dil_columns = kernel.shape
# Functionn for the dilation of the image

#kernel = np.ones((5,5), np.uint8)

kernel_max_value = kernel_s*kernel_s*kernel_s
#print (kernel_max_value)



def dilation(original_image,kernel,final_image):
    for i in range(0,new_rows-dil_rows+1):
        for j in range(0,new_columns-dil_columns+1):
            temp = original_image[i:i+dil_rows, j:j+dil_columns]
            total = temp.dot(kernel)
            sum_is=(sum(sum(total)))
            #if (total.any()>0):a
            if sum_is > 0:
                final_image[i,j] = 1
            else:
                final_image[i,j] = 0
    print("dilation")
    return(final_image)


def erosion(original_image,kernel,final_image):
    for i in range(0,new_rows-dil_rows+1):
        for j in range(0,new_columns-dil_columns+1):
            temp = original_image[i:i+dil_rows, j:j+dil_columns]
            total = temp.dot(kernel)
            sum_is=(sum(sum(total)))
            if(sum_is == kernel_max_value):
                #print(sum_is)
                final_image[i,j] = 1
            else:
                final_image[i,j] = 0
    print("erosion")
    return(final_image)

root.mainloop()
sys.exit()

#dilation plotting

#a_dilated_image = dilation(new_binary_image,kernel,temporary_image
a_dilated_image = dilation(new_binary_image,kernel,image_zeros)


#kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(binary, kernel, iterations=1)

fig = plt.figure(figsize=(20,10))
a=fig.add_subplot(2,4,1)
imgplot = plt.imshow(img_dilation,cmap=plt.cm.gray)
a.set_title('using cv2.dilate()')

a=fig.add_subplot(2,4,2)
imgplot = plt.imshow(a_dilated_image,cmap=plt.cm.gray)
a.set_title('custom dilated')
plt.show()


#Erosion plotting


a_eroded_image = erosion(new_binary_image1,kernel,image_zeros)
kernel = np.ones((5,5),np.uint8)
img_eroded = cv2.erode(binary,kernel,iterations=1)

print(a_eroded_image)

fig = plt.figure()
a=fig.add_subplot(2,4,3)
imgplot = plt.imshow(img_eroded,cmap=plt.cm.gray)
a.set_title('using cv2.erode()')

a=fig.add_subplot(2,4,4)
imgplot = plt.imshow(a_eroded_image,cmap=plt.cm.gray)
a.set_title('custom eroded')
plt.show()


#plotting opening


img_opening=cv2.dilate(img_eroded, kernel, iterations=1)
a=fig.add_subplot(2,4,5)
imgplot = plt.imshow(img_opening,cmap=plt.cm.gray)
a.set_title('using cv2.dilation(erode)')
#plt.show()

custom_erosion = erosion(new_binary_image2,kernel,image_zeros)

opened_image = dilation(custom_erosion,kernel,image_zeros)
a=fig.add_subplot(2,4,6)
imgplot = plt.imshow(opened_image,cmap=plt.cm.gray)
a.set_title('custom opened')
#plt.show()


#plotting closing


img_closing=cv2.erode(img_dilation, kernel, iterations=1)
a=fig.add_subplot(2,4,7)
imgplot = plt.imshow(img_closing,cmap=plt.cm.gray)
a.set_title('using cv2.erode(dilation)')

custom_dilation = dilation(new_binary_image3,kernel,image_zeros)

closed_image = erosion(custom_dilation,kernel,image_zeros)
a=fig.add_subplot(2,4,8)
imgplot = plt.imshow(closed_image,cmap=plt.cm.gray)
a.set_title('custom closed')
plt.show()



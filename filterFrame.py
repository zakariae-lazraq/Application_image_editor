from tkinter import Toplevel, Button, RIGHT, LEFT

from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import cv2



class FilterFrame(Toplevel):


    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)
        self.original_image = self.master.processed_image
        self.filtered_image = None


        self.negative_button = Button(master=self, text="Negative",width=17,fg='white',bg='#34495E')
        #self.negative_button.grid(padx=2, pady=2)
        self.black_white_button = Button(master=self, text="Black White",width=17,fg='white',bg='#34495E')
        #self.black_white_button.grid(padx=2, pady=2)
        self.sepia_button = Button(master=self, text="Sepia",width=17,fg='white',bg='#34495E')
        #self.sepia_button.grid(padx=2, pady=2)
        self.emboss_button = Button(master=self, text="Emboss",width=17,fg='white',bg='#34495E')
        #self.emboss_button.grid(padx=2, pady=2)
        self.gaussian_blur_button = Button(master=self, text="Gaussian Blur",width=17,fg='white',bg='#34495E')
        #self.gaussian_blur_button.grid(padx=2, pady=2)
        self.median_blur_button = Button(master=self, text="Median Blur",width=17,fg='white',bg='#34495E')
        #self.median_blur_button.grid(padx=2, pady=2)
        self.detector_button = Button(master=self, text="detector Button ",width=17,fg='white',bg='#34495E')
        #self.detector_button.grid(padx=2, pady=2)
        self.brighteness_négative_button = Button(master=self, text="brighteness_négative ",width=17,fg='white',bg='#34495E')
        #self.brighteness_négative_button.grid(padx=2, pady=2)
        self.brighteness_positive_button = Button(master=self, text="brighteness_positive ",width=17,fg='white',bg='#34495E')
        #self.brighteness_positive_button.grid(padx=2, pady=2)
        self.Sharpen_button = Button(master=self, text="Sharpen  ",width=17,fg='white',bg='#34495E')
        #self.Sharpen_button.grid(padx=2, pady=2)
        self.Pencil_button = Button(master=self, text="Pencil_Sketch  ",width=17,fg='white',bg='#34495E')
        #self.Pencil_button.grid(padx=2, pady=2)
        self.Hdr_button = Button(master=self, text="HDR",width=17,fg='white',bg='#34495E')
        #self.Hdr_button.grid(padx=2, pady=2)
        self.Summer_button = Button(master=self, text="Summer",width=17,fg='white',bg='#34495E')
        #self.Summer_button.grid(padx=2, pady=2)
        self.Winter_button = Button(master=self, text="Winter",width=17,fg='white',bg='#34495E')
        #self.Winter_button.grid(padx=2, pady=2)
        self.cartoon_button = Button(master=self, text="Cartoon", width=17,fg='white',bg='#34495E')
        #self.cartoon_button.grid(padx=2, pady=2)
        self.bilateral_button = Button(master=self, text="bilateral Blur ", width=17, fg='white', bg='#34495E')
        ###################################################
        self.D2_button = Button(master=self, text="2D Convolution ", width=17, fg='white', bg='#34495E')
        #self.D2_button.grid(padx=2, pady=2)
        self.blur_button = Button(master=self, text=" blur ", width=17, fg='white', bg='#34495E')
        #self.blur_button.grid(padx=2, pady=2)
        self.cancel_button = Button(master=self, text="Cancel",bg="#555",width=7)
        #self.cancel_button.grid(padx=2, pady=2)
        self.apply_button = Button(master=self, text="Apply",bg="#555",width=7)
        #self.apply_button.grid(padx=2, pady=2)
        self.rotation_button = Button(master=self, text="180°",bg="#555",width=7)
        #self.rotation_button.grid(padx=2, pady=2)
        self.rotation_90_button = Button(master=self, text="90°", bg="#555",width=7)
        #self.rotation_90_button.grid(padx=2, pady=2)
        self.add_text_button = Button(master=self, text="Put text", bg="#555", width=7)
        #self.add_text_button.grid(padx=2, pady=2)
        self.denoising_button = Button(master=self, text="Denoising", bg="#555", width=7)
        #self.denoising_button.grid(padx=2, pady=2)
        self.Psnr_button = Button(master=self, text="PSNR", bg="#555", width=7)


        self.negative_button.bind("<ButtonRelease>", self.negative_button_released)
        self.black_white_button.bind("<ButtonRelease>", self.black_white_released)
        self.sepia_button.bind("<ButtonRelease>", self.sepia_button_released)
        self.emboss_button.bind("<ButtonRelease>", self.emboss_button_released)
        self.gaussian_blur_button.bind("<ButtonRelease>", self.gaussian_blur_button_released)
        self.median_blur_button.bind("<ButtonRelease>", self.median_blur_button_released)
        self.detector_button.bind("<ButtonRelease>", self.detector_button_released)
        self.Sharpen_button.bind("<ButtonRelease>", self.Sharpen_button_released)
        self.brighteness_négative_button.bind("<ButtonRelease>", self.brighteness_négative_button_released)
        self.brighteness_positive_button.bind("<ButtonRelease>", self.brighteness_positive_button_released)
        self.Pencil_button.bind("<ButtonRelease>", self.Pencil_button_released)
        self.Hdr_button.bind("<ButtonRelease>", self.Hdr_button_released)
        self.Summer_button.bind("<ButtonRelease>", self.Summer_button_released)
        self.Winter_button.bind("<ButtonRelease>", self.Winter_button_released)
        self.cartoon_button.bind("<ButtonRelease>", self.cartoon_button_released)
        self.bilateral_button.bind("<ButtonRelease>", self.bilateral_button_released)
        self.D2_button.bind("<ButtonRelease>", self.D2_button_released)
        self.blur_button.bind("<ButtonRelease>", self.blur_button_released)
        self.apply_button.bind("<ButtonRelease>", self.apply_button_released)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)
        self.rotation_button.bind("<ButtonRelease>", self.rotation_button_released)
        self.rotation_90_button.bind("<ButtonRelease>", self.rotation_90_button_released)
        self.add_text_button.bind("<ButtonRelease>", self.add_text_button_released)
        self.denoising_button.bind("<ButtonRelease>", self.denoising_button_released)
        self.Psnr_button.bind("<ButtonRelease>", self.Psnr_button_released)



        self.black_white_button.pack()
        self.negative_button.pack()
        self.sepia_button.pack()
        self.emboss_button.pack()
        self.gaussian_blur_button.pack()
        self.median_blur_button.pack()
        self.detector_button.pack()
        self.Sharpen_button.pack()
        self.Hdr_button.pack()
        self.Pencil_button.pack()
        self.Summer_button.pack()
        self.Winter_button.pack()
        self.brighteness_négative_button.pack()
        self.brighteness_positive_button.pack()
        self.cartoon_button.pack()
        self.bilateral_button.pack()
        self.D2_button.pack()
        self.blur_button.pack()
        self.cancel_button.pack(side=LEFT)
        self.apply_button.pack(side=LEFT)
        self.rotation_button.pack(side=RIGHT)
        self.rotation_90_button.pack(side=LEFT)
        self.add_text_button.pack(side=LEFT)
        self.denoising_button.pack(side=LEFT)
        self.Psnr_button.pack(side=LEFT)



    def negative_button_released(self, event):
        self.negative()
        self.show_image()

    def bilateral_button_released(self, event):
        self.bilateral()
        self.show_image()

    def D2_button_released(self, event):
        self.D2()
        self.show_image()

    def blur_button_released(self, event):
        self.blur()
        self.show_image()

    def black_white_released(self, event):
        self.black_white()
        self.show_image()

    def sepia_button_released(self, event):
        self.sepia()
        self.show_image()

    def emboss_button_released(self, event):
        self.emboss()
        self.show_image()

    def gaussian_blur_button_released(self, event):
        self.gaussian_blur()
        self.show_image()

    def median_blur_button_released(self, event):
        self.gaussian_blur()
        self.show_image()

    def detector_button_released(self, event):
        self.detector()
        self.show_image()

    def Sharpen_button_released(self, event):
        self.Sharpen()
        self.show_image()
    def Pencil_button_released(self,event):
        self.Pencil()
        self.show_image()

    def Hdr_button_released(self, event):
        self.Hdr()
        self.show_image()

    def Summer_button_released(self, event):
        self.Summer()
        self.show_image()

    def Winter_button_released(self, event):
        self.Winter()
        self.show_image()

    def brighteness_positive_button_released(self,event):
        self.brighteness_positive()
        self.show_image()

    def brighteness_négative_button_released(self, event):
        self.brighteness_négative()
        self.show_image()

    def lynn_button_released(self, event):
        self.lynn()
        self.show_image()


    def apply_button_released(self, event):
        self.master.processed_image = self.filtered_image
        self.show_image()
        self.close()
    def rotation_button_released(self,event):
        self.rotation()
        self.show_image()

    def rotation_90_button_released(self,event):
        self.rotation_90()
        self.show_image()
    def add_text_button_released(self,event):
        self.add_text()
        self.show_image()
    def cartoon_button_released(self,event):
        self.cartoon()
        self.show_image()
    def denoising_button_released(self,event):
        self.denoising()
        self.show_image()




    def cancel_button_released(self, event):
        self.master.image_viewer.show_image()
        self.close()

    def show_image(self):
        self.master.image_viewer.show_image(img=self.filtered_image)
    def negative(self):
        self.filtered_image = cv2.bitwise_not(self.original_image)
        '''permet d'inverser valeurs des pixels cela peut etre fait en soustraire des pixels par 255 avec
        la fonction ci dessus
        '''

    def black_white(self):
        self.filtered_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.filtered_image = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2BGR)

    def sepia(self):
        img=self.original_image
        img_sepia = np.array(img, dtype=np.float64)  # converting to float to prevent loss
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                        [0.349, 0.686, 0.168],
                                                        [0.393, 0.769,
                                                         0.189]]))  # multipying image with special sepia matrix
        img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        self.filtered_image=img_sepia
    def Sharpen(self):
        img = self.original_image
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img_sharpen = cv2.filter2D(img, -1, kernel)
        self.filtered_image=img_sharpen
    def Pencil(self):
        '''img = self.original_image
        sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
        self.filtered_image=sk_gray'''
        img=self.original_image
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(grey_img)
        blur = cv2.GaussianBlur(invert, (31, 31), 0)
        invertedBlur = cv2.bitwise_not(blur)
        sketch = cv2.divide(grey_img, invertedBlur, scale=256.0)
        self.filtered_image=sketch

    def Hdr(self):
        '''permet de augmenter le niveau de detail de l'image avec la fonction ci dessous
        _Sigma_s: détermine la quantité de lisage  1 filtre de lissage remplace chaque pixel par la somme
        ponderré de ses voisins si le voisinage d'image est grand ,l'image filtré est lisse ,pour cela n'utilise
        sigma_r qui permet de remplacer 1 px par la moyenne des pixels des voisins
        _shade_factor: range 0 to 1 , représente échelle de intensité d'image ,si plus élevé la résultat est brillant


        '''

        img = self.original_image
        hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
        self.filtered_image = hdr


    def LookupTable(self,x,y):
        spline = UnivariateSpline(x,y)
        return spline(range(256))
        '''en utilisant la fonction unvariate spline pour entrainer les pixels
        pour implementer summer effect il faut incrémenter la valeur de rouge et décrementer la valeurde bleu
        et pour appliquer winter effect en fait inverse
        '''

    def Summer(self):
        img=self.original_image
        increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        sum1 = cv2.merge((blue_channel, green_channel, red_channel))
        self.filtered_image=sum1

    def Winter(self):
        img = self.original_image
        increaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        win = cv2.merge((blue_channel, green_channel, red_channel))
        self.filtered_image =win
    def emboss(self):
        kernel = np.array([[0, -1, -1],
                           [1, 0, -1],
                           [1, 1, 0]])

        self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)

    def gaussian_blur(self):
        self.filtered_image = cv2.GaussianBlur(self.original_image, (41, 41), 0)
        ''' Gaussien Blur : image est convolu par un motif qui supprime les pixels plus grands
            en utilise la fonction ci dessus'''

    def median_blur(self):
        self.filtered_image = cv2.medianBlur(self.original_image, 41)
    def detector (self):
        self.filtered_image = cv2.Canny(self.original_image, 100, 200)

    def brighteness_négative(self):
        img = self.original_image
        self.filtered_image = cv2.convertScaleAbs(img, beta=-60)

    def brighteness_positive(self):
        img=self.original_image
        self.filtered_image  = cv2.convertScaleAbs(img, beta=60)
    def rotation(self):
        image = self.filtered_image
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), 360 // 2, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        self.filtered_image=rotated

    def rotation_90(self):
        image = self.filtered_image
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        self.filtered_image = rotated
    def cartoon(self):
        img = self.original_image
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
        edgeImg = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
        output = cv2.bitwise_and(img, img, mask=edgeImg)
        self.filtered_image =output

    def bilateral(self):
        self.filtered_image = cv2.bilateralFilter(self.original_image, 9, 75, 75)

    def D2(self):
        kernel = np.ones((5, 5), np.float32) / 25
        self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)

    def blur(self):
        self.filtered_image = cv2.blur(self.original_image, (5, 5))

    def add_text(self):
        img=self.filtered_image

        font = cv2.FONT_HERSHEY_SIMPLEX

        i = 10
        while (1):
            cv2.imshow('img', img)

            k = cv2.waitKey(33)
            if k == 27:  # Esc key to stop
                break
            elif k == -1:  # normally -1 returned,so don't print it
                continue
            else:
                print(k)  # else print its value
            cv2.putText(img, chr(k), (i, 50), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            i += 15
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    def denoising(self):
        img=self.original_image
        s=img.shape
        dn = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        self.filtered_image=dn
    def close(self):
        self.destroy()

    def Psnr(self,original, compressed):
        from math import log10, sqrt
        mse = np.mean((original - compressed) ** 2)
        if (mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def Psnr_button_released(self,event):
        original = self.original_image
        median = self.filtered_image
        compressed = median
        value = self.Psnr(original, compressed)
        print(f"PSNR value is {value} dB")

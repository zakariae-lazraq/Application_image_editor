from tkinter import Toplevel, Label, Scale, Button, HORIZONTAL, RIGHT
import cv2
from PIL import Image


class AdjustFrame(Toplevel):

    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        self.brightness_value = 0
        self.previous_brightness_value = 0

        self.original_image = self.master.processed_image
        self.processing_image = self.master.processed_image

        self.brightness_label = Label(self, text="Brightness")
        self.brightness_scale = Scale(self, from_=0, to_=2, length=250, resolution=0.1,
                                      orient=HORIZONTAL)
        self.r_label = Label(self, text="R")
        self.r_scale = Scale(self, from_=-100, to_=100, length=250, resolution=1,
                             orient=HORIZONTAL)
        self.g_label = Label(self, text="G")
        self.g_scale = Scale(self, from_=-100, to_=100, length=250, resolution=1,
                             orient=HORIZONTAL)
        self.b_label = Label(self, text="B")
        self.b_scale = Scale(self, from_=-100, to_=100, length=250, resolution=1,
                             orient=HORIZONTAL)
        self.apply_button = Button(self, text="Apply")
        self.preview_button = Button(self, text="Preview")
        self.zoom_button = Button(self, text="Zoom")
        self.background_button=Button(self,text="Background")
        self.concat_tile_button = Button(self, text="Concat")
        self.flip_button = Button(self, text="Flip")
        self.cancel_button = Button(self, text="Cancel")
        self.brightness_scale.set(1)

        self.apply_button.bind("<ButtonRelease>", self.apply_button_released)
        self.preview_button.bind("<ButtonRelease>", self.show_button_release)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)
        self.zoom_button.bind("<ButtonRelease>", self.zoom_button_released)
        self.background_button.bind("<ButtonRelease>", self.background_button_released)
        self.concat_tile_button.bind("<ButtonRelease>", self.concat_tile_button_released)
        self.flip_button.bind("<ButtonRelease>", self.flip_button_released)
        self.brightness_label.pack()
        self.brightness_scale.pack()
        self.r_label.pack()
        self.r_scale.pack()
        self.g_label.pack()
        self.g_scale.pack()
        self.b_label.pack()
        self.b_scale.pack()
        self.cancel_button.pack(side=RIGHT)
        self.preview_button.pack(side=RIGHT)
        self.zoom_button.pack(side=RIGHT)
        self.background_button.pack(side=RIGHT)
        self.concat_tile_button.pack(side=RIGHT)
        self.flip_button.pack(side=RIGHT)

        self.apply_button.pack(side=RIGHT)

    def apply_button_released(self, event):
        self.master.processed_image = self.processing_image
        self.close()

    def show_button_release(self, event):
        self.processing_image = cv2.convertScaleAbs(self.original_image, alpha=self.brightness_scale.get())
        b, g, r = cv2.split(self.processing_image)

        for b_value in b:
            cv2.add(b_value, self.b_scale.get(), b_value)
        for g_value in g:
            cv2.add(g_value, self.g_scale.get(), g_value)
        for r_value in r:
            cv2.add(r_value, self.r_scale.get(), r_value)

        self.processing_image = cv2.merge((b, g, r))
        self.show_image(self.processing_image)

    def cancel_button_released(self, event):
        self.close()
    def zoom_button_released(self, event):
        self.processing_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(self.processing_image)
        im_pil.show()


    def background_button_released(self,event):
        # variable
        RED = 255
        GREEN = 255
        BLUE = 0
        ALPHA = 200
        img=cv2.imread(self.original_image)
        Image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        trasn_mask = Image[:, :, 3] == 0
        Image[trasn_mask] = [BLUE, GREEN, RED, ALPHA]
        resized = cv2.resize(Image, None, fx=0.1, fy=0.1)
        self.processing_image=resized
        self.show_image(self.processing_image)

    def concat_tile(self,im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
    def concat_tile_button_released(self,event):
        im1 = self.original_image
        im1_s = cv2.resize(im1, dsize=(0, 0), fx=0.5, fy=0.5)
        im_tile = self.concat_tile([[im1_s, im1_s, im1_s, im1_s],
                           [im1_s, im1_s, im1_s, im1_s],
                           [im1_s, im1_s, im1_s, im1_s]])
        self.processing_image=im_tile
        self.show_image(self.processing_image)
    def flip_button_released(self,event):
        img=self.original_image
        image = cv2.flip(img, 1)
        self.processing_image = image
        self.show_image(self.processing_image)

    def show_image(self, img=None):
        self.master.image_viewer.show_image(img=img)

    def close(self):
        self.show_image()
        self.destroy()



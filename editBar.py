from tkinter import *
from tkinter import Frame, Button, LEFT
from tkinter import filedialog,messagebox
from filterFrame import FilterFrame
from adjustFrame import AdjustFrame
from tkinter import font as tkFont  # for convenience
import random
import numpy as np
import cv2
import PIL.Image
import PIL.ImageTk
from PIL import Image



class EditBar(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        helv36 = tkFont.Font(family='Oswald')

        self.new_button = Button(self, text="New",fg='white',bg='#34495E')
        self.new_button['font']=helv36
        self.save_button = Button(self, text="Save",fg='white',bg='#34495E')
        self.save_button['font']=helv36
        self.save_as_button = Button(self, text="Save As",fg='white',bg='#34495E')
        self.save_as_button['font']=helv36
        self.draw_button = Button(self, text="Draw",fg='white',bg='#34495E')
        self.draw_button['font']=helv36
        self.crop_button = Button(self, text="Crop",fg='white',bg='#34495E')
        self.crop_button['font']=helv36
        self.filter_button = Button(self, text="Filter",fg='white',bg='#34495E')
        self.filter_button['font']=helv36
        self.adjust_button = Button(self, text="Adjust",fg='white',bg='#34495E')
        self.adjust_button['font']=helv36
        self.clear_button = Button(self, text="Clear",fg='white',bg='#34495E')
        self.clear_button['font']=helv36
        self.Denoise_button = Button(self, text="Denoise", fg='white', bg='#34495E')
        self.Denoise_button['font'] = helv36

        self.new_button.bind("<ButtonRelease>", self.new_button_released)
        self.save_button.bind("<ButtonRelease>", self.save_button_released)
        self.save_as_button.bind("<ButtonRelease>", self.save_as_button_released)
        self.draw_button.bind("<ButtonRelease>", self.draw_button_released)
        self.crop_button.bind("<ButtonRelease>", self.crop_button_released)
        self.filter_button.bind("<ButtonRelease>", self.filter_button_released)


        self.adjust_button.bind("<ButtonRelease>", self.adjust_button_released)
        self.clear_button.bind("<ButtonRelease>", self.clear_button_released)
        self.Denoise_button.bind("<ButtonRelease>", self.Denoise_button_released)

        self.new_button.pack(side=LEFT)
        self.save_button.pack(side=LEFT)
        self.save_as_button.pack(side=LEFT)
        self.draw_button.pack(side=LEFT)
        self.crop_button.pack(side=LEFT)

        self.filter_button.pack(side=LEFT)


        self.adjust_button.pack(side=LEFT)
        self.clear_button.pack(side=RIGHT)
        self.Denoise_button.pack()
        self.ImageIsSelected = False




    def new_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.new_button:
            if self.master.is_draw_state:
                self.master.image_viewer.deactivate_draw()
            if self.master.is_crop_state:
                self.master.image_viewer.deactivate_crop()

            filename = filedialog.askopenfilename()
            image = cv2.imread(filename)

            if image is not None:
                self.master.filename = filename
                self.master.original_image = image.copy()
                self.master.processed_image = image.copy()
                self.master.image_viewer.show_image()
                self.master.is_image_selected = True

    def save_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()

                save_image = self.master.processed_image
                image_filename = self.master.filename
                cv2.imwrite(image_filename, save_image)

    def save_as_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_as_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                original_file_type = self.master.filename.split('.')[-1]
                filename = filedialog.asksaveasfilename()
                filename = filename + "." + original_file_type

                save_image = self.master.processed_image
                cv2.imwrite(filename, save_image)

                self.master.filename = filename

    def draw_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.draw_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                else:
                    self.master.image_viewer.activate_draw()



    def crop_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.crop_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                else:
                    self.master.image_viewer.activate_crop()


    def filter_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.filter_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                self.master.filter_frame = FilterFrame(master=self.master)
                self.master.filter_frame.grab_set()

    def adjust_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.adjust_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                self.master.adjust_frame = AdjustFrame(master=self.master)
                self.master.adjust_frame.grab_set()



    def clear_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.clear_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                self.master.processed_image = self.master.original_image.copy()
                self.master.image_viewer.show_image()
    def Denoise_button_released(self, event):
        import webbrowser
        a=webbrowser.open("http://localhost:8501/")
        return a





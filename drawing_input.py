# opens a tkinter sketch pad and converts a user drawn digit to a 28x28 matrix (grayscale) 
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw, ImageOps

class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.save_posn)
        self.bind("<B1-Motion>", self.add_line)
        
        self.img = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.img)
        self.matrix = None  # used for storing the matrix after root is destroyed

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        # put user drawing in background tag
        self.create_line(self.lastx, self.lasty, event.x, event.y, tags="stroke", width=20, smooth=True, capstyle=ROUND)
        self.draw.line((self.lastx, self.lasty, event.x, event.y),fill=0, width=20) # draw on the hidden PIL image as well
        self.save_posn(event)
        # Ensure the button and border stays on top
        self.tag_raise("border")
        self.tag_raise("submit")
        self.tag_raise("reset")
    def reset(self):
        self.delete("stroke")
        self.img = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.img)
    def get_matrix(self):
        small = self.img.resize((28, 28), Image.LANCZOS) # downsample to 28x28
        arr = np.array(small) # convert to numpy array
        arr = 255 - arr # invert so black=1, white=0

        return arr
def run():
    root = Tk()
    root.geometry("310x310")
    root.minsize(310, 310)
    root.minsize(310, 310)
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    sketch = Sketchpad(root)
    sketch.grid(row=0, column=0, sticky="nsew")

    # Create submit button
    sketch.create_rectangle(0, 0, 30, 30, fill="green", tags="submit")
    sketch.create_line(10, 25, 25, 10, fill="white", width=3, tags="submit")
    sketch.create_line(10, 25, 5, 20, fill="white", width=3, tags="submit")
    # Create reset button
    sketch.create_rectangle(280, 0, 310, 30, fill="red", tags="reset")
    sketch.create_line(285, 5, 305, 25, fill="white", width=3, tags="reset")
    sketch.create_line(285, 25, 305, 5, fill="white", width=3, tags="reset")
    # Create a border around a 280x280 area
    sketch.create_rectangle(0, 0, 30, 310, fill="black", tags="border") # left border
    sketch.create_rectangle(0, 0, 310, 30, fill="black", tags="border") # top border
    sketch.create_rectangle(280, 0, 310, 310, fill="black", tags="border") # right border
    sketch.create_rectangle(0, 280, 310, 310, fill="black", tags="border") # bottom border

    sketch.tag_raise("border")
    sketch.tag_raise("submit")
    sketch.tag_raise("reset")

    def reset_canvas(event):
        sketch.reset()
    def get_matrix(event):
        matrix = sketch.get_matrix()
        np.set_printoptions(linewidth=500)
        sketch.matrix = matrix
        root.destroy()
        
    sketch.tag_bind("submit", "<Button-1>", get_matrix)
    sketch.tag_bind("reset", "<Button-1>", reset_canvas)
    root.mainloop()
    matrix = sketch.matrix
    if matrix is not None:
        return matrix

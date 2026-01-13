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
        self.tag_raise("label")
    def reset(self):
        self.delete("stroke")
        self.img = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.img)
    def get_matrix(self):
        small = self.img.resize((28, 28), Image.LANCZOS) # downsample to 28x28
        arr = np.array(small) # convert to numpy array
        arr = 255 - arr # invert so black=1, white=0

        return arr
def run(network):
    root = Tk()
    root.geometry("310x310")
    root.minsize(310, 310)
    root.minsize(310, 310)
    root.title("Digit Classifier")
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    sketch = Sketchpad(root)
    sketch.grid(row=0, column=0, sticky="nsew")

    # Create submit button
    sketch.create_rectangle(0, 280, 30, 310, fill="green", tags="submit")
    sketch.create_line(10, 305, 25, 290, fill="white", width=3, tags="submit")
    sketch.create_line(10, 305, 5, 300, fill="white", width=3, tags="submit")
    # Create reset button
    sketch.create_rectangle(280, 280, 310, 310, fill="red", tags="reset")
    sketch.create_line(285, 285, 305, 305, fill="white", width=3, tags="reset")
    sketch.create_line(285, 305, 305, 285, fill="white", width=3, tags="reset")
    # Create a border around a 280x280 area
    sketch.create_rectangle(0, 0, 30, 310, fill="black", tags="border") # left border
    sketch.create_rectangle(0, 0, 310, 30, fill="black", tags="border") # top border
    sketch.create_rectangle(280, 0, 310, 310, fill="black", tags="border") # right border
    sketch.create_rectangle(0, 280, 310, 310, fill="black", tags="border") # bottom border

    # Text label at the bottom center
    result_text = sketch.create_text(
        155, 15,
        text="Draw a digit (0-9)",
        fill="white",
        font=("Arial", 12, "bold"),
        tags="label"
    )

    sketch.tag_raise("border")
    sketch.tag_raise("submit")
    sketch.tag_raise("reset")
    sketch.tag_raise("label")

    def reset_canvas(event):
        sketch.reset()
        sketch.itemconfig(result_text, text="Draw a digit (0-9)")
    def get_matrix(event):
        matrix = sketch.get_matrix()
        np.set_printoptions(linewidth=500)
        sketch.matrix = matrix
        
        o = network.apply(matrix)
        confidence, prediction = find_winner(o)
        sketch.itemconfig(result_text, text=f"Class: {prediction}, Confidence: {(confidence * 100):.2f}%")

        #root.destroy() # closes the window
        
    sketch.tag_bind("submit", "<Button-1>", get_matrix)
    sketch.tag_bind("reset", "<Button-1>", reset_canvas)
    root.mainloop()
    matrix = sketch.matrix
    if matrix is not None:
        return matrix

# Take highest value as the winner
def find_winner(o):
    high = o[0]
    high_index = 0
    for i in range(len(o)):
        if o[i] > high:
            high = o[i]
            high_index = i

    return high, high_index

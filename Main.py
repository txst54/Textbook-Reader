import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import pytesseract
import cv2
from PIL import ImageTk, Image
from threading import *
import queue
from time import sleep

root = Tk()
root.title("Page Summarizer")
frame = Frame(root)
canvas = Canvas(frame)
scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

canvas.configure(yscrollcommand=scrollbar.set, width=720, height=720)
frame.pack()
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
pytesseract.pytesseract.tesseract_cmd = r'C:\\Tesseract-OCR\\tesseract.exe'
intext = ""

def readPoints(inImg):
    global intext
    img = cv2.imread(inImg)
    string = str(pytesseract.image_to_string(img, lang="eng", config="--psm 1"))
    intext = string
    Display.delete(1.0, END)
    Display.insert(END, string)


def upload():
    global filename
    yourname = askopenfilename(title="Import Image to Summarize", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("TIFF Image", "*.tif")])
    filename = yourname
    img = Image.open(yourname)
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((420, 594), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    panel = Label(scrollable_frame, image=img)

    # set the image as img
    panel.image = img
    panel.pack(side=TOP)


def splitter(n, s, l):
    #print(f"len of s: {len(s)}")
    #print(f"len of l: {len(l)}")
    if len(s) < n:
        l.append(" ".join(s))
        #print(len(l))
        return l
    else:
        l.append(" ".join(s[1:n-1]))
        s = s[n-1:]
        return splitter(n, s, l)


def Savetext():
    global intext
    intext = Display.get(1.0, END)


def Main():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    pieces = intext.split()
    outext = splitter(100, pieces, [])
    ZaOutput = []
    for text in outext:
        preprocess_text = text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        #print ("original text preprocessed: \n", preprocess_text)

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


        # summmarize
        summary_ids = model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=7,
                                            max_length=25,
                                            early_stopping=False)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        ZaOutput.append(output)
        #print ("\n\nSummarized text: \n",output)
    ZaOutput = " ".join(ZaOutput)
    SummarizeDisplay.delete(1.0, END)
    SummarizeDisplay.insert(END, ZaOutput)

Upload = Button(scrollable_frame, text="Upload Image", command=upload)
Upload.pack(side=TOP, pady=10)
SummarizeDisplay = Text(scrollable_frame, height=10, width=90)
SummarizeDisplay.insert(END, "")
SummarizeDisplay.pack(side=BOTTOM, pady=10)
Summarize = Button(scrollable_frame, text="Summarize", command=Main)
Summarize.pack(side=BOTTOM, pady=10)
Save = Button(scrollable_frame, text="Save Text", command=Savetext)
Save.pack(side=BOTTOM, pady=10)
Display = Text(scrollable_frame, height=10, width=90)
Display.insert(END, "")
Display.pack(side=BOTTOM, pady=10)
Scan = Button(scrollable_frame, text="Scan Image", command=lambda: readPoints(filename))
Scan.pack(side=BOTTOM, pady=10)
root.mainloop()

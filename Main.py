import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from tkinter.filedialog import askopenfilename
from Tesseract import readImg

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')
tess = readImg()

filename = askopenfilename(title="Import Image to Summarize", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("TIFF Image", "*.tif")])
print(filename)

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


intext = tess.readPoints(inImg=str(filename))
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
print(ZaOutput)
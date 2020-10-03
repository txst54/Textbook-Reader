import cv2
import pytesseract


class readImg:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Tesseract-OCR\\tesseract.exe'

    def readPoints(self, inImg):
        img = cv2.imread(inImg)
        string = str(pytesseract.image_to_string(img, lang="eng", config="--psm 1"))
        return string


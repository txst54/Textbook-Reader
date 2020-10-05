import cx_Freeze
import sys

base = None

if sys.platform == 'win32':
    base = "Win32GUI"

executables = [cx_Freeze.Executable("PageSummarizer.py", base=base, icon="TextbookIcon.ico")]

cx_Freeze.setup(
    name="PageSummarizer",
    options={"build_exe": {"packages": ["tkinter", "torch", "transformers", "multiprocessing", "pytesseract", "cv2", "PIL"], "include_files": ["TextbookIcon.ico"]}},
    version="0.01",
    description="Summarizes an image of text for efficiency",
    executables=executables
)

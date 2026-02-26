import os
from PyPDF2 import PdfMerger

FOLDER_PATH = r"C:\Users\liu17\Documents\billing_checker\notes"  # <-- change this
OUTPUT_FILE = os.path.join(FOLDER_PATH, "merged_notesf.pdf")

merger = PdfMerger()

for filename in sorted(os.listdir(FOLDER_PATH)):
    if filename.lower().endswith(".pdf"):
        merger.append(os.path.join(FOLDER_PATH, filename))

merger.write(OUTPUT_FILE)
merger.close()

print("Merged PDF saved to:", OUTPUT_FILE)

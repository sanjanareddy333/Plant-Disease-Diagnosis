from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from textwrap import wrap
import os

def wrap_text(text, width=90):
    return wrap(text, width)

def generate_pdf_report(output_path, image_path, gradcam_path, plant, disease, confidence, cause, natural, pesticide):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    margin = 50
    cursor = height - margin

    def draw_heading(text, size=16):
        nonlocal cursor
        c.setFont("Helvetica-Bold", size)
        c.drawString(margin, cursor, text)
        cursor -= 25

    def draw_paragraph(title, content, bold=False):
        nonlocal cursor
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, cursor, title)
        cursor -= 15
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, 11)
        lines = wrap_text(content)
        for line in lines:
            c.drawString(margin + 20, cursor, line)
            cursor -= 14
        cursor -= 10

    # Title
    draw_heading(" Plant Disease Diagnosis Report", 16)

    # Plant Info
    c.setFont("Helvetica", 12)
    c.drawString(margin, cursor, f"Plant: {plant}")
    cursor -= 15
    c.drawString(margin, cursor, f"Disease: {disease}")
    cursor -= 15
    c.drawString(margin, cursor, f"Confidence: {confidence}%")
    cursor -= 25

    # Cause and Remedies
    draw_paragraph(" Cause:", cause)
    draw_paragraph(" Natural Remedy:", natural, bold=True)
    draw_paragraph(" Pesticide Remedy:", pesticide, bold=True)

    # Images
    try:
        c.drawImage(ImageReader(image_path), margin, cursor - 140, width=200, height=120)
        c.drawImage(ImageReader(gradcam_path), margin + 220, cursor - 140, width=200, height=120)
        cursor -= 160
    except Exception as e:
        print("Error drawing image:", e)

    c.save()
from flask import Flask, render_template, request, send_file
import os
import requests
import yfinance as yf
from Presentation.PresentLatex import calc_Ratios_with_growth, create_presentation, process

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        symbol = request.form.get("symbol")
        pdf_link=request.form.get("pdf_file")

        output_path = process_f(symbol,pdf_link)
        return send_file(output_path, as_attachment=True)

    return render_template("index.html")

def process_f(symbol,pdf):
    output_presentation=process(symbol,pdf)
    # Create presentation


    return 'financial_presentation.pptx'

if __name__ == "__main__":
    app.run(debug=True)

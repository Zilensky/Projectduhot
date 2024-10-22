import subprocess

import matplotlib.pyplot as plt
import yfinance as yf
import requests
from pylatex import Document, Section, Command, Figure, Package, Tabular
from pylatex.utils import NoEscape, bold

from Anlysis.Finnancial_Ratios.Calc_Altman import CalcAltman
from Anlysis.Finnancial_Ratios.scraperyahoo import calc_Ratios_with_growth
from Anlysis.VerticalAnlysis.Vertical import retriveData, vertical


def generate_line_chart():
    # Generate a simple line chart
    x = [1, 2, 3, 4]
    y1 = [10, 15, 13, 17]
    y2 = [5, 7, 8, 9]

    plt.plot(x, y1, label='Line 1')
    plt.plot(x, y2, label='Line 2')

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Simple Line Chart')
    plt.legend()

    # Save the chart as an image
    plt.savefig('line_chart.png')
    plt.close()


def create_presentation(ratios,symbol):
    # Create a beamer document for the presentation
    doc = Document(documentclass=NoEscape("beamer"))

    # Title page
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('xcolor'))  # Package for color
    doc.packages.append(Package('tcolorbox'))

    company_name = "Yair company"

    # Add title, author, and date
    doc.preamble.append(Command('title', company_name))  # Title for the presentation
    doc.preamble.append(Command('author', 'Your Name'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))

    # Begin the document


    # Create a slide for the title card with an orange background and white text


    # Make title slide


    # Section 1: Introduction (1 page)
    with doc.create(Section('Title')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{}'))

        # Company name and sector
        doc.append(bold("Company Name: "))
        doc.append(symbol)  # Company Name in Hebrew
        doc.append(bold(" Industry: "))
        doc.append("Construction")  # Industry in Hebrew

        # Insert the downloaded image
        with doc.create(Figure(position='h!')) as graph:
            graph.add_image(image_path, width='30mm')  # Adjust the width as needed

        doc.append(NoEscape(r'\end{frame}'))
    with doc.create(Section('Introduction')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Introduction}'))


        # Insert the line chart image
        with doc.create(Figure(position='h!')) as graph:
            graph.add_image('intro.png', width='100mm')  # Add the image generated earlier

        doc.append(NoEscape(r'\end{frame}'))


    with doc.create(Section('Financial Ratios')):
        # Define frames for each category of financial ratios
        categories = {
            "Liquidity Ratios": {
                "Current Ratio": ratios.get('current_ratio', 0),
                "Quick Ratio": ratios.get('quick_ratio', 0),
                "Immediate Liquidity Ratio": ratios.get('immediate_liquidity_ratio', 0),
                "Cashflow to Sales Ratio": ratios.get('cashflow_to_sales_ratio', 0)
            },
            "Profitability Ratios": {
                "Net Profit Margin": ratios.get('net_profit_margin', 0),
                "Operating Profit Margin": ratios.get('operating_profit_margin', 0),
                "EBITDA Ratio": ratios.get('ebitda_ratio', 0),
                "Return on Equity (ROE)": ratios.get('roe', 0),
                "Return on Assets (ROA)": ratios.get('roa', 0)
            },
            "Leverage Ratios": {
                "Leverage Ratio": ratios.get('leverage_ratio', 0),
                "Equity to Assets Ratio": ratios.get('equity_to_assets_ratio', 0)
            },
            "Efficiency Ratios": {
                "Receivables Ratio": ratios.get('receivables_ratio', 0),
                "Customers Ratio": ratios.get('customers_ratio', 0),
                "Inventory Ratio": ratios.get('inventory_ratio', 0),
                "Inventory Turnover Ratio": ratios.get('inventory_turnover_ratio', 0),
                "Inventory Days": ratios.get('inventory_days', 0),
                "Payables Days": ratios.get('payables_days', 0)
            }
        }
        # Iterate over each category and create a frame for it
        for category, ratios in categories.items():
            doc.append(NoEscape(r'\begin{frame}'))
            doc.append(NoEscape(fr'\frametitle{{{category}}}'))
            doc.append(f"This slide presents the calculated {category.lower()}:")

            # Format the ratios in LaTeX itemize environment
            doc.append(NoEscape(r'\begin{itemize}'))
            for key, value in ratios.items():
                rounded_value = f"{value:.2f}"
                doc.append(NoEscape(fr'\item \textbf{{{key.replace("_", " ").title()}}}: {rounded_value}'))
            doc.append(NoEscape(r'\end{itemize}'))

            doc.append(NoEscape(r'\end{frame}'))

   #Altman score
    z_score, status = CalcAltman(symbol)
    with doc.create(Section('Altman Z-Score')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Altman Z-Score}'))
        doc.append(f"Altman Z-Score for {symbol}: {z_score:.2f} \n")
      #  doc.append(f"Financial Status: {status.title()} \n")
        doc.append(NoEscape(r'\end{frame}'))
    # Section 5: Visualizations (Simple Line Graph)
    result_analysis = retriveData(stock)
    doc.preamble.append(Package('booktabs'))
    with doc.create(Section('RESULTS')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{RESULTS}'))

        # Ensure the result_analysis content is appended as LaTeX
        doc.append(NoEscape(result_analysis))

        doc.append(NoEscape(r'\end{frame}'))
    vertical_anlysis=vertical(stock)
    with doc.create(Section('Vertical')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Vertical}'))

        # Ensure the result_analysis content is appended as LaTeX
        doc.append(NoEscape(vertical_anlysis))

        doc.append(NoEscape(r'\end{frame}'))
    with doc.create(Section('Analysis')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Analysis}'))


        # Insert the line chart image
        with doc.create(Figure(position='h!')) as graph:
            graph.add_image('Ratios_Anlysis.png', width='80mm')  # Add the image generated earlier

        doc.append(NoEscape(r'\end{frame}'))
    with doc.create(Section('Directorial Report')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Directorial Report}'))


        # Insert the line chart image
        with doc.create(Figure(position='h!')) as graph:
            graph.add_image('Directorion.png', width='80mm')  # Add the image generated earlier

        doc.append(NoEscape(r'\end{frame}'))


    # Section 8: Conclusions
    with doc.create(Section('Market Capitalization')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Conclusions}'))
        with doc.create(Figure(position='h!')) as graph:
            graph.add_image('valuation_comparison.png', width='80mm')  # Add the image generated earlier

        doc.append(NoEscape(r'\end{frame}'))

    # Generate the LaTeX file and PDF
    doc.generate_tex('presentation')

    # Run the pdflatex command to compile the .tex file manually
    try:
        subprocess.run(['pdflatex', 'presentation.tex'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

image_url = "https://mayafiles.tase.co.il/logos/he-IL/000373.jpg"
image_path = "company_logo.jpg"
response = requests.get(image_url)
with open(image_path, 'wb') as file:
    file.write(response.content)
# Generate the line chart image
generate_line_chart()
symbol='AURA.TA'
stock = yf.Ticker(symbol)
ratios=calc_Ratios_with_growth(stock,symbol)
# Create the presentation
create_presentation(ratios,symbol)

def exa(doc):
    with doc.create(Section('Company Name')):

        doc.append(NoEscape(r'''
         \begin{frame}{ Open-SQL Framework - Results}
    \begin{itemize}
        \item \textbf{Tested Models:}
        \begin{itemize}
            \item Llama2-7B and Code-Llama-7B
            \item Fine-tuned using proposed COT (SFTI-COT-SK-FULL)
            \item Zero-Shot learning
        \end{itemize}

   
        \item \textbf{Performance improvement:}
        \begin{itemize}
            \item \textbf{Llama2-7B}  from \(2.54\%\) to \(41.04\%\)
            \item \textbf{Code Llama-7B}  from \(14.54\%\) to \(48.24\%\)
        \end{itemize}
    \end{itemize}
\end{frame}
        '''))

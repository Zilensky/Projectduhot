import subprocess

import matplotlib.pyplot as plt
from pylatex import Document, Section, Command, Figure
from pylatex.utils import NoEscape


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


def create_presentation():
    # Create a beamer document for the presentation
    doc = Document(documentclass=NoEscape("beamer"))

    # Title page
    company_name = "Yair company"
    doc.preamble.append(Command('title', company_name))
    doc.preamble.append(Command('author', 'Your Name'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))

    # Begin the document and add the title page
    doc.append(NoEscape(r'\maketitle'))

    # Section 1: Introduction (1 page)
    with doc.create(Section('Introduction')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Introduction}'))
        doc.append('This slide introduces the company and provides context.')
        doc.append(NoEscape(r'\end{frame}'))

    with doc.create(Section('Data')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Data}'))
        doc.append('This slide contains the data used in the analysis.')

        # Adding the list of items
        doc.append(NoEscape(r'\begin{itemize}'))
        doc.append(NoEscape(r'\item Hey'))
        doc.append(NoEscape(r'\item $2^4 = ?$'))  # LaTeX syntax for equation
        doc.append(NoEscape(r'\end{itemize}'))

        doc.append(NoEscape(r'\end{frame}'))

    # Section 3: Analysis
    with doc.create(Section('Analysis')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Analysis}'))
        doc.append('This slide presents the analysis of the data.')
        doc.append(NoEscape(r'\end{frame}'))

    # Section 4: Classifications
    with doc.create(Section('Classifications')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Classifications}'))
        doc.append('This slide presents the classifications based on the analysis.')
        doc.append(NoEscape(r'\end{frame}'))

    # Section 5: Visualizations (Simple Line Graph)
    with doc.create(Section('Visualizations')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Visualizations}'))
        doc.append('This slide contains a simple line graph.')

        # Insert the line chart image
        with doc.create(Figure(position='h!')) as graph:
            graph.add_image('line_chart.png', width='80mm')  # Add the image generated earlier

        doc.append(NoEscape(r'\end{frame}'))

    # Section 6: Anomalies
    with doc.create(Section('Anomalies')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Anomalies}'))
        doc.append('This slide presents the detected anomalies in the data.')
        doc.append(NoEscape(r'\end{frame}'))

    # Section 7: Prediction
    with doc.create(Section('Prediction')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Prediction}'))
        doc.append('This slide presents predictions based on the data.')
        doc.append(NoEscape(r'\end{frame}'))

    # Section 8: Conclusions
    with doc.create(Section('Conclusions')):
        doc.append(NoEscape(r'\begin{frame}'))
        doc.append(NoEscape(r'\frametitle{Conclusions}'))
        doc.append('This slide summarizes the key takeaways from the analysis.')
        doc.append(NoEscape(r'\end{frame}'))

    # Generate the LaTeX file and PDF
    doc.generate_tex('presentation')

    # Run the pdflatex command to compile the .tex file manually
    try:
        subprocess.run(['pdflatex', 'presentation.tex'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


# Generate the line chart image
generate_line_chart()

# Create the presentation
create_presentation()

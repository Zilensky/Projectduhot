# Install necessary packages in Colab


import pdfplumber
import fitz  # PyMuPDF for image extraction
import pandas as pd
import os

import re


def is_hebrew(s):
    """Check if the string contains Hebrew characters."""
    return bool(re.search(r'[\u0590-\u05FF]', s))


def reverse_hebrew_text(s):
    """Reverse Hebrew text in a string."""
    if is_hebrew(s):
        return s[::-1]  # Reverse string
    return s


def extract_tables(pdf_path):
    # Extract tables from the PDF and return as DataFrames
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            table = page.extract_table()
            if table:
                # Clean up the table by removing rows/columns with None values
                clean_table = [[cell if cell is not None else "" for cell in row] for row in table]
                for i, row in enumerate(clean_table):
                    clean_table[i] = [reverse_hebrew_text(cell) for cell in row]

                # Convert to DataFrame
                df = pd.DataFrame(clean_table[1:], columns=clean_table[0])
                search_value = "מזומנים ושווה מזומנים"
                matching_rows = df[df.apply(lambda row: row.astype(str).str.contains(search_value).any(), axis=1)]

                print(matching_rows)
                        # Print and store DataFrame
           #     print(f"Table found on page {page_num}:")
            #    print(df)  # Display without the DataFrame index
                # Convert the DataFrame to a JSON file and save it


                tables.append((page_num, df))
    return tables


def extract_images(pdf_path, output_folder='extracted_images'):
    # Extract images from the PDF and save them
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_count = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_folder}/page{page_num + 1}_image{img_index + 1}.{image_ext}"

            with open(image_filename, "wb") as f:
                f.write(image_bytes)
            image_count += 1
            print(f"Saved image {image_filename}")

    print(f"Total images extracted: {image_count}")




if __name__ == "__main__":
    # Path to the PDF file on your local machine
    pdf_path = r"C:\Users\yairb\Downloads\P1612163-00.pdf"  # Use raw string (r) to handle backslashes

    # Extract and print tables from the PDF
    tables = extract_tables(pdf_path)
    # Save the first DataFrame in the `tables` list to an Excel file
    excel_filename = 'yair_tables.xlsx'

    with pd.ExcelWriter(excel_filename) as writer:
        for i, (page_num, df) in enumerate(tables, start=1):
            sheet_name = f'Page_{page_num}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Tables saved to {excel_filename}")



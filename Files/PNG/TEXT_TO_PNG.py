import re

from PIL import Image, ImageDraw, ImageFont


def save_text_as_png(text, file_path='Directorion.png', width=1200, height=1200, initial_font_size=24):
    # Reverse text for proper Hebrew display
    text = text[::-1]

    # Set up image with white background
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Define font and adjust size to fit within width and height
    font_size = initial_font_size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Split text into lines to fit within width
    lines = []
    current_line = ""
    for word in text.split():
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        if test_width <= width - 20:  # Allowing a bit of margin
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    # Adjust font size if text exceeds the image height
    while True:
        total_text_height = sum(
            draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in
            lines) + len(lines) * 5
        if total_text_height <= height - 20:  # Allowing margin
            break
        font_size -= 1
        font = ImageFont.truetype("arial.ttf", font_size)

    # Calculate the y starting position
    y_text = (height - total_text_height) // 2

    lines=lines[::-1]

    def is_hebrew(word):
        """Check if a word is in Hebrew."""
        return bool(re.search(r'[\u0590-\u05FF]', word))

    def reverse_non_hebrew_words(lines):
        reversed_lines = []
        for line in lines:
            new_line = []
            for word in line.split():  # Split the line into words
                if is_hebrew(word):
                    new_line.append(word)  # Keep Hebrew words as they are
                else:
                    new_line.append(word[::-1])  # Reverse non-Hebrew words
            reversed_lines.append(" ".join(new_line))  # Join words back into a line
        return reversed_lines

    lines=reverse_non_hebrew_words(lines)
    # Draw each line of text
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_text = (width - text_width) // 2
        draw.text((x_text, y_text), line, fill="black", font=font)
        y_text += bbox[3] - bbox[1] + 5  # Adding line spacing

    # Save the image
    image.save(file_path)
    print(f"Image saved to {file_path}")


# Hebrew text to be saved as an image
text = '''בדוח העסקי של חברת אאורה השקעות בע"מ ודוח הדירקטוריון שלה, ניכרת ההתמקדות בצמיחת החברה והשקעותיה בפרויקטים נרחבים למגורים בישראל, בעיקר בתחומי ההתחדשות העירונית והקמת שכונות מגורים חדשות. החברה עברה מהלכים אסטרטגיים שונים, ובתוך כך חתמה על שיתופי פעולה חדשים והרחיבה את פעילותה הכלכלית.

מצב החברה ופעילותה:

תחום הנדל"ן למגורים והתחדשות עירונית - החברה מדווחת על פעילות ענפה בתחומי הייזום, התכנון והבנייה עם דגש על פרויקטים לשכונות מגורים והתחדשות עירונית. חלק משמעותי מהפעילות מתמקד בפרויקטים של בנייה ושיווק דירות חדשות, כולל בנייה ירוקה ותשתיות.

פעילות כלכלית וצמיחה - אאורה השקעות חתמה על הסכמי מימון רחבי היקף, כולל הרחבת מסגרת ההסכם עם הפניקס ביטוח בע"מ בכ-750 מיליון ש"ח לטובת מימון פרויקטים חדשים. הרחבה זו מעידה על גידול בפעילות והביטחון של השותפים הפיננסיים בהצלחת החברה.

רכישת מגידו י.ק בע"מ - החברה השלימה את רכישת חברת מגידו, מהלך המרחיב את יכולות הבנייה והיזמות שלה במגוון תחומים בנדל"ן, ומאפשר לה כניסה לפרויקטים נוספים וחדשניים.

סיכום המצב הכלכלי - הדירקטוריון מצביע על יציבות כלכלית וביטחון בהשקעות החברה, תוך ציפייה להמשך ביקוש גבוה לפרויקטים למגורים. יחד עם זאת, החברה עוקבת אחר תנודות אפשריות בשוק הנדל"ן ובמצב הכלכלה המקומית והבינלאומית, אשר עשויות להשפיע על יכולות המכירה והבנייה של פרויקטים עתידיים.'''

save_text_as_png(text)

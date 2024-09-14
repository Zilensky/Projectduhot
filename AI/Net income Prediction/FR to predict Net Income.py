import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def predict_net_income_from_excel(file_path,excel):
    # קריאה מקובץ ה-Excel
    df = pd.read_csv(file_path)
    df_excel=pd.read_excel(excel)

    # סינון נתוני 2022 עבור תכונות ונתוני 2023 עבור תחזית
    df_2022 = df[df['Year'] == 2022].drop(['Symbol', 'Year'], axis=1)
    df_2023 = df_excel[df['Year'] == 2023]['Net Income']

    # בדיקה שהקבצים מכילים נתונים מספיקים
    if df_2022.empty or df_2023.empty:
        print("הנתונים לא מספיקים על מנת לבצע רגרסיה.")
        return

    # חלוקה לסט אימון ובדיקה (80% אימון, 20% בדיקה)
    X_train, X_test, y_train, y_test = train_test_split(df_2022, df_2023, test_size=0.2, random_state=42)

    # יצירת מודל רגרסיה
    model = LinearRegression()

    # אימון המודל
    model.fit(X_train, y_train)

    # חיזוי Net Income ל-2023
    y_pred = model.predict(X_test)

    # חישוב Mean Squared Error ו-R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2 * 100:.2f}%')

    # חיזוי עבור כל החברות ב-2022 ושמירת התוצאה כקובץ CSV
    df['Net Income 2023 Prediction'] = model.predict(df_2022)
    df.to_csv('financial_predictions.csv', index=False)

    print("התחזית נשמרה בקובץ financial_predictions.csv")

# שימוש בפונקציה
file_path = 'financial_comparisons.csv'  # הקובץ שלך עם נתוני 2022 ו-2023
excel_file='../Files/combined_financial_data_all_stocks.xlsx'
predict_net_income_from_excel(file_path,excel_file)

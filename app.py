from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

DATA_FILE = 'data/expenses.csv'

# Load models
anomaly_model = pickle.load(open('model/anomaly_model.pkl', 'rb'))
category_model = pickle.load(open('model/category_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# -----------------------------
# RULE-BASED CATEGORY KEYWORDS
# -----------------------------
CATEGORY_KEYWORDS = {
    "Food": [
        "food", "lunch", "dinner", "breakfast", "restaurant", "cafe", "coffee",
        "tea", "snack", "pizza", "burger", "biryani", "swiggy", "zomato",
        "eat", "meal", "hotel", "canteen", "tiffin", "juice", "bakery",
        "groceries", "grocery", "vegetables", "rice", "dal", "curry", "bread",
        "milk", "egg", "chicken", "mutton", "fish", "dosa", "idli", "paratha"
    ],
    "Transport": [
        "transport", "bus", "auto", "cab", "uber", "ola", "taxi", "metro",
        "train", "fuel", "petrol", "diesel", "bike", "ticket", "travel",
        "fare", "commute", "rapido", "rickshaw", "flight", "airways", "toll"
    ],
    "Shopping": [
        "shopping", "clothes", "shirt", "shoes", "amazon", "flipkart", "myntra",
        "dress", "jeans", "bag", "watch", "accessories", "mall", "store",
        "buy", "purchase", "kurti", "saree", "trouser", "jacket", "cap", "belt"
    ],
    "Entertainment": [
        "movie", "cinema", "netflix", "spotify", "game", "gaming", "concert",
        "show", "ott", "prime", "hotstar", "fun", "outing", "party", "night out",
        "club", "bar", "bowling", "amusement", "theme park", "youtube premium"
    ],
    "Health": [
        "medicine", "doctor", "hospital", "clinic", "pharmacy", "medical",
        "health", "gym", "fitness", "tablet", "injection", "checkup", "apollo",
        "diagnostic", "test", "xray", "scan", "dentist", "eye", "spectacles"
    ],
    "Education": [
        "book", "course", "tuition", "school", "college", "fees", "udemy",
        "study", "pen", "notebook", "stationery", "exam", "coaching", "class",
        "coursera", "certificate", "training", "workshop", "seminar", "library"
    ],
    "Utilities": [
        "electricity", "water", "bill", "recharge", "mobile", "internet",
        "wifi", "broadband", "phone", "jio", "airtel", "bsnl", "gas", "lpg",
        "vi", "vodafone", "postpaid", "prepaid", "dth", "cable", "subscription"
    ],
    "Rent": [
        "rent", "house", "room", "flat", "hostel", "pg", "accommodation",
        "lease", "deposit", "maintenance", "society", "apartment"
    ],
}

def rule_based_category(description):
    desc = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in desc:
                return category
    return None

def smart_category(description):
    # Rule-based first (more reliable for common descriptions)
    rule_result = rule_based_category(description)
    if rule_result:
        return rule_result
    # Fall back to ML model
    X = vectorizer.transform([description])
    return category_model.predict(X)[0]

# -----------------------------
# CHARTS
# -----------------------------
def generate_charts(df):
    if not os.path.exists('static'):
        os.makedirs('static')

    df.groupby('Category')['Amount'].sum().plot.pie(autopct='%1.1f%%')
    plt.title("Category Distribution")
    plt.savefig('static/pie.png')
    plt.close()

    df.groupby('Date')['Amount'].sum().plot(kind='bar')
    plt.title("Daily Spending")
    plt.savefig('static/bar.png')
    plt.close()

# -----------------------------
# PREDICTION
# -----------------------------
def predict_expense(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date')
    df['Day'] = np.arange(len(df))

    X = df[['Day']]
    y = df['Amount']

    model = LinearRegression()
    model.fit(X, y)

    next_day = pd.DataFrame([[len(df)]], columns=['Day'])
    return round(model.predict(next_day)[0], 2)

# -----------------------------
# SCORE
# -----------------------------
def calculate_score(df):
    total = df['Amount'].sum()
    avg = df['Amount'].mean()

    score = 100
    if avg > 1000:
        score -= 30
    elif avg > 500:
        score -= 15
    if total > 10000:
        score -= 30

    return max(score, 0)

# -----------------------------
# IMPROVED CHATBOT
# -----------------------------
def chatbot_response(msg, df):
    msg_lower = msg.lower().strip()

    total       = round(df['Amount'].sum(), 2)
    avg         = round(df['Amount'].mean(), 2)
    count       = len(df)
    cat_summary = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    max_cat     = cat_summary.idxmax()
    min_cat     = cat_summary.idxmin()
    score       = calculate_score(df)
    prediction  = predict_expense(df.copy())
    latest      = df.tail(1).iloc[0]

    # Greeting
    if any(w in msg_lower for w in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
        return (f"Hello! 👋 I'm your Expense Guardian AI.\n"
                f"You've spent ₹{total} so far across {count} transactions.\n"
                f"How can I help you today?")

    # Thank you
    elif any(w in msg_lower for w in ["thank", "thanks", "great", "nice", "awesome", "perfect"]):
        return "You're welcome! 😊 Feel free to ask anything about your finances."

    # Total spending
    elif any(w in msg_lower for w in ["total", "overall", "how much", "all expenses", "sum"]):
        return (f"Your total spending is ₹{total}.\n"
                f"This is across {count} recorded transaction(s).")

    # Average
    elif any(w in msg_lower for w in ["average", "avg", "per transaction", "mean", "typical"]):
        return (f"Your average spending per transaction is ₹{avg}.\n"
                f"{'⚠️ This is on the higher side.' if avg > 800 else '✅ This looks reasonable.'}")

    # Highest category
    elif any(w in msg_lower for w in ["highest", "most", "top", "maximum", "max", "biggest", "largest"]):
        amt = round(cat_summary.iloc[0], 2)
        return (f"Your highest spending category is {max_cat} at ₹{amt}.\n"
                f"Consider reviewing your {max_cat} expenses to save more.")

    # Lowest category
    elif any(w in msg_lower for w in ["lowest", "least", "minimum", "min", "smallest"]):
        amt = round(cat_summary.iloc[-1], 2)
        return f"Your lowest spending category is {min_cat} at ₹{amt}."

    # Category breakdown
    elif any(w in msg_lower for w in ["category", "categories", "breakdown", "split", "distribution", "where"]):
        lines = [f"  • {cat}: ₹{round(amt, 2)}" for cat, amt in cat_summary.items()]
        return "Your spending breakdown by category:\n" + "\n".join(lines)

    # Overspending
    elif any(w in msg_lower for w in ["overspend", "over spend", "too much", "control", "limit", "spending habit"]):
        if avg > 1000:
            return (f"⚠️ Yes, you are overspending!\n"
                    f"Your average of ₹{avg} per transaction is very high.\n"
                    f"Try to cut down on {max_cat} first.")
        elif avg > 500:
            return (f"⚠️ You're spending moderately — average is ₹{avg}.\n"
                    f"Keep an eye on {max_cat} to stay in budget.")
        else:
            return (f"✅ Your spending is under control!\n"
                    f"Average is ₹{avg} per transaction. Keep it up!")

    # Save / budget tips
    elif any(w in msg_lower for w in ["save", "saving", "reduce", "cut", "budget", "tip", "advice", "suggest", "improve"]):
        estimated_saving = round(avg * 0.2 * count, 2)
        return (f"💡 Here are some tips to save more:\n"
                f"  • {max_cat} is your biggest expense — reduce it by 20%.\n"
                f"  • That could save you around ₹{estimated_saving} overall.\n"
                f"  • Try to keep average below ₹500 per transaction.\n"
                f"  • Your current average: ₹{avg}.")

    # Prediction
    elif any(w in msg_lower for w in ["predict", "forecast", "future", "expect", "upcoming", "next expense", "next month"]):
        return (f"📈 Based on your spending trend, your next predicted expense is around ₹{prediction}.\n"
                f"This is calculated using your past {count} transaction(s).")

    # Health score
    elif any(w in msg_lower for w in ["score", "health", "rating", "status", "performance", "financial"]):
        if score >= 70:
            status = "Good 🟢 — you're managing your money well!"
        elif score >= 40:
            status = "Moderate 🟡 — try to reduce high-value expenses."
        else:
            status = "Poor 🔴 — urgent attention needed on your spending."
        return f"Your financial health score is {score}/100.\nStatus: {status}"

    # Latest expense
    elif any(w in msg_lower for w in ["last", "latest", "recent", "yesterday", "today", "previous"]):
        return (f"Your most recent expense:\n"
                f"  • Date: {latest['Date']}\n"
                f"  • Amount: ₹{latest['Amount']}\n"
                f"  • Category: {latest['Category']}\n"
                f"  • Description: {latest.get('Description', 'N/A')}")

    # Count
    elif any(w in msg_lower for w in ["count", "how many", "number of", "entries", "records", "transactions"]):
        return f"You have recorded {count} expense(s) in total."

    # Spent on a specific category
    elif "spend on" in msg_lower or "spent on" in msg_lower:
        for cat in cat_summary.index:
            if cat.lower() in msg_lower:
                amt = round(cat_summary[cat], 2)
                return f"You have spent ₹{amt} on {cat} so far."
        return "I couldn't find that category. Try: Food, Transport, Shopping, Health, Entertainment, Education, Utilities, or Rent."

    # Default with helpful suggestions
    else:
        return ("I'm not sure about that. Try asking me:\n"
                "  • 'What is my total spending?'\n"
                "  • 'Show my category breakdown'\n"
                "  • 'Am I overspending?'\n"
                "  • 'How can I save money?'\n"
                "  • 'What is my financial health score?'\n"
                "  • 'What was my last expense?'\n"
                "  • 'How much did I spend on Food?'")

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def add():
    amount = float(request.form['amount'])
    desc = request.form['description']

    # Smart category: rule-based first, ML fallback
    category = smart_category(desc)

    anomaly = anomaly_model.predict([[amount]])[0]
    alert = "⚠️ Unusual!" if anomaly == -1 else "Normal"

    df = pd.read_csv(DATA_FILE)

    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Amount": amount,
        "Category": category,
        "Description": desc
    }

    df = pd.concat([df, pd.DataFrame([new_row])])
    df.to_csv(DATA_FILE, index=False)

    return jsonify({"category": category, "alert": alert})

@app.route('/dashboard')
def dashboard():
    df = pd.read_csv(DATA_FILE)

    generate_charts(df)
    prediction = predict_expense(df)
    score = calculate_score(df)

    return render_template('dashboard.html',
                           total=df['Amount'].sum(),
                           avg=df['Amount'].mean(),
                           prediction=prediction,
                           score=score,
                           data=df.to_dict(orient='records'))

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json['message']
    df = pd.read_csv(DATA_FILE)
    reply = chatbot_response(msg, df)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
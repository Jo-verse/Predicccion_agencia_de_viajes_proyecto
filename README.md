## 🧳 Use Case Description: 🧭 Intelligent Travel Destination Prediction

In the competitive tourism industry, understanding and anticipating customer needs is key to improving user experience, optimizing marketing campaigns, and increasing sales conversion. This project aims to build a web-based tool for travel agencies that recommends personalized destinations using machine learning with an XGBoost model and an interactive form.

### 🎯 Project Objective

The main goal is to develop a machine learning model capable of **predicting whether a customer will complete a travel booking**, based on variables such as:

- Customer's age, gender, and nationality.
- Channel used to book (web, physical agency, phone, etc.).
- Number of companions, preferred destination, type of travel package.
- History of previous interactions.
- Travel dates and duration, etc.

This type of prediction can help the agency to:

- **Identify potential customers** with a higher likelihood of conversion.
- **Design personalized offers** based on the customer profile.
- **Optimize customer service resources**, prioritizing the most qualified leads.
- **Reduce advertising costs** by better segmenting the target audience.

### 🧠 Business Benefit

By integrating this model into the agency’s processes, it enables **greater operational efficiency and improved customer experience**, ultimately resulting in **more confirmed bookings** and **higher revenue**. It also allows the agency to be more proactive and strategic in its business decisions.

### 🚀 Technologies Used

- **Python** (Pandas, NumPy, XGBoost, Scikit-learn)
- **SQL** (for structured data querying)
- **Streamlit** (web interface)
- **Jupyter Notebook** (exploration, analysis, and modeling)
- **Git & GitHub**

### 🤖 Prediction Model

The recommendation engine is based on an **XGBoost Classifier**, chosen for its high accuracy and performance in complex classification tasks. The model was trained using features such as:

- Customer's age and budget
- Interests (nature, culture, gastronomy, etc.)
- Preferred travel type (adventure, leisure, family, luxury)
- History of visited destinations
- Time of year

The goal is to predict the most suitable destination for each user profile.

### ⚙️ Installation & Usage

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app/main.py

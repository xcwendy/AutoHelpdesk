from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# 加载模型和向量器
model = joblib.load('fault_classifier.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')


# 文本清洗函数
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", str(text))
    stopwords = ["的", "我", "怎么办", "电脑", "软件", "故障"]
    text = " ".join([w for w in text.split() if w not in stopwords])
    text = " ".join(jieba.cut(text))
    return text

def get_solution_by_type(predicted_type):
    df = pd.read_csv('solutions.csv')
    solution = df[df['fault_type'] == predicted_type]['solution'].values
    if len(solution) > 0:
        return solution[0]
    else:
        return "未找到对应的解决方案"

@app.route('/get_solution', methods=['POST'])
def get_solution():
    user_question = request.json.get('question')
    clean_question = clean_text(user_question)
    question_vector = tfidf.transform([clean_question]).toarray()
    predicted_type = model.predict(question_vector)[0]
    # 这里假设你有一个根据预测类型找解决方案的函数get_solution_by_type
    solution = get_solution_by_type(predicted_type)
    return jsonify({'solution': solution, 'predicted_type': predicted_type})


if __name__ == '__main__':
    app.run(debug=True)
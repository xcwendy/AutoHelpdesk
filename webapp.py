from flask import Flask, request, jsonify,render_template
import joblib
import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
import logging

app = Flask(__name__)

# 加载模型和向量器
model = joblib.load('fault_classifier.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')
type_question_vectors = joblib.load('type_question_vectors.pkl')

# 文本清洗函数
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", str(text))
    stopwords = ["的", "我", "怎么办", "电脑", "软件", "故障"]
    text = " ".join([w for w in text.split() if w not in stopwords])
    text = " ".join(jieba.cut(text))
    return text


#添加返回html的路由
@app.route('/')
def index():
    return render_template('index.html')

logging.basicConfig(level = logging.INFO)

@app.route('/get_solution', methods=['POST'])
def get_solution():
    user_question = request.json.get('question')
    clean_question = clean_text(user_question)
    question_vector = tfidf.transform([clean_question]).toarray()
    predicted_index = model.predict(question_vector)[0]
    logging.info(f"Predicted index: {predicted_index}")
    predicted_type = le.inverse_transform([predicted_index])[0]
    logging.info(f"Predicted type: {predicted_type}")

    # 计算用户问题与该类型下问题的余弦相似度
    similarities = cosine_similarity(question_vector, type_question_vectors[predicted_type])[0]
    most_similar_index = similarities.argmax()


    # 从MySQL数据库获取solution
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='9696',
        database='qapairs',
        charset='utf8mb4'
    )

    try:
        with conn.cursor() as cursor:
            select_query = "SELECT id, fault_type, solution FROM combined_fault_data WHERE fault_type = %s"
            cursor.execute(select_query, (predicted_type,))
            data = cursor.fetchall()

            ids = [row[0] for row in data]
            solutions = [row[2] for row in data]

            id_value = ids[most_similar_index]
            solution = solutions[most_similar_index]
            logging.info(f"Predicted type: {predicted_type}, Matched id: {id_value}")

    finally:
        conn.close()

    return jsonify({'solution': solution, 'predicted_type': predicted_type, 'id': id_value})


if __name__ == '__main__':
    app.run(debug=True)
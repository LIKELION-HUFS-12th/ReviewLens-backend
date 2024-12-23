import os
import re
import json
import pandas as pd
import numpy as np
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from django.conf import settings
from matplotlib import font_manager, rc
import matplotlib
from django.conf import settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from django.conf import settings
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
matplotlib.use('Agg')  # GUI 백엔드 사용하지 않도록 설정

# 폰트 설정
font_path = os.path.join(settings.BASE_DIR, 'analysis', 'fonts', 'NGULIM.TTF')  # 폰트 파일 경로
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)  
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    print(f"Font '{font_name}' successfully set for matplotlib.")
else:
    print(f"Font not found at {font_path}. Using default font.")


def load_file(file_path):
    text_data = pd.read_excel(file_path)
    product_names = text_data['상품명'].astype(str).tolist()
    reviews = text_data['상품평']
    review_list = reviews.astype(str).tolist()
    print("파일 로딩 완료.")

    return review_list, product_names


def create_test_data(review_list, product_names, sample_size=5):
    if sample_size == 'max':
        max_size = len(review_list)
    else:
        max_size = min(sample_size, len(review_list), len(product_names))

    review_list_sliced = review_list[:max_size]
    product_list_sliced = product_names[:max_size]

    return review_list_sliced, product_list_sliced


def analyze_review(review, host, api_key, api_key_primary_val, request_id):
    preset_text = [
        {"role": "system", "content": "이것은 상품 리뷰에 대한 감정 분석기입니다. 리뷰가 긍정적이라면 'positive', 중립이면 'neutral', 부정적이면 'negative'만으로 답변해주세요. 세 가지 외 다른 답변이 나오면 안됩니다."},
        {"role": "user", "content": review}
    ]
    
    request_data = {
        'messages': preset_text,
        'topP': 0.6,
        'topK': 0,
        'maxTokens': 20,  
        'temperature': 0.1,
        'repeatPenalty': 1.2,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
    
    headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': api_key,
        'X-NCP-APIGW-API-KEY': api_key_primary_val,
        'X-NCP-CLOVASTUDIO-REQUEST-ID': request_id,
        'Content-Type': 'application/json; charset=utf-8',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.post(host + '/testapp/v1/chat-completions/HCX-DASH-001', headers=headers, json=request_data)
        response.raise_for_status()
        
        result = response.json()
        message_content = result['result']['message']['content'].strip()
        
        if message_content in ["positive", "neutral", "negative"]:
            return {"review": review, "sentiment": message_content}
        else:
            print(f"예상치 않은 응답: {message_content}. 기본값 'neutral'로 설정.")
            return {"review": review, "sentiment": "neutral"}
    
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 에러: {e}.")
        return {"review": review, "sentiment": "error"}
    except requests.exceptions.RequestException as e:
        print(f"API 요청 에러: {e}.")
        return {"review": review, "sentiment": "error"}

def analyze_reviews_clova_studio(review_list_sliced):
    print("데이터 전처리 완료. 감정 분석 시작...")
    start_time = time.time()

    host = 'https://clovastudio.stream.ntruss.com'
    api_key = 'NTA0MjU2MWZlZTcxNDJiYzCfHM1duMGVmI101pNbw6DRY8rHVXsyr1bq0e2r332L'
    api_key_primary_val = 'QjtD5GeFK6qBSlyFwpYo50Vrn6aURfdCG6SySOUE'
    request_id = '26eb069872414eb480d79bd6ccf640d1'

    result_list = []
    
    # 병렬 처리를 위한 ThreadPoolExecutor 사용
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_review = {
            executor.submit(analyze_review, review, host, api_key, api_key_primary_val, request_id): review
            for review in review_list_sliced
        }
        
        for i, future in enumerate(as_completed(future_to_review)):
            try:
                result = future.result()
                result_list.append(result)
                progress = (i + 1) / len(review_list_sliced) * 100
                print(f'전체 {len(review_list_sliced)}개 데이터 중 {i+1}번 째 데이터 {progress:.2f}% 완료')
            except Exception as e:
                print(f"에러 발생: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"감정 분석 완료. 총 소요 시간: {total_time:.2f}초")
    return result_list


def analyze_reviews_with_model(preprocessed_reviews_sliced, batch_size=32):
    print("데이터 전처리 완료. 감정 분석 시작...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 장치: {device}")

    model_name = "rlawltjd/kobert-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    start_time = time.time()

    result_list = []

    total_reviews = len(preprocessed_reviews_sliced)
    for start_idx in range(0, total_reviews, batch_size):
        end_idx = min(start_idx + batch_size, total_reviews)
        batch = preprocessed_reviews_sliced[start_idx:end_idx]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predicted_classes = logits.argmax(dim=-1).tolist()

        for review, predicted_class, prob in zip(batch, predicted_classes, probs):
            sentiment = label_map[predicted_class]
            confidence = prob[predicted_class].item()  
            result_list.append({"review": review, "sentiment": sentiment, "confidence": confidence})

        progress = (end_idx / total_reviews) * 100
        print(f"전체 {total_reviews}개 데이터 중 {end_idx}개 완료 ({progress:.2f}%)")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"감정 분석 완료. 총 소요 시간: {total_time:.2f}초")

    return result_list



def process_sentiment_analysis(sentiment_data_list, original_reviews, product_list_test):
    summary = []

    for i, sentiment_data in enumerate(sentiment_data_list):
        document_sentiment = sentiment_data.get('sentiment', 'neutral')
        product_name_cleaned = product_list_test[i].replace(" ", "")

        document_summary = {
            "product_name": product_name_cleaned,
            "original_content": original_reviews[i],
            "document_sentiment": document_sentiment
        }
        summary.append(document_summary)

    result = json.dumps(summary, ensure_ascii=False, indent=4)

    output_file_path = os.path.join(settings.BASE_DIR, 'media', 'result.json')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(result)

    print(f"결과가 {output_file_path}에 저장되었습니다.")
    return summary

def remove_stopwords(text, stopwords):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stopwords) + r')\b'
    return re.sub(pattern, '', text)

def wordcloud_stopwords():
    return list(set(['아', '게', '있는','이런','되지','내가', '구매했네요', '제가', '좋습니다', '좋네요','좋구', '써보니', '쓰던', '없어서', '없네요', '넘', '앞으로', '다른', 'OOO', '많이', '않아', '같이', '같아요', '전', '것', '않고', '진짜', '좀', '정말', '더', '않아요', '있어', '있습니다', '좋아요', '좋고', '잘', '많이', '너무너무', '원래', '샀는데', '같습니다', '좋다길래', '괜찮아요', '있어요', '사봤어요', '보다', '요건', '그래서', '별로', '따지면', '같구요', '선택했는데적당한것', '하고', '말이죠', '구매', '너무많아서', '바르면', '입니다', '바르고', '광고', '첨에', '엄청', '열심히', '바르다', '생각보단', '요거', '제형', '00', '얼굴에', '너무', '같아요', '없고', '없어요', '좋은거', '해서', '안', '아주', '그냥', '많아', '처음', '들어있는지', '쓰고', '않을거라고', '있음', '들어요', '있고', '수', '제형의', '좋은', '근데', '않을거라고']))


def total_sentiment_count(results):
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    for review in results:
        sentiment = review['document_sentiment']
        if sentiment == 'positive':
            sentiment_counts['Positive'] += 1
        elif sentiment == 'neutral':
            sentiment_counts['Neutral'] += 1
        elif sentiment == 'negative':
            sentiment_counts['Negative'] += 1
            
    print(f"리뷰 감정 분석 결과 전체 리뷰 개수 {len(results)}개 중")
    print(f"긍정적인 리뷰 {sentiment_counts['Positive']}개")
    print(f"부정적인 리뷰 {sentiment_counts['Negative']}개")
    print(f"중립적인 리뷰 {sentiment_counts['Neutral']}개 입니다.")
    return sentiment_counts

def count_for_indiv(results):
    ps_counts = {}

    for data in results:
        product = data['product_name']
        sentiment = data['document_sentiment']

        if product not in ps_counts:
            ps_counts[product] = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

        if sentiment == 'positive':
            ps_counts[product]['Positive'] += 1
        elif sentiment == 'neutral':
            ps_counts[product]['Neutral'] += 1
        elif sentiment == 'negative':
            ps_counts[product]['Negative'] += 1
            
    return ps_counts


def generate_wordcloud_directly_to_pdf(pdf_canvas, text, title='', font_path='', y_position=600):
    try:
        stopwords = wordcloud_stopwords()
        filtered_text = remove_stopwords(text, stopwords)

        wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(filtered_text)

        image_stream = BytesIO()
        wordcloud.to_image().save(image_stream, format='PNG')
        image_stream.seek(0)

        img = ImageReader(image_stream)
        pdf_canvas.drawImage(img, x=50, y=y_position, width=500, height=200)

        if title:
            pdf_canvas.setFont("Helvetica-Bold", 14)
            pdf_canvas.drawString(30, y_position + 220, title)

    except Exception as e:
        print(f"워드클라우드 생성 중 오류 발생: {e}")



def create_sentiment_report_pdf_directly(summary, sentiment_counts, ps_counts, output_path):
    font_path = os.path.join(settings.BASE_DIR, 'analysis', 'fonts', 'NanumGothicCoding.TTF')
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        font_name = 'NanumGothic'
    else:
        font_name = 'Helvetica'

    # PDF 생성
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont(font_name, 12)

    # 첫 번째 페이지: 분석 결과 요약 및 전체 파이 차트
    c.drawString(30, 750, "Sentiment Analysis Report")
    c.drawString(30, 735, "------------------------------------------")
    total_reviews = sum(sentiment_counts.values())
    positive_percent = (sentiment_counts['Positive'] / total_reviews) * 100
    neutral_percent = (sentiment_counts['Neutral'] / total_reviews) * 100
    negative_percent = (sentiment_counts['Negative'] / total_reviews) * 100

    c.drawString(30, 710, f"총 {total_reviews}개의 리뷰 분석 결과:")
    c.drawString(30, 695, f"긍정 리뷰: {sentiment_counts['Positive']}개 ({positive_percent:.2f}%)")
    c.drawString(30, 680, f"중립 리뷰: {sentiment_counts['Neutral']}개 ({neutral_percent:.2f}%)")
    c.drawString(30, 665, f"부정 리뷰: {sentiment_counts['Negative']}개 ({negative_percent:.2f}%)")
    c.drawString(30, 650, "------------------------------------------")

    add_pie_chart_to_pdf(
        c,
        sentiment_counts,
        title="Sentiment Distribution",
        x=80,
        y=300,
        width=400,
        height=300
    )
    c.showPage()

    # 두 번째 페이지: 긍정 및 부정 워드클라우드
    positive_text = " ".join([review["original_content"] for review in summary if review["document_sentiment"] == "positive"])
    generate_wordcloud_directly_to_pdf(c, positive_text, title="Positive Reviews Word Cloud", font_path=font_path, y_position=500)

    negative_text = " ".join([review["original_content"] for review in summary if review["document_sentiment"] == "negative"])
    generate_wordcloud_directly_to_pdf(c, negative_text, title="Negative Reviews Word Cloud", font_path=font_path, y_position=250)
    c.showPage()

    # 세 번째 페이지부터: 개별 제품 차트 (바 차트 + 파이 차트)
    products_per_page = 4  
    chart_y_positions = [600, 400, 200, 0] 
    product_items = list(ps_counts.items())

    for page_idx in range(0, len(product_items), products_per_page):
        for chart_idx, (product, counts) in enumerate(product_items[page_idx:page_idx + products_per_page]):
            y_position = chart_y_positions[chart_idx]
            add_combined_chart_to_pdf(
                c,
                counts,
                product_title=product,
                x=50,
                y=y_position,
                bar_width=250,
                pie_width=250,
                height=180
            )

        c.showPage()


    c.save()
    print(f"PDF가 '{output_path}'에 저장되었습니다.")





def add_pie_chart_to_pdf(pdf_canvas, data, title='', x=50, y=400, width=400, height=300, dpi=300):
    try:
        # 파이차트 생성
        fig, ax = plt.subplots(figsize=(8, 6))  
        ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#99ff99', '#ff9999'])
        ax.set_title(title, fontsize=14)

        image_stream = BytesIO()
        plt.savefig(image_stream, format='PNG', bbox_inches='tight', dpi=dpi)  # DPI 설정 추가
        plt.close(fig)
        image_stream.seek(0)

        img = ImageReader(image_stream)
        pdf_canvas.drawImage(img, x, y, width, height)

    except Exception as e:
        print(f"파이차트 생성 중 오류 발생: {e}")


def add_combined_chart_to_pdf(pdf_canvas, data, product_title='', x=50, y=200, bar_width=250, pie_width=250, height=250, dpi=300):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 바 차트
        axes[0].bar(data.keys(), data.values(), color=['#66b3ff', '#99ff99', '#ff9999'])
        axes[0].set_title(f"{product_title}", fontsize=14)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_xlabel('Sentiment', fontsize=12)
        
        # 파이 차트
        axes[1].pie(
            data.values(),
            labels=data.keys(),
            autopct='%1.1f%%',
            startangle=140,
            colors=['#66b3ff', '#99ff99', '#ff9999']
        )
        
        # 차트를 메모리에 저장
        image_stream = BytesIO()
        plt.savefig(image_stream, format='PNG', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        # PDF에 추가
        image_stream.seek(0)
        img = ImageReader(image_stream)
        pdf_canvas.drawImage(img, x, y, width=bar_width + pie_width, height=height)
    
    except Exception as e:
        print(f"차트 생성 중 오류 발생: {e}")

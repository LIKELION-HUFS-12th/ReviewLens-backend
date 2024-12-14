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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 장치: {device}")

model_name = "rlawltjd/kobert-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
label_map = {0: "negative", 1: "neutral", 2: "positive"}


def load_file(file_path):
    text_data = pd.read_excel(file_path)
    product_names = text_data['상품명'].astype(str).tolist()
    reviews = text_data['상품평']
    review_list = reviews.astype(str).tolist()
    return review_list, product_names


def preprocess(review_list):
    print("파일 로딩 완료. 데이터 전처리 중...")
    processed_reviews = []
    original_reviews = []

    for review in review_list:
        cleaned_original_review = re.sub(r'\s+', ' ', review).strip()
        original_reviews.append(cleaned_original_review)

        review_cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', cleaned_original_review)
        processed_reviews.append(review_cleaned)

    return original_reviews, processed_reviews


def create_test_data(preprocessed_reviews, original_reviews, product_names, sample_size=5):
    if sample_size == 'max':
        max_size = len(preprocessed_reviews)
    else:
        max_size = min(sample_size, len(preprocessed_reviews), len(original_reviews), len(product_names))

    preprocessed_reviews_sliced = preprocessed_reviews[:max_size]
    original_review_list_sliced = original_reviews[:max_size]
    product_list_sliced = product_names[:max_size]

    return preprocessed_reviews_sliced, original_review_list_sliced, product_list_sliced


def analyze_reviews_with_model(preprocessed_reviews_sliced, batch_size=32):
    print("데이터 전처리 완료. 감정 분석 시작...")
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

        # 결과 저장
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

    output_file_path = os.path.join(settings.BASE_DIR, 'analysis', 'results', 'sentiment_analysis_result_kobert.json')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(result)

    print(f"결과가 {output_file_path}에 저장되었습니다.")
    return summary

def generate_sentiment_wordclouds(summary):
    # 긍정적인 리뷰 필터링 및 불용어 제거
    positive_reviews = [f"{review['product_name']} {review['original_content']}" for review in summary if review["document_sentiment"] == "positive"]
    positive_text = " ".join(positive_reviews)
    positive_filtered_text = remove_stopwords(positive_text, wordcloud_stopwords())

    # 부정적인 리뷰 필터링 및 불용어 제거
    negative_reviews = [f"{review['product_name']} {review['original_content']}" for review in summary if review["document_sentiment"] == "negative"]
    negative_text = " ".join(negative_reviews)
    negative_filtered_text = remove_stopwords(negative_text, wordcloud_stopwords())

    # 긍정적인 리뷰 워드클라우드 생성 및 저장
    print("긍정적인 리뷰 워드클라우드 생성됨")
    generate_wordcloud(positive_filtered_text, title='Positive Reviews Word Cloud', sentiment='positive', font_path = os.path.join(settings.BASE_DIR,'analysis', 'fonts', 'NanumGothicCoding.TTF'))

    # 부정적인 리뷰 워드클라우드 생성 및 저장
    print("부정적인 리뷰 워드클라우드 생성됨")
    generate_wordcloud(negative_filtered_text, title='Negative Reviews Word Cloud', sentiment='negative', font_path = os.path.join(settings.BASE_DIR, 'analysis', 'fonts', 'NanumGothicCoding.TTF'))

def generate_wordcloud(text, title='', sentiment='', font_path=''):
    try:
        # 워드클라우드 생성
        wordcloud = WordCloud(font_path=font_path,
                              width=800, height=400, background_color='white').generate(text)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off') 
        if title:  
            plt.title(title, fontsize=16)
        
        file_dir = os.path.join(settings.MEDIA_ROOT, 'wordcloud')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = os.path.join(file_dir, f"{sentiment}_wordcloud.png")
        plt.savefig(file_path, bbox_inches='tight', dpi=300)  
        plt.close()
        print(f"워드클라우드가 '{file_path}'에 저장되었습니다.")
        return file_path
    
    except Exception as e:
        print(f"워드클라우드 생성 중 오류 발생: {e}")
        return None


def remove_stopwords(text, stopwords):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stopwords) + r')\b'
    return re.sub(pattern, '', text)

def wordcloud_stopwords():
    return list(set(['아', '구매했네요', '제가', '좋습니다', '좋네요','좋구', '써보니', '쓰던', '없어서', '없네요', '넘', '앞으로', '다른', 'OOO', '많이', '않아', '같이', '같아요', '전', '것', '않고', '진짜', '좀', '정말', '더', '않아요', '있어', '있습니다', '좋아요', '좋고', '잘', '많이', '너무너무', '원래', '샀는데', '같습니다', '좋다길래', '괜찮아요', '있어요', '사봤어요', '보다', '요건', '그래서', '별로', '따지면', '같구요', '선택했는데적당한것', '하고', '말이죠', '구매', '너무많아서', '바르면', '입니다', '바르고', '광고', '첨에', '엄청', '열심히', '바르다', '생각보단', '요거', '제형', '00', '얼굴에', '너무', '같아요', '없고', '없어요', '좋은거', '해서', '안', '아주', '그냥', '많아', '처음', '들어있는지', '쓰고', '않을거라고', '있음', '들어요', '있고', '수', '제형의', '좋은', '근데', '않을거라고']))

def load_result(path):
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

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

def plot_total_sentiment_pie_chart(sentiment_counts, output_path):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')
    plt.title("Sentiment Analysis Total Result")

    output_dir = os.path.join(settings.MEDIA_ROOT, 'charts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, os.path.basename(output_path))
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"파이차트가 '{output_file}'에 저장되었습니다.")

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

def charts(ps_counts, output_file_prefix):
    num_products = len(ps_counts)
    max_products_per_image = 5  # 한 이미지에 포함할 최대 제품 수
    total_images = (num_products // max_products_per_image) + (1 if num_products % max_products_per_image > 0 else 0)

    for img_index in range(total_images):
        start = img_index * max_products_per_image
        end = min(start + max_products_per_image, num_products)

        output_dir = os.path.join(settings.MEDIA_ROOT, 'charts', 'individual')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if end - start <= 0:
            break
        
        fig, axes = plt.subplots(end - start, 2, figsize=(12, 5 * (end - start)))
        fig.tight_layout(pad=5.0)

        for i, (product, counts) in enumerate(list(ps_counts.items())[start:end]):
            sentiments = list(counts.keys())
            values = list(counts.values())
            colors = ['#66b3ff', '#99ff99', '#ff9999']

            # 막대그래프 그리기
            axes[i, 0].bar(sentiments, values, color=colors)
            axes[i, 0].set_title(f"{product}", loc='left')
            axes[i, 0].set_xlabel("Sentiment")
            axes[i, 0].set_ylabel("Count")

            # 파이차트 그리기 전에 합계가 0인지 확인
            if np.sum(values) == 0:
                axes[i, 1].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
                axes[i, 1].set_title(f"Sentiment Analysis (Pie Chart) for {product}", loc='left')
                axes[i, 1].axis('off')
            else:
                axes[i, 1].pie(values, labels=sentiments, autopct='%1.1f%%', startangle=90, colors=colors)
                axes[i, 1].set_title(f"Sentiment Analysis (Pie Chart) for {product}", loc='left')
                axes[i, 1].axis('equal')

        # 이미지 저장
        output_file = os.path.join(output_dir, f"{output_file_prefix}_page_{img_index + 1}.png")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {output_file}")
        
        


def create_sentiment_report_pdf(output_path, charts_dir, wordcloud_dir, total_chart_path, sentiment_counts):
    # 한글 폰트 등록
    font_path = os.path.join(settings.BASE_DIR, 'analysis', 'fonts', 'NanumGothicCoding.TTF')
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        font_name = 'NanumGothic'
    else:
        font_name = 'Helvetica'

    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont(font_name, 12)

    # 첫 번째 페이지: 분석 결과 요약 + 전체 파이 차트
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

    if os.path.exists(total_chart_path):
        try:
            img = ImageReader(total_chart_path)
            img_width, img_height = img.getSize()
            aspect_ratio = img_height / img_width

            max_width = 400
            max_height = max_width * aspect_ratio
            chart_x_position = 100  
            chart_y_position = 300 

            c.drawImage(total_chart_path, chart_x_position, chart_y_position, width=max_width, height=max_height)
        except Exception as e:
            print(f"전체 감정 비율 파이 차트 추가 중 오류 발생: {e}")

    c.showPage()

    # 두 번째 페이지: 긍정/부정 워드클라우드
    c.setFont(font_name, 12)
    c.drawString(30, 750, "Sentiment Analysis Word Clouds")
    c.drawString(30, 735, "------------------------------------------")

    wordcloud_files = [
        os.path.join(wordcloud_dir, filename)
        for filename in os.listdir(wordcloud_dir) if filename.endswith(".png")
    ]

    y_positions = [450, 150]
    for idx, wordcloud_path in enumerate(wordcloud_files):
        try:
            img = ImageReader(wordcloud_path)
            img_width, img_height = img.getSize()
            aspect_ratio = img_height / img_width
            max_width = 400
            max_height = max_width * aspect_ratio
            y_position = y_positions[idx % 2]
            c.drawImage(wordcloud_path, 100, y_position, width=max_width, height=max_height)
        except Exception as e:
            print(f"워드클라우드 추가 중 오류 발생: {e}")


    # 세 번째 페이지부터: 개별 제품 차트
    c.setFont(font_name, 12)

    # 개별 차트 파일 로드
    individual_chart_files = [
        os.path.join(charts_dir, filename)
        for filename in os.listdir(charts_dir) if filename.endswith(".png")
    ]


    page_width, page_height = letter  
    for chart_path in individual_chart_files:
        try:
            c.showPage()

            img = ImageReader(chart_path)
            img_width, img_height = img.getSize()
            aspect_ratio = img_height / img_width

            # 이미지 크기 조정 (페이지에 꽉 차게)
            max_width = page_width - 100  # 양쪽 여백 50씩
            max_height = page_height - 100  # 위아래 여백 50씩
            if aspect_ratio > 1:  # 세로가 더 긴 경우
                scaled_height = max_height
                scaled_width = scaled_height / aspect_ratio
            else:  # 가로가 더 긴 경우
                scaled_width = max_width
                scaled_height = scaled_width * aspect_ratio

            # 이미지 배치 위치
            x_position = (page_width - scaled_width) / 2
            y_position = (page_height - scaled_height) / 2

            # 이미지 추가
            c.drawImage(chart_path, x_position, y_position, width=scaled_width, height=scaled_height)

        except Exception as e:
            print(f"개별 차트 추가 중 오류 발생: {e}")

    c.save()
    return output_path

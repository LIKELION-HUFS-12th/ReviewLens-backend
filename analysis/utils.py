# utils.py
import os
import re
import json
import pandas as pd
import requests
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from django.conf import settings
from matplotlib import font_manager, rc
import numpy as np
import matplotlib
from django.conf import settings
matplotlib.use('Agg')  # GUI 백엔드 사용하지 않도록 설정


# 폰트 설정
font_path = os.path.join(settings.BASE_DIR, 'analysis', 'fonts', 'NGULIM.TTF')  # 폰트 파일 경로
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)  # 폰트를 matplotlib에 추가
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
    return review_list, product_names

def preprocess(review_list):
    print("파일 로딩 완료. 데이터 전처리 중...")
    stopwords = ['아', '하다', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게', '뿐이다', '의거하여', '근거하여', '입각하여', '기준으로', '예하면', '예를 들면', '예를 들자면', '저', '소인', '소생', '저희', '지말고', '하지마', '하지마라', '다른', '물론', '또한', '그리고', '비길수 없다', '해서는 안된다', '뿐만 아니라', '만이 아니다', '만은 아니다', '막론하고', '관계없이', '그치지 않다', '그러나', '그런데', '하지만', '든간에', '논하지 않다', '따지지 않다', '설사', '비록', '더라도', '아니면', '만 못하다', '하는 편이 낫다', '불문하고', '향하여', '향해서', '향하다', '쪽으로', '틈타', '이용하여', '타다', '오르다', '제외하고', '이 외에', '이 밖에', '하여야', '비로소', '한다면 몰라도', '외에도', '이곳', '여기', '부터', '기점으로', '따라서', '할 생각이다', '하려고하다', '이리하여', '그리하여', '그렇게 함으로써', '하지만', '일때', '할때', '앞에서', '중에서', '보는데서', '으로써', '로써', '까지', '해야한다', '일것이다', '반드시', '할줄알다', '할수있다', '할수있어', '임에 틀림없다', '한다면', '등', '등등', '제', '겨우', '단지', '다만', '할뿐', '딩동', '댕그', '대해서', '대하여', '대하면', '훨씬', '얼마나', '얼마만큼', '얼마큼', '남짓', '여', '얼마간', '약간', '다소', '좀', '조금', '다수', '몇', '얼마', '지만', '하물며', '또한', '그러나', '그렇지만', '하지만', '이외에도', '대해 말하자면', '뿐이다', '다음에', '반대로', '반대로 말하자면', '이와 반대로', '바꾸어서 말하면', '바꾸어서 한다면', '만약', '그렇지않으면']  # 생략된 불용어들
    processed_reviews = []
    original_reviews = []

    for review in review_list:
        cleaned_original_review = re.sub(r'\s+', ' ', review).strip()
        original_reviews.append(cleaned_original_review)

        review_cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', cleaned_original_review)
        tokens = review_cleaned.split()
        filtered_tokens = [word for word in tokens if word not in stopwords]
        processed_reviews.append(' '.join(filtered_tokens))

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

def analyze_reviews_clova_studio(preprocessed_reviews_sliced):
    print("데이터 전처리 완료. 감정 분석 시작...")
    start_time = time.time()

    host = 'https://clovastudio.stream.ntruss.com'
    api_key = 'NTA0MjU2MWZlZTcxNDJiYzCfHM1duMGVmI101pNbw6DRY8rHVXsyr1bq0e2r332L'
    api_key_primary_val = 'QjtD5GeFK6qBSlyFwpYo50Vrn6aURfdCG6SySOUE'
    request_id = '26eb069872414eb480d79bd6ccf640d1'

    result_list = []
    for i, review in enumerate(preprocessed_reviews_sliced):
        progress = (i + 1) / len(preprocessed_reviews_sliced) * 100
        print(f'전체 {len(preprocessed_reviews_sliced)}개 데이터 중 {i+1}번 째 데이터 {progress:.2f}% 완료')

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
                sentiment = message_content
            else:
                print(f"예상치 않은 응답: {message_content}. 기본값 'neutral'로 설정.")
                sentiment = "neutral"

        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 에러: {e}.")
            sentiment = "error"
        except requests.exceptions.RequestException as e:
            print(f"API 요청 에러: {e}.")
            sentiment = "error"

        result_list.append({"review": review, "sentiment": sentiment})

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

    output_file_path = os.path.join(settings.BASE_DIR, 'analysis', 'results', 'sentiment_analysis_result_clovastudio.json')
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
        wordcloud = WordCloud(font_path=font_path, 
                              width=800, height=400, background_color='white').generate(text)

        # 저장 경로 설정 및 폴더 생성
        file_dir = os.path.join(settings.MEDIA_ROOT, 'wordcloud')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        
        file_path = os.path.join(file_dir, f"{sentiment}_wordcloud.png")
        wordcloud.to_file(file_path)
        print(f"워드클라우드가 '{file_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"워드클라우드 생성 중 오류 발생: {e}")

def remove_stopwords(text, stopwords):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stopwords) + r')\b'
    return re.sub(pattern, '', text)

def wordcloud_stopwords():
    return list(set(['아', '구매했네요', '제가', '좋구', '써보니', '쓰던', '없어서', '없네요', '넘', '앞으로', '다른', 'OOO', '많이', '않아', '같이', '같아요', '전', '것', '않고', '진짜', '좀', '정말', '더', '않아요', '있어', '있습니다', '좋아요', '좋고', '잘', '많이', '너무너무', '원래', '샀는데', '같습니다', '좋다길래', '괜찮아요', '있어요', '사봤어요', '보다', '요건', '그래서', '별로', '따지면', '같구요', '선택했는데적당한것', '하고', '말이죠', '구매', '너무많아서', '바르면', '입니다', '바르고', '광고', '첨에', '엄청', '열심히', '바르다', '생각보단', '요거', '제형', '00', '얼굴에', '너무', '같아요', '없고', '없어요', '좋은거', '해서', '안', '아주', '그냥', '많아', '처음', '들어있는지', '쓰고', '않을거라고', '있음', '들어요', '있고', '수', '제형의', '좋은', '근데', '않을거라고']))

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

    # 저장 경로 설정 및 폴더 생성
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
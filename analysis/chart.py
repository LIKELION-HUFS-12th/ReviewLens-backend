import json
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
import numpy as np
import matplotlib

font_path = "fonts/NGULIM.TTF"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


def load_result(path):
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

# 전체 긍정/부정/중립 카운트
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

# 전체 긍정/부정/중립 비율 파이차트로 저장
def plot_total_sentiment_pie_chart(sentiment_counts, output_path):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')
    plt.title("Sentiment Analysis Total Result")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"파이차트가 '{output_path}'에 저장되었습니다.")

# 개별 상품 리뷰 감정분석 결과 카운트
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

# 개별 상품 리뷰 감정분석 결과 시각화(막대그래프, 파이차트)
def charts(ps_counts, output_file_prefix):
    num_products = len(ps_counts)
    max_products_per_image = 5  # 한 이미지에 포함할 최대 제품 수
    total_images = (num_products // max_products_per_image) + (1 if num_products % max_products_per_image > 0 else 0)

    for img_index in range(total_images):
        start = img_index * max_products_per_image
        end = min(start + max_products_per_image, num_products)

        output_dir = os.path.dirname(output_file_prefix)
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
            axes[i, 0].set_title(f"Sentiment Analysis (Bar Chart) for {product}", loc='left')
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
        output_file = f"{output_file_prefix}_page_{img_index + 1}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {output_file}")



# 감정 분석 결과 파일 로드
results_path = 'results/sentiment_analysis_result_clovastudio.json'
results = load_result(results_path)

# 전체 감정 비율 파이 차트로 저장
total_output_path = 'results/chart/total_sentiment_pie_chart.png'
sentiment_counts = total_sentiment_count(results)
plot_total_sentiment_pie_chart(sentiment_counts, total_output_path)

# 개별 결과 파이 차트로 저장
output_file = 'results/chart/individual/individual_viz_charts.png'
ps_counts = count_for_indiv(results)
charts(ps_counts, output_file)

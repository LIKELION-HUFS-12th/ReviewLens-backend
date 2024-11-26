import re
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 데이터 로드
def load_summary_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        summary = json.load(file)
    return summary

# 긍정적인 리뷰만 필터링 
def filter_positive_reviews(summary):
    positive_reviews = [f"{review['product_name']} {review['original_content']}" for review in summary if review["document_sentiment"] == "positive"]
    return positive_reviews

# 부정적인 리뷰만 필터링 
def filter_negative_reviews(summary):
    negative_reviews = [f"{review['product_name']} {review['original_content']}" for review in summary if review["document_sentiment"] == "negative"]
    return negative_reviews

# 워드클라우드 생성 및 저장 함수
def generate_wordcloud(text, title='', sentiment=''):
    wordcloud = WordCloud(font_path='fonts/NanumGothicCoding.ttf', 
                          width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    
    file_path = f"result/wordcloud/{sentiment}_wordcloud.png"
    wordcloud.to_file(file_path)
    print(f"워드클라우드가 '{file_path}'에 저장되었습니다.")

# 불용어 제거 함수
def remove_stopwords(text, stopwords):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stopwords) + r')\b'
    return re.sub(pattern, '', text)


def generate_sentiment_wordclouds(summary, wordcloud_stopwords):
    # 긍정적인 리뷰 필터링 및 불용어 제거
    positive_reviews = filter_positive_reviews(summary)
    positive_text = " ".join(positive_reviews)
    positive_filtered_text = remove_stopwords(positive_text, wordcloud_stopwords)

    # 부정적인 리뷰 필터링 및 불용어 제거
    negative_reviews = filter_negative_reviews(summary)
    negative_text = " ".join(negative_reviews)
    negative_filtered_text = remove_stopwords(negative_text, wordcloud_stopwords)
    
    # 긍정적인 리뷰 워드클라우드 생성 및 저장
    print("긍정적인 리뷰 워드클라우드 생성됨")
    generate_wordcloud(positive_filtered_text, title='Positive Reviews Word Cloud', sentiment='positive')
    
    # 부정적인 리뷰 워드클라우드 생성 및 저장
    print("부정적인 리뷰 워드클라우드 생성됨")
    generate_wordcloud(negative_filtered_text, title='Negative Reviews Word Cloud', sentiment='negative')
    
    


summary = load_summary_from_json('project/analysis/results/sentiment_analysis_result_clovastudio.json')
wordcloud_stopwords = list(set(['아', '구매했네요', '제가', '좋구', '써보니', '쓰던', '없어서', '없네요', '넘', '앞으로', '다른', 'OOO', '많이', '않아', '같이', '같아요', '전', '것', '않고', '진짜', '좀', '정말', '더', '않아요', '있어', '있습니다', '좋아요', '좋고', '잘', '많이', '너무너무', '원래', '샀는데', '같습니다', '좋다길래', '괜찮아요', '있어요', '사봤어요', '보다', '요건', '그래서', '별로', '따지면', '같구요', '선택했는데적당한것', '하고', '말이죠', '구매', '너무많아서', '바르면', '입니다', '바르고', '광고', '첨에', '엄청', '열심히', '바르다', '생각보단', '요거', '제형' '00', '얼굴에', '너무', '같아요', '없고', '없어요', '좋은거', '해서', '안', '아주', '그냥', '많아', '처음', '들어있는지', '쓰고', '않을거라고', '있음', '들어요', '있고', '수', '제형의', '좋은', '근데', '않을거라고', 'OO인', '맘에', 'ㅇㅇㅇ', '넘', '이건', '좋았습니다', '만족해요', '제품입니다', '항상', '느낌이', '않아요', '주문했는데', '없는', '쓰다가','후기','되는', '다만', '사실', '집에', '특히', '쓸', '좋으네요', '거기다', '안녕하세요', '후기가', '좋아요', '좋네요', '좋습니다', '엄마가', '있어요', '같습니다','OO', '늘', '오늘은', 'OOO', 'OOOO', 'O', '소개해', '어떤', '사용하기', '오늘은' '싶어서' '받은', '구매해서', '안녕하세요', '도움이','좋아요', '좋고', '잘', '많이', '너무너무', '원래', '샀는데', '같습니다', '좋다길래', '괜찮아요', '있어요', '사봤어요', '보다', '요건', '그래서', '별로', '따지면', '같구요', '선택했는데적당한것', '하고', '말이죠', '구매', '너무많아서', '바르면', '입니다', '바르고', '광고', '첨에', '엄청', '열심히', '바르다', '생각보단', '요거', '제형' '00', '얼굴에', '너무', '같아요','같아요 ', ' 같아요', '없고', '없어요', '좋은거', '해서', '안', '아주', '그냥', '많아', '처음', '들어있는지', '쓰고', '않을거라고', '있음', '들어요', '있고', '수', '제형의', '좋은', '근데', '않을거라고', '이', '거', '이거', '제가', '있는', '것', '진짜', '마음에', '것 같아서', '계속', '그리고', '있어서', '정말', '그런데', '마음에', '있습니다', '마음에 듭니다', '사용할', '또', '내', '입니다', '굳이', '새로', '뭐', '같아서', '있습니다', '듭니다', '만족합니다', '없이', '좋네요', '들고', '걸', '보니까', '제품이', '수도', '다', '더', '한', '좀', '일단', '제대로', '산', '노트북을', '노트북이', '노트북은', '할', '때', '딱', '바로', '큰', '그', '조금', '제', '굉장히', '다른', 'ㅎㅎ', '이번에', '제품을', '가장', '하는', '노트북으로', '있을', '이렇게', '오래', '생각이', '같습니다', '저는', '같은', '매우', '때문에', '가지고', '않고', '게', '하나', '따로', '거의', '확실히', '것이', '않아서', '그래도', '있다는', '노트북의', '그럼', '점이', '생각보다', '전혀', '구매하게', '모두', '구매했습니다.', '그런', '좋아서', '그런지', '되어', '좋습니다', '있습니다.', '이런', '없어서', '정도', '하지만', '완전', '사용하고', '될', '저의', '건', '되어서', '전에', '있어', '볼', '것도', '함께', '훨씬', '쓰던', '아니라', '나서', '보니', '제품은', '또한', '여러', '드는', '보고', '일이', '했는데', '꼭', '제품', '같습니다', '합니다', '같아요', '하네요', '집에서', '다시', '문제가', 'ㅜㅜ', '지금', '하면', '있네요', '드네요', '않습니다', '아닙니다', '크게', '도대체', '그리고', '있는데', '부분이', '이제', '커야', '중', '되니까', '꽤', '들어볼까', '역시', '선택했습니다', '하는데', '드네요', '않는', '한번', '편이에요', '하게', '건지', '아닙니다', '괜히', '분들은', '좋은데', '있으면', '좋지', '제발', '하게', '살', '줄', '구입했는데', '정도입니다', '살짝', '그렇게', '아무리', '하지', '어렵지', '되네요', '차라리', '아쉬운', '때가', '되었는데', '아쉬운', '짠', '구매했는데', '같이', '된', '같은데', '왜', '보입니다', '이게', '아무래도', '좋을', '혼자', '건데', 'TV를', 'TV는', '않은', '전체적으로', '않네요', '보는', '게다가', '기분이', '제품에', '자주', '됩니다', '잘못', '마음이', '자꾸', '같네요', '어떻게', '노트북에', '게다가', '있었는데', '구매한', '내가', '기분이', '사서' ,'수가', '요즘', '구매했습니다', '구매했어요', 'ㅠㅠ', '000', '00', '0000', '0', '좋아요', '것','좋고', '잘', '많이', '너무너무', '원래', '샀는데', '같습니다', '좋다길래', '괜찮아요', '있어요', '사봤어요', '보다', '요건', '그래서', '별로', '따지면', '같구요', '선택했는데적당한것', '하고', '말이죠', '구매', '너무많아서', '바르면', '입니다', '바르고', '광고', '첨에', '엄청', '열심히', '바르다', '생각보단', '요거', '제형', '00', '얼굴에', '너무', '같아요', '없고', '없어요', '좋은거', '해서', '안', '아주', '그냥', '많아', '처음', '들어있는지', '쓰고', '않을거라고', '있음', '들어요', '있고', '수', '제형의', '좋은', '근데', '않을거라고','마침','합니다','갑자기','이게','가장','좋네요','이번', '좀','있어서','해주셔서','진짜','않아서','완전','마음에듭니다','정말','역시','좋습니다','만족합니다','넘','생각보다','더','그런데','계속','제품','있어', '조금','좋다고','훨씬','하네요','보고','편하고','화장실','사용하기','같네요','것','꼭','물이']))
generate_sentiment_wordclouds(summary, wordcloud_stopwords)



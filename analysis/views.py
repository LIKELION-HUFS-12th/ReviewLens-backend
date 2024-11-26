from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import pandas as pd
import os
from django.conf import settings
from .serializers import FileUploadSerializer
from .utils import preprocess, analyze_reviews_clova_studio, process_sentiment_analysis, create_test_data

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']
            
            # 파일을 저장할 경로 설정
            file_path = os.path.join(settings.MEDIA_ROOT, file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            
            # 파일 로드 및 감정 분석 처리
            try:
                review_list, product_names = self.load_file(file_path)
                original_reviews, preprocessed_reviews = preprocess(review_list)
                review_list_test, original_review_list_test, product_list_test = create_test_data(preprocessed_reviews, original_reviews, product_names, sample_size='max')
                result_list = analyze_reviews_clova_studio(review_list_test)
                sentiment_summary = process_sentiment_analysis(result_list, original_review_list_test, product_list_test)
                
                return Response(sentiment_summary, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def load_file(self, file_path):
        # 파일 로딩 함수
        text_data = pd.read_excel(file_path)
        product_names = text_data['상품명'].astype(str).tolist()
        reviews = text_data['상품평']
        review_list = reviews.astype(str).tolist()
        return review_list, product_names

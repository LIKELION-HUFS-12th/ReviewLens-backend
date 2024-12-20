from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import os
from django.conf import settings
from .serializers import FileUploadSerializer
from .utils import *
import traceback
from django.conf import settings
from django.http import JsonResponse
from django.http import FileResponse

def get_json_results(request, filename):
    if not filename.endswith('.json'):
        filename += '.json'

    file_path = os.path.join(settings.BASE_DIR, 'analysis', 'results', filename)
    print(f"DEBUG: file_path = {file_path}")

    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return JsonResponse(data, safe=False)
    else:
        return JsonResponse({'error': '파일을 찾을 수 없습니다.'}, status=404)


class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']

            file_path = os.path.join(settings.MEDIA_ROOT, file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            try:
                # 데이터 전처리 및 감정 분석 실행
                print("데이터 전처리 시작...")
                review_list, product_names = load_file(file_path)
                original_reviews, preprocessed_reviews = preprocess(review_list)
                review_list_test, original_review_list_test, product_list_test = create_test_data(
                    preprocessed_reviews, original_reviews, product_names, sample_size='max'
                )

                print("감정 분석 시작...")
                result_list = analyze_reviews_with_model(review_list_test)
                sentiment_summary = process_sentiment_analysis(result_list, original_review_list_test, product_list_test)

                # 워드클라우드 생성 및 저장
                print("워드클라우드 생성 시작...")
                generate_sentiment_wordclouds(sentiment_summary)

                # 차트 생성 및 저장
                print("차트 생성 시작...")
                results_path = os.path.join(settings.BASE_DIR, 'analysis', 'results', 'sentiment_analysis_result_kobert.json')
                results = load_result(results_path)

                # 전체 감정 비율 파이 차트 생성
                total_output_path = os.path.join(settings.MEDIA_ROOT, 'charts', 'total_sentiment_pie_chart.png')
                sentiment_counts = total_sentiment_count(results)
                plot_total_sentiment_pie_chart(sentiment_counts, total_output_path)

                # 개별 제품 리뷰 감정분석 결과 시각화
                charts_dir = os.path.join(settings.MEDIA_ROOT, 'charts', 'individual')
                output_file_prefix = os.path.join(charts_dir, 'individual_viz_charts')
                ps_counts = count_for_indiv(results)
                charts(ps_counts, output_file_prefix)

                # PDF 생성
                print("PDF 생성 시작...")
                pdf_output_path = os.path.join(settings.MEDIA_ROOT, "sentiment_report.pdf")
                wordcloud_dir = os.path.join(settings.MEDIA_ROOT, 'wordcloud')
                create_sentiment_report_pdf(
                    output_path=pdf_output_path,
                    charts_dir=charts_dir,
                    wordcloud_dir=wordcloud_dir,
                    total_chart_path=total_output_path,
                    sentiment_counts=sentiment_counts
                )

                print("모든 작업 완료")
                return Response({
                    'message': '파일이 성공적으로 업로드되고 분석, 워드클라우드, 차트 및 PDF 생성이 완료되었습니다.',
                    'file_path': file_path,
                    'pdf_report_path': pdf_output_path
                }, status=status.HTTP_200_OK)

            except Exception as e:
                traceback_str = traceback.format_exc()
                print(traceback_str)
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DownloadPDFView(APIView):
    def get(self, request, *args, **kwargs):
        pdf_path = os.path.join(settings.MEDIA_ROOT, "sentiment_report.pdf")
        if os.path.exists(pdf_path):
            return FileResponse(open(pdf_path, 'rb'), as_attachment=True, filename="sentiment_report.pdf")
        return Response({'error': 'PDF 파일을 찾을 수 없습니다.'}, status=status.HTTP_404_NOT_FOUND)
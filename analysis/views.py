import os
import json
import shutil
import threading
import traceback
import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.conf import settings
from django.http import JsonResponse, FileResponse
from .serializers import FileUploadSerializer
from .utils import *


def get_user_directory(user_id):
    """Get or create a directory for a specific user."""
    user_dir = os.path.join(settings.MEDIA_ROOT, f"user_{user_id}")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir


def get_json_results(request, filename):
    """Fetch JSON results for the authenticated user."""
    if not filename.endswith('.json'):
        filename += '.json'

    user_dir = get_user_directory(request.user.id)
    file_path = os.path.join(user_dir, filename)

    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return JsonResponse(data, safe=False)
    else:
        return JsonResponse({'error': '파일을 찾을 수 없습니다.'}, status=404)


class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']
            user_dir = get_user_directory(request.user.id)
            file_path = os.path.join(user_dir, file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            try:
                # 데이터 로딩 및 감정 분석 실행
                print(f"데이터 로딩 시작: {file_path}")
                review_list, product_names = load_file(file_path)
                review_list_test, product_list_test = create_test_data(
                    review_list, product_names, sample_size=50
                )

                print("감정 분석 시작...")
                result_list = analyze_reviews_clova_studio(review_list_test)  # Clova Studio API로 감성분석
                sentiment_summary = process_sentiment_analysis(result_list, review_list_test, product_list_test)

                print("결과 요약 생성 중...")
                sentiment_counts = total_sentiment_count(sentiment_summary)
                ps_counts = count_for_indiv(sentiment_summary)

                # PDF 생성
                print("PDF 생성 시작...")
                pdf_output_path = os.path.join(user_dir, "sentiment_report.pdf")
                create_sentiment_report_pdf_directly(
                    summary=sentiment_summary,
                    sentiment_counts=sentiment_counts,
                    ps_counts=ps_counts,
                    output_path=pdf_output_path
                )

                print("모든 작업 완료")
                return Response({
                    'message': '파일이 성공적으로 업로드되고 분석, PDF 생성이 완료되었습니다.',
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
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, *args, **kwargs):
        user_dir = get_user_directory(request.user.id)
        pdf_path = os.path.join(user_dir, "sentiment_report.pdf")

        if os.path.exists(pdf_path):
            file_handle = open(pdf_path, 'rb')
            response = FileResponse(file_handle, as_attachment=True, filename="sentiment_report.pdf")

            # Cleanup 사용자 디렉토리
            def delayed_cleanup():
                try:
                    time.sleep(1)
                    for root, dirs, files in os.walk(user_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            shutil.rmtree(dir_path)
                    print(f"사용자 {request.user.id}의 데이터가 삭제되었습니다.")
                except Exception as e:
                    print(f"사용자 데이터 삭제 중 오류 발생: {e}")

            threading.Thread(target=delayed_cleanup).start()
            return response

        return Response({'error': 'PDF 파일을 찾을 수 없습니다.'}, status=status.HTTP_404_NOT_FOUND)

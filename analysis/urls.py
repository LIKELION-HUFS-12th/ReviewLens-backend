from django.urls import path
from .views import *

urlpatterns = [
    path('upload/', FileUploadView.as_view()),  # 파일 업로드 및 분석
    path('<str:filename>.json/', get_json_results),  # JSON 결과 반환
    path('download/', DownloadPDFView.as_view()),  # PDF 다운로드
]

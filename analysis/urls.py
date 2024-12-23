from django.urls import path
from .views import *

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file_upload'),  # 파일 업로드 및 분석
    path('download/', DownloadPDFView.as_view(), name='download_pdf'),  # PDF 다운로드
    path('<str:filename>/', GetJsonView.as_view(), name='get_json'),  # JSON 결과 반환
]

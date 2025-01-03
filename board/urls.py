from django.urls import path
from . import views

app_name = 'board'

urlpatterns = [
    path('posting/', views.PostingView.as_view()),
    path('postdetail/<int:pk>/', views.PostDetailView.as_view()),
    path('postlist/', views.UserPostListView.as_view()),
    path('comment/<int:pk>/', views.CommentView.as_view()),
    path('comment/<int:post_id>/<int:comment_id>/', views.CommentDeleteView.as_view()),
]
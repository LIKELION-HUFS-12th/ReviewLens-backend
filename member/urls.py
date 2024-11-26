from django.urls import path
from . import views

app_name = 'member'

urlpatterns = [
    path('login/', views.Login.as_view()),
    path('myprofile/', views.Myprofile.as_view()),
]
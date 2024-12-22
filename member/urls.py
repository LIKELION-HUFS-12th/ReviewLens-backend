from django.urls import path
from . import views

app_name = 'member'

urlpatterns = [
    path('register/', views.RegisterView.as_view()),
    path('login/', views.LoginView.as_view()),
    path('logout/', views.LogoutView.as_view()),
    path('myprofile/', views.MyprofileView.as_view()),
]
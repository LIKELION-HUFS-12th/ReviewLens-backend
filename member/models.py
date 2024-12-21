from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.

class User(AbstractUser):
    #user_id = models.BigAutoField(primary_key=True)
    username = models.CharField(max_length=20)
    email = models.EmailField(unique=True) #이메일로 메일 전송 한다는 가정 하에 unique=True로 함
    agreement = models.BooleanField(default=False, verbose_name="이메일 수신 동의여부")

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.username
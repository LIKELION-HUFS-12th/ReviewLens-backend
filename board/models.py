from django.db import models
from member.models import User

# Create your models here.
class Post(models.Model):
    user = models.ForeignKey(User, null=False, on_delete=models.CASCADE)
    title = models.CharField(max_length=100, default = '')
    body = models.TextField(default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
    
class Comment(models.Model):
   post = models.ForeignKey(Post, null=False, on_delete=models.CASCADE, related_name="comments")
   comment = models.TextField(default='')
   created_at = models.DateTimeField(auto_now_add=True)
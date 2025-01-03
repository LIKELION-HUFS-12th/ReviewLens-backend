from rest_framework import serializers
from .models import Post, Comment
from django.utils import timezone

class PostSerializer(serializers.ModelSerializer):
    created_at = serializers.SerializerMethodField()
    class Meta:
        model = Post
        fields = ['id', 'title', 'body', 'created_at']
    def get_created_at(self, obj):
        time = timezone.localtime(obj.created_at)
        return time.strftime('%Y-%m-%d')

class CommentRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ['comment']

class CommentSerializer(serializers.ModelSerializer):
    created_at = serializers.SerializerMethodField()
    class Meta:
        model = Comment
        fields = ['id', 'user', 'comment', 'created_at']
    def get_created_at(self, obj):
        time = timezone.localtime(obj.created_at)
        return time.strftime('%Y-%m-%d')

class PostListSerializer(serializers.ModelSerializer):
    created_at = serializers.SerializerMethodField()
    class Meta:
        model = Post
        fields = ['id', 'user', 'title', 'created_at']
    def get_created_at(self, obj):
        time = timezone.localtime(obj.created_at)
        return time.strftime('%Y-%m-%d')

class PostDetailSerializer(serializers.ModelSerializer):
    created_at = serializers.SerializerMethodField()
    class Meta:
        model = Post
        fields = ['id', 'user', 'title', 'body', 'created_at', 'comments']
    def get_created_at(self, obj):
        time = timezone.localtime(obj.created_at)
        return time.strftime('%Y-%m-%d')
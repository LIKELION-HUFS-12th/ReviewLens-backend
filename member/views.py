from django.shortcuts import render
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from .models import *
from .serializers import *

# Create your views here.

class Login(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data['email']
        password = request.data['password']

        user = User.objects.get(email=email)

        if user is None:
            return Response({"message": "User not found."}, status=status.HTTP_400_BAD_REQUEST)
        if not user.check_password(password):
            print(user.check_password(password))
            return Response({"message": "Check your password."}, status=status.HTTP_400_BAD_REQUEST)
        if user is not None:
            token = TokenObtainPairSerializer.get_token(user)
            refresh_token = str(token)
            access_token = str(token.access_token)

            response = Response({
                "access" : access_token,
                "refresh" : refresh_token,
                "user" : UserLoginSerializer(user).data
            }, status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        
class Myprofile(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = User.objects.get(user_id=request.user.user_id)
        serializer = UserLoginSerializer(user, context={'request': request})
        return Response(serializer.data)
    
    def patch(self, request):
        user = User.objects.get(user_id=request.user.user_id)
        serializer = UserLoginSerializer(user, context={'request': request}, partial=True)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
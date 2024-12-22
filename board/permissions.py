from rest_framework import permissions

class IsOwnerOrUserOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        if not request.user.is_authenticated:
            return False
        if request.method == "POST" and request.user.is_authenticated:
            return True
        return obj.user == request.user
        
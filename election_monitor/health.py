from django.http import JsonResponse


class HealthCheckMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path == "/health/" and request.method in {"GET", "HEAD"}:
            return JsonResponse({"status": "ok"})

        return self.get_response(request)

from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import TemplateView, View
from django.core.files.storage import default_storage
from django.conf import settings
import os

from .models import DataUpload, ProcessedPost
from .utils.csv_processor import process_uploaded_csv


class UploadDataView(TemplateView):
    """UI for uploading CSV files"""
    template_name = 'dashboard/upload_data.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_tab'] = 'upload'
        context['tabs'] = [
            {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
            {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
            {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
            {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
            {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
            {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
        ]
        context['recent_uploads'] = DataUpload.objects.all()[:10]
        return context


class ProcessUploadView(View):
    """Handle CSV upload and processing"""
    
    def post(self, request):
        uploaded_file = request.FILES.get('csv_file')
        data_type = request.POST.get('data_type', 'custom')
        source_name = request.POST.get('source_name', 'User Upload')
        
        if not uploaded_file:
            messages.error(request, "No file uploaded")
            return redirect('upload_data')
        
        # Save uploaded file temporarily
        file_path = default_storage.save(f'uploads/{uploaded_file.name}', uploaded_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # Create upload record
        upload = DataUpload.objects.create(
            uploaded_file=file_path,
            original_filename=uploaded_file.name,
            uploaded_by=request.user.username if request.user.is_authenticated else 'anonymous',
            data_type=data_type,
            status='processing'
        )
        
        # Process the file
        success, message, count = process_uploaded_csv(full_path, data_type, source_name)
        
        # Update upload record
        upload.status = 'completed' if success else 'failed'
        upload.processing_log = message
        upload.records_processed = count if success else 0
        upload.save()
        
        # Show result
        if success:
            messages.success(request, f"✅ {message}")
        else:
            messages.error(request, f"❌ {message}")
        
        return redirect('upload_data')

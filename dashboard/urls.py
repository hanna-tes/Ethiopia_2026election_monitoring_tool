from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # === MAIN VIEWS ===
    path('', views.HomeView.as_view(), name='home'),  # ✅ This is the missing one!
    path('narratives/', views.NarrativesView.as_view(), name='narratives'),
    path('lexicons/', views.LexiconsView.as_view(), name='lexicons'),
    path('peps/', views.PEPsView.as_view(), name='peps'),
    path('networks/', views.NetworksView.as_view(), name='networks'),
    path('lexicon-management/', views.LexiconManagementView.as_view(), name='lexicon_management'),
    
    # === UPLOAD URLS ===
    path('upload/', views.UploadDataView.as_view(), name='upload_data'),
    path('upload/process/', views.ProcessUploadView.as_view(), name='process_upload'),
    path('upload/clear/', views.ClearDataView.as_view(), name='clear_data'),
    
    # === API ENDPOINTS ===
    path('api/scan-text/', views.scan_text_api, name='scan_text_api'),
    path('api/export-posts/', views.export_posts_api, name='export_posts_api'),
    path('api/network-graph/', views.generate_network_graph, name='generate_network_graph'),
]

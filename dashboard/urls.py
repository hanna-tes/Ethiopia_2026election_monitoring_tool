from django.urls import path
from . import views

# This creates the 'dashboard:' namespace for all URLs below
#app_name = 'dashboard'

urlpatterns = [
    # === MAIN DASHBOARD VIEWS ===
    # These match the 'url_name' used in your navigation tabs
    path('', views.HomeView.as_view(), name='home'),
    path('narratives/', views.NarrativesView.as_view(), name='narratives'),
    path('lexicons/', views.LexiconsView.as_view(), name='lexicons'),
    path('peps/', views.PEPsView.as_view(), name='peps'),
    path('networks/', views.NetworksView.as_view(), name='networks'),
    path('lexicon-management/', views.LexiconManagementView.as_view(), name='lexicon_management'),
    path('peps/', views.PEPsHubView.as_view(), name='peps_hub'),
    path('peps/data/', views.PEPsDataView.as_view(), name='peps_data'),
    
    # === UPLOAD & DATA MANAGEMENT ===
    # These handle the sidebar form actions
    path('upload/', views.UploadDataView.as_view(), name='upload_data'),
    path('upload/process/', views.ProcessUploadView.as_view(), name='process_upload'),
    path('upload/clear/', views.ClearDataView.as_view(), name='clear_data'),
    
    # === API ENDPOINTS ===
    # Used for AJAX calls like the Network Graph and Real-time Scanning
    path('api/scan-text/', views.scan_text_api, name='scan_text_api'),
    path('api/export-posts/', views.export_posts_api, name='export_posts_api'),
    path('api/network-graph/', views.generate_network_graph, name='generate_network_graph'),
]

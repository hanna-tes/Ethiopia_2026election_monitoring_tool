from django.urls import path
from . import views

urlpatterns = [
    # === MAIN DASHBOARD VIEWS ===
    path('', views.HomeView.as_view(), name='home'),
    path('narratives/', views.NarrativesView.as_view(), name='narratives'),
    path('lexicons/', views.LexiconsView.as_view(), name='lexicons'),
    path('peps/', views.PEPsView.as_view(), name='peps'),          # Main PEPs tab
    path('networks/', views.NetworksView.as_view(), name='networks'),
    path('lexicon-management/', views.LexiconManagementView.as_view(), name='lexicon_management'),
    
    # === PEPs DATA VIEWS (FIXED CONFLICT) ===
    path('peps/registry/', views.PEPsHubView.as_view(), name='peps_hub'),   # File cards (HoPR, RC, Executive)
    path('peps/data/', views.PEPsDataView.as_view(), name='peps_data'),     # Spreadsheet view
    
    # === UPLOAD & DATA MANAGEMENT ===
    path('upload/', views.UploadDataView.as_view(), name='upload_data'),
    path('upload/process/', views.ProcessUploadView.as_view(), name='process_upload'),
    path('upload/clear/', views.ClearDataView.as_view(), name='clear_data'),
    
    # === API ENDPOINTS ===
    path('api/scan-text/', views.scan_text_api, name='scan_text_api'),
    path('api/export-posts/', views.export_posts_api, name='export_posts_api'),
    path('api/network-graph/', views.generate_network_graph, name='generate_network_graph'),
]

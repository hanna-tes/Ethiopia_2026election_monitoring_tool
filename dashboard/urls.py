from django.urls import path
from . import views

urlpatterns = [
    # Existing tab URLs...
    path('', views.HomeView.as_view(), name='home'),
    path('narratives/', views.NarrativesView.as_view(), name='narratives'),
    path('lexicons/', views.LexiconsView.as_view(), name='lexicons'),
    path('peps/', views.PEPsView.as_view(), name='peps'),
    path('networks/', views.NetworksView.as_view(), name='networks'),
    path('lexicon-management/', views.LexiconManagementView.as_view(), name='lexicon_management'),
    
    # Upload URLs
    path('upload/', views.UploadDataView.as_view(), name='upload_data'),
    path('upload/process/', views.ProcessUploadView.as_view(), name='process_upload'),
]

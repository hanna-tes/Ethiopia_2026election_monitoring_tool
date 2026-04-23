import os
from pathlib import Path
from dotenv import load_dotenv

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY', default='django-insecure-change-this-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env('DEBUG', default=True)  # Set to False in production

ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=[
    'localhost', 
    '127.0.0.1',
    # Add your EC2 public IP here
    '34.253.216.206',  # Replace with your actual EC2 IP
])

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party
    'rest_framework',
    'corsheaders',
    
    # Local
    'dashboard',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # For static files
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'election_monitor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'election_monitor.wsgi.application'

# Database - Use SQLite for simplicity (you can switch to PostgreSQL later)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
# ... (keep default)

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Africa/Addis_Ababa'  # Ethiopia timezone
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
}

# CORS Settings (for development)
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOWED_ORIGINS = env.list('CORS_ALLOWED_ORIGINS', default=[
        f'http://{ALLOWED_HOSTS[0]}:8504',
    ])

# Groq API
GROQ_API_KEY = env('GROQ_API_KEY', default='')
GROQ_MODEL = env('GROQ_MODEL', default='llama-3.1-8b-instant')

# Data source URLs (your GitHub links)
MELTWATER_URL = env('MELTWATER_URL', default='https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/Ethiopia_NOV2025_Apri2026X.csv')
CIVICSIGNALS_URL = env('CIVICSIGNALS_URL', default='https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaCivicsignalApril17.csv')
TIKTOK_URL = env('TIKTOK_URL', default='https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaTikTokApril17.csv')
OPENMEASURES_URL = env('OPENMEASURES_URL', default='https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaopenmeasuresApri17.csv')
ORIGINAL_POSTS_URL = env('ORIGINAL_POSTS_URL', default='https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/EthiopiaMeltwaterApril17Original1.csv')
PEPS_CSV_URL = env('PEPS_CSV_URL', default='https://raw.githubusercontent.com/hanna-tes/Ethiopia-election-monitoring/refs/heads/main/Ethiopian%20PEP%20Social%20Media%20Verification%20Dataset%20-%20Dataset.csv')

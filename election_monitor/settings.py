"""
Django settings for Ethiopia Election Monitor project.
"""
import os
from pathlib import Path
import environ

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

#LOAD .ENV FILE HERE
from dotenv import load_dotenv
load_dotenv(BASE_DIR / '.env')


# Initialize environment variables
env = environ.Env(
    DEBUG=(bool, True),
    SECRET_KEY=(str, 'django-insecure-change-this-in-production'),
)

# CSRF Configuration for Nginx Reverse Proxy
CSRF_TRUSTED_ORIGINS = [
    'https://ethio-monitor.investigate.africa',
    'http://ethio-monitor.investigate.africa',
    'http://localhost',
    'http://127.0.0.1',
]

# Trust the X-Forwarded-Proto header from Nginx
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# CSRF cookie settings
CSRF_COOKIE_SECURE = False  # Set to True if using HTTPS
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = 'Lax'


# Read .env file if it exists (optional)
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# SECURITY
SECRET_KEY = env('SECRET_KEY')
DEBUG = env('DEBUG')
ALLOWED_HOSTS = ['*', 'localhost', '127.0.0.1', '0.0.0.0', '52.49.201.188', 'ethio-monitor.investigate.africa']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'huey.contrib.djhuey',
    'django_q',
    
    # Third-party
    'rest_framework',
    'corsheaders',
    
    # Local
    'dashboard',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
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
        'DIRS': [BASE_DIR / 'templates'],  # or your template dirs
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            
            'builtins': ['dashboard.templatetags.custom_filters'],
        },
    },
]

WSGI_APPLICATION = 'election_monitor.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'ethiopia_election_db',
        'USER': 'ethiopia_user',
        'PASSWORD': 'ElectionDbSecure2026!',  
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Africa/Addis_Ababa'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'dashboard' / 'static',  
]

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# paths to your PEP Excel files (local or absolute)
PEP_FILES = {
    'hopr': os.getenv('HOPR_FILE_PATH', str(BASE_DIR / 'media' / 'peps' / 'HoPR_Candidates.xlsx')),
    'regional': os.getenv('REGIONAL_FILE_PATH', str(BASE_DIR / 'media' / 'peps' / 'Regional_Candidates.xlsx')),
    'executive': os.getenv('EXECUTIVE_FILE_PATH', str(BASE_DIR / 'media' / 'peps' / 'Executive_Members.xlsx')),
}

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
}

# CORS
CORS_ALLOW_ALL_ORIGINS = True

# Django-Q2 Configuration using Redis Broker
Q_CLUSTER = {
    'name': 'ethiopia_monitor_cluster',
    'workers': 2,                # Number of parallel worker processes
    'recycle': 500,              # Restart workers after 500 tasks to prevent memory leaks
    'timeout': 600,              # Automatically time out a task if it takes more than 10 mins
    'retry': 700,                # Retry time slightly higher than timeout
    'django_redis': 'default',   # Tells Django-Q to use your existing Redis cache config
}

# ── REDIS CACHE ──────────────────────────────────────────────
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "SOCKET_CONNECT_TIMEOUT": 5,
            "SOCKET_TIMEOUT": 5,
        }
    }
}

# Groq API
GROQ_API_KEY = env('GROQ_API_KEY', default='')
GROQ_MODEL = env('GROQ_MODEL', default='llama-3.3-70b-versatile')

# Data source URLs
MELTWATER_URL = env('MELTWATER_URL', default='')
CIVICSIGNALS_URL = env('CIVICSIGNALS_URL', default='')
TIKTOK_URL = env('TIKTOK_URL', default='')
OPENMEASURES_URL = env('OPENMEASURES_URL', default='')
ORIGINAL_POSTS_URL = env('ORIGINAL_POSTS_URL', default='')
PEPS_CSV_URL = env('PEPS_CSV_URL', default='')

# Allow uploads up to 100MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 534773760  # 510 MB (510 * 1024 * 1024)

FILE_UPLOAD_MAX_MEMORY_SIZE = 26214400  # 25 MB

# Gama model
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
GEMMA_BASE_MODEL_PATH = 'unsloth/gemma-4-e4b-it-unsloth-bnb-4bit'
GEMMA_LORA_ADAPTER_PATH = './model_cache/gemma-lora-hate-speech'
GEMMA_TTP_MODEL_PATH = './model_cache/gemma-merged-4bit'
GEMMA_LOKA_MODEL_PATH = os.path.join(BASE_DIR, 'dashboard', 'model_cache', 'gemma_4_multiclass_hate_lexicon_lora')

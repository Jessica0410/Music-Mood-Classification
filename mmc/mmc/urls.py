"""
URL configuration for mmc project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from django.urls import re_path as url
from Test import views 

urlpatterns = [
    path('', views.audioUpload, name='AudioUpload'),
    url(r'^ModelSelection',views.toModelSelection,name="toModelSelection"),
    path('', views.getBack, name="toAudioUpload"),
    url(r'^Prediction', views.toPrediction, name="toPrediction"),
    path('returnModelSelection/', views.returnModelSelection, name="RMS"),
    path('', views.getBack, name="end"),
]
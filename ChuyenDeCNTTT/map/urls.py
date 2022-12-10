from django.contrib import admin
from django.urls import path
from map import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.preview, name='preview'),
    path('data', views.predata, name='predata'),
    path('extract', views.extract, name='extract')
]
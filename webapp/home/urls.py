from django.urls import path 
from . import views


urlpatterns = [
    path('', views.index, name='home'),
    path('upload', views.upload_dataset, name='upload'),
    path('train', views.train, name='train'),
     path('predictions', views.predictions, name='predictions'),
]
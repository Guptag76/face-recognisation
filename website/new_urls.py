from django.urls import path
from website import views

urlpatterns=[
    path('',views.index2,name='index2'),
]
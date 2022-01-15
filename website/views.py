from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from website.form import FaceRecognisationForm
from website.machine_learning import pipeline_model
from website.models import FaceRecognisation
import os
from django.conf import settings





def index(request):
    form = FaceRecognisationForm()
    
    if request.method =='POST':
        form = FaceRecognisationForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            save_img=form.save(commit=True)
            
            primary_key=save_img.pk
            imgobj=FaceRecognisation.objects.get(pk=primary_key)
            fileroot=str(imgobj.image)
            file_path=os.path.join(settings.MEDIA_ROOT,fileroot)
            result=pipeline_model(file_path)
            print(result)
    
            return render(request,'index.html',{'form':form,'upload':True,'results':result})

    
    return render(request,'index.html',{'form':form,'upload':False})

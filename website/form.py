from django import forms
from website.models import FaceRecognisation

class FaceRecognisationForm(forms.ModelForm):
    
    class Meta:
        model = FaceRecognisation
        fields = ['image']
        
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fields['image'].widget.attrs.update({'class':'form-control'})
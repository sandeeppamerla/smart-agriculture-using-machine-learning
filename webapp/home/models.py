from django.db import models
import os

# Create your models here.
 
class dataset(models.Model):
    file = models.FileField(upload_to="home/static/home/dataset")

    def filename(self):
        return os.path.basename(self.file.name)
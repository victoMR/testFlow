from django.db import models

class Documento(models.Model):
    nombre = models.CharField(max_length=100)
    archivo = models.FileField(upload_to='documentos/')
    procesado = models.BooleanField(default=False)

    def __str__(self):
        return self.nombre

class ProblemaMatematico(models.Model):
    id = models.AutoField(primary_key=True)
    latex = models.TextField()
    tipo_problema = models.CharField(max_length=100)
    fecha_escaneo = models.DateTimeField(auto_now_add=True)
    usos = models.IntegerField(default=0)

    def __str__(self):
        return f"Problema {self.id}: {self.tipo_problema}"
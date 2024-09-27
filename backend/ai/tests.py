from django.test import TestCase
from .models import Documento, ProblemaMatematico

class DocumentoModelTest(TestCase):
    def test_documento_creation(self):
        documento = Documento.objects.create(nombre="Test Documento")
        self.assertEqual(documento.nombre, "Test Documento")
        self.assertFalse(documento.procesado)

class ProblemaMatematicoModelTest(TestCase):
    def test_problema_matematico_creation(self):
        problema = ProblemaMatematico.objects.create(
            latex="x^2 + y^2 = r^2",
            tipo_problema="Ecuación"
        )
        self.assertEqual(problema.latex, "x^2 + y^2 = r^2")
        self.assertEqual(problema.tipo_problema, "Ecuación")
        self.assertEqual(problema.usos, 0)

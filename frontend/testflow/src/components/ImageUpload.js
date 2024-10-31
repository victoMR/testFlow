import React, { useState } from 'react';
import { 
  Button, 
  Container, 
  Typography, 
  Alert, 
  Paper,
  Box,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Chip
} from '@mui/material';
import { Upload, Image as ImageIcon, AlertCircle } from 'lucide-react';

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);

  const validateImage = (file) => {
    if (!file) return "Selecciona una imagen";
    if (!ALLOWED_TYPES.includes(file.type)) 
      return "Solo se permiten imágenes en formato JPG, PNG o WebP";
    if (file.size > MAX_FILE_SIZE) 
      return "La imagen no debe superar 5MB";
    return null;
  };

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    setError(null);
    setResult(null);
    
    const validationError = validateImage(selectedFile);
    if (validationError) {
      setError(validationError);
      setFile(null);
      setPreview(null);
      return;
    }

    // Crear preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
    
    setFile(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('imagen', file);

    setLoading(true);
    setError(null);
    setProgress(0);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/procesar_imagen/', {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Error al procesar la imagen');
      }

      setResult(data.problema);
      console.log("Respuesta del servidor:", data);
    } catch (error) {
      console.error("Error al enviar la imagen:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Procesamiento de Imagen
        </Typography>

        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 2 }}
            icon={<AlertCircle />}
          >
            {error}
          </Alert>
        )}

        <Box sx={{ mb: 3 }}>
          <input
            accept="image/jpeg,image/png,image/webp"
            style={{ display: 'none' }}
            id="image-upload"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="image-upload">
            <Button
              variant="outlined"
              component="span"
              fullWidth
              startIcon={<Upload />}
              sx={{ mb: 1 }}
            >
              Seleccionar Imagen
            </Button>
          </label>

          {preview && (
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <img 
                src={preview} 
                alt="Preview" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '300px',
                  borderRadius: '4px'
                }}
              />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                {file?.name} ({(file?.size / 1024 / 1024).toFixed(2)}MB)
              </Typography>
            </Box>
          )}
        </Box>

        {loading && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress />
            <Typography variant="body2" color="textSecondary" align="center">
              Procesando imagen...
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={handleSubmit}
          disabled={!file || loading}
          startIcon={<ImageIcon />}
        >
          Procesar Imagen
        </Button>

        {result && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Fórmula Detectada
            </Typography>
            <Paper elevation={1} sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Typography variant="body1">
                  {result.formula}
                </Typography>
                <Chip 
                  label={result.tipo} 
                  size="small" 
                  color="primary" 
                  variant="outlined"
                />
              </Box>
              {result.latex_image && (
                <Box sx={{ mt: 2, textAlign: 'center' }}>
                  <img 
                    src={`data:image/png;base64,${result.latex_image}`}
                    alt="Fórmula renderizada"
                    style={{ maxWidth: '100%' }}
                  />
                </Box>
              )}
            </Paper>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ImageUpload;

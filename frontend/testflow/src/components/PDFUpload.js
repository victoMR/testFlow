import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { 
  Button, 
  Container, 
  Typography, 
  LinearProgress, 
  Alert, 
  Paper,
  Box,
  List,
  ListItem,
  ListItemText,
  Chip
} from '@mui/material';
import { Upload, FileText, AlertCircle } from 'lucide-react';

const PDFUpload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);
  const [pageCount, setPageCount] = useState(0);

  const validatePDF = (file) => {
    if (!file) return "Selecciona un archivo PDF";
    if (file.type !== "application/pdf") return "El archivo debe ser PDF";
    if (file.size > 50 * 1024 * 1024) return "El archivo no debe superar 50MB";
    return null;
  };

  const checkPageCount = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const typedarray = new Uint8Array(e.target.result);
        try {
          const pdf = await window.pdfjsLib.getDocument(typedarray).promise;
          const numPages = pdf.numPages;
          if (numPages > 50) reject("El PDF no debe tener más de 50 páginas");
          if (numPages < 1) reject("El PDF debe tener al menos 1 página");
          resolve(numPages);
        } catch (error) {
          reject("Error al leer el PDF");
        }
      };
      reader.readAsArrayBuffer(file);
    });
  };

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    setError(null);
    setResults([]);
    
    const validationError = validatePDF(selectedFile);
    if (validationError) {
      setError(validationError);
      return;
    }

    try {
      const numPages = await checkPageCount(selectedFile);
      setPageCount(numPages);
      setFile(selectedFile);
    } catch (err) {
      setError(err);
      setFile(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('pdf', file);

    setLoading(true);
    setError(null);
    setProgress(0);

    try {
      const response = await axios.post(
        'http://127.0.0.1:8000/api/procesar_pdf/', 
        formData,
        {
          headers: { 
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
          withCredentials: true
        }
      );

      setResults(response.data.problemas);
      console.log("Respuesta del servidor:", response.data);
    } catch (error) {
      console.error("Error al enviar el PDF:", error);
      setError(error.response?.data?.error || "Error al procesar el PDF");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Procesamiento de PDF
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
            accept="application/pdf"
            style={{ display: 'none' }}
            id="pdf-upload"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="pdf-upload">
            <Button
              variant="outlined"
              component="span"
              fullWidth
              startIcon={<Upload />}
              sx={{ mb: 1 }}
            >
              Seleccionar PDF
            </Button>
          </label>

          {file && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="body2" color="textSecondary">
                Archivo: {file.name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Páginas: {pageCount}
              </Typography>
            </Box>
          )}
        </Box>

        {loading && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="body2" color="textSecondary" align="center">
              Procesando... {progress}%
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={handleSubmit}
          disabled={!file || loading}
          startIcon={<FileText />}
        >
          Procesar PDF
        </Button>

        {results.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Fórmulas Detectadas
            </Typography>
            <List>
              {results.map((problema, index) => (
                <ListItem key={index} divider>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body1">
                          {problema.formula}
                        </Typography>
                        <Chip 
                          label={problema.tipo} 
                          size="small" 
                          color="primary" 
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={`Confianza: ${problema.confidence || 'N/A'}%`}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default PDFUpload;

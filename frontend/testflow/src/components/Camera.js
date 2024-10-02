import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera } from 'lucide-react';
import { 
  Container, 
  Typography, 
  Button, 
  Alert, 
  Box, 
  CircularProgress,
  Paper
} from '@mui/material';

const CameraPage = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [frameCaptureInterval, setFrameCaptureInterval] = useState(null);

  const startVideo = useCallback(async () => {
    console.log("Intentando acceder a la cámara...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          console.log("Metadatos del video cargados, iniciando reproducción...");
          videoRef.current.play().catch(err => {
            console.error("Error al reproducir el video: ", err);
          });
          setIsStreaming(true);
          console.log("La cámara está transmitiendo...");
        };
      }
    } catch (err) {
      console.error("Error al acceder a la cámara: ", err);
      setError("No se pudo acceder a la cámara. Asegúrate de que está conectada y permite el acceso.");
    }
  }, []);

  useEffect(() => {
    console.log("Inicializando flujo de video...");
    startVideo();
    return () => {
      const currentVideo = videoRef.current;
      if (currentVideo && currentVideo.srcObject) {
        const tracks = currentVideo.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        currentVideo.srcObject = null; // Limpiar la fuente
        console.log("Pistas de video detenidas y limpiadas.");
      }
      // Limpiar el intervalo si existe
      if (frameCaptureInterval) {
        clearInterval(frameCaptureInterval);
        console.log("Intervalo de captura de frame limpiado.");
      }
    };
  }, [startVideo, frameCaptureInterval]);

  const captureFrame = useCallback(async () => {
    if (!canvasRef.current || !videoRef.current) {
      console.warn("Referencia de canvas o video no disponible.");
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    console.log("Frame capturado desde el video.");

    canvas.toBlob(async (blob) => {
      if (!blob) {
        console.warn("Creación de blob fallida.");
        return;
      }

      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');

      setLoading(true);
      console.log("Enviando frame al backend...");

      try {
        // Reemplaza con tu endpoint API real
        const response = await fetch('https://your-backend.com/api/process-frame', {
          method: 'POST',
          body: formData,
        });
        if (response.ok) {
          console.log("Frame enviado con éxito.");
        } else {
          console.error("Error al enviar frame: ", response.statusText);
          setError("Error al enviar frame. Por favor, inténtalo de nuevo.");
        }
      } catch (error) {
        console.error("Error al enviar frame: ", error);
        setError("Error al enviar frame. Por favor, inténtalo de nuevo.");
      } finally {
        setLoading(false);
      }
    }, 'image/jpeg');
  }, []);

  const startFrameCapture = useCallback(() => {
    if (!isStreaming) return;
    console.log("Iniciando captura de frame cada 2 segundos...");
    const id = setInterval(captureFrame, 2000);
    setFrameCaptureInterval(id);
  }, [isStreaming, captureFrame]);

  useEffect(() => {
    if (isStreaming) {
      startFrameCapture();
    }
    return () => {
      if (frameCaptureInterval) {
        clearInterval(frameCaptureInterval);
        console.log("Intervalo de captura de frame limpiado al desmontar el componente.");
      }
    };
  }, [isStreaming, startFrameCapture, frameCaptureInterval]);

  return (
    <Container maxWidth="sm">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Cámara en Tiempo Real
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Paper 
          elevation={3} 
          sx={{ 
            position: 'relative', 
            aspectRatio: '16/9', 
            overflow: 'hidden', 
            mb: 2 
          }}
        >
          <video 
            ref={videoRef} 
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            autoPlay
            playsInline
            muted
          />
          {loading && (
            <Box 
              sx={{ 
                position: 'absolute', 
                top: 0, 
                left: 0, 
                right: 0, 
                bottom: 0, 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center', 
                backgroundColor: 'rgba(0, 0, 0, 0.5)' 
              }}
            >
              <CircularProgress />
            </Box>
          )}
        </Paper>

        <canvas ref={canvasRef} style={{ display: 'none' }} />
        
        <Typography variant="body1" align="center" gutterBottom>
          Capturando frames y enviándolos al backend cada 2 segundos...
        </Typography>
        
        <Button 
          variant="contained" 
          color="primary" 
          fullWidth 
          onClick={captureFrame}
          disabled={loading || !isStreaming}
          startIcon={<Camera />}
        >
          Capturar Frame Manualmente
        </Button>
      </Box>
    </Container>
  );
};

export default CameraPage;

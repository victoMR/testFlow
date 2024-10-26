import React, { useRef, useEffect, useState, useCallback } from "react";
import { Camera } from "lucide-react";
import {
  Container,
  Typography,
  Button,
  Alert,
  Box,
  CircularProgress,
  Paper,
} from "@mui/material";

const CameraPage = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Función para iniciar el stream de la cámara
  const startVideo = useCallback(async () => {
    try {
      setError(null); // Reiniciar error
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current
            .play()
            .then(() => setIsStreaming(true))
            .catch((err) => {
              console.error("Error al reproducir el video: ", err);
              setError("No se pudo iniciar la reproducción del video.");
            });
        };
      }
    } catch (err) {
      console.error("Error al acceder a la cámara: ", err);
      setError(
        "No se pudo acceder a la cámara. Asegúrate de que está conectada y permite el acceso."
      );
    }
  }, []);

  // Funcion para apagar la cámara
  const stopVideo = useCallback(() => {
    const currentVideo = videoRef.current;
    if (currentVideo && currentVideo.srcObject) {
      const tracks = currentVideo.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      currentVideo.srcObject = null; // Limpiar la fuente
    }
    setIsStreaming(false);
  }, []);

  // Hook para iniciar la cámara al cargar el componente
  useEffect(() => {
    startVideo();

    return () => {
      stopVideo(); // Apaga la cámara al desmontar el componente
    };
  }, [startVideo, stopVideo]);

  // Función para capturar un fotograma y enviarlo al backend
  const captureFrame = useCallback(async () => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;

      setLoading(true);
      try {
        const response = await fetch(
          "http://127.0.0.1:8000/api/procesar_fotograma/",
          {
            method: "POST",
            body: blob,
            headers: {
              "Content-Type": "image/jpeg",
              "Access-Control-Allow-Origin": "*",
              "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
              "Access-Control-Allow-Headers": "Content-Type",
            },
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Respuesta del servidor:", data);
      } catch (err) {
        console.error("Error al enviar frame:", err);
        setError(`Error al enviar frame: ${err.message}`);
      } finally {
        setLoading(false);
      }
    }, "image/jpeg");
  }, []);

  // Función para iniciar la captura de fotogramas en intervalos definidos
  const startFrameCapture = useCallback(() => {
    const captureInterval = setInterval(() => {
      if (isStreaming) {
        captureFrame();
      }
    }, 5000); // Capturar cada 5 segundos (5000 ms)

    return () => clearInterval(captureInterval);
  }, [isStreaming, captureFrame]);

  // Hook para iniciar la captura automática cuando la cámara comienza a transmitir
  useEffect(() => {
    if (isStreaming) {
      const stopCapture = startFrameCapture();
      return () => stopCapture();
    }
  }, [isStreaming, startFrameCapture]);

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
            position: "relative",
            aspectRatio: "16/9",
            overflow: "hidden",
            mb: 2,
          }}
        >
          <video
            ref={videoRef}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            autoPlay
            playsInline
            muted
          />
          {loading && (
            <Box
              sx={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: "rgba(0, 0, 0, 0.5)",
              }}
            >
              <CircularProgress />
            </Box>
          )}
        </Paper>
        <canvas ref={canvasRef} style={{ display: "none" }} />
        <Typography variant="body1" align="center" gutterBottom>
          Capturando frames y enviándolos al backend cada 5 segundos...
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
        <Button
          variant="outlined"
          color="error" // Color de botón corregido
          fullWidth
          onClick={stopVideo}
          disabled={!isStreaming}
          sx={{ mt: 2 }}
        >
          Apagar Cámara {/* Añadido el texto del botón */}
        </Button>
      </Box>
    </Container>
  );
};

export default CameraPage;

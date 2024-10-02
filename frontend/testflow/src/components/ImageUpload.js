import React, { useState } from 'react';
import axios from 'axios';
import { Button, Container, Typography, TextField } from '@mui/material';

const ImageUpload = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('imagen', file);

    try {
      const response = await axios.post('https://tu-backend.com/procesar-imagen/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log("Respuesta del servidor:", response.data);
    } catch (error) {
      console.error("Error al enviar la imagen:", error);
    }
  };

  return (
    <Container>
      <Typography variant="h4" component="h1" gutterBottom>
        Subir y Procesar Imagen
      </Typography>
      <form onSubmit={handleSubmit}>
        <TextField type="file" accept="image/*" onChange={handleFileChange} />
        <Button variant="contained" color="primary" type="submit" style={{ marginTop: 10 }}>
          Procesar Imagen
        </Button>
      </form>
    </Container>
  );
};

export default ImageUpload;

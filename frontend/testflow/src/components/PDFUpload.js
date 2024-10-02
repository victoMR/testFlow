import React, { useState } from 'react';
import axios from 'axios';
import { Button, Container, Typography, TextField } from '@mui/material';

const PDFUpload = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('pdf', file);

    try {
      const response = await axios.post('https://tu-backend.com/procesar-pdf/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log("Respuesta del servidor:", response.data);
    } catch (error) {
      console.error("Error al enviar el PDF:", error);
    }
  };

  return (
    <Container>
      <Typography variant="h4" component="h1" gutterBottom>
        Subir y Procesar PDF
      </Typography>
      <form onSubmit={handleSubmit}>
        <TextField type="file" accept="application/pdf" onChange={handleFileChange} />
        <Button variant="contained" color="primary" type="submit" style={{ marginTop: 10 }}>
          Procesar PDF
        </Button>
      </form>
    </Container>
  );
};

export default PDFUpload;

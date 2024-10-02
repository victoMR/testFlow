import React from 'react';
import { Container, Button, Typography, Box, AppBar, Toolbar, IconButton } from '@mui/material';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { Home as HomeIcon } from 'lucide-react';
import CameraPage from './components/Camera';
import PdfProcessor from './components/PDFUpload';
import ImageProcessor from './components/ImageUpload';

const NavButton = ({ to, color, children }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Button
      variant={isActive ? "contained" : "outlined"}
      color={color}
      component={Link}
      to={to}
      fullWidth
      sx={{ mb: 2 }}
    >
      {children}
    </Button>
  );
};

const App = () => {
  return (
    <Router>
      <AppBar position="static" color="primary" sx={{ mb: 4 }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" component={Link} to="/">
            <HomeIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Detección de Problemas Matemáticos
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="sm">
        {/* Página de inicio */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/abrir-camara" element={<CameraPage />} />
          <Route path="/procesar-pdf" element={<PdfProcessor />} />
          <Route path="/procesar-imagen" element={<ImageProcessor />} />
        </Routes>

        <Box sx={{ my: 4 }}>
          {/* Botones de navegación debajo de Home */}
          <NavButton to="/abrir-camara" color="primary">Abrir Cámara</NavButton>
          <NavButton to="/procesar-pdf" color="secondary">Procesar PDF</NavButton>
          <NavButton to="/procesar-imagen" color="success">Procesar Imagen</NavButton>
        </Box>
      </Container>
    </Router>
  );
};

const Home = () => {
  return (
    <Container>
      <Typography variant="h4" component="h1" gutterBottom>
        Bienvenido a la Aplicación de Procesamiento
      </Typography>
      <Typography variant="body1" gutterBottom>
        Seleccione una opción:
      </Typography>
    </Container>
  );
};

export default App;

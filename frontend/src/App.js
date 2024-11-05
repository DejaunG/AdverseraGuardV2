import React, { useState } from 'react';
import {
  Button,
  Card,
  CardContent,
  Container,
  Grid,
  MenuItem,
  Select,
  TextField,
  Typography,
  Switch,
  FormControlLabel,
  Collapse,
  Snackbar,
  FormControl,
  InputLabel
} from '@mui/material';
import MuiAlert from '@mui/material/Alert';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4caf50',
    },
    secondary: {
      main: '#81c784',
    },
    background: {
      default: '#1e1e1e',
      paper: '#2e2e2e',
    },
  },
});

const AdversaGuardUI = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [adversarialImage, setAdversarialImage] = useState(null);
  const [method, setMethod] = useState('fgsm');
  const [epsilon, setEpsilon] = useState(0.03);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [stealthMode, setStealthMode] = useState(false);
  const [params, setParams] = useState({
    alpha: 0.01,
    num_iter: 40,
    num_classes: 1000,
    overshoot: 0.02,
    max_iter: 50,
    pixels: 1,
    pop_size: 400,
    delta: 0.2,
    max_iter_uni: 50,
    max_iter_df: 100
  });
  const [error, setError] = useState(null);
  const [originalPrediction, setOriginalPrediction] = useState(null);
  const [adversarialPrediction, setAdversarialPrediction] = useState(null);
  const [imageType, setImageType] = useState('auto');

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      console.log("Image uploaded:", file.name);
      if (imageType === 'auto') {
        autoDetectImageType(file);
      }
    }
  };

  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setParams({ ...params, [name]: parseFloat(value) });
  };

  const autoDetectImageType = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/detect_image_type', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setImageType(data.image_type);
    } catch (e) {
      console.error("Error detecting image type:", e);
      setError(`Failed to detect image type. Error: ${e.message}`);
    }
  };

  const generateAdversarial = async () => {
    console.log("Generate button clicked");
    if (!selectedImage) {
      setError("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    try {
      const imageFile = await fetch(selectedImage).then(r => r.blob());
      formData.append('file', imageFile, 'image.jpg');
      formData.append('method', method);
      formData.append('epsilon', epsilon);
      formData.append('stealth_mode', stealthMode);
      formData.append('image_type', imageType === 'auto' ? 'detect' : imageType);
      Object.keys(params).forEach(key => {
        formData.append(key, params[key]);
      });

      console.log("Sending request to backend");
      console.log("FormData contents:", Object.fromEntries(formData));

      const response = await fetch('http://127.0.0.1:8000/generate_adversarial', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      console.log("Received response from backend:", data);

      setAdversarialImage(`data:image/png;base64,${data.adversarial_image}`);
      setOriginalPrediction(data.original_prediction);
      setAdversarialPrediction(data.adversarial_prediction);
    } catch (e) {
      console.error("Error generating adversarial image:", e);
      setError(`Failed to generate adversarial image. Error: ${e.message}`);
    }
  };

  const methodOptions = [
    { value: 'fgsm', label: 'FGSM' },
    { value: 'pgd', label: 'PGD' },
    { value: 'deepfool', label: 'DeepFool' },
    { value: 'one_pixel', label: 'One Pixel' },
    { value: 'universal', label: 'Universal' },
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Typography variant="h2" align="center" gutterBottom style={{ color: '#4caf50' }}>
          AdversaGuard
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Original Image
                </Typography>
                {selectedImage && (
                  <img src={selectedImage} alt="Original" style={{ width: '100%', marginBottom: '1rem' }} />
                )}
                {originalPrediction && (
                  <Typography variant="subtitle1" gutterBottom>
                    Classification: {originalPrediction}
                  </Typography>
                )}
                <Button
                  variant="contained"
                  component="label"
                  fullWidth
                  color="primary"
                >
                  Upload Image
                  <input
                    type="file"
                    hidden
                    onChange={handleImageUpload}
                  />
                </Button>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Adversarial Image
                </Typography>
                {adversarialImage && (
                  <img src={adversarialImage} alt="Adversarial" style={{ width: '100%', marginBottom: '1rem' }} />
                )}
                {adversarialPrediction && (
                  <Typography variant="subtitle1" gutterBottom>
                    Classification: {adversarialPrediction}
                  </Typography>
                )}
                <Select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  fullWidth
                  style={{ marginBottom: '1rem' }}
                >
                  {methodOptions.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
                <TextField
                  label="Epsilon"
                  type="number"
                  value={epsilon}
                  onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                  fullWidth
                  margin="normal"
                />
                <FormControl fullWidth margin="normal">
                  <InputLabel id="image-type-label">Image Type</InputLabel>
                  <Select
                    labelId="image-type-label"
                    value={imageType}
                    onChange={(e) => setImageType(e.target.value)}
                    label="Image Type"
                  >
                    <MenuItem value="auto">Auto Detect</MenuItem>
                    <MenuItem value="fish_eye">Fish Eye</MenuItem>
                    <MenuItem value="mushroom">Mushroom</MenuItem>
                  </Select>
                </FormControl>
                <FormControlLabel
                  control={
                    <Switch
                      checked={stealthMode}
                      onChange={(e) => setStealthMode(e.target.checked)}
                    />
                  }
                  label="Stealth Mode"
                />
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  fullWidth
                  style={{ marginTop: '1rem', marginBottom: '1rem' }}
                >
                  {showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Options'}
                </Button>
                <Collapse in={showAdvanced}>
                  {Object.keys(params).map(key => (
                    <TextField
                      key={key}
                      label={key}
                      type="number"
                      name={key}
                      value={params[key]}
                      onChange={handleParamChange}
                      fullWidth
                      margin="normal"
                    />
                  ))}
                </Collapse>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={generateAdversarial}
                  fullWidth
                  style={{ marginTop: '1rem' }}
                >
                  Generate Adversarial
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError(null)}>
          <MuiAlert elevation={6} variant="filled" severity="error" onClose={() => setError(null)}>
            {error}
          </MuiAlert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
};

export default AdversaGuardUI;
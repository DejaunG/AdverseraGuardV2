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

const methodConfigs = {
  fgsm: {
    params: ['epsilon'],
    defaults: {
      epsilon: 0.03
    },
    labels: {
      epsilon: 'Perturbation Size (ε)'
    }
  },
  pgd: {
    params: ['epsilon', 'alpha', 'num_iter'],
    defaults: {
      epsilon: 0.03,
      alpha: 0.01,
      num_iter: 40
    },
    labels: {
      epsilon: 'Perturbation Size (ε)',
      alpha: 'Step Size (α)',
      num_iter: 'Number of Iterations'
    }
  },
  deepfool: {
    params: ['num_classes', 'overshoot', 'max_iter'],
    defaults: {
      num_classes: 1000,
      overshoot: 0.02,
      max_iter: 50
    },
    labels: {
      num_classes: 'Number of Classes',
      overshoot: 'Overshoot Parameter',
      max_iter: 'Maximum Iterations'
    }
  },
  one_pixel: {
    params: ['pixels', 'max_iter', 'pop_size'],
    defaults: {
      pixels: 1,
      max_iter: 100,
      pop_size: 400
    },
    labels: {
      pixels: 'Number of Pixels to Modify',
      max_iter: 'Maximum Iterations',
      pop_size: 'Population Size'
    }
  },
  universal: {
    params: ['epsilon', 'delta', 'max_iter_uni', 'num_classes'],
    defaults: {
      epsilon: 0.1,
      delta: 0.2,
      max_iter_uni: 50,
      num_classes: 1000
    },
    labels: {
      epsilon: 'Perturbation Size (ε)',
      delta: 'Fooling Rate Threshold (δ)',
      max_iter_uni: 'Maximum Iterations',
      num_classes: 'Number of Classes'
    }
  }
};

const AdversaGuardUI = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [adversarialImage, setAdversarialImage] = useState(null);
  const [method, setMethod] = useState('fgsm');
  const [stealthMode, setStealthMode] = useState(false);
  const [params, setParams] = useState(methodConfigs.fgsm.defaults);
  const [error, setError] = useState(null);
  const [originalPrediction, setOriginalPrediction] = useState(null);
  const [adversarialPrediction, setAdversarialPrediction] = useState(null);
  const [imageType, setImageType] = useState('auto');
  const [isLoading, setIsLoading] = useState(false);

  const handleMethodChange = (e) => {
    const newMethod = e.target.value;
    setMethod(newMethod);
    setParams(methodConfigs[newMethod].defaults);
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
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
      setError(`Failed to detect image type. Error: ${e.message}`);
    }
  };

const generateAdversarial = async () => {
  if (!selectedImage) {
    setError("Please upload an image first.");
    return;
  }

  setIsLoading(true);
  const formData = new FormData();

  try {
    const imageFile = await fetch(selectedImage).then(r => r.blob());
    formData.append('file', imageFile, 'image.jpg');
    formData.append('method', method);
    formData.append('stealth_mode', stealthMode.toString());
    formData.append('image_type', imageType === 'auto' ? 'detect' : imageType);

    // Method-specific parameter handling
    switch (method) {
      case 'fgsm':
        formData.append('epsilon', params.epsilon.toString());
        break;

      case 'pgd':
        formData.append('epsilon', params.epsilon.toString());
        formData.append('alpha', params.alpha.toString());
        formData.append('num_iter', params.num_iter.toString());
        break;

      case 'deepfool':
        formData.append('num_classes', params.num_classes.toString());
        formData.append('overshoot', params.overshoot.toString());
        formData.append('max_iter', params.max_iter.toString());
        // Add required default parameters
        formData.append('epsilon', '0.03'); // Default epsilon for stability
        break;

      case 'one_pixel':
        formData.append('pixels', params.pixels.toString());
        formData.append('max_iter', params.max_iter.toString());
        formData.append('pop_size', params.pop_size.toString());
        // Add required default parameters
        formData.append('epsilon', '0.03'); // Default epsilon for stability
        break;

      case 'universal':
        formData.append('epsilon', params.epsilon.toString());
        formData.append('delta', params.delta.toString());
        formData.append('max_iter_uni', params.max_iter_uni.toString());
        formData.append('num_classes', params.num_classes.toString());
        break;

      default:
        throw new Error(`Unknown attack method: ${method}`);
    }

    // Log request parameters for debugging
    const formDataObj = {};
    formData.forEach((value, key) => {
      formDataObj[key] = value;
    });
    console.log('Sending request with params:', formDataObj);

    const response = await fetch('http://127.0.0.1:8000/generate_adversarial', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    const data = await response.json();
    setAdversarialImage(`data:image/png;base64,${data.adversarial_image}`);
    setOriginalPrediction(data.original_prediction);
    setAdversarialPrediction(data.adversarial_prediction);
  } catch (e) {
    console.error('Error details:', e);
    setError(`Failed to generate adversarial image. Error: ${e.message}`);
  } finally {
    setIsLoading(false);
  }
};

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
                    accept="image/*"
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
                <FormControl fullWidth style={{ marginBottom: '1rem' }}>
                  <InputLabel>Attack Method</InputLabel>
                  <Select
                    value={method}
                    onChange={handleMethodChange}
                    label="Attack Method"
                  >
                    <MenuItem value="fgsm">FGSM (Fast Gradient Sign Method)</MenuItem>
                    <MenuItem value="pgd">PGD (Projected Gradient Descent)</MenuItem>
                    <MenuItem value="universal">Universal Adversarial Perturbation</MenuItem>
                    {/* Temporarily hidden attacks
                  <MenuItem value="deepfool">DeepFool</MenuItem>
                  <MenuItem value="one_pixel">One Pixel Attack</MenuItem>
                  */}
                  </Select>
                </FormControl>

                <FormControl fullWidth margin="normal">
                  <InputLabel>Image Type</InputLabel>
                  <Select
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

                <Typography variant="h6" gutterBottom style={{ marginTop: '1rem' }}>
                  Attack Parameters
                </Typography>
                {methodConfigs[method].params.map((param) => (
                  <TextField
                    key={param}
                    label={methodConfigs[method].labels[param]}
                    type="number"
                    name={param}
                    value={params[param]}
                    onChange={handleParamChange}
                    fullWidth
                    margin="normal"
                    inputProps={{
                      step: param.includes('iter') || param === 'pixels' || param === 'pop_size' || param === 'num_classes' ? 1 : 0.01,
                      min: 0,
                      max: param === 'epsilon' || param === 'delta' ? 1 : undefined
                    }}
                  />
                ))}

                <Button
                  variant="contained"
                  color="primary"
                  onClick={generateAdversarial}
                  fullWidth
                  style={{ marginTop: '1rem' }}
                  disabled={isLoading || !selectedImage}
                >
                  {isLoading ? 'Generating...' : 'Generate Adversarial'}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Snackbar
          open={!!error}
          autoHideDuration={6000}
          onClose={() => setError(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <MuiAlert elevation={6} variant="filled" severity="error" onClose={() => setError(null)}>
            {error}
          </MuiAlert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
};

export default AdversaGuardUI;
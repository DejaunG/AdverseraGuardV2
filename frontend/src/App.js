import React, { useState, useEffect } from 'react';
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
  InputLabel,
  Tooltip,
  IconButton
} from '@mui/material';
import MuiAlert from '@mui/material/Alert';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import InfoIcon from '@mui/icons-material/Info';

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

const methodDescriptions = {
  fgsm: "A quick and simple attack that slightly modifies the entire image in a single step. Good for testing basic model vulnerabilities. Fast but may be less effective.",
  pgd: "A stronger attack that gradually modifies the image over multiple steps. More likely to fool the model than FGSM but takes longer to run. Best for thorough testing.",
  universal: "Creates a pattern that can fool the model when applied to many different images. Useful for testing overall model security. Takes longer but works on multiple images.",
  deepfool: "Finds the smallest changes needed to fool the model. Good for understanding model weaknesses with minimal image changes.",
  one_pixel: "Changes only a few pixels to fool the model. Shows how sensitive the model can be to tiny changes."
};

const parameterDescriptions = {
  epsilon: "How much the image can be changed overall. Higher values (like 0.1) make more visible changes but are more likely to work. Lower values (like 0.01) make subtle changes but might be less effective.",
  alpha: "Size of each change step. Smaller values make more precise changes but take longer. Like walking with smaller steps to reach a destination more accurately.",
  num_iter: "Number of attempts to modify the image. More attempts usually give better results but take longer. Start with 40-50 and increase if needed.",
  num_classes: "How many different types of classifications to consider. Higher numbers are more thorough but slower.",
  overshoot: "How aggressive the attack should be. Higher values make stronger attacks but more visible changes.",
  max_iter: "Maximum number of tries before stopping. Increase this if the attack isn't successful enough.",
  pixels: "Number of pixels to change. More pixels = stronger attack but more visible changes.",
  pop_size: "Number of different variations to try. Larger numbers give better results but take longer.",
  delta: "How often the attack should successfully fool the model (0-1). Higher values make stronger but more visible attacks.",
  max_iter_uni: "How many times to try improving the attack. More attempts = better results but longer runtime."
};

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
  // Add animation state
  const [visible, setVisible] = useState(false);

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

  // Trigger animation on mount
  useEffect(() => {
    setVisible(true);
  }, []);

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
          formData.append('epsilon', '0.03');
          break;

        case 'one_pixel':
          formData.append('pixels', params.pixels.toString());
          formData.append('max_iter', params.max_iter.toString());
          formData.append('pop_size', params.pop_size.toString());
          formData.append('epsilon', '0.03');
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

  const InfoTooltip = ({ title }) => (
    <Tooltip title={title}>
      <IconButton size="small" sx={{ ml: 1, color: 'primary.light', padding: 0 }}>
        <InfoIcon sx={{ fontSize: 16 }} />
      </IconButton>
    </Tooltip>
  );

  const containerStyles = {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    paddingTop: '2rem',
    paddingBottom: '2rem',
    opacity: visible ? 1 : 0,
    transform: visible ? 'translateY(0)' : 'translateY(20px)',
    transition: 'opacity 0.5s ease-in-out, transform 0.7s ease-out'
  };

  const cardStyles = {
    height: '100%',
    transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
    '&:hover': {
      transform: 'scale(1.01)',
      boxShadow: '0 8px 16px rgba(0,0,0,0.2)'
    }
  };

  const imageStyles = {
    width: '100%',
    marginBottom: '1rem',
    borderRadius: '4px',
    transition: 'opacity 0.3s ease-in-out',
    opacity: 1
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div style={{ backgroundColor: theme.palette.background.default }}>
        <Container maxWidth="lg" sx={containerStyles}>
          <div>
            <Typography
              variant="h2"
              align="center"
              gutterBottom
              sx={{
                color: '#4caf50',
                marginBottom: '2rem',
                transition: 'opacity 0.5s ease-in-out',
                opacity: visible ? 1 : 0
              }}
            >
              AdversaGuard
            </Typography>
            <Grid container spacing={4}>
              <Grid item xs={12} md={6}>
                <Card sx={cardStyles}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Original Image
                    </Typography>
                    {selectedImage && (
                      <img src={selectedImage} alt="Original" style={imageStyles} />
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
                      sx={{ transition: 'transform 0.2s ease' }}
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
                <Card sx={cardStyles}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Adversarial Image
                    </Typography>
                    {adversarialImage && (
                      <img src={adversarialImage} alt="Adversarial" style={imageStyles} />
                    )}
                    {adversarialPrediction && (
                      <Typography variant="subtitle1" gutterBottom>
                        Classification: {adversarialPrediction}
                      </Typography>
                    )}
                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel>Attack Method</InputLabel>
                      <Select
                        value={method}
                        onChange={handleMethodChange}
                        label="Attack Method"
                      >
                        <MenuItem value="fgsm">
                          Fast Gradient Sign Method (FGSM)
                          <InfoTooltip title={methodDescriptions.fgsm} />
                        </MenuItem>
                        <MenuItem value="pgd">
                          Projected Gradient Descent (PGD)
                          <InfoTooltip title={methodDescriptions.pgd} />
                        </MenuItem>
                        <MenuItem value="universal">
                          Universal Adversarial Perturbation
                          <InfoTooltip title={methodDescriptions.universal} />
                        </MenuItem>
                      </Select>
                    </FormControl>

                    <FormControl fullWidth sx={{ mb: 2 }}>
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
                      label={
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                          Stealth Mode
                          <InfoTooltip title="When turned on, tries to make changes to the image that are harder for humans to notice." />
                        </div>
                      }
                      sx={{ mb: 2 }}
                    />

                    <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                      Attack Parameters
                    </Typography>
                    {methodConfigs[method].params.map((param) => (
                      <div key={param} style={{
                        display: 'flex',
                        alignItems: 'center',
                        transition: 'opacity 0.3s ease-in-out'
                      }}>
                        <TextField
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
                        <InfoTooltip title={parameterDescriptions[param]} />
                      </div>
                    ))}

                    <Button
                      variant="contained"
                      color="primary"
                      onClick={generateAdversarial}
                      fullWidth
                      sx={{
                        mt: 2,
                        transition: 'all 0.3s ease-in-out',
                        '&:hover': {
                          transform: 'scale(1.02)'
                        }
                      }}
                      disabled={isLoading || !selectedImage}
                    >
                      {isLoading ? 'Generating...' : 'Generate Adversarial'}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </div>
        </Container>
        <Snackbar
          open={!!error}
          autoHideDuration={6000}
          onClose={() => setError(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <MuiAlert
            elevation={6}
            variant="filled"
            severity="error"
            onClose={() => setError(null)}
            sx={{ opacity: error ? 1 : 0, transition: 'opacity 0.3s ease-in-out' }}
          >
            {error}
          </MuiAlert>
        </Snackbar>
      </div>
    </ThemeProvider>
  );
};

export default AdversaGuardUI;
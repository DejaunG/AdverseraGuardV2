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
  IconButton,
  Chip,
  Box,
  Stack,
} from '@mui/material';
import MuiAlert from '@mui/material/Alert';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';

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
    fgsm: "Fast Gradient Sign Method (FGSM) - A quick attack that modifies the image based on the gradient of the loss. Regular mode creates visible changes while stealth mode makes subtle perturbations. Fast to compute but may be less effective at fooling the model.",
    pgd: "Projected Gradient Descent (PGD) - An iterative attack that gradually modifies the image to fool the model. Regular mode produces visible alterations while stealth mode creates minimal visual changes. More effective than FGSM but takes longer to compute.",
    deepfool: "DeepFool - A sophisticated attack that finds the minimal perturbation needed to cross the decision boundary. Creates highly targeted modifications that may be visually noticeable. Good for understanding model decision boundaries.",
    one_pixel: "One Pixel Attack - Attempts to fool the model by modifying a small number of pixels. Shows how sensitive the model can be to tiny, localized changes. Progress bar shows attack status. More effective on simpler images.",
    universal: "Universal Adversarial Perturbation - Creates a pattern that can potentially fool the model across multiple images. Takes longer to compute but can reveal systematic model vulnerabilities. Works best with consistent image types."
};

const imageTypeDescriptions = {
    auto: "Auto-detection uses our custom trained EfficientNet-B3 model to identify if the image is a fish eye or mushroom.",
    fish_eye: "Fish eye images are classified as either fresh or non-fresh based on their visual characteristics.",
    mushroom: "Mushroom images are classified as either poisonous or non-poisonous based on their features.",
};

const parameterDescriptions = {
    epsilon: "Controls the magnitude of image modification. Higher values (like 0.1) create more noticeable changes but are more likely to succeed. Lower values (like 0.01) create subtler changes but might be less effective. In stealth mode, this value is automatically scaled for normalized image space.",
    alpha: "Step size for iterative attacks like PGD. Smaller values make more precise changes but take longer. Like walking with smaller steps to reach a destination more accurately. Automatically scaled in stealth mode.",
    num_iter: "Number of iterations for PGD attack. More iterations usually give better results but take longer. Start with 40-50 and increase if needed. Affects both regular and stealth modes equally.",
    num_classes: "Number of different classes to consider when computing the attack. Higher numbers are more thorough but slower. Used primarily in DeepFool attack.",
    overshoot: "Used in DeepFool to control how far past the decision boundary to push. Higher values make stronger attacks but more visible changes.",
    max_iter: "Maximum number of iterations before stopping an attack. Increase this if the attack isn't successful enough.",
    pixels: "Number of pixels to modify in One Pixel attack. More pixels = stronger attack but more visible changes.",
    pop_size: "Population size for evolutionary algorithms in One Pixel attack. Larger numbers give better results but take longer.",
    delta: "Target success rate for Universal attack (0-1). Higher values make stronger but more visible attacks.",
    max_iter_uni: "Maximum iterations for Universal attack. More attempts = better results but longer runtime."
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
  const [isDetecting, setIsDetecting] = useState(false);

  useEffect(() => {
    setVisible(true);
  }, []);

  const ClassificationDisplay = ({ original, adversarial }) => {
    if (!original && !adversarial) return null;

    const hasChanged = original !== adversarial;
    const getChipColor = (isOriginal) => {
      if (!hasChanged) return "success";
      return isOriginal ? "success" : "warning";
    };

    return (
      <Box
        sx={{
          mt: 2,
          mb: 2,
          backgroundColor: 'background.paper',
          p: 2,
          borderRadius: 1,
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'scale(1.01)',
            boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
          }
        }}
      >
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          alignItems="center"
          justifyContent="center"
          divider={<CompareArrowsIcon color="action" sx={{ display: { xs: 'none', sm: 'block' } }} />}
        >
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Original
            </Typography>
            <Chip
              label={original || 'No prediction'}
              color={getChipColor(true)}
              icon={<CheckCircleIcon />}
              variant="filled"
              sx={{
                minWidth: 120,
                transition: 'all 0.3s ease-in-out',
                '& .MuiChip-icon': { color: 'inherit' }
              }}
            />
          </Box>

          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Adversarial
            </Typography>
            <Chip
              label={adversarial || 'No prediction'}
              color={getChipColor(false)}
              icon={hasChanged ? <WarningIcon /> : <CheckCircleIcon />}
              variant="filled"
              sx={{
                minWidth: 120,
                transition: 'all 0.3s ease-in-out',
                '& .MuiChip-icon': { color: 'inherit' }
              }}
            />
          </Box>
        </Stack>

        {hasChanged && (
          <Typography
            variant="body2"
            color="warning.main"
            sx={{
              mt: 2,
              textAlign: 'center',
              opacity: 0.9,
              animation: 'pulse 2s infinite'
            }}
          >
            Classification Changed Successfully
          </Typography>
        )}
      </Box>
    );
  };
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
    setIsDetecting(true);
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
    } finally {
      setIsDetecting(false);
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

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div style={{ backgroundColor: theme.palette.background.default }}>
        <Container maxWidth="lg" sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          paddingTop: '2rem',
          paddingBottom: '2rem',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(20px)',
          transition: 'opacity 0.5s ease-in-out, transform 0.7s ease-out'
        }}>
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

            <ClassificationDisplay
              original={originalPrediction}
              adversarial={adversarialPrediction}
            />

            <Grid container spacing={4}>
              <Grid item xs={12} md={6}>
                <Card sx={{
                  height: '100%',
                  transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
                  '&:hover': {
                    transform: 'scale(1.01)',
                    boxShadow: '0 8px 16px rgba(0,0,0,0.2)'
                  }
                }}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Original Image
                    </Typography>
                    {selectedImage && (
                      <img
                        src={selectedImage}
                        alt="Original"
                        style={{
                          width: '100%',
                          marginBottom: '1rem',
                          borderRadius: '4px',
                          transition: 'opacity 0.3s ease-in-out',
                          opacity: 1
                        }}
                      />
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
                <Card sx={{
                  height: '100%',
                  transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
                  '&:hover': {
                    transform: 'scale(1.01)',
                    boxShadow: '0 8px 16px rgba(0,0,0,0.2)'
                  }
                }}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Adversarial Image
                    </Typography>
                    {adversarialImage && (
                      <img
                        src={adversarialImage}
                        alt="Adversarial"
                        style={{
                          width: '100%',
                          marginBottom: '1rem',
                          borderRadius: '4px',
                          transition: 'opacity 0.3s ease-in-out',
                          opacity: 1
                        }}
                      />
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
                        <MenuItem value="deepfool">
                          DeepFool
                          <InfoTooltip title={methodDescriptions.deepfool} />
                        </MenuItem>
                        <MenuItem value="one_pixel">
                          One Pixel Attack
                          <InfoTooltip title={methodDescriptions.one_pixel} />
                        </MenuItem>
                        <MenuItem value="universal">
                          Universal Adversarial Perturbation
                          <InfoTooltip title={methodDescriptions.universal} />
                        </MenuItem>
                      </Select>
                    </FormControl>

                    <FormControl fullWidth margin="normal">
                      <InputLabel>Image Type</InputLabel>
                      <Select
                        value={imageType}
                        onChange={(e) => setImageType(e.target.value)}
                        label="Image Type"
                      >
                        <MenuItem value="auto">
                          Auto Detect
                          <InfoTooltip title={imageTypeDescriptions.auto} />
                        </MenuItem>
                        <MenuItem value="fish_eye">
                          Fish Eye
                          <InfoTooltip title={imageTypeDescriptions.fish_eye} />
                        </MenuItem>
                        <MenuItem value="mushroom">
                          Mushroom
                          <InfoTooltip title={imageTypeDescriptions.mushroom} />
                        </MenuItem>
                      </Select>
                      {isDetecting && (
                        <Typography variant="caption" color="textSecondary">
                          Detecting image type...
                        </Typography>
                      )}
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
                          <InfoTooltip
                            title="When enabled, perturbations are carefully scaled in normalized image space to maintain image appearance while still attempting to fool the model. When disabled, creates more visible changes that clearly show the adversarial effect. Currently affects FGSM and PGD attacks."
                          />
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
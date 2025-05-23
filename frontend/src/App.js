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
  Collapse,
  Snackbar,
  FormControl,
  InputLabel,
  Divider,
  Box,
  Tab,
  Tabs,
  Paper
} from '@mui/material';
import MuiAlert from '@mui/material/Alert';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import ModelInfo from './components/ModelInfo';
import ModelSelector from './components/ModelSelector';
import ModelComparer from './components/ModelComparer';
import { getAvailableModels, selectModel } from './api';

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

const Alert = React.forwardRef(function Alert(props, ref) {
  return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
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
  const [currentModelId, setCurrentModelId] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [tabValue, setTabValue] = useState(0);

  // Load available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        console.log("App: Fetching available models...");
        const models = await getAvailableModels();
        console.log("App: Models received:", models);
        
        if (!models || models.length === 0) {
          console.warn("App: No models received from API");
          setError("No models available. Using fallback models.");
          
          // Create fallback models
          const fallbackModels = [
            {
              id: "imagenet",
              name: "ImageNet Default Model",
              type: "Pre-trained ImageNet Model"
            },
            {
              id: "federated_adversera_model.pth",
              name: "Federated Adversera Model",
              type: "Federated Learning Model"
            },
            {
              id: "optimized_adversera_model.pth",
              name: "Optimized Adversera Model",
              type: "Optimized Centralized Model"
            }
          ];
          
          setAvailableModels(fallbackModels);
          setCurrentModelId(fallbackModels[0].id);
          return;
        }
        
        setAvailableModels(models);
        
        // Check if there's a model in localStorage
        const savedModelId = window.localStorage.getItem('currentModelId');
        if (savedModelId && models.some(m => m.id === savedModelId)) {
          console.log(`App: Using saved model from localStorage: ${savedModelId}`);
          setCurrentModelId(savedModelId);
          
          try {
            await selectModel(savedModelId);
          } catch (selectError) {
            console.warn(`App: Error selecting saved model: ${selectError.message}`);
            // Continue anyway
          }
        } else {
          // Set default model if available
          if (models.length > 0) {
            // Try to find a federated model first
            const federatedModel = models.find(m => m.type && m.type.toLowerCase().includes('federated'));
            if (federatedModel) {
              console.log(`App: Using federated model: ${federatedModel.id}`);
              setCurrentModelId(federatedModel.id);
              
              try {
                await selectModel(federatedModel.id);
              } catch (selectError) {
                console.warn(`App: Error selecting federated model: ${selectError.message}`);
                // Continue anyway
              }
            } else {
              console.log(`App: Using default model: ${models[0].id}`);
              setCurrentModelId(models[0].id);
              
              try {
                await selectModel(models[0].id);
              } catch (selectError) {
                console.warn(`App: Error selecting default model: ${selectError.message}`);
                // Continue anyway
              }
            }
          }
        }
      } catch (error) {
        console.error('App: Error fetching models:', error);
        setError(`Failed to load models. Using defaults.`);
        
        // Use fallback models
        const fallbackModels = [
          {
            id: "imagenet",
            name: "ImageNet Default Model",
            type: "Pre-trained ImageNet Model"
          },
          {
            id: "federated_adversera_model.pth",
            name: "Federated Adversera Model",
            type: "Federated Learning Model"
          },
          {
            id: "optimized_adversera_model.pth",
            name: "Optimized Adversera Model",
            type: "Optimized Centralized Model"
          }
        ];
        
        setAvailableModels(fallbackModels);
        setCurrentModelId(fallbackModels[0].id);
      }
    };
    
    fetchModels();
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check if file is an image
      if (!file.type.startsWith('image/')) {
        setError("Please upload an image file (JPEG, PNG, etc.)");
        return;
      }
      
      // Create a local URL for the image
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      
      // Reset any previous results
      setAdversarialImage(null);
      setOriginalPrediction(null);
      setAdversarialPrediction(null);
      
      console.log("Image uploaded:", file.name, "Type:", file.type);
      
      // Store the file in a data attribute for later use
      window.uploadedImageFile = file;
      
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

  const handleModelChange = async (modelId) => {
    try {
      console.log(`Switching to model: ${modelId}`);
      setCurrentModelId(modelId);
      
      // Clear any previous results when switching models
      setAdversarialImage(null);
      setOriginalPrediction(null);
      setAdversarialPrediction(null);
      
      // Call the API to select the model
      const result = await selectModel(modelId);
      console.log("Model selection result:", result);
      
      // Show a brief success message
      const modelName = availableModels.find(m => m.id === modelId)?.name || modelId;
      setError(`Model switched to: ${modelName}`);
      
      // Clear the message after 3 seconds
      setTimeout(() => {
        if (setError) setError(null);
      }, 3000);
    } catch (error) {
      console.error('Error switching model:', error);
      setError(`Failed to switch model: ${error.message}`);
    }
  };

  const generateAdversarial = async () => {
    console.log("Generate button clicked");
    if (!selectedImage) {
      setError("Please upload an image first.");
      return;
    }

    // Get the stored file
    const uploadedFile = window.uploadedImageFile;
    if (!uploadedFile) {
      setError("Image file not found. Please try uploading again.");
      return;
    }

    const formData = new FormData();
    try {
      // Use the actual file object instead of fetching the URL
      formData.append('file', uploadedFile);
      formData.append('method', method);
      formData.append('epsilon', epsilon);
      formData.append('stealth_mode', stealthMode.toString());  // Ensure boolean is converted to string
      formData.append('image_type', imageType === 'auto' ? 'detect' : imageType);
      
      // Add advanced parameters
      Object.keys(params).forEach(key => {
        formData.append(key, params[key].toString());  // Convert numbers to strings
      });

      console.log("Sending request to backend");
      console.log("FormData method:", method);
      console.log("FormData epsilon:", epsilon);
      console.log("FormData stealth_mode:", stealthMode.toString());
      console.log("FormData image_type:", imageType === 'auto' ? 'detect' : imageType);

      // Show loading state
      setAdversarialImage(null);
      setOriginalPrediction("Processing...");
      setAdversarialPrediction("Processing...");

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

      // Set results
      setAdversarialImage(`data:image/png;base64,${data.adversarial_image}`);
      setOriginalPrediction(data.original_prediction);
      setAdversarialPrediction(data.adversarial_prediction);
    } catch (e) {
      console.error("Error generating adversarial image:", e);
      setError(`Failed to generate adversarial image. Error: ${e.message}`);
      // Reset loading state
      setOriginalPrediction(null);
      setAdversarialPrediction(null);
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

        <Paper sx={{ mb: 3 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            variant="fullWidth"
            indicatorColor="primary"
            textColor="primary"
          >
            <Tab label="Attack Generator" />
            <Tab label="Model Management" />
            <Tab label="Model Comparison" />
          </Tabs>
        </Paper>

        {tabValue === 0 && (
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
                    <>
                      <img src={adversarialImage} alt="Adversarial" style={{ width: '100%', marginBottom: '1rem' }} />
                      <Button 
                        variant="outlined" 
                        color="secondary" 
                        fullWidth 
                        style={{ marginBottom: '1rem' }}
                        onClick={() => {
                          try {
                            console.log("Download button clicked");
                            
                            // Try to convert base64 to blob for better download handling
                            const imageData = adversarialImage.split(',')[1]; // Remove the data:image/png;base64, part
                            const byteCharacters = atob(imageData);
                            const byteArrays = [];
                            for (let i = 0; i < byteCharacters.length; i += 512) {
                              const slice = byteCharacters.slice(i, i + 512);
                              const byteNumbers = new Array(slice.length);
                              for (let j = 0; j < slice.length; j++) {
                                byteNumbers[j] = slice.charCodeAt(j);
                              }
                              const byteArray = new Uint8Array(byteNumbers);
                              byteArrays.push(byteArray);
                            }
                            
                            const blob = new Blob(byteArrays, {type: 'image/png'});
                            const blobUrl = URL.createObjectURL(blob);
                            
                            // Create a temporary anchor element to trigger download
                            const downloadLink = document.createElement('a');
                            downloadLink.href = blobUrl;
                            downloadLink.download = `adversarial_image_${method}_${new Date().getTime()}.png`;
                            document.body.appendChild(downloadLink);
                            downloadLink.click();
                            document.body.removeChild(downloadLink);
                            
                            // Clean up the blob URL
                            setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
                            
                            console.log("Download initiated");
                          } catch (e) {
                            console.error("Error downloading image:", e);
                            // Fallback to original method
                            const downloadLink = document.createElement('a');
                            downloadLink.href = adversarialImage;
                            downloadLink.download = `adversarial_image_${method}.png`;
                            document.body.appendChild(downloadLink);
                            downloadLink.click();
                            document.body.removeChild(downloadLink);
                          }
                        }}
                      >
                        Download Adversarial Image
                      </Button>
                    </>
                  )}
                  {adversarialPrediction && (
                    <Typography variant="subtitle1" gutterBottom>
                      Classification: {adversarialPrediction}
                    </Typography>
                  )}

                  <FormControl fullWidth style={{ marginBottom: '1rem' }}>
                    <InputLabel id="model-label">Selected Model</InputLabel>
                    <Select
                      labelId="model-label"
                      value={currentModelId}
                      onChange={(e) => handleModelChange(e.target.value)}
                      label="Selected Model"
                    >
                      {availableModels.map(model => (
                        <MenuItem key={model.id} value={model.id}>
                          {model.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

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
                      <MenuItem value="text">Text Image</MenuItem>
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
        )}

        {tabValue === 1 && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <ModelSelector onModelChange={handleModelChange} />
            </Grid>
            <Grid item xs={12} md={6}>
              <ModelInfo />
            </Grid>
          </Grid>
        )}

        {tabValue === 2 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <ModelComparer />
            </Grid>
          </Grid>
        )}

        <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError(null)}>
          <Alert elevation={6} variant="filled" severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
};

export default AdversaGuardUI;
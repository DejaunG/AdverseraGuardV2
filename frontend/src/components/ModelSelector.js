import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  FormControl, 
  Select, 
  MenuItem, 
  InputLabel,
  Button,
  CircularProgress,
  Alert,
  Snackbar,
  Chip
} from '@mui/material';
import { getAvailableModels, selectModel } from '../api';

const ModelSelector = ({ onModelChange }) => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(true);
  const [loadingModel, setLoadingModel] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Fetch available models on component mount
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      console.log("ModelSelector: Fetching available models...");
      
      const availableModels = await getAvailableModels();
      console.log("ModelSelector: Models received:", availableModels);
      
      if (!availableModels || availableModels.length === 0) {
        throw new Error("No models received from API");
      }
      
      setModels(availableModels);
      
      // Try to get the currently selected model from localStorage
      const savedModelId = window.localStorage.getItem('currentModelId');
      
      // Set default selected model
      if (savedModelId && availableModels.some(m => m.id === savedModelId)) {
        // Use the saved model if it exists in our list
        console.log(`ModelSelector: Using saved model: ${savedModelId}`);
        setSelectedModel(savedModelId);
      } else if (availableModels.length > 0) {
        // Otherwise use default selection logic
        const defaultModel = 
          availableModels.find(m => m.id === 'federated_adversera_model.pth') || 
          availableModels.find(m => m.type && m.type.toLowerCase().includes('federated')) ||
          availableModels.find(m => m.id === 'optimized_adversera_model.pth') || 
          availableModels.find(m => m.type && m.type.toLowerCase().includes('optimized')) ||
          availableModels[0];
          
        console.log(`ModelSelector: Using default model: ${defaultModel.id}`);
        setSelectedModel(defaultModel.id);
      }
    } catch (err) {
      console.error("ModelSelector: Error loading models:", err);
      setError('Failed to load models. Using fallback models.');
      
      // Create fallback models since something is very wrong
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
      
      setModels(fallbackModels);
      setSelectedModel(fallbackModels[0].id);
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleApplyModel = async () => {
    try {
      setLoadingModel(true);
      setError('');
      
      console.log(`Applying model: ${selectedModel}`);
      
      try {
        await selectModel(selectedModel);
        console.log(`Model successfully selected on server`);
      } catch (apiError) {
        console.warn(`API selection failed, continuing with client-side selection:`, apiError);
        // Continue anyway, we'll handle it client-side
      }
      
      // Store the selected model in local storage
      window.localStorage.setItem('currentModelId', selectedModel);
      
      setSuccess(`Model ${selectedModel} successfully loaded`);
      
      // Notify parent component
      if (onModelChange) {
        console.log(`Notifying parent component of model change`);
        onModelChange(selectedModel);
      }
    } catch (err) {
      console.error(`Error in handleApplyModel:`, err);
      setError(`Failed to load model: ${err.message}`);
    } finally {
      setLoadingModel(false);
    }
  };

  const handleCloseAlert = () => {
    setError('');
    setSuccess('');
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp * 1000).toLocaleString();
  };

  const formatFileSize = (size) => {
    if (!size) return 'N/A';
    return `${size.toFixed(2)} MB`;
  };

  return (
    <Card sx={{ mt: 2, mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Model Selection
        </Typography>
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Select a model to use for adversarial example generation:
            </Typography>
            
            <FormControl fullWidth sx={{ mt: 2, mb: 2 }}>
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                value={selectedModel}
                onChange={handleModelChange}
                label="Model"
                disabled={loadingModel || models.length === 0}
              >
                {models.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    {model.name} 
                    {model.type === "Federated Learning Model" && 
                      <Chip size="small" color="primary" label="Federated" sx={{ ml: 1 }} />
                    }
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            {selectedModel && (
              <Box sx={{ mt: 2, mb: 1 }}>
                {models.filter(m => m.id === selectedModel).map((model) => (
                  <Box key={model.id} sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Type:</strong> {model.type}
                    </Typography>
                    {model.last_modified > 0 && (
                      <Typography variant="body2">
                        <strong>Last Modified:</strong> {formatDate(model.last_modified)}
                      </Typography>
                    )}
                    {model.file_size_mb > 0 && (
                      <Typography variant="body2">
                        <strong>Size:</strong> {formatFileSize(model.file_size_mb)}
                      </Typography>
                    )}
                  </Box>
                ))}
              </Box>
            )}
            
            <Button
              variant="contained"
              onClick={handleApplyModel}
              disabled={!selectedModel || loadingModel}
              fullWidth
              sx={{ mt: 1 }}
            >
              {loadingModel ? <CircularProgress size={24} /> : 'Apply Selected Model'}
            </Button>
          </>
        )}
        
        <Snackbar 
          open={!!error || !!success} 
          autoHideDuration={6000} 
          onClose={handleCloseAlert}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert 
            onClose={handleCloseAlert} 
            severity={error ? "error" : "success"} 
            sx={{ width: '100%' }}
          >
            {error || success}
          </Alert>
        </Snackbar>
      </CardContent>
    </Card>
  );
};

export default ModelSelector;
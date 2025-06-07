import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Divider, 
  LinearProgress, 
  Box, 
  Chip,
  Stack 
} from '@mui/material';
import { getModelInfo, getFederatedTrainingStatus } from '../api';

// Format timestamp to readable date
const formatDate = (timestamp) => {
  if (!timestamp) return 'Unknown';
  return new Date(timestamp * 1000).toLocaleString();
};

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [federatedStatus, setFederatedStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [modelData, federatedData] = await Promise.all([
          getModelInfo(),
          getFederatedTrainingStatus()
        ]);
        setModelInfo(modelData);
        setFederatedStatus(federatedData);
      } catch (err) {
        setError('Failed to load model information');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Refresh every minute
    const intervalId = setInterval(fetchData, 60000);
    return () => clearInterval(intervalId);
  }, []);

  if (loading) {
    return (
      <Card sx={{ mt: 2, mb: 2 }}>
        <CardContent>
          <Typography variant="h6">Model Information</Typography>
          <Box sx={{ width: '100%', mt: 2 }}>
            <LinearProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ mt: 2, mb: 2 }}>
        <CardContent>
          <Typography variant="h6" color="error">
            {error}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ mt: 2, mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Model Information
        </Typography>
        
        <Divider sx={{ mt: 1, mb: 2 }} />
        
        <Typography variant="subtitle1" fontWeight="bold">
          Currently Active: {modelInfo?.model_type || 'Unknown'}
        </Typography>
        
        <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: 'wrap', gap: 1 }}>
          <Chip 
            label={modelInfo?.using_cuda ? 'Using GPU' : 'Using CPU'} 
            color={modelInfo?.using_cuda ? 'success' : 'default'}
          />
          <Chip 
            label={`Device: ${modelInfo?.device || 'Unknown'}`} 
            variant="outlined" 
          />
        </Stack>
        
        {modelInfo?.model_path !== 'N/A' && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>File Path:</strong> {modelInfo?.model_path}
            </Typography>
            <Typography variant="body2">
              <strong>Last Modified:</strong> {formatDate(modelInfo?.last_modified)}
            </Typography>
            <Typography variant="body2">
              <strong>Size:</strong> {modelInfo?.file_size_mb?.toFixed(2)} MB
            </Typography>
          </Box>
        )}
        
        <Divider sx={{ mt: 2, mb: 2 }} />
        
        <Typography variant="subtitle1" fontWeight="bold">
          Federated Learning Status
        </Typography>
        
        <Box sx={{ mt: 1 }}>
          <Chip 
            label={federatedStatus?.trained ? 'Model Trained' : 'Not Trained'} 
            color={federatedStatus?.trained ? 'success' : 'warning'}
            sx={{ mr: 1, mb: 1 }}
          />
          
          {federatedStatus?.trained ? (
            <>
              <Typography variant="body2">
                <strong>Created:</strong> {formatDate(federatedStatus?.created_at)}
              </Typography>
              <Typography variant="body2">
                <strong>Last Updated:</strong> {formatDate(federatedStatus?.last_modified)}
              </Typography>
              <Typography variant="body2">
                <strong>Size:</strong> {federatedStatus?.file_size_mb?.toFixed(2)} MB
              </Typography>
            </>
          ) : (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {federatedStatus?.message || 'Federated model has not been trained yet.'}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ModelInfo;
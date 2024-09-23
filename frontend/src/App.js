import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  AppBar,
  Toolbar,
  CircularProgress,
} from '@mui/material';
import UploadComponent from './components/UploadComponent';
import AttackMethodSelector from './components/AttackMethodSelector';
import DisplayImages from './components/DisplayImages';
import Results from './components/Results';
import { processImage } from './utils/api';

function App() {
  const [attackMethod, setAttackMethod] = useState('fgsm');
  const [originalImage, setOriginalImage] = useState(null);
  const [advImage, setAdvImage] = useState(null);
  const [originalResult, setOriginalResult] = useState('');
  const [advResult, setAdvResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (imageFile) => {
    setOriginalImage(imageFile);
    setAdvImage(null);
    setOriginalResult('');
    setAdvResult('');
  };

  const handleAttackChange = (method) => {
    setAttackMethod(method);
  };

  const handleSubmit = async () => {
    if (!originalImage) {
      alert('Please upload an image.');
      return;
    }

    setLoading(true);
    const response = await processImage(originalImage, attackMethod);
    if (response) {
      setAdvImage(`data:image/jpeg;base64,${response.adv_image}`);
      setOriginalResult(response.original_result);
      setAdvResult(response.adv_result);
    }
    setLoading(false);
  };

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">AdverseraGuard</Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="md">
        <Box my={4}>
          <Typography variant="h4" gutterBottom align="center">
            Adversarial Image Generator
          </Typography>
          <UploadComponent onImageUpload={handleImageUpload} />
          {originalImage && (
            <Box my={2}>
              <Typography variant="h6" align="center">
                Uploaded Image Preview
              </Typography>
              <Box display="flex" justifyContent="center" mt={2}>
                <img
                  src={URL.createObjectURL(originalImage)}
                  alt="Uploaded"
                  style={{ maxWidth: '100%', maxHeight: 400 }}
                />
              </Box>
            </Box>
          )}
          <AttackMethodSelector
            attackMethod={attackMethod}
            onAttackChange={handleAttackChange}
          />
          <Box textAlign="center" my={2}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <>
                  <CircularProgress size={24} color="inherit" />
                  &nbsp;Processing...
                </>
              ) : (
                'Generate Adversarial Image'
              )}
            </Button>
          </Box>
          {advImage && (
            <DisplayImages originalImage={originalImage} advImage={advImage} />
          )}
          {originalResult && advResult && (
            <Results originalResult={originalResult} advResult={advResult} />
          )}
        </Box>
      </Container>
    </>
  );
}

export default App;
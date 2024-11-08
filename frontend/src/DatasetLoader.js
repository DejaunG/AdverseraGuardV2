import React, { useState } from 'react';
import {
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  Grid,
  Box,
  Typography,
  IconButton,
  Tab,
  Tabs,
  Card,
  CardMedia,
  CardActionArea,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Close, FolderOpen } from '@mui/icons-material';

const API_BASE_URL = 'http://127.0.0.1:8000';

const DatasetLoader = ({ onImageSelect }) => {
  const [open, setOpen] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);
  const [images, setImages] = useState({
    fresh: [],
    'non-fresh': [],
    edible: [],
    poisonous: []
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleOpen = async () => {
    setOpen(true);
    setLoading(true);
    setError(null);

    try {
      const categories = ['fresh', 'non-fresh', 'edible', 'poisonous'];
      const imageData = {};

      for (const category of categories) {
        console.log(`Fetching images for category: ${category}`);
        const response = await fetch(`${API_BASE_URL}/gallery/${category}`);

        if (!response.ok) {
          throw new Error(`Failed to load ${category} images: ${response.statusText}`);
        }

        const data = await response.json();
        console.log(`Received ${data.images.length} images for ${category}`);
        imageData[category] = data.images;
      }

      setImages(imageData);
    } catch (error) {
      console.error('Error loading dataset images:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setOpen(false);
    setError(null);
  };

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const handleImageSelect = async (category, filename) => {
    try {
      setLoading(true);
      const imagePath = `${API_BASE_URL}/gallery-image/${category}/${encodeURIComponent(filename)}`;

      const response = await fetch(imagePath);
      if (!response.ok) {
        throw new Error(`Failed to load image: ${response.statusText}`);
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      onImageSelect(imageUrl);
      handleClose();
    } catch (error) {
      console.error('Error loading image:', error);
      setError('Failed to load image');
    } finally {
      setLoading(false);
    }
  };

  const categories = ['fresh', 'non-fresh', 'edible', 'poisonous'];
  const categoryLabels = {
    'fresh': 'Fresh Fish',
    'non-fresh': 'Non-Fresh Fish',
    'edible': 'Edible Mushroom',
    'poisonous': 'Poisonous Mushroom'
  };

  return (
    <>
      <Button
        variant="contained"
        startIcon={<FolderOpen />}
        onClick={handleOpen}
        fullWidth
        sx={{ mb: 2 }}
      >
        Load from Dataset
      </Button>

      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Select Image from Dataset</Typography>
            <IconButton onClick={handleClose} size="small">
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>

        <DialogContent>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {loading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              <Tabs
                value={currentTab}
                onChange={handleTabChange}
                variant="scrollable"
                scrollButtons="auto"
                sx={{ mb: 2 }}
              >
                {categories.map((category, index) => (
                  <Tab
                    key={category}
                    label={categoryLabels[category]}
                    id={`gallery-tab-${index}`}
                  />
                ))}
              </Tabs>

              <Grid container spacing={2}>
                {images[categories[currentTab]]?.map((filename, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card>
                      <CardActionArea
                        onClick={() => handleImageSelect(categories[currentTab], filename)}
                      >
                        <CardMedia
                          component="img"
                          height="140"
                          image={`${API_BASE_URL}/gallery-image/${categories[currentTab]}/${encodeURIComponent(filename)}`}
                          alt={`Dataset image ${index + 1}`}
                          sx={{ objectFit: 'cover' }}
                        />
                      </CardActionArea>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};

export default DatasetLoader;
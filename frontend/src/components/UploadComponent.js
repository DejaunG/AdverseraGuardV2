import React from 'react';
import { Box, Button } from '@mui/material';
import { CloudUpload } from '@mui/icons-material';

function UploadComponent({ onImageUpload }) {
  const handleUploadClick = (event) => {
    onImageUpload(event.target.files[0]);
  };

  return (
    <Box textAlign="center" my={2}>
      <input
        accept="image/*"
        style={{ display: 'none' }}
        id="upload-image"
        type="file"
        onChange={handleUploadClick}
      />
      <label htmlFor="upload-image">
        <Button variant="contained" component="span" startIcon={<CloudUpload />}>
          Upload Image
        </Button>
      </label>
    </Box>
  );
}

export default UploadComponent;
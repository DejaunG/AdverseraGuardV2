import React from 'react';
import { Grid, Typography, Box, Paper } from '@mui/material';

function DisplayImages({ originalImage, advImage }) {
  const imageStyle = {
    width: '100%',
    height: '300px',
    objectFit: 'cover',
    borderRadius: '8px',
  };

  return (
    <Box my={4}>
      <Grid container spacing={4}>
        <Grid item xs={12} sm={6}>
          <Paper elevation={3} sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" align="center" gutterBottom>
              Original Image
            </Typography>
            <Box flexGrow={1} display="flex" alignItems="center" justifyContent="center">
              <img
                src={originalImage}
                alt="Original"
                style={imageStyle}
              />
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Paper elevation={3} sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" align="center" gutterBottom>
              Adversarial Image
            </Typography>
            <Box flexGrow={1} display="flex" alignItems="center" justifyContent="center">
              <img
                src={advImage}
                alt="Adversarial"
                style={imageStyle}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default DisplayImages;
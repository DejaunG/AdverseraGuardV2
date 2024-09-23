import React from 'react';
import { Box, Typography, Grid } from '@mui/material';

function Results({ originalResult, advResult }) {
  return (
    <Box textAlign="center" my={4}>
      <Typography variant="h5" gutterBottom>Classification Results:</Typography>
      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12} sm={6}>
          <Typography variant="h6">Original Image:</Typography>
          <Typography variant="body1">{originalResult}</Typography>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Typography variant="h6">Adversarial Image:</Typography>
          <Typography variant="body1">{advResult}</Typography>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Results;
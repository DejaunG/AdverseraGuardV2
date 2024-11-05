import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Science } from '@mui/icons-material';

function Results({ result }) {
  return (
    <Box textAlign="center" my={4}>
      <Paper elevation={3} sx={{ p: 3, backgroundColor: 'background.paper', maxWidth: '500px', margin: 'auto' }}>
        <Typography variant="h5" gutterBottom color="primary">
          <Science sx={{ verticalAlign: 'middle', mr: 1 }} />
          Classification Result
        </Typography>
        <Typography variant="h6" color="text.primary">
          {result}
        </Typography>
      </Paper>
    </Box>
  );
}

export default Results;
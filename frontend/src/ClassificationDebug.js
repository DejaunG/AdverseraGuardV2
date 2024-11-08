import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Button,
  Collapse,
  Fade
} from '@mui/material';
import {
  BugReport,
  KeyboardArrowDown,
  KeyboardArrowUp
} from '@mui/icons-material';

const ClassificationDebug = ({
  originalPrediction,
  adversarialPrediction,
  method,
  stealthMode,
  params,
  isLoading
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        bgcolor: 'background.default',
        borderTop: '1px solid rgba(255,255,255,0.1)',
        p: 2
      }}
    >
      <Box sx={{ maxWidth: 'lg', mx: 'auto' }}>
        <Button
          startIcon={<BugReport />}
          endIcon={expanded ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
          onClick={() => setExpanded(!expanded)}
          variant="outlined"
          color="primary"
          size="small"
          sx={{ mb: expanded ? 1 : 0 }}
        >
          Debug Info
        </Button>

        <Collapse in={expanded}>
          <Fade in={expanded}>
            <Paper
              sx={{
                p: 2,
                backgroundColor: 'rgba(0,0,0,0.1)',
                border: '1px solid rgba(255,255,255,0.1)',
                transition: 'all 0.3s ease-in-out',
                mt: 1
              }}
            >
              <Typography variant="h6" gutterBottom color="primary">
                Debug Information
              </Typography>

              <Box sx={{ my: 1 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Original Prediction:
                </Typography>
                <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                  {originalPrediction || 'No prediction'}
                </Typography>
              </Box>

              <Divider sx={{ my: 1 }} />

              <Box sx={{ my: 1 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Adversarial Prediction:
                </Typography>
                <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                  {adversarialPrediction || 'No prediction'}
                </Typography>
              </Box>

              <Divider sx={{ my: 1 }} />

              <Box sx={{ my: 1 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Attack Configuration:
                </Typography>
                <Typography variant="body2">
                  Method: {method}
                  <br />
                  Stealth Mode: {stealthMode ? 'On' : 'Off'}
                  <br />
                  Loading State: {isLoading ? 'Loading' : 'Idle'}
                  <br />
                  Parameters: {Object.entries(params).map(([key, value]) =>
                    `${key}: ${value}`
                  ).join(', ')}
                </Typography>
              </Box>
            </Paper>
          </Fade>
        </Collapse>
      </Box>
    </Box>
  );
};

export default ClassificationDebug;
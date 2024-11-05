import React from 'react';
import { FormControl, InputLabel, Select, MenuItem, Box } from '@mui/material';

function AttackMethodSelector({ attackMethod, onAttackChange }) {
  const handleChange = (event) => {
    onAttackChange(event.target.value);
  };

  return (
    <Box textAlign="center" my={2}>
      <FormControl variant="outlined" style={{ minWidth: 200 }}>
        <InputLabel id="attack-method-label">Attack Method</InputLabel>
        <Select
          labelId="attack-method-label"
          value={attackMethod}
          onChange={handleChange}
          label="Attack Method"
        >
          <MenuItem value="fgsm">FGSM</MenuItem>
          <MenuItem value="pgd">PGD</MenuItem>
          <MenuItem value="deepfool">DeepFool</MenuItem>
          <MenuItem value="one_pixel">One Pixel</MenuItem>
          <MenuItem value="universal">Universal</MenuItem>
        </Select>
      </FormControl>
    </Box>
  );
}

export default AttackMethodSelector;
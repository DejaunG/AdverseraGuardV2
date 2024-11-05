import axios from 'axios';

export const processImage = async (imageFile, attackMethod, stealthMode) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('attack_method', attackMethod);
  formData.append('stealth_mode', stealthMode);

  try {
    const response = await axios.post('http://localhost:8000/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    console.error('Error processing image:', error);
    throw new Error('An error occurred while processing the image.');
  }
};
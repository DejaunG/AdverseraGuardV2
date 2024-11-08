const API_BASE_URL = 'http://localhost:8000';

export const detectImageType = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/detect_image_type`, {
      method: 'POST',
      body: formData,
      // Add explicit CORS headers
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server responded with ${response.status}: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    if (error.message.includes('Failed to fetch')) {
      throw new Error(
        'Unable to connect to the server. Please ensure the backend is running on http://localhost:8000'
      );
    }
    throw error;
  }
};

export const generateAdversarial = async (formData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate_adversarial`, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server responded with ${response.status}: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    if (error.message.includes('Failed to fetch')) {
      throw new Error(
        'Unable to connect to the server. Please ensure the backend is running on http://localhost:8000'
      );
    }
    throw error;
  }
};
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2ecc71', // Bright green
      light: '#58d68d',
      dark: '#27ae60',
    },
    secondary: {
      main: '#1abc9c', // Teal
      light: '#48c9b0',
      dark: '#16a085',
    },
    background: {
      default: '#0a2f1f', // Dark green background
      paper: '#103626',
    },
    text: {
      primary: '#ecf0f1', // Light text for dark background
      secondary: '#bdc3c7',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#145a32', // Darker green for AppBar
        },
      },
    },
  },
});

export default theme;
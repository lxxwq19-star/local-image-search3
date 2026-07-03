import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Tauri expects a fixed port (for development)
  server: {
    port: 1420,
    strictPort: true,
  },
  // Enables better support for mobile devices
  optimizeDeps: {
    include: ['@tauri-apps/api'],
  },
});

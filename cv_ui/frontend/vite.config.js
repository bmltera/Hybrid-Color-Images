import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    fs: {
      allow: [
        // Allow serving files from the project root and parent
        path.resolve(__dirname, '..'),
      ],
    },
  },
  assetsInclude: ['**/*.png'],
})

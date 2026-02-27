import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/legen/web/',
  plugins: [react()],
  server: {
    host: true,
    port: 4173,
  },
})

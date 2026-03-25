import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
	plugins: [react()],
	server: {
		port: 5173,
		watch: {
			ignored: [
				'**/.lake/**',
				'**/target/**',
				'**/*.llbc',
				'**/*.ullbc',
				'**/demos/**',
			],
		},
	},
	publicDir: 'static',
	resolve: {
		alias: {
			$lib: resolve(__dirname, './src/lib'),
		},
	},
});

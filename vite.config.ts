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
				'**/lean_python_demo/**',
				'**/lean_ts_demo/**',
				'**/rust_aeneas_demo/**',
				'**/vec2_standalone_probe/**',
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

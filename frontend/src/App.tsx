import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import V2Page from './pages/V2Page';

function App() {
	return (
		<BrowserRouter>
			<Routes>
				<Route path="/v2" element={<V2Page />} />
				<Route path="/" element={<Navigate to="/v2" replace />} />
			</Routes>
		</BrowserRouter>
	);
}

export default App;

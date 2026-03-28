import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import V2Page from './pages/V2Page';
import V3Page from './pages/V3Page';

function App() {
	return (
		<BrowserRouter>
			<Routes>
				<Route path="/v3" element={<V3Page />} />
				<Route path="/v2" element={<V2Page />} />
				<Route path="/" element={<Navigate to="/v3" replace />} />
			</Routes>
		</BrowserRouter>
	);
}

export default App;

import './App.css';
import PredictionForm from './components/PredictionForm';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Rating Prediction</h1>
      </header>
      <main>
        <PredictionForm />
      </main>
    </div>
  );
}

export default App;
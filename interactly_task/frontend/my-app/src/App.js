import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [query, setQuery] = useState('');
  const [matches, setMatches] = useState([]);

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/match_candidates', { job_description: query });
      setMatches(response.data);
    } catch (error) {
      console.error(error);
      // Handle error
    }
  };

  return (
    <div className="App">
      <h1>Candidate Matching System</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="query">Enter job description:</label>
        <textarea id="query" value={query} onChange={(e) => setQuery(e.target.value)} />
        <button type="submit">Submit</button>
      </form>
      {matches.length > 0 && (
        <div>
          <h2>Top Matches</h2>
          <ul>
            {matches.map((match, index) => (
              <li key={index}>
                <strong>Candidate {index + 1}:</strong>
                {/* <p>Name: {match.Name}</p>
                <p>Skills: {match.Job Skills}</p>
                <p>Experience: {match.Experience}</p>
                <p>Similarity: {match.similarity}</p> */}
                <p>{match}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;

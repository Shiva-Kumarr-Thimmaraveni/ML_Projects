import React, { useState } from 'react'
import './App.css'
import progress from '../src/assets/progress2.gif';  // Make sure you have a loading.gif in your src folder


function App() {
  const [prompt, setPrompt] = useState('')
  const [dataFrame, setDataFrame] = useState([])
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    const res = await fetch('http://localhost:5000/getData', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    })

    if (res.ok) {
      const data = await res.json()
      setDataFrame(data.dataframe)
    } else {
      console.error('Error fetching data from backend')
    }
    setLoading(false)
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Recruiter Pilot</h1>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="prompt"></label>
            <textarea
              type="text"
              className="inputBox"
              id="prompt"
              placeholder="Data Scientist with 2+ Years of Experience."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              required
            ></textarea>
          </div>
          <button type="submit">Submit</button>
        </form>
        {loading && (
          <img src={progress} alt="Loading..." className="loading-gif" />
        )}

        {!loading && dataFrame.length > 0 && (
          <table>
            <thead>
              <tr>
                {Object.keys(dataFrame[0]).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dataFrame.map((row, index) => (
                <tr key={index}>
                  {Object.values(row).map((value, i) => (
                    <td key={i}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </header>
    </div>
  )
}

export default App

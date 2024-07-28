import react, {useEffect, useState} from 'react'
function App(){
  const [data, setData] = useState({})
  useEffect(()=> {
    fetchData()
  },[])

  const fetchData = async ()=>{
    try{
      const res = await fetch('https://laughing-carnival-r4469x69566xcxvpj-5000.app.github.dev/')
      const jsonData = await  res.json()
      setData(jsonData)
    }catch(error){
      console.log('error',error)
    }
  }
  return(
    <div className='App'>
      <h1>front end</h1>
      <h2>{data.message}</h2>
    </div>
  )
}

export default App
import styles from './App.module.css';
import { MdOutlineSearch, MdWaterDrop } from 'react-icons/md'
import { useState } from "react"
import Map, {Marker} from 'react-map-gl';

function App() {
  const MAPBOX_TOKEN = 'pk.eyJ1IjoiZ2Vvcmdpb3MtdWJlciIsImEiOiJjanZidTZzczAwajMxNGVwOGZrd2E5NG90In0.gdsRu_UeU_uPi9IulBruXA'; // Set your mapbox token here

  const [Zip, ChangeZip] = useState()
  const [Text, ChangeText] = useState("")
  const [alertVisible, setAlert] = useState(false)

  const validateInput = () => {
      if(/^\d+$/.test(Text) && Text.length == 5){
        console.log("23")
        const getZipData = {
          "3": 3
        }
        if(getZipData.length == 0) {
          setAlert(true)
        } else {
          setAlert(false)
        }
      } else {
        setAlert(true)
      }
  }

  return (
    <>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link href="https://fonts.googleapis.com/css2?family=Lobster&family=Saira:ital,wght@0,100;0,200;0,400;0,500;1,100;1,200;1,400;1,500&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Lobster&family=Rubik+Wet+Paint&family=Saira:ital,wght@0,100;0,200;0,400;0,500;1,100;1,200;1,400;1,500&family=Titan+One&display=swap" rel="stylesheet"></link>
      </head>
      <div className = {styles.HeaderBox}>
        H2G<MdWaterDrop className = {styles.drop} size = {60}/><MdWaterDrop size = {60} className = {styles.drop}/>d
      </div>
      <div className = {styles.AppContainer}>
        <div className={styles.SearchContainer}>
          <div className = {styles.SearchBarContainer}>
            <input className = {styles.SearchBar} value = {Text} onChange = {(inp) => ChangeText(inp.target.value)}></input>
            <button className= {styles.SubmitSearch} onClick = {() => validateInput()}><MdOutlineSearch size = {20}/></button>
          </div>
          {alertVisible ? <h2 className = {styles.Alert}>Invalid Zip</h2>: <></>}
        </div>
        <div className={styles.MapContainer} onClick = {validateInput}>
            <Map
              initialViewState={{
                latitude: 37.8,
                longitude: -122.4,
                zoom: 14
              }}
              mapStyle="mapbox://styles/mapbox/streets-v9"
              mapboxAccessToken={MAPBOX_TOKEN}
            >
              <Marker longitude={-122.4} latitude={37.8} color="red" />
            </Map>
        </div>

      </div>
    </>
  );
}

export default App;

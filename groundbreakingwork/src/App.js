import { MdOutlineSearch, MdWaterDrop } from 'react-icons/md'
import * as React from 'react';
import {useState, useMemo, useCallback, useEffect} from 'react';
import Map, {Popup, Source, Layer} from 'react-map-gl';
import {countiesLayer, highlightLayer} from './map-style';
import styles from "./App.module.css"

const data = [
  {
    "ID": 188449,
    "DateTime": "2018-12-02",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 2.9,
    "Conductance": 386,
    "Dissolved_oxygen": 12.5,
    "PH": 8.2,
    "Turbidity": 38.5
  },
  {
    "ID": 188450,
    "DateTime": "2018-12-03",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 2.3,
    "Conductance": 367,
    "Dissolved_oxygen": 12.6,
    "PH": 8.1,
    "Turbidity": 34.9
  },
  {
    "ID": 188451,
    "DateTime": "2018-12-04",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 1,
    "Conductance": 374,
    "Dissolved_oxygen": 13.3,
    "PH": 8.2,
    "Turbidity": 15.5
  },
  {
    "ID": 188452,
    "DateTime": "2018-12-05",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 1.2,
    "Conductance": 388,
    "Dissolved_oxygen": 13.4,
    "PH": 8.2,
    "Turbidity": 10.1
  },
  {
    "ID": 188453,
    "DateTime": "2018-12-06",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 1.2,
    "Conductance": 398,
    "Dissolved_oxygen": 13.5,
    "PH": 8.2,
    "Turbidity": 8.2
  },
  {
    "ID": 188454,
    "DateTime": "2018-12-07",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 0.6,
    "Conductance": 405,
    "Dissolved_oxygen": 13.9,
    "PH": 8.3,
    "Turbidity": 6.7
  },
  {
    "ID": 188455,
    "DateTime": "2018-12-08",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 0,
    "Conductance": 413,
    "Dissolved_oxygen": 14.2,
    "PH": 8.3,
    "Turbidity": 6
  },
  {
    "ID": 188456,
    "DateTime": "2018-12-09",
    "Latitude": 44.0725203,
    "Longitude": -84.0199939,
    "Temperature": 0,
    "Conductance": 422,
    "Dissolved_oxygen": 14.2,
    "PH": 8.3,
    "Turbidity": 5.4
  }
]

function App() {
  const MAPBOX_TOKEN = 'pk.eyJ1IjoiZ2Vvcmdpb3MtdWJlciIsImEiOiJjanZidTZzczAwajMxNGVwOGZrd2E5NG90In0.gdsRu_UeU_uPi9IulBruXA'
  const [Text, ChangeText] = useState("")
  const [alertVisible, setAlert] = useState(false)
  const [hoverInfo, setHoverInfo] = useState(null);
  const [countyData, setCountyData] = useState(data)

  const onHover = useCallback(event => {
    const county = event.features && event.features[0];
    setHoverInfo({
      longitude: event.lngLat.lng,
      latitude: event.lngLat.lat,
      countyName: county && county.properties.COUNTY
    });
  }, []);
 

  const selectedCounty = (hoverInfo && hoverInfo.countyName) || '';
  const filter = useMemo(() => ['in', 'COUNTY', selectedCounty], [selectedCounty]);

  function isInteger (value) {
    if (!isNaN(value) && !isNaN(parseFloat(value)) && Number(value) % 1 === 0){
        return true;
    } else {
        return false;
    }
  }

  const updateText = (str) => {
    ChangeText(str)
    if(isInteger(str)){
      setAlert(false)
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
            <input className = {styles.SearchBar} value = {Text} onChange = {(inp) => updateText(inp.target.value)}></input>
            <button className= {styles.SubmitSearch}><MdOutlineSearch size = {20}/></button>
          </div>
          {!alertVisible ? <h2 className = {styles.Years}>{Text ? Text : 0} Years</h2>: <></>}
          {alertVisible ? <h2 className = {styles.Alert}>Invalid Time</h2>: <></>}
        </div>
        <div className={styles.MapContainer}>
            <Map
            initialViewState={{
              latitude: 38.88,
              longitude: -98,
              zoom: 3
            }}
            onMouseMove={onHover}
            minZoom={4}
            mapStyle="mapbox://styles/mapbox/light-v9"
            mapboxAccessToken={MAPBOX_TOKEN}
            interactiveLayerIds={['counties']}
          >
            <Source type="vector" url="mapbox://mapbox.82pkq93d">
              <Layer beforeId="waterway-label" {...countiesLayer} />
              <Layer beforeId="waterway-label" {...highlightLayer} filter={filter} />
            </Source>
            {selectedCounty && (
              <Popup
                longitude={hoverInfo.longitude}
                latitude={hoverInfo.latitude}
                offset={[0, -10]}
                closeButton={false}
                className="county-info"
              >
                {selectedCounty}
              </Popup>
            )}
          </Map>
        </div>

      </div>
    </>
  );
}

export default App;

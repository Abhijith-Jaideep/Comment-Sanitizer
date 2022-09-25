import React from 'react'
import ReactDOM from "react-dom/client"
import App from './Components/App'
import AlertStates from './context/Alert/AlertStates'
import UserStates from './context/User/UserStates'

const root = ReactDOM.createRoot(document.getElementById("root"))
root.render(<UserStates><AlertStates><App /></AlertStates></UserStates>)
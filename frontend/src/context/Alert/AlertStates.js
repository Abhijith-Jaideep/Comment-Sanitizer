import React, { useState } from 'react'
import AlertContext from './AlertContext'

const AlertStates = (props) => {

    const [show, setshow] = useState(false)
    const [msg, setmsg] = useState({ msg: "", type: "" })
    const [mode, setmode] = useState(localStorage.getItem("mode"))

    const showAlert = (msg, type) => {
        setmsg({ msg, type })
        setshow(true)
        setTimeout(() => { setshow(false) }, 2000)
    }

    const changeMode = () => {
        if (mode === "light") {
            setmode("dark")
            localStorage.setItem("mode", "dark")
        }
        else {
            setmode("light")
            localStorage.setItem("mode", "light")
        }
    }

    return (
        <AlertContext.Provider value={{ msg, show, showAlert, mode, changeMode }}>
            {props.children}
        </AlertContext.Provider>
    )
}

export default AlertStates
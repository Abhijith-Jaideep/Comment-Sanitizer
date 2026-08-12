import React, { useState } from 'react'
import AlertContext from './AlertContext'

/**
 * Dark mode has been removed. `mode` stays in the context as a constant so the
 * components that read it keep rendering their light branch, rather than
 * editing seventeen files to strip a ternary out of each one.
 *
 * This also fixes a real bug. `mode` used to be seeded from
 * localStorage.getItem("mode"), which is null on a first visit, so every
 * `mode === "light"` check fell through to the dark branch and class names
 * rendered as `bg-null`. A first-time visitor saw a half-broken dark page.
 */
const MODE = "light"

const AlertStates = (props) => {

    const [show, setshow] = useState(false)
    const [msg, setmsg] = useState({ msg: "", type: "" })

    const showAlert = (msg, type) => {
        setmsg({ msg, type })
        setshow(true)
        setTimeout(() => { setshow(false) }, 2000)
    }

    return (
        <AlertContext.Provider value={{ msg, show, showAlert, mode: MODE }}>
            {props.children}
        </AlertContext.Provider>
    )
}

export default AlertStates

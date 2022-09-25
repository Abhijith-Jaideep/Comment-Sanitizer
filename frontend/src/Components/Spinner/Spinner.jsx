import React from 'react'
import spinner from "./spinner.gif"

const Spinner = () => {
    return (
        <div style={{display:"flex",justifyContent:"center",width:"100%"}}>
            <img src={spinner} alt="spinner" />
        </div>

    )
}

export default Spinner
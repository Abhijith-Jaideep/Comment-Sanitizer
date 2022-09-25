import React from 'react'

const Alert = (props) => {
    
    return (
        
        <div className={`alert alert-${props.msg.type} d-flex align-items-center role="alert" m-0`}>
            <div style={{fontWeight:"bold"}}>
                {props.msg.msg}
            </div>
        </div>
    )
}

export default Alert
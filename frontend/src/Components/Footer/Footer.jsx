import React from 'react'
import "./footer.css"

const Footer = (props) => {
  return (
    <div className="Footer"  style={{color:props.mode==="light"?"black":"darkorange",backgroundColor:props.mode==="light"?"darkorange":"#212529",borderTop:props.mode==="dark"?"0.5px solid darkorange":""}}>
      <p><i className="fa-solid fa-copyright"></i> 2022 All rights reserved</p>
      <p>Blogger.in</p>
    </div>
  )
}

export default Footer
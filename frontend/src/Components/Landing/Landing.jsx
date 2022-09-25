import React from 'react'
import { Link } from 'react-router-dom'
import "./landing.css"


const Landing = (props) => {
  return (
    <div className={`Landing bg-${props.mode}`}>
      <div className='container'>

        <div className='heading mt-3'>
          
          <h1>
            Inc. This Morning<br />
            <span style={{ color: props.mode === "dark" ? "white" : 'grey' }}>"</span>Blog<span style={{ color: props.mode === "dark" ? "white" : 'grey' }}>"</span>
          </h1>

          <p style={{ color: props.mode === "dark" ? "white" : 'grey' }}>
            Awesome place to make oneself<br />productive and entertained through daily updates
          </p>

        </div>

        <div className='image'>
          <div className='transparent'>
           
            <h2>The home for great readers and writers</h2><hr/>

            <Link to="/feed"><button className='btn mb-2 btn-dark' style={{ backgroundColor: "darkorange" }}>Start Reading</button></Link>
            <Link to={localStorage.getItem("token") ? "/write" : "/login"}><button className="btn btn-secondary">Write A Post</button></Link>

          </div>
        </div>

      </div>
    </div>
  )
}

export default Landing
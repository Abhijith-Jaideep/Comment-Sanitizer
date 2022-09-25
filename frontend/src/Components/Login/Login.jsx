import React, { useContext, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import AlertContext from '../../context/Alert/AlertContext'
import UserContext from '../../context/User/UserContext'
import picture from "./login.jpg"
import "./login.css"
const Login = () => {

  const navigate = useNavigate()

  const usercontext = useContext(UserContext)
  const alertcontext = useContext(AlertContext)
  const { login } = usercontext
  const { showAlert, mode ,} = alertcontext

  const [info, setinfo] = useState({ username: "", password: "" })

  const onChange = (e) => {
    setinfo({ ...info, [e.target.name]: e.target.value })
  }


  const submit = async (e) => {
    e.preventDefault()
    await login(info.username, info.password)
    const token = localStorage.getItem("token")
    if (!token) {
      showAlert("Invalid Credentials", "danger")
    }
    else {
      showAlert("Logged in successfully", "success")
      navigate("/feed")
      
    }

  }

  return (
    <div className="Login h-100 ">

      <div className={`bg-${mode} text-${mode === "light" ? "dark" : "light"} h-100 p-5`} >

        <div className='logincard container d-flex p-0' style={{ backgroundColor: mode === "light" ? "white" : "#303443" }}>


          <div className="loginform p-3">

            <form onSubmit={submit} autoComplete="off">

              <h1 align="center">Welcome Back!...</h1>

              <div className="mb-3 mt-5 w-100">
                <label htmlFor="username" className="form-label">Username</label>
                <input type="text" minLength={3} required className="form-control" id="username" onChange={onChange} name="username" aria-describedby="username" />
              </div>
              
              <div className="mb-5 w-100">
                <label htmlFor="password" className="form-label">Password</label>
                <input type="password" minLength={8} required className="form-control" name="password" onChange={onChange} id="password" />
              </div>

              <button type="submit" className="btn w-50 p-2">Login</button>

            </form>
          </div>

          <div className='pic w-50'>
            <img src={picture} alt="login side pic" width="100%" height="100%" />
          </div>

        </div>


      </div>

    </div>
  )
}

export default Login
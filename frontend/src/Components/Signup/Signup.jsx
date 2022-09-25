import React, { useContext, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import AlertContext from '../../context/Alert/AlertContext'
import UserContext from '../../context/User/UserContext'
import picture from './signup.jpg'
import "./signup.css"

const Signup = () => {

  const navigate = useNavigate()

  const context = useContext(UserContext)
  const alertcontext = useContext(AlertContext)
  const { signup } = context
  const { showAlert, mode } = alertcontext
  const [file, setFile] = useState(null)

  const [info, setinfo] = useState({ firstname: "", lastname: "", username: "", password: "", confirmpassword: '' })

  const onChange = (e) => {
    setinfo({ ...info, [e.target.name]: e.target.value })
  }

  const onFileChange = (e) => {
    setFile(e.target.files[0])
  }


  const submit = async(e) => {
    if (info.password !== info.confirmpassword) {
      showAlert("password mismatch!", 'danger')
      return 0;
    }
    else {
      e.preventDefault()
      const msg = await signup(info.firstname, info.lastname, info.username, info.password, file)
      if(msg==="User already exists"){
        showAlert(msg, "danger")
      }
      else{
        showAlert(msg,"success")
        navigate(-1)
      }
    }
  }

  return (
    <div className="signup h-100">

      <div className={`bg-${mode} text-${mode === "light" ? "dark" : "light"} h-100 p-5`} >


        <div className='signupcard container d-flex p-0' style={{ backgroundColor: mode === "light" ? "white" : "#303443" }}>

          <div className='pic w-50'>
            <img src={picture} alt="signup side pic" width="100%" height="100%" />
          </div>

          <div className="signupform p-3 ">

            <form onSubmit={submit} autoComplete="off" encType='multipart/form-data'>

              <h1 align="center">Join Us...</h1>



              <div className="mt-3 w-100 d-flex" style={{ flexDirection: "column", alignItems: "center" }}>
                <i className="fa-solid fa-circle-user fa-6x"></i>
                <label htmlFor="formFile" className="form-label"><center>Upload Profile Picture</center></label>
                <input className="form-control w-50" type="file" id="formFile" onChange={onFileChange} />
              </div>
              
              <div className='name'>

                <div className="mt-2 w-50">
                  <label htmlFor="firstname" className="form-label">First Name</label>
                  <input type="text" minLength={3} required className="form-control" id="firstname" onChange={onChange} name="firstname" />
                </div>

                <div className="mt-2 w-50">
                  <label htmlFor="lastname" className="form-label">Last Name</label>
                  <input type="text" minLength={3} required className="form-control" id="lastname" onChange={onChange} name="lastname"/>
                </div>
              </div>

              <div className="mb-2 mt-3 w-100">
                <label htmlFor="username" className="form-label">Username</label>
                <input type="text" minLength={3} required className="form-control" id="username" onChange={onChange} name="username" aria-describedby="username" />
              </div>

              <div className="mb-2 w-100">
                <label htmlFor="password" className="form-label">Password</label>
                <input type="password" minLength={8} required className="form-control" name="password" onChange={onChange} id="password" />
              </div>

              <div className="w-100 my-3">
                <label htmlFor="confirmpassword" className="form-label">Confirm Password</label>
                <input type="password" minLength={8} required className="form-control" name="confirmpassword" onChange={onChange} id="confirmpassword" />
              </div>


              <button type="submit" className="btn w-50 p-2 ">Sign Up</button>

            </form>
          </div>

        </div>

      </div>
    </div>

  )
}

export default Signup
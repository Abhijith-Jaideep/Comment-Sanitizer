import React, { useContext, useEffect, useState } from 'react'
import "./navbar.css"
import { NavLink, Link } from "react-router-dom"
import AlertContext from '../../context/Alert/AlertContext'
import UserContext from '../../context/User/UserContext'

const NavBar = (props) => {

    const [icon, seticon] = useState("fa-solid fa-moon")
    const context = useContext(AlertContext)
    const { showAlert, changeMode, mode } = context

    const usercontext = useContext(UserContext)
    const { userdata, fetchuserdata } = usercontext

    useEffect(() => {
        if (props.token)
            fetchuserdata()
        // eslint-disable-next-line
    }, [props.token])




    return (
        <nav className={`navbar navbar-${mode}`} style={{ backgroundColor: mode === "light" ? "darkorange" : "#212529" }} >


            <div className='nav-links'>
                <div className='d-flex'>



                    <div className="form-check form-switch navbar-brand m-0">
                        <input className="form-check-input" style={{ backgroundColor: mode === "light" ? "white" : "darkorange" }} onClick={() => {
                            changeMode();

                            if (icon === "fa-solid fa-moon")
                                seticon("fa-solid fa-sun")
                            else
                                seticon("fa-solid fa-moon")
                            showAlert(`${mode === "light" ? "Enabled" : "Disabled"} dark mode`, "primary")
                        }} type="checkbox" role="switch" id="flexSwitchCheckDefault" defaultChecked={mode === "dark" ? true : false} />

                    </div>
                    <div className="navbar-brand">
                        <i style={{ color: mode === "dark" ? "darkorange" : "black" }} className={`${icon}`}></i>
                    </div>

                    <NavLink className="navbar-brand" to="/"><i className="fa-solid fa-house-chimney"></i> Home</NavLink>
                    {props.token &&
                        <div className='d-flex'>
                            <Link className="navbar-brand" to="/write"><i className="fa-solid fa-feather"></i><span> Write</span></Link>
                            <Link className="navbar-brand" to="/feed" ><i className="fa-solid fa-book-open"></i><span> Feed</span></Link>
                        </div>
                    }
                </div>

                <div className='d-flex'>

                    {!props.token && <>
                        <Link className="navbar-brand" to="/signup"><i className="fa-solid fa-user-plus"></i> Signup</Link>
                        <Link className="navbar-brand" to="/login"><i className="fa-solid fa-user-pen"></i> Login</Link>
                    </>}

                    {props.token && <Link to="/profile" className='navbar-brand'>
                            {userdata.profilepic && <img src={`data:image/jpeg;base64,${userdata.profilepic}`} alt="profilepic" style={{ borderRadius: "50%", height: "36px", width: "36px", border: "2px solid black" }} />}
                            {!userdata.profilepic && <i className="fa-solid fa-circle-user"></i>}
                            <span style={{ margin: "10%" }}>{userdata.username}</span>
                    </Link>
                    }
                </div>
            </div>

        </nav>
    )
}

export default NavBar
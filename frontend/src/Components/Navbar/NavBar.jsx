import React, { useContext, useEffect } from 'react'
import "./navbar.css"
import { NavLink, Link } from "react-router-dom"
import UserContext from '../../context/User/UserContext'

/**
 * Bootstrap's collapse handles small screens. The bundle is already loaded in
 * public/index.html, so the toggler needs no extra JavaScript here.
 *
 * The dark mode switch that used to sit at the far left has been removed.
 */
const NavBar = (props) => {

    const usercontext = useContext(UserContext)
    const { userdata, fetchuserdata } = usercontext

    useEffect(() => {
        if (props.token)
            fetchuserdata()
        // eslint-disable-next-line
    }, [props.token])

    return (
        <nav
            className="navbar navbar-expand-lg navbar-light"
            style={{ backgroundColor: "darkorange" }}
        >
            <div className="container-fluid">

                {/* Home stays outside the collapse so there is always a way
                    back on a phone, even with the menu shut. */}
                <NavLink className="navbar-brand" to="/">
                    <i className="fa-solid fa-house-chimney"></i> Home
                </NavLink>

                <button
                    className="navbar-toggler"
                    type="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#mainNav"
                    aria-controls="mainNav"
                    aria-expanded="false"
                    aria-label="Toggle navigation"
                >
                    <span className="navbar-toggler-icon"></span>
                </button>

                <div className="collapse navbar-collapse" id="mainNav">

                    <div className="navbar-nav me-auto">
                        {props.token && <>
                            <Link className="nav-link" to="/write">
                                <i className="fa-solid fa-feather"></i><span> Write</span>
                            </Link>
                            <Link className="nav-link" to="/feed">
                                <i className="fa-solid fa-book-open"></i><span> Feed</span>
                            </Link>
                        </>}
                    </div>

                    <div className="navbar-nav ms-auto align-items-lg-center">
                        {!props.token && <>
                            <Link className="nav-link" to="/signup">
                                <i className="fa-solid fa-user-plus"></i> Signup
                            </Link>
                            <Link className="nav-link" to="/login">
                                <i className="fa-solid fa-user-pen"></i> Login
                            </Link>
                        </>}

                        {props.token &&
                            <Link to="/profile" className="nav-link d-flex align-items-center gap-2">
                                {userdata.profilepic &&
                                    <img
                                        src={`data:image/jpeg;base64,${userdata.profilepic}`}
                                        alt="profilepic"
                                        style={{ borderRadius: "50%", height: "36px", width: "36px", border: "2px solid black" }}
                                    />}
                                {!userdata.profilepic && <i className="fa-solid fa-circle-user"></i>}
                                <span>{userdata.username}</span>
                            </Link>
                        }
                    </div>

                </div>
            </div>
        </nav>
    )
}

export default NavBar

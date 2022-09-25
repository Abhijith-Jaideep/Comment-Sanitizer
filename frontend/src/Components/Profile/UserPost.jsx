import React from 'react'
import { useState } from 'react'
import { useEffect } from 'react'
import {Link} from 'react-router-dom'

const UserPost = (props) => {

    const [date, setdate] = useState('')



    useEffect(() => {
        setdate(new Date(props.timestamp).toLocaleString("en-In", { timeZone: "Asia/Kolkata" }))
        // eslint-disable-next-line
    }, [])


    return (
        <div className={`card  bg-${props.mode === "dark" ? "black" : "white"} my-3 w-75`} style={{ boxShadow: 'rgba(0, 0, 0, 0.16) 0px 3px 6px, rgba(0, 0, 0, 0.23) 0px 3px 6px', flexWrap: "wrap" }}>
            <div className="card-body">
                <div className="content d-flex"  style={{ justifyContent: "space-between" }}>

                    <h3 className="card-title w-50">{props.title}</h3>
                    {date}
                    <div>
                        <i className="fa-solid fa-trash-can fa-2x mx-4" style={{ color: "red", cursor: "pointer" }} data-bs-toggle="modal" data-bs-target="#exampleModal" onClick={() => { props.postid(props.id, props.title, props.description) }}></i>
                        <i className="fa-solid fa-pen-to-square fa-2x" style={{ color: "green", cursor: "pointer" }} data-bs-toggle="modal" data-bs-target="#exampleModal2" onClick={() => { props.postid(props.id, props.title, props.description) }}></i>
                    </div>

                </div>
                <div>
                    {props.description}
                </div>
                <div className='mt-1'>
                    <Link onClick={() => { localStorage.setItem("postid", props.id) }} to="/feed/post"><button type="button" class="btn" style={{backgroundColor:"darkorange"}}>Go To Post</button></Link>
                </div>

            </div>


        </div>
    )
}

export default UserPost
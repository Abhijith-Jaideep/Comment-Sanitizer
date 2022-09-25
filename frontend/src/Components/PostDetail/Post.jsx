import React, { useEffect, useState } from 'react'
import { useContext } from 'react'
import AlertContext from '../../context/Alert/AlertContext'
import CommentSection from "./CommentSection/CommentSection"
import "./post.css"

const Post = (props) => {

    const [date, setdate] = useState('')
    const context = useContext(AlertContext)
    const { mode } = context

    const [userdata, setuserdata] = useState({ profilepic: "something" })

    const [rerender, setrerender] = useState(0)

    const fetchuserdatabyusername = async () => {
        const response = await fetch(`http://localhost:5000/api/auth/fetchbyusername/${props.username}`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        })

        setuserdata(await response.json())

        setTimeout(() => {
            setrerender(1)
        }, 300)

    }

    useEffect(() => {
        setdate(new Date(props.timestamp).toLocaleString("en-IN", { timeZone: 'Asia/Kolkata' }))
        console.log(props.username)
        fetchuserdatabyusername()
        // eslint-disable-next-line
    }, [rerender])

    return (
        <div className='specificpostitem container my-5 d-flex' style={{ justifyContent: "center" }}>

            <div className={`card w-75 text-${mode === "dark" ? "white" : "dark"}`} style={{ backgroundColor: mode === "dark" ? "#303443" : "white" }}>

                <div className="card-header d-flex" style={{ backgroundColor: mode === "dark" ? "black" : "darkorange", alignItems: "center" }}>

                    <div className="user">

                        {userdata.profilepic !== "something" && userdata.profilepic &&
                            <img src={`data:image/jpeg;base64,${userdata.profilepic}`} width="36px" height="36px" style={{borderRadius:"100%"}} alt="postimg" />
                        }
                        {userdata.profilepic === null && <i className="fa-solid fa-circle-user"></i>}
                        <span className='mx-2'>{props.username}</span>
                    </div>

                    <div style={{ width: "fit-content" }}>
                        <b>{date}</b>
                    </div>

                </div>

                {props.postimg && <div style={{ display: 'flex', justifyContent: "center", height: "400px" }}>
                    <img src={`data:image/jpeg;base64,${props.postimg}`} width="100%" height="100%" alt="postimg" />
                </div>}
                <div className="card-body">

                    <h3>{props.title}</h3>
                    <div>
                        <p className="card-text">{props.description}</p>
                        <i className="fa-solid fa-heart fa-2x" style={{ cursor: "pointer" }}></i>
                    </div>
                </div>

                <div className="card-footer">
                    <CommentSection />
                </div>

            </div>
        </div>
    )
}

export default Post
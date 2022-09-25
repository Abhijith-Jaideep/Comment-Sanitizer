import React, { useEffect, useState } from 'react'
import { useContext } from 'react'
import { Link } from 'react-router-dom'
import AlertContext from '../../context/Alert/AlertContext'
import "./postitem.css"
import defaultimg from './postimg.jpg'


const PostItem = (props) => {

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
        fetchuserdatabyusername()
        // eslint-disable-next-line
    }, [rerender])


    return (
        <div className='postitem w-100 my-5 d-flex' style={{ justifyContent: "center" }}>

            <div className={`card w-100 text-${mode === "dark" ? "white" : "dark"}`} style={{ backgroundColor: mode === "dark" ? "#303443" : "white"}}>

                <div className="card-header d-flex" style={{ backgroundColor: mode === "dark" ? "black" : "darkorange", alignItems:"center" ,flexWrap:"wrap"}}>

                    <div className="user">
                        {userdata.profilepic !== "something" && userdata.profilepic &&
                            <img src={`data:image/jpeg;base64,${userdata.profilepic}`} width="24px" height="24px" alt="postimg" style={{borderRadius:"100%",border:"1px solid black"}} />
                        }
                        {userdata.profilepic === null && <i className="fa-solid fa-circle-user"></i>}
                        <span className='mx-2'>{props.username}</span>
                    </div>

                    <div style={{ width: "fit-content" }}>
                        <b>{date}</b>
                    </div>

                </div>

                {props.postimg && <div style={{ display: 'flex', justifyContent: "center" }}>
                    <img src={`data:image/jpeg;base64,${props.postimg}`} width="100%" height="250px" alt="postimg" />
                </div>}
                {!props.postimg && <div style={{ display: 'flex', justifyContent: "center" }}>
                    <img src={defaultimg} width="100%" height="250px" alt="postimg" />
                </div>}
                <div className="card-body">

                    <h2>{props.title}</h2>
                    <div>
                        <p className="card-text">{props.description}</p>
                        <Link to="./post" onClick={() => { localStorage.setItem("postid", props.id) }}><i className="fa-solid fa-2x fa-comment-dots" style={{ color: mode === 'light' ? 'black' : 'white', cursor: "pointer" }}></i></Link>
                        <i className="fa-solid fa-heart fa-2x mx-3" style={{ cursor: "pointer" }}></i>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default PostItem
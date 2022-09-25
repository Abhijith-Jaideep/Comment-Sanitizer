import React, { useContext, useState } from 'react'
import { useEffect } from 'react'
import UserContext from '../../../context/User/UserContext'

const CommentItem = (props) => {

    const usercontext = useContext(UserContext)
    const { userdata } = usercontext

    const [fetchdata, setfetchdata] = useState({ profilepic: "something" })

    const [rerender, setrerender] = useState(0)

    const fetchuserdatabyusername = async () => {
        const response = await fetch(`http://localhost:5000/api/auth/fetchbyusername/${props.username}`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        })

        setfetchdata(await response.json())

        setTimeout(() => {
            setrerender(1)
        }, 300)

    }


    useEffect(() => {
        fetchuserdatabyusername()
        // eslint-disable-next-line
    }, [rerender])


    return (
        <div>

            <div className="d-flex" style={{ justifyContent: "space-between", }}>
                <div>
                    {!fetchdata.profilepic && <i className="fa-solid fa-circle-user"></i>}
                    {fetchdata.profilepic && <img src={`data:image/jpeg;base64,${fetchdata.profilepic}`} alt="profilepic" height="36px" width="36px" style={{borderRadius:"100%"}}></img>}
                    <b className="mx-2">{props.username}</b>
                </div>
                <div className='d-flex'>{
                    new Date(props.timestamp).toLocaleString("en-In", { timeZone: "Asia/Kolkata" })}
                    {userdata.username === props.username && <div>
                        <i className="fa-solid fa-trash-can mx-3" style={{ color: "red", cursor: "pointer" }} data-bs-toggle="modal" data-bs-target="#exampleModal" onClick={() => { localStorage.setItem("cid", props.id) }}></i>
                        <i className="fa-solid fa-pen-to-square mx-3" style={{ color: "green", cursor: "pointer" }} data-bs-toggle="modal" data-bs-target="#exampleModal2" onClick={() => { localStorage.setItem("cid", props.id) }}></i>
                    </div>}
                </div>
            </div>
            {props.comment}
        </div>
    )
}

export default CommentItem
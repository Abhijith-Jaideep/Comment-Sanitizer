import React from 'react'
import { useContext } from 'react'
import { useState } from 'react'
import { useEffect } from 'react'
import AlertContext from '../../../context/Alert/AlertContext'
import CommentItem from './CommentItem'

const CommentSection = () => {

    const [comments, setcomments] = useState([])
    const [updateComment, setupdateComment] = useState('')
    const [rerender, setrerender] = useState(0)

    const [postcomment, setpostcomment] = useState('')
    const context = useContext(AlertContext)
    
    const { mode, showAlert } = context


    const postComment = async () => {

        const postid = localStorage.getItem("postid")
        const response = await fetch(`http://localhost:5000/api/comments/postComment/${postid}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "auth-token": localStorage.getItem("token")
            },
            body: JSON.stringify({ comment: postcomment })
        })

        const res = await response.json()
        return res


    }

    const fetchComments = async () => {
        const id = localStorage.getItem("postid")
        const response = await fetch(`http://localhost:5000/api/comments/fetchComments/${id}`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        })


        setcomments(await response.json())

        setTimeout(() => {
            if (setrerender === 0)
                setrerender(rerender + 1)
        }, 200)
    }


    const onChange = (e) => {
        setpostcomment(e.target.value)
    }

    const onUpdateChange = (e) => {
        setupdateComment(e.target.value)
    }

    const handleClickUpdate = async () => {
        const response = await fetch(`http://localhost:5000/api/comments/updateComment/${localStorage.getItem("cid")}`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
                "auth-token": localStorage.getItem("token")
            },
            body: JSON.stringify({ comment: updateComment })
        })

        const json = await response.json()



        setTimeout(() => {
            if (rerender === 1)
                setrerender(0)
            else setrerender(1)
        }, 100)

        if (json.msg === "Comment Updated!")
            showAlert(json.msg, "warning")
        else
            showAlert(json.msg, "danger")
    }





    const handleClickDelete = async () => {
        const response = await fetch(`http://localhost:5000/api/comments/deleteComment/${localStorage.getItem("cid")}`, {
            method: "DELETE",
            headers: {
                "Content-Type": "application/json",
                "auth-token": localStorage.getItem("token")
            }
        })

        const json = await response.json()

        setTimeout(() => {
            if (rerender === 1)
                setrerender(0)
            else setrerender(1)
        }, 100)

        if (json.msg === "Comment Deleted!")
            showAlert(json.msg, "warning");
        else
            showAlert(json.msg, "danger");

    }


    const onSubmit = async (e) => {
        e.preventDefault()
        const res = await postComment()
        if (res.msg === "Negative comment") {

            showAlert(res.msg, "danger")
        }
        else {
            showAlert("Comment Posted!", "primary")
        }

        setTimeout(() => {
            setrerender(rerender + 1)
        }, 200)

    }

    useEffect(() => {
        fetchComments()
        // eslint-disable-next-line
    }, [rerender])

    return (

        <div>
            {comments.length === 0 && <h3>No comments for this post</h3>}
            {comments.length !== 0 && <div className={`card bg-${mode}`} style={{ height: "50vh", overflowY: "scroll", boxShadow: "none" }}>
                <ul className="list-group list-group-flush">
                    {comments.map((Element) => {
                        return (
                            <div key={Element._id}>
                                <li className={`list-group-item bg-${mode === "dark" ? "secondary" : "white"} text-${mode === "dark" ? "white" : "black"}`} >
                                   <CommentItem id={Element._id} username={Element.username} timestamp = {Element.timestamp} comment={Element.comment}/>
                                </li>
                            </div>
                        )
                    })}
                </ul>
            </div>}

            {!localStorage.getItem("token") && <h4>Login to post comments</h4>}
            {localStorage.getItem("token") && <form onSubmit={onSubmit} className='mt-3'>
                <label htmlFor="comment" className="form-label">Post your comment</label>
                <div className='d-flex'>

                    <input autoComplete="off" type="text" id="comment" className="form-control" onChange={onChange} aria-describedby="passwordHelpBlock" />
                    <button className=' mx-2 btn' style={{backgroundColor:"darkorange"}}>Send<i className="fa-solid fa-paper-plane"></i></button>
                </div>

                <div id="passwordHelpBlock" className={`form-text text-${mode === "dark" ? "white" : "black"}`}>
                    Please Keep The Comment Section Clean and Professional
                </div>
            </form>}








            <div className="modal fade" id="exampleModal" tabIndex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
                <div className="modal-dialog">
                    <div className={`modal-content  bg-${mode}`}>
                        <div className={`modal-header bg-danger`}>
                            <h5 className="modal-title" id="exampleModalLabel">DELETE Comment?</h5>
                            <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div className={`modal-body bg-${mode}`}>
                            Are you sure you want to delete?
                        </div>
                        <div className={`modal-footer bg-${mode}`}>
                            <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" className="btn btn-danger" onClick={handleClickDelete} data-bs-dismiss="modal">Delete</button>
                        </div>
                    </div>
                </div>
            </div>



            <div className="modal fade " id="exampleModal2" tabIndex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
                <div className="modal-dialog">
                    <div className={`modal-content  bg-${mode}`}>
                        <div className={`modal-header  bg-success`}>
                            <h5 className="modal-title" id="exampleModalLabel">Update Comment</h5>
                            <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div className={`modal-body bg-${mode}`}>
                            <div className="form-group">
                                <label htmlFor="title">Comment</label>
                                <input onChange={onUpdateChange} type="text" className="form-control" name="updateComment" id="updateComment" aria-describedby="updateComment" value={updateComment} />
                            </div>
                            <div className={`modal-footer  bg-${mode}`}>
                                <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                <button type="button" className="btn btn-success" onClick={handleClickUpdate} data-bs-dismiss="modal">Update</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>


        </div>








    )
}

export default CommentSection
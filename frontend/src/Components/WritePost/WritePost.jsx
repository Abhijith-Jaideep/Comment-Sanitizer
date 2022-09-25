import React, { useContext, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PostContext from '../../context/Post/PostContext'
import AlertContext from "../../context/Alert/AlertContext"
import "./writepost.css"
import { useEffect } from 'react'

const WritePost = () => {

  const navigate = useNavigate()

  const context = useContext(PostContext)
  const alertcontext = useContext(AlertContext)


  const { createPost } = context
  const { showAlert, mode } = alertcontext

  const [data, setdata] = useState({ title: "", description: "" })
  const [file, setFile] = useState([])

  const onChange = (e) => {
    setdata({ ...data, [e.target.name]: e.target.value })
  }

  const onFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const submit = async (e) => {
    e.preventDefault()
    const msg = await createPost(data.title, data.description, file)
    if (msg === "Posted Successfully!") {

      showAlert(msg, "success")
      navigate("/")
    }
    else{
      showAlert(msg,"danger")
    }

  }

  useEffect(() => {
    if (!localStorage.getItem("token")) {
      navigate("/login")
      showAlert("Login to use this feature", "danger")
    }

    //eslint-disable-next-line
  }, [])


  return (
    <div className={`writepost h-100 bg-${mode} text-${mode === "light" ? "dark" : "light"} p-5`}>
      <div className="container">
        <form onSubmit={submit} encType="multipart/form-data">

          <div className="details">
            <h1>What is on your mind today...</h1>
            <div className="mb-3">
              <label htmlFor="title" className="form-label">Title</label>
              <input type="text" className="form-control bg-light" onChange={onChange} name="title" id="description" />
            </div>

            <div className="mb-3">
              <label htmlFor="description" className="form-label">Description</label>
              <textarea rows={4} name="description" className="form-control bg-light" onChange={onChange} id="description" />
            </div>

            <div className="mb-3">
              <div class="form-group">
                <label for="">Image</label>
                <input type="file" class="form-control-file" onChange={onFileChange} name="postimg" />
              </div>
            </div>


            <div style={{ display: "flex", justifyContent: "center" }}>
              <button type="submit" className="btn w-50 p-2 " style={{ borderRadius: "16px", fontWeight: "bold", backgroundColor: "darkorange" }}>Publish</button>
            </div>

          </div>
        </form>

      </div>

    </div>
  )
}

export default WritePost
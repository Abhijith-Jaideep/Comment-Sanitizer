import React, { useState } from 'react'
import PostContext from './PostContext'


const PostStates = (props) => {

    const [posts, setposts] = useState([])
    const [userpost, setuserpost] = useState([])

    const fetchAllPosts = async () => {
        const response = await fetch("http://localhost:5000/api/posts/allPosts", {
            method: "GET"
        })
        const json = await response.json()
        setposts(json)

    }

    const fetchUserPosts = async () => {
        const response = await fetch("http://localhost:5000/api/posts/getUserPosts", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "auth-token": localStorage.getItem("token")
            }
        })

        setuserpost(await response.json())
    }

    const createPost = async (title, description, file) => {

        const data = new FormData()
        data.append("title",title)
        data.append("description",description)
        data.append("postimg",file)

        const response = await fetch("http://localhost:5000/api/posts/createPost", {
            method: "POST",
            headers: {
                
                "auth-token": localStorage.getItem("token")
            },
            body: data
        })

        const json = await response.json()
        console.log(json.msg)
        return(json.msg)
    }

    const deletePost = async (id) => {
        const response = await fetch(`http://localhost:5000/api/posts/deletePost/${id}`, {
            method: "DELETE",
            headers: {
                "Content-Type": "application/json",
                "auth-token": localStorage.getItem("token")
            }
        })

        const json = await response.json()
        return(await json.msg)

    }

    const updatePost = async (title, description, id) => {

        const data = { title, description }
        const response = await fetch(`http://localhost:5000/api/posts/updatePost/${id}`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
                "auth-token": localStorage.getItem('token')
            },
            body: JSON.stringify(data)
        })

        const json = await response.json()
        return(await json.msg)
    }


    return (
        <PostContext.Provider value={{ posts, userpost, fetchUserPosts, fetchAllPosts, createPost, deletePost, updatePost }}>
            {props.children}
        </PostContext.Provider>
    )

}

export default PostStates
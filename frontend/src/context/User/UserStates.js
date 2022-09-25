import React, { useState } from 'react'
import UserContext from './UserContext'



const UserStates = (props) => {

    const [userdata, setUserdata] = useState([])
    

    const fetchuserdata = async () => {
        const response = await fetch("http://localhost:5000/api/auth/fetch", {
            method: "GET",
            headers: {
                "auth-token": localStorage.getItem('token')
            }
        })

        setUserdata(await response.json())

    }

   

    const signup = async (firstname,lastname,username, password,file) => {

        const data = new FormData()

        data.append("firstname",firstname)
        data.append("lastname",lastname)
        data.append("username",username)
        data.append("password",password)
        data.append("profilepic",file)


        const response = await fetch("http://localhost:5000/api/auth/signup", {
            method: "POST",
            body: data
        })
        const json = await response.json()

        return(await json.msg)
    }

    const login = async (username, password) => {

        const data = JSON.stringify({ username, password })

        const response = await fetch("http://localhost:5000/api/auth/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: data
        })

        const res = await response.json()
        if (res.token) {
            localStorage.setItem("token", res.token)
        }

    }

    return (
        <UserContext.Provider value={{ signup, login, fetchuserdata, userdata,}}>
            {props.children}
        </UserContext.Provider>
    )

}

export default UserStates
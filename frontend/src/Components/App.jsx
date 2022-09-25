import React, { useContext } from 'react'
import Footer from './Footer/Footer'
import NavBar from './Navbar/NavBar'
import "./app.css"
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Signup from './Signup/Signup'
import Login from './Login/Login'
import Alert from './Alert/Alert'
import AlertContext from '../context/Alert/AlertContext'
import WritePost from './WritePost/WritePost'
import Profile from './Profile/Profile'
import PostStates from '../context/Post/PostStates'
import Landing from './Landing/Landing'
import Feed from './Feed/Feed'
import PostDetail from './PostDetail/PostDetail'




const App = () => {

	const context = useContext(AlertContext)
	const { msg, show, mode } = context	


	document.getElementById("root").style.backgroundColor = mode === "light" ? "white" : "#212529"
	document.body.style.backgroundColor = mode === "light" ? "light" : "#212529"

	return (



		<Router>


			<div className={`app bg-${mode} `}>
				<div className="head fixed-top">
					<NavBar token={localStorage.getItem("token")}/>

					{show && <Alert msg={msg} />}
				</div>
				<div className="body">
					<Routes >

						<Route path="/" element={<Landing mode={mode} />} />
						<Route path="/feed" element={
							<PostStates>
								<Feed />
							</PostStates>
						} />
						<Route path="/signup" element={<Signup />} />
						<Route path="/login" element={<Login />} />
						<Route path="feed/post" element={<PostDetail />} />
						<Route path="/write" element={<PostStates><WritePost /></PostStates>} />
						<Route path="/profile" element={<PostStates><Profile /></PostStates>} />
					</Routes>
				</div>

				<div className="footer">
					<Footer mode={mode} />
				</div>
			</div>
		</Router >
	)
}

export default App
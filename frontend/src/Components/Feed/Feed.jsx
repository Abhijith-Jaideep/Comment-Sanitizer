import React, { useContext, useEffect, useState } from 'react'
import AlertContext from '../../context/Alert/AlertContext'
import PostContext from '../../context/Post/PostContext'
import Spinner from '../Spinner/Spinner'
import PostItem from './PostItem'

const Feed = () => {

  const context = useContext(PostContext)
  const { posts, fetchAllPosts } = context

  const alertContext = useContext(AlertContext)
  const { mode } = alertContext


  const [loading, setloading] = useState(false)

  const [searchText, setSearchText] = useState('')

  const [displayPosts, setDisplayPosts] = useState([])


  const fetchposts = async () => {
    setloading(true)
    await fetchAllPosts()
    setloading(false)
  }

  const search = (e) => {
    setSearchText(e.target.value)
    let l = posts.filter((Element) => {
      if (Element.title.toLowerCase().substr(0, searchText.length).includes(searchText.toLowerCase())) { return Element }
      return ""
    })
    setDisplayPosts(l)
  }


  useEffect(() => {
    fetchposts()
    // eslint-disable-next-line
  }, [])



  return (
    <div >



      <div className='mt-5'>
        <div class="input-group container mb-3">
          <span class="input-group-text" id="basic-addon3" style={{backgroundColor:"darkorange"}}>Search Posts</span>
          <input type="text" class="form-control" id="basic-url" onChange={search} value={searchText} aria-describedby="basic-addon3" />
        </div>
          <h3 className='container'>Search: {searchText}</h3>
      </div>


      <div className={`container text-${mode === "dark" ? "white" : "black"}`}>
        {posts.length === 0 && <h1>No Posts in the blog</h1>}
      </div>
      {loading && <Spinner />}
      <div className="row container-fluid row-cols-1 row-cols-md-3 g-4 w-100">
        {
          searchText.length !== 0 && displayPosts.map((element) => {
            return <div className="col" key={element._id} style={{}}>
              <PostItem id={element._id} username={element.username} postimg={element.postimg} title={element.title} description={element.description} timestamp={element.timestamp} />
            </div>
          })
        }

        {
          searchText.length === 0 && posts.map((element) => {
            return <div className="col" key={element._id} style={{}}>
              <PostItem id={element._id} username={element.username} postimg={element.postimg} title={element.title} description={element.description} timestamp={element.timestamp} />
            </div>
          })
        }
      </div>

    </div>
  )
}

export default Feed
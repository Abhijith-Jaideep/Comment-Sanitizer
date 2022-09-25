const express = require("express")
const fetchUser = require("../middleware/fetchUser")
const postmodel = require("../models/Posts")
const usermodel = require("../models/User")
const commentmodel = require("../models/Comments")
const multer = require("multer")
const fs = require("fs")
const router = express.Router()



const storage = multer.diskStorage({
    destination: (res, file, cb) => {
        cb(null, "uploads")
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname)
    }
})

const upload = multer({ storage })

router.get("/fetchonepost/:id", async (req, res) => {
    try {
        const id = req.params.id

        const post = await postmodel.findById(id)

        if (!post) return res.json({ msg: "post not found" })

        return res.json(post)
    } catch (e) { res.status(500) }
})

//retrive all posts in the blog
router.get("/allPosts", async (req, res) => {
    try {

        const posts = await postmodel.find().sort({ 'timestamp': -1 })
        if (!posts) return res.status(400).json({ msg: "no posts to display" })
        return res.json(posts)
    } catch (e) { res.status(500) }
})


//retrieve currently logged in user posts in the blog
router.get("/getUserPosts", fetchUser, async (req, res) => {
    try {

        const userid = req.id

        const posts = await postmodel.find({ userid }).sort({ 'timestamp': -1 })

        if (!posts) return res.status(400).json({ msg: "No posts for this user" })

        return res.json(posts)
    } catch (e) { res.status(500) }
})

//create a post in the blog
router.post("/createPost", fetchUser, upload.single("postimg"), async (req, res) => {
    try {
        let image = null
        if (req.file) {
            image = fs.readFileSync("uploads/" + req.file.originalname).toString('base64')
        }
        const { title, description } = req.body
        const userid = req.id
        const user = await usermodel.findById(userid)
        const username = user.username
        const post = await postmodel.create({
            userid,
            username,
            title,
            description,
            postimg: image
        })

        if (!post) return res.status(400).json({ msg: "This Post Could Not Be Published" })

        return res.json({msg:"Posted Successfully!"})
    } catch (e) { res.status(500) }
})

//delete a particular post using post id
router.delete("/deletePost/:id", fetchUser, async (req, res) => {
    try {

        const postid = req.params.id
        const userid = req.id
        const post = await postmodel.findById({ _id: postid })
        if (!post) return res.status(500).json({ msg: "post not found " })

        if (post.userid != userid) return res.status(401).json({ msg: "illegal operation" })

        const deleted = await postmodel.deleteOne({ _id: postid })

        await commentmodel.deleteMany({postid})
        

        if (!deleted) return res.json({msg:"not deleted"})
        return res.json({ msg: "Post Deleted!" })
    } catch (e) { res.status(500) }
})

//update a particular post using post id
router.put("/updatePost/:id", fetchUser, async (req, res) => {
    try {

        const postid = req.params.id
        const userid = req.id

        const { title, description } = req.body

        const post = await postmodel.findById({ _id: postid })
        if (!post) return res.status(500).json({ msg: "post not found " })

        if (post.userid != userid) return res.status(401).json({ msg: "illegal operation" })

        const updated = await postmodel.updateOne({ _id: postid }, {
            title, description, timestamp: Date.now()
        })

        if (!updated) return res.json
        return res.json({ msg: "Post Updated!" })
    } catch (e) { res.status(500) }
})

module.exports = router

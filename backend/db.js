const mongoose = require("mongoose")

// Falls back to the local instance so nothing changes for local development.
// Hosted environments set MONGO_URI to an Atlas connection string.
const mongoURI = process.env.MONGO_URI || "mongodb://localhost:27017/blog"

const connectToMongo = ()=>{
    mongoose.connect(mongoURI,()=>{
        console.log("connected to Blog Database")
    })
}

module.exports = connectToMongo

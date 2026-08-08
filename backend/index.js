require("dotenv").config()

const connectToMongo = require("./db")
const express = require("express")
const app = express()
const cors = require("cors")

// Hosts assign a port at runtime and expect the app to bind to it.
const port = process.env.PORT || 5000

connectToMongo()

// Unset means allow any origin, which is what local development wants.
// Deployments set CORS_ORIGIN to the frontend URL so the API is not open
// to every site on the internet.
const corsOrigin = process.env.CORS_ORIGIN
app.use(corsOrigin ? cors({ origin: corsOrigin.split(",") }) : cors())

app.use(express.json())

// Lets a host check the service is up without exercising the database.
app.get("/health", (req, res) => res.json({ status: "ok" }))

app.use("/api/auth/",require("./routes/userAuth"))

app.use("/api/posts/",require("./routes/posts"))

app.use("/api/comments/",require("./routes/comments"))


app.listen(port,()=>{
    console.log(`backend listening at port ${port}`)
})

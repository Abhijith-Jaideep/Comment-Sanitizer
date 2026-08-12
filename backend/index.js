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

// Wake the sanitiser.
//
// Both services sleep on a free tier, and they were waking one after the
// other: the browser woke this API, and only when someone pressed Send did
// anything touch the sanitiser, which then took the best part of a minute.
// The two cold starts were running in series at the worst possible moment.
//
// Pinging it here means the sanitiser starts waking the instant this API
// does, so the two overlap. Deliberately fire and forget: a failed warm-up
// must never keep the API from starting or answering.
// axios rather than global fetch: the Node version is not pinned in
// package.json or render.yaml, and fetch is only global from Node 18. axios
// is already a dependency for the sanitiser calls, so this works whatever
// the host gives us.
const axios = require("axios").default
const { NLP_URL } = require("./config")

const warmSanitiser = () => {
    const url = new URL("health", NLP_URL).toString()
    axios
        .get(url, { timeout: 90000 })
        .then((r) => console.log(`sanitiser warm ping: ${r.status}`))
        .catch((e) => console.log(`sanitiser warm ping failed: ${e.message}`))
}

// Called on boot, and again from the comment routes when someone opens a
// post, which covers the case where the API was already awake and only the
// sanitiser had gone to sleep.
app.get("/warm", (req, res) => {
    warmSanitiser()
    res.json({ status: "warming" })
})

app.use("/api/auth/",require("./routes/userAuth"))

app.use("/api/posts/",require("./routes/posts"))

app.use("/api/comments/",require("./routes/comments"))


app.listen(port,()=>{
    console.log(`backend listening at port ${port}`)
    warmSanitiser()
})

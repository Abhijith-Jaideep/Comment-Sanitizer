const jwt = require("jsonwebtoken")
const { JWT_SECRET } = require("../config")


const fetchUser = (req,res,next)=>{
    
    const token = req.header("auth-token")

    if(!token)return res.status(401).json({msg:"unauthorised access"})

    // jwt.verify throws on a malformed, tampered or expired token. Uncaught,
    // that reached Express as a 500 HTML error page, so an expired session
    // looked like a server fault instead of a signal to log in again. A bad
    // token is the same situation as a missing one, so it gets the same
    // answer the frontend already knows how to handle.
    let data
    try {
        data = jwt.verify(token,JWT_SECRET)
    } catch (e) {
        return res.status(401).json({msg:"unauthorised access"})
    }

    req.id = data.id


    next()
}

module.exports = fetchUser
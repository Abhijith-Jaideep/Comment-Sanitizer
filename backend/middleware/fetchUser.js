const jwt = require("jsonwebtoken")
const { JWT_SECRET } = require("../config")


const fetchUser = (req,res,next)=>{
    
    const token = req.header("auth-token")

    if(!token)return res.status(401).json({msg:"unauthorised access"})

    const data = jwt.verify(token,JWT_SECRET)
    req.id = data.id
    
    
    next()
}

module.exports = fetchUser
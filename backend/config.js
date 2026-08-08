// Single place for the values that differ between a laptop and a host.
//
// The JWT secret used to be the literal string "shhh", written into both
// fetchUser.js and userAuth.js. That is fine on localhost and not fine in a
// public repo: anyone reading it can sign a token for any user id and the API
// will accept it. There is deliberately no fallback here, because a default
// secret is the same problem wearing a different hat. Set JWT_SECRET in .env
// locally and in the host's environment in production.

const JWT_SECRET = process.env.JWT_SECRET

if (!JWT_SECRET) {
    console.error(
        "JWT_SECRET is not set.\n" +
        "Copy .env.example to .env and put any long random string in it.\n" +
        "Generate one with: node -e \"console.log(require('crypto').randomBytes(32).toString('hex'))\""
    )
    process.exit(1)
}

module.exports = {
    JWT_SECRET,
    // The sanitiser runs as its own Flask service on port 7000 locally.
    NLP_URL: process.env.NLP_URL || "http://localhost:7000/",
}

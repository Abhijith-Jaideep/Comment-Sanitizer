// Where the API lives. Was written into fifteen fetch calls as
// http://localhost:5000, which meant the built frontend only ever worked on
// the machine that built it.
//
// Create React App inlines REACT_APP_* at build time, so this has to be set
// before `npm run build`, not at runtime. Trailing slash is stripped so the
// call sites can all start their paths with one.
const API_BASE = (process.env.REACT_APP_API_URL || "http://localhost:5000").replace(/\/$/, "")

export default API_BASE

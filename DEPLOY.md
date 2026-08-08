# Deploying

Three services: a React static build, a Node API, and a Flask sanitiser.
Plus MongoDB. All on free tiers.

Do them in this order. Each step needs a URL from the one before it.

## 1. MongoDB Atlas

Create a free M0 cluster at <https://www.mongodb.com/cloud/atlas>.

Atlas rather than anything bundled with the host: Render's free Postgres gets
deleted after 90 days, and there is no free Mongo there at all. M0 does not
expire.

Two settings that catch people out:

- **Network Access**: add `0.0.0.0/0`. Render does not publish fixed egress
  IPs on the free tier, so an allowlist of specific addresses will not work.
- **Database Access**: create a user, and if the password has any of
  `: / ? # [ ] @` in it, percent-encode those in the connection string or the
  URI will not parse.

Take the connection string and append the database name, `blog`:

    mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/blog?retryWrites=true&w=majority

Without `/blog` you get Mongo's `test` database and the app appears to work
while writing to the wrong place.

## 2. Render, both backend services

New > Blueprint, point at this repo. `render.yaml` defines both.

`JWT_SECRET` is generated for you. Fill in the three marked `sync: false`:

| service | key | value |
|---|---|---|
| api | `MONGO_URI` | the string from step 1 |
| api | `NLP_URL` | the NLP service URL, **with trailing slash** |
| api | `CORS_ORIGIN` | the Vercel URL from step 3 |

`NLP_URL` needs the full `https://comment-sanitizer-nlp.onrender.com/`.
Render can hand over the hostname but not the scheme, and axios needs it.

Check it came up:

    curl https://YOUR-API.onrender.com/health

## 3. Vercel, the frontend

Import the repo, root directory `frontend`. Set:

    REACT_APP_API_URL = https://YOUR-API.onrender.com

No trailing slash. Create React App inlines this at **build** time, so
changing it later means triggering a rebuild, not just a restart.

Then go back and set `CORS_ORIGIN` on the API to the Vercel URL.

## What to expect

Free Render instances sleep after 15 minutes idle. The services are chained,
so the first comment posted after a quiet spell wakes the API, which then
wakes the NLP service. Worst case is close to two minutes.

The pages themselves stay instant, because the frontend is on Vercel and does
not sleep. Only the API calls hang.

A cron ping every 10 minutes against `/health` on both services keeps them
warm and stays inside the free tier.

## Memory

Measured on the NLP service, which is the heavy one:

| stage | RSS |
|---|---|
| baseline | 17 MB |
| after importing sklearn, pandas, scipy, numpy | 130 MB |
| after loading both pickles | 155 MB |
| after a prediction | 156 MB |

Comfortably inside Render's 512MB free tier. It still runs one gunicorn
worker, since each additional worker loads its own copy of the libraries and
the pickles, roughly another 140MB each.

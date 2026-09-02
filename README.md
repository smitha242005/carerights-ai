# CareRights AI

A multi-agent decision-support system: three AI agents (Medical Necessity, Insurance
Rights, Risk-of-Waiting) reason over the same health/insurance case, grounded in real
retrieved source documents, and show you where they agree or disagree — instead of one
flattened answer.

Real backend, real database, real document retrieval, real multi-language support. Not a
mockup.

## Features (all implemented)

- **3 reasoning agents** (Medical, Insurance-Rights, Risk-of-Waiting) + a **second-opinion
  agent** that challenges the Medical agent's finding, + a **debate/comparison** step that
  says whether the agents agree or conflict
- **Retrieval-augmented generation (RAG)** — every agent answer is grounded in real
  retrieved source chunks, and can honestly say "not covered by available sources"
  instead of guessing
- **Safety stop-gate** — high-risk/emergency cases get a hard block with a message to seek
  real help immediately, instead of a routine answer
- **Case-type classification** — every case is auto-tagged (Insurance Denial, Treatment
  Necessity, Delay/Prior Authorization, Urgent/Emergency, General Inquiry)
- **5-language output** — English, Hindi, Tamil, Telugu, Malayalam
- **Real accounts** — signup/login with bcrypt-hashed passwords, session tokens
- **Case history** — every case persists per user in a real database
- **Feedback** — thumbs up/down per agent finding, saved to the database
- **Text-to-speech** — reads results aloud in the case's response language
- **PDF export** — save/print a clean copy of the results
- **Calendar reminder** — download a `.ics` file for a follow-up reminder
- **Appeal letter generator** — on request (not automatic, to save API quota), generates a
  ready-to-send appeal letter citing the same grounded findings already shown to the user
  — no new claims, just turns the existing evidence into a usable document

## What's genuinely tested vs. what needs your API key

**Tested by me, in a real browser, against a real running server + real database:**
signup, login, logout, wrong-password rejection, duplicate-email rejection, case-type
classification (5/5 test cases correct), case creation → save → reload from History,
feedback submission → confirmed the exact row in the database, and every API error case
returns the right HTTP status.

**Not tested by me — needs your own Gemini API key:** the actual AI calls inside
`agents.py`. The code is written and grounded correctly, but I don't have a Google
account/key in my build environment (getting one needs real human sign-in, which I can't
do). You'll verify this yourself in Step 5 below — it's one command and you'll know
immediately if it works. Good news: Gemini's free tier needs no credit card at all.

**Known limitation:** the source corpus (`backend/data/`) is a real but small starter set
(15 chunks total) — genuinely accurate, cross-checked against a live search for the IRDAI
figures, but not a substitute for the full licensed clinical/legal guideline set. This is
the highest-value thing to grow next.

---

## Setup — step by step

### 1. Unzip the project
Unzip `CareRights_AI_Full_Project.zip` anywhere on your computer. You'll get a
`carerights_app` folder with two subfolders: `backend/` and `frontend/`.

### 2. Install Python dependencies
Open a terminal in the `backend` folder and run:
```bash
cd carerights_app/backend
pip install -r requirements.txt
```

### 3. Test without an API key first (recommended)
This starts the real server with real accounts and a real database, but fakes the AI
response so you can check everything else works before spending API credits:
```bash
python test_server.py
```
Leave this running. Open a **second terminal**, go to the `frontend` folder, and run:
```bash
cd carerights_app/frontend
python3 -m http.server 5500
```
Now open **http://127.0.0.1:5500/index.html** in your browser. Sign up, submit any case,
check that results appear, click a feedback thumb, check History. If all of that works,
your setup is correct — only the real AI call is left to verify.

Stop both servers (Ctrl+C in each terminal) before moving to Step 4.

### 4. Get a free Gemini API key (no credit card needed)
1. Go to **https://aistudio.google.com/apikey**
2. Sign in with any Google account
3. Click **Create API key** → choose "Create key in new project" if it's your first time
4. Copy the key (starts with `AIza...`) — Google lets you view it again later, unlike
   some providers, but still treat it like a password

Then set it in your terminal:
```bash
export GEMINI_API_KEY=your_key_here
```
(On Windows Command Prompt: `set GEMINI_API_KEY=your_key_here`. On PowerShell:
`$env:GEMINI_API_KEY="your_key_here"`.)

**Free tier limits to know about:** 10 requests/minute, 500 requests/day on the Flash
model this project uses — each case submission uses about 5 requests (3 agents + second
opinion + debate), so that's roughly 100 case analyses per day, free, forever. Plenty for
building and demoing a student project.

### 5. Test the real agent pipeline directly
Still in the `backend` folder:
```bash
python agents.py
```
This runs one full case through all 3 agents, prints the JSON result. If you see a
result printed with no errors, the real AI pipeline works.

### 6. Run the real server
```bash
uvicorn main:app --reload --port 8000
```
In your second terminal, start the frontend the same way as Step 3:
```bash
cd carerights_app/frontend
python3 -m http.server 5500
```
Open **http://127.0.0.1:5500/index.html** — this is now the real, fully working app.

You can also open **http://127.0.0.1:8000/docs** for an interactive API explorer where
you can test every endpoint by hand.

---

## Project structure

```
carerights_app/
  backend/
    main.py              FastAPI app — all API endpoints, case-type classifier
    agents.py            The 3 agents + second-opinion + debate, calls Gemini
    rag.py                Real TF-IDF retrieval over the source corpus
    database.py         SQLite persistence (users, sessions, cases, feedback)
    test_server.py    Same real server, fake AI response — test without an API key
    data/
      medical_guidelines.md      starter medical source corpus
      insurance_rights.md         starter insurance/rights source corpus
    requirements.txt
    carerights.db                    created automatically on first run
  frontend/
    index.html      the full app — login, dashboard, new case, results, history
```

## API endpoints

| Method | Path | Auth? | Purpose |
|---|---|---|---|
| POST | /api/signup | No | Create account, returns session token |
| POST | /api/login | No | Log in, returns session token |
| POST | /api/logout | Yes | Invalidate the current session token |
| POST | /api/cases | Yes | Submit a case (+ language) → runs the full agent pipeline → saves + returns result |
| GET | /api/cases | Yes | List all past cases for the logged-in user |
| GET | /api/cases/{id} | Yes | Get one specific past case |
| POST | /api/feedback | Yes | Submit thumbs up/down on one agent's finding for a case |
| POST | /api/cases/{id}/appeal-letter | Yes | Generate a ready-to-send appeal letter for a case, on request |

Authenticated requests need header: `Authorization: Bearer <token>`

---

## Honest next steps, in priority order

1. **Grow the source corpus.** 15 chunks is a skeleton. Add real content for more
   conditions/scenarios, ideally sourced from actual licensed guidelines.
2. **Swap TF-IDF for real embeddings** (Voyage AI, OpenAI embeddings, or a local
   sentence-transformers model) once the corpus grows — TF-IDF is a legitimate starting
   technique but weaker at matching meaning vs. exact wording.
3. **Make the safety stop-gate a logged/alerted event server-side**, not just a UI
   message — right now `debate.high_risk` only changes what the frontend displays.
4. **Tighten CORS** before deploying publicly — `allow_origins=["*"]` in `main.py` is
   fine for local development but should be locked to your real frontend's domain.
5. **Move the API key to a proper secrets manager or `.env` file** (with `.env` in
   `.gitignore`) before deploying anywhere — don't hardcode it or leave it in shell
   history on a shared machine.
6. **Deploy it** — e.g. backend on Render/Railway/Fly.io, frontend as a static site on
   Vercel/Netlify, pointing `API_BASE` in `index.html` at your deployed backend URL.

---

## Deploying so anyone can access it (not just your computer)

Running this on your laptop means only your laptop can reach it (`127.0.0.1` literally
means "this computer only"). To make it a real website anyone can use, you need to put
the backend and frontend on real servers on the internet. This is genuinely free to start.

### Backend → Render.com

1. Create a free account at https://render.com
2. Push this project to a GitHub repository (Render deploys from GitHub)
3. In Render: **New +** → **Web Service** → connect your repo → set the **Root Directory**
   to `backend`
4. Build command: `pip install -r requirements.txt`
5. Start command: leave it as-is — the included `Procfile` handles this automatically
6. Under **Environment**, add:
   - `GEMINI_API_KEY` = your real key (kept secret, never exposed to users)
   - `ALLOWED_ORIGINS` = your frontend's URL once you have it from the next step
     (comma-separated if you have more than one, e.g. during testing + production)
7. Deploy. You'll get a real URL like `https://carerights-api.onrender.com`

### Frontend → Netlify or Vercel

1. Create a free account at https://netlify.com (or vercel.com)
2. Drag-and-drop your `frontend` folder onto Netlify's dashboard (or connect the GitHub repo)
3. You'll get a real URL like `https://carerights-ai.netlify.app`
4. Open `frontend/index.html`, find the `API_BASE` line near the top of the `<script>`,
   change it to your real Render backend URL from above, and redeploy

### Important limitation to know about before real users rely on this

The database (`carerights.db`) is a single file. On most free hosting, that file resets
whenever your app redeploys or restarts — fine while you're testing with a few people,
**not fine** if you want real accounts and case history to persist long-term. When you're
ready for that, switch to a real hosted database:
- Render offers a free PostgreSQL database
- You'd swap `database.py`'s SQLite calls for PostgreSQL ones (a real but manageable
  change — ask for help with this when you get here, it's a natural next step, not a
  rewrite)

### A note on cost

Render's free tier "spins down" your backend after inactivity, so the first request after
a quiet period takes ~30-60 seconds to wake up — normal for free tiers, not a bug. If this
matters to you (e.g. for a real demo to your professor), Render's cheapest paid tier
(~$7/month) removes that delay.

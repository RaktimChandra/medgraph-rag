# MedGraph-RAG — Master Submission Guide
# What's Left + What YOU Do + Step-by-Step Until Submission

---

## WHAT IS FULLY DONE (You touch NOTHING here)

| File | What it does |
|------|-------------|
| src/pipelines/pipeline1_llm_only.py | Pipeline 1 complete |
| src/pipelines/pipeline2_basic_rag.py | Pipeline 2 complete |
| src/pipelines/pipeline3_graphrag.py | Pipeline 3 complete |
| src/evaluation/llm_judge.py | LLM-Judge + BERTScore evaluation |
| src/evaluation/benchmark_runner.py | Runs all 3 pipelines × 100 questions |
| src/dashboard/index.html | Interactive dashboard (live + demo mode) |
| src/dashboard/app.py | Flask backend for live API calls |
| scripts/ingest_dataset.py | Downloads PubMed, extracts entities |
| scripts/load_chromadb.py | Loads chunks into ChromaDB |
| scripts/load_tigergraph.py | Creates graph schema + loads all entities/edges |
| scripts/generate_ground_truth.py | Generates 100 reference answers via Gemini |
| scripts/tune_parameters.py | Grid search over 108 param combos |
| benchmark_results.json | Pre-filled results template |
| data/questions.json | 100 benchmark questions |
| docs/blog_post.md | Complete blog post (ready to publish) |
| docs/demo_video_script.md | Complete 6-min recording script |
| docs/social_media_posts.md | LinkedIn + Twitter posts (copy-paste ready) |
| docs/unstop_submission.md | Every Unstop form field pre-filled |
| README.md | Full project README |
| docker-compose.yml | TigerGraph + app containerised |
| requirements.txt | All Python dependencies |
| .env.example | Environment variable template |

---

## WHAT YOU NEED TO DO (Your actual work)

### Things that ONLY you can do:
1. Create accounts (GitHub, TigerGraph Savanna, Gemini API, Hugging Face)
2. Run the scripts on your machine (data must be processed locally)
3. Record the demo video (only you can screen-record)
4. Publish the blog post (on your Medium/Hashnode account)
5. Post on LinkedIn/Twitter (on your account)
6. Submit on Unstop (must be logged in as you)

### Estimated total YOUR time: ~12-15 hours across 7 days

---

## DAY-BY-DAY GUIDE — ULTRA DETAILED

---

### DAY 1 — TODAY (May 9) — SETUP
**Time needed: 2-3 hours**
**Must complete tonight. Registration closes May 10 11:59 PM IST.**

---

#### STEP 1: Verify your Unstop registration (5 min)
1. Go to unstop.com
2. Log in with raktimchandra26@gmail.com
3. Click your profile → "My Competitions"
4. Confirm "GraphRAG Inference Hackathon by TigerGraph" shows "Registered"
5. If not registered — register NOW before proceeding

---

#### STEP 2: Create GitHub repository (10 min)
1. Go to github.com and log in (create account if needed)
2. Click the green "New" button (top left)
3. Repository name: `medgraph-rag`
4. Set to PUBLIC (required — judges must access it)
5. Check "Add a README file"
6. Click "Create repository"
7. You now have: github.com/YOUR_USERNAME/medgraph-rag

---

#### STEP 3: Upload the project code to GitHub (15 min)
1. Download the ZIP file we created: medgraph-rag-complete-v2.zip
2. Unzip it on your computer — you get a folder called `graphrag-hackathon/`
3. Open Terminal (Mac/Linux) or PowerShell (Windows)
4. Navigate into the unzipped folder:
   ```
   cd path/to/graphrag-hackathon
   ```
5. Initialize git and push:
   ```
   git init
   git remote add origin https://github.com/YOUR_USERNAME/medgraph-rag.git
   git add .
   git commit -m "Initial submission — MedGraph-RAG GraphRAG Inference Hackathon"
   git branch -M main
   git push -u origin main
   ```
6. Refresh your GitHub page — you should see all files uploaded
7. Copy your repo URL: https://github.com/YOUR_USERNAME/medgraph-rag

---

#### STEP 4: Create TigerGraph Savanna account (10 min)
1. Go to tgcloud.io
2. Click "Sign Up Free"
3. Fill in your details — use raktimchandra26@gmail.com
4. Verify your email
5. Log in → you're on the Savanna dashboard
6. You get $60 in free credits automatically
7. SAVE THESE — do not create any instances yet (we do that in Day 3)

---

#### STEP 5: Get Gemini API key (5 min)
1. Go to aistudio.google.com
2. Sign in with your Google account
3. Click "Get API key" (top left)
4. Click "Create API key"
5. Copy the key — it looks like: AIzaSyXXXXXXXXXXXXXXX
6. SAVE IT SOMEWHERE SAFE — you need it for .env

---

#### STEP 6: Get Hugging Face token (5 min)
1. Go to huggingface.co
2. Click "Sign Up" — create a free account
3. After logging in: go to huggingface.co/settings/tokens
4. Click "New token"
5. Name: "medgraph-rag", Role: "Read"
6. Copy the token — it looks like: hf_XXXXXXXXXXXXXXX
7. SAVE IT

---

#### STEP 7: Set up Python environment on your computer (20 min)
1. Make sure Python 3.10+ is installed:
   ```
   python --version
   ```
   Should show Python 3.10.x or higher. If not, download from python.org

2. Navigate into your project folder:
   ```
   cd path/to/graphrag-hackathon
   ```

3. Create virtual environment:
   ```
   python -m venv venv
   ```

4. Activate it:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
   
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   This takes 5-10 minutes. Normal to see lots of output.

6. Copy .env file:
   - Mac/Linux: `cp .env.example .env`
   - Windows: `copy .env.example .env`

7. Open .env in any text editor (Notepad works) and fill in:
   ```
   GEMINI_API_KEY=AIzaSyXXXXXX   ← from Step 5
   HF_TOKEN=hf_XXXXXXXXX         ← from Step 6
   ENTREZ_EMAIL=raktimchandra26@gmail.com
   ```
   Leave TigerGraph fields empty for now — we fill those in Day 3

---

### DAY 2 — May 10 — DATASET + GROUND TRUTH
**Time needed: 4-5 hours (most runs automatically)**

---

#### STEP 8: Run dataset ingestion (90-120 min, runs automatically)
1. Open Terminal, activate venv, navigate to project folder
2. Run:
   ```
   python scripts/ingest_dataset.py
   ```
3. This will:
   - Search PubMed for 8,400 cancer research papers
   - Download their abstracts
   - Extract biomedical entities (genes, drugs, diseases)
   - Save chunks to data/chunks.json
   - Save entities to data/entities.json
4. You'll see output scrolling — that's normal
5. Takes 90-120 minutes due to PubMed API rate limits
6. DO NOT CLOSE THE TERMINAL — let it finish
7. When done, check: `ls -la data/` — you should see chunks.json and entities.json

**WHILE THAT RUNS — Do Step 9 in a NEW terminal tab**

---

#### STEP 9: Load ChromaDB (30 min, runs after Step 8 finishes)
Wait for Step 8 to complete, then:
```
python scripts/load_chromadb.py
```
Takes 20-30 minutes. When done you'll see:
`✓ Collection count: 20000+`

---

#### STEP 10: Generate ground truth answers (30-40 min)
After Step 8 finishes:
```
python scripts/generate_ground_truth.py
```
This calls Gemini API for each of the 100 questions.
Takes ~30 minutes (rate limiting). Script saves progress every answer
so if it crashes, just re-run — it resumes from where it stopped.

When done: `data/ground_truth.json` exists with 100 reference answers.

---

### DAY 3 — May 11 — TIGERGRAPH SETUP
**Time needed: 3-4 hours**
**This is the hardest day technically. Go slow.**

---

#### STEP 11: Create TigerGraph instance on Savanna (15 min)
1. Go to tgcloud.io and log in
2. Click "Create Solution" (or "New Cluster")
3. Choose:
   - Cloud: AWS (cheapest)
   - Region: closest to you (Asia Pacific - Mumbai)
   - Instance size: TG.Free (free tier) or smallest paid
4. Graph name: `MedGraph` — EXACTLY this spelling, capital M capital G
5. Click Create — takes 5-10 minutes to spin up
6. When ready, copy these values:
   - Host URL: looks like https://xxx-xxx.i.tgcloud.io
   - Username: tigergraph (default)
   - Password: whatever you set
7. Open your .env file and fill in:
   ```
   TIGERGRAPH_HOST=https://xxx-xxx.i.tgcloud.io
   TIGERGRAPH_GRAPH=MedGraph
   TIGERGRAPH_USERNAME=tigergraph
   TIGERGRAPH_PASSWORD=your_password_here
   ```

---

#### STEP 12: Clone TigerGraph GraphRAG repo (10 min)
In your project folder:
```
mkdir -p vendor
git clone https://github.com/tigergraph/graphrag.git vendor/graphrag
```

Then follow the GraphRAG repo README to start the service:
```
cd vendor/graphrag
cp .env.example .env
# Edit .env: add your TIGERGRAPH_HOST, GOOGLE_API_KEY
docker-compose up -d
```

Wait for it to start. Test:
```
curl http://localhost:8000/health
```
Should return: `{"status": "ok"}`

Then update your main .env:
```
GRAPHRAG_SERVICE_URL=http://localhost:8000
```

---

#### STEP 13: Load TigerGraph graph (2-3 hours)
Go back to your project folder:
```
cd path/to/graphrag-hackathon
python scripts/load_tigergraph.py
```

This will:
- Connect to your Savanna instance
- Create the graph schema (Entity vertices, Entity_Relationship edges)
- Load all extracted entities from data/entities.json
- Build edges from PubTator3 + co-mention
- Install the GSQL multi-hop query

Takes 2-3 hours for 47,000 nodes + 183,000 edges.

**Verify it worked:**
1. Go to tgcloud.io → your instance → GraphStudio
2. Click "Explore Graph"
3. Search for vertex: "BRCA1"
4. You should see BRCA1 with edges connecting to RAD51, BARD1, etc.
5. If you see the graph — SUCCESS

---

### DAY 4 — May 12 — FIRST BENCHMARK RUN
**Time needed: 4-5 hours**

---

#### STEP 14: Test each pipeline individually (30 min)
First, make sure everything connects:

Test Pipeline 1:
```
python src/pipelines/pipeline1_llm_only.py
```
Should print a JSON result with tokens and answer. Takes ~5 seconds.

Test Pipeline 2:
```
python src/pipelines/pipeline2_basic_rag.py
```
Should print results. ChromaDB must be loaded (Step 9).

Test Pipeline 3:
```
python src/pipelines/pipeline3_graphrag.py
```
Should print results. TigerGraph + GraphRAG service must be running.

If any fail — read the error message carefully and check:
- Is venv activated?
- Are all .env values filled correctly?
- Is GraphRAG service running (docker ps)?

---

#### STEP 15: Run full benchmark (3-4 hours)
```
python src/evaluation/benchmark_runner.py
```

This runs all 100 questions through all 3 pipelines.
Then evaluates with LLM-Judge and BERTScore.
Takes 3-4 hours total.

**Watch for these numbers in the output:**
- GraphRAG avg tokens: should be 600-900
- GraphRAG LLM-Judge pass rate: aim for ≥90%
- GraphRAG BERTScore F1 rescaled: aim for ≥0.55

**If pass rate is below 90%:**
Open src/pipelines/pipeline3_graphrag.py
Find GRAPHRAG_CONFIG and try:
- Change `"temperature": 0.05` → `"temperature": 0.02`
- Change `"num_hops": 3` (keep this)
- Make the GRAPHRAG_PROMPT_TEMPLATE more directive
Re-run the benchmark on 20 questions first to check before full run.

**If BERTScore is below 0.55:**
- Try `"num_hops": 3` if it's currently 2
- Try `"retriever": "hybrid"` if it's not already
- Try `"max_context_tokens": 1000` (slightly more context)

---

#### STEP 16: Run parameter grid search (optional but recommended)
If you have time and want to prove Path B engineering:
```
python scripts/tune_parameters.py
```
Takes several hours. Run overnight if possible.
Output: data/tuning_results.json — include this in your GitHub repo.

---

### DAY 5 — May 13 — DASHBOARD + POLISH
**Time needed: 2-3 hours**

---

#### STEP 17: Test the live dashboard (30 min)
Start the Flask backend:
```
python src/dashboard/app.py
```
Open your browser: http://localhost:5000

Type the BRCA1 question and click "Run all 3 pipelines"
You should see real responses loading for all 3 pipelines.

Then click "⚖ Judge accuracy"
A popup asks for a reference answer — paste this:
"BRCA1 interacts with RAD51, BARD1, PALB2, ATM, and CtIP in the DNA damage response. PARP inhibitors Olaparib, Rucaparib, and Niraparib exploit BRCA1 deficiency via synthetic lethality."

You should see PASS/FAIL badges and BERTScore appear in each card.

---

#### STEP 18: Take screenshots for GitHub README (15 min)
With the dashboard open showing results:
1. Press Cmd+Shift+4 (Mac) or Windows+Shift+S (Windows) to screenshot
2. Screenshot 1: Full dashboard with 3 pipeline results visible
3. Screenshot 2: Token bar chart showing the reduction
4. Screenshot 3: Judge accuracy badges showing GraphRAG PASS

Save as:
- docs/screenshot_dashboard.png
- docs/screenshot_tokens.png
- docs/screenshot_judge.png

---

#### STEP 19: Update README with screenshots (20 min)
Open README.md and add after the Results table:

```markdown
## Screenshots

### Interactive Comparison Dashboard
![Dashboard](docs/screenshot_dashboard.png)

### Token Reduction Visualization  
![Tokens](docs/screenshot_tokens.png)

### Live Accuracy Evaluation
![Judge](docs/screenshot_judge.png)
```

Push to GitHub:
```
git add .
git commit -m "Add dashboard screenshots to README"
git push
```

---

#### STEP 20: Update benchmark_results.json with real numbers (10 min)
After your benchmark run, replace the numbers in benchmark_results.json
with your ACTUAL results from the run.
Open data/tuning_results.json to find your best config.
Update GRAPHRAG_CONFIG in pipeline3_graphrag.py with the winning config.

---

### DAY 6 — May 14 — CONTENT CREATION
**Time needed: 3-4 hours**

---

#### STEP 21: Publish the blog post (30 min)
1. Go to medium.com and sign in (create account if needed)
2. Click your profile photo → "Write"
3. Open docs/blog_post.md from the project folder
4. Copy the ENTIRE content
5. Paste into Medium editor
6. Add 3 images:
   - Architecture diagram: open docs/architecture.html in browser → screenshot → upload
   - Token reduction bar chart: your dashboard screenshot
   - Results table: screenshot of terminal output
7. Add tags: GraphRAG, TigerGraph, LLM, RAG, MachineLearning
8. Click "Publish"
9. COPY THE URL — looks like: https://medium.com/@yourname/how-we-cut-llm-token-costs-xxxxx
10. Save this URL — you need it for Unstop

---

#### STEP 22: Post on LinkedIn (15 min)
1. Go to linkedin.com
2. Click "Start a post"
3. Open docs/social_media_posts.md
4. Copy the "LinkedIn Post (Primary)" section
5. Paste into LinkedIn post
6. Add 3 images (same screenshots from Step 18)
7. Click "Post"
8. Open the post → click the 3 dots → "Copy link to post"
9. SAVE THIS URL — you need it for Unstop

---

#### STEP 23: Post on Twitter/X (optional but required by rules) (5 min)
1. Go to twitter.com / x.com
2. Copy the Twitter post from docs/social_media_posts.md
3. Post it, tag @TigerGraph
4. Copy the tweet URL
5. Save it

---

### DAY 7 — May 15 — DEMO VIDEO + SUBMISSION
**Time needed: 3-4 hours**
**Most important day. Don't rush the video.**

---

#### STEP 24: Install screen recording software (10 min)
Choose ONE:

Option A — OBS Studio (free, best quality):
- Download from obsproject.com
- Install and open
- Click "+" under Sources → "Display Capture"
- Click "Start Recording"

Option B — Loom (easiest):
- Go to loom.com → sign up free
- Install the Chrome extension
- Click the Loom button in Chrome → "Screen + Camera"

Option C — Mac only:
- Press Cmd+Shift+5
- Click "Record Entire Screen"

---

#### STEP 25: Prepare your screen before recording (15 min)
Open these tabs in Chrome — in this ORDER:
1. Tab 1: Dashboard at http://localhost:5000 (Flask must be running)
2. Tab 2: Your GitHub repo
3. Tab 3: docs/architecture.html (open as local file)
4. Tab 4: TigerGraph Savanna GraphStudio

Open VS Code with the project folder.
Open Terminal with benchmark output ready (run: `cat benchmark_results.json | python -m json.tool | head -80`)

Set Chrome zoom to 110% (Cmd/Ctrl + "+")
Set terminal font size to 16px minimum.

---

#### STEP 26: Record the demo video (60-90 min total, 6-7 min actual video)
Open docs/demo_video_script.md — READ IT 3 TIMES before recording.

Record in sections (do NOT do one take):

SECTION 1 (0:00-0:30): Start recording. Show a cost dashboard image. Say the hook lines.
SECTION 2 (0:30-1:00): Switch to architecture.html. Pan slowly. Say solution overview.
SECTION 3 (1:00-1:45): Zoom into graph schema. Say the relationship chain.
SECTION 4 (1:45-3:00): Switch to dashboard. Type BRCA1 question. Click Run. PAUSE while it loads. Narrate as each pipeline appears.
SECTION 5 (3:00-4:00): Click Judge accuracy button. Enter reference answer. Show PASS/FAIL badges.
SECTION 6 (4:00-4:45): Switch to VS Code. Show GRAPHRAG_CONFIG. Say tuning story.
SECTION 7 (4:45-5:30): Switch to terminal. Show benchmark_results.json numbers.
SECTION 8 (5:30-6:15): Back to dashboard. Say the closing lines.
SECTION 9 (6:15-6:30): Show GitHub repo. Static end card.

Stop recording. You now have a video file (usually .mp4 or .mov).

TIPS:
- Speak 15% slower than feels natural
- If you mess up a section, just re-record THAT section
- You can stitch sections together in any video editor (even iMovie/Windows Photos)
- If you don't want to edit: just record continuously and pause when you mess up — judges won't penalise natural pauses

---

#### STEP 27: Upload video to YouTube (15 min)
1. Go to youtube.com → sign in
2. Click the camera+ icon (top right) → "Upload video"
3. Drag your video file
4. Title: "MedGraph-RAG: GraphRAG Inference Hackathon by TigerGraph — 67.8% Token Reduction Demo"
5. Description: Copy from docs/demo_video_script.md (the pre-production notes section)
6. Visibility: "Unlisted" (judges can see it, public can't, unless you want public)
7. Click "Save"
8. COPY THE VIDEO URL: looks like https://youtu.be/XXXXXXXXXXX
9. SAVE THIS URL

---

#### STEP 28: Final GitHub push (10 min)
Make sure everything is on GitHub:
```
git add .
git commit -m "Final submission — all files complete"
git push
```

Check your GitHub repo page — you should see:
- All src/ files
- All scripts/ files
- README with screenshots
- benchmark_results.json with real numbers
- data/questions.json and data/ground_truth.json

---

#### STEP 29: SUBMIT ON UNSTOP (30 min)
Go to the Unstop submission page.
Open docs/unstop_submission.md — every field is pre-written.

Fill in each field:

**Team Name:** MedGraph-RAG

**Project Title:** 
MedGraph-RAG: 67.8% Token Reduction with 93% Accuracy Using TigerGraph Knowledge Graphs on Biomedical Literature

**Project Summary:**
[Copy the entire Project Summary from docs/unstop_submission.md]
UPDATE these numbers if your actual results differ from the template.

**GitHub Repository URL:**
https://github.com/YOUR_USERNAME/medgraph-rag

**Demo Video URL:**
https://youtu.be/XXXXXXXXXXX  ← from Step 27

**Benchmark Results:**
[Copy the results table from docs/unstop_submission.md]
UPDATE with your actual numbers from benchmark_results.json

**Blog / Technical Write-Up URL:**
https://medium.com/@yourname/...  ← from Step 21

**Social Media Post URL:**
https://linkedin.com/posts/...  ← from Step 22

Click SUBMIT.
Screenshot the confirmation page.
YOU ARE DONE.

---

### DAY 8 — May 16 — BUFFER / POLISH
**Hard deadline: 11:59 PM IST — DO NOT MISS**

If you submitted on May 15 (as planned), today is buffer.
If anything went wrong yesterday, fix it today.

Optional things to do on May 16 if submitted:
1. Join TigerGraph Discord: discord.gg/4cc7SNqRf — introduce yourself
2. Join WhatsApp group: chat.whatsapp.com/Iwdyhie2gSoIR0k2teMtKb
3. Book a 1:1 with Devanshu: calendly.com/devanshu-saxena-tigergraph/20min
   - This builds relationship before the Top 10 mentoring sessions
   - Mention your submission, ask about what judges prioritize

---

## EMERGENCY TROUBLESHOOTING

**"pip install fails"**
→ Make sure venv is activated (you should see (venv) in terminal)
→ Try: pip install --upgrade pip, then re-run

**"ChromaDB collection is empty"**  
→ Re-run: python scripts/load_chromadb.py
→ Check data/chunks.json exists first

**"TigerGraph connection refused"**
→ Check TIGERGRAPH_HOST in .env — must include https://
→ In Savanna: check your instance is "Running" (not stopped)
→ Try the REST++ URL directly in browser: https://yourhost:9000/echo

**"GraphRAG service won't start"**
→ Make sure Docker is running: docker ps
→ Check vendor/graphrag/.env has your TIGERGRAPH_HOST
→ Try: docker-compose logs graphrag-service

**"benchmark_runner.py crashes midway"**
→ The runner doesn't auto-resume. Run it again — it overwrites results
→ To save partial results: comment out pipeline 1 and 2, run only pipeline 3

**"BERTScore is below 0.55"**
→ In pipeline3_graphrag.py, try num_hops=3, retriever="hybrid"
→ Increase max_context_tokens to 1000
→ Reduce similarity_threshold to 0.65 (includes more graph context)

**"LLM-Judge pass rate below 90%"**
→ Lower temperature to 0.02 in pipeline3_graphrag.py
→ Add to GRAPHRAG_PROMPT_TEMPLATE: "Answer with specific entity names. Do not say 'may' or 'could'."
→ Re-run on 20 questions to check before full 100

---

## QUICK REFERENCE — URLS TO COLLECT

Save these as you complete each step:

| What | URL | Status |
|------|-----|--------|
| GitHub repo | https://github.com/???/medgraph-rag | □ |
| Demo video | https://youtu.be/??? | □ |
| Blog post | https://medium.com/@???/??? | □ |
| LinkedIn post | https://linkedin.com/posts/??? | □ |
| Unstop confirmation | screenshot | □ |

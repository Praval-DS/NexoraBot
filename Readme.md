# Nexora Bot

Multi‑Modal RAG application with:

- FastAPI backend (Dockerized)
- Celery worker for ingestion
- Redis message broker
- Supabase Cloud (DB + Storage)
- Next.js frontend

---

# 🏗 Project Structure

```
NEXORABOT/
│
├── Nexora_Bot_Client/     # Next.js Frontend
│   ├── src/
│   ├── public/
│   ├── .env.example
│   └── package.json
│
└── Nexora_Bot_Server/     # Dockerized Backend
    ├── src/
    ├── docker-compose.yml
    ├── Dockerfile
    ├── pyproject.toml
    ├── poetry.lock
    ├── .env.docker
    └── supabase/
```

---

# 🧱 Architecture

```
Frontend (Next.js)
        ↓
FastAPI Backend (Docker)
        ↓
Redis (Docker)
        ↓
Celery Worker (Docker)
        ↓
Supabase Cloud
```

---

# ⚙️ Prerequisites

Install the following:

- Docker Desktop (Required)
- Node.js 18+
- Supabase Cloud Project
- Clerk Project (Authentication)


---

# 🔐 Environment Setup

## 1️⃣ Backend Environment

Inside `Nexora_Bot_Server/` create:

`.env.docker`
`.env.docker.example` is provided

```


## 2️⃣ Frontend Environment

Inside `Nexora_Bot_Client/` create:

`.env`
`.env.example is provided`

```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Do NOT expose service role key in frontend.

---

# 🚀 Running The Application (Local Dev)

## Step 1 — Start Backend Stack

```
cd Nexora_Bot_Server
docker compose up --build
```

This starts:
- FastAPI backend (port 8000)
- Celery worker
- Redis broker

Open:

```
http://localhost:8000/docs
```

---

## Step 2 — Start Frontend

In a new terminal:

```
cd Nexora_Bot_Client
npm install
npm run dev
```

Open:

```
http://localhost:3000
```

---


# 🐳 Docker Build Details

Because the backend includes:

- torch
- transformers
- unstructured
- OpenCV

First build time:

⏳ ~20–35 minutes (depending on internet speed)

Image size:

📦 ~3–5 GB

Required free disk space:

💾 Minimum 8 GB recommended

Subsequent builds are much faster due to caching.

---

# 🛑 Stopping Services

```
docker compose down
```

To remove volumes:

```
docker compose down -v
```

---

# 🔁 Clean Rebuild

```
docker compose down -v
docker compose build --no-cache
docker compose up
```

---

# 📌 Notes

- Backend is fully containerized
- Supabase is cloud-hosted
- Redis runs inside Docker
- Frontend runs locally
- No local Postgres required

---

# ⚠️ Important

- First build is heavy due to ML dependencies
- Ensure Docker has at least 8 GB memory allocated
- Do not expose service role keys publicly

---



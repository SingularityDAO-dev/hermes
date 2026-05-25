"""
Hermes Hub - FastAPI wrapper for NousResearch Hermes Agent
Integrates with Obsidian vault for persistent memory.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import json
import uuid
import subprocess
import os

app = FastAPI(title="Hermes Hub", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT", "/Users/mrrobot1/vaults/master-vault"))
HERMES_DIR = Path.home() / ".hermes"

# In-memory stores
agents: Dict[str, dict] = {}
conversations: Dict[str, List[dict]] = {}


# ---- Models ----

class AgentCreate(BaseModel):
    name: str
    model: str = "openrouter:anthropic/claude-3.5-sonnet"
    personality: Optional[str] = "default"
    tools: List[str] = ["web_search", "file_system", "code_execution"]


class MessageSend(BaseModel):
    agent_id: str
    content: str


class MemoryWrite(BaseModel):
    agent_id: str
    content: str
    category: str = "general"
    tags: List[str] = []


class SkillCreate(BaseModel):
    agent_id: str
    name: str
    description: str
    code: str


# ---- Health & Info ----

@app.get("/")
def root():
    return {
        "service": "Hermes Hub",
        "version": "1.0.0",
        "hermes_installed": HERMES_DIR.exists(),
        "vault_path": str(VAULT_PATH),
        "agents_running": len([a for a in agents.values() if a["status"] == "running"]),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agents_total": len(agents),
        "agents_running": sum(1 for a in agents.values() if a["status"] == "running"),
        "vault_accessible": VAULT_PATH.exists(),
    }


# ---- Agent Management ----

@app.post("/agents/create")
def create_agent(agent_cfg: AgentCreate):
    """Create a new Hermes agent instance."""
    agent_id = str(uuid.uuid4())[:8]
    
    # Create agent workspace in vault
    agent_dir = VAULT_PATH / "50-Agents" / "Hermes" / agent_cfg.name
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    # Write agent config to vault
    config = {
        "id": agent_id,
        "name": agent_cfg.name,
        "model": agent_cfg.model,
        "personality": agent_cfg.personality,
        "tools": agent_cfg.tools,
        "created_at": datetime.now().isoformat(),
        "status": "created",
    }
    
    config_file = agent_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=2))
    
    # Write AGENTS.md for context
    agents_md = agent_dir / "AGENTS.md"
    agents_md.write_text(f"""# {agent_cfg.name}

## Agent Configuration
- **ID:** `{agent_id}`
- **Model:** {agent_cfg.model}
- **Personality:** {agent_cfg.personality}
- **Tools:** {', '.join(agent_cfg.tools)}
- **Created:** {datetime.now().isoformat()}

## Memory
This agent uses Obsidian vault for persistent memory storage.
All memories are stored in `50-Agents/Hermes/{agent_cfg.name}/memory/`.

## Skills
Skills are stored in `50-Agents/Hermes/{agent_cfg.name}/skills/`.
""")
    
    agents[agent_id] = {
        **config,
        "vault_dir": str(agent_dir.relative_to(VAULT_PATH)),
        "process": None,
    }
    
    return {"success": True, "agent_id": agent_id, "config": config}


@app.get("/agents")
def list_agents():
    """List all Hermes agents."""
    return {
        "agents": [
            {
                "id": a["id"],
                "name": a["name"],
                "model": a["model"],
                "status": a["status"],
                "created_at": a["created_at"],
            }
            for a in agents.values()
        ]
    }


@app.get("/agents/{agent_id}")
def get_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agents[agent_id]


@app.post("/agents/{agent_id}/start")
def start_agent(agent_id: str):
    """Start a Hermes agent process."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    agent["status"] = "running"
    agent["started_at"] = datetime.now().isoformat()
    
    # Create conversation log
    conversations[agent_id] = []
    
    return {"success": True, "agent_id": agent_id, "status": "running"}


@app.post("/agents/{agent_id}/stop")
def stop_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agents[agent_id]["status"] = "stopped"
    agents[agent_id]["stopped_at"] = datetime.now().isoformat()
    
    return {"success": True, "status": "stopped"}


# ---- Conversation / Messaging ----

@app.post("/agents/{agent_id}/message")
def send_message(agent_id: str, msg: MessageSend):
    """Send a message to an agent and get response."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    
    # Log user message
    user_msg = {
        "id": str(uuid.uuid4())[:8],
        "role": "user",
        "content": msg.content,
        "timestamp": datetime.now().isoformat(),
    }
    
    if agent_id not in conversations:
        conversations[agent_id] = []
    conversations[agent_id].append(user_msg)
    
    # Simulate agent response (in production, this would call Hermes CLI)
    response = generate_agent_response(agent, msg.content)
    
    agent_msg = {
        "id": str(uuid.uuid4())[:8],
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
    }
    conversations[agent_id].append(agent_msg)
    
    # Write conversation to vault
    write_conversation_to_vault(agent_id, user_msg, agent_msg)
    
    return {
        "success": True,
        "message_id": agent_msg["id"],
        "response": response,
    }


def generate_agent_response(agent: dict, user_input: str) -> str:
    """Generate a simulated agent response. In production, calls Hermes."""
    # This is a placeholder - real implementation would call hermes CLI
    responses = {
        "hello": f"Hello! I'm {agent['name']}, powered by Hermes. How can I help you today?",
        "status": f"I'm running with model {agent['model']}. All systems operational.",
        "memory": "I can read and write to Obsidian vault for persistent memory. What would you like me to remember?",
    }
    
    lower_input = user_input.lower()
    for key, resp in responses.items():
        if key in lower_input:
            return resp
    
    return f"I received your message: '{user_input}'. I'm processing this through my Hermes agent core with {agent['model']}."


def write_conversation_to_vault(agent_id: str, user_msg: dict, agent_msg: dict):
    """Write conversation turn to Obsidian vault."""
    agent = agents[agent_id]
    conv_dir = VAULT_PATH / "50-Agents" / "Hermes" / agent["name"] / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    conv_file = conv_dir / f"{date_str}.md"
    
    entry = f"""
### {user_msg['timestamp']}

**User:** {user_msg['content']}

**{agent['name']}:** {agent_msg['content']}

---
"""
    
    if conv_file.exists():
        content = conv_file.read_text()
    else:
        content = f"# Conversation Log - {agent['name']}\n\n"
    
    content += entry
    conv_file.write_text(content)


@app.get("/agents/{agent_id}/conversation")
def get_conversation(agent_id: str):
    """Get conversation history for an agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {"messages": conversations.get(agent_id, [])}


# ---- Obsidian Memory ----

@app.post("/memory/write")
def write_memory(mem: MemoryWrite):
    """Write agent memory to Obsidian vault."""
    if mem.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[mem.agent_id]
    mem_dir = VAULT_PATH / "50-Agents" / "Hermes" / agent["name"] / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = f"{mem.category}_{timestamp}.md"
    filepath = mem_dir / filename
    
    frontmatter = f"""---
date: "{datetime.now().strftime('%Y-%m-%d')}"
time: "{datetime.now().strftime('%H:%M:%S')}"
agent: "{agent['name']}"
agent_id: "{mem.agent_id}"
category: "{mem.category}"
tags:
  - memory
  - hermes
{chr(10).join(f'  - {tag}' for tag in mem.tags)}
---

# {mem.category.title()} Memory

**Agent:** {agent['name']}
**Time:** {datetime.now().isoformat()}

{mem.content}
"""
    
    filepath.write_text(frontmatter)
    
    return {
        "success": True,
        "filepath": str(filepath.relative_to(VAULT_PATH)),
        "vault": str(VAULT_PATH),
    }


@app.get("/memory/read/{agent_id}")
def read_memory(agent_id: str, limit: int = 10):
    """Read agent memory from Obsidian vault."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    mem_dir = VAULT_PATH / "50-Agents" / "Hermes" / agent["name"] / "memory"
    
    if not mem_dir.exists():
        return {"memories": []}
    
    memories = []
    for f in sorted(mem_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        content = f.read_text()
        # Extract content after frontmatter
        if "---" in content:
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
            else:
                body = content
        else:
            body = content
        
        memories.append({
            "file": f.name,
            "preview": body[:300] + "..." if len(body) > 300 else body,
        })
        
        if len(memories) >= limit:
            break
    
    return {"memories": memories}


@app.get("/memory/search")
def search_memory(query: str):
    """Search memory across all agents."""
    hermes_dir = VAULT_PATH / "50-Agents" / "Hermes"
    if not hermes_dir.exists():
        return {"results": []}
    
    results = []
    for mem_file in hermes_dir.rglob("memory/*.md"):
        content = mem_file.read_text().lower()
        if query.lower() in content:
            rel_path = mem_file.relative_to(VAULT_PATH)
            results.append({
                "file": str(rel_path),
                "preview": content[:200] + "...",
            })
    
    return {"results": results[:20]}


# ---- Skills Management ----

@app.post("/skills/create")
def create_skill(skill: SkillCreate):
    """Create a skill for an agent (Hermes skills system)."""
    if skill.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[skill.agent_id]
    skills_dir = VAULT_PATH / "50-Agents" / "Hermes" / agent["name"] / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    
    skill_file = skills_dir / f"{skill.name}.py"
    skill_file.write_text(skill.code)
    
    # Write skill metadata
    meta_file = skills_dir / f"{skill.name}.json"
    meta_file.write_text(json.dumps({
        "name": skill.name,
        "description": skill.description,
        "created_at": datetime.now().isoformat(),
        "agent": agent["name"],
    }, indent=2))
    
    return {
        "success": True,
        "skill": skill.name,
        "filepath": str(skill_file.relative_to(VAULT_PATH)),
    }


@app.get("/skills/{agent_id}")
def list_skills(agent_id: str):
    """List skills for an agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    skills_dir = VAULT_PATH / "50-Agents" / "Hermes" / agent["name"] / "skills"
    
    if not skills_dir.exists():
        return {"skills": []}
    
    skills = []
    for meta_file in skills_dir.glob("*.json"):
        meta = json.loads(meta_file.read_text())
        skills.append(meta)
    
    return {"skills": skills}


# ---- Vault Stats ----

@app.get("/vault/stats")
def vault_stats():
    """Get Obsidian vault statistics for Hermes."""
    hermes_dir = VAULT_PATH / "50-Agents" / "Hermes"
    
    if not hermes_dir.exists():
        return {"hermes_dir": str(hermes_dir), "agents": 0, "memories": 0, "skills": 0}
    
    memories = list(hermes_dir.rglob("memory/*.md"))
    skills = list(hermes_dir.rglob("skills/*.py"))
    conversations = list(hermes_dir.rglob("conversations/*.md"))
    
    return {
        "vault_path": str(VAULT_PATH),
        "hermes_dir": str(hermes_dir.relative_to(VAULT_PATH)),
        "agents": len(agents),
        "memories": len(memories),
        "skills": len(skills),
        "conversations": len(conversations),
        "total_size_mb": round(
            sum(f.stat().st_size for f in hermes_dir.rglob("*") if f.is_file()) / (1024 * 1024), 2
        ),
    }

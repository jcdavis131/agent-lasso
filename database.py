import sqlite3
import threading
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import json

# ---------------------------------------------------------------------------
# SQLite file location
# ---------------------------------------------------------------------------
# Vercel (and most serverless platforms) mount the project root as **read-only**.
# The only writable directory is /tmp.  Detect such environments and store the
# SQLite file there; otherwise use the project root for local development.
# You can also override via the DATABASE_PATH env var.
# ---------------------------------------------------------------------------

if os.getenv("DATABASE_PATH"):
    DB_NAME = os.getenv("DATABASE_PATH")
else:
    running_on_vercel = any(
        key in os.environ for key in ("VERCEL", "VERCEL_URL", "VERCEL_ENV")
    )
    DB_NAME = "/tmp/agent_log.db" if running_on_vercel else "agent_log.db"

# Thread-local storage for database connections
local = threading.local()

def get_db():
    """Gets the database connection from thread-local storage, creating it if it doesn't exist."""
    if not hasattr(local, 'db'):
        local.db = sqlite3.connect(DB_NAME, check_same_thread=False)
        local.db.row_factory = sqlite3.Row
    return local.db

def init_db():
    """Initializes the database, creating tables if they don't exist."""
    db = get_db()
    cursor = db.cursor()
    
    # Table for conversation threads
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            FOREIGN KEY (agent_id) REFERENCES agent_configs (id) ON DELETE CASCADE
        )
    ''')
    
    # Enhanced messages table with thread_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tool_data TEXT,
            FOREIGN KEY (thread_id) REFERENCES conversation_threads (id) ON DELETE CASCADE
        )
    ''')
    
    # Table for persistent agent configurations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            tools TEXT NOT NULL, -- Storing as a JSON string of tool names
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add thread_id column to existing messages table if it doesn't exist
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN thread_id INTEGER")
        db.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    # Table for benchmark results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id TEXT PRIMARY KEY,
            agent_id INTEGER,
            test_type TEXT,
            score REAL,
            timestamp TEXT,
            details TEXT
        )
    ''')
    
    # Table for exam sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_sessions (
            session_id TEXT PRIMARY KEY,
            exam_slug TEXT NOT NULL,
            agent_id INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            accuracy REAL DEFAULT 0.0,
            total_score REAL DEFAULT 0.0,
            avg_response_time REAL DEFAULT 0.0,
            total_questions INTEGER DEFAULT 0,
            correct_answers INTEGER DEFAULT 0,
            metadata TEXT,
            FOREIGN KEY (agent_id) REFERENCES agent_configs (id)
        )
    ''')
    
    # Table for detailed exam results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            correct BOOLEAN NOT NULL,
            confidence REAL DEFAULT 0.0,
            response TEXT,
            expected TEXT,
            feedback TEXT,
            response_time REAL DEFAULT 0.0,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES exam_sessions (session_id)
        )
    ''')
    
    # Table for exam analytics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_slug TEXT NOT NULL,
            total_sessions INTEGER DEFAULT 0,
            avg_accuracy REAL DEFAULT 0.0,
            avg_response_time REAL DEFAULT 0.0,
            last_updated TEXT NOT NULL,
            UNIQUE(exam_slug)
        )
    ''')
    
    db.commit()

# --- Agent Configuration CRUD Functions ---

def dict_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a sqlite3.Row object to a dictionary."""
    if not row: return {}
    return dict(row)

def add_agent_config(name: str, description: str, provider: str, model: str, tools_json: str, system_prompt: str = "", api_keys: str = None) -> int:
    """Adds a new agent configuration to the database."""
    db = get_db()
    cursor = db.cursor()
    
    # Check if system_prompt column exists, add it if not
    try:
        cursor.execute("SELECT system_prompt FROM agent_configs LIMIT 1")
    except sqlite3.OperationalError:
        # Add system_prompt column
        cursor.execute("ALTER TABLE agent_configs ADD COLUMN system_prompt TEXT")
        db.commit()
    
    # Check if api_keys column exists, add it if not
    try:
        cursor.execute("SELECT api_keys FROM agent_configs LIMIT 1")
    except sqlite3.OperationalError:
        # Add api_keys column
        cursor.execute("ALTER TABLE agent_configs ADD COLUMN api_keys TEXT")
        db.commit()
    
    cursor.execute(
        "INSERT INTO agent_configs (name, description, provider, model, tools, system_prompt, api_keys) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name, description, provider, model, tools_json, system_prompt, api_keys)
    )
    db.commit()
    return cursor.lastrowid

def get_agent_configs() -> List[Dict[str, Any]]:
    """Retrieves all agent configurations from the database."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM agent_configs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return [dict_from_row(row) for row in rows]

def get_agent_config(agent_id: int) -> Dict[str, Any]:
    """Retrieves a single agent configuration by its ID."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM agent_configs WHERE id = ?", (agent_id,))
    row = cursor.fetchone()
    return dict_from_row(row)

def update_agent_config(agent_id: int, name: str, description: str, provider: str, model: str, tools_json: str, system_prompt: str = None, api_keys: str = None):
    """Updates an existing agent configuration."""
    db = get_db()
    cursor = db.cursor()
    
    # Build update query dynamically based on which fields are provided
    update_fields = []
    values = []
    
    update_fields.extend(["name = ?", "description = ?", "provider = ?", "model = ?", "tools = ?"])
    values.extend([name, description, provider, model, tools_json])
    
    if system_prompt is not None:
        update_fields.append("system_prompt = ?")
        values.append(system_prompt)
    
    if api_keys is not None:
        update_fields.append("api_keys = ?")
        values.append(api_keys)
    
    values.append(agent_id)
    
    cursor.execute(
        f"UPDATE agent_configs SET {', '.join(update_fields)} WHERE id = ?",
        values
    )
    db.commit()

def delete_agent_config(agent_id: int):
    """Deletes an agent configuration from the database."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM agent_configs WHERE id = ?", (agent_id,))
    db.commit()

# --- Conversation Thread Management ---

class ConversationThread(BaseModel):
    id: int
    agent_id: int
    name: str
    created_at: str
    updated_at: str
    message_count: int

def create_thread(agent_id: int, name: str = None) -> int:
    """Creates a new conversation thread for an agent."""
    db = get_db()
    cursor = db.cursor()
    
    # Check how many threads exist for this agent
    cursor.execute("SELECT COUNT(*) as count FROM conversation_threads WHERE agent_id = ?", (agent_id,))
    thread_count = cursor.fetchone()['count']
    
    if thread_count >= 3:
        raise ValueError("Maximum of 3 conversation threads allowed per agent")
    
    if name is None:
        name = f"Conversation {thread_count + 1}"
    
    cursor.execute(
        "INSERT INTO conversation_threads (agent_id, name) VALUES (?, ?)",
        (agent_id, name)
    )
    db.commit()
    return cursor.lastrowid

def get_threads(agent_id: int) -> List[ConversationThread]:
    """Retrieves all conversation threads for an agent."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """SELECT ct.*, COUNT(m.id) as message_count 
           FROM conversation_threads ct 
           LEFT JOIN messages m ON ct.id = m.thread_id 
           WHERE ct.agent_id = ? 
           GROUP BY ct.id 
           ORDER BY ct.updated_at DESC""",
        (agent_id,)
    )
    rows = cursor.fetchall()
    return [ConversationThread(**dict_from_row(row)) for row in rows]

def get_thread(thread_id: int) -> Optional[ConversationThread]:
    """Retrieves a specific conversation thread."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """SELECT ct.*, COUNT(m.id) as message_count 
           FROM conversation_threads ct 
           LEFT JOIN messages m ON ct.id = m.thread_id 
           WHERE ct.id = ? 
           GROUP BY ct.id""",
        (thread_id,)
    )
    row = cursor.fetchone()
    return ConversationThread(**dict_from_row(row)) if row else None

def update_thread_name(thread_id: int, name: str):
    """Updates the name of a conversation thread."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "UPDATE conversation_threads SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (name, thread_id)
    )
    db.commit()

def delete_thread(thread_id: int):
    """Deletes a conversation thread and all its messages."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM conversation_threads WHERE id = ?", (thread_id,))
    db.commit()

def update_thread_timestamp(thread_id: int):
    """Updates the updated_at timestamp for a thread."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "UPDATE conversation_threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (thread_id,)
    )
    db.commit()

# --- Enhanced Message CRUD Functions ---

class Message(BaseModel):
    id: int
    role: str
    content: str
    timestamp: str
    tool_data: Optional[str] = None
    thread_id: Optional[int] = None

def add_message(session_id: str, role: str, content: str, timestamp: str, tool_data: Optional[str] = None, thread_id: Optional[int] = None):
    """Adds a message to the database for a specific session and optionally a thread."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, tool_data, thread_id) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, role, content, timestamp, tool_data, thread_id)
    )
    
    # Update thread timestamp if thread_id is provided
    if thread_id:
        update_thread_timestamp(thread_id)
    
    db.commit()

def get_messages(session_id: str, thread_id: Optional[int] = None) -> List[Message]:
    """Retrieves all messages for a specific session and optionally filtered by thread."""
    db = get_db()
    cursor = db.cursor()
    
    if thread_id:
        cursor.execute(
            "SELECT id, role, content, timestamp, tool_data, thread_id FROM messages WHERE session_id = ? AND thread_id = ? ORDER BY timestamp",
            (session_id, thread_id)
        )
    else:
        cursor.execute(
            "SELECT id, role, content, timestamp, tool_data, thread_id FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
    
    rows = cursor.fetchall()
    return [Message(**dict_from_row(row)) for row in rows]

def get_thread_messages(thread_id: int) -> List[Message]:
    """Retrieves all messages for a specific thread."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, role, content, timestamp, tool_data, thread_id FROM messages WHERE thread_id = ? ORDER BY timestamp",
        (thread_id,)
    )
    rows = cursor.fetchall()
    return [Message(**dict_from_row(row)) for row in rows]

def clear_messages(session_id: str, thread_id: Optional[int] = None):
    """Clears all messages for a specific session and optionally for a specific thread."""
    db = get_db()
    cursor = db.cursor()
    
    if thread_id:
        cursor.execute("DELETE FROM messages WHERE session_id = ? AND thread_id = ?", (session_id, thread_id))
    else:
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    
    db.commit()

def clear_thread_messages(thread_id: int):
    """Clears all messages for a specific thread."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    db.commit()

# --- Benchmark result CRUD ---

def add_benchmark_result(id_: str, agent_id: int, test_type: str, score: float, details: dict):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO benchmark_results (id, agent_id, test_type, score, timestamp, details) VALUES (?,?,?,?,?,?)",
        (id_, agent_id, test_type, score, datetime.now().isoformat(), json.dumps(details))
    )
    db.commit()


def get_recent_benchmarks(limit: int = 5):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM benchmark_results ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    return [dict_from_row(r) for r in rows]

# NEW: Fetch benchmark history for specific agent

def get_benchmarks_for_agent(agent_id: int, limit: int = 20):
    """Retrieve the most recent benchmark results for a given agent."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM benchmark_results WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?",
        (agent_id, limit),
    )
    rows = cursor.fetchall()
    return [dict_from_row(r) for r in rows]


# NEW: Simple leaderboard helper – ranks agents by average score (higher is better)

def get_leaderboard(limit: int = 10):
    """Return a leaderboard of agents ranked by their average benchmark score.

    NOTE: This provides a naive average across *all* benchmark types where a higher
    score is considered better.  For benchmarks where a lower score is favourable
    (e.g. response time), you may wish to invert/normalise the values at the
    harness level before persisting them so that higher is always better.  This
    keeps the database layer simple.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT agent_id, AVG(score) as avg_score, COUNT(*) as runs, MAX(timestamp) as last_run
        FROM benchmark_results
        GROUP BY agent_id
        ORDER BY avg_score DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    return [dict_from_row(r) for r in rows]

# Add new functions for exam data management

def add_exam_session(session_id: str, exam_slug: str, agent_id: int, 
                    start_time: str, end_time: str = None, 
                    accuracy: float = 0.0, total_score: float = 0.0,
                    avg_response_time: float = 0.0, total_questions: int = 0,
                    correct_answers: int = 0, metadata: dict = None):
    """Add exam session to database"""
    db = get_db()
    cursor = db.cursor()
    
    metadata_json = json.dumps(metadata) if metadata else None
    
    cursor.execute('''
        INSERT OR REPLACE INTO exam_sessions 
        (session_id, exam_slug, agent_id, start_time, end_time, accuracy, 
         total_score, avg_response_time, total_questions, correct_answers, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, exam_slug, agent_id, start_time, end_time, accuracy,
          total_score, avg_response_time, total_questions, correct_answers, metadata_json))
    
    db.commit()

def add_exam_result(session_id: str, task_id: str, correct: bool, 
                   confidence: float, response: str, expected: str,
                   feedback: str, response_time: float, timestamp: str):
    """Add individual exam result to database"""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('''
        INSERT INTO exam_results 
        (session_id, task_id, correct, confidence, response, expected, feedback, response_time, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, task_id, correct, confidence, response, expected, feedback, response_time, timestamp))
    
    db.commit()

def get_exam_session(session_id: str):
    """Get exam session by ID"""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('SELECT * FROM exam_sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    
    if row:
        session_data = dict_from_row(row)
        
        # Get results for this session
        cursor.execute('SELECT * FROM exam_results WHERE session_id = ? ORDER BY timestamp', (session_id,))
        results = [dict_from_row(r) for r in cursor.fetchall()]
        
        session_data['results'] = results
        return session_data
    
    return None

def get_exam_sessions_by_agent(agent_id: int, limit: int = 20):
    """Get exam sessions for a specific agent"""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('''
        SELECT * FROM exam_sessions 
        WHERE agent_id = ? 
        ORDER BY start_time DESC 
        LIMIT ?
    ''', (agent_id, limit))
    
    return [dict_from_row(row) for row in cursor.fetchall()]

def get_exam_sessions_by_slug(exam_slug: str, limit: int = 20):
    """Get exam sessions for a specific exam"""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('''
        SELECT * FROM exam_sessions 
        WHERE exam_slug = ? 
        ORDER BY start_time DESC 
        LIMIT ?
    ''', (exam_slug, limit))
    
    return [dict_from_row(row) for row in cursor.fetchall()]

def update_exam_analytics(exam_slug: str):
    """Update analytics for an exam"""
    db = get_db()
    cursor = db.cursor()
    
    # Calculate analytics from sessions
    cursor.execute('''
        SELECT 
            COUNT(*) as total_sessions,
            AVG(accuracy) as avg_accuracy,
            AVG(avg_response_time) as avg_response_time
        FROM exam_sessions 
        WHERE exam_slug = ? AND end_time IS NOT NULL
    ''', (exam_slug,))
    
    stats = cursor.fetchone()
    
    if stats and stats['total_sessions'] > 0:
        cursor.execute('''
            INSERT OR REPLACE INTO exam_analytics 
            (exam_slug, total_sessions, avg_accuracy, avg_response_time, last_updated)
            VALUES (?, ?, ?, ?, ?)
        ''', (exam_slug, stats['total_sessions'], stats['avg_accuracy'], 
              stats['avg_response_time'], datetime.now().isoformat()))
        
        db.commit()

def get_exam_analytics(exam_slug: str):
    """Get analytics for an exam"""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('SELECT * FROM exam_analytics WHERE exam_slug = ?', (exam_slug,))
    row = cursor.fetchone()
    
    if row:
        return dict_from_row(row)
    
    # If no analytics exist, calculate them
    update_exam_analytics(exam_slug)
    
    cursor.execute('SELECT * FROM exam_analytics WHERE exam_slug = ?', (exam_slug,))
    row = cursor.fetchone()
    
    return dict_from_row(row) if row else None

def get_exam_leaderboard(exam_slug: str = None, limit: int = 10):
    """Get leaderboard for exams"""
    db = get_db()
    cursor = db.cursor()
    
    if exam_slug:
        # Leaderboard for specific exam
        cursor.execute('''
            SELECT 
                es.agent_id,
                ac.name as agent_name,
                ac.provider,
                ac.model,
                es.accuracy,
                es.total_score,
                es.avg_response_time,
                es.start_time
            FROM exam_sessions es
            JOIN agent_configs ac ON es.agent_id = ac.id
            WHERE es.exam_slug = ? AND es.end_time IS NOT NULL
            ORDER BY es.accuracy DESC, es.avg_response_time ASC
            LIMIT ?
        ''', (exam_slug, limit))
    else:
        # Overall leaderboard across all exams
        cursor.execute('''
            SELECT 
                es.agent_id,
                ac.name as agent_name,
                ac.provider,
                ac.model,
                AVG(es.accuracy) as avg_accuracy,
                AVG(es.total_score) as avg_total_score,
                AVG(es.avg_response_time) as avg_response_time,
                COUNT(*) as total_exams
            FROM exam_sessions es
            JOIN agent_configs ac ON es.agent_id = ac.id
            WHERE es.end_time IS NOT NULL
            GROUP BY es.agent_id
            ORDER BY AVG(es.accuracy) DESC, AVG(es.avg_response_time) ASC
            LIMIT ?
        ''', (limit,))
    
    return [dict_from_row(row) for row in cursor.fetchall()] 
# Silver Lasso Benchmark Exam System

A comprehensive benchmark testing center for evaluating AI agent performance across multiple dimensions and domains.

## 🎯 Overview

The Silver Lasso Benchmark Exam System transforms your AI agent development platform into a robust testing center that can:

- **Load and run standardized benchmark exams** from YAML files
- **Generate custom exams** using AI-powered ExamBuilder
- **Track performance metrics** with detailed analytics
- **Compare agents** with leaderboards and historical data
- **Integrate seamlessly** with existing benchmark harnesses

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │  Overview   │   Exams     │   Builder   │  Analytics  │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────┼─────────────────────────────────┐
│                    API Layer (FastAPI)                      │
│  /api/exams/*  │  /api/benchmark/*  │  /api/agents/*        │
└─────────────────────────────┼─────────────────────────────────┘
                             │
┌─────────────────────────────┼─────────────────────────────────┐
│                    Core Components                          │
│                             │                               │
│  ┌─────────────────────┐   │   ┌─────────────────────┐      │
│  │  BenchmarkExamEngine│───┼───│   ExamBuilder       │      │
│  │  • YAML Loading     │   │   │   • AI Generation   │      │
│  │  • Exam Execution   │   │   │   • Template System │      │
│  │  • Result Analysis  │   │   │   • Content Creation│      │
│  └─────────────────────┘   │   └─────────────────────┘      │
│                             │                               │
│  ┌─────────────────────┐   │   ┌─────────────────────┐      │
│  │  Benchmark Harness  │───┼───│   Database Layer    │      │
│  │  • Integration     │   │   │   • Session Storage │      │
│  │  • Plugin System    │   │   │   • Analytics       │      │
│  │  • Streaming        │   │   │   • Leaderboards    │      │
│  └─────────────────────┘   │   └─────────────────────┘      │
└─────────────────────────────┼─────────────────────────────────┘
                             │
┌─────────────────────────────┼─────────────────────────────────┐
│                     Data Layer                              │
│  benchmark_exams/*.yaml  │  agent_log.db  │  Generated Exams │
└─────────────────────────────┼─────────────────────────────────┘
```

## 📦 Core Components

### 1. BenchmarkExamEngine (`benchmark_exam_engine.py`)

**Purpose**: Core engine for loading, managing, and executing benchmark exams.

**Key Features**:
- **YAML Exam Loading**: Automatically discovers and loads exam files
- **Multiple Evaluation Methods**: equals, contains, regex, semantic matching, numeric tolerance
- **Session Management**: Tracks exam sessions with detailed results
- **Database Integration**: Persistent storage of sessions and analytics
- **Performance Metrics**: Accuracy, response time, scoring analysis

**Usage Example**:
```python
from benchmark_exam_engine import exam_engine

# Get available exams
exams = exam_engine.get_available_exams()

# Run an exam
async def agent_runner(prompt: str, session_id: str) -> str:
    return agent.run(prompt, session_id)

session = await exam_engine.run_exam("math_quiz", agent_id=1, agent_runner_func=agent_runner)
print(f"Accuracy: {session.accuracy:.1%}")
```

### 2. ExamBuilder (`exam_builder_agent.py`)

**Purpose**: AI-powered system for generating custom benchmark exams.

**Key Features**:
- **Template-Driven Generation**: Subject-specific templates (math, science, CS, reasoning)
- **Difficulty Progression**: Adaptive question complexity
- **Multiple Question Types**: computational, conceptual, problem-solving, analytical
- **Skill Targeting**: Focus on specific competencies
- **YAML Export**: Generated exams saved as standard YAML files

**Usage Example**:
```python
from exam_builder_agent import ExamBuilder, ExamRequirements, ExamType

builder = ExamBuilder()
requirements = ExamRequirements(
    subject="Calculus",
    difficulty="Advanced",
    num_questions=10,
    exam_type=ExamType.MATHEMATICAL,
    target_skills=["derivatives", "integrals", "optimization"]
)

exam_data = builder.generate_exam(requirements)
file_path = builder.save_exam_to_file(exam_data)
```

### 3. Database Integration (`database.py`)

**Purpose**: Persistent storage for exam sessions, results, and analytics.

**Schema**:
- `exam_sessions`: Complete exam session metadata
- `exam_results`: Individual task results with detailed feedback
- `exam_analytics`: Aggregated performance statistics

**Key Functions**:
```python
# Store exam session
add_exam_session(session_id, exam_slug, agent_id, start_time, ...)

# Store individual results
add_exam_result(session_id, task_id, correct, confidence, response, ...)

# Get analytics
analytics = get_exam_analytics(exam_slug)
leaderboard = get_exam_leaderboard(exam_slug, limit=10)
```

### 4. Benchmark Harness Integration (`benchmarks.py`)

**Purpose**: Seamless integration with existing benchmark infrastructure.

**Features**:
- **YAMLExamHarness**: Wrapper for YAML exams in benchmark system
- **Streaming Support**: Real-time progress updates during execution
- **Plugin Registration**: Automatic registration of YAML exams
- **Legacy Compatibility**: Works alongside existing benchmarks

## 📋 YAML Exam Format

Exams are defined in YAML files with the following structure:

```yaml
slug: exam_identifier
name: Human Readable Exam Name
description: Detailed description of what this exam tests
category: Mathematical Reasoning  # See ExamCategory enum
difficulty: Intermediate  # Basic|Intermediate|Advanced|Expert|College Level
timeout: 600  # Total time limit in seconds
source: "Source or author information"
metadata:
  version: "1.0"
  tags: ["tag1", "tag2"]

tasks:
  - id: unique_task_id
    prompt: "What is the question being asked?"
    answer: "Expected answer"
    eval: equals  # equals|contains|regex|semantic_match|numeric_tolerance
    explanation: "Why this answer is correct"
    difficulty_weight: 1.0  # Weighting for scoring
    time_limit: 60  # Optional per-task time limit
    hints: ["Optional hint 1", "Optional hint 2"]
    tags: ["arithmetic", "problem-solving"]
  
  - id: another_task
    prompt: "Another question..."
    answer: "Another answer"
    eval: contains
    # ... more task fields
```

## 🚀 API Endpoints

### Exam Management
- `GET /api/exams/available` - List all available exams
- `GET /api/exams/{exam_slug}` - Get detailed exam information
- `POST /api/exams/run` - Execute an exam with an agent
- `GET /api/exams/session/{session_id}` - Get session results

### ExamBuilder
- `POST /api/exams/build` - Generate a new exam from requirements

### Analytics
- `GET /api/exams/analytics/{exam_slug}` - Get exam performance analytics
- `GET /api/benchmark/leaderboard` - Get agent performance leaderboard

## 🎮 Web Dashboard

The enhanced benchmark dashboard provides a comprehensive interface with four main tabs:

### 1. Overview Tab
- **Quick Stats**: Total exams, sessions, average accuracy, active agents
- **Quick Test Run**: Select agent + exam and run immediately
- **Recent Activity**: Latest test results and top performers

### 2. Available Exams Tab
- **Exam Gallery**: Grid view of all available benchmark exams
- **Exam Details**: Category, difficulty, question count, descriptions
- **Quick Run**: One-click exam execution

### 3. Exam Builder Tab
- **Exam Creation Form**: Subject, difficulty, question count, time limits
- **Target Skills**: Specify competencies to focus on
- **Real-time Generation**: Progress tracking with status updates

### 4. Analytics Tab
- **Performance Analytics**: Detailed metrics and trends
- **Exam Leaderboards**: Top performing agents by exam
- **Historical Data**: Long-term performance tracking

## 📊 Evaluation Methods

The system supports multiple evaluation methods for maximum flexibility:

### 1. Equals (`equals`)
Exact string matching (case-insensitive)
```yaml
answer: "42"
eval: equals
# Matches: "42", "  42  ", "42.0"
```

### 2. Contains (`contains`)
Substring matching for longer responses
```yaml
answer: "photosynthesis"
eval: contains
# Matches: "The process is photosynthesis", "photosynthesis occurs when..."
```

### 3. Regex (`regex`)
Pattern matching with regular expressions
```yaml
answer: "\\d{4}-\\d{2}-\\d{2}"  # Date format YYYY-MM-DD
eval: regex
# Matches: "2024-03-15", "1999-12-31"
```

### 4. Semantic Match (`semantic_match`)
Keyword overlap analysis for conceptual answers
```yaml
answer: "gravitational force attraction"
eval: semantic_match
# Matches responses with 60%+ keyword overlap
```

### 5. Numeric Tolerance (`numeric_tolerance`)
Numerical answers with acceptable error margins
```yaml
answer: "3.14159"
eval: numeric_tolerance
# Matches: "3.14", "3.142", "3.14159265" (within tolerance)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_benchmark_system.py
```

**Test Coverage**:
- ✅ YAML exam loading and parsing
- ✅ All evaluation methods
- ✅ Exam execution workflows
- ✅ ExamBuilder generation
- ✅ Database operations
- ✅ API endpoint integration
- ✅ End-to-end workflows

## 🚀 Getting Started

### 1. Basic Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from database import init_db; init_db()"

# Start the server
python main.py
```

### 2. Create Your First Custom Exam
1. Navigate to the Benchmark Center → Exam Builder tab
2. Fill in exam requirements (subject, difficulty, questions)
3. Click "Generate Exam" and wait for completion
4. New exam appears in Available Exams tab

### 3. Run Your First Benchmark
1. Go to Benchmark Center → Overview tab
2. Select an agent from the dropdown
3. Choose an exam (try "sample_math_quiz" for testing)
4. Click "Run Test" and watch real-time progress

### 4. Analyze Results
1. Check Recent Test Results for immediate feedback
2. Visit Analytics tab for detailed performance data
3. View leaderboards to compare agent performance
4. Use dashboard metrics to guide agent improvements

---

🎉 **The Silver Lasso Benchmark Exam System is now ready to transform your AI agent development with comprehensive testing capabilities!**

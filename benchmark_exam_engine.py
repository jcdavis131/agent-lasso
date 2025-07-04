"""
Advanced Benchmark Exam Engine for Silver Lasso
Comprehensive testing center for watching and reviewing agent performance
"""

import os
import yaml
import json
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path
import sqlite3
import logging
from database import (
    add_exam_session, add_exam_result, get_exam_session, 
    get_exam_sessions_by_agent, get_exam_sessions_by_slug,
    update_exam_analytics, get_exam_analytics as get_db_exam_analytics,
    get_exam_leaderboard
)

logger = logging.getLogger(__name__)

class ExamDifficulty(Enum):
    BASIC = "Basic"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"
    COLLEGE_LEVEL = "College Level"

class ExamCategory(Enum):
    MATHEMATICAL_REASONING = "Mathematical Reasoning"
    SCIENTIFIC_REASONING = "Scientific Reasoning"
    LOGICAL_REASONING = "Logical Reasoning"
    COMMONSENSE_REASONING = "Commonsense Reasoning"
    LANGUAGE_UNDERSTANDING = "Language Understanding"
    AI_SAFETY_ALIGNMENT = "AI Safety & Alignment"
    ACADEMIC_KNOWLEDGE = "Academic Knowledge"
    TECHNICAL_CODING = "Technical & Coding"
    RESEARCH_SYNTHESIS = "Research & Synthesis"

class EvaluationMethod(Enum):
    EQUALS = "equals"
    CONTAINS = "contains"
    REGEX = "regex"
    SEMANTIC_MATCH = "semantic_match"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    CUSTOM = "custom"

@dataclass
class ExamTask:
    """Individual task within an exam"""
    id: str
    prompt: str
    answer: str
    eval_method: EvaluationMethod
    explanation: str = ""
    hints: List[str] = field(default_factory=list)
    difficulty_weight: float = 1.0
    time_limit: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    
    def evaluate_response(self, response: str) -> Tuple[bool, float, str]:
        """Evaluate agent response against expected answer"""
        response = response.strip()
        expected = self.answer.strip()
        
        if self.eval_method == EvaluationMethod.EQUALS:
            correct = response.lower() == expected.lower()
        elif self.eval_method == EvaluationMethod.CONTAINS:
            correct = expected.lower() in response.lower()
        elif self.eval_method == EvaluationMethod.REGEX:
            correct = bool(re.search(expected, response, re.IGNORECASE))
        elif self.eval_method == EvaluationMethod.SEMANTIC_MATCH:
            correct = self._semantic_match(response, expected)
        elif self.eval_method == EvaluationMethod.NUMERIC_TOLERANCE:
            correct = self._numeric_tolerance_match(response, expected)
        else:
            correct = response.lower() == expected.lower()
        
        confidence = 1.0 if correct else 0.0
        feedback = self.explanation if correct else f"Expected: {expected}, Got: {response}"
        
        return correct, confidence, feedback
    
    def _semantic_match(self, response: str, expected: str) -> bool:
        """Semantic matching using keyword overlap"""
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        overlap = len(response_words & expected_words)
        return overlap >= len(expected_words) * 0.6
    
    def _numeric_tolerance_match(self, response: str, expected: str, tolerance: float = 0.01) -> bool:
        """Numeric matching with tolerance"""
        try:
            resp_num = float(re.search(r'-?\d+\.?\d*', response).group())
            exp_num = float(re.search(r'-?\d+\.?\d*', expected).group())
            return abs(resp_num - exp_num) <= tolerance
        except:
            return False

@dataclass
class ExamResult:
    """Result of running an exam task"""
    task_id: str
    correct: bool
    confidence: float
    response: str
    expected: str
    feedback: str
    response_time: float
    timestamp: datetime
    
@dataclass
class ExamSession:
    """Complete exam session results"""
    exam_slug: str
    agent_id: int
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_score: float = 0.0
    accuracy: float = 0.0
    avg_response_time: float = 0.0
    results: List[ExamResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_metrics(self):
        """Calculate final metrics for the session"""
        if not self.results:
            return
            
        correct_count = sum(1 for r in self.results if r.correct)
        self.accuracy = correct_count / len(self.results)
        self.avg_response_time = sum(r.response_time for r in self.results) / len(self.results)
        self.total_score = self.accuracy * 100  # Basic scoring
        
        if self.end_time:
            self.metadata['total_duration'] = (self.end_time - self.start_time).total_seconds()

@dataclass
class BenchmarkExam:
    """Complete benchmark exam definition"""
    slug: str
    name: str
    description: str
    category: ExamCategory
    difficulty: ExamDifficulty
    timeout: int = 600
    source: str = ""
    tasks: List[ExamTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_task_count(self) -> int:
        return len(self.tasks)
    
    def get_difficulty_score(self) -> float:
        """Calculate weighted difficulty score"""
        if not self.tasks:
            return 1.0
        return sum(task.difficulty_weight for task in self.tasks) / len(self.tasks)

class BenchmarkExamEngine:
    """Main engine for loading and running benchmark exams"""
    
    def __init__(self, exam_directory: str = "benchmark_exams"):
        self.exam_directory = Path(exam_directory)
        self.exams: Dict[str, BenchmarkExam] = {}
        self.active_sessions: Dict[str, ExamSession] = {}
        self.load_all_exams()
    
    def load_all_exams(self):
        """Load all YAML exam files from the benchmark_exams directory"""
        logger.info(f"Loading benchmark exams from {self.exam_directory}")
        
        if not self.exam_directory.exists():
            logger.warning(f"Benchmark exams directory {self.exam_directory} does not exist")
            return
            
        yaml_files = list(self.exam_directory.glob("*.yaml")) + list(self.exam_directory.glob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                exam = self._load_exam_from_yaml(yaml_file)
                if exam:
                    self.exams[exam.slug] = exam
                    logger.info(f"Loaded exam: {exam.name} ({exam.slug})")
            except Exception as e:
                logger.error(f"Failed to load exam from {yaml_file}: {e}")
                
        logger.info(f"Successfully loaded {len(self.exams)} benchmark exams")
    
    def _load_exam_from_yaml(self, yaml_file: Path) -> Optional[BenchmarkExam]:
        """Load a single exam from YAML file"""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            return None
            
        # Parse exam metadata
        exam = BenchmarkExam(
            slug=data.get('slug', yaml_file.stem),
            name=data.get('name', 'Unnamed Exam'),
            description=data.get('description', ''),
            category=self._parse_category(data.get('category', '')),
            difficulty=self._parse_difficulty(data.get('difficulty', '')),
            timeout=data.get('timeout', 600),
            source=data.get('source', ''),
            metadata=data.get('metadata', {})
        )
        
        # Parse tasks
        tasks_data = data.get('tasks', [])
        for task_data in tasks_data:
            task = ExamTask(
                id=task_data.get('id', str(uuid.uuid4())),
                prompt=task_data.get('prompt', ''),
                answer=task_data.get('answer', ''),
                eval_method=self._parse_eval_method(task_data.get('eval', 'equals')),
                explanation=task_data.get('explanation', ''),
                hints=task_data.get('hints', []),
                difficulty_weight=task_data.get('difficulty_weight', 1.0),
                time_limit=task_data.get('time_limit'),
                tags=task_data.get('tags', [])
            )
            exam.tasks.append(task)
        
        return exam
    
    def _parse_category(self, category_str: str) -> ExamCategory:
        """Parse category string to enum"""
        category_map = {
            'Mathematical Reasoning': ExamCategory.MATHEMATICAL_REASONING,
            'Scientific Reasoning': ExamCategory.SCIENTIFIC_REASONING,
            'Logical Reasoning': ExamCategory.LOGICAL_REASONING,
            'Commonsense Reasoning': ExamCategory.COMMONSENSE_REASONING,
            'Language Understanding': ExamCategory.LANGUAGE_UNDERSTANDING,
            'AI Safety & Alignment': ExamCategory.AI_SAFETY_ALIGNMENT,
            'Academic Knowledge': ExamCategory.ACADEMIC_KNOWLEDGE,
            'Technical & Coding': ExamCategory.TECHNICAL_CODING,
            'Research & Synthesis': ExamCategory.RESEARCH_SYNTHESIS,
        }
        return category_map.get(category_str, ExamCategory.ACADEMIC_KNOWLEDGE)
    
    def _parse_difficulty(self, difficulty_str: str) -> ExamDifficulty:
        """Parse difficulty string to enum"""
        difficulty_map = {
            'Basic': ExamDifficulty.BASIC,
            'Intermediate': ExamDifficulty.INTERMEDIATE,
            'Advanced': ExamDifficulty.ADVANCED,
            'Expert': ExamDifficulty.EXPERT,
            'College Level': ExamDifficulty.COLLEGE_LEVEL,
        }
        return difficulty_map.get(difficulty_str, ExamDifficulty.INTERMEDIATE)
    
    def _parse_eval_method(self, eval_str: str) -> EvaluationMethod:
        """Parse evaluation method string to enum"""
        eval_map = {
            'equals': EvaluationMethod.EQUALS,
            'contains': EvaluationMethod.CONTAINS,
            'regex': EvaluationMethod.REGEX,
            'semantic_match': EvaluationMethod.SEMANTIC_MATCH,
            'numeric_tolerance': EvaluationMethod.NUMERIC_TOLERANCE,
        }
        return eval_map.get(eval_str, EvaluationMethod.EQUALS)
    
    def get_available_exams(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available exams with metadata"""
        return {
            slug: {
                'name': exam.name,
                'description': exam.description,
                'category': exam.category.value,
                'difficulty': exam.difficulty.value,
                'task_count': exam.get_task_count(),
                'timeout': exam.timeout,
                'difficulty_score': exam.get_difficulty_score(),
                'source': exam.source
            }
            for slug, exam in self.exams.items()
        }
    
    def get_exam_by_slug(self, slug: str) -> Optional[BenchmarkExam]:
        """Get exam by slug"""
        return self.exams.get(slug)
    
    def start_exam_session(self, exam_slug: str, agent_id: int) -> str:
        """Start a new exam session"""
        exam = self.exams.get(exam_slug)
        if not exam:
            raise ValueError(f"Exam not found: {exam_slug}")
        
        session_id = str(uuid.uuid4())
        session = ExamSession(
            exam_slug=exam_slug,
            agent_id=agent_id,
            session_id=session_id,
            start_time=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        return session_id
    
    async def run_exam(self, exam_slug: str, agent_id: int, agent_runner_func) -> ExamSession:
        """Run a complete exam session"""
        exam = self.exams.get(exam_slug)
        if not exam:
            raise ValueError(f"Exam not found: {exam_slug}")
        
        session_id = self.start_exam_session(exam_slug, agent_id)
        session = self.active_sessions[session_id]
        
        logger.info(f"Starting exam session {session_id} for agent {agent_id} on exam {exam_slug}")
        
        # Add session to database
        add_exam_session(
            session_id=session_id,
            exam_slug=exam_slug,
            agent_id=agent_id,
            start_time=session.start_time.isoformat(),
            total_questions=len(exam.tasks)
        )
        
        # Run each task
        for task in exam.tasks:
            start_time = time.time()
            
            try:
                # Run agent on task
                response = await agent_runner_func(task.prompt, session_id)
                response_time = time.time() - start_time
                
                # Evaluate response
                correct, confidence, feedback = task.evaluate_response(response)
                
                # Store result
                result = ExamResult(
                    task_id=task.id,
                    correct=correct,
                    confidence=confidence,
                    response=response,
                    expected=task.answer,
                    feedback=feedback,
                    response_time=response_time,
                    timestamp=datetime.now()
                )
                
                session.results.append(result)
                
                # Add individual result to database
                add_exam_result(
                    session_id=session_id,
                    task_id=task.id,
                    correct=correct,
                    confidence=confidence,
                    response=response,
                    expected=task.answer,
                    feedback=feedback,
                    response_time=response_time,
                    timestamp=result.timestamp.isoformat()
                )
                
                logger.info(f"Task {task.id}: {'✓' if correct else '✗'} ({response_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error running task {task.id}: {e}")
                result = ExamResult(
                    task_id=task.id,
                    correct=False,
                    confidence=0.0,
                    response="ERROR: " + str(e),
                    expected=task.answer,
                    feedback=f"Task failed with error: {e}",
                    response_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
                session.results.append(result)
                
                # Add failed result to database
                add_exam_result(
                    session_id=session_id,
                    task_id=task.id,
                    correct=False,
                    confidence=0.0,
                    response="ERROR: " + str(e),
                    expected=task.answer,
                    feedback=f"Task failed with error: {e}",
                    response_time=result.response_time,
                    timestamp=result.timestamp.isoformat()
                )
        
        # Finalize session
        session.end_time = datetime.now()
        session.calculate_metrics()
        
        # Update session in database
        add_exam_session(
            session_id=session_id,
            exam_slug=exam_slug,
            agent_id=agent_id,
            start_time=session.start_time.isoformat(),
            end_time=session.end_time.isoformat(),
            accuracy=session.accuracy,
            total_score=session.total_score,
            avg_response_time=session.avg_response_time,
            total_questions=len(session.results),
            correct_answers=sum(1 for r in session.results if r.correct),
            metadata=session.metadata
        )
        
        # Update analytics
        update_exam_analytics(exam_slug)
        
        logger.info(f"Exam session {session_id} completed: {session.accuracy:.1%} accuracy, {session.avg_response_time:.2f}s avg response time")
        
        return session
    
    def get_session_results(self, session_id: str) -> Optional[ExamSession]:
        """Get results for a specific session"""
        # Try in-memory first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try database
        session_data = get_exam_session(session_id)
        if session_data:
            # Convert database record back to ExamSession object
            session = ExamSession(
                exam_slug=session_data['exam_slug'],
                agent_id=session_data['agent_id'],
                session_id=session_data['session_id'],
                start_time=datetime.fromisoformat(session_data['start_time']),
                end_time=datetime.fromisoformat(session_data['end_time']) if session_data['end_time'] else None,
                total_score=session_data['total_score'],
                accuracy=session_data['accuracy'],
                avg_response_time=session_data['avg_response_time'],
                metadata=json.loads(session_data['metadata']) if session_data['metadata'] else {}
            )
            
            # Convert results
            for result_data in session_data.get('results', []):
                result = ExamResult(
                    task_id=result_data['task_id'],
                    correct=result_data['correct'],
                    confidence=result_data['confidence'],
                    response=result_data['response'],
                    expected=result_data['expected'],
                    feedback=result_data['feedback'],
                    response_time=result_data['response_time'],
                    timestamp=datetime.fromisoformat(result_data['timestamp'])
                )
                session.results.append(result)
            
            return session
        
        return None
    
    def get_exam_analytics(self, exam_slug: str) -> Dict[str, Any]:
        """Get analytics for a specific exam across all sessions"""
        # Get analytics from database
        db_analytics = get_db_exam_analytics(exam_slug)
        
        if db_analytics:
            return {
                'exam_slug': exam_slug,
                'total_sessions': db_analytics['total_sessions'],
                'avg_accuracy': db_analytics['avg_accuracy'],
                'avg_response_time': db_analytics['avg_response_time'],
                'last_updated': db_analytics['last_updated']
            }
        
        return {'message': 'No analytics available for this exam'}
    
    def get_exam_leaderboard(self, exam_slug: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get leaderboard for exams"""
        return get_exam_leaderboard(exam_slug, limit)
    
    def get_agent_exam_history(self, agent_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get exam history for a specific agent"""
        return get_exam_sessions_by_agent(agent_id, limit)
    
    def get_exam_history(self, exam_slug: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get session history for a specific exam"""
        return get_exam_sessions_by_slug(exam_slug, limit)

# Global instance
exam_engine = BenchmarkExamEngine() 
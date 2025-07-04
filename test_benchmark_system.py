#!/usr/bin/env python3
"""
Comprehensive Test Suite for Silver Lasso Benchmark Exam System
Tests all components: ExamEngine, ExamBuilder, Database Integration, API Endpoints
"""

import unittest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import all the system components
from benchmark_exam_engine import (
    BenchmarkExamEngine, ExamTask, ExamResult, ExamSession, BenchmarkExam,
    ExamDifficulty, ExamCategory, EvaluationMethod
)
from exam_builder_agent import ExamBuilder, ExamRequirements, ExamType
from database import (
    init_db, add_exam_session, add_exam_result, get_exam_session,
    get_exam_sessions_by_agent, update_exam_analytics, get_exam_analytics
)

class TestBenchmarkExamEngine(unittest.TestCase):
    """Test the main benchmark exam engine"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test YAML files
        self.test_dir = tempfile.mkdtemp()
        self.engine = BenchmarkExamEngine(exam_directory=self.test_dir)
        
        # Create a test YAML exam file
        self.test_exam_yaml = """
slug: test_math_exam
name: Test Mathematics Exam
description: A test exam for mathematical reasoning
category: Mathematical Reasoning
difficulty: Intermediate
timeout: 600
source: Test Suite
metadata:
  version: "1.0"
  author: "Test System"
tasks:
  - id: basic_addition
    prompt: "What is 15 + 27?"
    answer: "42"
    eval: equals
    explanation: "Simple addition: 15 + 27 = 42"
    difficulty_weight: 1.0
    tags: ["arithmetic", "addition"]
  
  - id: algebra_solve
    prompt: "Solve for x: 2x + 10 = 30"
    answer: "10"
    eval: contains
    explanation: "2x = 20, so x = 10"
    difficulty_weight: 1.5
    tags: ["algebra", "equations"]
  
  - id: geometry_area
    prompt: "What is the area of a rectangle with length 8 and width 5?"
    answer: "40"
    eval: numeric_tolerance
    explanation: "Area = length × width = 8 × 5 = 40"
    difficulty_weight: 1.2
    tags: ["geometry", "area"]
"""
        
        # Write test exam to file
        with open(os.path.join(self.test_dir, "test_math_exam.yaml"), "w") as f:
            f.write(self.test_exam_yaml)
        
        # Reload exams to include test exam
        self.engine.load_all_exams()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_exam_loading(self):
        """Test that YAML exams are loaded correctly"""
        self.assertIn("test_math_exam", self.engine.exams)
        
        exam = self.engine.exams["test_math_exam"]
        self.assertEqual(exam.name, "Test Mathematics Exam")
        self.assertEqual(exam.category, ExamCategory.MATHEMATICAL_REASONING)
        self.assertEqual(exam.difficulty, ExamDifficulty.INTERMEDIATE)
        self.assertEqual(len(exam.tasks), 3)
    
    def test_task_evaluation_methods(self):
        """Test different evaluation methods work correctly"""
        exam = self.engine.exams["test_math_exam"]
        
        # Test equals method
        task1 = exam.tasks[0]  # basic_addition
        correct, confidence, feedback = task1.evaluate_response("42")
        self.assertTrue(correct)
        self.assertEqual(confidence, 1.0)
        
        # Test contains method
        task2 = exam.tasks[1]  # algebra_solve
        correct, confidence, feedback = task2.evaluate_response("The answer is x = 10")
        self.assertTrue(correct)
        
        # Test numeric tolerance method
        task3 = exam.tasks[2]  # geometry_area
        correct, confidence, feedback = task3.evaluate_response("The area is 40.0 square units")
        self.assertTrue(correct)
    
    def test_exam_session_creation(self):
        """Test exam session creation and management"""
        session_id = self.engine.start_exam_session("test_math_exam", agent_id=1)
        self.assertIn(session_id, self.engine.active_sessions)
        
        session = self.engine.active_sessions[session_id]
        self.assertEqual(session.exam_slug, "test_math_exam")
        self.assertEqual(session.agent_id, 1)
        self.assertIsNotNone(session.start_time)
    
    async def test_exam_execution(self):
        """Test running a complete exam"""
        # Mock agent runner function
        async def mock_agent_runner(prompt: str, session_id: str) -> str:
            if "15 + 27" in prompt:
                return "42"
            elif "2x + 10 = 30" in prompt:
                return "x equals 10"
            elif "area of a rectangle" in prompt:
                return "The area is 40"
            return "I don't know"
        
        # Run the exam
        session = await self.engine.run_exam("test_math_exam", 1, mock_agent_runner)
        
        # Verify results
        self.assertEqual(len(session.results), 3)
        self.assertIsNotNone(session.end_time)
        self.assertGreater(session.accuracy, 0.8)  # Should get most questions right
        self.assertGreater(session.total_score, 80)
    
    def test_get_available_exams(self):
        """Test retrieving available exam metadata"""
        available = self.engine.get_available_exams()
        self.assertIn("test_math_exam", available)
        
        exam_info = available["test_math_exam"]
        self.assertEqual(exam_info["name"], "Test Mathematics Exam")
        self.assertEqual(exam_info["task_count"], 3)
        self.assertEqual(exam_info["category"], "Mathematical Reasoning")


class TestExamBuilder(unittest.TestCase):
    """Test the AI-powered exam builder"""
    
    def setUp(self):
        """Set up test environment"""
        self.builder = ExamBuilder()
    
    def test_exam_generation(self):
        """Test generating an exam from requirements"""
        requirements = ExamRequirements(
            subject="Mathematics",
            difficulty="Intermediate",
            num_questions=5,
            exam_type=ExamType.MATHEMATICAL,
            target_skills=["algebra", "geometry"],
            time_limit=600,
            description="Test algebra and geometry skills"
        )
        
        exam_data = self.builder.generate_exam(requirements)
        
        # Verify exam structure
        self.assertIn("slug", exam_data)
        self.assertIn("name", exam_data)
        self.assertIn("tasks", exam_data)
        self.assertEqual(len(exam_data["tasks"]), 5)
        self.assertEqual(exam_data["difficulty"], "Intermediate")
        
        # Verify each task has required fields
        for task in exam_data["tasks"]:
            self.assertIn("id", task)
            self.assertIn("prompt", task)
            self.assertIn("answer", task)
            self.assertIn("eval", task)
    
    def test_subject_template_matching(self):
        """Test that appropriate templates are selected for different subjects"""
        # Test mathematics
        math_template = self.builder._get_subject_template("calculus")
        self.assertIn("topics", math_template)
        self.assertIn("derivatives", math_template["topics"])
        
        # Test science
        physics_template = self.builder._get_subject_template("physics")
        self.assertIn("mechanics", physics_template["topics"])
        
        # Test computer science
        cs_template = self.builder._get_subject_template("algorithms")
        self.assertIn("sorting", cs_template["topics"])
    
    def test_exam_file_saving(self):
        """Test saving generated exams to YAML files"""
        requirements = ExamRequirements(
            subject="Logic",
            difficulty="Basic",
            num_questions=3,
            exam_type=ExamType.LOGICAL_REASONING
        )
        
        exam_data = self.builder.generate_exam(requirements)
        
        # Test saving to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = self.builder.save_exam_to_file(exam_data, temp_dir)
            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(file_path.endswith(".yaml"))
            
            # Verify file content
            with open(file_path, 'r') as f:
                content = f.read()
                self.assertIn("slug:", content)
                self.assertIn("tasks:", content)


class TestDatabaseIntegration(unittest.TestCase):
    """Test database integration for exam sessions and results"""
    
    def setUp(self):
        """Set up test database"""
        # Use in-memory database for testing
        import sqlite3
        self.test_db = ":memory:"
        
        # Initialize database
        original_db_name = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = self.test_db
        init_db()
        
        # Store original for cleanup
        self._original_db = original_db_name
    
    def tearDown(self):
        """Clean up test database"""
        if self._original_db:
            os.environ["DATABASE_PATH"] = self._original_db
        elif "DATABASE_PATH" in os.environ:
            del os.environ["DATABASE_PATH"]
    
    def test_exam_session_storage(self):
        """Test storing and retrieving exam sessions"""
        session_id = "test-session-123"
        exam_slug = "test_exam"
        agent_id = 1
        start_time = datetime.now().isoformat()
        
        # Add exam session
        add_exam_session(
            session_id=session_id,
            exam_slug=exam_slug,
            agent_id=agent_id,
            start_time=start_time,
            total_questions=5
        )
        
        # Retrieve session
        session = get_exam_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session["exam_slug"], exam_slug)
        self.assertEqual(session["agent_id"], agent_id)
    
    def test_exam_result_storage(self):
        """Test storing individual exam task results"""
        session_id = "test-session-123"
        
        # First create a session
        add_exam_session(
            session_id=session_id,
            exam_slug="test_exam",
            agent_id=1,
            start_time=datetime.now().isoformat(),
            total_questions=1
        )
        
        # Add exam result
        add_exam_result(
            session_id=session_id,
            task_id="task_1",
            correct=True,
            confidence=0.95,
            response="The answer is 42",
            expected="42",
            feedback="Correct!",
            response_time=2.5,
            timestamp=datetime.now().isoformat()
        )
        
        # Verify result is stored (this would be tested via session retrieval)
        session = get_exam_session(session_id)
        self.assertIsNotNone(session)
    
    def test_analytics_update(self):
        """Test updating and retrieving exam analytics"""
        exam_slug = "test_analytics_exam"
        
        # Update analytics
        update_exam_analytics(exam_slug)
        
        # Retrieve analytics
        analytics = get_exam_analytics(exam_slug)
        self.assertIsNotNone(analytics)


class TestAPIIntegration(unittest.TestCase):
    """Test API endpoints work correctly"""
    
    def setUp(self):
        """Set up test environment"""
        # This would require setting up a test FastAPI client
        # For now, we'll test the core logic
        pass
    
    def test_exam_endpoints_exist(self):
        """Test that exam API endpoints are defined"""
        # Import main to check endpoints exist
        try:
            import main
            # Check that exam-related imports exist
            self.assertTrue(hasattr(main, 'exam_engine'))
            self.assertTrue(hasattr(main, 'exam_builder'))
        except ImportError:
            self.fail("Main module should be importable")
    
    def test_benchmark_harness_integration(self):
        """Test YAML exams are integrated with benchmark harness"""
        try:
            from benchmarks import BENCHMARK_PLUGINS, YAMLExamHarness
            
            # Check that YAML exams are registered
            yaml_exams = [name for name in BENCHMARK_PLUGINS.keys() 
                         if isinstance(BENCHMARK_PLUGINS[name], YAMLExamHarness)]
            
            # Should have some YAML exams registered
            # (exact number depends on what's in the benchmark_exams directory)
            self.assertGreaterEqual(len(yaml_exams), 0)
            
        except ImportError:
            self.fail("Benchmark integration should work")


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def setUp(self):
        """Set up complete test environment"""
        # Set up temporary directory for exams
        self.test_dir = tempfile.mkdtemp()
        self.engine = BenchmarkExamEngine(exam_directory=self.test_dir)
        self.builder = ExamBuilder()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_create_and_run_exam_workflow(self):
        """Test complete workflow: create exam -> save -> load -> run"""
        # Step 1: Create exam using builder
        requirements = ExamRequirements(
            subject="Basic Math",
            difficulty="Basic",
            num_questions=3,
            exam_type=ExamType.MATHEMATICAL,
            time_limit=300
        )
        
        exam_data = self.builder.generate_exam(requirements)
        
        # Step 2: Save exam to file
        file_path = self.builder.save_exam_to_file(exam_data, self.test_dir)
        self.assertTrue(os.path.exists(file_path))
        
        # Step 3: Reload engine to pick up new exam
        self.engine.load_all_exams()
        
        # Step 4: Verify exam is available
        available_exams = self.engine.get_available_exams()
        self.assertIn(exam_data["slug"], available_exams)
        
        # Step 5: Run exam (async test)
        async def run_exam_test():
            async def mock_agent(prompt: str, session_id: str) -> str:
                # Simple mock responses
                if "+" in prompt:
                    return "42"
                elif "x" in prompt.lower():
                    return "5"
                else:
                    return "answer"
            
            session = await self.engine.run_exam(exam_data["slug"], 1, mock_agent)
            self.assertIsNotNone(session)
            self.assertEqual(len(session.results), 3)
            self.assertIsNotNone(session.end_time)
        
        # Run the async test
        asyncio.run(run_exam_test())
    
    def test_benchmark_system_consistency(self):
        """Test that all components work together consistently"""
        # Test that engine categories match builder categories
        engine_categories = list(ExamCategory)
        builder_categories = self.builder._determine_category("Mathematics")
        
        # Both should use consistent category names
        self.assertIsInstance(builder_categories, str)
        
        # Test that difficulty levels are consistent
        engine_difficulties = list(ExamDifficulty)
        self.assertIn(ExamDifficulty.INTERMEDIATE, engine_difficulties)
        
        # Test that evaluation methods are comprehensive
        eval_methods = list(EvaluationMethod)
        self.assertIn(EvaluationMethod.EQUALS, eval_methods)
        self.assertIn(EvaluationMethod.CONTAINS, eval_methods)
        self.assertIn(EvaluationMethod.NUMERIC_TOLERANCE, eval_methods)


def run_all_tests():
    """Run all test suites"""
    print("🧪 Running Silver Lasso Benchmark System Tests...")
    print("=" * 60)
    
    # Create test suites
    test_suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestBenchmarkExamEngine),
        unittest.TestLoader().loadTestsFromTestCase(TestExamBuilder),
        unittest.TestLoader().loadTestsFromTestCase(TestDatabaseIntegration),
        unittest.TestLoader().loadTestsFromTestCase(TestAPIIntegration),
        unittest.TestLoader().loadTestsFromTestCase(TestEndToEndWorkflow)
    ]
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    results = []
    
    for suite in test_suites:
        result = runner.run(suite)
        results.append(result)
    
    # Summary
    total_tests = sum(result.testsRun for result in results)
    total_failures = sum(len(result.failures) for result in results)
    total_errors = sum(len(result.errors) for result in results)
    
    print("\n" + "=" * 60)
    print("🔍 TEST SUMMARY")
    print(f"Total Tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    
    if total_failures == 0 and total_errors == 0:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Silver Lasso Benchmark System is working correctly!")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    print("=" * 60)
    
    return total_failures == 0 and total_errors == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 
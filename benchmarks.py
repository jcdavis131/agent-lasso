"""benchmarks.py
Pluggable benchmark harnesses for evaluating DaivisAgent instances.

Each benchmark should be implemented as a subclass of `BenchmarkHarness` and
registered in `BENCHMARK_PLUGINS`.
"""

from __future__ import annotations

import abc
import uuid
from datetime import datetime
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Base harness
# ---------------------------------------------------------------------------


class BenchmarkHarness(abc.ABC):
    """Abstract base class for a benchmark harness.

    A harness is responsible for executing a benchmark against a given
    `DaivisAgent` instance and returning a dictionary compatible with the
    `/api/benchmark` response schema.  Concrete subclasses should override
    `run`.
    """

    # Human-readable unique name used in API requests (e.g. "response_time")
    name: str

    # Short description displayed in UI / docs
    description: str

    # Optional default timeout (seconds) agents are allowed to run
    default_timeout: int = 600

    @abc.abstractmethod
    def run(self, agent, run_id: str) -> Dict[str, Any]:
        """Execute the benchmark.

        Args:
            agent: The fully initialised `DaivisAgent` instance.
            run_id: A unique id for this benchmark run.

        Returns:
            A dict with at minimum the keys ``score`` (float) and ``details``
            (dict).  Additional metadata is allowed but will be ignored by the
            caller.
        """

    def run_iter(self, agent, run_id: str):
        """Streaming alternative to run(). Default just yields final."""
        yield {"event": "final", "data": self.run(agent, run_id)}


# ---------------------------------------------------------------------------
# Simple built-in benchmarks (keep parity with legacy behaviour)
# ---------------------------------------------------------------------------


class ResponseTimeBenchmark(BenchmarkHarness):
    """Measure raw latency for a trivial arithmetic question."""

    name = "response_time"
    description = "Measure latency for answering a simple arithmetic question."

    def run(self, agent, run_id: str) -> Dict[str, Any]:
        start_time = datetime.now()
        test_message = "What is 2+2?"
        response = agent.run(test_message, run_id)
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        return {
            "score": response_time,
            "details": {
                "response_time_seconds": response_time,
                "test_message": test_message,
                "response": response,
            },
        }


class AccuracyBenchmark(BenchmarkHarness):
    """Basic arithmetic accuracy over three questions."""

    name = "accuracy"
    description = "Arithmetic accuracy over three deterministic math questions."

    _TEST_CASES: List[Dict[str, str]] = [
        {"question": "What is 5 + 3?", "expected": "8"},
        {"question": "What is 10 - 4?", "expected": "6"},
        {"question": "What is 7 * 2?", "expected": "14"},
    ]

    def run(self, agent, run_id: str) -> Dict[str, Any]:
        correct_answers: int = 0
        test_results: List[Dict[str, Any]] = []

        for idx, case in enumerate(self._TEST_CASES):
            sub_run_id = f"{run_id}_{idx}"
            response = agent.run(case["question"], sub_run_id)
            is_correct = str(case["expected"]) in str(response)
            if is_correct:
                correct_answers += 1
            test_results.append(
                {
                    "question": case["question"],
                    "expected": case["expected"],
                    "response": response,
                    "correct": is_correct,
                }
            )

        accuracy: float = correct_answers / len(self._TEST_CASES)
        return {
            "score": accuracy,
            "details": {
                "accuracy": accuracy,
                "correct_answers": correct_answers,
                "total_questions": len(self._TEST_CASES),
                "test_results": test_results,
            },
        }


# ---------------------------------------------------------------------------
# Place-holder for REAL benchmark (deterministic website tasks)
# ---------------------------------------------------------------------------


class REALBenchmark(BenchmarkHarness):
    """Stub for REAL benchmark integration.

    The REAL benchmark (arXiv:2504.11543) requires launching a deterministic
    browser environment.  Full integration will be added in a later patch.  At
    present we return a dummy result so that the API endpoint functions without
    crashing when this benchmark name is requested.
    """

    name = "real_web"
    description = "REAL: deterministic replicas of 11 real websites (stub)."

    def run(self, agent, run_id: str) -> Dict[str, Any]:
        return {
            "score": 0.0,
            "details": {
                "message": "REAL benchmark integration not implemented yet.",
            },
        }


# ---------------------------------------------------------------------------
# Placeholder for AgentBench OS, WebArena, and GTA
# ---------------------------------------------------------------------------


class AgentBenchOSBenchmark(BenchmarkHarness):
    """Integration with AgentBench OS tasks (requires `agentbench` package).

    This harness expects the AgentBench repository to be installed in the
    environment (either via `pip install agentbench` or by adding the repo to
    PYTHONPATH).  It then runs the standard `os-std` task set in evaluation
    mode and measures the success rate reported by AgentBench.
    """

    name = "agentbench_os"
    description = "AgentBench OS: command-line & filesystem tasks."

    def run(self, agent, run_id: str) -> Dict[str, Any]:
        try:
            import importlib
            ab = importlib.import_module("agentbench")
        except ImportError as exc:
            return {
                "score": 0.0,
                "details": {
                    "error": "AgentBench not installed. Please `pip install git+https://github.com/THUDM/AgentBench.git`.",
                    "trace": str(exc),
                },
            }

        # --- Minimal evaluation loop -------------------------------------------------
        from agentbench.envs.os import init_env as init_os_env  # type: ignore
        from agentbench.evaluate import evaluate_agent  # type: ignore

        # Initialize OS environment (standard split)
        env = init_os_env(split="dev", num_parallel=1)

        # Wrap DaivisAgent so it matches AgentBench agent interface
        class _WrapperAgent:
            def __init__(self, davis):
                self.davis = davis

            def act(self, obs: str) -> str:  # AgentBench expects act method
                return self.davis.run(obs, run_id)

        wrapped = _WrapperAgent(agent)

        metrics = evaluate_agent(wrapped, env)
        success_rate = metrics.get("success_rate", 0.0)

        return {
            "score": success_rate,
            "details": metrics,
        }


class WebArenaBenchmark(BenchmarkHarness):
    """Integration with WebArena benchmark (Open-Operator)."""

    name = "webarena"
    description = "WebArena: realistic multi-page web tasks."

    def run(self, agent, run_id: str) -> Dict[str, Any]:
        try:
            import importlib
            oa = importlib.import_module("open_operator")
        except ImportError as exc:
            return {
                "score": 0.0,
                "details": {
                    "error": "Open-Operator / WebArena not installed. `pip install git+https://github.com/All-Hands-AI/open-operator.git`.",
                    "trace": str(exc),
                },
            }

        from open_operator.benchmarks.webarena.runner import evaluate_on_webarena  # type: ignore

        class _WrapperAgent:
            def __init__(self, davis):
                self.davis = davis

            def chat(self, prompt: str) -> str:  # expected proxy API
                return self.davis.run(prompt, run_id)

        wrapped = _WrapperAgent(agent)
        metrics = evaluate_on_webarena(wrapped, split="dev", max_tasks=20)
        return {
            "score": metrics.get("success_rate", 0.0),
            "details": metrics,
        }


class GTABenchmark(BenchmarkHarness):
    """Integration with GTA benchmark."""

    name = "gta"
    description = "GTA: General Tool Agents benchmark."

    def run(self, agent, run_id: str) -> Dict[str, Any]:
        try:
            import importlib, os, json
            gta = importlib.import_module("gta_benchmark")
        except ImportError as exc:
            return {
                "score": 0.0,
                "details": {
                    "error": "GTA benchmark not installed. See https://github.com/open-compass/GTA",
                    "trace": str(exc),
                },
            }

        from gta_benchmark.runner import run_gta  # type: ignore

        class _WrapperAgent:
            def __init__(self, davis):
                self.davis = davis

            def act(self, query: str, tools: list | None = None):
                # GTA expects agent to decide tool usage; simply pass through prompt.
                return self.davis.run(query, run_id)

        wrapped = _WrapperAgent(agent)
        results = run_gta(wrapped, subset="mini")  # run small subset for latency
        score = results.get("overall", 0.0)

        return {
            "score": score,
            "details": results,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _missing_pkg(pkg: str, exc: Exception) -> Dict[str, Any]:
    """Return a standard response when an optional dependency is absent."""
    return {
        "score": 0.0,
        "details": {
            "error": f"{pkg} not installed. Install it to enable this benchmark.",
            "trace": str(exc),
        },
    }


# ---------------------------------------------------------------------------
# Generic static QA harness (lm-eval-harness)
# ---------------------------------------------------------------------------

class QaDatasetHarness(BenchmarkHarness):
    """Runs any lm-eval-harness task with DaivisAgent as the underlying model."""

    def __init__(self, task_name: str, shots: int = 0):
        self.task_name = task_name
        self.shots = shots
        self.name = task_name.lower()
        self.description = f"{task_name} static QA benchmark via lm-eval ({shots}-shot)."

    def run(self, agent, run_id: str) -> Dict[str, Any]:
        try:
            from lm_eval import evaluator, tasks
        except Exception as exc:
            return _missing_pkg("lm-eval-harness", exc)

        # Build callable that satisfies lm-eval's interface
        def _agent_completion(prompt: str, **kwargs):
            return agent.run(prompt, run_id)

        try:
            task_dict = tasks.get_task_dict([self.task_name])
        except KeyError as exc:
            return {
                "score": 0.0,
                "details": {"error": f"Task {self.task_name} not found in lm-eval", "trace": str(exc)},
            }

        results, _ = evaluator.evaluate(
            model=_agent_completion,
            model_args={},
            tasks=task_dict,
            num_fewshot=self.shots,
            limit=None,
            bootstrap_iters=0,
        )

        # Pick first metric ending with "accuracy" or fallback to first value
        score_key = next((k for k in results if k.endswith("accuracy")), None)
        if score_key is None:
            score_key = next(iter(results))
        return {"score": results[score_key], "details": results}

    def run_iter(self, agent, run_id: str):
        try:
            from lm_eval import tasks
        except Exception as exc:
            yield {"event": "error", "data": _missing_pkg("lm-eval-harness", exc)}
            return
        task_obj = tasks.get_task_dict([self.task_name])[self.task_name]
        correct = 0
        total = 0
        for doc in task_obj.validation_docs():
            prompt = task_obj.doc_to_text(doc)
            reference = task_obj.doc_to_target(doc).strip()
            prediction = agent.run(prompt, f"{run_id}_{total}").strip()
            is_correct = reference in prediction
            correct += is_correct
            total += 1
            yield {"event": "step", "data": {"idx": total-1, "question": prompt, "reference": reference, "prediction": prediction, "correct": is_correct}}
        final = {"score": correct / total if total else 0.0, "correct": correct, "total": total}
        yield {"event": "final", "data": final}


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


BENCHMARK_PLUGINS: Dict[str, BenchmarkHarness] = {
    bench.name: bench
    for bench in (
        ResponseTimeBenchmark(),
        AccuracyBenchmark(),
        REALBenchmark(),
        AgentBenchOSBenchmark(),
        WebArenaBenchmark(),
        GTABenchmark(),
    )
}

BENCHMARK_PLUGINS.update({
    # Static QA datasets
    "mmlu": QaDatasetHarness("mmlu", shots=5),
    "arb": QaDatasetHarness("arb"),
    "gpqa": QaDatasetHarness("gpqa"),
    "truthfulqa": QaDatasetHarness("truthfulqa"),
})

# Add import for the new exam engine
from benchmark_exam_engine import exam_engine, ExamSession
from daivis_agent import DaivisAgent

class YAMLExamHarness(BenchmarkHarness):
    """Harness for running YAML-based benchmark exams"""
    
    def __init__(self, exam_slug: str):
        self.exam_slug = exam_slug
        exam = exam_engine.get_exam_by_slug(exam_slug)
        if not exam:
            raise ValueError(f"Exam not found: {exam_slug}")
        
        self.name = exam_slug
        self.description = f"{exam.name} - {exam.description}"
        self.default_timeout = exam.timeout
        self.exam = exam
    
    def run(self, agent: DaivisAgent, run_id: str) -> Dict[str, Any]:
        """Run the YAML exam synchronously"""
        import asyncio
        
        # Create async wrapper for the agent
        async def agent_runner(prompt: str, session_id: str) -> str:
            return agent.run(prompt, session_id)
        
        # Run the exam
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            session = loop.run_until_complete(
                exam_engine.run_exam(self.exam_slug, agent.agent_id, agent_runner)
            )
            
            # Format results for benchmark system
            return {
                "score": session.accuracy,
                "details": {
                    "exam_slug": session.exam_slug,
                    "accuracy": session.accuracy,
                    "total_score": session.total_score,
                    "avg_response_time": session.avg_response_time,
                    "total_questions": len(session.results),
                    "correct_answers": sum(1 for r in session.results if r.correct),
                    "session_id": session.session_id,
                    "results": [
                        {
                            "task_id": r.task_id,
                            "correct": r.correct,
                            "response": r.response,
                            "expected": r.expected,
                            "feedback": r.feedback,
                            "response_time": r.response_time
                        }
                        for r in session.results
                    ]
                }
            }
        finally:
            loop.close()
    
    def run_iter(self, agent: DaivisAgent, run_id: str):
        """Stream benchmark execution step-by-step"""
        import asyncio
        
        # Create async wrapper for the agent
        async def agent_runner(prompt: str, session_id: str) -> str:
            return agent.run(prompt, session_id)
        
        # Run the exam with streaming
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            session_id = exam_engine.start_exam_session(self.exam_slug, agent.agent_id)
            session = exam_engine.active_sessions[session_id]
            
            yield {"event": "start", "data": {"exam_slug": self.exam_slug, "session_id": session_id}}
            
            # Run each task and yield progress
            for i, task in enumerate(self.exam.tasks):
                yield {"event": "task_start", "data": {"task_id": task.id, "prompt": task.prompt}}
                
                start_time = time.time()
                response = agent.run(task.prompt, session_id)
                response_time = time.time() - start_time
                
                correct, confidence, feedback = task.evaluate_response(response)
                
                result = {
                    "task_id": task.id,
                    "correct": correct,
                    "confidence": confidence,
                    "response": response,
                    "expected": task.answer,
                    "feedback": feedback,
                    "response_time": response_time,
                    "progress": (i + 1) / len(self.exam.tasks)
                }
                
                yield {"event": "task_complete", "data": result}
            
            # Calculate final metrics
            session.end_time = datetime.now()
            session.calculate_metrics()
            
            final_result = {
                "score": session.accuracy,
                "accuracy": session.accuracy,
                "total_score": session.total_score,
                "avg_response_time": session.avg_response_time,
                "total_questions": len(session.results),
                "correct_answers": sum(1 for r in session.results if r.correct)
            }
            
            yield {"event": "final", "data": final_result}
            
        finally:
            loop.close()

# Add all YAML exams to the benchmark plugins
def register_yaml_exams():
    """Register all YAML exams as benchmark plugins"""
    available_exams = exam_engine.get_available_exams()
    
    for exam_slug, exam_info in available_exams.items():
        try:
            harness = YAMLExamHarness(exam_slug)
            BENCHMARK_PLUGINS[exam_slug] = harness
        except Exception as e:
            logger.error(f"Failed to register YAML exam {exam_slug}: {e}")

# Register YAML exams when module is imported
register_yaml_exams() 
"""
ExamBuilder Agent - Intelligent Exam Generation for Silver Lasso
Generates comprehensive benchmark exams based on user specifications
"""

import yaml
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)

class ExamType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    CODING = "coding"
    MATHEMATICAL = "mathematical"
    LOGICAL_REASONING = "logical_reasoning"
    COMPREHENSION = "comprehension"

@dataclass
class ExamRequirements:
    """Requirements for generating an exam"""
    subject: str
    difficulty: str
    num_questions: int
    exam_type: ExamType
    target_skills: List[str] = field(default_factory=list)
    domain_knowledge: List[str] = field(default_factory=list)
    time_limit: int = 600
    description: str = ""
    source_materials: List[str] = field(default_factory=list)
    
class ExamBuilder:
    """AI-powered exam generation system"""
    
    def __init__(self):
        self.exam_templates = self._load_exam_templates()
        self.question_patterns = self._load_question_patterns()
        
    def _load_exam_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load exam templates for different subjects and types"""
        return {
            'mathematics': {
                'calculus': {
                    'topics': ['derivatives', 'integrals', 'limits', 'applications'],
                    'difficulty_progression': ['basic_rules', 'chain_rule', 'integration_by_parts', 'optimization'],
                    'question_types': ['computation', 'application', 'proof']
                },
                'algebra': {
                    'topics': ['polynomials', 'equations', 'inequalities', 'functions'],
                    'difficulty_progression': ['linear', 'quadratic', 'polynomial', 'rational'],
                    'question_types': ['solve', 'graph', 'analyze']
                },
                'statistics': {
                    'topics': ['distributions', 'hypothesis_testing', 'regression', 'probability'],
                    'difficulty_progression': ['descriptive', 'inferential', 'multivariate', 'bayesian'],
                    'question_types': ['calculate', 'interpret', 'design']
                }
            },
            'science': {
                'physics': {
                    'topics': ['mechanics', 'thermodynamics', 'electromagnetism', 'quantum'],
                    'difficulty_progression': ['kinematics', 'dynamics', 'fields', 'modern'],
                    'question_types': ['calculation', 'conceptual', 'experimental']
                },
                'chemistry': {
                    'topics': ['atomic_structure', 'bonding', 'reactions', 'thermochemistry'],
                    'difficulty_progression': ['basic_concepts', 'stoichiometry', 'equilibrium', 'kinetics'],
                    'question_types': ['mechanism', 'prediction', 'analysis']
                },
                'biology': {
                    'topics': ['cell_biology', 'genetics', 'ecology', 'evolution'],
                    'difficulty_progression': ['structure', 'function', 'interaction', 'systems'],
                    'question_types': ['identification', 'explanation', 'comparison']
                }
            },
            'computer_science': {
                'algorithms': {
                    'topics': ['sorting', 'searching', 'graph_algorithms', 'dynamic_programming'],
                    'difficulty_progression': ['basic', 'intermediate', 'advanced', 'expert'],
                    'question_types': ['implementation', 'analysis', 'optimization']
                },
                'data_structures': {
                    'topics': ['arrays', 'linked_lists', 'trees', 'graphs'],
                    'difficulty_progression': ['basic_ops', 'traversal', 'manipulation', 'optimization'],
                    'question_types': ['implementation', 'design', 'analysis']
                }
            },
            'reasoning': {
                'logical': {
                    'topics': ['propositional_logic', 'predicate_logic', 'inference', 'proofs'],
                    'difficulty_progression': ['basic_logic', 'compound_statements', 'quantifiers', 'formal_proofs'],
                    'question_types': ['truth_tables', 'logical_equivalence', 'proof_construction']
                },
                'critical_thinking': {
                    'topics': ['argument_analysis', 'fallacies', 'evidence_evaluation', 'decision_making'],
                    'difficulty_progression': ['identification', 'analysis', 'evaluation', 'construction'],
                    'question_types': ['analyze', 'evaluate', 'construct']
                }
            }
        }
    
    def _load_question_patterns(self) -> Dict[str, List[str]]:
        """Load question patterns for different types of questions"""
        return {
            'mathematical_computation': [
                "Calculate {expression}",
                "Find the value of {variable} when {condition}",
                "Solve for {variable} in the equation {equation}",
                "Evaluate {integral_or_derivative}",
                "Determine the {property} of {mathematical_object}"
            ],
            'conceptual_understanding': [
                "Explain why {concept} is important in {context}",
                "What happens when {condition} changes in {system}?",
                "Compare and contrast {concept_a} and {concept_b}",
                "Describe the relationship between {variable_a} and {variable_b}",
                "What are the implications of {assumption} for {conclusion}?"
            ],
            'problem_solving': [
                "A {scenario} has the following properties: {properties}. Find {target}",
                "Given {constraints}, what is the optimal {solution}?",
                "How would you approach solving {complex_problem}?",
                "Design a method to {objective} under {constraints}",
                "What strategy would you use to {goal}?"
            ],
            'analysis_evaluation': [
                "Analyze the strengths and weaknesses of {approach}",
                "Evaluate the effectiveness of {method} for {purpose}",
                "What are the potential risks of {decision}?",
                "How would you improve {system} to {objective}?",
                "What evidence supports or contradicts {claim}?"
            ]
        }
    
    def generate_exam(self, requirements: ExamRequirements) -> Dict[str, Any]:
        """Generate a complete exam based on requirements"""
        logger.info(f"Generating exam for {requirements.subject} at {requirements.difficulty} level")
        
        # Get subject template
        subject_template = self._get_subject_template(requirements.subject)
        
        # Generate questions
        questions = self._generate_questions(requirements, subject_template)
        
        # Create exam structure
        exam_data = {
            'slug': self._generate_slug(requirements),
            'name': self._generate_exam_name(requirements),
            'description': requirements.description or self._generate_description(requirements),
            'category': self._determine_category(requirements.subject),
            'difficulty': requirements.difficulty,
            'timeout': requirements.time_limit,
            'source': f"Generated by ExamBuilder on {datetime.now().strftime('%Y-%m-%d')}",
            'metadata': {
                'generated_by': 'ExamBuilder',
                'requirements': requirements.__dict__,
                'generation_timestamp': datetime.now().isoformat()
            },
            'tasks': questions
        }
        
        return exam_data
    
    def _get_subject_template(self, subject: str) -> Dict[str, Any]:
        """Get template for the specified subject"""
        subject_lower = subject.lower()
        
        # Try to find exact match
        for domain, subjects in self.exam_templates.items():
            if subject_lower in subjects:
                return subjects[subject_lower]
        
        # Try to find partial match
        for domain, subjects in self.exam_templates.items():
            for subj_key, template in subjects.items():
                if subject_lower in subj_key or subj_key in subject_lower:
                    return template
        
        # Default template
        return {
            'topics': ['general_knowledge', 'problem_solving', 'analysis'],
            'difficulty_progression': ['basic', 'intermediate', 'advanced'],
            'question_types': ['conceptual', 'analytical', 'application']
        }
    
    def _generate_questions(self, requirements: ExamRequirements, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate questions based on requirements and template"""
        questions = []
        topics = template.get('topics', ['general'])
        question_types = template.get('question_types', ['conceptual'])
        
        questions_per_topic = max(1, requirements.num_questions // len(topics))
        
        for i, topic in enumerate(topics):
            if len(questions) >= requirements.num_questions:
                break
                
            # Determine how many questions for this topic
            remaining_questions = requirements.num_questions - len(questions)
            topic_questions = min(questions_per_topic, remaining_questions)
            
            for j in range(topic_questions):
                question = self._generate_single_question(
                    topic, 
                    requirements, 
                    template, 
                    f"{topic}_{j+1}"
                )
                questions.append(question)
        
        return questions
    
    def _generate_single_question(self, topic: str, requirements: ExamRequirements, 
                                 template: Dict[str, Any], question_id: str) -> Dict[str, Any]:
        """Generate a single question for a specific topic"""
        
        # Select question type
        question_types = template.get('question_types', ['conceptual'])
        question_type = question_types[hash(question_id) % len(question_types)]
        
        # Generate question based on subject and type
        question_data = self._create_question_content(
            requirements.subject, 
            topic, 
            question_type, 
            requirements.difficulty,
            question_id
        )
        
        return {
            'id': question_id,
            'prompt': question_data['prompt'],
            'answer': question_data['answer'],
            'eval': question_data.get('eval', 'contains'),
            'explanation': question_data.get('explanation', ''),
            'tags': [topic, question_type, requirements.difficulty],
            'difficulty_weight': self._calculate_difficulty_weight(requirements.difficulty)
        }
    
    def _create_question_content(self, subject: str, topic: str, question_type: str, 
                               difficulty: str, question_id: str) -> Dict[str, Any]:
        """Create actual question content"""
        
        # This is where we'd integrate with an LLM to generate sophisticated questions
        # For now, using templates and patterns
        
        if 'math' in subject.lower():
            return self._generate_math_question(topic, question_type, difficulty, question_id)
        elif 'science' in subject.lower():
            return self._generate_science_question(topic, question_type, difficulty, question_id)
        elif 'computer' in subject.lower() or 'programming' in subject.lower():
            return self._generate_cs_question(topic, question_type, difficulty, question_id)
        elif 'reasoning' in subject.lower() or 'logic' in subject.lower():
            return self._generate_reasoning_question(topic, question_type, difficulty, question_id)
        else:
            return self._generate_general_question(topic, question_type, difficulty, question_id)
    
    def _generate_math_question(self, topic: str, question_type: str, difficulty: str, question_id: str) -> Dict[str, Any]:
        """Generate mathematics questions"""
        questions = {
            'calculus': {
                'basic': {
                    'prompt': "Find the derivative of f(x) = x³ + 2x² - 5x + 3",
                    'answer': "3x² + 4x - 5",
                    'eval': 'contains',
                    'explanation': "Using the power rule: d/dx(x³) = 3x², d/dx(2x²) = 4x, d/dx(-5x) = -5, d/dx(3) = 0"
                },
                'intermediate': {
                    'prompt': "Find the integral of ∫(2x + 1)e^x dx",
                    'answer': "(2x - 1)e^x + C",
                    'eval': 'contains',
                    'explanation': "Using integration by parts with u = 2x + 1, dv = e^x dx"
                }
            },
            'algebra': {
                'basic': {
                    'prompt': "Solve for x: 2x² - 8x + 6 = 0",
                    'answer': "x = 1 or x = 3",
                    'eval': 'contains',
                    'explanation': "Using the quadratic formula or factoring: 2(x-1)(x-3) = 0"
                }
            }
        }
        
        if topic in questions and difficulty in questions[topic]:
            return questions[topic][difficulty]
        
        return {
            'prompt': f"Solve the following {topic} problem at {difficulty} level: [Problem details would be generated here]",
            'answer': "[Answer would be generated here]",
            'eval': 'contains',
            'explanation': "[Explanation would be generated here]"
        }
    
    def _generate_science_question(self, topic: str, question_type: str, difficulty: str, question_id: str) -> Dict[str, Any]:
        """Generate science questions"""
        questions = {
            'physics': {
                'basic': {
                    'prompt': "A ball is thrown upward with initial velocity 20 m/s. What is its velocity after 2 seconds? (g = 9.8 m/s²)",
                    'answer': "0.4 m/s",
                    'eval': 'contains',
                    'explanation': "Using v = v₀ - gt: v = 20 - 9.8(2) = 20 - 19.6 = 0.4 m/s"
                }
            },
            'chemistry': {
                'basic': {
                    'prompt': "What is the molecular formula for glucose?",
                    'answer': "C₆H₁₂O₆",
                    'eval': 'contains',
                    'explanation': "Glucose has 6 carbon atoms, 12 hydrogen atoms, and 6 oxygen atoms"
                }
            }
        }
        
        if topic in questions and difficulty in questions[topic]:
            return questions[topic][difficulty]
        
        return {
            'prompt': f"Answer this {topic} question at {difficulty} level: [Question would be generated here]",
            'answer': "[Answer would be generated here]",
            'eval': 'contains',
            'explanation': "[Explanation would be generated here]"
        }
    
    def _generate_cs_question(self, topic: str, question_type: str, difficulty: str, question_id: str) -> Dict[str, Any]:
        """Generate computer science questions"""
        return {
            'prompt': f"What is the time complexity of {topic} operation in the worst case?",
            'answer': "O(n)",
            'eval': 'contains',
            'explanation': f"The time complexity depends on the specific implementation and data structure used for {topic}"
        }
    
    def _generate_reasoning_question(self, topic: str, question_type: str, difficulty: str, question_id: str) -> Dict[str, Any]:
        """Generate logical reasoning questions"""
        return {
            'prompt': f"If all A are B, and all B are C, what can we conclude about A and C?",
            'answer': "All A are C",
            'eval': 'contains',
            'explanation': "This is a valid syllogism using the transitivity of the universal quantifier"
        }
    
    def _generate_general_question(self, topic: str, question_type: str, difficulty: str, question_id: str) -> Dict[str, Any]:
        """Generate general knowledge questions"""
        return {
            'prompt': f"Explain the concept of {topic} and provide an example.",
            'answer': f"[Answer about {topic}]",
            'eval': 'contains',
            'explanation': f"This question tests understanding of {topic} concepts"
        }
    
    def _calculate_difficulty_weight(self, difficulty: str) -> float:
        """Calculate difficulty weight for scoring"""
        weights = {
            'basic': 1.0,
            'intermediate': 1.5,
            'advanced': 2.0,
            'expert': 2.5,
            'college level': 2.0
        }
        return weights.get(difficulty.lower(), 1.0)
    
    def _generate_slug(self, requirements: ExamRequirements) -> str:
        """Generate a unique slug for the exam"""
        subject_slug = requirements.subject.lower().replace(' ', '_')
        difficulty_slug = requirements.difficulty.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{subject_slug}_{difficulty_slug}_{timestamp}"
    
    def _generate_exam_name(self, requirements: ExamRequirements) -> str:
        """Generate a descriptive name for the exam"""
        return f"{requirements.subject} - {requirements.difficulty.title()} Level Assessment"
    
    def _generate_description(self, requirements: ExamRequirements) -> str:
        """Generate a description for the exam"""
        return f"A comprehensive {requirements.difficulty.lower()} level assessment covering {requirements.subject} with {requirements.num_questions} questions. Generated to test {', '.join(requirements.target_skills) if requirements.target_skills else 'core concepts'}."
    
    def _determine_category(self, subject: str) -> str:
        """Determine the category based on subject"""
        category_map = {
            'mathematics': 'Mathematical Reasoning',
            'math': 'Mathematical Reasoning',
            'physics': 'Scientific Reasoning',
            'chemistry': 'Scientific Reasoning',
            'biology': 'Scientific Reasoning',
            'science': 'Scientific Reasoning',
            'computer science': 'Technical & Coding',
            'programming': 'Technical & Coding',
            'logic': 'Logical Reasoning',
            'reasoning': 'Logical Reasoning',
            'philosophy': 'Logical Reasoning',
            'language': 'Language Understanding',
            'literature': 'Language Understanding',
            'history': 'Academic Knowledge',
            'geography': 'Academic Knowledge',
            'economics': 'Academic Knowledge'
        }
        
        subject_lower = subject.lower()
        for key, category in category_map.items():
            if key in subject_lower:
                return category
        
        return 'Academic Knowledge'
    
    def save_exam_to_file(self, exam_data: Dict[str, Any], output_dir: str = "benchmark_exams") -> str:
        """Save generated exam to YAML file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = f"{exam_data['slug']}.yaml"
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(exam_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved exam to {file_path}")
        return str(file_path)
    
    def generate_and_save_exam(self, requirements: ExamRequirements, output_dir: str = "benchmark_exams") -> str:
        """Generate exam and save it to file"""
        exam_data = self.generate_exam(requirements)
        return self.save_exam_to_file(exam_data, output_dir)

# Global instance
exam_builder = ExamBuilder()

# Convenience function for API use
def create_exam_from_requirements(
    subject: str,
    difficulty: str,
    num_questions: int,
    exam_type: str = "multiple_choice",
    target_skills: List[str] = None,
    time_limit: int = 600,
    description: str = ""
) -> str:
    """Create an exam from basic requirements"""
    
    requirements = ExamRequirements(
        subject=subject,
        difficulty=difficulty,
        num_questions=num_questions,
        exam_type=ExamType(exam_type),
        target_skills=target_skills or [],
        time_limit=time_limit,
        description=description
    )
    
    return exam_builder.generate_and_save_exam(requirements) 
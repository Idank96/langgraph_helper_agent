from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict
import re


class LLMJudgeEvaluator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    def _get_score(self, prompt: str) -> float:
        try:
            response = self.llm.invoke(prompt).content.strip()

            # Try to extract from <score>X.XX</score> tags first
            match = re.search(r'<score>([\d.]+)</score>', response)
            if match:
                score_str = match.group(1)
            else:
                # Fallback: try to find any float pattern in the response
                match = re.search(r'\b(\d+\.\d+)\b', response)
                if match:
                    score_str = match.group(1)
                else:
                    # Last resort: try direct parsing
                    score_str = response

            # Validate it's a valid float (only digits and decimal point)
            if not re.match(r'^\d+\.?\d*$', score_str):
                raise ValueError(f"Invalid score format: {score_str}")

            score = float(score_str)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Warning: Failed to parse score from response: '{response[:100]}'. Error: {e}")
            return 0.0

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""Act as a meticulous fact-checker who critically verifies every claim against the source material. Your task is to identify any inconsistencies, unsupported statements, or hallucinations.

                    Rate faithfulness of answer to context (0.0-1.0).
                    Context: {context}
                    Answer: {answer}

                    Return ONLY a decimal number between 0.0 and 1.0 (e.g., 0.75).
                    Do NOT include any explanations, reasoning, or additional text.
                    Format: <score>0.XX</score>"""
        return self._get_score(prompt)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""Act as a strict relevance analyst who demands that answers directly address the specific question asked. Be critical of tangential information, unnecessary details, or responses that miss the core intent.

                Rate answer relevancy to question (0.0-1.0).
                Question: {question}
                Answer: {answer}

                Return ONLY a decimal number between 0.0 and 1.0 (e.g., 0.75).
                Do NOT include any explanations, reasoning, or additional text.
                Format: <score>0.XX</score>"""
        return self._get_score(prompt)

    def evaluate_context_precision(self, question: str, context: str) -> float:
        prompt = f"""Act as a discerning information quality assessor who evaluates whether the provided context is precise and directly useful for answering the question. Be critical of irrelevant information, noise, or overly broad context.

                Rate context precision for question (0.0-1.0).
                Question: {question}
                Context: {context}

                Return ONLY a decimal number between 0.0 and 1.0 (e.g., 0.75).
                Do NOT include any explanations, reasoning, or additional text.
                Format: <score>0.XX</score>"""
        return self._get_score(prompt)

    def evaluate_all(self, question: str, context: str, answer: str) -> Dict[str, float]:
        return {
            "faithfulness": self.evaluate_faithfulness(answer, context),
            "answer_relevancy": self.evaluate_answer_relevancy(question, answer),
            "context_precision": self.evaluate_context_precision(question, context)
        }

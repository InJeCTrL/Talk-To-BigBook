"""LLM summarizer wrappers for Layered Summary System"""

import logging
import os
import requests
from typing import List
from google import genai

from .models import Node
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


class DashScopeSummarizer:
    """Wrapper for Aliyun DashScope API for online summarization."""
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "qwen2-7b-instruct",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        """
        Initialize DashScope client.
        
        Args:
            api_key: DashScope API key (or set DASHSCOPE_API_KEY env var)
            model: Model name (qwen2-7b-instruct, qwen-turbo, qwen-plus, etc.)
            base_url: DashScope API base URL
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key required. Set DASHSCOPE_API_KEY env var or pass api_key.")
        
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/chat/completions"
        
        logger.info(f"DashScopeSummarizer initialized: model={model}")
    
    def _build_prompt(self, text: str, language: str) -> str:
        """Build summarization prompt."""
        if language == "zh":
            return f"""请总结以下文本，重点包括：
- 关键实体（人物、组织、地点）
- 重要事件及其顺序
- 时间引用和时间关系
- 空间/位置信息
- 主要主题和概念

保持总结简洁但保留关键细节。

文本：
{text}

总结："""
        else:
            return f"""Summarize the following text, emphasizing:
- Key entities (people, organizations, locations)
- Important events and their sequence
- Time references and temporal relationships
- Spatial/location information
- Main themes and concepts

Keep the summary concise but preserve critical details.

Text:
{text}

Summary:"""
    
    @retry_with_backoff(max_retries=3, delays=[1, 2, 4])
    def summarize(self, text: str, language: str = "en", max_tokens: int = 500) -> str:
        """
        Generate summary using DashScope.
        
        Args:
            text: Text to summarize
            language: "en" or "zh" for language-specific prompts
            max_tokens: Maximum tokens in summary
            
        Returns:
            Summary text
        """
        prompt = self._build_prompt(text, language)
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()
            
            if not summary:
                raise ValueError("Empty summary returned from DashScope")
            
            logger.debug(f"Generated summary: {len(summary)} characters")
            return summary
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DashScope API request failed: {e}")
            raise Exception(f"DashScope API error: {e}")
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise


class OllamaSummarizer:
    """Wrapper for Ollama API for offline summarization."""
    
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            model: Ollama model name
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        logger.info(f"OllamaSummarizer initialized: model={model}, base_url={base_url}")
    
    def _build_prompt(self, text: str, language: str) -> str:
        """
        Build summarization prompt emphasizing key entities, events, time, location.
        
        Args:
            text: Text to summarize
            language: "en" or "zh"
            
        Returns:
            Formatted prompt
        """
        if language == "zh":
            return f"""请总结以下文本，重点包括：
- 关键实体（人物、组织、地点）
- 重要事件及其顺序
- 时间引用和时间关系
- 空间/位置信息
- 主要主题和概念

保持总结简洁但保留关键细节。

文本：
{text}

总结："""
        else:
            return f"""Summarize the following text, emphasizing:
- Key entities (people, organizations, locations)
- Important events and their sequence
- Time references and temporal relationships
- Spatial/location information
- Main themes and concepts

Keep the summary concise but preserve critical details.

Text:
{text}

Summary:"""
    
    @retry_with_backoff(max_retries=3, delays=[1, 2, 4])
    def summarize(self, text: str, language: str = "en", max_tokens: int = 500) -> str:
        """
        Generate summary using Ollama.
        
        Args:
            text: Text to summarize
            language: "en" or "zh" for language-specific prompts
            max_tokens: Maximum tokens in summary
            
        Returns:
            Summary text
            
        Raises:
            Exception: If API call fails after retries
        """
        prompt = self._build_prompt(text, language)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            summary = result.get("response", "").strip()
            
            if not summary:
                raise ValueError("Empty summary returned from Ollama")
            
            logger.debug(f"Generated summary: {len(summary)} characters")
            return summary
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise Exception(f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise


class GeminiRetriever:
    """Wrapper for Gemini API for online query processing."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key
            model: Gemini model name
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        
        logger.info(f"GeminiRetriever initialized: model={model}")
    
    def _build_node_selection_prompt(self, query: str, nodes: List[Node], level: int) -> str:
        """
        Build prompt for node selection.
        
        Args:
            query: User query
            nodes: Available nodes
            level: Current level
            
        Returns:
            Formatted prompt
        """
        summaries_text = "\n\n".join([
            f"ID: {node.id}\nContent: {node.text[:500]}..."  # Truncate for brevity
            for node in nodes
        ])
        
        return f"""Given the user query and the following summaries at Level {level}, select the IDs of the most relevant summaries that would help answer the query.

Query: {query}

Available summaries:
{summaries_text}

Return only the selected IDs as a JSON array, e.g., ["L{level}_0", "L{level}_3"]

Selected IDs:"""
    
    def _build_answer_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Build prompt for answer generation.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            
        Returns:
            Formatted prompt
        """
        contexts_text = "\n\n---\n\n".join(contexts)
        
        return f"""Based on the following context excerpts from a document, answer the user's question. If the context doesn't contain enough information, say so.

Question: {query}

Context:
{contexts_text}

Answer:"""
    
    @retry_with_backoff(max_retries=3, delays=[1, 2, 4])
    def select_relevant_nodes(self, query: str, nodes: List[Node], level: int) -> List[str]:
        """
        Ask LLM to select most relevant nodes for query.
        
        Args:
            query: User question
            nodes: Available nodes at current level
            level: Current level (3, 2, or 1)
            
        Returns:
            List of selected node IDs
            
        Raises:
            Exception: If API call fails after retries
        """
        prompt = self._build_node_selection_prompt(query, nodes, level)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            result_text = response.text.strip()
            
            # Parse JSON array from response
            import json
            import re
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*?\]', result_text, re.DOTALL)
            if json_match:
                selected_ids = json.loads(json_match.group())
            else:
                # Fallback: split by comma
                selected_ids = [id.strip().strip('"\'') for id in result_text.split(',')]
            
            logger.info(f"Selected {len(selected_ids)} nodes at level {level}: {selected_ids}")
            return selected_ids
            
        except Exception as e:
            logger.error(f"Node selection failed: {e}")
            raise Exception(f"Gemini API error during node selection: {e}")
    
    @retry_with_backoff(max_retries=3, delays=[1, 2, 4])
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate final answer based on retrieved contexts.
        
        Args:
            query: User question
            contexts: Retrieved text chunks
            
        Returns:
            Answer text
            
        Raises:
            Exception: If API call fails after retries
        """
        prompt = self._build_answer_prompt(query, contexts)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            answer = response.text.strip()
            
            if not answer:
                raise ValueError("Empty answer returned from Gemini")
            
            logger.info(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise Exception(f"Gemini API error during answer generation: {e}")

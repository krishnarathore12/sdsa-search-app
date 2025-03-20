from config.config import CONFIG
from loguru import logger
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere


class LLMManager:
    _llm_instance: Optional[BaseChatModel] = None

    @classmethod
    def get_llm(cls) -> BaseChatModel:
        """Returns a singleton LLM instance, using config settings."""
        if cls._llm_instance is None:
            llm_config = CONFIG.llm
            provider = llm_config.get("provider", "groq").lower()

            logger.info(f"Using {provider} LLM provider")
            
            if provider == "openai":
                cls._llm_instance = ChatOpenAI(
                    api_key=CONFIG.openai_api_key,
                    model=llm_config.get("model", "gpt-4-turbo-preview"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1000)
                )
            elif provider == "anthropic":
                cls._llm_instance = ChatAnthropic(
                    api_key=CONFIG.anthropic_api_key,
                    model=llm_config.get("model", "claude-3-opus-20240229"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1000)
                )
            elif provider == "google":
                cls._llm_instance = ChatGoogleGenerativeAI(
                    api_key=CONFIG.google_api_key,
                    model=llm_config.get("model", "gemini-pro"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_output_tokens=llm_config.get("max_tokens", 1000)
                )
            elif provider == "cohere":
                cls._llm_instance = ChatCohere(
                    api_key=CONFIG.cohere_api_key,
                    model=llm_config.get("model", "command-r-plus"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1000)
                )
            else:  # Default to Groq
                cls._llm_instance = ChatGroq(
                    api_key=CONFIG.groq_api_key,
                    model=llm_config.get("model", "llama3-70b-8192"),
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 1000)
                )

        return cls._llm_instance

    @classmethod
    def reset_llm(cls) -> None:
        """Reset the LLM instance to allow reconfiguration."""
        cls._llm_instance = None

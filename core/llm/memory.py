"""
Conversation memory management system
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, max_tokens: int = 1000):
        """
        Initialize conversation memory
        
        Args:
            max_tokens: Maximum tokens to keep in memory
        """
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, Any]] = []
        self.total_tokens = 0
    
    def add_message(self, human_message: str, ai_message: str):
        """
        Add a conversation turn to memory
        
        Args:
            human_message: User's message
            ai_message: AI's response
        """
        timestamp = datetime.now().isoformat()
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        human_tokens = len(human_message) // 4
        ai_tokens = len(ai_message) // 4
        
        message_entry = {
            "timestamp": timestamp,
            "human": human_message,
            "ai": ai_message,
            "tokens": human_tokens + ai_tokens
        }
        
        self.messages.append(message_entry)
        self.total_tokens += message_entry["tokens"]
        
        # Trim old messages if we exceed token limit
        self._trim_memory()
        
        logger.debug(f"Added message to memory. Total messages: {len(self.messages)}, Total tokens: {self.total_tokens}")
    
    def _trim_memory(self):
        """Remove old messages to stay within token limit"""
        while self.total_tokens > self.max_tokens and len(self.messages) > 1:
            removed_message = self.messages.pop(0)
            self.total_tokens -= removed_message["tokens"]
            logger.debug(f"Trimmed old message from memory. Remaining messages: {len(self.messages)}")
    
    def get_conversation_string(self, include_timestamps: bool = False, max_exchanges: int = 5) -> str:
        """
        Get conversation history as a formatted string
        
        Args:
            include_timestamps: Whether to include timestamps
            max_exchanges: Maximum number of recent exchanges to include
            
        Returns:
            Formatted conversation history
        """
        if not self.messages:
            return ""
        
        # Get recent messages
        recent_messages = self.messages[-max_exchanges:] if max_exchanges > 0 else self.messages
        
        conversation_parts = []
        for msg in recent_messages:
            if include_timestamps:
                conversation_parts.append(f"[{msg['timestamp']}]")
            conversation_parts.append(f"Human: {msg['human']}")
            conversation_parts.append(f"Assistant: {msg['ai']}")
            conversation_parts.append("")  # Empty line for readability
        
        return "\n".join(conversation_parts).strip()
    
    def get_summary(self) -> str:
        """
        Get a summary of the conversation
        
        Returns:
            Conversation summary
        """
        if not self.messages:
            return "No conversation history."
        
        total_exchanges = len(self.messages)
        recent_topics = []
        
        # Extract key topics from recent messages
        for msg in self.messages[-3:]:
            if len(msg["human"]) > 20:  # Focus on substantial messages
                topic = msg["human"][:50]
                if len(msg["human"]) > 50:
                    topic += "..."
                recent_topics.append(topic)
        
        summary = f"Conversation history: {total_exchanges} exchanges. "
        if recent_topics:
            summary += f"Recent topics: {'; '.join(recent_topics)}"
        
        return summary
    
    def save_to_file(self, session_id: str, conversations_dir: str = "data/conversations"):
        """
        Save conversation to file
        
        Args:
            session_id: Session identifier
            conversations_dir: Directory to save conversations
        """
        try:
            Path(conversations_dir).mkdir(parents=True, exist_ok=True)
            filepath = Path(conversations_dir) / f"{session_id}.json"
            
            conversation_data = {
                "session_id": session_id,
                "messages": self.messages,
                "total_tokens": self.total_tokens,
                "message_count": len(self.messages),
                "saved_at": datetime.now().isoformat(),
                "summary": self.get_summary()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Conversation saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
    
    def load_from_file(self, session_id: str, conversations_dir: str = "data/conversations"):
        """
        Load conversation from file
        
        Args:
            session_id: Session identifier
            conversations_dir: Directory containing conversations
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            filepath = Path(conversations_dir) / f"{session_id}.json"
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.messages = data.get("messages", [])
                self.total_tokens = data.get("total_tokens", 0)
                
                logger.info(f"Conversation loaded from {filepath} ({len(self.messages)} messages)")
                return True
            else:
                logger.debug(f"No saved conversation found for session {session_id}")
                return False
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False
    
    def clear(self):
        """Clear all conversation history"""
        self.messages = []
        self.total_tokens = 0
        logger.info("Conversation memory cleared")
    
    def get_stats(self) -> dict:
        """Get memory statistics"""
        return {
            "total_messages": len(self.messages),
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "memory_usage_percent": (self.total_tokens / self.max_tokens) * 100 if self.max_tokens > 0 else 0
        }

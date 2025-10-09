"""
Model layer for Human Review System (MVC Pattern).
Contains data structures, business logic, and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import uuid

from utils.logger import get_logger

logger = get_logger(__name__)


class ReviewDecision(Enum):
    """Available review decisions."""
    CONTINUE = "continue"
    RETRIEVE_MORE = "retrieve_more"
    STOP_ERROR = "stop_error"


class ReviewStatus(Enum):
    """Review session status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ReviewContext:
    """Context information for human review."""
    question: str
    context: str
    retrieved_docs_count: int = 0
    web_results_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    iteration_count: int = 1
    
    def get_context_preview(self, max_chars: int = 500) -> str:
        """Get a preview of the context."""
        if len(self.context) <= max_chars:
            return self.context
        return self.context[:max_chars] + "..."
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the context."""
        return {
            "context_length": len(self.context),
            "word_count": len(self.context.split()),
            "retrieved_docs": self.retrieved_docs_count,
            "web_results": self.web_results_count,
            "iteration": self.iteration_count,
            "has_metadata": bool(self.metadata)
        }


@dataclass
class ReviewSession:
    """Represents a complete review session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[ReviewContext] = None
    status: ReviewStatus = ReviewStatus.PENDING
    decision: Optional[ReviewDecision] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300  # 5 minutes default
    error_message: Optional[str] = None
    user_notes: Optional[str] = None
    
    def start_review(self) -> None:
        """Start the review session."""
        self.status = ReviewStatus.IN_PROGRESS
        logger.info(f"Review session {self.session_id} started")
    
    def complete_review(self, decision: ReviewDecision, user_notes: Optional[str] = None) -> None:
        """Complete the review session with a decision."""
        self.decision = decision
        self.status = ReviewStatus.COMPLETED
        self.completed_at = datetime.now()
        self.user_notes = user_notes
        logger.info(f"Review session {self.session_id} completed with decision: {decision.value}")
    
    def timeout_review(self) -> None:
        """Mark the review session as timed out."""
        self.status = ReviewStatus.TIMEOUT
        self.completed_at = datetime.now()
        self.decision = ReviewDecision.CONTINUE  # Default fallback
        logger.warning(f"Review session {self.session_id} timed out")
    
    def error_review(self, error_message: str) -> None:
        """Mark the review session as errored."""
        self.status = ReviewStatus.ERROR
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.decision = ReviewDecision.CONTINUE  # Default fallback
        logger.error(f"Review session {self.session_id} errored: {error_message}")
    
    def is_active(self) -> bool:
        """Check if the review session is still active."""
        return self.status in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]
    
    def get_duration(self) -> Optional[float]:
        """Get the duration of the review session in seconds."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None


class ReviewModel:
    """Model class that handles business logic for human reviews."""
    
    def __init__(self):
        self.active_sessions: Dict[str, ReviewSession] = {}
        self.completed_sessions: List[ReviewSession] = []
    
    def create_session(self, context: ReviewContext, timeout_seconds: int = 300) -> ReviewSession:
        """Create a new review session."""
        session = ReviewSession(
            context=context,
            timeout_seconds=timeout_seconds
        )
        
        # Validate the context
        if not self._validate_context(context):
            session.error_review("Invalid review context provided")
            return session
        
        self.active_sessions[session.session_id] = session
        logger.info(f"Created review session {session.session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ReviewSession]:
        """Get a review session by ID."""
        return self.active_sessions.get(session_id)
    
    def complete_session(self, session_id: str, decision: ReviewDecision, 
                        user_notes: Optional[str] = None) -> bool:
        """Complete a review session."""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        if not session.is_active():
            logger.warning(f"Session {session_id} is not active")
            return False
        
        session.complete_review(decision, user_notes)
        
        # Move to completed sessions
        self.completed_sessions.append(session)
        del self.active_sessions[session_id]
        
        return True
    
    def timeout_session(self, session_id: str) -> bool:
        """Mark a session as timed out."""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        session.timeout_review()
        self.completed_sessions.append(session)
        del self.active_sessions[session_id]
        
        return True
    
    def error_session(self, session_id: str, error_message: str) -> bool:
        """Mark a session as errored."""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        session.error_review(error_message)
        self.completed_sessions.append(session)
        del self.active_sessions[session_id]
        
        return True
    
    def get_active_sessions(self) -> List[ReviewSession]:
        """Get all active review sessions."""
        return list(self.active_sessions.values())
    
    def get_completed_sessions(self, limit: int = 100) -> List[ReviewSession]:
        """Get completed review sessions."""
        return self.completed_sessions[-limit:]
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about review sessions."""
        completed = self.completed_sessions
        
        if not completed:
            return {
                "total_sessions": 0,
                "avg_duration": 0,
                "decision_breakdown": {},
                "timeout_rate": 0,
                "error_rate": 0
            }
        
        # Calculate statistics
        total_sessions = len(completed)
        durations = [s.get_duration() for s in completed if s.get_duration()]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Decision breakdown
        decision_counts = {}
        timeout_count = 0
        error_count = 0
        
        for session in completed:
            if session.status == ReviewStatus.TIMEOUT:
                timeout_count += 1
            elif session.status == ReviewStatus.ERROR:
                error_count += 1
            elif session.decision:
                decision_counts[session.decision.value] = decision_counts.get(session.decision.value, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": len(self.active_sessions),
            "avg_duration": round(avg_duration, 2),
            "decision_breakdown": decision_counts,
            "timeout_rate": round((timeout_count / total_sessions) * 100, 2),
            "error_rate": round((error_count / total_sessions) * 100, 2)
        }
    
    def cleanup_old_sessions(self, max_completed: int = 1000) -> None:
        """Clean up old completed sessions to prevent memory issues."""
        if len(self.completed_sessions) > max_completed:
            # Keep only the most recent sessions
            self.completed_sessions = self.completed_sessions[-max_completed:]
            logger.info(f"Cleaned up old review sessions, keeping {max_completed} most recent")
    
    def _validate_context(self, context: ReviewContext) -> bool:
        """Validate the review context."""
        if not context.question or not context.question.strip():
            logger.error("Review context missing question")
            return False
        
        if not context.context or not context.context.strip():
            logger.error("Review context missing content")
            return False
        
        if context.retrieved_docs_count < 0 or context.web_results_count < 0:
            logger.error("Review context has negative counts")
            return False
        
        return True


# Global model instance
_review_model = ReviewModel()


def get_review_model() -> ReviewModel:
    """Get the global review model instance."""
    return _review_model

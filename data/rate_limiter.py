import time
from datetime import datetime, timedelta
from threading import Lock
from config import Config

class RateLimiter:
    """Handles rate limiting for API requests"""
    
    def __init__(self, requests_per_minute: int = None):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute (int): Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute or Config.REQUESTS_PER_MINUTE
        self.request_times = []
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to stay within rate limits"""
        with self.lock:
            now = datetime.now()
            self._cleanup_old_requests(now)
            
            if self._at_rate_limit():
                wait_time = self._calculate_wait_time(now)
                if wait_time > 0:
                    time.sleep(wait_time)
            
            self.request_times.append(now)
    
    def _cleanup_old_requests(self, now: datetime):
        """Remove requests older than 1 minute"""
        while self.request_times and now - self.request_times[0] > timedelta(minutes=1):
            self.request_times.pop(0)
    
    def _at_rate_limit(self) -> bool:
        """Check if we're at the rate limit"""
        return len(self.request_times) >= self.requests_per_minute
    
    def _calculate_wait_time(self, now: datetime) -> float:
        """Calculate how long to wait until oldest request expires"""
        if not self.request_times:
            return 0.0
        return (self.request_times[0] + timedelta(minutes=1) - now).total_seconds()
    
    def get_current_requests(self) -> int:
        """Get current number of requests in the window"""
        with self.lock:
            return len(self.request_times)
    
    def get_requests_remaining(self) -> int:
        """Get number of requests remaining in current window"""
        with self.lock:
            return max(0, self.requests_per_minute - len(self.request_times))
    
    def reset(self):
        """Reset the rate limiter (useful for testing or error recovery)"""
        with self.lock:
            self.request_times.clear()

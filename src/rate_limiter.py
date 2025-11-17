"""Rate limiting utilities for API calls."""
import time
from datetime import datetime, timedelta
from collections import deque


class RateLimiter:
    """Simple rate limiter to avoid exceeding API quotas."""

    def __init__(self, max_calls: int = 10, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds (default 60s = 1 minute)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()

    def wait_if_needed(self):
        """Wait if necessary to avoid exceeding rate limit."""
        now = datetime.now()

        # Remove calls outside the time window
        cutoff = now - timedelta(seconds=self.time_window)
        while self.call_times and self.call_times[0] < cutoff:
            self.call_times.popleft()

        # If we're at the limit, wait until we can make another call
        if len(self.call_times) >= self.max_calls:
            oldest_call = self.call_times[0]
            wait_until = oldest_call + timedelta(seconds=self.time_window)
            wait_seconds = (wait_until - now).total_seconds()

            if wait_seconds > 0:
                print(f"[RATE LIMITER] Waiting {wait_seconds:.1f}s to avoid rate limit...")
                time.sleep(wait_seconds + 0.5)  # Add small buffer

                # Clean up old calls after waiting
                now = datetime.now()
                cutoff = now - timedelta(seconds=self.time_window)
                while self.call_times and self.call_times[0] < cutoff:
                    self.call_times.popleft()

        # Record this call
        self.call_times.append(datetime.now())


# Global rate limiter for Gemini API (free tier: 15 RPM)
# Using conservative limit of 10 RPM to be safe
gemini_rate_limiter = RateLimiter(max_calls=10, time_window=60)

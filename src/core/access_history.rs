use serde::{Deserialize, Serialize};

/// Tracks access patterns for a memory node
///
/// This structure stores usage statistics that can be used to optimize
/// prefetching, caching, and other performance-enhancing strategies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessHistory {
    /// Total number of times this memory has been accessed
    access_count: u64,

    /// Timestamp of the last access
    last_access_timestamp: Option<u64>,

    /// Recent access timestamps, used for frequency analysis
    recent_accesses: Vec<u64>,

    /// Maximum number of recent accesses to store
    max_recent_accesses: usize,
}

impl AccessHistory {
    /// Creates a new access history tracker
    pub fn new() -> Self {
        Self {
            access_count: 0,
            last_access_timestamp: None,
            recent_accesses: Vec::new(),
            max_recent_accesses: 10, // Store the 10 most recent accesses
        }
    }

    /// Records an access to the associated memory node
    pub fn record_access(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.access_count += 1;
        self.last_access_timestamp = Some(now);

        // Add to recent accesses, maintaining the cap
        self.recent_accesses.push(now);
        if self.recent_accesses.len() > self.max_recent_accesses {
            self.recent_accesses.remove(0);
        }
    }

    /// Returns the total number of times this memory has been accessed
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Returns the timestamp of the last access, if any
    pub fn last_access_timestamp(&self) -> Option<u64> {
        self.last_access_timestamp
    }

    /// Returns a reference to the recent access timestamps
    pub fn recent_accesses(&self) -> &[u64] {
        &self.recent_accesses
    }

    /// Calculates the access frequency (accesses per hour) over recent history
    pub fn recent_frequency(&self) -> f64 {
        if self.recent_accesses.is_empty() {
            return 0.0;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Use the oldest stored access as our time window
        let oldest = self.recent_accesses[0];
        let window_seconds = (now - oldest) as f64;

        // Avoid division by zero
        if window_seconds < 1.0 {
            return 0.0;
        }

        // Convert to accesses per hour
        let hours = window_seconds / 3600.0;
        self.recent_accesses.len() as f64 / hours
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_access_history_creation() {
        let history = AccessHistory::new();

        assert_eq!(history.access_count(), 0);
        assert_eq!(history.last_access_timestamp(), None);
        assert!(history.recent_accesses().is_empty());
    }

    #[test]
    fn test_record_access() {
        let mut history = AccessHistory::new();

        // Record first access
        history.record_access();
        assert_eq!(history.access_count(), 1);
        assert!(history.last_access_timestamp().is_some());
        assert_eq!(history.recent_accesses().len(), 1);

        // Record second access
        sleep(Duration::from_millis(10)); // Ensure different timestamp
        history.record_access();
        assert_eq!(history.access_count(), 2);
        assert!(history.last_access_timestamp().unwrap() >= history.recent_accesses()[0]);
        assert_eq!(history.recent_accesses().len(), 2);
    }

    #[test]
    fn test_max_recent_accesses() {
        let mut history = AccessHistory::new();

        // Set a smaller cap for testing
        history.max_recent_accesses = 3;

        // Record more than the cap
        for _ in 0..5 {
            sleep(Duration::from_millis(10)); // Ensure different timestamps
            history.record_access();
        }

        // Check that only the most recent are kept
        assert_eq!(history.access_count(), 5);
        assert_eq!(history.recent_accesses().len(), 3);
    }
}

//! Utility functions and helpers
//!
//! This module contains various utility functions used throughout the system.

/// Time related utility functions
pub mod time {
    /// Converts a duration in seconds to a human-readable string
    ///
    /// # Arguments
    ///
    /// * `seconds` - Duration in seconds
    ///
    /// # Returns
    ///
    /// A human-readable string like "2 days 3 hours 45 minutes"
    pub fn format_duration(seconds: u64) -> String {
        if seconds == 0 {
            return "just now".to_string();
        }

        let days = seconds / (24 * 60 * 60);
        let hours = (seconds % (24 * 60 * 60)) / (60 * 60);
        let minutes = (seconds % (60 * 60)) / 60;
        let secs = seconds % 60;

        let mut parts = Vec::new();

        if days > 0 {
            parts.push(format!("{} day{}", days, if days == 1 { "" } else { "s" }));
        }

        if hours > 0 {
            parts.push(format!(
                "{} hour{}",
                hours,
                if hours == 1 { "" } else { "s" }
            ));
        }

        if minutes > 0 {
            parts.push(format!(
                "{} minute{}",
                minutes,
                if minutes == 1 { "" } else { "s" }
            ));
        }

        if secs > 0 && parts.is_empty() {
            parts.push(format!(
                "{} second{}",
                secs,
                if secs == 1 { "" } else { "s" }
            ));
        }

        parts.join(" ")
    }

    /// Gets the current timestamp in seconds since epoch
    pub fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Helper to create a timestamp for N seconds ago
    pub fn seconds_ago(seconds: u64) -> u64 {
        now().saturating_sub(seconds)
    }

    /// Helper to create a timestamp for N minutes ago
    pub fn minutes_ago(minutes: u64) -> u64 {
        seconds_ago(minutes * 60)
    }

    /// Helper to create a timestamp for N hours ago
    pub fn hours_ago(hours: u64) -> u64 {
        seconds_ago(hours * 60 * 60)
    }

    /// Helper to create a timestamp for N days ago
    pub fn days_ago(days: u64) -> u64 {
        seconds_ago(days * 24 * 60 * 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(time::format_duration(0), "just now");
        assert_eq!(time::format_duration(30), "30 seconds");
        assert_eq!(time::format_duration(60), "1 minute");
        assert_eq!(time::format_duration(90), "1 minute");
        assert_eq!(time::format_duration(120), "2 minutes");
        assert_eq!(time::format_duration(3600), "1 hour");
        assert_eq!(time::format_duration(3660), "1 hour 1 minute");
        assert_eq!(time::format_duration(86400), "1 day");
        assert_eq!(time::format_duration(90000), "1 day 1 hour");
        assert_eq!(time::format_duration(172800), "2 days");
        assert_eq!(time::format_duration(176460), "2 days 1 hour 1 minute");
    }

    #[test]
    fn test_time_helpers() {
        let now = time::now();

        // Allow a small margin of error for test execution time
        assert!(time::seconds_ago(10) <= now && time::seconds_ago(10) >= now - 11);
        assert!(time::minutes_ago(1) <= now && time::minutes_ago(1) >= now - 61);
        assert!(time::hours_ago(1) <= now && time::hours_ago(1) >= now - 3601);
        assert!(time::days_ago(1) <= now && time::days_ago(1) >= now - 86401);
    }
}

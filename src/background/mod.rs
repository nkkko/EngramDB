//! Background processing module for EngramDB.
//!
//! This module provides functionality for performing background computations
//! during idle periods, particularly for sleep-time compute operations like
//! summarization, relationship inference, and context enrichment.

pub mod llm;
pub mod task;
pub mod triggers;

// Export types from the module
pub use llm::LLMProcessor;
pub use task::{
    BackgroundTaskManager, Task, TaskId, TaskPriority, TaskResult, TaskStatus, TaskType,
};
pub use triggers::{IdleTrigger, PredictiveTrigger, TriggerType};

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Configuration for background tasks
#[derive(Debug, Clone)]
pub struct BackgroundTaskConfig {
    /// Threshold for considering the system idle (in seconds)
    pub idle_threshold: f64,

    /// Maximum number of tokens to use for background tasks
    pub max_tokens: Option<usize>,

    /// Maximum cost to incur for background tasks
    pub max_cost: Option<f64>,

    /// Maximum number of concurrent background tasks
    pub max_concurrent_tasks: usize,
}

impl Default for BackgroundTaskConfig {
    fn default() -> Self {
        Self {
            idle_threshold: 5.0,
            max_tokens: None,
            max_cost: None,
            max_concurrent_tasks: 2,
        }
    }
}

/// Activity tracker for the database to detect idle periods
#[derive(Debug)]
pub struct ActivityTracker {
    last_activity: Mutex<Instant>,
}

impl ActivityTracker {
    /// Create a new activity tracker
    pub fn new() -> Self {
        Self {
            last_activity: Mutex::new(Instant::now()),
        }
    }

    /// Record an activity
    pub fn record_activity(&self) {
        let mut last_activity = self.last_activity.lock().unwrap();
        *last_activity = Instant::now();
    }

    /// Check if the system is idle based on the configured threshold
    pub fn is_idle(&self, threshold: Duration) -> bool {
        let last_activity = self.last_activity.lock().unwrap();
        last_activity.elapsed() >= threshold
    }
}

impl Default for ActivityTracker {
    fn default() -> Self {
        Self::new()
    }
}

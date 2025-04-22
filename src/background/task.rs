//! Task management for background processing.

use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use uuid::Uuid;

/// Unique identifier for a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub Uuid);

impl TaskId {
    /// Create a new random task ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// Priority level for background tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPriority {
    /// Critical tasks that should run as soon as possible
    Critical = 0,

    /// High priority tasks
    High = 1,

    /// Normal priority tasks
    Normal = 2,

    /// Low priority tasks that can be deferred
    Low = 3,
}

impl PartialOrd for TaskPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaskPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower values have higher priority
        (*self as usize).cmp(&(*other as usize))
    }
}

/// Types of background tasks
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Summarize a set of memory nodes
    Summarize {
        node_ids: Vec<Uuid>,
        prompt: Option<String>,
    },

    /// Infer connections between memory nodes
    InferConnections {
        node_ids: Vec<Uuid>,
        prompt: Option<String>,
    },

    /// Enrich a memory node with additional context
    EnrichNode {
        node_id: Uuid,
        prompt: Option<String>,
    },

    /// Predict and pre-compute results for likely future queries
    PredictQueries { recent_queries: Option<Vec<String>> },
}

/// Status of a background task
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is queued but not yet started
    Queued,

    /// Task is currently running
    Running,

    /// Task completed successfully
    Completed,

    /// Task failed
    Failed,

    /// Task was cancelled
    Cancelled,
}

/// A background task
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique identifier for the task
    pub id: TaskId,

    /// Type of task
    pub task_type: TaskType,

    /// Priority of the task
    pub priority: TaskPriority,

    /// Current status of the task
    pub status: TaskStatus,

    /// When the task was created
    pub created_at: Instant,

    /// When the task started executing (if it has)
    pub started_at: Option<Instant>,

    /// When the task completed (if it has)
    pub completed_at: Option<Instant>,

    /// Result of the task (if it completed)
    pub result: Option<TaskResult>,
}

impl Task {
    /// Create a new task
    pub fn new(task_type: TaskType, priority: TaskPriority) -> Self {
        Self {
            id: TaskId::new(),
            task_type,
            priority,
            status: TaskStatus::Queued,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            result: None,
        }
    }

    /// Mark the task as running
    pub fn mark_running(&mut self) {
        self.status = TaskStatus::Running;
        self.started_at = Some(Instant::now());
    }

    /// Mark the task as completed
    pub fn mark_completed(&mut self, result: TaskResult) {
        self.status = TaskStatus::Completed;
        self.completed_at = Some(Instant::now());
        self.result = Some(result);
    }

    /// Mark the task as failed
    pub fn mark_failed(&mut self) {
        self.status = TaskStatus::Failed;
        self.completed_at = Some(Instant::now());
    }

    /// Mark the task as cancelled
    pub fn mark_cancelled(&mut self) {
        self.status = TaskStatus::Cancelled;
        self.completed_at = Some(Instant::now());
    }
}

/// For ordering in the priority queue (higher priority and older tasks come first)
impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.created_at == other.created_at
    }
}

impl Eq for Task {}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (lower number means higher priority)
        let priority_order = other.priority.cmp(&self.priority);
        if priority_order != Ordering::Equal {
            return priority_order;
        }

        // Then by creation time (older first)
        other.created_at.cmp(&self.created_at)
    }
}

/// Result of a background task
#[derive(Debug, Clone)]
pub enum TaskResult {
    /// A new summary node was created
    Summary {
        /// ID of the created summary node
        node_id: Uuid,
    },

    /// New connections were inferred
    Connections {
        /// IDs of the created connections
        connection_ids: Vec<Uuid>,
    },

    /// A node was enriched with additional context
    Enrichment {
        /// ID of the enriched node
        node_id: Uuid,
    },

    /// Query predictions were generated
    Predictions {
        /// Number of predictions generated
        count: usize,
    },
}

/// Manager for background tasks
#[derive(Debug)]
#[allow(dead_code)]
pub struct BackgroundTaskManager {
    /// Queue of pending tasks
    task_queue: Mutex<BinaryHeap<Task>>,

    /// Currently running tasks
    running_tasks: Mutex<Vec<Task>>,

    /// Completed tasks
    completed_tasks: Mutex<Vec<Task>>,

    /// Configuration for the task manager
    config: Arc<Mutex<super::BackgroundTaskConfig>>,
}

impl BackgroundTaskManager {
    /// Create a new background task manager
    pub fn new(config: super::BackgroundTaskConfig) -> Self {
        Self {
            task_queue: Mutex::new(BinaryHeap::new()),
            running_tasks: Mutex::new(Vec::new()),
            completed_tasks: Mutex::new(Vec::new()),
            config: Arc::new(Mutex::new(config)),
        }
    }

    /// Schedule a new task
    pub fn schedule_task(&self, task_type: TaskType, priority: TaskPriority) -> TaskId {
        let task = Task::new(task_type, priority);
        let task_id = task.id;

        let mut queue = self.task_queue.lock().unwrap();
        queue.push(task);

        task_id
    }

    /// Get the status of a task
    pub fn get_task_status(&self, task_id: TaskId) -> Option<TaskStatus> {
        // Check running tasks
        {
            let running = self.running_tasks.lock().unwrap();
            if let Some(task) = running.iter().find(|t| t.id == task_id) {
                return Some(task.status);
            }
        }

        // Check queued tasks
        {
            let queue = self.task_queue.lock().unwrap();
            if queue.iter().any(|t| t.id == task_id) {
                return Some(TaskStatus::Queued);
            }
        }

        // Check completed tasks
        {
            let completed = self.completed_tasks.lock().unwrap();
            if let Some(task) = completed.iter().find(|t| t.id == task_id) {
                return Some(task.status);
            }
        }

        None
    }

    /// Get the result of a completed task
    pub fn get_task_result(&self, task_id: TaskId) -> Option<TaskResult> {
        let completed = self.completed_tasks.lock().unwrap();
        if let Some(task) = completed.iter().find(|t| t.id == task_id) {
            return task.result.clone();
        }

        None
    }

    /// Cancel a pending task
    pub fn cancel_task(&self, task_id: TaskId) -> bool {
        let mut queue = self.task_queue.lock().unwrap();
        if let Some(pos) = queue.iter().position(|t| t.id == task_id) {
            let mut elements: Vec<_> = queue.drain().collect();
            let mut task = elements.remove(pos);
            task.mark_cancelled();

            let mut completed = self.completed_tasks.lock().unwrap();
            completed.push(task);

            for element in elements {
                queue.push(element);
            }

            return true;
        }

        false
    }

    /// Get the next task to execute
    #[allow(dead_code)]
    pub(crate) fn next_task(&self) -> Option<Task> {
        let mut queue = self.task_queue.lock().unwrap();
        let mut running = self.running_tasks.lock().unwrap();

        // Check if we've reached the limit of concurrent tasks
        let config = self.config.lock().unwrap();
        if running.len() >= config.max_concurrent_tasks {
            return None;
        }

        // Get the next task
        if let Some(mut task) = queue.pop() {
            task.mark_running();
            running.push(task.clone());
            Some(task)
        } else {
            None
        }
    }

    /// Complete a task with a result
    #[allow(dead_code)]
    pub(crate) fn complete_task(&self, task_id: TaskId, result: TaskResult) {
        let mut running = self.running_tasks.lock().unwrap();
        if let Some(pos) = running.iter().position(|t| t.id == task_id) {
            let mut task = running.remove(pos);
            task.mark_completed(result);

            let mut completed = self.completed_tasks.lock().unwrap();
            completed.push(task);
        }
    }

    /// Mark a task as failed
    #[allow(dead_code)]
    pub(crate) fn fail_task(&self, task_id: TaskId) {
        let mut running = self.running_tasks.lock().unwrap();
        if let Some(pos) = running.iter().position(|t| t.id == task_id) {
            let mut task = running.remove(pos);
            task.mark_failed();

            let mut completed = self.completed_tasks.lock().unwrap();
            completed.push(task);
        }
    }
}

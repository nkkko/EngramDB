use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1};
use uuid::Uuid;
use engramdb::{
    MemoryNode as EngramDbMemoryNode,
    Database as EngramDbDatabase,
    RelationshipType as EngramDbRelationshipType,
};
use engramdb::core::AttributeValue;
use std::path::PathBuf;

/// Simple function to test that the Python binding works
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> String {
    format!("{}", a + b)
}

/// A Python wrapper for the EngramDB MemoryNode class
#[pyclass]
struct MemoryNode {
    inner: EngramDbMemoryNode,
}

#[pymethods]
impl MemoryNode {
    /// Create a new memory node with the given embeddings
    #[new]
    fn new(embeddings: Vec<f32>) -> Self {
        Self {
            inner: EngramDbMemoryNode::new(embeddings),
        }
    }

    /// Get the ID of the memory node
    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Set an attribute on the memory node
    fn set_attribute(&mut self, key: &str, value: &PyAny) -> PyResult<()> {
        let attr_value = match value.extract::<String>() {
            Ok(s) => AttributeValue::String(s),
            Err(_) => match value.extract::<i64>() {
                Ok(i) => AttributeValue::Integer(i),
                Err(_) => match value.extract::<f64>() {
                    Ok(f) => AttributeValue::Float(f),
                    Err(_) => match value.extract::<bool>() {
                        Ok(b) => AttributeValue::Boolean(b),
                        Err(_) => return Err(PyValueError::new_err("Unsupported attribute value type")),
                    },
                },
            },
        };
        
        self.inner.set_attribute(key.to_string(), attr_value);
        Ok(())
    }

    /// Get an attribute from the memory node
    fn get_attribute(&self, key: &str) -> Option<PyObject> {
        Python::with_gil(|py| {
            self.inner.get_attribute(key).map(|attr| {
                match attr {
                    AttributeValue::String(s) => s.to_object(py),
                    AttributeValue::Integer(i) => i.to_object(py),
                    AttributeValue::Float(f) => f.to_object(py),
                    AttributeValue::Boolean(b) => b.to_object(py),
                    _ => py.None(),
                }
            })
        })
    }
    
    /// Get all attributes of the memory node
    fn attributes<'py>(&self, py: Python<'py>) -> PyObject {
        let mut dict = pyo3::types::PyDict::new(py);
        for (key, value) in self.inner.attributes() {
            let py_value = match value {
                AttributeValue::String(s) => s.to_object(py),
                AttributeValue::Integer(i) => i.to_object(py),
                AttributeValue::Float(f) => f.to_object(py),
                AttributeValue::Boolean(b) => b.to_object(py),
                _ => py.None(),
            };
            dict.set_item(key, py_value).unwrap();
        }
        dict.to_object(py)
    }

    /// Get the embeddings of this memory node
    fn get_embeddings<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.inner.embeddings().to_vec().into_pyarray(py)
    }
    
    /// Set new embeddings for this memory node
    fn set_embeddings(&mut self, embeddings: Vec<f32>) {
        self.inner.set_embeddings(embeddings);
    }
}

/// Python wrapper for the EngramDB Database
#[pyclass]
struct Database {
    inner: EngramDbDatabase,
}

#[pymethods]
impl Database {
    /// Create a new in-memory database
    #[staticmethod]
    fn in_memory() -> Self {
        Self {
            inner: EngramDbDatabase::in_memory(),
        }
    }

    /// Create a new file-based database at the given path
    #[staticmethod]
    fn file_based(path: &str) -> PyResult<Self> {
        match EngramDbDatabase::file_based(PathBuf::from(path)) {
            Ok(db) => Ok(Self { inner: db }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create file-based database: {}", e)))
        }
    }

    /// Save a memory node to the database
    fn save(&mut self, memory_node: &MemoryNode) -> PyResult<String> {
        match self.inner.save(&memory_node.inner) {
            Ok(id) => Ok(id.to_string()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to save node: {}", e)))
        }
    }

    /// Load a memory node from the database
    fn load(&self, id_str: &str) -> PyResult<MemoryNode> {
        let id = Uuid::parse_str(id_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.load(id) {
            Ok(node) => Ok(MemoryNode { inner: node }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to load node: {}", e)))
        }
    }

    /// Delete a memory node from the database
    fn delete(&mut self, id_str: &str) -> PyResult<bool> {
        let id = Uuid::parse_str(id_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.delete(id) {
            Ok(_) => Ok(true),
            Err(e) => Err(PyValueError::new_err(format!("Failed to delete node: {}", e)))
        }
    }

    /// List all memory node IDs in the database
    fn list_all(&self) -> PyResult<Vec<String>> {
        match self.inner.list_all() {
            Ok(ids) => Ok(ids.into_iter().map(|id| id.to_string()).collect()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to list nodes: {}", e)))
        }
    }

    /// Search for similar memory nodes using vector similarity
    fn search_similar(&self, query: Vec<f32>, limit: usize, threshold: f32) -> PyResult<Vec<(String, f32)>> {
        match self.inner.search_similar(&query, limit, threshold) {
            Ok(results) => Ok(results.into_iter().map(|(id, score)| (id.to_string(), score)).collect()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to search: {}", e)))
        }
    }
}

/// Python enum for relationship types between memory nodes
#[pyclass]
#[derive(Clone, Copy)]
enum RelationshipType {
    Association = 0,
    Causation = 1,
    Sequence = 2,
    Contains = 3,
}

impl From<RelationshipType> for EngramDbRelationshipType {
    fn from(py_type: RelationshipType) -> Self {
        match py_type {
            RelationshipType::Association => EngramDbRelationshipType::Association,
            RelationshipType::Causation => EngramDbRelationshipType::Causation,
            RelationshipType::Sequence => EngramDbRelationshipType::Sequence,
            RelationshipType::Contains => EngramDbRelationshipType::Contains,
        }
    }
}

/// Register the Python module
#[pymodule]
fn engramdb_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<MemoryNode>()?;
    m.add_class::<Database>()?;
    m.add_class::<RelationshipType>()?;
    Ok(())
}
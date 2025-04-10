/// Module for model information and configuration

/// Enum of supported embedding models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// E5 Multilingual Large Instruct
    E5MultilingualLargeInstruct,
    /// GTE Modern BERT Base
    GteModernBertBase, 
    /// Jina Embeddings V3
    JinaEmbeddingsV3,
    /// Custom model
    Custom,
}

impl EmbeddingModel {
    /// Get the model ID for loading from HuggingFace
    pub fn model_id(&self) -> &'static str {
        match self {
            EmbeddingModel::E5MultilingualLargeInstruct => "intfloat/multilingual-e5-large-instruct",
            EmbeddingModel::GteModernBertBase => "Alibaba-NLP/gte-modernbert-base",
            EmbeddingModel::JinaEmbeddingsV3 => "jinaai/jina-embeddings-v3",
            EmbeddingModel::Custom => "",
        }
    }

    /// Get the dimensions of embeddings for this model
    pub fn dimensions(&self) -> usize {
        match self {
            EmbeddingModel::E5MultilingualLargeInstruct => 1024,
            EmbeddingModel::GteModernBertBase => 768,
            EmbeddingModel::JinaEmbeddingsV3 => 768,
            EmbeddingModel::Custom => 0, // Will be set dynamically
        }
    }

    /// Create from a model name
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "e5" | "multilingual-e5" | "e5-multilingual" | "intfloat/multilingual-e5-large-instruct" => 
                EmbeddingModel::E5MultilingualLargeInstruct,
            "gte" | "gte-modernbert" | "gte-modernbert-base" | "alibaba-nlp/gte-modernbert-base" => 
                EmbeddingModel::GteModernBertBase,
            "jina" | "jina-embeddings" | "jina-v3" | "jinaai/jina-embeddings-v3" =>
                EmbeddingModel::JinaEmbeddingsV3,
            _ => {
                // If it's not one of the known models, assume it's a custom model
                EmbeddingModel::Custom
            }
        }
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        EmbeddingModel::E5MultilingualLargeInstruct
    }
}
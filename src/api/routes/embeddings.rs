//! Embedding-related routes

use crate::api::auth::{ApiKey, User};
use crate::api::error::{ApiError, ApiResult};
use crate::api::models::embedding_models::*;
use crate::embeddings::{self, EmbeddingService};
use rocket::serde::json::Json;

/// List available embedding models
#[rocket::get("/models")]
pub fn list_models(_api_key: ApiKey) -> ApiResult<Json<Vec<EmbeddingModelInfo>>> {
    // In a real implementation, we'd query available embedding models dynamically
    let mut models = Vec::new();

    // Add a default mock embedding model for demo purposes
    models.push(EmbeddingModelInfo {
        id: "default".to_string(),
        name: "Default Mock Embedding Model".to_string(),
        dimensions: 128,
        description: "A mock embedding model for demonstration purposes.".to_string(),
        provider: "EngramDB".to_string(),
        model_type: "single_vector".to_string(),
    });

    // Add E5 model if available
    #[cfg(feature = "embeddings")]
    models.push(EmbeddingModelInfo {
        id: "e5-small".to_string(),
        name: "E5 Small".to_string(),
        dimensions: 384,
        description: "E5 small embedding model from Microsoft.".to_string(),
        provider: "Microsoft".to_string(),
        model_type: "single_vector".to_string(),
    });

    // Add JINA ColBERT model if available
    #[cfg(feature = "embeddings")]
    models.push(EmbeddingModelInfo {
        id: "jina-colbert".to_string(),
        name: "JINA ColBERT".to_string(),
        dimensions: 512,
        description: "JINA ColBERT multi-vector embedding model.".to_string(),
        provider: "JINA AI".to_string(),
        model_type: "multi_vector".to_string(),
    });

    Ok(Json(models))
}

/// Generate embedding from text
#[rocket::post("/generate_embedding", data = "<input>")]
pub fn generate_embedding(
    _api_key: ApiKey,
    input: Json<GenerateEmbeddingInput>,
) -> ApiResult<Json<GeneratedEmbedding>> {
    // Create a mock embedding service
    let mock_service = embeddings::mock::MockEmbeddingService::new();

    // Generate embedding
    let vector = mock_service
        .embed_text(&input.content)
        .map_err(|_| ApiError::bad_request("Failed to generate embedding"))?;

    // In a real implementation, we'd use the specified model
    // let vector = match input.model.as_str() {
    //     "e5-small" => {
    //         let service = embeddings::providers::E5EmbeddingService::new()?;
    //         service.embed_text(&input.content)?
    //     }
    //     "jina-colbert" => {
    //         let service = embeddings::providers::JinaEmbeddingService::new()?;
    //         service.embed_text(&input.content)?.primary_vector()
    //     }
    //     _ => mock_service.embed_text(&input.content)?,
    // };

    Ok(Json(GeneratedEmbedding {
        vector: vector.to_vec(),
        model: input.model.clone(),
        dimensions: vector.len(),
    }))
}

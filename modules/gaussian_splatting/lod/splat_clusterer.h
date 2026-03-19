#ifndef SPLAT_CLUSTERER_H
#define SPLAT_CLUSTERER_H

#include "core/math/vector3.h"
#include "core/math/color.h"
#include "core/math/basis.h"
#include "core/templates/vector.h"
#include "core/templates/local_vector.h"
#include "../core/gaussian_data.h"

namespace GaussianSplatting {

class SplatClusterer {
public:
    struct ClusteredSplat {
        Vector3 position;           // Weighted average position
        Color color;                // Weighted average color
        float combined_opacity;     // Combined opacity
        Quaternion rotation;        // Average orientation
        Vector3 scale;             // Combined scale
        uint32_t source_count;     // Number of merged splats
        float importance_sum;      // Sum of importance values
        uint32_t original_index;   // Index of most important source splat
        PainterlyMetadata painterly; // Painterly metadata blended from sources

        // Covariance matrix for the clustered splat
        float covariance[6] = {};  // Upper triangular: xx, xy, xz, yy, yz, zz

        ClusteredSplat() : combined_opacity(0.0f), source_count(0),
                          importance_sum(0.0f), original_index(0) {
        }

        // Convert to GaussianData format
        GaussianData to_gaussian_data() const;

        // Compute combined covariance from scale and rotation
        void compute_covariance();
    };

    struct ClusteringParams {
        enum Method {
            SPATIAL_CLUSTERING,      // K-means based on position
            HIERARCHICAL_CLUSTERING, // Bottom-up merging
            GRID_BASED,             // Uniform grid clustering
            IMPORTANCE_WEIGHTED     // Weighted by visual importance
        };

        Method method = HIERARCHICAL_CLUSTERING;
        float cluster_radius = 1.0f;      // Maximum cluster size
        uint32_t target_count = 0;        // Target number of clusters (0 = auto)
        float merge_threshold = 0.5f;     // Similarity threshold for merging
        bool preserve_boundaries = true;   // Preserve object boundaries
        bool color_similarity = true;      // Consider color when clustering
        uint32_t min_cluster_size = 2;    // Minimum splats per cluster
        uint32_t max_cluster_size = 16;   // Maximum splats per cluster
    };

    struct ClusteringResult {
        Vector<ClusteredSplat> clusters;
        float reduction_ratio;      // 1.0 - (clusters / original)
        float quality_score;        // Estimated visual quality (0-1)

        struct Statistics {
            uint32_t original_count;
            uint32_t cluster_count;
            float avg_cluster_size;
            float max_cluster_radius;
            float processing_time_ms;
        } stats;
    };

public:
    SplatClusterer();
    ~SplatClusterer();

    // Main clustering method
    ClusteringResult cluster_splats(
        const Vector<GaussianData>& splats,
        const ClusteringParams& params
    );

    // Specific clustering algorithms
    ClusteringResult cluster_spatial(
        const Vector<GaussianData>& splats,
        float cluster_radius,
        uint32_t target_count
    );

    ClusteringResult cluster_hierarchical(
        const Vector<GaussianData>& splats,
        const ClusteringParams& params
    );

    ClusteringResult cluster_grid_based(
        const Vector<GaussianData>& splats,
        float grid_size
    );

    // Importance-aware clustering
    ClusteringResult cluster_by_importance(
        const Vector<GaussianData>& splats,
        const Vector<float>& importance_scores,
        const ClusteringParams& params
    );

    // LOD-specific clustering
    ClusteringResult generate_lod_clusters(
        const Vector<GaussianData>& splats,
        uint32_t lod_level
    );

    // Merge two splats into one
    ClusteredSplat merge_splats(
        const GaussianData& splat1,
        const GaussianData& splat2,
        float weight1 = 1.0f,
        float weight2 = 1.0f
    );

    // Merge multiple splats
    ClusteredSplat merge_splat_group(
        const Vector<GaussianData>& splats,
        const LocalVector<uint32_t>& indices,
        const Vector<float>* weights = nullptr
    );

    // Quality metrics
    float compute_cluster_quality(
        const ClusteredSplat& cluster,
        const Vector<GaussianData>& source_splats,
        const LocalVector<uint32_t>& source_indices
    );

    float compute_color_similarity(
        const GaussianData& splat1,
        const GaussianData& splat2
    );

    float compute_spatial_similarity(
        const GaussianData& splat1,
        const GaussianData& splat2,
        float max_distance
    );

private:
    // Hierarchical clustering helpers
    struct ClusterNode {
        LocalVector<uint32_t> splat_indices;
        Vector3 centroid;
        Color avg_color;
        float radius;
        float importance;
        bool merged;

        ClusterNode() : radius(0.0f), importance(0.0f), merged(false) {}
    };

    void build_cluster_hierarchy(
        Vector<ClusterNode>& nodes,
        const Vector<GaussianData>& splats,
        const ClusteringParams& params
    );

    bool can_merge_clusters(
        const ClusterNode& node1,
        const ClusterNode& node2,
        const ClusteringParams& params
    );

    ClusterNode merge_cluster_nodes(
        const ClusterNode& node1,
        const ClusterNode& node2,
        const Vector<GaussianData>& splats
    );

    // K-means helpers
    void kmeans_clustering(
        const Vector<GaussianData>& splats,
        uint32_t k,
        Vector<ClusteredSplat>& clusters
    );

    void update_cluster_centers(
        Vector<ClusteredSplat>& clusters,
        const Vector<GaussianData>& splats,
        const LocalVector<uint32_t>& assignments
    );

    // Grid-based helpers
    struct GridCell {
        LocalVector<uint32_t> splat_indices;
        Vector3 min_bound;
        Vector3 max_bound;
    };

    void build_spatial_grid(
        const Vector<GaussianData>& splats,
        float grid_size,
        Vector<GridCell>& grid
    );

    // Covariance operations
    void merge_covariances(
        float result[6],
        const float cov1[6],
        const float cov2[6],
        const Vector3& pos1,
        const Vector3& pos2,
        float weight1,
        float weight2
    );

    void add_covariance_matrices(
        float result[6],
        const float a[6],
        const float b[6]
    );

    void scale_covariance_matrix(
        float result[6],
        const float input[6],
        float scale
    );

    // Quality estimation
    float estimate_visual_quality(
        const ClusteringResult& result,
        const Vector<GaussianData>& original_splats
    );

    // Boundary preservation
    bool is_boundary_splat(
        const GaussianData& splat,
        const Vector<GaussianData>& neighbors
    );

    void detect_boundaries(
        const Vector<GaussianData>& splats,
        LocalVector<bool>& is_boundary
    );
};

// Inline implementations
inline GaussianData SplatClusterer::ClusteredSplat::to_gaussian_data() const {
    GaussianData gaussian;
    gaussian.position = position;
    gaussian.color = color;
    gaussian.rotation = rotation;
    gaussian.scale = scale;
    gaussian.index = original_index;
    gaussian.importance = importance_sum;
    gaussian.painterly = painterly;

    // Copy covariance
    memcpy(gaussian.covariance, covariance, sizeof(covariance));

    return gaussian;
}

inline void SplatClusterer::ClusteredSplat::compute_covariance() {
    // Convert scale and rotation to covariance matrix
    // This is a simplified version - full implementation would be more complex

    // Create scaling matrix
    Basis S = Basis::from_scale(Vector3(
        scale.x * scale.x,
        scale.y * scale.y,
        scale.z * scale.z));

    // Create rotation matrix from quaternion
    Basis R(rotation);

    // Covariance = R * S * R^T
    Basis cov_matrix = R * S * R.transposed();

    // Store upper triangular part
    covariance[0] = cov_matrix[0][0];  // xx
    covariance[1] = cov_matrix[0][1];  // xy
    covariance[2] = cov_matrix[0][2];  // xz
    covariance[3] = cov_matrix[1][1];  // yy
    covariance[4] = cov_matrix[1][2];  // yz
    covariance[5] = cov_matrix[2][2];  // zz
}

} // namespace GaussianSplatting

#endif // SPLAT_CLUSTERER_H

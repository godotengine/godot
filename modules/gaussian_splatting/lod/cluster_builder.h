#ifndef CLUSTER_BUILDER_H
#define CLUSTER_BUILDER_H

#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "../core/gaussian_data.h"
#include <cstdint>

namespace GaussianSplatting {

/**
 * @struct SplatCluster
 * @brief A group of spatially coherent splats with shared AABB for coarse culling.
 *
 * Clusters group 32-256 splats together to enable efficient frustum culling.
 * The GPU tests cluster AABBs first (O(clusters)) before testing individual
 * splats (O(splats)), reducing overall culling cost significantly.
 */
struct SplatCluster {
    AABB bounds;           ///< Axis-aligned bounding box encompassing all splats
    Vector3 center;        ///< Centroid of the cluster
    float radius;          ///< Bounding sphere radius (conservative)
    uint32_t splat_start;  ///< First splat index in sorted order
    uint32_t splat_count;  ///< Number of splats in this cluster
    float importance_sum;  ///< Sum of importance weights for LOD
    float max_splat_radius; ///< Largest splat radius in cluster (for culling slack)

    SplatCluster() :
        radius(0.0f),
        splat_start(0),
        splat_count(0),
        importance_sum(0.0f),
        max_splat_radius(0.0f) {}
};

/**
 * @struct ClusterBuildParams
 * @brief Parameters for cluster construction.
 */
struct ClusterBuildParams {
    uint32_t target_cluster_size = 128;  ///< Target splats per cluster (32-256 recommended)
    uint32_t min_cluster_size = 32;      ///< Minimum splats per cluster
    uint32_t max_cluster_size = 256;     ///< Maximum splats per cluster
    float spatial_weight = 1.0f;         ///< Weight for spatial coherence
    bool use_morton_order = true;        ///< Use Morton (Z-order) curve for better locality
    bool compute_importance = true;      ///< Compute importance weights per cluster
};

/**
 * @struct ClusterBuildResult
 * @brief Result of cluster building operation.
 */
struct ClusterBuildResult {
    Vector<SplatCluster> clusters;            ///< Built clusters
    LocalVector<uint32_t> splat_to_cluster;   ///< Mapping from splat index to cluster index
    LocalVector<uint32_t> sorted_splat_order; ///< Splat indices sorted by cluster assignment
    float build_time_ms;                      ///< Time to build clusters
    uint32_t original_splat_count;            ///< Number of input splats

    ClusterBuildResult() : build_time_ms(0.0f), original_splat_count(0) {}
};

/**
 * @class ClusterBuilder
 * @brief Builds spatial clusters of Gaussian splats for hierarchical culling.
 *
 * The ClusterBuilder organizes splats into groups of 32-256 for efficient
 * GPU frustum culling. By testing cluster AABBs first, the GPU can reject
 * large numbers of splats without testing each one individually.
 *
 * Algorithm:
 * 1. Compute Morton codes for all splat positions
 * 2. Sort splats by Morton code (preserves spatial locality)
 * 3. Group consecutive splats into clusters
 * 4. Compute AABB for each cluster
 *
 * Typical performance: 1M splats -> ~8K clusters (128 splats/cluster)
 */
class ClusterBuilder {
public:
    ClusterBuilder();
    ~ClusterBuilder();

    /**
     * @brief Builds clusters from Gaussian splat data.
     * @param p_gaussians Source Gaussian data.
     * @param p_params Build parameters.
     * @return Cluster build result including clusters and mappings.
     */
    ClusterBuildResult build_clusters(
        const LocalVector<Gaussian> &p_gaussians,
        const ClusterBuildParams &p_params = ClusterBuildParams()
    );

    /**
     * @brief Updates clusters incrementally when splats change.
     * @param p_gaussians Updated Gaussian data.
     * @param p_changed_indices Indices of splats that changed.
     * @param p_previous Existing clusters to update.
     * @return Updated cluster result.
     *
     * This is more efficient than full rebuild for small changes.
     */
    ClusterBuildResult update_clusters_incremental(
        const LocalVector<Gaussian> &p_gaussians,
        const LocalVector<uint32_t> &p_changed_indices,
        const ClusterBuildResult &p_previous
    );

    /**
     * @brief Computes the GPU buffer layout for cluster data.
     * @param p_clusters Source clusters.
     * @return Packed buffer data ready for GPU upload.
     *
     * GPU format per cluster (32 bytes):
     *   vec3 min_bounds (12 bytes)
     *   uint splat_start (4 bytes)
     *   vec3 max_bounds (12 bytes)
     *   uint splat_count (4 bytes)
     */
    Vector<uint8_t> pack_for_gpu(const Vector<SplatCluster> &p_clusters) const;

private:
    /**
     * @brief Computes Morton code for a 3D position.
     * @param p_position Position in world space.
     * @param p_bounds Bounding box for normalization.
     * @return 32-bit Morton code.
     */
    uint32_t compute_morton_code(const Vector3 &p_position, const AABB &p_bounds) const;

    /**
     * @brief Expands 10-bit integer to 30 bits (3-bit gaps for interleaving).
     */
    uint32_t expand_bits(uint32_t v) const;

    /**
     * @brief Computes AABB for a group of splats.
     */
    AABB compute_cluster_bounds(
        const LocalVector<Gaussian> &p_gaussians,
        const LocalVector<uint32_t> &p_sorted_order,
        uint32_t p_start,
        uint32_t p_count,
        float &r_max_radius
    ) const;
};

} // namespace GaussianSplatting

#endif // CLUSTER_BUILDER_H

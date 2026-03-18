#include "splat_clusterer.h"
#include "core/os/os.h"
#include "core/math/math_funcs.h"
#include "../logger/gs_logger.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace GaussianSplatting {

namespace {
static uint32_t mix_seed(uint32_t accumulator, uint32_t seed, float weight) {
    uint32_t quantized = uint32_t(CLAMP(weight, 0.0f, 1.0f) * 65535.0f + 0.5f);
    accumulator ^= seed + 0x9e3779b9u + (accumulator << 6) + (accumulator >> 2);
    accumulator ^= quantized + 0x85ebca6bu + (accumulator << 5) + (accumulator >> 3);
    return accumulator;
}

static PainterlyMetadata blend_metadata_pair(
    const PainterlyMetadata &a,
    const PainterlyMetadata &b,
    float weight_a,
    float weight_b) {

    PainterlyMetadata result;
    float total = weight_a + weight_b;
    if (total <= 0.0001f) {
        return a;
    }

    float wa = weight_a / total;
    float wb = weight_b / total;
    result.jitter = a.jitter * wa + b.jitter * wb;
    result.blue_noise = a.blue_noise * wa + b.blue_noise * wb;
    result.stroke_scale = a.stroke_scale * wa + b.stroke_scale * wb;

    Vector2 dir_a(Math::cos(a.stroke_angle), Math::sin(a.stroke_angle));
    Vector2 dir_b(Math::cos(b.stroke_angle), Math::sin(b.stroke_angle));
    Vector2 dir = dir_a * wa + dir_b * wb;
    if (dir.length_squared() > 0.0f) {
        result.stroke_angle = Math::atan2(dir.y, dir.x);
    } else {
        result.stroke_angle = a.stroke_angle;
    }

    result.stability = a.stability * wa + b.stability * wb;

    uint32_t combined_seed = 0x811C9DC5u;
    combined_seed = mix_seed(combined_seed, a.temporal_seed, wa);
    combined_seed = mix_seed(combined_seed, b.temporal_seed, wb);
    result.temporal_seed = combined_seed != 0 ? combined_seed : a.temporal_seed;

    return result;
}

static PainterlyMetadata blend_metadata_list(
    const Vector<GaussianData> &splats,
    const LocalVector<uint32_t> &indices,
    const LocalVector<float> &weights) {

    PainterlyMetadata result;
    if (indices.is_empty()) {
        return result;
    }

    Vector2 jitter_sum;
    Vector2 blue_sum;
    Vector2 angle_dir;
    float scale_sum = 0.0f;
    float stability_sum = 0.0f;
    uint32_t seed_mix = 0x9E3779B9u;

    for (uint32_t i = 0; i < indices.size(); i++) {
        const PainterlyMetadata &meta = splats[indices[i]].painterly;
        float weight = (i < weights.size()) ? weights[i] : 0.0f;
        jitter_sum += meta.jitter * weight;
        blue_sum += meta.blue_noise * weight;
        angle_dir += Vector2(Math::cos(meta.stroke_angle), Math::sin(meta.stroke_angle)) * weight;
        scale_sum += meta.stroke_scale * weight;
        stability_sum += meta.stability * weight;
        seed_mix = mix_seed(seed_mix, meta.temporal_seed, weight);
    }

    result.jitter = jitter_sum;
    result.blue_noise = blue_sum;
    result.stroke_scale = scale_sum;
    if (angle_dir.length_squared() > 0.0001f) {
        result.stroke_angle = Math::atan2(angle_dir.y, angle_dir.x);
    } else {
        result.stroke_angle = 0.0f;
    }
    result.stability = stability_sum;
    result.temporal_seed = seed_mix != 0 ? seed_mix : splats[indices[0]].painterly.temporal_seed;

    return result;
}

struct SpatialKey {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    bool operator==(const SpatialKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct SpatialKeyHasher {
    size_t operator()(const SpatialKey &key) const {
        const uint64_t hx = uint64_t(uint32_t(key.x)) * 73856093u;
        const uint64_t hy = uint64_t(uint32_t(key.y)) * 19349663u;
        const uint64_t hz = uint64_t(uint32_t(key.z)) * 83492791u;
        return size_t(hx ^ hy ^ hz);
    }
};

static inline SpatialKey spatial_key_from_position(const Vector3 &position, float inv_cell_size) {
    SpatialKey key;
    key.x = int32_t(Math::floor(position.x * inv_cell_size));
    key.y = int32_t(Math::floor(position.y * inv_cell_size));
    key.z = int32_t(Math::floor(position.z * inv_cell_size));
    return key;
}
} // namespace

SplatClusterer::SplatClusterer() {
}

SplatClusterer::~SplatClusterer() {
}

SplatClusterer::ClusteringResult SplatClusterer::cluster_splats(
    const Vector<GaussianData>& splats,
    const ClusteringParams& params) {

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    ClusteringResult result;
    result.stats.original_count = splats.size();

    if (splats.is_empty()) {
        return result;
    }

    // Choose clustering method
    switch (params.method) {
        case ClusteringParams::SPATIAL_CLUSTERING:
            result = cluster_spatial(splats, params.cluster_radius, params.target_count);
            break;

        case ClusteringParams::HIERARCHICAL_CLUSTERING:
            result = cluster_hierarchical(splats, params);
            break;

        case ClusteringParams::GRID_BASED:
            result = cluster_grid_based(splats, params.cluster_radius);
            break;

        case ClusteringParams::IMPORTANCE_WEIGHTED:
            // Generate default importance scores
            Vector<float> importance_scores;
            importance_scores.resize(splats.size());
            for (int i = 0; i < splats.size(); i++) {
                importance_scores.write[i] = splats[i].color.a;  // Use opacity as importance
            }
            result = cluster_by_importance(splats, importance_scores, params);
            break;
    }

    // Calculate statistics
    result.stats.cluster_count = result.clusters.size();
    result.reduction_ratio = 1.0f - (float(result.stats.cluster_count) / result.stats.original_count);

    float total_size = 0.0f;
    float max_radius = 0.0f;
    for (const auto& cluster : result.clusters) {
        total_size += cluster.source_count;
        float radius = cluster.scale.length();
        max_radius = MAX(max_radius, radius);
    }

    result.stats.avg_cluster_size = result.stats.cluster_count > 0 ?
        total_size / result.stats.cluster_count : 0.0f;
    result.stats.max_cluster_radius = max_radius;

    result.stats.processing_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;

    // Estimate quality
    result.quality_score = estimate_visual_quality(result, splats);

    return result;
}

SplatClusterer::ClusteringResult SplatClusterer::cluster_spatial(
    const Vector<GaussianData>& splats,
    float cluster_radius,
    uint32_t target_count) {

    ClusteringResult result;

    if (target_count == 0) {
        // Auto-determine target count based on reduction ratio
        target_count = MAX(1, splats.size() / 4);
    }

    // Use k-means clustering
    result.clusters.resize(target_count);
    kmeans_clustering(splats, target_count, result.clusters);

    return result;
}

SplatClusterer::ClusteringResult SplatClusterer::cluster_hierarchical(
    const Vector<GaussianData>& splats,
    const ClusteringParams& params) {

    ClusteringResult result;

    // Initialize cluster nodes
    Vector<ClusterNode> nodes;
    nodes.resize(splats.size());

    for (uint32_t i = 0; i < splats.size(); i++) {
        ClusterNode& node = nodes.write[i];
        node.splat_indices.push_back(i);
        node.centroid = splats[i].position;
        node.avg_color = splats[i].color;
        node.radius = splats[i].compute_radius();
        node.importance = splats[i].color.a;  // Use opacity as importance
        node.merged = false;
    }

    // Build hierarchy
    build_cluster_hierarchy(nodes, splats, params);

    // Convert remaining nodes to clusters
    for (const ClusterNode& node : nodes) {
        if (!node.merged && !node.splat_indices.is_empty()) {
            ClusteredSplat cluster = merge_splat_group(splats, node.splat_indices);
            result.clusters.push_back(cluster);
        }
    }

    return result;
}

SplatClusterer::ClusteringResult SplatClusterer::cluster_grid_based(
    const Vector<GaussianData>& splats,
    float grid_size) {

    ClusteringResult result;

    // Build spatial grid
    Vector<GridCell> grid;
    build_spatial_grid(splats, grid_size, grid);

    // Create cluster for each non-empty grid cell
    for (const GridCell& cell : grid) {
        if (!cell.splat_indices.is_empty()) {
            ClusteredSplat cluster = merge_splat_group(splats, cell.splat_indices);
            result.clusters.push_back(cluster);
        }
    }

    return result;
}

SplatClusterer::ClusteringResult SplatClusterer::cluster_by_importance(
    const Vector<GaussianData>& splats,
    const Vector<float>& importance_scores,
    const ClusteringParams& params) {

    ClusteringResult result;

    if (splats.size() != importance_scores.size()) {
        GS_LOG_RENDERER_ERROR("Importance scores size mismatch");
        return result;
    }

    // Sort splats by importance
    LocalVector<std::pair<float, uint32_t>> importance_pairs;
    importance_pairs.resize(splats.size());

    for (uint32_t i = 0; i < splats.size(); i++) {
        importance_pairs[i] = {importance_scores[i], i};
    }

    std::sort(
        importance_pairs.ptr(),
        importance_pairs.ptr() + importance_pairs.size(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Keep high-importance splats as-is, cluster low-importance ones
    LocalVector<bool> processed;
    processed.resize(splats.size());
    memset(processed.ptr(), 0, processed.size());

    float importance_threshold = 0.7f;  // Keep top 70% importance splats
    const float cell_size = MAX(params.cluster_radius, 0.001f);
    const float inv_cell_size = 1.0f / cell_size;
    const float cluster_radius_sq = params.cluster_radius * params.cluster_radius;

    std::unordered_map<SpatialKey, std::vector<uint32_t>, SpatialKeyHasher> low_importance_grid;
    low_importance_grid.reserve(splats.size());
    for (uint32_t i = 0; i < splats.size(); i++) {
        if (importance_scores[i] <= importance_threshold) {
            const SpatialKey key = spatial_key_from_position(splats[i].position, inv_cell_size);
            low_importance_grid[key].push_back(i);
        }
    }

    for (const auto& pair : importance_pairs) {
        uint32_t idx = pair.second;
        float importance = pair.first;

        if (processed[idx]) continue;

        if (importance > importance_threshold) {
            // High importance - keep as single cluster
            ClusteredSplat cluster;
            cluster.position = splats[idx].position;
            cluster.color = splats[idx].color;
            cluster.combined_opacity = splats[idx].color.a;
            cluster.rotation = splats[idx].rotation;
            cluster.scale = splats[idx].scale;
            cluster.source_count = 1;
            cluster.importance_sum = importance;
            cluster.original_index = idx;
            cluster.painterly = splats[idx].painterly;
            cluster.compute_covariance();

            result.clusters.push_back(cluster);
            processed[idx] = true;
        } else {
            // Low importance - cluster with neighbors
            LocalVector<uint32_t> cluster_indices;
            cluster_indices.push_back(idx);
            processed[idx] = true;

            // Find nearby low-importance splats via spatial hash buckets.
            const Vector3 center = splats[idx].position;
            const SpatialKey base_key = spatial_key_from_position(center, inv_cell_size);
            bool reached_cluster_limit = false;

            for (int32_t dz = -1; dz <= 1 && !reached_cluster_limit; dz++) {
                for (int32_t dy = -1; dy <= 1 && !reached_cluster_limit; dy++) {
                    for (int32_t dx = -1; dx <= 1; dx++) {
                        SpatialKey neighbor_key;
                        neighbor_key.x = base_key.x + dx;
                        neighbor_key.y = base_key.y + dy;
                        neighbor_key.z = base_key.z + dz;

                        auto bucket_it = low_importance_grid.find(neighbor_key);
                        if (bucket_it == low_importance_grid.end()) {
                            continue;
                        }

                        const std::vector<uint32_t> &bucket = bucket_it->second;
                        for (uint32_t candidate_idx : bucket) {
                            if (processed[candidate_idx] || importance_scores[candidate_idx] > importance_threshold) {
                                continue;
                            }

                            float dist_sq = (splats[candidate_idx].position - center).length_squared();
                            if (dist_sq < cluster_radius_sq) {
                                cluster_indices.push_back(candidate_idx);
                                processed[candidate_idx] = true;
                                if (cluster_indices.size() >= params.max_cluster_size) {
                                    reached_cluster_limit = true;
                                    break;
                                }
                            }
                        }

                        if (reached_cluster_limit) {
                            break;
                        }
                    }
                }
            }

            if (cluster_indices.size() >= params.min_cluster_size) {
                ClusteredSplat cluster = merge_splat_group(splats, cluster_indices, &importance_scores);
                result.clusters.push_back(cluster);
            } else {
                // Too small to cluster, keep as individual
                for (uint32_t k : cluster_indices) {
                    ClusteredSplat cluster;
                    cluster.position = splats[k].position;
                    cluster.color = splats[k].color;
                    cluster.combined_opacity = splats[k].color.a;
                    cluster.rotation = splats[k].rotation;
                    cluster.scale = splats[k].scale;
                    cluster.source_count = 1;
                    cluster.importance_sum = importance_scores[k];
                    cluster.original_index = k;
                    cluster.painterly = splats[k].painterly;
                    cluster.compute_covariance();

                    result.clusters.push_back(cluster);
                }
            }
        }
    }

    return result;
}

SplatClusterer::ClusteringResult SplatClusterer::generate_lod_clusters(
    const Vector<GaussianData>& splats,
    uint32_t lod_level) {

    ClusteringParams params;

    // Adjust parameters based on LOD level
    switch (lod_level) {
        case 0: {
            // No clustering for LOD 0
            ClusteringResult result;
            result.clusters.resize(splats.size());
            for (uint32_t i = 0; i < splats.size(); i++) {
                ClusteredSplat& cluster = result.clusters.write[i];
                cluster.position = splats[i].position;
                cluster.color = splats[i].color;
                cluster.combined_opacity = splats[i].color.a;
                cluster.rotation = splats[i].rotation;
                cluster.scale = splats[i].scale;
                cluster.source_count = 1;
                cluster.original_index = i;
                cluster.painterly = splats[i].painterly;
                cluster.compute_covariance();
            }
            return result;
        }

        case 1:
            params.cluster_radius = 2.0f;
            params.max_cluster_size = 4;
            params.target_count = splats.size() / 2;
            break;

        case 2:
            params.cluster_radius = 4.0f;
            params.max_cluster_size = 8;
            params.target_count = splats.size() / 4;
            break;

        case 3:
            params.cluster_radius = 8.0f;
            params.max_cluster_size = 16;
            params.target_count = splats.size() / 10;
            break;

        default:
            params.cluster_radius = 16.0f;
            params.max_cluster_size = 32;
            params.target_count = MAX(1, splats.size() / 20);
            break;
    }

    return cluster_splats(splats, params);
}

SplatClusterer::ClusteredSplat SplatClusterer::merge_splats(
    const GaussianData& splat1,
    const GaussianData& splat2,
    float weight1,
    float weight2) {

    ClusteredSplat result;

    float total_weight = weight1 + weight2;
    if (total_weight < 0.001f) {
        total_weight = 1.0f;
        weight1 = 0.5f;
        weight2 = 0.5f;
    }

    float norm_w1 = weight1 / total_weight;
    float norm_w2 = weight2 / total_weight;

    // Weighted average position
    result.position = splat1.position * norm_w1 + splat2.position * norm_w2;

    // Weighted average color
    result.color = splat1.color * norm_w1 + splat2.color * norm_w2;

    // Combined opacity (not just average)
    result.combined_opacity = 1.0f - (1.0f - splat1.color.a) * (1.0f - splat2.color.a);

    // Average rotation (simplified - should use quaternion slerp)
    result.rotation = splat1.rotation.slerp(splat2.rotation, norm_w2);

    // Combined scale (RMS for volume preservation)
    result.scale.x = sqrt(splat1.scale.x * splat1.scale.x * norm_w1 +
                          splat2.scale.x * splat2.scale.x * norm_w2);
    result.scale.y = sqrt(splat1.scale.y * splat1.scale.y * norm_w1 +
                          splat2.scale.y * splat2.scale.y * norm_w2);
    result.scale.z = sqrt(splat1.scale.z * splat1.scale.z * norm_w1 +
                          splat2.scale.z * splat2.scale.z * norm_w2);

    result.source_count = 2;
    result.importance_sum = weight1 + weight2;

    // Merge covariances
    merge_covariances(
        result.covariance,
        splat1.covariance,
        splat2.covariance,
        splat1.position,
        splat2.position,
        norm_w1,
        norm_w2
    );

    result.painterly = blend_metadata_pair(
        splat1.painterly,
        splat2.painterly,
        norm_w1,
        norm_w2
    );

    return result;
}

SplatClusterer::ClusteredSplat SplatClusterer::merge_splat_group(
    const Vector<GaussianData>& splats,
    const LocalVector<uint32_t>& indices,
    const Vector<float>* weights) {

    ClusteredSplat result;

    if (indices.is_empty()) {
        return result;
    }

    if (indices.size() == 1) {
        // Single splat
        uint32_t idx = indices[0];
        result.position = splats[idx].position;
        result.color = splats[idx].color;
        result.combined_opacity = splats[idx].color.a;
        result.rotation = splats[idx].rotation;
        result.scale = splats[idx].scale;
        result.source_count = 1;
        result.original_index = idx;
        result.painterly = splats[idx].painterly;
        memcpy(result.covariance, splats[idx].covariance, sizeof(result.covariance));
        return result;
    }

    // Calculate total weight
    float total_weight = 0.0f;
    LocalVector<float> normalized_weights;
    normalized_weights.resize(indices.size());

    if (weights && weights->size() > 0) {
        for (uint32_t i = 0; i < indices.size(); i++) {
            uint32_t idx = indices[i];
            float w = (idx < weights->size()) ? (*weights)[idx] : 1.0f;
            normalized_weights[i] = w;
            total_weight += w;
        }
    } else {
        // Use opacity as weight
        for (uint32_t i = 0; i < indices.size(); i++) {
            float w = splats[indices[i]].color.a;
            normalized_weights[i] = w;
            total_weight += w;
        }
    }

    if (total_weight < 0.001f) {
        total_weight = indices.size();
        for (uint32_t i = 0; i < indices.size(); i++) {
            normalized_weights[i] = 1.0f;
        }
    }

    // Normalize weights
    for (uint32_t i = 0; i < indices.size(); i++) {
        normalized_weights[i] /= total_weight;
    }

    // Compute weighted averages
    Vector3 avg_position;
    Color avg_color;
    Quaternion avg_rotation;
    Vector3 avg_scale;
    float opacity_product = 1.0f;
    float max_importance = 0.0f;
    uint32_t max_importance_idx = indices[0];

    for (uint32_t i = 0; i < indices.size(); i++) {
        uint32_t idx = indices[i];
        float w = normalized_weights[i];
        const GaussianData& splat = splats[idx];

        avg_position += splat.position * w;
        avg_color += splat.color * w;

        // For rotation, use slerp with first splat
        if (i == 0) {
            avg_rotation = splat.rotation;
        } else {
            avg_rotation = avg_rotation.slerp(splat.rotation, w);
        }

        // RMS for scale
        avg_scale.x += splat.scale.x * splat.scale.x * w;
        avg_scale.y += splat.scale.y * splat.scale.y * w;
        avg_scale.z += splat.scale.z * splat.scale.z * w;

        // Combined opacity
        opacity_product *= (1.0f - splat.color.a);

        // Track most important splat
        if (w > max_importance) {
            max_importance = w;
            max_importance_idx = idx;
        }
    }

    result.position = avg_position;
    result.color = avg_color;
    result.combined_opacity = 1.0f - opacity_product;
    result.rotation = avg_rotation.normalized();
    result.scale.x = sqrt(avg_scale.x);
    result.scale.y = sqrt(avg_scale.y);
   result.scale.z = sqrt(avg_scale.z);
    result.source_count = indices.size();
    result.importance_sum = total_weight;
    result.original_index = max_importance_idx;
    result.painterly = blend_metadata_list(splats, indices, normalized_weights);

    // Compute merged covariance
    // For simplicity, compute from final scale and rotation
    result.compute_covariance();

    return result;
}

void SplatClusterer::build_cluster_hierarchy(
    Vector<ClusterNode>& nodes,
    const Vector<GaussianData>& splats,
    const ClusteringParams& params) {

    bool any_merged = true;
    uint32_t iterations = 0;
    const uint32_t max_iterations = 100;
    const float cell_size = MAX(params.cluster_radius, 0.001f);
    const float inv_cell_size = 1.0f / cell_size;
    const uint32_t max_neighbor_checks = 64;

    while (any_merged && iterations < max_iterations) {
        any_merged = false;
        iterations++;

        std::unordered_map<SpatialKey, std::vector<int>, SpatialKeyHasher> active_grid;
        active_grid.reserve(nodes.size());
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes[i].merged) {
                continue;
            }
            const SpatialKey key = spatial_key_from_position(nodes[i].centroid, inv_cell_size);
            active_grid[key].push_back(i);
        }

        // Find best pair to merge
        float best_similarity = -1.0f;
        int best_i = -1, best_j = -1;

        for (int i = 0; i < nodes.size(); i++) {
            if (nodes[i].merged) continue;
            const SpatialKey base_key = spatial_key_from_position(nodes[i].centroid, inv_cell_size);

            uint32_t neighbor_checks = 0;
            bool reached_neighbor_budget = false;
            for (int32_t dz = -1; dz <= 1 && !reached_neighbor_budget; dz++) {
                for (int32_t dy = -1; dy <= 1 && !reached_neighbor_budget; dy++) {
                    for (int32_t dx = -1; dx <= 1; dx++) {
                        SpatialKey neighbor_key;
                        neighbor_key.x = base_key.x + dx;
                        neighbor_key.y = base_key.y + dy;
                        neighbor_key.z = base_key.z + dz;

                        auto bucket_it = active_grid.find(neighbor_key);
                        if (bucket_it == active_grid.end()) {
                            continue;
                        }

                        const std::vector<int> &bucket = bucket_it->second;
                        for (int j : bucket) {
                            if (j <= i || nodes[j].merged) {
                                continue;
                            }

                            if (!can_merge_clusters(nodes[i], nodes[j], params)) {
                                continue;
                            }

                            neighbor_checks++;
                            if (neighbor_checks > max_neighbor_checks) {
                                reached_neighbor_budget = true;
                                break;
                            }

                            float dist = (nodes[i].centroid - nodes[j].centroid).length();
                            float color_sim = compute_color_similarity(
                                splats[nodes[i].splat_indices[0]],
                                splats[nodes[j].splat_indices[0]]
                            );

                            float similarity = color_sim / (1.0f + dist);

                            if (similarity > best_similarity) {
                                best_similarity = similarity;
                                best_i = i;
                                best_j = j;
                            }
                        }

                        if (reached_neighbor_budget) {
                            break;
                        }
                    }
                }
            }
        }

        // Merge best pair
        if (best_i >= 0 && best_j >= 0 && best_similarity > params.merge_threshold) {
            ClusterNode merged = merge_cluster_nodes(
                nodes[best_i],
                nodes[best_j],
                splats
            );

            // Replace first node with merged, mark second as merged
            nodes.write[best_i] = merged;
            nodes.write[best_j].merged = true;
            any_merged = true;
        }

        // Check if we've reached target count
        if (params.target_count > 0) {
            uint32_t active_count = 0;
            for (const auto& node : nodes) {
                if (!node.merged) active_count++;
            }
            if (active_count <= params.target_count) {
                break;
            }
        }
    }
}

bool SplatClusterer::can_merge_clusters(
    const ClusterNode& node1,
    const ClusterNode& node2,
    const ClusteringParams& params) {

    // Check distance constraint
    float dist = (node1.centroid - node2.centroid).length();
    if (dist > params.cluster_radius) {
        return false;
    }

    // Check size constraint
    uint32_t combined_size = node1.splat_indices.size() + node2.splat_indices.size();
    if (combined_size > params.max_cluster_size) {
        return false;
    }

    // Check color similarity if required
    if (params.color_similarity) {
        Vector3 col1(node1.avg_color.r, node1.avg_color.g, node1.avg_color.b);
        Vector3 col2(node2.avg_color.r, node2.avg_color.g, node2.avg_color.b);
        float color_dist = (col1 - col2).length();
        if (color_dist > 0.5f) {  // Threshold for color difference
            return false;
        }
    }

    return true;
}

SplatClusterer::ClusterNode SplatClusterer::merge_cluster_nodes(
    const ClusterNode& node1,
    const ClusterNode& node2,
    const Vector<GaussianData>& splats) {

    ClusterNode result;

    // Combine splat indices
    result.splat_indices = node1.splat_indices;
    for (uint32_t idx : node2.splat_indices) {
        result.splat_indices.push_back(idx);
    }

    // Weighted average centroid
    float w1 = node1.splat_indices.size();
    float w2 = node2.splat_indices.size();
    float total = w1 + w2;

    result.centroid = (node1.centroid * w1 + node2.centroid * w2) / total;
    result.avg_color = (node1.avg_color * w1 + node2.avg_color * w2) / total;
    result.radius = MAX(node1.radius, node2.radius) +
                   (node1.centroid - node2.centroid).length() * 0.5f;
    result.importance = node1.importance + node2.importance;
    result.merged = false;

    return result;
}

void SplatClusterer::kmeans_clustering(
    const Vector<GaussianData>& splats,
    uint32_t k,
    Vector<ClusteredSplat>& clusters) {

    if (splats.is_empty() || k == 0) return;

    // Initialize cluster centers randomly
    LocalVector<Vector3> centers;
    centers.resize(k);

    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = Math::random(0, splats.size() - 1);
        centers[i] = splats[idx].position;
    }

    // K-means iterations
    const uint32_t max_iterations = 20;
    LocalVector<uint32_t> assignments;
    assignments.resize(splats.size());

    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        // Assign splats to nearest center
        for (uint32_t i = 0; i < splats.size(); i++) {
            float min_dist = FLT_MAX;
            uint32_t best_cluster = 0;

            for (uint32_t j = 0; j < k; j++) {
                float dist = (splats[i].position - centers[j]).length_squared();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update centers
        bool converged = true;
        for (uint32_t j = 0; j < k; j++) {
            Vector3 new_center;
            uint32_t count = 0;

            for (uint32_t i = 0; i < splats.size(); i++) {
                if (assignments[i] == j) {
                    new_center += splats[i].position;
                    count++;
                }
            }

            if (count > 0) {
                new_center /= float(count);
                if ((new_center - centers[j]).length() > 0.01f) {
                    converged = false;
                }
                centers[j] = new_center;
            }
        }

        if (converged) break;
    }

    // Create final clusters
    for (uint32_t j = 0; j < k; j++) {
        LocalVector<uint32_t> cluster_indices;
        for (uint32_t i = 0; i < splats.size(); i++) {
            if (assignments[i] == j) {
                cluster_indices.push_back(i);
            }
        }

        if (!cluster_indices.is_empty()) {
            ClusteredSplat cluster = merge_splat_group(splats, cluster_indices);
            clusters.write[j] = cluster;
        }
    }
}

void SplatClusterer::build_spatial_grid(
    const Vector<GaussianData>& splats,
    float grid_size,
    Vector<GridCell>& grid) {

    if (splats.is_empty() || grid_size <= 0.0f) return;

    // Find bounds
    AABB bounds;
    for (uint32_t i = 0; i < splats.size(); i++) {
        if (i == 0) {
            bounds.position = splats[i].position;
            bounds.size = Vector3();
        } else {
            bounds.expand_to(splats[i].position);
        }
    }

    // Calculate grid dimensions
    Vector3 grid_dims = bounds.size / grid_size;
    int grid_x = MAX(1, int(ceil(grid_dims.x)));
    int grid_y = MAX(1, int(ceil(grid_dims.y)));
    int grid_z = MAX(1, int(ceil(grid_dims.z)));

    uint32_t total_cells = grid_x * grid_y * grid_z;
    grid.resize(total_cells);

    // Initialize grid cells
    for (uint32_t i = 0; i < total_cells; i++) {
        uint32_t x = i % grid_x;
        uint32_t y = (i / grid_x) % grid_y;
        uint32_t z = i / (grid_x * grid_y);

        GridCell& cell = grid.write[i];
        cell.min_bound = bounds.position + Vector3(x, y, z) * grid_size;
        cell.max_bound = cell.min_bound + Vector3(grid_size, grid_size, grid_size);
    }

    // Assign splats to grid cells
    for (uint32_t i = 0; i < splats.size(); i++) {
        Vector3 rel_pos = splats[i].position - bounds.position;
        int x = MIN(grid_x - 1, int(rel_pos.x / grid_size));
        int y = MIN(grid_y - 1, int(rel_pos.y / grid_size));
        int z = MIN(grid_z - 1, int(rel_pos.z / grid_size));

        uint32_t cell_idx = x + y * grid_x + z * grid_x * grid_y;
        if (cell_idx < total_cells) {
            grid.write[cell_idx].splat_indices.push_back(i);
        }
    }
}

void SplatClusterer::merge_covariances(
    float result[6],
    const float cov1[6],
    const float cov2[6],
    const Vector3& pos1,
    const Vector3& pos2,
    float weight1,
    float weight2) {

    // Weighted average of covariances
    // This is a simplified approach - proper covariance merging is more complex

    for (int i = 0; i < 6; i++) {
        result[i] = cov1[i] * weight1 + cov2[i] * weight2;
    }

    // Add covariance due to position difference
    Vector3 delta = pos2 - pos1;
    float additional_cov[6] = {
        delta.x * delta.x * weight1 * weight2,  // xx
        delta.x * delta.y * weight1 * weight2,  // xy
        delta.x * delta.z * weight1 * weight2,  // xz
        delta.y * delta.y * weight1 * weight2,  // yy
        delta.y * delta.z * weight1 * weight2,  // yz
        delta.z * delta.z * weight1 * weight2   // zz
    };

    add_covariance_matrices(result, result, additional_cov);
}

void SplatClusterer::add_covariance_matrices(
    float result[6],
    const float a[6],
    const float b[6]) {

    for (int i = 0; i < 6; i++) {
        result[i] = a[i] + b[i];
    }
}

void SplatClusterer::scale_covariance_matrix(
    float result[6],
    const float input[6],
    float scale) {

    for (int i = 0; i < 6; i++) {
        result[i] = input[i] * scale;
    }
}

float SplatClusterer::compute_cluster_quality(
    const ClusteredSplat& cluster,
    const Vector<GaussianData>& source_splats,
    const LocalVector<uint32_t>& source_indices) {

    // Compute quality based on how well the cluster represents the source splats
    float quality = 0.0f;

    // Color variance
    Color avg_color = cluster.color;
    float color_variance = 0.0f;

    for (uint32_t idx : source_indices) {
        const GaussianData& splat = source_splats[idx];
        Vector3 color_diff(
            splat.color.r - avg_color.r,
            splat.color.g - avg_color.g,
            splat.color.b - avg_color.b
        );
        color_variance += color_diff.length_squared();
    }

    color_variance /= MAX(1u, source_indices.size());

    // Position variance
    float position_variance = 0.0f;
    for (uint32_t idx : source_indices) {
        const GaussianData& splat = source_splats[idx];
        float dist = (splat.position - cluster.position).length_squared();
        position_variance += dist;
    }
    position_variance /= MAX(1u, source_indices.size());

    // Quality score (lower variance = higher quality)
    quality = 1.0f / (1.0f + color_variance + position_variance * 0.01f);

    return quality;
}

float SplatClusterer::compute_color_similarity(
    const GaussianData& splat1,
    const GaussianData& splat2) {

    Vector3 col1(splat1.color.r, splat1.color.g, splat1.color.b);
    Vector3 col2(splat2.color.r, splat2.color.g, splat2.color.b);

    float dist = (col1 - col2).length();
    return 1.0f - MIN(1.0f, dist);
}

float SplatClusterer::compute_spatial_similarity(
    const GaussianData& splat1,
    const GaussianData& splat2,
    float max_distance) {

    float dist = (splat1.position - splat2.position).length();
    return 1.0f - MIN(1.0f, dist / max_distance);
}

float SplatClusterer::estimate_visual_quality(
    const ClusteringResult& result,
    const Vector<GaussianData>& original_splats) {

    // Estimate how well the clusters preserve visual appearance
    float quality = 1.0f;

    // Penalize high reduction ratio
    quality *= (1.0f - result.reduction_ratio * 0.5f);

    // Consider cluster sizes
    float avg_cluster_size = 0.0f;
    for (const auto& cluster : result.clusters) {
        avg_cluster_size += cluster.source_count;
    }
    avg_cluster_size /= MAX(1, result.clusters.size());

    // Prefer smaller clusters (better detail preservation)
    quality *= 1.0f / (1.0f + avg_cluster_size * 0.1f);

    return CLAMP(quality, 0.0f, 1.0f);
}

} // namespace GaussianSplatting

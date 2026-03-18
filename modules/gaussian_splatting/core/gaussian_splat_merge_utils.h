#ifndef GAUSSIAN_SPLAT_MERGE_UTILS_H
#define GAUSSIAN_SPLAT_MERGE_UTILS_H

#include "core/math/aabb.h"
#include "core/math/transform_3d.h"
#include "core/templates/vector.h"
#include "gaussian_data.h"
#include "gaussian_splat_asset.h"
#include "../renderer/gaussian_splat_renderer.h"

struct GaussianSplatMergeSource {
    Ref<GaussianSplatAsset> asset;
    Transform3D transform;
    bool is_2d = false;
};

struct GaussianSplatMergeResult {
    Ref<GaussianData> data;
    Vector<GaussianSplatRenderer::StaticChunk> chunks;
    AABB bounds;
};

bool gaussian_splat_merge_sources(const Vector<GaussianSplatMergeSource> &sources,
        float chunk_size, GaussianSplatMergeResult &out);

#endif // GAUSSIAN_SPLAT_MERGE_UTILS_H

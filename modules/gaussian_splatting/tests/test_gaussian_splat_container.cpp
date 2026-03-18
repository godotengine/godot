#include "test_macros.h"
#include "../nodes/gaussian_splat_container.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../core/gaussian_splat_asset.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

#ifdef TESTS_ENABLED

namespace {
Ref<GaussianSplatAsset> create_single_splat_asset(const Vector3 &p_position) {
    Ref<GaussianSplatAsset> asset;
    asset.instantiate();

    asset->set_splat_count(1);

    PackedFloat32Array positions;
    positions.resize(3);
    {
        float *ptr = positions.ptrw();
        ptr[0] = p_position.x;
        ptr[1] = p_position.y;
        ptr[2] = p_position.z;
    }
    asset->set_positions(positions);

    PackedFloat32Array scales;
    scales.resize(3);
    {
        float *ptr = scales.ptrw();
        ptr[0] = 1.0f;
        ptr[1] = 1.0f;
        ptr[2] = 1.0f;
    }
    asset->set_scales(scales);

    PackedFloat32Array rotations;
    rotations.resize(4);
    {
        float *ptr = rotations.ptrw();
        ptr[0] = 1.0f; // w
        ptr[1] = 0.0f;
        ptr[2] = 0.0f;
        ptr[3] = 0.0f;
    }
    asset->set_rotations(rotations);

    PackedFloat32Array sh_dc;
    sh_dc.resize(3);
    {
        float *ptr = sh_dc.ptrw();
        ptr[0] = 1.0f;
        ptr[1] = 1.0f;
        ptr[2] = 1.0f;
    }
    asset->set_sh_dc_coefficients(sh_dc);

    // Use opacity logits (sigmoid: opacity = 1/(1+exp(-logit)))
    // Large positive logit (~10) corresponds to opacity ~1.0
    PackedFloat32Array opacity_logits;
    opacity_logits.resize(1);
    opacity_logits.set(0, 10.0f);
    asset->set_opacity_logits(opacity_logits);

    return asset;
}
} // namespace

TEST_CASE("[GaussianSplatting][Container] Merges child splat nodes into chunked GaussianData") {
    SceneTree *tree = SceneTree::get_singleton();
    bool created_tree = false;

    if (!tree) {
        tree = memnew(SceneTree);
        tree->initialize();
        created_tree = true;
    }

    Window *root = tree->get_root();
    CHECK_MESSAGE(root != nullptr, "SceneTree root window must exist for GaussianSplatContainer test");
    if (root == nullptr) {
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }

    GaussianSplatContainer *container = memnew(GaussianSplatContainer);
    container->set_chunk_size(5.0f);
    root->add_child(container);

    const int splat_node_count = 100;
    const int grid_width = 10;
    const float spacing = 1.5f;

    for (int i = 0; i < splat_node_count; i++) {
        GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
        Vector3 local_position(0, 0, 0);
        Ref<GaussianSplatAsset> asset = create_single_splat_asset(local_position);
        node->set_splat_asset(asset);

        container->add_child(node);
        int x = i % grid_width;
        int z = i / grid_width;
        node->set_position(Vector3(x * spacing, 0.0f, z * spacing));
    }

    container->merge_children();

    Ref<GaussianData> merged = container->get_merged_data();
    CHECK(merged.is_valid());
    if (!merged.is_valid()) {
        root->remove_child(container);
        memdelete(container);
        if (created_tree) {
            tree->finalize();
            memdelete(tree);
        }
        return;
    }
    CHECK_EQ((int)merged->get_count(), splat_node_count);

    CHECK_LT(container->get_chunk_count(), 10);

    PackedInt32Array chunk_sizes = container->get_chunk_sizes();
    int total_indices = 0;
    for (int i = 0; i < chunk_sizes.size(); i++) {
        total_indices += chunk_sizes[i];
        CHECK_GT(chunk_sizes[i], 0);
    }
    CHECK_EQ(total_indices, splat_node_count);

    // Estimated workload: chunk count should meaningfully reduce frustum checks.
    int naive_checks = splat_node_count;
    int chunk_checks = container->get_chunk_count();
    CHECK_LT(chunk_checks, naive_checks);

    root->remove_child(container);
    memdelete(container);

    if (created_tree) {
        tree->finalize();
        memdelete(tree);
    }
}

#endif // TESTS_ENABLED

#ifndef GAUSSIAN_SPLAT_SCENE_DIRECTOR_H
#define GAUSSIAN_SPLAT_SCENE_DIRECTOR_H

#include "core/object/object.h"
#include "core/object/object_id.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/variant/variant.h"
#include "scene/resources/3d/world_3d.h"

#include "gaussian_data.h"
#include "gaussian_splat_asset.h"
#include "../lod/lod_config.h"
#include "../renderer/gaussian_splat_renderer.h"

class GaussianSplatSceneDirector : public Object {
    GDCLASS(GaussianSplatSceneDirector, Object);

public:
    enum InstanceWindMode : uint32_t {
        INSTANCE_WIND_INHERIT = 0u,
        INSTANCE_WIND_FORCE_DISABLED = 1u,
        INSTANCE_WIND_FORCE_ENABLED = 2u,
    };

    enum SubmissionResidencyHint : int32_t {
        SUBMISSION_RESIDENCY_HINT_RESIDENT = 0,
        SUBMISSION_RESIDENCY_HINT_STREAMING = 1,
    };

    struct InstanceSubmission {
        ObjectID node_id;
        RID scenario;
        Ref<GaussianSplatRenderer> renderer;
        Ref<GaussianSplatAsset> asset;
        Transform3D transform;
        float opacity = 1.0f;
        float lod_bias = 0.0f;
        float wind_intensity = 1.0f;
        uint32_t wind_mode = INSTANCE_WIND_INHERIT;
        Vector3 wind_direction = Vector3();
        float wind_frequency = 1.0f;
        uint32_t flags = 0;
        uint32_t last_lod = 0;
        bool casts_shadow = false;
        bool visible = true;
        bool has_desired_residency_hint = false;
        int32_t desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT;
    };

    struct WorldSubmission {
        ObjectID owner_id;
        RID scenario;
        Ref<GaussianData> gaussian_data;
        Vector<GaussianSplatRenderer::StaticChunk> static_chunks;
        AABB bounds;
        Dictionary metadata;
        bool has_desired_residency_hint = false;
        int32_t desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT;
        Dictionary desired_renderer_overrides;
    };

    struct SubmissionCounts {
        uint32_t instance_submissions = 0;
        uint32_t world_submissions = 0;
    };

    static GaussianSplatSceneDirector *get_singleton();

    GaussianSplatSceneDirector();
    ~GaussianSplatSceneDirector();

	void register_instance(ObjectID p_node_id, const Ref<GaussianSplatAsset> &p_asset, const Transform3D &p_transform,
			float p_opacity, float p_lod_bias, uint32_t p_flags, bool p_casts_shadow = false,
			float p_wind_intensity = 1.0f, uint32_t p_wind_mode = INSTANCE_WIND_INHERIT,
			const Vector3 &p_wind_direction = Vector3(), float p_wind_frequency = 1.0f,
			bool p_visible = true, bool p_has_desired_residency_hint = false,
			int32_t p_desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT);
	void update_instance_transform(ObjectID p_node_id, const Transform3D &p_transform);
	void update_instance_params(ObjectID p_node_id, float p_opacity, float p_lod_bias, uint32_t p_flags, bool p_casts_shadow = false,
			float p_wind_intensity = 1.0f, uint32_t p_wind_mode = INSTANCE_WIND_INHERIT,
			const Vector3 &p_wind_direction = Vector3(), float p_wind_frequency = 1.0f,
			bool p_visible = true, bool p_has_desired_residency_hint = false,
			int32_t p_desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT);
	void unregister_instance(ObjectID p_node_id);
	void update_instance_lods(const Vector3 &p_camera_pos, const LODConfig &p_lod_config, float p_hysteresis_zone);
    void update_instance_lods_for_renderer(const GaussianSplatRenderer *p_renderer, const Vector3 &p_camera_pos,
            const LODConfig &p_lod_config, float p_hysteresis_zone);
    void build_instance_buffer(LocalVector<InstanceDataGPU> &out) const;
	void build_instance_buffer_for_renderer(const GaussianSplatRenderer *p_renderer, LocalVector<InstanceDataGPU> &out,
			bool p_shadow_casters_only = false) const;
	uint32_t get_instance_count_for_renderer(const GaussianSplatRenderer *p_renderer) const;
	uint64_t get_instance_generation_for_renderer(const GaussianSplatRenderer *p_renderer) const;
    void register_instance_submission(ObjectID p_node_id, const Ref<GaussianSplatAsset> &p_asset,
            const Transform3D &p_transform, float p_opacity, float p_lod_bias, uint32_t p_flags,
            bool p_casts_shadow = false, float p_wind_intensity = 1.0f,
            uint32_t p_wind_mode = INSTANCE_WIND_INHERIT, const Vector3 &p_wind_direction = Vector3(),
            float p_wind_frequency = 1.0f, bool p_visible = true,
            bool p_has_desired_residency_hint = false,
            int32_t p_desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT);
    void update_instance_submission_transform(ObjectID p_node_id, const Transform3D &p_transform);
    void update_instance_submission_params(ObjectID p_node_id, float p_opacity, float p_lod_bias, uint32_t p_flags,
            bool p_casts_shadow = false, float p_wind_intensity = 1.0f,
            uint32_t p_wind_mode = INSTANCE_WIND_INHERIT, const Vector3 &p_wind_direction = Vector3(),
            float p_wind_frequency = 1.0f, bool p_visible = true,
            bool p_has_desired_residency_hint = false,
            int32_t p_desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT);
    void unregister_instance_submission(ObjectID p_node_id);
    bool get_instance_submission(ObjectID p_node_id, InstanceSubmission *r_submission) const;

	void collect_instance_assets_for_renderer(const GaussianSplatRenderer *p_renderer, LocalVector<InstanceAssetRegistration> &out,
			bool p_shadow_casters_only = false) const;
    // Runtime world-submission path. Applies the submitted payload to the shared renderer and
    // becomes the authoritative active world-backed source for the scenario.
    bool submit_world_submission(const WorldSubmission &p_submission);
    // Runtime inverse of submit_world_submission(). Clears renderer-owned world state and
    // releases the active world-backed source for this owner.
    void release_world_submission(ObjectID p_owner_id);
#if defined(TESTS_ENABLED) || defined(TOOLS_ENABLED)
    // Scaffolding/introspection helper. Stores or replaces the director-owned world submission
    // record without mutating renderer state. Kept public only for tools/test builds because
    // the current doctest umbrella still compiles against the editor/tools build surface.
    bool upsert_world_submission(const WorldSubmission &p_submission);
    // Scaffolding/introspection inverse of upsert_world_submission(). Removes only the stored
    // world submission record and intentionally leaves renderer state untouched.
    void unregister_world_submission(ObjectID p_owner_id);
#endif
    bool get_world_submission(ObjectID p_owner_id, WorldSubmission *r_submission) const;
    bool get_world_submission_for_scenario(const RID &p_scenario, WorldSubmission *r_submission) const;
    bool has_world_submission_for_renderer(const GaussianSplatRenderer *p_renderer) const;
    // Current hint precedence is active world > homogeneous instance submissions.
    // Conflicting instance submission hints return false with source "mixed_instance_submissions".
    bool get_submission_residency_hint_for_renderer(const GaussianSplatRenderer *p_renderer,
            int32_t *r_hint, String *r_source = nullptr) const;
    SubmissionCounts get_submission_counts() const;

    Ref<GaussianSplatRenderer> get_shared_renderer(World3D *p_world);

protected:
    static void _bind_methods();

private:
    struct InstanceRecord {
        ObjectID node_id;
        Transform3D transform;
        float opacity = 1.0f;
        float lod_bias = 0.0f;
        float wind_intensity = 1.0f;
        uint32_t wind_mode = INSTANCE_WIND_INHERIT;
		Vector3 wind_direction = Vector3();
		float wind_frequency = 1.0f;
		uint32_t asset_id = 0;
		uint32_t flags = 0;
        uint32_t last_lod = 0;
        bool casts_shadow = false;
        bool visible = true;
        bool has_desired_residency_hint = false;
        int32_t desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT;
        bool dirty = true;
	};

    struct SharedWorld {
        RID scenario;
        Ref<GaussianSplatRenderer> renderer;
        LocalVector<InstanceRecord> instances;
        HashMap<ObjectID, uint32_t> instance_lookup;
        uint64_t instance_generation = 1;
        struct WorldSubmissionRecord {
            ObjectID owner_id;
            Ref<GaussianData> gaussian_data;
            Vector<GaussianSplatRenderer::StaticChunk> static_chunks;
            AABB bounds;
            Dictionary metadata;
            bool has_desired_residency_hint = false;
            int32_t desired_residency_hint = SUBMISSION_RESIDENCY_HINT_RESIDENT;
            Dictionary desired_renderer_overrides;
            bool active = false;
        };
        WorldSubmissionRecord world_submission;
        struct AssetRecord {
            Ref<GaussianSplatAsset> asset;
            Ref<GaussianData> data;
            uint32_t refcount = 0;
            uint32_t edited_version = 0;
        };
        HashMap<uint32_t, AssetRecord> asset_records;
    };

    static GaussianSplatSceneDirector *singleton;

    mutable Mutex world_mutex;
    HashMap<RID, SharedWorld> worlds;

    SharedWorld *_get_or_create_world_for_scenario(const RID &p_scenario);
    SharedWorld *_get_or_create_world(World3D *p_world);
    SharedWorld *_get_world_for_instance(ObjectID p_node_id);
    SharedWorld *_find_world_for_instance(ObjectID p_node_id);
    SharedWorld *_find_world_for_renderer(const GaussianSplatRenderer *p_renderer);
    const SharedWorld *_find_world_for_renderer(const GaussianSplatRenderer *p_renderer) const;
    SharedWorld *_find_world_for_world_submission(ObjectID p_owner_id);
    const SharedWorld *_find_world_for_world_submission(ObjectID p_owner_id) const;

    static bool _populate_gaussian_data_from_asset(const Ref<GaussianSplatAsset> &p_asset, Ref<GaussianData> &r_data);
    static bool _retain_asset_record(SharedWorld &p_world, const Ref<GaussianSplatAsset> &p_asset, uint32_t p_asset_id);
    static bool _refresh_asset_record(SharedWorld &p_world, const Ref<GaussianSplatAsset> &p_asset, uint32_t p_asset_id);
    static void _release_asset_record(SharedWorld &p_world, uint32_t p_asset_id);
    static bool _is_world_submission_owner_live(ObjectID p_owner_id);
    static void _store_world_submission_record(SharedWorld::WorldSubmissionRecord &r_record, const WorldSubmission &p_submission);
    static void _copy_world_submission_record(const SharedWorld &p_world, const SharedWorld::WorldSubmissionRecord &p_record,
            WorldSubmission *r_submission);
    static void _clear_world_submission_renderer(SharedWorld &p_world);
    static bool _apply_world_submission_to_renderer(SharedWorld &p_world, const SharedWorld::WorldSubmissionRecord &p_record);
	bool _should_prune_world(const SharedWorld &p_world) const;
	void _prune_world_if_unused(const RID &p_scenario);
};

#endif // GAUSSIAN_SPLAT_SCENE_DIRECTOR_H

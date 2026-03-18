#ifndef GAUSSIAN_SPLAT_SCENE_DIRECTOR_H
#define GAUSSIAN_SPLAT_SCENE_DIRECTOR_H

#include "core/object/object.h"
#include "core/object/object_id.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/math/transform_3d.h"
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

    static GaussianSplatSceneDirector *get_singleton();

    GaussianSplatSceneDirector();
    ~GaussianSplatSceneDirector();

    void register_instance(ObjectID p_node_id, const Ref<GaussianSplatAsset> &p_asset, const Transform3D &p_transform,
            float p_opacity, float p_lod_bias, uint32_t p_flags,
            float p_wind_intensity = 1.0f, uint32_t p_wind_mode = INSTANCE_WIND_INHERIT,
            const Vector3 &p_wind_direction = Vector3(), float p_wind_frequency = 1.0f);
    void update_instance_transform(ObjectID p_node_id, const Transform3D &p_transform);
    void update_instance_params(ObjectID p_node_id, float p_opacity, float p_lod_bias, uint32_t p_flags,
            float p_wind_intensity = 1.0f, uint32_t p_wind_mode = INSTANCE_WIND_INHERIT,
            const Vector3 &p_wind_direction = Vector3(), float p_wind_frequency = 1.0f);
    void unregister_instance(ObjectID p_node_id);
    void update_instance_lods(const Vector3 &p_camera_pos, const LODConfig &p_lod_config, float p_hysteresis_zone);
    void update_instance_lods_for_renderer(const GaussianSplatRenderer *p_renderer, const Vector3 &p_camera_pos,
            const LODConfig &p_lod_config, float p_hysteresis_zone);
    void build_instance_buffer(LocalVector<InstanceDataGPU> &out) const;
    void build_instance_buffer_for_renderer(const GaussianSplatRenderer *p_renderer, LocalVector<InstanceDataGPU> &out) const;
    uint64_t get_instance_generation_for_renderer(const GaussianSplatRenderer *p_renderer) const;

    void collect_instance_assets_for_renderer(const GaussianSplatRenderer *p_renderer, LocalVector<InstanceAssetRegistration> &out) const;

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
        bool dirty = true;
    };

    struct SharedWorld {
        RID scenario;
        Ref<GaussianSplatRenderer> renderer;
        LocalVector<InstanceRecord> instances;
        HashMap<ObjectID, uint32_t> instance_lookup;
        uint64_t instance_generation = 1;
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

    SharedWorld *_get_or_create_world(World3D *p_world);
    SharedWorld *_get_world_for_instance(ObjectID p_node_id);
    SharedWorld *_find_world_for_instance(ObjectID p_node_id);
    SharedWorld *_find_world_for_renderer(const GaussianSplatRenderer *p_renderer);
    const SharedWorld *_find_world_for_renderer(const GaussianSplatRenderer *p_renderer) const;

    static bool _populate_gaussian_data_from_asset(const Ref<GaussianSplatAsset> &p_asset, Ref<GaussianData> &r_data);
    static bool _retain_asset_record(SharedWorld &p_world, const Ref<GaussianSplatAsset> &p_asset, uint32_t p_asset_id);
    static bool _refresh_asset_record(SharedWorld &p_world, const Ref<GaussianSplatAsset> &p_asset, uint32_t p_asset_id);
    static void _release_asset_record(SharedWorld &p_world, uint32_t p_asset_id);
};

#endif // GAUSSIAN_SPLAT_SCENE_DIRECTOR_H

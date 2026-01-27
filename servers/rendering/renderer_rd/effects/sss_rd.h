#pragma once

#include "servers/rendering/renderer_rd/pipeline_deferred_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/ss_shadow.glsl.gen.h"
#include "servers/rendering/rendering_server.h"

class RenderSceneBuffersRD;

namespace RendererRD {

class SSSEffects {
private:
    static SSSEffects *singleton;

    struct ScreenSpaceShadowsPushConstant {
        float screen_size_rcp[2];
        float screen_size[2];
        float light_dir[3];
        float thickness;
        float max_dist;
        float intensity;
        uint32_t sample_count;
        uint32_t use_normals;
        float camera_pos[3];
        uint32_t frame_count;
        float light_radius;
        float thickness_falloff;
        float contact_shadow_distance;
        float shadow_fade_range;
        float history_blend;
        uint32_t use_history;
        float history_pad[2];
    };

    struct ScreenSpaceShadowsData {
        float proj[16];
        float proj_inv[16];
        float view[16];
        float view_inv[16];
    };

    struct ScreenSpaceShadows {
        ScreenSpaceShadowsPushConstant push_constant;
        SsShadowShaderRD shader;
        RID shader_version;
        RID pipelines[1];
        RID ubo;
        RID history_textures[2];
        RID history_dummy;
        Size2i history_size;
        uint32_t history_index = 0;
    } screen_space_shadows;

public:
    static SSSEffects *get_singleton() { return singleton; }

    SSSEffects();
    ~SSSEffects();

    void gen_screen_space_shadows(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_normal_roughness, float p_depth_threshold, float p_trace_distance, float p_shadow_weight, int p_ray_steps, float p_light_radius, float p_thickness_falloff, float p_contact_distance, float p_fade_range, float p_history_weight, uint32_t p_frame_count, const Vector3 &p_camera_position, Transform3D light_dir, Projection p_projection, Transform3D p_view);
};

} // namespace RendererRD

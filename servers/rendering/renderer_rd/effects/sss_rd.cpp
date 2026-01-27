#include "sss_rd.h"

#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"

namespace RendererRD {

SSSEffects *SSSEffects::singleton = nullptr;

SSSEffects::SSSEffects() {
    singleton = this;

    Vector<String> sss_modes;
    sss_modes.push_back("\n");
    screen_space_shadows.shader.initialize(sss_modes);
    screen_space_shadows.shader_version = screen_space_shadows.shader.version_create();
    screen_space_shadows.pipelines[0] = RD::get_singleton()->compute_pipeline_create(screen_space_shadows.shader.version_get_shader(screen_space_shadows.shader_version, 0));
    screen_space_shadows.ubo = RD::get_singleton()->uniform_buffer_create(sizeof(ScreenSpaceShadowsData));

    RD::TextureFormat tf;
    tf.format = RD::DATA_FORMAT_R8_UNORM;
    tf.width = 1;
    tf.height = 1;
    tf.mipmaps = 1;
    tf.array_layers = 1;
    tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
    Vector<Vector<uint8_t>> dummy;
    dummy.resize(1);
    dummy.write[0].resize(1);
    dummy.write[0].write[0] = 0;
    screen_space_shadows.history_dummy = RD::get_singleton()->texture_create(tf, RD::TextureView(), dummy);
    RD::get_singleton()->texture_clear(screen_space_shadows.history_dummy, Color(0, 0, 0, 0), 0, 1, 0, 1);
}

SSSEffects::~SSSEffects() {
    screen_space_shadows.shader.version_free(screen_space_shadows.shader_version);
    RD::get_singleton()->free_rid(screen_space_shadows.pipelines[0]);
    RD::get_singleton()->free_rid(screen_space_shadows.ubo);

    for (int i = 0; i < 2; i++) {
        if (screen_space_shadows.history_textures[i].is_valid()) {
            RD::get_singleton()->free_rid(screen_space_shadows.history_textures[i]);
        }
    }
    if (screen_space_shadows.history_dummy.is_valid()) {
        RD::get_singleton()->free_rid(screen_space_shadows.history_dummy);
    }

    singleton = nullptr;
}

void SSSEffects::gen_screen_space_shadows(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_normal_roughness, float p_depth_threshold, float p_trace_distance, float p_shadow_weight, int p_ray_steps, float p_light_radius, float p_thickness_falloff, float p_contact_distance, float p_fade_range, float p_history_weight, uint32_t p_frame_count, const Vector3 &p_camera_position, Transform3D light_dir, Projection p_projection, Transform3D p_view) {
    UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
    ERR_FAIL_NULL(uniform_set_cache);
    MaterialStorage *material_storage = MaterialStorage::get_singleton();
    ERR_FAIL_NULL(material_storage);

    RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
    Size2i screen_size = p_render_buffers->get_internal_size();
    RID depth_texture = p_render_buffers->get_depth_texture();

    // Ensure shadow texture is allocated
    if (!p_render_buffers->has_texture(SNAME("SSS"), SNAME("Shadow"))) {
        p_render_buffers->create_texture(SNAME("SSS"), SNAME("Shadow"), RD::DATA_FORMAT_R8_UNORM, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT);
    }
    RID shadow_texture = p_render_buffers->get_texture(SNAME("SSS"), SNAME("Shadow"));
    RD::get_singleton()->texture_clear(shadow_texture, Color(1, 1, 1, 1), 0, 1, 0, 1);

    bool use_normals = p_normal_roughness.is_valid();
	RID normal_texture = use_normals ? p_normal_roughness : depth_texture;

    bool history_requested = (p_history_weight > 0.0f) && p_render_buffers->get_view_count() == 1;

    if (history_requested) {
		if (screen_space_shadows.history_textures[0].is_null() || screen_space_shadows.history_size != screen_size) {
			for (int i = 0; i < 2; i++) {
				if (screen_space_shadows.history_textures[i].is_valid()) {
					RD::get_singleton()->free_rid(screen_space_shadows.history_textures[i]);
				}
				RD::TextureFormat tf;
				tf.format = RD::DATA_FORMAT_R8_UNORM;
				tf.width = screen_size.x;
				tf.height = screen_size.y;
				tf.mipmaps = 1;
				tf.array_layers = 1;
				tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
				Vector<Vector<uint8_t>> dummy;
				screen_space_shadows.history_textures[i] = RD::get_singleton()->texture_create(tf, RD::TextureView(), dummy);
				RD::get_singleton()->texture_clear(screen_space_shadows.history_textures[i], Color(0, 0, 0, 0), 0, 1, 0, 1);
			}
			screen_space_shadows.history_size = screen_size;
			screen_space_shadows.history_index = 0;
		}
	}

    RID history_read = history_requested ? screen_space_shadows.history_textures[screen_space_shadows.history_index] : screen_space_shadows.history_dummy;
	RID history_write = history_requested ? screen_space_shadows.history_textures[screen_space_shadows.history_index ^ 1] : screen_space_shadows.history_dummy;


    // Keep the light direction in world space; the shader reconstructs positions in world space too.
    auto dir = light_dir.basis.get_quaternion().normalized().xform(Vector3(0, 0, -1));

    ScreenSpaceShadowsData data;
    MaterialStorage::store_camera(p_projection, data.proj);
    MaterialStorage::store_camera(p_projection.inverse(), data.proj_inv);
    MaterialStorage::store_transform(p_view, data.view_inv);
    MaterialStorage::store_transform(p_view.affine_inverse(), data.view);
    RD::get_singleton()->buffer_update(screen_space_shadows.ubo, 0, sizeof(ScreenSpaceShadowsData), &data);

    screen_space_shadows.push_constant.screen_size[0] = screen_size.x;
    screen_space_shadows.push_constant.screen_size[1] = screen_size.y;
    screen_space_shadows.push_constant.screen_size_rcp[0] = 1.0f / screen_size.x;
    screen_space_shadows.push_constant.screen_size_rcp[1] = 1.0f / screen_size.y;
    screen_space_shadows.push_constant.light_dir[0] = dir.x;
    screen_space_shadows.push_constant.light_dir[1] = dir.y;
    screen_space_shadows.push_constant.light_dir[2] = dir.z;
    screen_space_shadows.push_constant.intensity = CLAMP(p_shadow_weight, 0.0f, 1.0f);
    screen_space_shadows.push_constant.sample_count = MAX(1, p_ray_steps);
    screen_space_shadows.push_constant.use_normals = use_normals ? 1 : 0;
    screen_space_shadows.push_constant.light_radius = MAX(p_light_radius, 0.0f);
	screen_space_shadows.push_constant.thickness_falloff = MAX(p_thickness_falloff, 0.0f);
	screen_space_shadows.push_constant.contact_shadow_distance = MAX(p_contact_distance, 0.0f);
	screen_space_shadows.push_constant.shadow_fade_range = MAX(p_fade_range, 0.0f);
	bool history_available = history_requested;
	float history_strength = CLAMP(p_history_weight, 0.0f, 1.0f);
	screen_space_shadows.push_constant.history_blend = history_available ? (1.0f - history_strength) : 1.0f;
	screen_space_shadows.push_constant.use_history = history_available ? 1 : 0;
	screen_space_shadows.push_constant.camera_pos[0] = p_camera_position.x;
	screen_space_shadows.push_constant.camera_pos[1] = p_camera_position.y;
	screen_space_shadows.push_constant.camera_pos[2] = p_camera_position.z;
	screen_space_shadows.push_constant.frame_count = p_frame_count;
    
    RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

    RID shader = screen_space_shadows.shader.version_get_shader(screen_space_shadows.shader_version, 0);
    RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, screen_space_shadows.pipelines[0]);
    
    // Force recompile
    RD::Uniform u_depth_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, {default_sampler, depth_texture});
    RD::Uniform u_shadow_image(RD::UNIFORM_TYPE_IMAGE, 1, {shadow_texture});
    RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_depth_sampler, u_shadow_image), 0);

    RD::Uniform u_normal_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, {default_sampler, normal_texture});
    RD::Uniform u_history_sampler(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, {default_sampler, history_read});
    RD::Uniform u_history_write(RD::UNIFORM_TYPE_IMAGE, 2, {history_write});
    RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_normal_sampler, u_history_sampler, u_history_write), 1);
    
    RD::Uniform u_data(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, screen_space_shadows.ubo);
    RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 2, u_data), 2);

    RD::get_singleton()->compute_list_set_push_constant(compute_list, &screen_space_shadows.push_constant, sizeof(ScreenSpaceShadowsPushConstant));
    RD::get_singleton()->compute_list_dispatch_threads(compute_list, screen_size.x, screen_size.y, 1);
    RD::get_singleton()->compute_list_end();

    if (history_available) {
		screen_space_shadows.history_index ^= 1;
	}
}

} // namespace RendererRD

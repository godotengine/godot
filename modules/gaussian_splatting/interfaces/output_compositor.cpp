// Phase 9: Output compositor extracted from GaussianSplatRenderer
// Handles framebuffer caching, viewport blitting, and final output composition

#include "output_compositor.h"
#include "../logger/gs_logger.h"
#include "../core/gaussian_splat_manager.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/gpu_debug_utils.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/effects/copy_effects.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "../renderer/shader_compilation_helper.h"
#include "core/crypto/crypto_core.h"
#include "../shaders/viewport_blit.glsl.gen.h"

using GaussianSplatting::ScopedGpuMarker;

namespace {
constexpr char GS_COMPOSITE_DEPTH_TEST_SETTING[] = "rendering/gaussian_splatting/composite/depth_test";
constexpr bool GS_COMPOSITE_DEPTH_TEST_DEFAULT = true;
constexpr char GS_SCENE_COMPOSITE_DEPTH_POLICY_SETTING[] = "rendering/gaussian_splatting/composite/scene_depth_policy";

enum GSSceneCompositeDepthPolicy {
    GS_SCENE_COMPOSITE_DEPTH_POLICY_STRICT = 0,
    GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED = 1,
};

constexpr GSSceneCompositeDepthPolicy GS_SCENE_COMPOSITE_DEPTH_POLICY_DEFAULT = GS_SCENE_COMPOSITE_DEPTH_POLICY_STRICT;

bool gs_get_composite_depth_test_enabled() {
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (!project_settings || !project_settings->has_setting(GS_COMPOSITE_DEPTH_TEST_SETTING)) {
        return GS_COMPOSITE_DEPTH_TEST_DEFAULT;
    }

    return (bool)project_settings->get_setting_with_override(GS_COMPOSITE_DEPTH_TEST_SETTING);
}

GSSceneCompositeDepthPolicy gs_get_scene_composite_depth_policy() {
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (!project_settings || !project_settings->has_setting(GS_SCENE_COMPOSITE_DEPTH_POLICY_SETTING)) {
        return GS_SCENE_COMPOSITE_DEPTH_POLICY_DEFAULT;
    }

    Variant value = project_settings->get_setting_with_override(GS_SCENE_COMPOSITE_DEPTH_POLICY_SETTING);
    switch (value.get_type()) {
        case Variant::BOOL: {
            return value.operator bool() ? GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED : GS_SCENE_COMPOSITE_DEPTH_POLICY_STRICT;
        }
        case Variant::INT: {
            int64_t raw = value.operator int64_t();
            return raw >= GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED ? GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED : GS_SCENE_COMPOSITE_DEPTH_POLICY_STRICT;
        }
        case Variant::FLOAT: {
            double raw = value.operator double();
            return raw >= double(GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED) ? GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED : GS_SCENE_COMPOSITE_DEPTH_POLICY_STRICT;
        }
        case Variant::STRING:
        case Variant::STRING_NAME: {
            String policy = ((String)value).strip_edges().to_lower();
            if (policy == "relaxed") {
                return GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED;
            }
            if (policy == "strict") {
                return GS_SCENE_COMPOSITE_DEPTH_POLICY_STRICT;
            }
            break;
        }
        default:
            break;
    }

    return GS_SCENE_COMPOSITE_DEPTH_POLICY_DEFAULT;
}

void gs_log_scene_depth_contract_skip_once(bool p_missing_source_depth, bool p_missing_destination_depth) {
    static bool warned_once = false;
    if (warned_once) {
        return;
    }
    warned_once = true;

    String missing_inputs = "source_depth";
    if (p_missing_source_depth && p_missing_destination_depth) {
        missing_inputs = "source_depth + scene_depth";
    } else if (p_missing_destination_depth) {
        missing_inputs = "scene_depth";
    }

    GS_LOG_WARN(gs_logger::Category::COMPOSITOR,
            vformat("[OutputCompositor] Scene compositing skipped: missing required %s while strict depth policy is active. "
                    "Set %s=1 (or 'relaxed') to allow no-depth blend fallback.",
                    missing_inputs, GS_SCENE_COMPOSITE_DEPTH_POLICY_SETTING));
}
} // namespace

void OutputCompositor::_bind_methods() {
    // Bind methods for script access if needed
}

OutputCompositor::OutputCompositor() {
}

OutputCompositor::~OutputCompositor() {
    shutdown();
}

Error OutputCompositor::initialize(RenderingDevice *p_device) {
    if (!p_device) {
        return ERR_INVALID_PARAMETER;
    }

    rd = p_device;
    initialized = true;
    return OK;
}

void OutputCompositor::shutdown() {
    clear_cached_framebuffers();
    clear_viewport_blit_resources();

    if (viewport_blit_shader_source) {
        delete viewport_blit_shader_source;
        viewport_blit_shader_source = nullptr;
    }
    viewport_blit_shader_source_initialized = false;

    final_render_texture = RID();
    output_cache = OutputCacheState();
    cached_render_reuse_enabled = true;

    initialized = false;
    rd = nullptr;
}

void OutputCompositor::invalidate_cached_render() {
    output_cache.cached_render_valid = false;
    output_cache.cached_render_depth = RID();
    output_cache.cached_render_content_generation = 0;
    output_cache.cached_render_cull_config_signature = 0;
    output_cache.cached_render_color_grading_signature = 0;
    output_cache.cached_render_lighting_signature = 0;
}

bool OutputCompositor::_is_depth_texture_valid(const RID &p_depth_texture) const {
    if (!p_depth_texture.is_valid()) {
        return false;
    }
    if (rd && !rd->texture_is_valid(p_depth_texture)) {
        return false;
    }
    return true;
}

bool OutputCompositor::can_reuse_cached_render(const Transform3D &p_world_to_camera_to_world_transform, const Projection &p_projection,
        const Size2i &p_viewport_size, bool p_painterly_active, const RID &p_final_render_texture,
        uint64_t p_content_generation,
        uint64_t p_cull_config_signature,
        uint64_t p_color_grading_signature, uint64_t p_lighting_signature,
        bool p_require_valid_depth) const {
    if (!cached_render_reuse_enabled) {
        return false;
    }
    if (!output_cache.cached_render_valid || !output_cache.has_valid_render || !p_final_render_texture.is_valid()) {
        return false;
    }
    if (output_cache.cached_render_painterly != p_painterly_active) {
        return false;
    }
    if (output_cache.cached_render_viewport_size != p_viewport_size) {
        return false;
    }
    if (output_cache.cached_render_camera_projection != p_projection) {
        return false;
    }
    if (!output_cache.cached_render_camera_to_world_transform.is_equal_approx(p_world_to_camera_to_world_transform)) {
        return false;
    }
    if (output_cache.cached_render_content_generation != p_content_generation) {
        return false;
    }
    if (output_cache.cached_render_cull_config_signature != p_cull_config_signature) {
        return false;
    }
    if (output_cache.cached_render_color_grading_signature != p_color_grading_signature) {
        return false;
    }
    if (output_cache.cached_render_lighting_signature != p_lighting_signature) {
        return false;
    }
    if (p_require_valid_depth && !_is_depth_texture_valid(output_cache.cached_render_depth)) {
        return false;
    }
    return true;
}

void OutputCompositor::update_render_cache_signature(const Transform3D &p_world_to_camera_to_world_transform, const Projection &p_projection,
        const Size2i &p_viewport_size, bool p_painterly_active, const RID &p_cached_depth,
        const Size2i &p_internal_size, const RID &p_final_render_texture,
        uint64_t p_content_generation,
        uint64_t p_cull_config_signature,
        uint64_t p_color_grading_signature, uint64_t p_lighting_signature,
        bool p_require_valid_depth) {
    const bool cached_depth_valid = _is_depth_texture_valid(p_cached_depth);
    output_cache.cached_render_camera_to_world_transform = p_world_to_camera_to_world_transform;
    output_cache.cached_render_camera_projection = p_projection;
    output_cache.cached_render_viewport_size = p_viewport_size;
    output_cache.cached_render_internal_size = p_internal_size;
    output_cache.cached_render_painterly = p_painterly_active;
    output_cache.cached_render_depth = cached_depth_valid ? p_cached_depth : RID();
    output_cache.cached_render_valid = p_final_render_texture.is_valid() && (!p_require_valid_depth || cached_depth_valid);
    output_cache.cached_render_content_generation = p_content_generation;
    output_cache.cached_render_cull_config_signature = p_cull_config_signature;
    output_cache.cached_render_color_grading_signature = p_color_grading_signature;
    output_cache.cached_render_lighting_signature = p_lighting_signature;
}

#ifdef TESTS_ENABLED
void OutputCompositor::test_reset_last_viewport_copy_state() {
    output_cache.last_viewport_copy_success = false;
    output_cache.last_viewport_copy_source_size = Size2i();
    output_cache.last_viewport_copy_dest_size = Size2i();
}
#endif

void OutputCompositor::set_device_manager(Ref<RenderDeviceManager> p_device_manager) {
    device_manager = p_device_manager;
}

// Helper methods delegating to device manager or callbacks
RD::TextureFormat OutputCompositor::_get_texture_format(RenderingDevice *p_device, RID p_texture) const {
    if (texture_format_callback) {
        return texture_format_callback(p_device, p_texture);
    }

    if (!p_texture.is_valid() || !p_device) {
        return RD::TextureFormat();
    }

    if (p_device->texture_is_valid(p_texture)) {
        return p_device->texture_get_format(p_texture);
    }

    return RD::TextureFormat();
}

bool OutputCompositor::_is_texture_srgb(RenderingDevice *p_device, RID p_texture) {
    if (!p_texture.is_valid() || !p_device) {
        return false;
    }

    const uint64_t key = p_texture.get_id();
    if (bool *cached = srgb_format_cache.getptr(key)) {
        if (p_device->texture_is_valid(p_texture)) {
            return *cached;
        }
        srgb_format_cache.erase(key);
    }

    if (!p_device->texture_is_valid(p_texture)) {
        return false;
    }

    RD::TextureFormat format = _get_texture_format(p_device, p_texture);
    bool srgb = _is_srgb_format(format.format);
    srgb_format_cache.insert(key, srgb);
    return srgb;
}

void OutputCompositor::_track_resource(const RID &p_rid, RenderingDevice *p_device, bool p_owned, const char *p_label) {
    if (device_manager.is_valid()) {
        device_manager->track_resource(p_rid, p_device, p_owned, p_label);
    }
}

void OutputCompositor::_forget_resource(const RID &p_rid) {
    if (device_manager.is_valid()) {
        device_manager->forget_resource(p_rid);
    }
}

// Static helper methods
bool OutputCompositor::_is_depth_attachment_format(RD::DataFormat p_format) {
    switch (p_format) {
        case RD::DATA_FORMAT_D16_UNORM:
        case RD::DATA_FORMAT_X8_D24_UNORM_PACK32:
        case RD::DATA_FORMAT_D32_SFLOAT:
        case RD::DATA_FORMAT_S8_UINT:
        case RD::DATA_FORMAT_D16_UNORM_S8_UINT:
        case RD::DATA_FORMAT_D24_UNORM_S8_UINT:
        case RD::DATA_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            break;
    }
    return false;
}

bool OutputCompositor::_is_srgb_format(RD::DataFormat p_format) {
    switch (p_format) {
        case RD::DATA_FORMAT_R8_SRGB:
        case RD::DATA_FORMAT_R8G8_SRGB:
        case RD::DATA_FORMAT_R8G8B8_SRGB:
        case RD::DATA_FORMAT_B8G8R8_SRGB:
        case RD::DATA_FORMAT_R8G8B8A8_SRGB:
        case RD::DATA_FORMAT_B8G8R8A8_SRGB:
        case RD::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
            return true;
        default:
            break;
    }
    return false;
}

String OutputCompositor::_attachment_usage_label(bool p_is_depth) {
    return p_is_depth ? String("DEPTH_STENCIL_ATTACHMENT") : String("COLOR_ATTACHMENT");
}

bool OutputCompositor::_determine_viewport_blit_format(RD::DataFormat p_format, ViewportBlitFormat &r_format) {
    switch (p_format) {
        case RD::DATA_FORMAT_R8G8B8A8_UNORM:
        case RD::DATA_FORMAT_B8G8R8A8_UNORM:
        case RD::DATA_FORMAT_A8B8G8R8_UNORM_PACK32:
        case RD::DATA_FORMAT_R8G8B8A8_SRGB:
        case RD::DATA_FORMAT_B8G8R8A8_SRGB:
        case RD::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
            r_format = ViewportBlitFormat::RGBA8;
            return true;
        case RD::DATA_FORMAT_R16G16B16A16_SFLOAT:
            r_format = ViewportBlitFormat::RGBA16F;
            return true;
        case RD::DATA_FORMAT_R32G32B32A32_SFLOAT:
            r_format = ViewportBlitFormat::RGBA32F;
            return true;
        default:
            break;
    }
    return false;
}

const char *OutputCompositor::_viewport_blit_define(ViewportBlitFormat p_format) {
    switch (p_format) {
        case ViewportBlitFormat::RGBA8:
            return "#define VIEWPORT_BLIT_FORMAT 0";
        case ViewportBlitFormat::RGBA16F:
            return "#define VIEWPORT_BLIT_FORMAT 1";
        case ViewportBlitFormat::RGBA32F:
            return "#define VIEWPORT_BLIT_FORMAT 2";
        default:
            break;
    }
    return "#define VIEWPORT_BLIT_FORMAT 0";
}

uint64_t OutputCompositor::_viewport_blit_key(RenderingDevice *p_device, ViewportBlitFormat p_format) {
    uint64_t format_key = static_cast<uint64_t>(p_format) & 0xFFFFull;
    uint64_t device_key = reinterpret_cast<uint64_t>(p_device);
    return (device_key << 16) ^ format_key;
}

uint64_t OutputCompositor::_compute_framebuffer_validation_key(RenderingDevice *p_device, const Vector<RID> &p_attachments) const {
    uint64_t seed = hash64_murmur3_64(reinterpret_cast<uint64_t>(p_device), HASH_MURMUR3_SEED);
    seed = hash64_murmur3_64(static_cast<uint64_t>(p_attachments.size()), seed);
    for (const RID &attachment : p_attachments) {
        seed = hash64_murmur3_64(attachment.get_id(), seed);
    }
    return seed;
}

RID OutputCompositor::_ensure_viewport_blit_sampler(RenderingDevice *p_device) {
    if (!p_device) {
        return RID();
    }

    uint64_t device_key = reinterpret_cast<uint64_t>(p_device);
    if (ViewportBlitSampler *existing = viewport_blit_samplers.getptr(device_key)) {
        if (existing->sampler.is_valid() && existing->owner_device == p_device) {
            return existing->sampler;
        }

        RenderingDevice *owner_device = existing->owner_device ? existing->owner_device : p_device;
        if (existing->sampler.is_valid() && owner_device) {
            owner_device->free(existing->sampler);
        }
        _forget_resource(existing->sampler);
        viewport_blit_samplers.erase(device_key);
    }

    RD::SamplerState sampler_state;
    sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
    sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
    sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
    sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
    sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
    sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
    sampler_state.enable_compare = false;

    RID sampler = p_device->sampler_create(sampler_state);
    if (!sampler.is_valid()) {
        GS_LOG_WARN_DEFAULT("[OutputCompositor] Failed to create sampler for viewport compute blit");
        return RID();
    }
    p_device->set_resource_name(sampler, "GS_OutputCompositor_ViewportBlitSampler");

    _track_resource(sampler, p_device, true, "viewport_blit_sampler");
    ViewportBlitSampler sampler_entry;
    sampler_entry.sampler = sampler;
    sampler_entry.owner_device = p_device;
    viewport_blit_samplers.insert(device_key, sampler_entry);
    return sampler;
}

bool OutputCompositor::_ensure_viewport_blit_pipeline(RenderingDevice *p_device, RD::DataFormat p_format, RID &r_shader, RID &r_pipeline) {
    if (!p_device) {
        return false;
    }

    ViewportBlitFormat format;
    if (!_determine_viewport_blit_format(p_format, format)) {
        return false;
    }

    uint64_t key = _viewport_blit_key(p_device, format);
    if (ViewportBlitVariant *variant = viewport_blit_variants.getptr(key)) {
        if (variant->owner_device == p_device && variant->shader.is_valid() && variant->pipeline.is_valid()) {
            r_shader = variant->shader;
            r_pipeline = variant->pipeline;
            return true;
        }

        RenderingDevice *owner_device = variant->owner_device ? variant->owner_device : p_device;
        if (owner_device) {
            if (variant->pipeline.is_valid() && owner_device->compute_pipeline_is_valid(variant->pipeline)) {
                owner_device->free(variant->pipeline);
            }
            if (variant->shader.is_valid()) {
                owner_device->free(variant->shader);
            }
        }
        _forget_resource(variant->pipeline);
        _forget_resource(variant->shader);
        viewport_blit_variants.erase(key);
    }

    if (!viewport_blit_shader_source) {
        viewport_blit_shader_source = new ViewportBlitShaderRD();
    }

    if (!viewport_blit_shader_source_initialized) {
        Vector<String> versions;
        versions.push_back("");
        viewport_blit_shader_source->initialize(versions);
        viewport_blit_shader_source_initialized = true;
    }

    RID shader_version = viewport_blit_shader_source->version_create();
    if (!shader_version.is_valid()) {
        GS_LOG_WARN_DEFAULT("[OutputCompositor] Failed to create viewport blit shader version");
        return false;
    }

    Vector<String> stage_sources = viewport_blit_shader_source->version_build_variant_stage_sources(shader_version, 0);
    viewport_blit_shader_source->version_free(shader_version);
    if (stage_sources.size() <= RD::SHADER_STAGE_COMPUTE) {
        GS_LOG_WARN_DEFAULT("[OutputCompositor] Viewport blit shader missing compute stage");
        return false;
    }

    String compute_source = stage_sources[RD::SHADER_STAGE_COMPUTE];
    Vector<String> defines;
    defines.push_back(_viewport_blit_define(format));

    String compile_error;
    String processed_source;
    RID shader = ShaderCompilationHelper::compile_shader_on_device(p_device, compute_source, "viewport_blit", defines, &compile_error, &processed_source);
    if (!shader.is_valid()) {
        GS_LOG_WARN_DEFAULT(vformat("[OutputCompositor] Failed to compile viewport blit shader: %s",
                compile_error.is_empty() ? String("unknown error") : compile_error));
        return false;
    }

    RID pipeline = p_device->compute_pipeline_create(shader);
    if (!pipeline.is_valid()) {
        GS_LOG_WARN_DEFAULT("[OutputCompositor] Failed to create viewport blit compute pipeline");
        p_device->free(shader);
        return false;
    }

    _track_resource(shader, p_device, true, "viewport_blit_shader");
    _track_resource(pipeline, p_device, true, "viewport_blit_pipeline");

    ViewportBlitVariant new_variant;
    new_variant.shader = shader;
    new_variant.pipeline = pipeline;
    new_variant.owner_device = p_device;
    viewport_blit_variants.insert(key, new_variant);

    r_shader = shader;
    r_pipeline = pipeline;
    return true;
}

void OutputCompositor::clear_cached_framebuffers() {
    for (KeyValue<uint64_t, CachedFramebuffer> &E : output_cache.cached_framebuffers) {
        CachedFramebuffer &entry = E.value;
        RenderingDevice *fb_device = entry.device ? entry.device : rd;
        if (entry.framebuffer.is_valid() && fb_device && fb_device->framebuffer_is_valid(entry.framebuffer)) {
            fb_device->free(entry.framebuffer);
        }
        _forget_resource(entry.framebuffer);
    }

    output_cache.cached_framebuffers.clear();
    output_cache.framebuffer_validation_cache.clear();
    srgb_format_cache.clear();
}

void OutputCompositor::clear_viewport_blit_resources() {
    for (KeyValue<uint64_t, ViewportBlitVariant> &E : viewport_blit_variants) {
        ViewportBlitVariant &variant = E.value;
        RenderingDevice *owner_device = variant.owner_device ? variant.owner_device : rd;
        if (owner_device) {
            if (variant.pipeline.is_valid() && owner_device->compute_pipeline_is_valid(variant.pipeline)) {
                owner_device->free(variant.pipeline);
            }
            if (variant.shader.is_valid()) {
                owner_device->free(variant.shader);
            }
        }
        _forget_resource(variant.pipeline);
        _forget_resource(variant.shader);
    }

    for (KeyValue<uint64_t, ViewportBlitSampler> &E : viewport_blit_samplers) {
        ViewportBlitSampler &sampler_entry = E.value;
        RenderingDevice *owner_device = sampler_entry.owner_device ? sampler_entry.owner_device : rd;
        if (sampler_entry.sampler.is_valid() && owner_device) {
            owner_device->free(sampler_entry.sampler);
        }
        _forget_resource(sampler_entry.sampler);
    }

    viewport_blit_variants.clear();
    viewport_blit_samplers.clear();
}

RID OutputCompositor::get_cached_framebuffer(RenderingDevice *p_device, const RID &p_texture) {
    if (!p_device || !p_texture.is_valid()) {
        return RID();
    }

    uint64_t key = p_texture.get_id();
    if (CachedFramebuffer *entry = output_cache.cached_framebuffers.getptr(key)) {
        RenderingDevice *entry_device = entry->device ? entry->device : p_device;
        bool device_matches = entry_device == p_device;
        if (entry->framebuffer.is_valid() && device_matches &&
                entry_device && entry_device->framebuffer_is_valid(entry->framebuffer)) {
            return entry->framebuffer;
        }
        if (entry->framebuffer.is_valid() && entry_device && entry_device->framebuffer_is_valid(entry->framebuffer)) {
            entry_device->free(entry->framebuffer);
        }
        _forget_resource(entry->framebuffer);
        output_cache.cached_framebuffers.erase(key);
    }

    Vector<RID> attachments;
    attachments.push_back(p_texture);

    Vector<AttachmentValidationInfo> attachment_infos;
    Size2i attachment_extent;
    RD::TextureSamples attachment_samples = RD::TEXTURE_SAMPLES_1;
    String validation_error;
    if (!validate_framebuffer_attachments(p_device, attachments, attachment_infos, attachment_extent, attachment_samples,
                validation_error)) {
        GS_LOG_ERROR_DEFAULT(vformat("[OutputCompositor] Cached framebuffer validation failed for texture RID %s: %s",
                String::num_uint64(p_texture.get_id()), validation_error));
        return RID();
    }

    Vector<RID> framebuffer_attachments;
    framebuffer_attachments.push_back(p_texture);
    RID framebuffer = p_device->framebuffer_create(framebuffer_attachments);
    if (!framebuffer.is_valid()) {
        GS_LOG_ERROR_DEFAULT(vformat("[OutputCompositor] framebuffer_create failed for cached framebuffer (RID %s)",
                String::num_uint64(p_texture.get_id())));
        return RID();
    }
    p_device->set_resource_name(framebuffer, "GS_OutputCompositor_CachedFramebuffer");
    _track_resource(framebuffer, p_device, true, "cached_framebuffer");

    CachedFramebuffer new_entry;
    new_entry.framebuffer = framebuffer;
    new_entry.device = p_device;
    output_cache.cached_framebuffers.insert(key, new_entry);
    return framebuffer;
}

bool OutputCompositor::validate_framebuffer_attachments(RenderingDevice *p_device, const Vector<RID> &p_attachments,
        Vector<AttachmentValidationInfo> &r_infos, Size2i &r_extent, RD::TextureSamples &r_samples, String &r_error) {
    r_infos.clear();
    r_extent = Size2i();
    r_samples = RD::TEXTURE_SAMPLES_1;

    if (!p_device) {
        r_error = "Target RenderingDevice is null";
        return false;
    }

    if (p_attachments.is_empty()) {
        r_error = "No framebuffer attachments were provided";
        return false;
    }

    uint64_t cache_key = _compute_framebuffer_validation_key(p_device, p_attachments);
    if (FramebufferValidationCacheEntry *cache_entry = output_cache.framebuffer_validation_cache.getptr(cache_key)) {
        if (cache_entry->valid) {
            bool cache_still_valid = true;
            for (const AttachmentValidationInfo &cached_info : cache_entry->infos) {
                const RID &attachment = cached_info.original_attachment;
                if (!attachment.is_valid() || !p_device->texture_is_valid(attachment)) {
                    cache_still_valid = false;
                    break;
                }
            }
            if (cache_still_valid) {
                r_infos = cache_entry->infos;
                r_extent = cache_entry->extent;
                r_samples = cache_entry->samples;
                return true;
            }
        }
    }
    Vector<String> errors;

    bool extent_initialized = false;
    RD::TextureSamples expected_samples = RD::TEXTURE_SAMPLES_1;

    for (int i = 0; i < p_attachments.size(); i++) {
        const RID &attachment = p_attachments[i];
        AttachmentValidationInfo info;
        info.original_attachment = attachment;

        if (!attachment.is_valid()) {
            errors.push_back(vformat("Attachment %d is invalid (RID %s)", i, String::num_uint64(attachment.get_id())));
            continue;
        }

        if (!p_device->texture_is_valid(attachment)) {
            errors.push_back(vformat("Attachment %d (RID %s) is not valid on the target RenderingDevice", i,
                    String::num_uint64(attachment.get_id())));
            continue;
        }

        RD::TextureFormat texture_format = p_device->texture_get_format(attachment);
        if (texture_format.format == RD::DATA_FORMAT_MAX) {
            errors.push_back(vformat("Attachment %d (RID %s) reports an unknown texture format", i,
                    String::num_uint64(attachment.get_id())));
            continue;
        }

        info.original_format = texture_format;
        info.is_depth = _is_depth_attachment_format(texture_format.format);

        if (texture_format.width == 0 || texture_format.height == 0) {
            errors.push_back(vformat("Attachment %d (RID %s) has zero extent", i, String::num_uint64(attachment.get_id())));
        }
        if (texture_format.array_layers == 0) {
            errors.push_back(vformat("Attachment %d (RID %s) has no array layers", i, String::num_uint64(attachment.get_id())));
        }
        if (texture_format.mipmaps == 0) {
            errors.push_back(vformat("Attachment %d (RID %s) has no mipmap levels", i,
                    String::num_uint64(attachment.get_id())));
        }
        if (texture_format.texture_type != RD::TEXTURE_TYPE_2D && texture_format.texture_type != RD::TEXTURE_TYPE_2D_ARRAY) {
            errors.push_back(vformat("Attachment %d (RID %s) uses unsupported texture type %d", i,
                    String::num_uint64(attachment.get_id()), (int)texture_format.texture_type));
        }
        if (texture_format.texture_type == RD::TEXTURE_TYPE_2D && texture_format.depth > 1) {
            errors.push_back(vformat("Attachment %d (RID %s) declares depth %d for 2D texture", i,
                    String::num_uint64(attachment.get_id()), texture_format.depth));
        }

        Size2i attachment_extent(texture_format.width, texture_format.height);
        if (!extent_initialized) {
            r_extent = attachment_extent;
            extent_initialized = true;
        } else if (r_extent != attachment_extent) {
            errors.push_back(vformat("Attachment %d (RID %s) size %dx%d does not match expected %dx%d", i,
                    String::num_uint64(attachment.get_id()), attachment_extent.x, attachment_extent.y, r_extent.x, r_extent.y));
        }

        if (i == 0) {
            expected_samples = texture_format.samples;
            r_samples = texture_format.samples;
        } else if (texture_format.samples != expected_samples) {
            errors.push_back(vformat("Attachment %d (RID %s) sample count %d mismatches expected %d", i,
                    String::num_uint64(attachment.get_id()), (int)texture_format.samples, (int)expected_samples));
        }

        BitField<RD::TextureUsageBits> required_usage = info.is_depth ? BitField<RD::TextureUsageBits>(RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                                                                      : BitField<RD::TextureUsageBits>(RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT);
        uint32_t required_usage_bits = info.is_depth ? (uint32_t)RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                                     : (uint32_t)RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

        bool has_required_usage = (texture_format.usage_bits & required_usage_bits) != 0;
        bool format_supported = p_device->texture_is_format_supported_for_usage(texture_format.format, required_usage);

        if (!has_required_usage || !format_supported) {
            errors.push_back(vformat("Attachment %d (RID %s) lacks required usage %s or is unsupported by the device",
                    i, String::num_uint64(attachment.get_id()), _attachment_usage_label(info.is_depth)));
        }

        r_infos.push_back(info);
    }

    if (!errors.is_empty()) {
        String error_text;
        for (int i = 0; i < errors.size(); i++) {
            if (i > 0) {
                error_text += "\n";
            }
            error_text += errors[i];
        }
        r_error = error_text;
        return false;
    }

    FramebufferValidationCacheEntry cache_entry;
    cache_entry.valid = true;
    cache_entry.extent = r_extent;
    cache_entry.samples = r_samples;
    cache_entry.infos = r_infos;
    output_cache.framebuffer_validation_cache.insert(cache_key, cache_entry);

    return true;
}

bool OutputCompositor::_copy_final_output_compute(RenderingDevice *p_device, RID p_source, RID p_destination,
        const Size2i &p_source_extent, const Size2i &p_copy_extent, const Vector3i &p_destination_offset,
        bool p_composite_with_destination, bool p_source_is_premultiplied, bool p_destination_is_srgb,
        const RD::TextureFormat &p_destination_format, RID p_source_depth, RID p_destination_depth,
        bool p_depth_test_enabled, bool p_depth_is_orthogonal, float p_z_near, float p_z_far,
        float p_depth_linearize_mul, float p_depth_linearize_add, float p_depth_epsilon) {
    if (!p_device || !p_source.is_valid() || !p_destination.is_valid()) {
        return false;
    }

    if (p_copy_extent.x <= 0 || p_copy_extent.y <= 0) {
        return false;
    }

    RID shader;
    RID pipeline;
    if (!_ensure_viewport_blit_pipeline(p_device, p_destination_format.format, shader, pipeline)) {
        return false;
    }

    RID sampler = _ensure_viewport_blit_sampler(p_device);
    if (!sampler.is_valid()) {
        return false;
    }

    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    RID fallback_depth = texture_storage ? texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_DEPTH) : RID();

    RID source_depth = p_source_depth.is_valid() ? p_source_depth : fallback_depth;
    RID destination_depth = p_destination_depth.is_valid() ? p_destination_depth : fallback_depth;

    bool depth_test_enabled = p_depth_test_enabled && p_source_depth.is_valid() && p_destination_depth.is_valid() &&
            source_depth.is_valid() && destination_depth.is_valid();
    if (!depth_test_enabled) {
        source_depth = fallback_depth;
        destination_depth = fallback_depth;
    }

    RD::Uniform sampler_uniform;
    sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    sampler_uniform.binding = 0;
    sampler_uniform.append_id(sampler);
    sampler_uniform.append_id(p_source);

    RD::Uniform image_uniform;
    image_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
    image_uniform.binding = 1;
    image_uniform.append_id(p_destination);

    RD::Uniform source_depth_uniform;
    source_depth_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    source_depth_uniform.binding = 2;
    source_depth_uniform.append_id(sampler);
    source_depth_uniform.append_id(source_depth);

    RD::Uniform destination_depth_uniform;
    destination_depth_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    destination_depth_uniform.binding = 3;
    destination_depth_uniform.append_id(sampler);
    destination_depth_uniform.append_id(destination_depth);

    UniformSetCacheRD *uniform_cache = UniformSetCacheRD::get_singleton();
    RID uniform_set;
    bool manual_uniform_set = false;
    if (uniform_cache) {
        uniform_set = uniform_cache->get_cache(shader, 0, sampler_uniform, image_uniform, source_depth_uniform, destination_depth_uniform);
    } else {
        Vector<RD::Uniform> uniforms;
        uniforms.push_back(sampler_uniform);
        uniforms.push_back(image_uniform);
        uniforms.push_back(source_depth_uniform);
        uniforms.push_back(destination_depth_uniform);
        uniform_set = p_device->uniform_set_create(uniforms, shader, 0);
        manual_uniform_set = true;
        if (!uniform_set.is_valid()) {
            return false;
        }
        p_device->set_resource_name(uniform_set, "GS_OutputCompositor_CopyFinalOutputUniformSet");
    }

    // GPU Debug: Output composition pass (Purple tones for compositing)
    ScopedGpuMarker composite_marker(p_device, "GS_Composite", Color(0.7f, 0.3f, 0.9f, 1.0f));

    RD::ComputeListID compute_list = p_device->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        if (manual_uniform_set && uniform_set.is_valid() && p_device->uniform_set_is_valid(uniform_set)) {
            p_device->free(uniform_set);
        }
        return false;
    }

    p_device->compute_list_bind_compute_pipeline(compute_list, pipeline);
    p_device->compute_list_bind_uniform_set(compute_list, uniform_set, 0);

    struct ViewportBlitPushConstant {
        int32_t copy_size[2];
        int32_t source_size[2];
        int32_t destination_size[2];
        int32_t destination_offset[2];
        int32_t composite_with_destination;
        int32_t destination_is_srgb;
        int32_t source_is_premultiplied;
        int32_t depth_test_enabled;
        int32_t depth_is_orthogonal;
        float z_near;
        float z_far;
        float depth_epsilon;
        float depth_linearize_mul;
        float depth_linearize_add;
        float pad0;
        float pad1;
    } params = {};

    params.copy_size[0] = p_copy_extent.x;
    params.copy_size[1] = p_copy_extent.y;
    params.source_size[0] = MAX(p_source_extent.x, 1);
    params.source_size[1] = MAX(p_source_extent.y, 1);

    int32_t dest_width = p_destination_format.width > 0 ? (int32_t)p_destination_format.width
                                                        : MAX(p_destination_offset.x + p_copy_extent.x, 1);
    int32_t dest_height = p_destination_format.height > 0 ? (int32_t)p_destination_format.height
                                                          : MAX(p_destination_offset.y + p_copy_extent.y, 1);
    params.destination_size[0] = dest_width;
    params.destination_size[1] = dest_height;
    params.destination_offset[0] = p_destination_offset.x;
    params.destination_offset[1] = p_destination_offset.y;
    params.composite_with_destination = p_composite_with_destination ? 1 : 0;
    params.destination_is_srgb = p_destination_is_srgb ? 1 : 0;
    params.source_is_premultiplied = p_source_is_premultiplied ? 1 : 0;
    params.depth_test_enabled = depth_test_enabled ? 1 : 0;
    params.depth_is_orthogonal = p_depth_is_orthogonal ? 1 : 0;
    params.z_near = p_z_near;
    params.z_far = p_z_far;
    params.depth_epsilon = p_depth_epsilon;
    params.depth_linearize_mul = p_depth_linearize_mul;
    params.depth_linearize_add = p_depth_linearize_add;

    p_device->compute_list_set_push_constant(compute_list, &params, sizeof(ViewportBlitPushConstant));

    uint32_t groups_x = (uint32_t)((p_copy_extent.x + 7) / 8);
    uint32_t groups_y = (uint32_t)((p_copy_extent.y + 7) / 8);
    groups_x = MAX(groups_x, 1u);
    groups_y = MAX(groups_y, 1u);

    p_device->compute_list_dispatch(compute_list, groups_x, groups_y, 1);
    p_device->compute_list_end();

    if (manual_uniform_set && uniform_set.is_valid() && p_device->uniform_set_is_valid(uniform_set)) {
        p_device->free(uniform_set);
    }

    return true;
}

bool OutputCompositor::copy_to_framebuffer(const FramebufferCopyParams &p_params) {
    output_cache.last_viewport_copy_success = false;
    output_cache.last_viewport_copy_source_size = Size2i();
    output_cache.last_viewport_copy_dest_size = Size2i();

    if (!p_params.source_texture.is_valid() || !p_params.framebuffer.is_valid()) {
        return false;
    }

    RenderingDevice *main_rd = RenderingDevice::get_singleton();
    if (device_manager.is_valid()) {
        if (RenderingDevice *main_device = device_manager->get_main_device()) {
            main_rd = main_device;
        }
    }
    if (!main_rd) {
        return false;
    }

    if (!main_rd->texture_is_valid(p_params.source_texture)) {
        GS_LOG_ERROR_DEFAULT("[OutputCompositor] Source texture is not valid on main RenderingDevice; framebuffer copy unsupported");
        return false;
    }

    RendererRD::CopyEffects *copy_effects = RendererRD::CopyEffects::get_singleton();
    if (!copy_effects) {
        GS_LOG_WARN_DEFAULT("[OutputCompositor] CopyEffects unavailable for framebuffer copy");
        return false;
    }

    Size2i viewport_size = p_params.viewport_size;
    if (viewport_size.x <= 0 || viewport_size.y <= 0) {
        viewport_size = Size2i(1920, 1080); // Fallback
    }

    // Get source texture format to determine sRGB handling
    RenderingDevice *check_rd = main_rd;
    bool srgb_destination = false;
    if (check_rd) {
        srgb_destination = _is_texture_srgb(check_rd, p_params.source_texture);
    }

    Rect2i dest_rect(Vector2i(0, 0), Size2i(viewport_size.x, viewport_size.y));
    Rect2 src_rect(Vector2(0.0f, 0.0f), Vector2(1.0f, 1.0f));

    bool enable_blend = p_params.composite_with_destination;
    bool use_premultiplied_alpha = enable_blend && p_params.source_is_premultiplied;

    copy_effects->copy_to_fb_rect(p_params.source_texture, p_params.framebuffer, dest_rect, false, false, false, srgb_destination, RID(), false, false, false, false, src_rect, enable_blend, use_premultiplied_alpha);

    output_cache.last_viewport_copy_success = true;
    output_cache.last_viewport_copy_source_size = viewport_size;
    output_cache.last_viewport_copy_dest_size = viewport_size;
    return true;
}

OutputCopyResult OutputCompositor::copy_to_render_target(const OutputCopyParams &p_params) {
    OutputCopyResult result;
    result.success = false;
    output_cache.last_viewport_copy_success = false;
    output_cache.last_viewport_copy_source_size = Size2i();
    output_cache.last_viewport_copy_dest_size = Size2i();

    if (!p_params.source_texture.is_valid() || !p_params.destination_texture.is_valid()) {
        result.error = "Invalid source or destination texture";
        return result;
    }

    if (!rd) {
        result.error = "RenderingDevice not initialized";
        return result;
    }

    RenderingDevice *main_rd = RenderingDevice::get_singleton();
    if (device_manager.is_valid()) {
        if (RenderingDevice *main_device = device_manager->get_main_device()) {
            main_rd = main_device;
        }
    }
    if (!main_rd) {
        result.error = "Main RenderingDevice unavailable";
        return result;
    }

    RenderingDevice *copy_device = main_rd; // Copy executed on main RenderingDevice
    if (!copy_device->texture_is_valid(p_params.source_texture) ||
            !copy_device->texture_is_valid(p_params.destination_texture)) {
        result.error = "Output textures are not valid on main RenderingDevice";
        return result;
    }

    Size2i destination_extent = p_params.viewport_size;
    if (destination_extent.x <= 0 || destination_extent.y <= 0) {
        destination_extent = Size2i(1, 1);
    }

    RD::TextureFormat destination_format = _get_texture_format(copy_device, p_params.destination_texture);
    if (destination_format.width > 0 && destination_format.height > 0) {
        destination_extent.x = destination_format.width;
        destination_extent.y = destination_format.height;
    }

    RD::TextureFormat source_format = _get_texture_format(copy_device, p_params.source_texture);
    Size2i source_extent(source_format.width, source_format.height);
    if (source_extent.x <= 0 || source_extent.y <= 0) {
        if (internal_render_size.x > 0 && internal_render_size.y > 0) {
            source_extent = internal_render_size;
        }
    }
    if (source_extent.x <= 0 || source_extent.y <= 0) {
        source_extent = destination_extent;
    }

    output_cache.last_viewport_copy_source_size = source_extent;
    result.source_size = source_extent;

    Size2i copy_extent = source_extent;
    if (destination_extent.x > 0 && destination_extent.y > 0) {
        copy_extent.x = MIN(copy_extent.x, destination_extent.x);
        copy_extent.y = MIN(copy_extent.y, destination_extent.y);
    }

    output_cache.last_viewport_copy_dest_size = copy_extent;
    result.dest_size = copy_extent;

    if (copy_extent.x <= 0 || copy_extent.y <= 0) {
        result.error = "Copy extent is zero";
        return result;
    }

    // Check format compatibility
    bool format_mismatch = false;
    if (destination_format.format != RD::DATA_FORMAT_MAX && source_format.format != RD::DATA_FORMAT_MAX) {
        if (destination_format.format != source_format.format) {
            format_mismatch = true;
        }
    }

    bool sample_mismatch = destination_format.samples != source_format.samples;

    // Check texture usage flags
    uint64_t source_usage = (uint64_t)source_format.usage_bits;
    uint64_t destination_usage = (uint64_t)destination_format.usage_bits;
    bool source_can_copy = (source_usage & RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT) != 0;
    bool destination_can_copy = (destination_usage & RD::TEXTURE_USAGE_CAN_COPY_TO_BIT) != 0;
    bool source_can_sample = (source_usage & RD::TEXTURE_USAGE_SAMPLING_BIT) != 0;

    bool can_direct_copy = !p_params.composite_with_destination && !format_mismatch && !sample_mismatch && source_can_copy && destination_can_copy;

    // Try direct copy first
    if (can_direct_copy) {
        Vector3i src_offset(0, 0, 0);
        Vector3i dst_offset(0, 0, 0);
        if (destination_extent.x > copy_extent.x) {
            dst_offset.x = (destination_extent.x - copy_extent.x) / 2;
        }
        if (destination_extent.y > copy_extent.y) {
            dst_offset.y = (destination_extent.y - copy_extent.y) / 2;
        }

        Vector3i region(copy_extent.x, copy_extent.y, 1);
        Error copy_err = copy_device->texture_copy(p_params.source_texture, p_params.destination_texture, src_offset, dst_offset, region, 0, 0, 0, 0);
        if (copy_err == OK) {
            result.success = true;
            output_cache.last_viewport_copy_success = (copy_extent == source_extent);
            return result;
        }
    }

    // Depth-aware composite (compute) if requested.
    if (p_params.depth_test_enabled) {
        Vector3i dst_offset(0, 0, 0);
        if (destination_extent.x > copy_extent.x) {
            dst_offset.x = (destination_extent.x - copy_extent.x) / 2;
        }
        if (destination_extent.y > copy_extent.y) {
            dst_offset.y = (destination_extent.y - copy_extent.y) / 2;
        }

        bool ok = _copy_final_output_compute(copy_device, p_params.source_texture, p_params.destination_texture,
                source_extent, copy_extent, dst_offset, p_params.composite_with_destination,
                p_params.source_is_premultiplied, _is_srgb_format(destination_format.format),
                destination_format, p_params.source_depth, p_params.destination_depth, true,
                p_params.depth_is_orthogonal, p_params.z_near, p_params.z_far,
                p_params.depth_linearize_mul, p_params.depth_linearize_add, p_params.depth_epsilon);
        if (ok) {
            result.success = true;
            output_cache.last_viewport_copy_success = (copy_extent == source_extent);
            return result;
        }
    }

    // Use CopyEffects blit fallback
    RendererRD::CopyEffects *copy_effects = RendererRD::CopyEffects::get_singleton();
    if (!copy_effects) {
        result.error = "CopyEffects subsystem unavailable";
        return result;
    }

    // Create framebuffer from destination texture
    RID framebuffer = get_cached_framebuffer(main_rd, p_params.destination_texture);
    if (!framebuffer.is_valid()) {
        result.error = "Failed to create framebuffer for destination";
        return result;
    }

    // Determine sRGB handling
    bool srgb_destination = _is_srgb_format(destination_format.format);
    if (destination_format.format != RD::DATA_FORMAT_MAX &&
            p_params.destination_texture.is_valid() &&
            copy_device->texture_is_valid(p_params.destination_texture)) {
        srgb_format_cache.insert(p_params.destination_texture.get_id(), srgb_destination);
    }

    Vector3i dst_offset(0, 0, 0);
    if (destination_extent.x > copy_extent.x) {
        dst_offset.x = (destination_extent.x - copy_extent.x) / 2;
    }
    if (destination_extent.y > copy_extent.y) {
        dst_offset.y = (destination_extent.y - copy_extent.y) / 2;
    }

    Rect2i dest_rect(Vector2i(dst_offset.x, dst_offset.y), Size2i(copy_extent.x, copy_extent.y));
    Vector2 src_scale(1.0f, 1.0f);
    if (source_extent.x > 0) {
        src_scale.x = float(copy_extent.x) / float(source_extent.x);
    }
    if (source_extent.y > 0) {
        src_scale.y = float(copy_extent.y) / float(source_extent.y);
    }
    Rect2 src_rect(Vector2(0.0f, 0.0f), src_scale);

    bool enable_blend = p_params.composite_with_destination;
    bool use_premultiplied_alpha = enable_blend && p_params.source_is_premultiplied;

    copy_effects->copy_to_fb_rect(p_params.source_texture, framebuffer, dest_rect, false, false, false, srgb_destination, RID(), false, false, false, false, src_rect, enable_blend, use_premultiplied_alpha);

    result.success = true;
    output_cache.last_viewport_copy_success = (copy_extent == source_extent);
    return result;
}

void OutputCompositor::integrate_final_output(GaussianSplatRenderer *p_renderer, RenderDataRD *p_render_data, RenderSceneBuffersRD *render_buffers_rd,
        const RID &p_final_output, RID &r_render_target, const Size2i &p_viewport_size, bool p_defer_commit,
        bool p_painterly_active, const RID &p_cached_depth) {
    if (!p_renderer) {
        return;
    }
    (void)p_defer_commit;

    final_render_texture = p_final_output;
    output_cache.has_valid_render = p_final_output.is_valid();

    if (render_buffers_rd && p_final_output.is_valid()) {
        Size2i viewport_size = p_viewport_size;
        if (viewport_size.x <= 0 || viewport_size.y <= 0) {
            viewport_size = render_buffers_rd->get_internal_size();
        }
        if (viewport_size.x <= 0 || viewport_size.y <= 0) {
            viewport_size = Size2i(1, 1);
        }

        RID render_target = r_render_target;
        RID render_target_framebuffer;

        // CRITICAL FIX: ALWAYS try to get Godot's actual render target framebuffer.
        // Even if we have the texture, we need the framebuffer to draw into the viewport.
        RID godot_render_target = render_buffers_rd->get_render_target();
        if (godot_render_target.is_valid()) {
            RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
            if (texture_storage) {
                render_target_framebuffer = texture_storage->render_target_get_rd_framebuffer(godot_render_target);
                if (!render_target.is_valid()) {
                    render_target = texture_storage->render_target_get_rd_texture(godot_render_target);
                }
            }
        }

        output_cache.last_render_target = render_target;

        bool composited = false;
        auto &subsystem_state = p_renderer->get_subsystem_state();
        if (p_painterly_active && subsystem_state.painterly_renderer.is_valid()) {
            RID depth_for_composite = p_renderer->get_painterly_depth_texture();
            if (!depth_for_composite.is_valid()) {
                depth_for_composite = p_cached_depth;
            }
            if (depth_for_composite.is_valid()) {
                composited = subsystem_state.painterly_renderer->composite_painterly_output(
                        p_renderer, p_render_data, p_final_output, depth_for_composite, viewport_size);
                if (composited) {
                    output_cache.last_viewport_copy_success = true;
                }
            }
        }

        const bool scene_depth_test_requested = gs_get_composite_depth_test_enabled();
        const GSSceneCompositeDepthPolicy scene_depth_policy = gs_get_scene_composite_depth_policy();

        RID scene_depth = render_buffers_rd ? render_buffers_rd->get_depth_texture() : RID();
        const bool has_source_depth = p_cached_depth.is_valid();
        const bool has_scene_depth = scene_depth.is_valid();
        const bool depth_test_enabled = scene_depth_test_requested && has_source_depth && has_scene_depth;
        const bool missing_required_scene_depth = scene_depth_test_requested && (!has_source_depth || !has_scene_depth);
        const bool allow_scene_blend_fallback = scene_depth_policy == GS_SCENE_COMPOSITE_DEPTH_POLICY_RELAXED;

        if (!composited && render_target.is_valid()) {
            if (missing_required_scene_depth && !allow_scene_blend_fallback) {
                gs_log_scene_depth_contract_skip_once(!has_source_depth, !has_scene_depth);
                output_cache.last_viewport_copy_success = false;
            } else if (render_target_framebuffer.is_valid() && !depth_test_enabled) {
                FramebufferCopyParams params;
                params.source_texture = p_final_output;
                params.framebuffer = render_target_framebuffer;
                params.viewport_size = viewport_size;
                params.composite_with_destination = true;
                params.source_is_premultiplied = true;
                copy_to_framebuffer(params);
            } else {
                OutputCopyParams params;
                params.source_texture = p_final_output;
                params.source_depth = p_cached_depth;
                params.destination_texture = render_target;
                params.destination_depth = scene_depth;
                params.viewport_size = viewport_size;
                params.composite_with_destination = true;
                params.source_is_premultiplied = true;
                params.depth_test_enabled = depth_test_enabled;
                params.depth_linearize_mul = params.z_near;
                params.depth_linearize_add = params.z_far;
                if (p_render_data && p_render_data->scene_data) {
                    params.z_near = p_render_data->scene_data->z_near;
                    params.z_far = p_render_data->scene_data->z_far;
                    params.depth_is_orthogonal = p_render_data->scene_data->cam_orthogonal;
                    params.depth_linearize_mul = params.z_near;
                    params.depth_linearize_add = params.z_far;
                    if (!params.depth_is_orthogonal) {
                        Projection correction;
                        correction.set_depth_correction(false);
                        Projection temp = correction * p_render_data->scene_data->cam_projection;
                        params.depth_linearize_mul = -temp.columns[3][2];
                        params.depth_linearize_add = temp.columns[2][2];
                        if (params.depth_linearize_mul * params.depth_linearize_add < 0.0f) {
                            params.depth_linearize_add = -params.depth_linearize_add;
                        }
                    }
                }
                copy_to_render_target(params);
            }
        }
        r_render_target = render_target;
    }

    output_cache.render_buffers_commit_pending = false;
    output_cache.pending_painterly_commit = false;
}

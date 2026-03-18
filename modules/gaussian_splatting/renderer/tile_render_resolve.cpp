/**
 * tile_render_resolve.cpp — TileRenderer::TileResolveStage method implementations.
 *
 * Companion .cpp for tile_renderer.h / tile_render_stages.h.
 * Contains the tile resolve pass: texture management, pipeline setup,
 * sampler creation, fallback lighting buffers, uniform set creation,
 * push constants, and the resolve dispatch.
 *
 * Pattern 11 (RAII for RIDs): resolve textures, samplers, and fallback
 * lighting buffers are owned RIDs created/freed by this stage.
 * Pattern 12 (Ownership graph): resolved_texture_external may be shared
 * with the main RenderingDevice; ownership tracked via BufferOwnership.
 */

#include "tile_renderer.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_method_pointer.h"
#include "servers/rendering/rendering_device.h"
#include "core/templates/hash_map.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"

#include "gpu_debug_utils.h"
#include "gpu_performance_monitor.h"
#include "gpu_sorter.h"
#include "gpu_sorting_config.h"
#include "quantization_config.h"
#include "resource_owner_mismatch_contract.h"
#include "../logger/gs_logger.h"
#include "gaussian_gpu_layout.h"
#include "pipeline_io_contracts.h"
#include "shader_compilation_helper.h"
#include "sh_config.h"
#include "tile_prefix_scan_utils.h"
#include "../interfaces/sync_policy.h"
#include "../shaders/tile_resolve.glsl.gen.h"

using GaussianSplatting::PassColors;
using GaussianSplatting::ScopedGpuMarker;
using GaussianSplatting::ScopedGpuMarkerEx;

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

static constexpr uint32_t kFallbackSceneUniformBytes = 8192; // SceneDataBlock (SceneData + prev_data) ~6KB; pad for safety.
static constexpr uint32_t kFallbackDirectionalUniformBytes = 2048; // One DirectionalLightData entry (std140), padded.
static constexpr uint32_t kFallbackLightStorageBytes = 1024; // Minimal SSBO for omni/spot/reflection arrays.
static constexpr uint32_t kFallbackClusterStorageBytes = 1024; // Minimal SSBO for cluster data.
static constexpr int32_t kFallbackDecalTextureSize = 4;
static constexpr int32_t kFallbackReflectionTextureSize = 4;
static constexpr int32_t kFallbackShadowTextureSize = 1;

static RD::DataFormat _resolve_storage_format(RD::DataFormat p_format, String *r_reason = nullptr) {
    if (r_reason) {
        *r_reason = String();
    }
    switch (p_format) {
        case RD::DATA_FORMAT_R8G8B8A8_UNORM:
        case RD::DATA_FORMAT_R16G16B16A16_SFLOAT:
        case RD::DATA_FORMAT_R32G32B32A32_SFLOAT:
            return p_format;
        case RD::DATA_FORMAT_R8G8B8A8_SRGB:
            if (r_reason) {
                *r_reason = "sRGB resolve storage is unsupported";
            }
            return RD::DATA_FORMAT_R8G8B8A8_UNORM;
        case RD::DATA_FORMAT_B8G8R8A8_UNORM:
        case RD::DATA_FORMAT_B8G8R8A8_SRGB:
        case RD::DATA_FORMAT_A8B8G8R8_UNORM_PACK32:
        case RD::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
            if (r_reason) {
                *r_reason = "BGRA resolve storage is unsupported";
            }
            return RD::DATA_FORMAT_R8G8B8A8_UNORM;
        default:
            if (r_reason) {
                *r_reason = "unsupported resolve storage format";
            }
            return RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }
}

static void _log_resolve_format_fallback_once(const char *p_context, RD::DataFormat p_requested, RD::DataFormat p_resolved, const String &p_reason) {
    WARN_PRINT_ONCE(vformat("[TileRenderer] %s format fallback: requested=%d resolved=%d reason=%s",
            String(p_context ? p_context : "resolve"), int(p_requested), int(p_resolved), p_reason));
}

static bool _is_resolve_contract_fallback_accepted(RD::DataFormat p_requested, RD::DataFormat p_resolved) {
	switch (p_requested) {
		case RD::DATA_FORMAT_R8G8B8A8_SRGB:
		case RD::DATA_FORMAT_B8G8R8A8_UNORM:
		case RD::DATA_FORMAT_B8G8R8A8_SRGB:
		case RD::DATA_FORMAT_A8B8G8R8_UNORM_PACK32:
		case RD::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
			return p_resolved == RD::DATA_FORMAT_R8G8B8A8_UNORM;
		default:
			break;
	}
	return false;
}

} // namespace

void TileRenderer::TileResolveStage::destroy_resolve_textures() {
    RenderingDevice *resource_device = owner._get_resource_device();

    auto safe_texture_free = [](RenderingDevice *p_device, RID &p_rid) {
        if (!p_device || !p_rid.is_valid()) {
            p_rid = RID();
            return;
        }
        if (p_device->texture_is_valid(p_rid)) {
            p_device->free(p_rid);
        }
        p_rid = RID();
    };

    if (resource_device) {
        safe_texture_free(resource_device, owner.render_targets.resolved_texture);
        safe_texture_free(resource_device, owner.render_targets.resolved_depth_texture);
    }
    owner.render_targets.resolved_texture = RID();
    owner.render_targets.resolved_depth_texture = RID();

    // Only free external resolved textures if we own them (not shared from compositor/viewport)
    // The main device textures are managed by the viewport/compositor lifecycle
    if (owner.render_targets.resolved_texture_external.is_valid() &&
            owner.render_targets.resolved_texture_owner.matches(resource_device)) {
        safe_texture_free(resource_device, owner.render_targets.resolved_texture_external);
    } else {
        owner.render_targets.resolved_texture_external = RID();
    }
    if (owner.render_targets.resolved_depth_texture_external.is_valid() &&
            owner.render_targets.resolved_depth_texture_owner.matches(resource_device)) {
        safe_texture_free(resource_device, owner.render_targets.resolved_depth_texture_external);
    } else {
        owner.render_targets.resolved_depth_texture_external = RID();
    }
    owner.render_targets.resolved_texture_owner.clear();
    owner.render_targets.resolved_texture_local_owner = nullptr;
    owner.render_targets.resolved_depth_texture_owner.clear();
    owner.render_targets.resolved_depth_texture_local_owner = nullptr;
    owner.shader_resources.resolve_pipeline_initialized = false;
}

void TileRenderer::TileResolveStage::ensure_resolve_resources(const Vector2i &p_size, RD::DataFormat p_format) {
    RenderingDevice *device = owner._get_resource_device();
    if (!device || p_size.x <= 0 || p_size.y <= 0) {
        return;
    }

    RD::DataFormat requested_format = p_format;
    if (requested_format == RD::DATA_FORMAT_MAX) {
        requested_format = owner.config_state.desired_output_format != RD::DATA_FORMAT_MAX
                ? owner.config_state.desired_output_format
                : RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }
    String resolve_fallback_reason;
    RD::DataFormat resolved_storage_format = _resolve_storage_format(requested_format, &resolve_fallback_reason);
    if (resolved_storage_format != requested_format) {
        _log_resolve_format_fallback_once("resolve_resources", requested_format, resolved_storage_format, resolve_fallback_reason);
        const bool accepted_fallback = _is_resolve_contract_fallback_accepted(requested_format, resolved_storage_format);
        if (!accepted_fallback) {
            ERR_PRINT_ONCE(vformat("[TileRenderer] resolve_resources contract violation: requested_format=%d resolved_format=%d reason=%s. Resolve stage disabled.",
                    int(requested_format), int(resolved_storage_format), resolve_fallback_reason));
            owner.config_state.resolve_target_format = RD::DATA_FORMAT_MAX;
            owner.shader_resources.resolve_pipeline_initialized = false;
            owner.render_targets.depth_texture_copy_compatible = false;
            destroy_resolve_textures();
            return;
        }
    }
    owner.config_state.resolve_target_format = resolved_storage_format;

    RenderingDevice *main_device = RenderingDevice::get_singleton();

    auto texture_matches = [&](RID p_texture, RD::DataFormat p_expected_format) -> bool {
        if (!p_texture.is_valid()) {
            return false;
        }
        RD::TextureFormat fmt = device->texture_get_format(p_texture);
        return fmt.width == p_size.x && fmt.height == p_size.y && fmt.format == p_expected_format;
    };

    auto update_depth_copy_contract = [&]() {
        owner.render_targets.depth_texture_copy_compatible = false;
        RID resolved_depth_external = owner.render_targets.resolved_depth_texture_external.is_valid()
                ? owner.render_targets.resolved_depth_texture_external
                : owner.render_targets.resolved_depth_texture;
        if (!resolved_depth_external.is_valid()) {
            return;
        }

        RenderingDevice *resolved_depth_owner = owner.render_targets.resolved_depth_texture_owner.device
                ? owner.render_targets.resolved_depth_texture_owner.device
                : device;
        if (!main_device) {
            owner.render_targets.depth_texture_copy_compatible = true;
            return;
        }

        const bool resolved_depth_on_main = (resolved_depth_owner == main_device) ||
                main_device->texture_is_valid(resolved_depth_external);
        owner.render_targets.depth_texture_copy_compatible = resolved_depth_on_main;
    };

    bool color_valid = texture_matches(owner.render_targets.resolved_texture, resolved_storage_format);
    bool depth_valid = texture_matches(owner.render_targets.resolved_depth_texture, RD::DATA_FORMAT_R32_SFLOAT);

    if (color_valid && depth_valid) {
        update_depth_copy_contract();
        ensure_resolve_pipeline(device, owner.config_state.resolve_target_format);
        return;
    }

    destroy_resolve_textures();

    bool can_share_with_main = main_device != nullptr && main_device != device;

    auto share_texture_with_main = [&](const RD::TextureFormat &fmt, RID &r_local, RID &r_external,
                                            RenderingDevice *&r_owner, RenderingDevice *&r_local_owner) -> bool {
        if (!can_share_with_main) {
            return false;
        }

        RID main_texture = main_device->texture_create(fmt, RD::TextureView());
        if (!main_texture.is_valid()) {
            return false;
        }

        uint64_t driver_handle = main_device->get_driver_resource(RenderingDevice::DRIVER_RESOURCE_TEXTURE, main_texture, 0);
        if (driver_handle == 0) {
            main_device->free(main_texture);
            return false;
        }

        RID local_reference = device->texture_create_from_extension(fmt.texture_type, fmt.format, fmt.samples, fmt.usage_bits,
                driver_handle, fmt.width, fmt.height, fmt.depth, fmt.array_layers, fmt.mipmaps);
        if (!local_reference.is_valid()) {
            main_device->free(main_texture);
            return false;
        }

        r_local = local_reference;
        r_external = main_texture;
        r_owner = main_device;
        r_local_owner = device;
        return true;
    };

    RD::TextureFormat resolve_color_format;
    resolve_color_format.width = p_size.x;
    resolve_color_format.height = p_size.y;
    resolve_color_format.depth = 1;
    resolve_color_format.array_layers = 1;
    resolve_color_format.mipmaps = 1;
    resolve_color_format.texture_type = RD::TEXTURE_TYPE_2D;
    resolve_color_format.samples = RD::TEXTURE_SAMPLES_1;
    resolve_color_format.format = resolved_storage_format;
    resolve_color_format.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT |
            RD::TEXTURE_USAGE_SAMPLING_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

    owner.render_targets.resolved_texture_owner.clear();
    RenderingDevice *shared_resolve_owner = nullptr;
    bool shared_resolve = share_texture_with_main(resolve_color_format, owner.render_targets.resolved_texture, owner.render_targets.resolved_texture_external,
            shared_resolve_owner, owner.render_targets.resolved_texture_local_owner);
    if (!shared_resolve) {
        owner.render_targets.resolved_texture = device->texture_create(resolve_color_format, RD::TextureView());
        if (!owner.render_targets.resolved_texture.is_valid()) {
            GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to allocate resolve color texture");
        } else {
            owner.render_targets.resolved_texture_external = owner.render_targets.resolved_texture;
            owner.render_targets.resolved_texture_owner.set(device);
            owner.render_targets.resolved_texture_local_owner = device;
        }
    } else if (shared_resolve_owner) {
        owner.render_targets.resolved_texture_owner.set(shared_resolve_owner);
    }
    if (owner.render_targets.resolved_texture.is_valid()) {
        device->set_resource_name(owner.render_targets.resolved_texture, "GS_TileRenderer_ResolveColorTexture");
    }
    if (owner.render_targets.resolved_texture_external.is_valid() && owner.render_targets.resolved_texture_owner.device &&
            owner.render_targets.resolved_texture_external != owner.render_targets.resolved_texture) {
        owner.render_targets.resolved_texture_owner.device->set_resource_name(owner.render_targets.resolved_texture_external,
                "GS_TileRenderer_ResolveColorTextureExternal");
    }

    RD::TextureFormat resolve_depth_format;
    resolve_depth_format.width = p_size.x;
    resolve_depth_format.height = p_size.y;
    resolve_depth_format.depth = 1;
    resolve_depth_format.array_layers = 1;
    resolve_depth_format.mipmaps = 1;
    resolve_depth_format.texture_type = RD::TEXTURE_TYPE_2D;
    resolve_depth_format.samples = RD::TEXTURE_SAMPLES_1;
    resolve_depth_format.format = RD::DATA_FORMAT_R32_SFLOAT;
    resolve_depth_format.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT |
            RD::TEXTURE_USAGE_SAMPLING_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

    owner.render_targets.resolved_depth_texture_owner.clear();
    RenderingDevice *shared_resolve_depth_owner = nullptr;
    bool shared_depth = share_texture_with_main(resolve_depth_format, owner.render_targets.resolved_depth_texture,
            owner.render_targets.resolved_depth_texture_external, shared_resolve_depth_owner, owner.render_targets.resolved_depth_texture_local_owner);
    if (!shared_depth) {
        owner.render_targets.resolved_depth_texture = device->texture_create(resolve_depth_format, RD::TextureView());
        if (!owner.render_targets.resolved_depth_texture.is_valid()) {
            GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to allocate resolve depth texture");
        } else {
            owner.render_targets.resolved_depth_texture_external = owner.render_targets.resolved_depth_texture;
            owner.render_targets.resolved_depth_texture_owner.set(device);
            owner.render_targets.resolved_depth_texture_local_owner = device;
        }
    } else if (shared_resolve_depth_owner) {
        owner.render_targets.resolved_depth_texture_owner.set(shared_resolve_depth_owner);
    }
    if (owner.render_targets.resolved_depth_texture.is_valid()) {
        device->set_resource_name(owner.render_targets.resolved_depth_texture, "GS_TileRenderer_ResolveDepthTexture");
    }
    if (owner.render_targets.resolved_depth_texture_external.is_valid() && owner.render_targets.resolved_depth_texture_owner.device &&
            owner.render_targets.resolved_depth_texture_external != owner.render_targets.resolved_depth_texture) {
        owner.render_targets.resolved_depth_texture_owner.device->set_resource_name(owner.render_targets.resolved_depth_texture_external,
                "GS_TileRenderer_ResolveDepthTextureExternal");
    }

    update_depth_copy_contract();

    ensure_resolve_pipeline(device, owner.config_state.resolve_target_format);
}

void TileRenderer::TileResolveStage::ensure_resolve_pipeline(RenderingDevice *p_device, RD::DataFormat p_format) {
    RenderingDevice *device = p_device ? p_device : owner._get_resource_device();
    if (!device) {
        return;
    }

    RD::DataFormat requested_format = p_format;
    if (requested_format == RD::DATA_FORMAT_MAX) {
        requested_format = owner.config_state.resolve_target_format != RD::DATA_FORMAT_MAX
                ? owner.config_state.resolve_target_format
                : RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }
    String pipeline_fallback_reason;
    RD::DataFormat pipeline_format = _resolve_storage_format(requested_format, &pipeline_fallback_reason);
    if (pipeline_format != requested_format) {
        _log_resolve_format_fallback_once("resolve_pipeline", requested_format, pipeline_format, pipeline_fallback_reason);
        const bool accepted_fallback = _is_resolve_contract_fallback_accepted(requested_format, pipeline_format);
        if (!accepted_fallback) {
            ERR_PRINT_ONCE(vformat("[TileRenderer] resolve_pipeline contract violation: requested_format=%d resolved_format=%d reason=%s. Resolve pipeline disabled.",
                    int(requested_format), int(pipeline_format), pipeline_fallback_reason));
            owner.shader_resources.resolve_pipeline_initialized = false;
            owner.config_state.resolve_target_format = RD::DATA_FORMAT_MAX;
            owner.render_targets.depth_texture_copy_compatible = false;
            return;
        }
    }

	if (!owner.shader_resources.tile_resolve_shader_initialized) {
		ERR_FAIL_NULL(owner.shader_resources.tile_resolve_shader_source.get());
		Vector<String> variants;
		variants.push_back("");
		owner.shader_resources.tile_resolve_shader_source->initialize(variants);
		owner.shader_resources.tile_resolve_shader_initialized = true;
	}

    if (!owner.shader_resources.tile_resolve_shader.is_valid() || owner.shader_resources.resolve_pipeline_format != pipeline_format) {
        // Map the actual resolve format to shader define.
        Vector<String> resolve_defines = owner._build_common_shader_defines(false);
        int format_define = 0;
        if (pipeline_format == RD::DATA_FORMAT_R16G16B16A16_SFLOAT) {
            format_define = 1;
        } else if (pipeline_format == RD::DATA_FORMAT_R32G32B32A32_SFLOAT) {
            format_define = 2;
        }
        resolve_defines.push_back(vformat("#define TILE_RESOLVE_FORMAT %d\n", format_define));

		ERR_FAIL_NULL(owner.shader_resources.tile_resolve_shader_source.get());
		RID resolve_version = owner.shader_resources.tile_resolve_shader_source->version_create();
		if (!resolve_version.is_valid()) {
			GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to create resolve shader variant");
			return;
		}

		Vector<String> resolve_stage_sources = owner.shader_resources.tile_resolve_shader_source->version_build_variant_stage_sources(resolve_version, 0);
		owner.shader_resources.tile_resolve_shader_source->version_free(resolve_version);
        if (resolve_stage_sources.size() <= RD::SHADER_STAGE_COMPUTE) {
            GS_LOG_ERROR_DEFAULT("[TileRenderer] Resolve shader missing compute stage");
            return;
        }

        String resolve_error;
        owner.shader_resources.tile_resolve_shader = ShaderCompilationHelper::compile_shader_on_device(device,
                resolve_stage_sources[RD::SHADER_STAGE_COMPUTE],
                "tile_resolve", resolve_defines, &resolve_error);
        if (!owner.shader_resources.tile_resolve_shader.is_valid()) {
            GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] Failed to compile tile_resolve.glsl: %s", resolve_error));
            return;
        }
        owner.shader_resources.resolve_pipeline_format = pipeline_format;
    }

	if (!owner.shader_resources.tile_resolve_pipeline.is_valid() || !owner.shader_resources.tile_resolve_shader.is_valid()) {
		if (owner.shader_resources.tile_resolve_pipeline.is_valid()) {
			if (device->compute_pipeline_is_valid(owner.shader_resources.tile_resolve_pipeline)) {
				device->free(owner.shader_resources.tile_resolve_pipeline);
			}
		}
		owner.shader_resources.tile_resolve_pipeline = device->compute_pipeline_create(owner.shader_resources.tile_resolve_shader);
        if (!owner.shader_resources.tile_resolve_pipeline.is_valid()) {
            GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to create compute pipeline for tile resolve stage");
            return;
        }
    }

    owner.shader_resources.resolve_pipeline_initialized = owner.shader_resources.tile_resolve_pipeline.is_valid();
}

bool TileRenderer::TileResolveStage::ensure_resolve_sampler(RenderingDevice *p_device) {
    if (!p_device) {
        return false;
    }
    if (!resolve_sampler.is_valid() || !resolve_sampler_owner.matches(p_device)) {
        if (resolve_sampler.is_valid() && resolve_sampler_owner.device && !resolve_sampler_owner.matches(p_device)) {
            resolve_sampler_owner.device->free(resolve_sampler);
        }

        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.enable_compare = false;
        resolve_sampler = p_device->sampler_create(sampler_state);
        resolve_sampler_owner.set(p_device);
    }

    return resolve_sampler.is_valid();
}

bool TileRenderer::TileResolveStage::ensure_shadow_sampler(RenderingDevice *p_device) {
    if (!p_device) {
        return false;
    }
    if (!shadow_sampler.is_valid() || !shadow_sampler_owner.matches(p_device)) {
        if (shadow_sampler.is_valid() && shadow_sampler_owner.device && !shadow_sampler_owner.matches(p_device)) {
            shadow_sampler_owner.device->free(shadow_sampler);
        }

        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.enable_compare = true;
        sampler_state.compare_op = RD::COMPARE_OP_GREATER;
        shadow_sampler = p_device->sampler_create(sampler_state);
        shadow_sampler_owner.set(p_device);
    }

    return shadow_sampler.is_valid();
}

RID TileRenderer::TileResolveStage::create_resolve_uniform_set(RenderingDevice *p_device) {
    ERR_FAIL_NULL_V(p_device, RID());
    if (!ensure_resolve_sampler(p_device)) {
        return RID();
    }

    Vector<RD::Uniform> uniforms;

    RD::Uniform input_color_uniform;
    input_color_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    input_color_uniform.binding = 0;
    input_color_uniform.append_id(resolve_sampler);
    input_color_uniform.append_id(owner.render_targets.output_texture);
    uniforms.push_back(input_color_uniform);

    RD::Uniform input_depth_uniform;
    input_depth_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    input_depth_uniform.binding = 1;
    input_depth_uniform.append_id(resolve_sampler);
    input_depth_uniform.append_id(owner.render_targets.depth_texture);
    uniforms.push_back(input_depth_uniform);

    RD::Uniform input_normal_uniform;
    input_normal_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    input_normal_uniform.binding = 4;
    input_normal_uniform.append_id(resolve_sampler);
    input_normal_uniform.append_id(owner.render_targets.normal_texture);
    uniforms.push_back(input_normal_uniform);

    RD::Uniform output_color_uniform;
    output_color_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
    output_color_uniform.binding = 2;
    output_color_uniform.append_id(owner.render_targets.resolved_texture);
    uniforms.push_back(output_color_uniform);

    RD::Uniform output_depth_uniform;
    output_depth_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
    output_depth_uniform.binding = 3;
    output_depth_uniform.append_id(owner.render_targets.resolved_depth_texture);
    uniforms.push_back(output_depth_uniform);

    // ISSUE-002: Verify all textures belong to p_device before creating resolve uniform set.
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.output_texture, "resolve:output_texture"), RID());
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.depth_texture, "resolve:depth_texture"), RID());
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.normal_texture, "resolve:normal_texture"), RID());
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.resolved_texture, "resolve:resolved_texture"), RID());
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.resolved_depth_texture, "resolve:resolved_depth_texture"), RID());

    RID resolve_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_resolve_shader, 0);
    if (resolve_uniform_set.is_valid()) {
        p_device->set_resource_name(resolve_uniform_set, "GS_TileRenderer_ResolveSet");
    }
    return resolve_uniform_set;
}

RID TileRenderer::TileResolveStage::create_resolve_param_uniform_set(RenderingDevice *p_device) {
    ERR_FAIL_NULL_V(p_device, RID());
    ERR_FAIL_COND_V(!owner.uniform_buffers.param_uniform_buffer.is_valid(), RID());

    Vector<RD::Uniform> uniforms;
    RD::Uniform params_uniform;
    params_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    params_uniform.binding = 0;
    params_uniform.append_id(owner.uniform_buffers.param_uniform_buffer);
    uniforms.push_back(params_uniform);

    RID param_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_resolve_shader, 1);
    if (param_uniform_set.is_valid()) {
        p_device->set_resource_name(param_uniform_set, "GS_TileRenderer_ResolveParamSet");
    }
    return param_uniform_set;
}

void TileRenderer::TileResolveStage::free_fallback_lighting_buffers(RenderingDevice *p_device) {
    RenderingDevice *device = fallback_lighting_owner.device ? fallback_lighting_owner.device : p_device;
    if (!device) {
        fallback_scene_uniform_buffer = RID();
        fallback_directional_light_buffer = RID();
        fallback_omni_light_buffer = RID();
        fallback_spot_light_buffer = RID();
        fallback_reflection_buffer = RID();
        fallback_cluster_buffer = RID();
        fallback_decal_texture = RID();
        fallback_reflection_texture = RID();
        fallback_lighting_owner.clear();
        return;
    }

    auto free_resource = [&](RID &p_rid) {
        if (p_rid.is_valid()) {
            device->free(p_rid);
            p_rid = RID();
        }
    };

    free_resource(fallback_scene_uniform_buffer);
    free_resource(fallback_directional_light_buffer);
    free_resource(fallback_omni_light_buffer);
    free_resource(fallback_spot_light_buffer);
    free_resource(fallback_reflection_buffer);
    free_resource(fallback_cluster_buffer);
    free_resource(fallback_decal_texture);
    free_resource(fallback_reflection_texture);
    free_resource(fallback_shadow_texture);
    free_resource(fallback_directional_shadow_texture);
    fallback_lighting_owner.clear();
}

bool TileRenderer::TileResolveStage::ensure_fallback_lighting_buffers(RenderingDevice *p_device) {
    if (!p_device) {
        return false;
    }

    const bool owner_matches = fallback_lighting_owner.matches(p_device);
    const bool buffers_ready = fallback_scene_uniform_buffer.is_valid() &&
            fallback_directional_light_buffer.is_valid() &&
            fallback_omni_light_buffer.is_valid() &&
            fallback_spot_light_buffer.is_valid() &&
            fallback_reflection_buffer.is_valid() &&
            fallback_cluster_buffer.is_valid() &&
            fallback_decal_texture.is_valid() &&
            fallback_reflection_texture.is_valid() &&
            fallback_shadow_texture.is_valid() &&
            fallback_directional_shadow_texture.is_valid();
    if (owner_matches && buffers_ready) {
        return true;
    }

    free_fallback_lighting_buffers(p_device);

    Vector<uint8_t> zero_data;
    zero_data.resize(kFallbackSceneUniformBytes);
    zero_data.fill(0);
    fallback_scene_uniform_buffer = p_device->uniform_buffer_create(zero_data.size(), zero_data);
    if (!fallback_scene_uniform_buffer.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_scene_uniform_buffer, "GS_TileResolve_FallbackSceneUBO");

    zero_data.resize(kFallbackDirectionalUniformBytes);
    zero_data.fill(0);
    fallback_directional_light_buffer = p_device->uniform_buffer_create(zero_data.size(), zero_data);
    if (!fallback_directional_light_buffer.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_directional_light_buffer, "GS_TileResolve_FallbackDirectionalUBO");

    zero_data.resize(kFallbackLightStorageBytes);
    zero_data.fill(0);
    fallback_omni_light_buffer = p_device->storage_buffer_create(zero_data.size(), zero_data);
    if (!fallback_omni_light_buffer.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_omni_light_buffer, "GS_TileResolve_FallbackOmniSSBO");

    fallback_spot_light_buffer = p_device->storage_buffer_create(zero_data.size(), zero_data);
    if (!fallback_spot_light_buffer.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_spot_light_buffer, "GS_TileResolve_FallbackSpotSSBO");

    fallback_reflection_buffer = p_device->storage_buffer_create(zero_data.size(), zero_data);
    if (!fallback_reflection_buffer.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_reflection_buffer, "GS_TileResolve_FallbackReflectionSSBO");

    zero_data.resize(kFallbackClusterStorageBytes);
    zero_data.fill(0);
    fallback_cluster_buffer = p_device->storage_buffer_create(zero_data.size(), zero_data);
    if (!fallback_cluster_buffer.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_cluster_buffer, "GS_TileResolve_FallbackClusterSSBO");

    RD::TextureFormat decal_format;
    decal_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    decal_format.width = kFallbackDecalTextureSize;
    decal_format.height = kFallbackDecalTextureSize;
    decal_format.depth = 1;
    decal_format.array_layers = 1;
    decal_format.mipmaps = 1;
    decal_format.texture_type = RD::TEXTURE_TYPE_2D;
    decal_format.samples = RD::TEXTURE_SAMPLES_1;
    decal_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;

    Vector<uint8_t> decal_pixel_data;
    decal_pixel_data.resize(kFallbackDecalTextureSize * kFallbackDecalTextureSize * 4);
    for (int i = 0; i < kFallbackDecalTextureSize * kFallbackDecalTextureSize; ++i) {
        decal_pixel_data.set(i * 4 + 0, 0);
        decal_pixel_data.set(i * 4 + 1, 0);
        decal_pixel_data.set(i * 4 + 2, 0);
        decal_pixel_data.set(i * 4 + 3, 255);
    }

    Vector<Vector<uint8_t>> decal_layers;
    decal_layers.push_back(decal_pixel_data);
    fallback_decal_texture = p_device->texture_create(decal_format, RD::TextureView(), decal_layers);
    if (!fallback_decal_texture.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_decal_texture, "GS_TileResolve_FallbackDecalTexture");

    RD::TextureFormat reflection_format;
    reflection_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    reflection_format.width = kFallbackReflectionTextureSize;
    reflection_format.height = kFallbackReflectionTextureSize;
    reflection_format.depth = 1;
    reflection_format.array_layers = 6;
    reflection_format.mipmaps = 1;
    reflection_format.texture_type = RD::TEXTURE_TYPE_CUBE_ARRAY;
    reflection_format.samples = RD::TEXTURE_SAMPLES_1;
    reflection_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;

    Vector<uint8_t> reflection_pixel_data;
    reflection_pixel_data.resize(kFallbackReflectionTextureSize * kFallbackReflectionTextureSize * 4);
    for (int i = 0; i < kFallbackReflectionTextureSize * kFallbackReflectionTextureSize; ++i) {
        reflection_pixel_data.set(i * 4 + 0, 0);
        reflection_pixel_data.set(i * 4 + 1, 0);
        reflection_pixel_data.set(i * 4 + 2, 0);
        reflection_pixel_data.set(i * 4 + 3, 255);
    }

    Vector<Vector<uint8_t>> reflection_layers;
    reflection_layers.resize(6);
    for (int i = 0; i < 6; ++i) {
        reflection_layers.set(i, reflection_pixel_data);
    }
    fallback_reflection_texture = p_device->texture_create(reflection_format, RD::TextureView(), reflection_layers);
    if (!fallback_reflection_texture.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_reflection_texture, "GS_TileResolve_FallbackReflectionTexture");

    RD::TextureFormat shadow_format;
    shadow_format.format = RD::DATA_FORMAT_D32_SFLOAT;
    shadow_format.width = kFallbackShadowTextureSize;
    shadow_format.height = kFallbackShadowTextureSize;
    shadow_format.depth = 1;
    shadow_format.array_layers = 1;
    shadow_format.mipmaps = 1;
    shadow_format.texture_type = RD::TEXTURE_TYPE_2D;
    shadow_format.samples = RD::TEXTURE_SAMPLES_1;
    shadow_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;

    Vector<uint8_t> shadow_pixel_data;
    shadow_pixel_data.resize(sizeof(float) * kFallbackShadowTextureSize * kFallbackShadowTextureSize);
    shadow_pixel_data.fill(0); // Depth = 0 -> compare GREATER yields lit
    Vector<Vector<uint8_t>> shadow_layers;
    shadow_layers.push_back(shadow_pixel_data);

    fallback_shadow_texture = p_device->texture_create(shadow_format, RD::TextureView(), shadow_layers);
    if (!fallback_shadow_texture.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_shadow_texture, "GS_TileResolve_FallbackShadowAtlas");

    fallback_directional_shadow_texture = p_device->texture_create(shadow_format, RD::TextureView(), shadow_layers);
    if (!fallback_directional_shadow_texture.is_valid()) {
        free_fallback_lighting_buffers(p_device);
        return false;
    }
    p_device->set_resource_name(fallback_directional_shadow_texture, "GS_TileResolve_FallbackDirectionalShadowAtlas");

    fallback_lighting_owner.set(p_device);
    return true;
}

RID TileRenderer::TileResolveStage::create_lighting_uniform_set(RenderingDevice *p_device, const RenderParams &p_params) {
    ERR_FAIL_NULL_V(p_device, RID());
    if (!ensure_resolve_sampler(p_device)) {
        return RID();
    }
    if (!ensure_shadow_sampler(p_device)) {
        return RID();
    }

    RID scene_buffer = p_params.scene_uniform_buffer;
    bool using_fallback_scene = false;
    if (!scene_buffer.is_valid()) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        scene_buffer = fallback_scene_uniform_buffer;
        using_fallback_scene = true;
    }

    RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton();
    RID directional_buffer = p_params.directional_light_buffer;
    if (!directional_buffer.is_valid()) {
        directional_buffer = light_storage ? light_storage->get_directional_light_buffer() : RID();
    }
    if (!directional_buffer.is_valid()) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        directional_buffer = fallback_directional_light_buffer;
    }

    RID omni_buffer = light_storage ? light_storage->get_omni_light_buffer() : RID();
    if (!omni_buffer.is_valid()) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        omni_buffer = fallback_omni_light_buffer;
    }

    RID spot_buffer = light_storage ? light_storage->get_spot_light_buffer() : RID();
    if (!spot_buffer.is_valid()) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        spot_buffer = fallback_spot_light_buffer;
    }

    RID reflection_buffer = light_storage ? light_storage->get_reflection_probe_buffer() : RID();
    if (!reflection_buffer.is_valid()) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        reflection_buffer = fallback_reflection_buffer;
    }

    RID cluster_buffer = p_params.cluster_buffer;
    if (!cluster_buffer.is_valid()) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        cluster_buffer = fallback_cluster_buffer;
    }

    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    RID decal_texture;
    RID reflection_texture;
    RID shadow_atlas_texture;
    RID directional_shadow_texture;
    RID default_depth_texture;
    if (texture_storage) {
        decal_texture = texture_storage->decal_atlas_get_texture_srgb();
        if (!decal_texture.is_valid()) {
            decal_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
        }
        reflection_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK);
        default_depth_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_DEPTH);
    }

    if (light_storage && p_params.shadow_atlas.is_valid()) {
        shadow_atlas_texture = light_storage->shadow_atlas_get_texture(p_params.shadow_atlas);
    }
    if (light_storage) {
        directional_shadow_texture = light_storage->directional_shadow_get_texture();
    }
    if (!shadow_atlas_texture.is_valid()) {
        shadow_atlas_texture = default_depth_texture;
    }
    if (!directional_shadow_texture.is_valid()) {
        directional_shadow_texture = default_depth_texture;
    }

    if (!decal_texture.is_valid() || !p_device->texture_is_valid(decal_texture)) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        decal_texture = fallback_decal_texture;
    }

    if (!reflection_texture.is_valid() || !p_device->texture_is_valid(reflection_texture)) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        reflection_texture = fallback_reflection_texture;
    }

    if (!shadow_atlas_texture.is_valid() || !p_device->texture_is_valid(shadow_atlas_texture)) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        shadow_atlas_texture = fallback_shadow_texture;
    }

    if (!directional_shadow_texture.is_valid() || !p_device->texture_is_valid(directional_shadow_texture)) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        directional_shadow_texture = fallback_directional_shadow_texture;
    }

    Vector<RD::Uniform> uniforms;
    RD::Uniform scene_uniform;
    scene_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    scene_uniform.binding = 0;
    scene_uniform.append_id(scene_buffer);
    uniforms.push_back(scene_uniform);

    RD::Uniform directional_uniform;
    directional_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    directional_uniform.binding = 1;
    directional_uniform.append_id(directional_buffer);
    uniforms.push_back(directional_uniform);

    RD::Uniform omni_uniform;
    omni_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    omni_uniform.binding = 2;
    omni_uniform.append_id(omni_buffer);
    uniforms.push_back(omni_uniform);

    RD::Uniform spot_uniform;
    spot_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    spot_uniform.binding = 3;
    spot_uniform.append_id(spot_buffer);
    uniforms.push_back(spot_uniform);

    RD::Uniform reflection_uniform;
    reflection_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    reflection_uniform.binding = 4;
    reflection_uniform.append_id(reflection_buffer);
    uniforms.push_back(reflection_uniform);

    RD::Uniform cluster_uniform;
    cluster_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    cluster_uniform.binding = 9;
    cluster_uniform.append_id(cluster_buffer);
    uniforms.push_back(cluster_uniform);

    RD::Uniform decal_uniform;
    decal_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
    decal_uniform.binding = 5;
    decal_uniform.append_id(decal_texture);
    uniforms.push_back(decal_uniform);

    RD::Uniform reflection_atlas_uniform;
    reflection_atlas_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
    reflection_atlas_uniform.binding = 6;
    reflection_atlas_uniform.append_id(reflection_texture);
    uniforms.push_back(reflection_atlas_uniform);

    RD::Uniform projector_sampler_uniform;
    projector_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
    projector_sampler_uniform.binding = 7;
    projector_sampler_uniform.append_id(resolve_sampler);
    uniforms.push_back(projector_sampler_uniform);

    RD::Uniform default_sampler_uniform;
    default_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
    default_sampler_uniform.binding = 8;
    default_sampler_uniform.append_id(resolve_sampler);
    uniforms.push_back(default_sampler_uniform);

    RD::Uniform shadow_sampler_uniform;
    shadow_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
    shadow_sampler_uniform.binding = 10;
    shadow_sampler_uniform.append_id(shadow_sampler);
    uniforms.push_back(shadow_sampler_uniform);

    RD::Uniform shadow_atlas_uniform;
    shadow_atlas_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
    shadow_atlas_uniform.binding = 11;
    shadow_atlas_uniform.append_id(shadow_atlas_texture);
    uniforms.push_back(shadow_atlas_uniform);

    RD::Uniform directional_shadow_uniform;
    directional_shadow_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
    directional_shadow_uniform.binding = 12;
    directional_shadow_uniform.append_id(directional_shadow_texture);
    uniforms.push_back(directional_shadow_uniform);

    RD::Uniform linear_clamp_uniform;
    linear_clamp_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
    linear_clamp_uniform.binding = 13;
    linear_clamp_uniform.append_id(resolve_sampler);
    uniforms.push_back(linear_clamp_uniform);

    RID lighting_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_resolve_shader, 2);
    if (!lighting_uniform_set.is_valid()) {
        if (!ensure_fallback_lighting_buffers(p_device)) {
            return RID();
        }
        uniforms.clear();

        RD::Uniform fallback_scene_uniform;
        fallback_scene_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
        fallback_scene_uniform.binding = 0;
        fallback_scene_uniform.append_id(fallback_scene_uniform_buffer);
        uniforms.push_back(fallback_scene_uniform);

        RD::Uniform fallback_directional_uniform;
        fallback_directional_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
        fallback_directional_uniform.binding = 1;
        fallback_directional_uniform.append_id(fallback_directional_light_buffer);
        uniforms.push_back(fallback_directional_uniform);

        RD::Uniform fallback_omni_uniform;
        fallback_omni_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
        fallback_omni_uniform.binding = 2;
        fallback_omni_uniform.append_id(fallback_omni_light_buffer);
        uniforms.push_back(fallback_omni_uniform);

        RD::Uniform fallback_spot_uniform;
        fallback_spot_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
        fallback_spot_uniform.binding = 3;
        fallback_spot_uniform.append_id(fallback_spot_light_buffer);
        uniforms.push_back(fallback_spot_uniform);

        RD::Uniform fallback_reflection_uniform;
        fallback_reflection_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
        fallback_reflection_uniform.binding = 4;
        fallback_reflection_uniform.append_id(fallback_reflection_buffer);
        uniforms.push_back(fallback_reflection_uniform);

        RD::Uniform fallback_cluster_uniform;
        fallback_cluster_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
        fallback_cluster_uniform.binding = 9;
        fallback_cluster_uniform.append_id(fallback_cluster_buffer);
        uniforms.push_back(fallback_cluster_uniform);

        RD::Uniform fallback_decal_uniform;
        fallback_decal_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
        fallback_decal_uniform.binding = 5;
        fallback_decal_uniform.append_id(decal_texture);
        uniforms.push_back(fallback_decal_uniform);

        RD::Uniform fallback_reflection_atlas_uniform;
        fallback_reflection_atlas_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
        fallback_reflection_atlas_uniform.binding = 6;
        fallback_reflection_atlas_uniform.append_id(reflection_texture);
        uniforms.push_back(fallback_reflection_atlas_uniform);

        RD::Uniform fallback_projector_sampler_uniform;
        fallback_projector_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
        fallback_projector_sampler_uniform.binding = 7;
        fallback_projector_sampler_uniform.append_id(resolve_sampler);
        uniforms.push_back(fallback_projector_sampler_uniform);

        RD::Uniform fallback_default_sampler_uniform;
        fallback_default_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
        fallback_default_sampler_uniform.binding = 8;
        fallback_default_sampler_uniform.append_id(resolve_sampler);
        uniforms.push_back(fallback_default_sampler_uniform);

        RD::Uniform fallback_shadow_sampler_uniform;
        fallback_shadow_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
        fallback_shadow_sampler_uniform.binding = 10;
        fallback_shadow_sampler_uniform.append_id(shadow_sampler);
        uniforms.push_back(fallback_shadow_sampler_uniform);

        RD::Uniform fallback_shadow_atlas_uniform;
        fallback_shadow_atlas_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
        fallback_shadow_atlas_uniform.binding = 11;
        fallback_shadow_atlas_uniform.append_id(fallback_shadow_texture);
        uniforms.push_back(fallback_shadow_atlas_uniform);

        RD::Uniform fallback_directional_shadow_uniform;
        fallback_directional_shadow_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
        fallback_directional_shadow_uniform.binding = 12;
        fallback_directional_shadow_uniform.append_id(fallback_directional_shadow_texture);
        uniforms.push_back(fallback_directional_shadow_uniform);

        RD::Uniform fallback_linear_clamp_uniform;
        fallback_linear_clamp_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
        fallback_linear_clamp_uniform.binding = 13;
        fallback_linear_clamp_uniform.append_id(resolve_sampler);
        uniforms.push_back(fallback_linear_clamp_uniform);

        lighting_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_resolve_shader, 2);
    }
    if (lighting_uniform_set.is_valid()) {
        p_device->set_resource_name(lighting_uniform_set, "GS_TileRenderer_ResolveLightingSet");
    }
    return lighting_uniform_set;
}

TileRenderer::TileResolveStage::ResolvePushConstants TileRenderer::TileResolveStage::build_push_constants(
        const Vector2i &p_viewport, int p_tile_size, bool p_output_is_premultiplied) const {
    ResolvePushConstants push_constants;
    push_constants.viewport_width = p_viewport.x;
    push_constants.viewport_height = p_viewport.y;
    push_constants.tile_size_pixels = p_tile_size > 0 ? p_tile_size : owner.config_state.tile_size;
    if (push_constants.tile_size_pixels <= 0) {
        push_constants.tile_size_pixels = 1;
    }
    float max_feather = float(push_constants.tile_size_pixels) * 0.5f;
    push_constants.feather_pixels = CLAMP(owner.render_settings.resolve_feather_pixels, 0.0f, max_feather);
    push_constants.tiles_x = int32_t(owner.grid_state.tiles_x);
    push_constants.tiles_y = int32_t(owner.grid_state.tiles_y);

    auto compute_last_extent = [](int32_t p_view_size, int32_t p_tile_size_pixels, int32_t p_tiles) -> int32_t {
        if (p_tiles <= 0 || p_tile_size_pixels <= 0) {
            return p_tile_size_pixels;
        }
        int32_t covered = p_tile_size_pixels * (p_tiles - 1);
        int32_t remainder = p_view_size - covered;
        if (remainder <= 0) {
            remainder = p_tile_size_pixels;
        }
        return remainder;
    };

    push_constants.last_tile_width = compute_last_extent(push_constants.viewport_width, push_constants.tile_size_pixels,
            push_constants.tiles_x);
    push_constants.last_tile_height = compute_last_extent(push_constants.viewport_height, push_constants.tile_size_pixels,
            push_constants.tiles_y);
    push_constants.debug_visualize_tiles = owner.diagnostics.resolve_debug_visualize_tiles ? 1 : 0;
    push_constants.use_texel_fetch_sampling = owner.diagnostics.resolve_use_texel_fetch_sampling ? 1 : 0;
    push_constants.output_is_premultiplied = p_output_is_premultiplied ? 1 : 0;
    return push_constants;
}

void TileRenderer::TileResolveStage::dispatch_tile_resolve(const Vector2i &p_viewport, int p_tile_size,
        bool p_output_is_premultiplied, const RenderParams &p_params) {
    if (owner.render_settings.resolve_debug_mode == RESOLVE_DEBUG_INPUT) {
        return; // Bypass resolve stage and consume raw raster outputs.
    }

    if (!owner.render_targets.resolved_texture.is_valid() || !owner.render_targets.resolved_depth_texture.is_valid() ||
            !owner.render_targets.output_texture.is_valid() || !owner.render_targets.depth_texture.is_valid() ||
            !owner.render_targets.normal_texture.is_valid()) {
        WARN_PRINT_ONCE("[TileRenderer] Resolve contract unavailable: required resolve/raster targets are missing; resolve pass skipped");
        owner.render_targets.depth_texture_copy_compatible = false;
        return;
    }

    RenderingDevice *resource_device = owner._get_resource_device();
    if (!resource_device) {
        WARN_PRINT_ONCE("[TileRenderer] Resolve contract unavailable: resource device missing; resolve pass skipped");
        owner.render_targets.depth_texture_copy_compatible = false;
        return;
    }

    // Keep resolve pipeline format aligned with the actual resolved texture format.
    RD::DataFormat configured_format = owner.config_state.resolve_target_format;
    RD::DataFormat texture_format = configured_format;
    if (resource_device->texture_is_valid(owner.render_targets.resolved_texture)) {
        RD::TextureFormat fmt = resource_device->texture_get_format(owner.render_targets.resolved_texture);
        texture_format = fmt.format;
    }
    String drift_fallback_reason;
    RD::DataFormat resolved_storage_format = _resolve_storage_format(texture_format, &drift_fallback_reason);

    // If the resolved texture format drifted, recreate resources/pipeline for the actual format.
    if (texture_format != configured_format || resolved_storage_format != configured_format) {
        if (resolved_storage_format != texture_format) {
            _log_resolve_format_fallback_once("resolve_drift", texture_format, resolved_storage_format, drift_fallback_reason);
        } else {
            WARN_PRINT_ONCE(vformat("[TileRenderer] resolve_drift format fallback: requested=%d resolved=%d reason=drift from configured resolve format %d",
                    int(texture_format), int(resolved_storage_format), int(configured_format)));
        }
        ensure_resolve_resources(p_viewport, texture_format);
    }

    ensure_resolve_pipeline(resource_device, owner.config_state.resolve_target_format);
    if (!owner.shader_resources.resolve_pipeline_initialized || !owner.shader_resources.tile_resolve_pipeline.is_valid()) {
        WARN_PRINT_ONCE("[TileRenderer] Resolve pipeline unavailable after contract validation; resolve pass skipped");
        owner.render_targets.depth_texture_copy_compatible = false;
        return;
    }

    uint32_t dispatch_x = (uint32_t)((p_viewport.x + 7) / 8);
    uint32_t dispatch_y = (uint32_t)((p_viewport.y + 7) / 8);
    if (dispatch_x == 0 || dispatch_y == 0) {
        return;
    }

    RenderingDevice *submission_device = owner._acquire_submission_device();
    if (!submission_device) {
        submission_device = resource_device;
    }
    if (!submission_device) {
        return;
    }

    ensure_resolve_pipeline(submission_device, owner.config_state.resolve_target_format);
    if (!owner.shader_resources.resolve_pipeline_initialized || !owner.shader_resources.tile_resolve_pipeline.is_valid()) {
        WARN_PRINT_ONCE("[TileRenderer] Resolve pipeline unavailable on submission device; resolve pass skipped");
        owner.render_targets.depth_texture_copy_compatible = false;
        return;
    }

    RID resolve_uniform_set = create_resolve_uniform_set(submission_device);
    if (!resolve_uniform_set.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to create resolve uniform set");
        return;
    }
    RID param_uniform_set = create_resolve_param_uniform_set(submission_device);
    if (!param_uniform_set.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to create resolve param uniform set");
        if (submission_device->uniform_set_is_valid(resolve_uniform_set)) {
            submission_device->free(resolve_uniform_set);
        }
        return;
    }
    RID lighting_uniform_set = create_lighting_uniform_set(submission_device, p_params);
    if (!lighting_uniform_set.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to create resolve lighting uniform set");
        if (submission_device->uniform_set_is_valid(resolve_uniform_set)) {
            submission_device->free(resolve_uniform_set);
        }
        if (submission_device->uniform_set_is_valid(param_uniform_set)) {
            submission_device->free(param_uniform_set);
        }
        return;
    }

    RD::ComputeListID compute_list = submission_device->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        ERR_PRINT_ONCE("[TileRenderer] Failed to begin resolve compute list");
        if (submission_device->uniform_set_is_valid(resolve_uniform_set)) {
            submission_device->free(resolve_uniform_set);
        }
        if (submission_device->uniform_set_is_valid(param_uniform_set)) {
            submission_device->free(param_uniform_set);
        }
        if (submission_device->uniform_set_is_valid(lighting_uniform_set)) {
            submission_device->free(lighting_uniform_set);
        }
        return;
    }
    uint32_t timestamp_base = submission_device->get_captured_timestamps_count();
    String resolve_label = "TileResolve_" + String::num_uint64(owner.frame_state.current_frame_serial);
    submission_device->capture_timestamp(resolve_label + String("_Begin"));
    ScopedGpuMarker resolve_marker(submission_device, "GS_TileResolve", Color(0.3f, 0.6f, 1.0f, 1.0f));

    submission_device->compute_list_bind_compute_pipeline(compute_list, owner.shader_resources.tile_resolve_pipeline);
    submission_device->compute_list_bind_uniform_set(compute_list, resolve_uniform_set, 0);
    submission_device->compute_list_bind_uniform_set(compute_list, param_uniform_set, 1);
    submission_device->compute_list_bind_uniform_set(compute_list, lighting_uniform_set, 2);

    ResolvePushConstants push_constants = build_push_constants(p_viewport, p_tile_size, p_output_is_premultiplied);

    if (owner.diagnostics.debug_log_resolve) {
        GS_LOG_EVERY_N(gs_logger::Category::RENDERER, gs_logger::Level::DEBUG,
                tile_resolve_params_log_counter, owner.diagnostics.debug_log_resolve_interval_frames,
                vformat("[TileResolve] params frame=%s viewport=%dx%d tile=%d feather=%.2f tile_count=%dx%d last_tile=%dx%d overlay=%s texel_fetch=%s dispatch=%sx%s",
                        String::num_uint64(owner.frame_state.current_frame_serial),
                        push_constants.viewport_width,
                        push_constants.viewport_height,
                        push_constants.tile_size_pixels,
                        push_constants.feather_pixels,
                        push_constants.tiles_x,
                        push_constants.tiles_y,
                        push_constants.last_tile_width,
                        push_constants.last_tile_height,
                        owner.diagnostics.resolve_debug_visualize_tiles ? "on" : "off",
                        owner.diagnostics.resolve_use_texel_fetch_sampling ? "on" : "off",
                        String::num_uint64(dispatch_x),
                        String::num_uint64(dispatch_y)));
    }

    submission_device->compute_list_set_push_constant(compute_list, &push_constants, sizeof(ResolvePushConstants));
    submission_device->compute_list_dispatch(compute_list, dispatch_x, dispatch_y, 1);
    submission_device->compute_list_end();
    submission_device->capture_timestamp(resolve_label + String("_End"));

    owner.timing_state.resolve_timestamp.device = submission_device;
    owner.timing_state.resolve_timestamp.start_index = timestamp_base;
    owner.timing_state.resolve_timestamp.end_index = timestamp_base + 1;
    owner.timing_state.resolve_timestamp.label = resolve_label;

    // Main RenderingDevice submission is frame-owned by Godot.
    // For local submission devices we must submit + sync explicitly so resolve
    // writes are visible before the main frame consumes resolved textures.
    if (!submission_device->is_main_rendering_device()) {
        owner._queue_submission(submission_device, false);
        gs_device_utils::safe_sync(submission_device);
    }

    if (resolve_uniform_set.is_valid() && submission_device->uniform_set_is_valid(resolve_uniform_set)) {
        submission_device->free(resolve_uniform_set);
    }
    if (param_uniform_set.is_valid() && submission_device->uniform_set_is_valid(param_uniform_set)) {
        submission_device->free(param_uniform_set);
    }
    if (lighting_uniform_set.is_valid() && submission_device->uniform_set_is_valid(lighting_uniform_set)) {
        submission_device->free(lighting_uniform_set);
    }
}

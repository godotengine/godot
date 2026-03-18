#include "shader_compilation_helper.h"

#include "core/error/error_macros.h"
#include "core/string/string_builder.h"
#include "../logger/gs_logger.h"
#include "quantization_config.h"
#include "tile_prefix_scan_utils.h"
#include "tile_renderer.h"
#include "../shaders/tile_binning.glsl.gen.h"
#include "../shaders/tile_prefix_scan.glsl.gen.h"
#include "../shaders/tile_rasterizer.glsl.gen.h"
#include "../shaders/tile_rasterizer_compute.glsl.gen.h"

namespace {
static constexpr int SHADER_ERROR_PREVIEW_LINES = 80;

static String _build_source_preview(const String &p_source, int p_line_limit) {
    if (p_source.is_empty() || p_line_limit <= 0) {
        return String("<no source available>");
    }

    StringBuilder preview_builder;
    int current_line = 1;
    int position = 0;
    const int length = p_source.length();

    while (position <= length && current_line <= p_line_limit) {
        int newline_pos = p_source.find("\n", position);
        bool has_newline = newline_pos != -1;
        int end_pos = has_newline ? newline_pos : length;

        String line = p_source.substr(position, end_pos - position);
        preview_builder.append(vformat("%4d | %s\n", current_line, line));

        if (!has_newline) {
            break;
        }

        position = newline_pos + 1;
        current_line++;
    }

    if (current_line > p_line_limit && position < length) {
        preview_builder.append("...\n");
    }

    return preview_builder.as_string();
}

static void _log_shader_failure(const String &p_shader_name, const String &p_stage, const String &p_error, const String &p_source) {
    String error_message = p_error.is_empty() ? String("<no error message>") : p_error;
    String preview = _build_source_preview(p_source, SHADER_ERROR_PREVIEW_LINES);
    GS_LOG_RENDERER_ERROR(vformat("[GaussianSplatting] %s shader '%s' failed to compile: %s\n--- Processed GLSL (first %d lines) ---\n%s",
            p_stage, p_shader_name, error_message, SHADER_ERROR_PREVIEW_LINES, preview));
}

static String _process_shader_source(const String &p_glsl_source, const Vector<String> &p_defines, String *r_error) {
    StringBuilder source_builder;
    const int source_length = p_glsl_source.length();

    int metadata_block_end = 0;
    int metadata_scan_pos = 0;
    while (metadata_scan_pos < source_length) {
        int newline_pos = p_glsl_source.find("\n", metadata_scan_pos);
        if (newline_pos == -1) {
            newline_pos = source_length;
        }
        String line = p_glsl_source.substr(metadata_scan_pos, newline_pos - metadata_scan_pos);
        String trimmed_line = line.strip_edges(true, false);
        if (!trimmed_line.begins_with("#[")) {
            break;
        }

        metadata_scan_pos = (newline_pos < source_length) ? newline_pos + 1 : source_length;
        metadata_block_end = metadata_scan_pos;
    }

    if (metadata_block_end > 0) {
        source_builder.append(p_glsl_source.substr(0, metadata_block_end));
    }

    int version_pos = -1;
    int check_pos = metadata_block_end;
    while (check_pos < source_length) {
        const char32_t c = p_glsl_source[check_pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            check_pos++;
            continue;
        }
        if (check_pos + 8 <= source_length && p_glsl_source.substr(check_pos, 8) == "#version") {
            version_pos = check_pos;
        }
        break;
    }

    const bool version_at_start = version_pos != -1;

    if (version_at_start) {
        if (metadata_block_end == 0 && version_pos > 0) {
            source_builder.append(p_glsl_source.substr(0, version_pos));
        }

        int newline_pos = p_glsl_source.find("\n", version_pos);
        String version_line;
        String remainder;

        if (newline_pos == -1) {
            version_line = p_glsl_source.substr(version_pos, source_length - version_pos);
            remainder = String();
        } else {
            version_line = p_glsl_source.substr(version_pos, newline_pos - version_pos);
            remainder = p_glsl_source.substr(newline_pos + 1, source_length - newline_pos - 1);
        }

        source_builder.append(version_line);
        source_builder.append("\n");

        for (int i = 0; i < p_defines.size(); i++) {
            const String &define = p_defines[i];
            if (define.is_empty()) {
                continue;
            }
            source_builder.append(define);
            if (!define.ends_with("\n")) {
                source_builder.append("\n");
            }
        }

        if (!remainder.is_empty()) {
            source_builder.append(remainder);
        }
    } else {
        for (int i = 0; i < p_defines.size(); i++) {
            const String &define = p_defines[i];
            if (define.is_empty()) {
                continue;
            }
            source_builder.append(define);
            if (!define.ends_with("\n")) {
                source_builder.append("\n");
            }
        }

        if (metadata_block_end > 0) {
            source_builder.append(p_glsl_source.substr(metadata_block_end, source_length - metadata_block_end));
        } else {
            source_builder.append(p_glsl_source);
        }
    }

    String final_source = source_builder.as_string();

    int first_directive_index = 0;
    const int final_length = final_source.length();
    while (first_directive_index < final_length) {
        const char32_t c = final_source[first_directive_index];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            first_directive_index++;
            continue;
        }
        if (c == '#' && first_directive_index + 1 < final_length && final_source[first_directive_index + 1] == '[') {
            int newline_pos = final_source.find("\n", first_directive_index);
            if (newline_pos == -1) {
                first_directive_index = final_length;
            } else {
                first_directive_index = newline_pos + 1;
            }
            continue;
        }
        if (c == '/' && first_directive_index + 1 < final_length) {
            const char32_t next = final_source[first_directive_index + 1];
            if (next == '/') {
                int newline_pos = final_source.find("\n", first_directive_index);
                if (newline_pos == -1) {
                    first_directive_index = final_length;
                } else {
                    first_directive_index = newline_pos + 1;
                }
                continue;
            } else if (next == '*') {
                first_directive_index += 2;
                while (first_directive_index < final_length) {
                    if (final_source[first_directive_index] == '*' && first_directive_index + 1 < final_length && final_source[first_directive_index + 1] == '/') {
                        first_directive_index += 2;
                        break;
                    }
                    first_directive_index++;
                }
                continue;
            }
        }
        break;
    }

    const bool first_directive_is_version = (first_directive_index + 8 <= final_length) &&
            final_source.substr(first_directive_index, 8) == "#version";
    if (!first_directive_is_version) {
        if (r_error) {
            *r_error = "Shader source must start with #version directive after metadata and comments.";
        }
        return String();
    }

    return final_source;
}
} // namespace

RID ShaderCompilationHelper::compile_shader_on_device(RenderingDevice *p_device, const String &p_glsl_source,
        const String &p_shader_name, const Vector<String> &p_defines, String *r_error, String *r_processed_source) {
    ERR_FAIL_NULL_V(p_device, RID());

    String processed_source = _process_shader_source(p_glsl_source, p_defines, r_error);
    ERR_FAIL_COND_V(processed_source.is_empty(), RID());
    if (r_processed_source) {
        *r_processed_source = processed_source;
    }

    String compile_error;
    Vector<uint8_t> spirv = p_device->shader_compile_spirv_from_source(RenderingDevice::SHADER_STAGE_COMPUTE, processed_source,
            RenderingDevice::SHADER_LANGUAGE_GLSL, &compile_error);
    if (r_error) {
        *r_error = compile_error;
    }
    if (spirv.is_empty()) {
        _log_shader_failure(p_shader_name, "compute", compile_error, processed_source);
        return RID();
    }

    RenderingDevice::ShaderStageSPIRVData stage_data;
    stage_data.shader_stage = RenderingDevice::SHADER_STAGE_COMPUTE;
    stage_data.spirv = spirv;

    Vector<RenderingDevice::ShaderStageSPIRVData> stages;
    stages.push_back(stage_data);

    RID shader = p_device->shader_create_from_spirv(stages, p_shader_name);
    return shader;
}

RID ShaderCompilationHelper::compile_graphics_shader_on_device(RenderingDevice *p_device, const String &p_vertex_source,
        const String &p_fragment_source, const String &p_shader_name, const Vector<String> &p_defines, String *r_error,
        String *r_processed_vertex, String *r_processed_fragment) {
    ERR_FAIL_NULL_V(p_device, RID());

    String vertex_error;
    String fragment_error;
    String processed_vertex = _process_shader_source(p_vertex_source, p_defines, &vertex_error);
    ERR_FAIL_COND_V_MSG(processed_vertex.is_empty(), RID(), "Vertex shader preprocessing failed: " + vertex_error);
    String processed_fragment = _process_shader_source(p_fragment_source, p_defines, &fragment_error);
    ERR_FAIL_COND_V_MSG(processed_fragment.is_empty(), RID(), "Fragment shader preprocessing failed: " + fragment_error);

    if (r_processed_vertex) {
        *r_processed_vertex = processed_vertex;
    }
    if (r_processed_fragment) {
        *r_processed_fragment = processed_fragment;
    }

    String vertex_compile_error;
    Vector<uint8_t> vertex_spirv = p_device->shader_compile_spirv_from_source(RenderingDevice::SHADER_STAGE_VERTEX, processed_vertex,
            RenderingDevice::SHADER_LANGUAGE_GLSL, &vertex_compile_error);
    if (vertex_spirv.is_empty()) {
        if (r_error) {
            *r_error = vertex_compile_error;
        }
        _log_shader_failure(p_shader_name, "vertex", vertex_compile_error, processed_vertex);
        return RID();
    }

    String fragment_compile_error;
    Vector<uint8_t> fragment_spirv = p_device->shader_compile_spirv_from_source(RenderingDevice::SHADER_STAGE_FRAGMENT, processed_fragment,
            RenderingDevice::SHADER_LANGUAGE_GLSL, &fragment_compile_error);
    if (fragment_spirv.is_empty()) {
        if (r_error) {
            *r_error = fragment_compile_error;
        }
        _log_shader_failure(p_shader_name, "fragment", fragment_compile_error, processed_fragment);
        return RID();
    }

    Vector<RenderingDevice::ShaderStageSPIRVData> stages;
    RenderingDevice::ShaderStageSPIRVData vertex_stage;
    vertex_stage.shader_stage = RenderingDevice::SHADER_STAGE_VERTEX;
    vertex_stage.spirv = vertex_spirv;
    stages.push_back(vertex_stage);

    RenderingDevice::ShaderStageSPIRVData fragment_stage;
    fragment_stage.shader_stage = RenderingDevice::SHADER_STAGE_FRAGMENT;
    fragment_stage.spirv = fragment_spirv;
    stages.push_back(fragment_stage);

    return p_device->shader_create_from_spirv(stages, p_shader_name);
}

ShaderCompilationManager::ShaderCompilationManager(TileRenderer &p_owner) :
		owner(p_owner) {
}

Error ShaderCompilationManager::compile_all_shaders(RenderingDevice *p_device_hint) {
	RenderingDevice *device = owner._acquire_submission_device();
	if (!device) {
		device = p_device_hint ? p_device_hint : owner._get_resource_device();
	}
	ERR_FAIL_NULL_V(device, ERR_UNCONFIGURED);

	owner._detect_subgroup_support(device);

	const bool want_compute_raster = owner.render_settings.allow_compute_raster;
	if (owner._check_pipeline_validity(device, want_compute_raster)) {
		return OK;
	}

	owner._initialize_shader_sources();
	RenderingDevice *shader_owner = owner.shader_resources.shader_device ? owner.shader_resources.shader_device : device;
	owner._free_existing_pipelines(shader_owner);
	owner._invalidate_descriptor_cache(); // Pipelines are changing; cached sets must be rebuilt.

	Error err = compile_binning_shaders(device);
	if (err != OK) {
		return err;
	}
	err = compile_prefix_shaders(device);
	if (err != OK) {
		return err;
	}
	err = compile_raster_shaders(device, want_compute_raster);
	if (err != OK) {
		return err;
	}

	// Legacy per-tile sort pipeline removed; global composite sort is the only supported path.
	owner.shader_resources.shader_device = device;
	owner.shader_resources.shader_device_instance = device->get_device_instance_id();
	owner.shader_resources.quantized_storage_enabled = g_quantization_config.per_chunk_quantization;
	owner.shader_resources.shader_defines_hash = owner._compute_shader_defines_hash();
	owner.config_state.effective_splat_capacity = TileRenderer::MAX_SPLATS_PER_TILE;

	return OK;
}

Error ShaderCompilationManager::compile_binning_shaders(RenderingDevice *p_device) {
	const int variant_index = TileShaderCompilation::DEFAULT_VARIANT_INDEX;
	ERR_FAIL_NULL_V(owner.shader_resources.tile_binning_shader_source.get(), ERR_UNCONFIGURED);

	Vector<String> binning_stage_sources;
	ERR_FAIL_COND_V(!_build_variant_stage_sources(*owner.shader_resources.tile_binning_shader_source, variant_index,
							 RD::SHADER_STAGE_COMPUTE, binning_stage_sources),
			ERR_CANT_CREATE);

	ShaderVariant binning_variant;
	binning_variant.label = "tile_binning";
	binning_variant.index = variant_index;
	binning_variant.defines = owner._build_binning_shader_defines();

	CompilationResult binning_result = _compile_compute_pipeline(p_device, binning_stage_sources[RD::SHADER_STAGE_COMPUTE],
			"tile_binning", binning_variant);
	owner.shader_resources.tile_binning_shader = binning_result.shader;
	ERR_FAIL_COND_V_MSG(!owner.shader_resources.tile_binning_shader.is_valid(), ERR_CANT_CREATE,
			"[TileRenderer] Failed to compile tile_binning.glsl: " + binning_result.error_message);
	owner.shader_resources.tile_binning_pipeline = binning_result.pipeline;
	ERR_FAIL_COND_V(!owner.shader_resources.tile_binning_pipeline.is_valid(), ERR_CANT_CREATE);

	ShaderVariant binning_count_variant;
	binning_count_variant.label = "tile_binning_count";
	binning_count_variant.index = variant_index;
	binning_count_variant.defines = _build_binning_count_defines();

	CompilationResult count_result = _compile_compute_pipeline(p_device, binning_stage_sources[RD::SHADER_STAGE_COMPUTE],
			"tile_binning_count", binning_count_variant);
	owner.shader_resources.tile_binning_count_shader = count_result.shader;
	ERR_FAIL_COND_V_MSG(!owner.shader_resources.tile_binning_count_shader.is_valid(), ERR_CANT_CREATE,
			"[TileRenderer] Failed to compile tile_binning.glsl (count pass): " + count_result.error_message);
	owner.shader_resources.tile_binning_count_pipeline = count_result.pipeline;
	ERR_FAIL_COND_V(!owner.shader_resources.tile_binning_count_pipeline.is_valid(), ERR_CANT_CREATE);

	return OK;
}

Error ShaderCompilationManager::compile_prefix_shaders(RenderingDevice *p_device) {
	const int variant_index = TileShaderCompilation::DEFAULT_VARIANT_INDEX;
	ERR_FAIL_NULL_V(owner.shader_resources.tile_prefix_shader_source.get(), ERR_UNCONFIGURED);

	Vector<String> prefix_stage_sources;
	ERR_FAIL_COND_V(!_build_variant_stage_sources(*owner.shader_resources.tile_prefix_shader_source, variant_index,
							 RD::SHADER_STAGE_COMPUTE, prefix_stage_sources),
			ERR_CANT_CREATE);

	auto compile_prefix = [&](int p_pass, int p_local_size, RID &r_shader, RID &r_pipeline, const char *p_label) -> bool {
		ShaderVariant prefix_variant;
		prefix_variant.label = p_label;
		prefix_variant.index = variant_index;
		prefix_variant.defines = _build_prefix_defines(p_pass, p_local_size);

		CompilationResult result = _compile_compute_pipeline(p_device, prefix_stage_sources[RD::SHADER_STAGE_COMPUTE],
				p_label, prefix_variant);
		if (!result.shader.is_valid()) {
			GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] Failed to compile %s: %s", p_label, result.error_message));
			return false;
		}
		if (!owner.shader_resources.tile_prefix_shader.is_valid()) {
			owner.shader_resources.tile_prefix_shader = result.shader;
		}
		r_shader = result.shader;
		r_pipeline = result.pipeline;
		if (!r_pipeline.is_valid()) {
			GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] Failed to create pipeline for %s", p_label));
			return false;
		}
		return true;
	};

	ERR_FAIL_COND_V(!compile_prefix(1, int(GaussianSplatting::kTilePrefixPassLocalSize),
							 owner.shader_resources.tile_prefix_shader,
							 owner.shader_resources.tile_prefix_pipeline_pass1,
							 "tile_prefix_pass1"),
			ERR_CANT_CREATE);
	ERR_FAIL_COND_V(!compile_prefix(2, int(GaussianSplatting::kTilePrefixPassLocalSize),
							 owner.shader_resources.tile_prefix_shader_pass2,
							 owner.shader_resources.tile_prefix_pipeline_pass2,
							 "tile_prefix_pass2"),
			ERR_CANT_CREATE);
	ERR_FAIL_COND_V(!compile_prefix(3, int(GaussianSplatting::kTilePrefixPassLocalSize),
							 owner.shader_resources.tile_prefix_shader_pass3,
							 owner.shader_resources.tile_prefix_pipeline_pass3,
							 "tile_prefix_pass3"),
			ERR_CANT_CREATE);

	return OK;
}

Error ShaderCompilationManager::compile_raster_shaders(RenderingDevice *p_device, bool p_want_compute_raster) {
	const int variant_index = TileShaderCompilation::DEFAULT_VARIANT_INDEX;
	ERR_FAIL_NULL_V(owner.shader_resources.tile_raster_shader_source.get(), ERR_UNCONFIGURED);
	ERR_FAIL_NULL_V(owner.shader_resources.tile_raster_compute_shader_source.get(), ERR_UNCONFIGURED);

	Vector<String> raster_defines = owner._build_raster_shader_defines();

	Vector<String> raster_stage_sources;
	ERR_FAIL_COND_V(!_build_variant_stage_sources(*owner.shader_resources.tile_raster_shader_source, variant_index,
							 RD::SHADER_STAGE_FRAGMENT, raster_stage_sources),
			ERR_CANT_CREATE);

	ShaderVariant raster_variant;
	raster_variant.label = "tile_rasterizer";
	raster_variant.index = variant_index;
	raster_variant.defines = raster_defines;

	// Optional compile logging disabled for performance.
	String raster_error;
	owner.shader_resources.tile_raster_shader = _compile_graphics_shader(p_device,
			raster_stage_sources[RD::SHADER_STAGE_VERTEX],
			raster_stage_sources[RD::SHADER_STAGE_FRAGMENT],
			"tile_rasterizer", raster_variant, &raster_error);
	ERR_FAIL_COND_V_MSG(!owner.shader_resources.tile_raster_shader.is_valid(), ERR_CANT_CREATE,
			"[TileRenderer] Failed to compile tile_rasterizer.glsl: " + raster_error);
	// Note: shader_resources.tile_raster_pipeline is created lazily in render() because it needs the framebuffer format.

	if (!p_want_compute_raster) {
		owner.shader_resources.tile_raster_compute_shader = RID();
		owner.shader_resources.tile_raster_compute_pipeline = RID();
		return OK;
	}

	// Runtime check: verify shared memory requirement against device limit.
	// tile_rasterizer_compute.glsl allocates:
	//   SPLATS_PER_TILE * (sizeof(uint) + sizeof(ProjectedGaussian)) + 4 * sizeof(uint)
	// ProjectedGaussian = 9 uints (36 bytes) unpacked, 8 uints (32 bytes) packed.
	{
		const uint32_t splats_per_tile = TileRenderer::MAX_SPLATS_PER_TILE;
		const uint32_t projected_gaussian_bytes = 9 * sizeof(uint32_t); // Conservative: unpacked layout
		const uint32_t shared_bytes_required = splats_per_tile * (sizeof(uint32_t) + projected_gaussian_bytes) + 4 * sizeof(uint32_t);
		const uint64_t device_shared_memory = p_device->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE);
		if (device_shared_memory > 0 && shared_bytes_required > device_shared_memory) {
			WARN_PRINT(vformat("[TileRenderer] Compute rasterizer shared memory requirement (%d bytes) "
					"exceeds device maxComputeSharedMemorySize (%d bytes). "
					"SPLATS_PER_TILE=%d. Compute rasterizer may fail on this GPU.",
					shared_bytes_required, (uint32_t)device_shared_memory, splats_per_tile));
		}
	}

	Vector<String> compute_stage_sources;
	if (!_build_variant_stage_sources(*owner.shader_resources.tile_raster_compute_shader_source, variant_index,
				RD::SHADER_STAGE_COMPUTE, compute_stage_sources)) {
		owner.shader_resources.tile_raster_compute_shader = RID();
		owner.shader_resources.tile_raster_compute_pipeline = RID();
		return OK;
	}

	ShaderVariant raster_compute_variant;
	raster_compute_variant.label = "tile_rasterizer_compute";
	raster_compute_variant.index = variant_index;
	raster_compute_variant.defines = raster_defines;
	raster_compute_variant.defines.push_back("#define GS_TILE_RASTER_COMPUTE 1\n");

	CompilationResult compute_result = _compile_compute_pipeline(p_device,
			compute_stage_sources[RD::SHADER_STAGE_COMPUTE],
			"tile_rasterizer_compute", raster_compute_variant);
	owner.shader_resources.tile_raster_compute_shader = compute_result.shader;
	if (!owner.shader_resources.tile_raster_compute_shader.is_valid()) {
		_log_shader_compilation_failure("tile_rasterizer_compute", variant_index, raster_compute_variant.defines,
				compute_stage_sources[RD::SHADER_STAGE_COMPUTE], compute_result.error_message, p_device);
		owner.shader_resources.tile_raster_compute_pipeline = RID();
		return OK;
	}

	owner.shader_resources.tile_raster_compute_pipeline = compute_result.pipeline;
	if (!owner.shader_resources.tile_raster_compute_pipeline.is_valid()) {
		_log_pipeline_creation_failure("tile_rasterizer_compute", variant_index, raster_compute_variant.defines, p_device);
	}

	return OK;
}

Vector<String> ShaderCompilationManager::_build_prefix_defines(int p_pass, int p_local_size) const {
	Vector<String> defines = owner._build_common_shader_defines(true);
	defines.push_back("#define GS_TILE_GLOBAL_SORT 1\n");
	defines.push_back(vformat("#define GS_TILE_PREFIX_PASS_%d 1\n", p_pass));
	defines.push_back(vformat("#define GS_PREFIX_LOCAL_SIZE %d\n", p_local_size));
	defines.push_back(vformat("#define GS_TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP %d\n",
			int(GaussianSplatting::TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP)));
	defines.push_back(vformat("#define GS_TILE_PREFIX_PASS2_OP_EXCLUSIVE_SHIFT %d\n",
			int(GaussianSplatting::TILE_PREFIX_PASS2_OP_EXCLUSIVE_SHIFT)));
	defines.push_back(vformat("#define GS_TILE_PREFIX_PASS2_OP_COPY %d\n",
			int(GaussianSplatting::TILE_PREFIX_PASS2_OP_COPY)));
	defines.push_back(vformat("#define GS_TILE_PREFIX_PASS2_SOURCE_WG_SUMS %d\n",
			int(GaussianSplatting::TILE_PREFIX_PASS2_SOURCE_WG_SUMS)));
	defines.push_back(vformat("#define GS_TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS %d\n",
			int(GaussianSplatting::TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS)));
	return defines;
}

Vector<String> ShaderCompilationManager::_build_binning_count_defines() const {
	Vector<String> defines = owner._build_common_shader_defines(true);
	defines.push_back(vformat("#define GS_TILE_SPLAT_CAPACITY %d\n", TileRenderer::MAX_SPLATS_PER_TILE));
	defines.push_back("#define GS_TILE_GLOBAL_SORT 1\n");
	defines.push_back("#define GS_TILE_GLOBAL_SORT_COUNT_PASS 1\n");
	return defines;
}

ShaderCompilationManager::CompilationResult ShaderCompilationManager::_compile_compute_pipeline(RenderingDevice *p_device,
		const String &p_glsl_source, const String &p_shader_name, const ShaderVariant &p_variant) const {
	CompilationResult result;
	ERR_FAIL_NULL_V(p_device, result);

	result.shader = ShaderCompilationHelper::compile_shader_on_device(p_device, p_glsl_source, p_shader_name, p_variant.defines,
			&result.error_message);
	if (result.shader.is_valid()) {
		result.pipeline = p_device->compute_pipeline_create(result.shader);
	}

	return result;
}

RID ShaderCompilationManager::_compile_graphics_shader(RenderingDevice *p_device, const String &p_vertex_source,
		const String &p_fragment_source, const String &p_shader_name, const ShaderVariant &p_variant, String *r_error) const {
	return ShaderCompilationHelper::compile_graphics_shader_on_device(p_device, p_vertex_source, p_fragment_source,
			p_shader_name, p_variant.defines, r_error);
}

String ShaderCompilationManager::_get_first_n_lines(const String &p_source, int p_line_count) {
	if (p_line_count <= 0 || p_source.is_empty()) {
		return String();
	}

	String result;
	int remaining = p_line_count;
	int start = 0;
	int length = p_source.length();

	for (int i = 0; i <= length && remaining > 0; i++) {
		bool end_of_line = (i == length) || (p_source[i] == '\n');
		if (!end_of_line) {
			continue;
		}

		String line = p_source.substr(start, i - start);
		result += line;
		remaining--;
		if (remaining > 0) {
			result += "\n";
		}
		start = i + 1;
	}

	return result;
}

String ShaderCompilationManager::_format_shader_define_list(const Vector<String> &p_defines) {
	if (p_defines.is_empty()) {
		return String("<none>");
	}

	String defines_text;
	for (int i = 0; i < p_defines.size(); i++) {
		const String &define = p_defines[i];
		if (define.is_empty()) {
			continue;
		}
		defines_text += define;
		if (!define.ends_with("\n")) {
			defines_text += "\n";
		}
	}

	if (defines_text.is_empty()) {
		return String("<none>");
	}

	if (defines_text.ends_with("\n")) {
		defines_text = defines_text.substr(0, defines_text.length() - 1);
	}

	return defines_text;
}

void ShaderCompilationManager::_log_shader_compilation_failure(const String &p_shader_name, int p_variant, const Vector<String> &p_defines,
		const String &p_source_preview, const String &p_error_message, RenderingDevice *p_device) const {
	String device_label = owner._is_main_rendering_device(p_device) ? "singleton" : "local";
	String snippet = _get_first_n_lines(p_source_preview, SHADER_LOG_PREVIEW_LINES);
	if (snippet.is_empty()) {
		snippet = "<no source preview>";
	}

	String defines_text = _format_shader_define_list(p_defines);
	String error_text = p_error_message.is_empty() ? String("<no error message provided>") : p_error_message;

	GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] Shader compilation failed for %s variant %d on %s rendering device.\n"
					  "Defines:\n%s\n"
					  "First 20 lines:\n%s\n"
					  "SPIR-V error: %s",
			p_shader_name, p_variant, device_label, defines_text, snippet, error_text));
}

void ShaderCompilationManager::_log_pipeline_creation_failure(const String &p_shader_name, int p_variant, const Vector<String> &p_defines,
		RenderingDevice *p_device) const {
	String device_label = owner._is_main_rendering_device(p_device) ? "singleton" : "local";
	String defines_text = _format_shader_define_list(p_defines);

	GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] Compute pipeline creation failed for %s variant %d on %s rendering device.\n"
					  "Defines:\n%s",
			p_shader_name, p_variant, device_label, defines_text));
}

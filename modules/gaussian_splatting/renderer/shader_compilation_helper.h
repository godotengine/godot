#ifndef SHADER_COMPILATION_HELPER_H
#define SHADER_COMPILATION_HELPER_H

#include "core/error/error_list.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"

#include "shader_compilation_types.h"

class TileRenderer;

class ShaderCompilationHelper {
public:
    static RID compile_shader_on_device(RenderingDevice *p_device, const String &p_glsl_source,
            const String &p_shader_name, const Vector<String> &p_defines = Vector<String>(), String *r_error = nullptr,
            String *r_processed_source = nullptr);

    static RID compile_graphics_shader_on_device(RenderingDevice *p_device, const String &p_vertex_source,
            const String &p_fragment_source, const String &p_shader_name, const Vector<String> &p_defines = Vector<String>(),
            String *r_error = nullptr, String *r_processed_vertex = nullptr, String *r_processed_fragment = nullptr);
};

class ShaderCompilationManager {
public:
	explicit ShaderCompilationManager(TileRenderer &p_owner);

	Error compile_all_shaders(RenderingDevice *p_device_hint);
	Error compile_binning_shaders(RenderingDevice *p_device);
	Error compile_prefix_shaders(RenderingDevice *p_device);
	Error compile_raster_shaders(RenderingDevice *p_device, bool p_want_compute_raster);

private:
	using ShaderVariant = TileShaderCompilation::ShaderVariant;
	using CompilationResult = TileShaderCompilation::CompilationResult;

	static constexpr int SHADER_LOG_PREVIEW_LINES = 20;

	TileRenderer &owner;

	Vector<String> _build_prefix_defines(int p_pass, int p_local_size) const;
	Vector<String> _build_binning_count_defines() const;
	CompilationResult _compile_compute_pipeline(RenderingDevice *p_device, const String &p_glsl_source,
			const String &p_shader_name, const ShaderVariant &p_variant) const;
	RID _compile_graphics_shader(RenderingDevice *p_device, const String &p_vertex_source,
			const String &p_fragment_source, const String &p_shader_name, const ShaderVariant &p_variant,
			String *r_error) const;
	static String _get_first_n_lines(const String &p_source, int p_line_count);
	static String _format_shader_define_list(const Vector<String> &p_defines);
	void _log_shader_compilation_failure(const String &p_shader_name, int p_variant, const Vector<String> &p_defines,
			const String &p_source_preview, const String &p_error_message, RenderingDevice *p_device) const;
	void _log_pipeline_creation_failure(const String &p_shader_name, int p_variant, const Vector<String> &p_defines,
			RenderingDevice *p_device) const;

	template <typename TShaderSource>
	static bool _build_variant_stage_sources(TShaderSource &p_shader_source, int p_variant, int p_required_stage,
			Vector<String> &r_stage_sources) {
		RID version = p_shader_source.version_create();
		if (!version.is_valid()) {
			return false;
		}
		r_stage_sources = p_shader_source.version_build_variant_stage_sources(version, p_variant);
		p_shader_source.version_free(version);
		if (r_stage_sources.size() <= p_required_stage) {
			return false;
		}
		return true;
	}
};

#endif // SHADER_COMPILATION_HELPER_H

/**************************************************************************/
/*  rendering_shader_container_metal.mm                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#import "rendering_shader_container_metal.h"

#import "metal_utils.h"

#import "core/io/file_access.h"
#import "core/io/marshalls.h"
#import "core/templates/fixed_vector.h"
#import "servers/rendering/rendering_device.h"

#include "thirdparty/spirv-reflect/spirv_reflect.h"

#import <Metal/Metal.h>
#import <spirv.hpp>
#import <spirv_msl.hpp>
#import <spirv_parser.hpp>

void RenderingShaderContainerMetal::_initialize_toolchain_properties() {
	if (compiler_props.is_valid()) {
		return;
	}

	String sdk;
	switch (device_profile->platform) {
		case MetalDeviceProfile::Platform::macOS:
			sdk = "macosx";
			break;
		case MetalDeviceProfile::Platform::iOS:
			sdk = "iphoneos";
			break;
		case MetalDeviceProfile::Platform::visionOS:
			sdk = "xros";
			break;
	}

	Vector<String> parts{ "echo", R"("")", "|", "/usr/bin/xcrun", "-sdk", sdk, "metal", "-E", "-dM", "-x", "metal" };

	switch (device_profile->platform) {
		case MetalDeviceProfile::Platform::macOS: {
			parts.push_back("-mtargetos=macos" + device_profile->min_os_version.to_compiler_os_version());
			break;
		}
		case MetalDeviceProfile::Platform::iOS: {
			parts.push_back("-mtargetos=ios" + device_profile->min_os_version.to_compiler_os_version());
			break;
		}
		case MetalDeviceProfile::Platform::visionOS: {
			parts.push_back("-mtargetos=xros" + device_profile->min_os_version.to_compiler_os_version());
			break;
		}
	}

	parts.append_array({ "-", "|", "grep", "-E", R"(\"__METAL_VERSION__|__ENVIRONMENT_OS\")" });

	List<String> args = { "-c", String(" ").join(parts) };

	String r_pipe;
	int exit_code;
	Error err = OS::get_singleton()->execute("sh", args, &r_pipe, &exit_code, true);
	ERR_FAIL_COND_MSG(err != OK, "Failed to determine Metal toolchain properties");

	// Parse the lines, which are in the form:
	//
	// #define VARNAME VALUE
	Vector<String> lines = r_pipe.split("\n", false);
	for (String &line : lines) {
		Vector<String> name_val = line.trim_prefix("#define ").split(" ");
		if (name_val.size() != 2) {
			continue;
		}
		if (name_val[0] == "__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__") {
			compiler_props.os_version_min_required = MinOsVersion((uint32_t)name_val[1].to_int());
		} else if (name_val[0] == "__METAL_VERSION__") {
			uint32_t ver = (uint32_t)name_val[1].to_int();
			uint32_t maj = ver / 100;
			uint32_t min = (ver % 100) / 10;
			compiler_props.metal_version = make_msl_version(maj, min);
		}

		if (compiler_props.is_valid()) {
			break;
		}
	}
}

Error RenderingShaderContainerMetal::compile_metal_source(const char *p_source, const StageData &p_stage_data, Vector<uint8_t> &r_binary_data) {
	String name(shader_name.ptr());
	if (name.contains_char(':')) {
		name = name.replace_char(':', '_');
	}
	Error r_error;
	Ref<FileAccess> source_file = FileAccess::create_temp(FileAccess::ModeFlags::READ_WRITE,
			name + "_" + itos(p_stage_data.hash.short_sha()),
			"metal", false, &r_error);
	ERR_FAIL_COND_V_MSG(r_error != OK, r_error, "Unable to create temporary source file.");
	if (!source_file->store_buffer((const uint8_t *)p_source, strlen(p_source))) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Unable to write temporary source file");
	}
	source_file->flush();
	Ref<FileAccess> result_file = FileAccess::create_temp(FileAccess::ModeFlags::READ_WRITE,
			name + "_" + itos(p_stage_data.hash.short_sha()),
			"metallib", false, &r_error);

	ERR_FAIL_COND_V_MSG(r_error != OK, r_error, "Unable to create temporary target file");

	String sdk;
	switch (device_profile->platform) {
		case MetalDeviceProfile::Platform::macOS:
			sdk = "macosx";
			break;
		case MetalDeviceProfile::Platform::iOS:
			sdk = "iphoneos";
			break;
		case MetalDeviceProfile::Platform::visionOS:
			sdk = "xros";
			break;
	}

	// Build the .metallib binary.
	{
		List<String> args{ "-sdk", sdk, "metal", "-O3" };

		// Compile metal shaders for the minimum supported target instead of the host machine.
		switch (device_profile->platform) {
			case MetalDeviceProfile::Platform::macOS: {
				args.push_back("-mtargetos=macos" + device_profile->min_os_version.to_compiler_os_version());
				break;
			}
			case MetalDeviceProfile::Platform::iOS: {
				args.push_back("-mtargetos=ios" + device_profile->min_os_version.to_compiler_os_version());
				break;
			}
			case MetalDeviceProfile::Platform::visionOS: {
				args.push_back("-mtargetos=xros" + device_profile->min_os_version.to_compiler_os_version());
				break;
			}
		}

		if (p_stage_data.is_position_invariant) {
			args.push_back("-fpreserve-invariance");
		}
		args.push_back("-fmetal-math-mode=fast");
		args.push_back(source_file->get_path_absolute());
		args.push_back("-o");
		args.push_back(result_file->get_path_absolute());
		String r_pipe;
		int exit_code;
		Error err = OS::get_singleton()->execute("/usr/bin/xcrun", args, &r_pipe, &exit_code, true);
		if (!r_pipe.is_empty()) {
			print_line(r_pipe);
		}
		if (err != OK) {
			ERR_PRINT(vformat("Metal compiler returned error code: %d", err));
		}

		if (exit_code != 0) {
			ERR_PRINT(vformat("Metal compiler exited with error code: %d", exit_code));
		}
		int len = result_file->get_length();
		ERR_FAIL_COND_V_MSG(len == 0, ERR_CANT_CREATE, "Metal compiler created empty library");
	}

	// Strip the source from the binary.
	{
		List<String> args{ "-sdk", sdk, "metal-dsymutil", "--remove-source", result_file->get_path_absolute() };
		String r_pipe;
		int exit_code;
		Error err = OS::get_singleton()->execute("/usr/bin/xcrun", args, &r_pipe, &exit_code, true);
		if (!r_pipe.is_empty()) {
			print_line(r_pipe);
		}
		if (err != OK) {
			ERR_PRINT(vformat("metal-dsymutil tool returned error code: %d", err));
		}

		if (exit_code != 0) {
			ERR_PRINT(vformat("metal-dsymutil Compiler exited with error code: %d", exit_code));
		}
		int len = result_file->get_length();
		ERR_FAIL_COND_V_MSG(len == 0, ERR_CANT_CREATE, "metal-dsymutil tool created empty library");
	}

	r_binary_data = result_file->get_buffer(result_file->get_length());

	return OK;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability"

static spv::ExecutionModel SHADER_STAGE_REMAP[RDD::SHADER_STAGE_MAX] = {
	[RDD::SHADER_STAGE_VERTEX] = spv::ExecutionModelVertex,
	[RDD::SHADER_STAGE_FRAGMENT] = spv::ExecutionModelFragment,
	[RDD::SHADER_STAGE_TESSELATION_CONTROL] = spv::ExecutionModelTessellationControl,
	[RDD::SHADER_STAGE_TESSELATION_EVALUATION] = spv::ExecutionModelTessellationEvaluation,
	[RDD::SHADER_STAGE_COMPUTE] = spv::ExecutionModelGLCompute,
};

spv::ExecutionModel get_stage(uint32_t p_stages_mask, RDD::ShaderStage p_stage) {
	if (p_stages_mask & (1 << p_stage)) {
		return SHADER_STAGE_REMAP[p_stage];
	}
	return spv::ExecutionModel::ExecutionModelMax;
}

spv::ExecutionModel map_stage(RDD::ShaderStage p_stage) {
	return SHADER_STAGE_REMAP[p_stage];
}

bool RenderingShaderContainerMetal::_set_code_from_spirv(const ReflectShader &p_shader) {
	using namespace spirv_cross;
	using spirv_cross::CompilerMSL;
	using spirv_cross::Resource;

	const LocalVector<ReflectShaderStage> &p_spirv = p_shader.shader_stages;

	if (export_mode) {
		_initialize_toolchain_properties();
	}

	// initialize Metal-specific reflection data
	shaders.resize(p_spirv.size());
	mtl_shaders.resize(p_spirv.size());
	mtl_reflection_binding_set_uniforms_data.resize(reflection_binding_set_uniforms_data.size());

	mtl_reflection_data.set_needs_view_mask_buffer(reflection_data.has_multiview);
	mtl_reflection_data.profile = *device_profile;

	CompilerMSL::Options msl_options{};

	// Determine Metal language version.
	uint32_t msl_version = 0;
	{
		if (export_mode && compiler_props.is_valid()) {
			// Use the properties determined by the toolchain and minimum OS version.
			msl_version = compiler_props.metal_version;
			mtl_reflection_data.os_min_version = compiler_props.os_version_min_required;
		} else {
			msl_version = device_profile->features.msl_version;
			mtl_reflection_data.os_min_version = MinOsVersion();
		}
		uint32_t msl_ver_maj = 0;
		uint32_t msl_ver_min = 0;
		parse_msl_version(msl_version, msl_ver_maj, msl_ver_min);
		msl_options.set_msl_version(msl_ver_maj, msl_ver_min);
		mtl_reflection_data.msl_version = msl_version;
	}

	msl_options.platform = device_profile->platform == MetalDeviceProfile::Platform::macOS ? CompilerMSL::Options::macOS : CompilerMSL::Options::iOS;

	if (device_profile->platform == MetalDeviceProfile::Platform::iOS) {
		msl_options.ios_use_simdgroup_functions = device_profile->features.simdPermute;
		msl_options.ios_support_base_vertex_instance = true;
	}

	if (device_profile->features.use_argument_buffers) {
		msl_options.argument_buffers_tier = CompilerMSL::Options::ArgumentBuffersTier::Tier2;
		msl_options.argument_buffers = true;
		mtl_reflection_data.set_uses_argument_buffers(true);
	} else {
		msl_options.argument_buffers_tier = CompilerMSL::Options::ArgumentBuffersTier::Tier1;
		// Tier 1 argument buffers don't support writable textures, so we disable them completely.
		msl_options.argument_buffers = false;
		mtl_reflection_data.set_uses_argument_buffers(false);
	}
	msl_options.force_active_argument_buffer_resources = true;
	msl_options.pad_argument_buffer_resources = true;
	msl_options.texture_buffer_native = true; // Enable texture buffer support.
	msl_options.use_framebuffer_fetch_subpasses = false;
	msl_options.pad_fragment_output_components = true;
	msl_options.r32ui_alignment_constant_id = R32UI_ALIGNMENT_CONSTANT_ID;
	msl_options.agx_manual_cube_grad_fixup = true;
	if (reflection_data.has_multiview) {
		msl_options.multiview = true;
		msl_options.multiview_layered_rendering = true;
		msl_options.view_mask_buffer_index = VIEW_MASK_BUFFER_INDEX;
	}
	if (msl_version >= MSL_VERSION_32) {
		// All 3.2+ versions support device coherence, so we can disable texture fences.
		msl_options.readwrite_texture_fences = false;
	}

	CompilerGLSL::Options options{};
	options.vertex.flip_vert_y = true;
#if DEV_ENABLED
	options.emit_line_directives = true;
#endif

	// Assign MSL bindings for all the descriptor sets.
	typedef std::pair<MSLResourceBinding, uint32_t> MSLBindingInfo;
	LocalVector<MSLBindingInfo> spirv_bindings;
	MSLResourceBinding push_constant_resource_binding;
	{
		enum IndexType {
			Texture,
			Buffer,
			Sampler,
			Max,
		};

		uint32_t dset_count = p_shader.uniform_sets.size();
		uint32_t size = reflection_binding_set_uniforms_data.size();
		spirv_bindings.resize(size);

		uint32_t indices[IndexType::Max] = { 0 };
		auto next_index = [&indices](IndexType p_t, uint32_t p_stride) -> uint32_t {
			uint32_t v = indices[p_t];
			indices[p_t] += p_stride;
			return v;
		};

		uint32_t idx_dset = 0;
		MSLBindingInfo *iter = spirv_bindings.ptr();
		UniformData *found = mtl_reflection_binding_set_uniforms_data.ptrw();
		UniformData::IndexType shader_index_type = msl_options.argument_buffers ? UniformData::IndexType::ARG : UniformData::IndexType::SLOT;

		for (const ReflectDescriptorSet &dset : p_shader.uniform_sets) {
			// Reset the index count for each descriptor set, as this is an index in to the argument table.
			uint32_t next_arg_buffer_index = 0;
			auto next_arg_index = [&next_arg_buffer_index](uint32_t p_stride) -> uint32_t {
				uint32_t v = next_arg_buffer_index;
				next_arg_buffer_index += p_stride;
				return v;
			};

			for (const ReflectUniform &uniform : dset) {
				const SpvReflectDescriptorBinding &binding = uniform.get_spv_reflect();

				found->active_stages = uniform.stages;

				RD::UniformType type = RD::UniformType(uniform.type);
				uint32_t binding_stride = 1; // If this is an array, stride will be the length of the array.
				if (uniform.length > 1) {
					switch (type) {
						case RDC::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC:
						case RDC::UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC:
						case RDC::UNIFORM_TYPE_UNIFORM_BUFFER:
						case RDC::UNIFORM_TYPE_STORAGE_BUFFER:
							// Buffers's length is its size, in bytes, so there is no stride.
							break;
						default: {
							binding_stride = uniform.length;
							found->array_length = uniform.length;
						} break;
					}
				}

				// Determine access type.
				switch (binding.descriptor_type) {
					case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
						if (!(binding.decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE)) {
							if (!(binding.decoration_flags & SPV_REFLECT_DECORATION_NON_READABLE)) {
								found->access = MTLBindingAccessReadWrite;
							} else {
								found->access = MTLBindingAccessWriteOnly;
							}
						}
					} break;
					case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
					case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
						if (!(binding.decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE) && !(binding.block.decoration_flags & SPV_REFLECT_DECORATION_NON_WRITABLE)) {
							if (!(binding.decoration_flags & SPV_REFLECT_DECORATION_NON_READABLE) && !(binding.block.decoration_flags & SPV_REFLECT_DECORATION_NON_READABLE)) {
								found->access = MTLBindingAccessReadWrite;
							} else {
								found->access = MTLBindingAccessWriteOnly;
							}
						}
					} break;
					default:
						break;
				}

				switch (found->access) {
					case MTLBindingAccessReadOnly:
						found->usage = MTLResourceUsageRead;
						break;
					case MTLBindingAccessWriteOnly:
						found->usage = MTLResourceUsageWrite;
						break;
					case MTLBindingAccessReadWrite:
						found->usage = MTLResourceUsageRead | MTLResourceUsageWrite;
						break;
				}

				iter->second = uniform.stages;
				MSLResourceBinding &rb = iter->first;
				rb.desc_set = idx_dset;
				rb.binding = uniform.binding;
				rb.count = binding_stride;

				switch (type) {
					case RDC::UNIFORM_TYPE_SAMPLER: {
						found->data_type = MTLDataTypeSampler;
						found->get_indexes(UniformData::IndexType::SLOT).sampler = next_index(Sampler, binding_stride);
						found->get_indexes(UniformData::IndexType::ARG).sampler = next_arg_index(binding_stride);

						rb.basetype = SPIRType::BaseType::Sampler;

					} break;
					case RDC::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE:
					case RDC::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER: {
						found->data_type = MTLDataTypeTexture;
						found->get_indexes(UniformData::IndexType::SLOT).texture = next_index(Texture, binding_stride);
						found->get_indexes(UniformData::IndexType::SLOT).sampler = next_index(Sampler, binding_stride);
						found->get_indexes(UniformData::IndexType::ARG).texture = next_arg_index(binding_stride);
						found->get_indexes(UniformData::IndexType::ARG).sampler = next_arg_index(binding_stride);
						rb.basetype = SPIRType::BaseType::SampledImage;
					} break;
					case RDC::UNIFORM_TYPE_TEXTURE:
					case RDC::UNIFORM_TYPE_IMAGE:
					case RDC::UNIFORM_TYPE_TEXTURE_BUFFER: {
						found->data_type = MTLDataTypeTexture;
						found->get_indexes(UniformData::IndexType::SLOT).texture = next_index(Texture, binding_stride);
						found->get_indexes(UniformData::IndexType::ARG).texture = next_arg_index(binding_stride);
						rb.basetype = SPIRType::BaseType::Image;
					} break;
					case RDC::UNIFORM_TYPE_IMAGE_BUFFER:
						CRASH_NOW_MSG("Unimplemented!"); // TODO.
						break;
					case RDC::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC:
					case RDC::UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC:
					case RDC::UNIFORM_TYPE_UNIFORM_BUFFER:
					case RDC::UNIFORM_TYPE_STORAGE_BUFFER: {
						found->data_type = MTLDataTypePointer;
						found->get_indexes(UniformData::IndexType::SLOT).buffer = next_index(Buffer, binding_stride);
						found->get_indexes(UniformData::IndexType::ARG).buffer = next_arg_index(binding_stride);
						rb.basetype = SPIRType::BaseType::Void;
					} break;
					case RDC::UNIFORM_TYPE_INPUT_ATTACHMENT: {
						found->data_type = MTLDataTypeTexture;
						found->get_indexes(UniformData::IndexType::SLOT).texture = next_index(Texture, binding_stride);
						found->get_indexes(UniformData::IndexType::ARG).texture = next_arg_index(binding_stride);
						rb.basetype = SPIRType::BaseType::Image;
					} break;
					case RDC::UNIFORM_TYPE_MAX:
					default:
						CRASH_NOW_MSG("Unreachable");
				}

				// Specify the MSL resource bindings based on how the binding mode used by the shader.
				rb.msl_buffer = found->get_indexes(shader_index_type).buffer;
				rb.msl_texture = found->get_indexes(shader_index_type).texture;
				rb.msl_sampler = found->get_indexes(shader_index_type).sampler;

				if (found->data_type == MTLDataTypeTexture) {
					const SpvReflectImageTraits &image = uniform.get_spv_reflect().image;

					switch (image.dim) {
						case SpvDim1D: {
							if (image.arrayed) {
								found->texture_type = MTLTextureType1DArray;
							} else {
								found->texture_type = MTLTextureType1D;
							}
						} break;
						case SpvDimSubpassData:
						case SpvDim2D: {
							if (image.arrayed && image.ms) {
								found->texture_type = MTLTextureType2DMultisampleArray;
							} else if (image.arrayed) {
								found->texture_type = MTLTextureType2DArray;
							} else if (image.ms) {
								found->texture_type = MTLTextureType2DMultisample;
							} else {
								found->texture_type = MTLTextureType2D;
							}
						} break;
						case SpvDim3D: {
							found->texture_type = MTLTextureType3D;
						} break;
						case SpvDimCube: {
							if (image.arrayed) {
								found->texture_type = MTLTextureTypeCubeArray;
							} else {
								found->texture_type = MTLTextureTypeCube;
							}
						} break;
						case SpvDimRect: {
							// Ignored.
						} break;
						case SpvDimBuffer: {
							found->texture_type = MTLTextureTypeTextureBuffer;
						} break;
						case SpvDimTileImageDataEXT: {
							// Godot does not use this extension.
							// See: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_shader_tile_image.html
						} break;
						case SpvDimMax: {
							// Add all enumerations to silence the compiler warning
							// and generate future warnings, should a new one be added.
						} break;
					}
				}

				iter++;
				found++;
			}
			idx_dset++;
		}

		if (reflection_data.push_constant_size > 0) {
			push_constant_resource_binding.desc_set = ResourceBindingPushConstantDescriptorSet;
			push_constant_resource_binding.basetype = SPIRType::BaseType::Void;
			if (msl_options.argument_buffers) {
				push_constant_resource_binding.msl_buffer = dset_count;
			} else {
				push_constant_resource_binding.msl_buffer = next_index(Buffer, 1);
			}
			mtl_reflection_data.push_constant_binding = push_constant_resource_binding.msl_buffer;
		}
	}

	for (uint32_t i = 0; i < p_spirv.size(); i++) {
		StageData &stage_data = mtl_shaders.write[i];
		const ReflectShaderStage &v = p_spirv[i];
		RD::ShaderStage stage = v.shader_stage;
		Span<uint32_t> spirv = v.spirv();
		Parser parser(spirv.ptr(), spirv.size());
		try {
			parser.parse();
		} catch (CompilerError &e) {
			ERR_FAIL_V_MSG(false, "Failed to parse IR at stage " + String(RD::SHADER_STAGE_NAMES[stage]) + ": " + e.what());
		}

		CompilerMSL compiler(std::move(parser.get_parsed_ir()));
		compiler.set_msl_options(msl_options);
		compiler.set_common_options(options);

		spv::ExecutionModel execution_model = map_stage(stage);
		for (uint32_t jj = 0; jj < spirv_bindings.size(); jj++) {
			MSLResourceBinding &rb = spirv_bindings.ptr()[jj].first;
			rb.stage = execution_model;
			compiler.add_msl_resource_binding(rb);
		}

		if (push_constant_resource_binding.desc_set == ResourceBindingPushConstantDescriptorSet) {
			push_constant_resource_binding.stage = execution_model;
			compiler.add_msl_resource_binding(push_constant_resource_binding);
		}

		std::unordered_set<VariableID> active = compiler.get_active_interface_variables();
		ShaderResources resources = compiler.get_shader_resources();

		std::string source;
		try {
			source = compiler.compile();
		} catch (CompilerError &e) {
			ERR_FAIL_V_MSG(false, "Failed to compile stage " + String(RD::SHADER_STAGE_NAMES[stage]) + ": " + e.what());
		}

		ERR_FAIL_COND_V_MSG(compiler.get_entry_points_and_stages().size() != 1, false, "Expected a single entry point and stage.");

		SmallVector<EntryPoint> entry_pts_stages = compiler.get_entry_points_and_stages();
		EntryPoint &entry_point_stage = entry_pts_stages.front();
		SPIREntryPoint &entry_point = compiler.get_entry_point(entry_point_stage.name, entry_point_stage.execution_model);

		for (auto ext : compiler.get_declared_extensions()) {
			if (ext == "SPV_KHR_non_semantic_info" || ext == "SPV_KHR_printf") {
				mtl_reflection_data.set_needs_debug_logging(true);
				break;
			}
		}

		if (!resources.stage_inputs.empty()) {
			for (Resource const &res : resources.stage_inputs) {
				uint32_t binding = compiler.get_automatic_msl_resource_binding(res.id);
				if (binding != (uint32_t)-1) {
					stage_data.vertex_input_binding_mask |= 1 << binding;
				}
			}
		}

		stage_data.is_position_invariant = compiler.is_position_invariant();
		stage_data.supports_fast_math = !entry_point.flags.get(spv::ExecutionModeSignedZeroInfNanPreserve);
		stage_data.hash = SHA256Digest(source.c_str(), source.length());
		stage_data.source_size = source.length();
		::Vector<uint8_t> binary_data;
		binary_data.resize(stage_data.source_size);
		memcpy(binary_data.ptrw(), source.c_str(), stage_data.source_size);

		if (export_mode) {
			if (compiler_props.is_valid()) {
				// Try to compile the Metal source code.
				::Vector<uint8_t> library_data;
				Error compile_err = compile_metal_source(source.c_str(), stage_data, library_data);
				if (compile_err == OK) {
					// If we successfully compiled to a `.metallib`, there are greater restrictions on target platforms,
					// so we must update the properties.
					stage_data.library_size = library_data.size();
					binary_data.resize(stage_data.source_size + stage_data.library_size);
					memcpy(binary_data.ptrw() + stage_data.source_size, library_data.ptr(), stage_data.library_size);
				}
			} else {
				WARN_PRINT_ONCE("Metal shader baking limited to SPIR-V: Unable to determine toolchain properties to compile .metallib");
			}
		}

		uint32_t binary_data_size = binary_data.size();
		Shader &shader = shaders.write[i];
		shader.shader_stage = stage;
		shader.code_decompressed_size = binary_data_size;
		shader.code_compressed_bytes.resize(binary_data_size);

		uint32_t compressed_size = 0;
		bool compressed = compress_code(binary_data.ptr(), binary_data_size, shader.code_compressed_bytes.ptrw(), &compressed_size, &shader.code_compression_flags);
		ERR_FAIL_COND_V_MSG(!compressed, false, vformat("Failed to compress native code to native for SPIR-V #%d.", i));

		shader.code_compressed_bytes.resize(compressed_size);
	}

	return true;
}

#pragma clang diagnostic pop

uint32_t RenderingShaderContainerMetal::_to_bytes_reflection_extra_data(uint8_t *p_bytes) const {
	if (p_bytes != nullptr) {
		*(HeaderData *)p_bytes = mtl_reflection_data;
	}
	return sizeof(HeaderData);
}

uint32_t RenderingShaderContainerMetal::_to_bytes_reflection_binding_uniform_extra_data(uint8_t *p_bytes, uint32_t p_index) const {
	if (p_bytes != nullptr) {
		*(UniformData *)p_bytes = mtl_reflection_binding_set_uniforms_data[p_index];
	}
	return sizeof(UniformData);
}

uint32_t RenderingShaderContainerMetal::_to_bytes_shader_extra_data(uint8_t *p_bytes, uint32_t p_index) const {
	if (p_bytes != nullptr) {
		*(StageData *)p_bytes = mtl_shaders[p_index];
	}
	return sizeof(StageData);
}

uint32_t RenderingShaderContainerMetal::_from_bytes_reflection_extra_data(const uint8_t *p_bytes) {
	mtl_reflection_data = *(HeaderData *)p_bytes;
	return sizeof(HeaderData);
}

uint32_t RenderingShaderContainerMetal::_from_bytes_reflection_binding_uniform_extra_data_start(const uint8_t *p_bytes) {
	mtl_reflection_binding_set_uniforms_data.resize(reflection_binding_set_uniforms_data.size());
	return 0;
}

uint32_t RenderingShaderContainerMetal::_from_bytes_reflection_binding_uniform_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	mtl_reflection_binding_set_uniforms_data.ptrw()[p_index] = *(UniformData *)p_bytes;
	return sizeof(UniformData);
}

uint32_t RenderingShaderContainerMetal::_from_bytes_shader_extra_data_start(const uint8_t *p_bytes) {
	mtl_shaders.resize(shaders.size());
	return 0;
}

uint32_t RenderingShaderContainerMetal::_from_bytes_shader_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	mtl_shaders.ptrw()[p_index] = *(StageData *)p_bytes;
	return sizeof(StageData);
}

RenderingShaderContainerMetal::MetalShaderReflection RenderingShaderContainerMetal::get_metal_shader_reflection() const {
	MetalShaderReflection res;

	uint32_t uniform_set_count = reflection_binding_set_uniforms_count.size();
	uint32_t start = 0;
	res.uniform_sets.resize(uniform_set_count);
	for (uint32_t i = 0; i < uniform_set_count; i++) {
		Vector<UniformData> &set = res.uniform_sets.ptrw()[i];
		uint32_t count = reflection_binding_set_uniforms_count.get(i);
		set.resize(count);
		memcpy(set.ptrw(), &mtl_reflection_binding_set_uniforms_data.ptr()[start], count * sizeof(UniformData));
		start += count;
	}

	return res;
}

uint32_t RenderingShaderContainerMetal::_format() const {
	return 0x42424242;
}

uint32_t RenderingShaderContainerMetal::_format_version() const {
	return FORMAT_VERSION;
}

Ref<RenderingShaderContainer> RenderingShaderContainerFormatMetal::create_container() const {
	Ref<RenderingShaderContainerMetal> result;
	result.instantiate();
	result->set_export_mode(export_mode);
	result->set_device_profile(device_profile);
	return result;
}

RenderingDeviceCommons::ShaderLanguageVersion RenderingShaderContainerFormatMetal::get_shader_language_version() const {
	return SHADER_LANGUAGE_VULKAN_VERSION_1_1;
}

RenderingDeviceCommons::ShaderSpirvVersion RenderingShaderContainerFormatMetal::get_shader_spirv_version() const {
	return SHADER_SPIRV_VERSION_1_6;
}

RenderingShaderContainerFormatMetal::RenderingShaderContainerFormatMetal(const MetalDeviceProfile *p_device_profile, bool p_export) :
		export_mode(p_export), device_profile(p_device_profile) {
}

String MinOsVersion::to_compiler_os_version() const {
	if (version == UINT32_MAX) {
		return "";
	}

	uint32_t major = version / 10000;
	uint32_t minor = (version % 10000) / 100;
	return vformat("%d.%d", major, minor);
}

MinOsVersion::MinOsVersion(const String &p_version) {
	int pos = p_version.find_char('.');
	if (pos > 0) {
		version = (uint32_t)(p_version.substr(0, pos).to_int() * 10000 +
				p_version.substr(pos + 1).to_int() * 100);
	} else {
		version = (uint32_t)(p_version.to_int() * 10000);
	}

	if (version == 0) {
		version = UINT32_MAX;
	}
}

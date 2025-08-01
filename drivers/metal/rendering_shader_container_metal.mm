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

#include "rendering_shader_container_metal.h"

#include "servers/rendering/rendering_device.h"

#import "core/io/marshalls.h"

#import <Metal/Metal.h>
#import <spirv.hpp>
#import <spirv_msl.hpp>
#import <spirv_parser.hpp>

Mutex MetalDeviceProfile::profiles_lock;
HashMap<uint32_t, MetalDeviceProfile> MetalDeviceProfile::profiles;

const MetalDeviceProfile *MetalDeviceProfile::get_profile(MetalDeviceProfile::Platform p_platform, MetalDeviceProfile::GPU p_gpu) {
	DEV_ASSERT(p_platform == Platform::macOS || p_platform == Platform::iOS);

	MutexLock lock(profiles_lock);

	uint32_t key = (uint32_t)p_platform << 16 | (uint32_t)p_gpu;
	if (MetalDeviceProfile *profile = profiles.getptr(key)) {
		return profile;
	}

	MetalDeviceProfile res;
	res.platform = p_platform;
	res.gpu = p_gpu;
	if (p_platform == Platform::macOS) {
		res.features.mslVersionMajor = 3;
		res.features.mslVersionMinor = 2;
		res.features.argument_buffers_tier = ArgumentBuffersTier::Tier2;
		res.features.simdPermute = true;
	} else if (p_platform == Platform::iOS) {
		switch (p_gpu) {
			case GPU::Apple1:
			case GPU::Apple2:
			case GPU::Apple3:
			case GPU::Apple4:
			case GPU::Apple5: {
				res.features.simdPermute = false;
				res.features.argument_buffers_tier = ArgumentBuffersTier::Tier1;
			} break;
			case GPU::Apple6:
			case GPU::Apple7:
			case GPU::Apple8:
			case GPU::Apple9: {
				res.features.argument_buffers_tier = ArgumentBuffersTier::Tier2;
				res.features.simdPermute = true;
			} break;
		}
		res.features.mslVersionMajor = 3;
		res.features.mslVersionMinor = 2;
	}

	return &profiles.insert(key, res)->value;
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
	}

	// Build the metallib binary.
	{
		List<String> args{ "-sdk", sdk, "metal", "-O3" };
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

bool RenderingShaderContainerMetal::_set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) {
	using namespace spirv_cross;
	using spirv_cross::CompilerMSL;
	using spirv_cross::Resource;

	// initialize Metal-specific reflection data
	shaders.resize(p_spirv.size());
	mtl_shaders.resize(p_spirv.size());
	mtl_reflection_binding_set_uniforms_data.resize(reflection_binding_set_uniforms_data.size());
	mtl_reflection_specialization_data.resize(reflection_specialization_data.size());

	mtl_reflection_data.set_needs_view_mask_buffer(reflection_data.has_multiview);

	// set_indexes will contain the starting offsets of each descriptor set in the binding set uniforms data
	// including the last one, which is the size of reflection_binding_set_uniforms_count.
	LocalVector<uint32_t> set_indexes;
	uint32_t set_indexes_size = reflection_binding_set_uniforms_count.size() + 1;
	{
		// calculate the starting offsets of each descriptor set in the binding set uniforms data
		uint32_t size = reflection_binding_set_uniforms_count.size();
		set_indexes.resize(set_indexes_size);
		uint32_t offset = 0;
		for (uint32_t i = 0; i < size; i++) {
			set_indexes[i] = offset;
			offset += reflection_binding_set_uniforms_count.get(i);
		}
		set_indexes[set_indexes_size - 1] = offset;
	}
	CompilerMSL::Options msl_options{};
	// MAJOR * 10000 + MINOR * 100
	uint32_t msl_version = CompilerMSL::Options::make_msl_version(device_profile->features.mslVersionMajor, device_profile->features.mslVersionMinor);
	msl_options.set_msl_version(device_profile->features.mslVersionMajor, device_profile->features.mslVersionMinor);
	mtl_reflection_data.msl_version = msl_options.msl_version;
	msl_options.platform = device_profile->platform == MetalDeviceProfile::Platform::macOS ? CompilerMSL::Options::macOS : CompilerMSL::Options::iOS;

	if (device_profile->platform == MetalDeviceProfile::Platform::iOS) {
		msl_options.ios_use_simdgroup_functions = device_profile->features.simdPermute;
		msl_options.ios_support_base_vertex_instance = true;
	}

	bool disable_argument_buffers = false;
	if (String v = OS::get_singleton()->get_environment("GODOT_MTL_DISABLE_ARGUMENT_BUFFERS"); v == "1") {
		disable_argument_buffers = true;
	}

	if (device_profile->features.argument_buffers_tier >= MetalDeviceProfile::ArgumentBuffersTier::Tier2 && !disable_argument_buffers) {
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
	// We can't use this, as we have to add the descriptor sets via compiler.add_msl_resource_binding.
	// msl_options.pad_argument_buffer_resources = true;
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
	if (msl_version >= CompilerMSL::Options::make_msl_version(3, 2)) {
		// All 3.2+ versions support device coherence, so we can disable texture fences.
		msl_options.readwrite_texture_fences = false;
	}

	CompilerGLSL::Options options{};
	options.vertex.flip_vert_y = true;
#if DEV_ENABLED
	options.emit_line_directives = true;
#endif

	for (uint32_t i = 0; i < p_spirv.size(); i++) {
		StageData &stage_data = mtl_shaders.write[i];
		RD::ShaderStageSPIRVData const &v = p_spirv[i];
		RD::ShaderStage stage = v.shader_stage;
		char const *stage_name = RD::SHADER_STAGE_NAMES[stage];
		uint32_t const *const ir = reinterpret_cast<uint32_t const *const>(v.spirv.ptr());
		size_t word_count = v.spirv.size() / sizeof(uint32_t);
		Parser parser(ir, word_count);
		try {
			parser.parse();
		} catch (CompilerError &e) {
			ERR_FAIL_V_MSG(false, "Failed to parse IR at stage " + String(RD::SHADER_STAGE_NAMES[stage]) + ": " + e.what());
		}

		CompilerMSL compiler(std::move(parser.get_parsed_ir()));
		compiler.set_msl_options(msl_options);
		compiler.set_common_options(options);

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

		// Process specialization constants.
		if (!compiler.get_specialization_constants().empty()) {
			uint32_t size = reflection_specialization_data.size();
			for (SpecializationConstant const &constant : compiler.get_specialization_constants()) {
				uint32_t j = 0;
				while (j < size) {
					const ReflectionSpecializationData &res = reflection_specialization_data.ptr()[j];
					if (res.constant_id == constant.constant_id) {
						mtl_reflection_specialization_data.ptrw()[j].used_stages |= 1 << stage;
						// emulate labeled for loop and continue
						goto outer_continue;
					}
					++j;
				}
				if (j == size) {
					WARN_PRINT(String(stage_name) + ": unable to find constant_id: " + itos(constant.constant_id));
				}
			outer_continue:;
			}
		}

		// Process bindings.
		uint32_t uniform_sets_size = reflection_binding_set_uniforms_count.size();
		using BT = SPIRType::BaseType;

		// Always clearer than a boolean.
		enum class Writable {
			No,
			Maybe,
		};

		// Returns a std::optional containing the value of the
		// decoration, if it exists.
		auto get_decoration = [&compiler](spirv_cross::ID id, spv::Decoration decoration) {
			uint32_t res = -1;
			if (compiler.has_decoration(id, decoration)) {
				res = compiler.get_decoration(id, decoration);
			}
			return res;
		};

		auto descriptor_bindings = [&compiler, &active, this, &set_indexes, uniform_sets_size, stage, &get_decoration](SmallVector<Resource> &p_resources, Writable p_writable) {
			for (Resource const &res : p_resources) {
				uint32_t dset = get_decoration(res.id, spv::DecorationDescriptorSet);
				uint32_t dbin = get_decoration(res.id, spv::DecorationBinding);
				UniformData *found = nullptr;
				if (dset != (uint32_t)-1 && dbin != (uint32_t)-1 && dset < uniform_sets_size) {
					uint32_t begin = set_indexes[dset];
					uint32_t end = set_indexes[dset + 1];
					for (uint32_t j = begin; j < end; j++) {
						const ReflectionBindingData &ref_bind = reflection_binding_set_uniforms_data[j];
						if (dbin == ref_bind.binding) {
							found = &mtl_reflection_binding_set_uniforms_data.write[j];
							break;
						}
					}
				}

				ERR_FAIL_NULL_V_MSG(found, ERR_CANT_CREATE, "UniformData not found");

				bool is_active = active.find(res.id) != active.end();
				if (is_active) {
					found->active_stages |= 1 << stage;
				}

				BindingInfoData &primary = found->get_binding_for_stage(stage);

				SPIRType const &a_type = compiler.get_type(res.type_id);
				BT basetype = a_type.basetype;

				switch (basetype) {
					case BT::Struct: {
						primary.data_type = MTLDataTypePointer;
					} break;

					case BT::Image:
					case BT::SampledImage: {
						primary.data_type = MTLDataTypeTexture;
					} break;

					case BT::Sampler: {
						primary.data_type = MTLDataTypeSampler;
						primary.array_length = 1;
						for (uint32_t const &a : a_type.array) {
							primary.array_length *= a;
						}
					} break;

					default: {
						ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Unexpected BaseType");
					} break;
				}

				// Find array length of image.
				if (basetype == BT::Image || basetype == BT::SampledImage) {
					primary.array_length = 1;
					for (uint32_t const &a : a_type.array) {
						primary.array_length *= a;
					}
					primary.is_multisampled = a_type.image.ms;

					SPIRType::ImageType const &image = a_type.image;
					primary.image_format = image.format;

					switch (image.dim) {
						case spv::Dim1D: {
							if (image.arrayed) {
								primary.texture_type = MTLTextureType1DArray;
							} else {
								primary.texture_type = MTLTextureType1D;
							}
						} break;
						case spv::DimSubpassData: {
							[[fallthrough]];
						}
						case spv::Dim2D: {
							if (image.arrayed && image.ms) {
								primary.texture_type = MTLTextureType2DMultisampleArray;
							} else if (image.arrayed) {
								primary.texture_type = MTLTextureType2DArray;
							} else if (image.ms) {
								primary.texture_type = MTLTextureType2DMultisample;
							} else {
								primary.texture_type = MTLTextureType2D;
							}
						} break;
						case spv::Dim3D: {
							primary.texture_type = MTLTextureType3D;
						} break;
						case spv::DimCube: {
							if (image.arrayed) {
								primary.texture_type = MTLTextureTypeCube;
							}
						} break;
						case spv::DimRect: {
						} break;
						case spv::DimBuffer: {
							// VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
							primary.texture_type = MTLTextureTypeTextureBuffer;
						} break;
						case spv::DimTileImageDataEXT: {
							// Godot does not use this extension.
							// See: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_shader_tile_image.html
						} break;
						case spv::DimMax: {
							// Add all enumerations to silence the compiler warning
							// and generate future warnings, should a new one be added.
						} break;
					}
				}

				// Update writable.
				if (p_writable == Writable::Maybe) {
					if (basetype == BT::Struct) {
						Bitset flags = compiler.get_buffer_block_flags(res.id);
						if (!flags.get(spv::DecorationNonWritable)) {
							if (flags.get(spv::DecorationNonReadable)) {
								primary.access = MTLBindingAccessWriteOnly;
							} else {
								primary.access = MTLBindingAccessReadWrite;
							}
						}
					} else if (basetype == BT::Image) {
						switch (a_type.image.access) {
							case spv::AccessQualifierWriteOnly:
								primary.access = MTLBindingAccessWriteOnly;
								break;
							case spv::AccessQualifierReadWrite:
								primary.access = MTLBindingAccessReadWrite;
								break;
							case spv::AccessQualifierReadOnly:
								break;
							case spv::AccessQualifierMax:
								[[fallthrough]];
							default:
								if (!compiler.has_decoration(res.id, spv::DecorationNonWritable)) {
									if (compiler.has_decoration(res.id, spv::DecorationNonReadable)) {
										primary.access = MTLBindingAccessWriteOnly;
									} else {
										primary.access = MTLBindingAccessReadWrite;
									}
								}
								break;
						}
					}
				}

				switch (primary.access) {
					case MTLBindingAccessReadOnly:
						primary.usage = MTLResourceUsageRead;
						break;
					case MTLBindingAccessWriteOnly:
						primary.usage = MTLResourceUsageWrite;
						break;
					case MTLBindingAccessReadWrite:
						primary.usage = MTLResourceUsageRead | MTLResourceUsageWrite;
						break;
				}

				primary.index = compiler.get_automatic_msl_resource_binding(res.id);

				// A sampled image contains two bindings, the primary
				// is to the image, and the secondary is to the associated sampler.
				if (basetype == BT::SampledImage) {
					uint32_t binding = compiler.get_automatic_msl_resource_binding_secondary(res.id);
					if (binding != (uint32_t)-1) {
						BindingInfoData &secondary = found->get_secondary_binding_for_stage(stage);
						secondary.data_type = MTLDataTypeSampler;
						secondary.index = binding;
						secondary.access = MTLBindingAccessReadOnly;
					}
				}

				// An image may have a secondary binding if it is used
				// for atomic operations.
				if (basetype == BT::Image) {
					uint32_t binding = compiler.get_automatic_msl_resource_binding_secondary(res.id);
					if (binding != (uint32_t)-1) {
						BindingInfoData &secondary = found->get_secondary_binding_for_stage(stage);
						secondary.data_type = MTLDataTypePointer;
						secondary.index = binding;
						secondary.access = MTLBindingAccessReadWrite;
					}
				}
			}
			return Error::OK;
		};

		if (!resources.uniform_buffers.empty()) {
			Error err = descriptor_bindings(resources.uniform_buffers, Writable::No);
			ERR_FAIL_COND_V(err != OK, false);
		}
		if (!resources.storage_buffers.empty()) {
			Error err = descriptor_bindings(resources.storage_buffers, Writable::Maybe);
			ERR_FAIL_COND_V(err != OK, false);
		}
		if (!resources.storage_images.empty()) {
			Error err = descriptor_bindings(resources.storage_images, Writable::Maybe);
			ERR_FAIL_COND_V(err != OK, false);
		}
		if (!resources.sampled_images.empty()) {
			Error err = descriptor_bindings(resources.sampled_images, Writable::No);
			ERR_FAIL_COND_V(err != OK, false);
		}
		if (!resources.separate_images.empty()) {
			Error err = descriptor_bindings(resources.separate_images, Writable::No);
			ERR_FAIL_COND_V(err != OK, false);
		}
		if (!resources.separate_samplers.empty()) {
			Error err = descriptor_bindings(resources.separate_samplers, Writable::No);
			ERR_FAIL_COND_V(err != OK, false);
		}
		if (!resources.subpass_inputs.empty()) {
			Error err = descriptor_bindings(resources.subpass_inputs, Writable::No);
			ERR_FAIL_COND_V(err != OK, false);
		}

		if (!resources.push_constant_buffers.empty()) {
			for (Resource const &res : resources.push_constant_buffers) {
				uint32_t binding = compiler.get_automatic_msl_resource_binding(res.id);
				if (binding != (uint32_t)-1) {
					stage_data.push_constant_binding = binding;
				}
			}
		}

		ERR_FAIL_COND_V_MSG(!resources.atomic_counters.empty(), false, "Atomic counters not supported");
		ERR_FAIL_COND_V_MSG(!resources.acceleration_structures.empty(), false, "Acceleration structures not supported");
		ERR_FAIL_COND_V_MSG(!resources.shader_record_buffers.empty(), false, "Shader record buffers not supported");

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
			// Try to compile the Metal source code
			::Vector<uint8_t> library_data;
			Error compile_err = compile_metal_source(source.c_str(), stage_data, library_data);
			if (compile_err == OK) {
				stage_data.library_size = library_data.size();
				binary_data.resize(stage_data.source_size + stage_data.library_size);
				memcpy(binary_data.ptrw() + stage_data.source_size, library_data.ptr(), stage_data.library_size);
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

uint32_t RenderingShaderContainerMetal::_to_bytes_reflection_specialization_extra_data(uint8_t *p_bytes, uint32_t p_index) const {
	if (p_bytes != nullptr) {
		*(SpecializationData *)p_bytes = mtl_reflection_specialization_data[p_index];
	}
	return sizeof(SpecializationData);
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

uint32_t RenderingShaderContainerMetal::_from_bytes_reflection_specialization_extra_data_start(const uint8_t *p_bytes) {
	mtl_reflection_specialization_data.resize(reflection_specialization_data.size());
	return 0;
}

uint32_t RenderingShaderContainerMetal::_from_bytes_reflection_specialization_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	mtl_reflection_specialization_data.ptrw()[p_index] = *(SpecializationData *)p_bytes;
	return sizeof(SpecializationData);
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

	res.specialization_constants = mtl_reflection_specialization_data;
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

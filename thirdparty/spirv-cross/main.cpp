/*
 * Copyright 2015-2019 Arm Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spirv_cpp.hpp"
#include "spirv_cross_util.hpp"
#include "spirv_glsl.hpp"
#include "spirv_hlsl.hpp"
#include "spirv_msl.hpp"
#include "spirv_parser.hpp"
#include "spirv_reflect.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;
using namespace std;

#ifdef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
static inline void THROW(const char *str)
{
	fprintf(stderr, "SPIRV-Cross will abort: %s\n", str);
	fflush(stderr);
	abort();
}
#else
#define THROW(x) throw runtime_error(x)
#endif

struct CLIParser;
struct CLICallbacks
{
	void add(const char *cli, const function<void(CLIParser &)> &func)
	{
		callbacks[cli] = func;
	}
	unordered_map<string, function<void(CLIParser &)>> callbacks;
	function<void()> error_handler;
	function<void(const char *)> default_handler;
};

struct CLIParser
{
	CLIParser(CLICallbacks cbs_, int argc_, char *argv_[])
	    : cbs(move(cbs_))
	    , argc(argc_)
	    , argv(argv_)
	{
	}

	bool parse()
	{
#ifndef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
		try
#endif
		{
			while (argc && !ended_state)
			{
				const char *next = *argv++;
				argc--;

				if (*next != '-' && cbs.default_handler)
				{
					cbs.default_handler(next);
				}
				else
				{
					auto itr = cbs.callbacks.find(next);
					if (itr == ::end(cbs.callbacks))
					{
						THROW("Invalid argument");
					}

					itr->second(*this);
				}
			}

			return true;
		}
#ifndef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
		catch (...)
		{
			if (cbs.error_handler)
			{
				cbs.error_handler();
			}
			return false;
		}
#endif
	}

	void end()
	{
		ended_state = true;
	}

	uint32_t next_uint()
	{
		if (!argc)
		{
			THROW("Tried to parse uint, but nothing left in arguments");
		}

		uint64_t val = stoul(*argv);
		if (val > numeric_limits<uint32_t>::max())
		{
			THROW("next_uint() out of range");
		}

		argc--;
		argv++;

		return uint32_t(val);
	}

	double next_double()
	{
		if (!argc)
		{
			THROW("Tried to parse double, but nothing left in arguments");
		}

		double val = stod(*argv);

		argc--;
		argv++;

		return val;
	}

	// Return a string only if it's not prefixed with `--`, otherwise return the default value
	const char *next_value_string(const char *default_value)
	{
		if (!argc)
		{
			return default_value;
		}

		if (0 == strncmp("--", *argv, 2))
		{
			return default_value;
		}

		return next_string();
	}

	const char *next_string()
	{
		if (!argc)
		{
			THROW("Tried to parse string, but nothing left in arguments");
		}

		const char *ret = *argv;
		argc--;
		argv++;
		return ret;
	}

	CLICallbacks cbs;
	int argc;
	char **argv;
	bool ended_state = false;
};

static vector<uint32_t> read_spirv_file(const char *path)
{
	FILE *file = fopen(path, "rb");
	if (!file)
	{
		fprintf(stderr, "Failed to open SPIR-V file: %s\n", path);
		return {};
	}

	fseek(file, 0, SEEK_END);
	long len = ftell(file) / sizeof(uint32_t);
	rewind(file);

	vector<uint32_t> spirv(len);
	if (fread(spirv.data(), sizeof(uint32_t), len, file) != size_t(len))
		spirv.clear();

	fclose(file);
	return spirv;
}

static bool write_string_to_file(const char *path, const char *string)
{
	FILE *file = fopen(path, "w");
	if (!file)
	{
		fprintf(stderr, "Failed to write file: %s\n", path);
		return false;
	}

	fprintf(file, "%s", string);
	fclose(file);
	return true;
}

static void print_resources(const Compiler &compiler, const char *tag, const SmallVector<Resource> &resources)
{
	fprintf(stderr, "%s\n", tag);
	fprintf(stderr, "=============\n\n");
	bool print_ssbo = !strcmp(tag, "ssbos");

	for (auto &res : resources)
	{
		auto &type = compiler.get_type(res.type_id);

		if (print_ssbo && compiler.buffer_is_hlsl_counter_buffer(res.id))
			continue;

		// If we don't have a name, use the fallback for the type instead of the variable
		// for SSBOs and UBOs since those are the only meaningful names to use externally.
		// Push constant blocks are still accessed by name and not block name, even though they are technically Blocks.
		bool is_push_constant = compiler.get_storage_class(res.id) == StorageClassPushConstant;
		bool is_block = compiler.get_decoration_bitset(type.self).get(DecorationBlock) ||
		                compiler.get_decoration_bitset(type.self).get(DecorationBufferBlock);
		bool is_sized_block = is_block && (compiler.get_storage_class(res.id) == StorageClassUniform ||
		                                   compiler.get_storage_class(res.id) == StorageClassUniformConstant);
		uint32_t fallback_id = !is_push_constant && is_block ? res.base_type_id : res.id;

		uint32_t block_size = 0;
		uint32_t runtime_array_stride = 0;
		if (is_sized_block)
		{
			auto &base_type = compiler.get_type(res.base_type_id);
			block_size = uint32_t(compiler.get_declared_struct_size(base_type));
			runtime_array_stride = uint32_t(compiler.get_declared_struct_size_runtime_array(base_type, 1) -
			                                compiler.get_declared_struct_size_runtime_array(base_type, 0));
		}

		Bitset mask;
		if (print_ssbo)
			mask = compiler.get_buffer_block_flags(res.id);
		else
			mask = compiler.get_decoration_bitset(res.id);

		string array;
		for (auto arr : type.array)
			array = join("[", arr ? convert_to_string(arr) : "", "]") + array;

		fprintf(stderr, " ID %03u : %s%s", res.id,
		        !res.name.empty() ? res.name.c_str() : compiler.get_fallback_name(fallback_id).c_str(), array.c_str());

		if (mask.get(DecorationLocation))
			fprintf(stderr, " (Location : %u)", compiler.get_decoration(res.id, DecorationLocation));
		if (mask.get(DecorationDescriptorSet))
			fprintf(stderr, " (Set : %u)", compiler.get_decoration(res.id, DecorationDescriptorSet));
		if (mask.get(DecorationBinding))
			fprintf(stderr, " (Binding : %u)", compiler.get_decoration(res.id, DecorationBinding));
		if (mask.get(DecorationInputAttachmentIndex))
			fprintf(stderr, " (Attachment : %u)", compiler.get_decoration(res.id, DecorationInputAttachmentIndex));
		if (mask.get(DecorationNonReadable))
			fprintf(stderr, " writeonly");
		if (mask.get(DecorationNonWritable))
			fprintf(stderr, " readonly");
		if (is_sized_block)
		{
			fprintf(stderr, " (BlockSize : %u bytes)", block_size);
			if (runtime_array_stride)
				fprintf(stderr, " (Unsized array stride: %u bytes)", runtime_array_stride);
		}

		uint32_t counter_id = 0;
		if (print_ssbo && compiler.buffer_get_hlsl_counter_buffer(res.id, counter_id))
			fprintf(stderr, " (HLSL counter buffer ID: %u)", counter_id);
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "=============\n\n");
}

static const char *execution_model_to_str(spv::ExecutionModel model)
{
	switch (model)
	{
	case spv::ExecutionModelVertex:
		return "vertex";
	case spv::ExecutionModelTessellationControl:
		return "tessellation control";
	case ExecutionModelTessellationEvaluation:
		return "tessellation evaluation";
	case ExecutionModelGeometry:
		return "geometry";
	case ExecutionModelFragment:
		return "fragment";
	case ExecutionModelGLCompute:
		return "compute";
	case ExecutionModelRayGenerationNV:
		return "raygenNV";
	case ExecutionModelIntersectionNV:
		return "intersectionNV";
	case ExecutionModelCallableNV:
		return "callableNV";
	case ExecutionModelAnyHitNV:
		return "anyhitNV";
	case ExecutionModelClosestHitNV:
		return "closesthitNV";
	case ExecutionModelMissNV:
		return "missNV";
	default:
		return "???";
	}
}

static void print_resources(const Compiler &compiler, const ShaderResources &res)
{
	auto &modes = compiler.get_execution_mode_bitset();

	fprintf(stderr, "Entry points:\n");
	auto entry_points = compiler.get_entry_points_and_stages();
	for (auto &e : entry_points)
		fprintf(stderr, "  %s (%s)\n", e.name.c_str(), execution_model_to_str(e.execution_model));
	fprintf(stderr, "\n");

	fprintf(stderr, "Execution modes:\n");
	modes.for_each_bit([&](uint32_t i) {
		auto mode = static_cast<ExecutionMode>(i);
		uint32_t arg0 = compiler.get_execution_mode_argument(mode, 0);
		uint32_t arg1 = compiler.get_execution_mode_argument(mode, 1);
		uint32_t arg2 = compiler.get_execution_mode_argument(mode, 2);

		switch (static_cast<ExecutionMode>(i))
		{
		case ExecutionModeInvocations:
			fprintf(stderr, "  Invocations: %u\n", arg0);
			break;

		case ExecutionModeLocalSize:
			fprintf(stderr, "  LocalSize: (%u, %u, %u)\n", arg0, arg1, arg2);
			break;

		case ExecutionModeOutputVertices:
			fprintf(stderr, "  OutputVertices: %u\n", arg0);
			break;

#define CHECK_MODE(m)                  \
	case ExecutionMode##m:             \
		fprintf(stderr, "  %s\n", #m); \
		break
			CHECK_MODE(SpacingEqual);
			CHECK_MODE(SpacingFractionalEven);
			CHECK_MODE(SpacingFractionalOdd);
			CHECK_MODE(VertexOrderCw);
			CHECK_MODE(VertexOrderCcw);
			CHECK_MODE(PixelCenterInteger);
			CHECK_MODE(OriginUpperLeft);
			CHECK_MODE(OriginLowerLeft);
			CHECK_MODE(EarlyFragmentTests);
			CHECK_MODE(PointMode);
			CHECK_MODE(Xfb);
			CHECK_MODE(DepthReplacing);
			CHECK_MODE(DepthGreater);
			CHECK_MODE(DepthLess);
			CHECK_MODE(DepthUnchanged);
			CHECK_MODE(LocalSizeHint);
			CHECK_MODE(InputPoints);
			CHECK_MODE(InputLines);
			CHECK_MODE(InputLinesAdjacency);
			CHECK_MODE(Triangles);
			CHECK_MODE(InputTrianglesAdjacency);
			CHECK_MODE(Quads);
			CHECK_MODE(Isolines);
			CHECK_MODE(OutputPoints);
			CHECK_MODE(OutputLineStrip);
			CHECK_MODE(OutputTriangleStrip);
			CHECK_MODE(VecTypeHint);
			CHECK_MODE(ContractionOff);

		default:
			break;
		}
	});
	fprintf(stderr, "\n");

	print_resources(compiler, "subpass inputs", res.subpass_inputs);
	print_resources(compiler, "inputs", res.stage_inputs);
	print_resources(compiler, "outputs", res.stage_outputs);
	print_resources(compiler, "textures", res.sampled_images);
	print_resources(compiler, "separate images", res.separate_images);
	print_resources(compiler, "separate samplers", res.separate_samplers);
	print_resources(compiler, "images", res.storage_images);
	print_resources(compiler, "ssbos", res.storage_buffers);
	print_resources(compiler, "ubos", res.uniform_buffers);
	print_resources(compiler, "push", res.push_constant_buffers);
	print_resources(compiler, "counters", res.atomic_counters);
	print_resources(compiler, "acceleration structures", res.acceleration_structures);
}

static void print_push_constant_resources(const Compiler &compiler, const SmallVector<Resource> &res)
{
	for (auto &block : res)
	{
		auto ranges = compiler.get_active_buffer_ranges(block.id);
		fprintf(stderr, "Active members in buffer: %s\n",
		        !block.name.empty() ? block.name.c_str() : compiler.get_fallback_name(block.id).c_str());

		fprintf(stderr, "==================\n\n");
		for (auto &range : ranges)
		{
			const auto &name = compiler.get_member_name(block.base_type_id, range.index);

			fprintf(stderr, "Member #%3u (%s): Offset: %4u, Range: %4u\n", range.index,
			        !name.empty() ? name.c_str() : compiler.get_fallback_member_name(range.index).c_str(),
			        unsigned(range.offset), unsigned(range.range));
		}
		fprintf(stderr, "==================\n\n");
	}
}

static void print_spec_constants(const Compiler &compiler)
{
	auto spec_constants = compiler.get_specialization_constants();
	fprintf(stderr, "Specialization constants\n");
	fprintf(stderr, "==================\n\n");
	for (auto &c : spec_constants)
		fprintf(stderr, "ID: %u, Spec ID: %u\n", c.id, c.constant_id);
	fprintf(stderr, "==================\n\n");
}

static void print_capabilities_and_extensions(const Compiler &compiler)
{
	fprintf(stderr, "Capabilities\n");
	fprintf(stderr, "============\n");
	for (auto &capability : compiler.get_declared_capabilities())
		fprintf(stderr, "Capability: %u\n", static_cast<unsigned>(capability));
	fprintf(stderr, "============\n\n");

	fprintf(stderr, "Extensions\n");
	fprintf(stderr, "============\n");
	for (auto &ext : compiler.get_declared_extensions())
		fprintf(stderr, "Extension: %s\n", ext.c_str());
	fprintf(stderr, "============\n\n");
}

struct PLSArg
{
	PlsFormat format;
	string name;
};

struct Remap
{
	string src_name;
	string dst_name;
	unsigned components;
};

struct VariableTypeRemap
{
	string variable_name;
	string new_variable_type;
};

struct InterfaceVariableRename
{
	StorageClass storageClass;
	uint32_t location;
	string variable_name;
};

struct CLIArguments
{
	const char *input = nullptr;
	const char *output = nullptr;
	const char *cpp_interface_name = nullptr;
	uint32_t version = 0;
	uint32_t shader_model = 0;
	uint32_t msl_version = 0;
	bool es = false;
	bool set_version = false;
	bool set_shader_model = false;
	bool set_msl_version = false;
	bool set_es = false;
	bool dump_resources = false;
	bool force_temporary = false;
	bool flatten_ubo = false;
	bool fixup = false;
	bool yflip = false;
	bool sso = false;
	bool support_nonzero_baseinstance = true;
	bool msl_capture_output_to_buffer = false;
	bool msl_swizzle_texture_samples = false;
	bool msl_ios = false;
	bool msl_pad_fragment_output = false;
	bool msl_domain_lower_left = false;
	bool msl_argument_buffers = false;
	bool msl_texture_buffer_native = false;
	bool glsl_emit_push_constant_as_ubo = false;
	bool glsl_emit_ubo_as_plain_uniforms = false;
	SmallVector<uint32_t> msl_discrete_descriptor_sets;
	SmallVector<PLSArg> pls_in;
	SmallVector<PLSArg> pls_out;
	SmallVector<Remap> remaps;
	SmallVector<string> extensions;
	SmallVector<VariableTypeRemap> variable_type_remaps;
	SmallVector<InterfaceVariableRename> interface_variable_renames;
	SmallVector<HLSLVertexAttributeRemap> hlsl_attr_remap;
	string entry;
	string entry_stage;

	struct Rename
	{
		string old_name;
		string new_name;
		ExecutionModel execution_model;
	};
	SmallVector<Rename> entry_point_rename;

	uint32_t iterations = 1;
	bool cpp = false;
	string reflect;
	bool msl = false;
	bool hlsl = false;
	bool hlsl_compat = false;
	bool hlsl_support_nonzero_base = false;
	bool vulkan_semantics = false;
	bool flatten_multidimensional_arrays = false;
	bool use_420pack_extension = true;
	bool remove_unused = false;
	bool combined_samplers_inherit_bindings = false;
};

static void print_help()
{
	fprintf(stderr, "Usage: spirv-cross\n"
	                "\t[--output <output path>]\n"
	                "\t[SPIR-V file]\n"
	                "\t[--es]\n"
	                "\t[--no-es]\n"
	                "\t[--version <GLSL version>]\n"
	                "\t[--dump-resources]\n"
	                "\t[--help]\n"
	                "\t[--force-temporary]\n"
	                "\t[--vulkan-semantics]\n"
	                "\t[--flatten-ubo]\n"
	                "\t[--fixup-clipspace]\n"
	                "\t[--flip-vert-y]\n"
	                "\t[--iterations iter]\n"
	                "\t[--cpp]\n"
	                "\t[--cpp-interface-name <name>]\n"
	                "\t[--glsl-emit-push-constant-as-ubo]\n"
	                "\t[--glsl-emit-ubo-as-plain-uniforms]\n"
	                "\t[--msl]\n"
	                "\t[--msl-version <MMmmpp>]\n"
	                "\t[--msl-capture-output]\n"
	                "\t[--msl-swizzle-texture-samples]\n"
	                "\t[--msl-ios]\n"
	                "\t[--msl-pad-fragment-output]\n"
	                "\t[--msl-domain-lower-left]\n"
	                "\t[--msl-argument-buffers]\n"
	                "\t[--msl-texture-buffer-native]\n"
	                "\t[--msl-discrete-descriptor-set <index>]\n"
	                "\t[--hlsl]\n"
	                "\t[--reflect]\n"
	                "\t[--shader-model]\n"
	                "\t[--hlsl-enable-compat]\n"
	                "\t[--hlsl-support-nonzero-basevertex-baseinstance]\n"
	                "\t[--separate-shader-objects]\n"
	                "\t[--pls-in format input-name]\n"
	                "\t[--pls-out format output-name]\n"
	                "\t[--remap source_name target_name components]\n"
	                "\t[--extension ext]\n"
	                "\t[--entry name]\n"
	                "\t[--stage <stage (vert, frag, geom, tesc, tese comp)>]\n"
	                "\t[--remove-unused-variables]\n"
	                "\t[--flatten-multidimensional-arrays]\n"
	                "\t[--no-420pack-extension]\n"
	                "\t[--remap-variable-type <variable_name> <new_variable_type>]\n"
	                "\t[--rename-interface-variable <in|out> <location> <new_variable_name>]\n"
	                "\t[--set-hlsl-vertex-input-semantic <location> <semantic>]\n"
	                "\t[--rename-entry-point <old> <new> <stage>]\n"
	                "\t[--combined-samplers-inherit-bindings]\n"
	                "\t[--no-support-nonzero-baseinstance]\n"
	                "\n");
}

static bool remap_generic(Compiler &compiler, const SmallVector<Resource> &resources, const Remap &remap)
{
	auto itr =
	    find_if(begin(resources), end(resources), [&remap](const Resource &res) { return res.name == remap.src_name; });

	if (itr != end(resources))
	{
		compiler.set_remapped_variable_state(itr->id, true);
		compiler.set_name(itr->id, remap.dst_name);
		compiler.set_subpass_input_remapped_components(itr->id, remap.components);
		return true;
	}
	else
		return false;
}

static vector<PlsRemap> remap_pls(const SmallVector<PLSArg> &pls_variables, const SmallVector<Resource> &resources,
                                  const SmallVector<Resource> *secondary_resources)
{
	vector<PlsRemap> ret;

	for (auto &pls : pls_variables)
	{
		bool found = false;
		for (auto &res : resources)
		{
			if (res.name == pls.name)
			{
				ret.push_back({ res.id, pls.format });
				found = true;
				break;
			}
		}

		if (!found && secondary_resources)
		{
			for (auto &res : *secondary_resources)
			{
				if (res.name == pls.name)
				{
					ret.push_back({ res.id, pls.format });
					found = true;
					break;
				}
			}
		}

		if (!found)
			fprintf(stderr, "Did not find stage input/output/target with name \"%s\".\n", pls.name.c_str());
	}

	return ret;
}

static PlsFormat pls_format(const char *str)
{
	if (!strcmp(str, "r11f_g11f_b10f"))
		return PlsR11FG11FB10F;
	else if (!strcmp(str, "r32f"))
		return PlsR32F;
	else if (!strcmp(str, "rg16f"))
		return PlsRG16F;
	else if (!strcmp(str, "rg16"))
		return PlsRG16;
	else if (!strcmp(str, "rgb10_a2"))
		return PlsRGB10A2;
	else if (!strcmp(str, "rgba8"))
		return PlsRGBA8;
	else if (!strcmp(str, "rgba8i"))
		return PlsRGBA8I;
	else if (!strcmp(str, "rgba8ui"))
		return PlsRGBA8UI;
	else if (!strcmp(str, "rg16i"))
		return PlsRG16I;
	else if (!strcmp(str, "rgb10_a2ui"))
		return PlsRGB10A2UI;
	else if (!strcmp(str, "rg16ui"))
		return PlsRG16UI;
	else if (!strcmp(str, "r32ui"))
		return PlsR32UI;
	else
		return PlsNone;
}

static ExecutionModel stage_to_execution_model(const std::string &stage)
{
	if (stage == "vert")
		return ExecutionModelVertex;
	else if (stage == "frag")
		return ExecutionModelFragment;
	else if (stage == "comp")
		return ExecutionModelGLCompute;
	else if (stage == "tesc")
		return ExecutionModelTessellationControl;
	else if (stage == "tese")
		return ExecutionModelTessellationEvaluation;
	else if (stage == "geom")
		return ExecutionModelGeometry;
	else
		SPIRV_CROSS_THROW("Invalid stage.");
}

static string compile_iteration(const CLIArguments &args, std::vector<uint32_t> spirv_file)
{
	Parser spirv_parser(move(spirv_file));
	spirv_parser.parse();

	unique_ptr<CompilerGLSL> compiler;
	bool combined_image_samplers = false;
	bool build_dummy_sampler = false;

	if (args.cpp)
	{
		compiler.reset(new CompilerCPP(move(spirv_parser.get_parsed_ir())));
		if (args.cpp_interface_name)
			static_cast<CompilerCPP *>(compiler.get())->set_interface_name(args.cpp_interface_name);
	}
	else if (args.msl)
	{
		compiler.reset(new CompilerMSL(move(spirv_parser.get_parsed_ir())));

		auto *msl_comp = static_cast<CompilerMSL *>(compiler.get());
		auto msl_opts = msl_comp->get_msl_options();
		if (args.set_msl_version)
			msl_opts.msl_version = args.msl_version;
		msl_opts.capture_output_to_buffer = args.msl_capture_output_to_buffer;
		msl_opts.swizzle_texture_samples = args.msl_swizzle_texture_samples;
		if (args.msl_ios)
			msl_opts.platform = CompilerMSL::Options::iOS;
		msl_opts.pad_fragment_output_components = args.msl_pad_fragment_output;
		msl_opts.tess_domain_origin_lower_left = args.msl_domain_lower_left;
		msl_opts.argument_buffers = args.msl_argument_buffers;
		msl_opts.texture_buffer_native = args.msl_texture_buffer_native;
		msl_comp->set_msl_options(msl_opts);
		for (auto &v : args.msl_discrete_descriptor_sets)
			msl_comp->add_discrete_descriptor_set(v);
	}
	else if (args.hlsl)
		compiler.reset(new CompilerHLSL(move(spirv_parser.get_parsed_ir())));
	else
	{
		combined_image_samplers = !args.vulkan_semantics;
		if (!args.vulkan_semantics)
			build_dummy_sampler = true;
		compiler.reset(new CompilerGLSL(move(spirv_parser.get_parsed_ir())));
	}

	if (!args.variable_type_remaps.empty())
	{
		auto remap_cb = [&](const SPIRType &, const string &name, string &out) -> void {
			for (const VariableTypeRemap &remap : args.variable_type_remaps)
				if (name == remap.variable_name)
					out = remap.new_variable_type;
		};

		compiler->set_variable_type_remap_callback(move(remap_cb));
	}

	for (auto &rename : args.entry_point_rename)
		compiler->rename_entry_point(rename.old_name, rename.new_name, rename.execution_model);

	auto entry_points = compiler->get_entry_points_and_stages();
	auto entry_point = args.entry;
	ExecutionModel model = ExecutionModelMax;

	if (!args.entry_stage.empty())
	{
		model = stage_to_execution_model(args.entry_stage);
		if (entry_point.empty())
		{
			// Just use the first entry point with this stage.
			for (auto &e : entry_points)
			{
				if (e.execution_model == model)
				{
					entry_point = e.name;
					break;
				}
			}

			if (entry_point.empty())
			{
				fprintf(stderr, "Could not find an entry point with stage: %s\n", args.entry_stage.c_str());
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			// Make sure both stage and name exists.
			bool exists = false;
			for (auto &e : entry_points)
			{
				if (e.execution_model == model && e.name == entry_point)
				{
					exists = true;
					break;
				}
			}

			if (!exists)
			{
				fprintf(stderr, "Could not find an entry point %s with stage: %s\n", entry_point.c_str(),
				        args.entry_stage.c_str());
				exit(EXIT_FAILURE);
			}
		}
	}
	else if (!entry_point.empty())
	{
		// Make sure there is just one entry point with this name, or the stage
		// is ambiguous.
		uint32_t stage_count = 0;
		for (auto &e : entry_points)
		{
			if (e.name == entry_point)
			{
				stage_count++;
				model = e.execution_model;
			}
		}

		if (stage_count == 0)
		{
			fprintf(stderr, "There is no entry point with name: %s\n", entry_point.c_str());
			exit(EXIT_FAILURE);
		}
		else if (stage_count > 1)
		{
			fprintf(stderr, "There is more than one entry point with name: %s. Use --stage.\n", entry_point.c_str());
			exit(EXIT_FAILURE);
		}
	}

	if (!entry_point.empty())
		compiler->set_entry_point(entry_point, model);

	if (!args.set_version && !compiler->get_common_options().version)
	{
		fprintf(stderr, "Didn't specify GLSL version and SPIR-V did not specify language.\n");
		print_help();
		exit(EXIT_FAILURE);
	}

	CompilerGLSL::Options opts = compiler->get_common_options();
	if (args.set_version)
		opts.version = args.version;
	if (args.set_es)
		opts.es = args.es;
	opts.force_temporary = args.force_temporary;
	opts.separate_shader_objects = args.sso;
	opts.flatten_multidimensional_arrays = args.flatten_multidimensional_arrays;
	opts.enable_420pack_extension = args.use_420pack_extension;
	opts.vulkan_semantics = args.vulkan_semantics;
	opts.vertex.fixup_clipspace = args.fixup;
	opts.vertex.flip_vert_y = args.yflip;
	opts.vertex.support_nonzero_base_instance = args.support_nonzero_baseinstance;
	opts.emit_push_constant_as_uniform_buffer = args.glsl_emit_push_constant_as_ubo;
	opts.emit_uniform_buffer_as_plain_uniforms = args.glsl_emit_ubo_as_plain_uniforms;
	compiler->set_common_options(opts);

	// Set HLSL specific options.
	if (args.hlsl)
	{
		auto *hlsl = static_cast<CompilerHLSL *>(compiler.get());
		auto hlsl_opts = hlsl->get_hlsl_options();
		if (args.set_shader_model)
		{
			if (args.shader_model < 30)
			{
				fprintf(stderr, "Shader model earlier than 30 (3.0) not supported.\n");
				exit(EXIT_FAILURE);
			}

			hlsl_opts.shader_model = args.shader_model;
		}

		if (args.hlsl_compat)
		{
			// Enable all compat options.
			hlsl_opts.point_size_compat = true;
			hlsl_opts.point_coord_compat = true;
		}

		if (hlsl_opts.shader_model <= 30)
		{
			combined_image_samplers = true;
			build_dummy_sampler = true;
		}

		hlsl_opts.support_nonzero_base_vertex_base_instance = args.hlsl_support_nonzero_base;
		hlsl->set_hlsl_options(hlsl_opts);
	}

	if (build_dummy_sampler)
	{
		uint32_t sampler = compiler->build_dummy_sampler_for_combined_images();
		if (sampler != 0)
		{
			// Set some defaults to make validation happy.
			compiler->set_decoration(sampler, DecorationDescriptorSet, 0);
			compiler->set_decoration(sampler, DecorationBinding, 0);
		}
	}

	ShaderResources res;
	if (args.remove_unused)
	{
		auto active = compiler->get_active_interface_variables();
		res = compiler->get_shader_resources(active);
		compiler->set_enabled_interface_variables(move(active));
	}
	else
		res = compiler->get_shader_resources();

	if (args.flatten_ubo)
	{
		for (auto &ubo : res.uniform_buffers)
			compiler->flatten_buffer_block(ubo.id);
		for (auto &ubo : res.push_constant_buffers)
			compiler->flatten_buffer_block(ubo.id);
	}

	auto pls_inputs = remap_pls(args.pls_in, res.stage_inputs, &res.subpass_inputs);
	auto pls_outputs = remap_pls(args.pls_out, res.stage_outputs, nullptr);
	compiler->remap_pixel_local_storage(move(pls_inputs), move(pls_outputs));

	for (auto &ext : args.extensions)
		compiler->require_extension(ext);

	for (auto &remap : args.remaps)
	{
		if (remap_generic(*compiler, res.stage_inputs, remap))
			continue;
		if (remap_generic(*compiler, res.stage_outputs, remap))
			continue;
		if (remap_generic(*compiler, res.subpass_inputs, remap))
			continue;
	}

	for (auto &rename : args.interface_variable_renames)
	{
		if (rename.storageClass == StorageClassInput)
			spirv_cross_util::rename_interface_variable(*compiler, res.stage_inputs, rename.location,
			                                            rename.variable_name);
		else if (rename.storageClass == StorageClassOutput)
			spirv_cross_util::rename_interface_variable(*compiler, res.stage_outputs, rename.location,
			                                            rename.variable_name);
		else
		{
			fprintf(stderr, "error at --rename-interface-variable <in|out> ...\n");
			exit(EXIT_FAILURE);
		}
	}

	if (args.dump_resources)
	{
		print_resources(*compiler, res);
		print_push_constant_resources(*compiler, res.push_constant_buffers);
		print_spec_constants(*compiler);
		print_capabilities_and_extensions(*compiler);
	}

	if (combined_image_samplers)
	{
		compiler->build_combined_image_samplers();
		if (args.combined_samplers_inherit_bindings)
			spirv_cross_util::inherit_combined_sampler_bindings(*compiler);

		// Give the remapped combined samplers new names.
		for (auto &remap : compiler->get_combined_image_samplers())
		{
			compiler->set_name(remap.combined_id, join("SPIRV_Cross_Combined", compiler->get_name(remap.image_id),
			                                           compiler->get_name(remap.sampler_id)));
		}
	}

	if (args.hlsl)
	{
		auto *hlsl_compiler = static_cast<CompilerHLSL *>(compiler.get());
		uint32_t new_builtin = hlsl_compiler->remap_num_workgroups_builtin();
		if (new_builtin)
		{
			hlsl_compiler->set_decoration(new_builtin, DecorationDescriptorSet, 0);
			hlsl_compiler->set_decoration(new_builtin, DecorationBinding, 0);
		}
	}

	if (args.hlsl)
	{
		for (auto &remap : args.hlsl_attr_remap)
			static_cast<CompilerHLSL *>(compiler.get())->add_vertex_attribute_remap(remap);
	}

	return compiler->compile();
}

static int main_inner(int argc, char *argv[])
{
	CLIArguments args;
	CLICallbacks cbs;

	cbs.add("--help", [](CLIParser &parser) {
		print_help();
		parser.end();
	});
	cbs.add("--output", [&args](CLIParser &parser) { args.output = parser.next_string(); });
	cbs.add("--es", [&args](CLIParser &) {
		args.es = true;
		args.set_es = true;
	});
	cbs.add("--no-es", [&args](CLIParser &) {
		args.es = false;
		args.set_es = true;
	});
	cbs.add("--version", [&args](CLIParser &parser) {
		args.version = parser.next_uint();
		args.set_version = true;
	});
	cbs.add("--dump-resources", [&args](CLIParser &) { args.dump_resources = true; });
	cbs.add("--force-temporary", [&args](CLIParser &) { args.force_temporary = true; });
	cbs.add("--flatten-ubo", [&args](CLIParser &) { args.flatten_ubo = true; });
	cbs.add("--fixup-clipspace", [&args](CLIParser &) { args.fixup = true; });
	cbs.add("--flip-vert-y", [&args](CLIParser &) { args.yflip = true; });
	cbs.add("--iterations", [&args](CLIParser &parser) { args.iterations = parser.next_uint(); });
	cbs.add("--cpp", [&args](CLIParser &) { args.cpp = true; });
	cbs.add("--reflect", [&args](CLIParser &parser) { args.reflect = parser.next_value_string("json"); });
	cbs.add("--cpp-interface-name", [&args](CLIParser &parser) { args.cpp_interface_name = parser.next_string(); });
	cbs.add("--metal", [&args](CLIParser &) { args.msl = true; }); // Legacy compatibility
	cbs.add("--glsl-emit-push-constant-as-ubo", [&args](CLIParser &) { args.glsl_emit_push_constant_as_ubo = true; });
	cbs.add("--glsl-emit-ubo-as-plain-uniforms", [&args](CLIParser &) { args.glsl_emit_ubo_as_plain_uniforms = true; });
	cbs.add("--msl", [&args](CLIParser &) { args.msl = true; });
	cbs.add("--hlsl", [&args](CLIParser &) { args.hlsl = true; });
	cbs.add("--hlsl-enable-compat", [&args](CLIParser &) { args.hlsl_compat = true; });
	cbs.add("--hlsl-support-nonzero-basevertex-baseinstance",
	        [&args](CLIParser &) { args.hlsl_support_nonzero_base = true; });
	cbs.add("--vulkan-semantics", [&args](CLIParser &) { args.vulkan_semantics = true; });
	cbs.add("--flatten-multidimensional-arrays", [&args](CLIParser &) { args.flatten_multidimensional_arrays = true; });
	cbs.add("--no-420pack-extension", [&args](CLIParser &) { args.use_420pack_extension = false; });
	cbs.add("--msl-capture-output", [&args](CLIParser &) { args.msl_capture_output_to_buffer = true; });
	cbs.add("--msl-swizzle-texture-samples", [&args](CLIParser &) { args.msl_swizzle_texture_samples = true; });
	cbs.add("--msl-ios", [&args](CLIParser &) { args.msl_ios = true; });
	cbs.add("--msl-pad-fragment-output", [&args](CLIParser &) { args.msl_pad_fragment_output = true; });
	cbs.add("--msl-domain-lower-left", [&args](CLIParser &) { args.msl_domain_lower_left = true; });
	cbs.add("--msl-argument-buffers", [&args](CLIParser &) { args.msl_argument_buffers = true; });
	cbs.add("--msl-discrete-descriptor-set",
	        [&args](CLIParser &parser) { args.msl_discrete_descriptor_sets.push_back(parser.next_uint()); });
	cbs.add("--msl-texture-buffer-native", [&args](CLIParser &) { args.msl_texture_buffer_native = true; });
	cbs.add("--extension", [&args](CLIParser &parser) { args.extensions.push_back(parser.next_string()); });
	cbs.add("--rename-entry-point", [&args](CLIParser &parser) {
		auto old_name = parser.next_string();
		auto new_name = parser.next_string();
		auto model = stage_to_execution_model(parser.next_string());
		args.entry_point_rename.push_back({ old_name, new_name, move(model) });
	});
	cbs.add("--entry", [&args](CLIParser &parser) { args.entry = parser.next_string(); });
	cbs.add("--stage", [&args](CLIParser &parser) { args.entry_stage = parser.next_string(); });
	cbs.add("--separate-shader-objects", [&args](CLIParser &) { args.sso = true; });
	cbs.add("--set-hlsl-vertex-input-semantic", [&args](CLIParser &parser) {
		HLSLVertexAttributeRemap remap;
		remap.location = parser.next_uint();
		remap.semantic = parser.next_string();
		args.hlsl_attr_remap.push_back(move(remap));
	});

	cbs.add("--remap", [&args](CLIParser &parser) {
		string src = parser.next_string();
		string dst = parser.next_string();
		uint32_t components = parser.next_uint();
		args.remaps.push_back({ move(src), move(dst), components });
	});

	cbs.add("--remap-variable-type", [&args](CLIParser &parser) {
		string var_name = parser.next_string();
		string new_type = parser.next_string();
		args.variable_type_remaps.push_back({ move(var_name), move(new_type) });
	});

	cbs.add("--rename-interface-variable", [&args](CLIParser &parser) {
		StorageClass cls = StorageClassMax;
		string clsStr = parser.next_string();
		if (clsStr == "in")
			cls = StorageClassInput;
		else if (clsStr == "out")
			cls = StorageClassOutput;

		uint32_t loc = parser.next_uint();
		string var_name = parser.next_string();
		args.interface_variable_renames.push_back({ cls, loc, move(var_name) });
	});

	cbs.add("--pls-in", [&args](CLIParser &parser) {
		auto fmt = pls_format(parser.next_string());
		auto name = parser.next_string();
		args.pls_in.push_back({ move(fmt), move(name) });
	});
	cbs.add("--pls-out", [&args](CLIParser &parser) {
		auto fmt = pls_format(parser.next_string());
		auto name = parser.next_string();
		args.pls_out.push_back({ move(fmt), move(name) });
	});
	cbs.add("--shader-model", [&args](CLIParser &parser) {
		args.shader_model = parser.next_uint();
		args.set_shader_model = true;
	});
	cbs.add("--msl-version", [&args](CLIParser &parser) {
		args.msl_version = parser.next_uint();
		args.set_msl_version = true;
	});

	cbs.add("--remove-unused-variables", [&args](CLIParser &) { args.remove_unused = true; });
	cbs.add("--combined-samplers-inherit-bindings",
	        [&args](CLIParser &) { args.combined_samplers_inherit_bindings = true; });

	cbs.add("--no-support-nonzero-baseinstance", [&](CLIParser &) { args.support_nonzero_baseinstance = false; });

	cbs.default_handler = [&args](const char *value) { args.input = value; };
	cbs.error_handler = [] { print_help(); };

	CLIParser parser{ move(cbs), argc - 1, argv + 1 };
	if (!parser.parse())
		return EXIT_FAILURE;
	else if (parser.ended_state)
		return EXIT_SUCCESS;

	if (!args.input)
	{
		fprintf(stderr, "Didn't specify input file.\n");
		print_help();
		return EXIT_FAILURE;
	}

	auto spirv_file = read_spirv_file(args.input);
	if (spirv_file.empty())
		return EXIT_FAILURE;

	// Special case reflection because it has little to do with the path followed by code-outputting compilers
	if (!args.reflect.empty())
	{
		Parser spirv_parser(move(spirv_file));
		spirv_parser.parse();

		CompilerReflection compiler(move(spirv_parser.get_parsed_ir()));
		compiler.set_format(args.reflect);
		auto json = compiler.compile();
		if (args.output)
			write_string_to_file(args.output, json.c_str());
		else
			printf("%s", json.c_str());
		return EXIT_SUCCESS;
	}

	string compiled_output;

	if (args.iterations == 1)
		compiled_output = compile_iteration(args, move(spirv_file));
	else
	{
		for (unsigned i = 0; i < args.iterations; i++)
			compiled_output = compile_iteration(args, spirv_file);
	}

	if (args.output)
		write_string_to_file(args.output, compiled_output.c_str());
	else
		printf("%s", compiled_output.c_str());

	return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
#ifdef SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	return main_inner(argc, argv);
#else
	// Make sure we catch the exception or it just disappears into the aether on Windows.
	try
	{
		return main_inner(argc, argv);
	}
	catch (const std::exception &e)
	{
		fprintf(stderr, "SPIRV-Cross threw an exception: %s\n", e.what());
		return EXIT_FAILURE;
	}
#endif
}

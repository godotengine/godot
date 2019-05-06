/*
 * Copyright 2018-2019 Bradley Austin Davis
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

#include "spirv_reflect.hpp"
#include "spirv_glsl.hpp"
#include <iomanip>

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;
using namespace std;

namespace simple_json
{
enum class Type
{
	Object,
	Array,
};

using State = std::pair<Type, bool>;
using Stack = std::stack<State>;

class Stream
{
	Stack stack;
	StringStream<> buffer;
	uint32_t indent{ 0 };
	char current_locale_radix_character = '.';

public:
	void set_current_locale_radix_character(char c)
	{
		current_locale_radix_character = c;
	}

	void begin_json_object();
	void end_json_object();
	void emit_json_key(const std::string &key);
	void emit_json_key_value(const std::string &key, const std::string &value);
	void emit_json_key_value(const std::string &key, bool value);
	void emit_json_key_value(const std::string &key, uint32_t value);
	void emit_json_key_value(const std::string &key, int32_t value);
	void emit_json_key_value(const std::string &key, float value);
	void emit_json_key_object(const std::string &key);
	void emit_json_key_array(const std::string &key);

	void begin_json_array();
	void end_json_array();
	void emit_json_array_value(const std::string &value);
	void emit_json_array_value(uint32_t value);

	std::string str() const
	{
		return buffer.str();
	}

private:
	inline void statement_indent()
	{
		for (uint32_t i = 0; i < indent; i++)
			buffer << "    ";
	}

	template <typename T>
	inline void statement_inner(T &&t)
	{
		buffer << std::forward<T>(t);
	}

	template <typename T, typename... Ts>
	inline void statement_inner(T &&t, Ts &&... ts)
	{
		buffer << std::forward<T>(t);
		statement_inner(std::forward<Ts>(ts)...);
	}

	template <typename... Ts>
	inline void statement(Ts &&... ts)
	{
		statement_indent();
		statement_inner(std::forward<Ts>(ts)...);
		buffer << '\n';
	}

	template <typename... Ts>
	void statement_no_return(Ts &&... ts)
	{
		statement_indent();
		statement_inner(std::forward<Ts>(ts)...);
	}
};
} // namespace simple_json

using namespace simple_json;

// Hackery to emit JSON without using nlohmann/json C++ library (which requires a
// higher level of compiler compliance than is required by SPIRV-Cross
void Stream::begin_json_array()
{
	if (!stack.empty() && stack.top().second)
	{
		statement_inner(",\n");
	}
	statement("[");
	++indent;
	stack.emplace(Type::Array, false);
}

void Stream::end_json_array()
{
	if (stack.empty() || stack.top().first != Type::Array)
		SPIRV_CROSS_THROW("Invalid JSON state");
	if (stack.top().second)
	{
		statement_inner("\n");
	}
	--indent;
	statement_no_return("]");
	stack.pop();
	if (!stack.empty())
	{
		stack.top().second = true;
	}
}

void Stream::emit_json_array_value(const std::string &value)
{
	if (stack.empty() || stack.top().first != Type::Array)
		SPIRV_CROSS_THROW("Invalid JSON state");

	if (stack.top().second)
		statement_inner(",\n");

	statement_no_return("\"", value, "\"");
	stack.top().second = true;
}

void Stream::emit_json_array_value(uint32_t value)
{
	if (stack.empty() || stack.top().first != Type::Array)
		SPIRV_CROSS_THROW("Invalid JSON state");
	if (stack.top().second)
		statement_inner(",\n");
	statement_no_return(std::to_string(value));
	stack.top().second = true;
}

void Stream::begin_json_object()
{
	if (!stack.empty() && stack.top().second)
	{
		statement_inner(",\n");
	}
	statement("{");
	++indent;
	stack.emplace(Type::Object, false);
}

void Stream::end_json_object()
{
	if (stack.empty() || stack.top().first != Type::Object)
		SPIRV_CROSS_THROW("Invalid JSON state");
	if (stack.top().second)
	{
		statement_inner("\n");
	}
	--indent;
	statement_no_return("}");
	stack.pop();
	if (!stack.empty())
	{
		stack.top().second = true;
	}
}

void Stream::emit_json_key(const std::string &key)
{
	if (stack.empty() || stack.top().first != Type::Object)
		SPIRV_CROSS_THROW("Invalid JSON state");

	if (stack.top().second)
		statement_inner(",\n");
	statement_no_return("\"", key, "\" : ");
	stack.top().second = true;
}

void Stream::emit_json_key_value(const std::string &key, const std::string &value)
{
	emit_json_key(key);
	statement_inner("\"", value, "\"");
}

void Stream::emit_json_key_value(const std::string &key, uint32_t value)
{
	emit_json_key(key);
	statement_inner(value);
}

void Stream::emit_json_key_value(const std::string &key, int32_t value)
{
	emit_json_key(key);
	statement_inner(value);
}

void Stream::emit_json_key_value(const std::string &key, float value)
{
	emit_json_key(key);
	statement_inner(convert_to_string(value, current_locale_radix_character));
}

void Stream::emit_json_key_value(const std::string &key, bool value)
{
	emit_json_key(key);
	statement_inner(value ? "true" : "false");
}

void Stream::emit_json_key_object(const std::string &key)
{
	emit_json_key(key);
	statement_inner("{\n");
	++indent;
	stack.emplace(Type::Object, false);
}

void Stream::emit_json_key_array(const std::string &key)
{
	emit_json_key(key);
	statement_inner("[\n");
	++indent;
	stack.emplace(Type::Array, false);
}

void CompilerReflection::set_format(const std::string &format)
{
	if (format != "json")
	{
		SPIRV_CROSS_THROW("Unsupported format");
	}
}

string CompilerReflection::compile()
{
	json_stream = std::make_shared<simple_json::Stream>();
	json_stream->set_current_locale_radix_character(current_locale_radix_character);
	json_stream->begin_json_object();
	fixup_type_alias();
	reorder_type_alias();
	emit_entry_points();
	emit_types();
	emit_resources();
	emit_specialization_constants();
	json_stream->end_json_object();
	return json_stream->str();
}

void CompilerReflection::emit_types()
{
	bool emitted_open_tag = false;

	ir.for_each_typed_id<SPIRType>([&](uint32_t, SPIRType &type) {
		if (type.basetype == SPIRType::Struct && !type.pointer && type.array.empty())
			emit_type(type, emitted_open_tag);
	});

	if (emitted_open_tag)
	{
		json_stream->end_json_object();
	}
}

void CompilerReflection::emit_type(const SPIRType &type, bool &emitted_open_tag)
{
	auto name = type_to_glsl(type);

	if (type.type_alias != 0)
		return;

	if (!emitted_open_tag)
	{
		json_stream->emit_json_key_object("types");
		emitted_open_tag = true;
	}
	json_stream->emit_json_key_object("_" + std::to_string(type.self));
	json_stream->emit_json_key_value("name", name);
	json_stream->emit_json_key_array("members");
	// FIXME ideally we'd like to emit the size of a structure as a
	// convenience to people parsing the reflected JSON.  The problem
	// is that there's no implicit size for a type.  It's final size
	// will be determined by the top level declaration in which it's
	// included.  So there might be one size for the struct if it's
	// included in a std140 uniform block and another if it's included
	// in a std430 uniform block.
	// The solution is to include *all* potential sizes as a map of
	// layout type name to integer, but that will probably require
	// some additional logic being written in this class, or in the
	// parent CompilerGLSL class.
	auto size = type.member_types.size();
	for (uint32_t i = 0; i < size; ++i)
	{
		emit_type_member(type, i);
	}
	json_stream->end_json_array();
	json_stream->end_json_object();
}

void CompilerReflection::emit_type_member(const SPIRType &type, uint32_t index)
{
	auto &membertype = get<SPIRType>(type.member_types[index]);
	json_stream->begin_json_object();
	auto name = to_member_name(type, index);
	// FIXME we'd like to emit the offset of each member, but such offsets are
	// context dependent.  See the comment above regarding structure sizes
	json_stream->emit_json_key_value("name", name);
	if (membertype.basetype == SPIRType::Struct)
	{
		json_stream->emit_json_key_value("type", "_" + std::to_string(membertype.self));
	}
	else
	{
		json_stream->emit_json_key_value("type", type_to_glsl(membertype));
	}
	emit_type_member_qualifiers(type, index);
	json_stream->end_json_object();
}

void CompilerReflection::emit_type_array(const SPIRType &type)
{
	if (!type.array.empty())
	{
		json_stream->emit_json_key_array("array");
		// Note that we emit the zeros here as a means of identifying
		// unbounded arrays.  This is necessary as otherwise there would
		// be no way of differentiating between float[4] and float[4][]
		for (const auto &value : type.array)
			json_stream->emit_json_array_value(value);
		json_stream->end_json_array();
	}
}

void CompilerReflection::emit_type_member_qualifiers(const SPIRType &type, uint32_t index)
{
	auto flags = combined_decoration_for_member(type, index);
	if (flags.get(DecorationRowMajor))
		json_stream->emit_json_key_value("row_major", true);

	auto &membertype = get<SPIRType>(type.member_types[index]);
	emit_type_array(membertype);
	auto &memb = ir.meta[type.self].members;
	if (index < memb.size())
	{
		auto &dec = memb[index];
		if (dec.decoration_flags.get(DecorationLocation))
			json_stream->emit_json_key_value("location", dec.location);
		if (dec.decoration_flags.get(DecorationOffset))
			json_stream->emit_json_key_value("offset", dec.offset);
	}
}

string CompilerReflection::execution_model_to_str(spv::ExecutionModel model)
{
	switch (model)
	{
	case ExecutionModelVertex:
		return "vert";
	case ExecutionModelTessellationControl:
		return "tesc";
	case ExecutionModelTessellationEvaluation:
		return "tese";
	case ExecutionModelGeometry:
		return "geom";
	case ExecutionModelFragment:
		return "frag";
	case ExecutionModelGLCompute:
		return "comp";
	case ExecutionModelRayGenerationNV:
		return "rgen";
	case ExecutionModelIntersectionNV:
		return "rint";
	case ExecutionModelAnyHitNV:
		return "rahit";
	case ExecutionModelClosestHitNV:
		return "rchit";
	case ExecutionModelMissNV:
		return "rmiss";
	case ExecutionModelCallableNV:
		return "rcall";
	default:
		return "???";
	}
}

// FIXME include things like the local_size dimensions, geometry output vertex count, etc
void CompilerReflection::emit_entry_points()
{
	auto entries = get_entry_points_and_stages();
	if (!entries.empty())
	{
		// Needed to make output deterministic.
		sort(begin(entries), end(entries), [](const EntryPoint &a, const EntryPoint &b) -> bool {
			if (a.execution_model < b.execution_model)
				return true;
			else if (a.execution_model > b.execution_model)
				return false;
			else
				return a.name < b.name;
		});

		json_stream->emit_json_key_array("entryPoints");
		for (auto &e : entries)
		{
			json_stream->begin_json_object();
			json_stream->emit_json_key_value("name", e.name);
			json_stream->emit_json_key_value("mode", execution_model_to_str(e.execution_model));
			json_stream->end_json_object();
		}
		json_stream->end_json_array();
	}
}

void CompilerReflection::emit_resources()
{
	auto res = get_shader_resources();
	emit_resources("subpass_inputs", res.subpass_inputs);
	emit_resources("inputs", res.stage_inputs);
	emit_resources("outputs", res.stage_outputs);
	emit_resources("textures", res.sampled_images);
	emit_resources("separate_images", res.separate_images);
	emit_resources("separate_samplers", res.separate_samplers);
	emit_resources("images", res.storage_images);
	emit_resources("ssbos", res.storage_buffers);
	emit_resources("ubos", res.uniform_buffers);
	emit_resources("push_constants", res.push_constant_buffers);
	emit_resources("counters", res.atomic_counters);
	emit_resources("acceleration_structures", res.acceleration_structures);
}

void CompilerReflection::emit_resources(const char *tag, const SmallVector<Resource> &resources)
{
	if (resources.empty())
	{
		return;
	}

	json_stream->emit_json_key_array(tag);
	for (auto &res : resources)
	{
		auto &type = get_type(res.type_id);
		auto typeflags = ir.meta[type.self].decoration.decoration_flags;
		auto &mask = get_decoration_bitset(res.id);

		// If we don't have a name, use the fallback for the type instead of the variable
		// for SSBOs and UBOs since those are the only meaningful names to use externally.
		// Push constant blocks are still accessed by name and not block name, even though they are technically Blocks.
		bool is_push_constant = get_storage_class(res.id) == StorageClassPushConstant;
		bool is_block = get_decoration_bitset(type.self).get(DecorationBlock) ||
		                get_decoration_bitset(type.self).get(DecorationBufferBlock);

		uint32_t fallback_id = !is_push_constant && is_block ? res.base_type_id : res.id;

		json_stream->begin_json_object();

		if (type.basetype == SPIRType::Struct)
		{
			json_stream->emit_json_key_value("type", "_" + std::to_string(res.base_type_id));
		}
		else
		{
			json_stream->emit_json_key_value("type", type_to_glsl(type));
		}

		json_stream->emit_json_key_value("name", !res.name.empty() ? res.name : get_fallback_name(fallback_id));
		{
			bool ssbo_block = type.storage == StorageClassStorageBuffer ||
			                  (type.storage == StorageClassUniform && typeflags.get(DecorationBufferBlock));
			if (ssbo_block)
			{
				auto buffer_flags = get_buffer_block_flags(res.id);
				if (buffer_flags.get(DecorationNonReadable))
					json_stream->emit_json_key_value("writeonly", true);
				if (buffer_flags.get(DecorationNonWritable))
					json_stream->emit_json_key_value("readonly", true);
				if (buffer_flags.get(DecorationRestrict))
					json_stream->emit_json_key_value("restrict", true);
				if (buffer_flags.get(DecorationCoherent))
					json_stream->emit_json_key_value("coherent", true);
			}
		}

		emit_type_array(type);

		{
			bool is_sized_block = is_block && (get_storage_class(res.id) == StorageClassUniform ||
			                                   get_storage_class(res.id) == StorageClassUniformConstant ||
			                                   get_storage_class(res.id) == StorageClassStorageBuffer);
			if (is_sized_block)
			{
				uint32_t block_size = uint32_t(get_declared_struct_size(get_type(res.base_type_id)));
				json_stream->emit_json_key_value("block_size", block_size);
			}
		}

		if (type.storage == StorageClassPushConstant)
			json_stream->emit_json_key_value("push_constant", true);
		if (mask.get(DecorationLocation))
			json_stream->emit_json_key_value("location", get_decoration(res.id, DecorationLocation));
		if (mask.get(DecorationRowMajor))
			json_stream->emit_json_key_value("row_major", true);
		if (mask.get(DecorationColMajor))
			json_stream->emit_json_key_value("column_major", true);
		if (mask.get(DecorationIndex))
			json_stream->emit_json_key_value("index", get_decoration(res.id, DecorationIndex));
		if (type.storage != StorageClassPushConstant && mask.get(DecorationDescriptorSet))
			json_stream->emit_json_key_value("set", get_decoration(res.id, DecorationDescriptorSet));
		if (mask.get(DecorationBinding))
			json_stream->emit_json_key_value("binding", get_decoration(res.id, DecorationBinding));
		if (mask.get(DecorationInputAttachmentIndex))
			json_stream->emit_json_key_value("input_attachment_index",
			                                 get_decoration(res.id, DecorationInputAttachmentIndex));
		if (mask.get(DecorationOffset))
			json_stream->emit_json_key_value("offset", get_decoration(res.id, DecorationOffset));

		// For images, the type itself adds a layout qualifer.
		// Only emit the format for storage images.
		if (type.basetype == SPIRType::Image && type.image.sampled == 2)
		{
			const char *fmt = format_to_glsl(type.image.format);
			if (fmt != nullptr)
				json_stream->emit_json_key_value("format", std::string(fmt));
		}
		json_stream->end_json_object();
	}
	json_stream->end_json_array();
}

void CompilerReflection::emit_specialization_constants()
{
	auto specialization_constants = get_specialization_constants();
	if (specialization_constants.empty())
		return;

	json_stream->emit_json_key_array("specialization_constants");
	for (const auto spec_const : specialization_constants)
	{
		auto &c = get<SPIRConstant>(spec_const.id);
		auto type = get<SPIRType>(c.constant_type);
		json_stream->begin_json_object();
		json_stream->emit_json_key_value("id", spec_const.constant_id);
		json_stream->emit_json_key_value("type", type_to_glsl(type));
		switch (type.basetype)
		{
		case SPIRType::UInt:
			json_stream->emit_json_key_value("default_value", c.scalar());
			break;

		case SPIRType::Int:
			json_stream->emit_json_key_value("default_value", c.scalar_i32());
			break;

		case SPIRType::Float:
			json_stream->emit_json_key_value("default_value", c.scalar_f32());
			break;

		case SPIRType::Boolean:
			json_stream->emit_json_key_value("default_value", c.scalar() != 0);
			break;

		default:
			break;
		}
		json_stream->end_json_object();
	}
	json_stream->end_json_array();
}

string CompilerReflection::to_member_name(const SPIRType &type, uint32_t index) const
{
	auto *type_meta = ir.find_meta(type.self);

	if (type_meta)
	{
		auto &memb = type_meta->members;
		if (index < memb.size() && !memb[index].alias.empty())
			return memb[index].alias;
		else
			return join("_m", index);
	}
	else
		return join("_m", index);
}

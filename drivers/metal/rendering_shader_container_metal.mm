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

#import "metal_objects.h"

#import "core/io/marshalls.h"

#import <Metal/MTLTexture.h>
#import <Metal/Metal.h>
#import <os/log.h>
#import <os/signpost.h>
#import <spirv.hpp>
#import <spirv_msl.hpp>
#import <spirv_parser.hpp>

const uint32_t SHADER_BINARY_VERSION = 4;

// region Serialization

class BufWriter;

template <typename T>
concept Serializable = requires(T t, BufWriter &p_writer) {
	{
		t.serialize_size()
	} -> std::same_as<size_t>;
	{
		t.serialize(p_writer)
	} -> std::same_as<void>;
};

class BufWriter {
	uint8_t *data = nullptr;
	uint64_t length = 0; // Length of data.
	uint64_t pos = 0;

public:
	BufWriter(uint8_t *p_data, uint64_t p_length) :
			data(p_data), length(p_length) {}

	template <Serializable T>
	void write(T const &p_value) {
		p_value.serialize(*this);
	}

	_FORCE_INLINE_ void write(uint32_t p_value) {
		DEV_ASSERT(pos + sizeof(uint32_t) <= length);
		pos += encode_uint32(p_value, data + pos);
	}

	_FORCE_INLINE_ void write(RD::ShaderStage p_value) {
		write((uint32_t)p_value);
	}

	_FORCE_INLINE_ void write(bool p_value) {
		DEV_ASSERT(pos + sizeof(uint8_t) <= length);
		*(data + pos) = p_value ? 1 : 0;
		pos += 1;
	}

	_FORCE_INLINE_ void write(int p_value) {
		write((uint32_t)p_value);
	}

	_FORCE_INLINE_ void write(uint64_t p_value) {
		DEV_ASSERT(pos + sizeof(uint64_t) <= length);
		pos += encode_uint64(p_value, data + pos);
	}

	_FORCE_INLINE_ void write(float p_value) {
		DEV_ASSERT(pos + sizeof(float) <= length);
		pos += encode_float(p_value, data + pos);
	}

	_FORCE_INLINE_ void write(double p_value) {
		DEV_ASSERT(pos + sizeof(double) <= length);
		pos += encode_double(p_value, data + pos);
	}

	void write_compressed(CharString const &p_string) {
		write(p_string.length()); // Uncompressed size.

		DEV_ASSERT(pos + sizeof(uint32_t) + Compression::get_max_compressed_buffer_size(p_string.length(), Compression::MODE_ZSTD) <= length);

		// Save pointer for compressed size.
		uint8_t *dst_size_ptr = data + pos; // Compressed size.
		pos += sizeof(uint32_t);

		int dst_size = Compression::compress(data + pos, reinterpret_cast<uint8_t const *>(p_string.ptr()), p_string.length(), Compression::MODE_ZSTD);
		encode_uint32(dst_size, dst_size_ptr);
		pos += dst_size;
	}

	void write(CharString const &p_string) {
		write_buffer(reinterpret_cast<const uint8_t *>(p_string.ptr()), p_string.length());
	}

	template <typename T>
	void write(VectorView<T> p_vector) {
		write(p_vector.size());
		for (uint32_t i = 0; i < p_vector.size(); i++) {
			T const &e = p_vector[i];
			write(e);
		}
	}

	void write(VectorView<uint8_t> p_vector) {
		write_buffer(p_vector.ptr(), p_vector.size());
	}

	template <typename K, typename V>
	void write(HashMap<K, V> const &p_map) {
		write(p_map.size());
		for (KeyValue<K, V> const &e : p_map) {
			write(e.key);
			write(e.value);
		}
	}

	uint64_t get_pos() const {
		return pos;
	}

	uint64_t get_length() const {
		return length;
	}

private:
	void write_buffer(uint8_t const *p_buffer, uint32_t p_length) {
		write(p_length);

		DEV_ASSERT(pos + p_length <= length);
		memcpy(data + pos, p_buffer, p_length);
		pos += p_length;
	}
};

class BufReader;

template <typename T>
concept Deserializable = requires(T t, BufReader &p_reader) {
	{
		t.serialize_size()
	} -> std::same_as<size_t>;
	{
		t.deserialize(p_reader)
	} -> std::same_as<void>;
};

class BufReader {
	uint8_t const *data = nullptr;
	uint64_t length = 0;
	uint64_t pos = 0;

	bool check_length(size_t p_size) {
		if (status != Status::OK) {
			return false;
		}

		if (pos + p_size > length) {
			status = Status::SHORT_BUFFER;
			return false;
		}
		return true;
	}

#define CHECK(p_size)          \
	if (!check_length(p_size)) \
	return

public:
	enum class Status {
		OK,
		SHORT_BUFFER,
		BAD_COMPRESSION,
	};

	Status status = Status::OK;

	BufReader(uint8_t const *p_data, uint64_t p_length) :
			data(p_data), length(p_length) {}

	template <Deserializable T>
	void read(T &p_value) {
		p_value.deserialize(*this);
	}

	_FORCE_INLINE_ void read(uint32_t &p_val) {
		CHECK(sizeof(uint32_t));

		p_val = decode_uint32(data + pos);
		pos += sizeof(uint32_t);
	}

	_FORCE_INLINE_ void read(RD::ShaderStage &p_val) {
		uint32_t val;
		read(val);
		p_val = (RD::ShaderStage)val;
	}

	_FORCE_INLINE_ void read(bool &p_val) {
		CHECK(sizeof(uint8_t));

		p_val = *(data + pos) > 0;
		pos += 1;
	}

	_FORCE_INLINE_ void read(uint64_t &p_val) {
		CHECK(sizeof(uint64_t));

		p_val = decode_uint64(data + pos);
		pos += sizeof(uint64_t);
	}

	_FORCE_INLINE_ void read(float &p_val) {
		CHECK(sizeof(float));

		p_val = decode_float(data + pos);
		pos += sizeof(float);
	}

	_FORCE_INLINE_ void read(double &p_val) {
		CHECK(sizeof(double));

		p_val = decode_double(data + pos);
		pos += sizeof(double);
	}

	void read(CharString &p_val) {
		uint32_t len;
		read(len);
		CHECK(len);
		p_val.resize(len + 1 /* NUL */);
		memcpy(p_val.ptrw(), data + pos, len);
		p_val.set(len, 0);
		pos += len;
	}

	void read_compressed(CharString &p_val) {
		uint32_t len;
		read(len);
		uint32_t comp_size;
		read(comp_size);

		CHECK(comp_size);

		p_val.resize(len + 1 /* NUL */);
		uint32_t bytes = (uint32_t)Compression::decompress(reinterpret_cast<uint8_t *>(p_val.ptrw()), len, data + pos, comp_size, Compression::MODE_ZSTD);
		if (bytes != len) {
			status = Status::BAD_COMPRESSION;
			return;
		}
		p_val.set(len, 0);
		pos += comp_size;
	}

	void read(LocalVector<uint8_t> &p_val) {
		uint32_t len;
		read(len);
		CHECK(len);
		p_val.resize(len);
		memcpy(p_val.ptr(), data + pos, len);
		pos += len;
	}

	template <typename T>
	void read(LocalVector<T> &p_val) {
		uint32_t len;
		read(len);
		CHECK(len);
		p_val.resize(len);
		for (uint32_t i = 0; i < len; i++) {
			read(p_val[i]);
		}
	}

	template <typename K, typename V>
	void read(HashMap<K, V> &p_map) {
		uint32_t len;
		read(len);
		CHECK(len);
		p_map.reserve(len);
		for (uint32_t i = 0; i < len; i++) {
			K key;
			read(key);
			V value;
			read(value);
			p_map[key] = value;
		}
	}

#undef CHECK
};

struct ComputeSize {
	uint32_t x = 0;
	uint32_t y = 0;
	uint32_t z = 0;

	size_t serialize_size() const {
		return sizeof(uint32_t) * 3;
	}

	void serialize(BufWriter &p_writer) const {
		p_writer.write(x);
		p_writer.write(y);
		p_writer.write(z);
	}

	void deserialize(BufReader &p_reader) {
		p_reader.read(x);
		p_reader.read(y);
		p_reader.read(z);
	}
};

struct ShaderStageData {
	RD::ShaderStage stage = RD::ShaderStage::SHADER_STAGE_MAX;
	uint32_t is_position_invariant = UINT32_MAX;
	uint32_t supports_fast_math = UINT32_MAX;
	CharString entry_point_name;
	CharString source;
	LocalVector<uint8_t> binary;
	SHA256Digest key;
	bool binary_mode = false;

	size_t serialize_size() const {
		size_t result = sizeof(uint32_t) // Stage.
				+ sizeof(uint32_t) // is_position_invariant
				+ sizeof(uint32_t) // supports_fast_math
				+ sizeof(uint32_t) /* entry_point_name.utf8().length */
				+ entry_point_name.length() + sizeof(uint8_t) // binary_mode;
				+ sizeof(uint32_t) // key size
				+ SHA256Digest::serialized_size();
		if (binary_mode) {
			result += sizeof(uint32_t) /* binary size */ + binary.size();
		} else {
			int comp_size = Compression::get_max_compressed_buffer_size(source.length(), Compression::MODE_ZSTD);
			result += sizeof(uint32_t) /* uncompressed size */ + sizeof(uint32_t) /* compressed size */ + comp_size;
		}
		return result;
	}

	void serialize(BufWriter &p_writer) const {
		p_writer.write((uint32_t)stage);
		p_writer.write(is_position_invariant);
		p_writer.write(supports_fast_math);
		p_writer.write(entry_point_name);
		p_writer.write(binary_mode);
		p_writer.write(key.serialize());
		if (binary_mode) {
			p_writer.write(binary);
		} else {
			p_writer.write_compressed(source);
		}
	}

	void deserialize(BufReader &p_reader) {
		p_reader.read((uint32_t &)stage);
		p_reader.read(is_position_invariant);
		p_reader.read(supports_fast_math);
		p_reader.read(entry_point_name);
		p_reader.read(binary_mode);
		LocalVector<uint8_t> key_ser;
		p_reader.read(key_ser);
		key = SHA256Digest((const char *)key_ser.ptr());
		if (binary_mode) {
			p_reader.read(binary);
		} else {
			p_reader.read_compressed(source);
		}
	}
};

struct SpecializationConstantData {
	uint32_t constant_id = UINT32_MAX;
	RD::PipelineSpecializationConstantType type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT;
	ShaderStageUsage stages = ShaderStageUsage::None;
	// Specifies the stages the constant is used by Metal.
	ShaderStageUsage used_stages = ShaderStageUsage::None;
	uint32_t int_value = UINT32_MAX;

	size_t serialize_size() const {
		return sizeof(constant_id) + sizeof(uint32_t) // type
				+ sizeof(stages) + sizeof(used_stages) // used_stages
				+ sizeof(int_value); // int_value
	}

	void serialize(BufWriter &p_writer) const {
		p_writer.write(constant_id);
		p_writer.write((uint32_t)type);
		p_writer.write(stages);
		p_writer.write(used_stages);
		p_writer.write(int_value);
	}

	void deserialize(BufReader &p_reader) {
		p_reader.read(constant_id);
		p_reader.read((uint32_t &)type);
		p_reader.read((uint32_t &)stages);
		p_reader.read((uint32_t &)used_stages);
		p_reader.read(int_value);
	}
};

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) UniformData {
	RD::UniformType type = RD::UniformType::UNIFORM_TYPE_MAX;
	uint32_t binding = UINT32_MAX;
	bool writable = false;
	uint32_t length = UINT32_MAX;
	ShaderStageUsage stages = ShaderStageUsage::None;
	// Specifies the stages the uniform data is
	// used by the Metal shader.
	ShaderStageUsage active_stages = ShaderStageUsage::None;
	BindingInfoMap bindings;
	BindingInfoMap bindings_secondary;

	size_t serialize_size() const {
		size_t size = 0;
		size += sizeof(uint32_t); // type
		size += sizeof(uint32_t); // binding
		size += sizeof(uint32_t); // writable
		size += sizeof(uint32_t); // length
		size += sizeof(uint32_t); // stages
		size += sizeof(uint32_t); // active_stages
		size += sizeof(uint32_t); // bindings.size()
		size += sizeof(uint32_t) * bindings.size(); // Total size of keys.
		for (KeyValue<RD::ShaderStage, BindingInfo> const &e : bindings) {
			size += e.value.serialize_size();
		}
		size += sizeof(uint32_t); // bindings_secondary.size()
		size += sizeof(uint32_t) * bindings_secondary.size(); // Total size of keys.
		for (KeyValue<RD::ShaderStage, BindingInfo> const &e : bindings_secondary) {
			size += e.value.serialize_size();
		}
		return size;
	}

	void serialize(BufWriter &p_writer) const {
		p_writer.write((uint32_t)type);
		p_writer.write(binding);
		p_writer.write(writable);
		p_writer.write(length);
		p_writer.write(stages);
		p_writer.write(active_stages);
		p_writer.write(bindings);
		p_writer.write(bindings_secondary);
	}

	void deserialize(BufReader &p_reader) {
		p_reader.read((uint32_t &)type);
		p_reader.read(binding);
		p_reader.read(writable);
		p_reader.read(length);
		p_reader.read((uint32_t &)stages);
		p_reader.read((uint32_t &)active_stages);
		p_reader.read(bindings);
		p_reader.read(bindings_secondary);
	}
};

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) UniformSetData {
	uint32_t index = UINT32_MAX;
	LocalVector<UniformData> uniforms;

	size_t serialize_size() const {
		size_t size = 0;
		size += sizeof(uint32_t); // index
		size += sizeof(uint32_t); // uniforms.size()
		for (UniformData const &e : uniforms) {
			size += e.serialize_size();
		}
		return size;
	}

	void serialize(BufWriter &p_writer) const {
		p_writer.write(index);
		p_writer.write(VectorView(uniforms));
	}

	void deserialize(BufReader &p_reader) {
		p_reader.read(index);
		p_reader.read(uniforms);
	}
	UniformSetData() = default;
	UniformSetData(uint32_t p_index) :
			index(p_index) {}
};

struct PushConstantData {
	uint32_t size = UINT32_MAX;
	ShaderStageUsage stages = ShaderStageUsage::None;
	ShaderStageUsage used_stages = ShaderStageUsage::None;
	HashMap<RD::ShaderStage, uint32_t> msl_binding;

	size_t serialize_size() const {
		return sizeof(uint32_t) // size
				+ sizeof(uint32_t) // stages
				+ sizeof(uint32_t) // used_stages
				+ sizeof(uint32_t) // msl_binding.size()
				+ sizeof(uint32_t) * msl_binding.size() // keys
				+ sizeof(uint32_t) * msl_binding.size(); // values
	}

	void serialize(BufWriter &p_writer) const {
		p_writer.write(size);
		p_writer.write((uint32_t)stages);
		p_writer.write((uint32_t)used_stages);
		p_writer.write(msl_binding);
	}

	void deserialize(BufReader &p_reader) {
		p_reader.read(size);
		p_reader.read((uint32_t &)stages);
		p_reader.read((uint32_t &)used_stages);
		p_reader.read(msl_binding);
	}
};

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) ShaderBinaryData {
	enum Flags : uint32_t {
		NONE = 0,
		NEEDS_VIEW_MASK_BUFFER = 1 << 0,
		USES_ARGUMENT_BUFFERS = 1 << 1,
	};
	CharString shader_name;
	// The Metal language version specified when compiling SPIR-V to MSL.
	// Format is major * 10000 + minor * 100 + patch.
	uint32_t msl_version = UINT32_MAX;
	uint32_t vertex_input_mask = UINT32_MAX;
	uint32_t fragment_output_mask = UINT32_MAX;
	uint32_t spirv_specialization_constants_ids_mask = UINT32_MAX;
	uint32_t flags = NONE;
	ComputeSize compute_local_size;
	PushConstantData push_constant;
	LocalVector<ShaderStageData> stages;
	LocalVector<SpecializationConstantData> constants;
	LocalVector<UniformSetData> uniforms;

	MTLLanguageVersion get_msl_version() const {
		uint32_t major = msl_version / 10000;
		uint32_t minor = (msl_version / 100) % 100;
		return MTLLanguageVersion((major << 0x10) + minor);
	}

	bool is_compute() const {
		return std::any_of(stages.begin(), stages.end(), [](ShaderStageData const &e) {
			return e.stage == RD::ShaderStage::SHADER_STAGE_COMPUTE;
		});
	}

	bool needs_view_mask_buffer() const {
		return flags & NEEDS_VIEW_MASK_BUFFER;
	}

	void set_needs_view_mask_buffer(bool p_value) {
		if (p_value) {
			flags |= NEEDS_VIEW_MASK_BUFFER;
		} else {
			flags &= ~NEEDS_VIEW_MASK_BUFFER;
		}
	}

	bool uses_argument_buffers() const {
		return flags & USES_ARGUMENT_BUFFERS;
	}

	void set_uses_argument_buffers(bool p_value) {
		if (p_value) {
			flags |= USES_ARGUMENT_BUFFERS;
		} else {
			flags &= ~USES_ARGUMENT_BUFFERS;
		}
	}

	size_t serialize_size() const {
		size_t size = 0;
		size += sizeof(uint32_t) + shader_name.length(); // shader_name
		size += sizeof(msl_version);
		size += sizeof(vertex_input_mask);
		size += sizeof(fragment_output_mask);
		size += sizeof(spirv_specialization_constants_ids_mask);
		size += sizeof(flags);
		size += compute_local_size.serialize_size();
		size += push_constant.serialize_size();
		size += sizeof(uint32_t); // stages.size()
		for (ShaderStageData const &e : stages) {
			size += e.serialize_size();
		}
		size += sizeof(uint32_t); // constants.size()
		for (SpecializationConstantData const &e : constants) {
			size += e.serialize_size();
		}
		size += sizeof(uint32_t); // uniforms.size()
		for (UniformSetData const &e : uniforms) {
			size += e.serialize_size();
		}
		return size;
	}

	void serialize(BufWriter &p_writer) const {
		p_writer.write(shader_name);
		p_writer.write(msl_version);
		p_writer.write(vertex_input_mask);
		p_writer.write(fragment_output_mask);
		p_writer.write(spirv_specialization_constants_ids_mask);
		p_writer.write(flags);
		p_writer.write(compute_local_size);
		p_writer.write(push_constant);
		p_writer.write(VectorView(stages));
		p_writer.write(VectorView(constants));
		p_writer.write(VectorView(uniforms));
	}

	void deserialize(BufReader &p_reader) {
		p_reader.read(shader_name);
		p_reader.read(msl_version);
		p_reader.read(vertex_input_mask);
		p_reader.read(fragment_output_mask);
		p_reader.read(spirv_specialization_constants_ids_mask);
		p_reader.read(flags);
		p_reader.read(compute_local_size);
		p_reader.read(push_constant);
		p_reader.read(stages);
		p_reader.read(constants);
		p_reader.read(uniforms);
	}
};

/// Contains additional metadata about the shader.
struct ShaderMeta {
	/// Indicates whether the shader uses multiview.
	bool has_multiview = false;
};

Error _reflect_spirv16(VectorView<RD::ShaderStageSPIRVData> p_spirv, RD::ShaderReflection &r_reflection, ShaderMeta &r_shader_meta) {
	using namespace spirv_cross;
	using spirv_cross::Resource;

	r_reflection = {};
	r_shader_meta = {};

	for (uint32_t i = 0; i < p_spirv.size(); i++) {
		RD::ShaderStageSPIRVData const &v = p_spirv[i];
		RD::ShaderStage stage = v.shader_stage;
		uint32_t const *const ir = reinterpret_cast<uint32_t const *const>(v.spirv.ptr());
		size_t word_count = v.spirv.size() / sizeof(uint32_t);
		Parser parser(ir, word_count);
		try {
			parser.parse();
		} catch (CompilerError &e) {
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Failed to parse IR at stage " + String(RD::SHADER_STAGE_NAMES[stage]) + ": " + e.what());
		}

		RD::ShaderStage stage_flag = (RD::ShaderStage)(1 << p_spirv[i].shader_stage);

		if (p_spirv[i].shader_stage == RD::SHADER_STAGE_COMPUTE) {
			r_reflection.is_compute = true;
			ERR_FAIL_COND_V_MSG(p_spirv.size() != 1, FAILED,
					"Compute shaders can only receive one stage, dedicated to compute.");
		}
		ERR_FAIL_COND_V_MSG(r_reflection.stages_bits.has_flag(stage_flag), FAILED,
				"Stage " + String(RD::SHADER_STAGE_NAMES[p_spirv[i].shader_stage]) + " submitted more than once.");

		ParsedIR &pir = parser.get_parsed_ir();
		using BT = SPIRType::BaseType;

		Compiler compiler(std::move(pir));

		if (r_reflection.is_compute) {
			r_reflection.compute_local_size[0] = compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, 0);
			r_reflection.compute_local_size[1] = compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, 1);
			r_reflection.compute_local_size[2] = compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, 2);
		}

		// Parse bindings.

		auto get_decoration = [&compiler](spirv_cross::ID id, spv::Decoration decoration) {
			uint32_t res = -1;
			if (compiler.has_decoration(id, decoration)) {
				res = compiler.get_decoration(id, decoration);
			}
			return res;
		};

		// Always clearer than a boolean.
		enum class Writable {
			No,
			Maybe,
		};

		// clang-format off
		enum {
		  SPIRV_WORD_SIZE      = sizeof(uint32_t),
		  SPIRV_DATA_ALIGNMENT = 4 * SPIRV_WORD_SIZE,
		};
		// clang-format on

		auto process_uniforms = [&r_reflection, &compiler, &get_decoration, stage, stage_flag](SmallVector<Resource> &resources, Writable writable, std::function<RDD::UniformType(SPIRType const &)> uniform_type) {
			for (Resource const &res : resources) {
				RD::ShaderUniform uniform;

				std::string const &name = compiler.get_name(res.id);
				uint32_t set = get_decoration(res.id, spv::DecorationDescriptorSet);
				ERR_FAIL_COND_V_MSG(set == (uint32_t)-1, FAILED, "No descriptor set found");
				ERR_FAIL_COND_V_MSG(set >= RenderingDeviceCommons::MAX_UNIFORM_SETS, FAILED, "On shader stage '" + String(RD::SHADER_STAGE_NAMES[stage]) + "', uniform '" + name.c_str() + "' uses a set (" + itos(set) + ") index larger than what is supported (" + itos(RenderingDeviceCommons::MAX_UNIFORM_SETS) + ").");

				uniform.binding = get_decoration(res.id, spv::DecorationBinding);
				ERR_FAIL_COND_V_MSG(uniform.binding == (uint32_t)-1, FAILED, "No binding found");

				SPIRType const &a_type = compiler.get_type(res.type_id);
				uniform.type = uniform_type(a_type);

				// Update length.
				switch (a_type.basetype) {
					case BT::Struct: {
						if (uniform.type == RD::UNIFORM_TYPE_STORAGE_BUFFER) {
							// Consistent with spirv_reflect.
							uniform.length = 0;
						} else {
							uniform.length = round_up_to_alignment(compiler.get_declared_struct_size(a_type), SPIRV_DATA_ALIGNMENT);
						}
					} break;
					case BT::Image:
					case BT::Sampler:
					case BT::SampledImage: {
						uniform.length = 1;
						for (uint32_t const &a : a_type.array) {
							uniform.length *= a;
						}
					} break;
					default:
						break;
				}

				// Update writable.
				if (writable == Writable::Maybe) {
					if (a_type.basetype == BT::Struct) {
						Bitset flags = compiler.get_buffer_block_flags(res.id);
						uniform.writable = !compiler.has_decoration(res.id, spv::DecorationNonWritable) && !flags.get(spv::DecorationNonWritable);
					} else if (a_type.basetype == BT::Image) {
						if (a_type.image.access == spv::AccessQualifierMax) {
							uniform.writable = !compiler.has_decoration(res.id, spv::DecorationNonWritable);
						} else {
							uniform.writable = a_type.image.access != spv::AccessQualifierReadOnly;
						}
					}
				}

				if (set < (uint32_t)r_reflection.uniform_sets.size()) {
					// Check if this already exists.
					bool exists = false;
					for (uint32_t k = 0; k < r_reflection.uniform_sets[set].size(); k++) {
						if (r_reflection.uniform_sets[set][k].binding == uniform.binding) {
							// Already exists, verify that it's the same type.
							ERR_FAIL_COND_V_MSG(r_reflection.uniform_sets[set][k].type != uniform.type, FAILED,
									"On shader stage '" + String(RD::SHADER_STAGE_NAMES[stage]) + "', uniform '" + name.c_str() + "' trying to reuse location for set=" + itos(set) + ", binding=" + itos(uniform.binding) + " with different uniform type.");

							// Also, verify that it's the same size.
							ERR_FAIL_COND_V_MSG(r_reflection.uniform_sets[set][k].length != uniform.length, FAILED,
									"On shader stage '" + String(RD::SHADER_STAGE_NAMES[stage]) + "', uniform '" + name.c_str() + "' trying to reuse location for set=" + itos(set) + ", binding=" + itos(uniform.binding) + " with different uniform size.");

							// Also, verify that it has the same writability.
							ERR_FAIL_COND_V_MSG(r_reflection.uniform_sets[set][k].writable != uniform.writable, FAILED,
									"On shader stage '" + String(RD::SHADER_STAGE_NAMES[stage]) + "', uniform '" + name.c_str() + "' trying to reuse location for set=" + itos(set) + ", binding=" + itos(uniform.binding) + " with different writability.");

							// Just append stage mask and continue.
							r_reflection.uniform_sets.write[set].write[k].stages.set_flag(stage_flag);
							exists = true;
							break;
						}
					}

					if (exists) {
						continue; // Merged.
					}
				}

				uniform.stages.set_flag(stage_flag);

				if (set >= (uint32_t)r_reflection.uniform_sets.size()) {
					r_reflection.uniform_sets.resize(set + 1);
				}

				r_reflection.uniform_sets.write[set].push_back(uniform);
			}

			return OK;
		};

		ShaderResources resources = compiler.get_shader_resources();

		process_uniforms(resources.uniform_buffers, Writable::No, [](SPIRType const &a_type) {
			DEV_ASSERT(a_type.basetype == BT::Struct);
			return RD::UNIFORM_TYPE_UNIFORM_BUFFER;
		});

		process_uniforms(resources.storage_buffers, Writable::Maybe, [](SPIRType const &a_type) {
			DEV_ASSERT(a_type.basetype == BT::Struct);
			return RD::UNIFORM_TYPE_STORAGE_BUFFER;
		});

		process_uniforms(resources.storage_images, Writable::Maybe, [](SPIRType const &a_type) {
			DEV_ASSERT(a_type.basetype == BT::Image);
			if (a_type.image.dim == spv::DimBuffer) {
				return RD::UNIFORM_TYPE_IMAGE_BUFFER;
			} else {
				return RD::UNIFORM_TYPE_IMAGE;
			}
		});

		process_uniforms(resources.sampled_images, Writable::No, [](SPIRType const &a_type) {
			DEV_ASSERT(a_type.basetype == BT::SampledImage);
			return RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
		});

		process_uniforms(resources.separate_images, Writable::No, [](SPIRType const &a_type) {
			DEV_ASSERT(a_type.basetype == BT::Image);
			if (a_type.image.dim == spv::DimBuffer) {
				return RD::UNIFORM_TYPE_TEXTURE_BUFFER;
			} else {
				return RD::UNIFORM_TYPE_TEXTURE;
			}
		});

		process_uniforms(resources.separate_samplers, Writable::No, [](SPIRType const &a_type) {
			DEV_ASSERT(a_type.basetype == BT::Sampler);
			return RD::UNIFORM_TYPE_SAMPLER;
		});

		process_uniforms(resources.subpass_inputs, Writable::No, [](SPIRType const &a_type) {
			DEV_ASSERT(a_type.basetype == BT::Image && a_type.image.dim == spv::DimSubpassData);
			return RD::UNIFORM_TYPE_INPUT_ATTACHMENT;
		});

		if (!resources.push_constant_buffers.empty()) {
			// There can be only one push constant block.
			Resource const &res = resources.push_constant_buffers.front();

			size_t push_constant_size = round_up_to_alignment(compiler.get_declared_struct_size(compiler.get_type(res.base_type_id)), SPIRV_DATA_ALIGNMENT);
			ERR_FAIL_COND_V_MSG(r_reflection.push_constant_size && r_reflection.push_constant_size != push_constant_size, FAILED,
					"Reflection of SPIR-V shader stage '" + String(RD::SHADER_STAGE_NAMES[p_spirv[i].shader_stage]) + "': Push constant block must be the same across shader stages.");

			r_reflection.push_constant_size = push_constant_size;
			r_reflection.push_constant_stages.set_flag(stage_flag);
		}

		ERR_FAIL_COND_V_MSG(!resources.atomic_counters.empty(), FAILED, "Atomic counters not supported");
		ERR_FAIL_COND_V_MSG(!resources.acceleration_structures.empty(), FAILED, "Acceleration structures not supported");
		ERR_FAIL_COND_V_MSG(!resources.shader_record_buffers.empty(), FAILED, "Shader record buffers not supported");

		if (stage == RD::SHADER_STAGE_VERTEX && !resources.stage_inputs.empty()) {
			for (Resource const &res : resources.stage_inputs) {
				SPIRType a_type = compiler.get_type(res.base_type_id);
				uint32_t loc = get_decoration(res.id, spv::DecorationLocation);
				if (loc != (uint32_t)-1) {
					r_reflection.vertex_input_mask |= 1 << loc;
				}
			}
		}

		if (stage == RD::SHADER_STAGE_FRAGMENT && !resources.stage_outputs.empty()) {
			for (Resource const &res : resources.stage_outputs) {
				SPIRType a_type = compiler.get_type(res.base_type_id);
				uint32_t loc = get_decoration(res.id, spv::DecorationLocation);
				uint32_t built_in = spv::BuiltIn(get_decoration(res.id, spv::DecorationBuiltIn));
				if (loc != (uint32_t)-1 && built_in != spv::BuiltInFragDepth) {
					r_reflection.fragment_output_mask |= 1 << loc;
				}
			}
		}

		for (const BuiltInResource &res : resources.builtin_inputs) {
			if (res.builtin == spv::BuiltInViewIndex || res.builtin == spv::BuiltInViewportIndex) {
				r_shader_meta.has_multiview = true;
			}
		}

		if (!r_shader_meta.has_multiview) {
			for (const BuiltInResource &res : resources.builtin_outputs) {
				if (res.builtin == spv::BuiltInViewIndex || res.builtin == spv::BuiltInViewportIndex) {
					r_shader_meta.has_multiview = true;
				}
			}
		}

		// Specialization constants.
		for (SpecializationConstant const &constant : compiler.get_specialization_constants()) {
			int32_t existing = -1;
			RD::ShaderSpecializationConstant sconst;
			SPIRConstant &spc = compiler.get_constant(constant.id);
			SPIRType const &spct = compiler.get_type(spc.constant_type);

			sconst.constant_id = constant.constant_id;
			sconst.int_value = 0;

			switch (spct.basetype) {
				case BT::Boolean: {
					sconst.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
					sconst.bool_value = spc.scalar() != 0;
				} break;
				case BT::Int:
				case BT::UInt: {
					sconst.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;
					sconst.int_value = spc.scalar();
				} break;
				case BT::Float: {
					sconst.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT;
					sconst.float_value = spc.scalar_f32();
				} break;
				default:
					ERR_FAIL_V_MSG(FAILED, "Unsupported specialization constant type");
			}
			sconst.stages.set_flag(stage_flag);

			for (uint32_t k = 0; k < r_reflection.specialization_constants.size(); k++) {
				if (r_reflection.specialization_constants[k].constant_id == sconst.constant_id) {
					ERR_FAIL_COND_V_MSG(r_reflection.specialization_constants[k].type != sconst.type, FAILED, "More than one specialization constant used for id (" + itos(sconst.constant_id) + "), but their types differ.");
					ERR_FAIL_COND_V_MSG(r_reflection.specialization_constants[k].int_value != sconst.int_value, FAILED, "More than one specialization constant used for id (" + itos(sconst.constant_id) + "), but their default values differ.");
					existing = k;
					break;
				}
			}

			if (existing > 0) {
				r_reflection.specialization_constants.write[existing].stages.set_flag(stage_flag);
			} else {
				r_reflection.specialization_constants.push_back(sconst);
			}
		}

		r_reflection.stages_bits.set_flag(stage_flag);
	}

	// Sort all uniform_sets.
	for (uint32_t i = 0; i < r_reflection.uniform_sets.size(); i++) {
		r_reflection.uniform_sets.write[i].sort();
	}

	return OK;
}

Error compile_metal_source(String p_shader_name, const char *p_source, ShaderStageData &p_stage_data) {
	//static Mutex mutex;
	//MutexLock mLock(mutex);
	Error r_error;
	Ref<FileAccess> source_file = FileAccess::create_temp(FileAccess::ModeFlags::READ_WRITE,
			p_shader_name + "_" + itos(p_stage_data.key.short_sha()),
			"metal", false, &r_error);
	if (r_error != OK) {
		ERR_FAIL_V_MSG(r_error, "Unable to create temporary source file");
	}
	if (!source_file->store_buffer((const uint8_t *)p_source, strlen(p_source))) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Unable to write temporary source file");
	}
	source_file->flush();
	Ref<FileAccess> result_file = FileAccess::create_temp(FileAccess::ModeFlags::READ_WRITE,
			p_shader_name + "_" + itos(p_stage_data.key.short_sha()),
			"metallib", false, &r_error);

	if (r_error != OK) {
		ERR_FAIL_V_MSG(r_error, "Unable to create temporary target file");
	}
	{
		List<String> args{ "-sdk", "iphoneos", "metal", "-Os", "-target", "air64-apple-ios16.0" };
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
			ERR_PRINT(vformat("Metal Compiler exited with error code: %d", exit_code));
		}
		int len = result_file->get_length();
		if (len == 0) {
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Metal Compiler created empty library");
		}
	}
	{
		List<String> args{ "-sdk", "iphoneos", "metal-dsymutil", "-remove-source", result_file->get_path_absolute() };
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
		if (len == 0) {
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "metal-dsymutil tool created empty library");
		}
	}
	p_stage_data.binary = result_file->get_buffer(result_file->get_length());
	p_stage_data.binary_mode = true;
	return OK;
}

Vector<uint8_t> shader_compile_binary_from_spirv(MetalDeviceProperties *device_properties, const Vector<RD::ShaderStageSPIRVData> &p_spirv, const String &p_shader_name, bool p_export_mode) {
	using Result = ::Vector<uint8_t>;
	using namespace spirv_cross;
	using spirv_cross::CompilerMSL;
	using spirv_cross::Resource;

	RD::ShaderReflection spirv_data;
	ShaderMeta shader_meta;
	ERR_FAIL_COND_V(_reflect_spirv16(p_spirv, spirv_data, shader_meta), Result());

	ShaderBinaryData bin_data{};
	if (!p_shader_name.is_empty()) {
		bin_data.shader_name = p_shader_name.utf8();
	} else {
		bin_data.shader_name = "unnamed";
	}

	bin_data.vertex_input_mask = spirv_data.vertex_input_mask;
	bin_data.fragment_output_mask = spirv_data.fragment_output_mask;
	bin_data.compute_local_size = ComputeSize{
		.x = spirv_data.compute_local_size[0],
		.y = spirv_data.compute_local_size[1],
		.z = spirv_data.compute_local_size[2],
	};
	bin_data.push_constant.size = spirv_data.push_constant_size;
	bin_data.push_constant.stages = (ShaderStageUsage)(uint8_t)spirv_data.push_constant_stages;
	bin_data.set_needs_view_mask_buffer(shader_meta.has_multiview);

	for (uint32_t i = 0; i < spirv_data.uniform_sets.size(); i++) {
		const ::Vector<RD::ShaderUniform> &spirv_set = spirv_data.uniform_sets[i];
		UniformSetData set(i);
		for (const RD::ShaderUniform &spirv_uniform : spirv_set) {
			UniformData binding{};
			binding.type = spirv_uniform.type;
			binding.binding = spirv_uniform.binding;
			binding.writable = spirv_uniform.writable;
			binding.stages = (ShaderStageUsage)(uint8_t)spirv_uniform.stages;
			binding.length = spirv_uniform.length;
			set.uniforms.push_back(binding);
		}
		bin_data.uniforms.push_back(set);
	}

	for (const RD::ShaderSpecializationConstant &spirv_sc : spirv_data.specialization_constants) {
		SpecializationConstantData spec_constant{};
		spec_constant.type = spirv_sc.type;
		spec_constant.constant_id = spirv_sc.constant_id;
		spec_constant.int_value = spirv_sc.int_value;
		spec_constant.stages = (ShaderStageUsage)(uint8_t)spirv_sc.stages;
		bin_data.constants.push_back(spec_constant);
		bin_data.spirv_specialization_constants_ids_mask |= (1 << spirv_sc.constant_id);
	}

	// Reflection using SPIRV-Cross:
	// https://github.com/KhronosGroup/SPIRV-Cross/wiki/Reflection-API-user-guide

	MTLCompileOptions *compile_options = [MTLCompileOptions new];
	uint32_t version_major = (compile_options.languageVersion >> 0x10) & 0xff;
	uint32_t version_minor = (compile_options.languageVersion >> 0x00) & 0xff;

	CompilerMSL::Options msl_options{};
	msl_options.set_msl_version(version_major, version_minor);
	bin_data.msl_version = msl_options.msl_version;
#if TARGET_OS_OSX
	if (p_export_mode) {
		msl_options.platform = CompilerMSL::Options::iOS;
	} else {
		msl_options.platform = CompilerMSL::Options::macOS;
	}
#else
	msl_options.platform = CompilerMSL::Options::iOS;
#endif

	if (p_export_mode) {
		msl_options.ios_use_simdgroup_functions = (*device_properties).features.simdPermute;
		msl_options.ios_support_base_vertex_instance = true;
	}
#if TARGET_OS_IPHONE
	msl_options.ios_use_simdgroup_functions = (*device_properties).features.simdPermute;
	msl_options.ios_support_base_vertex_instance = true;
#endif

	bool disable_argument_buffers = false;
	if (String v = OS::get_singleton()->get_environment(U"GODOT_DISABLE_ARGUMENT_BUFFERS"); v == U"1") {
		disable_argument_buffers = true;
	}

	if (device_properties->features.argument_buffers_tier >= MTLArgumentBuffersTier2 && !disable_argument_buffers) {
		msl_options.argument_buffers_tier = CompilerMSL::Options::ArgumentBuffersTier::Tier2;
		msl_options.argument_buffers = true;
		bin_data.set_uses_argument_buffers(true);
	} else {
		msl_options.argument_buffers_tier = CompilerMSL::Options::ArgumentBuffersTier::Tier1;
		// Tier 1 argument buffers don't support writable textures, so we disable them completely.
		msl_options.argument_buffers = false;
		bin_data.set_uses_argument_buffers(false);
	}
	msl_options.force_active_argument_buffer_resources = true;
	// We can't use this, as we have to add the descriptor sets via compiler.add_msl_resource_binding.
	// msl_options.pad_argument_buffer_resources = true;
	msl_options.texture_buffer_native = true; // Enable texture buffer support.
	msl_options.use_framebuffer_fetch_subpasses = false;
	msl_options.pad_fragment_output_components = true;
	msl_options.r32ui_alignment_constant_id = R32UI_ALIGNMENT_CONSTANT_ID;
	msl_options.agx_manual_cube_grad_fixup = true;
	if (shader_meta.has_multiview) {
		msl_options.multiview = true;
		msl_options.multiview_layered_rendering = true;
		msl_options.view_mask_buffer_index = VIEW_MASK_BUFFER_INDEX;
	}

	CompilerGLSL::Options options{};
	options.vertex.flip_vert_y = true;
#if DEV_ENABLED
	options.emit_line_directives = true;
#endif

	for (uint32_t i = 0; i < p_spirv.size(); i++) {
		RD::ShaderStageSPIRVData const &v = p_spirv[i];
		RD::ShaderStage stage = v.shader_stage;
		char const *stage_name = RD::SHADER_STAGE_NAMES[stage];
		uint32_t const *const ir = reinterpret_cast<uint32_t const *const>(v.spirv.ptr());
		size_t word_count = v.spirv.size() / sizeof(uint32_t);
		Parser parser(ir, word_count);
		try {
			parser.parse();
		} catch (CompilerError &e) {
			ERR_FAIL_V_MSG(Result(), "Failed to parse IR at stage " + String(RD::SHADER_STAGE_NAMES[stage]) + ": " + e.what());
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
			ERR_FAIL_V_MSG(Result(), "Failed to compile stage " + String(RD::SHADER_STAGE_NAMES[stage]) + ": " + e.what());
		}

		ERR_FAIL_COND_V_MSG(compiler.get_entry_points_and_stages().size() != 1, Result(), "Expected a single entry point and stage.");

		SmallVector<EntryPoint> entry_pts_stages = compiler.get_entry_points_and_stages();
		EntryPoint &entry_point_stage = entry_pts_stages.front();
		SPIREntryPoint &entry_point = compiler.get_entry_point(entry_point_stage.name, entry_point_stage.execution_model);

		// Process specialization constants.
		if (!compiler.get_specialization_constants().empty()) {
			for (SpecializationConstant const &constant : compiler.get_specialization_constants()) {
				LocalVector<SpecializationConstantData>::Iterator res = bin_data.constants.begin();
				while (res != bin_data.constants.end()) {
					if (res->constant_id == constant.constant_id) {
						res->used_stages |= 1 << stage;
						break;
					}
					++res;
				}
				if (res == bin_data.constants.end()) {
					WARN_PRINT(String(stage_name) + ": unable to find constant_id: " + itos(constant.constant_id));
				}
			}
		}

		// Process bindings.

		LocalVector<UniformSetData> &uniform_sets = bin_data.uniforms;
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

		auto descriptor_bindings = [&compiler, &active, &uniform_sets, stage, &get_decoration](SmallVector<Resource> &p_resources, Writable p_writable) {
			for (Resource const &res : p_resources) {
				uint32_t dset = get_decoration(res.id, spv::DecorationDescriptorSet);
				uint32_t dbin = get_decoration(res.id, spv::DecorationBinding);
				UniformData *found = nullptr;
				if (dset != (uint32_t)-1 && dbin != (uint32_t)-1 && dset < uniform_sets.size()) {
					UniformSetData &set = uniform_sets[dset];
					LocalVector<UniformData>::Iterator pos = set.uniforms.begin();
					while (pos != set.uniforms.end()) {
						if (dbin == pos->binding) {
							found = &(*pos);
							break;
						}
						++pos;
					}
				}

				ERR_FAIL_NULL_V_MSG(found, ERR_CANT_CREATE, "UniformData not found");

				bool is_active = active.find(res.id) != active.end();
				if (is_active) {
					found->active_stages |= 1 << stage;
				}

				BindingInfo primary{};

				SPIRType const &a_type = compiler.get_type(res.type_id);
				BT basetype = a_type.basetype;

				switch (basetype) {
					case BT::Struct: {
						primary.dataType = MTLDataTypePointer;
					} break;

					case BT::Image:
					case BT::SampledImage: {
						primary.dataType = MTLDataTypeTexture;
					} break;

					case BT::Sampler: {
						primary.dataType = MTLDataTypeSampler;
						primary.arrayLength = 1;
						for (uint32_t const &a : a_type.array) {
							primary.arrayLength *= a;
						}
					} break;

					default: {
						ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Unexpected BaseType");
					} break;
				}

				// Find array length of image.
				if (basetype == BT::Image || basetype == BT::SampledImage) {
					primary.arrayLength = 1;
					for (uint32_t const &a : a_type.array) {
						primary.arrayLength *= a;
					}
					primary.isMultisampled = a_type.image.ms;

					SPIRType::ImageType const &image = a_type.image;
					primary.imageFormat = image.format;

					switch (image.dim) {
						case spv::Dim1D: {
							if (image.arrayed) {
								primary.textureType = MTLTextureType1DArray;
							} else {
								primary.textureType = MTLTextureType1D;
							}
						} break;
						case spv::DimSubpassData: {
							DISPATCH_FALLTHROUGH;
						}
						case spv::Dim2D: {
							if (image.arrayed && image.ms) {
								primary.textureType = MTLTextureType2DMultisampleArray;
							} else if (image.arrayed) {
								primary.textureType = MTLTextureType2DArray;
							} else if (image.ms) {
								primary.textureType = MTLTextureType2DMultisample;
							} else {
								primary.textureType = MTLTextureType2D;
							}
						} break;
						case spv::Dim3D: {
							primary.textureType = MTLTextureType3D;
						} break;
						case spv::DimCube: {
							if (image.arrayed) {
								primary.textureType = MTLTextureTypeCube;
							}
						} break;
						case spv::DimRect: {
						} break;
						case spv::DimBuffer: {
							// VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
							primary.textureType = MTLTextureTypeTextureBuffer;
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
								DISPATCH_FALLTHROUGH;
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

				found->bindings[stage] = primary;

				// A sampled image contains two bindings, the primary
				// is to the image, and the secondary is to the associated sampler.
				if (basetype == BT::SampledImage) {
					uint32_t binding = compiler.get_automatic_msl_resource_binding_secondary(res.id);
					if (binding != (uint32_t)-1) {
						found->bindings_secondary[stage] = BindingInfo{
							.dataType = MTLDataTypeSampler,
							.index = binding,
							.access = MTLBindingAccessReadOnly,
						};
					}
				}

				// An image may have a secondary binding if it is used
				// for atomic operations.
				if (basetype == BT::Image) {
					uint32_t binding = compiler.get_automatic_msl_resource_binding_secondary(res.id);
					if (binding != (uint32_t)-1) {
						found->bindings_secondary[stage] = BindingInfo{
							.dataType = MTLDataTypePointer,
							.index = binding,
							.access = MTLBindingAccessReadWrite,
						};
					}
				}
			}
			return Error::OK;
		};

		if (!resources.uniform_buffers.empty()) {
			Error err = descriptor_bindings(resources.uniform_buffers, Writable::No);
			ERR_FAIL_COND_V(err != OK, Result());
		}
		if (!resources.storage_buffers.empty()) {
			Error err = descriptor_bindings(resources.storage_buffers, Writable::Maybe);
			ERR_FAIL_COND_V(err != OK, Result());
		}
		if (!resources.storage_images.empty()) {
			Error err = descriptor_bindings(resources.storage_images, Writable::Maybe);
			ERR_FAIL_COND_V(err != OK, Result());
		}
		if (!resources.sampled_images.empty()) {
			Error err = descriptor_bindings(resources.sampled_images, Writable::No);
			ERR_FAIL_COND_V(err != OK, Result());
		}
		if (!resources.separate_images.empty()) {
			Error err = descriptor_bindings(resources.separate_images, Writable::No);
			ERR_FAIL_COND_V(err != OK, Result());
		}
		if (!resources.separate_samplers.empty()) {
			Error err = descriptor_bindings(resources.separate_samplers, Writable::No);
			ERR_FAIL_COND_V(err != OK, Result());
		}
		if (!resources.subpass_inputs.empty()) {
			Error err = descriptor_bindings(resources.subpass_inputs, Writable::No);
			ERR_FAIL_COND_V(err != OK, Result());
		}

		if (!resources.push_constant_buffers.empty()) {
			for (Resource const &res : resources.push_constant_buffers) {
				uint32_t binding = compiler.get_automatic_msl_resource_binding(res.id);
				if (binding != (uint32_t)-1) {
					bin_data.push_constant.used_stages |= 1 << stage;
					bin_data.push_constant.msl_binding[stage] = binding;
				}
			}
		}

		ERR_FAIL_COND_V_MSG(!resources.atomic_counters.empty(), Result(), "Atomic counters not supported");
		ERR_FAIL_COND_V_MSG(!resources.acceleration_structures.empty(), Result(), "Acceleration structures not supported");
		ERR_FAIL_COND_V_MSG(!resources.shader_record_buffers.empty(), Result(), "Shader record buffers not supported");

		if (!resources.stage_inputs.empty()) {
			for (Resource const &res : resources.stage_inputs) {
				uint32_t binding = compiler.get_automatic_msl_resource_binding(res.id);
				if (binding != (uint32_t)-1) {
					bin_data.vertex_input_mask |= 1 << binding;
				}
			}
		}

		ShaderStageData stage_data;
		stage_data.stage = v.shader_stage;
		stage_data.is_position_invariant = compiler.is_position_invariant();
		stage_data.supports_fast_math = !entry_point.flags.get(spv::ExecutionModeSignedZeroInfNanPreserve);
		stage_data.entry_point_name = entry_point.name.c_str();
		stage_data.key = SHA256Digest(source.c_str(), source.length());
		if (p_export_mode) {
			// Try to compile the Metal source code
			Error compile_err = compile_metal_source(p_shader_name, source.c_str(), stage_data);
			if (compile_err == OK) {
				stage_data.binary_mode = true;
			}
		}
		if (!stage_data.binary_mode) {
			stage_data.source = source.c_str();
		}
		bin_data.stages.push_back(stage_data);
	}

	size_t vec_size = bin_data.serialize_size() + 8;

	::Vector<uint8_t> ret;
	ret.resize(vec_size);
	BufWriter writer(ret.ptrw(), vec_size);
	const uint8_t HEADER[4] = { 'G', 'M', 'S', 'L' };
	writer.write(*(uint32_t *)HEADER);
	writer.write(SHADER_BINARY_VERSION);
	bin_data.serialize(writer);
	ret.resize(writer.get_pos());

	return ret;
}

bool RenderingShaderContainerMetal::_set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) {
	PackedByteArray code_bytes;
	shaders.resize(1);
	Vector<uint8_t> bin = shader_compile_binary_from_spirv(owner->device_properties, p_spirv, String::utf8(shader_name), export_mode);
	RenderingShaderContainer::Shader &shader = shaders.ptrw()[0];
	shader.code_decompressed_size = 0;
	shader.code_compression_flags = 0;
	shader.code_compressed_bytes = bin;

	return true;
}

RDD::ShaderID RenderingShaderContainerMetal::create_shader(const Vector<RDD::ImmutableSampler> &p_immutable_samplers) {
	RenderingShaderContainer::Shader &container_shader = shaders.ptrw()[0];

	const uint8_t *binptr = container_shader.code_compressed_bytes.ptr();
	uint32_t binsize = container_shader.code_compressed_bytes.size();

	BufReader reader(binptr, binsize);
	uint8_t header[4];
	reader.read((uint32_t &)header);
	ERR_FAIL_COND_V_MSG(memcmp(header, "GMSL", 4) != 0, RDD::ShaderID(), "Invalid header");
	uint32_t version = 0;
	reader.read(version);
	ERR_FAIL_COND_V_MSG(version != SHADER_BINARY_VERSION, RDD::ShaderID(), "Invalid shader binary version");

	ShaderBinaryData binary_data;
	binary_data.deserialize(reader);
	switch (reader.status) {
		case BufReader::Status::OK:
			break;
		case BufReader::Status::BAD_COMPRESSION:
			ERR_FAIL_V_MSG(RDD::ShaderID(), "Invalid compressed data");
		case BufReader::Status::SHORT_BUFFER:
			ERR_FAIL_V_MSG(RDD::ShaderID(), "Unexpected end of buffer");
	}

	// We need to regenerate the shader if the cache is moved to an incompatible device.
	ERR_FAIL_COND_V_MSG(owner->device_properties->features.argument_buffers_tier < MTLArgumentBuffersTier2 && binary_data.uses_argument_buffers(),
			RDD::ShaderID(),
			"Shader was generated with argument buffers, but device has limited support");

	MTLCompileOptions *options = [MTLCompileOptions new];
	options.languageVersion = binary_data.get_msl_version();
	HashMap<RD::ShaderStage, MDLibrary *> libraries;

	for (ShaderStageData &shader_data : binary_data.stages) {
		if (ShaderCacheEntry **p = RenderingShaderContainerFormatMetal::_shader_cache.getptr(shader_data.key); p != nullptr) {
			libraries[shader_data.stage] = (*p)->library;
			continue;
		}

		ShaderCacheEntry *cd = memnew(ShaderCacheEntry(shader_data.key));
		cd->name = binary_data.shader_name;
		cd->stage = shader_data.stage;
		MDLibrary *library = nil;
		if (shader_data.binary_mode) {
			cd->binary = shader_data.binary;
			dispatch_data_t binary = dispatch_data_create(cd->binary.ptr(), cd->binary.size(), dispatch_get_main_queue(), ^{
																							   });
			library = [MDLibrary newLibraryWithCacheEntry:cd
												   device:owner->device_properties->device
													 data:binary];
		} else {
			NSString *source = [[NSString alloc] initWithBytes:(void *)shader_data.source.ptr()
														length:shader_data.source.length()
													  encoding:NSUTF8StringEncoding];
			options.preserveInvariance = shader_data.is_position_invariant;
			options.fastMathEnabled = YES;
			library = [MDLibrary newLibraryWithCacheEntry:cd
												   device:owner->device_properties->device
												   source:source
												  options:options
												 strategy:ShaderLoadStrategy::LAZY];
		}
		RenderingShaderContainerFormatMetal::_shader_cache[shader_data.key] = cd;
		libraries[shader_data.stage] = library;
	}

	Vector<UniformSet> uniform_sets;
	uniform_sets.resize(binary_data.uniforms.size());

	// r_shader_desc.uniform_sets.resize(binary_data.uniforms.size());

	// Create sets.
	for (UniformSetData &uniform_set : binary_data.uniforms) {
		UniformSet &set = uniform_sets.write[uniform_set.index];
		set.uniforms.resize(uniform_set.uniforms.size());

		// Vector<ShaderUniform> &uset = r_shader_desc.uniform_sets.write[uniform_set.index];
		// uset.resize(uniform_set.uniforms.size());

		for (uint32_t i = 0; i < uniform_set.uniforms.size(); i++) {
			UniformData &uniform = uniform_set.uniforms[i];

			RD::ShaderUniform su;
			su.type = uniform.type;
			su.writable = uniform.writable;
			su.length = uniform.length;
			su.binding = uniform.binding;
			su.stages = uniform.stages;
			// uset.write[i] = su;

			UniformInfo ui;
			ui.binding = uniform.binding;
			ui.active_stages = uniform.active_stages;
			for (KeyValue<RDC::ShaderStage, BindingInfo> &kv : uniform.bindings) {
				ui.bindings.insert(kv.key, kv.value);
			}
			for (KeyValue<RDC::ShaderStage, BindingInfo> &kv : uniform.bindings_secondary) {
				ui.bindings_secondary.insert(kv.key, kv.value);
			}
			set.uniforms[i] = ui;
		}
	}
	for (UniformSetData &uniform_set : binary_data.uniforms) {
		UniformSet &set = uniform_sets.write[uniform_set.index];

		// Make encoders.
		for (ShaderStageData const &stage_data : binary_data.stages) {
			RD::ShaderStage stage = stage_data.stage;
			NSMutableArray<MTLArgumentDescriptor *> *descriptors = [NSMutableArray new];

			for (UniformInfo const &uniform : set.uniforms) {
				BindingInfo const *binding_info = uniform.bindings.getptr(stage);
				if (binding_info == nullptr) {
					continue;
				}

				[descriptors addObject:binding_info->new_argument_descriptor()];
				BindingInfo const *secondary_binding_info = uniform.bindings_secondary.getptr(stage);
				if (secondary_binding_info != nullptr) {
					[descriptors addObject:secondary_binding_info->new_argument_descriptor()];
				}
			}

			if (descriptors.count == 0) {
				// No bindings.
				continue;
			}
			// Sort by index.
			[descriptors sortUsingComparator:^NSComparisonResult(MTLArgumentDescriptor *a, MTLArgumentDescriptor *b) {
				if (a.index < b.index) {
					return NSOrderedAscending;
				} else if (a.index > b.index) {
					return NSOrderedDescending;
				} else {
					return NSOrderedSame;
				}
			}];

			id<MTLArgumentEncoder> enc = [owner->device_properties->device newArgumentEncoderWithArguments:descriptors];
			set.encoders[stage] = enc;
			set.offsets[stage] = set.buffer_size;
			set.buffer_size += enc.encodedLength;
		}
	}

	// r_shader_desc.specialization_constants.resize(binary_data.constants.size());
	for (uint32_t i = 0; i < binary_data.constants.size(); i++) {
		SpecializationConstantData &c = binary_data.constants[i];

		RD::ShaderSpecializationConstant sc;
		sc.type = c.type;
		sc.constant_id = c.constant_id;
		sc.int_value = c.int_value;
		sc.stages = c.stages;
		// r_shader_desc.specialization_constants.write[i] = sc;
	}

	MDShader *shader = nullptr;
	if (binary_data.is_compute()) {
		MDComputeShader *cs = new MDComputeShader(
				binary_data.shader_name,
				uniform_sets,
				binary_data.uses_argument_buffers(),
				libraries[RD::ShaderStage::SHADER_STAGE_COMPUTE]);

		uint32_t *binding = binary_data.push_constant.msl_binding.getptr(RD::SHADER_STAGE_COMPUTE);
		if (binding) {
			cs->push_constants.size = binary_data.push_constant.size;
			cs->push_constants.binding = *binding;
		}

		cs->local = MTLSizeMake(binary_data.compute_local_size.x, binary_data.compute_local_size.y, binary_data.compute_local_size.z);
#if DEV_ENABLED
		cs->kernel_source = binary_data.stages[0].source;
#endif
		shader = cs;
	} else {
		MDRenderShader *rs = new MDRenderShader(
				binary_data.shader_name,
				uniform_sets,
				binary_data.needs_view_mask_buffer(),
				binary_data.uses_argument_buffers(),
				libraries[RD::ShaderStage::SHADER_STAGE_VERTEX],
				libraries[RD::ShaderStage::SHADER_STAGE_FRAGMENT]);

		uint32_t *vert_binding = binary_data.push_constant.msl_binding.getptr(RD::SHADER_STAGE_VERTEX);
		if (vert_binding) {
			rs->push_constants.vert.size = binary_data.push_constant.size;
			rs->push_constants.vert.binding = *vert_binding;
		}
		uint32_t *frag_binding = binary_data.push_constant.msl_binding.getptr(RD::SHADER_STAGE_FRAGMENT);
		if (frag_binding) {
			rs->push_constants.frag.size = binary_data.push_constant.size;
			rs->push_constants.frag.binding = *frag_binding;
		}

#if DEV_ENABLED
		for (ShaderStageData &stage_data : binary_data.stages) {
			if (stage_data.stage == RD::ShaderStage::SHADER_STAGE_VERTEX) {
				rs->vert_source = stage_data.source;
			} else if (stage_data.stage == RD::ShaderStage::SHADER_STAGE_FRAGMENT) {
				rs->frag_source = stage_data.source;
			}
		}
#endif
		shader = rs;
	}

	// r_shader_desc.vertex_input_mask = binary_data.vertex_input_mask;
	// r_shader_desc.fragment_output_mask = binary_data.fragment_output_mask;
	// r_shader_desc.is_compute = binary_data.is_compute();
	// r_shader_desc.compute_local_size[0] = binary_data.compute_local_size.x;
	// r_shader_desc.compute_local_size[1] = binary_data.compute_local_size.y;
	// r_shader_desc.compute_local_size[2] = binary_data.compute_local_size.z;
	// r_shader_desc.push_constant_size = binary_data.push_constant.size;

	return RDD::ShaderID(shader);
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
	result->set_owner(this);
	return result;
}

RenderingDeviceCommons::ShaderLanguageVersion RenderingShaderContainerFormatMetal::get_shader_language_version() const {
	return SHADER_LANGUAGE_VULKAN_VERSION_1_1;
}

RenderingDeviceCommons::ShaderSpirvVersion RenderingShaderContainerFormatMetal::get_shader_spirv_version() const {
	return SHADER_SPIRV_VERSION_1_5;
}

static void mvkDispatchToMainAndWait(dispatch_block_t block) {
	if (NSThread.isMainThread) {
		block();
	} else {
		dispatch_sync(dispatch_get_main_queue(), block);
	}
}

RenderingShaderContainerFormatMetal::RenderingShaderContainerFormatMetal(bool p_export) {
	export_mode = p_export;
	id<MTLDevice> __block device = nil;
	mvkDispatchToMainAndWait(^{
		device = MTLCreateSystemDefaultDevice();
	});
	device_properties = memnew(MetalDeviceProperties(device));
}

RenderingShaderContainerFormatMetal::~RenderingShaderContainerFormatMetal() {
}

void RenderingShaderContainerFormatMetal::shader_cache_free_entry(const SHA256Digest &key) {
	if (ShaderCacheEntry **pentry = _shader_cache.getptr(key); pentry != nullptr) {
		ShaderCacheEntry *entry = *pentry;
		_shader_cache.erase(key);
		entry->library = nil;
		memdelete(entry);
	}
}

void RenderingShaderContainerFormatMetal::clear_shader_cache() {
	for (KeyValue<SHA256Digest, ShaderCacheEntry *> &kv : _shader_cache) {
		memdelete(kv.value);
	}
}

#include "gaussian_splat_world_io.h"

#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/io/compression.h"
#include "core/string/ustring.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "io_settings_utils.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_world.h"
#include "../interfaces/gpu_culler.h"
#include "../logger/gs_logger.h"

#include <cstring>
#include <cstdint>

namespace {

static constexpr uint32_t kWorldMagic = 0x57505347; // 'GSPW' little-endian.
static constexpr uint32_t kWorldVersion = 1;
static constexpr uint32_t kMaxShDegree = 3;
static constexpr uint32_t kFlagHasMetadata = 1u << 0u;
static constexpr uint32_t kFlagIs2D = 1u << 1u;
static constexpr uint32_t kFlagHasChunks = 1u << 2u;
static constexpr uint32_t kFlagHasHighSh = 1u << 3u;
static constexpr uint32_t kFlagCompressed = 1u << 4u;
static constexpr uint64_t kHeaderSizeBytes = 104u;

struct ChunkRecord {
	Vector3 bounds_pos;
	Vector3 bounds_size;
	Vector3 center;
	float radius = 0.0f;
	uint64_t indices_offset = 0;
	uint32_t index_count = 0;
	uint32_t reserved = 0;
};

static bool _read_exact(Ref<FileAccess> p_file, void *p_dst, uint64_t p_size) {
	ERR_FAIL_COND_V(p_file.is_null(), false);
	uint8_t *dst = static_cast<uint8_t *>(p_dst);
	const uint64_t read = p_file->get_buffer(dst, p_size);
	return read == p_size;
}

static void _write_vec3(Ref<FileAccess> p_file, const Vector3 &p_value) {
	p_file->store_float(p_value.x);
	p_file->store_float(p_value.y);
	p_file->store_float(p_value.z);
}

static Vector3 _read_vec3(Ref<FileAccess> p_file) {
	return Vector3(p_file->get_float(), p_file->get_float(), p_file->get_float());
}

static void _write_chunk_record(Ref<FileAccess> p_file, const ChunkRecord &p_record) {
	_write_vec3(p_file, p_record.bounds_pos);
	_write_vec3(p_file, p_record.bounds_size);
	_write_vec3(p_file, p_record.center);
	p_file->store_float(p_record.radius);
	p_file->store_64(p_record.indices_offset);
	p_file->store_32(p_record.index_count);
	p_file->store_32(p_record.reserved);
}

static ChunkRecord _read_chunk_record(Ref<FileAccess> p_file) {
	ChunkRecord record;
	record.bounds_pos = _read_vec3(p_file);
	record.bounds_size = _read_vec3(p_file);
	record.center = _read_vec3(p_file);
	record.radius = p_file->get_float();
	record.indices_offset = p_file->get_64();
	record.index_count = p_file->get_32();
	record.reserved = p_file->get_32();
	return record;
}

// Compression helpers
static PackedByteArray _compress_data(const uint8_t *p_data, uint64_t p_size) {
	PackedByteArray result;
	if (p_size == 0) return result;
	
	int64_t max_compressed = Compression::get_max_compressed_buffer_size(p_size, Compression::MODE_GZIP);
	result.resize(max_compressed);
	
	int64_t compressed_size = Compression::compress(result.ptrw(), p_data, p_size, Compression::MODE_GZIP);
	if (compressed_size <= 0 || compressed_size > max_compressed) {
		return PackedByteArray(); // Compression failed
	}
	
	result.resize(compressed_size);
	return result;
}

static bool _decompress_data(const uint8_t *p_compressed, uint64_t p_compressed_size, uint8_t *p_dst, uint64_t p_original_size) {
	if (p_compressed_size == 0 || p_original_size == 0) return false;
	
	int64_t result = Compression::decompress(p_dst, p_original_size, p_compressed, p_compressed_size, Compression::MODE_GZIP);
	return result == static_cast<int64_t>(p_original_size);
}

static bool _is_world_compression_enabled() {
	if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
		return GaussianSplattingIO::get_bool_setting(ps,
				"rendering/gaussian_splatting/import/gsplatworld_compression_enabled",
				false);
	}
	return false;
}

static Error _ensure_file_write_ok(const Ref<FileAccess> &p_file, const char *p_context) {
	ERR_FAIL_COND_V_MSG(p_file.is_null(), ERR_INVALID_PARAMETER, "FileAccess is null while checking write status.");
	const Error io_error = p_file->get_error();
	if (io_error != OK) {
		ERR_PRINT(vformat("[GaussianSplatWorldIO] Write failure in %s (error=%d).",
				p_context ? p_context : "unknown",
				(int)io_error));
		return io_error;
	}
	return OK;
}

} // namespace

Ref<Resource> ResourceFormatLoaderGaussianSplatWorld::load(const String &p_path, const String &p_original_path,
		Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	(void)p_original_path;
	(void)p_use_sub_threads;
	(void)p_cache_mode;
	if (r_progress) {
		*r_progress = 0.0f;
	}

	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	if (file.is_null()) {
		if (r_error) {
			*r_error = ERR_FILE_CANT_READ;
		}
		return Ref<Resource>();
	}

	const uint64_t file_len = file->get_length();
	if (file_len < kHeaderSizeBytes) {
		if (r_error) {
			*r_error = ERR_FILE_CORRUPT;
		}
		return Ref<Resource>();
	}

	const uint32_t magic = file->get_32();
	if (magic != kWorldMagic) {
		if (r_error) {
			*r_error = ERR_FILE_UNRECOGNIZED;
		}
		return Ref<Resource>();
	}

	const uint32_t version = file->get_32();
	if (version != kWorldVersion) {
		if (r_error) {
			*r_error = ERR_FILE_CORRUPT;
		}
		return Ref<Resource>();
	}

	const uint32_t flags = file->get_32();
	const uint32_t splat_count = file->get_32();
	const uint32_t sh_degree = file->get_32();
	if (sh_degree > kMaxShDegree) {
		if (r_error) {
			*r_error = ERR_FILE_CORRUPT;
		}
		return Ref<Resource>();
	}
	const uint32_t sh_first_order = file->get_32();
	const uint32_t sh_high_order = file->get_32();
	const Vector3 bounds_pos = _read_vec3(file);
	const Vector3 bounds_size = _read_vec3(file);
	const uint32_t chunk_count = file->get_32();
	const uint64_t gaussian_offset = file->get_64();
	const uint64_t sh_offset = file->get_64();
	const uint64_t chunk_table_offset = file->get_64();
	const uint64_t indices_offset = file->get_64();
	const uint64_t metadata_offset = file->get_64();
	const uint64_t metadata_size = file->get_64();

	if (gaussian_offset >= file_len || gaussian_offset < kHeaderSizeBytes) {
		if (r_error) {
			*r_error = ERR_FILE_CORRUPT;
		}
		return Ref<Resource>();
	}

	const uint64_t gaussian_bytes = uint64_t(splat_count) * sizeof(Gaussian);
	const bool gaussian_data_compressed = (flags & kFlagCompressed) != 0;
	if (gaussian_data_compressed) {
		if (gaussian_offset > file_len || gaussian_offset > file_len - sizeof(uint64_t)) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}

		const uint64_t saved_pos = file->get_position();
		file->seek(gaussian_offset);
		const uint64_t compressed_size = file->get_64();
		file->seek(saved_pos);

		const uint64_t compressed_capacity = file_len - (gaussian_offset + sizeof(uint64_t));
		if ((gaussian_bytes > 0 && compressed_size == 0) || compressed_size > compressed_capacity) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
	} else {
		if (gaussian_bytes > file_len - gaussian_offset) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
	}

	if ((flags & kFlagHasHighSh) != 0 && sh_high_order > 0) {
		const uint64_t sh_count = uint64_t(splat_count) * uint64_t(sh_high_order);
		const uint64_t sh_bytes = sh_count * sizeof(Vector3);
		if (sh_offset + sh_bytes > file_len) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
	}

	if ((flags & kFlagHasChunks) != 0 && chunk_count > 0) {
		const uint64_t chunk_table_bytes = uint64_t(chunk_count) * 56u;
		if (chunk_table_offset + chunk_table_bytes > file_len) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
		if (indices_offset >= file_len) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
	}

	if ((flags & kFlagHasMetadata) != 0 && metadata_size > 0) {
		if (metadata_offset + metadata_size > file_len) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
	}

	LocalVector<Gaussian> gaussians;
	gaussians.resize(splat_count);
	file->seek(gaussian_offset);

	if ((flags & kFlagCompressed) != 0) {
		// Compressed format: [8 bytes compressed_size][compressed_data]
		const uint64_t compressed_size = file->get_64();
		const uint64_t compressed_data_offset = file->get_position();
		if (compressed_data_offset > file_len ||
				compressed_size > (file_len - compressed_data_offset)) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
		PackedByteArray compressed_data;
		compressed_data.resize(compressed_size);
		if (!_read_exact(file, compressed_data.ptrw(), compressed_size)) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
		if (!_decompress_data(compressed_data.ptr(), compressed_size,
				reinterpret_cast<uint8_t *>(gaussians.ptr()), gaussian_bytes)) {
			GS_LOG_ERROR_DEFAULT("Failed to decompress gaussian data");
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
		GS_LOG_STREAMING_INFO(vformat("Decompressed gaussian data: %d KB -> %d KB (%.1fx)",
				int(compressed_size / 1024), int(gaussian_bytes / 1024),
				float(gaussian_bytes) / float(compressed_size)));
	} else {
		// Uncompressed format: raw gaussian data
		if (!_read_exact(file, gaussians.ptr(), gaussian_bytes)) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
	}

	LocalVector<Vector3> sh_high_coeffs;
	if ((flags & kFlagHasHighSh) != 0 && sh_high_order > 0) {
		const uint64_t sh_count = uint64_t(splat_count) * uint64_t(sh_high_order);
		sh_high_coeffs.resize(sh_count);
		file->seek(sh_offset);
		const uint64_t sh_bytes = sh_count * sizeof(Vector3);
		if (!_read_exact(file, sh_high_coeffs.ptr(), sh_bytes)) {
			if (r_error) {
				*r_error = ERR_FILE_CORRUPT;
			}
			return Ref<Resource>();
		}
	}

	Vector<ChunkRecord> chunk_records;
	if ((flags & kFlagHasChunks) != 0 && chunk_count > 0) {
		chunk_records.resize(chunk_count);
		file->seek(chunk_table_offset);
		for (uint32_t i = 0; i < chunk_count; i++) {
			chunk_records.write[i] = _read_chunk_record(file);
		}
	}

	Vector<uint32_t> all_indices;
	if (!chunk_records.is_empty()) {
		uint64_t total_indices = 0;
		for (int i = 0; i < chunk_records.size(); i++) {
			const ChunkRecord &record = chunk_records[i];
			if (record.indices_offset > UINT64_MAX - uint64_t(record.index_count)) {
				if (r_error) {
					*r_error = ERR_FILE_CORRUPT;
				}
				ERR_PRINT(vformat("[GaussianSplatWorld] Corrupt chunk index range: chunk=%d offset=%s count=%u overflows uint64.",
						i, String::num_uint64(record.indices_offset), record.index_count));
				return Ref<Resource>();
			}
			total_indices = MAX<uint64_t>(total_indices, record.indices_offset + uint64_t(record.index_count));
		}
		if (total_indices > 0) {
			all_indices.resize(total_indices);
			file->seek(indices_offset);
			const uint64_t bytes = total_indices * sizeof(uint32_t);
			if (!_read_exact(file, all_indices.ptrw(), bytes)) {
				if (r_error) {
					*r_error = ERR_FILE_CORRUPT;
				}
				return Ref<Resource>();
			}
		}
	}

	Ref<GaussianData> gaussian_data;
	gaussian_data.instantiate();
	gaussian_data->set_gaussian_payload(gaussians, sh_high_coeffs, sh_first_order, sh_high_order, (flags & kFlagIs2D) != 0);

	Vector<StaticChunk> chunks;
	if (!chunk_records.is_empty()) {
		chunks.resize(chunk_records.size());
		for (int i = 0; i < chunk_records.size(); i++) {
			const ChunkRecord &record = chunk_records[i];
			StaticChunk chunk;
			chunk.bounds = AABB(record.bounds_pos, record.bounds_size);
			chunk.center = record.center;
			chunk.radius = record.radius;
			if (record.index_count > 0 && !all_indices.is_empty()) {
				chunk.indices.resize(record.index_count);
				const uint64_t src_offset = record.indices_offset;
				const uint64_t src_size = uint64_t(record.index_count);
				if (src_offset <= uint64_t(all_indices.size()) && src_size <= uint64_t(all_indices.size()) - src_offset) {
					memcpy(chunk.indices.ptrw(),
							all_indices.ptr() + size_t(src_offset),
							uint64_t(record.index_count) * sizeof(uint32_t));
				} else {
					if (r_error) {
						*r_error = ERR_FILE_CORRUPT;
					}
					ERR_PRINT(vformat("[GaussianSplatWorld] Corrupt chunk indices: chunk=%d offset=%s count=%u total=%d",
							i, String::num_uint64(src_offset), record.index_count, all_indices.size()));
					return Ref<Resource>();
				}
			}
			chunks.write[i] = chunk;
		}
	}

	Dictionary file_metadata;
	if ((flags & kFlagHasMetadata) != 0 && metadata_size > 0) {
		file->seek(metadata_offset);
		PackedByteArray metadata_bytes = file->get_buffer(metadata_size);
		String metadata_text = String::utf8((const char *)metadata_bytes.ptr(), metadata_bytes.size());
		Variant parsed = JSON::parse_string(metadata_text);
		if (parsed.get_type() == Variant::DICTIONARY) {
			file_metadata = parsed;
		}
	}

	Ref<GaussianSplatWorld> world;
	world.instantiate();
	world->set_gaussian_data(gaussian_data);
	world->set_bounds(AABB(bounds_pos, bounds_size));
	world->set_metadata(file_metadata);
	world->set_static_chunks(chunks);

	if (GaussianSplattingIO::is_data_log_enabled()) {
		GS_LOG_STREAMING_INFO(vformat("[GaussianSplatWorld] Loaded: %d splats, %d spatial chunks, bounds: pos=%s size=%s",
				splat_count, chunks.size(), bounds_pos, bounds_size));
	}

	if (r_error) {
		*r_error = OK;
	}
	if (r_progress) {
		*r_progress = 1.0f;
	}
	return world;
}

void ResourceFormatLoaderGaussianSplatWorld::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gsplatworld");
}

bool ResourceFormatLoaderGaussianSplatWorld::handles_type(const String &p_type) const {
	return p_type == "GaussianSplatWorld";
}

String ResourceFormatLoaderGaussianSplatWorld::get_resource_type(const String &p_path) const {
	String ext = p_path.get_extension().to_lower();
	if (ext == "gsplatworld") {
		return "GaussianSplatWorld";
	}
	return "";
}

Error ResourceFormatSaverGaussianSplatWorld::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	(void)p_flags;
	const GaussianSplatWorld *world = Object::cast_to<GaussianSplatWorld>(*p_resource);
	ERR_FAIL_COND_V_MSG(world == nullptr, ERR_INVALID_PARAMETER, "Resource is not a GaussianSplatWorld.");

	const Ref<GaussianData> gaussian_data = world->get_gaussian_data();
	ERR_FAIL_COND_V_MSG(gaussian_data.is_null(), ERR_INVALID_DATA, "GaussianSplatWorld has no GaussianData.");
	Error err = OK;

	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(file.is_null(), ERR_FILE_CANT_WRITE, vformat("Cannot write gsplatworld file: %s", p_path));

	const uint32_t splat_count = gaussian_data->get_count();
	const uint32_t sh_degree = gaussian_data->get_sh_degree();
	const uint32_t sh_first_order = gaussian_data->get_sh_first_order_count();
	const uint32_t sh_high_order = gaussian_data->get_sh_high_order_count();
	const bool is_2d = gaussian_data->get_2d_mode();
	const Vector<StaticChunk> &chunks = world->get_static_chunks();
	const uint32_t chunk_count = chunks.size();

	uint64_t total_indices = 0;
	for (uint32_t i = 0; i < chunk_count; i++) {
		total_indices += chunks[i].indices.size();
	}

	uint32_t flags = 0;
	if (!world->get_metadata().is_empty()) {
		flags |= kFlagHasMetadata;
	}
	if (is_2d) {
		flags |= kFlagIs2D;
	}
	if (chunk_count > 0) {
		flags |= kFlagHasChunks;
	}
	if (sh_high_order > 0) {
		flags |= kFlagHasHighSh;
	}

	String metadata_json;
	if ((flags & kFlagHasMetadata) != 0) {
		metadata_json = JSON::stringify(world->get_metadata());
	}
	PackedByteArray metadata_bytes = metadata_json.to_utf8_buffer();

	const uint64_t gaussian_bytes = uint64_t(splat_count) * sizeof(Gaussian);
	const uint64_t sh_bytes = uint64_t(splat_count) * uint64_t(sh_high_order) * sizeof(Vector3);
	const uint64_t chunk_table_bytes = uint64_t(chunk_count) * 56u;
	const uint64_t indices_bytes = total_indices * sizeof(uint32_t);

	// Compression is configurable; cache-heavy workflows can disable it to minimize CPU load times.
	PackedByteArray compressed_gaussians;
	bool use_compression = _is_world_compression_enabled() && gaussian_bytes > 1024; // Only compress if >1KB
	if (use_compression && gaussian_bytes > 0) {
		compressed_gaussians = _compress_data(
				reinterpret_cast<const uint8_t *>(gaussian_data->get_gaussian_storage().ptr()),
				gaussian_bytes);
		if (compressed_gaussians.is_empty()) {
			use_compression = false; // Compression failed, fall back to uncompressed
		} else {
			flags |= kFlagCompressed;
			GS_LOG_STREAMING_INFO(vformat("Compressed gaussian data: %d KB -> %d KB (%.1fx)",
					int(gaussian_bytes / 1024), int(compressed_gaussians.size() / 1024),
					float(gaussian_bytes) / float(compressed_gaussians.size())));
		}
	}

	// Calculate stored size (compressed + 8 byte header, or uncompressed)
	const uint64_t gaussian_stored_bytes = use_compression
			? (8 + compressed_gaussians.size())
			: gaussian_bytes;

	const uint64_t metadata_offset = (metadata_bytes.is_empty()) ? 0u
			: (kHeaderSizeBytes + gaussian_stored_bytes + sh_bytes + chunk_table_bytes + indices_bytes);

	const uint64_t gaussian_offset = kHeaderSizeBytes;
	const uint64_t sh_offset = gaussian_offset + gaussian_stored_bytes;
	const uint64_t chunk_table_offset = sh_offset + sh_bytes;
	const uint64_t indices_offset = chunk_table_offset + chunk_table_bytes;
	const uint64_t metadata_size = metadata_bytes.size();

	file->store_32(kWorldMagic);
	file->store_32(kWorldVersion);
	file->store_32(flags);
	file->store_32(splat_count);
	file->store_32(sh_degree);
	file->store_32(sh_first_order);
	file->store_32(sh_high_order);
	_write_vec3(file, world->get_bounds().position);
	_write_vec3(file, world->get_bounds().size);
	file->store_32(chunk_count);
	file->store_64(gaussian_offset);
	file->store_64(sh_offset);
	file->store_64(chunk_table_offset);
	file->store_64(indices_offset);
	file->store_64(metadata_offset);
	file->store_64(metadata_size);
	err = _ensure_file_write_ok(file, "save(header)");
	if (err != OK) {
		return err;
	}

	if (gaussian_bytes > 0) {
		if (use_compression && !compressed_gaussians.is_empty()) {
			// Write compressed format: [8 bytes size][compressed data]
			file->store_64(compressed_gaussians.size());
			file->store_buffer(compressed_gaussians.ptr(), compressed_gaussians.size());
		} else {
			// Write uncompressed format: raw gaussian data
			file->store_buffer(reinterpret_cast<const uint8_t *>(gaussian_data->get_gaussian_storage().ptr()),
					gaussian_bytes);
		}
		err = _ensure_file_write_ok(file, "save(gaussian_data)");
		if (err != OK) {
			return err;
		}
	}

	if (sh_bytes > 0) {
		const Vector3 *sh_ptr = gaussian_data->get_sh_high_order_coefficients_ptr();
		ERR_FAIL_COND_V_MSG(sh_ptr == nullptr, ERR_INVALID_DATA,
				"GaussianSplatWorld missing high-order SH coefficients while sh_high_order_count > 0.");
		file->store_buffer(reinterpret_cast<const uint8_t *>(sh_ptr), sh_bytes);
		err = _ensure_file_write_ok(file, "save(sh_data)");
		if (err != OK) {
			return err;
		}
	}

	if (chunk_count > 0) {
		uint64_t indices_cursor = 0;
		for (uint32_t i = 0; i < chunk_count; i++) {
			const StaticChunk &chunk = chunks[i];
			ChunkRecord record;
			record.bounds_pos = chunk.bounds.position;
			record.bounds_size = chunk.bounds.size;
			record.center = chunk.center;
			record.radius = chunk.radius;
			record.indices_offset = indices_cursor;
			record.index_count = chunk.indices.size();
			_write_chunk_record(file, record);
			indices_cursor += record.index_count;
		}
		err = _ensure_file_write_ok(file, "save(chunk_table)");
		if (err != OK) {
			return err;
		}

		for (uint32_t i = 0; i < chunk_count; i++) {
			const StaticChunk &chunk = chunks[i];
			if (chunk.indices.is_empty()) {
				continue;
			}
			file->store_buffer(reinterpret_cast<const uint8_t *>(chunk.indices.ptr()),
					uint64_t(chunk.indices.size()) * sizeof(uint32_t));
			err = _ensure_file_write_ok(file, "save(chunk_indices)");
			if (err != OK) {
				return err;
			}
		}
	}

	if (metadata_size > 0) {
		file->store_buffer(metadata_bytes.ptr(), metadata_size);
		err = _ensure_file_write_ok(file, "save(metadata)");
		if (err != OK) {
			return err;
		}
	}

	return _ensure_file_write_ok(file, "save(final)");
}

void ResourceFormatSaverGaussianSplatWorld::get_recognized_extensions(const Ref<Resource> &p_resource,
		List<String> *p_extensions) const {
	if (recognize(p_resource)) {
		p_extensions->push_back("gsplatworld");
	}
}

bool ResourceFormatSaverGaussianSplatWorld::recognize(const Ref<Resource> &p_resource) const {
	return Object::cast_to<GaussianSplatWorld>(*p_resource) != nullptr;
}

bool ResourceFormatSaverGaussianSplatWorld::recognize_path(const Ref<Resource> &p_resource, const String &p_path) const {
	return p_path.get_extension().to_lower() == "gsplatworld" && recognize(p_resource);
}

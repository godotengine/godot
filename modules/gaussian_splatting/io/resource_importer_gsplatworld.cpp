#include "resource_importer_gsplatworld.h"

#ifdef TOOLS_ENABLED

#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "../core/gaussian_data.h"
#include "../logger/gs_logger.h"
#include <cstdint>

namespace {

static bool _range_in_file(uint64_t p_offset, uint64_t p_size, uint64_t p_file_size) {
	if (p_offset > p_file_size) {
		return false;
	}
	return p_size <= (p_file_size - p_offset);
}

static bool _checked_mul_u64(uint64_t p_a, uint64_t p_b, uint64_t &r_result) {
	if (p_a != 0u && p_b > (UINT64_MAX / p_a)) {
		return false;
	}
	r_result = p_a * p_b;
	return true;
}

static Error _validate_gsplatworld_header(const String &p_source_file) {
	constexpr uint32_t world_magic = 0x57505347u; // 'GSPW' little-endian.
	constexpr uint32_t world_version = 1u;
	constexpr uint32_t max_sh_degree = 3u;
	constexpr uint32_t flag_has_metadata = 1u << 0u;
	constexpr uint32_t flag_has_chunks = 1u << 2u;
	constexpr uint32_t flag_has_high_sh = 1u << 3u;
	constexpr uint32_t flag_compressed = 1u << 4u;
	constexpr uint64_t header_size_bytes = 104u;
	constexpr uint64_t chunk_record_size_bytes = 56u;

	Error open_err = OK;
	Ref<FileAccess> file = FileAccess::open(p_source_file, FileAccess::READ, &open_err);
	if (file.is_null()) {
		return open_err != OK ? open_err : ERR_CANT_OPEN;
	}

	const uint64_t file_size = file->get_length();
	if (file_size < header_size_bytes) {
		return ERR_FILE_CORRUPT;
	}

	const uint32_t magic = file->get_32();
	if (magic != world_magic) {
		return ERR_FILE_UNRECOGNIZED;
	}

	const uint32_t version = file->get_32();
	if (version != world_version) {
		return ERR_FILE_CORRUPT;
	}

	const uint32_t flags = file->get_32();
	const uint32_t splat_count = file->get_32();

	const uint32_t sh_degree = file->get_32();
	if (sh_degree > max_sh_degree) {
		return ERR_FILE_CORRUPT;
	}

	(void)file->get_32(); // sh_first_order
	const uint32_t sh_high_order = file->get_32();

	file->seek(file->get_position() + 12u); // bounds_pos
	file->seek(file->get_position() + 12u); // bounds_size

	const uint32_t chunk_count = file->get_32();
	const uint64_t gaussian_offset = file->get_64();
	const uint64_t sh_offset = file->get_64();
	const uint64_t chunk_table_offset = file->get_64();
	const uint64_t indices_offset = file->get_64();
	const uint64_t metadata_offset = file->get_64();
	const uint64_t metadata_size = file->get_64();

	if (gaussian_offset < header_size_bytes || gaussian_offset >= file_size) {
		return ERR_FILE_CORRUPT;
	}

	uint64_t gaussian_bytes = 0u;
	if (!_checked_mul_u64(uint64_t(splat_count), sizeof(Gaussian), gaussian_bytes)) {
		return ERR_FILE_CORRUPT;
	}
	if ((flags & flag_compressed) != 0u) {
		if (!_range_in_file(gaussian_offset, sizeof(uint64_t), file_size)) {
			return ERR_FILE_CORRUPT;
		}
		file->seek(gaussian_offset);
		const uint64_t compressed_size = file->get_64();
		if (!_range_in_file(gaussian_offset + sizeof(uint64_t), compressed_size, file_size)) {
			return ERR_FILE_CORRUPT;
		}
	} else {
		if (!_range_in_file(gaussian_offset, gaussian_bytes, file_size)) {
			return ERR_FILE_CORRUPT;
		}
	}

	if ((flags & flag_has_high_sh) != 0u && sh_high_order > 0u) {
		uint64_t sh_count = 0u;
		if (!_checked_mul_u64(uint64_t(splat_count), uint64_t(sh_high_order), sh_count)) {
			return ERR_FILE_CORRUPT;
		}
		uint64_t sh_bytes = 0u;
		if (!_checked_mul_u64(sh_count, sizeof(Vector3), sh_bytes)) {
			return ERR_FILE_CORRUPT;
		}
		if (!_range_in_file(sh_offset, sh_bytes, file_size)) {
			return ERR_FILE_CORRUPT;
		}
	}

	if ((flags & flag_has_chunks) != 0u && chunk_count > 0u) {
		if (chunk_count > (UINT64_MAX / chunk_record_size_bytes)) {
			return ERR_FILE_CORRUPT;
		}
		const uint64_t chunk_table_size = uint64_t(chunk_count) * chunk_record_size_bytes;
		if (!_range_in_file(chunk_table_offset, chunk_table_size, file_size)) {
			return ERR_FILE_CORRUPT;
		}
		if (indices_offset >= file_size) {
			return ERR_FILE_CORRUPT;
		}
	}

	if ((flags & flag_has_metadata) != 0u && metadata_size > 0u) {
		if (!_range_in_file(metadata_offset, metadata_size, file_size)) {
			return ERR_FILE_CORRUPT;
		}
	}

	return OK;
}

static Error _copy_binary_file(const String &p_source_file, const String &p_dest_file) {
	Error read_err = OK;
	Ref<FileAccess> src = FileAccess::open(p_source_file, FileAccess::READ, &read_err);
	if (src.is_null()) {
		return read_err != OK ? read_err : ERR_CANT_OPEN;
	}

	Error write_err = OK;
	Ref<FileAccess> dst = FileAccess::open(p_dest_file, FileAccess::WRITE, &write_err);
	if (dst.is_null()) {
		return write_err != OK ? write_err : ERR_CANT_OPEN;
	}

	const uint64_t file_size = src->get_length();
	constexpr uint64_t chunk_size = 1024 * 1024; // 1 MiB
	PackedByteArray buffer;
	buffer.resize(chunk_size);
	uint8_t *buffer_ptr = buffer.ptrw();

	while (src->get_position() < file_size) {
		const uint64_t remaining = file_size - src->get_position();
		const uint64_t to_read = remaining < chunk_size ? remaining : chunk_size;
		const uint64_t read = src->get_buffer(buffer_ptr, to_read);
		if (read != to_read) {
			return ERR_FILE_CORRUPT;
		}
		if (!dst->store_buffer(buffer_ptr, to_read)) {
			const Error dst_err = dst->get_error();
			return dst_err != OK ? dst_err : ERR_FILE_CANT_WRITE;
		}
	}

	return OK;
}

} // namespace

String ResourceImporterGSplatWorld::get_importer_name() const {
	return "gaussian_splat_world";
}

String ResourceImporterGSplatWorld::get_visible_name() const {
	return "Gaussian Splat World";
}

void ResourceImporterGSplatWorld::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gsplatworld");
}

String ResourceImporterGSplatWorld::get_save_extension() const {
	// Preserve binary world payloads during import. Saving as .tres drops
	// gaussian/chunk data because Generic text serialization cannot encode
	// GaussianSplatWorld payload arrays.
	return "gsplatworld";
}

String ResourceImporterGSplatWorld::get_resource_type() const {
	return "GaussianSplatWorld";
}

int ResourceImporterGSplatWorld::get_preset_count() const {
	return 0;
}

String ResourceImporterGSplatWorld::get_preset_name(int p_idx) const {
	(void)p_idx;
	return String();
}

void ResourceImporterGSplatWorld::get_import_options(const String &p_path, List<ImportOption> *r_options,
		int p_preset) const {
	(void)p_path;
	(void)r_options;
	(void)p_preset;
}

bool ResourceImporterGSplatWorld::get_option_visibility(const String &p_path, const String &p_option,
		const HashMap<StringName, Variant> &p_options) const {
	(void)p_path;
	(void)p_option;
	(void)p_options;
	return true;
}

Error ResourceImporterGSplatWorld::import(ResourceUID::ID p_source_id, const String &p_source_file,
		const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants,
		List<String> *r_gen_files, Variant *r_metadata) {
	(void)p_source_id;
	(void)p_options;
	(void)r_gen_files;
	if (r_platform_variants) {
		r_platform_variants->clear();
	}

	Error validation_err = _validate_gsplatworld_header(p_source_file);
	if (validation_err != OK) {
		GS_LOG_ERROR_DEFAULT(vformat("GaussianSplatWorld importer rejected invalid payload %s (error %d)",
				p_source_file, validation_err));
		return validation_err;
	}

	String save_path = p_save_path + "." + get_save_extension();
	// Keep imports cheap for large worlds: source/destination formats are identical.
	Error err = _copy_binary_file(p_source_file, save_path);
	if (err != OK) {
		GS_LOG_ERROR_DEFAULT(vformat("GaussianSplatWorld importer failed to copy %s -> %s (error %d)",
				p_source_file, save_path, err));
		return err;
	}
	Error imported_decode_err = OK;
	Ref<Resource> imported_world = ResourceLoader::load(
			save_path, "GaussianSplatWorld", ResourceFormatLoader::CACHE_MODE_IGNORE, &imported_decode_err);
	if (imported_world.is_null()) {
		const Error final_err = imported_decode_err != OK ? imported_decode_err : ERR_FILE_CORRUPT;
		GS_LOG_ERROR_DEFAULT(vformat("GaussianSplatWorld importer produced unreadable output %s (error %d)",
				save_path, final_err));
		return final_err;
	}

	if (r_metadata) {
		Dictionary import_metadata;
		import_metadata[StringName("source_path")] = p_source_file;
		import_metadata[StringName("resource_path")] = save_path;
		*r_metadata = import_metadata;
	}

	return OK;
}

#endif // TOOLS_ENABLED

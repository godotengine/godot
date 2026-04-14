#include "streaming_chunk_payload_source.h"

#include "core/io/file_access.h"
#include "core/error/error_macros.h"
#include "../logger/gs_logger.h"

// ---------------------------------------------------------------------------
// ChunkPayloadSource
// ---------------------------------------------------------------------------

void ChunkPayloadSource::_bind_methods() {}

// ---------------------------------------------------------------------------
// InMemoryChunkPayloadSource
// ---------------------------------------------------------------------------

void InMemoryChunkPayloadSource::_bind_methods() {}

bool InMemoryChunkPayloadSource::capture_chunk_snapshot(uint32_t p_start, uint32_t p_count,
		LocalVector<Gaussian> &r_gaussians,
		LocalVector<Vector3> &r_sh_high_order,
		uint32_t &r_sh_first_order_count,
		uint32_t &r_sh_high_order_count) const {
	if (!data.is_valid()) {
		return false;
	}
	return data->capture_chunk_snapshot(p_start, p_count,
			r_gaussians, r_sh_high_order, r_sh_first_order_count, r_sh_high_order_count);
}

bool InMemoryChunkPayloadSource::capture_indexed_chunk_snapshot(const uint32_t *p_indices, uint32_t p_count,
		LocalVector<Gaussian> &r_gaussians,
		LocalVector<Vector3> &r_sh_high_order,
		uint32_t &r_sh_first_order_count,
		uint32_t &r_sh_high_order_count) const {
	if (!data.is_valid()) {
		return false;
	}
	return data->capture_indexed_chunk_snapshot(p_indices, p_count,
			r_gaussians, r_sh_high_order, r_sh_first_order_count, r_sh_high_order_count);
}

uint32_t InMemoryChunkPayloadSource::get_count() const {
	return data.is_valid() ? data->get_count() : 0;
}

uint32_t InMemoryChunkPayloadSource::get_sh_degree() const {
	return data.is_valid() ? data->get_sh_degree() : 0;
}

AABB InMemoryChunkPayloadSource::get_bounds() const {
	return data.is_valid() ? data->get_aabb() : AABB();
}

bool InMemoryChunkPayloadSource::is_valid() const {
	return data.is_valid() && data->get_count() > 0;
}

// ---------------------------------------------------------------------------
// StagedFileChunkPayloadSource
// ---------------------------------------------------------------------------

void StagedFileChunkPayloadSource::_bind_methods() {}

void StagedFileChunkPayloadSource::configure(const String &p_path,
		uint64_t p_gaussian_offset,
		uint64_t p_sh_offset,
		uint32_t p_splat_count,
		uint32_t p_sh_degree,
		uint32_t p_sh_first_order,
		uint32_t p_sh_high_order,
		const AABB &p_bounds) {
	file_path = p_path;
	gaussian_data_offset = p_gaussian_offset;
	sh_data_offset = p_sh_offset;
	splat_count = p_splat_count;
	sh_degree = p_sh_degree;
	sh_first_order = p_sh_first_order;
	sh_high_order = p_sh_high_order;
	bounds = p_bounds;
}

bool StagedFileChunkPayloadSource::capture_chunk_snapshot(uint32_t p_start, uint32_t p_count,
		LocalVector<Gaussian> &r_gaussians,
		LocalVector<Vector3> &r_sh_high_order_out,
		uint32_t &r_sh_first_order_count,
		uint32_t &r_sh_high_order_count) const {
	if (file_path.is_empty() || p_count == 0) {
		return false;
	}
	if (uint64_t(p_start) + uint64_t(p_count) > uint64_t(splat_count)) {
		ERR_PRINT(vformat("[StagedFileSource] Range out of bounds: start=%d count=%d total=%d",
				p_start, p_count, splat_count));
		return false;
	}

	// Each call opens its own FileAccess handle. The OS handles concurrent
	// reads on separate handles efficiently; no cross-thread mutex needed.
	// Serializing all pack workers on a single mutex was the primary source
	// of pack throughput stalls under bounded-VRAM corridor churn.
	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
	if (file.is_null()) {
		ERR_PRINT(vformat("[StagedFileSource] Cannot open staged world file: %s", file_path));
		return false;
	}

	// Read gaussian data.
	const uint64_t gaussian_byte_offset = gaussian_data_offset + uint64_t(p_start) * sizeof(Gaussian);
	const uint64_t gaussian_byte_count = uint64_t(p_count) * sizeof(Gaussian);
	file->seek(gaussian_byte_offset);

	r_gaussians.resize(p_count);
	uint64_t got = file->get_buffer(reinterpret_cast<uint8_t *>(r_gaussians.ptr()), gaussian_byte_count);
	if (got != gaussian_byte_count) {
		ERR_PRINT(vformat("[StagedFileSource] Short read on gaussians: expected %d got %d",
				gaussian_byte_count, got));
		return false;
	}

	// Read SH high-order coefficients if present.
	r_sh_first_order_count = sh_first_order;
	r_sh_high_order_count = sh_high_order;

	if (sh_high_order > 0 && sh_data_offset > 0) {
		const uint64_t sh_per_splat = uint64_t(sh_high_order);
		const uint64_t sh_byte_offset = sh_data_offset + uint64_t(p_start) * sh_per_splat * sizeof(Vector3);
		const uint64_t sh_byte_count = uint64_t(p_count) * sh_per_splat * sizeof(Vector3);
		file->seek(sh_byte_offset);

		r_sh_high_order_out.resize(uint32_t(p_count * sh_per_splat));
		got = file->get_buffer(reinterpret_cast<uint8_t *>(r_sh_high_order_out.ptr()), sh_byte_count);
		if (got != sh_byte_count) {
			ERR_PRINT(vformat("[StagedFileSource] Short read on SH data: expected %d got %d",
					sh_byte_count, got));
			return false;
		}
	} else {
		r_sh_high_order_out.clear();
	}

	return true;
}

bool StagedFileChunkPayloadSource::capture_indexed_chunk_snapshot(const uint32_t *p_indices, uint32_t p_count,
		LocalVector<Gaussian> &r_gaussians,
		LocalVector<Vector3> &r_sh_high_order_out,
		uint32_t &r_sh_first_order_count,
		uint32_t &r_sh_high_order_count) const {
	if (file_path.is_empty() || p_count == 0 || p_indices == nullptr) {
		return false;
	}

	// Find min/max indices to determine the contiguous read range.
	uint32_t min_idx = p_indices[0];
	uint32_t max_idx = p_indices[0];
	for (uint32_t i = 1; i < p_count; i++) {
		min_idx = MIN(min_idx, p_indices[i]);
		max_idx = MAX(max_idx, p_indices[i]);
	}
	if (max_idx >= splat_count) {
		ERR_PRINT(vformat("[StagedFileSource] Index out of bounds: max=%d total=%d",
				max_idx, splat_count));
		return false;
	}

	const uint32_t range_count = max_idx - min_idx + 1;

	// No mutex: see capture_chunk_snapshot above. Independent file handles
	// per call allow parallel reads from the staged world file.
	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
	if (file.is_null()) {
		ERR_PRINT(vformat("[StagedFileSource] Cannot open staged world file: %s", file_path));
		return false;
	}

	// Read the contiguous gaussian range covering all requested indices.
	const uint64_t gaussian_byte_offset = gaussian_data_offset + uint64_t(min_idx) * sizeof(Gaussian);
	const uint64_t gaussian_byte_count = uint64_t(range_count) * sizeof(Gaussian);
	file->seek(gaussian_byte_offset);

	LocalVector<Gaussian> range_buf;
	range_buf.resize(range_count);
	uint64_t got = file->get_buffer(reinterpret_cast<uint8_t *>(range_buf.ptr()), gaussian_byte_count);
	if (got != gaussian_byte_count) {
		ERR_PRINT(vformat("[StagedFileSource] Short read on gaussians: expected %d got %d",
				gaussian_byte_count, got));
		return false;
	}

	// Extract the requested gaussians.
	r_gaussians.resize(p_count);
	for (uint32_t i = 0; i < p_count; i++) {
		r_gaussians[i] = range_buf[p_indices[i] - min_idx];
	}

	// Read SH data.
	r_sh_first_order_count = sh_first_order;
	r_sh_high_order_count = sh_high_order;

	if (sh_high_order > 0 && sh_data_offset > 0) {
		const uint64_t sh_per_splat = uint64_t(sh_high_order);
		const uint64_t sh_byte_offset = sh_data_offset + uint64_t(min_idx) * sh_per_splat * sizeof(Vector3);
		const uint64_t sh_byte_count = uint64_t(range_count) * sh_per_splat * sizeof(Vector3);
		file->seek(sh_byte_offset);

		LocalVector<Vector3> sh_range_buf;
		sh_range_buf.resize(uint32_t(range_count * sh_per_splat));
		got = file->get_buffer(reinterpret_cast<uint8_t *>(sh_range_buf.ptr()), sh_byte_count);
		if (got != sh_byte_count) {
			ERR_PRINT(vformat("[StagedFileSource] Short read on SH data: expected %d got %d",
					sh_byte_count, got));
			return false;
		}

		r_sh_high_order_out.resize(uint32_t(p_count * sh_per_splat));
		for (uint32_t i = 0; i < p_count; i++) {
			const uint32_t src_base = (p_indices[i] - min_idx) * uint32_t(sh_per_splat);
			const uint32_t dst_base = i * uint32_t(sh_per_splat);
			for (uint32_t c = 0; c < uint32_t(sh_per_splat); c++) {
				r_sh_high_order_out[dst_base + c] = sh_range_buf[src_base + c];
			}
		}
	} else {
		r_sh_high_order_out.clear();
	}

	return true;
}

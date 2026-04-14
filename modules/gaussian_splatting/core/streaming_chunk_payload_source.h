#ifndef STREAMING_CHUNK_PAYLOAD_SOURCE_H
#define STREAMING_CHUNK_PAYLOAD_SOURCE_H

#include "core/math/aabb.h"
#include "core/object/ref_counted.h"
#include "core/os/mutex.h"
#include "core/templates/local_vector.h"
#include "gaussian_data.h"
#include <cstdint>

// Abstract source of chunk gaussian payloads for the streaming system.
// Implementations provide either in-memory reads (wrapping GaussianData)
// or on-demand file-backed reads for out-of-core large-world streaming.
class ChunkPayloadSource : public RefCounted {
	GDCLASS(ChunkPayloadSource, RefCounted);

protected:
	static void _bind_methods();

public:
	virtual ~ChunkPayloadSource() = default;

	virtual bool capture_chunk_snapshot(uint32_t p_start, uint32_t p_count,
			LocalVector<Gaussian> &r_gaussians,
			LocalVector<Vector3> &r_sh_high_order,
			uint32_t &r_sh_first_order_count,
			uint32_t &r_sh_high_order_count) const = 0;

	virtual bool capture_indexed_chunk_snapshot(const uint32_t *p_indices, uint32_t p_count,
			LocalVector<Gaussian> &r_gaussians,
			LocalVector<Vector3> &r_sh_high_order,
			uint32_t &r_sh_first_order_count,
			uint32_t &r_sh_high_order_count) const = 0;

	virtual uint32_t get_count() const = 0;
	virtual uint32_t get_sh_degree() const = 0;
	virtual AABB get_bounds() const = 0;
	virtual bool is_valid() const = 0;
};

// Wraps an in-memory GaussianData. This is the default path for small
// datasets and smoke tests — identical behaviour to the pre-existing code.
class InMemoryChunkPayloadSource : public ChunkPayloadSource {
	GDCLASS(InMemoryChunkPayloadSource, ChunkPayloadSource);
	Ref<GaussianData> data;

protected:
	static void _bind_methods();

public:
	InMemoryChunkPayloadSource() = default;
	explicit InMemoryChunkPayloadSource(const Ref<GaussianData> &p_data) :
			data(p_data) {}

	void set_data(const Ref<GaussianData> &p_data) { data = p_data; }
	Ref<GaussianData> get_data() const { return data; }

	bool capture_chunk_snapshot(uint32_t p_start, uint32_t p_count,
			LocalVector<Gaussian> &r_gaussians,
			LocalVector<Vector3> &r_sh_high_order,
			uint32_t &r_sh_first_order_count,
			uint32_t &r_sh_high_order_count) const override;

	bool capture_indexed_chunk_snapshot(const uint32_t *p_indices, uint32_t p_count,
			LocalVector<Gaussian> &r_gaussians,
			LocalVector<Vector3> &r_sh_high_order,
			uint32_t &r_sh_first_order_count,
			uint32_t &r_sh_high_order_count) const override;

	uint32_t get_count() const override;
	uint32_t get_sh_degree() const override;
	AABB get_bounds() const override;
	bool is_valid() const override;
};

// Reads chunk gaussian payloads directly from an uncompressed
// .gsplatworld file on demand.  Only the file path, header offsets,
// and lightweight metadata are kept in memory — the full gaussian
// array is never loaded, so process memory does not scale with
// total world splat count.
class StagedFileChunkPayloadSource : public ChunkPayloadSource {
	GDCLASS(StagedFileChunkPayloadSource, ChunkPayloadSource);
	String file_path;

protected:
	static void _bind_methods();
	uint64_t gaussian_data_offset = 0; // byte offset of gaussian array in file
	uint64_t sh_data_offset = 0; // byte offset of SH array (0 = no SH)
	uint32_t splat_count = 0;
	uint32_t sh_degree = 0;
	uint32_t sh_first_order = 0;
	uint32_t sh_high_order = 0;
	AABB bounds;
	mutable Mutex file_mutex;

public:
	StagedFileChunkPayloadSource() = default;

	void configure(const String &p_path,
			uint64_t p_gaussian_offset,
			uint64_t p_sh_offset,
			uint32_t p_splat_count,
			uint32_t p_sh_degree,
			uint32_t p_sh_first_order,
			uint32_t p_sh_high_order,
			const AABB &p_bounds);

	bool capture_chunk_snapshot(uint32_t p_start, uint32_t p_count,
			LocalVector<Gaussian> &r_gaussians,
			LocalVector<Vector3> &r_sh_high_order,
			uint32_t &r_sh_first_order_count,
			uint32_t &r_sh_high_order_count) const override;

	bool capture_indexed_chunk_snapshot(const uint32_t *p_indices, uint32_t p_count,
			LocalVector<Gaussian> &r_gaussians,
			LocalVector<Vector3> &r_sh_high_order,
			uint32_t &r_sh_first_order_count,
			uint32_t &r_sh_high_order_count) const override;

	uint32_t get_count() const override { return splat_count; }
	uint32_t get_sh_degree() const override { return sh_degree; }
	AABB get_bounds() const override { return bounds; }
	bool is_valid() const override { return !file_path.is_empty() && splat_count > 0; }

	const String &get_file_path() const { return file_path; }
};

#endif // STREAMING_CHUNK_PAYLOAD_SOURCE_H

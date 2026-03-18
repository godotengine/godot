#ifndef STREAMING_ATLAS_H
#define STREAMING_ATLAS_H

/**************************************************************************/
/* streaming_atlas.h                                                      */
/*                                                                        */
/* Atlas slot allocator and global atlas state for the streaming system.   */
/*                                                                        */
/* Pattern 10 (Flyweight + GPU resource cache): The atlas allocator maps   */
/* many streaming chunks to a fixed pool of GPU atlas slots. Each slot is  */
/* a flyweight reference into a single, shared GPU buffer.                 */
/*                                                                        */
/* Pattern 12 (Ownership graph): GlobalAtlasState holds RIDs that the     */
/* streaming system owns (created/freed in _sync_global_atlas_state).     */
/* The atlas_gaussian_buffer RID is borrowed from the persistent buffer   */
/* managed by GaussianStreamingSystem.                                    */
/*                                                                        */
/* Pattern 2 (Strong types): Slot indices are uint32_t, chunk keys are    */
/* uint64_t (asset_id << 32 | chunk_idx encoding).                        */
/**************************************************************************/

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid.h"

// Global atlas buffer bundle for the instance pipeline.
//
// Ownership (Pattern 12):
//   atlas_gaussian_buffer   - BORROWED from GaussianStreamingSystem::persistent_buffer
//   asset_meta_buffer       - OWNED: created/freed in _sync_global_atlas_state
//   chunk_meta_buffer       - OWNED: created/freed in _sync_global_atlas_state
//   asset_chunk_index_buffer- OWNED: created/freed in _sync_global_atlas_state
//   quantization_buffer     - OWNED: created/freed in _upload/_release_quantization_buffer
struct GlobalAtlasState {
	RID atlas_gaussian_buffer;
	uint32_t atlas_gaussian_count = 0; // Capacity upper-bound (persistent buffer size / PackedGaussian)
	RID asset_meta_buffer;
	RID chunk_meta_buffer;
	RID asset_chunk_index_buffer;
	RID quantization_buffer;
	uint64_t atlas_generation = 0;
};

// Global atlas slot allocator (stable slot IDs, no defrag yet).
//
// Pattern 10 (Flyweight + GPU resource cache): Maps chunk keys to
// reusable atlas slot indices. Slots are recycled via a free-list so
// that GPU buffer offsets remain stable across load/evict cycles.
//
// Pattern 11 (RAII for RIDs): The allocator itself holds no RIDs;
// it only manages logical slot indices. The owning streaming system
// is responsible for the GPU buffer lifetime.
class GaussianAtlasAllocator {
public:
	void reset(uint32_t p_slot_count);
	bool has_free_slots() const { return !free_slots.is_empty(); }
	uint32_t get_free_slot_count() const { return free_slots.size(); }
	uint32_t get_capacity() const { return capacity; }
	bool allocate_slot(uint64_t p_chunk_key, uint32_t &r_slot);
	void release_slot(uint64_t p_chunk_key);
	bool get_slot(uint64_t p_chunk_key, uint32_t &r_slot) const;
	void clear();

private:
	uint32_t capacity = 0;
	LocalVector<uint32_t> free_slots;
	HashMap<uint64_t, uint32_t> slot_map;
};

#endif // STREAMING_ATLAS_H

#ifndef STREAMING_ATLAS_H
#define STREAMING_ATLAS_H

/**************************************************************************/
/* streaming_atlas.h                                                      */
/*                                                                        */
/* Atlas slot allocator for the streaming system.                         */
/*                                                                        */
/* Pattern 10 (Flyweight + GPU resource cache): The atlas allocator maps   */
/* many streaming chunks to a fixed pool of GPU atlas slots. Each slot is  */
/* a flyweight reference into a single, shared GPU buffer.                 */
/*                                                                        */
/* Pattern 2 (Strong types): Slot indices are uint32_t, chunk keys are    */
/* uint64_t (asset_id << 32 | chunk_idx encoding).                        */
/**************************************************************************/

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include <cstdint>

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

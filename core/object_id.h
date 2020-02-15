#ifndef OBJECT_ID_H
#define OBJECT_ID_H

#include "core/typedefs.h"

// Class to store an object ID (int64)
// needs to be compatile with int64 because this is what Variant uses
// Also, need to be explicitly only castable to 64 bits integer types
// to avoid bugs due to loss of precision

class ObjectID {
	uint64_t id = 0;

public:
	_ALWAYS_INLINE_ bool is_reference() const { return (id & (uint64_t(1) << 63)) != 0; }
	_ALWAYS_INLINE_ bool is_valid() const { return id != 0; }
	_ALWAYS_INLINE_ bool is_null() const { return id == 0; }
	_ALWAYS_INLINE_ operator uint64_t() const { return id; }
	_ALWAYS_INLINE_ operator int64_t() const { return id; }

	_ALWAYS_INLINE_ bool operator==(const ObjectID &p_id) const { return id == p_id.id; }
	_ALWAYS_INLINE_ bool operator!=(const ObjectID &p_id) const { return id != p_id.id; }
	_ALWAYS_INLINE_ bool operator<(const ObjectID &p_id) const { return id < p_id.id; }

	_ALWAYS_INLINE_ void operator=(int64_t p_int64) { id = p_int64; }
	_ALWAYS_INLINE_ void operator=(uint64_t p_uint64) { id = p_uint64; }

	_ALWAYS_INLINE_ ObjectID() {}
	_ALWAYS_INLINE_ explicit ObjectID(const uint64_t p_id) { id = p_id; }
	_ALWAYS_INLINE_ explicit ObjectID(const int64_t p_id) { id = p_id; }
};

#endif // OBJECT_ID_H

// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

namespace meshopt
{

static unsigned int hashUpdate4(unsigned int h, const unsigned char* key, size_t len)
{
	// MurmurHash2
	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	while (len >= 4)
	{
		unsigned int k = *reinterpret_cast<const unsigned int*>(key);

		k *= m;
		k ^= k >> r;
		k *= m;

		h *= m;
		h ^= k;

		key += 4;
		len -= 4;
	}

	return h;
}

struct VertexHasher
{
	const unsigned char* vertices;
	size_t vertex_size;
	size_t vertex_stride;

	size_t hash(unsigned int index) const
	{
		return hashUpdate4(0, vertices + index * vertex_stride, vertex_size);
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		return memcmp(vertices + lhs * vertex_stride, vertices + rhs * vertex_stride, vertex_size) == 0;
	}
};

struct VertexStreamHasher
{
	const meshopt_Stream* streams;
	size_t stream_count;

	size_t hash(unsigned int index) const
	{
		unsigned int h = 0;

		for (size_t i = 0; i < stream_count; ++i)
		{
			const meshopt_Stream& s = streams[i];
			const unsigned char* data = static_cast<const unsigned char*>(s.data);

			h = hashUpdate4(h, data + index * s.stride, s.size);
		}

		return h;
	}

	bool equal(unsigned int lhs, unsigned int rhs) const
	{
		for (size_t i = 0; i < stream_count; ++i)
		{
			const meshopt_Stream& s = streams[i];
			const unsigned char* data = static_cast<const unsigned char*>(s.data);

			if (memcmp(data + lhs * s.stride, data + rhs * s.stride, s.size) != 0)
				return false;
		}

		return true;
	}
};

static size_t hashBuckets(size_t count)
{
	size_t buckets = 1;
	while (buckets < count)
		buckets *= 2;

	return buckets;
}

template <typename T, typename Hash>
static T* hashLookup(T* table, size_t buckets, const Hash& hash, const T& key, const T& empty)
{
	assert(buckets > 0);
	assert((buckets & (buckets - 1)) == 0);

	size_t hashmod = buckets - 1;
	size_t bucket = hash.hash(key) & hashmod;

	for (size_t probe = 0; probe <= hashmod; ++probe)
	{
		T& item = table[bucket];

		if (item == empty)
			return &item;

		if (hash.equal(item, key))
			return &item;

		// hash collision, quadratic probing
		bucket = (bucket + probe + 1) & hashmod;
	}

	assert(false && "Hash table is full"); // unreachable
	return 0;
}

} // namespace meshopt

size_t meshopt_generateVertexRemap(unsigned int* destination, const unsigned int* indices, size_t index_count, const void* vertices, size_t vertex_count, size_t vertex_size)
{
	using namespace meshopt;

	assert(indices || index_count == vertex_count);
	assert(index_count % 3 == 0);
	assert(vertex_size > 0 && vertex_size <= 256);

	meshopt_Allocator allocator;

	memset(destination, -1, vertex_count * sizeof(unsigned int));

	VertexHasher hasher = {static_cast<const unsigned char*>(vertices), vertex_size, vertex_size};

	size_t table_size = hashBuckets(vertex_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);
	memset(table, -1, table_size * sizeof(unsigned int));

	unsigned int next_vertex = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices ? indices[i] : unsigned(i);
		assert(index < vertex_count);

		if (destination[index] == ~0u)
		{
			unsigned int* entry = hashLookup(table, table_size, hasher, index, ~0u);

			if (*entry == ~0u)
			{
				*entry = index;

				destination[index] = next_vertex++;
			}
			else
			{
				assert(destination[*entry] != ~0u);

				destination[index] = destination[*entry];
			}
		}
	}

	assert(next_vertex <= vertex_count);

	return next_vertex;
}

size_t meshopt_generateVertexRemapMulti(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, const struct meshopt_Stream* streams, size_t stream_count)
{
	using namespace meshopt;

	assert(indices || index_count == vertex_count);
	assert(index_count % 3 == 0);
	assert(stream_count > 0 && stream_count <= 16);

	for (size_t i = 0; i < stream_count; ++i)
	{
		assert(streams[i].size > 0 && streams[i].size <= 256);
		assert(streams[i].size <= streams[i].stride);
	}

	meshopt_Allocator allocator;

	memset(destination, -1, vertex_count * sizeof(unsigned int));

	VertexStreamHasher hasher = {streams, stream_count};

	size_t table_size = hashBuckets(vertex_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);
	memset(table, -1, table_size * sizeof(unsigned int));

	unsigned int next_vertex = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices ? indices[i] : unsigned(i);
		assert(index < vertex_count);

		if (destination[index] == ~0u)
		{
			unsigned int* entry = hashLookup(table, table_size, hasher, index, ~0u);

			if (*entry == ~0u)
			{
				*entry = index;

				destination[index] = next_vertex++;
			}
			else
			{
				assert(destination[*entry] != ~0u);

				destination[index] = destination[*entry];
			}
		}
	}

	assert(next_vertex <= vertex_count);

	return next_vertex;
}

void meshopt_remapVertexBuffer(void* destination, const void* vertices, size_t vertex_count, size_t vertex_size, const unsigned int* remap)
{
	assert(vertex_size > 0 && vertex_size <= 256);

	meshopt_Allocator allocator;

	// support in-place remap
	if (destination == vertices)
	{
		unsigned char* vertices_copy = allocator.allocate<unsigned char>(vertex_count * vertex_size);
		memcpy(vertices_copy, vertices, vertex_count * vertex_size);
		vertices = vertices_copy;
	}

	for (size_t i = 0; i < vertex_count; ++i)
	{
		if (remap[i] != ~0u)
		{
			assert(remap[i] < vertex_count);

			memcpy(static_cast<unsigned char*>(destination) + remap[i] * vertex_size, static_cast<const unsigned char*>(vertices) + i * vertex_size, vertex_size);
		}
	}
}

void meshopt_remapIndexBuffer(unsigned int* destination, const unsigned int* indices, size_t index_count, const unsigned int* remap)
{
	assert(index_count % 3 == 0);

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices ? indices[i] : unsigned(i);
		assert(remap[index] != ~0u);

		destination[i] = remap[index];
	}
}

void meshopt_generateShadowIndexBuffer(unsigned int* destination, const unsigned int* indices, size_t index_count, const void* vertices, size_t vertex_count, size_t vertex_size, size_t vertex_stride)
{
	using namespace meshopt;

	assert(indices);
	assert(index_count % 3 == 0);
	assert(vertex_size > 0 && vertex_size <= 256);
	assert(vertex_size <= vertex_stride);

	meshopt_Allocator allocator;

	unsigned int* remap = allocator.allocate<unsigned int>(vertex_count);
	memset(remap, -1, vertex_count * sizeof(unsigned int));

	VertexHasher hasher = {static_cast<const unsigned char*>(vertices), vertex_size, vertex_stride};

	size_t table_size = hashBuckets(vertex_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);
	memset(table, -1, table_size * sizeof(unsigned int));

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		if (remap[index] == ~0u)
		{
			unsigned int* entry = hashLookup(table, table_size, hasher, index, ~0u);

			if (*entry == ~0u)
				*entry = index;

			remap[index] = *entry;
		}

		destination[i] = remap[index];
	}
}

void meshopt_generateShadowIndexBufferMulti(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, const struct meshopt_Stream* streams, size_t stream_count)
{
	using namespace meshopt;

	assert(indices);
	assert(index_count % 3 == 0);
	assert(stream_count > 0 && stream_count <= 16);

	for (size_t i = 0; i < stream_count; ++i)
	{
		assert(streams[i].size > 0 && streams[i].size <= 256);
		assert(streams[i].size <= streams[i].stride);
	}

	meshopt_Allocator allocator;

	unsigned int* remap = allocator.allocate<unsigned int>(vertex_count);
	memset(remap, -1, vertex_count * sizeof(unsigned int));

	VertexStreamHasher hasher = {streams, stream_count};

	size_t table_size = hashBuckets(vertex_count);
	unsigned int* table = allocator.allocate<unsigned int>(table_size);
	memset(table, -1, table_size * sizeof(unsigned int));

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		if (remap[index] == ~0u)
		{
			unsigned int* entry = hashLookup(table, table_size, hasher, index, ~0u);

			if (*entry == ~0u)
				*entry = index;

			remap[index] = *entry;
		}

		destination[i] = remap[index];
	}
}

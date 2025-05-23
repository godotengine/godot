// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <math.h>
#include <string.h>

namespace meshopt
{

struct ClusterAdjacency
{
	unsigned int* offsets;
	unsigned int* clusters;
	unsigned int* shared;
};

static void buildClusterAdjacency(ClusterAdjacency& adjacency, const unsigned int* cluster_indices, const unsigned int* cluster_offsets, size_t cluster_count, unsigned char* used, size_t vertex_count, meshopt_Allocator& allocator)
{
	unsigned int* ref_offsets = allocator.allocate<unsigned int>(vertex_count + 1);

	// compute number of clusters referenced by each vertex
	memset(ref_offsets, 0, vertex_count * sizeof(unsigned int));

	for (size_t i = 0; i < cluster_count; ++i)
	{
		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
		{
			unsigned int v = cluster_indices[j];
			assert(v < vertex_count);

			ref_offsets[v] += 1 - used[v];
			used[v] = 1;
		}

		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
			used[cluster_indices[j]] = 0;
	}

	// compute (worst-case) number of adjacent clusters for each cluster
	size_t total_adjacency = 0;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		size_t count = 0;
		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
		{
			unsigned int v = cluster_indices[j];
			assert(v < vertex_count);

			// worst case is every vertex has a disjoint cluster list
			count += used[v] ? 0 : ref_offsets[v] - 1;
			used[v] = 1;
		}

		// ... but only every other cluster can be adjacent in the end
		total_adjacency += count < cluster_count - 1 ? count : cluster_count - 1;

		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
			used[cluster_indices[j]] = 0;
	}

	// we can now allocate adjacency buffers
	adjacency.offsets = allocator.allocate<unsigned int>(cluster_count + 1);
	adjacency.clusters = allocator.allocate<unsigned int>(total_adjacency);
	adjacency.shared = allocator.allocate<unsigned int>(total_adjacency);

	// convert ref counts to offsets
	size_t total_refs = 0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		size_t count = ref_offsets[i];
		ref_offsets[i] = unsigned(total_refs);
		total_refs += count;
	}

	unsigned int* ref_data = allocator.allocate<unsigned int>(total_refs);

	// fill cluster refs for each vertex
	for (size_t i = 0; i < cluster_count; ++i)
	{
		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
		{
			unsigned int v = cluster_indices[j];
			assert(v < vertex_count);

			if (used[v])
				continue;

			ref_data[ref_offsets[v]++] = unsigned(i);
			used[v] = 1;
		}

		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
			used[cluster_indices[j]] = 0;
	}

	// after the previous pass, ref_offsets contain the end of the data for each vertex; shift it forward to get the start
	memmove(ref_offsets + 1, ref_offsets, vertex_count * sizeof(unsigned int));
	ref_offsets[0] = 0;

	// fill cluster adjacency for each cluster...
	adjacency.offsets[0] = 0;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		unsigned int* adj = adjacency.clusters + adjacency.offsets[i];
		unsigned int* shd = adjacency.shared + adjacency.offsets[i];
		size_t count = 0;

		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
		{
			unsigned int v = cluster_indices[j];
			assert(v < vertex_count);

			if (used[v])
				continue;

			// merge the entire cluster list of each vertex into current list
			for (size_t k = ref_offsets[v]; k < ref_offsets[v + 1]; ++k)
			{
				unsigned int c = ref_data[k];
				assert(c < cluster_count);

				if (c == unsigned(i))
					continue;

				// if the cluster is already in the list, increment the shared count
				bool found = false;
				for (size_t l = 0; l < count; ++l)
					if (adj[l] == c)
					{
						found = true;
						shd[l]++;
						break;
					}

				// .. or append a new cluster
				if (!found)
				{
					adj[count] = c;
					shd[count] = 1;
					count++;
				}
			}

			used[v] = 1;
		}

		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
			used[cluster_indices[j]] = 0;

		// mark the end of the adjacency list; the next cluster will start there as well
		adjacency.offsets[i + 1] = adjacency.offsets[i] + unsigned(count);
	}

	assert(adjacency.offsets[cluster_count] <= total_adjacency);

	// ref_offsets can't be deallocated as it was allocated before adjacency
	allocator.deallocate(ref_data);
}

struct ClusterGroup
{
	int group;
	int next;
	unsigned int size; // 0 unless root
	unsigned int vertices;
};

struct GroupOrder
{
	unsigned int id;
	int order;
};

static void heapPush(GroupOrder* heap, size_t size, GroupOrder item)
{
	// insert a new element at the end (breaks heap invariant)
	heap[size++] = item;

	// bubble up the new element to its correct position
	size_t i = size - 1;
	while (i > 0 && heap[i].order < heap[(i - 1) / 2].order)
	{
		size_t p = (i - 1) / 2;

		GroupOrder temp = heap[i];
		heap[i] = heap[p];
		heap[p] = temp;
		i = p;
	}
}

static GroupOrder heapPop(GroupOrder* heap, size_t size)
{
	assert(size > 0);
	GroupOrder top = heap[0];

	// move the last element to the top (breaks heap invariant)
	heap[0] = heap[--size];

	// bubble down the new top element to its correct position
	size_t i = 0;
	while (i * 2 + 1 < size)
	{
		// find the smallest child
		size_t j = i * 2 + 1;
		j += (j + 1 < size && heap[j + 1].order < heap[j].order);

		// if the parent is already smaller than both children, we're done
		if (heap[j].order >= heap[i].order)
			break;

		// otherwise, swap the parent and child and continue
		GroupOrder temp = heap[i];
		heap[i] = heap[j];
		heap[j] = temp;
		i = j;
	}

	return top;
}

static unsigned int countTotal(const ClusterGroup* groups, int id, const unsigned int* cluster_indices, const unsigned int* cluster_offsets, unsigned char* used)
{
	unsigned int total = 0;

	for (int i = id; i >= 0; i = groups[i].next)
	{
		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
		{
			unsigned int v = cluster_indices[j];
			total += 1 - used[v];
			used[v] = 1;
		}
	}

	for (int i = id; i >= 0; i = groups[i].next)
	{
		for (size_t j = cluster_offsets[i]; j < cluster_offsets[i + 1]; ++j)
			used[cluster_indices[j]] = 0;
	}

	return total;
}

static unsigned int countShared(const ClusterGroup* groups, int group1, int group2, const ClusterAdjacency& adjacency)
{
	unsigned int total = 0;

	for (int i1 = group1; i1 >= 0; i1 = groups[i1].next)
		for (int i2 = group2; i2 >= 0; i2 = groups[i2].next)
		{
			for (unsigned int adj = adjacency.offsets[i1]; adj < adjacency.offsets[i1 + 1]; ++adj)
				if (adjacency.clusters[adj] == unsigned(i2))
				{
					total += adjacency.shared[adj];
					break;
				}
		}

	return total;
}

static int pickGroupToMerge(const ClusterGroup* groups, int id, const ClusterAdjacency& adjacency, size_t max_partition_size)
{
	assert(groups[id].size > 0);

	float group_rsqrt = 1.f / sqrtf(float(int(groups[id].vertices)));

	int best_group = -1;
	float best_score = 0;

	for (int ci = id; ci >= 0; ci = groups[ci].next)
	{
		for (unsigned int adj = adjacency.offsets[ci]; adj != adjacency.offsets[ci + 1]; ++adj)
		{
			int other = groups[adjacency.clusters[adj]].group;
			if (other < 0)
				continue;

			assert(groups[other].size > 0);
			if (groups[id].size + groups[other].size > max_partition_size)
				continue;

			unsigned int shared = countShared(groups, id, other, adjacency);
			float other_rsqrt = 1.f / sqrtf(float(int(groups[other].vertices)));

			// normalize shared count by the expected boundary of each group (+ keeps scoring symmetric)
			float score = float(int(shared)) * (group_rsqrt + other_rsqrt);

			if (score > best_score)
			{
				best_group = other;
				best_score = score;
			}
		}
	}

	return best_group;
}

} // namespace meshopt

size_t meshopt_partitionClusters(unsigned int* destination, const unsigned int* cluster_indices, size_t total_index_count, const unsigned int* cluster_index_counts, size_t cluster_count, size_t vertex_count, size_t target_partition_size)
{
	using namespace meshopt;

	assert(target_partition_size > 0);

	size_t max_partition_size = target_partition_size + target_partition_size * 3 / 8;

	meshopt_Allocator allocator;

	unsigned char* used = allocator.allocate<unsigned char>(vertex_count);
	memset(used, 0, vertex_count);

	// build cluster index offsets as a prefix sum
	unsigned int* cluster_offsets = allocator.allocate<unsigned int>(cluster_count + 1);
	unsigned int cluster_nextoffset = 0;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		assert(cluster_index_counts[i] > 0);

		cluster_offsets[i] = cluster_nextoffset;
		cluster_nextoffset += cluster_index_counts[i];
	}

	assert(cluster_nextoffset == total_index_count);
	cluster_offsets[cluster_count] = unsigned(total_index_count);

	// build cluster adjacency along with edge weights (shared vertex count)
	ClusterAdjacency adjacency = {};
	buildClusterAdjacency(adjacency, cluster_indices, cluster_offsets, cluster_count, used, vertex_count, allocator);

	ClusterGroup* groups = allocator.allocate<ClusterGroup>(cluster_count);

	GroupOrder* order = allocator.allocate<GroupOrder>(cluster_count);
	size_t pending = 0;

	// create a singleton group for each cluster and order them by priority
	for (size_t i = 0; i < cluster_count; ++i)
	{
		groups[i].group = int(i);
		groups[i].next = -1;
		groups[i].size = 1;
		groups[i].vertices = countTotal(groups, int(i), cluster_indices, cluster_offsets, used);

		GroupOrder item = {};
		item.id = unsigned(i);
		item.order = groups[i].vertices;

		heapPush(order, pending++, item);
	}

	// iteratively merge the smallest group with the best group
	while (pending)
	{
		GroupOrder top = heapPop(order, pending--);

		// this group was merged into another group earlier
		if (groups[top.id].size == 0)
			continue;

		// disassociate clusters from the group to prevent them from being merged again; we will re-associate them if the group is reinserted
		for (int i = top.id; i >= 0; i = groups[i].next)
		{
			assert(groups[i].group == int(top.id));
			groups[i].group = -1;
		}

		// the group is large enough, emit as is
		if (groups[top.id].size >= target_partition_size)
			continue;

		int best_group = pickGroupToMerge(groups, top.id, adjacency, max_partition_size);

		// we can't grow the group any more, emit as is
		if (best_group == -1)
			continue;

		// compute shared vertices to adjust the total vertices estimate after merging
		unsigned int shared = countShared(groups, top.id, best_group, adjacency);

		// combine groups by linking them together
		assert(groups[best_group].size > 0);

		for (int i = top.id; i >= 0; i = groups[i].next)
			if (groups[i].next < 0)
			{
				groups[i].next = best_group;
				break;
			}

		// update group sizes; note, the vertex update is an approximation which avoids recomputing the true size via countTotal
		groups[top.id].size += groups[best_group].size;
		groups[top.id].vertices += groups[best_group].vertices;
		groups[top.id].vertices = (groups[top.id].vertices > shared) ? groups[top.id].vertices - shared : 1;

		groups[best_group].size = 0;
		groups[best_group].vertices = 0;

		// re-associate all clusters back to the merged group
		for (int i = top.id; i >= 0; i = groups[i].next)
			groups[i].group = int(top.id);

		top.order = groups[top.id].vertices;
		heapPush(order, pending++, top);
	}

	size_t next_group = 0;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		if (groups[i].size == 0)
			continue;

		for (int j = int(i); j >= 0; j = groups[j].next)
			destination[j] = unsigned(next_group);

		next_group++;
	}

	assert(next_group <= cluster_count);
	return next_group;
}

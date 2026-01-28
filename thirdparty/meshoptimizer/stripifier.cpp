// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <limits.h>
#include <string.h>

// This work is based on:
// Francine Evans, Steven Skiena and Amitabh Varshney. Optimizing Triangle Strips for Fast Rendering. 1996
namespace meshopt
{

static unsigned int findStripFirst(const unsigned int buffer[][3], unsigned int buffer_size, const unsigned char* valence)
{
	unsigned int index = 0;
	unsigned int iv = ~0u;

	for (size_t i = 0; i < buffer_size; ++i)
	{
		unsigned char va = valence[buffer[i][0]], vb = valence[buffer[i][1]], vc = valence[buffer[i][2]];
		unsigned int v = (va < vb && va < vc) ? va : (vb < vc ? vb : vc);

		if (v < iv)
		{
			index = unsigned(i);
			iv = v;
		}
	}

	return index;
}

static int findStripNext(const unsigned int buffer[][3], unsigned int buffer_size, unsigned int e0, unsigned int e1)
{
	for (size_t i = 0; i < buffer_size; ++i)
	{
		unsigned int a = buffer[i][0], b = buffer[i][1], c = buffer[i][2];

		if (e0 == a && e1 == b)
			return (int(i) << 2) | 2;
		else if (e0 == b && e1 == c)
			return (int(i) << 2) | 0;
		else if (e0 == c && e1 == a)
			return (int(i) << 2) | 1;
	}

	return -1;
}

} // namespace meshopt

size_t meshopt_stripify(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, unsigned int restart_index)
{
	assert(destination != indices);
	assert(index_count % 3 == 0);

	using namespace meshopt;

	meshopt_Allocator allocator;

	const size_t buffer_capacity = 8;

	unsigned int buffer[buffer_capacity][3] = {};
	unsigned int buffer_size = 0;

	size_t index_offset = 0;

	unsigned int strip[2] = {};
	unsigned int parity = 0;

	size_t strip_size = 0;

	// compute vertex valence; this is used to prioritize starting triangle for strips
	// note: we use 8-bit counters for performance; for outlier vertices the valence is incorrect but that just affects the heuristic
	unsigned char* valence = allocator.allocate<unsigned char>(vertex_count);
	memset(valence, 0, vertex_count);

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		valence[index]++;
	}

	int next = -1;

	while (buffer_size > 0 || index_offset < index_count)
	{
		assert(next < 0 || (size_t(next >> 2) < buffer_size && (next & 3) < 3));

		// fill triangle buffer
		while (buffer_size < buffer_capacity && index_offset < index_count)
		{
			buffer[buffer_size][0] = indices[index_offset + 0];
			buffer[buffer_size][1] = indices[index_offset + 1];
			buffer[buffer_size][2] = indices[index_offset + 2];

			buffer_size++;
			index_offset += 3;
		}

		assert(buffer_size > 0);

		if (next >= 0)
		{
			unsigned int i = next >> 2;
			unsigned int a = buffer[i][0], b = buffer[i][1], c = buffer[i][2];
			unsigned int v = buffer[i][next & 3];

			// ordered removal from the buffer
			memmove(buffer[i], buffer[i + 1], (buffer_size - i - 1) * sizeof(buffer[0]));
			buffer_size--;

			// update vertex valences for strip start heuristic
			valence[a]--;
			valence[b]--;
			valence[c]--;

			// find next triangle (note that edge order flips on every iteration)
			// in some cases we need to perform a swap to pick a different outgoing triangle edge
			// for [a b c], the default strip edge is [b c], but we might want to use [a c]
			int cont = findStripNext(buffer, buffer_size, parity ? strip[1] : v, parity ? v : strip[1]);
			int swap = cont < 0 ? findStripNext(buffer, buffer_size, parity ? v : strip[0], parity ? strip[0] : v) : -1;

			if (cont < 0 && swap >= 0)
			{
				// [a b c] => [a b a c]
				destination[strip_size++] = strip[0];
				destination[strip_size++] = v;

				// next strip has same winding
				// ? a b => b a v
				strip[1] = v;

				next = swap;
			}
			else
			{
				// emit the next vertex in the strip
				destination[strip_size++] = v;

				// next strip has flipped winding
				strip[0] = strip[1];
				strip[1] = v;
				parity ^= 1;

				next = cont;
			}
		}
		else
		{
			// if we didn't find anything, we need to find the next new triangle
			// we use a heuristic to maximize the strip length
			unsigned int i = findStripFirst(buffer, buffer_size, valence);
			unsigned int a = buffer[i][0], b = buffer[i][1], c = buffer[i][2];

			// ordered removal from the buffer
			memmove(buffer[i], buffer[i + 1], (buffer_size - i - 1) * sizeof(buffer[0]));
			buffer_size--;

			// update vertex valences for strip start heuristic
			valence[a]--;
			valence[b]--;
			valence[c]--;

			// we need to pre-rotate the triangle so that we will find a match in the existing buffer on the next iteration
			int ea = findStripNext(buffer, buffer_size, c, b);
			int eb = findStripNext(buffer, buffer_size, a, c);
			int ec = findStripNext(buffer, buffer_size, b, a);

			// in some cases we can have several matching edges; since we can pick any edge, we pick the one with the smallest
			// triangle index in the buffer. this reduces the effect of stripification on ACMR and additionally - for unclear
			// reasons - slightly improves the stripification efficiency
			int mine = INT_MAX;
			mine = (ea >= 0 && mine > ea) ? ea : mine;
			mine = (eb >= 0 && mine > eb) ? eb : mine;
			mine = (ec >= 0 && mine > ec) ? ec : mine;

			if (ea == mine)
			{
				// keep abc
				next = ea;
			}
			else if (eb == mine)
			{
				// abc -> bca
				unsigned int t = a;
				a = b, b = c, c = t;

				next = eb;
			}
			else if (ec == mine)
			{
				// abc -> cab
				unsigned int t = c;
				c = b, b = a, a = t;

				next = ec;
			}

			if (restart_index)
			{
				if (strip_size)
					destination[strip_size++] = restart_index;

				destination[strip_size++] = a;
				destination[strip_size++] = b;
				destination[strip_size++] = c;

				// new strip always starts with the same edge winding
				strip[0] = b;
				strip[1] = c;
				parity = 1;
			}
			else
			{
				if (strip_size)
				{
					// connect last strip using degenerate triangles
					destination[strip_size++] = strip[1];
					destination[strip_size++] = a;
				}

				// note that we may need to flip the emitted triangle based on parity
				// we always end up with outgoing edge "cb" in the end
				unsigned int e0 = parity ? c : b;
				unsigned int e1 = parity ? b : c;

				destination[strip_size++] = a;
				destination[strip_size++] = e0;
				destination[strip_size++] = e1;

				strip[0] = e0;
				strip[1] = e1;
				parity ^= 1;
			}
		}
	}

	return strip_size;
}

size_t meshopt_stripifyBound(size_t index_count)
{
	assert(index_count % 3 == 0);

	// worst case without restarts is 2 degenerate indices and 3 indices per triangle
	// worst case with restarts is 1 restart index and 3 indices per triangle
	return (index_count / 3) * 5;
}

size_t meshopt_unstripify(unsigned int* destination, const unsigned int* indices, size_t index_count, unsigned int restart_index)
{
	assert(destination != indices);

	size_t offset = 0;
	size_t start = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		if (restart_index && indices[i] == restart_index)
		{
			start = i + 1;
		}
		else if (i - start >= 2)
		{
			unsigned int a = indices[i - 2], b = indices[i - 1], c = indices[i];

			// flip winding for odd triangles
			if ((i - start) & 1)
			{
				unsigned int t = a;
				a = b, b = t;
			}

			// although we use restart indices, strip swaps still produce degenerate triangles, so skip them
			if (a != b && a != c && b != c)
			{
				destination[offset + 0] = a;
				destination[offset + 1] = b;
				destination[offset + 2] = c;
				offset += 3;
			}
		}
	}

	return offset;
}

size_t meshopt_unstripifyBound(size_t index_count)
{
	assert(index_count == 0 || index_count >= 3);

	return (index_count == 0) ? 0 : (index_count - 2) * 3;
}

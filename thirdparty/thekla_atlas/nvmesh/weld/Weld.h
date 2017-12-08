// This code is in the public domain -- castanyo@yahoo.es

#ifndef NV_MESH_WELD_H
#define NV_MESH_WELD_H

#include "nvcore/Array.h"
#include "nvcore/Hash.h"
#include "nvcore/Utils.h" // nextPowerOfTwo

#include <string.h> // for memset, memcmp, memcpy

// Weld function to remove array duplicates in linear time using hashing.

namespace nv
{

/// Generic welding routine. This function welds the elements of the array p
/// and returns the cross references in the xrefs array. To compare the elements
/// it uses the given hash and equal functors.
/// 
/// This code is based on the ideas of Ville Miettinen and Pierre Terdiman.
template <class T, class H=Hash<T>, class E=Equal<T> >
struct Weld
{
	// xrefs maps old elements to new elements
	uint operator()(Array<T> & p, Array<uint> & xrefs)
	{
		const uint N = p.size();							// # of input vertices.
		uint outputCount = 0;								// # of output vertices
		uint hashSize = nextPowerOfTwo(N);					// size of the hash table
		uint * hashTable = new uint[hashSize + N];			// hash table + linked list
		uint * next = hashTable + hashSize;					// use bottom part as linked list

		xrefs.resize(N);
		memset( hashTable, NIL, hashSize*sizeof(uint) );	// init hash table (NIL = 0xFFFFFFFF so memset works)

		H hash;
		E equal;
		for (uint i = 0; i < N; i++)
		{
			const T & e = p[i];
			uint32 hashValue = hash(e) & (hashSize-1);
			uint offset = hashTable[hashValue];

			// traverse linked list
			while( offset != NIL && !equal(p[offset], e) )
			{
				offset = next[offset];
			}

			xrefs[i] = offset;

			// no match found - copy vertex & add to hash
			if( offset == NIL )
			{
				// save xref
				xrefs[i] = outputCount;

				// copy element
				p[outputCount] = e;

				// link to hash table
				next[outputCount] = hashTable[hashValue];

				// update hash heads and increase output counter
				hashTable[hashValue] = outputCount++;
			}
		}

		// cleanup
		delete [] hashTable;

		p.resize(outputCount);
		
		// number of output vertices
		return outputCount;
	}
};


/// Reorder the given array accoding to the indices given in xrefs.
template <class T>
void reorderArray(Array<T> & array, const Array<uint> & xrefs)
{
	const uint count = xrefs.count();
	Array<T> new_array;
    new_array.resize(count);

	for(uint i = 0; i < count; i++) {
		new_array[i] = array[xrefs[i]];
	}

	swap(array, new_array);
}

/// Reverse the given array so that new indices point to old indices.
inline void reverseXRefs(Array<uint> & xrefs, uint count)
{
	Array<uint> new_xrefs;
    new_xrefs.resize(count);
	
	for(uint i = 0; i < xrefs.count(); i++) {
		new_xrefs[xrefs[i]] = i;
	}
	
	swap(xrefs, new_xrefs);
}



//
struct WeldN
{
    uint vertexSize;

    WeldN(uint n) : vertexSize(n) {}

	// xrefs maps old elements to new elements
	uint operator()(uint8 * ptr, uint N, Array<uint> & xrefs)
	{
		uint outputCount = 0;								// # of output vertices
		uint hashSize = nextPowerOfTwo(N);					// size of the hash table
		uint * hashTable = new uint[hashSize + N];			// hash table + linked list
		uint * next = hashTable + hashSize;					// use bottom part as linked list

		xrefs.resize(N);
		memset( hashTable, NIL, hashSize*sizeof(uint) );	// init hash table (NIL = 0xFFFFFFFF so memset works)

		for (uint i = 0; i < N; i++)
		{
			const uint8 * vertex = ptr + i * vertexSize;
			uint32 hashValue = sdbmHash(vertex, vertexSize) & (hashSize-1);
			uint offset = hashTable[hashValue];

			// traverse linked list
			while (offset != NIL && memcmp(ptr + offset * vertexSize, vertex, vertexSize) != 0)
			{
				offset = next[offset];
			}

			xrefs[i] = offset;

			// no match found - copy vertex & add to hash
			if (offset == NIL)
			{
				// save xref
				xrefs[i] = outputCount;

				// copy element
                memcpy(ptr + outputCount * vertexSize, vertex, vertexSize);

				// link to hash table
				next[outputCount] = hashTable[hashValue];

				// update hash heads and increase output counter
				hashTable[hashValue] = outputCount++;
			}
		}

		// cleanup
		delete [] hashTable;

		// number of output vertices
		return outputCount;
	}
};


} // nv namespace

#endif // NV_MESH_WELD_H

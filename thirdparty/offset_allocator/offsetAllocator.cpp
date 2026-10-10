// (C) Sebastian Aaltonen 2023
// MIT License (see file: LICENSE)

#include "offsetAllocator.hpp"

#ifdef DEBUG
#include <assert.h>
#define ASSERT(x) assert(x)
//#define DEBUG_VERBOSE
#else
#define ASSERT(x)
#endif

#ifdef DEBUG_VERBOSE
#include <stdio.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstring>

namespace OffsetAllocator
{
    inline uint32 lzcnt_nonzero(uint32 v)
    {
#ifdef _MSC_VER
        unsigned long retVal;
        _BitScanReverse(&retVal, v);
        return 31 - retVal;
#else
        return __builtin_clz(v);
#endif
    }

    inline uint32 tzcnt_nonzero(uint32 v)
    {
#ifdef _MSC_VER
        unsigned long retVal;
        _BitScanForward(&retVal, v);
        return retVal;
#else
        return __builtin_ctz(v);
#endif
    }

    namespace SmallFloat
    {
        static constexpr uint32 MANTISSA_BITS = 3;
        static constexpr uint32 MANTISSA_VALUE = 1 << MANTISSA_BITS;
        static constexpr uint32 MANTISSA_MASK = MANTISSA_VALUE - 1;
    
        // Bin sizes follow floating point (exponent + mantissa) distribution (piecewise linear log approx)
        // This ensures that for each size class, the average overhead percentage stays the same
        uint32 uintToFloatRoundUp(uint32 size)
        {
            uint32 exp = 0;
            uint32 mantissa = 0;
            
            if (size < MANTISSA_VALUE)
            {
                // Denorm: 0..(MANTISSA_VALUE-1)
                mantissa = size;
            }
            else
            {
                // Normalized: Hidden high bit always 1. Not stored. Just like float.
                uint32 leadingZeros = lzcnt_nonzero(size);
                uint32 highestSetBit = 31 - leadingZeros;
                
                uint32 mantissaStartBit = highestSetBit - MANTISSA_BITS;
                exp = mantissaStartBit + 1;
                mantissa = (size >> mantissaStartBit) & MANTISSA_MASK;
                
                uint32 lowBitsMask = (uint32(1) << mantissaStartBit) - 1;
                
                // Round up!
                if ((size & lowBitsMask) != 0)
                    mantissa++;
            }
            
            return (exp << MANTISSA_BITS) + mantissa; // + allows mantissa->exp overflow for round up
        }

        uint32 uintToFloatRoundDown(uint32 size)
        {
            uint32 exp = 0;
            uint32 mantissa = 0;
            
            if (size < MANTISSA_VALUE)
            {
                // Denorm: 0..(MANTISSA_VALUE-1)
                mantissa = size;
            }
            else
            {
                // Normalized: Hidden high bit always 1. Not stored. Just like float.
                uint32 leadingZeros = lzcnt_nonzero(size);
                uint32 highestSetBit = 31 - leadingZeros;
                
                uint32 mantissaStartBit = highestSetBit - MANTISSA_BITS;
                exp = mantissaStartBit + 1;
                mantissa = (size >> mantissaStartBit) & MANTISSA_MASK;
            }
            
            return (exp << MANTISSA_BITS) | mantissa;
        }
    
        uint32 floatToUint(uint32 floatValue)
        {
            uint32 exponent = floatValue >> MANTISSA_BITS;
            uint32 mantissa = floatValue & MANTISSA_MASK;
            if (exponent == 0)
            {
                // Denorms
                return mantissa;
            }
            else
            {
                return (mantissa | MANTISSA_VALUE) << (exponent - 1);
            }
        }
    }

    // Utility functions
    uint32 findLowestSetBitAfter(uint32 bitMask, uint32 startBitIndex)
    {
        if (startBitIndex >= 32) return Allocation::NO_SPACE;
        uint32 maskBeforeStartIndex = (uint32(1) << startBitIndex) - 1;
        uint32 maskAfterStartIndex = ~maskBeforeStartIndex;
        uint32 bitsAfter = bitMask & maskAfterStartIndex;
        if (bitsAfter == 0) return Allocation::NO_SPACE;
        return tzcnt_nonzero(bitsAfter);
    }

    // Allocator...
    Allocator::Allocator(uint32 size, uint32 maxAllocs) :
        m_size(size),
        m_maxAllocs(maxAllocs),
        m_nodes(nullptr),
        m_freeNodes(nullptr)
    {
        if (sizeof(NodeIndex) == 2)
        {
            ASSERT(maxAllocs <= static_cast<uint32>(Node::unused) - 1);
        }
        reset();
    }

    Allocator::Allocator(Allocator &&other) :
        m_size(other.m_size),
        m_maxAllocs(other.m_maxAllocs),
        m_freeStorage(other.m_freeStorage),
        m_usedBinsTop(other.m_usedBinsTop),
        m_nodes(other.m_nodes),
        m_freeNodes(other.m_freeNodes),
        m_freeOffset(other.m_freeOffset)
    {
        memcpy(m_usedBins, other.m_usedBins, sizeof(uint8) * NUM_TOP_BINS);
        memcpy(m_binIndices, other.m_binIndices, sizeof(NodeIndex) * NUM_LEAF_BINS);

        other.m_nodes = nullptr;
        other.m_freeNodes = nullptr;
        other.m_freeOffset = 0;
        other.m_maxAllocs = 0;
        other.m_usedBinsTop = 0;
    }

    void Allocator::reset()
    {
        m_freeStorage = 0;
        m_usedBinsTop = 0;
        m_freeOffset = m_maxAllocs;

        for (uint32 i = 0 ; i < NUM_TOP_BINS; i++)
            m_usedBins[i] = 0;
        
        for (uint32 i = 0 ; i < NUM_LEAF_BINS; i++)
            m_binIndices[i] = Node::unused;
        
        if (m_nodes) delete[] m_nodes;
        if (m_freeNodes) delete[] m_freeNodes;

        m_nodes = new Node[m_maxAllocs + 1];
        m_freeNodes = new NodeIndex[m_maxAllocs + 1];
        
        // Freelist is a stack. Nodes in inverse order so that [0] pops first.
        for (uint32 i = 0; i < m_maxAllocs + 1; i++)
        {
            m_freeNodes[i] = static_cast<NodeIndex>(m_maxAllocs - i);
        }
        
        // Start state: Whole storage as one big node
        // Algorithm will split remainders and push them back as smaller nodes
        insertNodeIntoBin(m_size, 0);
    }

    Allocator::~Allocator()
    {        
        delete[] m_nodes;
        delete[] m_freeNodes;
    }
    
    Allocation Allocator::allocate(uint32 size)
    {
        // Out of allocations?
        if (m_freeOffset == Allocation::NO_SPACE)
        {
            return {.offset = Allocation::NO_SPACE, .metadata = Allocation::NO_SPACE_METADATA};
        }
        
        // Round up to bin index to ensure that alloc >= bin
        // Gives us min bin index that fits the size
        uint32 minBinIndex = SmallFloat::uintToFloatRoundUp(size);
        if (minBinIndex == 240) minBinIndex = 239; // 4 GiB bin overflows uint32
        
        uint32 minTopBinIndex = minBinIndex >> TOP_BINS_INDEX_SHIFT;
        uint32 minLeafBinIndex = minBinIndex & LEAF_BINS_INDEX_MASK;
        
        uint32 topBinIndex = minTopBinIndex;
        uint32 leafBinIndex = Allocation::NO_SPACE;

        // If top bin exists, scan its leaf bin. This can fail (NO_SPACE).
        if (m_usedBinsTop & (uint32(1) << topBinIndex))
        {
            leafBinIndex = findLowestSetBitAfter(m_usedBins[topBinIndex], minLeafBinIndex);
        }
    
        // If we didn't find space in top bin, we search top bin from +1
        if (leafBinIndex == Allocation::NO_SPACE)
        {
            topBinIndex = findLowestSetBitAfter(m_usedBinsTop, minTopBinIndex + 1);
            
            // Out of space?
            if (topBinIndex == Allocation::NO_SPACE)
            {
                return {.offset = Allocation::NO_SPACE, .metadata = Allocation::NO_SPACE_METADATA};
            }

            // All leaf bins here fit the alloc, since the top bin was rounded up. Start leaf search from bit 0.
            // NOTE: This search can't fail since at least one leaf bit was set because the top bit was set.
            leafBinIndex = tzcnt_nonzero(m_usedBins[topBinIndex]);
        }
                
        uint32 binIndex = (topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex;
        
        // Pop the top node of the bin. Bin top = node.next.
        NodeIndex nodeIndex = m_binIndices[binIndex];
        Node& node = m_nodes[nodeIndex];
        uint32 nodeTotalSize = node.dataSize;
        if (nodeTotalSize < size)
        {
            return {.offset = Allocation::NO_SPACE, .metadata = Allocation::NO_SPACE_METADATA};
        }
        node.dataSize = size;
        node.used = true;
        m_binIndices[binIndex] = node.binListNext;
        if (node.binListNext != Node::unused) m_nodes[node.binListNext].binListPrev = Node::unused;
        m_freeStorage -= nodeTotalSize;
#ifdef DEBUG_VERBOSE
        printf("Free storage: %u (-%u) (allocate)\n", m_freeStorage, nodeTotalSize);
#endif

        // Bin empty?
        if (m_binIndices[binIndex] == Node::unused)
        {
            // Remove a leaf bin mask bit
            m_usedBins[topBinIndex] &= ~(uint32(1) << leafBinIndex);
            
            // All leaf bins empty?
            if (m_usedBins[topBinIndex] == 0)
            {
                // Remove a top bin mask bit
                m_usedBinsTop &= ~(uint32(1) << topBinIndex);
            }
        }
        
        // Push back reminder N elements to a lower bin
        uint32 reminderSize = nodeTotalSize - size;
        if (reminderSize > 0)
        {
            NodeIndex newNodeIndex = insertNodeIntoBin(reminderSize, node.dataOffset + size);
            
            // Link nodes next to each other so that we can merge them later if both are free
            // And update the old next neighbor to point to the new node (in middle)
            if (node.neighborNext != Node::unused) m_nodes[node.neighborNext].neighborPrev = newNodeIndex;
            m_nodes[newNodeIndex].neighborPrev = nodeIndex;
            m_nodes[newNodeIndex].neighborNext = node.neighborNext;
            node.neighborNext = newNodeIndex;
        }
        
        return {.offset = node.dataOffset, .metadata = nodeIndex};
    }
    
    void Allocator::free(Allocation allocation)
    {
        ASSERT(allocation.metadata != Allocation::NO_SPACE_METADATA);
        if (!m_nodes) return;
        
        NodeIndex nodeIndex = allocation.metadata;
        Node& node = m_nodes[nodeIndex];
        
        // Double delete check
        ASSERT(node.used == true);
        
        // Merge with neighbors...
        uint32 offset = node.dataOffset;
        uint32 size = node.dataSize;
        
        if ((node.neighborPrev != Node::unused) && (m_nodes[node.neighborPrev].used == false))
        {
            // Previous (contiguous) free node: Change offset to previous node offset. Sum sizes
            Node& prevNode = m_nodes[node.neighborPrev];
            offset = prevNode.dataOffset;
            size += prevNode.dataSize;
            
            // Remove node from the bin linked list and put it in the freelist
            removeNodeFromBin(node.neighborPrev);
            
            ASSERT(prevNode.neighborNext == nodeIndex);
            node.neighborPrev = prevNode.neighborPrev;
        }
        
        if ((node.neighborNext != Node::unused) && (m_nodes[node.neighborNext].used == false))
        {
            // Next (contiguous) free node: Offset remains the same. Sum sizes.
            Node& nextNode = m_nodes[node.neighborNext];
            size += nextNode.dataSize;
            
            // Remove node from the bin linked list and put it in the freelist
            removeNodeFromBin(node.neighborNext);
            
            ASSERT(nextNode.neighborPrev == nodeIndex);
            node.neighborNext = nextNode.neighborNext;
        }

        NodeIndex neighborNext = node.neighborNext;
        NodeIndex neighborPrev = node.neighborPrev;
        
        // Insert the removed node to freelist
#ifdef DEBUG_VERBOSE
        printf("Putting node %u into freelist[%u] (free)\n", nodeIndex, m_freeOffset + 1);
#endif
        m_freeNodes[++m_freeOffset] = nodeIndex;

        // Insert the (combined) free node to bin
        NodeIndex combinedNodeIndex = insertNodeIntoBin(size, offset);

        // Connect neighbors with the new combined node
        if (neighborNext != Node::unused)
        {
            m_nodes[combinedNodeIndex].neighborNext = neighborNext;
            m_nodes[neighborNext].neighborPrev = combinedNodeIndex;
        }
        if (neighborPrev != Node::unused)
        {
            m_nodes[combinedNodeIndex].neighborPrev = neighborPrev;
            m_nodes[neighborPrev].neighborNext = combinedNodeIndex;
        }
    }

    NodeIndex Allocator::insertNodeIntoBin(uint32 size, uint32 dataOffset)
    {
        // Round down to bin index to ensure that bin >= alloc
        uint32 binIndex = SmallFloat::uintToFloatRoundDown(size);
        
        uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
        uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;
        
        // Bin was empty before?
        if (m_binIndices[binIndex] == Node::unused)
        {
            // Set bin mask bits
            m_usedBins[topBinIndex] |= uint32(1) << leafBinIndex;
            m_usedBinsTop |= uint32(1) << topBinIndex;
        }
        
        // Take a freelist node and insert on top of the bin linked list (next = old top)
        NodeIndex topNodeIndex = m_binIndices[binIndex];
        NodeIndex nodeIndex = m_freeNodes[m_freeOffset--];
#ifdef DEBUG_VERBOSE
        printf("Getting node %u from freelist[%u]\n", nodeIndex, m_freeOffset + 1);
#endif
        m_nodes[nodeIndex] = {.dataOffset = dataOffset, .dataSize = size, .binListNext = topNodeIndex};
        if (topNodeIndex != Node::unused) m_nodes[topNodeIndex].binListPrev = nodeIndex;
        m_binIndices[binIndex] = nodeIndex;
        
        m_freeStorage += size;
#ifdef DEBUG_VERBOSE
        printf("Free storage: %u (+%u) (insertNodeIntoBin)\n", m_freeStorage, size);
#endif

        return nodeIndex;
    }
    
    void Allocator::removeNodeFromBin(NodeIndex nodeIndex)
    {
        Node &node = m_nodes[nodeIndex];
        
        if (node.binListPrev != Node::unused)
        {
            // Easy case: We have previous node. Just remove this node from the middle of the list.
            m_nodes[node.binListPrev].binListNext = node.binListNext;
            if (node.binListNext != Node::unused) m_nodes[node.binListNext].binListPrev = node.binListPrev;
        }
        else
        {
            // Hard case: We are the first node in a bin. Find the bin.
            
            // Round down to bin index to ensure that bin >= alloc
            uint32 binIndex = SmallFloat::uintToFloatRoundDown(node.dataSize);
            
            uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
            uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;
            
            m_binIndices[binIndex] = node.binListNext;
            if (node.binListNext != Node::unused) m_nodes[node.binListNext].binListPrev = Node::unused;

            // Bin empty?
            if (m_binIndices[binIndex] == Node::unused)
            {
                // Remove a leaf bin mask bit
                m_usedBins[topBinIndex] &= ~(uint32(1) << leafBinIndex);
                
                // All leaf bins empty?
                if (m_usedBins[topBinIndex] == 0)
                {
                    // Remove a top bin mask bit
                    m_usedBinsTop &= ~(uint32(1) << topBinIndex);
                }
            }
        }
        
        // Insert the node to freelist
#ifdef DEBUG_VERBOSE
        printf("Putting node %u into freelist[%u] (removeNodeFromBin)\n", nodeIndex, m_freeOffset + 1);
#endif
        m_freeNodes[++m_freeOffset] = nodeIndex;

        m_freeStorage -= node.dataSize;
#ifdef DEBUG_VERBOSE
        printf("Free storage: %u (-%u) (removeNodeFromBin)\n", m_freeStorage, node.dataSize);
#endif
    }

    uint32 Allocator::allocationSize(Allocation allocation) const
    {
        if (allocation.metadata == Allocation::NO_SPACE_METADATA) return 0;
        if (!m_nodes) return 0;
        
        return m_nodes[allocation.metadata].dataSize;
    }

    StorageReport Allocator::storageReport() const
    {
        uint32 largestFreeRegion = 0;
        uint32 freeStorage = 0;
        
        // Out of allocations? -> Zero free space
        if ((m_nodes != nullptr) && (m_freeOffset != Allocation::NO_SPACE))
        {
            freeStorage = m_freeStorage;
            if (m_usedBinsTop)
            {
                uint32 topBinIndex = 31 - lzcnt_nonzero(m_usedBinsTop);
                uint32 leafBinIndex = 31 - lzcnt_nonzero(m_usedBins[topBinIndex]);
                largestFreeRegion = SmallFloat::floatToUint((topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex);
                ASSERT(freeStorage >= largestFreeRegion);
            }
        }

        return {.totalFreeSpace = freeStorage, .largestFreeRegion = largestFreeRegion};
    }

    StorageReportFull Allocator::storageReportFull() const
    {
        StorageReportFull report;
        for (uint32 i = 0; i < NUM_LEAF_BINS; i++)
        {
            uint32 count = 0;
            NodeIndex nodeIndex = m_binIndices[i];
            while (nodeIndex != Node::unused)
            {
                nodeIndex = m_nodes[nodeIndex].binListNext;
                count++;
            }
            report.freeRegions[i] = { .size = SmallFloat::floatToUint(i), .count = count };
        }
        return report;
    }
}

//
// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

/** \mainpage D3D12 Memory Allocator

<b>Version 3.1.0-development</b> (2025-XX-XX)

Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved. \n
License: MIT

Documentation of all members: D3D12MemAlloc.h

\section main_table_of_contents Table of contents

- \subpage faq
- \subpage quick_start
    - [Project setup](@ref quick_start_project_setup)
    - [Creating resources](@ref quick_start_creating_resources)
    - [Resource reference counting](@ref quick_start_resource_reference_counting)
    - [Mapping memory](@ref quick_start_mapping_memory)
    - [Helper structures](@ref quick_start_helper_structures)
- \subpage custom_pools
- \subpage optimal_allocation
    - [Avoiding running out of memory](@ref optimal_allocation_avoiding_running_out_of_memory)
    - [Allocation performance](@ref optimal_allocation_allocation_Performance)
    - [Sub-allocating buffers](@ref optimal_allocation_suballocating_buffers)
    - [Residency priority](@ref optimal_allocation_residency_priority)
    - [GPU upload heap](@ref optimal_allocation_gpu_upload_heap)
    - [Committed versus placed resources](@ref optimal_allocation_committed_vs_placed)
    - [Resource alignment](@ref optimal_allocation_resource_alignment)
- \subpage defragmentation
- \subpage statistics
- \subpage resource_aliasing
- \subpage linear_algorithm
- \subpage virtual_allocator
- \subpage configuration
  - [Custom CPU memory allocator](@ref custom_memory_allocator)
  - [Debug margins](@ref debug_margins)
- \subpage general_considerations
  - [Thread safety](@ref general_considerations_thread_safety)
  - [Versioning and compatibility](@ref general_considerations_versioning_and_compatibility)
  - [Features not supported](@ref general_considerations_features_not_supported)
        
\section main_see_also Web links

- [Direct3D 12 Memory Allocator at GPUOpen.com](https://gpuopen.com/d3d12-memory-allocator/) - product page
- [GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator at GitHub.com](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator) - source code repository
*/

// If using this library on a platform different than Windows PC or want to use different version of DXGI,
// you should include D3D12-compatible headers before this library on your own and define 
// D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED.
// Alternatively, if you are targeting the open sourced DirectX headers, defining D3D12MA_USING_DIRECTX_HEADERS
// will include them rather the ones provided by the Windows SDK.
#ifndef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
    #if defined(D3D12MA_USING_DIRECTX_HEADERS)
        #include <directx/d3d12.h>
        #include <dxguids/dxguids.h>
    #else
        #include <d3d12.h>
    #endif
    
    #include <dxgi1_4.h>
#endif

#ifndef D3D12MA_DXGI_1_4
    #ifdef __IDXGIAdapter3_INTERFACE_DEFINED__
        /// Define this macro to 0 to disable usage of DXGI 1.4 (which is used for `IDXGIAdapter3` and query for memory budget).
        #define D3D12MA_DXGI_1_4 1
    #else
        #define D3D12MA_DXGI_1_4 0
    #endif
#endif

#ifndef D3D12MA_CREATE_NOT_ZEROED_AVAILABLE
    #ifdef __ID3D12Device8_INTERFACE_DEFINED__
        /// This macro is defined to 0 or 1 automatically. Define it to 0 to disable support for `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED`.
        #define D3D12MA_CREATE_NOT_ZEROED_AVAILABLE 1
    #else
        #define D3D12MA_CREATE_NOT_ZEROED_AVAILABLE 0
    #endif
#endif

#ifndef D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT
    /** \brief
    When defined to value other than 0, the library will try to use
    `D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT` or `D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT`
    for created textures when possible, which can save memory because some small textures
    may get their alignment 4 KB and their size a multiply of 4 KB instead of 64 KB.

    - `#define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 0` -
      Disables small texture alignment.
    - `#define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 1` (the default) -
      Enables conservative algorithm that will use small alignment only for some textures
      that are surely known to support it.
    - `#define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 2` -
      Enables query for small alignment to D3D12 (based on Microsoft sample) which will
      enable small alignment for more textures, but will also generate D3D Debug Layer
      error #721 on call to `ID3D12Device::GetResourceAllocationInfo`, which you should just
      ignore.
    */
    #define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 1
#endif

#ifndef D3D12MA_RECOMMENDED_ALLOCATOR_FLAGS
    /// Set of flags recommended for use in D3D12MA::ALLOCATOR_DESC::Flags for optimal performance.
    #define D3D12MA_RECOMMENDED_ALLOCATOR_FLAGS (D3D12MA::ALLOCATOR_FLAG_DEFAULT_POOLS_NOT_ZEROED | D3D12MA::ALLOCATOR_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED)
#endif

#ifndef D3D12MA_RECOMMENDED_HEAP_FLAGS
    #if D3D12MA_CREATE_NOT_ZEROED_AVAILABLE
        #define D3D12MA_RECOMMENDED_HEAP_FLAGS (D3D12_HEAP_FLAG_CREATE_NOT_ZEROED)
    #else
        /// Set of flags recommended for use in D3D12MA::POOL_DESC::HeapFlags for optimal performance.
        #define D3D12MA_RECOMMENDED_HEAP_FLAGS (D3D12_HEAP_FLAG_NONE)
    #endif
#endif

#ifndef D3D12MA_RECOMMENDED_POOL_FLAGS
    /// Set of flags recommended for use in D3D12MA::POOL_DESC::Flags for optimal performance.
    #define D3D12MA_RECOMMENDED_POOL_FLAGS (D3D12MA::POOL_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED)
#endif


/// \cond INTERNAL

#define D3D12MA_CLASS_NO_COPY(className) \
    private: \
        className(const className&) = delete; \
        className(className&&) = delete; \
        className& operator=(const className&) = delete; \
        className& operator=(className&&) = delete;

// To be used with MAKE_HRESULT to define custom error codes.
#define FACILITY_D3D12MA 3542

/*
If providing your own implementation, you need to implement a subset of std::atomic.
*/
#if !defined(D3D12MA_ATOMIC_UINT32) || !defined(D3D12MA_ATOMIC_UINT64)
    #include <atomic>
#endif

#ifndef D3D12MA_ATOMIC_UINT32
    #define D3D12MA_ATOMIC_UINT32 std::atomic<UINT>
#endif

#ifndef D3D12MA_ATOMIC_UINT64
    #define D3D12MA_ATOMIC_UINT64 std::atomic<UINT64>
#endif

#ifdef D3D12MA_EXPORTS
    #define D3D12MA_API __declspec(dllexport)
#elif defined(D3D12MA_IMPORTS)
    #define D3D12MA_API __declspec(dllimport)
#else
    #define D3D12MA_API
#endif

// Forward declaration if ID3D12ProtectedResourceSession is not defined inside the headers (older SDK, pre ID3D12Device4)
struct ID3D12ProtectedResourceSession;

// Define this enum even if SDK doesn't provide it, to simplify the API.
#ifndef __ID3D12Device1_INTERFACE_DEFINED__
typedef enum D3D12_RESIDENCY_PRIORITY
{
    D3D12_RESIDENCY_PRIORITY_MINIMUM = 0x28000000,
    D3D12_RESIDENCY_PRIORITY_LOW = 0x50000000,
    D3D12_RESIDENCY_PRIORITY_NORMAL = 0x78000000,
    D3D12_RESIDENCY_PRIORITY_HIGH = 0xa0010000,
    D3D12_RESIDENCY_PRIORITY_MAXIMUM = 0xc8000000
} D3D12_RESIDENCY_PRIORITY;
#endif

namespace D3D12MA
{
class D3D12MA_API IUnknownImpl : public IUnknown
{
public:
    virtual ~IUnknownImpl() = default;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
protected:
    virtual void ReleaseThis() { delete this; }
private:
    D3D12MA_ATOMIC_UINT32 m_RefCount = {1};
};
} // namespace D3D12MA

/// \endcond

namespace D3D12MA
{

/// \cond INTERNAL
class DefragmentationContextPimpl;
class AllocatorPimpl;
class PoolPimpl;
class NormalBlock;
class BlockVector;
class CommittedAllocationList;
class JsonWriter;
class VirtualBlockPimpl;
/// \endcond

class Pool;
class Allocator;
struct Statistics;
struct DetailedStatistics;
struct TotalStatistics;

/// \brief Unique identifier of single allocation done inside the memory heap.
typedef UINT64 AllocHandle;

/// Pointer to custom callback function that allocates CPU memory.
using ALLOCATE_FUNC_PTR = void* (*)(size_t Size, size_t Alignment, void* pPrivateData);
/**
\brief Pointer to custom callback function that deallocates CPU memory.

`pMemory = null` should be accepted and ignored.
*/
using FREE_FUNC_PTR = void (*)(void* pMemory, void* pPrivateData);

/// Custom callbacks to CPU memory allocation functions.
struct ALLOCATION_CALLBACKS
{
    /// %Allocation function.
    ALLOCATE_FUNC_PTR pAllocate;
    /// Dellocation function.
    FREE_FUNC_PTR pFree;
    /// Custom data that will be passed to allocation and deallocation functions as `pUserData` parameter.
    void* pPrivateData;
};


/// \brief Bit flags to be used with ALLOCATION_DESC::Flags.
enum ALLOCATION_FLAGS
{
    /// Zero
    ALLOCATION_FLAG_NONE = 0,

    /**
    Set this flag if the allocation should have its own dedicated memory allocation (committed resource with implicit heap).
    
    Use it for special, big resources, like fullscreen textures used as render targets.

    - When used with functions like D3D12MA::Allocator::CreateResource, it will use `ID3D12Device::CreateCommittedResource`,
      so the created allocation will contain a resource (D3D12MA::Allocation::GetResource() `!= NULL`) but will not have
      a heap (D3D12MA::Allocation::GetHeap() `== NULL`), as the heap is implicit.
    - When used with raw memory allocation like D3D12MA::Allocator::AllocateMemory, it will use `ID3D12Device::CreateHeap`,
      so the created allocation will contain a heap (D3D12MA::Allocation::GetHeap() `!= NULL`) and its offset will always be 0.
    */
    ALLOCATION_FLAG_COMMITTED = 0x1,

    /**
    Set this flag to only try to allocate from existing memory heaps and never create new such heap.

    If new allocation cannot be placed in any of the existing heaps, allocation
    fails with `E_OUTOFMEMORY` error.

    You should not use D3D12MA::ALLOCATION_FLAG_COMMITTED and
    D3D12MA::ALLOCATION_FLAG_NEVER_ALLOCATE at the same time. It makes no sense.
    */
    ALLOCATION_FLAG_NEVER_ALLOCATE = 0x2,

    /** Create allocation only if additional memory required for it, if any, won't exceed
    memory budget. Otherwise return `E_OUTOFMEMORY`.
    */
    ALLOCATION_FLAG_WITHIN_BUDGET = 0x4,

    /** Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for custom pools created with #POOL_FLAG_ALGORITHM_LINEAR flag.
    */
    ALLOCATION_FLAG_UPPER_ADDRESS = 0x8,

    /** Set this flag if the allocated memory will have aliasing resources.
    
    Use this when calling D3D12MA::Allocator::CreateResource() and similar to
    guarantee creation of explicit heap for desired allocation and prevent it from using `CreateCommittedResource`,
    so that new allocation object will always have `allocation->GetHeap() != NULL`.
    */
    ALLOCATION_FLAG_CAN_ALIAS = 0x10,

    /** %Allocation strategy that chooses smallest possible free range for the allocation
    to minimize memory usage and fragmentation, possibly at the expense of allocation time.
    */
    ALLOCATION_FLAG_STRATEGY_MIN_MEMORY = 0x00010000,

    /** %Allocation strategy that chooses first suitable free range for the allocation -
    not necessarily in terms of the smallest offset but the one that is easiest and fastest to find
    to minimize allocation time, possibly at the expense of allocation quality.
    */
    ALLOCATION_FLAG_STRATEGY_MIN_TIME = 0x00020000,

    /** %Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    Used internally by defragmentation, not recomended in typical usage.
    */
    ALLOCATION_FLAG_STRATEGY_MIN_OFFSET = 0x0004000,

    /// Alias to #ALLOCATION_FLAG_STRATEGY_MIN_MEMORY.
    ALLOCATION_FLAG_STRATEGY_BEST_FIT = ALLOCATION_FLAG_STRATEGY_MIN_MEMORY,
    /// Alias to #ALLOCATION_FLAG_STRATEGY_MIN_TIME.
    ALLOCATION_FLAG_STRATEGY_FIRST_FIT = ALLOCATION_FLAG_STRATEGY_MIN_TIME,

    /// A bit mask to extract only `STRATEGY` bits from entire set of flags.
    ALLOCATION_FLAG_STRATEGY_MASK =
        ALLOCATION_FLAG_STRATEGY_MIN_MEMORY |
        ALLOCATION_FLAG_STRATEGY_MIN_TIME |
        ALLOCATION_FLAG_STRATEGY_MIN_OFFSET,
};

/// \brief Parameters of created D3D12MA::Allocation object. To be used with Allocator::CreateResource.
struct ALLOCATION_DESC
{
    /// Flags for the allocation.
    ALLOCATION_FLAGS Flags;
    /** \brief The type of memory heap where the new allocation should be placed.

    It must be one of: `D3D12_HEAP_TYPE_DEFAULT`, `D3D12_HEAP_TYPE_UPLOAD`, `D3D12_HEAP_TYPE_READBACK`.

    When D3D12MA::ALLOCATION_DESC::CustomPool != NULL this member is ignored.
    */
    D3D12_HEAP_TYPE HeapType;
    /** \brief Additional heap flags to be used when allocating memory.

    In most cases it can be 0.
    
    - If you use D3D12MA::Allocator::CreateResource(), you don't need to care.
      Necessary flag `D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS`, `D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES`,
      or `D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES` is added automatically.
    - If you use D3D12MA::Allocator::AllocateMemory(), you should specify one of those `ALLOW_ONLY` flags.
      Except when you validate that D3D12MA::Allocator::GetD3D12Options()`.ResourceHeapTier == D3D12_RESOURCE_HEAP_TIER_1` -
      then you can leave it 0.
    - You can specify additional flags if needed. Then the memory will always be allocated as
      separate block using `D3D12Device::CreateCommittedResource` or `CreateHeap`, not as part of an existing larget block.

    When D3D12MA::ALLOCATION_DESC::CustomPool != NULL this member is ignored.
    */
    D3D12_HEAP_FLAGS ExtraHeapFlags;
    /** \brief Custom pool to place the new resource in. Optional.

    When not null, the resource will be created inside specified custom pool.
    Members `HeapType`, `ExtraHeapFlags` are then ignored.
    */
    Pool* CustomPool;
    /// Custom general-purpose pointer that will be stored in D3D12MA::Allocation.
    void* pPrivateData;
};

/** \brief Calculated statistics of memory usage e.g. in a specific memory heap type,
memory segment group, custom pool, or total.

These are fast to calculate.
See functions: D3D12MA::Allocator::GetBudget(), D3D12MA::Pool::GetStatistics().
*/
struct Statistics
{
    /** \brief Number of D3D12 memory blocks allocated - `ID3D12Heap` objects and committed resources.
    */
    UINT BlockCount;
    /** \brief Number of D3D12MA::Allocation objects allocated.

    Committed allocations have their own blocks, so each one adds 1 to `AllocationCount` as well as `BlockCount`.
    */
    UINT AllocationCount;
    /** \brief Number of bytes allocated in memory blocks.
    */
    UINT64 BlockBytes;
    /** \brief Total number of bytes occupied by all D3D12MA::Allocation objects.

    Always less or equal than `BlockBytes`.
    Difference `(BlockBytes - AllocationBytes)` is the amount of memory allocated from D3D12
    but unused by any D3D12MA::Allocation.
    */
    UINT64 AllocationBytes;
};

/** \brief More detailed statistics than D3D12MA::Statistics.

These are slower to calculate. Use for debugging purposes.
See functions: D3D12MA::Allocator::CalculateStatistics(), D3D12MA::Pool::CalculateStatistics().

Averages are not provided because they can be easily calculated as:

\code
UINT64 AllocationSizeAvg = DetailedStats.Statistics.AllocationBytes / detailedStats.Statistics.AllocationCount;
UINT64 UnusedBytes = DetailedStats.Statistics.BlockBytes - DetailedStats.Statistics.AllocationBytes;
UINT64 UnusedRangeSizeAvg = UnusedBytes / DetailedStats.UnusedRangeCount;
\endcode
*/
struct DetailedStatistics
{
    /// Basic statistics.
    Statistics Stats;
    /// Number of free ranges of memory between allocations.
    UINT UnusedRangeCount;
    /// Smallest allocation size. `UINT64_MAX` if there are 0 allocations.
    UINT64 AllocationSizeMin;
    /// Largest allocation size. 0 if there are 0 allocations.
    UINT64 AllocationSizeMax;
    /// Smallest empty range size. `UINT64_MAX` if there are 0 empty ranges.
    UINT64 UnusedRangeSizeMin;
    /// Largest empty range size. 0 if there are 0 empty ranges.
    UINT64 UnusedRangeSizeMax;
};

/** \brief  General statistics from current state of the allocator -
total memory usage across all memory heaps and segments.

These are slower to calculate. Use for debugging purposes.
See function D3D12MA::Allocator::CalculateStatistics().
*/
struct TotalStatistics
{
    /** \brief One element for each type of heap located at the following indices:

    - 0 = `D3D12_HEAP_TYPE_DEFAULT`
    - 1 = `D3D12_HEAP_TYPE_UPLOAD`
    - 2 = `D3D12_HEAP_TYPE_READBACK`
    - 3 = `D3D12_HEAP_TYPE_CUSTOM`
    - 4 = `D3D12_HEAP_TYPE_GPU_UPLOAD`
    */
    DetailedStatistics HeapType[5];
    /** \brief One element for each memory segment group located at the following indices:

    - 0 = `DXGI_MEMORY_SEGMENT_GROUP_LOCAL`
    - 1 = `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL`

    Meaning of these segment groups is:

    - When `IsUMA() == FALSE` (discrete graphics card):
      - `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` (index 0) represents GPU memory
        (resources allocated in `D3D12_HEAP_TYPE_DEFAULT`, `D3D12_HEAP_TYPE_GPU_UPLOAD` or `D3D12_MEMORY_POOL_L1`).
      - `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL` (index 1) represents system memory
        (resources allocated in `D3D12_HEAP_TYPE_UPLOAD`, `D3D12_HEAP_TYPE_READBACK`, or `D3D12_MEMORY_POOL_L0`).
    - When `IsUMA() == TRUE` (integrated graphics chip):
      - `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` = (index 0) represents memory shared for all the resources.
      - `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL` = (index 1) is unused and always 0.
    */
    DetailedStatistics MemorySegmentGroup[2];
    /// Total statistics from all memory allocated from D3D12.
    DetailedStatistics Total;
};

/** \brief %Statistics of current memory usage and available budget for a specific memory segment group.

These are fast to calculate. See function D3D12MA::Allocator::GetBudget().
*/
struct Budget
{
    /** \brief %Statistics fetched from the library.
    */
    Statistics Stats;
    /** \brief Estimated current memory usage of the program.

    Fetched from system using `IDXGIAdapter3::QueryVideoMemoryInfo` if possible.

    It might be different than `BlockBytes` (usually higher) due to additional implicit objects
    also occupying the memory, like swapchain, pipeline state objects, descriptor heaps, command lists, or
    heaps and resources allocated outside of this library, if any.
    */
    UINT64 UsageBytes;
    /** \brief Estimated amount of memory available to the program.

    Fetched from system using `IDXGIAdapter3::QueryVideoMemoryInfo` if possible.

    It might be different (most probably smaller) than memory capacity returned
    by D3D12MA::Allocator::GetMemoryCapacity() due to factors
    external to the program, decided by the operating system.
    Difference `BudgetBytes - UsageBytes` is the amount of additional memory that can probably
    be allocated without problems. Exceeding the budget may result in various problems.
    */
    UINT64 BudgetBytes;
};


/// \brief Represents single memory allocation done inside VirtualBlock.
struct D3D12MA_API VirtualAllocation
{
    /// \brief Unique idenitfier of current allocation. 0 means null/invalid.
    AllocHandle AllocHandle;
};

/** \brief Represents single memory allocation.

It may be either implicit memory heap dedicated to a single resource or a
specific region of a bigger heap plus unique offset.

To create such object, fill structure D3D12MA::ALLOCATION_DESC and call function
Allocator::CreateResource.

The object remembers size and some other information.
To retrieve this information, use methods of this class.

The object also remembers `ID3D12Resource` and "owns" a reference to it,
so it calls `%Release()` on the resource when destroyed.
*/
class D3D12MA_API Allocation : public IUnknownImpl
{
public:
    /** \brief Returns offset in bytes from the start of memory heap.

    You usually don't need to use this offset. If you create a buffer or a texture together with the allocation using function
    D3D12MA::Allocator::CreateResource, functions that operate on that resource refer to the beginning of the resource,
    not entire memory heap.

    If the Allocation represents committed resource with implicit heap, returns 0.
    */
    UINT64 GetOffset() const;

    /// Returns alignment that resource was created with.
    UINT64 GetAlignment() const { return m_Alignment; }

    /** \brief Returns size in bytes of the allocation.

    - If you created a buffer or a texture together with the allocation using function D3D12MA::Allocator::CreateResource,
      this is the size of the resource returned by `ID3D12Device::GetResourceAllocationInfo`.
    - For allocations made out of bigger memory blocks, this also is the size of the memory region assigned exclusively to this allocation.
    - For resources created as committed, this value may not be accurate. DirectX implementation may optimize memory usage internally
      so that you may even observe regions of `ID3D12Resource::GetGPUVirtualAddress()` + Allocation::GetSize() to overlap in memory and still work correctly.
    */
    UINT64 GetSize() const { return m_Size; }

    /** \brief Returns D3D12 resource associated with this object.

    Calling this method doesn't increment resource's reference counter.
    */
    ID3D12Resource* GetResource() const { return m_Resource; }

    /** \brief Releases the resource currently pointed by the allocation (if not null), sets it to new one, incrementing its reference counter (if not null).
    
    \warning
    This is an advanced feature that should be used only in special cases, e.g. during \subpage defragmentation.
    Typically, an allocation object should reference the resource that was created together with it.
    If you swap it to another resource of different size, \subpage statistics and budgets can be calculated incorrectly.
    */
    void SetResource(ID3D12Resource* pResource);

    /** \brief Returns memory heap that the resource is created in.

    If the Allocation represents committed resource with implicit heap, returns NULL.
    */
    ID3D12Heap* GetHeap() const;

    /// Changes custom pointer for an allocation to a new value.
    void SetPrivateData(void* pPrivateData) { m_pPrivateData = pPrivateData; }

    /// Get custom pointer associated with the allocation.
    void* GetPrivateData() const { return m_pPrivateData; }

    /** \brief Associates a name with the allocation object. This name is for use in debug diagnostics and tools.

    Internal copy of the string is made, so the memory pointed by the argument can be
    changed of freed immediately after this call.

    `Name` can be null.
    */
    void SetName(LPCWSTR Name);

    /** \brief Returns the name associated with the allocation object.

    Returned string points to an internal copy.

    If no name was associated with the allocation, returns null.
    */
    LPCWSTR GetName() const { return m_Name; }

protected:
    void ReleaseThis() override;

private:
    friend class AllocatorPimpl;
    friend class BlockVector;
    friend class CommittedAllocationList;
    friend class JsonWriter;
    friend class BlockMetadata_Linear;
    friend class DefragmentationContextPimpl;
    friend struct CommittedAllocationListItemTraits;
    template<typename T> friend void D3D12MA_DELETE(const ALLOCATION_CALLBACKS&, T*);
    template<typename T> friend class PoolAllocator;

    enum Type
    {
        TYPE_COMMITTED,
        TYPE_PLACED,
        TYPE_HEAP,
        TYPE_COUNT
    };

    AllocatorPimpl* m_Allocator;
    UINT64 m_Size;
    UINT64 m_Alignment;
    ID3D12Resource* m_Resource;
    void* m_pPrivateData;
    wchar_t* m_Name;

    union
    {
        struct
        {
            CommittedAllocationList* list;
            Allocation* prev;
            Allocation* next;
        } m_Committed;

        struct
        {
            AllocHandle allocHandle;
            NormalBlock* block;
        } m_Placed;

        struct
        {
            // Beginning must be compatible with m_Committed.
            CommittedAllocationList* list;
            Allocation* prev;
            Allocation* next;
            ID3D12Heap* heap;
        } m_Heap;
    };

    struct PackedData
    {
    public:
        PackedData() :
            m_Type(0), m_ResourceDimension(0), m_ResourceFlags(0), m_TextureLayout(0) { }

        Type GetType() const { return (Type)m_Type; }
        D3D12_RESOURCE_DIMENSION GetResourceDimension() const { return (D3D12_RESOURCE_DIMENSION)m_ResourceDimension; }
        D3D12_RESOURCE_FLAGS GetResourceFlags() const { return (D3D12_RESOURCE_FLAGS)m_ResourceFlags; }
        D3D12_TEXTURE_LAYOUT GetTextureLayout() const { return (D3D12_TEXTURE_LAYOUT)m_TextureLayout; }

        void SetType(Type type);
        void SetResourceDimension(D3D12_RESOURCE_DIMENSION resourceDimension);
        void SetResourceFlags(D3D12_RESOURCE_FLAGS resourceFlags);
        void SetTextureLayout(D3D12_TEXTURE_LAYOUT textureLayout);

    private:
        UINT m_Type : 2;               // enum Type
        UINT m_ResourceDimension : 3;  // enum D3D12_RESOURCE_DIMENSION
        UINT m_ResourceFlags : 24;     // flags D3D12_RESOURCE_FLAGS
        UINT m_TextureLayout : 9;      // enum D3D12_TEXTURE_LAYOUT
    } m_PackedData;

    Allocation(AllocatorPimpl* allocator, UINT64 size, UINT64 alignment);
    //  Nothing here, everything already done in Release.
    virtual ~Allocation() = default;

    void InitCommitted(CommittedAllocationList* list);
    void InitPlaced(AllocHandle allocHandle, NormalBlock* block);
    void InitHeap(CommittedAllocationList* list, ID3D12Heap* heap);
    void SwapBlockAllocation(Allocation* allocation);
    // If the Allocation represents committed resource with implicit heap, returns UINT64_MAX.
    AllocHandle GetAllocHandle() const;
    NormalBlock* GetBlock();
    template<typename D3D12_RESOURCE_DESC_T>
    void SetResourcePointer(ID3D12Resource* resource, const D3D12_RESOURCE_DESC_T* pResourceDesc);
    void FreeName();

    D3D12MA_CLASS_NO_COPY(Allocation)
};


/// Flags to be passed as DEFRAGMENTATION_DESC::Flags.
enum DEFRAGMENTATION_FLAGS
{
    /** Use simple but fast algorithm for defragmentation.
    May not achieve best results but will require least time to compute and least allocations to copy.
    */
    DEFRAGMENTATION_FLAG_ALGORITHM_FAST = 0x1,
    /** Default defragmentation algorithm, applied also when no `ALGORITHM` flag is specified.
    Offers a balance between defragmentation quality and the amount of allocations and bytes that need to be moved.
    */
    DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED = 0x2,
    /** Perform full defragmentation of memory.
    Can result in notably more time to compute and allocations to copy, but will achieve best memory packing.
    */
    DEFRAGMENTATION_FLAG_ALGORITHM_FULL = 0x4,

    /// A bit mask to extract only `ALGORITHM` bits from entire set of flags.
    DEFRAGMENTATION_FLAG_ALGORITHM_MASK =
        DEFRAGMENTATION_FLAG_ALGORITHM_FAST |
        DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED |
        DEFRAGMENTATION_FLAG_ALGORITHM_FULL
};

/** \brief Parameters for defragmentation.

To be used with functions Allocator::BeginDefragmentation() and Pool::BeginDefragmentation().
*/
struct DEFRAGMENTATION_DESC
{
    /// Flags.
    DEFRAGMENTATION_FLAGS Flags;
    /** \brief Maximum numbers of bytes that can be copied during single pass, while moving allocations to different places.

    0 means no limit.
    */
    UINT64 MaxBytesPerPass;
    /** \brief Maximum number of allocations that can be moved during single pass to a different place.

    0 means no limit.
    */
    UINT32 MaxAllocationsPerPass;
};

/// Operation performed on single defragmentation move.
enum DEFRAGMENTATION_MOVE_OPERATION
{
    /** Resource has been recreated at `pDstTmpAllocation`, data has been copied, old resource has been destroyed.
    `pSrcAllocation` will be changed to point to the new place. This is the default value set by DefragmentationContext::BeginPass().
    */
    DEFRAGMENTATION_MOVE_OPERATION_COPY = 0,
    /// Set this value if you cannot move the allocation. New place reserved at `pDstTmpAllocation` will be freed. `pSrcAllocation` will remain unchanged.
    DEFRAGMENTATION_MOVE_OPERATION_IGNORE = 1,
    /// Set this value if you decide to abandon the allocation and you destroyed the resource. New place reserved `pDstTmpAllocation` will be freed, along with `pSrcAllocation`.
    DEFRAGMENTATION_MOVE_OPERATION_DESTROY = 2,
};

/// Single move of an allocation to be done for defragmentation.
struct DEFRAGMENTATION_MOVE
{
    /** \brief Operation to be performed on the allocation by DefragmentationContext::EndPass().
    Default value is #DEFRAGMENTATION_MOVE_OPERATION_COPY. You can modify it.
    */
    DEFRAGMENTATION_MOVE_OPERATION Operation;
    /// %Allocation that should be moved.
    Allocation* pSrcAllocation;
    /** \brief Temporary allocation pointing to destination memory that will replace `pSrcAllocation`.

    Use it to retrieve new `ID3D12Heap` and offset to create new `ID3D12Resource` and then store it here via Allocation::SetResource().

    \warning Do not store this allocation in your data structures! It exists only temporarily, for the duration of the defragmentation pass,
    to be used for storing newly created resource. DefragmentationContext::EndPass() will destroy it and make `pSrcAllocation` point to this memory.
    */
    Allocation* pDstTmpAllocation;
};

/** \brief Parameters for incremental defragmentation steps.

To be used with function DefragmentationContext::BeginPass().
*/
struct DEFRAGMENTATION_PASS_MOVE_INFO
{
    /// Number of elements in the `pMoves` array.
    UINT32 MoveCount;
    /** \brief Array of moves to be performed by the user in the current defragmentation pass.

    Pointer to an array of `MoveCount` elements, owned by %D3D12MA, created in DefragmentationContext::BeginPass(), destroyed in DefragmentationContext::EndPass().

    For each element, you should:

    1. Create a new resource in the place pointed by `pMoves[i].pDstTmpAllocation->GetHeap()` + `pMoves[i].pDstTmpAllocation->GetOffset()`.
    2. Store new resource in `pMoves[i].pDstTmpAllocation` by using Allocation::SetResource(). It will later replace old resource from `pMoves[i].pSrcAllocation`.
    3. Copy data from the `pMoves[i].pSrcAllocation` e.g. using `D3D12GraphicsCommandList::CopyResource`.
    4. Make sure these commands finished executing on the GPU.

    Only then you can finish defragmentation pass by calling DefragmentationContext::EndPass().
    After this call, the allocation will point to the new place in memory.

    Alternatively, if you cannot move specific allocation,
    you can set DEFRAGMENTATION_MOVE::Operation to D3D12MA::DEFRAGMENTATION_MOVE_OPERATION_IGNORE.

    Alternatively, if you decide you want to completely remove the allocation,
    set DEFRAGMENTATION_MOVE::Operation to D3D12MA::DEFRAGMENTATION_MOVE_OPERATION_DESTROY.
    Then, after DefragmentationContext::EndPass() the allocation will be released.
    */
    DEFRAGMENTATION_MOVE* pMoves;
};

/// %Statistics returned for defragmentation process by function DefragmentationContext::GetStats().
struct DEFRAGMENTATION_STATS
{
    /// Total number of bytes that have been copied while moving allocations to different places.
    UINT64 BytesMoved;
    /// Total number of bytes that have been released to the system by freeing empty heaps.
    UINT64 BytesFreed;
    /// Number of allocations that have been moved to different places.
    UINT32 AllocationsMoved;
    /// Number of empty `ID3D12Heap` objects that have been released to the system.
    UINT32 HeapsFreed;
};

/** \brief Represents defragmentation process in progress.

You can create this object using Allocator::BeginDefragmentation (for default pools) or
Pool::BeginDefragmentation (for a custom pool).
*/
class D3D12MA_API DefragmentationContext : public IUnknownImpl
{
public:
    /** \brief Starts single defragmentation pass.

    \param[out] pPassInfo Computed informations for current pass.
    \returns
    - `S_OK` if no more moves are possible. Then you can omit call to DefragmentationContext::EndPass() and simply end whole defragmentation.
    - `S_FALSE` if there are pending moves returned in `pPassInfo`. You need to perform them, call DefragmentationContext::EndPass(),
      and then preferably try another pass with DefragmentationContext::BeginPass().
    */
    HRESULT BeginPass(DEFRAGMENTATION_PASS_MOVE_INFO* pPassInfo);
    /** \brief Ends single defragmentation pass.

    \param pPassInfo Computed informations for current pass filled by DefragmentationContext::BeginPass() and possibly modified by you.
    \return Returns `S_OK` if no more moves are possible or `S_FALSE` if more defragmentations are possible.

    Ends incremental defragmentation pass and commits all defragmentation moves from `pPassInfo`.
    After this call:

    - %Allocation at `pPassInfo[i].pSrcAllocation` that had `pPassInfo[i].Operation ==` #DEFRAGMENTATION_MOVE_OPERATION_COPY
      (which is the default) will be pointing to the new destination place.
    - %Allocation at `pPassInfo[i].pSrcAllocation` that had `pPassInfo[i].operation ==` #DEFRAGMENTATION_MOVE_OPERATION_DESTROY
      will be released.

    If no more moves are possible you can end whole defragmentation.
    */
    HRESULT EndPass(DEFRAGMENTATION_PASS_MOVE_INFO* pPassInfo);
    /** \brief Returns statistics of the defragmentation performed so far.
    */
    void GetStats(DEFRAGMENTATION_STATS* pStats);

protected:
    void ReleaseThis() override;

private:
    friend class Pool;
    friend class Allocator;
    template<typename T> friend void D3D12MA_DELETE(const ALLOCATION_CALLBACKS&, T*);

    DefragmentationContextPimpl* m_Pimpl;

    DefragmentationContext(AllocatorPimpl* allocator,
        const DEFRAGMENTATION_DESC& desc,
        BlockVector* poolVector);
    ~DefragmentationContext();

    D3D12MA_CLASS_NO_COPY(DefragmentationContext)
};

/// \brief Bit flags to be used with POOL_DESC::Flags.
enum POOL_FLAGS
{
    /// Zero
    POOL_FLAG_NONE = 0,

    /** Enables alternative, linear allocation algorithm in this pool.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    POOL_FLAG_ALGORITHM_LINEAR = 0x1,

    /** Optimization, allocate MSAA textures as committed resources always.
    
    Specify this flag to create MSAA textures with implicit heaps, as if they were created
    with flag D3D12MA::ALLOCATION_FLAG_COMMITTED. Usage of this flags enables pool to create its heaps
    on smaller alignment not suitable for MSAA textures.

    You should always use this flag unless you really need to create some MSAA textures in this pool as placed.
    */
    POOL_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED = 0x2,
    /** Every allocation made in this pool will be created as a committed resource - will have its own memory block.
    
    There is also an equivalent flag for the entire allocator: D3D12MA::ALLOCATOR_FLAG_ALWAYS_COMMITTED.
    */
    POOL_FLAG_ALWAYS_COMMITTED = 0x4,

    // Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    POOL_FLAG_ALGORITHM_MASK = POOL_FLAG_ALGORITHM_LINEAR
};

/// \brief Parameters of created D3D12MA::Pool object. To be used with D3D12MA::Allocator::CreatePool.
struct POOL_DESC
{
    /** \brief Flags for the heap.
    
    It is recommended to use #D3D12MA_RECOMMENDED_HEAP_FLAGS.
    */
    POOL_FLAGS Flags;
    /** \brief The parameters of memory heap where allocations of this pool should be placed.

    In the simplest case, just fill it with zeros and set `Type` to one of: `D3D12_HEAP_TYPE_DEFAULT`,
    `D3D12_HEAP_TYPE_UPLOAD`, `D3D12_HEAP_TYPE_READBACK`. Additional parameters can be used e.g. to utilize UMA.
    */
    D3D12_HEAP_PROPERTIES HeapProperties;
    /** \brief Heap flags to be used when allocating heaps of this pool.

    It should contain one of these values, depending on type of resources you are going to create in this heap:
    `D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS`,
    `D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES`,
    `D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES`.
    Except if ResourceHeapTier = 2, then it may be `D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES` = 0.

    It is recommended to also add #D3D12MA_RECOMMENDED_POOL_FLAGS.
    You can specify additional flags if needed.
    */
    D3D12_HEAP_FLAGS HeapFlags;
    /** \brief Size of a single heap (memory block) to be allocated as part of this pool, in bytes. Optional.

    Specify nonzero to set explicit, constant size of memory blocks used by this pool.
    Leave 0 to use default and let the library manage block sizes automatically.
    Then sizes of particular blocks may vary.
    */
    UINT64 BlockSize;
    /** \brief Minimum number of heaps (memory blocks) to be always allocated in this pool, even if they stay empty. Optional.

    Set to 0 to have no preallocated blocks and allow the pool be completely empty.
    */
    UINT MinBlockCount;
    /** \brief Maximum number of heaps (memory blocks) that can be allocated in this pool. Optional.

    Set to 0 to use default, which is `UINT64_MAX`, which means no limit.

    Set to same value as D3D12MA::POOL_DESC::MinBlockCount to have fixed amount of memory allocated
    throughout whole lifetime of this pool.
    */
    UINT MaxBlockCount;
    /** \brief Additional minimum alignment to be used for all allocations created from this pool. Can be 0.
    
    Leave 0 (default) not to impose any additional alignment. If not 0, it must be a power of two.
    */
    UINT64 MinAllocationAlignment;
    /** \brief Additional parameter allowing pool to create resources with passed protected session.
    
    If not null then all the heaps and committed resources will be created with this parameter.
    Valid only if ID3D12Device4 interface is present in current Windows SDK!
    */
    ID3D12ProtectedResourceSession* pProtectedSession;
    /** \brief Residency priority to be set for all allocations made in this pool. Optional.
    
    Set this parameter to one of the possible enum values e.g. `D3D12_RESIDENCY_PRIORITY_HIGH`
    to apply specific residency priority to all allocations made in this pool:
    `ID3D12Heap` memory blocks used to sub-allocate for placed resources, as well as
    committed resources or heaps created when D3D12MA::ALLOCATION_FLAG_COMMITTED is used.
    This can increase/decrease chance that the memory will be pushed out from VRAM
    to system RAM when the system runs out of memory, which is invisible to the developer
    using D3D12 API while it can degrade performance.

    Priority is set using function `ID3D12Device1::SetResidencyPriority`.
    It is performed only when `ID3D12Device1` interface is defined and successfully obtained.
    Otherwise, this parameter is ignored.

    This parameter is optional. If you set it to `D3D12_RESIDENCY_PRIORITY(0)`,
    residency priority will not be set for allocations made in this pool.

    There is no equivalent parameter for allocations made in default pools.
    If you want to set residency priority for such allocation, you need to do it manually:
    allocate with D3D12MA::ALLOCATION_FLAG_COMMITTED and call
    `ID3D12Device1::SetResidencyPriority`, passing `allocation->GetResource()`.
    */
    D3D12_RESIDENCY_PRIORITY ResidencyPriority;
};

/** \brief Custom memory pool

Represents a separate set of heaps (memory blocks) that can be used to create
D3D12MA::Allocation-s and resources in it. Usually there is no need to create custom
pools - creating resources in default pool is sufficient.

To create custom pool, fill D3D12MA::POOL_DESC and call D3D12MA::Allocator::CreatePool.
*/
class D3D12MA_API Pool : public IUnknownImpl
{
public:
    /** \brief Returns copy of parameters of the pool.

    These are the same parameters as passed to D3D12MA::Allocator::CreatePool.
    */
    POOL_DESC GetDesc() const;

    /** \brief Retrieves basic statistics of the custom pool that are fast to calculate.

    \param[out] pStats %Statistics of the current pool.
    */
    void GetStatistics(Statistics* pStats);

    /** \brief Retrieves detailed statistics of the custom pool that are slower to calculate.

    \param[out] pStats %Statistics of the current pool.
    */
    void CalculateStatistics(DetailedStatistics* pStats);

    /** \brief Associates a name with the pool. This name is for use in debug diagnostics and tools.

    Internal copy of the string is made, so the memory pointed by the argument can be
    changed of freed immediately after this call.

    `Name` can be NULL.
    */
    void SetName(LPCWSTR Name);

    /** \brief Returns the name associated with the pool object.

    Returned string points to an internal copy.

    If no name was associated with the allocation, returns NULL.
    */
    LPCWSTR GetName() const;

    /** \brief Begins defragmentation process of the current pool.

    \param pDesc Structure filled with parameters of defragmentation.
    \param[out] ppContext Context object that will manage defragmentation.
    \returns
    - `S_OK` if defragmentation can begin.
    - `E_NOINTERFACE` if defragmentation is not supported.

    For more information about defragmentation, see documentation chapter:
    [Defragmentation](@ref defragmentation).
    */
    HRESULT BeginDefragmentation(const DEFRAGMENTATION_DESC* pDesc, DefragmentationContext** ppContext);

protected:
    void ReleaseThis() override;

private:
    friend class Allocator;
    friend class AllocatorPimpl;
    template<typename T> friend void D3D12MA_DELETE(const ALLOCATION_CALLBACKS&, T*);

    PoolPimpl* m_Pimpl;

    Pool(Allocator* allocator, const POOL_DESC &desc);
    ~Pool();

    D3D12MA_CLASS_NO_COPY(Pool)
};


/// \brief Bit flags to be used with ALLOCATOR_DESC::Flags.
enum ALLOCATOR_FLAGS
{
    /// Zero
    ALLOCATOR_FLAG_NONE = 0,

    /**
    Allocator and all objects created from it will not be synchronized internally,
    so you must guarantee they are used from only one thread at a time or
    synchronized by you.

    Using this flag may increase performance because internal mutexes are not used.
    */
    ALLOCATOR_FLAG_SINGLETHREADED = 0x1,

    /** Every allocation will be created as a committed resource - will have its own memory block.
    
    Affects both default pools and custom pools.
    To be used for debugging purposes only.
    There is also an equivalent flag for custom pools: D3D12MA::POOL_FLAG_ALWAYS_COMMITTED.
    */
    ALLOCATOR_FLAG_ALWAYS_COMMITTED = 0x2,

    /**
    Heaps created for the default pools will be created with flag `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED`,
    allowing for their memory to be not zeroed by the system if possible,
    which can speed up allocation.

    Only affects default pools.
    To use the flag with @ref custom_pools, you need to add it manually:

    \code
    poolDesc.heapFlags |= D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
    \endcode

    Only avaiable if `ID3D12Device8` is present. Otherwise, the flag is ignored.
    */
    ALLOCATOR_FLAG_DEFAULT_POOLS_NOT_ZEROED = 0x4,

    /** Optimization, allocate MSAA textures as committed resources always.

    Specify this flag to create MSAA textures with implicit heaps, as if they were created
    with flag D3D12MA::ALLOCATION_FLAG_COMMITTED. Usage of this flags enables all default pools
    to create its heaps on smaller alignment not suitable for MSAA textures.

    You should always use this flag unless you really need to create some MSAA textures as placed.
    */
    ALLOCATOR_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED = 0x8,
    /** Disable optimization that prefers creating small buffers as committed to avoid 64 KB alignment.
    
    By default, the library prefers creating small buffers <= 32 KB as committed,
    because drivers tend to pack them better, while placed buffers require 64 KB alignment.
    This, however, may decrease performance, as creating committed resources involves allocation of implicit heaps,
    which may take longer than creating placed resources in existing heaps.
    Passing this flag will disable this committed preference globally for the allocator.
    It can also be disabled for a single allocation by using #ALLOCATION_FLAG_STRATEGY_MIN_TIME.

    If the tight resource alignment feature is used by the library (which happens automatically whenever supported,
    unless you use flag #ALLOCATOR_FLAG_DONT_USE_TIGHT_ALIGNMENT), then small buffers are not preferred as committed.
    Long story short, you don't need to specify any of these flags.
    The library chooses the most optimal method automatically.
    */
    ALLOCATOR_FLAG_DONT_PREFER_SMALL_BUFFERS_COMMITTED = 0x10,
    /** Disables the use of the tight alignment feature even when it is supported on the current system.
    By default, the feature is used whenever available.

    Support can be checked by D3D12MA::Allocator::IsTightAlignmentSupported() regardless of using this flag.
    */
    ALLOCATOR_FLAG_DONT_USE_TIGHT_ALIGNMENT = 0x20,
};

/// \brief Parameters of created Allocator object. To be used with CreateAllocator().
struct ALLOCATOR_DESC
{
    /** \brief Flags for the entire allocator.
    
    It is recommended to use #D3D12MA_RECOMMENDED_ALLOCATOR_FLAGS.
    */
    ALLOCATOR_FLAGS Flags;
    
    /** Direct3D device object that the allocator should be attached to.

    Allocator is doing `AddRef`/`Release` on this object.
    */
    ID3D12Device* pDevice;
    
    /** \brief Preferred size of a single `ID3D12Heap` block to be allocated.
    
    Set to 0 to use default, which is currently 64 MiB.
    */
    UINT64 PreferredBlockSize;
    
    /** \brief Custom CPU memory allocation callbacks. Optional.

    Optional, can be null. When specified, will be used for all CPU-side memory allocations.
    */
    const ALLOCATION_CALLBACKS* pAllocationCallbacks;

    /** DXGI Adapter object that you use for D3D12 and this allocator.

    Allocator is doing `AddRef`/`Release` on this object.
    */
    IDXGIAdapter* pAdapter;
};

/**
\brief Represents main object of this library initialized for particular `ID3D12Device`.

Fill structure D3D12MA::ALLOCATOR_DESC and call function CreateAllocator() to create it.
Call method `Release()` to destroy it.

It is recommended to create just one object of this type per `ID3D12Device` object,
right after Direct3D 12 is initialized and keep it alive until before Direct3D device is destroyed.
*/
class D3D12MA_API Allocator : public IUnknownImpl
{
public:
    /// Returns cached options retrieved from D3D12 device.
    const D3D12_FEATURE_DATA_D3D12_OPTIONS& GetD3D12Options() const;
    /** \brief Returns true if `D3D12_FEATURE_DATA_ARCHITECTURE1::UMA` was found to be true.
    
    For more information about how to use it, see articles in Microsoft Docs articles:

    - "UMA Optimizations: CPU Accessible Textures and Standard Swizzle"
    - "D3D12_FEATURE_DATA_ARCHITECTURE structure (d3d12.h)"
    - "ID3D12Device::GetCustomHeapProperties method (d3d12.h)"
    */
    BOOL IsUMA() const;
    /** \brief Returns true if `D3D12_FEATURE_DATA_ARCHITECTURE1::CacheCoherentUMA` was found to be true.

    For more information about how to use it, see articles in Microsoft Docs articles:

    - "UMA Optimizations: CPU Accessible Textures and Standard Swizzle"
    - "D3D12_FEATURE_DATA_ARCHITECTURE structure (d3d12.h)"
    - "ID3D12Device::GetCustomHeapProperties method (d3d12.h)"
    */
    BOOL IsCacheCoherentUMA() const;
    /** \brief Returns true if GPU Upload Heaps are supported on the current system.

    When true, you can use `D3D12_HEAP_TYPE_GPU_UPLOAD`.

    This flag is fetched from `D3D12_FEATURE_D3D12_OPTIONS16::GPUUploadHeapSupported`.
    */
    BOOL IsGPUUploadHeapSupported() const;
    /** \brief Returns true if resource tight alignment is supported on the current system.
    When supported, it is automatically used by the library, unless
    #ALLOCATOR_FLAG_DONT_USE_TIGHT_ALIGNMENT flag was specified on allocator creation.
    This flag is fetched from `D3D12_FEATURE_DATA_TIGHT_ALIGNMENT::SupportTier`.
    */
    BOOL IsTightAlignmentSupported() const;
    /** \brief Returns total amount of memory of specific segment group, in bytes.
    
    \param memorySegmentGroup use `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` or `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL`.

    This information is taken from `DXGI_ADAPTER_DESC`.
    It is not recommended to use this number.
    You should preferably call GetBudget() and limit memory usage to D3D12MA::Budget::BudgetBytes instead.

    - When IsUMA() `== FALSE` (discrete graphics card):
      - `GetMemoryCapacity(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)` returns the size of the video memory.
      - `GetMemoryCapacity(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)` returns the size of the system memory available for D3D12 resources.
    - When IsUMA() `== TRUE` (integrated graphics chip):
      - `GetMemoryCapacity(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)` returns the size of the shared memory available for all D3D12 resources.
        All memory is considered "local".
      - `GetMemoryCapacity(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)` is not applicable and returns 0.
    */
    UINT64 GetMemoryCapacity(UINT memorySegmentGroup) const;

    /** \brief Allocates memory and creates a D3D12 resource (buffer or texture). This is the main allocation function.

    The function is similar to `ID3D12Device::CreateCommittedResource`, but it may
    really call `ID3D12Device::CreatePlacedResource` to assign part of a larger,
    existing memory heap to the new resource, which is the main purpose of this
    whole library.

    If `ppvResource` is null, you receive only `ppAllocation` object from this function.
    It holds pointer to `ID3D12Resource` that can be queried using function D3D12MA::Allocation::GetResource().
    Reference count of the resource object is 1.
    It is automatically destroyed when you destroy the allocation object.

    If `ppvResource` is not null, you receive pointer to the resource next to allocation object.
    Reference count of the resource object is then increased by calling `QueryInterface`, so you need to manually `Release` it
    along with the allocation.

    \param pAllocDesc   Parameters of the allocation.
    \param pResourceDesc   Description of created resource.
    \param InitialResourceState   Initial resource state.
    \param pOptimizedClearValue   Optional. Either null or optimized clear value.
    \param[out] ppAllocation   Filled with pointer to new allocation object created.
    \param riidResource   IID of a resource to be returned via `ppvResource`.
    \param[out] ppvResource   Optional. If not null, filled with pointer to new resouce created.

    \note This function creates a new resource. Sub-allocation of parts of one large buffer,
    although recommended as a good practice, is out of scope of this library and could be implemented
    by the user as a higher-level logic on top of it, e.g. using the \ref virtual_allocator feature.
    */
    HRESULT CreateResource(
        const ALLOCATION_DESC* pAllocDesc,
        const D3D12_RESOURCE_DESC* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation,
        REFIID riidResource,
        void** ppvResource);

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    /** \brief Similar to Allocator::CreateResource, but supports new structure `D3D12_RESOURCE_DESC1`.
    
    It internally uses `ID3D12Device8::CreateCommittedResource2` or `ID3D12Device8::CreatePlacedResource1`.

    To work correctly, `ID3D12Device8` interface must be available in the current system. Otherwise, `E_NOINTERFACE` is returned.
    */
    HRESULT CreateResource2(
        const ALLOCATION_DESC* pAllocDesc,
        const D3D12_RESOURCE_DESC1* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        Allocation** ppAllocation,
        REFIID riidResource,
        void** ppvResource);
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

#ifdef __ID3D12Device10_INTERFACE_DEFINED__
    /** \brief Similar to Allocator::CreateResource2, but there are initial layout instead of state and 
    castable formats list

    It internally uses `ID3D12Device10::CreateCommittedResource3` or `ID3D12Device10::CreatePlacedResource2`.

    To work correctly, `ID3D12Device10` interface must be available in the current system. Otherwise, `E_NOINTERFACE` is returned.
    If you use `pCastableFormats`, `ID3D12Device12` must albo be available.
    */
    HRESULT CreateResource3(const ALLOCATION_DESC* pAllocDesc,
        const D3D12_RESOURCE_DESC1* pResourceDesc,
        D3D12_BARRIER_LAYOUT InitialLayout,
        const D3D12_CLEAR_VALUE* pOptimizedClearValue,
        UINT32 NumCastableFormats,
        const DXGI_FORMAT* pCastableFormats,
        Allocation** ppAllocation,
        REFIID riidResource,
        void** ppvResource);
#endif  // #ifdef __ID3D12Device10_INTERFACE_DEFINED__

    /** \brief Allocates memory without creating any resource placed in it.

    This function is similar to `ID3D12Device::CreateHeap`, but it may really assign
    part of a larger, existing heap to the allocation.

    `pAllocDesc->heapFlags` should contain one of these values, depending on type of resources you are going to create in this memory:
    `D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS`,
    `D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES`,
    `D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES`.
    Except if you validate that ResourceHeapTier = 2 - then `heapFlags`
    may be `D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES` = 0.
    Additional flags in `heapFlags` are allowed as well.

    `pAllocInfo->SizeInBytes` must be multiply of 64KB.
    `pAllocInfo->Alignment` must be one of the legal values as described in documentation of `D3D12_HEAP_DESC`.

    If you use D3D12MA::ALLOCATION_FLAG_COMMITTED you will get a separate memory block -
    a heap that always has offset 0.
    */
    HRESULT AllocateMemory(
        const ALLOCATION_DESC* pAllocDesc,
        const D3D12_RESOURCE_ALLOCATION_INFO* pAllocInfo,
        Allocation** ppAllocation);

    /** \brief Creates a new resource in place of an existing allocation. This is useful for memory aliasing.

    \param pAllocation Existing allocation indicating the memory where the new resource should be created.
        It can be created using D3D12MA::Allocator::CreateResource and already have a resource bound to it,
        or can be a raw memory allocated with D3D12MA::Allocator::AllocateMemory.
        It must not be created as committed so that `ID3D12Heap` is available and not implicit.
    \param AllocationLocalOffset Additional offset in bytes to be applied when allocating the resource.
        Local from the start of `pAllocation`, not the beginning of the whole `ID3D12Heap`!
        If the new resource should start from the beginning of the `pAllocation` it should be 0.
    \param pResourceDesc Description of the new resource to be created.
    \param InitialResourceState
    \param pOptimizedClearValue
    \param riidResource
    \param[out] ppvResource Returns pointer to the new resource.
        The resource is not bound with `pAllocation`.
        This pointer must not be null - you must get the resource pointer and `Release` it when no longer needed.

    Memory requirements of the new resource are checked for validation.
    If its size exceeds the end of `pAllocation` or required alignment is not fulfilled
    considering `pAllocation->GetOffset() + AllocationLocalOffset`, the function
    returns `E_INVALIDARG`.
    */
    HRESULT CreateAliasingResource(
        Allocation* pAllocation,
        UINT64 AllocationLocalOffset,
        const D3D12_RESOURCE_DESC* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE *pOptimizedClearValue,
        REFIID riidResource,
        void** ppvResource);

#ifdef __ID3D12Device8_INTERFACE_DEFINED__
    /** \brief Similar to Allocator::CreateAliasingResource, but supports new structure `D3D12_RESOURCE_DESC1`.
    
    It internally uses `ID3D12Device8::CreatePlacedResource1`.

    To work correctly, `ID3D12Device8` interface must be available in the current system. Otherwise, `E_NOINTERFACE` is returned.
    */
    HRESULT CreateAliasingResource1(Allocation* pAllocation,
        UINT64 AllocationLocalOffset,
        const D3D12_RESOURCE_DESC1* pResourceDesc,
        D3D12_RESOURCE_STATES InitialResourceState,
        const D3D12_CLEAR_VALUE* pOptimizedClearValue,
        REFIID riidResource,
        void** ppvResource);
#endif // #ifdef __ID3D12Device8_INTERFACE_DEFINED__

#ifdef __ID3D12Device10_INTERFACE_DEFINED__
    /** \brief Similar to Allocator::CreateAliasingResource1, but there are initial layout instead of state and 
    castable formats list.

    It internally uses `ID3D12Device10::CreatePlacedResource2`.

    To work correctly, `ID3D12Device10` interface must be available in the current system. Otherwise, `E_NOINTERFACE` is returned.
    If you use `pCastableFormats`, `ID3D12Device12` must albo be available.
    */
    HRESULT CreateAliasingResource2(Allocation* pAllocation,
        UINT64 AllocationLocalOffset,
        const D3D12_RESOURCE_DESC1* pResourceDesc,
        D3D12_BARRIER_LAYOUT InitialLayout,
        const D3D12_CLEAR_VALUE* pOptimizedClearValue,
        UINT32 NumCastableFormats,
        const DXGI_FORMAT* pCastableFormats,
        REFIID riidResource,
        void** ppvResource);
#endif  // #ifdef __ID3D12Device10_INTERFACE_DEFINED__

    /** \brief Creates custom pool.
    */
    HRESULT CreatePool(
        const POOL_DESC* pPoolDesc,
        Pool** ppPool);

    /** \brief Sets the index of the current frame.

    This function is used to set the frame index in the allocator when a new game frame begins.
    */
    void SetCurrentFrameIndex(UINT frameIndex);

    /** \brief Retrieves information about current memory usage and budget.

    \param[out] pLocalBudget Optional, can be null.
    \param[out] pNonLocalBudget Optional, can be null.

    - When IsUMA() `== FALSE` (discrete graphics card):
      - `pLocalBudget` returns the budget of the video memory.
      - `pNonLocalBudget` returns the budget of the system memory available for D3D12 resources.
    - When IsUMA() `== TRUE` (integrated graphics chip):
      - `pLocalBudget` returns the budget of the shared memory available for all D3D12 resources.
         All memory is considered "local".
      - `pNonLocalBudget` is not applicable and returns zeros.

    This function is called "get" not "calculate" because it is very fast, suitable to be called
    every frame or every allocation. For more detailed statistics use CalculateStatistics().

    Note that when using allocator from multiple threads, returned information may immediately
    become outdated.
    */
    void GetBudget(Budget* pLocalBudget, Budget* pNonLocalBudget);

    /** \brief Retrieves statistics from current state of the allocator.

    This function is called "calculate" not "get" because it has to traverse all
    internal data structures, so it may be quite slow. Use it for debugging purposes.
    For faster but more brief statistics suitable to be called every frame or every allocation,
    use GetBudget().

    Note that when using allocator from multiple threads, returned information may immediately
    become outdated.
    */
    void CalculateStatistics(TotalStatistics* pStats);

    /** \brief Builds and returns statistics as a string in JSON format.
    * 
    @param[out] ppStatsString Must be freed using Allocator::FreeStatsString.
    @param DetailedMap `TRUE` to include full list of allocations (can make the string quite long), `FALSE` to only return statistics.
    */
    void BuildStatsString(WCHAR** ppStatsString, BOOL DetailedMap) const;

    /// Frees memory of a string returned from Allocator::BuildStatsString.
    void FreeStatsString(WCHAR* pStatsString) const;

    /** \brief Begins defragmentation process of the default pools.

    \param pDesc Structure filled with parameters of defragmentation.
    \param[out] ppContext Context object that will manage defragmentation.

    For more information about defragmentation, see documentation chapter:
    [Defragmentation](@ref defragmentation).
    */
    void BeginDefragmentation(const DEFRAGMENTATION_DESC* pDesc, DefragmentationContext** ppContext);

protected:
    void ReleaseThis() override;

private:
    friend D3D12MA_API HRESULT CreateAllocator(const ALLOCATOR_DESC*, Allocator**);
    template<typename T> friend void D3D12MA_DELETE(const ALLOCATION_CALLBACKS&, T*);
    friend class DefragmentationContext;
    friend class Pool;

    Allocator(const ALLOCATION_CALLBACKS& allocationCallbacks, const ALLOCATOR_DESC& desc);
    ~Allocator();
    
    AllocatorPimpl* m_Pimpl;
    
    D3D12MA_CLASS_NO_COPY(Allocator)
};


/// \brief Bit flags to be used with VIRTUAL_BLOCK_DESC::Flags.
enum VIRTUAL_BLOCK_FLAGS
{
    /// Zero
    VIRTUAL_BLOCK_FLAG_NONE = 0,

    /** \brief Enables alternative, linear allocation algorithm in this virtual block.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    VIRTUAL_BLOCK_FLAG_ALGORITHM_LINEAR = POOL_FLAG_ALGORITHM_LINEAR,

    // Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    VIRTUAL_BLOCK_FLAG_ALGORITHM_MASK = POOL_FLAG_ALGORITHM_MASK
};

/// Parameters of created D3D12MA::VirtualBlock object to be passed to CreateVirtualBlock().
struct VIRTUAL_BLOCK_DESC
{
    /// Flags.
    VIRTUAL_BLOCK_FLAGS Flags;
    /** \brief Total size of the block.

    Sizes can be expressed in bytes or any units you want as long as you are consistent in using them.
    For example, if you allocate from some array of structures, 1 can mean single instance of entire structure.
    */
    UINT64 Size;
    /** \brief Custom CPU memory allocation callbacks. Optional.

    Optional, can be null. When specified, will be used for all CPU-side memory allocations.
    */
    const ALLOCATION_CALLBACKS* pAllocationCallbacks;
};

/// \brief Bit flags to be used with VIRTUAL_ALLOCATION_DESC::Flags.
enum VIRTUAL_ALLOCATION_FLAGS
{
    /// Zero
    VIRTUAL_ALLOCATION_FLAG_NONE = 0,

    /** \brief Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for virtual blocks created with #VIRTUAL_BLOCK_FLAG_ALGORITHM_LINEAR flag.
    */
    VIRTUAL_ALLOCATION_FLAG_UPPER_ADDRESS = ALLOCATION_FLAG_UPPER_ADDRESS,

    /// %Allocation strategy that tries to minimize memory usage.
    VIRTUAL_ALLOCATION_FLAG_STRATEGY_MIN_MEMORY = ALLOCATION_FLAG_STRATEGY_MIN_MEMORY,
    /// %Allocation strategy that tries to minimize allocation time.
    VIRTUAL_ALLOCATION_FLAG_STRATEGY_MIN_TIME = ALLOCATION_FLAG_STRATEGY_MIN_TIME,
    /** %Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    */
    VIRTUAL_ALLOCATION_FLAG_STRATEGY_MIN_OFFSET = ALLOCATION_FLAG_STRATEGY_MIN_OFFSET,
    /** \brief A bit mask to extract only `STRATEGY` bits from entire set of flags.

    These strategy flags are binary compatible with equivalent flags in #ALLOCATION_FLAGS.
    */
    VIRTUAL_ALLOCATION_FLAG_STRATEGY_MASK = ALLOCATION_FLAG_STRATEGY_MASK,
};

/// Parameters of created virtual allocation to be passed to VirtualBlock::Allocate().
struct VIRTUAL_ALLOCATION_DESC
{
    /// Flags for the virtual allocation.
    VIRTUAL_ALLOCATION_FLAGS Flags;
    /** \brief Size of the allocation.
    
    Cannot be zero.
    */
    UINT64 Size;
    /** \brief Required alignment of the allocation.
    
    Must be power of two. Special value 0 has the same meaning as 1 - means no special alignment is required, so allocation can start at any offset.
    */
    UINT64 Alignment;
    /** \brief Custom pointer to be associated with the allocation.

    It can be fetched or changed later.
    */
    void* pPrivateData;
};

/// Parameters of an existing virtual allocation, returned by VirtualBlock::GetAllocationInfo().
struct VIRTUAL_ALLOCATION_INFO
{
    /// \brief Offset of the allocation.
    UINT64 Offset;
    /** \brief Size of the allocation.

    Same value as passed in VIRTUAL_ALLOCATION_DESC::Size.
    */
    UINT64 Size;
    /** \brief Custom pointer associated with the allocation.

    Same value as passed in VIRTUAL_ALLOCATION_DESC::pPrivateData or VirtualBlock::SetAllocationPrivateData().
    */
    void* pPrivateData;
};

/** \brief Represents pure allocation algorithm and a data structure with allocations in some memory block, without actually allocating any GPU memory.

This class allows to use the core algorithm of the library custom allocations e.g. CPU memory or
sub-allocation regions inside a single GPU buffer.

To create this object, fill in D3D12MA::VIRTUAL_BLOCK_DESC and call CreateVirtualBlock().
To destroy it, call its method `VirtualBlock::Release()`.
You need to free all the allocations within this block or call Clear() before destroying it.

This object is not thread-safe - should not be used from multiple threads simultaneously, must be synchronized externally.
*/
class D3D12MA_API VirtualBlock : public IUnknownImpl
{
public:
    /** \brief Returns true if the block is empty - contains 0 allocations.
    */
    BOOL IsEmpty() const;
    /** \brief Returns information about an allocation - its offset, size and custom pointer.
    */
    void GetAllocationInfo(VirtualAllocation allocation, VIRTUAL_ALLOCATION_INFO* pInfo) const;

    /** \brief Creates new allocation.
    \param pDesc
    \param[out] pAllocation Unique indentifier of the new allocation within single block.
    \param[out] pOffset Returned offset of the new allocation. Optional, can be null.
    \return `S_OK` if allocation succeeded, `E_OUTOFMEMORY` if it failed.

    If the allocation failed, `pAllocation->AllocHandle` is set to 0 and `pOffset`, if not null, is set to `UINT64_MAX`.
    */
    HRESULT Allocate(const VIRTUAL_ALLOCATION_DESC* pDesc, VirtualAllocation* pAllocation, UINT64* pOffset);
    /** \brief Frees the allocation.
    
    Calling this function with `allocation.AllocHandle == 0` is correct and does nothing.
    */
    void FreeAllocation(VirtualAllocation allocation);
    /** \brief Frees all the allocations.
    */
    void Clear();
    /** \brief Changes custom pointer for an allocation to a new value.
    */
    void SetAllocationPrivateData(VirtualAllocation allocation, void* pPrivateData);
    /** \brief Retrieves basic statistics of the virtual block that are fast to calculate.

    \param[out] pStats %Statistics of the virtual block.
    */
    void GetStatistics(Statistics* pStats) const;
    /** \brief Retrieves detailed statistics of the virtual block that are slower to calculate.

    \param[out] pStats %Statistics of the virtual block.
    */
    void CalculateStatistics(DetailedStatistics* pStats) const;

    /** \brief Builds and returns statistics as a string in JSON format, including the list of allocations with their parameters.
    @param[out] ppStatsString Must be freed using VirtualBlock::FreeStatsString.
    */
    void BuildStatsString(WCHAR** ppStatsString) const;

    /** \brief Frees memory of a string returned from VirtualBlock::BuildStatsString.
    */
    void FreeStatsString(WCHAR* pStatsString) const;
   
protected:
    void ReleaseThis() override;

private:
    friend D3D12MA_API HRESULT CreateVirtualBlock(const VIRTUAL_BLOCK_DESC*, VirtualBlock**);
    template<typename T> friend void D3D12MA_DELETE(const ALLOCATION_CALLBACKS&, T*);

    VirtualBlockPimpl* m_Pimpl;

    VirtualBlock(const ALLOCATION_CALLBACKS& allocationCallbacks, const VIRTUAL_BLOCK_DESC& desc);
    ~VirtualBlock();

    D3D12MA_CLASS_NO_COPY(VirtualBlock)
};


/** \brief Creates new main D3D12MA::Allocator object and returns it through `ppAllocator`.

You normally only need to call it once and keep a single Allocator object for your `ID3D12Device`.
*/
D3D12MA_API HRESULT CreateAllocator(const ALLOCATOR_DESC* pDesc, Allocator** ppAllocator);

/** \brief Creates new D3D12MA::VirtualBlock object and returns it through `ppVirtualBlock`.

Note you don't need to create D3D12MA::Allocator to use virtual blocks.
*/
D3D12MA_API HRESULT CreateVirtualBlock(const VIRTUAL_BLOCK_DESC* pDesc, VirtualBlock** ppVirtualBlock);

#ifndef D3D12MA_NO_HELPERS

/** \brief Helper structure that helps with complete and conscise initialization of the D3D12MA::ALLOCATION_DESC structure.
 */
struct CALLOCATION_DESC : public ALLOCATION_DESC
{
    /// Default constructor. Leaves the structure uninitialized.
    CALLOCATION_DESC() = default;
    /// Constructor initializing from the base D3D12MA::ALLOCATION_DESC structure.
    explicit CALLOCATION_DESC(const ALLOCATION_DESC& o) noexcept
        : ALLOCATION_DESC(o)
    {
    }
    /// Constructor initializing description of an allocation to be created in a specific custom pool.
    explicit CALLOCATION_DESC(Pool* customPool,
        ALLOCATION_FLAGS flags = ALLOCATION_FLAG_NONE,
        void* privateData = NULL) noexcept
    {
        Flags = flags;
        HeapType = (D3D12_HEAP_TYPE)0;
        ExtraHeapFlags = D3D12_HEAP_FLAG_NONE;
        CustomPool = customPool;
        pPrivateData = privateData;
    }
    /// Constructor initializing description of an allocation to be created in a default pool of a specific `D3D12_HEAP_TYPE`.
    explicit CALLOCATION_DESC(D3D12_HEAP_TYPE heapType,
        ALLOCATION_FLAGS flags = ALLOCATION_FLAG_NONE,
        void* privateData = NULL,
        D3D12_HEAP_FLAGS extraHeapFlags = D3D12MA_RECOMMENDED_HEAP_FLAGS) noexcept
    {
        Flags = flags;
        HeapType = heapType;
        ExtraHeapFlags = extraHeapFlags;
        CustomPool = NULL;
        pPrivateData = privateData;
    }
};

/** \brief Helper structure that helps with complete and conscise initialization of the D3D12MA::POOL_DESC structure.
 */
struct CPOOL_DESC : public POOL_DESC
{
    /// Default constructor. Leaves the structure uninitialized.
    CPOOL_DESC() = default;
    /// Constructor initializing from the base D3D12MA::POOL_DESC structure.
    explicit CPOOL_DESC(const POOL_DESC& o) noexcept
        : POOL_DESC(o)
    {
    }
    /// Constructor initializing description of a custom pool created in one of the standard `D3D12_HEAP_TYPE`.
    explicit CPOOL_DESC(D3D12_HEAP_TYPE heapType,
        D3D12_HEAP_FLAGS heapFlags,
        POOL_FLAGS flags = D3D12MA_RECOMMENDED_POOL_FLAGS,
        UINT64 blockSize = 0,
        UINT minBlockCount = 0,
        UINT maxBlockCount = UINT_MAX,
        D3D12_RESIDENCY_PRIORITY residencyPriority = D3D12_RESIDENCY_PRIORITY_NORMAL) noexcept
    {
        Flags = flags;
        HeapProperties = {};
        HeapProperties.Type = heapType;
        HeapFlags = heapFlags;
        BlockSize = blockSize;
        MinBlockCount = minBlockCount;
        MaxBlockCount = maxBlockCount;
        MinAllocationAlignment = 0;
        pProtectedSession = NULL;
        ResidencyPriority = residencyPriority;
    }
    /// Constructor initializing description of a custom pool created with custom `D3D12_HEAP_PROPERTIES`.
    explicit CPOOL_DESC(const D3D12_HEAP_PROPERTIES heapProperties,
        D3D12_HEAP_FLAGS heapFlags,
        POOL_FLAGS flags = D3D12MA_RECOMMENDED_POOL_FLAGS,
        UINT64 blockSize = 0,
        UINT minBlockCount = 0,
        UINT maxBlockCount = UINT_MAX,
        D3D12_RESIDENCY_PRIORITY residencyPriority = D3D12_RESIDENCY_PRIORITY_NORMAL) noexcept
    {
        Flags = flags;
        HeapProperties = heapProperties;
        HeapFlags = heapFlags;
        BlockSize = blockSize;
        MinBlockCount = minBlockCount;
        MaxBlockCount = maxBlockCount;
        MinAllocationAlignment = 0;
        pProtectedSession = NULL;
        ResidencyPriority = residencyPriority;
    }
};

/** \brief Helper structure that helps with complete and conscise initialization of the D3D12MA::VIRTUAL_BLOCK_DESC structure.
 */
struct CVIRTUAL_BLOCK_DESC : public VIRTUAL_BLOCK_DESC
{
    /// Default constructor. Leaves the structure uninitialized.
    CVIRTUAL_BLOCK_DESC() = default;
    /// Constructor initializing from the base D3D12MA::VIRTUAL_BLOCK_DESC structure.
    explicit CVIRTUAL_BLOCK_DESC(const VIRTUAL_BLOCK_DESC& o) noexcept
        : VIRTUAL_BLOCK_DESC(o)
    {
    }
    /// Constructor initializing description of a virtual block with given parameters.
    explicit CVIRTUAL_BLOCK_DESC(UINT64 size,
        VIRTUAL_BLOCK_FLAGS flags = VIRTUAL_BLOCK_FLAG_NONE,
        const ALLOCATION_CALLBACKS* allocationCallbacks = NULL) noexcept
    {
        Flags = flags;
        Size = size;
        pAllocationCallbacks = allocationCallbacks;
    }
};

/** \brief Helper structure that helps with complete and conscise initialization of the D3D12MA::VIRTUAL_ALLOCATION_DESC structure.
 */
struct CVIRTUAL_ALLOCATION_DESC : public VIRTUAL_ALLOCATION_DESC
{
    /// Default constructor. Leaves the structure uninitialized.
    CVIRTUAL_ALLOCATION_DESC() = default;
    /// Constructor initializing from the base D3D12MA::VIRTUAL_ALLOCATION_DESC structure.
    explicit CVIRTUAL_ALLOCATION_DESC(const VIRTUAL_ALLOCATION_DESC& o) noexcept
        : VIRTUAL_ALLOCATION_DESC(o)
    {
    }
    /// Constructor initializing description of a virtual allocation with given parameters.
    explicit CVIRTUAL_ALLOCATION_DESC(UINT64 size, UINT64 alignment,
        VIRTUAL_ALLOCATION_FLAGS flags = VIRTUAL_ALLOCATION_FLAG_NONE,
        void* privateData = NULL) noexcept
    {
        Flags = flags;
        Size = size;
        Alignment = alignment;
        pPrivateData = privateData;
    }
};

#endif // #ifndef D3D12MA_NO_HELPERS

} // namespace D3D12MA

/// \cond INTERNAL
DEFINE_ENUM_FLAG_OPERATORS(D3D12MA::ALLOCATION_FLAGS);
DEFINE_ENUM_FLAG_OPERATORS(D3D12MA::DEFRAGMENTATION_FLAGS);
DEFINE_ENUM_FLAG_OPERATORS(D3D12MA::ALLOCATOR_FLAGS);
DEFINE_ENUM_FLAG_OPERATORS(D3D12MA::POOL_FLAGS);
DEFINE_ENUM_FLAG_OPERATORS(D3D12MA::VIRTUAL_BLOCK_FLAGS);
DEFINE_ENUM_FLAG_OPERATORS(D3D12MA::VIRTUAL_ALLOCATION_FLAGS);
/// \endcond

/**
\page faq Frequently asked questions

<b>What is %D3D12MA?</b>

D3D12 Memory Allocator (%D3D12MA) is a software library for developers who use the DirectX(R) 12 graphics API in their code.
It is written in C++.

<b>What is the license of %D3D12MA?</b>

%D3D12MA is licensed under MIT, which means it is open source and free software.

<b>What is the purpose of %D3D12MA?</b>

%D3D12MA helps with handling one aspect of DX12 usage, which is GPU memory management -
allocation of `ID3D12Heap` objects and creation of `ID3D12Resource` objects - buffers and textures.

<b>Do I need to use %D3D12MA?</b>

You don't need to, but it may be beneficial in many cases.
DX12 is a complex and low-level API, so libraries like this that abstract certain aspects of the API
and bring them to a higher level are useful.
When developing any non-trivial graphics application, you may benefit from using a memory allocator.
Using %D3D12MA can save time compared to implementing your own.

In DX12 you can create each resource separately with its own implicit memory heap by calling `CreateCommittedResource`,
but this may not be the optimal solution.
For more information, see [Committed versus placed resources](@ref optimal_allocation_committed_vs_placed).

<b>When should I not use %D3D12MA?</b>

While %D3D12MA is useful for many applications that use the DX12 API, there are cases
when it may be a better choice not to use it.
For example, if the application is very simple, e.g. serving as a sample or a learning exercise
to help you understand or teach others the basics of DX12,
and it creates only a small number of buffers and textures, then including %D3D12MA may be an overkill.
Developing your own memory allocator may also be a good learning exercise.

<b>What are the benefits of using %D3D12MA?</b>

-# %D3D12MA allocates large blocks of `ID3D12Heap` memory and sub-allocates parts of them to create your placed resources.
   Allocating a new block of GPU memory may be a time-consuming operation.
   Sub-allocating parts of a memory block requires implementing an allocation algorithm,
   which is a non-trivial task.
   %D3D12MA does that, using an advanced and efficient algorithm that works well in various use cases.
-# %D3D12MA offers a simple API that allows creating placed buffers and textures within one function call
   like D3D12MA::Allocator::CreateResource.

The library is doing much more under the hood.
For example, it keeps buffers separate from textures when needed, respecting `D3D12_RESOURCE_HEAP_TIER`.
It also makes use of the "small texture alignment" automatically, so you don't need to think about it.

<b>Which version should I pick?</b>

You can just pick [the latest version from the "master" branch](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator).
It is kept in a good shape most of the time, compiling and working correctly,
with no compatibility-breaking changes and no unfinished code.

If you want an even more stable version, you can pick
[the latest official release](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator/releases).
Current code from the master branch is occasionally tagged as a release,
with [CHANGELOG](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator/blob/master/CHANGELOG.md)
carefully curated to enumerate all important changes since the previous version.

The library uses [Semantic Versioning](https://semver.org/),
which means versions that only differ in the patch number are forward and backward compatible
(e.g., only fixing some bugs), while versions that differ in the minor number are backward compatible
(e.g., only adding new functions to the API, but not removing or changing existing ones).

<b>How to integrate it with my code?</b>

%D3D12MA is an small library fully implemented in a single pair of CPP + H files.

You can pull the entire GitHub repository, e.g. using Git submodules.
The repository contains ancillary files like the Cmake script, Doxygen config file,
sample application, test suite, and others.
You can compile it as a library and link with your project.

However, a simpler way is taking only files "include\D3D12MemAlloc.h", "src\D3D12MemAlloc.cpp"
and including them in your project.
These files contain all you need: a copyright notice,
declarations of the public library interface (API), its internal implementation,
and even the documentation in form of Doxygen-style comments.

<b>I am not a fan of modern C++. Can I still use it?</b>

Very likely yes.
We acknowledge that many C++ developers, especially in the games industry,
do not appreciate all the latest features that the language has to offer.

- %D3D12MA doesn't throw or catch any C++ exceptions.
  It reports errors by returning a `HRESULT` value instead, just like DX12.
  If you don't use exceptions in your project, your code is not exception-safe,
  or even if you disable exception handling in the compiler options, you can still use %D3D12MA.
- %D3D12MA doesn't use C++ run-time type information like `typeid` or `dynamic_cast`,
  so if you disable RTTI in the compiler options, you can still use the library.
- %D3D12MA uses only a limited subset of standard C and C++ library.
  It doesn't use STL containers like `std::vector`, `map`, or `string`,
  either in the public interface nor in the internal implementation.
  It implements its own containers instead.
- If you don't use the default heap memory allocator through `malloc/free` or `new/delete`
  but implement your own allocator instead, you can pass it to %D3D12MA as
  D3D12MA::ALLOCATOR_DESC::pAllocationCallbacks
  and the library will use your functions for every dynamic heap allocation made internally.

<b>Is it available for other programming languages?</b>

%D3D12MA is a C++ library in similar style as DX12.
Bindings to other programming languages are out of scope of this project,
but they are welcome as external projects.
Some of them are listed in [README.md, "See also" section](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator/?tab=readme-ov-file#see-also),
including binding to C.
Before using any of them, please check if they are still maintained and updated to use a recent version of %D3D12MA.

<b>What platforms does it support?</b>

%D3D12MA relies only on DX12 and some parts of the standard C and C++ library,
so it could support any platform where a C++ compiler and DX12 are available.
However, it is developed and tested only on Microsoft(R) Windows(R).

<b>Does it only work on AMD GPUs?</b>

No! While %D3D12MA is published by AMD, it works on any GPU that supports DX12,
whether a discrete PC graphics card or a processor integrated graphics.
It doesn't give AMD GPUs any advantage over any other GPUs.

<b>What DirectX 12 versions are supported?</b>

%D3D12MA is updated to support latest versions of DirectX 12, as available through recent retail versions of the
[DirectX 12 Agility SDK](https://devblogs.microsoft.com/directx/directx12agility/).
Support for new features added in the preview version of the Agility SDK is developed on separate branches until they are included in the retail version.

The library also supports older versions down to the base DX12 shipping with Windows SDK.
Features added by later versions of the Agility SDK are automatically enabled conditionally using
`#ifdef` preprocessor macros depending on the version of the SDK that you compile your project with.

<b>Does it support other graphics APIs, like Vulkan(R)?</b>

No, but we offer an equivalent library for Vulkan:
[Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator).
It uses the same core allocation algorithm.
It also shares many features with %D3D12MA, like the support for custom pools and virtual allocator.
However, it is not identical in terms of the features supported.
Its API also looks different, because while the interface of %D3D12MA is similar in style to DX12,
the interface of VMA is similar to Vulkan.

<b>Is the library lightweight?</b>

Yes.
%D3D12MA is implemented with high-performance and real-time applications like video games in mind.
The CPU performance overhead of using this library is low.
It uses a high-quality allocation algorithm called Two-Level Segregated Fit (TLSF),
which in most cases can find a free place for a new allocation in few steps.
The library also doesn't perform too many CPU heap allocations.
In many cases, the allocation happens with 0 new CPU heap allocations performed by the library.
Even the creation of a D3D12MA::Allocation object doesn't typically feature an CPU allocation,
because these objects are returned out of a dedicated memory pool.

That said, %D3D12MA needs some extra memory and extra time
to maintain the metadata about the occupied and free regions of the memory blocks,
and the algorithms and data structures used must be generic enough to work well in most cases.

<b>Does it have a documentation?</b>

Yes! %D3D12MA comes with full documentation of all elements of the API (classes, structures, enums),
as well as many generic chapters that provide an introduction,
describe core concepts of the library, good practices, etc.
The entire documentation is written in form of code comments inside "D3D12MemAlloc.h", in Doxygen format.
You can access it in multiple ways:

- Browsable online: https://gpuopen-librariesandsdks.github.io/D3D12MemoryAllocator/html/
- Local HTML pages available after you clone the repository and open file "docs\html\index.html".
- You can rebuild the documentation in HTML or some other format from the source code using Doxygen.
  Configuration file "Doxyfile" is part of the repository.
- Finally, you can just read the comments preceding declarations of any public classes and functions of the library.

<b>Is it a mature project?</b>

Yes! The library is in development since May 2019, has over 300 commits, and multiple contributors.
It is used by many software projects, including some large and popular ones like Qt or Godot Engine,
as well as some AAA games.

<b>How can I contribute to the project?</b>

If you have an idea for improvement or a feature request,
you can go to [the library repository](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator)
and create an Issue ticket, describing your idea.
You can also implement it yourself by forking the repository, making changes to the code,
and creating a Pull request.

If you want to ask a question, you can also create a ticket the same way.
Before doing this, please make sure you read the relevant part of the DX12 documentation and %D3D12MA documentation,
where you may find the answers to your question.

If you want to report a suspected bug, you can also create a ticket the same way.
Before doing this, please put some effort into the investigation of whether the bug is really
in the library and not in your code or in the DX12 implementation (the GPU driver) on your platform:

- Enable D3D Debug Layer and make sure it is free from any errors.
- Make sure `D3D12MA_ASSERT` is defined to an implementation that can report a failure and not ignore it.
- Try making your allocation using pure DX12 functions like `CreateCommittedResource()` rather than %D3D12MA and see if the bug persists.

<b>I found some compilation warnings. How can we fix them?</b>

Seeing compiler warnings may be annoying to some developers,
but it is a design decision to not fix all of them.
Due to the nature of the C++ language, certain preprocessor macros can make some variables unused,
function parameters unreferenced, or conditional expressions constant in some configurations.
The code of this library should not be bigger or more complicated just to silence these warnings.
It is recommended to disable such warnings instead.
For more information, see [Features not supported](@ref general_considerations_features_not_supported).

However, if you observe a warning that is really dangerous, e.g.,
about an implicit conversion from a larger to a smaller integer type, please report it and it will be fixed ASAP.


\page quick_start Quick start

\section quick_start_project_setup Project setup and initialization

This is a small, standalone C++ library. It consists of 2 files:
"D3D12MemAlloc.h" header file with public interface and "D3D12MemAlloc.cpp" with
internal implementation. The only external dependencies are WinAPI, Direct3D 12,
and parts of C/C++ standard library (but STL containers, exceptions, or RTTI are
not used).

The library is developed and tested using Microsoft Visual Studio 2022, but it
should work with other compilers as well. It is designed for 64-bit code.

To use the library in your project:

(1.) Copy files `D3D12MemAlloc.cpp`, `%D3D12MemAlloc.h` to your project.

(2.) Make `D3D12MemAlloc.cpp` compiling as part of the project, as C++ code.

(3.) Include library header in each CPP file that needs to use the library.

\code
#include "D3D12MemAlloc.h"
\endcode

(4.) Right after you created `ID3D12Device`, fill D3D12MA::ALLOCATOR_DESC
structure and call function D3D12MA::CreateAllocator to create the main
D3D12MA::Allocator object.

Please note that all symbols of the library are declared inside #D3D12MA namespace.

\code
IDXGIAdapter* adapter = ...
ID3D12Device* device = ...

D3D12MA::ALLOCATOR_DESC allocatorDesc = {};
allocatorDesc.pDevice = device;
allocatorDesc.pAdapter = adapter;
allocatorDesc.Flags = D3D12MA_RECOMMENDED_ALLOCATOR_FLAGS;

D3D12MA::Allocator* allocator;
HRESULT hr = D3D12MA::CreateAllocator(&allocatorDesc, &allocator);
// Check hr...
\endcode

(5.) Right before destroying the D3D12 device, destroy the allocator object.

\code
allocator->Release();
\endcode

Objects of this library must be destroyed by calling `Release` method.
They are somewhat compatible with COM: they implement `IUnknown` interface with its virtual methods: `AddRef`, `Release`, `QueryInterface`,
and they are reference-counted internally.
You can use smart pointers designed for COM with objects of this library - e.g. `CComPtr` or `Microsoft::WRL::ComPtr`.
The reference counter is thread-safe.
`QueryInterface` method supports only `IUnknown`, as classes of this library don't define their own GUIDs.


\section quick_start_creating_resources Creating resources

To use the library for creating resources (textures and buffers), call method
D3D12MA::Allocator::CreateResource in the place where you would previously call
`ID3D12Device::CreateCommittedResource`.

The function has similar syntax, but it expects structure D3D12MA::ALLOCATION_DESC
to be passed along with `D3D12_RESOURCE_DESC` and other parameters for created
resource. This structure describes parameters of the desired memory allocation,
including choice of `D3D12_HEAP_TYPE`.

The function returns a new object of type D3D12MA::Allocation.
It represents allocated memory and can be queried for size, offset, `ID3D12Heap`.
It also holds a reference to the `ID3D12Resource`, which can be accessed by calling D3D12MA::Allocation::GetResource().

\code
D3D12_RESOURCE_DESC resourceDesc = {};
resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
resourceDesc.Alignment = 0;
resourceDesc.Width = 1024;
resourceDesc.Height = 1024;
resourceDesc.DepthOrArraySize = 1;
resourceDesc.MipLevels = 1;
resourceDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
resourceDesc.SampleDesc.Count = 1;
resourceDesc.SampleDesc.Quality = 0;
resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

D3D12MA::ALLOCATION_DESC allocationDesc = {};
allocationDesc.HeapType = D3D12_HEAP_TYPE_DEFAULT;

D3D12MA::Allocation* allocation;
HRESULT hr = allocator->CreateResource(
    &allocationDesc,
    &resourceDesc,
    D3D12_RESOURCE_STATE_COPY_DEST,
    NULL,
    &allocation,
    IID_NULL, NULL);
// Check hr...

ID3D12Resource* res = allocation->GetResource();
// Use res...
\endcode

You need to release the allocation object when no longer needed.
This will also release the D3D12 resource.

\code
allocation->Release();
\endcode

The advantage of using the allocator instead of creating committed resource, and
the main purpose of this library, is that it can decide to allocate bigger memory
heap internally using `ID3D12Device::CreateHeap` and place multiple resources in
it, at different offsets, using `ID3D12Device::CreatePlacedResource`. The library
manages its own collection of allocated memory blocks (heaps) and remembers which
parts of them are occupied and which parts are free to be used for new resources.

It is important to remember that resources created as placed don't have their memory
initialized to zeros, but may contain garbage data, so they need to be fully initialized
before usage, e.g. using Clear (`ClearRenderTargetView`), Discard (`DiscardResource`),
or Copy (`CopyResource`).

The library also automatically handles resource heap tier.
When `D3D12_FEATURE_DATA_D3D12_OPTIONS::ResourceHeapTier == D3D12_RESOURCE_HEAP_TIER_1`,
resources of 3 types: buffers, textures that are render targets or depth-stencil,
and other textures must be kept in separate heaps. When `D3D12_RESOURCE_HEAP_TIER_2`,
they can be kept together. By using this library, you don't need to handle this
manually.


\section quick_start_resource_reference_counting Resource reference counting

`ID3D12Resource` and other interfaces of Direct3D 12 use COM, so they are reference-counted.
Objects of this library are reference-counted as well.
An object of type D3D12MA::Allocation remembers the resource (buffer or texture)
that was created together with this memory allocation
and holds a reference to the `ID3D12Resource` object.
(Note this is a difference to Vulkan Memory Allocator, where a `VmaAllocation` object has no connection
with the buffer or image that was created with it.)
Thus, it is important to manage the resource reference counter properly.

<b>The simplest use case</b> is shown in the code snippet above.
When only D3D12MA::Allocation object is obtained from a function call like D3D12MA::Allocator::CreateResource,
it remembers the `ID3D12Resource` that was created with it and holds a reference to it.
The resource can be obtained by calling `allocation->GetResource()`, which doesn't increment the resource
reference counter.
Calling `allocation->Release()` will decrease the resource reference counter, which is 1 in this case,
so the resource will be released.

<b>Second option</b> is to retrieve a pointer to the resource along with D3D12MA::Allocation.
Last parameters of the resource creation function can be used for this purpose.

\code
D3D12MA::Allocation* allocation;
ID3D12Resource* resource;
HRESULT hr = allocator->CreateResource(
    &allocationDesc,
    &resourceDesc,
    D3D12_RESOURCE_STATE_COPY_DEST,
    NULL,
    &allocation,
    IID_PPV_ARGS(&resource));

// Use resource...
\endcode

In this case, returned pointer `resource` is equal to `allocation->GetResource()`,
but the creation function additionally increases resource reference counter for the purpose of returning it from this call
(it actually calls `QueryInterface` internally), so the resource will have the counter = 2.
The resource then need to be released along with the allocation, in this particular order,
to make sure the resource is destroyed before its memory heap can potentially be freed.

\code
resource->Release();
allocation->Release();
\endcode

<b>More advanced use cases</b> are possible when we consider that an D3D12MA::Allocation object can just hold
a reference to any resource.
It can be changed by calling D3D12MA::Allocation::SetResource. This function
releases the old resource and calls `AddRef` on the new one.

Special care must be taken when performing <b>defragmentation</b>.
The new resource created at the destination place should be set as `pass.pMoves[i].pDstTmpAllocation->SetResource(newRes)`,
but it is moved to the source allocation at end of the defragmentation pass,
while the old resource accessible through `pass.pMoves[i].pSrcAllocation->GetResource()` is then released.
For more information, see documentation chapter \ref defragmentation.


\section quick_start_mapping_memory Mapping memory

The process of getting regular CPU-side pointer to the memory of a resource in
Direct3D is called "mapping". There are rules and restrictions to this process,
as described in D3D12 documentation of `ID3D12Resource::Map` method.

Mapping happens on the level of particular resources, not entire memory heaps,
and so it is out of scope of this library. Just as the documentation of the `Map` function says:

- Returned pointer refers to data of particular subresource, not entire memory heap.
- You can map same resource multiple times. It is ref-counted internally.
- Mapping is thread-safe.
- Unmapping is not required before resource destruction.
- Unmapping may not be required before using written data - some heap types on
  some platforms support resources persistently mapped.

When using this library, you can map and use your resources normally without
considering whether they are created as committed resources or placed resources in one large heap.

Example for buffer created and filled in `UPLOAD` heap type:

\code
const UINT64 bufSize = 65536;
const float* bufData = ...;

D3D12_RESOURCE_DESC resourceDesc = {};
resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
resourceDesc.Alignment = 0;
resourceDesc.Width = bufSize;
resourceDesc.Height = 1;
resourceDesc.DepthOrArraySize = 1;
resourceDesc.MipLevels = 1;
resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
resourceDesc.SampleDesc.Count = 1;
resourceDesc.SampleDesc.Quality = 0;
resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

D3D12MA::ALLOCATION_DESC allocationDesc = {};
allocationDesc.HeapType = D3D12_HEAP_TYPE_UPLOAD;

D3D12Resource* resource;
D3D12MA::Allocation* allocation;
HRESULT hr = allocator->CreateResource(
    &allocationDesc,
    &resourceDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    NULL,
    &allocation,
    IID_PPV_ARGS(&resource));

D3D12_RANGE emptyRange = {0, 0};
void* mappedPtr;
hr = resource->Map(0, &emptyRange, &mappedPtr);

memcpy(mappedPtr, bufData, bufSize);

resource->Unmap(0, NULL);
\endcode


\section quick_start_helper_structures Helper structures

DirectX 12 Agility SDK offers a library of helpers in files "build\native\include\d3dx12\*.h".
They include structures that help with complete and concise initialization of the core D3D12 `*_DESC` structures
by using some basic C++ features (constructors, static methods, default parameters).
They inherit from these structures, so they support implicit casting to them.
For example, structure `CD3DX12_RESOURCE_DESC` can be used to conveniently fill in structure `D3D12_RESOURCE_DESC`.

Similarly, this library provides a set of helper structures that aid in initialization of some of the `*_DESC` structures defined in the library.
These are:

- D3D12MA::CALLOCATION_DESC, which inherits from D3D12MA::ALLOCATION_DESC.
- D3D12MA::CPOOL_DESC, which inherits from D3D12MA::POOL_DESC.
- D3D12MA::CVIRTUAL_BLOCK_DESC, which inherits from D3D12MA::VIRTUAL_BLOCK_DESC.
- D3D12MA::CVIRTUAL_ALLOCATION_DESC, which inherits from D3D12MA::VIRTUAL_ALLOCATION_DESC.

For example, when you want to create a buffer in the `UPLAOD` heap using minimal allocation time, you can use base structures:

\code
D3D12MA::ALLOCATION_DESC allocDesc;
allocDesc.Flags = D3D12MA::ALLOCATION_FLAG_STRATEGY_MIN_TIME;
allocDesc.HeapType = D3D12_HEAP_TYPE_UPLOAD;
allocDesc.ExtraHeapFlags = D3D12_HEAP_FLAG_NONE;
allocDesc.CustomPool = NULL;
allocDesc.pPrivateData = NULL;

D3D12_RESOURCE_DESC resDesc;
resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
resDesc.Alignment = 0;
resDesc.Width = myBufSize;
resDesc.Height = 1;
resDesc.DepthOrArraySize = 1;
resDesc.MipLevels = 1;
resDesc.Format = DXGI_FORMAT_UNKNOWN;
resDesc.SampleDesc.Count = 1;
resDesc.SampleDesc.Quality = 0;
resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
resDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

D3D12MA::Allocation* alloc;
ID3D12Resource* res;
HRESULT hr = allocator->CreateResource(&allocDesc, &resDesc,
    D3D12_RESOURCE_STATE_COMMON, NULL, &alloc, IID_PPV_ARGS(&res));
// Check hr...
\endcode

Or you can use helper structs from both D3X12 library and this library to make the code shorter:

\code
D3D12MA::CALLOCATION_DESC allocDesc = D3D12MA::CALLOCATION_DESC{
    D3D12_HEAP_TYPE_UPLOAD,
    D3D12MA::ALLOCATION_FLAG_STRATEGY_MIN_TIME };

CD3DX12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(myBufSize);

D3D12MA::Allocation* alloc;
ID3D12Resource* res;
HRESULT hr = allocator->CreateResource(&allocDesc, &resDesc,
    D3D12_RESOURCE_STATE_COMMON, NULL, &alloc, IID_PPV_ARGS(&res));
// Check hr...
\endcode

\page custom_pools Custom memory pools

A "pool" is a collection of memory blocks that share certain properties.
Allocator creates 3 default pools: for `D3D12_HEAP_TYPE_DEFAULT`, `UPLOAD`, `READBACK`.
A default pool automatically grows in size. Size of allocated blocks is also variable and managed automatically.
Typical allocations are created in these pools. You can also create custom pools.

\section custom_pools_usage Usage

To create a custom pool, fill in structure D3D12MA::POOL_DESC and call function D3D12MA::Allocator::CreatePool
to obtain object D3D12MA::Pool. Example:

\code
POOL_DESC poolDesc = {};
poolDesc.HeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
poolDesc.Flags = D3D12MA_RECOMMENDED_POOL_FLAGS;
poolDesc.HeapFlags = D3D12MA_RECOMMENDED_HEAP_FLAGS | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

Pool* pool;
HRESULT hr = allocator->CreatePool(&poolDesc, &pool);
\endcode

To allocate resources out of a custom pool, only set member D3D12MA::ALLOCATION_DESC::CustomPool.
Example:

\code
ALLOCATION_DESC allocDesc = {};
allocDesc.CustomPool = pool;

D3D12_RESOURCE_DESC resDesc = ...
Allocation* alloc;
hr = allocator->CreateResource(&allocDesc, &resDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &alloc, IID_NULL, NULL);
\endcode

All allocations must be released before releasing the pool.
The pool must be released before relasing the allocator.

\code
alloc->Release();
pool->Release();
\endcode

\section custom_pools_features_and_benefits Features and benefits

While it is recommended to use default pools whenever possible for simplicity and to give the allocator
more opportunities for internal optimizations, custom pools may be useful in following cases:

- To keep some resources separate from others in memory.
- To keep track of memory usage of just a specific group of resources. %Statistics can be queried using
  D3D12MA::Pool::CalculateStatistics.
- To use specific size of a memory block (`ID3D12Heap`). To set it, use member D3D12MA::POOL_DESC::BlockSize.
  When set to 0, the library uses automatically determined, variable block sizes.
- To reserve some minimum amount of memory allocated. To use it, set member D3D12MA::POOL_DESC::MinBlockCount.
- To limit maximum amount of memory allocated. To use it, set member D3D12MA::POOL_DESC::MaxBlockCount.
- To use extended parameters of the D3D12 memory allocation. While resources created from default pools
  can only specify `D3D12_HEAP_TYPE_DEFAULT`, `UPLOAD`, `READBACK`, a custom pool may use non-standard
  `D3D12_HEAP_PROPERTIES` (member D3D12MA::POOL_DESC::HeapProperties) and `D3D12_HEAP_FLAGS`
  (D3D12MA::POOL_DESC::HeapFlags), which is useful e.g. for cross-adapter sharing or UMA
  (see also D3D12MA::Allocator::IsUMA).

New versions of this library support creating **committed allocations in custom pools**.
It is supported only when D3D12MA::POOL_DESC::BlockSize = 0.
To use this feature, set D3D12MA::ALLOCATION_DESC::CustomPool to the pointer to your custom pool and
D3D12MA::ALLOCATION_DESC::Flags to D3D12MA::ALLOCATION_FLAG_COMMITTED. Example:

\code
ALLOCATION_DESC allocDesc = {};
allocDesc.CustomPool = pool;
allocDesc.Flags = ALLOCATION_FLAG_COMMITTED;

D3D12_RESOURCE_DESC resDesc = ...
Allocation* alloc;
ID3D12Resource* res;
hr = allocator->CreateResource(&allocDesc, &resDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &alloc, IID_PPV_ARGS(&res));
\endcode

This feature may seem unnecessary, but creating committed allocations from custom pools may be useful
in some cases, e.g. to have separate memory usage statistics for some group of resources or to use
extended allocation parameters, like custom `D3D12_HEAP_PROPERTIES`, which are available only in custom pools.


\page optimal_allocation Optimal resource allocation

This library tries to automatically make optimal choices for the resources you create,
so you don't need to care about them.
There are some advanced features of Direct3D 12 that you may use to optimize your memory management.
There are also some settings in %D3D12MA that you may change to alter its default behavior.
This page provides miscellaneous advice about features of D3D12 and %D3D12MA that are
non-essential, but may improve the stability or performance of your app.

\section optimal_allocation_avoiding_running_out_of_memory Avoiding running out of memory

When trying to allocate more memory than available in the current heap
(e.g., video memory on the graphics card, system memory), one of few bad things can happen:

- The allocation (resource creation) function call can fail with `HRESULT` value other than `S_OK`.
- The allocation may succeed, but take long time (even a significant fraction of a second).
- Some resources are automatically demoted from video memory to system memory, degrading the app performance.
- Even a crash of the entire graphics driver can happen, resulting in the D3D12 "device removal", which is usually
  catastrophic for the application.

Unfortunately, there is no way to be 100% protected against memory overcommitment.
The best approach is to avoid allocating too much memory.

The full capacity of the memory can be queried using function D3D12MA::Allocator::GetMemoryCapacity.
However, it is not recommended, because the amount of memory available to the application
is typically smaller than the full capacity, as some portion of it is reserved by the operating system
or used by other processes.

Because of this, the recommended way of fetching the **memory budget** available to the application
is using function D3D12MA::Allocator::GetBudget.
Preventing value D3D12MA::Budget::UsageBytes from exceeding the D3D12MA::Budget::BudgetBytes
is probably the best we can do in trying to avoid the consequences of over-commitment.
For more information, see also: \subpage statistics.

Example:

\code
D3D12MA::Budget videoMemBudget = {};
allocator->GetBudget(&videoMemBudget, NULL);

UINT64 freeBytes = videoMemBudget.BudgetBytes - videoMemBudget.UsageBytes;
gameStreamingSystem->SetAvailableFreeMemory(freeBytes);
\endcode

\par Implementation detail
DXGI interface offers function `IDXGIAdapter3::QueryVideoMemoryInfo` that queries the current memory usage and budget.
This library automatically makes use of it when available (when you use recent enough version of the DirectX SDK).
If not, it falls back to estimating the usage and budget based on the total amount of the allocated memory
and 80% of the full memory capacity, respectively.

\par Implementation detail
Allocating large heaps and creating placed resources in them is one of the main features of this library.
However, if allocating new such block would exceed the budget, it will automatically prefer creating the resource as committed
to have exactly the right size, which can lower the chance of getting into trouble in case of over-commitment.

When creating non-essential resources, you can use D3D12MA::ALLOCATION_FLAG_WITHIN_BUDGET.
Then, in case the allocation would exceed the budget, the library will return failure from the function
without attempting to allocate the actual D3D12 memory.

It may also be a good idea to support failed resource creation.
For non-essential resources, when function D3D12MA::Allocator::CreateResource fails with a result other than `S_OK`,
it is worth implementing some way of recovery instead of terminating or crashing the entire app.

\section optimal_allocation_allocation_Performance Allocation performance

Creating D3D12 resources (buffers and textures) can be a time-consuming operation.
The duration can be unpredictable, spanning from a small fraction of a millisecond to a significant fraction of a second.
Thus, it is recommended to allocate all the memory and create all the resources needed upfront
rather than doing it during application runtime.
For example, a video game can try to create its resources on startup or when loading a new level.
Of course, is is not always possible.
For example, open-world games may require loading and unloading some graphical assets in the background (often called "streaming").

Creating and releasing D3D12 resources **on a separate thread** in the background may help.
Both `ID3D12Device` and D3D12MA::Allocator objects are thread-safe, synchronized internally.
However, cases were observed where resource creation calls like `ID3D12Device::CreateCommittedResource`
were blocking other D3D12 calls like `ExecuteCommandLists` or `Present`
somewhere inside the graphics driver, so hitches can happen even when using multithreading.

The most expensive part is typically **the allocation of a new D3D12 memory heap**.
This library tackles this problem by automatically allocating large heaps (64 MB by default)
and creating resources as placed inside of them.
When a new requested resource can be placed in a free space of an existing heap and doesn't require allocating a new heap,
this operation is typically much faster, as it only requires creating a new `ID3D12Resource` object
and not allocating new memory.
This is the main benefit of using %D3D12MA compared to the naive approach of using Direct3D 12 directly
and creating each resource as committed with `CreateCommittedResource`, which would result in a separate allocation of an implicit heap every time.

When **a large number of small buffers** needs to be created, the overhead of creating even just separate `ID3D12Resource` objects can be significant.
It can be avoided by creating one or few larger buffers and manually sub-allocating parts of them for specific needs.
This library can also help with it. See section "Sub-allocating buffers" below.

\par Implementation detail
The CPU performance overhead of using this library is low.
It uses a high-quality allocation algorithm called Two-Level Segregated Fit (TLSF),
which in most cases can find a free place for a new allocation in few steps.
The library also doesn't perform too many CPU heap allocations.
In may cases, the allocation happens with 0 new CPU heap allocations performed by the library.
Even the creation of a D3D12MA::Allocation object itself doesn't typically feature an CPU allocation,
because these objects are returned out of a dedicated memory pool.

Another reason for the slowness of D3D12 memory allocation is the guarantee that the **newly allocated memory is filled with zeros**.
When creating and destroying resources placed in an existing heap, this overhead is not present,
and the memory is not zeroed - it may contain random data left by the resource previously allocated in that place.
In recent versions of the DirectX 12 SDK, clearing the memory of the newly created D3D12 heaps can also be disabled for the improved performance.
%D3D12MA can use this feature when:

- D3D12MA::ALLOCATOR_FLAG_DEFAULT_POOLS_NOT_ZEROED is used during the creation of the main allocator object.
- `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` is passed to D3D12MA::POOL_DESC::HeapFlags during the creation of a custom pool.

It is recommended to always use these flags.
The downside is that when the memory is not filled with zeros, while you don't properly clear it or otherwise initialize its content before use
(which is required by D3D12), you may observe incorrect behavior.
This problem mostly affects render-target and depth-stencil textures.

When an allocation needs to be made in a performance-critical code, you can use D3D12MA::ALLOCATION_FLAG_STRATEGY_MIN_TIME.
In influences multiple heuristics inside the library to prefer faster allocation
at the expense of possibly less optimal placement in the memory.

If the resource to be created is non-essential, while the performance is paramount,
you can also use D3D12MA::ALLOCATION_FLAG_NEVER_ALLOCATE.
It will create the resource only if it can be placed inside and existing memory heap
and return failure from the function if a new heap would need to be allocated,
which should guarantee good performance of such function call.

\section optimal_allocation_suballocating_buffers Sub-allocating buffers

When a large number of small buffers needs to be created, the overhead of creating separate `ID3D12Resource` objects can be significant.
It can also cause a significant waste of memory, as placed buffers need to be aligned to `D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT` = 64 KB by default.
These problems can be avoided by creating one or few larger buffers and manually sub-allocating parts of them for specific needs.

It requires implementing a custom allocator for the data inside the buffer and using offsets to individual regions.
When all the regions can be allocated linearly and freed all at once, implementing such allocator is trivial.
When every region has the same size, implementing an allocator is also quite simple when using a "free list" algorithm.
However, when regions can have different sizes and can be allocated and freed in random order,
it requires a full allocation algorithm.
%D3D12MA can help with it by exposing its core allocation algorithm for custom usages.
For more details and example code, see chapter: \subpage virtual_allocator.
It can be used for all the cases mentioned above without too much performance overhead,
because the D3D12MA::VirtualAllocation object is just a lightweight handle.

When sub-allocating a buffer, you need to remember to explicitly request proper alignment required for each region.
For example, data used as a constant buffer must be aligned to `D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT` = 256 B.

\section optimal_allocation_residency_priority Residency priority

When too much video memory is allocated, one of the things that can happen is the system
demoting some heaps to the system memory.
Moving data between memory pools or reaching out directly to the system memory through PCI Express bus can have large performance overhead,
which can slow down the application, or even make the game unplayable any more.
Unfortunately, it is not possible to fully control or prevent this demotion.
Best thing to do is avoiding memory over-commitment.
For more information, see section "Avoiding running out of memory" above.

Recent versions of DirectX 12 SDK offer function `ID3D12Device1::SetResidencyPriority` that sets a hint
about the priority of a resource - how important it is to stay resident in the video memory.
Setting the priority happens at the level of an entire memory heap.
%D3D12MA offers an interface to set this priority in form of D3D12MA::POOL_DESC::ResidencyPriority parameter.
It affects all allocations made out of the custom pool created with it, both placed inside large heaps
and created as committed.

It is recommended to create a custom pool for the purpose of using high residency priority
of all resources that are critical for the performance, especially those that are written by the GPU,
like render-target, depth-stencil textures, UAV textures and buffers.
It is also worth creating them as committed, so that each one will have its own implicit heap.
This can minimize the chance that an entire large heap is demoted to system memory, degrading performance
of all the resources placed in it.

Example:

\code
D3D12MA::CPOOL_DESC poolDesc = D3D12MA::CPOOL_DESC{
    D3D12_HEAP_TYPE_DEFAULT,
    D3D12MA_RECOMMENDED_HEAP_FLAGS | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
poolDesc.ResidencyPriority = D3D12_RESIDENCY_PRIORITY_HIGH; // !!!

D3D12MA::Pool* pool;
HRESULT hr = allocator->CreatePool(&poolDesc, &pool);
// Check hr...

D3D12MA::CALLOCATION_DESC allocDesc = D3D12MA::CALLOCATION_DESC{
    pool,
    ALLOCATION_FLAG_COMMITTED }; // !!!

CD3DX12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(
    1048576); // Requested buffer size.

D3D12MA::Allocation* alloc;
hr = allocator->CreateResource(&allocDesc, &resDesc, D3D12_RESOURCE_STATE_COMMON,
    NULL, &alloc, IID_NULL, NULL);
// Check hr...
\endcode

When you have a committed allocation created, you can also set the residency priority of its resource
using the D3D12 function:

\code
D3D12MA::Allocation* committedAlloc = ...
ID3D12Pageable* res = committedAlloc->GetResource();
D3D12_RESIDENCY_PRIORITY priority = D3D12_RESIDENCY_PRIORITY_HIGH;
device1->SetResidencyPriority(1, &res, &priority);
\endcode

Note this is not the same as explicit eviction controlled using `ID3D12Device::Evict` and `MakeResident` functions.
Resources evicted explicitly are illegal to access until they are made resident again,
while the demotion described here happens automatically and only slows down the execution.

\section optimal_allocation_gpu_upload_heap GPU upload heap

Direct3D 12 offers a fixed set of memory heap types:

- `D3D12_HEAP_TYPE_DEFAULT`: Represents the video memory. It is available and fast to access for the GPU.
  It should be used for all resources that are written by the GPU (like render-target and depth-stencil textures,
  UAV) and resources that are frequently read by the GPU (like textures intended for sampling,
  vertex, index, and constant buffers).
- `D3D12_HEAP_TYPE_UPLOAD`: Represents the system memory that is uncached and write-combined.
  It can be mapped and accessed by the CPU code using a pointer.
  It supports only buffers, not textures.
  It is intended for "staging buffers" that are filled by the CPU code and then used as a source of copy operations to the `DEFAULT` heap.
  It can also be accessed directly by the GPU - shaders can read from buffers created in this memory.
- `D3D12_HEAP_TYPE_READBACK`: Represents the system memory that is cached.
  It is intended for buffers used as a destination of copy operations from the `DEFAULT` heap.

Note that in systems with a discrete graphics card, access to system memory is fast from the CPU code
(like the C++ code mapping D3D12 buffers and accessing them through a pointer),
while access to the video memory is fast from the GPU code (like shaders reading and writing buffers and textures).
Any copy operation or direct access between these memory heap types happens through PCI Express bus, which can be relatively slow.

Modern systems offer a feature called **Resizable BAR (ReBAR)** that gives the CPU direct access to the full video memory.
To be available, this feature needs to be supported by the whole hardware-software environment, including:

- Supporting motherboard and its UEFI.
- Supporting graphics card and its graphics driver.
- Supporting operating system.
- The feature needs to be enabled in the UEFI settings. It is typically called "Above 4G Decoding" and "Resizable Bar".

Recent versions of DirectX 12 SDK give access to this feature in form of a new, 4th memory pool: `D3D12_HEAP_TYPE_GPU_UPLOAD`.
Resources created in it behave logically similar to the `D3D12_HEAP_TYPE_UPLOAD` heap:

- They support mapping and direct access from the CPU code through a pointer.
- The mapped memory is uncached and write-combined, so it should be only written sequentially
  (e.g., number-by-number or using `memcpy`). It shouldn't be accessed randomly or read,
  because it is extremely slow for uncached memory.
- Only buffers are supported.
- Those buffers can be used as a source of copy operations or directly accessed by the GPU.

The main difference is that resources created in the new `D3D12_HEAP_TYPE_GPU_UPLOAD` are placed in the video memory,
while resources created in the old `D3D12_HEAP_TYPE_UPLOAD` are placed in the system memory.
This implies which budgets are consumed by new resources allocated in those heaps.
This also implies which operations involve transferring data through the PCI Express bus.

- As `D3D12_HEAP_TYPE_UPLOAD` uses the system memory, writes from the CPU code through a mapped pointer are faster,
  while copies or direct access from the GPU are slower because they need to go through PCIe.
- As the new `D3D12_HEAP_TYPE_GPU_UPLOAD` uses the video memory,
  copies or direct access from the GPU are faster,
  while writes from the CPU code through a mapped pointer can be slower, because they need to go through PCIe.
  For maximum performance of copy operations from this heap, a graphics or compute queue should be used, not a copy queue.

GPU Upload Heap can be used for performance optimization of some resources that need to be written by the CPU and read by the GPU.
It can be beneficial especially for resources that need to change frequently (often called "dynamic").

%D3D12MA supports GPU upload heap when recent enough version of DirectX 12 SDK is used and when the current system supports it.
The support can be queried using function D3D12MA::Allocator::IsGPUUploadHeapSupported().
When it returns `TRUE`, you can create resources using `D3D12_HEAP_TYPE_GPU_UPLOAD`.
You can also just try creating such resource. Example:

\code
    D3D12MA::CALLOCATION_DESC allocDesc = D3D12MA::CALLOCATION_DESC{
        D3D12_HEAP_TYPE_GPU_UPLOAD }; // !!!

    CD3DX12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(
        1048576); // Requested buffer size.

    D3D12MA::Allocation* alloc;
    ID3D12Resource* res;
    hr = allocator->CreateResource(&allocDesc, &resDesc, D3D12_RESOURCE_STATE_COMMON,
        NULL, &alloc, IID_PPV_ARGS(&res));
    if(SUCCEEDED(hr))
    {
        // Fast path for data upload.

        D3D12_RANGE emptyRange = {0, 0};
        void* mappedPtr = NULL;
        hr = res->Map(0, &emptyRange, &mappedPtr);
        memcpy(mappedPtr, srcData, 1048576);
        res->Unmap(0, NULL); // Optional. You can leave it persistently mapped.

        D3D12_GPU_VIRTUAL_ADDRESS gpuva = res->GetGPUVirtualAddress();
        // Use gpuva to access the buffer on the GPU...
    }
    else if(hr == E_NOTIMPL)
    {
        // GPU Upload Heap not supported in this system.
        // Fall back to creating a staging buffer in UPLOAD and another copy in DEFAULT.
        
        allocDesc.HeapType = D3D12_HEAP_TYPE_UPLOAD;
        // ...
    }
    else
        // Some other error code e.g., out of memory...
\endcode

\section optimal_allocation_committed_vs_placed Committed versus placed resources

When using D3D12 API directly, there are 3 ways of creating resources:

1. **Committed**, using function `ID3D12Device::CreateCommittedResource`.
   It creates the resource with its own memory heap, which is called an "implicit heap" and cannot be accessed directly.
2. **Placed**, using function `ID3D12Device::CreatePlacedResource`.
   A `ID3D12Heap` needs to be created beforehand using `ID3D12Device::CreateHeap`.
   Then, the resource can be created as placed inside the heap at a specific offset.
3. **Reserved**, using function `ID3D12Device::CreateReservedResource`.
   This library doesn't support them directly.

A naive solution would be to create all the resources as committed.
It works, because in D3D12 there is no strict limit on the number of resources or heaps that can be created.
However, there are certain advantages and disadvantages of using committed versus placed resources:

- The biggest advantage of using placed resources is the allocation performance.
  Once a heap is allocated, creating and releasing resources placed in it can be much faster than
  creating them as committed, which would involve allocating a new heap for each resource.
  - Using large number of small heaps can put an extra burden on the software stack,
    including D3D12 runtime, graphics driver, operating system, and developer tools like Radeon Memory Visualizer (RMV).
- The advantage of committed resources is that their implicit heaps have exactly the right size,
  while creating resources as placed inside larger heaps can lead to some memory wasted because:
  - Some part of the allocated heap memory is unused.
  - After placed resources of various sizes are created and released in random order,
    gaps between remaining resources can be too small to fit new allocations.
    This is also known as "fragmentation". A solution to this problem is implementing \subpage defragmentation.
  - The alignment required by placed resources can leave gaps between them, while the driver can pack individual committed resources better.
    For details, see section "Resource alignment" below.
- The advantage of committed resources is that they are always created with a new heap, which is initialized with zeros.
  When a resource is created as placed, the memory may contain random data left by the resource previously allocated in that place.
  When the memory is not filled with zeros, while you don't properly clear it or otherwise initialize its content before use
  (which is required by D3D12), you may observe incorrect behavior.
  On the other hand, using committed resources and having every new resource filled with zeros can leave this kind of bugs undetected.
- Manual eviction with `ID3D12Device::Evict` and `MakeResident` functions work at the level of the entire heap,
  and so does `ID3D12Device1::SetResidencyPriority`, so creating resources as committed allows more fine-grained control
  over the eviction and residency priority of individual resources.
- The advantage of placed resources is that they can be created in a region of a heap overlapping with some other resources.
  This approach is commonly called "aliasing".
  It can save memory, but it needs careful control over the resources that overlap in memory
  to make sure they are not used at the same time, there is an aliasing barrier issued between their usage,
  and the resource used after aliasing is correctly cleared every time.
  Committed resources don't offer this possibility, because every committed resource has its own exclusive memory heap.
  For more information, see chapter \subpage resource_aliasing.

When creating resources with the help of %D3D12MA using function D3D12MA::Allocator::CreateResource,
you typically don't need to care about all this.
The library automatically makes the choice of creating the new resource as committed or placed.
However, in cases when you need the information or the control over this choice between committed and placed,
the library offers facilities to do that, described below.

\par Implementation detail
%D3D12MA creates large heaps (default size is 64 MB) and creates resources as placed in them.
However, it may decide that it is required or preferred to create the specific resource as committed for many reasons, including:
- When the resource is large (larger than half of the default heap size).
- When allocating an entire new heap would exceed the current budget or when we are already over the budget.
- When the resource is a very small buffer. Placed buffers need to be aligned to 64 KB by default,
  while creating them as committed can allow the driver to pack them better.
  This heuristics can be disabled for an individual resource by using D3D12MA::ALLOCATION_FLAG_STRATEGY_MIN_TIME
  and for the entire allocator by using D3D12MA::ALLOCATOR_FLAG_DONT_PREFER_SMALL_BUFFERS_COMMITTED.
- When the resource uses non-standard flags specified via D3D12MA::ALLOCATION_DESC::ExtraHeapFlags.

<b>You can check whether an allocation was created as a committed resource</b> by checking if its heap is null.
Committed resources have an implicit heap that is not directly accessible.

\code
bool isCommitted = allocation->GetHeap() == NULL;
\endcode

<b>You can request a new resource to be created as committed</b> by using D3D12MA::ALLOCATION_FLAG_COMMITTED.
Note that committed resources can also be created out of \subpage custom_pools.

You can also request all resources to be created as committed globally for the entire allocator
by using D3D12MA::ALLOCATOR_FLAG_ALWAYS_COMMITTED.
However, this contradicts the main purpose of using this library.
It can also prevent certain other features of the library to be used.
This flag should be used only for debugging purposes.

You can create a custom pool with an explicit block size by specifying non-zero D3D12MA::POOL_DESC::BlockSize.
When doing this, all **resources created in such pool are placed** in those blocks (heaps) and never created as committed.
Example:

\code
D3D12MA::CPOOL_DESC poolDesc = D3D12MA::CPOOL_DESC{
    D3D12_HEAP_TYPE_DEFAULT,
    D3D12MA_RECOMMENDED_HEAP_FLAGS | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
poolDesc.BlockSize = 100llu * 1024 * 1024; // 100 MB. Explicit BlockSize guarantees placed.

D3D12MA::Pool* pool;
HRESULT hr = allocator->CreatePool(&poolDesc, &pool);
// Check hr...

D3D12MA::CALLOCATION_DESC allocDesc = D3D12MA::CALLOCATION_DESC{ pool };

CD3DX12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(
    90llu * 1024 * 1024); // 90 MB

D3D12MA::Allocation* alloc;
ID3D12Resource* res;
hr = allocator->CreateResource(&allocDesc, &resDesc, D3D12_RESOURCE_STATE_COMMON,
    NULL, &alloc, IID_PPV_ARGS(&res));
// Check hr...

// Even a large buffer like this, filling 90% of the block, was created as placed!
assert(alloc->GetHeap() != NULL);
\endcode

<b>You can request a new resource to be created as placed</b> by using D3D12MA::ALLOCATION_FLAG_CAN_ALIAS.
This is required especially if you plan to create another resource in the same region of memory, aliasing with your resource -
hence the name of this flag.

Note D3D12MA::ALLOCATION_FLAG_CAN_ALIAS can be even combined with D3D12MA::ALLOCATION_FLAG_COMMITTED.
In this case, the resource is not created as committed, but it is also not placed as part of a larger heap.
What happens instead is that a new heap is created with the exact size required for the resource,
and the resource is created in it, placed at offset 0.

\section optimal_allocation_resource_alignment Resource alignment

Certain types of resources require certain alignment in memory.
An alignment is a requirement for the address or offset to the beginning of the resource to be a multiply of some value, which is always a power of 2.
For committed resources, the problem is non-existent, because committed resources have their own implicit heaps
where they are created at offset 0, which meets any alignment requirement.
For placed resources, %D3D12MA takes care of the alignment automatically.

\par Implementation detail
Default alignment required MSAA textures is `D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT` = 4 MB.
Default alignment required for buffers and other textures is `D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT` = 64 KB.

Because the alignment required for buffers is 64 KB, **small buffers** can waste a lot of memory in between when created as placed.
When such small buffers are created as committed, some graphics drivers are able to pack them better.
%D3D12MA automatically takes advantage of this by preferring to create small buffers as committed.
This heuristics is enabled by default. It is also a tradeoff - it can make the allocation of these buffers slower.
It can be disabled for an individual resource by using D3D12MA::ALLOCATION_FLAG_STRATEGY_MIN_TIME
and for the entire allocator by using D3D12MA::ALLOCATOR_FLAG_DONT_PREFER_SMALL_BUFFERS_COMMITTED.

For certain textures that meet a complex set of requirements, special **"small alignment"** can be applied.
Details can be found in Microsoft documentation of the `D3D12_RESOURCE_DESC` structure.
For MSAA textures, the small alignment is `D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT` = 64 KB.
For other textures, the small alignment is `D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT` = 4 KB.
%D3D12MA uses this feature automatically.
Detailed behavior can be disabled or controlled by predefining macro #D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT.

D3D12 also has a concept of **alignment of the entire heap**, passed through `D3D12_HEAP_DESC::Alignment`.
This library automatically sets the alignment as small as possible.
Unfortunately, any heap that has a chance of hosting an MSAA texture needs to have the alignment set to 4 MB.
This problem can be overcome by passing D3D12MA::ALLOCATOR_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED on the creation of the main allocator object
and D3D12MA::POOL_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED on the creation of any custom heap that supports textures, not only buffers.
With those flags, the alignment of the heaps created by %D3D12MA can be lower, but any MSAA textures are created as committed.
You should always use these flags in your code unless you really need to create some MSAA textures as placed.

With DirectX 12 Agility SDK 1.618.1, Microsoft added a new feature called **"tight alignment"**.
Note this is a separate feature than the "small alignment" described earlier.
When using this new SDK and a compatible graphics driver, the API exposes support for this new feature.
Then, a new flag `D3D12_RESOURCE_FLAG_USE_TIGHT_ALIGNMENT` can be added when creating a resource.
D3D12 can then return the alignment required for the resource smaller than the default ones described above.
This library automatically makes use of the tight alignment feature when available and adds that new resource flag.
When the tight alignment is enabled, the heuristics that creates small buffers as committed described above is deactivated,
as it is no longer needed.

You can check if the tight alignment it is available in the current system by calling D3D12MA::Allocator::IsTightAlignmentSupported().
You can tell the library to not use it by specifying D3D12MA::ALLOCATOR_FLAG_DONT_USE_TIGHT_ALIGNMENT.
Typically, you don't need to do any of those.

The library automatically aligns all buffers to at least 256 B, even when the system supports smaller alignment.
This is the alignment required for constant buffers, expressed by `D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT` constant.
You can override this logic for \subpage custom_pools with a specific D3D12MA::POOL_DESC::MinAllocationAlignment.

\page defragmentation Defragmentation

Interleaved allocations and deallocations of many objects of varying size can
cause fragmentation over time, which can lead to a situation where the library is unable
to find a continuous range of free memory for a new allocation despite there is
enough free space, just scattered across many small free ranges between existing
allocations.

To mitigate this problem, you can use defragmentation feature.
It doesn't happen automatically though and needs your cooperation,
because %D3D12MA is a low level library that only allocates memory.
It cannot recreate buffers and textures in a new place as it doesn't remember the contents of `D3D12_RESOURCE_DESC` structure.
It cannot copy their contents as it doesn't record any commands to a command list.

Example:

\code
D3D12MA::DEFRAGMENTATION_DESC defragDesc = {};
defragDesc.Flags = D3D12MA::DEFRAGMENTATION_FLAG_ALGORITHM_FAST;

D3D12MA::DefragmentationContext* defragCtx;
allocator->BeginDefragmentation(&defragDesc, &defragCtx);

for(;;)
{
    D3D12MA::DEFRAGMENTATION_PASS_MOVE_INFO pass;
    HRESULT hr = defragCtx->BeginPass(&pass);
    if(hr == S_OK)
        break;
    else if(hr != S_FALSE)
        // Handle error...

    for(UINT i = 0; i < pass.MoveCount; ++i)
    {
        // Inspect pass.pMoves[i].pSrcAllocation, identify what buffer/texture it represents.
        MyEngineResourceData* resData = (MyEngineResourceData*)pMoves[i].pSrcAllocation->GetPrivateData();
            
        // Recreate this buffer/texture as placed at pass.pMoves[i].pDstTmpAllocation.
        D3D12_RESOURCE_DESC resDesc = ...
        ID3D12Resource* newRes;
        hr = device->CreatePlacedResource(
            pass.pMoves[i].pDstTmpAllocation->GetHeap(),
            pass.pMoves[i].pDstTmpAllocation->GetOffset(), &resDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, NULL, IID_PPV_ARGS(&newRes));
        // Check hr...

        // Store new resource in the pDstTmpAllocation.
        pass.pMoves[i].pDstTmpAllocation->SetResource(newRes);

        // Copy its content to the new place.
        cmdList->CopyResource(
            pass.pMoves[i].pDstTmpAllocation->GetResource(),
            pass.pMoves[i].pSrcAllocation->GetResource());
    }
        
    // Make sure the copy commands finished executing.
    cmdQueue->ExecuteCommandLists(...);
    // ...
    WaitForSingleObject(fenceEvent, INFINITE);

    // Update appropriate descriptors to point to the new places...
        
    hr = defragCtx->EndPass(&pass);
    if(hr == S_OK)
        break;
    else if(hr != S_FALSE)
        // Handle error...
}

defragCtx->Release();
\endcode

Although functions like D3D12MA::Allocator::CreateResource()
create an allocation and a buffer/texture at once, these are just a shortcut for
allocating memory and creating a placed resource.
Defragmentation works on memory allocations only. You must handle the rest manually.
Defragmentation is an iterative process that should repreat "passes" as long as related functions
return `S_FALSE` not `S_OK`.
In each pass:

1. D3D12MA::DefragmentationContext::BeginPass() function call:
   - Calculates and returns the list of allocations to be moved in this pass.
     Note this can be a time-consuming process.
   - Reserves destination memory for them by creating temporary destination allocations
     that you can query for their `ID3D12Heap` + offset using methods like D3D12MA::Allocation::GetHeap().
2. Inside the pass, **you should**:
   - Inspect the returned list of allocations to be moved.
   - Create new buffers/textures as placed at the returned destination temporary allocations.
   - Copy data from source to destination resources if necessary.
   - Store the pointer to the new resource in the temporary destination allocation.
3. D3D12MA::DefragmentationContext::EndPass() function call:
   - Frees the source memory reserved for the allocations that are moved.
   - Modifies source D3D12MA::Allocation objects that are moved to point to the destination reserved memory
     and destination resource, while source resource is released.
   - Frees `ID3D12Heap` blocks that became empty.

Defragmentation algorithm tries to move all suitable allocations.
You can, however, refuse to move some of them inside a defragmentation pass, by setting
`pass.pMoves[i].Operation` to D3D12MA::DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
This is not recommended and may result in suboptimal packing of the allocations after defragmentation.
If you cannot ensure any allocation can be moved, it is better to keep movable allocations separate in a custom pool.

Inside a pass, for each allocation that should be moved:

- You should copy its data from the source to the destination place by calling e.g. `CopyResource()`.
  - You need to make sure these commands finished executing before the source buffers/textures are released by D3D12MA::DefragmentationContext::EndPass().
- If a resource doesn't contain any meaningful data, e.g. it is a transient render-target texture to be cleared,
  filled, and used temporarily in each rendering frame, you can just recreate this texture
  without copying its data.
- If the resource is in `D3D12_HEAP_TYPE_READBACK` memory, you can copy its data on the CPU
  using `memcpy()`.
- If you cannot move the allocation, you can set `pass.pMoves[i].Operation` to D3D12MA::DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
  This will cancel the move.
  - D3D12MA::DefragmentationContext::EndPass() will then free the destination memory
    not the source memory of the allocation, leaving it unchanged.
- If you decide the allocation is unimportant and can be destroyed instead of moved (e.g. it wasn't used for long time),
  you can set `pass.pMoves[i].Operation` to D3D12MA::DEFRAGMENTATION_MOVE_OPERATION_DESTROY.
  - D3D12MA::DefragmentationContext::EndPass() will then free both source and destination memory, and will destroy the source D3D12MA::Allocation object.

You can defragment a specific custom pool by calling D3D12MA::Pool::BeginDefragmentation
or all the default pools by calling D3D12MA::Allocator::BeginDefragmentation (like in the example above).

Defragmentation is always performed in each pool separately.
Allocations are never moved between different heap types.
The size of the destination memory reserved for a moved allocation is the same as the original one.
Alignment of an allocation as it was determined using `GetResourceAllocationInfo()` is also respected after defragmentation.
Buffers/textures should be recreated with the same `D3D12_RESOURCE_DESC` parameters as the original ones.

You can perform the defragmentation incrementally to limit the number of allocations and bytes to be moved
in each pass, e.g. to call it in sync with render frames and not to experience too big hitches.
See members: D3D12MA::DEFRAGMENTATION_DESC::MaxBytesPerPass, D3D12MA::DEFRAGMENTATION_DESC::MaxAllocationsPerPass.

<b>Thread safety:</b>
It is safe to perform the defragmentation asynchronously to render frames and other Direct3D 12 and %D3D12MA
usage, possibly from multiple threads, with the exception that allocations
returned in D3D12MA::DEFRAGMENTATION_PASS_MOVE_INFO::pMoves shouldn't be released until the defragmentation pass is ended.
During the call to D3D12MA::DefragmentationContext::BeginPass(), any operations on the memory pool
affected by the defragmentation are blocked by a mutex.

What it means in practice is that you shouldn't free any allocations from the defragmented pool
since the moment a call to `BeginPass` begins. Otherwise, a thread performing the `allocation->Release()`
would block for the time `BeginPass` executes and then free the allocation when it finishes, while the allocation
could have ended up on the list of allocations to move.
A solution to freeing allocations during defragmentation is to find such allocation on the list
`pass.pMoves[i]` and set its operation to D3D12MA::DEFRAGMENTATION_MOVE_OPERATION_DESTROY instead of
calling `allocation->Release()`, or simply deferring the release to the time after defragmentation finished.

<b>Mapping</b> is out of scope of this library and so it is not preserved after an allocation is moved during defragmentation.
You need to map the new resource yourself if needed.

\note Defragmentation is not supported in custom pools created with D3D12MA::POOL_FLAG_ALGORITHM_LINEAR.


\page statistics Statistics

This library contains several functions that return information about its internal state,
especially the amount of memory allocated from D3D12.

\section statistics_numeric_statistics Numeric statistics

If you need to obtain basic statistics about memory usage per memory segment group, together with current budget,
you can call function D3D12MA::Allocator::GetBudget() and inspect structure D3D12MA::Budget.
This is useful to keep track of memory usage and stay withing budget.
Example:

\code
D3D12MA::Budget localBudget;
allocator->GetBudget(&localBudget, NULL);

printf("My GPU memory currently has %u allocations taking %llu B,\n",
    localBudget.Statistics.AllocationCount,
    localBudget.Statistics.AllocationBytes);
printf("allocated out of %u D3D12 memory heaps taking %llu B,\n",
    localBudget.Statistics.BlockCount,
    localBudget.Statistics.BlockBytes);
printf("D3D12 reports total usage %llu B with budget %llu B.\n",
    localBudget.UsageBytes,
    localBudget.BudgetBytes);
\endcode

You can query for more detailed statistics per heap type, memory segment group, and totals,
including minimum and maximum allocation size and unused range size,
by calling function D3D12MA::Allocator::CalculateStatistics() and inspecting structure D3D12MA::TotalStatistics.
This function is slower though, as it has to traverse all the internal data structures,
so it should be used only for debugging purposes.

You can query for statistics of a custom pool using function D3D12MA::Pool::GetStatistics()
or D3D12MA::Pool::CalculateStatistics().

You can query for information about a specific allocation using functions of the D3D12MA::Allocation class,
e.g. `GetSize()`, `GetOffset()`, `GetHeap()`.

\section statistics_json_dump JSON dump

You can dump internal state of the allocator to a string in JSON format using function D3D12MA::Allocator::BuildStatsString().
The result is guaranteed to be correct JSON.
It uses Windows Unicode (UTF-16) encoding.
Any strings provided by user (see D3D12MA::Allocation::SetName())
are copied as-is and properly escaped for JSON.
It must be freed using function D3D12MA::Allocator::FreeStatsString().

The format of this JSON string is not part of official documentation of the library,
but it will not change in backward-incompatible way without increasing library major version number
and appropriate mention in changelog.

The JSON string contains all the data that can be obtained using D3D12MA::Allocator::CalculateStatistics().
It can also contain detailed map of allocated memory blocks and their regions -
free and occupied by allocations.
This allows e.g. to visualize the memory or assess fragmentation.


\page resource_aliasing Resource aliasing (overlap)

New explicit graphics APIs (Vulkan and Direct3D 12), thanks to manual memory
management, give an opportunity to alias (overlap) multiple resources in the
same region of memory - a feature not available in the old APIs (Direct3D 11, OpenGL).
It can be useful to save video memory, but it must be used with caution.

For example, if you know the flow of your whole render frame in advance, you
are going to use some intermediate textures or buffers only during a small range of render passes,
and you know these ranges don't overlap in time, you can create these resources in
the same place in memory, even if they have completely different parameters (width, height, format etc.).

![Resource aliasing (overlap)](../gfx/Aliasing.png)

Such scenario is possible using D3D12MA, but you need to create your resources
using special function D3D12MA::Allocator::CreateAliasingResource.
Before that, you need to allocate memory with parameters calculated using formula:

- allocation size = max(size of each resource)
- allocation alignment = max(alignment of each resource)

Following example shows two different textures created in the same place in memory,
allocated to fit largest of them.

\code
D3D12_RESOURCE_DESC resDesc1 = {};
resDesc1.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
resDesc1.Alignment = 0;
resDesc1.Width = 1920;
resDesc1.Height = 1080;
resDesc1.DepthOrArraySize = 1;
resDesc1.MipLevels = 1;
resDesc1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
resDesc1.SampleDesc.Count = 1;
resDesc1.SampleDesc.Quality = 0;
resDesc1.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
resDesc1.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

D3D12_RESOURCE_DESC resDesc2 = {};
resDesc2.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
resDesc2.Alignment = 0;
resDesc2.Width = 1024;
resDesc2.Height = 1024;
resDesc2.DepthOrArraySize = 1;
resDesc2.MipLevels = 0;
resDesc2.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
resDesc2.SampleDesc.Count = 1;
resDesc2.SampleDesc.Quality = 0;
resDesc2.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
resDesc2.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

const D3D12_RESOURCE_ALLOCATION_INFO allocInfo1 =
    device->GetResourceAllocationInfo(0, 1, &resDesc1);
const D3D12_RESOURCE_ALLOCATION_INFO allocInfo2 =
    device->GetResourceAllocationInfo(0, 1, &resDesc2);

D3D12_RESOURCE_ALLOCATION_INFO finalAllocInfo = {};
finalAllocInfo.Alignment = std::max(allocInfo1.Alignment, allocInfo2.Alignment);
finalAllocInfo.SizeInBytes = std::max(allocInfo1.SizeInBytes, allocInfo2.SizeInBytes);

D3D12MA::ALLOCATION_DESC allocDesc = {};
allocDesc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
allocDesc.ExtraHeapFlags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;

D3D12MA::Allocation* alloc;
hr = allocator->AllocateMemory(&allocDesc, &finalAllocInfo, &alloc);
assert(alloc != NULL && alloc->GetHeap() != NULL);

ID3D12Resource* res1;
hr = allocator->CreateAliasingResource(
    alloc,
    0, // AllocationLocalOffset
    &resDesc1,
    D3D12_RESOURCE_STATE_COMMON,
    NULL, // pOptimizedClearValue
    IID_PPV_ARGS(&res1));

ID3D12Resource* res2;
hr = allocator->CreateAliasingResource(
    alloc,
    0, // AllocationLocalOffset
    &resDesc2,
    D3D12_RESOURCE_STATE_COMMON,
    NULL, // pOptimizedClearValue
    IID_PPV_ARGS(&res2));

// You can use res1 and res2, but not at the same time!

res2->Release();
res1->Release();
alloc->Release();
\endcode

Remember that using resouces that alias in memory requires proper synchronization.
You need to issue a special barrier of type `D3D12_RESOURCE_BARRIER_TYPE_ALIASING`.
You also need to treat a resource after aliasing as uninitialized - containing garbage data.
For example, if you use `res1` and then want to use `res2`, you need to first initialize `res2`
using either Clear, Discard, or Copy to the entire resource.

Additional considerations:

- D3D12 also allows to interpret contents of memory between aliasing resources consistently in some cases,
  which is called "data inheritance". For details, see
  Microsoft documentation chapter "Memory Aliasing and Data Inheritance".
- You can create more complex layout where different textures and buffers are bound
  at different offsets inside one large allocation. For example, one can imagine
  a big texture used in some render passes, aliasing with a set of many small buffers
  used in some further passes. To bind a resource at non-zero offset of an allocation,
  call D3D12MA::Allocator::CreateAliasingResource with appropriate value of `AllocationLocalOffset` parameter.
- Resources of the three categories: buffers, textures with `RENDER_TARGET` or `DEPTH_STENCIL` flags, and all other textures,
  can be placed in the same memory only when `allocator->GetD3D12Options().ResourceHeapTier >= D3D12_RESOURCE_HEAP_TIER_2`.
  Otherwise they must be placed in different memory heap types, and thus aliasing them is not possible.


\page linear_algorithm Linear allocation algorithm

Each D3D12 memory block managed by this library has accompanying metadata that
keeps track of used and unused regions. By default, the metadata structure and
algorithm tries to find best place for new allocations among free regions to
optimize memory usage. This way you can allocate and free objects in any order.

![Default allocation algorithm](../gfx/Linear_allocator_1_algo_default.png)

Sometimes there is a need to use simpler, linear allocation algorithm. You can
create custom pool that uses such algorithm by adding flag
D3D12MA::POOL_FLAG_ALGORITHM_LINEAR to D3D12MA::POOL_DESC::Flags while creating
D3D12MA::Pool object. Then an alternative metadata management is used. It always
creates new allocations after last one and doesn't reuse free regions after
allocations freed in the middle. It results in better allocation performance and
less memory consumed by metadata.

![Linear allocation algorithm](../gfx/Linear_allocator_2_algo_linear.png)

With this one flag, you can create a custom pool that can be used in many ways:
free-at-once, stack, double stack, and ring buffer. See below for details.
You don't need to specify explicitly which of these options you are going to use - it is detected automatically.

\section linear_algorithm_free_at_once Free-at-once

In a pool that uses linear algorithm, you still need to free all the allocations
individually by calling `allocation->Release()`. You can free
them in any order. New allocations are always made after last one - free space
in the middle is not reused. However, when you release all the allocation and
the pool becomes empty, allocation starts from the beginning again. This way you
can use linear algorithm to speed up creation of allocations that you are going
to release all at once.

![Free-at-once](../gfx/Linear_allocator_3_free_at_once.png)

This mode is also available for pools created with D3D12MA::POOL_DESC::MaxBlockCount
value that allows multiple memory blocks.

\section linear_algorithm_stack Stack

When you free an allocation that was created last, its space can be reused.
Thanks to this, if you always release allocations in the order opposite to their
creation (LIFO - Last In First Out), you can achieve behavior of a stack.

![Stack](../gfx/Linear_allocator_4_stack.png)

This mode is also available for pools created with D3D12MA::POOL_DESC::MaxBlockCount
value that allows multiple memory blocks.

\section linear_algorithm_double_stack Double stack

The space reserved by a custom pool with linear algorithm may be used by two
stacks:

- First, default one, growing up from offset 0.
- Second, "upper" one, growing down from the end towards lower offsets.

To make allocation from the upper stack, add flag D3D12MA::ALLOCATION_FLAG_UPPER_ADDRESS
to D3D12MA::ALLOCATION_DESC::Flags.

![Double stack](../gfx/Linear_allocator_7_double_stack.png)

Double stack is available only in pools with one memory block -
D3D12MA::POOL_DESC::MaxBlockCount must be 1. Otherwise behavior is undefined.

When the two stacks' ends meet so there is not enough space between them for a
new allocation, such allocation fails with usual `E_OUTOFMEMORY` error.

\section linear_algorithm_ring_buffer Ring buffer

When you free some allocations from the beginning and there is not enough free space
for a new one at the end of a pool, allocator's "cursor" wraps around to the
beginning and starts allocation there. Thanks to this, if you always release
allocations in the same order as you created them (FIFO - First In First Out),
you can achieve behavior of a ring buffer / queue.

![Ring buffer](../gfx/Linear_allocator_5_ring_buffer.png)

Ring buffer is available only in pools with one memory block -
D3D12MA::POOL_DESC::MaxBlockCount must be 1. Otherwise behavior is undefined.

\section linear_algorithm_additional_considerations Additional considerations

Linear algorithm can also be used with \ref virtual_allocator.
See flag D3D12MA::VIRTUAL_BLOCK_FLAG_ALGORITHM_LINEAR.


\page virtual_allocator Virtual allocator

As an extra feature, the core allocation algorithm of the library is exposed through a simple and convenient API of "virtual allocator".
It doesn't allocate any real GPU memory. It just keeps track of used and free regions of a "virtual block".
You can use it to allocate your own memory or other objects, even completely unrelated to D3D12.
A common use case is sub-allocation of pieces of one large GPU buffer.
Another suggested use case is allocating descriptors in a `ID3D12DescriptorHeap`.

\section virtual_allocator_creating_virtual_block Creating virtual block

To use this functionality, there is no main "allocator" object.
You don't need to have D3D12MA::Allocator object created.
All you need to do is to create a separate D3D12MA::VirtualBlock object for each block of memory you want to be managed by the allocator:

-# Fill in D3D12MA::ALLOCATOR_DESC structure.
-# Call D3D12MA::CreateVirtualBlock. Get new D3D12MA::VirtualBlock object.

Example:

\code
D3D12MA::VIRTUAL_BLOCK_DESC blockDesc = {};
blockDesc.Size = 1048576; // 1 MB

D3D12MA::VirtualBlock *block;
HRESULT hr = CreateVirtualBlock(&blockDesc, &block);
\endcode

\section virtual_allocator_making_virtual_allocations Making virtual allocations

D3D12MA::VirtualBlock object contains internal data structure that keeps track of free and occupied regions
using the same code as the main D3D12 memory allocator.
A single allocation is identified by a lightweight structure D3D12MA::VirtualAllocation.
You will also likely want to know the offset at which the allocation was made in the block.

In order to make an allocation:

-# Fill in D3D12MA::VIRTUAL_ALLOCATION_DESC structure.
-# Call D3D12MA::VirtualBlock::Allocate. Get new D3D12MA::VirtualAllocation value that identifies the allocation.

Example:

\code
D3D12MA::VIRTUAL_ALLOCATION_DESC allocDesc = {};
allocDesc.Size = 4096; // 4 KB

D3D12MA::VirtualAllocation alloc;
UINT64 allocOffset;
hr = block->Allocate(&allocDesc, &alloc, &allocOffset);
if(SUCCEEDED(hr))
{
    // Use the 4 KB of your memory starting at allocOffset.
}
else
{
    // Allocation failed - no space for it could be found. Handle this error!
}
\endcode

\section virtual_allocator_deallocation Deallocation

When no longer needed, an allocation can be freed by calling D3D12MA::VirtualBlock::FreeAllocation.

When whole block is no longer needed, the block object can be released by calling `block->Release()`.
All allocations must be freed before the block is destroyed, which is checked internally by an assert.
However, if you don't want to call `block->FreeAllocation` for each allocation, you can use D3D12MA::VirtualBlock::Clear to free them all at once -
a feature not available in normal D3D12 memory allocator.

Example:

\code
block->FreeAllocation(alloc);
block->Release();
\endcode

\section virtual_allocator_allocation_parameters Allocation parameters

You can attach a custom pointer to each allocation by using D3D12MA::VirtualBlock::SetAllocationPrivateData.
Its default value is `NULL`.
It can be used to store any data that needs to be associated with that allocation - e.g. an index, a handle, or a pointer to some
larger data structure containing more information. Example:

\code
struct CustomAllocData
{
    std::string m_AllocName;
};
CustomAllocData* allocData = new CustomAllocData();
allocData->m_AllocName = "My allocation 1";
block->SetAllocationPrivateData(alloc, allocData);
\endcode

The pointer can later be fetched, along with allocation offset and size, by passing the allocation handle to function
D3D12MA::VirtualBlock::GetAllocationInfo and inspecting returned structure D3D12MA::VIRTUAL_ALLOCATION_INFO.
If you allocated a new object to be used as the custom pointer, don't forget to delete that object before freeing the allocation!
Example:

\code
VIRTUAL_ALLOCATION_INFO allocInfo;
block->GetAllocationInfo(alloc, &allocInfo);
delete (CustomAllocData*)allocInfo.pPrivateData;

block->FreeAllocation(alloc);
\endcode

\section virtual_allocator_alignment_and_units Alignment and units

It feels natural to express sizes and offsets in bytes.
If an offset of an allocation needs to be aligned to a multiply of some number (e.g. 4 bytes), you can fill optional member
D3D12MA::VIRTUAL_ALLOCATION_DESC::Alignment to request it. Example:

\code
D3D12MA::VIRTUAL_ALLOCATION_DESC allocDesc = {};
allocDesc.Size = 4096; // 4 KB
allocDesc.Alignment = 4; // Returned offset must be a multiply of 4 B

D3D12MA::VirtualAllocation alloc;
UINT64 allocOffset;
hr = block->Allocate(&allocDesc, &alloc, &allocOffset);
\endcode

Alignments of different allocations made from one block may vary.
However, if all alignments and sizes are always multiply of some size e.g. 4 B or `sizeof(MyDataStruct)`,
you can express all sizes, alignments, and offsets in multiples of that size instead of individual bytes.
It might be more convenient, but you need to make sure to use this new unit consistently in all the places:

- D3D12MA::VIRTUAL_BLOCK_DESC::Size
- D3D12MA::VIRTUAL_ALLOCATION_DESC::Size and D3D12MA::VIRTUAL_ALLOCATION_DESC::Alignment
- Using offset returned by D3D12MA::VirtualBlock::Allocate and D3D12MA::VIRTUAL_ALLOCATION_INFO::Offset

\section virtual_allocator_statistics Statistics

You can obtain brief statistics of a virtual block using D3D12MA::VirtualBlock::GetStatistics().
The function fills structure D3D12MA::Statistics - same as used by the normal D3D12 memory allocator.
Example:

\code
D3D12MA::Statistics stats;
block->GetStatistics(&stats);
printf("My virtual block has %llu bytes used by %u virtual allocations\n",
    stats.AllocationBytes, stats.AllocationCount);
\endcode

More detailed statistics can be obtained using function D3D12MA::VirtualBlock::CalculateStatistics(),
but they are slower to calculate.

You can also request a full list of allocations and free regions as a string in JSON format by calling
D3D12MA::VirtualBlock::BuildStatsString.
Returned string must be later freed using D3D12MA::VirtualBlock::FreeStatsString.
The format of this string may differ from the one returned by the main D3D12 allocator, but it is similar.

\section virtual_allocator_additional_considerations Additional considerations

Alternative, linear algorithm can be used with virtual allocator - see flag
D3D12MA::VIRTUAL_BLOCK_FLAG_ALGORITHM_LINEAR and documentation: \ref linear_algorithm.

Note that the "virtual allocator" functionality is implemented on a level of individual memory blocks.
Keeping track of a whole collection of blocks, allocating new ones when out of free space,
deleting empty ones, and deciding which one to try first for a new allocation must be implemented by the user.


\page configuration Configuration

Please check file `D3D12MemAlloc.cpp` lines between "Configuration Begin" and
"Configuration End" to find macros that you can define to change the behavior of
the library, primarily for debugging purposes.

\section custom_memory_allocator Custom CPU memory allocator

If you use custom allocator for CPU memory rather than default C++ operator `new`
and `delete` or `malloc` and `free` functions, you can make this library using
your allocator as well by filling structure D3D12MA::ALLOCATION_CALLBACKS and
passing it as optional member D3D12MA::ALLOCATOR_DESC::pAllocationCallbacks.
Functions pointed there will be used by the library to make any CPU-side
allocations. Example:

\code
#include <malloc.h>

void* CustomAllocate(size_t Size, size_t Alignment, void* pPrivateData)
{
    void* memory = _aligned_malloc(Size, Alignment);
    // Your extra bookkeeping here...
    return memory;
}

void CustomFree(void* pMemory, void* pPrivateData)
{
    // Your extra bookkeeping here...
    _aligned_free(pMemory);
}

...

D3D12MA::ALLOCATION_CALLBACKS allocationCallbacks = {};
allocationCallbacks.pAllocate = &CustomAllocate;
allocationCallbacks.pFree = &CustomFree;

D3D12MA::ALLOCATOR_DESC allocatorDesc = {};
allocatorDesc.pDevice = device;
allocatorDesc.pAdapter = adapter;
allocatorDesc.Flags = D3D12MA_RECOMMENDED_ALLOCATOR_FLAGS;
allocatorDesc.pAllocationCallbacks = &allocationCallbacks;

D3D12MA::Allocator* allocator;
HRESULT hr = D3D12MA::CreateAllocator(&allocatorDesc, &allocator);
// Check hr...
\endcode


\section debug_margins Debug margins

By default, allocations are laid out in memory blocks next to each other if possible
(considering required alignment returned by `ID3D12Device::GetResourceAllocationInfo`).

![Allocations without margin](../gfx/Margins_1.png)

Define macro `D3D12MA_DEBUG_MARGIN` to some non-zero value (e.g. 16) inside "D3D12MemAlloc.cpp"
to enforce specified number of bytes as a margin after every allocation.

![Allocations with margin](../gfx/Margins_2.png)

If your bug goes away after enabling margins, it means it may be caused by memory
being overwritten outside of allocation boundaries. It is not 100% certain though.
Change in application behavior may also be caused by different order and distribution
of allocations across memory blocks after margins are applied.

Margins work with all memory heap types.

Margin is applied only to placed allocations made out of memory heaps and not to committed
allocations, which have their own, implicit memory heap of specific size.
It is thus not applied to allocations made using D3D12MA::ALLOCATION_FLAG_COMMITTED flag
or those automatically decided to put into committed allocations, e.g. due to its large size.

Margins appear in [JSON dump](@ref statistics_json_dump) as part of free space.

Note that enabling margins increases memory usage and fragmentation.

Margins do not apply to \ref virtual_allocator.


\page general_considerations General considerations

\section general_considerations_thread_safety Thread safety

- The library has no global state, so separate D3D12MA::Allocator objects can be used independently.
  In typical applications there should be no need to create multiple such objects though - one per `ID3D12Device` is enough.
- All calls to methods of D3D12MA::Allocator class are safe to be made from multiple
  threads simultaneously because they are synchronized internally when needed.
- When the allocator is created with D3D12MA::ALLOCATOR_FLAG_SINGLETHREADED,
  calls to methods of D3D12MA::Allocator class must be made from a single thread or synchronized by the user.
  Using this flag may improve performance.
- D3D12MA::VirtualBlock is not safe to be used from multiple threads simultaneously.

\section general_considerations_versioning_and_compatibility Versioning and compatibility

The library uses [**Semantic Versioning**](https://semver.org/),
which means version numbers follow convention: Major.Minor.Patch (e.g. 2.3.0), where:

- Incremented Patch version means a release is backward- and forward-compatible,
  introducing only some internal improvements, bug fixes, optimizations etc.
  or changes that are out of scope of the official API described in this documentation.
- Incremented Minor version means a release is backward-compatible,
  so existing code that uses the library should continue to work, while some new
  symbols could have been added: new structures, functions, new values in existing
  enums and bit flags, new structure members, but not new function parameters.
- Incrementing Major version means a release could break some backward compatibility.

All changes between official releases are documented in file "CHANGELOG.md".

\warning Backward compatiblity is considered on the level of C++ source code, not binary linkage.
Adding new members to existing structures is treated as backward compatible if initializing
the new members to binary zero results in the old behavior.
You should always fully initialize all library structures to zeros and not rely on their
exact binary size.

\section general_considerations_features_not_supported Features not supported

Features deliberately excluded from the scope of this library:

- **Descriptor allocation.** Although also called "heaps", objects that represent
  descriptors are separate part of the D3D12 API from buffers and textures.
  You can still use \ref virtual_allocator to manage descriptors and their ranges inside a descriptor heap.
- **Support for reserved (tiled) resources.** We don't recommend using them. For more information, see [1].
- Support for `ID3D12Device::Evict` and `MakeResident`. We don't recommend using them.
  You can call them on the D3D12 objects manually.
  Plese keep in mind, however, that eviction happens on the level of entire `ID3D12Heap` memory blocks
  and not individual buffers or textures which may be placed inside them.
- **Handling CPU memory allocation failures.** When dynamically creating small C++
  objects in CPU memory (not the GPU memory), allocation failures are not
  handled gracefully, because that would complicate code significantly and
  is usually not needed in desktop PC applications anyway.
  Success of an allocation is just checked with an assert.
- **Code free of any compiler warnings.**
  There are many preprocessor macros that make some variables unused, function parameters unreferenced,
  or conditional expressions constant in some configurations.
  The code of this library should not be bigger or more complicated just to silence these warnings.
  It is recommended to disable such warnings instead.
- This is a C++ library. **Bindings or ports to any other programming languages** are welcome as external projects but
  are not going to be included into this repository.

[1] Antoine Richermoz, Fabrice Neyret. The Sad State of Hardware Virtual Textures. UGA - Universite Grenoble Alpes; INRIA Grenoble - Rhone-Alpes. 2025, pp.13. hal-05138369
*/

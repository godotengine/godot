//
// Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

<b>Version 2.0.1</b> (2022-04-05)

Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved. \n
License: MIT

Documentation of all members: D3D12MemAlloc.h

\section main_table_of_contents Table of contents

- \subpage quick_start
    - [Project setup](@ref quick_start_project_setup)
    - [Creating resources](@ref quick_start_creating_resources)
    - [Resource reference counting](@ref quick_start_resource_reference_counting)
    - [Mapping memory](@ref quick_start_mapping_memory)
- \subpage custom_pools
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
		
\section main_see_also See also

- [Product page on GPUOpen](https://gpuopen.com/gaming-product/d3d12-memory-allocator/)
- [Source repository on GitHub](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator)
*/

// If using this library on a platform different than Windows PC or want to use different version of DXGI,
// you should include D3D12-compatible headers before this library on your own and define this macro.
#ifndef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
    #include <d3d12.h>
    #include <dxgi1_4.h>
#endif

// Define this macro to 0 to disable usage of DXGI 1.4 (needed for IDXGIAdapter3 and query for memory budget).
#ifndef D3D12MA_DXGI_1_4
    #ifdef __IDXGIAdapter3_INTERFACE_DEFINED__
        #define D3D12MA_DXGI_1_4 1
    #else
        #define D3D12MA_DXGI_1_4 0
    #endif
#endif

/*
When defined to value other than 0, the library will try to use
D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT or D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
for created textures when possible, which can save memory because some small textures
may get their alignment 4K and their size a multiply of 4K instead of 64K.

#define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 0
    Disables small texture alignment.
#define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 1
    Enables conservative algorithm that will use small alignment only for some textures
    that are surely known to support it.
#define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 2
    Enables query for small alignment to D3D12 (based on Microsoft sample) which will
    enable small alignment for more textures, but will also generate D3D Debug Layer
    error #721 on call to ID3D12Device::GetResourceAllocationInfo, which you should just
    ignore.
*/
#ifndef D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT
    #define D3D12MA_USE_SMALL_RESOURCE_PLACEMENT_ALIGNMENT 1
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

namespace D3D12MA
{
class D3D12MA_API IUnknownImpl : public IUnknown
{
public:
    virtual ~IUnknownImpl() = default;
    virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject);
    virtual ULONG STDMETHODCALLTYPE AddRef();
    virtual ULONG STDMETHODCALLTYPE Release();
protected:
    virtual void ReleaseThis() { delete this; }
private:
    D3D12MA_ATOMIC_UINT32 m_RefCount = 1;
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

    /** Allocation strategy that chooses smallest possible free range for the allocation
    to minimize memory usage and fragmentation, possibly at the expense of allocation time.
    */
    ALLOCATION_FLAG_STRATEGY_MIN_MEMORY = 0x00010000,

    /** Allocation strategy that chooses first suitable free range for the allocation -
    not necessarily in terms of the smallest offset but the one that is easiest and fastest to find
    to minimize allocation time, possibly at the expense of allocation quality.
    */
    ALLOCATION_FLAG_STRATEGY_MIN_TIME = 0x00020000,

    /** Allocation strategy that chooses always the lowest offset in available space.
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
    /// Flags.
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

    When not NULL, the resource will be created inside specified custom pool.
    It will then never be created as committed.
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
    */
    DetailedStatistics HeapType[4];
    /** \brief One element for each memory segment group located at the following indices:

    - 0 = `DXGI_MEMORY_SEGMENT_GROUP_LOCAL`
    - 1 = `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL`

    Meaning of these segment groups is:

    - When `IsUMA() == FALSE` (discrete graphics card):
      - `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` (index 0) represents GPU memory
      (resources allocated in `D3D12_HEAP_TYPE_DEFAULT` or `D3D12_MEMORY_POOL_L1`).
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

    /// Releases the resource currently pointed by the allocation (if any), sets it to new one, incrementing its reference counter (if not null).
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

    /** \brief Returns `TRUE` if the memory of the allocation was filled with zeros when the allocation was created.

    Returns `TRUE` only if the allocator is sure that the entire memory where the
    allocation was created was filled with zeros at the moment the allocation was made.
    
    Returns `FALSE` if the memory could potentially contain garbage data.
    If it's a render-target or depth-stencil texture, it then needs proper
    initialization with `ClearRenderTargetView`, `ClearDepthStencilView`, `DiscardResource`,
    or a copy operation, as described on page
    "ID3D12Device::CreatePlacedResource method - Notes on the required resource initialization" in Microsoft documentation.
    Please note that rendering a fullscreen triangle or quad to the texture as
    a render target is not a proper way of initialization!

    See also articles:

    - "Coming to DirectX 12: More control over memory allocation" on DirectX Developer Blog
    - ["Initializing DX12 Textures After Allocation and Aliasing"](https://asawicki.info/news_1724_initializing_dx12_textures_after_allocation_and_aliasing).
    */
    BOOL WasZeroInitialized() const { return m_PackedData.WasZeroInitialized(); }

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
            m_Type(0), m_ResourceDimension(0), m_ResourceFlags(0), m_TextureLayout(0), m_WasZeroInitialized(0) { }

        Type GetType() const { return (Type)m_Type; }
        D3D12_RESOURCE_DIMENSION GetResourceDimension() const { return (D3D12_RESOURCE_DIMENSION)m_ResourceDimension; }
        D3D12_RESOURCE_FLAGS GetResourceFlags() const { return (D3D12_RESOURCE_FLAGS)m_ResourceFlags; }
        D3D12_TEXTURE_LAYOUT GetTextureLayout() const { return (D3D12_TEXTURE_LAYOUT)m_TextureLayout; }
        BOOL WasZeroInitialized() const { return (BOOL)m_WasZeroInitialized; }

        void SetType(Type type);
        void SetResourceDimension(D3D12_RESOURCE_DIMENSION resourceDimension);
        void SetResourceFlags(D3D12_RESOURCE_FLAGS resourceFlags);
        void SetTextureLayout(D3D12_TEXTURE_LAYOUT textureLayout);
        void SetWasZeroInitialized(BOOL wasZeroInitialized) { m_WasZeroInitialized = wasZeroInitialized ? 1 : 0; }

    private:
        UINT m_Type : 2;               // enum Type
        UINT m_ResourceDimension : 3;  // enum D3D12_RESOURCE_DIMENSION
        UINT m_ResourceFlags : 24;     // flags D3D12_RESOURCE_FLAGS
        UINT m_TextureLayout : 9;      // enum D3D12_TEXTURE_LAYOUT
        UINT m_WasZeroInitialized : 1; // BOOL
    } m_PackedData;

    Allocation(AllocatorPimpl* allocator, UINT64 size, UINT64 alignment, BOOL wasZeroInitialized);
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

    /** \brief Enables alternative, linear allocation algorithm in this pool.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    POOL_FLAG_ALGORITHM_LINEAR = 0x1,

    /** \brief Optimization, allocate MSAA textures as committed resources always.
    
    Specify this flag to create MSAA textures with implicit heaps, as if they were created
    with flag ALLOCATION_FLAG_COMMITTED. Usage of this flags enables pool to create its heaps
    on smaller alignment not suitable for MSAA textures.
    */
    POOL_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED = 0x2,

    // Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    POOL_FLAG_ALGORITHM_MASK = POOL_FLAG_ALGORITHM_LINEAR
};

/// \brief Parameters of created D3D12MA::Pool object. To be used with D3D12MA::Allocator::CreatePool.
struct POOL_DESC
{
    /// Flags.
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

    /**
    Every allocation will have its own memory block.
    To be used for debugging purposes.
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

    /** \brief Optimization, allocate MSAA textures as committed resources always.

    Specify this flag to create MSAA textures with implicit heaps, as if they were created
    with flag ALLOCATION_FLAG_COMMITTED. Usage of this flags enables all default pools
    to create its heaps on smaller alignment not suitable for MSAA textures.
    */
    ALLOCATOR_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED = 0x8,
};

/// \brief Parameters of created Allocator object. To be used with CreateAllocator().
struct ALLOCATOR_DESC
{
    /// Flags.
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
    /** \brief Returns total amount of memory of specific segment group, in bytes.
    
    \param memorySegmentGroup use `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` or DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL`.

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
#endif // #ifdef __ID3D12Device4_INTERFACE_DEFINED__

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

    /// Allocation strategy that tries to minimize memory usage.
    VIRTUAL_ALLOCATION_FLAG_STRATEGY_MIN_MEMORY = ALLOCATION_FLAG_STRATEGY_MIN_MEMORY,
    /// Allocation strategy that tries to minimize allocation time.
    VIRTUAL_ALLOCATION_FLAG_STRATEGY_MIN_TIME = ALLOCATION_FLAG_STRATEGY_MIN_TIME,
    /** \brief Allocation strategy that chooses always the lowest offset in available space.
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
    /// Flags.
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
\page quick_start Quick start

\section quick_start_project_setup Project setup and initialization

This is a small, standalone C++ library. It consists of a pair of 2 files:
"D3D12MemAlloc.h" header file with public interface and "D3D12MemAlloc.cpp" with
internal implementation. The only external dependencies are WinAPI, Direct3D 12,
and parts of C/C++ standard library (but STL containers, exceptions, or RTTI are
not used).

The library is developed and tested using Microsoft Visual Studio 2019, but it
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
IDXGIAdapter* adapter = (...)
ID3D12Device* device = (...)

D3D12MA::ALLOCATOR_DESC allocatorDesc = {};
allocatorDesc.pDevice = device;
allocatorDesc.pAdapter = adapter;

D3D12MA::Allocator* allocator;
HRESULT hr = D3D12MA::CreateAllocator(&allocatorDesc, &allocator);
\endcode

(5.) Right before destroying the D3D12 device, destroy the allocator object.

Objects of this library must be destroyed by calling `Release` method.
They are somewhat compatible with COM: they implement `IUnknown` interface with its virtual methods: `AddRef`, `Release`, `QueryInterface`,
and they are reference-counted internally.
You can use smart pointers designed for COM with objects of this library - e.g. `CComPtr` or `Microsoft::WRL::ComPtr`.
The reference counter is thread-safe.
`QueryInterface` method supports only `IUnknown`, as classes of this library don't define their own GUIDs.

\code
allocator->Release();
\endcode


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

// Use allocation->GetResource()...
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
or copy (`CopyResource`).

The library also automatically handles resource heap tier.
When `D3D12_FEATURE_DATA_D3D12_OPTIONS::ResourceHeapTier` equals `D3D12_RESOURCE_HEAP_TIER_1`,
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
Calling `allocation->Release()` will decrease the resource reference counter, which is = 1 in this case,
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
const float* bufData = (...);

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

void* mappedPtr;
hr = resource->Map(0, NULL, &mappedPtr);

memcpy(mappedPtr, bufData, bufSize);

resource->Unmap(0, NULL);
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

It is also safe to perform the defragmentation asynchronously to render frames and other Direct3D 12 and %D3D12MA
usage, possibly from multiple threads, with the exception that allocations
returned in D3D12MA::DEFRAGMENTATION_PASS_MOVE_INFO::pMoves shouldn't be released until the defragmentation pass is ended.

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

(...)

D3D12MA::ALLOCATION_CALLBACKS allocationCallbacks = {};
allocationCallbacks.pAllocate = &CustomAllocate;
allocationCallbacks.pFree = &CustomFree;

D3D12MA::ALLOCATOR_DESC allocatorDesc = {};
allocatorDesc.pDevice = device;
allocatorDesc.pAdapter = adapter;
allocatorDesc.pAllocationCallbacks = &allocationCallbacks;

D3D12MA::Allocator* allocator;
HRESULT hr = D3D12MA::CreateAllocator(&allocatorDesc, &allocator);
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
- **Support for reserved (tiled) resources.** We don't recommend using them.
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
*/

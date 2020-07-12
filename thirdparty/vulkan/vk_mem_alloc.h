//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AMD_VULKAN_MEMORY_ALLOCATOR_H
#define AMD_VULKAN_MEMORY_ALLOCATOR_H

#ifdef __cplusplus
extern "C" {
#endif

/** \mainpage Vulkan Memory Allocator

<b>Version 2.3.0</b> (2019-12-04)

Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved. \n
License: MIT

Documentation of all members: vk_mem_alloc.h

\section main_table_of_contents Table of contents

- <b>User guide</b>
  - \subpage quick_start
    - [Project setup](@ref quick_start_project_setup)
    - [Initialization](@ref quick_start_initialization)
    - [Resource allocation](@ref quick_start_resource_allocation)
  - \subpage choosing_memory_type
    - [Usage](@ref choosing_memory_type_usage)
    - [Required and preferred flags](@ref choosing_memory_type_required_preferred_flags)
    - [Explicit memory types](@ref choosing_memory_type_explicit_memory_types)
    - [Custom memory pools](@ref choosing_memory_type_custom_memory_pools)
    - [Dedicated allocations](@ref choosing_memory_type_dedicated_allocations)
  - \subpage memory_mapping
    - [Mapping functions](@ref memory_mapping_mapping_functions)
    - [Persistently mapped memory](@ref memory_mapping_persistently_mapped_memory)
    - [Cache flush and invalidate](@ref memory_mapping_cache_control)
    - [Finding out if memory is mappable](@ref memory_mapping_finding_if_memory_mappable)
  - \subpage staying_within_budget
    - [Querying for budget](@ref staying_within_budget_querying_for_budget)
    - [Controlling memory usage](@ref staying_within_budget_controlling_memory_usage)
  - \subpage custom_memory_pools
    - [Choosing memory type index](@ref custom_memory_pools_MemTypeIndex)
    - [Linear allocation algorithm](@ref linear_algorithm)
      - [Free-at-once](@ref linear_algorithm_free_at_once)
      - [Stack](@ref linear_algorithm_stack)
      - [Double stack](@ref linear_algorithm_double_stack)
      - [Ring buffer](@ref linear_algorithm_ring_buffer)
    - [Buddy allocation algorithm](@ref buddy_algorithm)
  - \subpage defragmentation
  	- [Defragmenting CPU memory](@ref defragmentation_cpu)
  	- [Defragmenting GPU memory](@ref defragmentation_gpu)
  	- [Additional notes](@ref defragmentation_additional_notes)
  	- [Writing custom allocation algorithm](@ref defragmentation_custom_algorithm)
  - \subpage lost_allocations
  - \subpage statistics
    - [Numeric statistics](@ref statistics_numeric_statistics)
    - [JSON dump](@ref statistics_json_dump)
  - \subpage allocation_annotation
    - [Allocation user data](@ref allocation_user_data)
    - [Allocation names](@ref allocation_names)
  - \subpage debugging_memory_usage
    - [Memory initialization](@ref debugging_memory_usage_initialization)
    - [Margins](@ref debugging_memory_usage_margins)
    - [Corruption detection](@ref debugging_memory_usage_corruption_detection)
  - \subpage record_and_replay
- \subpage usage_patterns
  - [Common mistakes](@ref usage_patterns_common_mistakes)
  - [Simple patterns](@ref usage_patterns_simple)
  - [Advanced patterns](@ref usage_patterns_advanced)
- \subpage configuration
  - [Pointers to Vulkan functions](@ref config_Vulkan_functions)
  - [Custom host memory allocator](@ref custom_memory_allocator)
  - [Device memory allocation callbacks](@ref allocation_callbacks)
  - [Device heap memory limit](@ref heap_memory_limit)
  - \subpage vk_khr_dedicated_allocation
- \subpage general_considerations
  - [Thread safety](@ref general_considerations_thread_safety)
  - [Validation layer warnings](@ref general_considerations_validation_layer_warnings)
  - [Allocation algorithm](@ref general_considerations_allocation_algorithm)
  - [Features not supported](@ref general_considerations_features_not_supported)

\section main_see_also See also

- [Product page on GPUOpen](https://gpuopen.com/gaming-product/vulkan-memory-allocator/)
- [Source repository on GitHub](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)




\page quick_start Quick start

\section quick_start_project_setup Project setup

Vulkan Memory Allocator comes in form of a "stb-style" single header file.
You don't need to build it as a separate library project.
You can add this file directly to your project and submit it to code repository next to your other source files.

"Single header" doesn't mean that everything is contained in C/C++ declarations,
like it tends to be in case of inline functions or C++ templates.
It means that implementation is bundled with interface in a single file and needs to be extracted using preprocessor macro.
If you don't do it properly, you will get linker errors.

To do it properly:

-# Include "vk_mem_alloc.h" file in each CPP file where you want to use the library.
   This includes declarations of all members of the library.
-# In exacly one CPP file define following macro before this include.
   It enables also internal definitions.

\code
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
\endcode

It may be a good idea to create dedicated CPP file just for this purpose.

Note on language: This library is written in C++, but has C-compatible interface.
Thus you can include and use vk_mem_alloc.h in C or C++ code, but full
implementation with `VMA_IMPLEMENTATION` macro must be compiled as C++, NOT as C.

Please note that this library includes header `<vulkan/vulkan.h>`, which in turn
includes `<windows.h>` on Windows. If you need some specific macros defined
before including these headers (like `WIN32_LEAN_AND_MEAN` or
`WINVER` for Windows, `VK_USE_PLATFORM_WIN32_KHR` for Vulkan), you must define
them before every `#include` of this library.


\section quick_start_initialization Initialization

At program startup:

-# Initialize Vulkan to have `VkPhysicalDevice` and `VkDevice` object.
-# Fill VmaAllocatorCreateInfo structure and create #VmaAllocator object by
   calling vmaCreateAllocator().

\code
VmaAllocatorCreateInfo allocatorInfo = {};
allocatorInfo.physicalDevice = physicalDevice;
allocatorInfo.device = device;

VmaAllocator allocator;
vmaCreateAllocator(&allocatorInfo, &allocator);
\endcode

\section quick_start_resource_allocation Resource allocation

When you want to create a buffer or image:

-# Fill `VkBufferCreateInfo` / `VkImageCreateInfo` structure.
-# Fill VmaAllocationCreateInfo structure.
-# Call vmaCreateBuffer() / vmaCreateImage() to get `VkBuffer`/`VkImage` with memory
   already allocated and bound to it.

\code
VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufferInfo.size = 65536;
bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

Don't forget to destroy your objects when no longer needed:

\code
vmaDestroyBuffer(allocator, buffer, allocation);
vmaDestroyAllocator(allocator);
\endcode


\page choosing_memory_type Choosing memory type

Physical devices in Vulkan support various combinations of memory heaps and
types. Help with choosing correct and optimal memory type for your specific
resource is one of the key features of this library. You can use it by filling
appropriate members of VmaAllocationCreateInfo structure, as described below.
You can also combine multiple methods.

-# If you just want to find memory type index that meets your requirements, you
   can use function: vmaFindMemoryTypeIndex(), vmaFindMemoryTypeIndexForBufferInfo(),
   vmaFindMemoryTypeIndexForImageInfo().
-# If you want to allocate a region of device memory without association with any
   specific image or buffer, you can use function vmaAllocateMemory(). Usage of
   this function is not recommended and usually not needed.
   vmaAllocateMemoryPages() function is also provided for creating multiple allocations at once,
   which may be useful for sparse binding.
-# If you already have a buffer or an image created, you want to allocate memory
   for it and then you will bind it yourself, you can use function
   vmaAllocateMemoryForBuffer(), vmaAllocateMemoryForImage().
   For binding you should use functions: vmaBindBufferMemory(), vmaBindImageMemory()
   or their extended versions: vmaBindBufferMemory2(), vmaBindImageMemory2().
-# If you want to create a buffer or an image, allocate memory for it and bind
   them together, all in one call, you can use function vmaCreateBuffer(),
   vmaCreateImage(). This is the easiest and recommended way to use this library.

When using 3. or 4., the library internally queries Vulkan for memory types
supported for that buffer or image (function `vkGetBufferMemoryRequirements()`)
and uses only one of these types.

If no memory type can be found that meets all the requirements, these functions
return `VK_ERROR_FEATURE_NOT_PRESENT`.

You can leave VmaAllocationCreateInfo structure completely filled with zeros.
It means no requirements are specified for memory type.
It is valid, although not very useful.

\section choosing_memory_type_usage Usage

The easiest way to specify memory requirements is to fill member
VmaAllocationCreateInfo::usage using one of the values of enum #VmaMemoryUsage.
It defines high level, common usage types.
For more details, see description of this enum.

For example, if you want to create a uniform buffer that will be filled using
transfer only once or infrequently and used for rendering every frame, you can
do it using following code:

\code
VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufferInfo.size = 65536;
bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

\section choosing_memory_type_required_preferred_flags Required and preferred flags

You can specify more detailed requirements by filling members
VmaAllocationCreateInfo::requiredFlags and VmaAllocationCreateInfo::preferredFlags
with a combination of bits from enum `VkMemoryPropertyFlags`. For example,
if you want to create a buffer that will be persistently mapped on host (so it
must be `HOST_VISIBLE`) and preferably will also be `HOST_COHERENT` and `HOST_CACHED`,
use following code:

\code
VmaAllocationCreateInfo allocInfo = {};
allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
allocInfo.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

A memory type is chosen that has all the required flags and as many preferred
flags set as possible.

If you use VmaAllocationCreateInfo::usage, it is just internally converted to
a set of required and preferred flags.

\section choosing_memory_type_explicit_memory_types Explicit memory types

If you inspected memory types available on the physical device and you have
a preference for memory types that you want to use, you can fill member
VmaAllocationCreateInfo::memoryTypeBits. It is a bit mask, where each bit set
means that a memory type with that index is allowed to be used for the
allocation. Special value 0, just like `UINT32_MAX`, means there are no
restrictions to memory type index.

Please note that this member is NOT just a memory type index.
Still you can use it to choose just one, specific memory type.
For example, if you already determined that your buffer should be created in
memory type 2, use following code:

\code
uint32_t memoryTypeIndex = 2;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.memoryTypeBits = 1u << memoryTypeIndex;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

\section choosing_memory_type_custom_memory_pools Custom memory pools

If you allocate from custom memory pool, all the ways of specifying memory
requirements described above are not applicable and the aforementioned members
of VmaAllocationCreateInfo structure are ignored. Memory type is selected
explicitly when creating the pool and then used to make all the allocations from
that pool. For further details, see \ref custom_memory_pools.

\section choosing_memory_type_dedicated_allocations Dedicated allocations

Memory for allocations is reserved out of larger block of `VkDeviceMemory`
allocated from Vulkan internally. That's the main feature of this whole library.
You can still request a separate memory block to be created for an allocation,
just like you would do in a trivial solution without using any allocator.
In that case, a buffer or image is always bound to that memory at offset 0.
This is called a "dedicated allocation".
You can explicitly request it by using flag #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
The library can also internally decide to use dedicated allocation in some cases, e.g.:

- When the size of the allocation is large.
- When [VK_KHR_dedicated_allocation](@ref vk_khr_dedicated_allocation) extension is enabled
  and it reports that dedicated allocation is required or recommended for the resource.
- When allocation of next big memory block fails due to not enough device memory,
  but allocation with the exact requested size succeeds.


\page memory_mapping Memory mapping

To "map memory" in Vulkan means to obtain a CPU pointer to `VkDeviceMemory`,
to be able to read from it or write to it in CPU code.
Mapping is possible only of memory allocated from a memory type that has
`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` flag.
Functions `vkMapMemory()`, `vkUnmapMemory()` are designed for this purpose.
You can use them directly with memory allocated by this library,
but it is not recommended because of following issue:
Mapping the same `VkDeviceMemory` block multiple times is illegal - only one mapping at a time is allowed.
This includes mapping disjoint regions. Mapping is not reference-counted internally by Vulkan.
Because of this, Vulkan Memory Allocator provides following facilities:

\section memory_mapping_mapping_functions Mapping functions

The library provides following functions for mapping of a specific #VmaAllocation: vmaMapMemory(), vmaUnmapMemory().
They are safer and more convenient to use than standard Vulkan functions.
You can map an allocation multiple times simultaneously - mapping is reference-counted internally.
You can also map different allocations simultaneously regardless of whether they use the same `VkDeviceMemory` block.
The way it's implemented is that the library always maps entire memory block, not just region of the allocation.
For further details, see description of vmaMapMemory() function.
Example:

\code
// Having these objects initialized:

struct ConstantBuffer
{
    ...
};
ConstantBuffer constantBufferData;

VmaAllocator allocator;
VkBuffer constantBuffer;
VmaAllocation constantBufferAllocation;

// You can map and fill your buffer using following code:

void* mappedData;
vmaMapMemory(allocator, constantBufferAllocation, &mappedData);
memcpy(mappedData, &constantBufferData, sizeof(constantBufferData));
vmaUnmapMemory(allocator, constantBufferAllocation);
\endcode

When mapping, you may see a warning from Vulkan validation layer similar to this one:

<i>Mapping an image with layout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL can result in undefined behavior if this memory is used by the device. Only GENERAL or PREINITIALIZED should be used.</i>

It happens because the library maps entire `VkDeviceMemory` block, where different
types of images and buffers may end up together, especially on GPUs with unified memory like Intel.
You can safely ignore it if you are sure you access only memory of the intended
object that you wanted to map.


\section memory_mapping_persistently_mapped_memory Persistently mapped memory

Kepping your memory persistently mapped is generally OK in Vulkan.
You don't need to unmap it before using its data on the GPU.
The library provides a special feature designed for that:
Allocations made with #VMA_ALLOCATION_CREATE_MAPPED_BIT flag set in
VmaAllocationCreateInfo::flags stay mapped all the time,
so you can just access CPU pointer to it any time
without a need to call any "map" or "unmap" function.
Example:

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = sizeof(ConstantBuffer);
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

// Buffer is already mapped. You can access its memory.
memcpy(allocInfo.pMappedData, &constantBufferData, sizeof(constantBufferData));
\endcode

There are some exceptions though, when you should consider mapping memory only for a short period of time:

- When operating system is Windows 7 or 8.x (Windows 10 is not affected because it uses WDDM2),
  device is discrete AMD GPU,
  and memory type is the special 256 MiB pool of `DEVICE_LOCAL + HOST_VISIBLE` memory
  (selected when you use #VMA_MEMORY_USAGE_CPU_TO_GPU),
  then whenever a memory block allocated from this memory type stays mapped
  for the time of any call to `vkQueueSubmit()` or `vkQueuePresentKHR()`, this
  block is migrated by WDDM to system RAM, which degrades performance. It doesn't
  matter if that particular memory block is actually used by the command buffer
  being submitted.
- On Mac/MoltenVK there is a known bug - [Issue #175](https://github.com/KhronosGroup/MoltenVK/issues/175)
  which requires unmapping before GPU can see updated texture.
- Keeping many large memory blocks mapped may impact performance or stability of some debugging tools.

\section memory_mapping_cache_control Cache flush and invalidate
  
Memory in Vulkan doesn't need to be unmapped before using it on GPU,
but unless a memory types has `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` flag set,
you need to manually **invalidate** cache before reading of mapped pointer
and **flush** cache after writing to mapped pointer.
Map/unmap operations don't do that automatically.
Vulkan provides following functions for this purpose `vkFlushMappedMemoryRanges()`,
`vkInvalidateMappedMemoryRanges()`, but this library provides more convenient
functions that refer to given allocation object: vmaFlushAllocation(),
vmaInvalidateAllocation().

Regions of memory specified for flush/invalidate must be aligned to
`VkPhysicalDeviceLimits::nonCoherentAtomSize`. This is automatically ensured by the library.
In any memory type that is `HOST_VISIBLE` but not `HOST_COHERENT`, all allocations
within blocks are aligned to this value, so their offsets are always multiply of
`nonCoherentAtomSize` and two different allocations never share same "line" of this size.

Please note that memory allocated with #VMA_MEMORY_USAGE_CPU_ONLY is guaranteed to be `HOST_COHERENT`.

Also, Windows drivers from all 3 **PC** GPU vendors (AMD, Intel, NVIDIA)
currently provide `HOST_COHERENT` flag on all memory types that are
`HOST_VISIBLE`, so on this platform you may not need to bother.

\section memory_mapping_finding_if_memory_mappable Finding out if memory is mappable

It may happen that your allocation ends up in memory that is `HOST_VISIBLE` (available for mapping)
despite it wasn't explicitly requested.
For example, application may work on integrated graphics with unified memory (like Intel) or
allocation from video memory might have failed, so the library chose system memory as fallback.

You can detect this case and map such allocation to access its memory on CPU directly,
instead of launching a transfer operation.
In order to do that: inspect `allocInfo.memoryType`, call vmaGetMemoryTypeProperties(),
and look for `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` flag in properties of that memory type.

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = sizeof(ConstantBuffer);
bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
allocCreateInfo.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

VkMemoryPropertyFlags memFlags;
vmaGetMemoryTypeProperties(allocator, allocInfo.memoryType, &memFlags);
if((memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
{
    // Allocation ended up in mappable memory. You can map it and access it directly.
    void* mappedData;
    vmaMapMemory(allocator, alloc, &mappedData);
    memcpy(mappedData, &constantBufferData, sizeof(constantBufferData));
    vmaUnmapMemory(allocator, alloc);
}
else
{
    // Allocation ended up in non-mappable memory.
    // You need to create CPU-side buffer in VMA_MEMORY_USAGE_CPU_ONLY and make a transfer.
}
\endcode

You can even use #VMA_ALLOCATION_CREATE_MAPPED_BIT flag while creating allocations
that are not necessarily `HOST_VISIBLE` (e.g. using #VMA_MEMORY_USAGE_GPU_ONLY).
If the allocation ends up in memory type that is `HOST_VISIBLE`, it will be persistently mapped and you can use it directly.
If not, the flag is just ignored.
Example:

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = sizeof(ConstantBuffer);
bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

if(allocInfo.pUserData != nullptr)
{
    // Allocation ended up in mappable memory.
    // It's persistently mapped. You can access it directly.
    memcpy(allocInfo.pMappedData, &constantBufferData, sizeof(constantBufferData));
}
else
{
    // Allocation ended up in non-mappable memory.
    // You need to create CPU-side buffer in VMA_MEMORY_USAGE_CPU_ONLY and make a transfer.
}
\endcode


\page staying_within_budget Staying within budget

When developing a graphics-intensive game or program, it is important to avoid allocating
more GPU memory than it's physically available. When the memory is over-committed,
various bad things can happen, depending on the specific GPU, graphics driver, and
operating system:

- It may just work without any problems.
- The application may slow down because some memory blocks are moved to system RAM
  and the GPU has to access them through PCI Express bus.
- A new allocation may take very long time to complete, even few seconds, and possibly
  freeze entire system.
- The new allocation may fail with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
- It may even result in GPU crash (TDR), observed as `VK_ERROR_DEVICE_LOST`
  returned somewhere later.

\section staying_within_budget_querying_for_budget Querying for budget

To query for current memory usage and available budget, use function vmaGetBudget().
Returned structure #VmaBudget contains quantities expressed in bytes, per Vulkan memory heap.

Please note that this function returns different information and works faster than
vmaCalculateStats(). vmaGetBudget() can be called every frame or even before every
allocation, while vmaCalculateStats() is intended to be used rarely,
only to obtain statistical information, e.g. for debugging purposes.

It is recommended to use <b>VK_EXT_memory_budget</b> device extension to obtain information
about the budget from Vulkan device. VMA is able to use this extension automatically.
When not enabled, the allocator behaves same way, but then it estimates current usage
and available budget based on its internal information and Vulkan memory heap sizes,
which may be less precise. In order to use this extension:

1. Make sure extensions VK_EXT_memory_budget and VK_KHR_get_physical_device_properties2
   required by it are available and enable them. Please note that the first is a device
   extension and the second is instance extension!
2. Use flag #VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT when creating #VmaAllocator object.
3. Make sure to call vmaSetCurrentFrameIndex() every frame. Budget is queried from
   Vulkan inside of it to avoid overhead of querying it with every allocation.

\section staying_within_budget_controlling_memory_usage Controlling memory usage

There are many ways in which you can try to stay within the budget.

First, when making new allocation requires allocating a new memory block, the library
tries not to exceed the budget automatically. If a block with default recommended size
(e.g. 256 MB) would go over budget, a smaller block is allocated, possibly even
dedicated memory for just this resource.

If the size of the requested resource plus current memory usage is more than the
budget, by default the library still tries to create it, leaving it to the Vulkan
implementation whether the allocation succeeds or fails. You can change this behavior
by using #VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT flag. With it, the allocation is
not made if it would exceed the budget or if the budget is already exceeded.
Some other allocations become lost instead to make room for it, if the mechanism of
[lost allocations](@ref lost_allocations) is used.
If that is not possible, the allocation fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
Example usage pattern may be to pass the #VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT flag
when creating resources that are not essential for the application (e.g. the texture
of a specific object) and not to pass it when creating critically important resources
(e.g. render targets).

Finally, you can also use #VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT flag to make sure
a new allocation is created only when it fits inside one of the existing memory blocks.
If it would require to allocate a new block, if fails instead with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
This also ensures that the function call is very fast because it never goes to Vulkan
to obtain a new block.

Please note that creating \ref custom_memory_pools with VmaPoolCreateInfo::minBlockCount
set to more than 0 will try to allocate memory blocks without checking whether they
fit within budget.


\page custom_memory_pools Custom memory pools

A memory pool contains a number of `VkDeviceMemory` blocks.
The library automatically creates and manages default pool for each memory type available on the device.
Default memory pool automatically grows in size.
Size of allocated blocks is also variable and managed automatically.

You can create custom pool and allocate memory out of it.
It can be useful if you want to:

- Keep certain kind of allocations separate from others.
- Enforce particular, fixed size of Vulkan memory blocks.
- Limit maximum amount of Vulkan memory allocated for that pool.
- Reserve minimum or fixed amount of Vulkan memory always preallocated for that pool.

To use custom memory pools:

-# Fill VmaPoolCreateInfo structure.
-# Call vmaCreatePool() to obtain #VmaPool handle.
-# When making an allocation, set VmaAllocationCreateInfo::pool to this handle.
   You don't need to specify any other parameters of this structure, like `usage`.

Example:

\code
// Create a pool that can have at most 2 blocks, 128 MiB each.
VmaPoolCreateInfo poolCreateInfo = {};
poolCreateInfo.memoryTypeIndex = ...
poolCreateInfo.blockSize = 128ull * 1024 * 1024;
poolCreateInfo.maxBlockCount = 2;

VmaPool pool;
vmaCreatePool(allocator, &poolCreateInfo, &pool);

// Allocate a buffer out of it.
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 1024;
bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.pool = pool;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);
\endcode

You have to free all allocations made from this pool before destroying it.

\code
vmaDestroyBuffer(allocator, buf, alloc);
vmaDestroyPool(allocator, pool);
\endcode

\section custom_memory_pools_MemTypeIndex Choosing memory type index

When creating a pool, you must explicitly specify memory type index.
To find the one suitable for your buffers or images, you can use helper functions
vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo().
You need to provide structures with example parameters of buffers or images
that you are going to create in that pool.

\code
VkBufferCreateInfo exampleBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
exampleBufCreateInfo.size = 1024; // Whatever.
exampleBufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // Change if needed.

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; // Change if needed.

uint32_t memTypeIndex;
vmaFindMemoryTypeIndexForBufferInfo(allocator, &exampleBufCreateInfo, &allocCreateInfo, &memTypeIndex);

VmaPoolCreateInfo poolCreateInfo = {};
poolCreateInfo.memoryTypeIndex = memTypeIndex;
// ...
\endcode

When creating buffers/images allocated in that pool, provide following parameters:

- `VkBufferCreateInfo`: Prefer to pass same parameters as above.
  Otherwise you risk creating resources in a memory type that is not suitable for them, which may result in undefined behavior.
  Using different `VK_BUFFER_USAGE_` flags may work, but you shouldn't create images in a pool intended for buffers
  or the other way around.
- VmaAllocationCreateInfo: You don't need to pass same parameters. Fill only `pool` member.
  Other members are ignored anyway.

\section linear_algorithm Linear allocation algorithm

Each Vulkan memory block managed by this library has accompanying metadata that
keeps track of used and unused regions. By default, the metadata structure and
algorithm tries to find best place for new allocations among free regions to
optimize memory usage. This way you can allocate and free objects in any order.

![Default allocation algorithm](../gfx/Linear_allocator_1_algo_default.png)

Sometimes there is a need to use simpler, linear allocation algorithm. You can
create custom pool that uses such algorithm by adding flag
#VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT to VmaPoolCreateInfo::flags while creating
#VmaPool object. Then an alternative metadata management is used. It always
creates new allocations after last one and doesn't reuse free regions after
allocations freed in the middle. It results in better allocation performance and
less memory consumed by metadata.

![Linear allocation algorithm](../gfx/Linear_allocator_2_algo_linear.png)

With this one flag, you can create a custom pool that can be used in many ways:
free-at-once, stack, double stack, and ring buffer. See below for details.

\subsection linear_algorithm_free_at_once Free-at-once

In a pool that uses linear algorithm, you still need to free all the allocations
individually, e.g. by using vmaFreeMemory() or vmaDestroyBuffer(). You can free
them in any order. New allocations are always made after last one - free space
in the middle is not reused. However, when you release all the allocation and
the pool becomes empty, allocation starts from the beginning again. This way you
can use linear algorithm to speed up creation of allocations that you are going
to release all at once.

![Free-at-once](../gfx/Linear_allocator_3_free_at_once.png)

This mode is also available for pools created with VmaPoolCreateInfo::maxBlockCount
value that allows multiple memory blocks.

\subsection linear_algorithm_stack Stack

When you free an allocation that was created last, its space can be reused.
Thanks to this, if you always release allocations in the order opposite to their
creation (LIFO - Last In First Out), you can achieve behavior of a stack.

![Stack](../gfx/Linear_allocator_4_stack.png)

This mode is also available for pools created with VmaPoolCreateInfo::maxBlockCount
value that allows multiple memory blocks.

\subsection linear_algorithm_double_stack Double stack

The space reserved by a custom pool with linear algorithm may be used by two
stacks:

- First, default one, growing up from offset 0.
- Second, "upper" one, growing down from the end towards lower offsets.

To make allocation from upper stack, add flag #VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT
to VmaAllocationCreateInfo::flags.

![Double stack](../gfx/Linear_allocator_7_double_stack.png)

Double stack is available only in pools with one memory block -
VmaPoolCreateInfo::maxBlockCount must be 1. Otherwise behavior is undefined.

When the two stacks' ends meet so there is not enough space between them for a
new allocation, such allocation fails with usual
`VK_ERROR_OUT_OF_DEVICE_MEMORY` error.

\subsection linear_algorithm_ring_buffer Ring buffer

When you free some allocations from the beginning and there is not enough free space
for a new one at the end of a pool, allocator's "cursor" wraps around to the
beginning and starts allocation there. Thanks to this, if you always release
allocations in the same order as you created them (FIFO - First In First Out),
you can achieve behavior of a ring buffer / queue.

![Ring buffer](../gfx/Linear_allocator_5_ring_buffer.png)

Pools with linear algorithm support [lost allocations](@ref lost_allocations) when used as ring buffer.
If there is not enough free space for a new allocation, but existing allocations
from the front of the queue can become lost, they become lost and the allocation
succeeds.

![Ring buffer with lost allocations](../gfx/Linear_allocator_6_ring_buffer_lost.png)

Ring buffer is available only in pools with one memory block -
VmaPoolCreateInfo::maxBlockCount must be 1. Otherwise behavior is undefined.

\section buddy_algorithm Buddy allocation algorithm

There is another allocation algorithm that can be used with custom pools, called
"buddy". Its internal data structure is based on a tree of blocks, each having
size that is a power of two and a half of its parent's size. When you want to
allocate memory of certain size, a free node in the tree is located. If it's too
large, it is recursively split into two halves (called "buddies"). However, if
requested allocation size is not a power of two, the size of a tree node is
aligned up to the nearest power of two and the remaining space is wasted. When
two buddy nodes become free, they are merged back into one larger node.

![Buddy allocator](../gfx/Buddy_allocator.png)

The advantage of buddy allocation algorithm over default algorithm is faster
allocation and deallocation, as well as smaller external fragmentation. The
disadvantage is more wasted space (internal fragmentation).

For more information, please read ["Buddy memory allocation" on Wikipedia](https://en.wikipedia.org/wiki/Buddy_memory_allocation)
or other sources that describe this concept in general.

To use buddy allocation algorithm with a custom pool, add flag
#VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT to VmaPoolCreateInfo::flags while creating
#VmaPool object.

Several limitations apply to pools that use buddy algorithm:

- It is recommended to use VmaPoolCreateInfo::blockSize that is a power of two.
  Otherwise, only largest power of two smaller than the size is used for
  allocations. The remaining space always stays unused.
- [Margins](@ref debugging_memory_usage_margins) and
  [corruption detection](@ref debugging_memory_usage_corruption_detection)
  don't work in such pools.
- [Lost allocations](@ref lost_allocations) don't work in such pools. You can
  use them, but they never become lost. Support may be added in the future.
- [Defragmentation](@ref defragmentation) doesn't work with allocations made from
  such pool.

\page defragmentation Defragmentation

Interleaved allocations and deallocations of many objects of varying size can
cause fragmentation over time, which can lead to a situation where the library is unable
to find a continuous range of free memory for a new allocation despite there is
enough free space, just scattered across many small free ranges between existing
allocations.

To mitigate this problem, you can use defragmentation feature:
structure #VmaDefragmentationInfo2, function vmaDefragmentationBegin(), vmaDefragmentationEnd().
Given set of allocations, 
this function can move them to compact used memory, ensure more continuous free
space and possibly also free some `VkDeviceMemory` blocks.

What the defragmentation does is:

- Updates #VmaAllocation objects to point to new `VkDeviceMemory` and offset.
  After allocation has been moved, its VmaAllocationInfo::deviceMemory and/or
  VmaAllocationInfo::offset changes. You must query them again using
  vmaGetAllocationInfo() if you need them.
- Moves actual data in memory.

What it doesn't do, so you need to do it yourself:

- Recreate buffers and images that were bound to allocations that were defragmented and
  bind them with their new places in memory.
  You must use `vkDestroyBuffer()`, `vkDestroyImage()`,
  `vkCreateBuffer()`, `vkCreateImage()`, vmaBindBufferMemory(), vmaBindImageMemory()
  for that purpose and NOT vmaDestroyBuffer(),
  vmaDestroyImage(), vmaCreateBuffer(), vmaCreateImage(), because you don't need to
  destroy or create allocation objects!
- Recreate views and update descriptors that point to these buffers and images.

\section defragmentation_cpu Defragmenting CPU memory

Following example demonstrates how you can run defragmentation on CPU.
Only allocations created in memory types that are `HOST_VISIBLE` can be defragmented.
Others are ignored.

The way it works is:

- It temporarily maps entire memory blocks when necessary.
- It moves data using `memmove()` function.

\code
// Given following variables already initialized:
VkDevice device;
VmaAllocator allocator;
std::vector<VkBuffer> buffers;
std::vector<VmaAllocation> allocations;


const uint32_t allocCount = (uint32_t)allocations.size();
std::vector<VkBool32> allocationsChanged(allocCount);

VmaDefragmentationInfo2 defragInfo = {};
defragInfo.allocationCount = allocCount;
defragInfo.pAllocations = allocations.data();
defragInfo.pAllocationsChanged = allocationsChanged.data();
defragInfo.maxCpuBytesToMove = VK_WHOLE_SIZE; // No limit.
defragInfo.maxCpuAllocationsToMove = UINT32_MAX; // No limit.

VmaDefragmentationContext defragCtx;
vmaDefragmentationBegin(allocator, &defragInfo, nullptr, &defragCtx);
vmaDefragmentationEnd(allocator, defragCtx);

for(uint32_t i = 0; i < allocCount; ++i)
{
    if(allocationsChanged[i])
    {
        // Destroy buffer that is immutably bound to memory region which is no longer valid.
        vkDestroyBuffer(device, buffers[i], nullptr);

        // Create new buffer with same parameters.
        VkBufferCreateInfo bufferInfo = ...;
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffers[i]);
            
        // You can make dummy call to vkGetBufferMemoryRequirements here to silence validation layer warning.
            
        // Bind new buffer to new memory region. Data contained in it is already moved.
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(allocator, allocations[i], &allocInfo);
        vmaBindBufferMemory(allocator, allocations[i], buffers[i]);
    }
}
\endcode

Setting VmaDefragmentationInfo2::pAllocationsChanged is optional.
This output array tells whether particular allocation in VmaDefragmentationInfo2::pAllocations at the same index
has been modified during defragmentation.
You can pass null, but you then need to query every allocation passed to defragmentation
for new parameters using vmaGetAllocationInfo() if you might need to recreate and rebind a buffer or image associated with it.

If you use [Custom memory pools](@ref choosing_memory_type_custom_memory_pools),
you can fill VmaDefragmentationInfo2::poolCount and VmaDefragmentationInfo2::pPools
instead of VmaDefragmentationInfo2::allocationCount and VmaDefragmentationInfo2::pAllocations
to defragment all allocations in given pools.
You cannot use VmaDefragmentationInfo2::pAllocationsChanged in that case.
You can also combine both methods.

\section defragmentation_gpu Defragmenting GPU memory

It is also possible to defragment allocations created in memory types that are not `HOST_VISIBLE`.
To do that, you need to pass a command buffer that meets requirements as described in
VmaDefragmentationInfo2::commandBuffer. The way it works is:

- It creates temporary buffers and binds them to entire memory blocks when necessary.
- It issues `vkCmdCopyBuffer()` to passed command buffer.

Example:

\code
// Given following variables already initialized:
VkDevice device;
VmaAllocator allocator;
VkCommandBuffer commandBuffer;
std::vector<VkBuffer> buffers;
std::vector<VmaAllocation> allocations;


const uint32_t allocCount = (uint32_t)allocations.size();
std::vector<VkBool32> allocationsChanged(allocCount);

VkCommandBufferBeginInfo cmdBufBeginInfo = ...;
vkBeginCommandBuffer(commandBuffer, &cmdBufBeginInfo);

VmaDefragmentationInfo2 defragInfo = {};
defragInfo.allocationCount = allocCount;
defragInfo.pAllocations = allocations.data();
defragInfo.pAllocationsChanged = allocationsChanged.data();
defragInfo.maxGpuBytesToMove = VK_WHOLE_SIZE; // Notice it's "GPU" this time.
defragInfo.maxGpuAllocationsToMove = UINT32_MAX; // Notice it's "GPU" this time.
defragInfo.commandBuffer = commandBuffer;

VmaDefragmentationContext defragCtx;
vmaDefragmentationBegin(allocator, &defragInfo, nullptr, &defragCtx);

vkEndCommandBuffer(commandBuffer);

// Submit commandBuffer.
// Wait for a fence that ensures commandBuffer execution finished.

vmaDefragmentationEnd(allocator, defragCtx);

for(uint32_t i = 0; i < allocCount; ++i)
{
    if(allocationsChanged[i])
    {
        // Destroy buffer that is immutably bound to memory region which is no longer valid.
        vkDestroyBuffer(device, buffers[i], nullptr);

        // Create new buffer with same parameters.
        VkBufferCreateInfo bufferInfo = ...;
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffers[i]);
            
        // You can make dummy call to vkGetBufferMemoryRequirements here to silence validation layer warning.
            
        // Bind new buffer to new memory region. Data contained in it is already moved.
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(allocator, allocations[i], &allocInfo);
        vmaBindBufferMemory(allocator, allocations[i], buffers[i]);
    }
}
\endcode

You can combine these two methods by specifying non-zero `maxGpu*` as well as `maxCpu*` parameters.
The library automatically chooses best method to defragment each memory pool.

You may try not to block your entire program to wait until defragmentation finishes,
but do it in the background, as long as you carefully fullfill requirements described
in function vmaDefragmentationBegin().

\section defragmentation_additional_notes Additional notes

It is only legal to defragment allocations bound to:

- buffers
- images created with `VK_IMAGE_CREATE_ALIAS_BIT`, `VK_IMAGE_TILING_LINEAR`, and
  being currently in `VK_IMAGE_LAYOUT_GENERAL` or `VK_IMAGE_LAYOUT_PREINITIALIZED`.

Defragmentation of images created with `VK_IMAGE_TILING_OPTIMAL` or in any other
layout may give undefined results.

If you defragment allocations bound to images, new images to be bound to new
memory region after defragmentation should be created with `VK_IMAGE_LAYOUT_PREINITIALIZED`
and then transitioned to their original layout from before defragmentation if
needed using an image memory barrier.

While using defragmentation, you may experience validation layer warnings, which you just need to ignore.
See [Validation layer warnings](@ref general_considerations_validation_layer_warnings).

Please don't expect memory to be fully compacted after defragmentation.
Algorithms inside are based on some heuristics that try to maximize number of Vulkan
memory blocks to make totally empty to release them, as well as to maximimze continuous
empty space inside remaining blocks, while minimizing the number and size of allocations that
need to be moved. Some fragmentation may still remain - this is normal.

\section defragmentation_custom_algorithm Writing custom defragmentation algorithm

If you want to implement your own, custom defragmentation algorithm,
there is infrastructure prepared for that,
but it is not exposed through the library API - you need to hack its source code.
Here are steps needed to do this:

-# Main thing you need to do is to define your own class derived from base abstract
   class `VmaDefragmentationAlgorithm` and implement your version of its pure virtual methods.
   See definition and comments of this class for details.
-# Your code needs to interact with device memory block metadata.
   If you need more access to its data than it's provided by its public interface,
   declare your new class as a friend class e.g. in class `VmaBlockMetadata_Generic`.
-# If you want to create a flag that would enable your algorithm or pass some additional
   flags to configure it, add them to `VmaDefragmentationFlagBits` and use them in
   VmaDefragmentationInfo2::flags.
-# Modify function `VmaBlockVectorDefragmentationContext::Begin` to create object
   of your new class whenever needed.


\page lost_allocations Lost allocations

If your game oversubscribes video memory, if may work OK in previous-generation
graphics APIs (DirectX 9, 10, 11, OpenGL) because resources are automatically
paged to system RAM. In Vulkan you can't do it because when you run out of
memory, an allocation just fails. If you have more data (e.g. textures) that can
fit into VRAM and you don't need it all at once, you may want to upload them to
GPU on demand and "push out" ones that are not used for a long time to make room
for the new ones, effectively using VRAM (or a cartain memory pool) as a form of
cache. Vulkan Memory Allocator can help you with that by supporting a concept of
"lost allocations".

To create an allocation that can become lost, include #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT
flag in VmaAllocationCreateInfo::flags. Before using a buffer or image bound to
such allocation in every new frame, you need to query it if it's not lost.
To check it, call vmaTouchAllocation().
If the allocation is lost, you should not use it or buffer/image bound to it.
You mustn't forget to destroy this allocation and this buffer/image.
vmaGetAllocationInfo() can also be used for checking status of the allocation.
Allocation is lost when returned VmaAllocationInfo::deviceMemory == `VK_NULL_HANDLE`.

To create an allocation that can make some other allocations lost to make room
for it, use #VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT flag. You will
usually use both flags #VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT and
#VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT at the same time.

Warning! Current implementation uses quite naive, brute force algorithm,
which can make allocation calls that use #VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT
flag quite slow. A new, more optimal algorithm and data structure to speed this
up is planned for the future.

<b>Q: When interleaving creation of new allocations with usage of existing ones,
how do you make sure that an allocation won't become lost while it's used in the
current frame?</b>

It is ensured because vmaTouchAllocation() / vmaGetAllocationInfo() not only returns allocation
status/parameters and checks whether it's not lost, but when it's not, it also
atomically marks it as used in the current frame, which makes it impossible to
become lost in that frame. It uses lockless algorithm, so it works fast and
doesn't involve locking any internal mutex.

<b>Q: What if my allocation may still be in use by the GPU when it's rendering a
previous frame while I already submit new frame on the CPU?</b>

You can make sure that allocations "touched" by vmaTouchAllocation() / vmaGetAllocationInfo() will not
become lost for a number of additional frames back from the current one by
specifying this number as VmaAllocatorCreateInfo::frameInUseCount (for default
memory pool) and VmaPoolCreateInfo::frameInUseCount (for custom pool).

<b>Q: How do you inform the library when new frame starts?</b>

You need to call function vmaSetCurrentFrameIndex().

Example code:

\code
struct MyBuffer
{
    VkBuffer m_Buf = nullptr;
    VmaAllocation m_Alloc = nullptr;

    // Called when the buffer is really needed in the current frame.
    void EnsureBuffer();
};

void MyBuffer::EnsureBuffer()
{
    // Buffer has been created.
    if(m_Buf != VK_NULL_HANDLE)
    {
        // Check if its allocation is not lost + mark it as used in current frame.
        if(vmaTouchAllocation(allocator, m_Alloc))
        {
            // It's all OK - safe to use m_Buf.
            return;
        }
    }

    // Buffer not yet exists or lost - destroy and recreate it.

    vmaDestroyBuffer(allocator, m_Buf, m_Alloc);

    VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufCreateInfo.size = 1024;
    bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT |
        VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT;

    vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &m_Buf, &m_Alloc, nullptr);
}
\endcode

When using lost allocations, you may see some Vulkan validation layer warnings
about overlapping regions of memory bound to different kinds of buffers and
images. This is still valid as long as you implement proper handling of lost
allocations (like in the example above) and don't use them.

You can create an allocation that is already in lost state from the beginning using function
vmaCreateLostAllocation(). It may be useful if you need a "dummy" allocation that is not null.

You can call function vmaMakePoolAllocationsLost() to set all eligible allocations
in a specified custom pool to lost state.
Allocations that have been "touched" in current frame or VmaPoolCreateInfo::frameInUseCount frames back
cannot become lost.

<b>Q: Can I touch allocation that cannot become lost?</b>

Yes, although it has no visible effect.
Calls to vmaGetAllocationInfo() and vmaTouchAllocation() update last use frame index
also for allocations that cannot become lost, but the only way to observe it is to dump
internal allocator state using vmaBuildStatsString().
You can use this feature for debugging purposes to explicitly mark allocations that you use
in current frame and then analyze JSON dump to see for how long each allocation stays unused.


\page statistics Statistics

This library contains functions that return information about its internal state,
especially the amount of memory allocated from Vulkan.
Please keep in mind that these functions need to traverse all internal data structures
to gather these information, so they may be quite time-consuming.
Don't call them too often.

\section statistics_numeric_statistics Numeric statistics

You can query for overall statistics of the allocator using function vmaCalculateStats().
Information are returned using structure #VmaStats.
It contains #VmaStatInfo - number of allocated blocks, number of allocations
(occupied ranges in these blocks), number of unused (free) ranges in these blocks,
number of bytes used and unused (but still allocated from Vulkan) and other information.
They are summed across memory heaps, memory types and total for whole allocator.

You can query for statistics of a custom pool using function vmaGetPoolStats().
Information are returned using structure #VmaPoolStats.

You can query for information about specific allocation using function vmaGetAllocationInfo().
It fill structure #VmaAllocationInfo.

\section statistics_json_dump JSON dump

You can dump internal state of the allocator to a string in JSON format using function vmaBuildStatsString().
The result is guaranteed to be correct JSON.
It uses ANSI encoding.
Any strings provided by user (see [Allocation names](@ref allocation_names))
are copied as-is and properly escaped for JSON, so if they use UTF-8, ISO-8859-2 or any other encoding,
this JSON string can be treated as using this encoding.
It must be freed using function vmaFreeStatsString().

The format of this JSON string is not part of official documentation of the library,
but it will not change in backward-incompatible way without increasing library major version number
and appropriate mention in changelog.

The JSON string contains all the data that can be obtained using vmaCalculateStats().
It can also contain detailed map of allocated memory blocks and their regions -
free and occupied by allocations.
This allows e.g. to visualize the memory or assess fragmentation.


\page allocation_annotation Allocation names and user data

\section allocation_user_data Allocation user data

You can annotate allocations with your own information, e.g. for debugging purposes.
To do that, fill VmaAllocationCreateInfo::pUserData field when creating
an allocation. It's an opaque `void*` pointer. You can use it e.g. as a pointer,
some handle, index, key, ordinal number or any other value that would associate
the allocation with your custom metadata.

\code
VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
// Fill bufferInfo...

MyBufferMetadata* pMetadata = CreateBufferMetadata();

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
allocCreateInfo.pUserData = pMetadata;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &buffer, &allocation, nullptr);
\endcode

The pointer may be later retrieved as VmaAllocationInfo::pUserData:

\code
VmaAllocationInfo allocInfo;
vmaGetAllocationInfo(allocator, allocation, &allocInfo);
MyBufferMetadata* pMetadata = (MyBufferMetadata*)allocInfo.pUserData;
\endcode

It can also be changed using function vmaSetAllocationUserData().

Values of (non-zero) allocations' `pUserData` are printed in JSON report created by
vmaBuildStatsString(), in hexadecimal form.

\section allocation_names Allocation names

There is alternative mode available where `pUserData` pointer is used to point to
a null-terminated string, giving a name to the allocation. To use this mode,
set #VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT flag in VmaAllocationCreateInfo::flags.
Then `pUserData` passed as VmaAllocationCreateInfo::pUserData or argument to
vmaSetAllocationUserData() must be either null or pointer to a null-terminated string.
The library creates internal copy of the string, so the pointer you pass doesn't need
to be valid for whole lifetime of the allocation. You can free it after the call.

\code
VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
// Fill imageInfo...

std::string imageName = "Texture: ";
imageName += fileName;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT;
allocCreateInfo.pUserData = imageName.c_str();

VkImage image;
VmaAllocation allocation;
vmaCreateImage(allocator, &imageInfo, &allocCreateInfo, &image, &allocation, nullptr);
\endcode

The value of `pUserData` pointer of the allocation will be different than the one
you passed when setting allocation's name - pointing to a buffer managed
internally that holds copy of the string.

\code
VmaAllocationInfo allocInfo;
vmaGetAllocationInfo(allocator, allocation, &allocInfo);
const char* imageName = (const char*)allocInfo.pUserData;
printf("Image name: %s\n", imageName);
\endcode

That string is also printed in JSON report created by vmaBuildStatsString().


\page debugging_memory_usage Debugging incorrect memory usage

If you suspect a bug with memory usage, like usage of uninitialized memory or
memory being overwritten out of bounds of an allocation,
you can use debug features of this library to verify this.

\section debugging_memory_usage_initialization Memory initialization

If you experience a bug with incorrect and nondeterministic data in your program and you suspect uninitialized memory to be used,
you can enable automatic memory initialization to verify this.
To do it, define macro `VMA_DEBUG_INITIALIZE_ALLOCATIONS` to 1.

\code
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#include "vk_mem_alloc.h"
\endcode

It makes memory of all new allocations initialized to bit pattern `0xDCDCDCDC`.
Before an allocation is destroyed, its memory is filled with bit pattern `0xEFEFEFEF`.
Memory is automatically mapped and unmapped if necessary.

If you find these values while debugging your program, good chances are that you incorrectly
read Vulkan memory that is allocated but not initialized, or already freed, respectively.

Memory initialization works only with memory types that are `HOST_VISIBLE`.
It works also with dedicated allocations.
It doesn't work with allocations created with #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT flag,
as they cannot be mapped.

\section debugging_memory_usage_margins Margins

By default, allocations are laid out in memory blocks next to each other if possible
(considering required alignment, `bufferImageGranularity`, and `nonCoherentAtomSize`).

![Allocations without margin](../gfx/Margins_1.png)

Define macro `VMA_DEBUG_MARGIN` to some non-zero value (e.g. 16) to enforce specified
number of bytes as a margin before and after every allocation.

\code
#define VMA_DEBUG_MARGIN 16
#include "vk_mem_alloc.h"
\endcode

![Allocations with margin](../gfx/Margins_2.png)

If your bug goes away after enabling margins, it means it may be caused by memory
being overwritten outside of allocation boundaries. It is not 100% certain though.
Change in application behavior may also be caused by different order and distribution
of allocations across memory blocks after margins are applied.

The margin is applied also before first and after last allocation in a block.
It may occur only once between two adjacent allocations.

Margins work with all types of memory.

Margin is applied only to allocations made out of memory blocks and not to dedicated
allocations, which have their own memory block of specific size.
It is thus not applied to allocations made using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT flag
or those automatically decided to put into dedicated allocations, e.g. due to its
large size or recommended by VK_KHR_dedicated_allocation extension.
Margins are also not active in custom pools created with #VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT flag.

Margins appear in [JSON dump](@ref statistics_json_dump) as part of free space.

Note that enabling margins increases memory usage and fragmentation.

\section debugging_memory_usage_corruption_detection Corruption detection

You can additionally define macro `VMA_DEBUG_DETECT_CORRUPTION` to 1 to enable validation
of contents of the margins.

\code
#define VMA_DEBUG_MARGIN 16
#define VMA_DEBUG_DETECT_CORRUPTION 1
#include "vk_mem_alloc.h"
\endcode

When this feature is enabled, number of bytes specified as `VMA_DEBUG_MARGIN`
(it must be multiply of 4) before and after every allocation is filled with a magic number.
This idea is also know as "canary".
Memory is automatically mapped and unmapped if necessary.

This number is validated automatically when the allocation is destroyed.
If it's not equal to the expected value, `VMA_ASSERT()` is executed.
It clearly means that either CPU or GPU overwritten the memory outside of boundaries of the allocation,
which indicates a serious bug.

You can also explicitly request checking margins of all allocations in all memory blocks
that belong to specified memory types by using function vmaCheckCorruption(),
or in memory blocks that belong to specified custom pool, by using function 
vmaCheckPoolCorruption().

Margin validation (corruption detection) works only for memory types that are
`HOST_VISIBLE` and `HOST_COHERENT`.


\page record_and_replay Record and replay

\section record_and_replay_introduction Introduction

While using the library, sequence of calls to its functions together with their
parameters can be recorded to a file and later replayed using standalone player
application. It can be useful to:

- Test correctness - check if same sequence of calls will not cause crash or
  failures on a target platform.
- Gather statistics - see number of allocations, peak memory usage, number of
  calls etc.
- Benchmark performance - see how much time it takes to replay the whole
  sequence.

\section record_and_replay_usage Usage

Recording functionality is disabled by default.
To enable it, define following macro before every include of this library:

\code
#define VMA_RECORDING_ENABLED 1
\endcode

<b>To record sequence of calls to a file:</b> Fill in
VmaAllocatorCreateInfo::pRecordSettings member while creating #VmaAllocator
object. File is opened and written during whole lifetime of the allocator.

<b>To replay file:</b> Use VmaReplay - standalone command-line program.
Precompiled binary can be found in "bin" directory.
Its source can be found in "src/VmaReplay" directory.
Its project is generated by Premake.
Command line syntax is printed when the program is launched without parameters.
Basic usage:

    VmaReplay.exe MyRecording.csv

<b>Documentation of file format</b> can be found in file: "docs/Recording file format.md".
It's a human-readable, text file in CSV format (Comma Separated Values).

\section record_and_replay_additional_considerations Additional considerations

- Replaying file that was recorded on a different GPU (with different parameters
  like `bufferImageGranularity`, `nonCoherentAtomSize`, and especially different
  set of memory heaps and types) may give different performance and memory usage
  results, as well as issue some warnings and errors.
- Current implementation of recording in VMA, as well as VmaReplay application, is
  coded and tested only on Windows. Inclusion of recording code is driven by
  `VMA_RECORDING_ENABLED` macro. Support for other platforms should be easy to
  add. Contributions are welcomed.


\page usage_patterns Recommended usage patterns

See also slides from talk:
[Sawicki, Adam. Advanced Graphics Techniques Tutorial: Memory management in Vulkan and DX12. Game Developers Conference, 2018](https://www.gdcvault.com/play/1025458/Advanced-Graphics-Techniques-Tutorial-New)


\section usage_patterns_common_mistakes Common mistakes

<b>Use of CPU_TO_GPU instead of CPU_ONLY memory</b>

#VMA_MEMORY_USAGE_CPU_TO_GPU is recommended only for resources that will be
mapped and written by the CPU, as well as read directly by the GPU - like some
buffers or textures updated every frame (dynamic). If you create a staging copy
of a resource to be written by CPU and then used as a source of transfer to
another resource placed in the GPU memory, that staging resource should be
created with #VMA_MEMORY_USAGE_CPU_ONLY. Please read the descriptions of these
enums carefully for details.

<b>Unnecessary use of custom pools</b>

\ref custom_memory_pools may be useful for special purposes - when you want to
keep certain type of resources separate e.g. to reserve minimum amount of memory
for them, limit maximum amount of memory they can occupy, or make some of them
push out the other through the mechanism of \ref lost_allocations. For most
resources this is not needed and so it is not recommended to create #VmaPool
objects and allocations out of them. Allocating from the default pool is sufficient.

\section usage_patterns_simple Simple patterns

\subsection usage_patterns_simple_render_targets Render targets

<b>When:</b>
Any resources that you frequently write and read on GPU,
e.g. images used as color attachments (aka "render targets"), depth-stencil attachments,
images/buffers used as storage image/buffer (aka "Unordered Access View (UAV)").

<b>What to do:</b>
Create them in video memory that is fastest to access from GPU using
#VMA_MEMORY_USAGE_GPU_ONLY.

Consider using [VK_KHR_dedicated_allocation](@ref vk_khr_dedicated_allocation) extension
and/or manually creating them as dedicated allocations using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
especially if they are large or if you plan to destroy and recreate them e.g. when
display resolution changes.
Prefer to create such resources first and all other GPU resources (like textures and vertex buffers) later.

\subsection usage_patterns_simple_immutable_resources Immutable resources

<b>When:</b>
Any resources that you fill on CPU only once (aka "immutable") or infrequently
and then read frequently on GPU,
e.g. textures, vertex and index buffers, constant buffers that don't change often.

<b>What to do:</b>
Create them in video memory that is fastest to access from GPU using
#VMA_MEMORY_USAGE_GPU_ONLY.

To initialize content of such resource, create a CPU-side (aka "staging") copy of it
in system memory - #VMA_MEMORY_USAGE_CPU_ONLY, map it, fill it,
and submit a transfer from it to the GPU resource.
You can keep the staging copy if you need it for another upload transfer in the future.
If you don't, you can destroy it or reuse this buffer for uploading different resource
after the transfer finishes.

Prefer to create just buffers in system memory rather than images, even for uploading textures.
Use `vkCmdCopyBufferToImage()`.
Dont use images with `VK_IMAGE_TILING_LINEAR`.

\subsection usage_patterns_dynamic_resources Dynamic resources

<b>When:</b>
Any resources that change frequently (aka "dynamic"), e.g. every frame or every draw call,
written on CPU, read on GPU.

<b>What to do:</b>
Create them using #VMA_MEMORY_USAGE_CPU_TO_GPU.
You can map it and write to it directly on CPU, as well as read from it on GPU.

This is a more complex situation. Different solutions are possible,
and the best one depends on specific GPU type, but you can use this simple approach for the start.
Prefer to write to such resource sequentially (e.g. using `memcpy`).
Don't perform random access or any reads from it on CPU, as it may be very slow.

\subsection usage_patterns_readback Readback

<b>When:</b>
Resources that contain data written by GPU that you want to read back on CPU,
e.g. results of some computations.

<b>What to do:</b>
Create them using #VMA_MEMORY_USAGE_GPU_TO_CPU.
You can write to them directly on GPU, as well as map and read them on CPU.

\section usage_patterns_advanced Advanced patterns

\subsection usage_patterns_integrated_graphics Detecting integrated graphics

You can support integrated graphics (like Intel HD Graphics, AMD APU) better
by detecting it in Vulkan.
To do it, call `vkGetPhysicalDeviceProperties()`, inspect
`VkPhysicalDeviceProperties::deviceType` and look for `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`.
When you find it, you can assume that memory is unified and all memory types are comparably fast
to access from GPU, regardless of `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.

You can then sum up sizes of all available memory heaps and treat them as useful for
your GPU resources, instead of only `DEVICE_LOCAL` ones.
You can also prefer to create your resources in memory types that are `HOST_VISIBLE` to map them
directly instead of submitting explicit transfer (see below).

\subsection usage_patterns_direct_vs_transfer Direct access versus transfer

For resources that you frequently write on CPU and read on GPU, many solutions are possible:

-# Create one copy in video memory using #VMA_MEMORY_USAGE_GPU_ONLY,
   second copy in system memory using #VMA_MEMORY_USAGE_CPU_ONLY and submit explicit tranfer each time.
-# Create just single copy using #VMA_MEMORY_USAGE_CPU_TO_GPU, map it and fill it on CPU,
   read it directly on GPU.
-# Create just single copy using #VMA_MEMORY_USAGE_CPU_ONLY, map it and fill it on CPU,
   read it directly on GPU.

Which solution is the most efficient depends on your resource and especially on the GPU.
It is best to measure it and then make the decision.
Some general recommendations:

- On integrated graphics use (2) or (3) to avoid unnecesary time and memory overhead
  related to using a second copy and making transfer.
- For small resources (e.g. constant buffers) use (2).
  Discrete AMD cards have special 256 MiB pool of video memory that is directly mappable.
  Even if the resource ends up in system memory, its data may be cached on GPU after first
  fetch over PCIe bus.
- For larger resources (e.g. textures), decide between (1) and (2).
  You may want to differentiate NVIDIA and AMD, e.g. by looking for memory type that is
  both `DEVICE_LOCAL` and `HOST_VISIBLE`. When you find it, use (2), otherwise use (1).

Similarly, for resources that you frequently write on GPU and read on CPU, multiple
solutions are possible:

-# Create one copy in video memory using #VMA_MEMORY_USAGE_GPU_ONLY,
   second copy in system memory using #VMA_MEMORY_USAGE_GPU_TO_CPU and submit explicit tranfer each time.
-# Create just single copy using #VMA_MEMORY_USAGE_GPU_TO_CPU, write to it directly on GPU,
   map it and read it on CPU.

You should take some measurements to decide which option is faster in case of your specific
resource.

If you don't want to specialize your code for specific types of GPUs, you can still make
an simple optimization for cases when your resource ends up in mappable memory to use it
directly in this case instead of creating CPU-side staging copy.
For details see [Finding out if memory is mappable](@ref memory_mapping_finding_if_memory_mappable).


\page configuration Configuration

Please check "CONFIGURATION SECTION" in the code to find macros that you can define
before each include of this file or change directly in this file to provide
your own implementation of basic facilities like assert, `min()` and `max()` functions,
mutex, atomic etc.
The library uses its own implementation of containers by default, but you can switch to using
STL containers instead.

For example, define `VMA_ASSERT(expr)` before including the library to provide
custom implementation of the assertion, compatible with your project.
By default it is defined to standard C `assert(expr)` in `_DEBUG` configuration
and empty otherwise.

\section config_Vulkan_functions Pointers to Vulkan functions

The library uses Vulkan functions straight from the `vulkan.h` header by default.
If you want to provide your own pointers to these functions, e.g. fetched using
`vkGetInstanceProcAddr()` and `vkGetDeviceProcAddr()`:

-# Define `VMA_STATIC_VULKAN_FUNCTIONS 0`.
-# Provide valid pointers through VmaAllocatorCreateInfo::pVulkanFunctions.

\section custom_memory_allocator Custom host memory allocator

If you use custom allocator for CPU memory rather than default operator `new`
and `delete` from C++, you can make this library using your allocator as well
by filling optional member VmaAllocatorCreateInfo::pAllocationCallbacks. These
functions will be passed to Vulkan, as well as used by the library itself to
make any CPU-side allocations.

\section allocation_callbacks Device memory allocation callbacks

The library makes calls to `vkAllocateMemory()` and `vkFreeMemory()` internally.
You can setup callbacks to be informed about these calls, e.g. for the purpose
of gathering some statistics. To do it, fill optional member
VmaAllocatorCreateInfo::pDeviceMemoryCallbacks.

\section heap_memory_limit Device heap memory limit

When device memory of certain heap runs out of free space, new allocations may
fail (returning error code) or they may succeed, silently pushing some existing
memory blocks from GPU VRAM to system RAM (which degrades performance). This
behavior is implementation-dependant - it depends on GPU vendor and graphics
driver.

On AMD cards it can be controlled while creating Vulkan device object by using
VK_AMD_memory_overallocation_behavior extension, if available.

Alternatively, if you want to test how your program behaves with limited amount of Vulkan device
memory available without switching your graphics card to one that really has
smaller VRAM, you can use a feature of this library intended for this purpose.
To do it, fill optional member VmaAllocatorCreateInfo::pHeapSizeLimit.



\page vk_khr_dedicated_allocation VK_KHR_dedicated_allocation

VK_KHR_dedicated_allocation is a Vulkan extension which can be used to improve
performance on some GPUs. It augments Vulkan API with possibility to query
driver whether it prefers particular buffer or image to have its own, dedicated
allocation (separate `VkDeviceMemory` block) for better efficiency - to be able
to do some internal optimizations.

The extension is supported by this library. It will be used automatically when
enabled. To enable it:

1 . When creating Vulkan device, check if following 2 device extensions are
supported (call `vkEnumerateDeviceExtensionProperties()`).
If yes, enable them (fill `VkDeviceCreateInfo::ppEnabledExtensionNames`).

- VK_KHR_get_memory_requirements2
- VK_KHR_dedicated_allocation

If you enabled these extensions:

2 . Use #VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT flag when creating
your #VmaAllocator`to inform the library that you enabled required extensions
and you want the library to use them.

\code
allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;

vmaCreateAllocator(&allocatorInfo, &allocator);
\endcode

That's all. The extension will be automatically used whenever you create a
buffer using vmaCreateBuffer() or image using vmaCreateImage().

When using the extension together with Vulkan Validation Layer, you will receive
warnings like this:

    vkBindBufferMemory(): Binding memory to buffer 0x33 but vkGetBufferMemoryRequirements() has not been called on that buffer.

It is OK, you should just ignore it. It happens because you use function
`vkGetBufferMemoryRequirements2KHR()` instead of standard
`vkGetBufferMemoryRequirements()`, while the validation layer seems to be
unaware of it.

To learn more about this extension, see:

- [VK_KHR_dedicated_allocation in Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.0-extensions/html/vkspec.html#VK_KHR_dedicated_allocation)
- [VK_KHR_dedicated_allocation unofficial manual](http://asawicki.info/articles/VK_KHR_dedicated_allocation.php5)



\page general_considerations General considerations

\section general_considerations_thread_safety Thread safety

- The library has no global state, so separate #VmaAllocator objects can be used
  independently.
  There should be no need to create multiple such objects though - one per `VkDevice` is enough.
- By default, all calls to functions that take #VmaAllocator as first parameter
  are safe to call from multiple threads simultaneously because they are
  synchronized internally when needed.
- When the allocator is created with #VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT
  flag, calls to functions that take such #VmaAllocator object must be
  synchronized externally.
- Access to a #VmaAllocation object must be externally synchronized. For example,
  you must not call vmaGetAllocationInfo() and vmaMapMemory() from different
  threads at the same time if you pass the same #VmaAllocation object to these
  functions.

\section general_considerations_validation_layer_warnings Validation layer warnings

When using this library, you can meet following types of warnings issued by
Vulkan validation layer. They don't necessarily indicate a bug, so you may need
to just ignore them.

- *vkBindBufferMemory(): Binding memory to buffer 0xeb8e4 but vkGetBufferMemoryRequirements() has not been called on that buffer.*
  - It happens when VK_KHR_dedicated_allocation extension is enabled.
    `vkGetBufferMemoryRequirements2KHR` function is used instead, while validation layer seems to be unaware of it.
- *Mapping an image with layout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL can result in undefined behavior if this memory is used by the device. Only GENERAL or PREINITIALIZED should be used.*
  - It happens when you map a buffer or image, because the library maps entire
    `VkDeviceMemory` block, where different types of images and buffers may end
    up together, especially on GPUs with unified memory like Intel.
- *Non-linear image 0xebc91 is aliased with linear buffer 0xeb8e4 which may indicate a bug.*
  - It happens when you use lost allocations, and a new image or buffer is
    created in place of an existing object that bacame lost.
  - It may happen also when you use [defragmentation](@ref defragmentation).

\section general_considerations_allocation_algorithm Allocation algorithm

The library uses following algorithm for allocation, in order:

-# Try to find free range of memory in existing blocks.
-# If failed, try to create a new block of `VkDeviceMemory`, with preferred block size.
-# If failed, try to create such block with size/2, size/4, size/8.
-# If failed and #VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT flag was
   specified, try to find space in existing blocks, possilby making some other
   allocations lost.
-# If failed, try to allocate separate `VkDeviceMemory` for this allocation,
   just like when you use #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
-# If failed, choose other memory type that meets the requirements specified in
   VmaAllocationCreateInfo and go to point 1.
-# If failed, return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

\section general_considerations_features_not_supported Features not supported

Features deliberately excluded from the scope of this library:

- Data transfer. Uploading (straming) and downloading data of buffers and images
  between CPU and GPU memory and related synchronization is responsibility of the user.
  Defining some "texture" object that would automatically stream its data from a
  staging copy in CPU memory to GPU memory would rather be a feature of another,
  higher-level library implemented on top of VMA.
- Allocations for imported/exported external memory. They tend to require
  explicit memory type index and dedicated allocation anyway, so they don't
  interact with main features of this library. Such special purpose allocations
  should be made manually, using `vkCreateBuffer()` and `vkAllocateMemory()`.
- Recreation of buffers and images. Although the library has functions for
  buffer and image creation (vmaCreateBuffer(), vmaCreateImage()), you need to
  recreate these objects yourself after defragmentation. That's because the big
  structures `VkBufferCreateInfo`, `VkImageCreateInfo` are not stored in
  #VmaAllocation object.
- Handling CPU memory allocation failures. When dynamically creating small C++
  objects in CPU memory (not Vulkan memory), allocation failures are not checked
  and handled gracefully, because that would complicate code significantly and
  is usually not needed in desktop PC applications anyway.
- Code free of any compiler warnings. Maintaining the library to compile and
  work correctly on so many different platforms is hard enough. Being free of 
  any warnings, on any version of any compiler, is simply not feasible.
- This is a C++ library with C interface.
  Bindings or ports to any other programming languages are welcomed as external projects and
  are not going to be included into this repository.

*/

/*
Define this macro to 0/1 to disable/enable support for recording functionality,
available through VmaAllocatorCreateInfo::pRecordSettings.
*/
#ifndef VMA_RECORDING_ENABLED
    #define VMA_RECORDING_ENABLED 0
#endif

#ifndef NOMINMAX
    #define NOMINMAX // For windows.h
#endif

#ifndef VULKAN_H_
    #include <vulkan/vulkan.h>
#endif

#if VMA_RECORDING_ENABLED
    #include <windows.h>
#endif

// Define this macro to declare maximum supported Vulkan version in format AAABBBCCC,
// where AAA = major, BBB = minor, CCC = patch.
// If you want to use version > 1.0, it still needs to be enabled via VmaAllocatorCreateInfo::vulkanApiVersion.
#if !defined(VMA_VULKAN_VERSION)
    #if defined(VK_VERSION_1_1)
        #define VMA_VULKAN_VERSION 1001000
    #else
        #define VMA_VULKAN_VERSION 1000000
    #endif
#endif

#if !defined(VMA_DEDICATED_ALLOCATION)
    #if VK_KHR_get_memory_requirements2 && VK_KHR_dedicated_allocation
        #define VMA_DEDICATED_ALLOCATION 1
    #else
        #define VMA_DEDICATED_ALLOCATION 0
    #endif
#endif

#if !defined(VMA_BIND_MEMORY2)
    #if VK_KHR_bind_memory2
        #define VMA_BIND_MEMORY2 1
    #else
        #define VMA_BIND_MEMORY2 0
    #endif
#endif

#if !defined(VMA_MEMORY_BUDGET)
    #if VK_EXT_memory_budget && (VK_KHR_get_physical_device_properties2 || VMA_VULKAN_VERSION >= 1001000)
        #define VMA_MEMORY_BUDGET 1
    #else
        #define VMA_MEMORY_BUDGET 0
    #endif
#endif

// Define these macros to decorate all public functions with additional code,
// before and after returned type, appropriately. This may be useful for
// exporing the functions when compiling VMA as a separate library. Example:
// #define VMA_CALL_PRE  __declspec(dllexport)
// #define VMA_CALL_POST __cdecl
#ifndef VMA_CALL_PRE
    #define VMA_CALL_PRE
#endif
#ifndef VMA_CALL_POST
    #define VMA_CALL_POST
#endif

/** \struct VmaAllocator
\brief Represents main object of this library initialized.

Fill structure #VmaAllocatorCreateInfo and call function vmaCreateAllocator() to create it.
Call function vmaDestroyAllocator() to destroy it.

It is recommended to create just one object of this type per `VkDevice` object,
right after Vulkan is initialized and keep it alive until before Vulkan device is destroyed.
*/
VK_DEFINE_HANDLE(VmaAllocator)

/// Callback function called after successful vkAllocateMemory.
typedef void (VKAPI_PTR *PFN_vmaAllocateDeviceMemoryFunction)(
    VmaAllocator      allocator,
    uint32_t          memoryType,
    VkDeviceMemory    memory,
    VkDeviceSize      size);
/// Callback function called before vkFreeMemory.
typedef void (VKAPI_PTR *PFN_vmaFreeDeviceMemoryFunction)(
    VmaAllocator      allocator,
    uint32_t          memoryType,
    VkDeviceMemory    memory,
    VkDeviceSize      size);

/** \brief Set of callbacks that the library will call for `vkAllocateMemory` and `vkFreeMemory`.

Provided for informative purpose, e.g. to gather statistics about number of
allocations or total amount of memory allocated in Vulkan.

Used in VmaAllocatorCreateInfo::pDeviceMemoryCallbacks.
*/
typedef struct VmaDeviceMemoryCallbacks {
    /// Optional, can be null.
    PFN_vmaAllocateDeviceMemoryFunction pfnAllocate;
    /// Optional, can be null.
    PFN_vmaFreeDeviceMemoryFunction pfnFree;
} VmaDeviceMemoryCallbacks;

/// Flags for created #VmaAllocator.
typedef enum VmaAllocatorCreateFlagBits {
    /** \brief Allocator and all objects created from it will not be synchronized internally, so you must guarantee they are used from only one thread at a time or synchronized externally by you.

    Using this flag may increase performance because internal mutexes are not used.
    */
    VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT = 0x00000001,
    /** \brief Enables usage of VK_KHR_dedicated_allocation extension.

    The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it's `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    Using this extenion will automatically allocate dedicated blocks of memory for
    some buffers and images instead of suballocating place for them out of bigger
    memory blocks (as if you explicitly used #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    flag) when it is recommended by the driver. It may improve performance on some
    GPUs.

    You may set this flag only if you found out that following device extensions are
    supported, you enabled them while creating Vulkan device passed as
    VmaAllocatorCreateInfo::device, and you want them to be used internally by this
    library:

    - VK_KHR_get_memory_requirements2 (device extension)
    - VK_KHR_dedicated_allocation (device extension)

    When this flag is set, you can experience following warnings reported by Vulkan
    validation layer. You can ignore them.

    > vkBindBufferMemory(): Binding memory to buffer 0x2d but vkGetBufferMemoryRequirements() has not been called on that buffer.
    */
    VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT = 0x00000002,
    /**
    Enables usage of VK_KHR_bind_memory2 extension.

    The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it's `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library.

    The extension provides functions `vkBindBufferMemory2KHR` and `vkBindImageMemory2KHR`,
    which allow to pass a chain of `pNext` structures while binding.
    This flag is required if you use `pNext` parameter in vmaBindBufferMemory2() or vmaBindImageMemory2().
    */
    VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT = 0x00000004,
    /**
    Enables usage of VK_EXT_memory_budget extension.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library, along with another instance extension
    VK_KHR_get_physical_device_properties2, which is required by it (or Vulkan 1.1, where this extension is promoted).

    The extension provides query for current memory usage and budget, which will probably
    be more accurate than an estimation used by the library otherwise.
    */
    VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT = 0x00000008,

    VMA_ALLOCATOR_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaAllocatorCreateFlagBits;
typedef VkFlags VmaAllocatorCreateFlags;

/** \brief Pointers to some Vulkan functions - a subset used by the library.

Used in VmaAllocatorCreateInfo::pVulkanFunctions.
*/
typedef struct VmaVulkanFunctions {
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    PFN_vkAllocateMemory vkAllocateMemory;
    PFN_vkFreeMemory vkFreeMemory;
    PFN_vkMapMemory vkMapMemory;
    PFN_vkUnmapMemory vkUnmapMemory;
    PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
    PFN_vkBindBufferMemory vkBindBufferMemory;
    PFN_vkBindImageMemory vkBindImageMemory;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
    PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
    PFN_vkCreateBuffer vkCreateBuffer;
    PFN_vkDestroyBuffer vkDestroyBuffer;
    PFN_vkCreateImage vkCreateImage;
    PFN_vkDestroyImage vkDestroyImage;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR;
    PFN_vkGetImageMemoryRequirements2KHR vkGetImageMemoryRequirements2KHR;
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    PFN_vkBindBufferMemory2KHR vkBindBufferMemory2KHR;
    PFN_vkBindImageMemory2KHR vkBindImageMemory2KHR;
#endif
#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;
#endif
} VmaVulkanFunctions;

/// Flags to be used in VmaRecordSettings::flags.
typedef enum VmaRecordFlagBits {
    /** \brief Enables flush after recording every function call.

    Enable it if you expect your application to crash, which may leave recording file truncated.
    It may degrade performance though.
    */
    VMA_RECORD_FLUSH_AFTER_CALL_BIT = 0x00000001,
    
    VMA_RECORD_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaRecordFlagBits;
typedef VkFlags VmaRecordFlags;

/// Parameters for recording calls to VMA functions. To be used in VmaAllocatorCreateInfo::pRecordSettings.
typedef struct VmaRecordSettings
{
    /// Flags for recording. Use #VmaRecordFlagBits enum.
    VmaRecordFlags flags;
    /** \brief Path to the file that should be written by the recording.

    Suggested extension: "csv".
    If the file already exists, it will be overwritten.
    It will be opened for the whole time #VmaAllocator object is alive.
    If opening this file fails, creation of the whole allocator object fails.
    */
    const char* pFilePath;
} VmaRecordSettings;

/// Description of a Allocator to be created.
typedef struct VmaAllocatorCreateInfo
{
    /// Flags for created allocator. Use #VmaAllocatorCreateFlagBits enum.
    VmaAllocatorCreateFlags flags;
    /// Vulkan physical device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkPhysicalDevice physicalDevice;
    /// Vulkan device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkDevice device;
    /// Preferred size of a single `VkDeviceMemory` block to be allocated from large heaps > 1 GiB. Optional.
    /** Set to 0 to use default, which is currently 256 MiB. */
    VkDeviceSize preferredLargeHeapBlockSize;
    /// Custom CPU memory allocation callbacks. Optional.
    /** Optional, can be null. When specified, will also be used for all CPU-side memory allocations. */
    const VkAllocationCallbacks* pAllocationCallbacks;
    /// Informative callbacks for `vkAllocateMemory`, `vkFreeMemory`. Optional.
    /** Optional, can be null. */
    const VmaDeviceMemoryCallbacks* pDeviceMemoryCallbacks;
    /** \brief Maximum number of additional frames that are in use at the same time as current frame.

    This value is used only when you make allocations with
    VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT flag. Such allocation cannot become
    lost if allocation.lastUseFrameIndex >= allocator.currentFrameIndex - frameInUseCount.

    For example, if you double-buffer your command buffers, so resources used for
    rendering in previous frame may still be in use by the GPU at the moment you
    allocate resources needed for the current frame, set this value to 1.

    If you want to allow any allocations other than used in the current frame to
    become lost, set this value to 0.
    */
    uint32_t frameInUseCount;
    /** \brief Either null or a pointer to an array of limits on maximum number of bytes that can be allocated out of particular Vulkan memory heap.

    If not NULL, it must be a pointer to an array of
    `VkPhysicalDeviceMemoryProperties::memoryHeapCount` elements, defining limit on
    maximum number of bytes that can be allocated out of particular Vulkan memory
    heap.

    Any of the elements may be equal to `VK_WHOLE_SIZE`, which means no limit on that
    heap. This is also the default in case of `pHeapSizeLimit` = NULL.

    If there is a limit defined for a heap:

    - If user tries to allocate more memory from that heap using this allocator,
      the allocation fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    - If the limit is smaller than heap size reported in `VkMemoryHeap::size`, the
      value of this limit will be reported instead when using vmaGetMemoryProperties().

    Warning! Using this feature may not be equivalent to installing a GPU with
    smaller amount of memory, because graphics driver doesn't necessary fail new
    allocations with `VK_ERROR_OUT_OF_DEVICE_MEMORY` result when memory capacity is
    exceeded. It may return success and just silently migrate some device memory
    blocks to system RAM. This driver behavior can also be controlled using
    VK_AMD_memory_overallocation_behavior extension.
    */
    const VkDeviceSize* pHeapSizeLimit;
    /** \brief Pointers to Vulkan functions. Can be null if you leave define `VMA_STATIC_VULKAN_FUNCTIONS 1`.

    If you leave define `VMA_STATIC_VULKAN_FUNCTIONS 1` in configuration section,
    you can pass null as this member, because the library will fetch pointers to
    Vulkan functions internally in a static way, like:

        vulkanFunctions.vkAllocateMemory = &vkAllocateMemory;

    Fill this member if you want to provide your own pointers to Vulkan functions,
    e.g. fetched using `vkGetInstanceProcAddr()` and `vkGetDeviceProcAddr()`.
    */
    const VmaVulkanFunctions* pVulkanFunctions;
    /** \brief Parameters for recording of VMA calls. Can be null.

    If not null, it enables recording of calls to VMA functions to a file.
    If support for recording is not enabled using `VMA_RECORDING_ENABLED` macro,
    creation of the allocator object fails with `VK_ERROR_FEATURE_NOT_PRESENT`.
    */
    const VmaRecordSettings* pRecordSettings;
    /** \brief Optional handle to Vulkan instance object.

    Optional, can be null. Must be set if #VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT flas is used
    or if `vulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)`.
    */
    VkInstance instance;
    /** \brief Optional. The highest version of Vulkan that the application is designed to use.
    
    It must be a value in the format as created by macro `VK_MAKE_VERSION` or a constant like: `VK_API_VERSION_1_1`, `VK_API_VERSION_1_0`.
    The patch version number specified is ignored. Only the major and minor versions are considered.
    It must be less or euqal (preferably equal) to value as passed to `vkCreateInstance` as `VkApplicationInfo::apiVersion`.
    Only versions 1.0 and 1.1 are supported by the current implementation.
    Leaving it initialized to zero is equivalent to `VK_API_VERSION_1_0`.
    */
    uint32_t vulkanApiVersion;
} VmaAllocatorCreateInfo;

/// Creates Allocator object.
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAllocator(
    const VmaAllocatorCreateInfo* pCreateInfo,
    VmaAllocator* pAllocator);

/// Destroys allocator object.
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyAllocator(
    VmaAllocator allocator);

/**
PhysicalDeviceProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPhysicalDeviceProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceProperties** ppPhysicalDeviceProperties);

/**
PhysicalDeviceMemoryProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceMemoryProperties** ppPhysicalDeviceMemoryProperties);

/**
\brief Given Memory Type Index, returns Property Flags of this memory type.

This is just a convenience function. Same information can be obtained using
vmaGetMemoryProperties().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryTypeProperties(
    VmaAllocator allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* pFlags);

/** \brief Sets index of the current frame.

This function must be used if you make allocations with
#VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT and
#VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT flags to inform the allocator
when a new frame begins. Allocations queried using vmaGetAllocationInfo() cannot
become lost in the current frame.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetCurrentFrameIndex(
    VmaAllocator allocator,
    uint32_t frameIndex);

/** \brief Calculated statistics of memory usage in entire allocator.
*/
typedef struct VmaStatInfo
{
    /// Number of `VkDeviceMemory` Vulkan memory blocks allocated.
    uint32_t blockCount;
    /// Number of #VmaAllocation allocation objects allocated.
    uint32_t allocationCount;
    /// Number of free ranges of memory between allocations.
    uint32_t unusedRangeCount;
    /// Total number of bytes occupied by all allocations.
    VkDeviceSize usedBytes;
    /// Total number of bytes occupied by unused ranges.
    VkDeviceSize unusedBytes;
    VkDeviceSize allocationSizeMin, allocationSizeAvg, allocationSizeMax;
    VkDeviceSize unusedRangeSizeMin, unusedRangeSizeAvg, unusedRangeSizeMax;
} VmaStatInfo;

/// General statistics from current state of Allocator.
typedef struct VmaStats
{
    VmaStatInfo memoryType[VK_MAX_MEMORY_TYPES];
    VmaStatInfo memoryHeap[VK_MAX_MEMORY_HEAPS];
    VmaStatInfo total;
} VmaStats;

/** \brief Retrieves statistics from current state of the Allocator.

This function is called "calculate" not "get" because it has to traverse all
internal data structures, so it may be quite slow. For faster but more brief statistics
suitable to be called every frame or every allocation, use vmaGetBudget().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCalculateStats(
    VmaAllocator allocator,
    VmaStats* pStats);

/** \brief Statistics of current memory usage and available budget, in bytes, for specific memory heap.
*/
typedef struct VmaBudget
{
    /** \brief Sum size of all `VkDeviceMemory` blocks allocated from particular heap, in bytes.
    */
    VkDeviceSize blockBytes;
    
    /** \brief Sum size of all allocations created in particular heap, in bytes.
    
    Usually less or equal than `blockBytes`.
    Difference `blockBytes - allocationBytes` is the amount of memory allocated but unused -
    available for new allocations or wasted due to fragmentation.
    
    It might be greater than `blockBytes` if there are some allocations in lost state, as they account
    to this value as well.
    */
    VkDeviceSize allocationBytes;
    
    /** \brief Estimated current memory usage of the program, in bytes.
    
    Fetched from system using `VK_EXT_memory_budget` extension if enabled.
    
    It might be different than `blockBytes` (usually higher) due to additional implicit objects
    also occupying the memory, like swapchain, pipelines, descriptor heaps, command buffers, or
    `VkDeviceMemory` blocks allocated outside of this library, if any.
    */
    VkDeviceSize usage;
    
    /** \brief Estimated amount of memory available to the program, in bytes.
    
    Fetched from system using `VK_EXT_memory_budget` extension if enabled.
    
    It might be different (most probably smaller) than `VkMemoryHeap::size[heapIndex]` due to factors
    external to the program, like other programs also consuming system resources.
    Difference `budget - usage` is the amount of additional memory that can probably
    be allocated without problems. Exceeding the budget may result in various problems.
    */
    VkDeviceSize budget;
} VmaBudget;

/** \brief Retrieves information about current memory budget for all memory heaps.

\param[out] pBudget Must point to array with number of elements at least equal to number of memory heaps in physical device used.

This function is called "get" not "calculate" because it is very fast, suitable to be called
every frame or every allocation. For more detailed statistics use vmaCalculateStats().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetBudget(
    VmaAllocator allocator,
    VmaBudget* pBudget);

#ifndef VMA_STATS_STRING_ENABLED
#define VMA_STATS_STRING_ENABLED 1
#endif

#if VMA_STATS_STRING_ENABLED

/// Builds and returns statistics as string in JSON format.
/** @param[out] ppStatsString Must be freed using vmaFreeStatsString() function.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaBuildStatsString(
    VmaAllocator allocator,
    char** ppStatsString,
    VkBool32 detailedMap);

VMA_CALL_PRE void VMA_CALL_POST vmaFreeStatsString(
    VmaAllocator allocator,
    char* pStatsString);

#endif // #if VMA_STATS_STRING_ENABLED

/** \struct VmaPool
\brief Represents custom memory pool

Fill structure VmaPoolCreateInfo and call function vmaCreatePool() to create it.
Call function vmaDestroyPool() to destroy it.

For more information see [Custom memory pools](@ref choosing_memory_type_custom_memory_pools).
*/
VK_DEFINE_HANDLE(VmaPool)

typedef enum VmaMemoryUsage
{
    /** No intended memory usage specified.
    Use other members of VmaAllocationCreateInfo to specify your requirements.
    */
    VMA_MEMORY_USAGE_UNKNOWN = 0,
    /** Memory will be used on device only, so fast access from the device is preferred.
    It usually means device-local GPU (video) memory.
    No need to be mappable on host.
    It is roughly equivalent of `D3D12_HEAP_TYPE_DEFAULT`.

    Usage:
    
    - Resources written and read by device, e.g. images used as attachments.
    - Resources transferred from host once (immutable) or infrequently and read by
      device multiple times, e.g. textures to be sampled, vertex buffers, uniform
      (constant) buffers, and majority of other types of resources used on GPU.

    Allocation may still end up in `HOST_VISIBLE` memory on some implementations.
    In such case, you are free to map it.
    You can use #VMA_ALLOCATION_CREATE_MAPPED_BIT with this usage type.
    */
    VMA_MEMORY_USAGE_GPU_ONLY = 1,
    /** Memory will be mappable on host.
    It usually means CPU (system) memory.
    Guarantees to be `HOST_VISIBLE` and `HOST_COHERENT`.
    CPU access is typically uncached. Writes may be write-combined.
    Resources created in this pool may still be accessible to the device, but access to them can be slow.
    It is roughly equivalent of `D3D12_HEAP_TYPE_UPLOAD`.

    Usage: Staging copy of resources used as transfer source.
    */
    VMA_MEMORY_USAGE_CPU_ONLY = 2,
    /**
    Memory that is both mappable on host (guarantees to be `HOST_VISIBLE`) and preferably fast to access by GPU.
    CPU access is typically uncached. Writes may be write-combined.

    Usage: Resources written frequently by host (dynamic), read by device. E.g. textures, vertex buffers, uniform buffers updated every frame or every draw call.
    */
    VMA_MEMORY_USAGE_CPU_TO_GPU = 3,
    /** Memory mappable on host (guarantees to be `HOST_VISIBLE`) and cached.
    It is roughly equivalent of `D3D12_HEAP_TYPE_READBACK`.

    Usage:

    - Resources written by device, read by host - results of some computations, e.g. screen capture, average scene luminance for HDR tone mapping.
    - Any resources read or accessed randomly on host, e.g. CPU-side copy of vertex buffer used as source of transfer, but also used for collision detection.
    */
    VMA_MEMORY_USAGE_GPU_TO_CPU = 4,
    /** CPU memory - memory that is preferably not `DEVICE_LOCAL`, but also not guaranteed to be `HOST_VISIBLE`.

    Usage: Staging copy of resources moved from GPU memory to CPU memory as part
    of custom paging/residency mechanism, to be moved back to GPU memory when needed.
    */
    VMA_MEMORY_USAGE_CPU_COPY = 5,
    /** Lazily allocated GPU memory having `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`.
    Exists mostly on mobile platforms. Using it on desktop PC or other GPUs with no such memory type present will fail the allocation.
    
    Usage: Memory for transient attachment images (color attachments, depth attachments etc.), created with `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`.

    Allocations with this usage are always created as dedicated - it implies #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    */
    VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED = 6,

    VMA_MEMORY_USAGE_MAX_ENUM = 0x7FFFFFFF
} VmaMemoryUsage;

/// Flags to be passed as VmaAllocationCreateInfo::flags.
typedef enum VmaAllocationCreateFlagBits {
    /** \brief Set this flag if the allocation should have its own memory block.
    
    Use it for special, big resources, like fullscreen images used as attachments.
   
    You should not use this flag if VmaAllocationCreateInfo::pool is not null.
    */
    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT = 0x00000001,

    /** \brief Set this flag to only try to allocate from existing `VkDeviceMemory` blocks and never create new such block.
    
    If new allocation cannot be placed in any of the existing blocks, allocation
    fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` error.
    
    You should not use #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT and
    #VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT at the same time. It makes no sense.
    
    If VmaAllocationCreateInfo::pool is not null, this flag is implied and ignored. */
    VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT = 0x00000002,
    /** \brief Set this flag to use a memory that will be persistently mapped and retrieve pointer to it.
    
    Pointer to mapped memory will be returned through VmaAllocationInfo::pMappedData.

    Is it valid to use this flag for allocation made from memory type that is not
    `HOST_VISIBLE`. This flag is then ignored and memory is not mapped. This is
    useful if you need an allocation that is efficient to use on GPU
    (`DEVICE_LOCAL`) and still want to map it directly if possible on platforms that
    support it (e.g. Intel GPU).

    You should not use this flag together with #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT.
    */
    VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004,
    /** Allocation created with this flag can become lost as a result of another
    allocation with #VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT flag, so you
    must check it before use.

    To check if allocation is not lost, call vmaGetAllocationInfo() and check if
    VmaAllocationInfo::deviceMemory is not `VK_NULL_HANDLE`.

    For details about supporting lost allocations, see Lost Allocations
    chapter of User Guide on Main Page.

    You should not use this flag together with #VMA_ALLOCATION_CREATE_MAPPED_BIT.
    */
    VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT = 0x00000008,
    /** While creating allocation using this flag, other allocations that were
    created with flag #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT can become lost.

    For details about supporting lost allocations, see Lost Allocations
    chapter of User Guide on Main Page.
    */
    VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT = 0x00000010,
    /** Set this flag to treat VmaAllocationCreateInfo::pUserData as pointer to a
    null-terminated string. Instead of copying pointer value, a local copy of the
    string is made and stored in allocation's `pUserData`. The string is automatically
    freed together with the allocation. It is also used in vmaBuildStatsString().
    */
    VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT = 0x00000020,
    /** Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT flag.
    */
    VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = 0x00000040,
    /** Create both buffer/image and allocation, but don't bind them together.
    It is useful when you want to bind yourself to do some more advanced binding, e.g. using some extensions.
    The flag is meaningful only with functions that bind by default: vmaCreateBuffer(), vmaCreateImage().
    Otherwise it is ignored.
    */
    VMA_ALLOCATION_CREATE_DONT_BIND_BIT = 0x00000080,
    /** Create allocation only if additional device memory required for it, if any, won't exceed
    memory budget. Otherwise return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    */
    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT = 0x00000100,

    /** Allocation strategy that chooses smallest possible free range for the
    allocation.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT  = 0x00010000,
    /** Allocation strategy that chooses biggest possible free range for the
    allocation.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT = 0x00020000,
    /** Allocation strategy that chooses first suitable free range for the
    allocation.

    "First" doesn't necessarily means the one with smallest offset in memory,
    but rather the one that is easiest and fastest to find.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT = 0x00040000,

    /** Allocation strategy that tries to minimize memory usage.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
    /** Allocation strategy that tries to minimize allocation time.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT,
    /** Allocation strategy that tries to minimize memory fragmentation.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT = VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT,

    /** A bit mask to extract only `STRATEGY` bits from entire set of flags.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MASK =
        VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT |
        VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT |
        VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT,

    VMA_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaAllocationCreateFlagBits;
typedef VkFlags VmaAllocationCreateFlags;

typedef struct VmaAllocationCreateInfo
{
    /// Use #VmaAllocationCreateFlagBits enum.
    VmaAllocationCreateFlags flags;
    /** \brief Intended usage of memory.
    
    You can leave #VMA_MEMORY_USAGE_UNKNOWN if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.
    */
    VmaMemoryUsage usage;
    /** \brief Flags that must be set in a Memory Type chosen for an allocation.
    
    Leave 0 if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.*/
    VkMemoryPropertyFlags requiredFlags;
    /** \brief Flags that preferably should be set in a memory type chosen for an allocation.
    
    Set to 0 if no additional flags are prefered. \n
    If `pool` is not null, this member is ignored. */
    VkMemoryPropertyFlags preferredFlags;
    /** \brief Bitmask containing one bit set for every memory type acceptable for this allocation.

    Value 0 is equivalent to `UINT32_MAX` - it means any memory type is accepted if
    it meets other requirements specified by this structure, with no further
    restrictions on memory type index. \n
    If `pool` is not null, this member is ignored.
    */
    uint32_t memoryTypeBits;
    /** \brief Pool that this allocation should be created in.

    Leave `VK_NULL_HANDLE` to allocate from default pool. If not null, members:
    `usage`, `requiredFlags`, `preferredFlags`, `memoryTypeBits` are ignored.
    */
    VmaPool pool;
    /** \brief Custom general-purpose pointer that will be stored in #VmaAllocation, can be read as VmaAllocationInfo::pUserData and changed using vmaSetAllocationUserData().
    
    If #VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT is used, it must be either
    null or pointer to a null-terminated string. The string will be then copied to
    internal buffer, so it doesn't need to be valid after allocation call.
    */
    void* pUserData;
} VmaAllocationCreateInfo;

/**
\brief Helps to find memoryTypeIndex, given memoryTypeBits and VmaAllocationCreateInfo.

This algorithm tries to find a memory type that:

- Is allowed by memoryTypeBits.
- Contains all the flags from pAllocationCreateInfo->requiredFlags.
- Matches intended usage.
- Has as many flags from pAllocationCreateInfo->preferredFlags as possible.

\return Returns VK_ERROR_FEATURE_NOT_PRESENT if not found. Receiving such result
from this function or any other allocating function probably means that your
device doesn't support any memory type with requested features for the specific
type of resource you want to use it for. Please check parameters of your
resource, like image layout (OPTIMAL versus LINEAR) or mip level count.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndex(
    VmaAllocator allocator,
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex);

/**
\brief Helps to find memoryTypeIndex, given VkBufferCreateInfo and VmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as VmaPoolCreateInfo::memoryTypeIndex.
It internally creates a temporary, dummy buffer that never has memory bound.
It is just a convenience function, equivalent to calling:

- `vkCreateBuffer`
- `vkGetBufferMemoryRequirements`
- `vmaFindMemoryTypeIndex`
- `vkDestroyBuffer`
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForBufferInfo(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex);

/**
\brief Helps to find memoryTypeIndex, given VkImageCreateInfo and VmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as VmaPoolCreateInfo::memoryTypeIndex.
It internally creates a temporary, dummy image that never has memory bound.
It is just a convenience function, equivalent to calling:

- `vkCreateImage`
- `vkGetImageMemoryRequirements`
- `vmaFindMemoryTypeIndex`
- `vkDestroyImage`
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForImageInfo(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex);

/// Flags to be passed as VmaPoolCreateInfo::flags.
typedef enum VmaPoolCreateFlagBits {
    /** \brief Use this flag if you always allocate only buffers and linear images or only optimal images out of this pool and so Buffer-Image Granularity can be ignored.

    This is an optional optimization flag.

    If you always allocate using vmaCreateBuffer(), vmaCreateImage(),
    vmaAllocateMemoryForBuffer(), then you don't need to use it because allocator
    knows exact type of your allocations so it can handle Buffer-Image Granularity
    in the optimal way.

    If you also allocate using vmaAllocateMemoryForImage() or vmaAllocateMemory(),
    exact type of such allocations is not known, so allocator must be conservative
    in handling Buffer-Image Granularity, which can lead to suboptimal allocation
    (wasted memory). In that case, if you can make sure you always allocate only
    buffers and linear images or only optimal images out of this pool, use this flag
    to make allocator disregard Buffer-Image Granularity and so make allocations
    faster and more optimal.
    */
    VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT = 0x00000002,

    /** \brief Enables alternative, linear allocation algorithm in this pool.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack. For details, see documentation chapter
    \ref linear_algorithm.

    When using this flag, you must specify VmaPoolCreateInfo::maxBlockCount == 1 (or 0 for default).

    For more details, see [Linear allocation algorithm](@ref linear_algorithm).
    */
    VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT = 0x00000004,

    /** \brief Enables alternative, buddy allocation algorithm in this pool.

    It operates on a tree of blocks, each having size that is a power of two and
    a half of its parent's size. Comparing to default algorithm, this one provides
    faster allocation and deallocation and decreased external fragmentation,
    at the expense of more memory wasted (internal fragmentation).

    For more details, see [Buddy allocation algorithm](@ref buddy_algorithm).
    */
    VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT = 0x00000008,

    /** Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    */
    VMA_POOL_CREATE_ALGORITHM_MASK =
        VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT |
        VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT,

    VMA_POOL_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaPoolCreateFlagBits;
typedef VkFlags VmaPoolCreateFlags;

/** \brief Describes parameter of created #VmaPool.
*/
typedef struct VmaPoolCreateInfo {
    /** \brief Vulkan memory type index to allocate this pool from.
    */
    uint32_t memoryTypeIndex;
    /** \brief Use combination of #VmaPoolCreateFlagBits.
    */
    VmaPoolCreateFlags flags;
    /** \brief Size of a single `VkDeviceMemory` block to be allocated as part of this pool, in bytes. Optional.

    Specify nonzero to set explicit, constant size of memory blocks used by this
    pool.

    Leave 0 to use default and let the library manage block sizes automatically.
    Sizes of particular blocks may vary.
    */
    VkDeviceSize blockSize;
    /** \brief Minimum number of blocks to be always allocated in this pool, even if they stay empty.

    Set to 0 to have no preallocated blocks and allow the pool be completely empty.
    */
    size_t minBlockCount;
    /** \brief Maximum number of blocks that can be allocated in this pool. Optional.

    Set to 0 to use default, which is `SIZE_MAX`, which means no limit.
    
    Set to same value as VmaPoolCreateInfo::minBlockCount to have fixed amount of memory allocated
    throughout whole lifetime of this pool.
    */
    size_t maxBlockCount;
    /** \brief Maximum number of additional frames that are in use at the same time as current frame.

    This value is used only when you make allocations with
    #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT flag. Such allocation cannot become
    lost if allocation.lastUseFrameIndex >= allocator.currentFrameIndex - frameInUseCount.

    For example, if you double-buffer your command buffers, so resources used for
    rendering in previous frame may still be in use by the GPU at the moment you
    allocate resources needed for the current frame, set this value to 1.

    If you want to allow any allocations other than used in the current frame to
    become lost, set this value to 0.
    */
    uint32_t frameInUseCount;
} VmaPoolCreateInfo;

/** \brief Describes parameter of existing #VmaPool.
*/
typedef struct VmaPoolStats {
    /** \brief Total amount of `VkDeviceMemory` allocated from Vulkan for this pool, in bytes.
    */
    VkDeviceSize size;
    /** \brief Total number of bytes in the pool not used by any #VmaAllocation.
    */
    VkDeviceSize unusedSize;
    /** \brief Number of #VmaAllocation objects created from this pool that were not destroyed or lost.
    */
    size_t allocationCount;
    /** \brief Number of continuous memory ranges in the pool not used by any #VmaAllocation.
    */
    size_t unusedRangeCount;
    /** \brief Size of the largest continuous free memory region available for new allocation.

    Making a new allocation of that size is not guaranteed to succeed because of
    possible additional margin required to respect alignment and buffer/image
    granularity.
    */
    VkDeviceSize unusedRangeSizeMax;
    /** \brief Number of `VkDeviceMemory` blocks allocated for this pool.
    */
    size_t blockCount;
} VmaPoolStats;

/** \brief Allocates Vulkan device memory and creates #VmaPool object.

@param allocator Allocator object.
@param pCreateInfo Parameters of pool to create.
@param[out] pPool Handle to created pool.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreatePool(
	VmaAllocator allocator,
	const VmaPoolCreateInfo* pCreateInfo,
	VmaPool* pPool);

/** \brief Destroys #VmaPool object and frees Vulkan device memory.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyPool(
    VmaAllocator allocator,
    VmaPool pool);

/** \brief Retrieves statistics of existing #VmaPool object.

@param allocator Allocator object.
@param pool Pool object.
@param[out] pPoolStats Statistics of specified pool.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolStats(
    VmaAllocator allocator,
    VmaPool pool,
    VmaPoolStats* pPoolStats);

/** \brief Marks all allocations in given pool as lost if they are not used in current frame or VmaPoolCreateInfo::frameInUseCount back from now.

@param allocator Allocator object.
@param pool Pool.
@param[out] pLostAllocationCount Number of allocations marked as lost. Optional - pass null if you don't need this information.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaMakePoolAllocationsLost(
    VmaAllocator allocator,
    VmaPool pool,
    size_t* pLostAllocationCount);

/** \brief Checks magic number in margins around all allocations in given memory pool in search for corruptions.

Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`VMA_DEBUG_MARGIN` is defined to nonzero and the pool is created in memory type that is
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for specified pool.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_VALIDATION_FAILED_EXT` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckPoolCorruption(VmaAllocator allocator, VmaPool pool);

/** \brief Retrieves name of a custom pool.

After the call `ppName` is either null or points to an internally-owned null-terminated string
containing name of the pool that was previously set. The pointer becomes invalid when the pool is
destroyed or its name is changed using vmaSetPoolName().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char** ppName);

/** \brief Sets name of a custom pool.

`pName` can be either null or pointer to a null-terminated string with new name for the pool.
Function makes internal copy of the string, so it can be changed or freed immediately after this call.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char* pName);

/** \struct VmaAllocation
\brief Represents single memory allocation.

It may be either dedicated block of `VkDeviceMemory` or a specific region of a bigger block of this type
plus unique offset.

There are multiple ways to create such object.
You need to fill structure VmaAllocationCreateInfo.
For more information see [Choosing memory type](@ref choosing_memory_type).

Although the library provides convenience functions that create Vulkan buffer or image,
allocate memory for it and bind them together,
binding of the allocation to a buffer or an image is out of scope of the allocation itself.
Allocation object can exist without buffer/image bound,
binding can be done manually by the user, and destruction of it can be done
independently of destruction of the allocation.

The object also remembers its size and some other information.
To retrieve this information, use function vmaGetAllocationInfo() and inspect
returned structure VmaAllocationInfo.

Some kinds allocations can be in lost state.
For more information, see [Lost allocations](@ref lost_allocations).
*/
VK_DEFINE_HANDLE(VmaAllocation)

/** \brief Parameters of #VmaAllocation objects, that can be retrieved using function vmaGetAllocationInfo().
*/
typedef struct VmaAllocationInfo {
    /** \brief Memory type index that this allocation was allocated from.
    
    It never changes.
    */
    uint32_t memoryType;
    /** \brief Handle to Vulkan memory object.

    Same memory object can be shared by multiple allocations.
    
    It can change after call to vmaDefragment() if this allocation is passed to the function, or if allocation is lost.

    If the allocation is lost, it is equal to `VK_NULL_HANDLE`.
    */
    VkDeviceMemory deviceMemory;
    /** \brief Offset into deviceMemory object to the beginning of this allocation, in bytes. (deviceMemory, offset) pair is unique to this allocation.

    It can change after call to vmaDefragment() if this allocation is passed to the function, or if allocation is lost.
    */
    VkDeviceSize offset;
    /** \brief Size of this allocation, in bytes.

    It never changes, unless allocation is lost.
    */
    VkDeviceSize size;
    /** \brief Pointer to the beginning of this allocation as mapped data.

    If the allocation hasn't been mapped using vmaMapMemory() and hasn't been
    created with #VMA_ALLOCATION_CREATE_MAPPED_BIT flag, this value null.

    It can change after call to vmaMapMemory(), vmaUnmapMemory().
    It can also change after call to vmaDefragment() if this allocation is passed to the function.
    */
    void* pMappedData;
    /** \brief Custom general-purpose pointer that was passed as VmaAllocationCreateInfo::pUserData or set using vmaSetAllocationUserData().

    It can change after call to vmaSetAllocationUserData() for this allocation.
    */
    void* pUserData;
} VmaAllocationInfo;

/** \brief General purpose memory allocation.

@param[out] pAllocation Handle to allocated memory.
@param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

You should free the memory using vmaFreeMemory() or vmaFreeMemoryPages().

It is recommended to use vmaAllocateMemoryForBuffer(), vmaAllocateMemoryForImage(),
vmaCreateBuffer(), vmaCreateImage() instead whenever possible.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemory(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo);

/** \brief General purpose memory allocation for multiple allocation objects at once.

@param allocator Allocator object.
@param pVkMemoryRequirements Memory requirements for each allocation.
@param pCreateInfo Creation parameters for each alloction.
@param allocationCount Number of allocations to make.
@param[out] pAllocations Pointer to array that will be filled with handles to created allocations.
@param[out] pAllocationInfo Optional. Pointer to array that will be filled with parameters of created allocations.

You should free the memory using vmaFreeMemory() or vmaFreeMemoryPages().

Word "pages" is just a suggestion to use this function to allocate pieces of memory needed for sparse binding.
It is just a general purpose allocation function able to make multiple allocations at once.
It may be internally optimized to be more efficient than calling vmaAllocateMemory() `allocationCount` times.

All allocations are made using same parameters. All of them are created out of the same memory pool and type.
If any allocation fails, all allocations already made within this function call are also freed, so that when
returned result is not `VK_SUCCESS`, `pAllocation` array is always entirely filled with `VK_NULL_HANDLE`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryPages(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    size_t allocationCount,
    VmaAllocation* pAllocations,
    VmaAllocationInfo* pAllocationInfo);

/**
@param[out] pAllocation Handle to allocated memory.
@param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

You should free the memory using vmaFreeMemory().
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo);

/// Function similar to vmaAllocateMemoryForBuffer().
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForImage(
    VmaAllocator allocator,
    VkImage image,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo);

/** \brief Frees memory previously allocated using vmaAllocateMemory(), vmaAllocateMemoryForBuffer(), or vmaAllocateMemoryForImage().

Passing `VK_NULL_HANDLE` as `allocation` is valid. Such function call is just skipped.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemory(
    VmaAllocator allocator,
    VmaAllocation allocation);

/** \brief Frees memory and destroys multiple allocations.

Word "pages" is just a suggestion to use this function to free pieces of memory used for sparse binding.
It is just a general purpose function to free memory and destroy allocations made using e.g. vmaAllocateMemory(),
vmaAllocateMemoryPages() and other functions.
It may be internally optimized to be more efficient than calling vmaFreeMemory() `allocationCount` times.

Allocations in `pAllocations` array can come from any memory pools and types.
Passing `VK_NULL_HANDLE` as elements of `pAllocations` array is valid. Such entries are just skipped.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemoryPages(
    VmaAllocator allocator,
    size_t allocationCount,
    VmaAllocation* pAllocations);

/** \brief Deprecated.

In version 2.2.0 it used to try to change allocation's size without moving or reallocating it.
In current version it returns `VK_SUCCESS` only if `newSize` equals current allocation's size.
Otherwise returns `VK_ERROR_OUT_OF_POOL_MEMORY`, indicating that allocation's size could not be changed.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaResizeAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize newSize);

/** \brief Returns current information about specified allocation and atomically marks it as used in current frame.

Current paramters of given allocation are returned in `pAllocationInfo`.

This function also atomically "touches" allocation - marks it as used in current frame,
just like vmaTouchAllocation().
If the allocation is in lost state, `pAllocationInfo->deviceMemory == VK_NULL_HANDLE`.

Although this function uses atomics and doesn't lock any mutex, so it should be quite efficient,
you can avoid calling it too often.

- You can retrieve same VmaAllocationInfo structure while creating your resource, from function
  vmaCreateBuffer(), vmaCreateImage(). You can remember it if you are sure parameters don't change
  (e.g. due to defragmentation or allocation becoming lost).
- If you just want to check if allocation is not lost, vmaTouchAllocation() will work faster.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VmaAllocationInfo* pAllocationInfo);

/** \brief Returns `VK_TRUE` if allocation is not lost and atomically marks it as used in current frame.

If the allocation has been created with #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT flag,
this function returns `VK_TRUE` if it's not in lost state, so it can still be used.
It then also atomically "touches" the allocation - marks it as used in current frame,
so that you can be sure it won't become lost in current frame or next `frameInUseCount` frames.

If the allocation is in lost state, the function returns `VK_FALSE`.
Memory of such allocation, as well as buffer or image bound to it, should not be used.
Lost allocation and the buffer/image still need to be destroyed.

If the allocation has been created without #VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT flag,
this function always returns `VK_TRUE`.
*/
VMA_CALL_PRE VkBool32 VMA_CALL_POST vmaTouchAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation);

/** \brief Sets pUserData in given allocation to new value.

If the allocation was created with VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT,
pUserData must be either null, or pointer to a null-terminated string. The function
makes local copy of the string and sets it as allocation's `pUserData`. String
passed as pUserData doesn't need to be valid for whole lifetime of the allocation -
you can free it after this call. String previously pointed by allocation's
pUserData is freed from memory.

If the flag was not used, the value of pointer `pUserData` is just copied to
allocation's `pUserData`. It is opaque, so you can use it however you want - e.g.
as a pointer, ordinal number or some handle to you own data.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationUserData(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void* pUserData);

/** \brief Creates new allocation that is in lost state from the beginning.

It can be useful if you need a dummy, non-null allocation.

You still need to destroy created object using vmaFreeMemory().

Returned allocation is not tied to any specific memory pool or memory type and
not bound to any image or buffer. It has size = 0. It cannot be turned into
a real, non-empty allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCreateLostAllocation(
    VmaAllocator allocator,
    VmaAllocation* pAllocation);

/** \brief Maps memory represented by given allocation and returns pointer to it.

Maps memory represented by given allocation to make it accessible to CPU code.
When succeeded, `*ppData` contains pointer to first byte of this memory.
If the allocation is part of bigger `VkDeviceMemory` block, the pointer is
correctly offseted to the beginning of region assigned to this particular
allocation.

Mapping is internally reference-counted and synchronized, so despite raw Vulkan
function `vkMapMemory()` cannot be used to map same block of `VkDeviceMemory`
multiple times simultaneously, it is safe to call this function on allocations
assigned to the same memory block. Actual Vulkan memory will be mapped on first
mapping and unmapped on last unmapping.

If the function succeeded, you must call vmaUnmapMemory() to unmap the
allocation when mapping is no longer needed or before freeing the allocation, at
the latest.

It also safe to call this function multiple times on the same allocation. You
must call vmaUnmapMemory() same number of times as you called vmaMapMemory().

It is also safe to call this function on allocation created with
#VMA_ALLOCATION_CREATE_MAPPED_BIT flag. Its memory stays mapped all the time.
You must still call vmaUnmapMemory() same number of times as you called
vmaMapMemory(). You must not call vmaUnmapMemory() additional time to free the
"0-th" mapping made automatically due to #VMA_ALLOCATION_CREATE_MAPPED_BIT flag.

This function fails when used on allocation made in memory type that is not
`HOST_VISIBLE`.

This function always fails when called for allocation that was created with
#VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT flag. Such allocations cannot be
mapped.

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use vmaInvalidateAllocation() / vmaFlushAllocation(), as required by Vulkan specification.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaMapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void** ppData);

/** \brief Unmaps memory represented by given allocation, mapped previously using vmaMapMemory().

For details, see description of vmaMapMemory().

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use vmaInvalidateAllocation() / vmaFlushAllocation(), as required by Vulkan specification.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaUnmapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation);

/** \brief Flushes memory of given allocation.

Calls `vkFlushMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called after writing to a mapped memory for memory types that are not `HOST_COHERENT`.
Unmap operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!
*/
VMA_CALL_PRE void VMA_CALL_POST vmaFlushAllocation(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size);

/** \brief Invalidates memory of given allocation.

Calls `vkInvalidateMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called before reading from a mapped memory for memory types that are not `HOST_COHERENT`.
Map operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!
*/
VMA_CALL_PRE void VMA_CALL_POST vmaInvalidateAllocation(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size);

/** \brief Checks magic number in margins around all allocations in given memory types (in both default and custom pools) in search for corruptions.

@param memoryTypeBits Bit mask, where each bit set means that a memory type with that index should be checked.

Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`VMA_DEBUG_MARGIN` is defined to nonzero and only for memory types that are
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for any of specified memory types.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_VALIDATION_FAILED_EXT` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckCorruption(VmaAllocator allocator, uint32_t memoryTypeBits);

/** \struct VmaDefragmentationContext
\brief Represents Opaque object that represents started defragmentation process.

Fill structure #VmaDefragmentationInfo2 and call function vmaDefragmentationBegin() to create it.
Call function vmaDefragmentationEnd() to destroy it.
*/
VK_DEFINE_HANDLE(VmaDefragmentationContext)

/// Flags to be used in vmaDefragmentationBegin(). None at the moment. Reserved for future use.
typedef enum VmaDefragmentationFlagBits {
    VMA_DEFRAGMENTATION_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaDefragmentationFlagBits;
typedef VkFlags VmaDefragmentationFlags;

/** \brief Parameters for defragmentation.

To be used with function vmaDefragmentationBegin().
*/
typedef struct VmaDefragmentationInfo2 {
    /** \brief Reserved for future use. Should be 0.
    */
    VmaDefragmentationFlags flags;
    /** \brief Number of allocations in `pAllocations` array.
    */
    uint32_t allocationCount;
    /** \brief Pointer to array of allocations that can be defragmented.

    The array should have `allocationCount` elements.
    The array should not contain nulls.
    Elements in the array should be unique - same allocation cannot occur twice.
    It is safe to pass allocations that are in the lost state - they are ignored.
    All allocations not present in this array are considered non-moveable during this defragmentation.
    */
    VmaAllocation* pAllocations;
    /** \brief Optional, output. Pointer to array that will be filled with information whether the allocation at certain index has been changed during defragmentation.

    The array should have `allocationCount` elements.
    You can pass null if you are not interested in this information.
    */
    VkBool32* pAllocationsChanged;
    /** \brief Numer of pools in `pPools` array.
    */
    uint32_t poolCount;
    /** \brief Either null or pointer to array of pools to be defragmented.

    All the allocations in the specified pools can be moved during defragmentation
    and there is no way to check if they were really moved as in `pAllocationsChanged`,
    so you must query all the allocations in all these pools for new `VkDeviceMemory`
    and offset using vmaGetAllocationInfo() if you might need to recreate buffers
    and images bound to them.

    The array should have `poolCount` elements.
    The array should not contain nulls.
    Elements in the array should be unique - same pool cannot occur twice.

    Using this array is equivalent to specifying all allocations from the pools in `pAllocations`.
    It might be more efficient.
    */
    VmaPool* pPools;
    /** \brief Maximum total numbers of bytes that can be copied while moving allocations to different places using transfers on CPU side, like `memcpy()`, `memmove()`.
    
    `VK_WHOLE_SIZE` means no limit.
    */
    VkDeviceSize maxCpuBytesToMove;
    /** \brief Maximum number of allocations that can be moved to a different place using transfers on CPU side, like `memcpy()`, `memmove()`.

    `UINT32_MAX` means no limit.
    */
    uint32_t maxCpuAllocationsToMove;
    /** \brief Maximum total numbers of bytes that can be copied while moving allocations to different places using transfers on GPU side, posted to `commandBuffer`.
    
    `VK_WHOLE_SIZE` means no limit.
    */
    VkDeviceSize maxGpuBytesToMove;
    /** \brief Maximum number of allocations that can be moved to a different place using transfers on GPU side, posted to `commandBuffer`.

    `UINT32_MAX` means no limit.
    */
    uint32_t maxGpuAllocationsToMove;
    /** \brief Optional. Command buffer where GPU copy commands will be posted.

    If not null, it must be a valid command buffer handle that supports Transfer queue type.
    It must be in the recording state and outside of a render pass instance.
    You need to submit it and make sure it finished execution before calling vmaDefragmentationEnd().

    Passing null means that only CPU defragmentation will be performed.
    */
    VkCommandBuffer commandBuffer;
} VmaDefragmentationInfo2;

/** \brief Deprecated. Optional configuration parameters to be passed to function vmaDefragment().

\deprecated This is a part of the old interface. It is recommended to use structure #VmaDefragmentationInfo2 and function vmaDefragmentationBegin() instead.
*/
typedef struct VmaDefragmentationInfo {
    /** \brief Maximum total numbers of bytes that can be copied while moving allocations to different places.
    
    Default is `VK_WHOLE_SIZE`, which means no limit.
    */
    VkDeviceSize maxBytesToMove;
    /** \brief Maximum number of allocations that can be moved to different place.

    Default is `UINT32_MAX`, which means no limit.
    */
    uint32_t maxAllocationsToMove;
} VmaDefragmentationInfo;

/** \brief Statistics returned by function vmaDefragment(). */
typedef struct VmaDefragmentationStats {
    /// Total number of bytes that have been copied while moving allocations to different places.
    VkDeviceSize bytesMoved;
    /// Total number of bytes that have been released to the system by freeing empty `VkDeviceMemory` objects.
    VkDeviceSize bytesFreed;
    /// Number of allocations that have been moved to different places.
    uint32_t allocationsMoved;
    /// Number of empty `VkDeviceMemory` objects that have been released to the system.
    uint32_t deviceMemoryBlocksFreed;
} VmaDefragmentationStats;

/** \brief Begins defragmentation process.

@param allocator Allocator object.
@param pInfo Structure filled with parameters of defragmentation.
@param[out] pStats Optional. Statistics of defragmentation. You can pass null if you are not interested in this information.
@param[out] pContext Context object that must be passed to vmaDefragmentationEnd() to finish defragmentation.
@return `VK_SUCCESS` and `*pContext == null` if defragmentation finished within this function call. `VK_NOT_READY` and `*pContext != null` if defragmentation has been started and you need to call vmaDefragmentationEnd() to finish it. Negative value in case of error.

Use this function instead of old, deprecated vmaDefragment().

Warning! Between the call to vmaDefragmentationBegin() and vmaDefragmentationEnd():

- You should not use any of allocations passed as `pInfo->pAllocations` or
  any allocations that belong to pools passed as `pInfo->pPools`,
  including calling vmaGetAllocationInfo(), vmaTouchAllocation(), or access
  their data.
- Some mutexes protecting internal data structures may be locked, so trying to
  make or free any allocations, bind buffers or images, map memory, or launch
  another simultaneous defragmentation in between may cause stall (when done on
  another thread) or deadlock (when done on the same thread), unless you are
  100% sure that defragmented allocations are in different pools.
- Information returned via `pStats` and `pInfo->pAllocationsChanged` are undefined.
  They become valid after call to vmaDefragmentationEnd().
- If `pInfo->commandBuffer` is not null, you must submit that command buffer
  and make sure it finished execution before calling vmaDefragmentationEnd().

For more information and important limitations regarding defragmentation, see documentation chapter:
[Defragmentation](@ref defragmentation).
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaDefragmentationBegin(
    VmaAllocator allocator,
    const VmaDefragmentationInfo2* pInfo,
    VmaDefragmentationStats* pStats,
    VmaDefragmentationContext *pContext);

/** \brief Ends defragmentation process.

Use this function to finish defragmentation started by vmaDefragmentationBegin().
It is safe to pass `context == null`. The function then does nothing.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaDefragmentationEnd(
    VmaAllocator allocator,
    VmaDefragmentationContext context);

/** \brief Deprecated. Compacts memory by moving allocations.

@param pAllocations Array of allocations that can be moved during this compation.
@param allocationCount Number of elements in pAllocations and pAllocationsChanged arrays.
@param[out] pAllocationsChanged Array of boolean values that will indicate whether matching allocation in pAllocations array has been moved. This parameter is optional. Pass null if you don't need this information.
@param pDefragmentationInfo Configuration parameters. Optional - pass null to use default values.
@param[out] pDefragmentationStats Statistics returned by the function. Optional - pass null if you don't need this information.
@return `VK_SUCCESS` if completed, negative error code in case of error.

\deprecated This is a part of the old interface. It is recommended to use structure #VmaDefragmentationInfo2 and function vmaDefragmentationBegin() instead.

This function works by moving allocations to different places (different
`VkDeviceMemory` objects and/or different offsets) in order to optimize memory
usage. Only allocations that are in `pAllocations` array can be moved. All other
allocations are considered nonmovable in this call. Basic rules:

- Only allocations made in memory types that have
  `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` and `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`
  flags can be compacted. You may pass other allocations but it makes no sense -
  these will never be moved.
- Custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT or
  #VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT flag are not defragmented. Allocations
  passed to this function that come from such pools are ignored.
- Allocations created with #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT or
  created as dedicated allocations for any other reason are also ignored.
- Both allocations made with or without #VMA_ALLOCATION_CREATE_MAPPED_BIT
  flag can be compacted. If not persistently mapped, memory will be mapped
  temporarily inside this function if needed.
- You must not pass same #VmaAllocation object multiple times in `pAllocations` array.

The function also frees empty `VkDeviceMemory` blocks.

Warning: This function may be time-consuming, so you shouldn't call it too often
(like after every resource creation/destruction).
You can call it on special occasions (like when reloading a game level or
when you just destroyed a lot of objects). Calling it every frame may be OK, but
you should measure that on your platform.

For more information, see [Defragmentation](@ref defragmentation) chapter.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaDefragment(
    VmaAllocator allocator,
    VmaAllocation* pAllocations,
    size_t allocationCount,
    VkBool32* pAllocationsChanged,
    const VmaDefragmentationInfo *pDefragmentationInfo,
    VmaDefragmentationStats* pDefragmentationStats);

/** \brief Binds buffer to allocation.

Binds specified buffer to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create a buffer, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindBufferMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function vmaCreateBuffer() instead of this one.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkBuffer buffer);

/** \brief Binds buffer to allocation with additional parameters.

@param allocationLocalOffset Additional offset to be added while binding, relative to the beginnig of the `allocation`. Normally it should be 0.
@param pNext A chain of structures to be attached to `VkBindBufferMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to vmaBindBufferMemory(), but it provides additional parameters.

If `pNext` is not null, #VmaAllocator object must have been created with #VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_1`. Otherwise the call fails.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer buffer,
    const void* pNext);

/** \brief Binds image to allocation.

Binds specified image to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create an image, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindImageMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function vmaCreateImage() instead of this one.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkImage image);

/** \brief Binds image to allocation with additional parameters.

@param allocationLocalOffset Additional offset to be added while binding, relative to the beginnig of the `allocation`. Normally it should be 0.
@param pNext A chain of structures to be attached to `VkBindImageMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to vmaBindImageMemory(), but it provides additional parameters.

If `pNext` is not null, #VmaAllocator object must have been created with #VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_1`. Otherwise the call fails.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage image,
    const void* pNext);

/**
@param[out] pBuffer Buffer that was created.
@param[out] pAllocation Allocation that was created.
@param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

This function automatically:

-# Creates buffer.
-# Allocates appropriate memory for it.
-# Binds the buffer with the memory.

If any of these operations fail, buffer and allocation are not created,
returned value is negative error code, *pBuffer and *pAllocation are null.

If the function succeeded, you must destroy both buffer and allocation when you
no longer need them using either convenience function vmaDestroyBuffer() or
separately, using `vkDestroyBuffer()` and vmaFreeMemory().

If VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT flag was used,
VK_KHR_dedicated_allocation extension is used internally to query driver whether
it requires or prefers the new buffer to have dedicated allocation. If yes,
and if dedicated allocation is possible (VmaAllocationCreateInfo::pool is null
and VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT is not used), it creates dedicated
allocation for this buffer, just like when using
VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBuffer(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkBuffer* pBuffer,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo);

/** \brief Destroys Vulkan buffer and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyBuffer(device, buffer, allocationCallbacks);
vmaFreeMemory(allocator, allocation);
\endcode

It it safe to pass null as buffer and/or allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    VmaAllocation allocation);

/// Function similar to vmaCreateBuffer().
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImage(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkImage* pImage,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo);

/** \brief Destroys Vulkan image and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyImage(device, image, allocationCallbacks);
vmaFreeMemory(allocator, allocation);
\endcode

It it safe to pass null as image and/or allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyImage(
    VmaAllocator allocator,
    VkImage image,
    VmaAllocation allocation);

#ifdef __cplusplus
}
#endif

#endif // AMD_VULKAN_MEMORY_ALLOCATOR_H

// For Visual Studio IntelliSense.
#if defined(__cplusplus) && defined(__INTELLISENSE__)
#define VMA_IMPLEMENTATION
#endif

#ifdef VMA_IMPLEMENTATION
#undef VMA_IMPLEMENTATION

#include <cstdint>
#include <cstdlib>
#include <cstring>

/*******************************************************************************
CONFIGURATION SECTION

Define some of these macros before each #include of this header or change them
here if you need other then default behavior depending on your environment.
*/

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

    vulkanFunctions.vkAllocateMemory = &vkAllocateMemory;

Define to 0 if you are going to provide you own pointers to Vulkan functions via
VmaAllocatorCreateInfo::pVulkanFunctions.
*/
#if !defined(VMA_STATIC_VULKAN_FUNCTIONS) && !defined(VK_NO_PROTOTYPES)
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#endif

// Define this macro to 1 to make the library use STL containers instead of its own implementation.
//#define VMA_USE_STL_CONTAINERS 1

/* Set this macro to 1 to make the library including and using STL containers:
std::pair, std::vector, std::list, std::unordered_map.

Set it to 0 or undefined to make the library using its own implementation of
the containers.
*/
#if VMA_USE_STL_CONTAINERS
   #define VMA_USE_STL_VECTOR 1
   #define VMA_USE_STL_UNORDERED_MAP 1
   #define VMA_USE_STL_LIST 1
#endif

#ifndef VMA_USE_STL_SHARED_MUTEX
    // Compiler conforms to C++17.
    #if __cplusplus >= 201703L
        #define VMA_USE_STL_SHARED_MUTEX 1
    // Visual studio defines __cplusplus properly only when passed additional parameter: /Zc:__cplusplus
    // Otherwise it's always 199711L, despite shared_mutex works since Visual Studio 2015 Update 2.
    // See: https://blogs.msdn.microsoft.com/vcblog/2018/04/09/msvc-now-correctly-reports-__cplusplus/
    #elif defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 190023918 && __cplusplus == 199711L && _MSVC_LANG >= 201703L
        #define VMA_USE_STL_SHARED_MUTEX 1
    #else
        #define VMA_USE_STL_SHARED_MUTEX 0
    #endif
#endif

/*
THESE INCLUDES ARE NOT ENABLED BY DEFAULT.
Library has its own container implementation.
*/
#if VMA_USE_STL_VECTOR
   #include <vector>
#endif

#if VMA_USE_STL_UNORDERED_MAP
   #include <unordered_map>
#endif

#if VMA_USE_STL_LIST
   #include <list>
#endif

/*
Following headers are used in this CONFIGURATION section only, so feel free to
remove them if not needed.
*/
#include <cassert> // for assert
#include <algorithm> // for min, max
#include <mutex>

#ifndef VMA_NULL
   // Value used as null pointer. Define it to e.g.: nullptr, NULL, 0, (void*)0.
   #define VMA_NULL   nullptr
#endif

#if defined(__ANDROID_API__) && (__ANDROID_API__ < 16)
#include <cstdlib>
void *aligned_alloc(size_t alignment, size_t size)
{
    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    return memalign(alignment, size);
}
#elif defined(__APPLE__) || defined(__ANDROID__) || (defined(__linux__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
#include <cstdlib>
void *aligned_alloc(size_t alignment, size_t size)
{
    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    void *pointer;
    if(posix_memalign(&pointer, alignment, size) == 0)
        return pointer;
    return VMA_NULL;
}
#endif

// If your compiler is not compatible with C++11 and definition of
// aligned_alloc() function is missing, uncommeting following line may help:

//#include <malloc.h>

// Normal assert to check for programmer's errors, especially in Debug configuration.
#ifndef VMA_ASSERT
   #ifdef _DEBUG
       #define VMA_ASSERT(expr)         assert(expr)
   #else
       #define VMA_ASSERT(expr)
   #endif
#endif

// Assert that will be called very often, like inside data structures e.g. operator[].
// Making it non-empty can make program slow.
#ifndef VMA_HEAVY_ASSERT
   #ifdef _DEBUG
       #define VMA_HEAVY_ASSERT(expr)   //VMA_ASSERT(expr)
   #else
       #define VMA_HEAVY_ASSERT(expr)
   #endif
#endif

#ifndef VMA_ALIGN_OF
   #define VMA_ALIGN_OF(type)       (__alignof(type))
#endif

#ifndef VMA_SYSTEM_ALIGNED_MALLOC
   #if defined(_WIN32)
       #define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment)   (_aligned_malloc((size), (alignment)))
   #else
       #define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment)   (aligned_alloc((alignment), (size) ))
   #endif
#endif

#ifndef VMA_SYSTEM_FREE
   #if defined(_WIN32)
       #define VMA_SYSTEM_FREE(ptr)   _aligned_free(ptr)
   #else
       #define VMA_SYSTEM_FREE(ptr)   free(ptr)
   #endif
#endif

#ifndef VMA_MIN
   #define VMA_MIN(v1, v2)    (std::min((v1), (v2)))
#endif

#ifndef VMA_MAX
   #define VMA_MAX(v1, v2)    (std::max((v1), (v2)))
#endif

#ifndef VMA_SWAP
   #define VMA_SWAP(v1, v2)   std::swap((v1), (v2))
#endif

#ifndef VMA_SORT
   #define VMA_SORT(beg, end, cmp)  std::sort(beg, end, cmp)
#endif

#ifndef VMA_DEBUG_LOG
   #define VMA_DEBUG_LOG(format, ...)
   /*
   #define VMA_DEBUG_LOG(format, ...) do { \
       printf(format, __VA_ARGS__); \
       printf("\n"); \
   } while(false)
   */
#endif

// Define this macro to 1 to enable functions: vmaBuildStatsString, vmaFreeStatsString.
#if VMA_STATS_STRING_ENABLED
    static inline void VmaUint32ToStr(char* outStr, size_t strLen, uint32_t num)
    {
        snprintf(outStr, strLen, "%u", static_cast<unsigned int>(num));
    }
    static inline void VmaUint64ToStr(char* outStr, size_t strLen, uint64_t num)
    {
        snprintf(outStr, strLen, "%llu", static_cast<unsigned long long>(num));
    }
    static inline void VmaPtrToStr(char* outStr, size_t strLen, const void* ptr)
    {
        snprintf(outStr, strLen, "%p", ptr);
    }
#endif

#ifndef VMA_MUTEX
    class VmaMutex
    {
    public:
        void Lock() { m_Mutex.lock(); }
        void Unlock() { m_Mutex.unlock(); }
    private:
        std::mutex m_Mutex;
    };
    #define VMA_MUTEX VmaMutex
#endif

// Read-write mutex, where "read" is shared access, "write" is exclusive access.
#ifndef VMA_RW_MUTEX
    #if VMA_USE_STL_SHARED_MUTEX
        // Use std::shared_mutex from C++17.
        #include <shared_mutex>
        class VmaRWMutex
        {
        public:
            void LockRead() { m_Mutex.lock_shared(); }
            void UnlockRead() { m_Mutex.unlock_shared(); }
            void LockWrite() { m_Mutex.lock(); }
            void UnlockWrite() { m_Mutex.unlock(); }
        private:
            std::shared_mutex m_Mutex;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #elif defined(_WIN32) && defined(WINVER) && WINVER >= 0x0600
        // Use SRWLOCK from WinAPI.
        // Minimum supported client = Windows Vista, server = Windows Server 2008.
        class VmaRWMutex
        {
        public:
            VmaRWMutex() { InitializeSRWLock(&m_Lock); }
            void LockRead() { AcquireSRWLockShared(&m_Lock); }
            void UnlockRead() { ReleaseSRWLockShared(&m_Lock); }
            void LockWrite() { AcquireSRWLockExclusive(&m_Lock); }
            void UnlockWrite() { ReleaseSRWLockExclusive(&m_Lock); }
        private:
            SRWLOCK m_Lock;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #else
        // Less efficient fallback: Use normal mutex.
        class VmaRWMutex
        {
        public:
            void LockRead() { m_Mutex.Lock(); }
            void UnlockRead() { m_Mutex.Unlock(); }
            void LockWrite() { m_Mutex.Lock(); }
            void UnlockWrite() { m_Mutex.Unlock(); }
        private:
            VMA_MUTEX m_Mutex;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #endif // #if VMA_USE_STL_SHARED_MUTEX
#endif // #ifndef VMA_RW_MUTEX

/*
If providing your own implementation, you need to implement a subset of std::atomic.
*/
#ifndef VMA_ATOMIC_UINT32
    #include <atomic>
    #define VMA_ATOMIC_UINT32 std::atomic<uint32_t>
#endif

#ifndef VMA_ATOMIC_UINT64
    #include <atomic>
    #define VMA_ATOMIC_UINT64 std::atomic<uint64_t>
#endif

#ifndef VMA_DEBUG_ALWAYS_DEDICATED_MEMORY
    /**
    Every allocation will have its own memory block.
    Define to 1 for debugging purposes only.
    */
    #define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY (0)
#endif

#ifndef VMA_DEBUG_ALIGNMENT
    /**
    Minimum alignment of all allocations, in bytes.
    Set to more than 1 for debugging purposes only. Must be power of two.
    */
    #define VMA_DEBUG_ALIGNMENT (1)
#endif

#ifndef VMA_DEBUG_MARGIN
    /**
    Minimum margin before and after every allocation, in bytes.
    Set nonzero for debugging purposes only.
    */
    #define VMA_DEBUG_MARGIN (0)
#endif

#ifndef VMA_DEBUG_INITIALIZE_ALLOCATIONS
    /**
    Define this macro to 1 to automatically fill new allocations and destroyed
    allocations with some bit pattern.
    */
    #define VMA_DEBUG_INITIALIZE_ALLOCATIONS (0)
#endif

#ifndef VMA_DEBUG_DETECT_CORRUPTION
    /**
    Define this macro to 1 together with non-zero value of VMA_DEBUG_MARGIN to
    enable writing magic value to the margin before and after every allocation and
    validating it, so that memory corruptions (out-of-bounds writes) are detected.
    */
    #define VMA_DEBUG_DETECT_CORRUPTION (0)
#endif

#ifndef VMA_DEBUG_GLOBAL_MUTEX
    /**
    Set this to 1 for debugging purposes only, to enable single mutex protecting all
    entry calls to the library. Can be useful for debugging multithreading issues.
    */
    #define VMA_DEBUG_GLOBAL_MUTEX (0)
#endif

#ifndef VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY
    /**
    Minimum value for VkPhysicalDeviceLimits::bufferImageGranularity.
    Set to more than 1 for debugging purposes only. Must be power of two.
    */
    #define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY (1)
#endif

#ifndef VMA_SMALL_HEAP_MAX_SIZE
   /// Maximum size of a memory heap in Vulkan to consider it "small".
   #define VMA_SMALL_HEAP_MAX_SIZE (1024ull * 1024 * 1024)
#endif

#ifndef VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE
   /// Default size of a block allocated as single VkDeviceMemory from a "large" heap.
   #define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (256ull * 1024 * 1024)
#endif

#ifndef VMA_CLASS_NO_COPY
    #define VMA_CLASS_NO_COPY(className) \
        private: \
            className(const className&) = delete; \
            className& operator=(const className&) = delete;
#endif

static const uint32_t VMA_FRAME_INDEX_LOST = UINT32_MAX;

// Decimal 2139416166, float NaN, little-endian binary 66 E6 84 7F.
static const uint32_t VMA_CORRUPTION_DETECTION_MAGIC_VALUE = 0x7F84E666;

static const uint8_t VMA_ALLOCATION_FILL_PATTERN_CREATED   = 0xDC;
static const uint8_t VMA_ALLOCATION_FILL_PATTERN_DESTROYED = 0xEF;

/*******************************************************************************
END OF CONFIGURATION
*/

static const uint32_t VMA_ALLOCATION_INTERNAL_STRATEGY_MIN_OFFSET = 0x10000000u;

static VkAllocationCallbacks VmaEmptyAllocationCallbacks = {
    VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL };

// Returns number of bits set to 1 in (v).
static inline uint32_t VmaCountBitsSet(uint32_t v)
{
	uint32_t c = v - ((v >> 1) & 0x55555555);
	c = ((c >>  2) & 0x33333333) + (c & 0x33333333);
	c = ((c >>  4) + c) & 0x0F0F0F0F;
	c = ((c >>  8) + c) & 0x00FF00FF;
	c = ((c >> 16) + c) & 0x0000FFFF;
	return c;
}

// Aligns given value up to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 16.
// Use types like uint32_t, uint64_t as T.
template <typename T>
static inline T VmaAlignUp(T val, T align)
{
	return (val + align - 1) / align * align;
}
// Aligns given value down to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 8.
// Use types like uint32_t, uint64_t as T.
template <typename T>
static inline T VmaAlignDown(T val, T align)
{
    return val / align * align;
}

// Division with mathematical rounding to nearest number.
template <typename T>
static inline T VmaRoundDiv(T x, T y)
{
	return (x + (y / (T)2)) / y;
}

/*
Returns true if given number is a power of two.
T must be unsigned integer number or signed integer but always nonnegative.
For 0 returns true.
*/
template <typename T>
inline bool VmaIsPow2(T x)
{
    return (x & (x-1)) == 0;
}

// Returns smallest power of 2 greater or equal to v.
static inline uint32_t VmaNextPow2(uint32_t v)
{
	v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
static inline uint64_t VmaNextPow2(uint64_t v)
{
	v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

// Returns largest power of 2 less or equal to v.
static inline uint32_t VmaPrevPow2(uint32_t v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v = v ^ (v >> 1);
    return v;
}
static inline uint64_t VmaPrevPow2(uint64_t v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v = v ^ (v >> 1);
    return v;
}

static inline bool VmaStrIsEmpty(const char* pStr)
{
    return pStr == VMA_NULL || *pStr == '\0';
}

#if VMA_STATS_STRING_ENABLED

static const char* VmaAlgorithmToStr(uint32_t algorithm)
{
    switch(algorithm)
    {
    case VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT:
        return "Linear";
    case VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT:
        return "Buddy";
    case 0:
        return "Default";
    default:
        VMA_ASSERT(0);
        return "";
    }
}

#endif // #if VMA_STATS_STRING_ENABLED

#ifndef VMA_SORT

template<typename Iterator, typename Compare>
Iterator VmaQuickSortPartition(Iterator beg, Iterator end, Compare cmp)
{
    Iterator centerValue = end; --centerValue;
    Iterator insertIndex = beg;
    for(Iterator memTypeIndex = beg; memTypeIndex < centerValue; ++memTypeIndex)
    {
        if(cmp(*memTypeIndex, *centerValue))
        {
            if(insertIndex != memTypeIndex)
            {
                VMA_SWAP(*memTypeIndex, *insertIndex);
            }
            ++insertIndex;
        }
    }
    if(insertIndex != centerValue)
    {
        VMA_SWAP(*insertIndex, *centerValue);
    }
    return insertIndex;
}

template<typename Iterator, typename Compare>
void VmaQuickSort(Iterator beg, Iterator end, Compare cmp)
{
    if(beg < end)
    {
        Iterator it = VmaQuickSortPartition<Iterator, Compare>(beg, end, cmp);
        VmaQuickSort<Iterator, Compare>(beg, it, cmp);
        VmaQuickSort<Iterator, Compare>(it + 1, end, cmp);
    }
}

#define VMA_SORT(beg, end, cmp) VmaQuickSort(beg, end, cmp)

#endif // #ifndef VMA_SORT

/*
Returns true if two memory blocks occupy overlapping pages.
ResourceA must be in less memory offset than ResourceB.

Algorithm is based on "Vulkan 1.0.39 - A Specification (with all registered Vulkan extensions)"
chapter 11.6 "Resource Memory Association", paragraph "Buffer-Image Granularity".
*/
static inline bool VmaBlocksOnSamePage(
    VkDeviceSize resourceAOffset,
    VkDeviceSize resourceASize,
    VkDeviceSize resourceBOffset,
    VkDeviceSize pageSize)
{
    VMA_ASSERT(resourceAOffset + resourceASize <= resourceBOffset && resourceASize > 0 && pageSize > 0);
    VkDeviceSize resourceAEnd = resourceAOffset + resourceASize - 1;
    VkDeviceSize resourceAEndPage = resourceAEnd & ~(pageSize - 1);
    VkDeviceSize resourceBStart = resourceBOffset;
    VkDeviceSize resourceBStartPage = resourceBStart & ~(pageSize - 1);
    return resourceAEndPage == resourceBStartPage;
}

enum VmaSuballocationType
{
    VMA_SUBALLOCATION_TYPE_FREE = 0,
    VMA_SUBALLOCATION_TYPE_UNKNOWN = 1,
    VMA_SUBALLOCATION_TYPE_BUFFER = 2,
    VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN = 3,
    VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR = 4,
    VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL = 5,
    VMA_SUBALLOCATION_TYPE_MAX_ENUM = 0x7FFFFFFF
};

/*
Returns true if given suballocation types could conflict and must respect
VkPhysicalDeviceLimits::bufferImageGranularity. They conflict if one is buffer
or linear image and another one is optimal image. If type is unknown, behave
conservatively.
*/
static inline bool VmaIsBufferImageGranularityConflict(
    VmaSuballocationType suballocType1,
    VmaSuballocationType suballocType2)
{
    if(suballocType1 > suballocType2)
    {
        VMA_SWAP(suballocType1, suballocType2);
    }
    
    switch(suballocType1)
    {
    case VMA_SUBALLOCATION_TYPE_FREE:
        return false;
    case VMA_SUBALLOCATION_TYPE_UNKNOWN:
        return true;
    case VMA_SUBALLOCATION_TYPE_BUFFER:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL:
        return false;
    default:
        VMA_ASSERT(0);
        return true;
    }
}

static void VmaWriteMagicValue(void* pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
    uint32_t* pDst = (uint32_t*)((char*)pData + offset);
    const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for(size_t i = 0; i < numberCount; ++i, ++pDst)
    {
        *pDst = VMA_CORRUPTION_DETECTION_MAGIC_VALUE;
    }
#else
    // no-op
#endif
}

static bool VmaValidateMagicValue(const void* pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
    const uint32_t* pSrc = (const uint32_t*)((const char*)pData + offset);
    const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for(size_t i = 0; i < numberCount; ++i, ++pSrc)
    {
        if(*pSrc != VMA_CORRUPTION_DETECTION_MAGIC_VALUE)
        {
            return false;
        }
    }
#endif
    return true;
}

/*
Fills structure with parameters of an example buffer to be used for transfers
during GPU memory defragmentation.
*/
static void VmaFillGpuDefragmentationBufferCreateInfo(VkBufferCreateInfo& outBufCreateInfo)
{
    memset(&outBufCreateInfo, 0, sizeof(outBufCreateInfo));
    outBufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    outBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    outBufCreateInfo.size = (VkDeviceSize)VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE; // Example size.
}

// Helper RAII class to lock a mutex in constructor and unlock it in destructor (at the end of scope).
struct VmaMutexLock
{
    VMA_CLASS_NO_COPY(VmaMutexLock)
public:
    VmaMutexLock(VMA_MUTEX& mutex, bool useMutex = true) :
        m_pMutex(useMutex ? &mutex : VMA_NULL)
    { if(m_pMutex) { m_pMutex->Lock(); } }
    ~VmaMutexLock()
    { if(m_pMutex) { m_pMutex->Unlock(); } }
private:
    VMA_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for reading.
struct VmaMutexLockRead
{
    VMA_CLASS_NO_COPY(VmaMutexLockRead)
public:
    VmaMutexLockRead(VMA_RW_MUTEX& mutex, bool useMutex) :
        m_pMutex(useMutex ? &mutex : VMA_NULL)
    { if(m_pMutex) { m_pMutex->LockRead(); } }
    ~VmaMutexLockRead() { if(m_pMutex) { m_pMutex->UnlockRead(); } }
private:
    VMA_RW_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for writing.
struct VmaMutexLockWrite
{
    VMA_CLASS_NO_COPY(VmaMutexLockWrite)
public:
    VmaMutexLockWrite(VMA_RW_MUTEX& mutex, bool useMutex) :
        m_pMutex(useMutex ? &mutex : VMA_NULL)
    { if(m_pMutex) { m_pMutex->LockWrite(); } }
    ~VmaMutexLockWrite() { if(m_pMutex) { m_pMutex->UnlockWrite(); } }
private:
    VMA_RW_MUTEX* m_pMutex;
};

#if VMA_DEBUG_GLOBAL_MUTEX
    static VMA_MUTEX gDebugGlobalMutex;
    #define VMA_DEBUG_GLOBAL_MUTEX_LOCK VmaMutexLock debugGlobalMutexLock(gDebugGlobalMutex, true);
#else
    #define VMA_DEBUG_GLOBAL_MUTEX_LOCK
#endif

// Minimum size of a free suballocation to register it in the free suballocation collection.
static const VkDeviceSize VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER = 16;

/*
Performs binary search and returns iterator to first element that is greater or
equal to (key), according to comparison (cmp).

Cmp should return true if first argument is less than second argument.

Returned value is the found element, if present in the collection or place where
new element with value (key) should be inserted.
*/
template <typename CmpLess, typename IterT, typename KeyT>
static IterT VmaBinaryFindFirstNotLess(IterT beg, IterT end, const KeyT &key, const CmpLess& cmp)
{
    size_t down = 0, up = (end - beg);
    while(down < up)
    {
        const size_t mid = (down + up) / 2;
        if(cmp(*(beg+mid), key))
        {
            down = mid + 1;
        }
        else
        {
            up = mid;
        }
    }
    return beg + down;
}

template<typename CmpLess, typename IterT, typename KeyT>
IterT VmaBinaryFindSorted(const IterT& beg, const IterT& end, const KeyT& value, const CmpLess& cmp)
{
    IterT it = VmaBinaryFindFirstNotLess<CmpLess, IterT, KeyT>(
        beg, end, value, cmp);
    if(it == end ||
        (!cmp(*it, value) && !cmp(value, *it)))
    {
        return it;
    }
    return end;
}

/*
Returns true if all pointers in the array are not-null and unique.
Warning! O(n^2) complexity. Use only inside VMA_HEAVY_ASSERT.
T must be pointer type, e.g. VmaAllocation, VmaPool.
*/
template<typename T>
static bool VmaValidatePointerArray(uint32_t count, const T* arr)
{
    for(uint32_t i = 0; i < count; ++i)
    {
        const T iPtr = arr[i];
        if(iPtr == VMA_NULL)
        {
            return false;
        }
        for(uint32_t j = i + 1; j < count; ++j)
        {
            if(iPtr == arr[j])
            {
                return false;
            }
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Memory allocation

static void* VmaMalloc(const VkAllocationCallbacks* pAllocationCallbacks, size_t size, size_t alignment)
{
    if((pAllocationCallbacks != VMA_NULL) &&
        (pAllocationCallbacks->pfnAllocation != VMA_NULL))
    {
        return (*pAllocationCallbacks->pfnAllocation)(
            pAllocationCallbacks->pUserData,
            size,
            alignment,
            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    else
    {
        return VMA_SYSTEM_ALIGNED_MALLOC(size, alignment);
    }
}

static void VmaFree(const VkAllocationCallbacks* pAllocationCallbacks, void* ptr)
{
    if((pAllocationCallbacks != VMA_NULL) &&
        (pAllocationCallbacks->pfnFree != VMA_NULL))
    {
        (*pAllocationCallbacks->pfnFree)(pAllocationCallbacks->pUserData, ptr);
    }
    else
    {
        VMA_SYSTEM_FREE(ptr);
    }
}

template<typename T>
static T* VmaAllocate(const VkAllocationCallbacks* pAllocationCallbacks)
{
    return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T), VMA_ALIGN_OF(T));
}

template<typename T>
static T* VmaAllocateArray(const VkAllocationCallbacks* pAllocationCallbacks, size_t count)
{
    return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T) * count, VMA_ALIGN_OF(T));
}

#define vma_new(allocator, type)   new(VmaAllocate<type>(allocator))(type)

#define vma_new_array(allocator, type, count)   new(VmaAllocateArray<type>((allocator), (count)))(type)

template<typename T>
static void vma_delete(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr)
{
    ptr->~T();
    VmaFree(pAllocationCallbacks, ptr);
}

template<typename T>
static void vma_delete_array(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr, size_t count)
{
    if(ptr != VMA_NULL)
    {
        for(size_t i = count; i--; )
        {
            ptr[i].~T();
        }
        VmaFree(pAllocationCallbacks, ptr);
    }
}

static char* VmaCreateStringCopy(const VkAllocationCallbacks* allocs, const char* srcStr)
{
    if(srcStr != VMA_NULL)
    {
        const size_t len = strlen(srcStr);
        char* const result = vma_new_array(allocs, char, len + 1);
        memcpy(result, srcStr, len + 1);
        return result;
    }
    else
    {
        return VMA_NULL;
    }
}

static void VmaFreeString(const VkAllocationCallbacks* allocs, char* str)
{
    if(str != VMA_NULL)
    {
        const size_t len = strlen(str);
        vma_delete_array(allocs, str, len + 1);
    }
}

// STL-compatible allocator.
template<typename T>
class VmaStlAllocator
{
public:
    const VkAllocationCallbacks* const m_pCallbacks;
    typedef T value_type;
    
    VmaStlAllocator(const VkAllocationCallbacks* pCallbacks) : m_pCallbacks(pCallbacks) { }
    template<typename U> VmaStlAllocator(const VmaStlAllocator<U>& src) : m_pCallbacks(src.m_pCallbacks) { }

    T* allocate(size_t n) { return VmaAllocateArray<T>(m_pCallbacks, n); }
    void deallocate(T* p, size_t n) { VmaFree(m_pCallbacks, p); }

    template<typename U>
    bool operator==(const VmaStlAllocator<U>& rhs) const
    {
        return m_pCallbacks == rhs.m_pCallbacks;
    }
    template<typename U>
    bool operator!=(const VmaStlAllocator<U>& rhs) const
    {
        return m_pCallbacks != rhs.m_pCallbacks;
    }

    VmaStlAllocator& operator=(const VmaStlAllocator& x) = delete;
};

#if VMA_USE_STL_VECTOR

#define VmaVector std::vector

template<typename T, typename allocatorT>
static void VmaVectorInsert(std::vector<T, allocatorT>& vec, size_t index, const T& item)
{
    vec.insert(vec.begin() + index, item);
}

template<typename T, typename allocatorT>
static void VmaVectorRemove(std::vector<T, allocatorT>& vec, size_t index)
{
    vec.erase(vec.begin() + index);
}

#else // #if VMA_USE_STL_VECTOR

/* Class with interface compatible with subset of std::vector.
T must be POD because constructors and destructors are not called and memcpy is
used for these objects. */
template<typename T, typename AllocatorT>
class VmaVector
{
public:
    typedef T value_type;

    VmaVector(const AllocatorT& allocator) :
        m_Allocator(allocator),
        m_pArray(VMA_NULL),
        m_Count(0),
        m_Capacity(0)
    {
    }

    VmaVector(size_t count, const AllocatorT& allocator) :
        m_Allocator(allocator),
        m_pArray(count ? (T*)VmaAllocateArray<T>(allocator.m_pCallbacks, count) : VMA_NULL),
        m_Count(count),
        m_Capacity(count)
    {
    }
    
    // This version of the constructor is here for compatibility with pre-C++14 std::vector.
    // value is unused.
    VmaVector(size_t count, const T& value, const AllocatorT& allocator)
        : VmaVector(count, allocator) {}
    
    VmaVector(const VmaVector<T, AllocatorT>& src) :
        m_Allocator(src.m_Allocator),
        m_pArray(src.m_Count ? (T*)VmaAllocateArray<T>(src.m_Allocator.m_pCallbacks, src.m_Count) : VMA_NULL),
        m_Count(src.m_Count),
        m_Capacity(src.m_Count)
    {
        if(m_Count != 0)
        {
            memcpy(m_pArray, src.m_pArray, m_Count * sizeof(T));
        }
    }
    
    ~VmaVector()
    {
        VmaFree(m_Allocator.m_pCallbacks, m_pArray);
    }

    VmaVector& operator=(const VmaVector<T, AllocatorT>& rhs)
    {
        if(&rhs != this)
        {
            resize(rhs.m_Count);
            if(m_Count != 0)
            {
                memcpy(m_pArray, rhs.m_pArray, m_Count * sizeof(T));
            }
        }
        return *this;
    }
    
    bool empty() const { return m_Count == 0; }
    size_t size() const { return m_Count; }
    T* data() { return m_pArray; }
    const T* data() const { return m_pArray; }
    
    T& operator[](size_t index)
    {
        VMA_HEAVY_ASSERT(index < m_Count);
        return m_pArray[index];
    }
    const T& operator[](size_t index) const
    {
        VMA_HEAVY_ASSERT(index < m_Count);
        return m_pArray[index];
    }

    T& front()
    {
        VMA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[0];
    }
    const T& front() const
    {
        VMA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[0];
    }
    T& back()
    {
        VMA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[m_Count - 1];
    }
    const T& back() const
    {
        VMA_HEAVY_ASSERT(m_Count > 0);
        return m_pArray[m_Count - 1];
    }

    void reserve(size_t newCapacity, bool freeMemory = false)
    {
        newCapacity = VMA_MAX(newCapacity, m_Count);
        
        if((newCapacity < m_Capacity) && !freeMemory)
        {
            newCapacity = m_Capacity;
        }
        
        if(newCapacity != m_Capacity)
        {
            T* const newArray = newCapacity ? VmaAllocateArray<T>(m_Allocator, newCapacity) : VMA_NULL;
            if(m_Count != 0)
            {
                memcpy(newArray, m_pArray, m_Count * sizeof(T));
            }
            VmaFree(m_Allocator.m_pCallbacks, m_pArray);
            m_Capacity = newCapacity;
            m_pArray = newArray;
        }
    }

    void resize(size_t newCount, bool freeMemory = false)
    {
        size_t newCapacity = m_Capacity;
        if(newCount > m_Capacity)
        {
            newCapacity = VMA_MAX(newCount, VMA_MAX(m_Capacity * 3 / 2, (size_t)8));
        }
        else if(freeMemory)
        {
            newCapacity = newCount;
        }

        if(newCapacity != m_Capacity)
        {
            T* const newArray = newCapacity ? VmaAllocateArray<T>(m_Allocator.m_pCallbacks, newCapacity) : VMA_NULL;
            const size_t elementsToCopy = VMA_MIN(m_Count, newCount);
            if(elementsToCopy != 0)
            {
                memcpy(newArray, m_pArray, elementsToCopy * sizeof(T));
            }
            VmaFree(m_Allocator.m_pCallbacks, m_pArray);
            m_Capacity = newCapacity;
            m_pArray = newArray;
        }

        m_Count = newCount;
    }

    void clear(bool freeMemory = false)
    {
        resize(0, freeMemory);
    }

    void insert(size_t index, const T& src)
    {
        VMA_HEAVY_ASSERT(index <= m_Count);
        const size_t oldCount = size();
        resize(oldCount + 1);
        if(index < oldCount)
        {
            memmove(m_pArray + (index + 1), m_pArray + index, (oldCount - index) * sizeof(T));
        }
        m_pArray[index] = src;
    }

    void remove(size_t index)
    {
        VMA_HEAVY_ASSERT(index < m_Count);
        const size_t oldCount = size();
        if(index < oldCount - 1)
        {
            memmove(m_pArray + index, m_pArray + (index + 1), (oldCount - index - 1) * sizeof(T));
        }
        resize(oldCount - 1);
    }

    void push_back(const T& src)
    {
        const size_t newIndex = size();
        resize(newIndex + 1);
        m_pArray[newIndex] = src;
    }

    void pop_back()
    {
        VMA_HEAVY_ASSERT(m_Count > 0);
        resize(size() - 1);
    }

    void push_front(const T& src)
    {
        insert(0, src);
    }

    void pop_front()
    {
        VMA_HEAVY_ASSERT(m_Count > 0);
        remove(0);
    }

    typedef T* iterator;

    iterator begin() { return m_pArray; }
    iterator end() { return m_pArray + m_Count; }

private:
    AllocatorT m_Allocator;
    T* m_pArray;
    size_t m_Count;
    size_t m_Capacity;
};

template<typename T, typename allocatorT>
static void VmaVectorInsert(VmaVector<T, allocatorT>& vec, size_t index, const T& item)
{
    vec.insert(index, item);
}

template<typename T, typename allocatorT>
static void VmaVectorRemove(VmaVector<T, allocatorT>& vec, size_t index)
{
    vec.remove(index);
}

#endif // #if VMA_USE_STL_VECTOR

template<typename CmpLess, typename VectorT>
size_t VmaVectorInsertSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    const size_t indexToInsert = VmaBinaryFindFirstNotLess(
        vector.data(),
        vector.data() + vector.size(),
        value,
        CmpLess()) - vector.data();
    VmaVectorInsert(vector, indexToInsert, value);
    return indexToInsert;
}

template<typename CmpLess, typename VectorT>
bool VmaVectorRemoveSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    CmpLess comparator;
    typename VectorT::iterator it = VmaBinaryFindFirstNotLess(
        vector.begin(),
        vector.end(),
        value,
        comparator);
    if((it != vector.end()) && !comparator(*it, value) && !comparator(value, *it))
    {
        size_t indexToRemove = it - vector.begin();
        VmaVectorRemove(vector, indexToRemove);
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////
// class VmaPoolAllocator

/*
Allocator for objects of type T using a list of arrays (pools) to speed up
allocation. Number of elements that can be allocated is not bounded because
allocator can create multiple blocks.
*/
template<typename T>
class VmaPoolAllocator
{
    VMA_CLASS_NO_COPY(VmaPoolAllocator)
public:
    VmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity);
    ~VmaPoolAllocator();
    T* Alloc();
    void Free(T* ptr);

private:
    union Item
    {
        uint32_t NextFreeIndex;
        alignas(T) char Value[sizeof(T)];
    };

    struct ItemBlock
    {
        Item* pItems;
        uint32_t Capacity;
        uint32_t FirstFreeIndex;
    };
    
    const VkAllocationCallbacks* m_pAllocationCallbacks;
    const uint32_t m_FirstBlockCapacity;
    VmaVector< ItemBlock, VmaStlAllocator<ItemBlock> > m_ItemBlocks;

    ItemBlock& CreateNewBlock();
};

template<typename T>
VmaPoolAllocator<T>::VmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity) :
    m_pAllocationCallbacks(pAllocationCallbacks),
    m_FirstBlockCapacity(firstBlockCapacity),
    m_ItemBlocks(VmaStlAllocator<ItemBlock>(pAllocationCallbacks))
{
    VMA_ASSERT(m_FirstBlockCapacity > 1);
}

template<typename T>
VmaPoolAllocator<T>::~VmaPoolAllocator()
{
    for(size_t i = m_ItemBlocks.size(); i--; )
        vma_delete_array(m_pAllocationCallbacks, m_ItemBlocks[i].pItems, m_ItemBlocks[i].Capacity);
    m_ItemBlocks.clear();
}

template<typename T>
T* VmaPoolAllocator<T>::Alloc()
{
    for(size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];
        // This block has some free items: Use first one.
        if(block.FirstFreeIndex != UINT32_MAX)
        {
            Item* const pItem = &block.pItems[block.FirstFreeIndex];
            block.FirstFreeIndex = pItem->NextFreeIndex;
            T* result = (T*)&pItem->Value;
            new(result)T(); // Explicit constructor call.
            return result;
        }
    }

    // No block has free item: Create new one and use it.
    ItemBlock& newBlock = CreateNewBlock();
    Item* const pItem = &newBlock.pItems[0];
    newBlock.FirstFreeIndex = pItem->NextFreeIndex;
    T* result = (T*)&pItem->Value;
    new(result)T(); // Explicit constructor call.
    return result;
}

template<typename T>
void VmaPoolAllocator<T>::Free(T* ptr)
{
    // Search all memory blocks to find ptr.
    for(size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];
        
        // Casting to union.
        Item* pItemPtr;
        memcpy(&pItemPtr, &ptr, sizeof(pItemPtr));
        
        // Check if pItemPtr is in address range of this block.
        if((pItemPtr >= block.pItems) && (pItemPtr < block.pItems + block.Capacity))
        {
            ptr->~T(); // Explicit destructor call.
            const uint32_t index = static_cast<uint32_t>(pItemPtr - block.pItems);
            pItemPtr->NextFreeIndex = block.FirstFreeIndex;
            block.FirstFreeIndex = index;
            return;
        }
    }
    VMA_ASSERT(0 && "Pointer doesn't belong to this memory pool.");
}

template<typename T>
typename VmaPoolAllocator<T>::ItemBlock& VmaPoolAllocator<T>::CreateNewBlock()
{
    const uint32_t newBlockCapacity = m_ItemBlocks.empty() ?
        m_FirstBlockCapacity : m_ItemBlocks.back().Capacity * 3 / 2;

    const ItemBlock newBlock = {
        vma_new_array(m_pAllocationCallbacks, Item, newBlockCapacity),
        newBlockCapacity,
        0 };

    m_ItemBlocks.push_back(newBlock);

    // Setup singly-linked list of all free items in this block.
    for(uint32_t i = 0; i < newBlockCapacity - 1; ++i)
        newBlock.pItems[i].NextFreeIndex = i + 1;
    newBlock.pItems[newBlockCapacity - 1].NextFreeIndex = UINT32_MAX;
    return m_ItemBlocks.back();
}

////////////////////////////////////////////////////////////////////////////////
// class VmaRawList, VmaList

#if VMA_USE_STL_LIST

#define VmaList std::list

#else // #if VMA_USE_STL_LIST

template<typename T>
struct VmaListItem
{
    VmaListItem* pPrev;
    VmaListItem* pNext;
    T Value;
};

// Doubly linked list.
template<typename T>
class VmaRawList
{
    VMA_CLASS_NO_COPY(VmaRawList)
public:
    typedef VmaListItem<T> ItemType;

    VmaRawList(const VkAllocationCallbacks* pAllocationCallbacks);
    ~VmaRawList();
    void Clear();

    size_t GetCount() const { return m_Count; }
    bool IsEmpty() const { return m_Count == 0; }

    ItemType* Front() { return m_pFront; }
    const ItemType* Front() const { return m_pFront; }
    ItemType* Back() { return m_pBack; }
    const ItemType* Back() const { return m_pBack; }

    ItemType* PushBack();
    ItemType* PushFront();
    ItemType* PushBack(const T& value);
    ItemType* PushFront(const T& value);
    void PopBack();
    void PopFront();
    
    // Item can be null - it means PushBack.
    ItemType* InsertBefore(ItemType* pItem);
    // Item can be null - it means PushFront.
    ItemType* InsertAfter(ItemType* pItem);

    ItemType* InsertBefore(ItemType* pItem, const T& value);
    ItemType* InsertAfter(ItemType* pItem, const T& value);

    void Remove(ItemType* pItem);

private:
    const VkAllocationCallbacks* const m_pAllocationCallbacks;
    VmaPoolAllocator<ItemType> m_ItemAllocator;
    ItemType* m_pFront;
    ItemType* m_pBack;
    size_t m_Count;
};

template<typename T>
VmaRawList<T>::VmaRawList(const VkAllocationCallbacks* pAllocationCallbacks) :
    m_pAllocationCallbacks(pAllocationCallbacks),
    m_ItemAllocator(pAllocationCallbacks, 128),
    m_pFront(VMA_NULL),
    m_pBack(VMA_NULL),
    m_Count(0)
{
}

template<typename T>
VmaRawList<T>::~VmaRawList()
{
    // Intentionally not calling Clear, because that would be unnecessary
    // computations to return all items to m_ItemAllocator as free.
}

template<typename T>
void VmaRawList<T>::Clear()
{
    if(IsEmpty() == false)
    {
        ItemType* pItem = m_pBack;
        while(pItem != VMA_NULL)
        {
            ItemType* const pPrevItem = pItem->pPrev;
            m_ItemAllocator.Free(pItem);
            pItem = pPrevItem;
        }
        m_pFront = VMA_NULL;
        m_pBack = VMA_NULL;
        m_Count = 0;
    }
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushBack()
{
    ItemType* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pNext = VMA_NULL;
    if(IsEmpty())
    {
        pNewItem->pPrev = VMA_NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pPrev = m_pBack;
        m_pBack->pNext = pNewItem;
        m_pBack = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushFront()
{
    ItemType* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pPrev = VMA_NULL;
    if(IsEmpty())
    {
        pNewItem->pNext = VMA_NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pNext = m_pFront;
        m_pFront->pPrev = pNewItem;
        m_pFront = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushBack(const T& value)
{
    ItemType* const pNewItem = PushBack();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushFront(const T& value)
{
    ItemType* const pNewItem = PushFront();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
void VmaRawList<T>::PopBack()
{
    VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const pBackItem = m_pBack;
    ItemType* const pPrevItem = pBackItem->pPrev;
    if(pPrevItem != VMA_NULL)
    {
        pPrevItem->pNext = VMA_NULL;
    }
    m_pBack = pPrevItem;
    m_ItemAllocator.Free(pBackItem);
    --m_Count;
}

template<typename T>
void VmaRawList<T>::PopFront()
{
    VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const pFrontItem = m_pFront;
    ItemType* const pNextItem = pFrontItem->pNext;
    if(pNextItem != VMA_NULL)
    {
        pNextItem->pPrev = VMA_NULL;
    }
    m_pFront = pNextItem;
    m_ItemAllocator.Free(pFrontItem);
    --m_Count;
}

template<typename T>
void VmaRawList<T>::Remove(ItemType* pItem)
{
    VMA_HEAVY_ASSERT(pItem != VMA_NULL);
    VMA_HEAVY_ASSERT(m_Count > 0);

    if(pItem->pPrev != VMA_NULL)
    {
        pItem->pPrev->pNext = pItem->pNext;
    }
    else
    {
        VMA_HEAVY_ASSERT(m_pFront == pItem);
        m_pFront = pItem->pNext;
    }

    if(pItem->pNext != VMA_NULL)
    {
        pItem->pNext->pPrev = pItem->pPrev;
    }
    else
    {
        VMA_HEAVY_ASSERT(m_pBack == pItem);
        m_pBack = pItem->pPrev;
    }

    m_ItemAllocator.Free(pItem);
    --m_Count;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertBefore(ItemType* pItem)
{
    if(pItem != VMA_NULL)
    {
        ItemType* const prevItem = pItem->pPrev;
        ItemType* const newItem = m_ItemAllocator.Alloc();
        newItem->pPrev = prevItem;
        newItem->pNext = pItem;
        pItem->pPrev = newItem;
        if(prevItem != VMA_NULL)
        {
            prevItem->pNext = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(m_pFront == pItem);
            m_pFront = newItem;
        }
        ++m_Count;
        return newItem;
    }
    else
        return PushBack();
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertAfter(ItemType* pItem)
{
    if(pItem != VMA_NULL)
    {
        ItemType* const nextItem = pItem->pNext;
        ItemType* const newItem = m_ItemAllocator.Alloc();
        newItem->pNext = nextItem;
        newItem->pPrev = pItem;
        pItem->pNext = newItem;
        if(nextItem != VMA_NULL)
        {
            nextItem->pPrev = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(m_pBack == pItem);
            m_pBack = newItem;
        }
        ++m_Count;
        return newItem;
    }
    else
        return PushFront();
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertBefore(ItemType* pItem, const T& value)
{
    ItemType* const newItem = InsertBefore(pItem);
    newItem->Value = value;
    return newItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertAfter(ItemType* pItem, const T& value)
{
    ItemType* const newItem = InsertAfter(pItem);
    newItem->Value = value;
    return newItem;
}

template<typename T, typename AllocatorT>
class VmaList
{
    VMA_CLASS_NO_COPY(VmaList)
public:
    class iterator
    {
    public:
        iterator() :
            m_pList(VMA_NULL),
            m_pItem(VMA_NULL)
        {
        }

        T& operator*() const
        {
            VMA_HEAVY_ASSERT(m_pItem != VMA_NULL);
            return m_pItem->Value;
        }
        T* operator->() const
        {
            VMA_HEAVY_ASSERT(m_pItem != VMA_NULL);
            return &m_pItem->Value;
        }

        iterator& operator++()
        {
            VMA_HEAVY_ASSERT(m_pItem != VMA_NULL);
            m_pItem = m_pItem->pNext;
            return *this;
        }
        iterator& operator--()
        {
            if(m_pItem != VMA_NULL)
            {
                m_pItem = m_pItem->pPrev;
            }
            else
            {
                VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
                m_pItem = m_pList->Back();
            }
            return *this;
        }

        iterator operator++(int)
        {
            iterator result = *this;
            ++*this;
            return result;
        }
        iterator operator--(int)
        {
            iterator result = *this;
            --*this;
            return result;
        }

        bool operator==(const iterator& rhs) const
        {
            VMA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem == rhs.m_pItem;
        }
        bool operator!=(const iterator& rhs) const
        {
            VMA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem != rhs.m_pItem;
        }
        
    private:
        VmaRawList<T>* m_pList;
        VmaListItem<T>* m_pItem;

        iterator(VmaRawList<T>* pList, VmaListItem<T>* pItem) :
            m_pList(pList),
            m_pItem(pItem)
        {
        }

        friend class VmaList<T, AllocatorT>;
    };

    class const_iterator
    {
    public:
        const_iterator() :
            m_pList(VMA_NULL),
            m_pItem(VMA_NULL)
        {
        }

        const_iterator(const iterator& src) :
            m_pList(src.m_pList),
            m_pItem(src.m_pItem)
        {
        }
        
        const T& operator*() const
        {
            VMA_HEAVY_ASSERT(m_pItem != VMA_NULL);
            return m_pItem->Value;
        }
        const T* operator->() const
        {
            VMA_HEAVY_ASSERT(m_pItem != VMA_NULL);
            return &m_pItem->Value;
        }

        const_iterator& operator++()
        {
            VMA_HEAVY_ASSERT(m_pItem != VMA_NULL);
            m_pItem = m_pItem->pNext;
            return *this;
        }
        const_iterator& operator--()
        {
            if(m_pItem != VMA_NULL)
            {
                m_pItem = m_pItem->pPrev;
            }
            else
            {
                VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
                m_pItem = m_pList->Back();
            }
            return *this;
        }

        const_iterator operator++(int)
        {
            const_iterator result = *this;
            ++*this;
            return result;
        }
        const_iterator operator--(int)
        {
            const_iterator result = *this;
            --*this;
            return result;
        }

        bool operator==(const const_iterator& rhs) const
        {
            VMA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem == rhs.m_pItem;
        }
        bool operator!=(const const_iterator& rhs) const
        {
            VMA_HEAVY_ASSERT(m_pList == rhs.m_pList);
            return m_pItem != rhs.m_pItem;
        }
        
    private:
        const_iterator(const VmaRawList<T>* pList, const VmaListItem<T>* pItem) :
            m_pList(pList),
            m_pItem(pItem)
        {
        }

        const VmaRawList<T>* m_pList;
        const VmaListItem<T>* m_pItem;

        friend class VmaList<T, AllocatorT>;
    };

    VmaList(const AllocatorT& allocator) : m_RawList(allocator.m_pCallbacks) { }

    bool empty() const { return m_RawList.IsEmpty(); }
    size_t size() const { return m_RawList.GetCount(); }

    iterator begin() { return iterator(&m_RawList, m_RawList.Front()); }
    iterator end() { return iterator(&m_RawList, VMA_NULL); }

    const_iterator cbegin() const { return const_iterator(&m_RawList, m_RawList.Front()); }
    const_iterator cend() const { return const_iterator(&m_RawList, VMA_NULL); }

    void clear() { m_RawList.Clear(); }
    void push_back(const T& value) { m_RawList.PushBack(value); }
    void erase(iterator it) { m_RawList.Remove(it.m_pItem); }
    iterator insert(iterator it, const T& value) { return iterator(&m_RawList, m_RawList.InsertBefore(it.m_pItem, value)); }

private:
    VmaRawList<T> m_RawList;
};

#endif // #if VMA_USE_STL_LIST

////////////////////////////////////////////////////////////////////////////////
// class VmaMap

// Unused in this version.
#if 0

#if VMA_USE_STL_UNORDERED_MAP

#define VmaPair std::pair

#define VMA_MAP_TYPE(KeyT, ValueT) \
    std::unordered_map< KeyT, ValueT, std::hash<KeyT>, std::equal_to<KeyT>, VmaStlAllocator< std::pair<KeyT, ValueT> > >

#else // #if VMA_USE_STL_UNORDERED_MAP

template<typename T1, typename T2>
struct VmaPair
{
    T1 first;
    T2 second;

    VmaPair() : first(), second() { }
    VmaPair(const T1& firstSrc, const T2& secondSrc) : first(firstSrc), second(secondSrc) { }
};

/* Class compatible with subset of interface of std::unordered_map.
KeyT, ValueT must be POD because they will be stored in VmaVector.
*/
template<typename KeyT, typename ValueT>
class VmaMap
{
public:
    typedef VmaPair<KeyT, ValueT> PairType;
    typedef PairType* iterator;

    VmaMap(const VmaStlAllocator<PairType>& allocator) : m_Vector(allocator) { }

    iterator begin() { return m_Vector.begin(); }
    iterator end() { return m_Vector.end(); }

    void insert(const PairType& pair);
    iterator find(const KeyT& key);
    void erase(iterator it);
    
private:
    VmaVector< PairType, VmaStlAllocator<PairType> > m_Vector;
};

#define VMA_MAP_TYPE(KeyT, ValueT) VmaMap<KeyT, ValueT>

template<typename FirstT, typename SecondT>
struct VmaPairFirstLess
{
    bool operator()(const VmaPair<FirstT, SecondT>& lhs, const VmaPair<FirstT, SecondT>& rhs) const
    {
        return lhs.first < rhs.first;
    }
    bool operator()(const VmaPair<FirstT, SecondT>& lhs, const FirstT& rhsFirst) const
    {
        return lhs.first < rhsFirst;
    }
};

template<typename KeyT, typename ValueT>
void VmaMap<KeyT, ValueT>::insert(const PairType& pair)
{
    const size_t indexToInsert = VmaBinaryFindFirstNotLess(
        m_Vector.data(),
        m_Vector.data() + m_Vector.size(),
        pair,
        VmaPairFirstLess<KeyT, ValueT>()) - m_Vector.data();
    VmaVectorInsert(m_Vector, indexToInsert, pair);
}

template<typename KeyT, typename ValueT>
VmaPair<KeyT, ValueT>* VmaMap<KeyT, ValueT>::find(const KeyT& key)
{
    PairType* it = VmaBinaryFindFirstNotLess(
        m_Vector.data(),
        m_Vector.data() + m_Vector.size(),
        key,
        VmaPairFirstLess<KeyT, ValueT>());
    if((it != m_Vector.end()) && (it->first == key))
    {
        return it;
    }
    else
    {
        return m_Vector.end();
    }
}

template<typename KeyT, typename ValueT>
void VmaMap<KeyT, ValueT>::erase(iterator it)
{
    VmaVectorRemove(m_Vector, it - m_Vector.begin());
}

#endif // #if VMA_USE_STL_UNORDERED_MAP

#endif // #if 0

////////////////////////////////////////////////////////////////////////////////

class VmaDeviceMemoryBlock;

enum VMA_CACHE_OPERATION { VMA_CACHE_FLUSH, VMA_CACHE_INVALIDATE };

struct VmaAllocation_T
{
private:
    static const uint8_t MAP_COUNT_FLAG_PERSISTENT_MAP = 0x80;

    enum FLAGS
    {
        FLAG_USER_DATA_STRING = 0x01,
    };

public:
    enum ALLOCATION_TYPE
    {
        ALLOCATION_TYPE_NONE,
        ALLOCATION_TYPE_BLOCK,
        ALLOCATION_TYPE_DEDICATED,
    };

    /*
    This struct is allocated using VmaPoolAllocator.
    */

    void Ctor(uint32_t currentFrameIndex, bool userDataString)
    {
        m_Alignment = 1;
        m_Size = 0;
        m_MemoryTypeIndex = 0;
        m_pUserData = VMA_NULL;
        m_LastUseFrameIndex = currentFrameIndex;
        m_Type = (uint8_t)ALLOCATION_TYPE_NONE;
        m_SuballocationType = (uint8_t)VMA_SUBALLOCATION_TYPE_UNKNOWN;
        m_MapCount = 0;
        m_Flags = userDataString ? (uint8_t)FLAG_USER_DATA_STRING : 0;

#if VMA_STATS_STRING_ENABLED
        m_CreationFrameIndex = currentFrameIndex;
        m_BufferImageUsage = 0;
#endif
    }

    void Dtor()
    {
        VMA_ASSERT((m_MapCount & ~MAP_COUNT_FLAG_PERSISTENT_MAP) == 0 && "Allocation was not unmapped before destruction.");

        // Check if owned string was freed.
        VMA_ASSERT(m_pUserData == VMA_NULL);
    }

    void InitBlockAllocation(
        VmaDeviceMemoryBlock* block,
        VkDeviceSize offset,
        VkDeviceSize alignment,
        VkDeviceSize size,
        uint32_t memoryTypeIndex,
        VmaSuballocationType suballocationType,
        bool mapped,
        bool canBecomeLost)
    {
        VMA_ASSERT(m_Type == ALLOCATION_TYPE_NONE);
        VMA_ASSERT(block != VMA_NULL);
        m_Type = (uint8_t)ALLOCATION_TYPE_BLOCK;
        m_Alignment = alignment;
        m_Size = size;
        m_MemoryTypeIndex = memoryTypeIndex;
        m_MapCount = mapped ? MAP_COUNT_FLAG_PERSISTENT_MAP : 0;
        m_SuballocationType = (uint8_t)suballocationType;
        m_BlockAllocation.m_Block = block;
        m_BlockAllocation.m_Offset = offset;
        m_BlockAllocation.m_CanBecomeLost = canBecomeLost;
    }

    void InitLost()
    {
        VMA_ASSERT(m_Type == ALLOCATION_TYPE_NONE);
        VMA_ASSERT(m_LastUseFrameIndex.load() == VMA_FRAME_INDEX_LOST);
        m_Type = (uint8_t)ALLOCATION_TYPE_BLOCK;
        m_MemoryTypeIndex = 0;
        m_BlockAllocation.m_Block = VMA_NULL;
        m_BlockAllocation.m_Offset = 0;
        m_BlockAllocation.m_CanBecomeLost = true;
    }

    void ChangeBlockAllocation(
        VmaAllocator hAllocator,
        VmaDeviceMemoryBlock* block,
        VkDeviceSize offset); 

    void ChangeOffset(VkDeviceSize newOffset);

    // pMappedData not null means allocation is created with MAPPED flag.
    void InitDedicatedAllocation(
        uint32_t memoryTypeIndex,
        VkDeviceMemory hMemory,
        VmaSuballocationType suballocationType,
        void* pMappedData,
        VkDeviceSize size)
    {
        VMA_ASSERT(m_Type == ALLOCATION_TYPE_NONE);
        VMA_ASSERT(hMemory != VK_NULL_HANDLE);
        m_Type = (uint8_t)ALLOCATION_TYPE_DEDICATED;
        m_Alignment = 0;
        m_Size = size;
        m_MemoryTypeIndex = memoryTypeIndex;
        m_SuballocationType = (uint8_t)suballocationType;
        m_MapCount = (pMappedData != VMA_NULL) ? MAP_COUNT_FLAG_PERSISTENT_MAP : 0;
        m_DedicatedAllocation.m_hMemory = hMemory;
        m_DedicatedAllocation.m_pMappedData = pMappedData;
    }

    ALLOCATION_TYPE GetType() const { return (ALLOCATION_TYPE)m_Type; }
    VkDeviceSize GetAlignment() const { return m_Alignment; }
    VkDeviceSize GetSize() const { return m_Size; }
    bool IsUserDataString() const { return (m_Flags & FLAG_USER_DATA_STRING) != 0; }
    void* GetUserData() const { return m_pUserData; }
    void SetUserData(VmaAllocator hAllocator, void* pUserData);
    VmaSuballocationType GetSuballocationType() const { return (VmaSuballocationType)m_SuballocationType; }

    VmaDeviceMemoryBlock* GetBlock() const
    {
        VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK);
        return m_BlockAllocation.m_Block;
    }
    VkDeviceSize GetOffset() const;
    VkDeviceMemory GetMemory() const;
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    bool IsPersistentMap() const { return (m_MapCount & MAP_COUNT_FLAG_PERSISTENT_MAP) != 0; }
    void* GetMappedData() const;
    bool CanBecomeLost() const;
    
    uint32_t GetLastUseFrameIndex() const
    {
        return m_LastUseFrameIndex.load();
    }
    bool CompareExchangeLastUseFrameIndex(uint32_t& expected, uint32_t desired)
    {
        return m_LastUseFrameIndex.compare_exchange_weak(expected, desired);
    }
    /*
    - If hAllocation.LastUseFrameIndex + frameInUseCount < allocator.CurrentFrameIndex,
      makes it lost by setting LastUseFrameIndex = VMA_FRAME_INDEX_LOST and returns true.
    - Else, returns false.
    
    If hAllocation is already lost, assert - you should not call it then.
    If hAllocation was not created with CAN_BECOME_LOST_BIT, assert.
    */
    bool MakeLost(uint32_t currentFrameIndex, uint32_t frameInUseCount);

    void DedicatedAllocCalcStatsInfo(VmaStatInfo& outInfo)
    {
        VMA_ASSERT(m_Type == ALLOCATION_TYPE_DEDICATED);
        outInfo.blockCount = 1;
        outInfo.allocationCount = 1;
        outInfo.unusedRangeCount = 0;
        outInfo.usedBytes = m_Size;
        outInfo.unusedBytes = 0;
        outInfo.allocationSizeMin = outInfo.allocationSizeMax = m_Size;
        outInfo.unusedRangeSizeMin = UINT64_MAX;
        outInfo.unusedRangeSizeMax = 0;
    }

    void BlockAllocMap();
    void BlockAllocUnmap();
    VkResult DedicatedAllocMap(VmaAllocator hAllocator, void** ppData);
    void DedicatedAllocUnmap(VmaAllocator hAllocator);

#if VMA_STATS_STRING_ENABLED
    uint32_t GetCreationFrameIndex() const { return m_CreationFrameIndex; }
    uint32_t GetBufferImageUsage() const { return m_BufferImageUsage; }

    void InitBufferImageUsage(uint32_t bufferImageUsage)
    {
        VMA_ASSERT(m_BufferImageUsage == 0);
        m_BufferImageUsage = bufferImageUsage;
    }

    void PrintParameters(class VmaJsonWriter& json) const;
#endif

private:
    VkDeviceSize m_Alignment;
    VkDeviceSize m_Size;
    void* m_pUserData;
    VMA_ATOMIC_UINT32 m_LastUseFrameIndex;
    uint32_t m_MemoryTypeIndex;
    uint8_t m_Type; // ALLOCATION_TYPE
    uint8_t m_SuballocationType; // VmaSuballocationType
    // Bit 0x80 is set when allocation was created with VMA_ALLOCATION_CREATE_MAPPED_BIT.
    // Bits with mask 0x7F are reference counter for vmaMapMemory()/vmaUnmapMemory().
    uint8_t m_MapCount;
    uint8_t m_Flags; // enum FLAGS

    // Allocation out of VmaDeviceMemoryBlock.
    struct BlockAllocation
    {
        VmaDeviceMemoryBlock* m_Block;
        VkDeviceSize m_Offset;
        bool m_CanBecomeLost;
    };

    // Allocation for an object that has its own private VkDeviceMemory.
    struct DedicatedAllocation
    {
        VkDeviceMemory m_hMemory;
        void* m_pMappedData; // Not null means memory is mapped.
    };

    union
    {
        // Allocation out of VmaDeviceMemoryBlock.
        BlockAllocation m_BlockAllocation;
        // Allocation for an object that has its own private VkDeviceMemory.
        DedicatedAllocation m_DedicatedAllocation;
    };

#if VMA_STATS_STRING_ENABLED
    uint32_t m_CreationFrameIndex;
    uint32_t m_BufferImageUsage; // 0 if unknown.
#endif

    void FreeUserDataString(VmaAllocator hAllocator);
};

/*
Represents a region of VmaDeviceMemoryBlock that is either assigned and returned as
allocated memory block or free.
*/
struct VmaSuballocation
{
    VkDeviceSize offset;
    VkDeviceSize size;
    VmaAllocation hAllocation;
    VmaSuballocationType type;
};

// Comparator for offsets.
struct VmaSuballocationOffsetLess
{
    bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const
    {
        return lhs.offset < rhs.offset;
    }
};
struct VmaSuballocationOffsetGreater
{
    bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const
    {
        return lhs.offset > rhs.offset;
    }
};

typedef VmaList< VmaSuballocation, VmaStlAllocator<VmaSuballocation> > VmaSuballocationList;

// Cost of one additional allocation lost, as equivalent in bytes.
static const VkDeviceSize VMA_LOST_ALLOCATION_COST = 1048576;

enum class VmaAllocationRequestType
{
    Normal,
    // Used by "Linear" algorithm.
    UpperAddress,
    EndOf1st,
    EndOf2nd,
};

/*
Parameters of planned allocation inside a VmaDeviceMemoryBlock.

If canMakeOtherLost was false:
- item points to a FREE suballocation.
- itemsToMakeLostCount is 0.

If canMakeOtherLost was true:
- item points to first of sequence of suballocations, which are either FREE,
  or point to VmaAllocations that can become lost.
- itemsToMakeLostCount is the number of VmaAllocations that need to be made lost for
  the requested allocation to succeed.
*/
struct VmaAllocationRequest
{
    VkDeviceSize offset;
    VkDeviceSize sumFreeSize; // Sum size of free items that overlap with proposed allocation.
    VkDeviceSize sumItemSize; // Sum size of items to make lost that overlap with proposed allocation.
    VmaSuballocationList::iterator item;
    size_t itemsToMakeLostCount;
    void* customData;
    VmaAllocationRequestType type;

    VkDeviceSize CalcCost() const
    {
        return sumItemSize + itemsToMakeLostCount * VMA_LOST_ALLOCATION_COST;
    }
};

/*
Data structure used for bookkeeping of allocations and unused ranges of memory
in a single VkDeviceMemory block.
*/
class VmaBlockMetadata
{
public:
    VmaBlockMetadata(VmaAllocator hAllocator);
    virtual ~VmaBlockMetadata() { }
    virtual void Init(VkDeviceSize size) { m_Size = size; }

    // Validates all data structures inside this object. If not valid, returns false.
    virtual bool Validate() const = 0;
    VkDeviceSize GetSize() const { return m_Size; }
    virtual size_t GetAllocationCount() const = 0;
    virtual VkDeviceSize GetSumFreeSize() const = 0;
    virtual VkDeviceSize GetUnusedRangeSizeMax() const = 0;
    // Returns true if this block is empty - contains only single free suballocation.
    virtual bool IsEmpty() const = 0;

    virtual void CalcAllocationStatInfo(VmaStatInfo& outInfo) const = 0;
    // Shouldn't modify blockCount.
    virtual void AddPoolStats(VmaPoolStats& inoutStats) const = 0;

#if VMA_STATS_STRING_ENABLED
    virtual void PrintDetailedMap(class VmaJsonWriter& json) const = 0;
#endif

    // Tries to find a place for suballocation with given parameters inside this block.
    // If succeeded, fills pAllocationRequest and returns true.
    // If failed, returns false.
    virtual bool CreateAllocationRequest(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VkDeviceSize bufferImageGranularity,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        bool canMakeOtherLost,
        // Always one of VMA_ALLOCATION_CREATE_STRATEGY_* or VMA_ALLOCATION_INTERNAL_STRATEGY_* flags.
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) = 0;

    virtual bool MakeRequestedAllocationsLost(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VmaAllocationRequest* pAllocationRequest) = 0;

    virtual uint32_t MakeAllocationsLost(uint32_t currentFrameIndex, uint32_t frameInUseCount) = 0;

    virtual VkResult CheckCorruption(const void* pBlockData) = 0;

    // Makes actual allocation based on request. Request must already be checked and valid.
    virtual void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        VkDeviceSize allocSize,
        VmaAllocation hAllocation) = 0;

    // Frees suballocation assigned to given memory region.
    virtual void Free(const VmaAllocation allocation) = 0;
    virtual void FreeAtOffset(VkDeviceSize offset) = 0;

protected:
    const VkAllocationCallbacks* GetAllocationCallbacks() const { return m_pAllocationCallbacks; }

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap_Begin(class VmaJsonWriter& json,
        VkDeviceSize unusedBytes,
        size_t allocationCount,
        size_t unusedRangeCount) const;
    void PrintDetailedMap_Allocation(class VmaJsonWriter& json,
        VkDeviceSize offset,
        VmaAllocation hAllocation) const;
    void PrintDetailedMap_UnusedRange(class VmaJsonWriter& json,
        VkDeviceSize offset,
        VkDeviceSize size) const;
    void PrintDetailedMap_End(class VmaJsonWriter& json) const;
#endif

private:
    VkDeviceSize m_Size;
    const VkAllocationCallbacks* m_pAllocationCallbacks;
};

#define VMA_VALIDATE(cond) do { if(!(cond)) { \
        VMA_ASSERT(0 && "Validation failed: " #cond); \
        return false; \
    } } while(false)

class VmaBlockMetadata_Generic : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY(VmaBlockMetadata_Generic)
public:
    VmaBlockMetadata_Generic(VmaAllocator hAllocator);
    virtual ~VmaBlockMetadata_Generic();
    virtual void Init(VkDeviceSize size);

    virtual bool Validate() const;
    virtual size_t GetAllocationCount() const { return m_Suballocations.size() - m_FreeCount; }
    virtual VkDeviceSize GetSumFreeSize() const { return m_SumFreeSize; }
    virtual VkDeviceSize GetUnusedRangeSizeMax() const;
    virtual bool IsEmpty() const;

    virtual void CalcAllocationStatInfo(VmaStatInfo& outInfo) const;
    virtual void AddPoolStats(VmaPoolStats& inoutStats) const;

#if VMA_STATS_STRING_ENABLED
    virtual void PrintDetailedMap(class VmaJsonWriter& json) const;
#endif

    virtual bool CreateAllocationRequest(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VkDeviceSize bufferImageGranularity,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        bool canMakeOtherLost,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);

    virtual bool MakeRequestedAllocationsLost(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VmaAllocationRequest* pAllocationRequest);

    virtual uint32_t MakeAllocationsLost(uint32_t currentFrameIndex, uint32_t frameInUseCount);

    virtual VkResult CheckCorruption(const void* pBlockData);

    virtual void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        VkDeviceSize allocSize,
        VmaAllocation hAllocation);

    virtual void Free(const VmaAllocation allocation);
    virtual void FreeAtOffset(VkDeviceSize offset);

    ////////////////////////////////////////////////////////////////////////////////
    // For defragmentation
    
    bool IsBufferImageGranularityConflictPossible(
        VkDeviceSize bufferImageGranularity,
        VmaSuballocationType& inOutPrevSuballocType) const;

private:
    friend class VmaDefragmentationAlgorithm_Generic;
    friend class VmaDefragmentationAlgorithm_Fast;

    uint32_t m_FreeCount;
    VkDeviceSize m_SumFreeSize;
    VmaSuballocationList m_Suballocations;
    // Suballocations that are free and have size greater than certain threshold.
    // Sorted by size, ascending.
    VmaVector< VmaSuballocationList::iterator, VmaStlAllocator< VmaSuballocationList::iterator > > m_FreeSuballocationsBySize;

    bool ValidateFreeSuballocationList() const;

    // Checks if requested suballocation with given parameters can be placed in given pFreeSuballocItem.
    // If yes, fills pOffset and returns true. If no, returns false.
    bool CheckAllocation(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VkDeviceSize bufferImageGranularity,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        VmaSuballocationList::const_iterator suballocItem,
        bool canMakeOtherLost,
        VkDeviceSize* pOffset,
        size_t* itemsToMakeLostCount,
        VkDeviceSize* pSumFreeSize,
        VkDeviceSize* pSumItemSize) const;
    // Given free suballocation, it merges it with following one, which must also be free.
    void MergeFreeWithNext(VmaSuballocationList::iterator item);
    // Releases given suballocation, making it free.
    // Merges it with adjacent free suballocations if applicable.
    // Returns iterator to new free suballocation at this place.
    VmaSuballocationList::iterator FreeSuballocation(VmaSuballocationList::iterator suballocItem);
    // Given free suballocation, it inserts it into sorted list of
    // m_FreeSuballocationsBySize if it's suitable.
    void RegisterFreeSuballocation(VmaSuballocationList::iterator item);
    // Given free suballocation, it removes it from sorted list of
    // m_FreeSuballocationsBySize if it's suitable.
    void UnregisterFreeSuballocation(VmaSuballocationList::iterator item);
};

/*
Allocations and their references in internal data structure look like this:

if(m_2ndVectorMode == SECOND_VECTOR_EMPTY):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
GetSize() +-------+

if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER):

        0 +-------+
          | Alloc |  2nd[0]
          +-------+
          | Alloc |  2nd[1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
GetSize() +-------+

if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[1]
          +-------+
          | Alloc |  2nd[0]
GetSize() +-------+

*/
class VmaBlockMetadata_Linear : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY(VmaBlockMetadata_Linear)
public:
    VmaBlockMetadata_Linear(VmaAllocator hAllocator);
    virtual ~VmaBlockMetadata_Linear();
    virtual void Init(VkDeviceSize size);

    virtual bool Validate() const;
    virtual size_t GetAllocationCount() const;
    virtual VkDeviceSize GetSumFreeSize() const { return m_SumFreeSize; }
    virtual VkDeviceSize GetUnusedRangeSizeMax() const;
    virtual bool IsEmpty() const { return GetAllocationCount() == 0; }

    virtual void CalcAllocationStatInfo(VmaStatInfo& outInfo) const;
    virtual void AddPoolStats(VmaPoolStats& inoutStats) const;

#if VMA_STATS_STRING_ENABLED
    virtual void PrintDetailedMap(class VmaJsonWriter& json) const;
#endif

    virtual bool CreateAllocationRequest(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VkDeviceSize bufferImageGranularity,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        bool canMakeOtherLost,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);

    virtual bool MakeRequestedAllocationsLost(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VmaAllocationRequest* pAllocationRequest);

    virtual uint32_t MakeAllocationsLost(uint32_t currentFrameIndex, uint32_t frameInUseCount);

    virtual VkResult CheckCorruption(const void* pBlockData);

    virtual void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        VkDeviceSize allocSize,
        VmaAllocation hAllocation);

    virtual void Free(const VmaAllocation allocation);
    virtual void FreeAtOffset(VkDeviceSize offset);

private:
    /*
    There are two suballocation vectors, used in ping-pong way.
    The one with index m_1stVectorIndex is called 1st.
    The one with index (m_1stVectorIndex ^ 1) is called 2nd.
    2nd can be non-empty only when 1st is not empty.
    When 2nd is not empty, m_2ndVectorMode indicates its mode of operation.
    */
    typedef VmaVector< VmaSuballocation, VmaStlAllocator<VmaSuballocation> > SuballocationVectorType;

    enum SECOND_VECTOR_MODE
    {
        SECOND_VECTOR_EMPTY,
        /*
        Suballocations in 2nd vector are created later than the ones in 1st, but they
        all have smaller offset.
        */
        SECOND_VECTOR_RING_BUFFER,
        /*
        Suballocations in 2nd vector are upper side of double stack.
        They all have offsets higher than those in 1st vector.
        Top of this stack means smaller offsets, but higher indices in this vector.
        */
        SECOND_VECTOR_DOUBLE_STACK,
    };

    VkDeviceSize m_SumFreeSize;
    SuballocationVectorType m_Suballocations0, m_Suballocations1;
    uint32_t m_1stVectorIndex;
    SECOND_VECTOR_MODE m_2ndVectorMode;

    SuballocationVectorType& AccessSuballocations1st() { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    SuballocationVectorType& AccessSuballocations2nd() { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }
    const SuballocationVectorType& AccessSuballocations1st() const { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    const SuballocationVectorType& AccessSuballocations2nd() const { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }
    
    // Number of items in 1st vector with hAllocation = null at the beginning.
    size_t m_1stNullItemsBeginCount;
    // Number of other items in 1st vector with hAllocation = null somewhere in the middle.
    size_t m_1stNullItemsMiddleCount;
    // Number of items in 2nd vector with hAllocation = null.
    size_t m_2ndNullItemsCount;

    bool ShouldCompact1st() const;
    void CleanupAfterFree();

    bool CreateAllocationRequest_LowerAddress(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VkDeviceSize bufferImageGranularity,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        bool canMakeOtherLost,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);
    bool CreateAllocationRequest_UpperAddress(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VkDeviceSize bufferImageGranularity,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        bool canMakeOtherLost,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);
};

/*
- GetSize() is the original size of allocated memory block.
- m_UsableSize is this size aligned down to a power of two.
  All allocations and calculations happen relative to m_UsableSize.
- GetUnusableSize() is the difference between them.
  It is repoted as separate, unused range, not available for allocations.

Node at level 0 has size = m_UsableSize.
Each next level contains nodes with size 2 times smaller than current level.
m_LevelCount is the maximum number of levels to use in the current object.
*/
class VmaBlockMetadata_Buddy : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY(VmaBlockMetadata_Buddy)
public:
    VmaBlockMetadata_Buddy(VmaAllocator hAllocator);
    virtual ~VmaBlockMetadata_Buddy();
    virtual void Init(VkDeviceSize size);

    virtual bool Validate() const;
    virtual size_t GetAllocationCount() const { return m_AllocationCount; }
    virtual VkDeviceSize GetSumFreeSize() const { return m_SumFreeSize + GetUnusableSize(); }
    virtual VkDeviceSize GetUnusedRangeSizeMax() const;
    virtual bool IsEmpty() const { return m_Root->type == Node::TYPE_FREE; }

    virtual void CalcAllocationStatInfo(VmaStatInfo& outInfo) const;
    virtual void AddPoolStats(VmaPoolStats& inoutStats) const;

#if VMA_STATS_STRING_ENABLED
    virtual void PrintDetailedMap(class VmaJsonWriter& json) const;
#endif

    virtual bool CreateAllocationRequest(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VkDeviceSize bufferImageGranularity,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        bool canMakeOtherLost,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);

    virtual bool MakeRequestedAllocationsLost(
        uint32_t currentFrameIndex,
        uint32_t frameInUseCount,
        VmaAllocationRequest* pAllocationRequest);

    virtual uint32_t MakeAllocationsLost(uint32_t currentFrameIndex, uint32_t frameInUseCount);

    virtual VkResult CheckCorruption(const void* pBlockData) { return VK_ERROR_FEATURE_NOT_PRESENT; }

    virtual void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        VkDeviceSize allocSize,
        VmaAllocation hAllocation);

    virtual void Free(const VmaAllocation allocation) { FreeAtOffset(allocation, allocation->GetOffset()); }
    virtual void FreeAtOffset(VkDeviceSize offset) { FreeAtOffset(VMA_NULL, offset); }

private:
    static const VkDeviceSize MIN_NODE_SIZE = 32;
    static const size_t MAX_LEVELS = 30;

    struct ValidationContext
    {
        size_t calculatedAllocationCount;
        size_t calculatedFreeCount;
        VkDeviceSize calculatedSumFreeSize;

        ValidationContext() :
            calculatedAllocationCount(0),
            calculatedFreeCount(0),
            calculatedSumFreeSize(0) { }
    };

    struct Node
    {
        VkDeviceSize offset;
        enum TYPE
        {
            TYPE_FREE,
            TYPE_ALLOCATION,
            TYPE_SPLIT,
            TYPE_COUNT
        } type;
        Node* parent;
        Node* buddy;

        union
        {
            struct
            {
                Node* prev;
                Node* next;
            } free;
            struct
            {
                VmaAllocation alloc;
            } allocation;
            struct
            {
                Node* leftChild;
            } split;
        };
    };

    // Size of the memory block aligned down to a power of two.
    VkDeviceSize m_UsableSize;
    uint32_t m_LevelCount;

    Node* m_Root;
    struct {
        Node* front;
        Node* back;
    } m_FreeList[MAX_LEVELS];
    // Number of nodes in the tree with type == TYPE_ALLOCATION.
    size_t m_AllocationCount;
    // Number of nodes in the tree with type == TYPE_FREE.
    size_t m_FreeCount;
    // This includes space wasted due to internal fragmentation. Doesn't include unusable size.
    VkDeviceSize m_SumFreeSize;

    VkDeviceSize GetUnusableSize() const { return GetSize() - m_UsableSize; }
    void DeleteNode(Node* node);
    bool ValidateNode(ValidationContext& ctx, const Node* parent, const Node* curr, uint32_t level, VkDeviceSize levelNodeSize) const;
    uint32_t AllocSizeToLevel(VkDeviceSize allocSize) const;
    inline VkDeviceSize LevelToNodeSize(uint32_t level) const { return m_UsableSize >> level; }
    // Alloc passed just for validation. Can be null.
    void FreeAtOffset(VmaAllocation alloc, VkDeviceSize offset);
    void CalcAllocationStatInfoNode(VmaStatInfo& outInfo, const Node* node, VkDeviceSize levelNodeSize) const;
    // Adds node to the front of FreeList at given level.
    // node->type must be FREE.
    // node->free.prev, next can be undefined.
    void AddToFreeListFront(uint32_t level, Node* node);
    // Removes node from FreeList at given level.
    // node->type must be FREE.
    // node->free.prev, next stay untouched.
    void RemoveFromFreeList(uint32_t level, Node* node);

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMapNode(class VmaJsonWriter& json, const Node* node, VkDeviceSize levelNodeSize) const;
#endif
};

/*
Represents a single block of device memory (`VkDeviceMemory`) with all the
data about its regions (aka suballocations, #VmaAllocation), assigned and free.

Thread-safety: This class must be externally synchronized.
*/
class VmaDeviceMemoryBlock
{
    VMA_CLASS_NO_COPY(VmaDeviceMemoryBlock)
public:
    VmaBlockMetadata* m_pMetadata;

    VmaDeviceMemoryBlock(VmaAllocator hAllocator);

    ~VmaDeviceMemoryBlock()
    {
        VMA_ASSERT(m_MapCount == 0 && "VkDeviceMemory block is being destroyed while it is still mapped.");
        VMA_ASSERT(m_hMemory == VK_NULL_HANDLE);
    }

    // Always call after construction.
    void Init(
        VmaAllocator hAllocator,
        VmaPool hParentPool,
        uint32_t newMemoryTypeIndex,
        VkDeviceMemory newMemory,
        VkDeviceSize newSize,
        uint32_t id,
        uint32_t algorithm);
    // Always call before destruction.
    void Destroy(VmaAllocator allocator);
    
    VmaPool GetParentPool() const { return m_hParentPool; }
    VkDeviceMemory GetDeviceMemory() const { return m_hMemory; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    uint32_t GetId() const { return m_Id; }
    void* GetMappedData() const { return m_pMappedData; }

    // Validates all data structures inside this object. If not valid, returns false.
    bool Validate() const;

    VkResult CheckCorruption(VmaAllocator hAllocator);

    // ppData can be null.
    VkResult Map(VmaAllocator hAllocator, uint32_t count, void** ppData);
    void Unmap(VmaAllocator hAllocator, uint32_t count);

    VkResult WriteMagicValueAroundAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);
    VkResult ValidateMagicValueAroundAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);

    VkResult BindBufferMemory(
        const VmaAllocator hAllocator,
        const VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult BindImageMemory(
        const VmaAllocator hAllocator,
        const VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);

private:
    VmaPool m_hParentPool; // VK_NULL_HANDLE if not belongs to custom pool.
    uint32_t m_MemoryTypeIndex;
    uint32_t m_Id;
    VkDeviceMemory m_hMemory;

    /*
    Protects access to m_hMemory so it's not used by multiple threads simultaneously, e.g. vkMapMemory, vkBindBufferMemory.
    Also protects m_MapCount, m_pMappedData.
    Allocations, deallocations, any change in m_pMetadata is protected by parent's VmaBlockVector::m_Mutex.
    */
    VMA_MUTEX m_Mutex;
    uint32_t m_MapCount;
    void* m_pMappedData;
};

struct VmaPointerLess
{
    bool operator()(const void* lhs, const void* rhs) const
    {
        return lhs < rhs;
    }
};

struct VmaDefragmentationMove
{
    size_t srcBlockIndex;
    size_t dstBlockIndex;
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
};

class VmaDefragmentationAlgorithm;

/*
Sequence of VmaDeviceMemoryBlock. Represents memory blocks allocated for a specific
Vulkan memory type.

Synchronized internally with a mutex.
*/
struct VmaBlockVector
{
    VMA_CLASS_NO_COPY(VmaBlockVector)
public:
    VmaBlockVector(
        VmaAllocator hAllocator,
        VmaPool hParentPool,
        uint32_t memoryTypeIndex,
        VkDeviceSize preferredBlockSize,
        size_t minBlockCount,
        size_t maxBlockCount,
        VkDeviceSize bufferImageGranularity,
        uint32_t frameInUseCount,
        bool explicitBlockSize,
        uint32_t algorithm);
    ~VmaBlockVector();

    VkResult CreateMinBlocks();

    VmaAllocator GetAllocator() const { return m_hAllocator; }
    VmaPool GetParentPool() const { return m_hParentPool; }
    bool IsCustomPool() const { return m_hParentPool != VMA_NULL; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    VkDeviceSize GetPreferredBlockSize() const { return m_PreferredBlockSize; }
    VkDeviceSize GetBufferImageGranularity() const { return m_BufferImageGranularity; }
    uint32_t GetFrameInUseCount() const { return m_FrameInUseCount; }
    uint32_t GetAlgorithm() const { return m_Algorithm; }

    void GetPoolStats(VmaPoolStats* pStats);

    bool IsEmpty();
    bool IsCorruptionDetectionEnabled() const;

    VkResult Allocate(
        uint32_t currentFrameIndex,
        VkDeviceSize size,
        VkDeviceSize alignment,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    void Free(const VmaAllocation hAllocation);

    // Adds statistics of this BlockVector to pStats.
    void AddStats(VmaStats* pStats);

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json);
#endif

    void MakePoolAllocationsLost(
        uint32_t currentFrameIndex,
        size_t* pLostAllocationCount);
    VkResult CheckCorruption();

    // Saves results in pCtx->res.
    void Defragment(
        class VmaBlockVectorDefragmentationContext* pCtx,
        VmaDefragmentationStats* pStats,
        VkDeviceSize& maxCpuBytesToMove, uint32_t& maxCpuAllocationsToMove,
        VkDeviceSize& maxGpuBytesToMove, uint32_t& maxGpuAllocationsToMove,
        VkCommandBuffer commandBuffer);
    void DefragmentationEnd(
        class VmaBlockVectorDefragmentationContext* pCtx,
        VmaDefragmentationStats* pStats);

    ////////////////////////////////////////////////////////////////////////////////
    // To be used only while the m_Mutex is locked. Used during defragmentation.

    size_t GetBlockCount() const { return m_Blocks.size(); }
    VmaDeviceMemoryBlock* GetBlock(size_t index) const { return m_Blocks[index]; }
    size_t CalcAllocationCount() const;
    bool IsBufferImageGranularityConflictPossible() const;

private:
    friend class VmaDefragmentationAlgorithm_Generic;

    const VmaAllocator m_hAllocator;
    const VmaPool m_hParentPool;
    const uint32_t m_MemoryTypeIndex;
    const VkDeviceSize m_PreferredBlockSize;
    const size_t m_MinBlockCount;
    const size_t m_MaxBlockCount;
    const VkDeviceSize m_BufferImageGranularity;
    const uint32_t m_FrameInUseCount;
    const bool m_ExplicitBlockSize;
    const uint32_t m_Algorithm;
    VMA_RW_MUTEX m_Mutex;

    /* There can be at most one allocation that is completely empty (except when minBlockCount > 0) -
    a hysteresis to avoid pessimistic case of alternating creation and destruction of a VkDeviceMemory. */
    bool m_HasEmptyBlock;
    // Incrementally sorted by sumFreeSize, ascending.
    VmaVector< VmaDeviceMemoryBlock*, VmaStlAllocator<VmaDeviceMemoryBlock*> > m_Blocks;
    uint32_t m_NextBlockId;

    VkDeviceSize CalcMaxBlockSize() const;

    // Finds and removes given block from vector.
    void Remove(VmaDeviceMemoryBlock* pBlock);

    // Performs single step in sorting m_Blocks. They may not be fully sorted
    // after this call.
    void IncrementallySortBlocks();

    VkResult AllocatePage(
        uint32_t currentFrameIndex,
        VkDeviceSize size,
        VkDeviceSize alignment,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        VmaAllocation* pAllocation);

    // To be used only without CAN_MAKE_OTHER_LOST flag.
    VkResult AllocateFromBlock(
        VmaDeviceMemoryBlock* pBlock,
        uint32_t currentFrameIndex,
        VkDeviceSize size,
        VkDeviceSize alignment,
        VmaAllocationCreateFlags allocFlags,
        void* pUserData,
        VmaSuballocationType suballocType,
        uint32_t strategy,
        VmaAllocation* pAllocation);

    VkResult CreateBlock(VkDeviceSize blockSize, size_t* pNewBlockIndex);

    // Saves result to pCtx->res.
    void ApplyDefragmentationMovesCpu(
        class VmaBlockVectorDefragmentationContext* pDefragCtx,
        const VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves);
    // Saves result to pCtx->res.
    void ApplyDefragmentationMovesGpu(
        class VmaBlockVectorDefragmentationContext* pDefragCtx,
        const VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
        VkCommandBuffer commandBuffer);

    /*
    Used during defragmentation. pDefragmentationStats is optional. It's in/out
    - updated with new data.
    */
    void FreeEmptyBlocks(VmaDefragmentationStats* pDefragmentationStats);

    void UpdateHasEmptyBlock();
};

struct VmaPool_T
{
    VMA_CLASS_NO_COPY(VmaPool_T)
public:
    VmaBlockVector m_BlockVector;

    VmaPool_T(
        VmaAllocator hAllocator,
        const VmaPoolCreateInfo& createInfo,
        VkDeviceSize preferredBlockSize);
    ~VmaPool_T();

    uint32_t GetId() const { return m_Id; }
    void SetId(uint32_t id) { VMA_ASSERT(m_Id == 0); m_Id = id; }

    const char* GetName() const { return m_Name; }
    void SetName(const char* pName);

#if VMA_STATS_STRING_ENABLED
    //void PrintDetailedMap(class VmaStringBuilder& sb);
#endif

private:
    uint32_t m_Id;
    char* m_Name;
};

/*
Performs defragmentation:

- Updates `pBlockVector->m_pMetadata`.
- Updates allocations by calling ChangeBlockAllocation() or ChangeOffset().
- Does not move actual data, only returns requested moves as `moves`.
*/
class VmaDefragmentationAlgorithm
{
    VMA_CLASS_NO_COPY(VmaDefragmentationAlgorithm)
public:
    VmaDefragmentationAlgorithm(
        VmaAllocator hAllocator,
        VmaBlockVector* pBlockVector,
        uint32_t currentFrameIndex) :
        m_hAllocator(hAllocator),
        m_pBlockVector(pBlockVector),
        m_CurrentFrameIndex(currentFrameIndex)
    {
    }
    virtual ~VmaDefragmentationAlgorithm()
    {
    }

    virtual void AddAllocation(VmaAllocation hAlloc, VkBool32* pChanged) = 0;
    virtual void AddAll() = 0;

    virtual VkResult Defragment(
        VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
        VkDeviceSize maxBytesToMove,
        uint32_t maxAllocationsToMove) = 0;

    virtual VkDeviceSize GetBytesMoved() const = 0;
    virtual uint32_t GetAllocationsMoved() const = 0;

protected:
    VmaAllocator const m_hAllocator;
    VmaBlockVector* const m_pBlockVector;
    const uint32_t m_CurrentFrameIndex;

    struct AllocationInfo
    {
        VmaAllocation m_hAllocation;
        VkBool32* m_pChanged;

        AllocationInfo() :
            m_hAllocation(VK_NULL_HANDLE),
            m_pChanged(VMA_NULL)
        {
        }
        AllocationInfo(VmaAllocation hAlloc, VkBool32* pChanged) :
            m_hAllocation(hAlloc),
            m_pChanged(pChanged)
        {
        }
    };
};

class VmaDefragmentationAlgorithm_Generic : public VmaDefragmentationAlgorithm
{
    VMA_CLASS_NO_COPY(VmaDefragmentationAlgorithm_Generic)
public:
    VmaDefragmentationAlgorithm_Generic(
        VmaAllocator hAllocator,
        VmaBlockVector* pBlockVector,
        uint32_t currentFrameIndex,
        bool overlappingMoveSupported);
    virtual ~VmaDefragmentationAlgorithm_Generic();

    virtual void AddAllocation(VmaAllocation hAlloc, VkBool32* pChanged);
    virtual void AddAll() { m_AllAllocations = true; }

    virtual VkResult Defragment(
        VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
        VkDeviceSize maxBytesToMove,
        uint32_t maxAllocationsToMove);

    virtual VkDeviceSize GetBytesMoved() const { return m_BytesMoved; }
    virtual uint32_t GetAllocationsMoved() const { return m_AllocationsMoved; }

private:
    uint32_t m_AllocationCount;
    bool m_AllAllocations;

    VkDeviceSize m_BytesMoved;
    uint32_t m_AllocationsMoved;

    struct AllocationInfoSizeGreater
    {
        bool operator()(const AllocationInfo& lhs, const AllocationInfo& rhs) const
        {
            return lhs.m_hAllocation->GetSize() > rhs.m_hAllocation->GetSize();
        }
    };

    struct AllocationInfoOffsetGreater
    {
        bool operator()(const AllocationInfo& lhs, const AllocationInfo& rhs) const
        {
            return lhs.m_hAllocation->GetOffset() > rhs.m_hAllocation->GetOffset();
        }
    };

    struct BlockInfo
    {
        size_t m_OriginalBlockIndex;
        VmaDeviceMemoryBlock* m_pBlock;
        bool m_HasNonMovableAllocations;
        VmaVector< AllocationInfo, VmaStlAllocator<AllocationInfo> > m_Allocations;

        BlockInfo(const VkAllocationCallbacks* pAllocationCallbacks) :
            m_OriginalBlockIndex(SIZE_MAX),
            m_pBlock(VMA_NULL),
            m_HasNonMovableAllocations(true),
            m_Allocations(pAllocationCallbacks)
        {
        }

        void CalcHasNonMovableAllocations()
        {
            const size_t blockAllocCount = m_pBlock->m_pMetadata->GetAllocationCount();
            const size_t defragmentAllocCount = m_Allocations.size();
            m_HasNonMovableAllocations = blockAllocCount != defragmentAllocCount;
        }

        void SortAllocationsBySizeDescending()
        {
            VMA_SORT(m_Allocations.begin(), m_Allocations.end(), AllocationInfoSizeGreater());
        }

        void SortAllocationsByOffsetDescending()
        {
            VMA_SORT(m_Allocations.begin(), m_Allocations.end(), AllocationInfoOffsetGreater());
        }
    };

    struct BlockPointerLess
    {
        bool operator()(const BlockInfo* pLhsBlockInfo, const VmaDeviceMemoryBlock* pRhsBlock) const
        {
            return pLhsBlockInfo->m_pBlock < pRhsBlock;
        }
        bool operator()(const BlockInfo* pLhsBlockInfo, const BlockInfo* pRhsBlockInfo) const
        {
            return pLhsBlockInfo->m_pBlock < pRhsBlockInfo->m_pBlock;
        }
    };

    // 1. Blocks with some non-movable allocations go first.
    // 2. Blocks with smaller sumFreeSize go first.
    struct BlockInfoCompareMoveDestination
    {
        bool operator()(const BlockInfo* pLhsBlockInfo, const BlockInfo* pRhsBlockInfo) const
        {
            if(pLhsBlockInfo->m_HasNonMovableAllocations && !pRhsBlockInfo->m_HasNonMovableAllocations)
            {
                return true;
            }
            if(!pLhsBlockInfo->m_HasNonMovableAllocations && pRhsBlockInfo->m_HasNonMovableAllocations)
            {
                return false;
            }
            if(pLhsBlockInfo->m_pBlock->m_pMetadata->GetSumFreeSize() < pRhsBlockInfo->m_pBlock->m_pMetadata->GetSumFreeSize())
            {
                return true;
            }
            return false;
        }
    };

    typedef VmaVector< BlockInfo*, VmaStlAllocator<BlockInfo*> > BlockInfoVector;
    BlockInfoVector m_Blocks;

    VkResult DefragmentRound(
        VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
        VkDeviceSize maxBytesToMove,
        uint32_t maxAllocationsToMove);

    size_t CalcBlocksWithNonMovableCount() const;

    static bool MoveMakesSense(
        size_t dstBlockIndex, VkDeviceSize dstOffset,
        size_t srcBlockIndex, VkDeviceSize srcOffset);
};

class VmaDefragmentationAlgorithm_Fast : public VmaDefragmentationAlgorithm
{
    VMA_CLASS_NO_COPY(VmaDefragmentationAlgorithm_Fast)
public:
    VmaDefragmentationAlgorithm_Fast(
        VmaAllocator hAllocator,
        VmaBlockVector* pBlockVector,
        uint32_t currentFrameIndex,
        bool overlappingMoveSupported);
    virtual ~VmaDefragmentationAlgorithm_Fast();

    virtual void AddAllocation(VmaAllocation hAlloc, VkBool32* pChanged) { ++m_AllocationCount; }
    virtual void AddAll() { m_AllAllocations = true; }

    virtual VkResult Defragment(
        VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
        VkDeviceSize maxBytesToMove,
        uint32_t maxAllocationsToMove);

    virtual VkDeviceSize GetBytesMoved() const { return m_BytesMoved; }
    virtual uint32_t GetAllocationsMoved() const { return m_AllocationsMoved; }

private:
    struct BlockInfo
    {
        size_t origBlockIndex;
    };

    class FreeSpaceDatabase
    {
    public:
        FreeSpaceDatabase()
        {
            FreeSpace s = {};
            s.blockInfoIndex = SIZE_MAX;
            for(size_t i = 0; i < MAX_COUNT; ++i)
            {
                m_FreeSpaces[i] = s;
            }
        }

        void Register(size_t blockInfoIndex, VkDeviceSize offset, VkDeviceSize size)
        {
            if(size < VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
            {
                return;
            }

            // Find first invalid or the smallest structure.
            size_t bestIndex = SIZE_MAX;
            for(size_t i = 0; i < MAX_COUNT; ++i)
            {
                // Empty structure.
                if(m_FreeSpaces[i].blockInfoIndex == SIZE_MAX)
                {
                    bestIndex = i;
                    break;
                }
                if(m_FreeSpaces[i].size < size &&
                    (bestIndex == SIZE_MAX || m_FreeSpaces[bestIndex].size > m_FreeSpaces[i].size))
                {
                    bestIndex = i;
                }
            }

            if(bestIndex != SIZE_MAX)
            {
                m_FreeSpaces[bestIndex].blockInfoIndex = blockInfoIndex;
                m_FreeSpaces[bestIndex].offset = offset;
                m_FreeSpaces[bestIndex].size = size;
            }
        }

        bool Fetch(VkDeviceSize alignment, VkDeviceSize size,
            size_t& outBlockInfoIndex, VkDeviceSize& outDstOffset)
        {
            size_t bestIndex = SIZE_MAX;
            VkDeviceSize bestFreeSpaceAfter = 0;
            for(size_t i = 0; i < MAX_COUNT; ++i)
            {
                // Structure is valid.
                if(m_FreeSpaces[i].blockInfoIndex != SIZE_MAX)
                {
                    const VkDeviceSize dstOffset = VmaAlignUp(m_FreeSpaces[i].offset, alignment);
                    // Allocation fits into this structure.
                    if(dstOffset + size <= m_FreeSpaces[i].offset + m_FreeSpaces[i].size)
                    {
                        const VkDeviceSize freeSpaceAfter = (m_FreeSpaces[i].offset + m_FreeSpaces[i].size) -
                            (dstOffset + size);
                        if(bestIndex == SIZE_MAX || freeSpaceAfter > bestFreeSpaceAfter)
                        {
                            bestIndex = i;
                            bestFreeSpaceAfter = freeSpaceAfter;
                        }
                    }
                }
            }
            
            if(bestIndex != SIZE_MAX)
            {
                outBlockInfoIndex = m_FreeSpaces[bestIndex].blockInfoIndex;
                outDstOffset = VmaAlignUp(m_FreeSpaces[bestIndex].offset, alignment);

                if(bestFreeSpaceAfter >= VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
                {
                    // Leave this structure for remaining empty space.
                    const VkDeviceSize alignmentPlusSize = (outDstOffset - m_FreeSpaces[bestIndex].offset) + size;
                    m_FreeSpaces[bestIndex].offset += alignmentPlusSize;
                    m_FreeSpaces[bestIndex].size -= alignmentPlusSize;
                }
                else
                {
                    // This structure becomes invalid.
                    m_FreeSpaces[bestIndex].blockInfoIndex = SIZE_MAX;
                }

                return true;
            }

            return false;
        }

    private:
        static const size_t MAX_COUNT = 4;

        struct FreeSpace
        {
            size_t blockInfoIndex; // SIZE_MAX means this structure is invalid.
            VkDeviceSize offset;
            VkDeviceSize size;
        } m_FreeSpaces[MAX_COUNT];
    };

    const bool m_OverlappingMoveSupported;

    uint32_t m_AllocationCount;
    bool m_AllAllocations;

    VkDeviceSize m_BytesMoved;
    uint32_t m_AllocationsMoved;

    VmaVector< BlockInfo, VmaStlAllocator<BlockInfo> > m_BlockInfos;

    void PreprocessMetadata();
    void PostprocessMetadata();
    void InsertSuballoc(VmaBlockMetadata_Generic* pMetadata, const VmaSuballocation& suballoc);
};

struct VmaBlockDefragmentationContext
{
    enum BLOCK_FLAG
    {
        BLOCK_FLAG_USED = 0x00000001,
    };
    uint32_t flags;
    VkBuffer hBuffer;
};

class VmaBlockVectorDefragmentationContext
{
    VMA_CLASS_NO_COPY(VmaBlockVectorDefragmentationContext)
public:
    VkResult res;
    bool mutexLocked;
    VmaVector< VmaBlockDefragmentationContext, VmaStlAllocator<VmaBlockDefragmentationContext> > blockContexts;

    VmaBlockVectorDefragmentationContext(
        VmaAllocator hAllocator,
        VmaPool hCustomPool, // Optional.
        VmaBlockVector* pBlockVector,
        uint32_t currFrameIndex);
    ~VmaBlockVectorDefragmentationContext();

    VmaPool GetCustomPool() const { return m_hCustomPool; }
    VmaBlockVector* GetBlockVector() const { return m_pBlockVector; }
    VmaDefragmentationAlgorithm* GetAlgorithm() const { return m_pAlgorithm; }

    void AddAllocation(VmaAllocation hAlloc, VkBool32* pChanged);
    void AddAll() { m_AllAllocations = true; }

    void Begin(bool overlappingMoveSupported);

private:
    const VmaAllocator m_hAllocator;
    // Null if not from custom pool.
    const VmaPool m_hCustomPool;
    // Redundant, for convenience not to fetch from m_hCustomPool->m_BlockVector or m_hAllocator->m_pBlockVectors.
    VmaBlockVector* const m_pBlockVector;
    const uint32_t m_CurrFrameIndex;
    // Owner of this object.
    VmaDefragmentationAlgorithm* m_pAlgorithm;

    struct AllocInfo
    {
        VmaAllocation hAlloc;
        VkBool32* pChanged;
    };
    // Used between constructor and Begin.
    VmaVector< AllocInfo, VmaStlAllocator<AllocInfo> > m_Allocations;
    bool m_AllAllocations;
};

struct VmaDefragmentationContext_T
{
private:
    VMA_CLASS_NO_COPY(VmaDefragmentationContext_T)
public:
    VmaDefragmentationContext_T(
        VmaAllocator hAllocator,
        uint32_t currFrameIndex,
        uint32_t flags,
        VmaDefragmentationStats* pStats);
    ~VmaDefragmentationContext_T();

    void AddPools(uint32_t poolCount, VmaPool* pPools);
    void AddAllocations(
        uint32_t allocationCount,
        VmaAllocation* pAllocations,
        VkBool32* pAllocationsChanged);

    /*
    Returns:
    - `VK_SUCCESS` if succeeded and object can be destroyed immediately.
    - `VK_NOT_READY` if succeeded but the object must remain alive until vmaDefragmentationEnd().
    - Negative value if error occured and object can be destroyed immediately.
    */
    VkResult Defragment(
        VkDeviceSize maxCpuBytesToMove, uint32_t maxCpuAllocationsToMove,
        VkDeviceSize maxGpuBytesToMove, uint32_t maxGpuAllocationsToMove,
        VkCommandBuffer commandBuffer, VmaDefragmentationStats* pStats);

private:
    const VmaAllocator m_hAllocator;
    const uint32_t m_CurrFrameIndex;
    const uint32_t m_Flags;
    VmaDefragmentationStats* const m_pStats;
    // Owner of these objects.
    VmaBlockVectorDefragmentationContext* m_DefaultPoolContexts[VK_MAX_MEMORY_TYPES];
    // Owner of these objects.
    VmaVector< VmaBlockVectorDefragmentationContext*, VmaStlAllocator<VmaBlockVectorDefragmentationContext*> > m_CustomPoolContexts;
};

#if VMA_RECORDING_ENABLED

class VmaRecorder
{
public:
    VmaRecorder();
    VkResult Init(const VmaRecordSettings& settings, bool useMutex);
    void WriteConfiguration(
        const VkPhysicalDeviceProperties& devProps,
        const VkPhysicalDeviceMemoryProperties& memProps,
        uint32_t vulkanApiVersion,
        bool dedicatedAllocationExtensionEnabled,
        bool bindMemory2ExtensionEnabled,
        bool memoryBudgetExtensionEnabled);
    ~VmaRecorder();

    void RecordCreateAllocator(uint32_t frameIndex);
    void RecordDestroyAllocator(uint32_t frameIndex);
    void RecordCreatePool(uint32_t frameIndex,
        const VmaPoolCreateInfo& createInfo,
        VmaPool pool);
    void RecordDestroyPool(uint32_t frameIndex, VmaPool pool);
    void RecordAllocateMemory(uint32_t frameIndex,
        const VkMemoryRequirements& vkMemReq,
        const VmaAllocationCreateInfo& createInfo,
        VmaAllocation allocation);
    void RecordAllocateMemoryPages(uint32_t frameIndex,
        const VkMemoryRequirements& vkMemReq,
        const VmaAllocationCreateInfo& createInfo,
        uint64_t allocationCount,
        const VmaAllocation* pAllocations);
    void RecordAllocateMemoryForBuffer(uint32_t frameIndex,
        const VkMemoryRequirements& vkMemReq,
        bool requiresDedicatedAllocation,
        bool prefersDedicatedAllocation,
        const VmaAllocationCreateInfo& createInfo,
        VmaAllocation allocation);
    void RecordAllocateMemoryForImage(uint32_t frameIndex,
        const VkMemoryRequirements& vkMemReq,
        bool requiresDedicatedAllocation,
        bool prefersDedicatedAllocation,
        const VmaAllocationCreateInfo& createInfo,
        VmaAllocation allocation);
    void RecordFreeMemory(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordFreeMemoryPages(uint32_t frameIndex,
        uint64_t allocationCount,
        const VmaAllocation* pAllocations);
    void RecordSetAllocationUserData(uint32_t frameIndex,
        VmaAllocation allocation,
        const void* pUserData);
    void RecordCreateLostAllocation(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordMapMemory(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordUnmapMemory(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordFlushAllocation(uint32_t frameIndex,
        VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size);
    void RecordInvalidateAllocation(uint32_t frameIndex,
        VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size);
    void RecordCreateBuffer(uint32_t frameIndex,
        const VkBufferCreateInfo& bufCreateInfo,
        const VmaAllocationCreateInfo& allocCreateInfo,
        VmaAllocation allocation);
    void RecordCreateImage(uint32_t frameIndex,
        const VkImageCreateInfo& imageCreateInfo,
        const VmaAllocationCreateInfo& allocCreateInfo,
        VmaAllocation allocation);
    void RecordDestroyBuffer(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordDestroyImage(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordTouchAllocation(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordGetAllocationInfo(uint32_t frameIndex,
        VmaAllocation allocation);
    void RecordMakePoolAllocationsLost(uint32_t frameIndex,
        VmaPool pool);
    void RecordDefragmentationBegin(uint32_t frameIndex,
        const VmaDefragmentationInfo2& info,
        VmaDefragmentationContext ctx);
    void RecordDefragmentationEnd(uint32_t frameIndex,
        VmaDefragmentationContext ctx);
    void RecordSetPoolName(uint32_t frameIndex,
        VmaPool pool,
        const char* name);

private:
    struct CallParams
    {
        uint32_t threadId;
        double time;
    };

    class UserDataString
    {
    public:
        UserDataString(VmaAllocationCreateFlags allocFlags, const void* pUserData);
        const char* GetString() const { return m_Str; }

    private:
        char m_PtrStr[17];
        const char* m_Str;
    };

    bool m_UseMutex;
    VmaRecordFlags m_Flags;
    FILE* m_File;
    VMA_MUTEX m_FileMutex;
    int64_t m_Freq;
    int64_t m_StartCounter;

    void GetBasicParams(CallParams& outParams);

    // T must be a pointer type, e.g. VmaAllocation, VmaPool.
    template<typename T>
    void PrintPointerList(uint64_t count, const T* pItems)
    {
        if(count)
        {
            fprintf(m_File, "%p", pItems[0]);
            for(uint64_t i = 1; i < count; ++i)
            {
                fprintf(m_File, " %p", pItems[i]);
            }
        }
    }

    void PrintPointerList(uint64_t count, const VmaAllocation* pItems);
    void Flush();
};

#endif // #if VMA_RECORDING_ENABLED

/*
Thread-safe wrapper over VmaPoolAllocator free list, for allocation of VmaAllocation_T objects.
*/
class VmaAllocationObjectAllocator
{
    VMA_CLASS_NO_COPY(VmaAllocationObjectAllocator)
public:
    VmaAllocationObjectAllocator(const VkAllocationCallbacks* pAllocationCallbacks);

    VmaAllocation Allocate();
    void Free(VmaAllocation hAlloc);

private:
    VMA_MUTEX m_Mutex;
    VmaPoolAllocator<VmaAllocation_T> m_Allocator;
};

struct VmaCurrentBudgetData
{
    VMA_ATOMIC_UINT64 m_BlockBytes[VK_MAX_MEMORY_HEAPS];
    VMA_ATOMIC_UINT64 m_AllocationBytes[VK_MAX_MEMORY_HEAPS];

#if VMA_MEMORY_BUDGET
    VMA_ATOMIC_UINT32 m_OperationsSinceBudgetFetch;
    VMA_RW_MUTEX m_BudgetMutex;
    uint64_t m_VulkanUsage[VK_MAX_MEMORY_HEAPS];
    uint64_t m_VulkanBudget[VK_MAX_MEMORY_HEAPS];
    uint64_t m_BlockBytesAtBudgetFetch[VK_MAX_MEMORY_HEAPS];
#endif // #if VMA_MEMORY_BUDGET

    VmaCurrentBudgetData()
    {
        for(uint32_t heapIndex = 0; heapIndex < VK_MAX_MEMORY_HEAPS; ++heapIndex)
        {
            m_BlockBytes[heapIndex] = 0;
            m_AllocationBytes[heapIndex] = 0;
#if VMA_MEMORY_BUDGET
            m_VulkanUsage[heapIndex] = 0;
            m_VulkanBudget[heapIndex] = 0;
            m_BlockBytesAtBudgetFetch[heapIndex] = 0;
#endif
        }

#if VMA_MEMORY_BUDGET
        m_OperationsSinceBudgetFetch = 0;
#endif
    }

    void AddAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
    {
        m_AllocationBytes[heapIndex] += allocationSize;
#if VMA_MEMORY_BUDGET
        ++m_OperationsSinceBudgetFetch;
#endif
    }

    void RemoveAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
    {
        VMA_ASSERT(m_AllocationBytes[heapIndex] >= allocationSize); // DELME
        m_AllocationBytes[heapIndex] -= allocationSize;
#if VMA_MEMORY_BUDGET
        ++m_OperationsSinceBudgetFetch;
#endif
    }
};

// Main allocator object.
struct VmaAllocator_T
{
    VMA_CLASS_NO_COPY(VmaAllocator_T)
public:
    bool m_UseMutex;
    uint32_t m_VulkanApiVersion;
    bool m_UseKhrDedicatedAllocation; // Can be set only if m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0).
    bool m_UseKhrBindMemory2; // Can be set only if m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0).
    bool m_UseExtMemoryBudget;
    VkDevice m_hDevice;
    VkInstance m_hInstance;
    bool m_AllocationCallbacksSpecified;
    VkAllocationCallbacks m_AllocationCallbacks;
    VmaDeviceMemoryCallbacks m_DeviceMemoryCallbacks;
    VmaAllocationObjectAllocator m_AllocationObjectAllocator;
    
    // Each bit (1 << i) is set if HeapSizeLimit is enabled for that heap, so cannot allocate more than the heap size.
    uint32_t m_HeapSizeLimitMask;

    VkPhysicalDeviceProperties m_PhysicalDeviceProperties;
    VkPhysicalDeviceMemoryProperties m_MemProps;

    // Default pools.
    VmaBlockVector* m_pBlockVectors[VK_MAX_MEMORY_TYPES];

    // Each vector is sorted by memory (handle value).
    typedef VmaVector< VmaAllocation, VmaStlAllocator<VmaAllocation> > AllocationVectorType;
    AllocationVectorType* m_pDedicatedAllocations[VK_MAX_MEMORY_TYPES];
    VMA_RW_MUTEX m_DedicatedAllocationsMutex[VK_MAX_MEMORY_TYPES];

    VmaCurrentBudgetData m_Budget;

    VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo);
    VkResult Init(const VmaAllocatorCreateInfo* pCreateInfo);
    ~VmaAllocator_T();

    const VkAllocationCallbacks* GetAllocationCallbacks() const
    {
        return m_AllocationCallbacksSpecified ? &m_AllocationCallbacks : 0;
    }
    const VmaVulkanFunctions& GetVulkanFunctions() const
    {
        return m_VulkanFunctions;
    }

    VkDeviceSize GetBufferImageGranularity() const
    {
        return VMA_MAX(
            static_cast<VkDeviceSize>(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY),
            m_PhysicalDeviceProperties.limits.bufferImageGranularity);
    }

    uint32_t GetMemoryHeapCount() const { return m_MemProps.memoryHeapCount; }
    uint32_t GetMemoryTypeCount() const { return m_MemProps.memoryTypeCount; }

    uint32_t MemoryTypeIndexToHeapIndex(uint32_t memTypeIndex) const
    {
        VMA_ASSERT(memTypeIndex < m_MemProps.memoryTypeCount);
        return m_MemProps.memoryTypes[memTypeIndex].heapIndex;
    }
    // True when specific memory type is HOST_VISIBLE but not HOST_COHERENT.
    bool IsMemoryTypeNonCoherent(uint32_t memTypeIndex) const
    {
        return (m_MemProps.memoryTypes[memTypeIndex].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }
    // Minimum alignment for all allocations in specific memory type.
    VkDeviceSize GetMemoryTypeMinAlignment(uint32_t memTypeIndex) const
    {
        return IsMemoryTypeNonCoherent(memTypeIndex) ?
            VMA_MAX((VkDeviceSize)VMA_DEBUG_ALIGNMENT, m_PhysicalDeviceProperties.limits.nonCoherentAtomSize) :
            (VkDeviceSize)VMA_DEBUG_ALIGNMENT;
    }

    bool IsIntegratedGpu() const
    {
        return m_PhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }

#if VMA_RECORDING_ENABLED
    VmaRecorder* GetRecorder() const { return m_pRecorder; }
#endif

    void GetBufferMemoryRequirements(
        VkBuffer hBuffer,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;
    void GetImageMemoryRequirements(
        VkImage hImage,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;

    // Main allocation function.
    VkResult AllocateMemory(
        const VkMemoryRequirements& vkMemReq,
        bool requiresDedicatedAllocation,
        bool prefersDedicatedAllocation,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    // Main deallocation function.
    void FreeMemory(
        size_t allocationCount,
        const VmaAllocation* pAllocations);

    VkResult ResizeAllocation(
        const VmaAllocation alloc,
        VkDeviceSize newSize);

    void CalculateStats(VmaStats* pStats);

    void GetBudget(
        VmaBudget* outBudget, uint32_t firstHeap, uint32_t heapCount);

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json);
#endif

    VkResult DefragmentationBegin(
        const VmaDefragmentationInfo2& info,
        VmaDefragmentationStats* pStats,
        VmaDefragmentationContext* pContext);
    VkResult DefragmentationEnd(
        VmaDefragmentationContext context);

    void GetAllocationInfo(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo);
    bool TouchAllocation(VmaAllocation hAllocation);

    VkResult CreatePool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool);
    void DestroyPool(VmaPool pool);
    void GetPoolStats(VmaPool pool, VmaPoolStats* pPoolStats);

    void SetCurrentFrameIndex(uint32_t frameIndex);
    uint32_t GetCurrentFrameIndex() const { return m_CurrentFrameIndex.load(); }

    void MakePoolAllocationsLost(
        VmaPool hPool,
        size_t* pLostAllocationCount);
    VkResult CheckPoolCorruption(VmaPool hPool);
    VkResult CheckCorruption(uint32_t memoryTypeBits);

    void CreateLostAllocation(VmaAllocation* pAllocation);

    // Call to Vulkan function vkAllocateMemory with accompanying bookkeeping.
    VkResult AllocateVulkanMemory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory);
    // Call to Vulkan function vkFreeMemory with accompanying bookkeeping.
    void FreeVulkanMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory);
    // Call to Vulkan function vkBindBufferMemory or vkBindBufferMemory2KHR.
    VkResult BindVulkanBuffer(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkBuffer buffer,
        const void* pNext);
    // Call to Vulkan function vkBindImageMemory or vkBindImageMemory2KHR.
    VkResult BindVulkanImage(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkImage image,
        const void* pNext);

    VkResult Map(VmaAllocation hAllocation, void** ppData);
    void Unmap(VmaAllocation hAllocation);

    VkResult BindBufferMemory(
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult BindImageMemory(
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);

    void FlushOrInvalidateAllocation(
        VmaAllocation hAllocation,
        VkDeviceSize offset, VkDeviceSize size,
        VMA_CACHE_OPERATION op);

    void FillAllocation(const VmaAllocation hAllocation, uint8_t pattern);

    /*
    Returns bit mask of memory types that can support defragmentation on GPU as
    they support creation of required buffer for copy operations.
    */
    uint32_t GetGpuDefragmentationMemoryTypeBits();

private:
    VkDeviceSize m_PreferredLargeHeapBlockSize;

    VkPhysicalDevice m_PhysicalDevice;
    VMA_ATOMIC_UINT32 m_CurrentFrameIndex;
    VMA_ATOMIC_UINT32 m_GpuDefragmentationMemoryTypeBits; // UINT32_MAX means uninitialized.
    
    VMA_RW_MUTEX m_PoolsMutex;
    // Protected by m_PoolsMutex. Sorted by pointer value.
    VmaVector<VmaPool, VmaStlAllocator<VmaPool> > m_Pools;
    uint32_t m_NextPoolId;

    VmaVulkanFunctions m_VulkanFunctions;

#if VMA_RECORDING_ENABLED
    VmaRecorder* m_pRecorder;
#endif

    void ImportVulkanFunctions(const VmaVulkanFunctions* pVulkanFunctions);

    VkDeviceSize CalcPreferredBlockSize(uint32_t memTypeIndex);

    VkResult AllocateMemoryOfType(
        VkDeviceSize size,
        VkDeviceSize alignment,
        bool dedicatedAllocation,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        const VmaAllocationCreateInfo& createInfo,
        uint32_t memTypeIndex,
        VmaSuballocationType suballocType,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    // Helper function only to be used inside AllocateDedicatedMemory.
    VkResult AllocateDedicatedMemoryPage(
        VkDeviceSize size,
        VmaSuballocationType suballocType,
        uint32_t memTypeIndex,
        const VkMemoryAllocateInfo& allocInfo,
        bool map,
        bool isUserDataString,
        void* pUserData,
        VmaAllocation* pAllocation);

    // Allocates and registers new VkDeviceMemory specifically for dedicated allocations.
    VkResult AllocateDedicatedMemory(
        VkDeviceSize size,
        VmaSuballocationType suballocType,
        uint32_t memTypeIndex,
        bool withinBudget,
        bool map,
        bool isUserDataString,
        void* pUserData,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    void FreeDedicatedMemory(const VmaAllocation allocation);

    /*
    Calculates and returns bit mask of memory types that can support defragmentation
    on GPU as they support creation of required buffer for copy operations.
    */
    uint32_t CalculateGpuDefragmentationMemoryTypeBits() const;

#if VMA_MEMORY_BUDGET
    void UpdateVulkanBudget();
#endif // #if VMA_MEMORY_BUDGET
};

////////////////////////////////////////////////////////////////////////////////
// Memory allocation #2 after VmaAllocator_T definition

static void* VmaMalloc(VmaAllocator hAllocator, size_t size, size_t alignment)
{
    return VmaMalloc(&hAllocator->m_AllocationCallbacks, size, alignment);
}

static void VmaFree(VmaAllocator hAllocator, void* ptr)
{
    VmaFree(&hAllocator->m_AllocationCallbacks, ptr);
}

template<typename T>
static T* VmaAllocate(VmaAllocator hAllocator)
{
    return (T*)VmaMalloc(hAllocator, sizeof(T), VMA_ALIGN_OF(T));
}

template<typename T>
static T* VmaAllocateArray(VmaAllocator hAllocator, size_t count)
{
    return (T*)VmaMalloc(hAllocator, sizeof(T) * count, VMA_ALIGN_OF(T));
}

template<typename T>
static void vma_delete(VmaAllocator hAllocator, T* ptr)
{
    if(ptr != VMA_NULL)
    {
        ptr->~T();
        VmaFree(hAllocator, ptr);
    }
}

template<typename T>
static void vma_delete_array(VmaAllocator hAllocator, T* ptr, size_t count)
{
    if(ptr != VMA_NULL)
    {
        for(size_t i = count; i--; )
            ptr[i].~T();
        VmaFree(hAllocator, ptr);
    }
}

////////////////////////////////////////////////////////////////////////////////
// VmaStringBuilder

#if VMA_STATS_STRING_ENABLED

class VmaStringBuilder
{
public:
    VmaStringBuilder(VmaAllocator alloc) : m_Data(VmaStlAllocator<char>(alloc->GetAllocationCallbacks())) { }
    size_t GetLength() const { return m_Data.size(); }
    const char* GetData() const { return m_Data.data(); }

    void Add(char ch) { m_Data.push_back(ch); }
    void Add(const char* pStr);
    void AddNewLine() { Add('\n'); }
    void AddNumber(uint32_t num);
    void AddNumber(uint64_t num);
    void AddPointer(const void* ptr);

private:
    VmaVector< char, VmaStlAllocator<char> > m_Data;
};

void VmaStringBuilder::Add(const char* pStr)
{
    const size_t strLen = strlen(pStr);
    if(strLen > 0)
    {
        const size_t oldCount = m_Data.size();
        m_Data.resize(oldCount + strLen);
        memcpy(m_Data.data() + oldCount, pStr, strLen);
    }
}

void VmaStringBuilder::AddNumber(uint32_t num)
{
    char buf[11];
    buf[10] = '\0';
    char *p = &buf[10];
    do
    {
        *--p = '0' + (num % 10);
        num /= 10;
    }
    while(num);
    Add(p);
}

void VmaStringBuilder::AddNumber(uint64_t num)
{
    char buf[21];
    buf[20] = '\0';
    char *p = &buf[20];
    do
    {
        *--p = '0' + (num % 10);
        num /= 10;
    }
    while(num);
    Add(p);
}

void VmaStringBuilder::AddPointer(const void* ptr)
{
    char buf[21];
    VmaPtrToStr(buf, sizeof(buf), ptr);
    Add(buf);
}

#endif // #if VMA_STATS_STRING_ENABLED

////////////////////////////////////////////////////////////////////////////////
// VmaJsonWriter

#if VMA_STATS_STRING_ENABLED

class VmaJsonWriter
{
    VMA_CLASS_NO_COPY(VmaJsonWriter)
public:
    VmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, VmaStringBuilder& sb);
    ~VmaJsonWriter();

    void BeginObject(bool singleLine = false);
    void EndObject();
    
    void BeginArray(bool singleLine = false);
    void EndArray();
    
    void WriteString(const char* pStr);
    void BeginString(const char* pStr = VMA_NULL);
    void ContinueString(const char* pStr);
    void ContinueString(uint32_t n);
    void ContinueString(uint64_t n);
    void ContinueString_Pointer(const void* ptr);
    void EndString(const char* pStr = VMA_NULL);
    
    void WriteNumber(uint32_t n);
    void WriteNumber(uint64_t n);
    void WriteBool(bool b);
    void WriteNull();

private:
    static const char* const INDENT;

    enum COLLECTION_TYPE
    {
        COLLECTION_TYPE_OBJECT,
        COLLECTION_TYPE_ARRAY,
    };
    struct StackItem
    {
        COLLECTION_TYPE type;
        uint32_t valueCount;
        bool singleLineMode;
    };

    VmaStringBuilder& m_SB;
    VmaVector< StackItem, VmaStlAllocator<StackItem> > m_Stack;
    bool m_InsideString;

    void BeginValue(bool isString);
    void WriteIndent(bool oneLess = false);
};

const char* const VmaJsonWriter::INDENT = "  ";

VmaJsonWriter::VmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, VmaStringBuilder& sb) :
    m_SB(sb),
    m_Stack(VmaStlAllocator<StackItem>(pAllocationCallbacks)),
    m_InsideString(false)
{
}

VmaJsonWriter::~VmaJsonWriter()
{
    VMA_ASSERT(!m_InsideString);
    VMA_ASSERT(m_Stack.empty());
}

void VmaJsonWriter::BeginObject(bool singleLine)
{
    VMA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add('{');

    StackItem item;
    item.type = COLLECTION_TYPE_OBJECT;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    m_Stack.push_back(item);
}

void VmaJsonWriter::EndObject()
{
    VMA_ASSERT(!m_InsideString);

    WriteIndent(true);
    m_SB.Add('}');

    VMA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_OBJECT);
    m_Stack.pop_back();
}

void VmaJsonWriter::BeginArray(bool singleLine)
{
    VMA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add('[');

    StackItem item;
    item.type = COLLECTION_TYPE_ARRAY;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    m_Stack.push_back(item);
}

void VmaJsonWriter::EndArray()
{
    VMA_ASSERT(!m_InsideString);

    WriteIndent(true);
    m_SB.Add(']');

    VMA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_ARRAY);
    m_Stack.pop_back();
}

void VmaJsonWriter::WriteString(const char* pStr)
{
    BeginString(pStr);
    EndString();
}

void VmaJsonWriter::BeginString(const char* pStr)
{
    VMA_ASSERT(!m_InsideString);

    BeginValue(true);
    m_SB.Add('"');
    m_InsideString = true;
    if(pStr != VMA_NULL && pStr[0] != '\0')
    {
        ContinueString(pStr);
    }
}

void VmaJsonWriter::ContinueString(const char* pStr)
{
    VMA_ASSERT(m_InsideString);

    const size_t strLen = strlen(pStr);
    for(size_t i = 0; i < strLen; ++i)
    {
        char ch = pStr[i];
        if(ch == '\\')
        {
            m_SB.Add("\\\\");
        }
        else if(ch == '"')
        {
            m_SB.Add("\\\"");
        }
        else if(ch >= 32)
        {
            m_SB.Add(ch);
        }
        else switch(ch)
        {
        case '\b':
            m_SB.Add("\\b");
            break;
        case '\f':
            m_SB.Add("\\f");
            break;
        case '\n':
            m_SB.Add("\\n");
            break;
        case '\r':
            m_SB.Add("\\r");
            break;
        case '\t':
            m_SB.Add("\\t");
            break;
        default:
            VMA_ASSERT(0 && "Character not currently supported.");
            break;
        }
    }
}

void VmaJsonWriter::ContinueString(uint32_t n)
{
    VMA_ASSERT(m_InsideString);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::ContinueString(uint64_t n)
{
    VMA_ASSERT(m_InsideString);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::ContinueString_Pointer(const void* ptr)
{
    VMA_ASSERT(m_InsideString);
    m_SB.AddPointer(ptr);
}

void VmaJsonWriter::EndString(const char* pStr)
{
    VMA_ASSERT(m_InsideString);
    if(pStr != VMA_NULL && pStr[0] != '\0')
    {
        ContinueString(pStr);
    }
    m_SB.Add('"');
    m_InsideString = false;
}

void VmaJsonWriter::WriteNumber(uint32_t n)
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::WriteNumber(uint64_t n)
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::WriteBool(bool b)
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.Add(b ? "true" : "false");
}

void VmaJsonWriter::WriteNull()
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.Add("null");
}

void VmaJsonWriter::BeginValue(bool isString)
{
    if(!m_Stack.empty())
    {
        StackItem& currItem = m_Stack.back();
        if(currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 == 0)
        {
            VMA_ASSERT(isString);
        }

        if(currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 != 0)
        {
            m_SB.Add(": ");
        }
        else if(currItem.valueCount > 0)
        {
            m_SB.Add(", ");
            WriteIndent();
        }
        else
        {
            WriteIndent();
        }
        ++currItem.valueCount;
    }
}

void VmaJsonWriter::WriteIndent(bool oneLess)
{
    if(!m_Stack.empty() && !m_Stack.back().singleLineMode)
    {
        m_SB.AddNewLine();
        
        size_t count = m_Stack.size();
        if(count > 0 && oneLess)
        {
            --count;
        }
        for(size_t i = 0; i < count; ++i)
        {
            m_SB.Add(INDENT);
        }
    }
}

#endif // #if VMA_STATS_STRING_ENABLED

////////////////////////////////////////////////////////////////////////////////

void VmaAllocation_T::SetUserData(VmaAllocator hAllocator, void* pUserData)
{
    if(IsUserDataString())
    {
        VMA_ASSERT(pUserData == VMA_NULL || pUserData != m_pUserData);

        FreeUserDataString(hAllocator);

        if(pUserData != VMA_NULL)
        {
            m_pUserData = VmaCreateStringCopy(hAllocator->GetAllocationCallbacks(), (const char*)pUserData);
        }
    }
    else
    {
        m_pUserData = pUserData;
    }
}

void VmaAllocation_T::ChangeBlockAllocation(
    VmaAllocator hAllocator,
    VmaDeviceMemoryBlock* block,
    VkDeviceSize offset)
{
    VMA_ASSERT(block != VMA_NULL);
    VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK);

    // Move mapping reference counter from old block to new block.
    if(block != m_BlockAllocation.m_Block)
    {
        uint32_t mapRefCount = m_MapCount & ~MAP_COUNT_FLAG_PERSISTENT_MAP;
        if(IsPersistentMap())
            ++mapRefCount;
        m_BlockAllocation.m_Block->Unmap(hAllocator, mapRefCount);
        block->Map(hAllocator, mapRefCount, VMA_NULL);
    }

    m_BlockAllocation.m_Block = block;
    m_BlockAllocation.m_Offset = offset;
}

void VmaAllocation_T::ChangeOffset(VkDeviceSize newOffset)
{
    VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK);
    m_BlockAllocation.m_Offset = newOffset;
}

VkDeviceSize VmaAllocation_T::GetOffset() const
{
    switch(m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Offset;
    case ALLOCATION_TYPE_DEDICATED:
        return 0;
    default:
        VMA_ASSERT(0);
        return 0;
    }
}

VkDeviceMemory VmaAllocation_T::GetMemory() const
{
    switch(m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->GetDeviceMemory();
    case ALLOCATION_TYPE_DEDICATED:
        return m_DedicatedAllocation.m_hMemory;
    default:
        VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

void* VmaAllocation_T::GetMappedData() const
{
    switch(m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        if(m_MapCount != 0)
        {
            void* pBlockData = m_BlockAllocation.m_Block->GetMappedData();
            VMA_ASSERT(pBlockData != VMA_NULL);
            return (char*)pBlockData + m_BlockAllocation.m_Offset;
        }
        else
        {
            return VMA_NULL;
        }
        break;
    case ALLOCATION_TYPE_DEDICATED:
        VMA_ASSERT((m_DedicatedAllocation.m_pMappedData != VMA_NULL) == (m_MapCount != 0));
        return m_DedicatedAllocation.m_pMappedData;
    default:
        VMA_ASSERT(0);
        return VMA_NULL;
    }
}

bool VmaAllocation_T::CanBecomeLost() const
{
    switch(m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_CanBecomeLost;
    case ALLOCATION_TYPE_DEDICATED:
        return false;
    default:
        VMA_ASSERT(0);
        return false;
    }
}

bool VmaAllocation_T::MakeLost(uint32_t currentFrameIndex, uint32_t frameInUseCount)
{
    VMA_ASSERT(CanBecomeLost());

    /*
    Warning: This is a carefully designed algorithm.
    Do not modify unless you really know what you're doing :)
    */
    uint32_t localLastUseFrameIndex = GetLastUseFrameIndex();
    for(;;)
    {
        if(localLastUseFrameIndex == VMA_FRAME_INDEX_LOST)
        {
            VMA_ASSERT(0);
            return false;
        }
        else if(localLastUseFrameIndex + frameInUseCount >= currentFrameIndex)
        {
            return false;
        }
        else // Last use time earlier than current time.
        {
            if(CompareExchangeLastUseFrameIndex(localLastUseFrameIndex, VMA_FRAME_INDEX_LOST))
            {
                // Setting hAllocation.LastUseFrameIndex atomic to VMA_FRAME_INDEX_LOST is enough to mark it as LOST.
                // Calling code just needs to unregister this allocation in owning VmaDeviceMemoryBlock.
                return true;
            }
        }
    }
}

#if VMA_STATS_STRING_ENABLED

// Correspond to values of enum VmaSuballocationType.
static const char* VMA_SUBALLOCATION_TYPE_NAMES[] = {
    "FREE",
    "UNKNOWN",
    "BUFFER",
    "IMAGE_UNKNOWN",
    "IMAGE_LINEAR",
    "IMAGE_OPTIMAL",
};

void VmaAllocation_T::PrintParameters(class VmaJsonWriter& json) const
{
    json.WriteString("Type");
    json.WriteString(VMA_SUBALLOCATION_TYPE_NAMES[m_SuballocationType]);

    json.WriteString("Size");
    json.WriteNumber(m_Size);

    if(m_pUserData != VMA_NULL)
    {
        json.WriteString("UserData");
        if(IsUserDataString())
        {
            json.WriteString((const char*)m_pUserData);
        }
        else
        {
            json.BeginString();
            json.ContinueString_Pointer(m_pUserData);
            json.EndString();
        }
    }

    json.WriteString("CreationFrameIndex");
    json.WriteNumber(m_CreationFrameIndex);

    json.WriteString("LastUseFrameIndex");
    json.WriteNumber(GetLastUseFrameIndex());

    if(m_BufferImageUsage != 0)
    {
        json.WriteString("Usage");
        json.WriteNumber(m_BufferImageUsage);
    }
}

#endif

void VmaAllocation_T::FreeUserDataString(VmaAllocator hAllocator)
{
    VMA_ASSERT(IsUserDataString());
    VmaFreeString(hAllocator->GetAllocationCallbacks(), (char*)m_pUserData);
    m_pUserData = VMA_NULL;
}

void VmaAllocation_T::BlockAllocMap()
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_BLOCK);

    if((m_MapCount & ~MAP_COUNT_FLAG_PERSISTENT_MAP) < 0x7F)
    {
        ++m_MapCount;
    }
    else
    {
        VMA_ASSERT(0 && "Allocation mapped too many times simultaneously.");
    }
}

void VmaAllocation_T::BlockAllocUnmap()
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_BLOCK);

    if((m_MapCount & ~MAP_COUNT_FLAG_PERSISTENT_MAP) != 0)
    {
        --m_MapCount;
    }
    else
    {
        VMA_ASSERT(0 && "Unmapping allocation not previously mapped.");
    }
}

VkResult VmaAllocation_T::DedicatedAllocMap(VmaAllocator hAllocator, void** ppData)
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_DEDICATED);

    if(m_MapCount != 0)
    {
        if((m_MapCount & ~MAP_COUNT_FLAG_PERSISTENT_MAP) < 0x7F)
        {
            VMA_ASSERT(m_DedicatedAllocation.m_pMappedData != VMA_NULL);
            *ppData = m_DedicatedAllocation.m_pMappedData;
            ++m_MapCount;
            return VK_SUCCESS;
        }
        else
        {
            VMA_ASSERT(0 && "Dedicated allocation mapped too many times simultaneously.");
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
    }
    else
    {
        VkResult result = (*hAllocator->GetVulkanFunctions().vkMapMemory)(
            hAllocator->m_hDevice,
            m_DedicatedAllocation.m_hMemory,
            0, // offset
            VK_WHOLE_SIZE,
            0, // flags
            ppData);
        if(result == VK_SUCCESS)
        {
            m_DedicatedAllocation.m_pMappedData = *ppData;
            m_MapCount = 1;
        }
        return result;
    }
}

void VmaAllocation_T::DedicatedAllocUnmap(VmaAllocator hAllocator)
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_DEDICATED);

    if((m_MapCount & ~MAP_COUNT_FLAG_PERSISTENT_MAP) != 0)
    {
        --m_MapCount;
        if(m_MapCount == 0)
        {
            m_DedicatedAllocation.m_pMappedData = VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(
                hAllocator->m_hDevice,
                m_DedicatedAllocation.m_hMemory);
        }
    }
    else
    {
        VMA_ASSERT(0 && "Unmapping dedicated allocation not previously mapped.");
    }
}

#if VMA_STATS_STRING_ENABLED

static void VmaPrintStatInfo(VmaJsonWriter& json, const VmaStatInfo& stat)
{
    json.BeginObject();

    json.WriteString("Blocks");
    json.WriteNumber(stat.blockCount);

    json.WriteString("Allocations");
    json.WriteNumber(stat.allocationCount);

    json.WriteString("UnusedRanges");
    json.WriteNumber(stat.unusedRangeCount);

    json.WriteString("UsedBytes");
    json.WriteNumber(stat.usedBytes);

    json.WriteString("UnusedBytes");
    json.WriteNumber(stat.unusedBytes);

    if(stat.allocationCount > 1)
    {
        json.WriteString("AllocationSize");
        json.BeginObject(true);
        json.WriteString("Min");
        json.WriteNumber(stat.allocationSizeMin);
        json.WriteString("Avg");
        json.WriteNumber(stat.allocationSizeAvg);
        json.WriteString("Max");
        json.WriteNumber(stat.allocationSizeMax);
        json.EndObject();
    }

    if(stat.unusedRangeCount > 1)
    {
        json.WriteString("UnusedRangeSize");
        json.BeginObject(true);
        json.WriteString("Min");
        json.WriteNumber(stat.unusedRangeSizeMin);
        json.WriteString("Avg");
        json.WriteNumber(stat.unusedRangeSizeAvg);
        json.WriteString("Max");
        json.WriteNumber(stat.unusedRangeSizeMax);
        json.EndObject();
    }

    json.EndObject();
}

#endif // #if VMA_STATS_STRING_ENABLED

struct VmaSuballocationItemSizeLess
{
    bool operator()(
        const VmaSuballocationList::iterator lhs,
        const VmaSuballocationList::iterator rhs) const
    {
        return lhs->size < rhs->size;
    }
    bool operator()(
        const VmaSuballocationList::iterator lhs,
        VkDeviceSize rhsSize) const
    {
        return lhs->size < rhsSize;
    }
};


////////////////////////////////////////////////////////////////////////////////
// class VmaBlockMetadata

VmaBlockMetadata::VmaBlockMetadata(VmaAllocator hAllocator) :
    m_Size(0),
    m_pAllocationCallbacks(hAllocator->GetAllocationCallbacks())
{
}

#if VMA_STATS_STRING_ENABLED

void VmaBlockMetadata::PrintDetailedMap_Begin(class VmaJsonWriter& json,
    VkDeviceSize unusedBytes,
    size_t allocationCount,
    size_t unusedRangeCount) const
{
    json.BeginObject();

    json.WriteString("TotalBytes");
    json.WriteNumber(GetSize());

    json.WriteString("UnusedBytes");
    json.WriteNumber(unusedBytes);

    json.WriteString("Allocations");
    json.WriteNumber((uint64_t)allocationCount);

    json.WriteString("UnusedRanges");
    json.WriteNumber((uint64_t)unusedRangeCount);

    json.WriteString("Suballocations");
    json.BeginArray();
}

void VmaBlockMetadata::PrintDetailedMap_Allocation(class VmaJsonWriter& json,
    VkDeviceSize offset,
    VmaAllocation hAllocation) const
{
    json.BeginObject(true);
        
    json.WriteString("Offset");
    json.WriteNumber(offset);

    hAllocation->PrintParameters(json);

    json.EndObject();
}

void VmaBlockMetadata::PrintDetailedMap_UnusedRange(class VmaJsonWriter& json,
    VkDeviceSize offset,
    VkDeviceSize size) const
{
    json.BeginObject(true);
        
    json.WriteString("Offset");
    json.WriteNumber(offset);

    json.WriteString("Type");
    json.WriteString(VMA_SUBALLOCATION_TYPE_NAMES[VMA_SUBALLOCATION_TYPE_FREE]);

    json.WriteString("Size");
    json.WriteNumber(size);

    json.EndObject();
}

void VmaBlockMetadata::PrintDetailedMap_End(class VmaJsonWriter& json) const
{
    json.EndArray();
    json.EndObject();
}

#endif // #if VMA_STATS_STRING_ENABLED

////////////////////////////////////////////////////////////////////////////////
// class VmaBlockMetadata_Generic

VmaBlockMetadata_Generic::VmaBlockMetadata_Generic(VmaAllocator hAllocator) :
    VmaBlockMetadata(hAllocator),
    m_FreeCount(0),
    m_SumFreeSize(0),
    m_Suballocations(VmaStlAllocator<VmaSuballocation>(hAllocator->GetAllocationCallbacks())),
    m_FreeSuballocationsBySize(VmaStlAllocator<VmaSuballocationList::iterator>(hAllocator->GetAllocationCallbacks()))
{
}

VmaBlockMetadata_Generic::~VmaBlockMetadata_Generic()
{
}

void VmaBlockMetadata_Generic::Init(VkDeviceSize size)
{
    VmaBlockMetadata::Init(size);

    m_FreeCount = 1;
    m_SumFreeSize = size;

    VmaSuballocation suballoc = {};
    suballoc.offset = 0;
    suballoc.size = size;
    suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
    suballoc.hAllocation = VK_NULL_HANDLE;

    VMA_ASSERT(size > VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER);
    m_Suballocations.push_back(suballoc);
    VmaSuballocationList::iterator suballocItem = m_Suballocations.end();
    --suballocItem;
    m_FreeSuballocationsBySize.push_back(suballocItem);
}

bool VmaBlockMetadata_Generic::Validate() const
{
    VMA_VALIDATE(!m_Suballocations.empty());
    
    // Expected offset of new suballocation as calculated from previous ones.
    VkDeviceSize calculatedOffset = 0;
    // Expected number of free suballocations as calculated from traversing their list.
    uint32_t calculatedFreeCount = 0;
    // Expected sum size of free suballocations as calculated from traversing their list.
    VkDeviceSize calculatedSumFreeSize = 0;
    // Expected number of free suballocations that should be registered in
    // m_FreeSuballocationsBySize calculated from traversing their list.
    size_t freeSuballocationsToRegister = 0;
    // True if previous visited suballocation was free.
    bool prevFree = false;

    for(VmaSuballocationList::const_iterator suballocItem = m_Suballocations.cbegin();
        suballocItem != m_Suballocations.cend();
        ++suballocItem)
    {
        const VmaSuballocation& subAlloc = *suballocItem;
        
        // Actual offset of this suballocation doesn't match expected one.
        VMA_VALIDATE(subAlloc.offset == calculatedOffset);

        const bool currFree = (subAlloc.type == VMA_SUBALLOCATION_TYPE_FREE);
        // Two adjacent free suballocations are invalid. They should be merged.
        VMA_VALIDATE(!prevFree || !currFree);

        VMA_VALIDATE(currFree == (subAlloc.hAllocation == VK_NULL_HANDLE));

        if(currFree)
        {
            calculatedSumFreeSize += subAlloc.size;
            ++calculatedFreeCount;
            if(subAlloc.size >= VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
            {
                ++freeSuballocationsToRegister;
            }

            // Margin required between allocations - every free space must be at least that large.
            VMA_VALIDATE(subAlloc.size >= VMA_DEBUG_MARGIN);
        }
        else
        {
            VMA_VALIDATE(subAlloc.hAllocation->GetOffset() == subAlloc.offset);
            VMA_VALIDATE(subAlloc.hAllocation->GetSize() == subAlloc.size);

            // Margin required between allocations - previous allocation must be free.
            VMA_VALIDATE(VMA_DEBUG_MARGIN == 0 || prevFree);
        }

        calculatedOffset += subAlloc.size;
        prevFree = currFree;
    }

    // Number of free suballocations registered in m_FreeSuballocationsBySize doesn't
    // match expected one.
    VMA_VALIDATE(m_FreeSuballocationsBySize.size() == freeSuballocationsToRegister);

    VkDeviceSize lastSize = 0;
    for(size_t i = 0; i < m_FreeSuballocationsBySize.size(); ++i)
    {
        VmaSuballocationList::iterator suballocItem = m_FreeSuballocationsBySize[i];
        
        // Only free suballocations can be registered in m_FreeSuballocationsBySize.
        VMA_VALIDATE(suballocItem->type == VMA_SUBALLOCATION_TYPE_FREE);
        // They must be sorted by size ascending.
        VMA_VALIDATE(suballocItem->size >= lastSize);

        lastSize = suballocItem->size;
    }

    // Check if totals match calculacted values.
    VMA_VALIDATE(ValidateFreeSuballocationList());
    VMA_VALIDATE(calculatedOffset == GetSize());
    VMA_VALIDATE(calculatedSumFreeSize == m_SumFreeSize);
    VMA_VALIDATE(calculatedFreeCount == m_FreeCount);

    return true;
}

VkDeviceSize VmaBlockMetadata_Generic::GetUnusedRangeSizeMax() const
{
    if(!m_FreeSuballocationsBySize.empty())
    {
        return m_FreeSuballocationsBySize.back()->size;
    }
    else
    {
        return 0;
    }
}

bool VmaBlockMetadata_Generic::IsEmpty() const
{
    return (m_Suballocations.size() == 1) && (m_FreeCount == 1);
}

void VmaBlockMetadata_Generic::CalcAllocationStatInfo(VmaStatInfo& outInfo) const
{
    outInfo.blockCount = 1;

    const uint32_t rangeCount = (uint32_t)m_Suballocations.size();
    outInfo.allocationCount = rangeCount - m_FreeCount;
    outInfo.unusedRangeCount = m_FreeCount;
    
    outInfo.unusedBytes = m_SumFreeSize;
    outInfo.usedBytes = GetSize() - outInfo.unusedBytes;

    outInfo.allocationSizeMin = UINT64_MAX;
    outInfo.allocationSizeMax = 0;
    outInfo.unusedRangeSizeMin = UINT64_MAX;
    outInfo.unusedRangeSizeMax = 0;

    for(VmaSuballocationList::const_iterator suballocItem = m_Suballocations.cbegin();
        suballocItem != m_Suballocations.cend();
        ++suballocItem)
    {
        const VmaSuballocation& suballoc = *suballocItem;
        if(suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            outInfo.allocationSizeMin = VMA_MIN(outInfo.allocationSizeMin, suballoc.size);
            outInfo.allocationSizeMax = VMA_MAX(outInfo.allocationSizeMax, suballoc.size);
        }
        else
        {
            outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, suballoc.size);
            outInfo.unusedRangeSizeMax = VMA_MAX(outInfo.unusedRangeSizeMax, suballoc.size);
        }
    }
}

void VmaBlockMetadata_Generic::AddPoolStats(VmaPoolStats& inoutStats) const
{
    const uint32_t rangeCount = (uint32_t)m_Suballocations.size();

    inoutStats.size += GetSize();
    inoutStats.unusedSize += m_SumFreeSize;
    inoutStats.allocationCount += rangeCount - m_FreeCount;
    inoutStats.unusedRangeCount += m_FreeCount;
    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, GetUnusedRangeSizeMax());
}

#if VMA_STATS_STRING_ENABLED

void VmaBlockMetadata_Generic::PrintDetailedMap(class VmaJsonWriter& json) const
{
    PrintDetailedMap_Begin(json,
        m_SumFreeSize, // unusedBytes
        m_Suballocations.size() - (size_t)m_FreeCount, // allocationCount
        m_FreeCount); // unusedRangeCount

    size_t i = 0;
    for(VmaSuballocationList::const_iterator suballocItem = m_Suballocations.cbegin();
        suballocItem != m_Suballocations.cend();
        ++suballocItem, ++i)
    {
        if(suballocItem->type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            PrintDetailedMap_UnusedRange(json, suballocItem->offset, suballocItem->size);
        }
        else
        {
            PrintDetailedMap_Allocation(json, suballocItem->offset, suballocItem->hAllocation);
        }
    }

    PrintDetailedMap_End(json);
}

#endif // #if VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Generic::CreateAllocationRequest(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VkDeviceSize bufferImageGranularity,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    bool canMakeOtherLost,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(allocSize > 0);
    VMA_ASSERT(!upperAddress);
    VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(pAllocationRequest != VMA_NULL);
    VMA_HEAVY_ASSERT(Validate());

    pAllocationRequest->type = VmaAllocationRequestType::Normal;

    // There is not enough total free space in this block to fullfill the request: Early return.
    if(canMakeOtherLost == false &&
        m_SumFreeSize < allocSize + 2 * VMA_DEBUG_MARGIN)
    {
        return false;
    }

    // New algorithm, efficiently searching freeSuballocationsBySize.
    const size_t freeSuballocCount = m_FreeSuballocationsBySize.size();
    if(freeSuballocCount > 0)
    {
        if(strategy == VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT)
        {
            // Find first free suballocation with size not less than allocSize + 2 * VMA_DEBUG_MARGIN.
            VmaSuballocationList::iterator* const it = VmaBinaryFindFirstNotLess(
                m_FreeSuballocationsBySize.data(),
                m_FreeSuballocationsBySize.data() + freeSuballocCount,
                allocSize + 2 * VMA_DEBUG_MARGIN,
                VmaSuballocationItemSizeLess());
            size_t index = it - m_FreeSuballocationsBySize.data();
            for(; index < freeSuballocCount; ++index)
            {
                if(CheckAllocation(
                    currentFrameIndex,
                    frameInUseCount,
                    bufferImageGranularity,
                    allocSize,
                    allocAlignment,
                    allocType,
                    m_FreeSuballocationsBySize[index],
                    false, // canMakeOtherLost
                    &pAllocationRequest->offset,
                    &pAllocationRequest->itemsToMakeLostCount,
                    &pAllocationRequest->sumFreeSize,
                    &pAllocationRequest->sumItemSize))
                {
                    pAllocationRequest->item = m_FreeSuballocationsBySize[index];
                    return true;
                }
            }
        }
        else if(strategy == VMA_ALLOCATION_INTERNAL_STRATEGY_MIN_OFFSET)
        {
            for(VmaSuballocationList::iterator it = m_Suballocations.begin();
                it != m_Suballocations.end();
                ++it)
            {
                if(it->type == VMA_SUBALLOCATION_TYPE_FREE && CheckAllocation(
                    currentFrameIndex,
                    frameInUseCount,
                    bufferImageGranularity,
                    allocSize,
                    allocAlignment,
                    allocType,
                    it,
                    false, // canMakeOtherLost
                    &pAllocationRequest->offset,
                    &pAllocationRequest->itemsToMakeLostCount,
                    &pAllocationRequest->sumFreeSize,
                    &pAllocationRequest->sumItemSize))
                {
                    pAllocationRequest->item = it;
                    return true;
                }
            }
        }
        else // WORST_FIT, FIRST_FIT
        {
            // Search staring from biggest suballocations.
            for(size_t index = freeSuballocCount; index--; )
            {
                if(CheckAllocation(
                    currentFrameIndex,
                    frameInUseCount,
                    bufferImageGranularity,
                    allocSize,
                    allocAlignment,
                    allocType,
                    m_FreeSuballocationsBySize[index],
                    false, // canMakeOtherLost
                    &pAllocationRequest->offset,
                    &pAllocationRequest->itemsToMakeLostCount,
                    &pAllocationRequest->sumFreeSize,
                    &pAllocationRequest->sumItemSize))
                {
                    pAllocationRequest->item = m_FreeSuballocationsBySize[index];
                    return true;
                }
            }
        }
    }

    if(canMakeOtherLost)
    {
        // Brute-force algorithm. TODO: Come up with something better.

        bool found = false;
        VmaAllocationRequest tmpAllocRequest = {};
        tmpAllocRequest.type = VmaAllocationRequestType::Normal;
        for(VmaSuballocationList::iterator suballocIt = m_Suballocations.begin();
            suballocIt != m_Suballocations.end();
            ++suballocIt)
        {
            if(suballocIt->type == VMA_SUBALLOCATION_TYPE_FREE ||
                suballocIt->hAllocation->CanBecomeLost())
            {
                if(CheckAllocation(
                    currentFrameIndex,
                    frameInUseCount,
                    bufferImageGranularity,
                    allocSize,
                    allocAlignment,
                    allocType,
                    suballocIt,
                    canMakeOtherLost,
                    &tmpAllocRequest.offset,
                    &tmpAllocRequest.itemsToMakeLostCount,
                    &tmpAllocRequest.sumFreeSize,
                    &tmpAllocRequest.sumItemSize))
                {
                    if(strategy == VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT)
                    {
                        *pAllocationRequest = tmpAllocRequest;
                        pAllocationRequest->item = suballocIt;
                        break;
                    }
                    if(!found || tmpAllocRequest.CalcCost() < pAllocationRequest->CalcCost())
                    {
                        *pAllocationRequest = tmpAllocRequest;
                        pAllocationRequest->item = suballocIt;
                        found = true;
                    }
                }
            }
        }

        return found;
    }

    return false;
}

bool VmaBlockMetadata_Generic::MakeRequestedAllocationsLost(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(pAllocationRequest && pAllocationRequest->type == VmaAllocationRequestType::Normal);

    while(pAllocationRequest->itemsToMakeLostCount > 0)
    {
        if(pAllocationRequest->item->type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            ++pAllocationRequest->item;
        }
        VMA_ASSERT(pAllocationRequest->item != m_Suballocations.end());
        VMA_ASSERT(pAllocationRequest->item->hAllocation != VK_NULL_HANDLE);
        VMA_ASSERT(pAllocationRequest->item->hAllocation->CanBecomeLost());
        if(pAllocationRequest->item->hAllocation->MakeLost(currentFrameIndex, frameInUseCount))
        {
            pAllocationRequest->item = FreeSuballocation(pAllocationRequest->item);
            --pAllocationRequest->itemsToMakeLostCount;
        }
        else
        {
            return false;
        }
    }

    VMA_HEAVY_ASSERT(Validate());
    VMA_ASSERT(pAllocationRequest->item != m_Suballocations.end());
    VMA_ASSERT(pAllocationRequest->item->type == VMA_SUBALLOCATION_TYPE_FREE);
    
    return true;
}

uint32_t VmaBlockMetadata_Generic::MakeAllocationsLost(uint32_t currentFrameIndex, uint32_t frameInUseCount)
{
    uint32_t lostAllocationCount = 0;
    for(VmaSuballocationList::iterator it = m_Suballocations.begin();
        it != m_Suballocations.end();
        ++it)
    {
        if(it->type != VMA_SUBALLOCATION_TYPE_FREE &&
            it->hAllocation->CanBecomeLost() &&
            it->hAllocation->MakeLost(currentFrameIndex, frameInUseCount))
        {
            it = FreeSuballocation(it);
            ++lostAllocationCount;
        }
    }
    return lostAllocationCount;
}

VkResult VmaBlockMetadata_Generic::CheckCorruption(const void* pBlockData)
{
    for(VmaSuballocationList::iterator it = m_Suballocations.begin();
        it != m_Suballocations.end();
        ++it)
    {
        if(it->type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if(!VmaValidateMagicValue(pBlockData, it->offset - VMA_DEBUG_MARGIN))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED BEFORE VALIDATED ALLOCATION!");
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
            if(!VmaValidateMagicValue(pBlockData, it->offset + it->size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
        }
    }

    return VK_SUCCESS;
}

void VmaBlockMetadata_Generic::Alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    VkDeviceSize allocSize,
    VmaAllocation hAllocation)
{
    VMA_ASSERT(request.type == VmaAllocationRequestType::Normal);
    VMA_ASSERT(request.item != m_Suballocations.end());
    VmaSuballocation& suballoc = *request.item;
    // Given suballocation is a free block.
    VMA_ASSERT(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);
    // Given offset is inside this suballocation.
    VMA_ASSERT(request.offset >= suballoc.offset);
    const VkDeviceSize paddingBegin = request.offset - suballoc.offset;
    VMA_ASSERT(suballoc.size >= paddingBegin + allocSize);
    const VkDeviceSize paddingEnd = suballoc.size - paddingBegin - allocSize;

    // Unregister this free suballocation from m_FreeSuballocationsBySize and update
    // it to become used.
    UnregisterFreeSuballocation(request.item);

    suballoc.offset = request.offset;
    suballoc.size = allocSize;
    suballoc.type = type;
    suballoc.hAllocation = hAllocation;

    // If there are any free bytes remaining at the end, insert new free suballocation after current one.
    if(paddingEnd)
    {
        VmaSuballocation paddingSuballoc = {};
        paddingSuballoc.offset = request.offset + allocSize;
        paddingSuballoc.size = paddingEnd;
        paddingSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
        VmaSuballocationList::iterator next = request.item;
        ++next;
        const VmaSuballocationList::iterator paddingEndItem =
            m_Suballocations.insert(next, paddingSuballoc);
        RegisterFreeSuballocation(paddingEndItem);
    }

    // If there are any free bytes remaining at the beginning, insert new free suballocation before current one.
    if(paddingBegin)
    {
        VmaSuballocation paddingSuballoc = {};
        paddingSuballoc.offset = request.offset - paddingBegin;
        paddingSuballoc.size = paddingBegin;
        paddingSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
        const VmaSuballocationList::iterator paddingBeginItem =
            m_Suballocations.insert(request.item, paddingSuballoc);
        RegisterFreeSuballocation(paddingBeginItem);
    }

    // Update totals.
    m_FreeCount = m_FreeCount - 1;
    if(paddingBegin > 0)
    {
        ++m_FreeCount;
    }
    if(paddingEnd > 0)
    {
        ++m_FreeCount;
    }
    m_SumFreeSize -= allocSize;
}

void VmaBlockMetadata_Generic::Free(const VmaAllocation allocation)
{
    for(VmaSuballocationList::iterator suballocItem = m_Suballocations.begin();
        suballocItem != m_Suballocations.end();
        ++suballocItem)
    {
        VmaSuballocation& suballoc = *suballocItem;
        if(suballoc.hAllocation == allocation)
        {
            FreeSuballocation(suballocItem);
            VMA_HEAVY_ASSERT(Validate());
            return;
        }
    }
    VMA_ASSERT(0 && "Not found!");
}

void VmaBlockMetadata_Generic::FreeAtOffset(VkDeviceSize offset)
{
    for(VmaSuballocationList::iterator suballocItem = m_Suballocations.begin();
        suballocItem != m_Suballocations.end();
        ++suballocItem)
    {
        VmaSuballocation& suballoc = *suballocItem;
        if(suballoc.offset == offset)
        {
            FreeSuballocation(suballocItem);
            return;
        }
    }
    VMA_ASSERT(0 && "Not found!");
}

bool VmaBlockMetadata_Generic::ValidateFreeSuballocationList() const
{
    VkDeviceSize lastSize = 0;
    for(size_t i = 0, count = m_FreeSuballocationsBySize.size(); i < count; ++i)
    {
        const VmaSuballocationList::iterator it = m_FreeSuballocationsBySize[i];

        VMA_VALIDATE(it->type == VMA_SUBALLOCATION_TYPE_FREE);
        VMA_VALIDATE(it->size >= VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER);
        VMA_VALIDATE(it->size >= lastSize);
        lastSize = it->size;
    }
    return true;
}

bool VmaBlockMetadata_Generic::CheckAllocation(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VkDeviceSize bufferImageGranularity,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    VmaSuballocationList::const_iterator suballocItem,
    bool canMakeOtherLost,
    VkDeviceSize* pOffset,
    size_t* itemsToMakeLostCount,
    VkDeviceSize* pSumFreeSize,
    VkDeviceSize* pSumItemSize) const
{
    VMA_ASSERT(allocSize > 0);
    VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(suballocItem != m_Suballocations.cend());
    VMA_ASSERT(pOffset != VMA_NULL);
    
    *itemsToMakeLostCount = 0;
    *pSumFreeSize = 0;
    *pSumItemSize = 0;

    if(canMakeOtherLost)
    {
        if(suballocItem->type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            *pSumFreeSize = suballocItem->size;
        }
        else
        {
            if(suballocItem->hAllocation->CanBecomeLost() &&
                suballocItem->hAllocation->GetLastUseFrameIndex() + frameInUseCount < currentFrameIndex)
            {
                ++*itemsToMakeLostCount;
                *pSumItemSize = suballocItem->size;
            }
            else
            {
                return false;
            }
        }

        // Remaining size is too small for this request: Early return.
        if(GetSize() - suballocItem->offset < allocSize)
        {
            return false;
        }

        // Start from offset equal to beginning of this suballocation.
        *pOffset = suballocItem->offset;
    
        // Apply VMA_DEBUG_MARGIN at the beginning.
        if(VMA_DEBUG_MARGIN > 0)
        {
            *pOffset += VMA_DEBUG_MARGIN;
        }
    
        // Apply alignment.
        *pOffset = VmaAlignUp(*pOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if(bufferImageGranularity > 1)
        {
            bool bufferImageGranularityConflict = false;
            VmaSuballocationList::const_iterator prevSuballocItem = suballocItem;
            while(prevSuballocItem != m_Suballocations.cbegin())
            {
                --prevSuballocItem;
                const VmaSuballocation& prevSuballoc = *prevSuballocItem;
                if(VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, *pOffset, bufferImageGranularity))
                {
                    if(VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if(bufferImageGranularityConflict)
            {
                *pOffset = VmaAlignUp(*pOffset, bufferImageGranularity);
            }
        }
    
        // Now that we have final *pOffset, check if we are past suballocItem.
        // If yes, return false - this function should be called for another suballocItem as starting point.
        if(*pOffset >= suballocItem->offset + suballocItem->size)
        {
            return false;
        }
    
        // Calculate padding at the beginning based on current offset.
        const VkDeviceSize paddingBegin = *pOffset - suballocItem->offset;

        // Calculate required margin at the end.
        const VkDeviceSize requiredEndMargin = VMA_DEBUG_MARGIN;

        const VkDeviceSize totalSize = paddingBegin + allocSize + requiredEndMargin;
        // Another early return check.
        if(suballocItem->offset + totalSize > GetSize())
        {
            return false;
        }

        // Advance lastSuballocItem until desired size is reached.
        // Update itemsToMakeLostCount.
        VmaSuballocationList::const_iterator lastSuballocItem = suballocItem;
        if(totalSize > suballocItem->size)
        {
            VkDeviceSize remainingSize = totalSize - suballocItem->size;
            while(remainingSize > 0)
            {
                ++lastSuballocItem;
                if(lastSuballocItem == m_Suballocations.cend())
                {
                    return false;
                }
                if(lastSuballocItem->type == VMA_SUBALLOCATION_TYPE_FREE)
                {
                    *pSumFreeSize += lastSuballocItem->size;
                }
                else
                {
                    VMA_ASSERT(lastSuballocItem->hAllocation != VK_NULL_HANDLE);
                    if(lastSuballocItem->hAllocation->CanBecomeLost() &&
                        lastSuballocItem->hAllocation->GetLastUseFrameIndex() + frameInUseCount < currentFrameIndex)
                    {
                        ++*itemsToMakeLostCount;
                        *pSumItemSize += lastSuballocItem->size;
                    }
                    else
                    {
                        return false;
                    }
                }
                remainingSize = (lastSuballocItem->size < remainingSize) ?
                    remainingSize - lastSuballocItem->size : 0;
            }
        }

        // Check next suballocations for BufferImageGranularity conflicts.
        // If conflict exists, we must mark more allocations lost or fail.
        if(bufferImageGranularity > 1)
        {
            VmaSuballocationList::const_iterator nextSuballocItem = lastSuballocItem;
            ++nextSuballocItem;
            while(nextSuballocItem != m_Suballocations.cend())
            {
                const VmaSuballocation& nextSuballoc = *nextSuballocItem;
                if(VmaBlocksOnSamePage(*pOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                {
                    if(VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                    {
                        VMA_ASSERT(nextSuballoc.hAllocation != VK_NULL_HANDLE);
                        if(nextSuballoc.hAllocation->CanBecomeLost() &&
                            nextSuballoc.hAllocation->GetLastUseFrameIndex() + frameInUseCount < currentFrameIndex)
                        {
                            ++*itemsToMakeLostCount;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    // Already on next page.
                    break;
                }
                ++nextSuballocItem;
            }
        }
    }
    else
    {
        const VmaSuballocation& suballoc = *suballocItem;
        VMA_ASSERT(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

        *pSumFreeSize = suballoc.size;

        // Size of this suballocation is too small for this request: Early return.
        if(suballoc.size < allocSize)
        {
            return false;
        }

        // Start from offset equal to beginning of this suballocation.
        *pOffset = suballoc.offset;
    
        // Apply VMA_DEBUG_MARGIN at the beginning.
        if(VMA_DEBUG_MARGIN > 0)
        {
            *pOffset += VMA_DEBUG_MARGIN;
        }
    
        // Apply alignment.
        *pOffset = VmaAlignUp(*pOffset, allocAlignment);
    
        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if(bufferImageGranularity > 1)
        {
            bool bufferImageGranularityConflict = false;
            VmaSuballocationList::const_iterator prevSuballocItem = suballocItem;
            while(prevSuballocItem != m_Suballocations.cbegin())
            {
                --prevSuballocItem;
                const VmaSuballocation& prevSuballoc = *prevSuballocItem;
                if(VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, *pOffset, bufferImageGranularity))
                {
                    if(VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if(bufferImageGranularityConflict)
            {
                *pOffset = VmaAlignUp(*pOffset, bufferImageGranularity);
            }
        }
    
        // Calculate padding at the beginning based on current offset.
        const VkDeviceSize paddingBegin = *pOffset - suballoc.offset;

        // Calculate required margin at the end.
        const VkDeviceSize requiredEndMargin = VMA_DEBUG_MARGIN;

        // Fail if requested size plus margin before and after is bigger than size of this suballocation.
        if(paddingBegin + allocSize + requiredEndMargin > suballoc.size)
        {
            return false;
        }

        // Check next suballocations for BufferImageGranularity conflicts.
        // If conflict exists, allocation cannot be made here.
        if(bufferImageGranularity > 1)
        {
            VmaSuballocationList::const_iterator nextSuballocItem = suballocItem;
            ++nextSuballocItem;
            while(nextSuballocItem != m_Suballocations.cend())
            {
                const VmaSuballocation& nextSuballoc = *nextSuballocItem;
                if(VmaBlocksOnSamePage(*pOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                {
                    if(VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                    {
                        return false;
                    }
                }
                else
                {
                    // Already on next page.
                    break;
                }
                ++nextSuballocItem;
            }
        }
    }

    // All tests passed: Success. pOffset is already filled.
    return true;
}

void VmaBlockMetadata_Generic::MergeFreeWithNext(VmaSuballocationList::iterator item)
{
    VMA_ASSERT(item != m_Suballocations.end());
    VMA_ASSERT(item->type == VMA_SUBALLOCATION_TYPE_FREE);
    
    VmaSuballocationList::iterator nextItem = item;
    ++nextItem;
    VMA_ASSERT(nextItem != m_Suballocations.end());
    VMA_ASSERT(nextItem->type == VMA_SUBALLOCATION_TYPE_FREE);

    item->size += nextItem->size;
    --m_FreeCount;
    m_Suballocations.erase(nextItem);
}

VmaSuballocationList::iterator VmaBlockMetadata_Generic::FreeSuballocation(VmaSuballocationList::iterator suballocItem)
{
    // Change this suballocation to be marked as free.
    VmaSuballocation& suballoc = *suballocItem;
    suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
    suballoc.hAllocation = VK_NULL_HANDLE;
    
    // Update totals.
    ++m_FreeCount;
    m_SumFreeSize += suballoc.size;

    // Merge with previous and/or next suballocation if it's also free.
    bool mergeWithNext = false;
    bool mergeWithPrev = false;
    
    VmaSuballocationList::iterator nextItem = suballocItem;
    ++nextItem;
    if((nextItem != m_Suballocations.end()) && (nextItem->type == VMA_SUBALLOCATION_TYPE_FREE))
    {
        mergeWithNext = true;
    }

    VmaSuballocationList::iterator prevItem = suballocItem;
    if(suballocItem != m_Suballocations.begin())
    {
        --prevItem;
        if(prevItem->type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            mergeWithPrev = true;
        }
    }

    if(mergeWithNext)
    {
        UnregisterFreeSuballocation(nextItem);
        MergeFreeWithNext(suballocItem);
    }

    if(mergeWithPrev)
    {
        UnregisterFreeSuballocation(prevItem);
        MergeFreeWithNext(prevItem);
        RegisterFreeSuballocation(prevItem);
        return prevItem;
    }
    else
    {
        RegisterFreeSuballocation(suballocItem);
        return suballocItem;
    }
}

void VmaBlockMetadata_Generic::RegisterFreeSuballocation(VmaSuballocationList::iterator item)
{
    VMA_ASSERT(item->type == VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(item->size > 0);

    // You may want to enable this validation at the beginning or at the end of
    // this function, depending on what do you want to check.
    VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());

    if(item->size >= VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
    {
        if(m_FreeSuballocationsBySize.empty())
        {
            m_FreeSuballocationsBySize.push_back(item);
        }
        else
        {
            VmaVectorInsertSorted<VmaSuballocationItemSizeLess>(m_FreeSuballocationsBySize, item);
        }
    }

    //VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());
}


void VmaBlockMetadata_Generic::UnregisterFreeSuballocation(VmaSuballocationList::iterator item)
{
    VMA_ASSERT(item->type == VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(item->size > 0);

    // You may want to enable this validation at the beginning or at the end of
    // this function, depending on what do you want to check.
    VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());

    if(item->size >= VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
    {
        VmaSuballocationList::iterator* const it = VmaBinaryFindFirstNotLess(
            m_FreeSuballocationsBySize.data(),
            m_FreeSuballocationsBySize.data() + m_FreeSuballocationsBySize.size(),
            item,
            VmaSuballocationItemSizeLess());
        for(size_t index = it - m_FreeSuballocationsBySize.data();
            index < m_FreeSuballocationsBySize.size();
            ++index)
        {
            if(m_FreeSuballocationsBySize[index] == item)
            {
                VmaVectorRemove(m_FreeSuballocationsBySize, index);
                return;
            }
            VMA_ASSERT((m_FreeSuballocationsBySize[index]->size == item->size) && "Not found.");
        }
        VMA_ASSERT(0 && "Not found.");
    }

    //VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());
}

bool VmaBlockMetadata_Generic::IsBufferImageGranularityConflictPossible(
    VkDeviceSize bufferImageGranularity,
    VmaSuballocationType& inOutPrevSuballocType) const
{
    if(bufferImageGranularity == 1 || IsEmpty())
    {
        return false;
    }

    VkDeviceSize minAlignment = VK_WHOLE_SIZE;
    bool typeConflictFound = false;
    for(VmaSuballocationList::const_iterator it = m_Suballocations.cbegin();
        it != m_Suballocations.cend();
        ++it)
    {
        const VmaSuballocationType suballocType = it->type;
        if(suballocType != VMA_SUBALLOCATION_TYPE_FREE)
        {
            minAlignment = VMA_MIN(minAlignment, it->hAllocation->GetAlignment());
            if(VmaIsBufferImageGranularityConflict(inOutPrevSuballocType, suballocType))
            {
                typeConflictFound = true;
            }
            inOutPrevSuballocType = suballocType;
        }
    }

    return typeConflictFound || minAlignment >= bufferImageGranularity;
}

////////////////////////////////////////////////////////////////////////////////
// class VmaBlockMetadata_Linear

VmaBlockMetadata_Linear::VmaBlockMetadata_Linear(VmaAllocator hAllocator) :
    VmaBlockMetadata(hAllocator),
    m_SumFreeSize(0),
    m_Suballocations0(VmaStlAllocator<VmaSuballocation>(hAllocator->GetAllocationCallbacks())),
    m_Suballocations1(VmaStlAllocator<VmaSuballocation>(hAllocator->GetAllocationCallbacks())),
    m_1stVectorIndex(0),
    m_2ndVectorMode(SECOND_VECTOR_EMPTY),
    m_1stNullItemsBeginCount(0),
    m_1stNullItemsMiddleCount(0),
    m_2ndNullItemsCount(0)
{
}

VmaBlockMetadata_Linear::~VmaBlockMetadata_Linear()
{
}

void VmaBlockMetadata_Linear::Init(VkDeviceSize size)
{
    VmaBlockMetadata::Init(size);
    m_SumFreeSize = size;
}

bool VmaBlockMetadata_Linear::Validate() const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    VMA_VALIDATE(suballocations2nd.empty() == (m_2ndVectorMode == SECOND_VECTOR_EMPTY));
    VMA_VALIDATE(!suballocations1st.empty() ||
        suballocations2nd.empty() ||
        m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER);

    if(!suballocations1st.empty())
    {
        // Null item at the beginning should be accounted into m_1stNullItemsBeginCount.
        VMA_VALIDATE(suballocations1st[m_1stNullItemsBeginCount].hAllocation != VK_NULL_HANDLE);
        // Null item at the end should be just pop_back().
        VMA_VALIDATE(suballocations1st.back().hAllocation != VK_NULL_HANDLE);
    }
    if(!suballocations2nd.empty())
    {
        // Null item at the end should be just pop_back().
        VMA_VALIDATE(suballocations2nd.back().hAllocation != VK_NULL_HANDLE);
    }

    VMA_VALIDATE(m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount <= suballocations1st.size());
    VMA_VALIDATE(m_2ndNullItemsCount <= suballocations2nd.size());

    VkDeviceSize sumUsedSize = 0;
    const size_t suballoc1stCount = suballocations1st.size();
    VkDeviceSize offset = VMA_DEBUG_MARGIN;

    if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for(size_t i = 0; i < suballoc2ndCount; ++i)
        {
            const VmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

            VMA_VALIDATE(currFree == (suballoc.hAllocation == VK_NULL_HANDLE));
            VMA_VALIDATE(suballoc.offset >= offset);

            if(!currFree)
            {
                VMA_VALIDATE(suballoc.hAllocation->GetOffset() == suballoc.offset);
                VMA_VALIDATE(suballoc.hAllocation->GetSize() == suballoc.size);
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + VMA_DEBUG_MARGIN;
        }

        VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    for(size_t i = 0; i < m_1stNullItemsBeginCount; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        VMA_VALIDATE(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE &&
            suballoc.hAllocation == VK_NULL_HANDLE);
    }

    size_t nullItem1stCount = m_1stNullItemsBeginCount;

    for(size_t i = m_1stNullItemsBeginCount; i < suballoc1stCount; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

        VMA_VALIDATE(currFree == (suballoc.hAllocation == VK_NULL_HANDLE));
        VMA_VALIDATE(suballoc.offset >= offset);
        VMA_VALIDATE(i >= m_1stNullItemsBeginCount || currFree);

        if(!currFree)
        {
            VMA_VALIDATE(suballoc.hAllocation->GetOffset() == suballoc.offset);
            VMA_VALIDATE(suballoc.hAllocation->GetSize() == suballoc.size);
            sumUsedSize += suballoc.size;
        }
        else
        {
            ++nullItem1stCount;
        }

        offset = suballoc.offset + suballoc.size + VMA_DEBUG_MARGIN;
    }
    VMA_VALIDATE(nullItem1stCount == m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount);

    if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for(size_t i = suballoc2ndCount; i--; )
        {
            const VmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

            VMA_VALIDATE(currFree == (suballoc.hAllocation == VK_NULL_HANDLE));
            VMA_VALIDATE(suballoc.offset >= offset);

            if(!currFree)
            {
                VMA_VALIDATE(suballoc.hAllocation->GetOffset() == suballoc.offset);
                VMA_VALIDATE(suballoc.hAllocation->GetSize() == suballoc.size);
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + VMA_DEBUG_MARGIN;
        }

        VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    VMA_VALIDATE(offset <= GetSize());
    VMA_VALIDATE(m_SumFreeSize == GetSize() - sumUsedSize);

    return true;
}

size_t VmaBlockMetadata_Linear::GetAllocationCount() const
{
    return AccessSuballocations1st().size() - (m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount) +
        AccessSuballocations2nd().size() - m_2ndNullItemsCount;
}

VkDeviceSize VmaBlockMetadata_Linear::GetUnusedRangeSizeMax() const
{
    const VkDeviceSize size = GetSize();

    /*
    We don't consider gaps inside allocation vectors with freed allocations because
    they are not suitable for reuse in linear allocator. We consider only space that
    is available for new allocations.
    */
    if(IsEmpty())
    {
        return size;
    }
    
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();

    switch(m_2ndVectorMode)
    {
    case SECOND_VECTOR_EMPTY:
        /*
        Available space is after end of 1st, as well as before beginning of 1st (which
        whould make it a ring buffer).
        */
        {
            const size_t suballocations1stCount = suballocations1st.size();
            VMA_ASSERT(suballocations1stCount > m_1stNullItemsBeginCount);
            const VmaSuballocation& firstSuballoc = suballocations1st[m_1stNullItemsBeginCount];
            const VmaSuballocation& lastSuballoc  = suballocations1st[suballocations1stCount - 1];
            return VMA_MAX(
                firstSuballoc.offset,
                size - (lastSuballoc.offset + lastSuballoc.size));
        }
        break;

    case SECOND_VECTOR_RING_BUFFER:
        /*
        Available space is only between end of 2nd and beginning of 1st.
        */
        {
            const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
            const VmaSuballocation& lastSuballoc2nd = suballocations2nd.back();
            const VmaSuballocation& firstSuballoc1st = suballocations1st[m_1stNullItemsBeginCount];
            return firstSuballoc1st.offset - (lastSuballoc2nd.offset + lastSuballoc2nd.size);
        }
        break;

    case SECOND_VECTOR_DOUBLE_STACK:
        /*
        Available space is only between end of 1st and top of 2nd.
        */
        {
            const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
            const VmaSuballocation& topSuballoc2nd = suballocations2nd.back();
            const VmaSuballocation& lastSuballoc1st = suballocations1st.back();
            return topSuballoc2nd.offset - (lastSuballoc1st.offset + lastSuballoc1st.size);
        }
        break;

    default:
        VMA_ASSERT(0);
        return 0;
    }
}

void VmaBlockMetadata_Linear::CalcAllocationStatInfo(VmaStatInfo& outInfo) const
{
    const VkDeviceSize size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    outInfo.blockCount = 1;
    outInfo.allocationCount = (uint32_t)GetAllocationCount();
    outInfo.unusedRangeCount = 0;
    outInfo.usedBytes = 0;
    outInfo.allocationSizeMin = UINT64_MAX;
    outInfo.allocationSizeMax = 0;
    outInfo.unusedRangeSizeMin = UINT64_MAX;
    outInfo.unusedRangeSizeMax = 0;

    VkDeviceSize lastOffset = 0;

    if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while(lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while(nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    ++outInfo.unusedRangeCount;
                    outInfo.unusedBytes += unusedRangeSize;
                    outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, unusedRangeSize);
                    outInfo.unusedRangeSizeMax = VMA_MIN(outInfo.unusedRangeSizeMax, unusedRangeSize);
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                outInfo.usedBytes += suballoc.size;
                outInfo.allocationSizeMin = VMA_MIN(outInfo.allocationSizeMin, suballoc.size);
                outInfo.allocationSizeMax = VMA_MIN(outInfo.allocationSizeMax, suballoc.size);
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                if(lastOffset < freeSpace2ndTo1stEnd)
                {
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    ++outInfo.unusedRangeCount;
                    outInfo.unusedBytes += unusedRangeSize;
                    outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, unusedRangeSize);
                    outInfo.unusedRangeSizeMax = VMA_MIN(outInfo.unusedRangeSizeMax, unusedRangeSize);
               }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while(lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while(nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].hAllocation == VK_NULL_HANDLE)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if(nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];
            
            // 1. Process free space before this allocation.
            if(lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                ++outInfo.unusedRangeCount;
                outInfo.unusedBytes += unusedRangeSize;
                outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, unusedRangeSize);
                outInfo.unusedRangeSizeMax = VMA_MIN(outInfo.unusedRangeSizeMax, unusedRangeSize);
            }
            
            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            outInfo.usedBytes += suballoc.size;
            outInfo.allocationSizeMin = VMA_MIN(outInfo.allocationSizeMin, suballoc.size);
            outInfo.allocationSizeMax = VMA_MIN(outInfo.allocationSizeMax, suballoc.size);
            
            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // There is free space from lastOffset to freeSpace1stTo2ndEnd.
            if(lastOffset < freeSpace1stTo2ndEnd)
            {
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                ++outInfo.unusedRangeCount;
                outInfo.unusedBytes += unusedRangeSize;
                outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, unusedRangeSize);
                outInfo.unusedRangeSizeMax = VMA_MIN(outInfo.unusedRangeSizeMax, unusedRangeSize);
           }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while(lastOffset < size)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while(nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    ++outInfo.unusedRangeCount;
                    outInfo.unusedBytes += unusedRangeSize;
                    outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, unusedRangeSize);
                    outInfo.unusedRangeSizeMax = VMA_MIN(outInfo.unusedRangeSizeMax, unusedRangeSize);
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                outInfo.usedBytes += suballoc.size;
                outInfo.allocationSizeMin = VMA_MIN(outInfo.allocationSizeMin, suballoc.size);
                outInfo.allocationSizeMax = VMA_MIN(outInfo.allocationSizeMax, suballoc.size);
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to size.
                if(lastOffset < size)
                {
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    ++outInfo.unusedRangeCount;
                    outInfo.unusedBytes += unusedRangeSize;
                    outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, unusedRangeSize);
                    outInfo.unusedRangeSizeMax = VMA_MIN(outInfo.unusedRangeSizeMax, unusedRangeSize);
               }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    outInfo.unusedBytes = size - outInfo.usedBytes;
}

void VmaBlockMetadata_Linear::AddPoolStats(VmaPoolStats& inoutStats) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const VkDeviceSize size = GetSize();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    inoutStats.size += size;

    VkDeviceSize lastOffset = 0;

    if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = m_1stNullItemsBeginCount;
        while(lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while(nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    inoutStats.unusedSize += unusedRangeSize;
                    ++inoutStats.unusedRangeCount;
                    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, unusedRangeSize);
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if(lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    inoutStats.unusedSize += unusedRangeSize;
                    ++inoutStats.unusedRangeCount;
                    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while(lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while(nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].hAllocation == VK_NULL_HANDLE)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if(nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];
            
            // 1. Process free space before this allocation.
            if(lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                inoutStats.unusedSize += unusedRangeSize;
                ++inoutStats.unusedRangeCount;
                inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, unusedRangeSize);
            }
            
            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++inoutStats.allocationCount;
            
            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if(lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                inoutStats.unusedSize += unusedRangeSize;
                ++inoutStats.unusedRangeCount;
                inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while(lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while(nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    inoutStats.unusedSize += unusedRangeSize;
                    ++inoutStats.unusedRangeCount;
                    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, unusedRangeSize);
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if(lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    inoutStats.unusedSize += unusedRangeSize;
                    ++inoutStats.unusedRangeCount;
                    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Linear::PrintDetailedMap(class VmaJsonWriter& json) const
{
    const VkDeviceSize size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    // FIRST PASS

    size_t unusedRangeCount = 0;
    VkDeviceSize usedBytes = 0;

    VkDeviceSize lastOffset = 0;

    size_t alloc2ndCount = 0;
    if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while(lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while(nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if(lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    size_t alloc1stCount = 0;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while(lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while(nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].hAllocation == VK_NULL_HANDLE)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if(nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];
            
            // 1. Process free space before this allocation.
            if(lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                ++unusedRangeCount;
            }
            
            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++alloc1stCount;
            usedBytes += suballoc.size;
            
            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if(lastOffset < size)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                ++unusedRangeCount;
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while(lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while(nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if(lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    const VkDeviceSize unusedBytes = size - usedBytes;
    PrintDetailedMap_Begin(json, unusedBytes, alloc1stCount + alloc2ndCount, unusedRangeCount);

    // SECOND PASS
    lastOffset = 0;

    if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while(lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while(nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.hAllocation);
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if(lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    nextAlloc1stIndex = m_1stNullItemsBeginCount;
    while(lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while(nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].hAllocation == VK_NULL_HANDLE)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if(nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];
            
            // 1. Process free space before this allocation.
            if(lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }
            
            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.hAllocation);
            
            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if(lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while(lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while(nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].hAllocation == VK_NULL_HANDLE)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if(nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];
            
                // 1. Process free space before this allocation.
                if(lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }
            
                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.hAllocation);
            
                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if(lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    PrintDetailedMap_End(json);
}
#endif // #if VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Linear::CreateAllocationRequest(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VkDeviceSize bufferImageGranularity,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    bool canMakeOtherLost,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(allocSize > 0);
    VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(pAllocationRequest != VMA_NULL);
    VMA_HEAVY_ASSERT(Validate());
    return upperAddress ?
        CreateAllocationRequest_UpperAddress(
            currentFrameIndex, frameInUseCount, bufferImageGranularity,
            allocSize, allocAlignment, allocType, canMakeOtherLost, strategy, pAllocationRequest) :
        CreateAllocationRequest_LowerAddress(
            currentFrameIndex, frameInUseCount, bufferImageGranularity,
            allocSize, allocAlignment, allocType, canMakeOtherLost, strategy, pAllocationRequest);
}

bool VmaBlockMetadata_Linear::CreateAllocationRequest_UpperAddress(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VkDeviceSize bufferImageGranularity,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    bool canMakeOtherLost,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize size = GetSize();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        VMA_ASSERT(0 && "Trying to use pool with linear algorithm as double stack, while it is already being used as ring buffer.");
        return false;
    }

    // Try to allocate before 2nd.back(), or end of block if 2nd.empty().
    if(allocSize > size)
    {
        return false;
    }
    VkDeviceSize resultBaseOffset = size - allocSize;
    if(!suballocations2nd.empty())
    {
        const VmaSuballocation& lastSuballoc = suballocations2nd.back();
        resultBaseOffset = lastSuballoc.offset - allocSize;
        if(allocSize > lastSuballoc.offset)
        {
            return false;
        }
    }

    // Start from offset equal to end of free space.
    VkDeviceSize resultOffset = resultBaseOffset;

    // Apply VMA_DEBUG_MARGIN at the end.
    if(VMA_DEBUG_MARGIN > 0)
    {
        if(resultOffset < VMA_DEBUG_MARGIN)
        {
            return false;
        }
        resultOffset -= VMA_DEBUG_MARGIN;
    }

    // Apply alignment.
    resultOffset = VmaAlignDown(resultOffset, allocAlignment);

    // Check next suballocations from 2nd for BufferImageGranularity conflicts.
    // Make bigger alignment if necessary.
    if(bufferImageGranularity > 1 && !suballocations2nd.empty())
    {
        bool bufferImageGranularityConflict = false;
        for(size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
        {
            const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
            if(VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
            {
                if(VmaIsBufferImageGranularityConflict(nextSuballoc.type, allocType))
                {
                    bufferImageGranularityConflict = true;
                    break;
                }
            }
            else
                // Already on previous page.
                break;
        }
        if(bufferImageGranularityConflict)
        {
            resultOffset = VmaAlignDown(resultOffset, bufferImageGranularity);
        }
    }

    // There is enough free space.
    const VkDeviceSize endOf1st = !suballocations1st.empty() ?
        suballocations1st.back().offset + suballocations1st.back().size :
        0;
    if(endOf1st + VMA_DEBUG_MARGIN <= resultOffset)
    {
        // Check previous suballocations for BufferImageGranularity conflicts.
        // If conflict exists, allocation cannot be made here.
        if(bufferImageGranularity > 1)
        {
            for(size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if(VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if(VmaIsBufferImageGranularityConflict(allocType, prevSuballoc.type))
                    {
                        return false;
                    }
                }
                else
                {
                    // Already on next page.
                    break;
                }
            }
        }

        // All tests passed: Success.
        pAllocationRequest->offset = resultOffset;
        pAllocationRequest->sumFreeSize = resultBaseOffset + allocSize - endOf1st;
        pAllocationRequest->sumItemSize = 0;
        // pAllocationRequest->item unused.
        pAllocationRequest->itemsToMakeLostCount = 0;
        pAllocationRequest->type = VmaAllocationRequestType::UpperAddress;
        return true;
    }

    return false;
}

bool VmaBlockMetadata_Linear::CreateAllocationRequest_LowerAddress(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VkDeviceSize bufferImageGranularity,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    bool canMakeOtherLost,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize size = GetSize();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if(m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        // Try to allocate at the end of 1st vector.

        VkDeviceSize resultBaseOffset = 0;
        if(!suballocations1st.empty())
        {
            const VmaSuballocation& lastSuballoc = suballocations1st.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply VMA_DEBUG_MARGIN at the beginning.
        if(VMA_DEBUG_MARGIN > 0)
        {
            resultOffset += VMA_DEBUG_MARGIN;
        }

        // Apply alignment.
        resultOffset = VmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if(bufferImageGranularity > 1 && !suballocations1st.empty())
        {
            bool bufferImageGranularityConflict = false;
            for(size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if(VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if(VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if(bufferImageGranularityConflict)
            {
                resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        const VkDeviceSize freeSpaceEnd = m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ?
            suballocations2nd.back().offset : size;

        // There is enough free space at the end after alignment.
        if(resultOffset + allocSize + VMA_DEBUG_MARGIN <= freeSpaceEnd)
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if(bufferImageGranularity > 1 && m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
            {
                for(size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
                {
                    const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
                    if(VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if(VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on previous page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->offset = resultOffset;
            pAllocationRequest->sumFreeSize = freeSpaceEnd - resultBaseOffset;
            pAllocationRequest->sumItemSize = 0;
            // pAllocationRequest->item, customData unused.
            pAllocationRequest->type = VmaAllocationRequestType::EndOf1st;
            pAllocationRequest->itemsToMakeLostCount = 0;
            return true;
        }
    }

    // Wrap-around to end of 2nd vector. Try to allocate there, watching for the
    // beginning of 1st vector as the end of free space.
    if(m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        VMA_ASSERT(!suballocations1st.empty());

        VkDeviceSize resultBaseOffset = 0;
        if(!suballocations2nd.empty())
        {
            const VmaSuballocation& lastSuballoc = suballocations2nd.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply VMA_DEBUG_MARGIN at the beginning.
        if(VMA_DEBUG_MARGIN > 0)
        {
            resultOffset += VMA_DEBUG_MARGIN;
        }

        // Apply alignment.
        resultOffset = VmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if(bufferImageGranularity > 1 && !suballocations2nd.empty())
        {
            bool bufferImageGranularityConflict = false;
            for(size_t prevSuballocIndex = suballocations2nd.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations2nd[prevSuballocIndex];
                if(VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if(VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if(bufferImageGranularityConflict)
            {
                resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        pAllocationRequest->itemsToMakeLostCount = 0;
        pAllocationRequest->sumItemSize = 0;
        size_t index1st = m_1stNullItemsBeginCount;

        if(canMakeOtherLost)
        {
            while(index1st < suballocations1st.size() &&
                resultOffset + allocSize + VMA_DEBUG_MARGIN > suballocations1st[index1st].offset)
            {
                // Next colliding allocation at the beginning of 1st vector found. Try to make it lost.
                const VmaSuballocation& suballoc = suballocations1st[index1st];
                if(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE)
                {
                    // No problem.
                }
                else
                {
                    VMA_ASSERT(suballoc.hAllocation != VK_NULL_HANDLE);
                    if(suballoc.hAllocation->CanBecomeLost() &&
                        suballoc.hAllocation->GetLastUseFrameIndex() + frameInUseCount < currentFrameIndex)
                    {
                        ++pAllocationRequest->itemsToMakeLostCount;
                        pAllocationRequest->sumItemSize += suballoc.size;
                    }
                    else
                    {
                        return false;
                    }
                }
                ++index1st;
            }

            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, we must mark more allocations lost or fail.
            if(bufferImageGranularity > 1)
            {
                while(index1st < suballocations1st.size())
                {
                    const VmaSuballocation& suballoc = suballocations1st[index1st];
                    if(VmaBlocksOnSamePage(resultOffset, allocSize, suballoc.offset, bufferImageGranularity))
                    {
                        if(suballoc.hAllocation != VK_NULL_HANDLE)
                        {
                            // Not checking actual VmaIsBufferImageGranularityConflict(allocType, suballoc.type).
                            if(suballoc.hAllocation->CanBecomeLost() &&
                                suballoc.hAllocation->GetLastUseFrameIndex() + frameInUseCount < currentFrameIndex)
                            {
                                ++pAllocationRequest->itemsToMakeLostCount;
                                pAllocationRequest->sumItemSize += suballoc.size;
                            }
                            else
                            {
                                return false;
                            }
                        }
                    }
                    else
                    {
                        // Already on next page.
                        break;
                    }
                    ++index1st;
                }
            }

            // Special case: There is not enough room at the end for this allocation, even after making all from the 1st lost.
            if(index1st == suballocations1st.size() &&
                resultOffset + allocSize + VMA_DEBUG_MARGIN > size)
            {
                // TODO: This is a known bug that it's not yet implemented and the allocation is failing.
                VMA_DEBUG_LOG("Unsupported special case in custom pool with linear allocation algorithm used as ring buffer with allocations that can be lost.");
            }
        }

        // There is enough free space at the end after alignment.
        if((index1st == suballocations1st.size() && resultOffset + allocSize + VMA_DEBUG_MARGIN <= size) ||
            (index1st < suballocations1st.size() && resultOffset + allocSize + VMA_DEBUG_MARGIN <= suballocations1st[index1st].offset))
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if(bufferImageGranularity > 1)
            {
                for(size_t nextSuballocIndex = index1st;
                    nextSuballocIndex < suballocations1st.size();
                    nextSuballocIndex++)
                {
                    const VmaSuballocation& nextSuballoc = suballocations1st[nextSuballocIndex];
                    if(VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if(VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on next page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->offset = resultOffset;
            pAllocationRequest->sumFreeSize =
                (index1st < suballocations1st.size() ? suballocations1st[index1st].offset : size)
                - resultBaseOffset
                - pAllocationRequest->sumItemSize;
            pAllocationRequest->type = VmaAllocationRequestType::EndOf2nd;
            // pAllocationRequest->item, customData unused.
            return true;
        }
    }

    return false;
}

bool VmaBlockMetadata_Linear::MakeRequestedAllocationsLost(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VmaAllocationRequest* pAllocationRequest)
{
    if(pAllocationRequest->itemsToMakeLostCount == 0)
    {
        return true;
    }

    VMA_ASSERT(m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER);
    
    // We always start from 1st.
    SuballocationVectorType* suballocations = &AccessSuballocations1st();
    size_t index = m_1stNullItemsBeginCount;
    size_t madeLostCount = 0;
    while(madeLostCount < pAllocationRequest->itemsToMakeLostCount)
    {
        if(index == suballocations->size())
        {
            index = 0;
            // If we get to the end of 1st, we wrap around to beginning of 2nd of 1st.
            if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
            {
                suballocations = &AccessSuballocations2nd();
            }
            // else: m_2ndVectorMode == SECOND_VECTOR_EMPTY:
            // suballocations continues pointing at AccessSuballocations1st().
            VMA_ASSERT(!suballocations->empty());
        }
        VmaSuballocation& suballoc = (*suballocations)[index];
        if(suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            VMA_ASSERT(suballoc.hAllocation != VK_NULL_HANDLE);
            VMA_ASSERT(suballoc.hAllocation->CanBecomeLost());
            if(suballoc.hAllocation->MakeLost(currentFrameIndex, frameInUseCount))
            {
                suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
                suballoc.hAllocation = VK_NULL_HANDLE;
                m_SumFreeSize += suballoc.size;
                if(suballocations == &AccessSuballocations1st())
                {
                    ++m_1stNullItemsMiddleCount;
                }
                else
                {
                    ++m_2ndNullItemsCount;
                }
                ++madeLostCount;
            }
            else
            {
                return false;
            }
        }
        ++index;
    }

    CleanupAfterFree();
    //VMA_HEAVY_ASSERT(Validate()); // Already called by ClanupAfterFree().
    
    return true;
}

uint32_t VmaBlockMetadata_Linear::MakeAllocationsLost(uint32_t currentFrameIndex, uint32_t frameInUseCount)
{
    uint32_t lostAllocationCount = 0;
    
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for(size_t i = m_1stNullItemsBeginCount, count = suballocations1st.size(); i < count; ++i)
    {
        VmaSuballocation& suballoc = suballocations1st[i];
        if(suballoc.type != VMA_SUBALLOCATION_TYPE_FREE &&
            suballoc.hAllocation->CanBecomeLost() &&
            suballoc.hAllocation->MakeLost(currentFrameIndex, frameInUseCount))
        {
            suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
            suballoc.hAllocation = VK_NULL_HANDLE;
            ++m_1stNullItemsMiddleCount;
            m_SumFreeSize += suballoc.size;
            ++lostAllocationCount;
        }
    }

    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for(size_t i = 0, count = suballocations2nd.size(); i < count; ++i)
    {
        VmaSuballocation& suballoc = suballocations2nd[i];
        if(suballoc.type != VMA_SUBALLOCATION_TYPE_FREE &&
            suballoc.hAllocation->CanBecomeLost() &&
            suballoc.hAllocation->MakeLost(currentFrameIndex, frameInUseCount))
        {
            suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
            suballoc.hAllocation = VK_NULL_HANDLE;
            ++m_2ndNullItemsCount;
            m_SumFreeSize += suballoc.size;
            ++lostAllocationCount;
        }
    }

    if(lostAllocationCount)
    {
        CleanupAfterFree();
    }

    return lostAllocationCount;
}

VkResult VmaBlockMetadata_Linear::CheckCorruption(const void* pBlockData)
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for(size_t i = m_1stNullItemsBeginCount, count = suballocations1st.size(); i < count; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        if(suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if(!VmaValidateMagicValue(pBlockData, suballoc.offset - VMA_DEBUG_MARGIN))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED BEFORE VALIDATED ALLOCATION!");
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
            if(!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
        }
    }

    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for(size_t i = 0, count = suballocations2nd.size(); i < count; ++i)
    {
        const VmaSuballocation& suballoc = suballocations2nd[i];
        if(suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if(!VmaValidateMagicValue(pBlockData, suballoc.offset - VMA_DEBUG_MARGIN))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED BEFORE VALIDATED ALLOCATION!");
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
            if(!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
        }
    }

    return VK_SUCCESS;
}

void VmaBlockMetadata_Linear::Alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    VkDeviceSize allocSize,
    VmaAllocation hAllocation)
{
    const VmaSuballocation newSuballoc = { request.offset, allocSize, hAllocation, type };

    switch(request.type)
    {
    case VmaAllocationRequestType::UpperAddress:
        {
            VMA_ASSERT(m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER &&
                "CRITICAL ERROR: Trying to use linear allocator as double stack while it was already used as ring buffer.");
            SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
            suballocations2nd.push_back(newSuballoc);
            m_2ndVectorMode = SECOND_VECTOR_DOUBLE_STACK;
        }
        break;
    case VmaAllocationRequestType::EndOf1st:
        {
            SuballocationVectorType& suballocations1st = AccessSuballocations1st();

            VMA_ASSERT(suballocations1st.empty() ||
                request.offset >= suballocations1st.back().offset + suballocations1st.back().size);
            // Check if it fits before the end of the block.
            VMA_ASSERT(request.offset + allocSize <= GetSize());

            suballocations1st.push_back(newSuballoc);
        }
        break;
    case VmaAllocationRequestType::EndOf2nd:
        {
            SuballocationVectorType& suballocations1st = AccessSuballocations1st();
            // New allocation at the end of 2-part ring buffer, so before first allocation from 1st vector.
            VMA_ASSERT(!suballocations1st.empty() &&
                request.offset + allocSize <= suballocations1st[m_1stNullItemsBeginCount].offset);
            SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

            switch(m_2ndVectorMode)
            {
            case SECOND_VECTOR_EMPTY:
                // First allocation from second part ring buffer.
                VMA_ASSERT(suballocations2nd.empty());
                m_2ndVectorMode = SECOND_VECTOR_RING_BUFFER;
                break;
            case SECOND_VECTOR_RING_BUFFER:
                // 2-part ring buffer is already started.
                VMA_ASSERT(!suballocations2nd.empty());
                break;
            case SECOND_VECTOR_DOUBLE_STACK:
                VMA_ASSERT(0 && "CRITICAL ERROR: Trying to use linear allocator as ring buffer while it was already used as double stack.");
                break;
            default:
                VMA_ASSERT(0);
            }

            suballocations2nd.push_back(newSuballoc);
        }
        break;
    default:
        VMA_ASSERT(0 && "CRITICAL INTERNAL ERROR.");
    }

    m_SumFreeSize -= newSuballoc.size;
}

void VmaBlockMetadata_Linear::Free(const VmaAllocation allocation)
{
    FreeAtOffset(allocation->GetOffset());
}

void VmaBlockMetadata_Linear::FreeAtOffset(VkDeviceSize offset)
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if(!suballocations1st.empty())
    {
        // First allocation: Mark it as next empty at the beginning.
        VmaSuballocation& firstSuballoc = suballocations1st[m_1stNullItemsBeginCount];
        if(firstSuballoc.offset == offset)
        {
            firstSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
            firstSuballoc.hAllocation = VK_NULL_HANDLE;
            m_SumFreeSize += firstSuballoc.size;
            ++m_1stNullItemsBeginCount;
            CleanupAfterFree();
            return;
        }
    }

    // Last allocation in 2-part ring buffer or top of upper stack (same logic).
    if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ||
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        VmaSuballocation& lastSuballoc = suballocations2nd.back();
        if(lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations2nd.pop_back();
            CleanupAfterFree();
            return;
        }
    }
    // Last allocation in 1st vector.
    else if(m_2ndVectorMode == SECOND_VECTOR_EMPTY)
    {
        VmaSuballocation& lastSuballoc = suballocations1st.back();
        if(lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations1st.pop_back();
            CleanupAfterFree();
            return;
        }
    }

    // Item from the middle of 1st vector.
    {
        VmaSuballocation refSuballoc;
        refSuballoc.offset = offset;
        // Rest of members stays uninitialized intentionally for better performance.
        SuballocationVectorType::iterator it = VmaBinaryFindSorted(
            suballocations1st.begin() + m_1stNullItemsBeginCount,
            suballocations1st.end(),
            refSuballoc,
            VmaSuballocationOffsetLess());
        if(it != suballocations1st.end())
        {
            it->type = VMA_SUBALLOCATION_TYPE_FREE;
            it->hAllocation = VK_NULL_HANDLE;
            ++m_1stNullItemsMiddleCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    if(m_2ndVectorMode != SECOND_VECTOR_EMPTY)
    {
        // Item from the middle of 2nd vector.
        VmaSuballocation refSuballoc;
        refSuballoc.offset = offset;
        // Rest of members stays uninitialized intentionally for better performance.
        SuballocationVectorType::iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetLess()) :
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetGreater());
        if(it != suballocations2nd.end())
        {
            it->type = VMA_SUBALLOCATION_TYPE_FREE;
            it->hAllocation = VK_NULL_HANDLE;
            ++m_2ndNullItemsCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    VMA_ASSERT(0 && "Allocation to free not found in linear allocator!");
}

bool VmaBlockMetadata_Linear::ShouldCompact1st() const
{
    const size_t nullItemCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
    const size_t suballocCount = AccessSuballocations1st().size();
    return suballocCount > 32 && nullItemCount * 2 >= (suballocCount - nullItemCount) * 3;
}

void VmaBlockMetadata_Linear::CleanupAfterFree()
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if(IsEmpty())
    {
        suballocations1st.clear();
        suballocations2nd.clear();
        m_1stNullItemsBeginCount = 0;
        m_1stNullItemsMiddleCount = 0;
        m_2ndNullItemsCount = 0;
        m_2ndVectorMode = SECOND_VECTOR_EMPTY;
    }
    else
    {
        const size_t suballoc1stCount = suballocations1st.size();
        const size_t nullItem1stCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
        VMA_ASSERT(nullItem1stCount <= suballoc1stCount);

        // Find more null items at the beginning of 1st vector.
        while(m_1stNullItemsBeginCount < suballoc1stCount &&
            suballocations1st[m_1stNullItemsBeginCount].hAllocation == VK_NULL_HANDLE)
        {
            ++m_1stNullItemsBeginCount;
            --m_1stNullItemsMiddleCount;
        }

        // Find more null items at the end of 1st vector.
        while(m_1stNullItemsMiddleCount > 0 &&
            suballocations1st.back().hAllocation == VK_NULL_HANDLE)
        {
            --m_1stNullItemsMiddleCount;
            suballocations1st.pop_back();
        }

        // Find more null items at the end of 2nd vector.
        while(m_2ndNullItemsCount > 0 &&
            suballocations2nd.back().hAllocation == VK_NULL_HANDLE)
        {
            --m_2ndNullItemsCount;
            suballocations2nd.pop_back();
        }

        // Find more null items at the beginning of 2nd vector.
        while(m_2ndNullItemsCount > 0 &&
            suballocations2nd[0].hAllocation == VK_NULL_HANDLE)
        {
            --m_2ndNullItemsCount;
            VmaVectorRemove(suballocations2nd, 0);
        }

        if(ShouldCompact1st())
        {
            const size_t nonNullItemCount = suballoc1stCount - nullItem1stCount;
            size_t srcIndex = m_1stNullItemsBeginCount;
            for(size_t dstIndex = 0; dstIndex < nonNullItemCount; ++dstIndex)
            {
                while(suballocations1st[srcIndex].hAllocation == VK_NULL_HANDLE)
                {
                    ++srcIndex;
                }
                if(dstIndex != srcIndex)
                {
                    suballocations1st[dstIndex] = suballocations1st[srcIndex];
                }
                ++srcIndex;
            }
            suballocations1st.resize(nonNullItemCount);
            m_1stNullItemsBeginCount = 0;
            m_1stNullItemsMiddleCount = 0;
        }

        // 2nd vector became empty.
        if(suballocations2nd.empty())
        {
            m_2ndVectorMode = SECOND_VECTOR_EMPTY;
        }

        // 1st vector became empty.
        if(suballocations1st.size() - m_1stNullItemsBeginCount == 0)
        {
            suballocations1st.clear();
            m_1stNullItemsBeginCount = 0;

            if(!suballocations2nd.empty() && m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
            {
                // Swap 1st with 2nd. Now 2nd is empty.
                m_2ndVectorMode = SECOND_VECTOR_EMPTY;
                m_1stNullItemsMiddleCount = m_2ndNullItemsCount;
                while(m_1stNullItemsBeginCount < suballocations2nd.size() &&
                    suballocations2nd[m_1stNullItemsBeginCount].hAllocation == VK_NULL_HANDLE)
                {
                    ++m_1stNullItemsBeginCount;
                    --m_1stNullItemsMiddleCount;
                }
                m_2ndNullItemsCount = 0;
                m_1stVectorIndex ^= 1;
            }
        }
    }

    VMA_HEAVY_ASSERT(Validate());
}


////////////////////////////////////////////////////////////////////////////////
// class VmaBlockMetadata_Buddy

VmaBlockMetadata_Buddy::VmaBlockMetadata_Buddy(VmaAllocator hAllocator) :
    VmaBlockMetadata(hAllocator),
    m_Root(VMA_NULL),
    m_AllocationCount(0),
    m_FreeCount(1),
    m_SumFreeSize(0)
{
    memset(m_FreeList, 0, sizeof(m_FreeList));
}

VmaBlockMetadata_Buddy::~VmaBlockMetadata_Buddy()
{
    DeleteNode(m_Root);
}

void VmaBlockMetadata_Buddy::Init(VkDeviceSize size)
{
    VmaBlockMetadata::Init(size);

    m_UsableSize = VmaPrevPow2(size);
    m_SumFreeSize = m_UsableSize;

    // Calculate m_LevelCount.
    m_LevelCount = 1;
    while(m_LevelCount < MAX_LEVELS &&
        LevelToNodeSize(m_LevelCount) >= MIN_NODE_SIZE)
    {
        ++m_LevelCount;
    }

    Node* rootNode = vma_new(GetAllocationCallbacks(), Node)();
    rootNode->offset = 0;
    rootNode->type = Node::TYPE_FREE;
    rootNode->parent = VMA_NULL;
    rootNode->buddy = VMA_NULL;

    m_Root = rootNode;
    AddToFreeListFront(0, rootNode);
}

bool VmaBlockMetadata_Buddy::Validate() const
{
    // Validate tree.
    ValidationContext ctx;
    if(!ValidateNode(ctx, VMA_NULL, m_Root, 0, LevelToNodeSize(0)))
    {
        VMA_VALIDATE(false && "ValidateNode failed.");
    }
    VMA_VALIDATE(m_AllocationCount == ctx.calculatedAllocationCount);
    VMA_VALIDATE(m_SumFreeSize == ctx.calculatedSumFreeSize);

    // Validate free node lists.
    for(uint32_t level = 0; level < m_LevelCount; ++level)
    {
        VMA_VALIDATE(m_FreeList[level].front == VMA_NULL ||
            m_FreeList[level].front->free.prev == VMA_NULL);

        for(Node* node = m_FreeList[level].front;
            node != VMA_NULL;
            node = node->free.next)
        {
            VMA_VALIDATE(node->type == Node::TYPE_FREE);
            
            if(node->free.next == VMA_NULL)
            {
                VMA_VALIDATE(m_FreeList[level].back == node);
            }
            else
            {
                VMA_VALIDATE(node->free.next->free.prev == node);
            }
        }
    }

    // Validate that free lists ar higher levels are empty.
    for(uint32_t level = m_LevelCount; level < MAX_LEVELS; ++level)
    {
        VMA_VALIDATE(m_FreeList[level].front == VMA_NULL && m_FreeList[level].back == VMA_NULL);
    }

    return true;
}

VkDeviceSize VmaBlockMetadata_Buddy::GetUnusedRangeSizeMax() const
{
    for(uint32_t level = 0; level < m_LevelCount; ++level)
    {
        if(m_FreeList[level].front != VMA_NULL)
        {
            return LevelToNodeSize(level);
        }
    }
    return 0;
}

void VmaBlockMetadata_Buddy::CalcAllocationStatInfo(VmaStatInfo& outInfo) const
{
    const VkDeviceSize unusableSize = GetUnusableSize();

    outInfo.blockCount = 1;

    outInfo.allocationCount = outInfo.unusedRangeCount = 0;
    outInfo.usedBytes = outInfo.unusedBytes = 0;

    outInfo.allocationSizeMax = outInfo.unusedRangeSizeMax = 0;
    outInfo.allocationSizeMin = outInfo.unusedRangeSizeMin = UINT64_MAX;
    outInfo.allocationSizeAvg = outInfo.unusedRangeSizeAvg = 0; // Unused.

    CalcAllocationStatInfoNode(outInfo, m_Root, LevelToNodeSize(0));

    if(unusableSize > 0)
    {
        ++outInfo.unusedRangeCount;
        outInfo.unusedBytes += unusableSize;
        outInfo.unusedRangeSizeMax = VMA_MAX(outInfo.unusedRangeSizeMax, unusableSize);
        outInfo.unusedRangeSizeMin = VMA_MIN(outInfo.unusedRangeSizeMin, unusableSize);
    }
}

void VmaBlockMetadata_Buddy::AddPoolStats(VmaPoolStats& inoutStats) const
{
    const VkDeviceSize unusableSize = GetUnusableSize();

    inoutStats.size += GetSize();
    inoutStats.unusedSize += m_SumFreeSize + unusableSize;
    inoutStats.allocationCount += m_AllocationCount;
    inoutStats.unusedRangeCount += m_FreeCount;
    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, GetUnusedRangeSizeMax());

    if(unusableSize > 0)
    {
        ++inoutStats.unusedRangeCount;
        // Not updating inoutStats.unusedRangeSizeMax with unusableSize because this space is not available for allocations.
    }
}

#if VMA_STATS_STRING_ENABLED

void VmaBlockMetadata_Buddy::PrintDetailedMap(class VmaJsonWriter& json) const
{
    // TODO optimize
    VmaStatInfo stat;
    CalcAllocationStatInfo(stat);

    PrintDetailedMap_Begin(
        json,
        stat.unusedBytes,
        stat.allocationCount,
        stat.unusedRangeCount);

    PrintDetailedMapNode(json, m_Root, LevelToNodeSize(0));

    const VkDeviceSize unusableSize = GetUnusableSize();
    if(unusableSize > 0)
    {
        PrintDetailedMap_UnusedRange(json,
            m_UsableSize, // offset
            unusableSize); // size
    }

    PrintDetailedMap_End(json);
}

#endif // #if VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Buddy::CreateAllocationRequest(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VkDeviceSize bufferImageGranularity,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    bool canMakeOtherLost,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(!upperAddress && "VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT can be used only with linear algorithm.");

    // Simple way to respect bufferImageGranularity. May be optimized some day.
    // Whenever it might be an OPTIMAL image...
    if(allocType == VMA_SUBALLOCATION_TYPE_UNKNOWN ||
        allocType == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
        allocType == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL)
    {
        allocAlignment = VMA_MAX(allocAlignment, bufferImageGranularity);
        allocSize = VMA_MAX(allocSize, bufferImageGranularity);
    }

    if(allocSize > m_UsableSize)
    {
        return false;
    }

    const uint32_t targetLevel = AllocSizeToLevel(allocSize);
    for(uint32_t level = targetLevel + 1; level--; )
    {
        for(Node* freeNode = m_FreeList[level].front;
            freeNode != VMA_NULL;
            freeNode = freeNode->free.next)
        {
            if(freeNode->offset % allocAlignment == 0)
            {
                pAllocationRequest->type = VmaAllocationRequestType::Normal;
                pAllocationRequest->offset = freeNode->offset;
                pAllocationRequest->sumFreeSize = LevelToNodeSize(level);
                pAllocationRequest->sumItemSize = 0;
                pAllocationRequest->itemsToMakeLostCount = 0;
                pAllocationRequest->customData = (void*)(uintptr_t)level;
                return true;
            }
        }
    }

    return false;
}

bool VmaBlockMetadata_Buddy::MakeRequestedAllocationsLost(
    uint32_t currentFrameIndex,
    uint32_t frameInUseCount,
    VmaAllocationRequest* pAllocationRequest)
{
    /*
    Lost allocations are not supported in buddy allocator at the moment.
    Support might be added in the future.
    */
    return pAllocationRequest->itemsToMakeLostCount == 0;
}

uint32_t VmaBlockMetadata_Buddy::MakeAllocationsLost(uint32_t currentFrameIndex, uint32_t frameInUseCount)
{
    /*
    Lost allocations are not supported in buddy allocator at the moment.
    Support might be added in the future.
    */
    return 0;
}

void VmaBlockMetadata_Buddy::Alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    VkDeviceSize allocSize,
    VmaAllocation hAllocation)
{
    VMA_ASSERT(request.type == VmaAllocationRequestType::Normal);

    const uint32_t targetLevel = AllocSizeToLevel(allocSize);
    uint32_t currLevel = (uint32_t)(uintptr_t)request.customData;
    
    Node* currNode = m_FreeList[currLevel].front;
    VMA_ASSERT(currNode != VMA_NULL && currNode->type == Node::TYPE_FREE);
    while(currNode->offset != request.offset)
    {
        currNode = currNode->free.next;
        VMA_ASSERT(currNode != VMA_NULL && currNode->type == Node::TYPE_FREE);
    }
    
    // Go down, splitting free nodes.
    while(currLevel < targetLevel)
    {
        // currNode is already first free node at currLevel.
        // Remove it from list of free nodes at this currLevel.
        RemoveFromFreeList(currLevel, currNode);
         
        const uint32_t childrenLevel = currLevel + 1;

        // Create two free sub-nodes.
        Node* leftChild = vma_new(GetAllocationCallbacks(), Node)();
        Node* rightChild = vma_new(GetAllocationCallbacks(), Node)();

        leftChild->offset = currNode->offset;
        leftChild->type = Node::TYPE_FREE;
        leftChild->parent = currNode;
        leftChild->buddy = rightChild;

        rightChild->offset = currNode->offset + LevelToNodeSize(childrenLevel);
        rightChild->type = Node::TYPE_FREE;
        rightChild->parent = currNode;
        rightChild->buddy = leftChild;

        // Convert current currNode to split type.
        currNode->type = Node::TYPE_SPLIT;
        currNode->split.leftChild = leftChild;

        // Add child nodes to free list. Order is important!
        AddToFreeListFront(childrenLevel, rightChild);
        AddToFreeListFront(childrenLevel, leftChild);

        ++m_FreeCount;
        //m_SumFreeSize -= LevelToNodeSize(currLevel) % 2; // Useful only when level node sizes can be non power of 2.
        ++currLevel;
        currNode = m_FreeList[currLevel].front;

        /*
        We can be sure that currNode, as left child of node previously split,
        also fullfills the alignment requirement.
        */
    }

    // Remove from free list.
    VMA_ASSERT(currLevel == targetLevel &&
        currNode != VMA_NULL &&
        currNode->type == Node::TYPE_FREE);
    RemoveFromFreeList(currLevel, currNode);

    // Convert to allocation node.
    currNode->type = Node::TYPE_ALLOCATION;
    currNode->allocation.alloc = hAllocation;

    ++m_AllocationCount;
    --m_FreeCount;
    m_SumFreeSize -= allocSize;
}

void VmaBlockMetadata_Buddy::DeleteNode(Node* node)
{
    if(node->type == Node::TYPE_SPLIT)
    {
        DeleteNode(node->split.leftChild->buddy);
        DeleteNode(node->split.leftChild);
    }

    vma_delete(GetAllocationCallbacks(), node);
}

bool VmaBlockMetadata_Buddy::ValidateNode(ValidationContext& ctx, const Node* parent, const Node* curr, uint32_t level, VkDeviceSize levelNodeSize) const
{
    VMA_VALIDATE(level < m_LevelCount);
    VMA_VALIDATE(curr->parent == parent);
    VMA_VALIDATE((curr->buddy == VMA_NULL) == (parent == VMA_NULL));
    VMA_VALIDATE(curr->buddy == VMA_NULL || curr->buddy->buddy == curr);
    switch(curr->type)
    {
    case Node::TYPE_FREE:
        // curr->free.prev, next are validated separately.
        ctx.calculatedSumFreeSize += levelNodeSize;
        ++ctx.calculatedFreeCount;
        break;
    case Node::TYPE_ALLOCATION:
        ++ctx.calculatedAllocationCount;
        ctx.calculatedSumFreeSize += levelNodeSize - curr->allocation.alloc->GetSize();
        VMA_VALIDATE(curr->allocation.alloc != VK_NULL_HANDLE);
        break;
    case Node::TYPE_SPLIT:
        {
            const uint32_t childrenLevel = level + 1;
            const VkDeviceSize childrenLevelNodeSize = levelNodeSize / 2;
            const Node* const leftChild = curr->split.leftChild;
            VMA_VALIDATE(leftChild != VMA_NULL);
            VMA_VALIDATE(leftChild->offset == curr->offset);
            if(!ValidateNode(ctx, curr, leftChild, childrenLevel, childrenLevelNodeSize))
            {
                VMA_VALIDATE(false && "ValidateNode for left child failed.");
            }
            const Node* const rightChild = leftChild->buddy;
            VMA_VALIDATE(rightChild->offset == curr->offset + childrenLevelNodeSize);
            if(!ValidateNode(ctx, curr, rightChild, childrenLevel, childrenLevelNodeSize))
            {
                VMA_VALIDATE(false && "ValidateNode for right child failed.");
            }
        }
        break;
    default:
        return false;
    }

    return true;
}

uint32_t VmaBlockMetadata_Buddy::AllocSizeToLevel(VkDeviceSize allocSize) const
{
    // I know this could be optimized somehow e.g. by using std::log2p1 from C++20.
    uint32_t level = 0;
    VkDeviceSize currLevelNodeSize = m_UsableSize;
    VkDeviceSize nextLevelNodeSize = currLevelNodeSize >> 1;
    while(allocSize <= nextLevelNodeSize && level + 1 < m_LevelCount)
    {
        ++level;
        currLevelNodeSize = nextLevelNodeSize;
        nextLevelNodeSize = currLevelNodeSize >> 1;
    }
    return level;
}

void VmaBlockMetadata_Buddy::FreeAtOffset(VmaAllocation alloc, VkDeviceSize offset)
{
    // Find node and level.
    Node* node = m_Root;
    VkDeviceSize nodeOffset = 0;
    uint32_t level = 0;
    VkDeviceSize levelNodeSize = LevelToNodeSize(0);
    while(node->type == Node::TYPE_SPLIT)
    {
        const VkDeviceSize nextLevelSize = levelNodeSize >> 1;
        if(offset < nodeOffset + nextLevelSize)
        {
            node = node->split.leftChild;
        }
        else
        {
            node = node->split.leftChild->buddy;
            nodeOffset += nextLevelSize;
        }
        ++level;
        levelNodeSize = nextLevelSize;
    }

    VMA_ASSERT(node != VMA_NULL && node->type == Node::TYPE_ALLOCATION);
    VMA_ASSERT(alloc == VK_NULL_HANDLE || node->allocation.alloc == alloc);

    ++m_FreeCount;
    --m_AllocationCount;
    m_SumFreeSize += alloc->GetSize();

    node->type = Node::TYPE_FREE;

    // Join free nodes if possible.
    while(level > 0 && node->buddy->type == Node::TYPE_FREE)
    {
        RemoveFromFreeList(level, node->buddy);
        Node* const parent = node->parent;

        vma_delete(GetAllocationCallbacks(), node->buddy);
        vma_delete(GetAllocationCallbacks(), node);
        parent->type = Node::TYPE_FREE;
        
        node = parent;
        --level;
        //m_SumFreeSize += LevelToNodeSize(level) % 2; // Useful only when level node sizes can be non power of 2.
        --m_FreeCount;
    }

    AddToFreeListFront(level, node);
}

void VmaBlockMetadata_Buddy::CalcAllocationStatInfoNode(VmaStatInfo& outInfo, const Node* node, VkDeviceSize levelNodeSize) const
{
    switch(node->type)
    {
    case Node::TYPE_FREE:
        ++outInfo.unusedRangeCount;
        outInfo.unusedBytes += levelNodeSize;
        outInfo.unusedRangeSizeMax = VMA_MAX(outInfo.unusedRangeSizeMax, levelNodeSize);
        outInfo.unusedRangeSizeMin = VMA_MAX(outInfo.unusedRangeSizeMin, levelNodeSize);
        break;
    case Node::TYPE_ALLOCATION:
        {
            const VkDeviceSize allocSize = node->allocation.alloc->GetSize();
            ++outInfo.allocationCount;
            outInfo.usedBytes += allocSize;
            outInfo.allocationSizeMax = VMA_MAX(outInfo.allocationSizeMax, allocSize);
            outInfo.allocationSizeMin = VMA_MAX(outInfo.allocationSizeMin, allocSize);

            const VkDeviceSize unusedRangeSize = levelNodeSize - allocSize;
            if(unusedRangeSize > 0)
            {
                ++outInfo.unusedRangeCount;
                outInfo.unusedBytes += unusedRangeSize;
                outInfo.unusedRangeSizeMax = VMA_MAX(outInfo.unusedRangeSizeMax, unusedRangeSize);
                outInfo.unusedRangeSizeMin = VMA_MAX(outInfo.unusedRangeSizeMin, unusedRangeSize);
            }
        }
        break;
    case Node::TYPE_SPLIT:
        {
            const VkDeviceSize childrenNodeSize = levelNodeSize / 2;
            const Node* const leftChild = node->split.leftChild;
            CalcAllocationStatInfoNode(outInfo, leftChild, childrenNodeSize);
            const Node* const rightChild = leftChild->buddy;
            CalcAllocationStatInfoNode(outInfo, rightChild, childrenNodeSize);
        }
        break;
    default:
        VMA_ASSERT(0);
    }
}

void VmaBlockMetadata_Buddy::AddToFreeListFront(uint32_t level, Node* node)
{
    VMA_ASSERT(node->type == Node::TYPE_FREE);

    // List is empty.
    Node* const frontNode = m_FreeList[level].front;
    if(frontNode == VMA_NULL)
    {
        VMA_ASSERT(m_FreeList[level].back == VMA_NULL);
        node->free.prev = node->free.next = VMA_NULL;
        m_FreeList[level].front = m_FreeList[level].back = node;
    }
    else
    {
        VMA_ASSERT(frontNode->free.prev == VMA_NULL);
        node->free.prev = VMA_NULL;
        node->free.next = frontNode;
        frontNode->free.prev = node;
        m_FreeList[level].front = node;
    }
}

void VmaBlockMetadata_Buddy::RemoveFromFreeList(uint32_t level, Node* node)
{
    VMA_ASSERT(m_FreeList[level].front != VMA_NULL);

    // It is at the front.
    if(node->free.prev == VMA_NULL)
    {
        VMA_ASSERT(m_FreeList[level].front == node);
        m_FreeList[level].front = node->free.next;
    }
    else
    {
        Node* const prevFreeNode = node->free.prev;
        VMA_ASSERT(prevFreeNode->free.next == node);
        prevFreeNode->free.next = node->free.next;
    }

    // It is at the back.
    if(node->free.next == VMA_NULL)
    {
        VMA_ASSERT(m_FreeList[level].back == node);
        m_FreeList[level].back = node->free.prev;
    }
    else
    {
        Node* const nextFreeNode = node->free.next;
        VMA_ASSERT(nextFreeNode->free.prev == node);
        nextFreeNode->free.prev = node->free.prev;
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Buddy::PrintDetailedMapNode(class VmaJsonWriter& json, const Node* node, VkDeviceSize levelNodeSize) const
{
    switch(node->type)
    {
    case Node::TYPE_FREE:
        PrintDetailedMap_UnusedRange(json, node->offset, levelNodeSize);
        break;
    case Node::TYPE_ALLOCATION:
        {   
            PrintDetailedMap_Allocation(json, node->offset, node->allocation.alloc);
            const VkDeviceSize allocSize = node->allocation.alloc->GetSize();
            if(allocSize < levelNodeSize)
            {
                PrintDetailedMap_UnusedRange(json, node->offset + allocSize, levelNodeSize - allocSize);
            }
        }
        break;
    case Node::TYPE_SPLIT:
        {
            const VkDeviceSize childrenNodeSize = levelNodeSize / 2;
            const Node* const leftChild = node->split.leftChild;
            PrintDetailedMapNode(json, leftChild, childrenNodeSize);
            const Node* const rightChild = leftChild->buddy;
            PrintDetailedMapNode(json, rightChild, childrenNodeSize);
        }
        break;
    default:
        VMA_ASSERT(0);
    }
}
#endif // #if VMA_STATS_STRING_ENABLED


////////////////////////////////////////////////////////////////////////////////
// class VmaDeviceMemoryBlock

VmaDeviceMemoryBlock::VmaDeviceMemoryBlock(VmaAllocator hAllocator) :
    m_pMetadata(VMA_NULL),
    m_MemoryTypeIndex(UINT32_MAX),
    m_Id(0),
    m_hMemory(VK_NULL_HANDLE),
    m_MapCount(0),
    m_pMappedData(VMA_NULL)
{
}

void VmaDeviceMemoryBlock::Init(
    VmaAllocator hAllocator,
    VmaPool hParentPool,
    uint32_t newMemoryTypeIndex,
    VkDeviceMemory newMemory,
    VkDeviceSize newSize,
    uint32_t id,
    uint32_t algorithm)
{
    VMA_ASSERT(m_hMemory == VK_NULL_HANDLE);

    m_hParentPool = hParentPool;
    m_MemoryTypeIndex = newMemoryTypeIndex;
    m_Id = id;
    m_hMemory = newMemory;

    switch(algorithm)
    {
    case VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT:
        m_pMetadata = vma_new(hAllocator, VmaBlockMetadata_Linear)(hAllocator);
        break;
    case VMA_POOL_CREATE_BUDDY_ALGORITHM_BIT:
        m_pMetadata = vma_new(hAllocator, VmaBlockMetadata_Buddy)(hAllocator);
        break;
    default:
        VMA_ASSERT(0);
        // Fall-through.
    case 0:
        m_pMetadata = vma_new(hAllocator, VmaBlockMetadata_Generic)(hAllocator);
    }
    m_pMetadata->Init(newSize);
}

void VmaDeviceMemoryBlock::Destroy(VmaAllocator allocator)
{
    // This is the most important assert in the entire library.
    // Hitting it means you have some memory leak - unreleased VmaAllocation objects.
    VMA_ASSERT(m_pMetadata->IsEmpty() && "Some allocations were not freed before destruction of this memory block!");

    VMA_ASSERT(m_hMemory != VK_NULL_HANDLE);
    allocator->FreeVulkanMemory(m_MemoryTypeIndex, m_pMetadata->GetSize(), m_hMemory);
    m_hMemory = VK_NULL_HANDLE;

    vma_delete(allocator, m_pMetadata);
    m_pMetadata = VMA_NULL;
}

bool VmaDeviceMemoryBlock::Validate() const
{
    VMA_VALIDATE((m_hMemory != VK_NULL_HANDLE) &&
        (m_pMetadata->GetSize() != 0));
    
    return m_pMetadata->Validate();
}

VkResult VmaDeviceMemoryBlock::CheckCorruption(VmaAllocator hAllocator)
{
    void* pData = nullptr;
    VkResult res = Map(hAllocator, 1, &pData);
    if(res != VK_SUCCESS)
    {
        return res;
    }

    res = m_pMetadata->CheckCorruption(pData);

    Unmap(hAllocator, 1);

    return res;
}

VkResult VmaDeviceMemoryBlock::Map(VmaAllocator hAllocator, uint32_t count, void** ppData)
{
    if(count == 0)
    {
        return VK_SUCCESS;
    }

    VmaMutexLock lock(m_Mutex, hAllocator->m_UseMutex);
    if(m_MapCount != 0)
    {
        m_MapCount += count;
        VMA_ASSERT(m_pMappedData != VMA_NULL);
        if(ppData != VMA_NULL)
        {
            *ppData = m_pMappedData;
        }
        return VK_SUCCESS;
    }
    else
    {
        VkResult result = (*hAllocator->GetVulkanFunctions().vkMapMemory)(
            hAllocator->m_hDevice,
            m_hMemory,
            0, // offset
            VK_WHOLE_SIZE,
            0, // flags
            &m_pMappedData);
        if(result == VK_SUCCESS)
        {
            if(ppData != VMA_NULL)
            {
                *ppData = m_pMappedData;
            }
            m_MapCount = count;
        }
        return result;
    }
}

void VmaDeviceMemoryBlock::Unmap(VmaAllocator hAllocator, uint32_t count)
{
    if(count == 0)
    {
        return;
    }

    VmaMutexLock lock(m_Mutex, hAllocator->m_UseMutex);
    if(m_MapCount >= count)
    {
        m_MapCount -= count;
        if(m_MapCount == 0)
        {
            m_pMappedData = VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(hAllocator->m_hDevice, m_hMemory);
        }
    }
    else
    {
        VMA_ASSERT(0 && "VkDeviceMemory block is being unmapped while it was not previously mapped.");
    }
}

VkResult VmaDeviceMemoryBlock::WriteMagicValueAroundAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    VMA_ASSERT(VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_MARGIN % 4 == 0 && VMA_DEBUG_DETECT_CORRUPTION);
    VMA_ASSERT(allocOffset >= VMA_DEBUG_MARGIN);

    void* pData;
    VkResult res = Map(hAllocator, 1, &pData);
    if(res != VK_SUCCESS)
    {
        return res;
    }

    VmaWriteMagicValue(pData, allocOffset - VMA_DEBUG_MARGIN);
    VmaWriteMagicValue(pData, allocOffset + allocSize);

    Unmap(hAllocator, 1);

    return VK_SUCCESS;
}

VkResult VmaDeviceMemoryBlock::ValidateMagicValueAroundAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    VMA_ASSERT(VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_MARGIN % 4 == 0 && VMA_DEBUG_DETECT_CORRUPTION);
    VMA_ASSERT(allocOffset >= VMA_DEBUG_MARGIN);

    void* pData;
    VkResult res = Map(hAllocator, 1, &pData);
    if(res != VK_SUCCESS)
    {
        return res;
    }

    if(!VmaValidateMagicValue(pData, allocOffset - VMA_DEBUG_MARGIN))
    {
        VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED BEFORE FREED ALLOCATION!");
    }
    else if(!VmaValidateMagicValue(pData, allocOffset + allocSize))
    {
        VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER FREED ALLOCATION!");
    }

    Unmap(hAllocator, 1);

    return VK_SUCCESS;
}

VkResult VmaDeviceMemoryBlock::BindBufferMemory(
    const VmaAllocator hAllocator,
    const VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    VMA_ASSERT(hAllocation->GetType() == VmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->GetBlock() == this);
    VMA_ASSERT(allocationLocalOffset < hAllocation->GetSize() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->GetOffset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    VmaMutexLock lock(m_Mutex, hAllocator->m_UseMutex);
    return hAllocator->BindVulkanBuffer(m_hMemory, memoryOffset, hBuffer, pNext);
}

VkResult VmaDeviceMemoryBlock::BindImageMemory(
    const VmaAllocator hAllocator,
    const VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    VMA_ASSERT(hAllocation->GetType() == VmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->GetBlock() == this);
    VMA_ASSERT(allocationLocalOffset < hAllocation->GetSize() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->GetOffset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    VmaMutexLock lock(m_Mutex, hAllocator->m_UseMutex);
    return hAllocator->BindVulkanImage(m_hMemory, memoryOffset, hImage, pNext);
}

static void InitStatInfo(VmaStatInfo& outInfo)
{
    memset(&outInfo, 0, sizeof(outInfo));
    outInfo.allocationSizeMin = UINT64_MAX;
    outInfo.unusedRangeSizeMin = UINT64_MAX;
}

// Adds statistics srcInfo into inoutInfo, like: inoutInfo += srcInfo.
static void VmaAddStatInfo(VmaStatInfo& inoutInfo, const VmaStatInfo& srcInfo)
{
    inoutInfo.blockCount += srcInfo.blockCount;
    inoutInfo.allocationCount += srcInfo.allocationCount;
    inoutInfo.unusedRangeCount += srcInfo.unusedRangeCount;
    inoutInfo.usedBytes += srcInfo.usedBytes;
    inoutInfo.unusedBytes += srcInfo.unusedBytes;
    inoutInfo.allocationSizeMin = VMA_MIN(inoutInfo.allocationSizeMin, srcInfo.allocationSizeMin);
    inoutInfo.allocationSizeMax = VMA_MAX(inoutInfo.allocationSizeMax, srcInfo.allocationSizeMax);
    inoutInfo.unusedRangeSizeMin = VMA_MIN(inoutInfo.unusedRangeSizeMin, srcInfo.unusedRangeSizeMin);
    inoutInfo.unusedRangeSizeMax = VMA_MAX(inoutInfo.unusedRangeSizeMax, srcInfo.unusedRangeSizeMax);
}

static void VmaPostprocessCalcStatInfo(VmaStatInfo& inoutInfo)
{
    inoutInfo.allocationSizeAvg = (inoutInfo.allocationCount > 0) ?
        VmaRoundDiv<VkDeviceSize>(inoutInfo.usedBytes, inoutInfo.allocationCount) : 0;
    inoutInfo.unusedRangeSizeAvg = (inoutInfo.unusedRangeCount > 0) ?
        VmaRoundDiv<VkDeviceSize>(inoutInfo.unusedBytes, inoutInfo.unusedRangeCount) : 0;
}

VmaPool_T::VmaPool_T(
    VmaAllocator hAllocator,
    const VmaPoolCreateInfo& createInfo,
    VkDeviceSize preferredBlockSize) :
    m_BlockVector(
        hAllocator,
        this, // hParentPool
        createInfo.memoryTypeIndex,
        createInfo.blockSize != 0 ? createInfo.blockSize : preferredBlockSize,
        createInfo.minBlockCount,
        createInfo.maxBlockCount,
        (createInfo.flags & VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT) != 0 ? 1 : hAllocator->GetBufferImageGranularity(),
        createInfo.frameInUseCount,
        createInfo.blockSize != 0, // explicitBlockSize
        createInfo.flags & VMA_POOL_CREATE_ALGORITHM_MASK), // algorithm
    m_Id(0),
    m_Name(VMA_NULL)
{
}

VmaPool_T::~VmaPool_T()
{
}

void VmaPool_T::SetName(const char* pName)
{
    const VkAllocationCallbacks* allocs = m_BlockVector.GetAllocator()->GetAllocationCallbacks();
    VmaFreeString(allocs, m_Name);
    
    if(pName != VMA_NULL)
    {
        m_Name = VmaCreateStringCopy(allocs, pName);
    }
    else
    {
        m_Name = VMA_NULL;
    }
}

#if VMA_STATS_STRING_ENABLED

#endif // #if VMA_STATS_STRING_ENABLED

VmaBlockVector::VmaBlockVector(
    VmaAllocator hAllocator,
    VmaPool hParentPool,
    uint32_t memoryTypeIndex,
    VkDeviceSize preferredBlockSize,
    size_t minBlockCount,
    size_t maxBlockCount,
    VkDeviceSize bufferImageGranularity,
    uint32_t frameInUseCount,
    bool explicitBlockSize,
    uint32_t algorithm) :
    m_hAllocator(hAllocator),
    m_hParentPool(hParentPool),
    m_MemoryTypeIndex(memoryTypeIndex),
    m_PreferredBlockSize(preferredBlockSize),
    m_MinBlockCount(minBlockCount),
    m_MaxBlockCount(maxBlockCount),
    m_BufferImageGranularity(bufferImageGranularity),
    m_FrameInUseCount(frameInUseCount),
    m_ExplicitBlockSize(explicitBlockSize),
    m_Algorithm(algorithm),
    m_HasEmptyBlock(false),
    m_Blocks(VmaStlAllocator<VmaDeviceMemoryBlock*>(hAllocator->GetAllocationCallbacks())),
    m_NextBlockId(0)
{
}

VmaBlockVector::~VmaBlockVector()
{
    for(size_t i = m_Blocks.size(); i--; )
    {
        m_Blocks[i]->Destroy(m_hAllocator);
        vma_delete(m_hAllocator, m_Blocks[i]);
    }
}

VkResult VmaBlockVector::CreateMinBlocks()
{
    for(size_t i = 0; i < m_MinBlockCount; ++i)
    {
        VkResult res = CreateBlock(m_PreferredBlockSize, VMA_NULL);
        if(res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

void VmaBlockVector::GetPoolStats(VmaPoolStats* pStats)
{
    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);

    const size_t blockCount = m_Blocks.size();

    pStats->size = 0;
    pStats->unusedSize = 0;
    pStats->allocationCount = 0;
    pStats->unusedRangeCount = 0;
    pStats->unusedRangeSizeMax = 0;
    pStats->blockCount = blockCount;

    for(uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        const VmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VMA_HEAVY_ASSERT(pBlock->Validate());
        pBlock->m_pMetadata->AddPoolStats(*pStats);
    }
}

bool VmaBlockVector::IsEmpty()
{
    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);
    return m_Blocks.empty();
}

bool VmaBlockVector::IsCorruptionDetectionEnabled() const
{
    const uint32_t requiredMemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return (VMA_DEBUG_DETECT_CORRUPTION != 0) &&
        (VMA_DEBUG_MARGIN > 0) &&
        (m_Algorithm == 0 || m_Algorithm == VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT) &&
        (m_hAllocator->m_MemProps.memoryTypes[m_MemoryTypeIndex].propertyFlags & requiredMemFlags) == requiredMemFlags;
}

static const uint32_t VMA_ALLOCATION_TRY_COUNT = 32;

VkResult VmaBlockVector::Allocate(
    uint32_t currentFrameIndex,
    VkDeviceSize size,
    VkDeviceSize alignment,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    size_t allocIndex;
    VkResult res = VK_SUCCESS;

    if(IsCorruptionDetectionEnabled())
    {
        size = VmaAlignUp<VkDeviceSize>(size, sizeof(VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
        alignment = VmaAlignUp<VkDeviceSize>(alignment, sizeof(VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
    }

    {
        VmaMutexLockWrite lock(m_Mutex, m_hAllocator->m_UseMutex);
        for(allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
        {
            res = AllocatePage(
                currentFrameIndex,
                size,
                alignment,
                createInfo,
                suballocType,
                pAllocations + allocIndex);
            if(res != VK_SUCCESS)
            {
                break;
            }
        }
    }

    if(res != VK_SUCCESS)
    {
        // Free all already created allocations.
        while(allocIndex--)
        {
            Free(pAllocations[allocIndex]);
        }
        memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);
    }

    return res;
}

VkResult VmaBlockVector::AllocatePage(
    uint32_t currentFrameIndex,
    VkDeviceSize size,
    VkDeviceSize alignment,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    VmaAllocation* pAllocation)
{
    const bool isUpperAddress = (createInfo.flags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;
    bool canMakeOtherLost = (createInfo.flags & VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT) != 0;
    const bool mapped = (createInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;
    const bool isUserDataString = (createInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0;
    
    const bool withinBudget = (createInfo.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0;
    VkDeviceSize freeMemory;
    {
        const uint32_t heapIndex = m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex);
        VmaBudget heapBudget = {};
        m_hAllocator->GetBudget(&heapBudget, heapIndex, 1);
        freeMemory = (heapBudget.usage < heapBudget.budget) ? (heapBudget.budget - heapBudget.usage) : 0;
    }
    
    const bool canFallbackToDedicated = !IsCustomPool();
    const bool canCreateNewBlock =
        ((createInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0) &&
        (m_Blocks.size() < m_MaxBlockCount) &&
        (freeMemory >= size || !canFallbackToDedicated);
    uint32_t strategy = createInfo.flags & VMA_ALLOCATION_CREATE_STRATEGY_MASK;

    // If linearAlgorithm is used, canMakeOtherLost is available only when used as ring buffer.
    // Which in turn is available only when maxBlockCount = 1.
    if(m_Algorithm == VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT && m_MaxBlockCount > 1)
    {
        canMakeOtherLost = false;
    }

    // Upper address can only be used with linear allocator and within single memory block.
    if(isUpperAddress &&
        (m_Algorithm != VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT || m_MaxBlockCount > 1))
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Validate strategy.
    switch(strategy)
    {
    case 0:
        strategy = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT;
        break;
    case VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT:
    case VMA_ALLOCATION_CREATE_STRATEGY_WORST_FIT_BIT:
    case VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT:
        break;
    default:
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Early reject: requested allocation size is larger that maximum block size for this block vector.
    if(size + 2 * VMA_DEBUG_MARGIN > m_PreferredBlockSize)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    /*
    Under certain condition, this whole section can be skipped for optimization, so
    we move on directly to trying to allocate with canMakeOtherLost. That's the case
    e.g. for custom pools with linear algorithm.
    */
    if(!canMakeOtherLost || canCreateNewBlock)
    {
        // 1. Search existing allocations. Try to allocate without making other allocations lost.
        VmaAllocationCreateFlags allocFlagsCopy = createInfo.flags;
        allocFlagsCopy &= ~VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT;

        if(m_Algorithm == VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
        {
            // Use only last block.
            if(!m_Blocks.empty())
            {
                VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks.back();
                VMA_ASSERT(pCurrBlock);
                VkResult res = AllocateFromBlock(
                    pCurrBlock,
                    currentFrameIndex,
                    size,
                    alignment,
                    allocFlagsCopy,
                    createInfo.pUserData,
                    suballocType,
                    strategy,
                    pAllocation);
                if(res == VK_SUCCESS)
                {
                    VMA_DEBUG_LOG("    Returned from last block #%u", pCurrBlock->GetId());
                    return VK_SUCCESS;
                }
            }
        }
        else
        {
            if(strategy == VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT)
            {
                // Forward order in m_Blocks - prefer blocks with smallest amount of free space.
                for(size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex )
                {
                    VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                    VMA_ASSERT(pCurrBlock);
                    VkResult res = AllocateFromBlock(
                        pCurrBlock,
                        currentFrameIndex,
                        size,
                        alignment,
                        allocFlagsCopy,
                        createInfo.pUserData,
                        suballocType,
                        strategy,
                        pAllocation);
                    if(res == VK_SUCCESS)
                    {
                        VMA_DEBUG_LOG("    Returned from existing block #%u", pCurrBlock->GetId());
                        return VK_SUCCESS;
                    }
                }
            }
            else // WORST_FIT, FIRST_FIT
            {
                // Backward order in m_Blocks - prefer blocks with largest amount of free space.
                for(size_t blockIndex = m_Blocks.size(); blockIndex--; )
                {
                    VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                    VMA_ASSERT(pCurrBlock);
                    VkResult res = AllocateFromBlock(
                        pCurrBlock,
                        currentFrameIndex,
                        size,
                        alignment,
                        allocFlagsCopy,
                        createInfo.pUserData,
                        suballocType,
                        strategy,
                        pAllocation);
                    if(res == VK_SUCCESS)
                    {
                        VMA_DEBUG_LOG("    Returned from existing block #%u", pCurrBlock->GetId());
                        return VK_SUCCESS;
                    }
                }
            }
        }

        // 2. Try to create new block.
        if(canCreateNewBlock)
        {
            // Calculate optimal size for new block.
            VkDeviceSize newBlockSize = m_PreferredBlockSize;
            uint32_t newBlockSizeShift = 0;
            const uint32_t NEW_BLOCK_SIZE_SHIFT_MAX = 3;

            if(!m_ExplicitBlockSize)
            {
                // Allocate 1/8, 1/4, 1/2 as first blocks.
                const VkDeviceSize maxExistingBlockSize = CalcMaxBlockSize();
                for(uint32_t i = 0; i < NEW_BLOCK_SIZE_SHIFT_MAX; ++i)
                {
                    const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                    if(smallerNewBlockSize > maxExistingBlockSize && smallerNewBlockSize >= size * 2)
                    {
                        newBlockSize = smallerNewBlockSize;
                        ++newBlockSizeShift;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            size_t newBlockIndex = 0;
            VkResult res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
                CreateBlock(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
            // Allocation of this size failed? Try 1/2, 1/4, 1/8 of m_PreferredBlockSize.
            if(!m_ExplicitBlockSize)
            {
                while(res < 0 && newBlockSizeShift < NEW_BLOCK_SIZE_SHIFT_MAX)
                {
                    const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                    if(smallerNewBlockSize >= size)
                    {
                        newBlockSize = smallerNewBlockSize;
                        ++newBlockSizeShift;
                        res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
                            CreateBlock(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            if(res == VK_SUCCESS)
            {
                VmaDeviceMemoryBlock* const pBlock = m_Blocks[newBlockIndex];
                VMA_ASSERT(pBlock->m_pMetadata->GetSize() >= size);

                res = AllocateFromBlock(
                    pBlock,
                    currentFrameIndex,
                    size,
                    alignment,
                    allocFlagsCopy,
                    createInfo.pUserData,
                    suballocType,
                    strategy,
                    pAllocation);
                if(res == VK_SUCCESS)
                {
                    VMA_DEBUG_LOG("    Created new block #%u Size=%llu", pBlock->GetId(), newBlockSize);
                    return VK_SUCCESS;
                }
                else
                {
                    // Allocation from new block failed, possibly due to VMA_DEBUG_MARGIN or alignment.
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
            }
        }
    }

    // 3. Try to allocate from existing blocks with making other allocations lost.
    if(canMakeOtherLost)
    {
        uint32_t tryIndex = 0;
        for(; tryIndex < VMA_ALLOCATION_TRY_COUNT; ++tryIndex)
        {
            VmaDeviceMemoryBlock* pBestRequestBlock = VMA_NULL;
            VmaAllocationRequest bestRequest = {};
            VkDeviceSize bestRequestCost = VK_WHOLE_SIZE;

            // 1. Search existing allocations.
            if(strategy == VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT)
            {
                // Forward order in m_Blocks - prefer blocks with smallest amount of free space.
                for(size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex )
                {
                    VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                    VMA_ASSERT(pCurrBlock);
                    VmaAllocationRequest currRequest = {};
                    if(pCurrBlock->m_pMetadata->CreateAllocationRequest(
                        currentFrameIndex,
                        m_FrameInUseCount,
                        m_BufferImageGranularity,
                        size,
                        alignment,
                        (createInfo.flags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0,
                        suballocType,
                        canMakeOtherLost,
                        strategy,
                        &currRequest))
                    {
                        const VkDeviceSize currRequestCost = currRequest.CalcCost();
                        if(pBestRequestBlock == VMA_NULL ||
                            currRequestCost < bestRequestCost)
                        {
                            pBestRequestBlock = pCurrBlock;
                            bestRequest = currRequest;
                            bestRequestCost = currRequestCost;

                            if(bestRequestCost == 0)
                            {
                                break;
                            }
                        }
                    }
                }
            }
            else // WORST_FIT, FIRST_FIT
            {
                // Backward order in m_Blocks - prefer blocks with largest amount of free space.
                for(size_t blockIndex = m_Blocks.size(); blockIndex--; )
                {
                    VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                    VMA_ASSERT(pCurrBlock);
                    VmaAllocationRequest currRequest = {};
                    if(pCurrBlock->m_pMetadata->CreateAllocationRequest(
                        currentFrameIndex,
                        m_FrameInUseCount,
                        m_BufferImageGranularity,
                        size,
                        alignment,
                        (createInfo.flags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0,
                        suballocType,
                        canMakeOtherLost,
                        strategy,
                        &currRequest))
                    {
                        const VkDeviceSize currRequestCost = currRequest.CalcCost();
                        if(pBestRequestBlock == VMA_NULL ||
                            currRequestCost < bestRequestCost ||
                            strategy == VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT)
                        {
                            pBestRequestBlock = pCurrBlock;
                            bestRequest = currRequest;
                            bestRequestCost = currRequestCost;

                            if(bestRequestCost == 0 ||
                                strategy == VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT)
                            {
                                break;
                            }
                        }
                    }
                }
            }

            if(pBestRequestBlock != VMA_NULL)
            {
                if(mapped)
                {
                    VkResult res = pBestRequestBlock->Map(m_hAllocator, 1, VMA_NULL);
                    if(res != VK_SUCCESS)
                    {
                        return res;
                    }
                }

                if(pBestRequestBlock->m_pMetadata->MakeRequestedAllocationsLost(
                    currentFrameIndex,
                    m_FrameInUseCount,
                    &bestRequest))
                {
                    // Allocate from this pBlock.
                    *pAllocation = m_hAllocator->m_AllocationObjectAllocator.Allocate();
                    (*pAllocation)->Ctor(currentFrameIndex, isUserDataString);
                    pBestRequestBlock->m_pMetadata->Alloc(bestRequest, suballocType, size, *pAllocation);
                    UpdateHasEmptyBlock();
                    (*pAllocation)->InitBlockAllocation(
                        pBestRequestBlock,
                        bestRequest.offset,
                        alignment,
                        size,
                        m_MemoryTypeIndex,
                        suballocType,
                        mapped,
                        (createInfo.flags & VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT) != 0);
                    VMA_HEAVY_ASSERT(pBestRequestBlock->Validate());
                    VMA_DEBUG_LOG("    Returned from existing block");
                    (*pAllocation)->SetUserData(m_hAllocator, createInfo.pUserData);
                    m_hAllocator->m_Budget.AddAllocation(m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex), size);
                    if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
                    {
                        m_hAllocator->FillAllocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
                    }
                    if(IsCorruptionDetectionEnabled())
                    {
                        VkResult res = pBestRequestBlock->WriteMagicValueAroundAllocation(m_hAllocator, bestRequest.offset, size);
                        VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to write magic value.");
                    }
                    return VK_SUCCESS;
                }
                // else: Some allocations must have been touched while we are here. Next try.
            }
            else
            {
                // Could not find place in any of the blocks - break outer loop.
                break;
            }
        }
        /* Maximum number of tries exceeded - a very unlike event when many other
        threads are simultaneously touching allocations making it impossible to make
        lost at the same time as we try to allocate. */
        if(tryIndex == VMA_ALLOCATION_TRY_COUNT)
        {
            return VK_ERROR_TOO_MANY_OBJECTS;
        }
    }

    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaBlockVector::Free(
    const VmaAllocation hAllocation)
{
    VmaDeviceMemoryBlock* pBlockToDelete = VMA_NULL;

    bool budgetExceeded = false;
    {
        const uint32_t heapIndex = m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex);
        VmaBudget heapBudget = {};
        m_hAllocator->GetBudget(&heapBudget, heapIndex, 1);
        budgetExceeded = heapBudget.usage >= heapBudget.budget;
    }

    // Scope for lock.
    {
        VmaMutexLockWrite lock(m_Mutex, m_hAllocator->m_UseMutex);

        VmaDeviceMemoryBlock* pBlock = hAllocation->GetBlock();

        if(IsCorruptionDetectionEnabled())
        {
            VkResult res = pBlock->ValidateMagicValueAroundAllocation(m_hAllocator, hAllocation->GetOffset(), hAllocation->GetSize());
            VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to validate magic value.");
        }

        if(hAllocation->IsPersistentMap())
        {
            pBlock->Unmap(m_hAllocator, 1);
        }

        pBlock->m_pMetadata->Free(hAllocation);
        VMA_HEAVY_ASSERT(pBlock->Validate());

        VMA_DEBUG_LOG("  Freed from MemoryTypeIndex=%u", m_MemoryTypeIndex);

        const bool canDeleteBlock = m_Blocks.size() > m_MinBlockCount;
        // pBlock became empty after this deallocation.
        if(pBlock->m_pMetadata->IsEmpty())
        {
            // Already has empty block. We don't want to have two, so delete this one.
            if((m_HasEmptyBlock || budgetExceeded) && canDeleteBlock)
            {
                pBlockToDelete = pBlock;
                Remove(pBlock);
            }
            // else: We now have an empty block - leave it.
        }
        // pBlock didn't become empty, but we have another empty block - find and free that one.
        // (This is optional, heuristics.)
        else if(m_HasEmptyBlock && canDeleteBlock)
        {
            VmaDeviceMemoryBlock* pLastBlock = m_Blocks.back();
            if(pLastBlock->m_pMetadata->IsEmpty())
            {
                pBlockToDelete = pLastBlock;
                m_Blocks.pop_back();
            }
        }

        UpdateHasEmptyBlock();
        IncrementallySortBlocks();
    }

    // Destruction of a free block. Deferred until this point, outside of mutex
    // lock, for performance reason.
    if(pBlockToDelete != VMA_NULL)
    {
        VMA_DEBUG_LOG("    Deleted empty block");
        pBlockToDelete->Destroy(m_hAllocator);
        vma_delete(m_hAllocator, pBlockToDelete);
    }
}

VkDeviceSize VmaBlockVector::CalcMaxBlockSize() const
{
    VkDeviceSize result = 0;
    for(size_t i = m_Blocks.size(); i--; )
    {
        result = VMA_MAX(result, m_Blocks[i]->m_pMetadata->GetSize());
        if(result >= m_PreferredBlockSize)
        {
            break;
        }
    }
    return result;
}

void VmaBlockVector::Remove(VmaDeviceMemoryBlock* pBlock)
{
    for(uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        if(m_Blocks[blockIndex] == pBlock)
        {
            VmaVectorRemove(m_Blocks, blockIndex);
            return;
        }
    }
    VMA_ASSERT(0);
}

void VmaBlockVector::IncrementallySortBlocks()
{
    if(m_Algorithm != VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
    {
        // Bubble sort only until first swap.
        for(size_t i = 1; i < m_Blocks.size(); ++i)
        {
            if(m_Blocks[i - 1]->m_pMetadata->GetSumFreeSize() > m_Blocks[i]->m_pMetadata->GetSumFreeSize())
            {
                VMA_SWAP(m_Blocks[i - 1], m_Blocks[i]);
                return;
            }
        }
    }
}

VkResult VmaBlockVector::AllocateFromBlock(
    VmaDeviceMemoryBlock* pBlock,
    uint32_t currentFrameIndex,
    VkDeviceSize size,
    VkDeviceSize alignment,
    VmaAllocationCreateFlags allocFlags,
    void* pUserData,
    VmaSuballocationType suballocType,
    uint32_t strategy,
    VmaAllocation* pAllocation)
{
    VMA_ASSERT((allocFlags & VMA_ALLOCATION_CREATE_CAN_MAKE_OTHER_LOST_BIT) == 0);
    const bool isUpperAddress = (allocFlags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;
    const bool mapped = (allocFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;
    const bool isUserDataString = (allocFlags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0;

    VmaAllocationRequest currRequest = {};
    if(pBlock->m_pMetadata->CreateAllocationRequest(
        currentFrameIndex,
        m_FrameInUseCount,
        m_BufferImageGranularity,
        size,
        alignment,
        isUpperAddress,
        suballocType,
        false, // canMakeOtherLost
        strategy,
        &currRequest))
    {
        // Allocate from pCurrBlock.
        VMA_ASSERT(currRequest.itemsToMakeLostCount == 0);

        if(mapped)
        {
            VkResult res = pBlock->Map(m_hAllocator, 1, VMA_NULL);
            if(res != VK_SUCCESS)
            {
                return res;
            }
        }
            
        *pAllocation = m_hAllocator->m_AllocationObjectAllocator.Allocate();
        (*pAllocation)->Ctor(currentFrameIndex, isUserDataString);
        pBlock->m_pMetadata->Alloc(currRequest, suballocType, size, *pAllocation);
        UpdateHasEmptyBlock();
        (*pAllocation)->InitBlockAllocation(
            pBlock,
            currRequest.offset,
            alignment,
            size,
            m_MemoryTypeIndex,
            suballocType,
            mapped,
            (allocFlags & VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT) != 0);
        VMA_HEAVY_ASSERT(pBlock->Validate());
        (*pAllocation)->SetUserData(m_hAllocator, pUserData);
        m_hAllocator->m_Budget.AddAllocation(m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex), size);
        if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
        {
            m_hAllocator->FillAllocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
        }
        if(IsCorruptionDetectionEnabled())
        {
            VkResult res = pBlock->WriteMagicValueAroundAllocation(m_hAllocator, currRequest.offset, size);
            VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to write magic value.");
        }
        return VK_SUCCESS;
    }
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult VmaBlockVector::CreateBlock(VkDeviceSize blockSize, size_t* pNewBlockIndex)
{
    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.memoryTypeIndex = m_MemoryTypeIndex;
    allocInfo.allocationSize = blockSize;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult res = m_hAllocator->AllocateVulkanMemory(&allocInfo, &mem);
    if(res < 0)
    {
        return res;
    }

    // New VkDeviceMemory successfully created.

    // Create new Allocation for it.
    VmaDeviceMemoryBlock* const pBlock = vma_new(m_hAllocator, VmaDeviceMemoryBlock)(m_hAllocator);
    pBlock->Init(
        m_hAllocator,
        m_hParentPool,
        m_MemoryTypeIndex,
        mem,
        allocInfo.allocationSize,
        m_NextBlockId++,
        m_Algorithm);

    m_Blocks.push_back(pBlock);
    if(pNewBlockIndex != VMA_NULL)
    {
        *pNewBlockIndex = m_Blocks.size() - 1;
    }

    return VK_SUCCESS;
}

void VmaBlockVector::ApplyDefragmentationMovesCpu(
    class VmaBlockVectorDefragmentationContext* pDefragCtx,
    const VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves)
{
    const size_t blockCount = m_Blocks.size();
    const bool isNonCoherent = m_hAllocator->IsMemoryTypeNonCoherent(m_MemoryTypeIndex);

    enum BLOCK_FLAG
    {
        BLOCK_FLAG_USED = 0x00000001,
        BLOCK_FLAG_MAPPED_FOR_DEFRAGMENTATION = 0x00000002,
    };

    struct BlockInfo
    {
        uint32_t flags;
        void* pMappedData;
    };
    VmaVector< BlockInfo, VmaStlAllocator<BlockInfo> >
        blockInfo(blockCount, BlockInfo(), VmaStlAllocator<BlockInfo>(m_hAllocator->GetAllocationCallbacks()));
    memset(blockInfo.data(), 0, blockCount * sizeof(BlockInfo));

    // Go over all moves. Mark blocks that are used with BLOCK_FLAG_USED.
    const size_t moveCount = moves.size();
    for(size_t moveIndex = 0; moveIndex < moveCount; ++moveIndex)
    {
        const VmaDefragmentationMove& move = moves[moveIndex];
        blockInfo[move.srcBlockIndex].flags |= BLOCK_FLAG_USED;
        blockInfo[move.dstBlockIndex].flags |= BLOCK_FLAG_USED;
    }

    VMA_ASSERT(pDefragCtx->res == VK_SUCCESS);

    // Go over all blocks. Get mapped pointer or map if necessary.
    for(size_t blockIndex = 0; pDefragCtx->res == VK_SUCCESS && blockIndex < blockCount; ++blockIndex)
    {
        BlockInfo& currBlockInfo = blockInfo[blockIndex];
        VmaDeviceMemoryBlock* pBlock = m_Blocks[blockIndex];
        if((currBlockInfo.flags & BLOCK_FLAG_USED) != 0)
        {
            currBlockInfo.pMappedData = pBlock->GetMappedData();
            // It is not originally mapped - map it.
            if(currBlockInfo.pMappedData == VMA_NULL)
            {
                pDefragCtx->res = pBlock->Map(m_hAllocator, 1, &currBlockInfo.pMappedData);
                if(pDefragCtx->res == VK_SUCCESS)
                {
                    currBlockInfo.flags |= BLOCK_FLAG_MAPPED_FOR_DEFRAGMENTATION;
                }
            }
        }
    }

    // Go over all moves. Do actual data transfer.
    if(pDefragCtx->res == VK_SUCCESS)
    {
        const VkDeviceSize nonCoherentAtomSize = m_hAllocator->m_PhysicalDeviceProperties.limits.nonCoherentAtomSize;
        VkMappedMemoryRange memRange = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };

        for(size_t moveIndex = 0; moveIndex < moveCount; ++moveIndex)
        {
            const VmaDefragmentationMove& move = moves[moveIndex];

            const BlockInfo& srcBlockInfo = blockInfo[move.srcBlockIndex];
            const BlockInfo& dstBlockInfo = blockInfo[move.dstBlockIndex];

            VMA_ASSERT(srcBlockInfo.pMappedData && dstBlockInfo.pMappedData);

            // Invalidate source.
            if(isNonCoherent)
            {
                VmaDeviceMemoryBlock* const pSrcBlock = m_Blocks[move.srcBlockIndex];
                memRange.memory = pSrcBlock->GetDeviceMemory();
                memRange.offset = VmaAlignDown(move.srcOffset, nonCoherentAtomSize);
                memRange.size = VMA_MIN(
                    VmaAlignUp(move.size + (move.srcOffset - memRange.offset), nonCoherentAtomSize),
                    pSrcBlock->m_pMetadata->GetSize() - memRange.offset);
                (*m_hAllocator->GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hAllocator->m_hDevice, 1, &memRange);
            }

            // THE PLACE WHERE ACTUAL DATA COPY HAPPENS.
            memmove(
                reinterpret_cast<char*>(dstBlockInfo.pMappedData) + move.dstOffset,
                reinterpret_cast<char*>(srcBlockInfo.pMappedData) + move.srcOffset,
                static_cast<size_t>(move.size));

            if(IsCorruptionDetectionEnabled())
            {
                VmaWriteMagicValue(dstBlockInfo.pMappedData, move.dstOffset - VMA_DEBUG_MARGIN);
                VmaWriteMagicValue(dstBlockInfo.pMappedData, move.dstOffset + move.size);
            }

            // Flush destination.
            if(isNonCoherent)
            {
                VmaDeviceMemoryBlock* const pDstBlock = m_Blocks[move.dstBlockIndex];
                memRange.memory = pDstBlock->GetDeviceMemory();
                memRange.offset = VmaAlignDown(move.dstOffset, nonCoherentAtomSize);
                memRange.size = VMA_MIN(
                    VmaAlignUp(move.size + (move.dstOffset - memRange.offset), nonCoherentAtomSize),
                    pDstBlock->m_pMetadata->GetSize() - memRange.offset);
                (*m_hAllocator->GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hAllocator->m_hDevice, 1, &memRange);
            }
        }
    }

    // Go over all blocks in reverse order. Unmap those that were mapped just for defragmentation.
    // Regardless of pCtx->res == VK_SUCCESS.
    for(size_t blockIndex = blockCount; blockIndex--; )
    {
        const BlockInfo& currBlockInfo = blockInfo[blockIndex];
        if((currBlockInfo.flags & BLOCK_FLAG_MAPPED_FOR_DEFRAGMENTATION) != 0)
        {
            VmaDeviceMemoryBlock* pBlock = m_Blocks[blockIndex];
            pBlock->Unmap(m_hAllocator, 1);
        }
    }
}

void VmaBlockVector::ApplyDefragmentationMovesGpu(
    class VmaBlockVectorDefragmentationContext* pDefragCtx,
    const VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
    VkCommandBuffer commandBuffer)
{
    const size_t blockCount = m_Blocks.size();

    pDefragCtx->blockContexts.resize(blockCount);
    memset(pDefragCtx->blockContexts.data(), 0, blockCount * sizeof(VmaBlockDefragmentationContext));

    // Go over all moves. Mark blocks that are used with BLOCK_FLAG_USED.
    const size_t moveCount = moves.size();
    for(size_t moveIndex = 0; moveIndex < moveCount; ++moveIndex)
    {
        const VmaDefragmentationMove& move = moves[moveIndex];
        pDefragCtx->blockContexts[move.srcBlockIndex].flags |= VmaBlockDefragmentationContext::BLOCK_FLAG_USED;
        pDefragCtx->blockContexts[move.dstBlockIndex].flags |= VmaBlockDefragmentationContext::BLOCK_FLAG_USED;
    }

    VMA_ASSERT(pDefragCtx->res == VK_SUCCESS);

    // Go over all blocks. Create and bind buffer for whole block if necessary.
    {
        VkBufferCreateInfo bufCreateInfo;
        VmaFillGpuDefragmentationBufferCreateInfo(bufCreateInfo);

        for(size_t blockIndex = 0; pDefragCtx->res == VK_SUCCESS && blockIndex < blockCount; ++blockIndex)
        {
            VmaBlockDefragmentationContext& currBlockCtx = pDefragCtx->blockContexts[blockIndex];
            VmaDeviceMemoryBlock* pBlock = m_Blocks[blockIndex];
            if((currBlockCtx.flags & VmaBlockDefragmentationContext::BLOCK_FLAG_USED) != 0)
            {
                bufCreateInfo.size = pBlock->m_pMetadata->GetSize();
                pDefragCtx->res = (*m_hAllocator->GetVulkanFunctions().vkCreateBuffer)(
                    m_hAllocator->m_hDevice, &bufCreateInfo, m_hAllocator->GetAllocationCallbacks(), &currBlockCtx.hBuffer);
                if(pDefragCtx->res == VK_SUCCESS)
                {
                    pDefragCtx->res = (*m_hAllocator->GetVulkanFunctions().vkBindBufferMemory)(
                        m_hAllocator->m_hDevice, currBlockCtx.hBuffer, pBlock->GetDeviceMemory(), 0);
                }
            }
        }
    }

    // Go over all moves. Post data transfer commands to command buffer.
    if(pDefragCtx->res == VK_SUCCESS)
    {
        for(size_t moveIndex = 0; moveIndex < moveCount; ++moveIndex)
        {
            const VmaDefragmentationMove& move = moves[moveIndex];

            const VmaBlockDefragmentationContext& srcBlockCtx = pDefragCtx->blockContexts[move.srcBlockIndex];
            const VmaBlockDefragmentationContext& dstBlockCtx = pDefragCtx->blockContexts[move.dstBlockIndex];

            VMA_ASSERT(srcBlockCtx.hBuffer && dstBlockCtx.hBuffer);

            VkBufferCopy region = {
                move.srcOffset,
                move.dstOffset,
                move.size };
            (*m_hAllocator->GetVulkanFunctions().vkCmdCopyBuffer)(
                commandBuffer, srcBlockCtx.hBuffer, dstBlockCtx.hBuffer, 1, &region);
        }
    }

    // Save buffers to defrag context for later destruction.
    if(pDefragCtx->res == VK_SUCCESS && moveCount > 0)
    {
        pDefragCtx->res = VK_NOT_READY;
    }
}

void VmaBlockVector::FreeEmptyBlocks(VmaDefragmentationStats* pDefragmentationStats)
{
    for(size_t blockIndex = m_Blocks.size(); blockIndex--; )
    {
        VmaDeviceMemoryBlock* pBlock = m_Blocks[blockIndex];
        if(pBlock->m_pMetadata->IsEmpty())
        {
            if(m_Blocks.size() > m_MinBlockCount)
            {
                if(pDefragmentationStats != VMA_NULL)
                {
                    ++pDefragmentationStats->deviceMemoryBlocksFreed;
                    pDefragmentationStats->bytesFreed += pBlock->m_pMetadata->GetSize();
                }

                VmaVectorRemove(m_Blocks, blockIndex);
                pBlock->Destroy(m_hAllocator);
                vma_delete(m_hAllocator, pBlock);
            }
            else
            {
                break;
            }
        }
    }
    UpdateHasEmptyBlock();
}

void VmaBlockVector::UpdateHasEmptyBlock()
{
    m_HasEmptyBlock = false;
    for(size_t index = 0, count = m_Blocks.size(); index < count; ++index)
    {
        VmaDeviceMemoryBlock* const pBlock = m_Blocks[index];
        if(pBlock->m_pMetadata->IsEmpty())
        {
            m_HasEmptyBlock = true;
            break;
        }
    }
}

#if VMA_STATS_STRING_ENABLED

void VmaBlockVector::PrintDetailedMap(class VmaJsonWriter& json)
{
    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);

    json.BeginObject();

    if(IsCustomPool())
    {
        const char* poolName = m_hParentPool->GetName();
        if(poolName != VMA_NULL && poolName[0] != '\0')
        {
            json.WriteString("Name");
            json.WriteString(poolName);
        }

        json.WriteString("MemoryTypeIndex");
        json.WriteNumber(m_MemoryTypeIndex);

        json.WriteString("BlockSize");
        json.WriteNumber(m_PreferredBlockSize);

        json.WriteString("BlockCount");
        json.BeginObject(true);
        if(m_MinBlockCount > 0)
        {
            json.WriteString("Min");
            json.WriteNumber((uint64_t)m_MinBlockCount);
        }
        if(m_MaxBlockCount < SIZE_MAX)
        {
            json.WriteString("Max");
            json.WriteNumber((uint64_t)m_MaxBlockCount);
        }
        json.WriteString("Cur");
        json.WriteNumber((uint64_t)m_Blocks.size());
        json.EndObject();

        if(m_FrameInUseCount > 0)
        {
            json.WriteString("FrameInUseCount");
            json.WriteNumber(m_FrameInUseCount);
        }

        if(m_Algorithm != 0)
        {
            json.WriteString("Algorithm");
            json.WriteString(VmaAlgorithmToStr(m_Algorithm));
        }
    }
    else
    {
        json.WriteString("PreferredBlockSize");
        json.WriteNumber(m_PreferredBlockSize);
    }

    json.WriteString("Blocks");
    json.BeginObject();
    for(size_t i = 0; i < m_Blocks.size(); ++i)
    {
        json.BeginString();
        json.ContinueString(m_Blocks[i]->GetId());
        json.EndString();

        m_Blocks[i]->m_pMetadata->PrintDetailedMap(json);
    }
    json.EndObject();

    json.EndObject();
}

#endif // #if VMA_STATS_STRING_ENABLED

void VmaBlockVector::Defragment(
    class VmaBlockVectorDefragmentationContext* pCtx,
    VmaDefragmentationStats* pStats,
    VkDeviceSize& maxCpuBytesToMove, uint32_t& maxCpuAllocationsToMove,
    VkDeviceSize& maxGpuBytesToMove, uint32_t& maxGpuAllocationsToMove,
    VkCommandBuffer commandBuffer)
{
    pCtx->res = VK_SUCCESS;
    
    const VkMemoryPropertyFlags memPropFlags =
        m_hAllocator->m_MemProps.memoryTypes[m_MemoryTypeIndex].propertyFlags;
    const bool isHostVisible = (memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    const bool canDefragmentOnCpu = maxCpuBytesToMove > 0 && maxCpuAllocationsToMove > 0 &&
        isHostVisible;
    const bool canDefragmentOnGpu = maxGpuBytesToMove > 0 && maxGpuAllocationsToMove > 0 &&
        !IsCorruptionDetectionEnabled() &&
        ((1u << m_MemoryTypeIndex) & m_hAllocator->GetGpuDefragmentationMemoryTypeBits()) != 0;

    // There are options to defragment this memory type.
    if(canDefragmentOnCpu || canDefragmentOnGpu)
    {
        bool defragmentOnGpu;
        // There is only one option to defragment this memory type.
        if(canDefragmentOnGpu != canDefragmentOnCpu)
        {
            defragmentOnGpu = canDefragmentOnGpu;
        }
        // Both options are available: Heuristics to choose the best one.
        else
        {
            defragmentOnGpu = (memPropFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 ||
                m_hAllocator->IsIntegratedGpu();
        }

        bool overlappingMoveSupported = !defragmentOnGpu;

        if(m_hAllocator->m_UseMutex)
        {
            m_Mutex.LockWrite();
            pCtx->mutexLocked = true;
        }

        pCtx->Begin(overlappingMoveSupported);

        // Defragment.

        const VkDeviceSize maxBytesToMove = defragmentOnGpu ? maxGpuBytesToMove : maxCpuBytesToMove;
        const uint32_t maxAllocationsToMove = defragmentOnGpu ? maxGpuAllocationsToMove : maxCpuAllocationsToMove;
        VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> > moves = 
            VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >(VmaStlAllocator<VmaDefragmentationMove>(m_hAllocator->GetAllocationCallbacks()));
        pCtx->res = pCtx->GetAlgorithm()->Defragment(moves, maxBytesToMove, maxAllocationsToMove);

        // Accumulate statistics.
        if(pStats != VMA_NULL)
        {
            const VkDeviceSize bytesMoved = pCtx->GetAlgorithm()->GetBytesMoved();
            const uint32_t allocationsMoved = pCtx->GetAlgorithm()->GetAllocationsMoved();
            pStats->bytesMoved += bytesMoved;
            pStats->allocationsMoved += allocationsMoved;
            VMA_ASSERT(bytesMoved <= maxBytesToMove);
            VMA_ASSERT(allocationsMoved <= maxAllocationsToMove);
            if(defragmentOnGpu)
            {
                maxGpuBytesToMove -= bytesMoved;
                maxGpuAllocationsToMove -= allocationsMoved;
            }
            else
            {
                maxCpuBytesToMove -= bytesMoved;
                maxCpuAllocationsToMove -= allocationsMoved;
            }
        }
    
        if(pCtx->res >= VK_SUCCESS)
        {
            if(defragmentOnGpu)
            {
                ApplyDefragmentationMovesGpu(pCtx, moves, commandBuffer);
            }
            else
            {
                ApplyDefragmentationMovesCpu(pCtx, moves);
            }
        }
    }
}

void VmaBlockVector::DefragmentationEnd(
    class VmaBlockVectorDefragmentationContext* pCtx,
    VmaDefragmentationStats* pStats)
{
    // Destroy buffers.
    for(size_t blockIndex = pCtx->blockContexts.size(); blockIndex--; )
    {
        VmaBlockDefragmentationContext& blockCtx = pCtx->blockContexts[blockIndex];
        if(blockCtx.hBuffer)
        {
            (*m_hAllocator->GetVulkanFunctions().vkDestroyBuffer)(
                m_hAllocator->m_hDevice, blockCtx.hBuffer, m_hAllocator->GetAllocationCallbacks());
        }
    }

    if(pCtx->res >= VK_SUCCESS)
    {
        FreeEmptyBlocks(pStats);
    }

    if(pCtx->mutexLocked)
    {
        VMA_ASSERT(m_hAllocator->m_UseMutex);
        m_Mutex.UnlockWrite();
    }
}

size_t VmaBlockVector::CalcAllocationCount() const
{
    size_t result = 0;
    for(size_t i = 0; i < m_Blocks.size(); ++i)
    {
        result += m_Blocks[i]->m_pMetadata->GetAllocationCount();
    }
    return result;
}

bool VmaBlockVector::IsBufferImageGranularityConflictPossible() const
{
    if(m_BufferImageGranularity == 1)
    {
        return false;
    }
    VmaSuballocationType lastSuballocType = VMA_SUBALLOCATION_TYPE_FREE;
    for(size_t i = 0, count = m_Blocks.size(); i < count; ++i)
    {
        VmaDeviceMemoryBlock* const pBlock = m_Blocks[i];
        VMA_ASSERT(m_Algorithm == 0);
        VmaBlockMetadata_Generic* const pMetadata = (VmaBlockMetadata_Generic*)pBlock->m_pMetadata;
        if(pMetadata->IsBufferImageGranularityConflictPossible(m_BufferImageGranularity, lastSuballocType))
        {
            return true;
        }
    }
    return false;
}

void VmaBlockVector::MakePoolAllocationsLost(
    uint32_t currentFrameIndex,
    size_t* pLostAllocationCount)
{
    VmaMutexLockWrite lock(m_Mutex, m_hAllocator->m_UseMutex);
    size_t lostAllocationCount = 0;
    for(uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        VmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        VMA_ASSERT(pBlock);
        lostAllocationCount += pBlock->m_pMetadata->MakeAllocationsLost(currentFrameIndex, m_FrameInUseCount);
    }
    if(pLostAllocationCount != VMA_NULL)
    {
        *pLostAllocationCount = lostAllocationCount;
    }
}

VkResult VmaBlockVector::CheckCorruption()
{
    if(!IsCorruptionDetectionEnabled())
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);
    for(uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        VmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VkResult res = pBlock->CheckCorruption(m_hAllocator);
        if(res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

void VmaBlockVector::AddStats(VmaStats* pStats)
{
    const uint32_t memTypeIndex = m_MemoryTypeIndex;
    const uint32_t memHeapIndex = m_hAllocator->MemoryTypeIndexToHeapIndex(memTypeIndex);

    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);

    for(uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        const VmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VMA_HEAVY_ASSERT(pBlock->Validate());
        VmaStatInfo allocationStatInfo;
        pBlock->m_pMetadata->CalcAllocationStatInfo(allocationStatInfo);
        VmaAddStatInfo(pStats->total, allocationStatInfo);
        VmaAddStatInfo(pStats->memoryType[memTypeIndex], allocationStatInfo);
        VmaAddStatInfo(pStats->memoryHeap[memHeapIndex], allocationStatInfo);
    }
}

////////////////////////////////////////////////////////////////////////////////
// VmaDefragmentationAlgorithm_Generic members definition

VmaDefragmentationAlgorithm_Generic::VmaDefragmentationAlgorithm_Generic(
    VmaAllocator hAllocator,
    VmaBlockVector* pBlockVector,
    uint32_t currentFrameIndex,
    bool overlappingMoveSupported) :
    VmaDefragmentationAlgorithm(hAllocator, pBlockVector, currentFrameIndex),
    m_AllocationCount(0),
    m_AllAllocations(false),
    m_BytesMoved(0),
    m_AllocationsMoved(0),
    m_Blocks(VmaStlAllocator<BlockInfo*>(hAllocator->GetAllocationCallbacks()))
{
    // Create block info for each block.
    const size_t blockCount = m_pBlockVector->m_Blocks.size();
    for(size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        BlockInfo* pBlockInfo = vma_new(m_hAllocator, BlockInfo)(m_hAllocator->GetAllocationCallbacks());
        pBlockInfo->m_OriginalBlockIndex = blockIndex;
        pBlockInfo->m_pBlock = m_pBlockVector->m_Blocks[blockIndex];
        m_Blocks.push_back(pBlockInfo);
    }

    // Sort them by m_pBlock pointer value.
    VMA_SORT(m_Blocks.begin(), m_Blocks.end(), BlockPointerLess());
}

VmaDefragmentationAlgorithm_Generic::~VmaDefragmentationAlgorithm_Generic()
{
    for(size_t i = m_Blocks.size(); i--; )
    {
        vma_delete(m_hAllocator, m_Blocks[i]);
    }
}

void VmaDefragmentationAlgorithm_Generic::AddAllocation(VmaAllocation hAlloc, VkBool32* pChanged)
{
    // Now as we are inside VmaBlockVector::m_Mutex, we can make final check if this allocation was not lost.
    if(hAlloc->GetLastUseFrameIndex() != VMA_FRAME_INDEX_LOST)
    {
        VmaDeviceMemoryBlock* pBlock = hAlloc->GetBlock();
        BlockInfoVector::iterator it = VmaBinaryFindFirstNotLess(m_Blocks.begin(), m_Blocks.end(), pBlock, BlockPointerLess());
        if(it != m_Blocks.end() && (*it)->m_pBlock == pBlock)
        {
            AllocationInfo allocInfo = AllocationInfo(hAlloc, pChanged);
            (*it)->m_Allocations.push_back(allocInfo);
        }
        else
        {
            VMA_ASSERT(0);
        }

        ++m_AllocationCount;
    }
}

VkResult VmaDefragmentationAlgorithm_Generic::DefragmentRound(
    VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
    VkDeviceSize maxBytesToMove,
    uint32_t maxAllocationsToMove)
{
    if(m_Blocks.empty())
    {
        return VK_SUCCESS;
    }

    // This is a choice based on research.
    // Option 1:
    uint32_t strategy = VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;
    // Option 2:
    //uint32_t strategy = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;
    // Option 3:
    //uint32_t strategy = VMA_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT;

    size_t srcBlockMinIndex = 0;
    // When FAST_ALGORITHM, move allocations from only last out of blocks that contain non-movable allocations.
    /*
    if(m_AlgorithmFlags & VMA_DEFRAGMENTATION_FAST_ALGORITHM_BIT)
    {
        const size_t blocksWithNonMovableCount = CalcBlocksWithNonMovableCount();
        if(blocksWithNonMovableCount > 0)
        {
            srcBlockMinIndex = blocksWithNonMovableCount - 1;
        }
    }
    */

    size_t srcBlockIndex = m_Blocks.size() - 1;
    size_t srcAllocIndex = SIZE_MAX;
    for(;;)
    {
        // 1. Find next allocation to move.
        // 1.1. Start from last to first m_Blocks - they are sorted from most "destination" to most "source".
        // 1.2. Then start from last to first m_Allocations.
        while(srcAllocIndex >= m_Blocks[srcBlockIndex]->m_Allocations.size())
        {
            if(m_Blocks[srcBlockIndex]->m_Allocations.empty())
            {
                // Finished: no more allocations to process.
                if(srcBlockIndex == srcBlockMinIndex)
                {
                    return VK_SUCCESS;
                }
                else
                {
                    --srcBlockIndex;
                    srcAllocIndex = SIZE_MAX;
                }
            }
            else
            {
                srcAllocIndex = m_Blocks[srcBlockIndex]->m_Allocations.size() - 1;
            }
        }
        
        BlockInfo* pSrcBlockInfo = m_Blocks[srcBlockIndex];
        AllocationInfo& allocInfo = pSrcBlockInfo->m_Allocations[srcAllocIndex];

        const VkDeviceSize size = allocInfo.m_hAllocation->GetSize();
        const VkDeviceSize srcOffset = allocInfo.m_hAllocation->GetOffset();
        const VkDeviceSize alignment = allocInfo.m_hAllocation->GetAlignment();
        const VmaSuballocationType suballocType = allocInfo.m_hAllocation->GetSuballocationType();

        // 2. Try to find new place for this allocation in preceding or current block.
        for(size_t dstBlockIndex = 0; dstBlockIndex <= srcBlockIndex; ++dstBlockIndex)
        {
            BlockInfo* pDstBlockInfo = m_Blocks[dstBlockIndex];
            VmaAllocationRequest dstAllocRequest;
            if(pDstBlockInfo->m_pBlock->m_pMetadata->CreateAllocationRequest(
                m_CurrentFrameIndex,
                m_pBlockVector->GetFrameInUseCount(),
                m_pBlockVector->GetBufferImageGranularity(),
                size,
                alignment,
                false, // upperAddress
                suballocType,
                false, // canMakeOtherLost
                strategy,
                &dstAllocRequest) &&
            MoveMakesSense(
                dstBlockIndex, dstAllocRequest.offset, srcBlockIndex, srcOffset))
            {
                VMA_ASSERT(dstAllocRequest.itemsToMakeLostCount == 0);

                // Reached limit on number of allocations or bytes to move.
                if((m_AllocationsMoved + 1 > maxAllocationsToMove) ||
                    (m_BytesMoved + size > maxBytesToMove))
                {
                    return VK_SUCCESS;
                }

                VmaDefragmentationMove move;
                move.srcBlockIndex = pSrcBlockInfo->m_OriginalBlockIndex;
                move.dstBlockIndex = pDstBlockInfo->m_OriginalBlockIndex;
                move.srcOffset = srcOffset;
                move.dstOffset = dstAllocRequest.offset;
                move.size = size;
                moves.push_back(move);

                pDstBlockInfo->m_pBlock->m_pMetadata->Alloc(
                    dstAllocRequest,
                    suballocType,
                    size,
                    allocInfo.m_hAllocation);
                pSrcBlockInfo->m_pBlock->m_pMetadata->FreeAtOffset(srcOffset);
                
                allocInfo.m_hAllocation->ChangeBlockAllocation(m_hAllocator, pDstBlockInfo->m_pBlock, dstAllocRequest.offset);

                if(allocInfo.m_pChanged != VMA_NULL)
                {
                    *allocInfo.m_pChanged = VK_TRUE;
                }

                ++m_AllocationsMoved;
                m_BytesMoved += size;

                VmaVectorRemove(pSrcBlockInfo->m_Allocations, srcAllocIndex);

                break;
            }
        }

        // If not processed, this allocInfo remains in pBlockInfo->m_Allocations for next round.

        if(srcAllocIndex > 0)
        {
            --srcAllocIndex;
        }
        else
        {
            if(srcBlockIndex > 0)
            {
                --srcBlockIndex;
                srcAllocIndex = SIZE_MAX;
            }
            else
            {
                return VK_SUCCESS;
            }
        }
    }
}

size_t VmaDefragmentationAlgorithm_Generic::CalcBlocksWithNonMovableCount() const
{
    size_t result = 0;
    for(size_t i = 0; i < m_Blocks.size(); ++i)
    {
        if(m_Blocks[i]->m_HasNonMovableAllocations)
        {
            ++result;
        }
    }
    return result;
}

VkResult VmaDefragmentationAlgorithm_Generic::Defragment(
    VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
    VkDeviceSize maxBytesToMove,
    uint32_t maxAllocationsToMove)
{
    if(!m_AllAllocations && m_AllocationCount == 0)
    {
        return VK_SUCCESS;
    }

    const size_t blockCount = m_Blocks.size();
    for(size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        BlockInfo* pBlockInfo = m_Blocks[blockIndex];

        if(m_AllAllocations)
        {
            VmaBlockMetadata_Generic* pMetadata = (VmaBlockMetadata_Generic*)pBlockInfo->m_pBlock->m_pMetadata;
            for(VmaSuballocationList::const_iterator it = pMetadata->m_Suballocations.begin();
                it != pMetadata->m_Suballocations.end();
                ++it)
            {
                if(it->type != VMA_SUBALLOCATION_TYPE_FREE)
                {
                    AllocationInfo allocInfo = AllocationInfo(it->hAllocation, VMA_NULL);
                    pBlockInfo->m_Allocations.push_back(allocInfo);
                }
            }
        }

        pBlockInfo->CalcHasNonMovableAllocations();
        
        // This is a choice based on research.
        // Option 1:
        pBlockInfo->SortAllocationsByOffsetDescending();
        // Option 2:
        //pBlockInfo->SortAllocationsBySizeDescending();
    }

    // Sort m_Blocks this time by the main criterium, from most "destination" to most "source" blocks.
    VMA_SORT(m_Blocks.begin(), m_Blocks.end(), BlockInfoCompareMoveDestination());

    // This is a choice based on research.
    const uint32_t roundCount = 2;

    // Execute defragmentation rounds (the main part).
    VkResult result = VK_SUCCESS;
    for(uint32_t round = 0; (round < roundCount) && (result == VK_SUCCESS); ++round)
    {
        result = DefragmentRound(moves, maxBytesToMove, maxAllocationsToMove);
    }

    return result;
}

bool VmaDefragmentationAlgorithm_Generic::MoveMakesSense(
        size_t dstBlockIndex, VkDeviceSize dstOffset,
        size_t srcBlockIndex, VkDeviceSize srcOffset)
{
    if(dstBlockIndex < srcBlockIndex)
    {
        return true;
    }
    if(dstBlockIndex > srcBlockIndex)
    {
        return false;
    }
    if(dstOffset < srcOffset)
    {
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////
// VmaDefragmentationAlgorithm_Fast

VmaDefragmentationAlgorithm_Fast::VmaDefragmentationAlgorithm_Fast(
    VmaAllocator hAllocator,
    VmaBlockVector* pBlockVector,
    uint32_t currentFrameIndex,
    bool overlappingMoveSupported) :
    VmaDefragmentationAlgorithm(hAllocator, pBlockVector, currentFrameIndex),
    m_OverlappingMoveSupported(overlappingMoveSupported),
    m_AllocationCount(0),
    m_AllAllocations(false),
    m_BytesMoved(0),
    m_AllocationsMoved(0),
    m_BlockInfos(VmaStlAllocator<BlockInfo>(hAllocator->GetAllocationCallbacks()))
{
    VMA_ASSERT(VMA_DEBUG_MARGIN == 0);

}

VmaDefragmentationAlgorithm_Fast::~VmaDefragmentationAlgorithm_Fast()
{
}

VkResult VmaDefragmentationAlgorithm_Fast::Defragment(
    VmaVector< VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove> >& moves,
    VkDeviceSize maxBytesToMove,
    uint32_t maxAllocationsToMove)
{
    VMA_ASSERT(m_AllAllocations || m_pBlockVector->CalcAllocationCount() == m_AllocationCount);

    const size_t blockCount = m_pBlockVector->GetBlockCount();
    if(blockCount == 0 || maxBytesToMove == 0 || maxAllocationsToMove == 0)
    {
        return VK_SUCCESS;
    }

    PreprocessMetadata();

    // Sort blocks in order from most destination.

    m_BlockInfos.resize(blockCount);
    for(size_t i = 0; i < blockCount; ++i)
    {
        m_BlockInfos[i].origBlockIndex = i;
    }

    VMA_SORT(m_BlockInfos.begin(), m_BlockInfos.end(), [this](const BlockInfo& lhs, const BlockInfo& rhs) -> bool {
        return m_pBlockVector->GetBlock(lhs.origBlockIndex)->m_pMetadata->GetSumFreeSize() <
            m_pBlockVector->GetBlock(rhs.origBlockIndex)->m_pMetadata->GetSumFreeSize();
    });

    // THE MAIN ALGORITHM

    FreeSpaceDatabase freeSpaceDb;

    size_t dstBlockInfoIndex = 0;
    size_t dstOrigBlockIndex = m_BlockInfos[dstBlockInfoIndex].origBlockIndex;
    VmaDeviceMemoryBlock* pDstBlock = m_pBlockVector->GetBlock(dstOrigBlockIndex);
    VmaBlockMetadata_Generic* pDstMetadata = (VmaBlockMetadata_Generic*)pDstBlock->m_pMetadata;
    VkDeviceSize dstBlockSize = pDstMetadata->GetSize();
    VkDeviceSize dstOffset = 0;

    bool end = false;
    for(size_t srcBlockInfoIndex = 0; !end && srcBlockInfoIndex < blockCount; ++srcBlockInfoIndex)
    {
        const size_t srcOrigBlockIndex = m_BlockInfos[srcBlockInfoIndex].origBlockIndex;
        VmaDeviceMemoryBlock* const pSrcBlock = m_pBlockVector->GetBlock(srcOrigBlockIndex);
        VmaBlockMetadata_Generic* const pSrcMetadata = (VmaBlockMetadata_Generic*)pSrcBlock->m_pMetadata;
        for(VmaSuballocationList::iterator srcSuballocIt = pSrcMetadata->m_Suballocations.begin();
            !end && srcSuballocIt != pSrcMetadata->m_Suballocations.end(); )
        {
            VmaAllocation_T* const pAlloc = srcSuballocIt->hAllocation;
            const VkDeviceSize srcAllocAlignment = pAlloc->GetAlignment();
            const VkDeviceSize srcAllocSize = srcSuballocIt->size;
            if(m_AllocationsMoved == maxAllocationsToMove ||
                m_BytesMoved + srcAllocSize > maxBytesToMove)
            {
                end = true;
                break;
            }
            const VkDeviceSize srcAllocOffset = srcSuballocIt->offset;

            // Try to place it in one of free spaces from the database.
            size_t freeSpaceInfoIndex;
            VkDeviceSize dstAllocOffset;
            if(freeSpaceDb.Fetch(srcAllocAlignment, srcAllocSize,
                freeSpaceInfoIndex, dstAllocOffset))
            {
                size_t freeSpaceOrigBlockIndex = m_BlockInfos[freeSpaceInfoIndex].origBlockIndex;
                VmaDeviceMemoryBlock* pFreeSpaceBlock = m_pBlockVector->GetBlock(freeSpaceOrigBlockIndex);
                VmaBlockMetadata_Generic* pFreeSpaceMetadata = (VmaBlockMetadata_Generic*)pFreeSpaceBlock->m_pMetadata;

                // Same block
                if(freeSpaceInfoIndex == srcBlockInfoIndex)
                {
                    VMA_ASSERT(dstAllocOffset <= srcAllocOffset);

                    // MOVE OPTION 1: Move the allocation inside the same block by decreasing offset.

                    VmaSuballocation suballoc = *srcSuballocIt;
                    suballoc.offset = dstAllocOffset;
                    suballoc.hAllocation->ChangeOffset(dstAllocOffset);
                    m_BytesMoved += srcAllocSize;
                    ++m_AllocationsMoved;
                    
                    VmaSuballocationList::iterator nextSuballocIt = srcSuballocIt;
                    ++nextSuballocIt;
                    pSrcMetadata->m_Suballocations.erase(srcSuballocIt);
                    srcSuballocIt = nextSuballocIt;

                    InsertSuballoc(pFreeSpaceMetadata, suballoc);

                    VmaDefragmentationMove move = {
                        srcOrigBlockIndex, freeSpaceOrigBlockIndex,
                        srcAllocOffset, dstAllocOffset,
                        srcAllocSize };
                    moves.push_back(move);
                }
                // Different block
                else
                {
                    // MOVE OPTION 2: Move the allocation to a different block.

                    VMA_ASSERT(freeSpaceInfoIndex < srcBlockInfoIndex);

                    VmaSuballocation suballoc = *srcSuballocIt;
                    suballoc.offset = dstAllocOffset;
                    suballoc.hAllocation->ChangeBlockAllocation(m_hAllocator, pFreeSpaceBlock, dstAllocOffset);
                    m_BytesMoved += srcAllocSize;
                    ++m_AllocationsMoved;

                    VmaSuballocationList::iterator nextSuballocIt = srcSuballocIt;
                    ++nextSuballocIt;
                    pSrcMetadata->m_Suballocations.erase(srcSuballocIt);
                    srcSuballocIt = nextSuballocIt;

                    InsertSuballoc(pFreeSpaceMetadata, suballoc);

                    VmaDefragmentationMove move = {
                        srcOrigBlockIndex, freeSpaceOrigBlockIndex,
                        srcAllocOffset, dstAllocOffset,
                        srcAllocSize };
                    moves.push_back(move);
                }
            }
            else
            {
                dstAllocOffset = VmaAlignUp(dstOffset, srcAllocAlignment);

                // If the allocation doesn't fit before the end of dstBlock, forward to next block.
                while(dstBlockInfoIndex < srcBlockInfoIndex &&
                    dstAllocOffset + srcAllocSize > dstBlockSize)
                {
                    // But before that, register remaining free space at the end of dst block.
                    freeSpaceDb.Register(dstBlockInfoIndex, dstOffset, dstBlockSize - dstOffset);

                    ++dstBlockInfoIndex;
                    dstOrigBlockIndex = m_BlockInfos[dstBlockInfoIndex].origBlockIndex;
                    pDstBlock = m_pBlockVector->GetBlock(dstOrigBlockIndex);
                    pDstMetadata = (VmaBlockMetadata_Generic*)pDstBlock->m_pMetadata;
                    dstBlockSize = pDstMetadata->GetSize();
                    dstOffset = 0;
                    dstAllocOffset = 0;
                }

                // Same block
                if(dstBlockInfoIndex == srcBlockInfoIndex)
                {
                    VMA_ASSERT(dstAllocOffset <= srcAllocOffset);

                    const bool overlap = dstAllocOffset + srcAllocSize > srcAllocOffset;

                    bool skipOver = overlap;
                    if(overlap && m_OverlappingMoveSupported && dstAllocOffset < srcAllocOffset)
                    {
                        // If destination and source place overlap, skip if it would move it
                        // by only < 1/64 of its size.
                        skipOver = (srcAllocOffset - dstAllocOffset) * 64 < srcAllocSize;
                    }

                    if(skipOver)
                    {
                        freeSpaceDb.Register(dstBlockInfoIndex, dstOffset, srcAllocOffset - dstOffset);

                        dstOffset = srcAllocOffset + srcAllocSize;
                        ++srcSuballocIt;
                    }
                    // MOVE OPTION 1: Move the allocation inside the same block by decreasing offset.
                    else
                    {
                        srcSuballocIt->offset = dstAllocOffset;
                        srcSuballocIt->hAllocation->ChangeOffset(dstAllocOffset);
                        dstOffset = dstAllocOffset + srcAllocSize;
                        m_BytesMoved += srcAllocSize;
                        ++m_AllocationsMoved;
                        ++srcSuballocIt;
                        VmaDefragmentationMove move = {
                            srcOrigBlockIndex, dstOrigBlockIndex,
                            srcAllocOffset, dstAllocOffset,
                            srcAllocSize };
                        moves.push_back(move);
                    }
                }
                // Different block
                else
                {
                    // MOVE OPTION 2: Move the allocation to a different block.

                    VMA_ASSERT(dstBlockInfoIndex < srcBlockInfoIndex);
                    VMA_ASSERT(dstAllocOffset + srcAllocSize <= dstBlockSize);

                    VmaSuballocation suballoc = *srcSuballocIt;
                    suballoc.offset = dstAllocOffset;
                    suballoc.hAllocation->ChangeBlockAllocation(m_hAllocator, pDstBlock, dstAllocOffset);
                    dstOffset = dstAllocOffset + srcAllocSize;
                    m_BytesMoved += srcAllocSize;
                    ++m_AllocationsMoved;

                    VmaSuballocationList::iterator nextSuballocIt = srcSuballocIt;
                    ++nextSuballocIt;
                    pSrcMetadata->m_Suballocations.erase(srcSuballocIt);
                    srcSuballocIt = nextSuballocIt;

                    pDstMetadata->m_Suballocations.push_back(suballoc);

                    VmaDefragmentationMove move = {
                        srcOrigBlockIndex, dstOrigBlockIndex,
                        srcAllocOffset, dstAllocOffset,
                        srcAllocSize };
                    moves.push_back(move);
                }
            }
        }
    }

    m_BlockInfos.clear();
    
    PostprocessMetadata();

    return VK_SUCCESS;
}

void VmaDefragmentationAlgorithm_Fast::PreprocessMetadata()
{
    const size_t blockCount = m_pBlockVector->GetBlockCount();
    for(size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        VmaBlockMetadata_Generic* const pMetadata =
            (VmaBlockMetadata_Generic*)m_pBlockVector->GetBlock(blockIndex)->m_pMetadata;
        pMetadata->m_FreeCount = 0;
        pMetadata->m_SumFreeSize = pMetadata->GetSize();
        pMetadata->m_FreeSuballocationsBySize.clear();
        for(VmaSuballocationList::iterator it = pMetadata->m_Suballocations.begin();
            it != pMetadata->m_Suballocations.end(); )
        {
            if(it->type == VMA_SUBALLOCATION_TYPE_FREE)
            {
                VmaSuballocationList::iterator nextIt = it;
                ++nextIt;
                pMetadata->m_Suballocations.erase(it);
                it = nextIt;
            }
            else
            {
                ++it;
            }
        }
    }
}

void VmaDefragmentationAlgorithm_Fast::PostprocessMetadata()
{
    const size_t blockCount = m_pBlockVector->GetBlockCount();
    for(size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        VmaBlockMetadata_Generic* const pMetadata =
            (VmaBlockMetadata_Generic*)m_pBlockVector->GetBlock(blockIndex)->m_pMetadata;
        const VkDeviceSize blockSize = pMetadata->GetSize();
        
        // No allocations in this block - entire area is free.
        if(pMetadata->m_Suballocations.empty())
        {
            pMetadata->m_FreeCount = 1;
            //pMetadata->m_SumFreeSize is already set to blockSize.
            VmaSuballocation suballoc = {
                0, // offset
                blockSize, // size
                VMA_NULL, // hAllocation
                VMA_SUBALLOCATION_TYPE_FREE };
            pMetadata->m_Suballocations.push_back(suballoc);
            pMetadata->RegisterFreeSuballocation(pMetadata->m_Suballocations.begin());
        }
        // There are some allocations in this block.
        else
        {
            VkDeviceSize offset = 0;
            VmaSuballocationList::iterator it;
            for(it = pMetadata->m_Suballocations.begin();
                it != pMetadata->m_Suballocations.end();
                ++it)
            {
                VMA_ASSERT(it->type != VMA_SUBALLOCATION_TYPE_FREE);
                VMA_ASSERT(it->offset >= offset);

                // Need to insert preceding free space.
                if(it->offset > offset)
                {
                    ++pMetadata->m_FreeCount;
                    const VkDeviceSize freeSize = it->offset - offset;
                    VmaSuballocation suballoc = {
                        offset, // offset
                        freeSize, // size
                        VMA_NULL, // hAllocation
                        VMA_SUBALLOCATION_TYPE_FREE };
                    VmaSuballocationList::iterator precedingFreeIt = pMetadata->m_Suballocations.insert(it, suballoc);
                    if(freeSize >= VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
                    {
                        pMetadata->m_FreeSuballocationsBySize.push_back(precedingFreeIt);
                    }
                }

                pMetadata->m_SumFreeSize -= it->size;
                offset = it->offset + it->size;
            }

            // Need to insert trailing free space.
            if(offset < blockSize)
            {
                ++pMetadata->m_FreeCount;
                const VkDeviceSize freeSize = blockSize - offset;
                VmaSuballocation suballoc = {
                    offset, // offset
                    freeSize, // size
                    VMA_NULL, // hAllocation
                    VMA_SUBALLOCATION_TYPE_FREE };
                VMA_ASSERT(it == pMetadata->m_Suballocations.end());
                VmaSuballocationList::iterator trailingFreeIt = pMetadata->m_Suballocations.insert(it, suballoc);
                if(freeSize > VMA_MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER)
                {
                    pMetadata->m_FreeSuballocationsBySize.push_back(trailingFreeIt);
                }
            }

            VMA_SORT(
                pMetadata->m_FreeSuballocationsBySize.begin(),
                pMetadata->m_FreeSuballocationsBySize.end(),
                VmaSuballocationItemSizeLess());
        }

        VMA_HEAVY_ASSERT(pMetadata->Validate());
    }
}

void VmaDefragmentationAlgorithm_Fast::InsertSuballoc(VmaBlockMetadata_Generic* pMetadata, const VmaSuballocation& suballoc)
{
    // TODO: Optimize somehow. Remember iterator instead of searching for it linearly.
    VmaSuballocationList::iterator it = pMetadata->m_Suballocations.begin();
    while(it != pMetadata->m_Suballocations.end())
    {
        if(it->offset < suballoc.offset)
        {
            ++it;
        }
    }
    pMetadata->m_Suballocations.insert(it, suballoc);
}

////////////////////////////////////////////////////////////////////////////////
// VmaBlockVectorDefragmentationContext

VmaBlockVectorDefragmentationContext::VmaBlockVectorDefragmentationContext(
    VmaAllocator hAllocator,
    VmaPool hCustomPool,
    VmaBlockVector* pBlockVector,
    uint32_t currFrameIndex) :
    res(VK_SUCCESS),
    mutexLocked(false),
    blockContexts(VmaStlAllocator<VmaBlockDefragmentationContext>(hAllocator->GetAllocationCallbacks())),
    m_hAllocator(hAllocator),
    m_hCustomPool(hCustomPool),
    m_pBlockVector(pBlockVector),
    m_CurrFrameIndex(currFrameIndex),
    m_pAlgorithm(VMA_NULL),
    m_Allocations(VmaStlAllocator<AllocInfo>(hAllocator->GetAllocationCallbacks())),
    m_AllAllocations(false)
{
}

VmaBlockVectorDefragmentationContext::~VmaBlockVectorDefragmentationContext()
{
    vma_delete(m_hAllocator, m_pAlgorithm);
}

void VmaBlockVectorDefragmentationContext::AddAllocation(VmaAllocation hAlloc, VkBool32* pChanged)
{
    AllocInfo info = { hAlloc, pChanged };
    m_Allocations.push_back(info);
}

void VmaBlockVectorDefragmentationContext::Begin(bool overlappingMoveSupported)
{
    const bool allAllocations = m_AllAllocations ||
        m_Allocations.size() == m_pBlockVector->CalcAllocationCount();

    /********************************
    HERE IS THE CHOICE OF DEFRAGMENTATION ALGORITHM.
    ********************************/

    /*
    Fast algorithm is supported only when certain criteria are met:
    - VMA_DEBUG_MARGIN is 0.
    - All allocations in this block vector are moveable.
    - There is no possibility of image/buffer granularity conflict.
    */
    if(VMA_DEBUG_MARGIN == 0 &&
        allAllocations &&
        !m_pBlockVector->IsBufferImageGranularityConflictPossible())
    {
        m_pAlgorithm = vma_new(m_hAllocator, VmaDefragmentationAlgorithm_Fast)(
            m_hAllocator, m_pBlockVector, m_CurrFrameIndex, overlappingMoveSupported);
    }
    else
    {
        m_pAlgorithm = vma_new(m_hAllocator, VmaDefragmentationAlgorithm_Generic)(
            m_hAllocator, m_pBlockVector, m_CurrFrameIndex, overlappingMoveSupported);
    }

    if(allAllocations)
    {
        m_pAlgorithm->AddAll();
    }
    else
    {
        for(size_t i = 0, count = m_Allocations.size(); i < count; ++i)
        {
            m_pAlgorithm->AddAllocation(m_Allocations[i].hAlloc, m_Allocations[i].pChanged);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// VmaDefragmentationContext

VmaDefragmentationContext_T::VmaDefragmentationContext_T(
    VmaAllocator hAllocator,
    uint32_t currFrameIndex,
    uint32_t flags,
    VmaDefragmentationStats* pStats) :
    m_hAllocator(hAllocator),
    m_CurrFrameIndex(currFrameIndex),
    m_Flags(flags),
    m_pStats(pStats),
    m_CustomPoolContexts(VmaStlAllocator<VmaBlockVectorDefragmentationContext*>(hAllocator->GetAllocationCallbacks()))
{
    memset(m_DefaultPoolContexts, 0, sizeof(m_DefaultPoolContexts));
}

VmaDefragmentationContext_T::~VmaDefragmentationContext_T()
{
    for(size_t i = m_CustomPoolContexts.size(); i--; )
    {
        VmaBlockVectorDefragmentationContext* pBlockVectorCtx = m_CustomPoolContexts[i];
        pBlockVectorCtx->GetBlockVector()->DefragmentationEnd(pBlockVectorCtx, m_pStats);
        vma_delete(m_hAllocator, pBlockVectorCtx);
    }
    for(size_t i = m_hAllocator->m_MemProps.memoryTypeCount; i--; )
    {
        VmaBlockVectorDefragmentationContext* pBlockVectorCtx = m_DefaultPoolContexts[i];
        if(pBlockVectorCtx)
        {
            pBlockVectorCtx->GetBlockVector()->DefragmentationEnd(pBlockVectorCtx, m_pStats);
            vma_delete(m_hAllocator, pBlockVectorCtx);
        }
    }
}

void VmaDefragmentationContext_T::AddPools(uint32_t poolCount, VmaPool* pPools)
{
    for(uint32_t poolIndex = 0; poolIndex < poolCount; ++poolIndex)
    {
        VmaPool pool = pPools[poolIndex];
        VMA_ASSERT(pool);
        // Pools with algorithm other than default are not defragmented.
        if(pool->m_BlockVector.GetAlgorithm() == 0)
        {
            VmaBlockVectorDefragmentationContext* pBlockVectorDefragCtx = VMA_NULL;
            
            for(size_t i = m_CustomPoolContexts.size(); i--; )
            {
                if(m_CustomPoolContexts[i]->GetCustomPool() == pool)
                {
                    pBlockVectorDefragCtx = m_CustomPoolContexts[i];
                    break;
                }
            }
            
            if(!pBlockVectorDefragCtx)
            {
                pBlockVectorDefragCtx = vma_new(m_hAllocator, VmaBlockVectorDefragmentationContext)(
                    m_hAllocator,
                    pool,
                    &pool->m_BlockVector,
                    m_CurrFrameIndex);
                m_CustomPoolContexts.push_back(pBlockVectorDefragCtx);
            }

            pBlockVectorDefragCtx->AddAll();
        }
    }
}

void VmaDefragmentationContext_T::AddAllocations(
    uint32_t allocationCount,
    VmaAllocation* pAllocations,
    VkBool32* pAllocationsChanged)
{
    // Dispatch pAllocations among defragmentators. Create them when necessary.
    for(uint32_t allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
    {
        const VmaAllocation hAlloc = pAllocations[allocIndex];
        VMA_ASSERT(hAlloc);
        // DedicatedAlloc cannot be defragmented.
        if((hAlloc->GetType() == VmaAllocation_T::ALLOCATION_TYPE_BLOCK) &&
            // Lost allocation cannot be defragmented.
            (hAlloc->GetLastUseFrameIndex() != VMA_FRAME_INDEX_LOST))
        {
            VmaBlockVectorDefragmentationContext* pBlockVectorDefragCtx = VMA_NULL;

            const VmaPool hAllocPool = hAlloc->GetBlock()->GetParentPool();
            // This allocation belongs to custom pool.
            if(hAllocPool != VK_NULL_HANDLE)
            {
                // Pools with algorithm other than default are not defragmented.
                if(hAllocPool->m_BlockVector.GetAlgorithm() == 0)
                {
                    for(size_t i = m_CustomPoolContexts.size(); i--; )
                    {
                        if(m_CustomPoolContexts[i]->GetCustomPool() == hAllocPool)
                        {
                            pBlockVectorDefragCtx = m_CustomPoolContexts[i];
                            break;
                        }
                    }
                    if(!pBlockVectorDefragCtx)
                    {
                        pBlockVectorDefragCtx = vma_new(m_hAllocator, VmaBlockVectorDefragmentationContext)(
                            m_hAllocator,
                            hAllocPool,
                            &hAllocPool->m_BlockVector,
                            m_CurrFrameIndex);
                        m_CustomPoolContexts.push_back(pBlockVectorDefragCtx);
                    }
                }
            }
            // This allocation belongs to default pool.
            else
            {
                const uint32_t memTypeIndex = hAlloc->GetMemoryTypeIndex();
                pBlockVectorDefragCtx = m_DefaultPoolContexts[memTypeIndex];
                if(!pBlockVectorDefragCtx)
                {
                    pBlockVectorDefragCtx = vma_new(m_hAllocator, VmaBlockVectorDefragmentationContext)(
                        m_hAllocator,
                        VMA_NULL, // hCustomPool
                        m_hAllocator->m_pBlockVectors[memTypeIndex],
                        m_CurrFrameIndex);
                    m_DefaultPoolContexts[memTypeIndex] = pBlockVectorDefragCtx;
                }
            }

            if(pBlockVectorDefragCtx)
            {
                VkBool32* const pChanged = (pAllocationsChanged != VMA_NULL) ?
                    &pAllocationsChanged[allocIndex] : VMA_NULL;
                pBlockVectorDefragCtx->AddAllocation(hAlloc, pChanged);
            }
        }
    }
}

VkResult VmaDefragmentationContext_T::Defragment(
    VkDeviceSize maxCpuBytesToMove, uint32_t maxCpuAllocationsToMove,
    VkDeviceSize maxGpuBytesToMove, uint32_t maxGpuAllocationsToMove,
    VkCommandBuffer commandBuffer, VmaDefragmentationStats* pStats)
{
    if(pStats)
    {
        memset(pStats, 0, sizeof(VmaDefragmentationStats));
    }

    if(commandBuffer == VK_NULL_HANDLE)
    {
        maxGpuBytesToMove = 0;
        maxGpuAllocationsToMove = 0;
    }

    VkResult res = VK_SUCCESS;

    // Process default pools.
    for(uint32_t memTypeIndex = 0;
        memTypeIndex < m_hAllocator->GetMemoryTypeCount() && res >= VK_SUCCESS;
        ++memTypeIndex)
    {
        VmaBlockVectorDefragmentationContext* pBlockVectorCtx = m_DefaultPoolContexts[memTypeIndex];
        if(pBlockVectorCtx)
        {
            VMA_ASSERT(pBlockVectorCtx->GetBlockVector());
            pBlockVectorCtx->GetBlockVector()->Defragment(
                pBlockVectorCtx,
                pStats,
                maxCpuBytesToMove, maxCpuAllocationsToMove,
                maxGpuBytesToMove, maxGpuAllocationsToMove,
                commandBuffer);
            if(pBlockVectorCtx->res != VK_SUCCESS)
            {
                res = pBlockVectorCtx->res;
            }
        }
    }

    // Process custom pools.
    for(size_t customCtxIndex = 0, customCtxCount = m_CustomPoolContexts.size();
        customCtxIndex < customCtxCount && res >= VK_SUCCESS;
        ++customCtxIndex)
    {
        VmaBlockVectorDefragmentationContext* pBlockVectorCtx = m_CustomPoolContexts[customCtxIndex];
        VMA_ASSERT(pBlockVectorCtx && pBlockVectorCtx->GetBlockVector());
        pBlockVectorCtx->GetBlockVector()->Defragment(
            pBlockVectorCtx,
            pStats,
            maxCpuBytesToMove, maxCpuAllocationsToMove,
            maxGpuBytesToMove, maxGpuAllocationsToMove,
            commandBuffer);
        if(pBlockVectorCtx->res != VK_SUCCESS)
        {
            res = pBlockVectorCtx->res;
        }
    }

    return res;
}

////////////////////////////////////////////////////////////////////////////////
// VmaRecorder

#if VMA_RECORDING_ENABLED

VmaRecorder::VmaRecorder() :
    m_UseMutex(true),
    m_Flags(0),
    m_File(VMA_NULL),
    m_Freq(INT64_MAX),
    m_StartCounter(INT64_MAX)
{
}

VkResult VmaRecorder::Init(const VmaRecordSettings& settings, bool useMutex)
{
    m_UseMutex = useMutex;
    m_Flags = settings.flags;

    QueryPerformanceFrequency((LARGE_INTEGER*)&m_Freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&m_StartCounter);

    // Open file for writing.
    errno_t err = fopen_s(&m_File, settings.pFilePath, "wb");
    if(err != 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Write header.
    fprintf(m_File, "%s\n", "Vulkan Memory Allocator,Calls recording");
    fprintf(m_File, "%s\n", "1,8");

    return VK_SUCCESS;
}

VmaRecorder::~VmaRecorder()
{
    if(m_File != VMA_NULL)
    {
        fclose(m_File);
    }
}

void VmaRecorder::RecordCreateAllocator(uint32_t frameIndex)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaCreateAllocator\n", callParams.threadId, callParams.time, frameIndex);
    Flush();
}

void VmaRecorder::RecordDestroyAllocator(uint32_t frameIndex)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaDestroyAllocator\n", callParams.threadId, callParams.time, frameIndex);
    Flush();
}

void VmaRecorder::RecordCreatePool(uint32_t frameIndex, const VmaPoolCreateInfo& createInfo, VmaPool pool)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaCreatePool,%u,%u,%llu,%llu,%llu,%u,%p\n", callParams.threadId, callParams.time, frameIndex,
        createInfo.memoryTypeIndex,
        createInfo.flags,
        createInfo.blockSize,
        (uint64_t)createInfo.minBlockCount,
        (uint64_t)createInfo.maxBlockCount,
        createInfo.frameInUseCount,
        pool);
    Flush();
}

void VmaRecorder::RecordDestroyPool(uint32_t frameIndex, VmaPool pool)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaDestroyPool,%p\n", callParams.threadId, callParams.time, frameIndex,
        pool);
    Flush();
}

void VmaRecorder::RecordAllocateMemory(uint32_t frameIndex,
        const VkMemoryRequirements& vkMemReq,
        const VmaAllocationCreateInfo& createInfo,
        VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    UserDataString userDataStr(createInfo.flags, createInfo.pUserData);
    fprintf(m_File, "%u,%.3f,%u,vmaAllocateMemory,%llu,%llu,%u,%u,%u,%u,%u,%u,%p,%p,%s\n", callParams.threadId, callParams.time, frameIndex,
        vkMemReq.size,
        vkMemReq.alignment,
        vkMemReq.memoryTypeBits,
        createInfo.flags,
        createInfo.usage,
        createInfo.requiredFlags,
        createInfo.preferredFlags,
        createInfo.memoryTypeBits,
        createInfo.pool,
        allocation,
        userDataStr.GetString());
    Flush();
}

void VmaRecorder::RecordAllocateMemoryPages(uint32_t frameIndex,
    const VkMemoryRequirements& vkMemReq,
    const VmaAllocationCreateInfo& createInfo,
    uint64_t allocationCount,
    const VmaAllocation* pAllocations)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    UserDataString userDataStr(createInfo.flags, createInfo.pUserData);
    fprintf(m_File, "%u,%.3f,%u,vmaAllocateMemoryPages,%llu,%llu,%u,%u,%u,%u,%u,%u,%p,", callParams.threadId, callParams.time, frameIndex,
        vkMemReq.size,
        vkMemReq.alignment,
        vkMemReq.memoryTypeBits,
        createInfo.flags,
        createInfo.usage,
        createInfo.requiredFlags,
        createInfo.preferredFlags,
        createInfo.memoryTypeBits,
        createInfo.pool);
    PrintPointerList(allocationCount, pAllocations);
    fprintf(m_File, ",%s\n", userDataStr.GetString());
    Flush();
}

void VmaRecorder::RecordAllocateMemoryForBuffer(uint32_t frameIndex,
    const VkMemoryRequirements& vkMemReq,
    bool requiresDedicatedAllocation,
    bool prefersDedicatedAllocation,
    const VmaAllocationCreateInfo& createInfo,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    UserDataString userDataStr(createInfo.flags, createInfo.pUserData);
    fprintf(m_File, "%u,%.3f,%u,vmaAllocateMemoryForBuffer,%llu,%llu,%u,%u,%u,%u,%u,%u,%u,%u,%p,%p,%s\n", callParams.threadId, callParams.time, frameIndex,
        vkMemReq.size,
        vkMemReq.alignment,
        vkMemReq.memoryTypeBits,
        requiresDedicatedAllocation ? 1 : 0,
        prefersDedicatedAllocation ? 1 : 0,
        createInfo.flags,
        createInfo.usage,
        createInfo.requiredFlags,
        createInfo.preferredFlags,
        createInfo.memoryTypeBits,
        createInfo.pool,
        allocation,
        userDataStr.GetString());
    Flush();
}

void VmaRecorder::RecordAllocateMemoryForImage(uint32_t frameIndex,
    const VkMemoryRequirements& vkMemReq,
    bool requiresDedicatedAllocation,
    bool prefersDedicatedAllocation,
    const VmaAllocationCreateInfo& createInfo,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    UserDataString userDataStr(createInfo.flags, createInfo.pUserData);
    fprintf(m_File, "%u,%.3f,%u,vmaAllocateMemoryForImage,%llu,%llu,%u,%u,%u,%u,%u,%u,%u,%u,%p,%p,%s\n", callParams.threadId, callParams.time, frameIndex,
        vkMemReq.size,
        vkMemReq.alignment,
        vkMemReq.memoryTypeBits,
        requiresDedicatedAllocation ? 1 : 0,
        prefersDedicatedAllocation ? 1 : 0,
        createInfo.flags,
        createInfo.usage,
        createInfo.requiredFlags,
        createInfo.preferredFlags,
        createInfo.memoryTypeBits,
        createInfo.pool,
        allocation,
        userDataStr.GetString());
    Flush();
}

void VmaRecorder::RecordFreeMemory(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaFreeMemory,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordFreeMemoryPages(uint32_t frameIndex,
    uint64_t allocationCount,
    const VmaAllocation* pAllocations)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaFreeMemoryPages,", callParams.threadId, callParams.time, frameIndex);
    PrintPointerList(allocationCount, pAllocations);
    fprintf(m_File, "\n");
    Flush();
}

void VmaRecorder::RecordSetAllocationUserData(uint32_t frameIndex,
    VmaAllocation allocation,
    const void* pUserData)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    UserDataString userDataStr(
        allocation->IsUserDataString() ? VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT : 0,
        pUserData);
    fprintf(m_File, "%u,%.3f,%u,vmaSetAllocationUserData,%p,%s\n", callParams.threadId, callParams.time, frameIndex,
        allocation,
        userDataStr.GetString());
    Flush();
}

void VmaRecorder::RecordCreateLostAllocation(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaCreateLostAllocation,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordMapMemory(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaMapMemory,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordUnmapMemory(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaUnmapMemory,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordFlushAllocation(uint32_t frameIndex,
    VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaFlushAllocation,%p,%llu,%llu\n", callParams.threadId, callParams.time, frameIndex,
        allocation,
        offset,
        size);
    Flush();
}

void VmaRecorder::RecordInvalidateAllocation(uint32_t frameIndex,
    VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaInvalidateAllocation,%p,%llu,%llu\n", callParams.threadId, callParams.time, frameIndex,
        allocation,
        offset,
        size);
    Flush();
}

void VmaRecorder::RecordCreateBuffer(uint32_t frameIndex,
    const VkBufferCreateInfo& bufCreateInfo,
    const VmaAllocationCreateInfo& allocCreateInfo,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    UserDataString userDataStr(allocCreateInfo.flags, allocCreateInfo.pUserData);
    fprintf(m_File, "%u,%.3f,%u,vmaCreateBuffer,%u,%llu,%u,%u,%u,%u,%u,%u,%u,%p,%p,%s\n", callParams.threadId, callParams.time, frameIndex,
        bufCreateInfo.flags,
        bufCreateInfo.size,
        bufCreateInfo.usage,
        bufCreateInfo.sharingMode,
        allocCreateInfo.flags,
        allocCreateInfo.usage,
        allocCreateInfo.requiredFlags,
        allocCreateInfo.preferredFlags,
        allocCreateInfo.memoryTypeBits,
        allocCreateInfo.pool,
        allocation,
        userDataStr.GetString());
    Flush();
}

void VmaRecorder::RecordCreateImage(uint32_t frameIndex,
    const VkImageCreateInfo& imageCreateInfo,
    const VmaAllocationCreateInfo& allocCreateInfo,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    UserDataString userDataStr(allocCreateInfo.flags, allocCreateInfo.pUserData);
    fprintf(m_File, "%u,%.3f,%u,vmaCreateImage,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%p,%p,%s\n", callParams.threadId, callParams.time, frameIndex,
        imageCreateInfo.flags,
        imageCreateInfo.imageType,
        imageCreateInfo.format,
        imageCreateInfo.extent.width,
        imageCreateInfo.extent.height,
        imageCreateInfo.extent.depth,
        imageCreateInfo.mipLevels,
        imageCreateInfo.arrayLayers,
        imageCreateInfo.samples,
        imageCreateInfo.tiling,
        imageCreateInfo.usage,
        imageCreateInfo.sharingMode,
        imageCreateInfo.initialLayout,
        allocCreateInfo.flags,
        allocCreateInfo.usage,
        allocCreateInfo.requiredFlags,
        allocCreateInfo.preferredFlags,
        allocCreateInfo.memoryTypeBits,
        allocCreateInfo.pool,
        allocation,
        userDataStr.GetString());
    Flush();
}

void VmaRecorder::RecordDestroyBuffer(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaDestroyBuffer,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordDestroyImage(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaDestroyImage,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordTouchAllocation(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaTouchAllocation,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordGetAllocationInfo(uint32_t frameIndex,
    VmaAllocation allocation)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaGetAllocationInfo,%p\n", callParams.threadId, callParams.time, frameIndex,
        allocation);
    Flush();
}

void VmaRecorder::RecordMakePoolAllocationsLost(uint32_t frameIndex,
    VmaPool pool)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaMakePoolAllocationsLost,%p\n", callParams.threadId, callParams.time, frameIndex,
        pool);
    Flush();
}

void VmaRecorder::RecordDefragmentationBegin(uint32_t frameIndex,
    const VmaDefragmentationInfo2& info,
    VmaDefragmentationContext ctx)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaDefragmentationBegin,%u,", callParams.threadId, callParams.time, frameIndex,
        info.flags);
    PrintPointerList(info.allocationCount, info.pAllocations);
    fprintf(m_File, ",");
    PrintPointerList(info.poolCount, info.pPools);
    fprintf(m_File, ",%llu,%u,%llu,%u,%p,%p\n",
        info.maxCpuBytesToMove,
        info.maxCpuAllocationsToMove,
        info.maxGpuBytesToMove,
        info.maxGpuAllocationsToMove,
        info.commandBuffer,
        ctx);
    Flush();
}

void VmaRecorder::RecordDefragmentationEnd(uint32_t frameIndex,
    VmaDefragmentationContext ctx)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaDefragmentationEnd,%p\n", callParams.threadId, callParams.time, frameIndex,
        ctx);
    Flush();
}

void VmaRecorder::RecordSetPoolName(uint32_t frameIndex,
    VmaPool pool,
    const char* name)
{
    CallParams callParams;
    GetBasicParams(callParams);

    VmaMutexLock lock(m_FileMutex, m_UseMutex);
    fprintf(m_File, "%u,%.3f,%u,vmaSetPoolName,%p,%s\n", callParams.threadId, callParams.time, frameIndex,
        pool, name != VMA_NULL ? name : "");
    Flush();
}

VmaRecorder::UserDataString::UserDataString(VmaAllocationCreateFlags allocFlags, const void* pUserData)
{
    if(pUserData != VMA_NULL)
    {
        if((allocFlags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0)
        {
            m_Str = (const char*)pUserData;
        }
        else
        {
            sprintf_s(m_PtrStr, "%p", pUserData);
            m_Str = m_PtrStr;
        }
    }
    else
    {
        m_Str = "";
    }
}

void VmaRecorder::WriteConfiguration(
    const VkPhysicalDeviceProperties& devProps,
    const VkPhysicalDeviceMemoryProperties& memProps,
    uint32_t vulkanApiVersion,
    bool dedicatedAllocationExtensionEnabled,
    bool bindMemory2ExtensionEnabled,
    bool memoryBudgetExtensionEnabled)
{
    fprintf(m_File, "Config,Begin\n");

    fprintf(m_File, "VulkanApiVersion,%u,%u\n", VK_VERSION_MAJOR(vulkanApiVersion), VK_VERSION_MINOR(vulkanApiVersion));

    fprintf(m_File, "PhysicalDevice,apiVersion,%u\n", devProps.apiVersion);
    fprintf(m_File, "PhysicalDevice,driverVersion,%u\n", devProps.driverVersion);
    fprintf(m_File, "PhysicalDevice,vendorID,%u\n", devProps.vendorID);
    fprintf(m_File, "PhysicalDevice,deviceID,%u\n", devProps.deviceID);
    fprintf(m_File, "PhysicalDevice,deviceType,%u\n", devProps.deviceType);
    fprintf(m_File, "PhysicalDevice,deviceName,%s\n", devProps.deviceName);

    fprintf(m_File, "PhysicalDeviceLimits,maxMemoryAllocationCount,%u\n", devProps.limits.maxMemoryAllocationCount);
    fprintf(m_File, "PhysicalDeviceLimits,bufferImageGranularity,%llu\n", devProps.limits.bufferImageGranularity);
    fprintf(m_File, "PhysicalDeviceLimits,nonCoherentAtomSize,%llu\n", devProps.limits.nonCoherentAtomSize);

    fprintf(m_File, "PhysicalDeviceMemory,HeapCount,%u\n", memProps.memoryHeapCount);
    for(uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
    {
        fprintf(m_File, "PhysicalDeviceMemory,Heap,%u,size,%llu\n", i, memProps.memoryHeaps[i].size);
        fprintf(m_File, "PhysicalDeviceMemory,Heap,%u,flags,%u\n", i, memProps.memoryHeaps[i].flags);
    }
    fprintf(m_File, "PhysicalDeviceMemory,TypeCount,%u\n", memProps.memoryTypeCount);
    for(uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        fprintf(m_File, "PhysicalDeviceMemory,Type,%u,heapIndex,%u\n", i, memProps.memoryTypes[i].heapIndex);
        fprintf(m_File, "PhysicalDeviceMemory,Type,%u,propertyFlags,%u\n", i, memProps.memoryTypes[i].propertyFlags);
    }

    fprintf(m_File, "Extension,VK_KHR_dedicated_allocation,%u\n", dedicatedAllocationExtensionEnabled ? 1 : 0);
    fprintf(m_File, "Extension,VK_KHR_bind_memory2,%u\n", bindMemory2ExtensionEnabled ? 1 : 0);
    fprintf(m_File, "Extension,VK_EXT_memory_budget,%u\n", memoryBudgetExtensionEnabled ? 1 : 0);

    fprintf(m_File, "Macro,VMA_DEBUG_ALWAYS_DEDICATED_MEMORY,%u\n", VMA_DEBUG_ALWAYS_DEDICATED_MEMORY ? 1 : 0);
    fprintf(m_File, "Macro,VMA_DEBUG_ALIGNMENT,%llu\n", (VkDeviceSize)VMA_DEBUG_ALIGNMENT);
    fprintf(m_File, "Macro,VMA_DEBUG_MARGIN,%llu\n", (VkDeviceSize)VMA_DEBUG_MARGIN);
    fprintf(m_File, "Macro,VMA_DEBUG_INITIALIZE_ALLOCATIONS,%u\n", VMA_DEBUG_INITIALIZE_ALLOCATIONS ? 1 : 0);
    fprintf(m_File, "Macro,VMA_DEBUG_DETECT_CORRUPTION,%u\n", VMA_DEBUG_DETECT_CORRUPTION ? 1 : 0);
    fprintf(m_File, "Macro,VMA_DEBUG_GLOBAL_MUTEX,%u\n", VMA_DEBUG_GLOBAL_MUTEX ? 1 : 0);
    fprintf(m_File, "Macro,VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY,%llu\n", (VkDeviceSize)VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY);
    fprintf(m_File, "Macro,VMA_SMALL_HEAP_MAX_SIZE,%llu\n", (VkDeviceSize)VMA_SMALL_HEAP_MAX_SIZE);
    fprintf(m_File, "Macro,VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE,%llu\n", (VkDeviceSize)VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE);

    fprintf(m_File, "Config,End\n");
}

void VmaRecorder::GetBasicParams(CallParams& outParams)
{
    outParams.threadId = GetCurrentThreadId();

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    outParams.time = (double)(counter.QuadPart - m_StartCounter) / (double)m_Freq;
}

void VmaRecorder::PrintPointerList(uint64_t count, const VmaAllocation* pItems)
{
    if(count)
    {
        fprintf(m_File, "%p", pItems[0]);
        for(uint64_t i = 1; i < count; ++i)
        {
            fprintf(m_File, " %p", pItems[i]);
        }
    }
}

void VmaRecorder::Flush()
{
    if((m_Flags & VMA_RECORD_FLUSH_AFTER_CALL_BIT) != 0)
    {
        fflush(m_File);
    }
}

#endif // #if VMA_RECORDING_ENABLED

////////////////////////////////////////////////////////////////////////////////
// VmaAllocationObjectAllocator

VmaAllocationObjectAllocator::VmaAllocationObjectAllocator(const VkAllocationCallbacks* pAllocationCallbacks) :
    m_Allocator(pAllocationCallbacks, 1024)
{
}

VmaAllocation VmaAllocationObjectAllocator::Allocate()
{
    VmaMutexLock mutexLock(m_Mutex);
    return m_Allocator.Alloc();
}

void VmaAllocationObjectAllocator::Free(VmaAllocation hAlloc)
{
    VmaMutexLock mutexLock(m_Mutex);
    m_Allocator.Free(hAlloc);
}

////////////////////////////////////////////////////////////////////////////////
// VmaAllocator_T

VmaAllocator_T::VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo) :
    m_UseMutex((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT) == 0),
    m_VulkanApiVersion(pCreateInfo->vulkanApiVersion != 0 ? pCreateInfo->vulkanApiVersion : VK_API_VERSION_1_0),
    m_UseKhrDedicatedAllocation((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0),
    m_UseKhrBindMemory2((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0),
    m_UseExtMemoryBudget((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0),
    m_hDevice(pCreateInfo->device),
    m_hInstance(pCreateInfo->instance),
    m_AllocationCallbacksSpecified(pCreateInfo->pAllocationCallbacks != VMA_NULL),
    m_AllocationCallbacks(pCreateInfo->pAllocationCallbacks ?
        *pCreateInfo->pAllocationCallbacks : VmaEmptyAllocationCallbacks),
    m_AllocationObjectAllocator(&m_AllocationCallbacks),
    m_HeapSizeLimitMask(0),
    m_PreferredLargeHeapBlockSize(0),
    m_PhysicalDevice(pCreateInfo->physicalDevice),
    m_CurrentFrameIndex(0),
    m_GpuDefragmentationMemoryTypeBits(UINT32_MAX),
    m_Pools(VmaStlAllocator<VmaPool>(GetAllocationCallbacks())),
    m_NextPoolId(0)
#if VMA_RECORDING_ENABLED
    ,m_pRecorder(VMA_NULL)
#endif
{
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        m_UseKhrDedicatedAllocation = false;
        m_UseKhrBindMemory2 = false;
    }

    if(VMA_DEBUG_DETECT_CORRUPTION)
    {
        // Needs to be multiply of uint32_t size because we are going to write VMA_CORRUPTION_DETECTION_MAGIC_VALUE to it.
        VMA_ASSERT(VMA_DEBUG_MARGIN % sizeof(uint32_t) == 0);
    }

    VMA_ASSERT(pCreateInfo->physicalDevice && pCreateInfo->device);

    if(m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0))
    {
#if !(VMA_DEDICATED_ALLOCATION)
        if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0)
        {
            VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT set but required extensions are disabled by preprocessor macros.");
        }
#endif
#if !(VMA_BIND_MEMORY2)
        if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0)
        {
            VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT set but required extension is disabled by preprocessor macros.");
        }
#endif
    }
#if !(VMA_MEMORY_BUDGET)
    if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT set but required extension is disabled by preprocessor macros.");
    }
#endif
#if VMA_VULKAN_VERSION < 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_ASSERT(0 && "vulkanApiVersion >= VK_API_VERSION_1_1 but required Vulkan version is disabled by preprocessor macros.");
    }
#endif

    memset(&m_DeviceMemoryCallbacks, 0 ,sizeof(m_DeviceMemoryCallbacks));
    memset(&m_PhysicalDeviceProperties, 0, sizeof(m_PhysicalDeviceProperties));
    memset(&m_MemProps, 0, sizeof(m_MemProps));
        
    memset(&m_pBlockVectors, 0, sizeof(m_pBlockVectors));
    memset(&m_pDedicatedAllocations, 0, sizeof(m_pDedicatedAllocations));
    memset(&m_VulkanFunctions, 0, sizeof(m_VulkanFunctions));

    if(pCreateInfo->pDeviceMemoryCallbacks != VMA_NULL)
    {
        m_DeviceMemoryCallbacks.pfnAllocate = pCreateInfo->pDeviceMemoryCallbacks->pfnAllocate;
        m_DeviceMemoryCallbacks.pfnFree = pCreateInfo->pDeviceMemoryCallbacks->pfnFree;
    }

    ImportVulkanFunctions(pCreateInfo->pVulkanFunctions);

    (*m_VulkanFunctions.vkGetPhysicalDeviceProperties)(m_PhysicalDevice, &m_PhysicalDeviceProperties);
    (*m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties)(m_PhysicalDevice, &m_MemProps);

    VMA_ASSERT(VmaIsPow2(VMA_DEBUG_ALIGNMENT));
    VMA_ASSERT(VmaIsPow2(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY));
    VMA_ASSERT(VmaIsPow2(m_PhysicalDeviceProperties.limits.bufferImageGranularity));
    VMA_ASSERT(VmaIsPow2(m_PhysicalDeviceProperties.limits.nonCoherentAtomSize));

    m_PreferredLargeHeapBlockSize = (pCreateInfo->preferredLargeHeapBlockSize != 0) ?
        pCreateInfo->preferredLargeHeapBlockSize : static_cast<VkDeviceSize>(VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE);

    if(pCreateInfo->pHeapSizeLimit != VMA_NULL)
    {
        for(uint32_t heapIndex = 0; heapIndex < GetMemoryHeapCount(); ++heapIndex)
        {
            const VkDeviceSize limit = pCreateInfo->pHeapSizeLimit[heapIndex];
            if(limit != VK_WHOLE_SIZE)
            {
                m_HeapSizeLimitMask |= 1u << heapIndex;
                if(limit < m_MemProps.memoryHeaps[heapIndex].size)
                {
                    m_MemProps.memoryHeaps[heapIndex].size = limit;
                }
            }
        }
    }

    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(memTypeIndex);

        m_pBlockVectors[memTypeIndex] = vma_new(this, VmaBlockVector)(
            this,
            VK_NULL_HANDLE, // hParentPool
            memTypeIndex,
            preferredBlockSize,
            0,
            SIZE_MAX,
            GetBufferImageGranularity(),
            pCreateInfo->frameInUseCount,
            false, // explicitBlockSize
            false); // linearAlgorithm
        // No need to call m_pBlockVectors[memTypeIndex][blockVectorTypeIndex]->CreateMinBlocks here,
        // becase minBlockCount is 0.
        m_pDedicatedAllocations[memTypeIndex] = vma_new(this, AllocationVectorType)(VmaStlAllocator<VmaAllocation>(GetAllocationCallbacks()));

    }
}

VkResult VmaAllocator_T::Init(const VmaAllocatorCreateInfo* pCreateInfo)
{
    VkResult res = VK_SUCCESS;

    if(pCreateInfo->pRecordSettings != VMA_NULL &&
        !VmaStrIsEmpty(pCreateInfo->pRecordSettings->pFilePath))
    {
#if VMA_RECORDING_ENABLED
        m_pRecorder = vma_new(this, VmaRecorder)();
        res = m_pRecorder->Init(*pCreateInfo->pRecordSettings, m_UseMutex);
        if(res != VK_SUCCESS)
        {
            return res;
        }
        m_pRecorder->WriteConfiguration(
            m_PhysicalDeviceProperties,
            m_MemProps,
            m_VulkanApiVersion,
            m_UseKhrDedicatedAllocation,
            m_UseKhrBindMemory2,
            m_UseExtMemoryBudget);
        m_pRecorder->RecordCreateAllocator(GetCurrentFrameIndex());
#else
        VMA_ASSERT(0 && "VmaAllocatorCreateInfo::pRecordSettings used, but not supported due to VMA_RECORDING_ENABLED not defined to 1.");
        return VK_ERROR_FEATURE_NOT_PRESENT;
#endif
    }

#if VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        UpdateVulkanBudget();
    }
#endif // #if VMA_MEMORY_BUDGET

    return res;
}

VmaAllocator_T::~VmaAllocator_T()
{
#if VMA_RECORDING_ENABLED
    if(m_pRecorder != VMA_NULL)
    {
        m_pRecorder->RecordDestroyAllocator(GetCurrentFrameIndex());
        vma_delete(this, m_pRecorder);
    }
#endif
    
    VMA_ASSERT(m_Pools.empty());

    for(size_t i = GetMemoryTypeCount(); i--; )
    {
        if(m_pDedicatedAllocations[i] != VMA_NULL && !m_pDedicatedAllocations[i]->empty())
        {
            VMA_ASSERT(0 && "Unfreed dedicated allocations found.");
        }

        vma_delete(this, m_pDedicatedAllocations[i]);
        vma_delete(this, m_pBlockVectors[i]);
    }
}

void VmaAllocator_T::ImportVulkanFunctions(const VmaVulkanFunctions* pVulkanFunctions)
{
#if VMA_STATIC_VULKAN_FUNCTIONS == 1
    m_VulkanFunctions.vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetPhysicalDeviceProperties;
    m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetPhysicalDeviceMemoryProperties;
    m_VulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkAllocateMemory;
    m_VulkanFunctions.vkFreeMemory = (PFN_vkFreeMemory)vkFreeMemory;
    m_VulkanFunctions.vkMapMemory = (PFN_vkMapMemory)vkMapMemory;
    m_VulkanFunctions.vkUnmapMemory = (PFN_vkUnmapMemory)vkUnmapMemory;
    m_VulkanFunctions.vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)vkFlushMappedMemoryRanges;
    m_VulkanFunctions.vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)vkInvalidateMappedMemoryRanges;
    m_VulkanFunctions.vkBindBufferMemory = (PFN_vkBindBufferMemory)vkBindBufferMemory;
    m_VulkanFunctions.vkBindImageMemory = (PFN_vkBindImageMemory)vkBindImageMemory;
    m_VulkanFunctions.vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vkGetBufferMemoryRequirements;
    m_VulkanFunctions.vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)vkGetImageMemoryRequirements;
    m_VulkanFunctions.vkCreateBuffer = (PFN_vkCreateBuffer)vkCreateBuffer;
    m_VulkanFunctions.vkDestroyBuffer = (PFN_vkDestroyBuffer)vkDestroyBuffer;
    m_VulkanFunctions.vkCreateImage = (PFN_vkCreateImage)vkCreateImage;
    m_VulkanFunctions.vkDestroyImage = (PFN_vkDestroyImage)vkDestroyImage;
    m_VulkanFunctions.vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)vkCmdCopyBuffer;
#if VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_ASSERT(m_hInstance != VK_NULL_HANDLE);
        m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR =
            (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(m_hDevice, "vkGetBufferMemoryRequirements2");
        m_VulkanFunctions.vkGetImageMemoryRequirements2KHR =
            (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(m_hDevice, "vkGetImageMemoryRequirements2");
        m_VulkanFunctions.vkBindBufferMemory2KHR =
            (PFN_vkBindBufferMemory2KHR)vkGetDeviceProcAddr(m_hDevice, "vkBindBufferMemory2");
        m_VulkanFunctions.vkBindImageMemory2KHR =
            (PFN_vkBindImageMemory2KHR)vkGetDeviceProcAddr(m_hDevice, "vkBindImageMemory2");
        m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR =
            (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr(m_hInstance, "vkGetPhysicalDeviceMemoryProperties2");
    }
#endif
#if VMA_DEDICATED_ALLOCATION
    if(m_UseKhrDedicatedAllocation)
    {
        m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR =
            (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(m_hDevice, "vkGetBufferMemoryRequirements2KHR");
        m_VulkanFunctions.vkGetImageMemoryRequirements2KHR =
            (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(m_hDevice, "vkGetImageMemoryRequirements2KHR");
    }
#endif
#if VMA_BIND_MEMORY2
    if(m_UseKhrBindMemory2)
    {
        m_VulkanFunctions.vkBindBufferMemory2KHR =
            (PFN_vkBindBufferMemory2KHR)vkGetDeviceProcAddr(m_hDevice, "vkBindBufferMemory2KHR");
        m_VulkanFunctions.vkBindImageMemory2KHR =
            (PFN_vkBindImageMemory2KHR)vkGetDeviceProcAddr(m_hDevice, "vkBindImageMemory2KHR");
    }
#endif // #if VMA_BIND_MEMORY2
#if VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget && m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_ASSERT(m_hInstance != VK_NULL_HANDLE);
        m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR =
            (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr(m_hInstance, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }
#endif // #if VMA_MEMORY_BUDGET
#endif // #if VMA_STATIC_VULKAN_FUNCTIONS == 1

#define VMA_COPY_IF_NOT_NULL(funcName) \
    if(pVulkanFunctions->funcName != VMA_NULL) m_VulkanFunctions.funcName = pVulkanFunctions->funcName;

    if(pVulkanFunctions != VMA_NULL)
    {
        VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceProperties);
        VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties);
        VMA_COPY_IF_NOT_NULL(vkAllocateMemory);
        VMA_COPY_IF_NOT_NULL(vkFreeMemory);
        VMA_COPY_IF_NOT_NULL(vkMapMemory);
        VMA_COPY_IF_NOT_NULL(vkUnmapMemory);
        VMA_COPY_IF_NOT_NULL(vkFlushMappedMemoryRanges);
        VMA_COPY_IF_NOT_NULL(vkInvalidateMappedMemoryRanges);
        VMA_COPY_IF_NOT_NULL(vkBindBufferMemory);
        VMA_COPY_IF_NOT_NULL(vkBindImageMemory);
        VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements);
        VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements);
        VMA_COPY_IF_NOT_NULL(vkCreateBuffer);
        VMA_COPY_IF_NOT_NULL(vkDestroyBuffer);
        VMA_COPY_IF_NOT_NULL(vkCreateImage);
        VMA_COPY_IF_NOT_NULL(vkDestroyImage);
        VMA_COPY_IF_NOT_NULL(vkCmdCopyBuffer);
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
        VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements2KHR);
        VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements2KHR);
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
        VMA_COPY_IF_NOT_NULL(vkBindBufferMemory2KHR);
        VMA_COPY_IF_NOT_NULL(vkBindImageMemory2KHR);
#endif
#if VMA_MEMORY_BUDGET
        VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties2KHR);
#endif
    }

#undef VMA_COPY_IF_NOT_NULL

    // If these asserts are hit, you must either #define VMA_STATIC_VULKAN_FUNCTIONS 1
    // or pass valid pointers as VmaAllocatorCreateInfo::pVulkanFunctions.
    VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceProperties != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkAllocateMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkFreeMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkMapMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkUnmapMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkFlushMappedMemoryRanges != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkInvalidateMappedMemoryRanges != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkCreateBuffer != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkDestroyBuffer != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkCreateImage != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkDestroyImage != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkCmdCopyBuffer != VMA_NULL);
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrDedicatedAllocation)
    {
        VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR != VMA_NULL);
        VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements2KHR != VMA_NULL);
    }
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrBindMemory2)
    {
        VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory2KHR != VMA_NULL);
        VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory2KHR != VMA_NULL);
    }
#endif
#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(m_UseExtMemoryBudget || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR != VMA_NULL);
    }
#endif
}

VkDeviceSize VmaAllocator_T::CalcPreferredBlockSize(uint32_t memTypeIndex)
{
    const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
    const VkDeviceSize heapSize = m_MemProps.memoryHeaps[heapIndex].size;
    const bool isSmallHeap = heapSize <= VMA_SMALL_HEAP_MAX_SIZE;
    return VmaAlignUp(isSmallHeap ? (heapSize / 8) : m_PreferredLargeHeapBlockSize, (VkDeviceSize)32);
}

VkResult VmaAllocator_T::AllocateMemoryOfType(
    VkDeviceSize size,
    VkDeviceSize alignment,
    bool dedicatedAllocation,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    const VmaAllocationCreateInfo& createInfo,
    uint32_t memTypeIndex,
    VmaSuballocationType suballocType,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    VMA_ASSERT(pAllocations != VMA_NULL);
    VMA_DEBUG_LOG("  AllocateMemory: MemoryTypeIndex=%u, AllocationCount=%zu, Size=%llu", memTypeIndex, allocationCount, size);

    VmaAllocationCreateInfo finalCreateInfo = createInfo;

    // If memory type is not HOST_VISIBLE, disable MAPPED.
    if((finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0 &&
        (m_MemProps.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
    {
        finalCreateInfo.flags &= ~VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    // If memory is lazily allocated, it should be always dedicated.
    if(finalCreateInfo.usage == VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED)
    {
        finalCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    VmaBlockVector* const blockVector = m_pBlockVectors[memTypeIndex];
    VMA_ASSERT(blockVector);

    const VkDeviceSize preferredBlockSize = blockVector->GetPreferredBlockSize();
    bool preferDedicatedMemory =
        VMA_DEBUG_ALWAYS_DEDICATED_MEMORY ||
        dedicatedAllocation ||
        // Heuristics: Allocate dedicated memory if requested size if greater than half of preferred block size.
        size > preferredBlockSize / 2;

    if(preferDedicatedMemory &&
        (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0 &&
        finalCreateInfo.pool == VK_NULL_HANDLE)
    {
        finalCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    if((finalCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
    {
        if((finalCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        else
        {
            return AllocateDedicatedMemory(
                size,
                suballocType,
                memTypeIndex,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
                finalCreateInfo.pUserData,
                dedicatedBuffer,
                dedicatedImage,
                allocationCount,
                pAllocations);
        }
    }
    else
    {
        VkResult res = blockVector->Allocate(
            m_CurrentFrameIndex.load(),
            size,
            alignment,
            finalCreateInfo,
            suballocType,
            allocationCount,
            pAllocations);
        if(res == VK_SUCCESS)
        {
            return res;
        }

        // 5. Try dedicated memory.
        if((finalCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        else
        {
            res = AllocateDedicatedMemory(
                size,
                suballocType,
                memTypeIndex,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
                finalCreateInfo.pUserData,
                dedicatedBuffer,
                dedicatedImage,
                allocationCount,
                pAllocations);
            if(res == VK_SUCCESS)
            {
                // Succeeded: AllocateDedicatedMemory function already filld pMemory, nothing more to do here.
                VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
                return VK_SUCCESS;
            }
            else
            {
                // Everything failed: Return error code.
                VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
                return res;
            }
        }
    }
}

VkResult VmaAllocator_T::AllocateDedicatedMemory(
    VkDeviceSize size,
    VmaSuballocationType suballocType,
    uint32_t memTypeIndex,
    bool withinBudget,
    bool map,
    bool isUserDataString,
    void* pUserData,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    VMA_ASSERT(allocationCount > 0 && pAllocations);

    if(withinBudget)
    {
        const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
        VmaBudget heapBudget = {};
        GetBudget(&heapBudget, heapIndex, 1);
        if(heapBudget.usage + size * allocationCount > heapBudget.budget)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.allocationSize = size;

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    VkMemoryDedicatedAllocateInfoKHR dedicatedAllocInfo = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR };
    if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        if(dedicatedBuffer != VK_NULL_HANDLE)
        {
            VMA_ASSERT(dedicatedImage == VK_NULL_HANDLE);
            dedicatedAllocInfo.buffer = dedicatedBuffer;
            allocInfo.pNext = &dedicatedAllocInfo;
        }
        else if(dedicatedImage != VK_NULL_HANDLE)
        {
            dedicatedAllocInfo.image = dedicatedImage;
            allocInfo.pNext = &dedicatedAllocInfo;
        }
    }
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000

    size_t allocIndex;
    VkResult res = VK_SUCCESS;
    for(allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
    {
        res = AllocateDedicatedMemoryPage(
            size,
            suballocType,
            memTypeIndex,
            allocInfo,
            map,
            isUserDataString,
            pUserData,
            pAllocations + allocIndex);
        if(res != VK_SUCCESS)
        {
            break;
        }
    }

    if(res == VK_SUCCESS)
    {
        // Register them in m_pDedicatedAllocations.
        {
            VmaMutexLockWrite lock(m_DedicatedAllocationsMutex[memTypeIndex], m_UseMutex);
            AllocationVectorType* pDedicatedAllocations = m_pDedicatedAllocations[memTypeIndex];
            VMA_ASSERT(pDedicatedAllocations);
            for(allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
            {
                VmaVectorInsertSorted<VmaPointerLess>(*pDedicatedAllocations, pAllocations[allocIndex]);
            }
        }

        VMA_DEBUG_LOG("    Allocated DedicatedMemory Count=%zu, MemoryTypeIndex=#%u", allocationCount, memTypeIndex);
    }
    else
    {
        // Free all already created allocations.
        while(allocIndex--)
        {
            VmaAllocation currAlloc = pAllocations[allocIndex];
            VkDeviceMemory hMemory = currAlloc->GetMemory();
    
            /*
            There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
            before vkFreeMemory.

            if(currAlloc->GetMappedData() != VMA_NULL)
            {
                (*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
            }
            */
    
            FreeVulkanMemory(memTypeIndex, currAlloc->GetSize(), hMemory);
            m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(memTypeIndex), currAlloc->GetSize());
            currAlloc->SetUserData(this, VMA_NULL);
            currAlloc->Dtor();
            m_AllocationObjectAllocator.Free(currAlloc);
        }

        memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);
    }

    return res;
}

VkResult VmaAllocator_T::AllocateDedicatedMemoryPage(
    VkDeviceSize size,
    VmaSuballocationType suballocType,
    uint32_t memTypeIndex,
    const VkMemoryAllocateInfo& allocInfo,
    bool map,
    bool isUserDataString,
    void* pUserData,
    VmaAllocation* pAllocation)
{
    VkDeviceMemory hMemory = VK_NULL_HANDLE;
    VkResult res = AllocateVulkanMemory(&allocInfo, &hMemory);
    if(res < 0)
    {
        VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
        return res;
    }

    void* pMappedData = VMA_NULL;
    if(map)
    {
        res = (*m_VulkanFunctions.vkMapMemory)(
            m_hDevice,
            hMemory,
            0,
            VK_WHOLE_SIZE,
            0,
            &pMappedData);
        if(res < 0)
        {
            VMA_DEBUG_LOG("    vkMapMemory FAILED");
            FreeVulkanMemory(memTypeIndex, size, hMemory);
            return res;
        }
    }

    *pAllocation = m_AllocationObjectAllocator.Allocate();
    (*pAllocation)->Ctor(m_CurrentFrameIndex.load(), isUserDataString);
    (*pAllocation)->InitDedicatedAllocation(memTypeIndex, hMemory, suballocType, pMappedData, size);
    (*pAllocation)->SetUserData(this, pUserData);
    m_Budget.AddAllocation(MemoryTypeIndexToHeapIndex(memTypeIndex), size);
    if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
    {
        FillAllocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
    }

    return VK_SUCCESS;
}

void VmaAllocator_T::GetBufferMemoryRequirements(
    VkBuffer hBuffer,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkBufferMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.buffer = hBuffer;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        memReq2.pNext = &memDedicatedReq;

        (*m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    {
        (*m_VulkanFunctions.vkGetBufferMemoryRequirements)(m_hDevice, hBuffer, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

void VmaAllocator_T::GetImageMemoryRequirements(
    VkImage hImage,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkImageMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.image = hImage;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        memReq2.pNext = &memDedicatedReq;

        (*m_VulkanFunctions.vkGetImageMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    {
        (*m_VulkanFunctions.vkGetImageMemoryRequirements)(m_hDevice, hImage, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

VkResult VmaAllocator_T::AllocateMemory(
    const VkMemoryRequirements& vkMemReq,
    bool requiresDedicatedAllocation,
    bool prefersDedicatedAllocation,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);

    VMA_ASSERT(VmaIsPow2(vkMemReq.alignment));

    if(vkMemReq.size == 0)
    {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    if((createInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
        (createInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
    {
        VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT together with VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT makes no sense.");
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if((createInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0 &&
        (createInfo.flags & VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT) != 0)
    {
        VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_MAPPED_BIT together with VMA_ALLOCATION_CREATE_CAN_BECOME_LOST_BIT is invalid.");
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if(requiresDedicatedAllocation)
    {
        if((createInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
        {
            VMA_ASSERT(0 && "VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT specified while dedicated allocation is required.");
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        if(createInfo.pool != VK_NULL_HANDLE)
        {
            VMA_ASSERT(0 && "Pool specified while dedicated allocation is required.");
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    if((createInfo.pool != VK_NULL_HANDLE) &&
        ((createInfo.flags & (VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT)) != 0))
    {
        VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT when pool != null is invalid.");
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    if(createInfo.pool != VK_NULL_HANDLE)
    {
        const VkDeviceSize alignmentForPool = VMA_MAX(
            vkMemReq.alignment,
            GetMemoryTypeMinAlignment(createInfo.pool->m_BlockVector.GetMemoryTypeIndex()));

        VmaAllocationCreateInfo createInfoForPool = createInfo;
        // If memory type is not HOST_VISIBLE, disable MAPPED.
        if((createInfoForPool.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0 &&
            (m_MemProps.memoryTypes[createInfo.pool->m_BlockVector.GetMemoryTypeIndex()].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            createInfoForPool.flags &= ~VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        return createInfo.pool->m_BlockVector.Allocate(
            m_CurrentFrameIndex.load(),
            vkMemReq.size,
            alignmentForPool,
            createInfoForPool,
            suballocType,
            allocationCount,
            pAllocations);
    }
    else
    {
        // Bit mask of memory Vulkan types acceptable for this allocation.
        uint32_t memoryTypeBits = vkMemReq.memoryTypeBits;
        uint32_t memTypeIndex = UINT32_MAX;
        VkResult res = vmaFindMemoryTypeIndex(this, memoryTypeBits, &createInfo, &memTypeIndex);
        if(res == VK_SUCCESS)
        {
            VkDeviceSize alignmentForMemType = VMA_MAX(
                vkMemReq.alignment,
                GetMemoryTypeMinAlignment(memTypeIndex));

            res = AllocateMemoryOfType(
                vkMemReq.size,
                alignmentForMemType,
                requiresDedicatedAllocation || prefersDedicatedAllocation,
                dedicatedBuffer,
                dedicatedImage,
                createInfo,
                memTypeIndex,
                suballocType,
                allocationCount,
                pAllocations);
            // Succeeded on first try.
            if(res == VK_SUCCESS)
            {
                return res;
            }
            // Allocation from this memory type failed. Try other compatible memory types.
            else
            {
                for(;;)
                {
                    // Remove old memTypeIndex from list of possibilities.
                    memoryTypeBits &= ~(1u << memTypeIndex);
                    // Find alternative memTypeIndex.
                    res = vmaFindMemoryTypeIndex(this, memoryTypeBits, &createInfo, &memTypeIndex);
                    if(res == VK_SUCCESS)
                    {
                        alignmentForMemType = VMA_MAX(
                            vkMemReq.alignment,
                            GetMemoryTypeMinAlignment(memTypeIndex));
                        
                        res = AllocateMemoryOfType(
                            vkMemReq.size,
                            alignmentForMemType,
                            requiresDedicatedAllocation || prefersDedicatedAllocation,
                            dedicatedBuffer,
                            dedicatedImage,
                            createInfo,
                            memTypeIndex,
                            suballocType,
                            allocationCount,
                            pAllocations);
                        // Allocation from this alternative memory type succeeded.
                        if(res == VK_SUCCESS)
                        {
                            return res;
                        }
                        // else: Allocation from this memory type failed. Try next one - next loop iteration.
                    }
                    // No other matching memory type index could be found.
                    else
                    {
                        // Not returning res, which is VK_ERROR_FEATURE_NOT_PRESENT, because we already failed to allocate once.
                        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                    }
                }
            }
        }
        // Can't find any single memory type maching requirements. res is VK_ERROR_FEATURE_NOT_PRESENT.
        else
            return res;
    }
}

void VmaAllocator_T::FreeMemory(
    size_t allocationCount,
    const VmaAllocation* pAllocations)
{
    VMA_ASSERT(pAllocations);

    for(size_t allocIndex = allocationCount; allocIndex--; )
    {
        VmaAllocation allocation = pAllocations[allocIndex];

        if(allocation != VK_NULL_HANDLE)
        {
            if(TouchAllocation(allocation))
            {
                if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
                {
                    FillAllocation(allocation, VMA_ALLOCATION_FILL_PATTERN_DESTROYED);
                }

                switch(allocation->GetType())
                {
                case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
                    {
                        VmaBlockVector* pBlockVector = VMA_NULL;
                        VmaPool hPool = allocation->GetBlock()->GetParentPool();
                        if(hPool != VK_NULL_HANDLE)
                        {
                            pBlockVector = &hPool->m_BlockVector;
                        }
                        else
                        {
                            const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
                            pBlockVector = m_pBlockVectors[memTypeIndex];
                        }
                        pBlockVector->Free(allocation);
                    }
                    break;
                case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
                    FreeDedicatedMemory(allocation);
                    break;
                default:
                    VMA_ASSERT(0);
                }
            }

            // Do this regardless of whether the allocation is lost. Lost allocations still account to Budget.AllocationBytes.
            m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(allocation->GetMemoryTypeIndex()), allocation->GetSize());
            allocation->SetUserData(this, VMA_NULL);
            allocation->Dtor();
            m_AllocationObjectAllocator.Free(allocation);
        }
    }
}

VkResult VmaAllocator_T::ResizeAllocation(
    const VmaAllocation alloc,
    VkDeviceSize newSize)
{
    // This function is deprecated and so it does nothing. It's left for backward compatibility.
    if(newSize == 0 || alloc->GetLastUseFrameIndex() == VMA_FRAME_INDEX_LOST)
    {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    if(newSize == alloc->GetSize())
    {
        return VK_SUCCESS;
    }
    return VK_ERROR_OUT_OF_POOL_MEMORY;
}

void VmaAllocator_T::CalculateStats(VmaStats* pStats)
{
    // Initialize.
    InitStatInfo(pStats->total);
    for(size_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
        InitStatInfo(pStats->memoryType[i]);
    for(size_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i)
        InitStatInfo(pStats->memoryHeap[i]);
    
    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        VmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
        VMA_ASSERT(pBlockVector);
        pBlockVector->AddStats(pStats);
    }

    // Process custom pools.
    {
        VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        for(size_t poolIndex = 0, poolCount = m_Pools.size(); poolIndex < poolCount; ++poolIndex)
        {
            m_Pools[poolIndex]->m_BlockVector.AddStats(pStats);
        }
    }

    // Process dedicated allocations.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        const uint32_t memHeapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
        VmaMutexLockRead dedicatedAllocationsLock(m_DedicatedAllocationsMutex[memTypeIndex], m_UseMutex);
        AllocationVectorType* const pDedicatedAllocVector = m_pDedicatedAllocations[memTypeIndex];
        VMA_ASSERT(pDedicatedAllocVector);
        for(size_t allocIndex = 0, allocCount = pDedicatedAllocVector->size(); allocIndex < allocCount; ++allocIndex)
        {
            VmaStatInfo allocationStatInfo;
            (*pDedicatedAllocVector)[allocIndex]->DedicatedAllocCalcStatsInfo(allocationStatInfo);
            VmaAddStatInfo(pStats->total, allocationStatInfo);
            VmaAddStatInfo(pStats->memoryType[memTypeIndex], allocationStatInfo);
            VmaAddStatInfo(pStats->memoryHeap[memHeapIndex], allocationStatInfo);
        }
    }

    // Postprocess.
    VmaPostprocessCalcStatInfo(pStats->total);
    for(size_t i = 0; i < GetMemoryTypeCount(); ++i)
        VmaPostprocessCalcStatInfo(pStats->memoryType[i]);
    for(size_t i = 0; i < GetMemoryHeapCount(); ++i)
        VmaPostprocessCalcStatInfo(pStats->memoryHeap[i]);
}

void VmaAllocator_T::GetBudget(VmaBudget* outBudget, uint32_t firstHeap, uint32_t heapCount)
{
#if VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        if(m_Budget.m_OperationsSinceBudgetFetch < 30)
        {
            VmaMutexLockRead lockRead(m_Budget.m_BudgetMutex, m_UseMutex);
            for(uint32_t i = 0; i < heapCount; ++i, ++outBudget)
            {
                const uint32_t heapIndex = firstHeap + i;

                outBudget->blockBytes = m_Budget.m_BlockBytes[heapIndex];
                outBudget->allocationBytes = m_Budget.m_AllocationBytes[heapIndex];

                if(m_Budget.m_VulkanUsage[heapIndex] + outBudget->blockBytes > m_Budget.m_BlockBytesAtBudgetFetch[heapIndex])
                {
                    outBudget->usage = m_Budget.m_VulkanUsage[heapIndex] +
                        outBudget->blockBytes - m_Budget.m_BlockBytesAtBudgetFetch[heapIndex];
                }
                else
                {
                    outBudget->usage = 0;
                }

                // Have to take MIN with heap size because explicit HeapSizeLimit is included in it.
                outBudget->budget = VMA_MIN(
                    m_Budget.m_VulkanBudget[heapIndex], m_MemProps.memoryHeaps[heapIndex].size);
            }
        }
        else
        {
            UpdateVulkanBudget(); // Outside of mutex lock
            GetBudget(outBudget, firstHeap, heapCount); // Recursion
        }
    }
    else
#endif
    {
        for(uint32_t i = 0; i < heapCount; ++i, ++outBudget)
        {
            const uint32_t heapIndex = firstHeap + i;

            outBudget->blockBytes = m_Budget.m_BlockBytes[heapIndex];
            outBudget->allocationBytes = m_Budget.m_AllocationBytes[heapIndex];

            outBudget->usage = outBudget->blockBytes;
            outBudget->budget = m_MemProps.memoryHeaps[heapIndex].size * 8 / 10; // 80% heuristics.
        }
    }
}

static const uint32_t VMA_VENDOR_ID_AMD = 4098;

VkResult VmaAllocator_T::DefragmentationBegin(
    const VmaDefragmentationInfo2& info,
    VmaDefragmentationStats* pStats,
    VmaDefragmentationContext* pContext)
{
    if(info.pAllocationsChanged != VMA_NULL)
    {
        memset(info.pAllocationsChanged, 0, info.allocationCount * sizeof(VkBool32));
    }

    *pContext = vma_new(this, VmaDefragmentationContext_T)(
        this, m_CurrentFrameIndex.load(), info.flags, pStats);

    (*pContext)->AddPools(info.poolCount, info.pPools);
    (*pContext)->AddAllocations(
        info.allocationCount, info.pAllocations, info.pAllocationsChanged);

    VkResult res = (*pContext)->Defragment(
        info.maxCpuBytesToMove, info.maxCpuAllocationsToMove,
        info.maxGpuBytesToMove, info.maxGpuAllocationsToMove,
        info.commandBuffer, pStats);

    if(res != VK_NOT_READY)
    {
        vma_delete(this, *pContext);
        *pContext = VMA_NULL;
    }

    return res;
}

VkResult VmaAllocator_T::DefragmentationEnd(
    VmaDefragmentationContext context)
{
    vma_delete(this, context);
    return VK_SUCCESS;
}

void VmaAllocator_T::GetAllocationInfo(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo)
{
    if(hAllocation->CanBecomeLost())
    {
        /*
        Warning: This is a carefully designed algorithm.
        Do not modify unless you really know what you're doing :)
        */
        const uint32_t localCurrFrameIndex = m_CurrentFrameIndex.load();
        uint32_t localLastUseFrameIndex = hAllocation->GetLastUseFrameIndex();
        for(;;)
        {
            if(localLastUseFrameIndex == VMA_FRAME_INDEX_LOST)
            {
                pAllocationInfo->memoryType = UINT32_MAX;
                pAllocationInfo->deviceMemory = VK_NULL_HANDLE;
                pAllocationInfo->offset = 0;
                pAllocationInfo->size = hAllocation->GetSize();
                pAllocationInfo->pMappedData = VMA_NULL;
                pAllocationInfo->pUserData = hAllocation->GetUserData();
                return;
            }
            else if(localLastUseFrameIndex == localCurrFrameIndex)
            {
                pAllocationInfo->memoryType = hAllocation->GetMemoryTypeIndex();
                pAllocationInfo->deviceMemory = hAllocation->GetMemory();
                pAllocationInfo->offset = hAllocation->GetOffset();
                pAllocationInfo->size = hAllocation->GetSize();
                pAllocationInfo->pMappedData = VMA_NULL;
                pAllocationInfo->pUserData = hAllocation->GetUserData();
                return;
            }
            else // Last use time earlier than current time.
            {
                if(hAllocation->CompareExchangeLastUseFrameIndex(localLastUseFrameIndex, localCurrFrameIndex))
                {
                    localLastUseFrameIndex = localCurrFrameIndex;
                }
            }
        }
    }
    else
    {
#if VMA_STATS_STRING_ENABLED
        uint32_t localCurrFrameIndex = m_CurrentFrameIndex.load();
        uint32_t localLastUseFrameIndex = hAllocation->GetLastUseFrameIndex();
        for(;;)
        {
            VMA_ASSERT(localLastUseFrameIndex != VMA_FRAME_INDEX_LOST);
            if(localLastUseFrameIndex == localCurrFrameIndex)
            {
                break;
            }
            else // Last use time earlier than current time.
            {
                if(hAllocation->CompareExchangeLastUseFrameIndex(localLastUseFrameIndex, localCurrFrameIndex))
                {
                    localLastUseFrameIndex = localCurrFrameIndex;
                }
            }
        }
#endif

        pAllocationInfo->memoryType = hAllocation->GetMemoryTypeIndex();
        pAllocationInfo->deviceMemory = hAllocation->GetMemory();
        pAllocationInfo->offset = hAllocation->GetOffset();
        pAllocationInfo->size = hAllocation->GetSize();
        pAllocationInfo->pMappedData = hAllocation->GetMappedData();
        pAllocationInfo->pUserData = hAllocation->GetUserData();
    }
}

bool VmaAllocator_T::TouchAllocation(VmaAllocation hAllocation)
{
    // This is a stripped-down version of VmaAllocator_T::GetAllocationInfo.
    if(hAllocation->CanBecomeLost())
    {
        uint32_t localCurrFrameIndex = m_CurrentFrameIndex.load();
        uint32_t localLastUseFrameIndex = hAllocation->GetLastUseFrameIndex();
        for(;;)
        {
            if(localLastUseFrameIndex == VMA_FRAME_INDEX_LOST)
            {
                return false;
            }
            else if(localLastUseFrameIndex == localCurrFrameIndex)
            {
                return true;
            }
            else // Last use time earlier than current time.
            {
                if(hAllocation->CompareExchangeLastUseFrameIndex(localLastUseFrameIndex, localCurrFrameIndex))
                {
                    localLastUseFrameIndex = localCurrFrameIndex;
                }
            }
        }
    }
    else
    {
#if VMA_STATS_STRING_ENABLED
        uint32_t localCurrFrameIndex = m_CurrentFrameIndex.load();
        uint32_t localLastUseFrameIndex = hAllocation->GetLastUseFrameIndex();
        for(;;)
        {
            VMA_ASSERT(localLastUseFrameIndex != VMA_FRAME_INDEX_LOST);
            if(localLastUseFrameIndex == localCurrFrameIndex)
            {
                break;
            }
            else // Last use time earlier than current time.
            {
                if(hAllocation->CompareExchangeLastUseFrameIndex(localLastUseFrameIndex, localCurrFrameIndex))
                {
                    localLastUseFrameIndex = localCurrFrameIndex;
                }
            }
        }
#endif

        return true;
    }
}

VkResult VmaAllocator_T::CreatePool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool)
{
    VMA_DEBUG_LOG("  CreatePool: MemoryTypeIndex=%u, flags=%u", pCreateInfo->memoryTypeIndex, pCreateInfo->flags);

    VmaPoolCreateInfo newCreateInfo = *pCreateInfo;

    if(newCreateInfo.maxBlockCount == 0)
    {
        newCreateInfo.maxBlockCount = SIZE_MAX;
    }
    if(newCreateInfo.minBlockCount > newCreateInfo.maxBlockCount)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(newCreateInfo.memoryTypeIndex);

    *pPool = vma_new(this, VmaPool_T)(this, newCreateInfo, preferredBlockSize);

    VkResult res = (*pPool)->m_BlockVector.CreateMinBlocks();
    if(res != VK_SUCCESS)
    {
        vma_delete(this, *pPool);
        *pPool = VMA_NULL;
        return res;
    }

    // Add to m_Pools.
    {
        VmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
        (*pPool)->SetId(m_NextPoolId++);
        VmaVectorInsertSorted<VmaPointerLess>(m_Pools, *pPool);
    }

    return VK_SUCCESS;
}

void VmaAllocator_T::DestroyPool(VmaPool pool)
{
    // Remove from m_Pools.
    {
        VmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
        bool success = VmaVectorRemoveSorted<VmaPointerLess>(m_Pools, pool);
        VMA_ASSERT(success && "Pool not found in Allocator.");
    }

    vma_delete(this, pool);
}

void VmaAllocator_T::GetPoolStats(VmaPool pool, VmaPoolStats* pPoolStats)
{
    pool->m_BlockVector.GetPoolStats(pPoolStats);
}

void VmaAllocator_T::SetCurrentFrameIndex(uint32_t frameIndex)
{
    m_CurrentFrameIndex.store(frameIndex);

#if VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        UpdateVulkanBudget();
    }
#endif // #if VMA_MEMORY_BUDGET
}

void VmaAllocator_T::MakePoolAllocationsLost(
    VmaPool hPool,
    size_t* pLostAllocationCount)
{
    hPool->m_BlockVector.MakePoolAllocationsLost(
        m_CurrentFrameIndex.load(),
        pLostAllocationCount);
}

VkResult VmaAllocator_T::CheckPoolCorruption(VmaPool hPool)
{
    return hPool->m_BlockVector.CheckCorruption();
}

VkResult VmaAllocator_T::CheckCorruption(uint32_t memoryTypeBits)
{
    VkResult finalRes = VK_ERROR_FEATURE_NOT_PRESENT;

    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        if(((1u << memTypeIndex) & memoryTypeBits) != 0)
        {
            VmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
            VMA_ASSERT(pBlockVector);
            VkResult localRes = pBlockVector->CheckCorruption();
            switch(localRes)
            {
            case VK_ERROR_FEATURE_NOT_PRESENT:
                break;
            case VK_SUCCESS:
                finalRes = VK_SUCCESS;
                break;
            default:
                return localRes;
            }
        }
    }

    // Process custom pools.
    {
        VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        for(size_t poolIndex = 0, poolCount = m_Pools.size(); poolIndex < poolCount; ++poolIndex)
        {
            if(((1u << m_Pools[poolIndex]->m_BlockVector.GetMemoryTypeIndex()) & memoryTypeBits) != 0)
            {
                VkResult localRes = m_Pools[poolIndex]->m_BlockVector.CheckCorruption();
                switch(localRes)
                {
                case VK_ERROR_FEATURE_NOT_PRESENT:
                    break;
                case VK_SUCCESS:
                    finalRes = VK_SUCCESS;
                    break;
                default:
                    return localRes;
                }
            }
        }
    }

    return finalRes;
}

void VmaAllocator_T::CreateLostAllocation(VmaAllocation* pAllocation)
{
    *pAllocation = m_AllocationObjectAllocator.Allocate();
    (*pAllocation)->Ctor(VMA_FRAME_INDEX_LOST, false);
    (*pAllocation)->InitLost();
}

VkResult VmaAllocator_T::AllocateVulkanMemory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory)
{
    const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(pAllocateInfo->memoryTypeIndex);

    // HeapSizeLimit is in effect for this heap.
    if((m_HeapSizeLimitMask & (1u << heapIndex)) != 0)
    {
        const VkDeviceSize heapSize = m_MemProps.memoryHeaps[heapIndex].size;
        VkDeviceSize blockBytes = m_Budget.m_BlockBytes[heapIndex];
        for(;;)
        {
            const VkDeviceSize blockBytesAfterAllocation = blockBytes + pAllocateInfo->allocationSize;
            if(blockBytesAfterAllocation > heapSize)
            {
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            }
            if(m_Budget.m_BlockBytes[heapIndex].compare_exchange_strong(blockBytes, blockBytesAfterAllocation))
            {
                break;
            }
        }
    }
    else
    {
        m_Budget.m_BlockBytes[heapIndex] += pAllocateInfo->allocationSize;
    }

    // VULKAN CALL vkAllocateMemory.
    VkResult res = (*m_VulkanFunctions.vkAllocateMemory)(m_hDevice, pAllocateInfo, GetAllocationCallbacks(), pMemory);

    if(res == VK_SUCCESS)
    {
#if VMA_MEMORY_BUDGET
        ++m_Budget.m_OperationsSinceBudgetFetch;
#endif

        // Informative callback.
        if(m_DeviceMemoryCallbacks.pfnAllocate != VMA_NULL)
        {
            (*m_DeviceMemoryCallbacks.pfnAllocate)(this, pAllocateInfo->memoryTypeIndex, *pMemory, pAllocateInfo->allocationSize);
        }
    }
    else
    {
        m_Budget.m_BlockBytes[heapIndex] -= pAllocateInfo->allocationSize;
    }

    return res;
}

void VmaAllocator_T::FreeVulkanMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory)
{
    // Informative callback.
    if(m_DeviceMemoryCallbacks.pfnFree != VMA_NULL)
    {
        (*m_DeviceMemoryCallbacks.pfnFree)(this, memoryType, hMemory, size);
    }

    // VULKAN CALL vkFreeMemory.
    (*m_VulkanFunctions.vkFreeMemory)(m_hDevice, hMemory, GetAllocationCallbacks());

    m_Budget.m_BlockBytes[MemoryTypeIndexToHeapIndex(memoryType)] -= size;
}

VkResult VmaAllocator_T::BindVulkanBuffer(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkBuffer buffer,
    const void* pNext)
{
    if(pNext != VMA_NULL)
    {
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
            m_VulkanFunctions.vkBindBufferMemory2KHR != VMA_NULL)
        {
            VkBindBufferMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.buffer = buffer;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*m_VulkanFunctions.vkBindBufferMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
        }
        else
#endif // #if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    else
    {
        return (*m_VulkanFunctions.vkBindBufferMemory)(m_hDevice, buffer, memory, memoryOffset);
    }
}

VkResult VmaAllocator_T::BindVulkanImage(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkImage image,
    const void* pNext)
{
    if(pNext != VMA_NULL)
    {
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
            m_VulkanFunctions.vkBindImageMemory2KHR != VMA_NULL)
        {
            VkBindImageMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.image = image;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*m_VulkanFunctions.vkBindImageMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
        }
        else
#endif // #if VMA_BIND_MEMORY2
        {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    else
    {
        return (*m_VulkanFunctions.vkBindImageMemory)(m_hDevice, image, memory, memoryOffset);
    }
}

VkResult VmaAllocator_T::Map(VmaAllocation hAllocation, void** ppData)
{
    if(hAllocation->CanBecomeLost())
    {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
            char *pBytes = VMA_NULL;
            VkResult res = pBlock->Map(this, 1, (void**)&pBytes);
            if(res == VK_SUCCESS)
            {
                *ppData = pBytes + (ptrdiff_t)hAllocation->GetOffset();
                hAllocation->BlockAllocMap();
            }
            return res;
        }
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        return hAllocation->DedicatedAllocMap(this, ppData);
    default:
        VMA_ASSERT(0);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
}

void VmaAllocator_T::Unmap(VmaAllocation hAllocation)
{
    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
            hAllocation->BlockAllocUnmap();
            pBlock->Unmap(this, 1);
        }
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        hAllocation->DedicatedAllocUnmap(this);
        break;
    default:
        VMA_ASSERT(0);
    }
}

VkResult VmaAllocator_T::BindBufferMemory(
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    VkResult res = VK_SUCCESS;
    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = BindVulkanBuffer(hAllocation->GetMemory(), allocationLocalOffset, hBuffer, pNext);
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
        VMA_ASSERT(pBlock && "Binding buffer to allocation that doesn't belong to any block. Is the allocation lost?");
        res = pBlock->BindBufferMemory(this, hAllocation, allocationLocalOffset, hBuffer, pNext);
        break;
    }
    default:
        VMA_ASSERT(0);
    }
    return res;
}

VkResult VmaAllocator_T::BindImageMemory(
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    VkResult res = VK_SUCCESS;
    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = BindVulkanImage(hAllocation->GetMemory(), allocationLocalOffset, hImage, pNext);
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        VmaDeviceMemoryBlock* pBlock = hAllocation->GetBlock();
        VMA_ASSERT(pBlock && "Binding image to allocation that doesn't belong to any block. Is the allocation lost?");
        res = pBlock->BindImageMemory(this, hAllocation, allocationLocalOffset, hImage, pNext);
        break;
    }
    default:
        VMA_ASSERT(0);
    }
    return res;
}

void VmaAllocator_T::FlushOrInvalidateAllocation(
    VmaAllocation hAllocation,
    VkDeviceSize offset, VkDeviceSize size,
    VMA_CACHE_OPERATION op)
{
    const uint32_t memTypeIndex = hAllocation->GetMemoryTypeIndex();
    if(size > 0 && IsMemoryTypeNonCoherent(memTypeIndex))
    {
        const VkDeviceSize allocationSize = hAllocation->GetSize();
        VMA_ASSERT(offset <= allocationSize);

        const VkDeviceSize nonCoherentAtomSize = m_PhysicalDeviceProperties.limits.nonCoherentAtomSize;

        VkMappedMemoryRange memRange = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
        memRange.memory = hAllocation->GetMemory();
        
        switch(hAllocation->GetType())
        {
        case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
            memRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                memRange.size = allocationSize - memRange.offset;
            }
            else
            {
                VMA_ASSERT(offset + size <= allocationSize);
                memRange.size = VMA_MIN(
                    VmaAlignUp(size + (offset - memRange.offset), nonCoherentAtomSize),
                    allocationSize - memRange.offset);
            }
            break;

        case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            // 1. Still within this allocation.
            memRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                size = allocationSize - offset;
            }
            else
            {
                VMA_ASSERT(offset + size <= allocationSize);
            }
            memRange.size = VmaAlignUp(size + (offset - memRange.offset), nonCoherentAtomSize);

            // 2. Adjust to whole block.
            const VkDeviceSize allocationOffset = hAllocation->GetOffset();
            VMA_ASSERT(allocationOffset % nonCoherentAtomSize == 0);
            const VkDeviceSize blockSize = hAllocation->GetBlock()->m_pMetadata->GetSize();
            memRange.offset += allocationOffset;
            memRange.size = VMA_MIN(memRange.size, blockSize - memRange.offset);
            
            break;
        }
        
        default:
            VMA_ASSERT(0);
        }

        switch(op)
        {
        case VMA_CACHE_FLUSH:
            (*GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hDevice, 1, &memRange);
            break;
        case VMA_CACHE_INVALIDATE:
            (*GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hDevice, 1, &memRange);
            break;
        default:
            VMA_ASSERT(0);
        }
    }
    // else: Just ignore this call.
}

void VmaAllocator_T::FreeDedicatedMemory(const VmaAllocation allocation)
{
    VMA_ASSERT(allocation && allocation->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);

    const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
    {
        VmaMutexLockWrite lock(m_DedicatedAllocationsMutex[memTypeIndex], m_UseMutex);
        AllocationVectorType* const pDedicatedAllocations = m_pDedicatedAllocations[memTypeIndex];
        VMA_ASSERT(pDedicatedAllocations);
        bool success = VmaVectorRemoveSorted<VmaPointerLess>(*pDedicatedAllocations, allocation);
        VMA_ASSERT(success);
    }

    VkDeviceMemory hMemory = allocation->GetMemory();
    
    /*
    There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
    before vkFreeMemory.

    if(allocation->GetMappedData() != VMA_NULL)
    {
        (*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
    }
    */
    
    FreeVulkanMemory(memTypeIndex, allocation->GetSize(), hMemory);

    VMA_DEBUG_LOG("    Freed DedicatedMemory MemoryTypeIndex=%u", memTypeIndex);
}

uint32_t VmaAllocator_T::CalculateGpuDefragmentationMemoryTypeBits() const
{
    VkBufferCreateInfo dummyBufCreateInfo;
    VmaFillGpuDefragmentationBufferCreateInfo(dummyBufCreateInfo);

    uint32_t memoryTypeBits = 0;

    // Create buffer.
    VkBuffer buf = VK_NULL_HANDLE;
    VkResult res = (*GetVulkanFunctions().vkCreateBuffer)(
        m_hDevice, &dummyBufCreateInfo, GetAllocationCallbacks(), &buf);
    if(res == VK_SUCCESS)
    {
        // Query for supported memory types.
        VkMemoryRequirements memReq;
        (*GetVulkanFunctions().vkGetBufferMemoryRequirements)(m_hDevice, buf, &memReq);
        memoryTypeBits = memReq.memoryTypeBits;

        // Destroy buffer.
        (*GetVulkanFunctions().vkDestroyBuffer)(m_hDevice, buf, GetAllocationCallbacks());
    }

    return memoryTypeBits;
}

#if VMA_MEMORY_BUDGET

void VmaAllocator_T::UpdateVulkanBudget()
{
    VMA_ASSERT(m_UseExtMemoryBudget);

    VkPhysicalDeviceMemoryProperties2KHR memProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR };

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
    memProps.pNext = &budgetProps;

    GetVulkanFunctions().vkGetPhysicalDeviceMemoryProperties2KHR(m_PhysicalDevice, &memProps);

    {
        VmaMutexLockWrite lockWrite(m_Budget.m_BudgetMutex, m_UseMutex);

        for(uint32_t heapIndex = 0; heapIndex < GetMemoryHeapCount(); ++heapIndex)
        {
            m_Budget.m_VulkanUsage[heapIndex] = budgetProps.heapUsage[heapIndex];
            m_Budget.m_VulkanBudget[heapIndex] = budgetProps.heapBudget[heapIndex];
            m_Budget.m_BlockBytesAtBudgetFetch[heapIndex] = m_Budget.m_BlockBytes[heapIndex].load();
        }
        m_Budget.m_OperationsSinceBudgetFetch = 0;
    }
}

#endif // #if VMA_MEMORY_BUDGET

void VmaAllocator_T::FillAllocation(const VmaAllocation hAllocation, uint8_t pattern)
{
    if(VMA_DEBUG_INITIALIZE_ALLOCATIONS &&
        !hAllocation->CanBecomeLost() &&
        (m_MemProps.memoryTypes[hAllocation->GetMemoryTypeIndex()].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
    {
        void* pData = VMA_NULL;
        VkResult res = Map(hAllocation, &pData);
        if(res == VK_SUCCESS)
        {
            memset(pData, (int)pattern, (size_t)hAllocation->GetSize());
            FlushOrInvalidateAllocation(hAllocation, 0, VK_WHOLE_SIZE, VMA_CACHE_FLUSH);
            Unmap(hAllocation);
        }
        else
        {
            VMA_ASSERT(0 && "VMA_DEBUG_INITIALIZE_ALLOCATIONS is enabled, but couldn't map memory to fill allocation.");
        }
    }
}

uint32_t VmaAllocator_T::GetGpuDefragmentationMemoryTypeBits()
{
    uint32_t memoryTypeBits = m_GpuDefragmentationMemoryTypeBits.load();
    if(memoryTypeBits == UINT32_MAX)
    {
        memoryTypeBits = CalculateGpuDefragmentationMemoryTypeBits();
        m_GpuDefragmentationMemoryTypeBits.store(memoryTypeBits);
    }
    return memoryTypeBits;
}

#if VMA_STATS_STRING_ENABLED

void VmaAllocator_T::PrintDetailedMap(VmaJsonWriter& json)
{
    bool dedicatedAllocationsStarted = false;
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        VmaMutexLockRead dedicatedAllocationsLock(m_DedicatedAllocationsMutex[memTypeIndex], m_UseMutex);
        AllocationVectorType* const pDedicatedAllocVector = m_pDedicatedAllocations[memTypeIndex];
        VMA_ASSERT(pDedicatedAllocVector);
        if(pDedicatedAllocVector->empty() == false)
        {
            if(dedicatedAllocationsStarted == false)
            {
                dedicatedAllocationsStarted = true;
                json.WriteString("DedicatedAllocations");
                json.BeginObject();
            }

            json.BeginString("Type ");
            json.ContinueString(memTypeIndex);
            json.EndString();
                
            json.BeginArray();

            for(size_t i = 0; i < pDedicatedAllocVector->size(); ++i)
            {
                json.BeginObject(true);
                const VmaAllocation hAlloc = (*pDedicatedAllocVector)[i];
                hAlloc->PrintParameters(json);
                json.EndObject();
            }

            json.EndArray();
        }
    }
    if(dedicatedAllocationsStarted)
    {
        json.EndObject();
    }

    {
        bool allocationsStarted = false;
        for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
        {
            if(m_pBlockVectors[memTypeIndex]->IsEmpty() == false)
            {
                if(allocationsStarted == false)
                {
                    allocationsStarted = true;
                    json.WriteString("DefaultPools");
                    json.BeginObject();
                }

                json.BeginString("Type ");
                json.ContinueString(memTypeIndex);
                json.EndString();

                m_pBlockVectors[memTypeIndex]->PrintDetailedMap(json);
            }
        }
        if(allocationsStarted)
        {
            json.EndObject();
        }
    }

    // Custom pools
    {
        VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        const size_t poolCount = m_Pools.size();
        if(poolCount > 0)
        {
            json.WriteString("Pools");
            json.BeginObject();
            for(size_t poolIndex = 0; poolIndex < poolCount; ++poolIndex)
            {
                json.BeginString();
                json.ContinueString(m_Pools[poolIndex]->GetId());
                json.EndString();

                m_Pools[poolIndex]->m_BlockVector.PrintDetailedMap(json);
            }
            json.EndObject();
        }
    }
}

#endif // #if VMA_STATS_STRING_ENABLED

////////////////////////////////////////////////////////////////////////////////
// Public interface

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAllocator(
    const VmaAllocatorCreateInfo* pCreateInfo,
    VmaAllocator* pAllocator)
{
    VMA_ASSERT(pCreateInfo && pAllocator);
    VMA_ASSERT(pCreateInfo->vulkanApiVersion == 0 ||
        (VK_VERSION_MAJOR(pCreateInfo->vulkanApiVersion) == 1 && VK_VERSION_MINOR(pCreateInfo->vulkanApiVersion) <= 1));
    VMA_DEBUG_LOG("vmaCreateAllocator");
    *pAllocator = vma_new(pCreateInfo->pAllocationCallbacks, VmaAllocator_T)(pCreateInfo);
    return (*pAllocator)->Init(pCreateInfo);
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyAllocator(
    VmaAllocator allocator)
{
    if(allocator != VK_NULL_HANDLE)
    {
        VMA_DEBUG_LOG("vmaDestroyAllocator");
        VkAllocationCallbacks allocationCallbacks = allocator->m_AllocationCallbacks;
        vma_delete(&allocationCallbacks, allocator);
    }
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPhysicalDeviceProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceProperties **ppPhysicalDeviceProperties)
{
    VMA_ASSERT(allocator && ppPhysicalDeviceProperties);
    *ppPhysicalDeviceProperties = &allocator->m_PhysicalDeviceProperties;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceMemoryProperties** ppPhysicalDeviceMemoryProperties)
{
    VMA_ASSERT(allocator && ppPhysicalDeviceMemoryProperties);
    *ppPhysicalDeviceMemoryProperties = &allocator->m_MemProps;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryTypeProperties(
    VmaAllocator allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* pFlags)
{
    VMA_ASSERT(allocator && pFlags);
    VMA_ASSERT(memoryTypeIndex < allocator->GetMemoryTypeCount());
    *pFlags = allocator->m_MemProps.memoryTypes[memoryTypeIndex].propertyFlags;
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetCurrentFrameIndex(
    VmaAllocator allocator,
    uint32_t frameIndex)
{
    VMA_ASSERT(allocator);
    VMA_ASSERT(frameIndex != VMA_FRAME_INDEX_LOST);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->SetCurrentFrameIndex(frameIndex);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculateStats(
    VmaAllocator allocator,
    VmaStats* pStats)
{
    VMA_ASSERT(allocator && pStats);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->CalculateStats(pStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetBudget(
    VmaAllocator allocator,
    VmaBudget* pBudget)
{
    VMA_ASSERT(allocator && pBudget);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->GetBudget(pBudget, 0, allocator->GetMemoryHeapCount());
}

#if VMA_STATS_STRING_ENABLED

VMA_CALL_PRE void VMA_CALL_POST vmaBuildStatsString(
    VmaAllocator allocator,
    char** ppStatsString,
    VkBool32 detailedMap)
{
    VMA_ASSERT(allocator && ppStatsString);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VmaStringBuilder sb(allocator);
    {
        VmaJsonWriter json(allocator->GetAllocationCallbacks(), sb);
        json.BeginObject();

        VmaBudget budget[VK_MAX_MEMORY_HEAPS];
        allocator->GetBudget(budget, 0, allocator->GetMemoryHeapCount());

        VmaStats stats;
        allocator->CalculateStats(&stats);

        json.WriteString("Total");
        VmaPrintStatInfo(json, stats.total);
    
        for(uint32_t heapIndex = 0; heapIndex < allocator->GetMemoryHeapCount(); ++heapIndex)
        {
            json.BeginString("Heap ");
            json.ContinueString(heapIndex);
            json.EndString();
            json.BeginObject();

            json.WriteString("Size");
            json.WriteNumber(allocator->m_MemProps.memoryHeaps[heapIndex].size);

            json.WriteString("Flags");
            json.BeginArray(true);
            if((allocator->m_MemProps.memoryHeaps[heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
            {
                json.WriteString("DEVICE_LOCAL");
            }
            json.EndArray();

            json.WriteString("Budget");
            json.BeginObject();
            {
                json.WriteString("BlockBytes");
                json.WriteNumber(budget[heapIndex].blockBytes);
                json.WriteString("AllocationBytes");
                json.WriteNumber(budget[heapIndex].allocationBytes);
                json.WriteString("Usage");
                json.WriteNumber(budget[heapIndex].usage);
                json.WriteString("Budget");
                json.WriteNumber(budget[heapIndex].budget);
            }
            json.EndObject();

            if(stats.memoryHeap[heapIndex].blockCount > 0)
            {
                json.WriteString("Stats");
                VmaPrintStatInfo(json, stats.memoryHeap[heapIndex]);
            }

            for(uint32_t typeIndex = 0; typeIndex < allocator->GetMemoryTypeCount(); ++typeIndex)
            {
                if(allocator->MemoryTypeIndexToHeapIndex(typeIndex) == heapIndex)
                {
                    json.BeginString("Type ");
                    json.ContinueString(typeIndex);
                    json.EndString();

                    json.BeginObject();

                    json.WriteString("Flags");
                    json.BeginArray(true);
                    VkMemoryPropertyFlags flags = allocator->m_MemProps.memoryTypes[typeIndex].propertyFlags;
                    if((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
                    {
                        json.WriteString("DEVICE_LOCAL");
                    }
                    if((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
                    {
                        json.WriteString("HOST_VISIBLE");
                    }
                    if((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
                    {
                        json.WriteString("HOST_COHERENT");
                    }
                    if((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0)
                    {
                        json.WriteString("HOST_CACHED");
                    }
                    if((flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0)
                    {
                        json.WriteString("LAZILY_ALLOCATED");
                    }
                    json.EndArray();

                    if(stats.memoryType[typeIndex].blockCount > 0)
                    {
                        json.WriteString("Stats");
                        VmaPrintStatInfo(json, stats.memoryType[typeIndex]);
                    }

                    json.EndObject();
                }
            }

            json.EndObject();
        }
        if(detailedMap == VK_TRUE)
        {
            allocator->PrintDetailedMap(json);
        }

        json.EndObject();
    }

    const size_t len = sb.GetLength();
    char* const pChars = vma_new_array(allocator, char, len + 1);
    if(len > 0)
    {
        memcpy(pChars, sb.GetData(), len);
    }
    pChars[len] = '\0';
    *ppStatsString = pChars;
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeStatsString(
    VmaAllocator allocator,
    char* pStatsString)
{
    if(pStatsString != VMA_NULL)
    {
        VMA_ASSERT(allocator);
        size_t len = strlen(pStatsString);
        vma_delete_array(allocator, pStatsString, len + 1);
    }
}

#endif // #if VMA_STATS_STRING_ENABLED

/*
This function is not protected by any mutex because it just reads immutable data.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndex(
    VmaAllocator allocator,
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    if(pAllocationCreateInfo->memoryTypeBits != 0)
    {
        memoryTypeBits &= pAllocationCreateInfo->memoryTypeBits;
    }
    
    uint32_t requiredFlags = pAllocationCreateInfo->requiredFlags;
    uint32_t preferredFlags = pAllocationCreateInfo->preferredFlags;
    uint32_t notPreferredFlags = 0;

    // Convert usage to requiredFlags and preferredFlags.
    switch(pAllocationCreateInfo->usage)
    {
    case VMA_MEMORY_USAGE_UNKNOWN:
        break;
    case VMA_MEMORY_USAGE_GPU_ONLY:
        if(!allocator->IsIntegratedGpu() || (preferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            preferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case VMA_MEMORY_USAGE_CPU_ONLY:
        requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case VMA_MEMORY_USAGE_CPU_TO_GPU:
        requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if(!allocator->IsIntegratedGpu() || (preferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            preferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case VMA_MEMORY_USAGE_GPU_TO_CPU:
        requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        preferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case VMA_MEMORY_USAGE_CPU_COPY:
        notPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED:
        requiredFlags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
        break;
    default:
        VMA_ASSERT(0);
        break;
    }

    *pMemoryTypeIndex = UINT32_MAX;
    uint32_t minCost = UINT32_MAX;
    for(uint32_t memTypeIndex = 0, memTypeBit = 1;
        memTypeIndex < allocator->GetMemoryTypeCount();
        ++memTypeIndex, memTypeBit <<= 1)
    {
        // This memory type is acceptable according to memoryTypeBits bitmask.
        if((memTypeBit & memoryTypeBits) != 0)
        {
            const VkMemoryPropertyFlags currFlags =
                allocator->m_MemProps.memoryTypes[memTypeIndex].propertyFlags;
            // This memory type contains requiredFlags.
            if((requiredFlags & ~currFlags) == 0)
            {
                // Calculate cost as number of bits from preferredFlags not present in this memory type.
                uint32_t currCost = VmaCountBitsSet(preferredFlags & ~currFlags) +
                    VmaCountBitsSet(currFlags & notPreferredFlags);
                // Remember memory type with lowest cost.
                if(currCost < minCost)
                {
                    *pMemoryTypeIndex = memTypeIndex;
                    if(currCost == 0)
                    {
                        return VK_SUCCESS;
                    }
                    minCost = currCost;
                }
            }
        }
    }
    return (*pMemoryTypeIndex != UINT32_MAX) ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForBufferInfo(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pBufferCreateInfo != VMA_NULL);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    const VkDevice hDev = allocator->m_hDevice;
    VkBuffer hBuffer = VK_NULL_HANDLE;
    VkResult res = allocator->GetVulkanFunctions().vkCreateBuffer(
        hDev, pBufferCreateInfo, allocator->GetAllocationCallbacks(), &hBuffer);
    if(res == VK_SUCCESS)
    {
        VkMemoryRequirements memReq = {};
        allocator->GetVulkanFunctions().vkGetBufferMemoryRequirements(
            hDev, hBuffer, &memReq);

        res = vmaFindMemoryTypeIndex(
            allocator,
            memReq.memoryTypeBits,
            pAllocationCreateInfo,
            pMemoryTypeIndex);

        allocator->GetVulkanFunctions().vkDestroyBuffer(
            hDev, hBuffer, allocator->GetAllocationCallbacks());
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForImageInfo(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pImageCreateInfo != VMA_NULL);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    const VkDevice hDev = allocator->m_hDevice;
    VkImage hImage = VK_NULL_HANDLE;
    VkResult res = allocator->GetVulkanFunctions().vkCreateImage(
        hDev, pImageCreateInfo, allocator->GetAllocationCallbacks(), &hImage);
    if(res == VK_SUCCESS)
    {
        VkMemoryRequirements memReq = {};
        allocator->GetVulkanFunctions().vkGetImageMemoryRequirements(
            hDev, hImage, &memReq);

        res = vmaFindMemoryTypeIndex(
            allocator,
            memReq.memoryTypeBits,
            pAllocationCreateInfo,
            pMemoryTypeIndex);

        allocator->GetVulkanFunctions().vkDestroyImage(
            hDev, hImage, allocator->GetAllocationCallbacks());
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreatePool(
	VmaAllocator allocator,
	const VmaPoolCreateInfo* pCreateInfo,
	VmaPool* pPool)
{
    VMA_ASSERT(allocator && pCreateInfo && pPool);
    
    VMA_DEBUG_LOG("vmaCreatePool");
    
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    
    VkResult res = allocator->CreatePool(pCreateInfo, pPool);
    
#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordCreatePool(allocator->GetCurrentFrameIndex(), *pCreateInfo, *pPool);
    }
#endif
    
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyPool(
    VmaAllocator allocator,
    VmaPool pool)
{
    VMA_ASSERT(allocator);
    
    if(pool == VK_NULL_HANDLE)
    {
        return;
    }
    
    VMA_DEBUG_LOG("vmaDestroyPool");
    
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    
#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordDestroyPool(allocator->GetCurrentFrameIndex(), pool);
    }
#endif

    allocator->DestroyPool(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolStats(
    VmaAllocator allocator,
    VmaPool pool,
    VmaPoolStats* pPoolStats)
{
    VMA_ASSERT(allocator && pool && pPoolStats);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->GetPoolStats(pool, pPoolStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaMakePoolAllocationsLost(
    VmaAllocator allocator,
    VmaPool pool,
    size_t* pLostAllocationCount)
{
    VMA_ASSERT(allocator && pool);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordMakePoolAllocationsLost(allocator->GetCurrentFrameIndex(), pool);
    }
#endif

    allocator->MakePoolAllocationsLost(pool, pLostAllocationCount);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckPoolCorruption(VmaAllocator allocator, VmaPool pool)
{
    VMA_ASSERT(allocator && pool);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VMA_DEBUG_LOG("vmaCheckPoolCorruption");

    return allocator->CheckPoolCorruption(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char** ppName)
{
    VMA_ASSERT(allocator && pool);
    
    VMA_DEBUG_LOG("vmaGetPoolName");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *ppName = pool->GetName();
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char* pName)
{
    VMA_ASSERT(allocator && pool);

    VMA_DEBUG_LOG("vmaSetPoolName");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    pool->SetName(pName);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordSetPoolName(allocator->GetCurrentFrameIndex(), pool, pName);
    }
#endif
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemory(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VkResult result = allocator->AllocateMemory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_UNKNOWN,
        1, // allocationCount
        pAllocation);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordAllocateMemory(
            allocator->GetCurrentFrameIndex(),
            *pVkMemoryRequirements,
            *pCreateInfo,
            *pAllocation);
    }
#endif
        
    if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

	return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryPages(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    size_t allocationCount,
    VmaAllocation* pAllocations,
    VmaAllocationInfo* pAllocationInfo)
{
    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocations);

    VMA_DEBUG_LOG("vmaAllocateMemoryPages");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VkResult result = allocator->AllocateMemory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_UNKNOWN,
        allocationCount,
        pAllocations);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordAllocateMemoryPages(
            allocator->GetCurrentFrameIndex(),
            *pVkMemoryRequirements,
            *pCreateInfo,
            (uint64_t)allocationCount,
            pAllocations);
    }
#endif
        
    if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
    {
        for(size_t i = 0; i < allocationCount; ++i)
        {
            allocator->GetAllocationInfo(pAllocations[i], pAllocationInfo + i);
        }
    }

	return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && buffer != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemoryForBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation = false;
    allocator->GetBufferMemoryRequirements(buffer, vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation);

    VkResult result = allocator->AllocateMemory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        buffer, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_BUFFER,
        1, // allocationCount
        pAllocation);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordAllocateMemoryForBuffer(
            allocator->GetCurrentFrameIndex(),
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            *pCreateInfo,
            *pAllocation);
    }
#endif

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

	return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForImage(
    VmaAllocator allocator,
    VkImage image,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && image != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemoryForImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation  = false;
    allocator->GetImageMemoryRequirements(image, vkMemReq,
        requiresDedicatedAllocation, prefersDedicatedAllocation);

    VkResult result = allocator->AllocateMemory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        VK_NULL_HANDLE, // dedicatedBuffer
        image, // dedicatedImage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN,
        1, // allocationCount
        pAllocation);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordAllocateMemoryForImage(
            allocator->GetCurrentFrameIndex(),
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            *pCreateInfo,
            *pAllocation);
    }
#endif

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

	return result;
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemory(
    VmaAllocator allocator,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator);
    
    if(allocation == VK_NULL_HANDLE)
    {
        return;
    }
    
    VMA_DEBUG_LOG("vmaFreeMemory");
    
    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordFreeMemory(
            allocator->GetCurrentFrameIndex(),
            allocation);
    }
#endif
    
    allocator->FreeMemory(
        1, // allocationCount
        &allocation);
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemoryPages(
    VmaAllocator allocator,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    if(allocationCount == 0)
    {
        return;
    }

    VMA_ASSERT(allocator);
    
    VMA_DEBUG_LOG("vmaFreeMemoryPages");
    
    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordFreeMemoryPages(
            allocator->GetCurrentFrameIndex(),
            (uint64_t)allocationCount,
            pAllocations);
    }
#endif
    
    allocator->FreeMemory(allocationCount, pAllocations);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaResizeAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize newSize)
{
    VMA_ASSERT(allocator && allocation);
    
    VMA_DEBUG_LOG("vmaResizeAllocation");
    
    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->ResizeAllocation(allocation, newSize);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && allocation && pAllocationInfo);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordGetAllocationInfo(
            allocator->GetCurrentFrameIndex(),
            allocation);
    }
#endif

    allocator->GetAllocationInfo(allocation, pAllocationInfo);
}

VMA_CALL_PRE VkBool32 VMA_CALL_POST vmaTouchAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordTouchAllocation(
            allocator->GetCurrentFrameIndex(),
            allocation);
    }
#endif

    return allocator->TouchAllocation(allocation);
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationUserData(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void* pUserData)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocation->SetUserData(allocator, pUserData);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordSetAllocationUserData(
            allocator->GetCurrentFrameIndex(),
            allocation,
            pUserData);
    }
#endif
}

VMA_CALL_PRE void VMA_CALL_POST vmaCreateLostAllocation(
    VmaAllocator allocator,
    VmaAllocation* pAllocation)
{
    VMA_ASSERT(allocator && pAllocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK;

    allocator->CreateLostAllocation(pAllocation);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordCreateLostAllocation(
            allocator->GetCurrentFrameIndex(),
            *pAllocation);
    }
#endif
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaMapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void** ppData)
{
    VMA_ASSERT(allocator && allocation && ppData);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult res = allocator->Map(allocation, ppData);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordMapMemory(
            allocator->GetCurrentFrameIndex(),
            allocation);
    }
#endif

    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaUnmapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordUnmapMemory(
            allocator->GetCurrentFrameIndex(),
            allocation);
    }
#endif

    allocator->Unmap(allocation);
}

VMA_CALL_PRE void VMA_CALL_POST vmaFlushAllocation(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_LOG("vmaFlushAllocation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->FlushOrInvalidateAllocation(allocation, offset, size, VMA_CACHE_FLUSH);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordFlushAllocation(
            allocator->GetCurrentFrameIndex(),
            allocation, offset, size);
    }
#endif
}

VMA_CALL_PRE void VMA_CALL_POST vmaInvalidateAllocation(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_LOG("vmaInvalidateAllocation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->FlushOrInvalidateAllocation(allocation, offset, size, VMA_CACHE_INVALIDATE);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordInvalidateAllocation(
            allocator->GetCurrentFrameIndex(),
            allocation, offset, size);
    }
#endif
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckCorruption(VmaAllocator allocator, uint32_t memoryTypeBits)
{
    VMA_ASSERT(allocator);

    VMA_DEBUG_LOG("vmaCheckCorruption");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->CheckCorruption(memoryTypeBits);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaDefragment(
    VmaAllocator allocator,
    VmaAllocation* pAllocations,
    size_t allocationCount,
    VkBool32* pAllocationsChanged,
    const VmaDefragmentationInfo *pDefragmentationInfo,
    VmaDefragmentationStats* pDefragmentationStats)
{
    // Deprecated interface, reimplemented using new one.

    VmaDefragmentationInfo2 info2 = {};
    info2.allocationCount = (uint32_t)allocationCount;
    info2.pAllocations = pAllocations;
    info2.pAllocationsChanged = pAllocationsChanged;
    if(pDefragmentationInfo != VMA_NULL)
    {
        info2.maxCpuAllocationsToMove = pDefragmentationInfo->maxAllocationsToMove;
        info2.maxCpuBytesToMove = pDefragmentationInfo->maxBytesToMove;
    }
    else
    {
        info2.maxCpuAllocationsToMove = UINT32_MAX;
        info2.maxCpuBytesToMove = VK_WHOLE_SIZE;
    }
    // info2.flags, maxGpuAllocationsToMove, maxGpuBytesToMove, commandBuffer deliberately left zero.

    VmaDefragmentationContext ctx;
    VkResult res = vmaDefragmentationBegin(allocator, &info2, pDefragmentationStats, &ctx);
    if(res == VK_NOT_READY)
    {
        res = vmaDefragmentationEnd( allocator, ctx);
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaDefragmentationBegin(
    VmaAllocator allocator,
    const VmaDefragmentationInfo2* pInfo,
    VmaDefragmentationStats* pStats,
    VmaDefragmentationContext *pContext)
{
    VMA_ASSERT(allocator && pInfo && pContext);

    // Degenerate case: Nothing to defragment.
    if(pInfo->allocationCount == 0 && pInfo->poolCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(pInfo->allocationCount == 0 || pInfo->pAllocations != VMA_NULL);
    VMA_ASSERT(pInfo->poolCount == 0 || pInfo->pPools != VMA_NULL);
    VMA_HEAVY_ASSERT(VmaValidatePointerArray(pInfo->allocationCount, pInfo->pAllocations));
    VMA_HEAVY_ASSERT(VmaValidatePointerArray(pInfo->poolCount, pInfo->pPools));

    VMA_DEBUG_LOG("vmaDefragmentationBegin");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult res = allocator->DefragmentationBegin(*pInfo, pStats, pContext);

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordDefragmentationBegin(
            allocator->GetCurrentFrameIndex(), *pInfo, *pContext);
    }
#endif

    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaDefragmentationEnd(
    VmaAllocator allocator,
    VmaDefragmentationContext context)
{
    VMA_ASSERT(allocator);

    VMA_DEBUG_LOG("vmaDefragmentationEnd");

    if(context != VK_NULL_HANDLE)
    {
        VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
        if(allocator->GetRecorder() != VMA_NULL)
        {
            allocator->GetRecorder()->RecordDefragmentationEnd(
                allocator->GetCurrentFrameIndex(), context);
        }
#endif

        return allocator->DefragmentationEnd(context);
    }
    else
    {
        return VK_SUCCESS;
    }
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkBuffer buffer)
{
    VMA_ASSERT(allocator && allocation && buffer);

    VMA_DEBUG_LOG("vmaBindBufferMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindBufferMemory(allocation, 0, buffer, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer buffer,
    const void* pNext)
{
    VMA_ASSERT(allocator && allocation && buffer);

    VMA_DEBUG_LOG("vmaBindBufferMemory2");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindBufferMemory(allocation, allocationLocalOffset, buffer, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkImage image)
{
    VMA_ASSERT(allocator && allocation && image);

    VMA_DEBUG_LOG("vmaBindImageMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindImageMemory(allocation, 0, image, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage image,
    const void* pNext)
{
    VMA_ASSERT(allocator && allocation && image);

    VMA_DEBUG_LOG("vmaBindImageMemory2");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

        return allocator->BindImageMemory(allocation, allocationLocalOffset, image, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBuffer(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkBuffer* pBuffer,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && pBuffer && pAllocation);

    if(pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    
    VMA_DEBUG_LOG("vmaCreateBuffer");
    
    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pBuffer = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkBuffer.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateBuffer)(
        allocator->m_hDevice,
        pBufferCreateInfo,
        allocator->GetAllocationCallbacks(),
        pBuffer);
    if(res >= 0)
    {
        // 2. vkGetBufferMemoryRequirements.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->GetBufferMemoryRequirements(*pBuffer, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        // 3. Allocate memory using allocator.
        res = allocator->AllocateMemory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            *pBuffer, // dedicatedBuffer
            VK_NULL_HANDLE, // dedicatedImage
            *pAllocationCreateInfo,
            VMA_SUBALLOCATION_TYPE_BUFFER,
            1, // allocationCount
            pAllocation);

#if VMA_RECORDING_ENABLED
        if(allocator->GetRecorder() != VMA_NULL)
        {
            allocator->GetRecorder()->RecordCreateBuffer(
                allocator->GetCurrentFrameIndex(),
                *pBufferCreateInfo,
                *pAllocationCreateInfo,
                *pAllocation);
        }
#endif

        if(res >= 0)
        {
            // 3. Bind buffer with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->BindBufferMemory(*pAllocation, 0, *pBuffer, VMA_NULL);
            }
            if(res >= 0)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->InitBufferImageUsage(pBufferCreateInfo->usage);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->FreeMemory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
            *pBuffer = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
        *pBuffer = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator);

    if(buffer == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordDestroyBuffer(
            allocator->GetCurrentFrameIndex(),
            allocation);
    }
#endif

    if(buffer != VK_NULL_HANDLE)
    {
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, buffer, allocator->GetAllocationCallbacks());
    }

    if(allocation != VK_NULL_HANDLE)
    {
        allocator->FreeMemory(
            1, // allocationCount
            &allocation);
    }
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImage(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkImage* pImage,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation);

    if(pImageCreateInfo->extent.width == 0 ||
        pImageCreateInfo->extent.height == 0 ||
        pImageCreateInfo->extent.depth == 0 ||
        pImageCreateInfo->mipLevels == 0 ||
        pImageCreateInfo->arrayLayers == 0)
    {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }

    VMA_DEBUG_LOG("vmaCreateImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pImage = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkImage.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateImage)(
        allocator->m_hDevice,
        pImageCreateInfo,
        allocator->GetAllocationCallbacks(),
        pImage);
    if(res >= 0)
    {
        VmaSuballocationType suballocType = pImageCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL ?
            VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL :
            VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR;
        
        // 2. Allocate memory using allocator.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->GetImageMemoryRequirements(*pImage, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        res = allocator->AllocateMemory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            VK_NULL_HANDLE, // dedicatedBuffer
            *pImage, // dedicatedImage
            *pAllocationCreateInfo,
            suballocType,
            1, // allocationCount
            pAllocation);

#if VMA_RECORDING_ENABLED
        if(allocator->GetRecorder() != VMA_NULL)
        {
            allocator->GetRecorder()->RecordCreateImage(
                allocator->GetCurrentFrameIndex(),
                *pImageCreateInfo,
                *pAllocationCreateInfo,
                *pAllocation);
        }
#endif

        if(res >= 0)
        {
            // 3. Bind image with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->BindImageMemory(*pAllocation, 0, *pImage, VMA_NULL);
            }
            if(res >= 0)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->InitBufferImageUsage(pImageCreateInfo->usage);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->FreeMemory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
            *pImage = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
        *pImage = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyImage(
    VmaAllocator allocator,
    VkImage image,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator);

    if(image == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

#if VMA_RECORDING_ENABLED
    if(allocator->GetRecorder() != VMA_NULL)
    {
        allocator->GetRecorder()->RecordDestroyImage(
            allocator->GetCurrentFrameIndex(),
            allocation);
    }
#endif

    if(image != VK_NULL_HANDLE)
    {
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, image, allocator->GetAllocationCallbacks());
    }
    if(allocation != VK_NULL_HANDLE)
    {
        allocator->FreeMemory(
            1, // allocationCount
            &allocation);
    }
}

#endif // #ifdef VMA_IMPLEMENTATION

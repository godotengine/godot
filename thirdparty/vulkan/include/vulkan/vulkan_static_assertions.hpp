// Copyright 2015-2024 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_STATIC_ASSERTIONS_HPP
#define VULKAN_STATIC_ASSERTIONS_HPP

#include <vulkan/vulkan.hpp>

//=========================
//=== static_assertions ===
//=========================

//=== VK_VERSION_1_0 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Extent2D ) == sizeof( VkExtent2D ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Extent2D>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Extent2D>::value, "Extent2D is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Extent3D ) == sizeof( VkExtent3D ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Extent3D>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Extent3D>::value, "Extent3D is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Offset2D ) == sizeof( VkOffset2D ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Offset2D>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Offset2D>::value, "Offset2D is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Offset3D ) == sizeof( VkOffset3D ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Offset3D>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Offset3D>::value, "Offset3D is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Rect2D ) == sizeof( VkRect2D ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Rect2D>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Rect2D>::value, "Rect2D is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BaseInStructure ) == sizeof( VkBaseInStructure ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BaseInStructure>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BaseInStructure>::value,
                          "BaseInStructure is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BaseOutStructure ) == sizeof( VkBaseOutStructure ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BaseOutStructure>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BaseOutStructure>::value,
                          "BaseOutStructure is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferMemoryBarrier ) == sizeof( VkBufferMemoryBarrier ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier>::value,
                          "BufferMemoryBarrier is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DispatchIndirectCommand ) == sizeof( VkDispatchIndirectCommand ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DispatchIndirectCommand>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DispatchIndirectCommand>::value,
                          "DispatchIndirectCommand is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrawIndexedIndirectCommand ) == sizeof( VkDrawIndexedIndirectCommand ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrawIndexedIndirectCommand>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrawIndexedIndirectCommand>::value,
                          "DrawIndexedIndirectCommand is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrawIndirectCommand ) == sizeof( VkDrawIndirectCommand ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrawIndirectCommand>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrawIndirectCommand>::value,
                          "DrawIndirectCommand is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageMemoryBarrier ) == sizeof( VkImageMemoryBarrier ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier>::value,
                          "ImageMemoryBarrier is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryBarrier ) == sizeof( VkMemoryBarrier ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryBarrier>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryBarrier>::value, "MemoryBarrier is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionOne ) == sizeof( VkPipelineCacheHeaderVersionOne ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionOne>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionOne>::value,
                          "PipelineCacheHeaderVersionOne is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AllocationCallbacks ) == sizeof( VkAllocationCallbacks ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AllocationCallbacks>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AllocationCallbacks>::value,
                          "AllocationCallbacks is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ApplicationInfo ) == sizeof( VkApplicationInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ApplicationInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ApplicationInfo>::value,
                          "ApplicationInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FormatProperties ) == sizeof( VkFormatProperties ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FormatProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FormatProperties>::value,
                          "FormatProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageFormatProperties ) == sizeof( VkImageFormatProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageFormatProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageFormatProperties>::value,
                          "ImageFormatProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Instance ) == sizeof( VkInstance ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Instance>::value, "Instance is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::InstanceCreateInfo ) == sizeof( VkInstanceCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::InstanceCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::InstanceCreateInfo>::value,
                          "InstanceCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryHeap ) == sizeof( VkMemoryHeap ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryHeap>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryHeap>::value, "MemoryHeap is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryType ) == sizeof( VkMemoryType ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryType>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryType>::value, "MemoryType is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevice ) == sizeof( VkPhysicalDevice ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevice>::value,
                          "PhysicalDevice is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures ) == sizeof( VkPhysicalDeviceFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures>::value,
                          "PhysicalDeviceFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceLimits ) == sizeof( VkPhysicalDeviceLimits ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceLimits>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceLimits>::value,
                          "PhysicalDeviceLimits is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties ) == sizeof( VkPhysicalDeviceMemoryProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties>::value,
                          "PhysicalDeviceMemoryProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties ) == sizeof( VkPhysicalDeviceProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties>::value,
                          "PhysicalDeviceProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseProperties ) == sizeof( VkPhysicalDeviceSparseProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseProperties>::value,
                          "PhysicalDeviceSparseProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueueFamilyProperties ) == sizeof( VkQueueFamilyProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueueFamilyProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueueFamilyProperties>::value,
                          "QueueFamilyProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Device ) == sizeof( VkDevice ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Device>::value, "Device is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceCreateInfo ) == sizeof( VkDeviceCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceCreateInfo>::value,
                          "DeviceCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceQueueCreateInfo ) == sizeof( VkDeviceQueueCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceQueueCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceQueueCreateInfo>::value,
                          "DeviceQueueCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExtensionProperties ) == sizeof( VkExtensionProperties ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExtensionProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExtensionProperties>::value,
                          "ExtensionProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LayerProperties ) == sizeof( VkLayerProperties ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LayerProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LayerProperties>::value,
                          "LayerProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Queue ) == sizeof( VkQueue ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Queue>::value, "Queue is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubmitInfo ) == sizeof( VkSubmitInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubmitInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubmitInfo>::value, "SubmitInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MappedMemoryRange ) == sizeof( VkMappedMemoryRange ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MappedMemoryRange>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MappedMemoryRange>::value,
                          "MappedMemoryRange is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryAllocateInfo ) == sizeof( VkMemoryAllocateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryAllocateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryAllocateInfo>::value,
                          "MemoryAllocateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceMemory ) == sizeof( VkDeviceMemory ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceMemory>::value, "DeviceMemory is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryRequirements ) == sizeof( VkMemoryRequirements ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryRequirements>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryRequirements>::value,
                          "MemoryRequirements is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindSparseInfo ) == sizeof( VkBindSparseInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindSparseInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindSparseInfo>::value,
                          "BindSparseInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageSubresource ) == sizeof( VkImageSubresource ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageSubresource>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageSubresource>::value,
                          "ImageSubresource is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseBufferMemoryBindInfo ) == sizeof( VkSparseBufferMemoryBindInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseBufferMemoryBindInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseBufferMemoryBindInfo>::value,
                          "SparseBufferMemoryBindInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseImageFormatProperties ) == sizeof( VkSparseImageFormatProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties>::value,
                          "SparseImageFormatProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseImageMemoryBind ) == sizeof( VkSparseImageMemoryBind ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseImageMemoryBind>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseImageMemoryBind>::value,
                          "SparseImageMemoryBind is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseImageMemoryBindInfo ) == sizeof( VkSparseImageMemoryBindInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseImageMemoryBindInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseImageMemoryBindInfo>::value,
                          "SparseImageMemoryBindInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements ) == sizeof( VkSparseImageMemoryRequirements ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements>::value,
                          "SparseImageMemoryRequirements is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseImageOpaqueMemoryBindInfo ) == sizeof( VkSparseImageOpaqueMemoryBindInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseImageOpaqueMemoryBindInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseImageOpaqueMemoryBindInfo>::value,
                          "SparseImageOpaqueMemoryBindInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseMemoryBind ) == sizeof( VkSparseMemoryBind ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseMemoryBind>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseMemoryBind>::value,
                          "SparseMemoryBind is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Fence ) == sizeof( VkFence ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Fence>::value, "Fence is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FenceCreateInfo ) == sizeof( VkFenceCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FenceCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FenceCreateInfo>::value,
                          "FenceCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Semaphore ) == sizeof( VkSemaphore ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Semaphore>::value, "Semaphore is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo ) == sizeof( VkSemaphoreCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo>::value,
                          "SemaphoreCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Event ) == sizeof( VkEvent ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Event>::value, "Event is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::EventCreateInfo ) == sizeof( VkEventCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::EventCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::EventCreateInfo>::value,
                          "EventCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueryPool ) == sizeof( VkQueryPool ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueryPool>::value, "QueryPool is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo ) == sizeof( VkQueryPoolCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo>::value,
                          "QueryPoolCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Buffer ) == sizeof( VkBuffer ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Buffer>::value, "Buffer is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCreateInfo ) == sizeof( VkBufferCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCreateInfo>::value,
                          "BufferCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferView ) == sizeof( VkBufferView ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferView>::value, "BufferView is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferViewCreateInfo ) == sizeof( VkBufferViewCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferViewCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferViewCreateInfo>::value,
                          "BufferViewCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Image ) == sizeof( VkImage ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Image>::value, "Image is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageCreateInfo ) == sizeof( VkImageCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageCreateInfo>::value,
                          "ImageCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubresourceLayout ) == sizeof( VkSubresourceLayout ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubresourceLayout>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubresourceLayout>::value,
                          "SubresourceLayout is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ComponentMapping ) == sizeof( VkComponentMapping ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ComponentMapping>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ComponentMapping>::value,
                          "ComponentMapping is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageSubresourceRange ) == sizeof( VkImageSubresourceRange ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageSubresourceRange>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageSubresourceRange>::value,
                          "ImageSubresourceRange is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageView ) == sizeof( VkImageView ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageView>::value, "ImageView is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewCreateInfo ) == sizeof( VkImageViewCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewCreateInfo>::value,
                          "ImageViewCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderModule ) == sizeof( VkShaderModule ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderModule>::value, "ShaderModule is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo ) == sizeof( VkShaderModuleCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo>::value,
                          "ShaderModuleCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCache ) == sizeof( VkPipelineCache ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCache>::value, "PipelineCache is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo ) == sizeof( VkPipelineCacheCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo>::value,
                          "PipelineCacheCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo ) == sizeof( VkComputePipelineCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo>::value,
                          "ComputePipelineCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo ) == sizeof( VkGraphicsPipelineCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo>::value,
                          "GraphicsPipelineCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Pipeline ) == sizeof( VkPipeline ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Pipeline>::value, "Pipeline is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineColorBlendAttachmentState ) == sizeof( VkPipelineColorBlendAttachmentState ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineColorBlendAttachmentState>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineColorBlendAttachmentState>::value,
                          "PipelineColorBlendAttachmentState is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateInfo ) == sizeof( VkPipelineColorBlendStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateInfo>::value,
                          "PipelineColorBlendStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateInfo ) == sizeof( VkPipelineDepthStencilStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateInfo>::value,
                          "PipelineDepthStencilStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateInfo ) == sizeof( VkPipelineDynamicStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateInfo>::value,
                          "PipelineDynamicStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateInfo ) == sizeof( VkPipelineInputAssemblyStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateInfo>::value,
                          "PipelineInputAssemblyStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateInfo ) == sizeof( VkPipelineMultisampleStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateInfo>::value,
                          "PipelineMultisampleStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateInfo ) == sizeof( VkPipelineRasterizationStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateInfo>::value,
                          "PipelineRasterizationStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo ) == sizeof( VkPipelineShaderStageCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo>::value,
                          "PipelineShaderStageCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateInfo ) == sizeof( VkPipelineTessellationStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateInfo>::value,
                          "PipelineTessellationStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateInfo ) == sizeof( VkPipelineVertexInputStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateInfo>::value,
                          "PipelineVertexInputStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateInfo ) == sizeof( VkPipelineViewportStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateInfo>::value,
                          "PipelineViewportStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SpecializationInfo ) == sizeof( VkSpecializationInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SpecializationInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SpecializationInfo>::value,
                          "SpecializationInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SpecializationMapEntry ) == sizeof( VkSpecializationMapEntry ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SpecializationMapEntry>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SpecializationMapEntry>::value,
                          "SpecializationMapEntry is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::StencilOpState ) == sizeof( VkStencilOpState ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::StencilOpState>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::StencilOpState>::value,
                          "StencilOpState is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription ) == sizeof( VkVertexInputAttributeDescription ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription>::value,
                          "VertexInputAttributeDescription is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VertexInputBindingDescription ) == sizeof( VkVertexInputBindingDescription ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription>::value,
                          "VertexInputBindingDescription is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Viewport ) == sizeof( VkViewport ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Viewport>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Viewport>::value, "Viewport is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineLayout ) == sizeof( VkPipelineLayout ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineLayout>::value,
                          "PipelineLayout is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo ) == sizeof( VkPipelineLayoutCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo>::value,
                          "PipelineLayoutCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PushConstantRange ) == sizeof( VkPushConstantRange ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PushConstantRange>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PushConstantRange>::value,
                          "PushConstantRange is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Sampler ) == sizeof( VkSampler ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Sampler>::value, "Sampler is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerCreateInfo ) == sizeof( VkSamplerCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerCreateInfo>::value,
                          "SamplerCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyDescriptorSet ) == sizeof( VkCopyDescriptorSet ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyDescriptorSet>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyDescriptorSet>::value,
                          "CopyDescriptorSet is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorBufferInfo ) == sizeof( VkDescriptorBufferInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorBufferInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorBufferInfo>::value,
                          "DescriptorBufferInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorImageInfo ) == sizeof( VkDescriptorImageInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorImageInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorImageInfo>::value,
                          "DescriptorImageInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorPool ) == sizeof( VkDescriptorPool ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorPool>::value,
                          "DescriptorPool is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo ) == sizeof( VkDescriptorPoolCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo>::value,
                          "DescriptorPoolCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorPoolSize ) == sizeof( VkDescriptorPoolSize ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorPoolSize>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorPoolSize>::value,
                          "DescriptorPoolSize is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSet ) == sizeof( VkDescriptorSet ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSet>::value, "DescriptorSet is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo ) == sizeof( VkDescriptorSetAllocateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo>::value,
                          "DescriptorSetAllocateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetLayout ) == sizeof( VkDescriptorSetLayout ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetLayout>::value,
                          "DescriptorSetLayout is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding ) == sizeof( VkDescriptorSetLayoutBinding ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding>::value,
                          "DescriptorSetLayoutBinding is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo ) == sizeof( VkDescriptorSetLayoutCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo>::value,
                          "DescriptorSetLayoutCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::WriteDescriptorSet ) == sizeof( VkWriteDescriptorSet ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::WriteDescriptorSet>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::WriteDescriptorSet>::value,
                          "WriteDescriptorSet is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentDescription ) == sizeof( VkAttachmentDescription ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentDescription>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentDescription>::value,
                          "AttachmentDescription is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentReference ) == sizeof( VkAttachmentReference ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentReference>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentReference>::value,
                          "AttachmentReference is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Framebuffer ) == sizeof( VkFramebuffer ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Framebuffer>::value, "Framebuffer is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FramebufferCreateInfo ) == sizeof( VkFramebufferCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FramebufferCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FramebufferCreateInfo>::value,
                          "FramebufferCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPass ) == sizeof( VkRenderPass ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPass>::value, "RenderPass is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassCreateInfo ) == sizeof( VkRenderPassCreateInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo>::value,
                          "RenderPassCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassDependency ) == sizeof( VkSubpassDependency ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassDependency>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassDependency>::value,
                          "SubpassDependency is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassDescription ) == sizeof( VkSubpassDescription ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassDescription>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassDescription>::value,
                          "SubpassDescription is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandPool ) == sizeof( VkCommandPool ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandPool>::value, "CommandPool is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo ) == sizeof( VkCommandPoolCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo>::value,
                          "CommandPoolCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBuffer ) == sizeof( VkCommandBuffer ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBuffer>::value, "CommandBuffer is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo ) == sizeof( VkCommandBufferAllocateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo>::value,
                          "CommandBufferAllocateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferBeginInfo ) == sizeof( VkCommandBufferBeginInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferBeginInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferBeginInfo>::value,
                          "CommandBufferBeginInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceInfo ) == sizeof( VkCommandBufferInheritanceInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceInfo>::value,
                          "CommandBufferInheritanceInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCopy ) == sizeof( VkBufferCopy ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCopy>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCopy>::value, "BufferCopy is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferImageCopy ) == sizeof( VkBufferImageCopy ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferImageCopy>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferImageCopy>::value,
                          "BufferImageCopy is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ClearAttachment ) == sizeof( VkClearAttachment ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ClearAttachment>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ClearAttachment>::value,
                          "ClearAttachment is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ClearColorValue ) == sizeof( VkClearColorValue ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ClearColorValue>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ClearColorValue>::value,
                          "ClearColorValue is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ClearDepthStencilValue ) == sizeof( VkClearDepthStencilValue ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ClearDepthStencilValue>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ClearDepthStencilValue>::value,
                          "ClearDepthStencilValue is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ClearRect ) == sizeof( VkClearRect ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ClearRect>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ClearRect>::value, "ClearRect is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ClearValue ) == sizeof( VkClearValue ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ClearValue>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ClearValue>::value, "ClearValue is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageBlit ) == sizeof( VkImageBlit ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageBlit>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageBlit>::value, "ImageBlit is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageCopy ) == sizeof( VkImageCopy ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageCopy>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageCopy>::value, "ImageCopy is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageResolve ) == sizeof( VkImageResolve ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageResolve>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageResolve>::value, "ImageResolve is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageSubresourceLayers ) == sizeof( VkImageSubresourceLayers ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageSubresourceLayers>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageSubresourceLayers>::value,
                          "ImageSubresourceLayers is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassBeginInfo ) == sizeof( VkRenderPassBeginInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassBeginInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassBeginInfo>::value,
                          "RenderPassBeginInfo is not nothrow_move_constructible!" );

//=== VK_VERSION_1_1 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupProperties ) == sizeof( VkPhysicalDeviceSubgroupProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupProperties>::value,
                          "PhysicalDeviceSubgroupProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo ) == sizeof( VkBindBufferMemoryInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo>::value,
                          "BindBufferMemoryInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindImageMemoryInfo ) == sizeof( VkBindImageMemoryInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindImageMemoryInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindImageMemoryInfo>::value,
                          "BindImageMemoryInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeatures ) == sizeof( VkPhysicalDevice16BitStorageFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeatures>::value,
                          "PhysicalDevice16BitStorageFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirements ) == sizeof( VkMemoryDedicatedRequirements ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirements>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirements>::value,
                          "MemoryDedicatedRequirements is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfo ) == sizeof( VkMemoryDedicatedAllocateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfo>::value,
                          "MemoryDedicatedAllocateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfo ) == sizeof( VkMemoryAllocateFlagsInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfo>::value,
                          "MemoryAllocateFlagsInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfo ) == sizeof( VkDeviceGroupRenderPassBeginInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfo>::value,
                          "DeviceGroupRenderPassBeginInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfo ) == sizeof( VkDeviceGroupCommandBufferBeginInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfo>::value,
                          "DeviceGroupCommandBufferBeginInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfo ) == sizeof( VkDeviceGroupSubmitInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfo>::value,
                          "DeviceGroupSubmitInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfo ) == sizeof( VkDeviceGroupBindSparseInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfo>::value,
                          "DeviceGroupBindSparseInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfo ) == sizeof( VkBindBufferMemoryDeviceGroupInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfo>::value,
                          "BindBufferMemoryDeviceGroupInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfo ) == sizeof( VkBindImageMemoryDeviceGroupInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfo>::value,
                          "BindImageMemoryDeviceGroupInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties ) == sizeof( VkPhysicalDeviceGroupProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>::value,
                          "PhysicalDeviceGroupProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfo ) == sizeof( VkDeviceGroupDeviceCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfo>::value,
                          "DeviceGroupDeviceCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2 ) == sizeof( VkBufferMemoryRequirementsInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2>::value,
                          "BufferMemoryRequirementsInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2 ) == sizeof( VkImageMemoryRequirementsInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2>::value,
                          "ImageMemoryRequirementsInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2 ) == sizeof( VkImageSparseMemoryRequirementsInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2>::value,
                          "ImageSparseMemoryRequirementsInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryRequirements2 ) == sizeof( VkMemoryRequirements2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryRequirements2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryRequirements2>::value,
                          "MemoryRequirements2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2 ) == sizeof( VkSparseImageMemoryRequirements2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>::value,
                          "SparseImageMemoryRequirements2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 ) == sizeof( VkPhysicalDeviceFeatures2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2>::value,
                          "PhysicalDeviceFeatures2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2 ) == sizeof( VkPhysicalDeviceProperties2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2>::value,
                          "PhysicalDeviceProperties2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FormatProperties2 ) == sizeof( VkFormatProperties2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FormatProperties2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FormatProperties2>::value,
                          "FormatProperties2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageFormatProperties2 ) == sizeof( VkImageFormatProperties2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageFormatProperties2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageFormatProperties2>::value,
                          "ImageFormatProperties2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2 ) == sizeof( VkPhysicalDeviceImageFormatInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2>::value,
                          "PhysicalDeviceImageFormatInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueueFamilyProperties2 ) == sizeof( VkQueueFamilyProperties2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>::value,
                          "QueueFamilyProperties2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2 ) == sizeof( VkPhysicalDeviceMemoryProperties2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2>::value,
                          "PhysicalDeviceMemoryProperties2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2 ) == sizeof( VkSparseImageFormatProperties2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>::value,
                          "SparseImageFormatProperties2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2 ) == sizeof( VkPhysicalDeviceSparseImageFormatInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2>::value,
                          "PhysicalDeviceSparseImageFormatInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingProperties ) == sizeof( VkPhysicalDevicePointClippingProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingProperties>::value,
                          "PhysicalDevicePointClippingProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfo ) == sizeof( VkRenderPassInputAttachmentAspectCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfo>::value,
                          "RenderPassInputAttachmentAspectCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::InputAttachmentAspectReference ) == sizeof( VkInputAttachmentAspectReference ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::InputAttachmentAspectReference>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::InputAttachmentAspectReference>::value,
                          "InputAttachmentAspectReference is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfo ) == sizeof( VkImageViewUsageCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfo>::value,
                          "ImageViewUsageCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfo ) ==
                            sizeof( VkPipelineTessellationDomainOriginStateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfo>::value,
                          "PipelineTessellationDomainOriginStateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfo ) == sizeof( VkRenderPassMultiviewCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfo>::value,
                          "RenderPassMultiviewCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeatures ) == sizeof( VkPhysicalDeviceMultiviewFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeatures>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeatures>::value,
                          "PhysicalDeviceMultiviewFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewProperties ) == sizeof( VkPhysicalDeviceMultiviewProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewProperties>::value,
                          "PhysicalDeviceMultiviewProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeatures ) == sizeof( VkPhysicalDeviceVariablePointersFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeatures>::value,
                          "PhysicalDeviceVariablePointersFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryFeatures ) == sizeof( VkPhysicalDeviceProtectedMemoryFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryFeatures>::value,
                          "PhysicalDeviceProtectedMemoryFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryProperties ) == sizeof( VkPhysicalDeviceProtectedMemoryProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryProperties>::value,
                          "PhysicalDeviceProtectedMemoryProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceQueueInfo2 ) == sizeof( VkDeviceQueueInfo2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceQueueInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceQueueInfo2>::value,
                          "DeviceQueueInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ProtectedSubmitInfo ) == sizeof( VkProtectedSubmitInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ProtectedSubmitInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ProtectedSubmitInfo>::value,
                          "ProtectedSubmitInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo ) == sizeof( VkSamplerYcbcrConversionCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo>::value,
                          "SamplerYcbcrConversionCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfo ) == sizeof( VkSamplerYcbcrConversionInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfo>::value,
                          "SamplerYcbcrConversionInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfo ) == sizeof( VkBindImagePlaneMemoryInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfo>::value,
                          "BindImagePlaneMemoryInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfo ) == sizeof( VkImagePlaneMemoryRequirementsInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfo>::value,
                          "ImagePlaneMemoryRequirementsInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeatures ) ==
                            sizeof( VkPhysicalDeviceSamplerYcbcrConversionFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeatures>::value,
                          "PhysicalDeviceSamplerYcbcrConversionFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatProperties ) ==
                            sizeof( VkSamplerYcbcrConversionImageFormatProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatProperties>::value,
                          "SamplerYcbcrConversionImageFormatProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion ) == sizeof( VkSamplerYcbcrConversion ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion>::value,
                          "SamplerYcbcrConversion is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate ) == sizeof( VkDescriptorUpdateTemplate ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate>::value,
                          "DescriptorUpdateTemplate is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntry ) == sizeof( VkDescriptorUpdateTemplateEntry ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntry>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntry>::value,
                          "DescriptorUpdateTemplateEntry is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo ) == sizeof( VkDescriptorUpdateTemplateCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo>::value,
                          "DescriptorUpdateTemplateCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalMemoryProperties ) == sizeof( VkExternalMemoryProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalMemoryProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalMemoryProperties>::value,
                          "ExternalMemoryProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfo ) == sizeof( VkPhysicalDeviceExternalImageFormatInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfo>::value,
                          "PhysicalDeviceExternalImageFormatInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalImageFormatProperties ) == sizeof( VkExternalImageFormatProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalImageFormatProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalImageFormatProperties>::value,
                          "ExternalImageFormatProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfo ) == sizeof( VkPhysicalDeviceExternalBufferInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfo>::value,
                          "PhysicalDeviceExternalBufferInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalBufferProperties ) == sizeof( VkExternalBufferProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalBufferProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalBufferProperties>::value,
                          "ExternalBufferProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceIDProperties ) == sizeof( VkPhysicalDeviceIDProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceIDProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceIDProperties>::value,
                          "PhysicalDeviceIDProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfo ) == sizeof( VkExternalMemoryImageCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfo>::value,
                          "ExternalMemoryImageCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfo ) == sizeof( VkExternalMemoryBufferCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfo>::value,
                          "ExternalMemoryBufferCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfo ) == sizeof( VkExportMemoryAllocateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfo>::value,
                          "ExportMemoryAllocateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfo ) == sizeof( VkPhysicalDeviceExternalFenceInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfo>::value,
                          "PhysicalDeviceExternalFenceInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalFenceProperties ) == sizeof( VkExternalFenceProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalFenceProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalFenceProperties>::value,
                          "ExternalFenceProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportFenceCreateInfo ) == sizeof( VkExportFenceCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportFenceCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportFenceCreateInfo>::value,
                          "ExportFenceCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfo ) == sizeof( VkExportSemaphoreCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfo>::value,
                          "ExportSemaphoreCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfo ) == sizeof( VkPhysicalDeviceExternalSemaphoreInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfo>::value,
                          "PhysicalDeviceExternalSemaphoreInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties ) == sizeof( VkExternalSemaphoreProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties>::value,
                          "ExternalSemaphoreProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3Properties ) == sizeof( VkPhysicalDeviceMaintenance3Properties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3Properties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3Properties>::value,
                          "PhysicalDeviceMaintenance3Properties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport ) == sizeof( VkDescriptorSetLayoutSupport ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport>::value,
                          "DescriptorSetLayoutSupport is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParametersFeatures ) == sizeof( VkPhysicalDeviceShaderDrawParametersFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParametersFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParametersFeatures>::value,
                          "PhysicalDeviceShaderDrawParametersFeatures is not nothrow_move_constructible!" );

//=== VK_VERSION_1_2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Features ) == sizeof( VkPhysicalDeviceVulkan11Features ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Features>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Features>::value,
                          "PhysicalDeviceVulkan11Features is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Properties ) == sizeof( VkPhysicalDeviceVulkan11Properties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Properties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Properties>::value,
                          "PhysicalDeviceVulkan11Properties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Features ) == sizeof( VkPhysicalDeviceVulkan12Features ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Features>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Features>::value,
                          "PhysicalDeviceVulkan12Features is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Properties ) == sizeof( VkPhysicalDeviceVulkan12Properties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Properties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Properties>::value,
                          "PhysicalDeviceVulkan12Properties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfo ) == sizeof( VkImageFormatListCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfo>::value,
                          "ImageFormatListCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2 ) == sizeof( VkRenderPassCreateInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2>::value,
                          "RenderPassCreateInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentDescription2 ) == sizeof( VkAttachmentDescription2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentDescription2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentDescription2>::value,
                          "AttachmentDescription2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentReference2 ) == sizeof( VkAttachmentReference2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentReference2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentReference2>::value,
                          "AttachmentReference2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassDescription2 ) == sizeof( VkSubpassDescription2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassDescription2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassDescription2>::value,
                          "SubpassDescription2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassDependency2 ) == sizeof( VkSubpassDependency2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassDependency2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassDependency2>::value,
                          "SubpassDependency2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassBeginInfo ) == sizeof( VkSubpassBeginInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassBeginInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassBeginInfo>::value,
                          "SubpassBeginInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassEndInfo ) == sizeof( VkSubpassEndInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassEndInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassEndInfo>::value,
                          "SubpassEndInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeatures ) == sizeof( VkPhysicalDevice8BitStorageFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeatures>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeatures>::value,
                          "PhysicalDevice8BitStorageFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ConformanceVersion ) == sizeof( VkConformanceVersion ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ConformanceVersion>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ConformanceVersion>::value,
                          "ConformanceVersion is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverProperties ) == sizeof( VkPhysicalDeviceDriverProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverProperties>::value,
                          "PhysicalDeviceDriverProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64Features ) == sizeof( VkPhysicalDeviceShaderAtomicInt64Features ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64Features>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64Features>::value,
                          "PhysicalDeviceShaderAtomicInt64Features is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8Features ) == sizeof( VkPhysicalDeviceShaderFloat16Int8Features ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8Features>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8Features>::value,
                          "PhysicalDeviceShaderFloat16Int8Features is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsProperties ) == sizeof( VkPhysicalDeviceFloatControlsProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsProperties>::value,
                          "PhysicalDeviceFloatControlsProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfo ) == sizeof( VkDescriptorSetLayoutBindingFlagsCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfo>::value,
                          "DescriptorSetLayoutBindingFlagsCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeatures ) == sizeof( VkPhysicalDeviceDescriptorIndexingFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeatures>::value,
                          "PhysicalDeviceDescriptorIndexingFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingProperties ) == sizeof( VkPhysicalDeviceDescriptorIndexingProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingProperties>::value,
                          "PhysicalDeviceDescriptorIndexingProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfo ) ==
                            sizeof( VkDescriptorSetVariableDescriptorCountAllocateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfo>::value,
                          "DescriptorSetVariableDescriptorCountAllocateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupport ) ==
                            sizeof( VkDescriptorSetVariableDescriptorCountLayoutSupport ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupport>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupport>::value,
                          "DescriptorSetVariableDescriptorCountLayoutSupport is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolve ) == sizeof( VkSubpassDescriptionDepthStencilResolve ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolve>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolve>::value,
                          "SubpassDescriptionDepthStencilResolve is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolveProperties ) ==
                            sizeof( VkPhysicalDeviceDepthStencilResolveProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolveProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolveProperties>::value,
                          "PhysicalDeviceDepthStencilResolveProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeatures ) == sizeof( VkPhysicalDeviceScalarBlockLayoutFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeatures>::value,
                          "PhysicalDeviceScalarBlockLayoutFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfo ) == sizeof( VkImageStencilUsageCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfo>::value,
                          "ImageStencilUsageCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfo ) == sizeof( VkSamplerReductionModeCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfo>::value,
                          "SamplerReductionModeCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxProperties ) ==
                            sizeof( VkPhysicalDeviceSamplerFilterMinmaxProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxProperties>::value,
                          "PhysicalDeviceSamplerFilterMinmaxProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeatures ) == sizeof( VkPhysicalDeviceVulkanMemoryModelFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeatures>::value,
                          "PhysicalDeviceVulkanMemoryModelFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeatures ) == sizeof( VkPhysicalDeviceImagelessFramebufferFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeatures>::value,
                          "PhysicalDeviceImagelessFramebufferFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfo ) == sizeof( VkFramebufferAttachmentsCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfo>::value,
                          "FramebufferAttachmentsCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfo ) == sizeof( VkFramebufferAttachmentImageInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfo>::value,
                          "FramebufferAttachmentImageInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfo ) == sizeof( VkRenderPassAttachmentBeginInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfo>::value,
                          "RenderPassAttachmentBeginInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeatures ) ==
                            sizeof( VkPhysicalDeviceUniformBufferStandardLayoutFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeatures>::value,
                          "PhysicalDeviceUniformBufferStandardLayoutFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeatures ) ==
                            sizeof( VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeatures>::value,
                          "PhysicalDeviceShaderSubgroupExtendedTypesFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeatures ) ==
                            sizeof( VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeatures>::value,
                          "PhysicalDeviceSeparateDepthStencilLayoutsFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayout ) == sizeof( VkAttachmentReferenceStencilLayout ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayout>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayout>::value,
                          "AttachmentReferenceStencilLayout is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayout ) == sizeof( VkAttachmentDescriptionStencilLayout ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayout>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayout>::value,
                          "AttachmentDescriptionStencilLayout is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeatures ) == sizeof( VkPhysicalDeviceHostQueryResetFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeatures>::value,
                          "PhysicalDeviceHostQueryResetFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeatures ) == sizeof( VkPhysicalDeviceTimelineSemaphoreFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeatures>::value,
                          "PhysicalDeviceTimelineSemaphoreFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreProperties ) == sizeof( VkPhysicalDeviceTimelineSemaphoreProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreProperties>::value,
                          "PhysicalDeviceTimelineSemaphoreProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfo ) == sizeof( VkSemaphoreTypeCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfo>::value,
                          "SemaphoreTypeCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfo ) == sizeof( VkTimelineSemaphoreSubmitInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfo>::value,
                          "TimelineSemaphoreSubmitInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreWaitInfo ) == sizeof( VkSemaphoreWaitInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreWaitInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreWaitInfo>::value,
                          "SemaphoreWaitInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreSignalInfo ) == sizeof( VkSemaphoreSignalInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreSignalInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreSignalInfo>::value,
                          "SemaphoreSignalInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeatures ) == sizeof( VkPhysicalDeviceBufferDeviceAddressFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeatures>::value,
                          "PhysicalDeviceBufferDeviceAddressFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfo ) == sizeof( VkBufferDeviceAddressInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfo>::value,
                          "BufferDeviceAddressInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfo ) == sizeof( VkBufferOpaqueCaptureAddressCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfo>::value,
                          "BufferOpaqueCaptureAddressCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfo ) == sizeof( VkMemoryOpaqueCaptureAddressAllocateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfo>::value,
                          "MemoryOpaqueCaptureAddressAllocateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfo ) == sizeof( VkDeviceMemoryOpaqueCaptureAddressInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfo>::value,
                          "DeviceMemoryOpaqueCaptureAddressInfo is not nothrow_move_constructible!" );

//=== VK_VERSION_1_3 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Features ) == sizeof( VkPhysicalDeviceVulkan13Features ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Features>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Features>::value,
                          "PhysicalDeviceVulkan13Features is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Properties ) == sizeof( VkPhysicalDeviceVulkan13Properties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Properties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Properties>::value,
                          "PhysicalDeviceVulkan13Properties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfo ) == sizeof( VkPipelineCreationFeedbackCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfo>::value,
                          "PipelineCreationFeedbackCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCreationFeedback ) == sizeof( VkPipelineCreationFeedback ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCreationFeedback>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCreationFeedback>::value,
                          "PipelineCreationFeedback is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeatures ) ==
                            sizeof( VkPhysicalDeviceShaderTerminateInvocationFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeatures>::value,
                          "PhysicalDeviceShaderTerminateInvocationFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceToolProperties ) == sizeof( VkPhysicalDeviceToolProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceToolProperties>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceToolProperties>::value,
                          "PhysicalDeviceToolProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeatures ) ==
                            sizeof( VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeatures>::value,
                          "PhysicalDeviceShaderDemoteToHelperInvocationFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeatures ) == sizeof( VkPhysicalDevicePrivateDataFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeatures>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeatures>::value,
                          "PhysicalDevicePrivateDataFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfo ) == sizeof( VkDevicePrivateDataCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfo>::value,
                          "DevicePrivateDataCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfo ) == sizeof( VkPrivateDataSlotCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfo>::value,
                          "PrivateDataSlotCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PrivateDataSlot ) == sizeof( VkPrivateDataSlot ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PrivateDataSlot>::value,
                          "PrivateDataSlot is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeatures ) ==
                            sizeof( VkPhysicalDevicePipelineCreationCacheControlFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeatures>::value,
                          "PhysicalDevicePipelineCreationCacheControlFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryBarrier2 ) == sizeof( VkMemoryBarrier2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryBarrier2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryBarrier2>::value,
                          "MemoryBarrier2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2 ) == sizeof( VkBufferMemoryBarrier2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2>::value,
                          "BufferMemoryBarrier2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2 ) == sizeof( VkImageMemoryBarrier2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2>::value,
                          "ImageMemoryBarrier2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DependencyInfo ) == sizeof( VkDependencyInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DependencyInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DependencyInfo>::value,
                          "DependencyInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubmitInfo2 ) == sizeof( VkSubmitInfo2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubmitInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubmitInfo2>::value, "SubmitInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfo ) == sizeof( VkSemaphoreSubmitInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfo>::value,
                          "SemaphoreSubmitInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfo ) == sizeof( VkCommandBufferSubmitInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfo>::value,
                          "CommandBufferSubmitInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2Features ) == sizeof( VkPhysicalDeviceSynchronization2Features ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2Features>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2Features>::value,
                          "PhysicalDeviceSynchronization2Features is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures ) ==
                            sizeof( VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures>::value,
                          "PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeatures ) == sizeof( VkPhysicalDeviceImageRobustnessFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeatures>::value,
                          "PhysicalDeviceImageRobustnessFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyBufferInfo2 ) == sizeof( VkCopyBufferInfo2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyBufferInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyBufferInfo2>::value,
                          "CopyBufferInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyImageInfo2 ) == sizeof( VkCopyImageInfo2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyImageInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyImageInfo2>::value,
                          "CopyImageInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2 ) == sizeof( VkCopyBufferToImageInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2>::value,
                          "CopyBufferToImageInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2 ) == sizeof( VkCopyImageToBufferInfo2 ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2>::value,
                          "CopyImageToBufferInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BlitImageInfo2 ) == sizeof( VkBlitImageInfo2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BlitImageInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BlitImageInfo2>::value,
                          "BlitImageInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ResolveImageInfo2 ) == sizeof( VkResolveImageInfo2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ResolveImageInfo2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ResolveImageInfo2>::value,
                          "ResolveImageInfo2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCopy2 ) == sizeof( VkBufferCopy2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCopy2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCopy2>::value, "BufferCopy2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageCopy2 ) == sizeof( VkImageCopy2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageCopy2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageCopy2>::value, "ImageCopy2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageBlit2 ) == sizeof( VkImageBlit2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageBlit2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageBlit2>::value, "ImageBlit2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferImageCopy2 ) == sizeof( VkBufferImageCopy2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferImageCopy2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferImageCopy2>::value,
                          "BufferImageCopy2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageResolve2 ) == sizeof( VkImageResolve2 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageResolve2>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageResolve2>::value, "ImageResolve2 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeatures ) == sizeof( VkPhysicalDeviceSubgroupSizeControlFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeatures>::value,
                          "PhysicalDeviceSubgroupSizeControlFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlProperties ) ==
                            sizeof( VkPhysicalDeviceSubgroupSizeControlProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlProperties>::value,
                          "PhysicalDeviceSubgroupSizeControlProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfo ) ==
                            sizeof( VkPipelineShaderStageRequiredSubgroupSizeCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfo>::value,
                          "PipelineShaderStageRequiredSubgroupSizeCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeatures ) == sizeof( VkPhysicalDeviceInlineUniformBlockFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeatures>::value,
                          "PhysicalDeviceInlineUniformBlockFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockProperties ) == sizeof( VkPhysicalDeviceInlineUniformBlockProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockProperties>::value,
                          "PhysicalDeviceInlineUniformBlockProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlock ) == sizeof( VkWriteDescriptorSetInlineUniformBlock ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlock>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlock>::value,
                          "WriteDescriptorSetInlineUniformBlock is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfo ) == sizeof( VkDescriptorPoolInlineUniformBlockCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfo>::value,
                          "DescriptorPoolInlineUniformBlockCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeatures ) ==
                            sizeof( VkPhysicalDeviceTextureCompressionASTCHDRFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeatures>::value,
                          "PhysicalDeviceTextureCompressionASTCHDRFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderingInfo ) == sizeof( VkRenderingInfo ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderingInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderingInfo>::value, "RenderingInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderingAttachmentInfo ) == sizeof( VkRenderingAttachmentInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderingAttachmentInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderingAttachmentInfo>::value,
                          "RenderingAttachmentInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfo ) == sizeof( VkPipelineRenderingCreateInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfo>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfo>::value,
                          "PipelineRenderingCreateInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeatures ) == sizeof( VkPhysicalDeviceDynamicRenderingFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeatures>::value,
                          "PhysicalDeviceDynamicRenderingFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfo ) == sizeof( VkCommandBufferInheritanceRenderingInfo ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfo>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfo>::value,
                          "CommandBufferInheritanceRenderingInfo is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeatures ) ==
                            sizeof( VkPhysicalDeviceShaderIntegerDotProductFeatures ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeatures>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeatures>::value,
                          "PhysicalDeviceShaderIntegerDotProductFeatures is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductProperties ) ==
                            sizeof( VkPhysicalDeviceShaderIntegerDotProductProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductProperties>::value,
                          "PhysicalDeviceShaderIntegerDotProductProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentProperties ) ==
                            sizeof( VkPhysicalDeviceTexelBufferAlignmentProperties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentProperties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentProperties>::value,
                          "PhysicalDeviceTexelBufferAlignmentProperties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FormatProperties3 ) == sizeof( VkFormatProperties3 ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FormatProperties3>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FormatProperties3>::value,
                          "FormatProperties3 is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Features ) == sizeof( VkPhysicalDeviceMaintenance4Features ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Features>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Features>::value,
                          "PhysicalDeviceMaintenance4Features is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Properties ) == sizeof( VkPhysicalDeviceMaintenance4Properties ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Properties>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Properties>::value,
                          "PhysicalDeviceMaintenance4Properties is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirements ) == sizeof( VkDeviceBufferMemoryRequirements ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirements>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirements>::value,
                          "DeviceBufferMemoryRequirements is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirements ) == sizeof( VkDeviceImageMemoryRequirements ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirements>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirements>::value,
                          "DeviceImageMemoryRequirements is not nothrow_move_constructible!" );

//=== VK_KHR_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceKHR ) == sizeof( VkSurfaceKHR ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceKHR>::value, "SurfaceKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR ) == sizeof( VkSurfaceCapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR>::value,
                          "SurfaceCapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceFormatKHR ) == sizeof( VkSurfaceFormatKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceFormatKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceFormatKHR>::value,
                          "SurfaceFormatKHR is not nothrow_move_constructible!" );

//=== VK_KHR_swapchain ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR ) == sizeof( VkSwapchainCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR>::value,
                          "SwapchainCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainKHR ) == sizeof( VkSwapchainKHR ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainKHR>::value, "SwapchainKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PresentInfoKHR ) == sizeof( VkPresentInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PresentInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PresentInfoKHR>::value,
                          "PresentInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR ) == sizeof( VkImageSwapchainCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR>::value,
                          "ImageSwapchainCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR ) == sizeof( VkBindImageMemorySwapchainInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR>::value,
                          "BindImageMemorySwapchainInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR ) == sizeof( VkAcquireNextImageInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR>::value,
                          "AcquireNextImageInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR ) == sizeof( VkDeviceGroupPresentCapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR>::value,
                          "DeviceGroupPresentCapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR ) == sizeof( VkDeviceGroupPresentInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR>::value,
                          "DeviceGroupPresentInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR ) == sizeof( VkDeviceGroupSwapchainCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR>::value,
                          "DeviceGroupSwapchainCreateInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_display ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayKHR ) == sizeof( VkDisplayKHR ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayKHR>::value, "DisplayKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR ) == sizeof( VkDisplayModeCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR>::value,
                          "DisplayModeCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayModeKHR ) == sizeof( VkDisplayModeKHR ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayModeKHR>::value,
                          "DisplayModeKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayModeParametersKHR ) == sizeof( VkDisplayModeParametersKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayModeParametersKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayModeParametersKHR>::value,
                          "DisplayModeParametersKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR ) == sizeof( VkDisplayModePropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR>::value,
                          "DisplayModePropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR ) == sizeof( VkDisplayPlaneCapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR>::value,
                          "DisplayPlaneCapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR ) == sizeof( VkDisplayPlanePropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR>::value,
                          "DisplayPlanePropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR ) == sizeof( VkDisplayPropertiesKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR>::value,
                          "DisplayPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR ) == sizeof( VkDisplaySurfaceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR>::value,
                          "DisplaySurfaceCreateInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_display_swapchain ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPresentInfoKHR ) == sizeof( VkDisplayPresentInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPresentInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPresentInfoKHR>::value,
                          "DisplayPresentInfoKHR is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_XLIB_KHR )
//=== VK_KHR_xlib_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR ) == sizeof( VkXlibSurfaceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR>::value,
                          "XlibSurfaceCreateInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
//=== VK_KHR_xcb_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR ) == sizeof( VkXcbSurfaceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR>::value,
                          "XcbSurfaceCreateInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
//=== VK_KHR_wayland_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR ) == sizeof( VkWaylandSurfaceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR>::value,
                          "WaylandSurfaceCreateInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
//=== VK_KHR_android_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR ) == sizeof( VkAndroidSurfaceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR>::value,
                          "AndroidSurfaceCreateInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_KHR_win32_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR ) == sizeof( VkWin32SurfaceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR>::value,
                          "Win32SurfaceCreateInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

//=== VK_EXT_debug_report ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT ) == sizeof( VkDebugReportCallbackEXT ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT>::value,
                          "DebugReportCallbackEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT ) == sizeof( VkDebugReportCallbackCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT>::value,
                          "DebugReportCallbackCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_AMD_rasterization_order ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRasterizationStateRasterizationOrderAMD ) ==
                            sizeof( VkPipelineRasterizationStateRasterizationOrderAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateRasterizationOrderAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateRasterizationOrderAMD>::value,
                          "PipelineRasterizationStateRasterizationOrderAMD is not nothrow_move_constructible!" );

//=== VK_EXT_debug_marker ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugMarkerObjectNameInfoEXT ) == sizeof( VkDebugMarkerObjectNameInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugMarkerObjectNameInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugMarkerObjectNameInfoEXT>::value,
                          "DebugMarkerObjectNameInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugMarkerObjectTagInfoEXT ) == sizeof( VkDebugMarkerObjectTagInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugMarkerObjectTagInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugMarkerObjectTagInfoEXT>::value,
                          "DebugMarkerObjectTagInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugMarkerMarkerInfoEXT ) == sizeof( VkDebugMarkerMarkerInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugMarkerMarkerInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugMarkerMarkerInfoEXT>::value,
                          "DebugMarkerMarkerInfoEXT is not nothrow_move_constructible!" );

//=== VK_KHR_video_queue ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoSessionKHR ) == sizeof( VkVideoSessionKHR ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoSessionKHR>::value,
                          "VideoSessionKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR ) == sizeof( VkVideoSessionParametersKHR ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR>::value,
                          "VideoSessionParametersKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueueFamilyQueryResultStatusPropertiesKHR ) == sizeof( VkQueueFamilyQueryResultStatusPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueueFamilyQueryResultStatusPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueueFamilyQueryResultStatusPropertiesKHR>::value,
                          "QueueFamilyQueryResultStatusPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueueFamilyVideoPropertiesKHR ) == sizeof( VkQueueFamilyVideoPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueueFamilyVideoPropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueueFamilyVideoPropertiesKHR>::value,
                          "QueueFamilyVideoPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoProfileInfoKHR ) == sizeof( VkVideoProfileInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoProfileInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoProfileInfoKHR>::value,
                          "VideoProfileInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoProfileListInfoKHR ) == sizeof( VkVideoProfileListInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoProfileListInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoProfileListInfoKHR>::value,
                          "VideoProfileListInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR ) == sizeof( VkVideoCapabilitiesKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR>::value,
                          "VideoCapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoFormatInfoKHR ) == sizeof( VkPhysicalDeviceVideoFormatInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoFormatInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoFormatInfoKHR>::value,
                          "PhysicalDeviceVideoFormatInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR ) == sizeof( VkVideoFormatPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR>::value,
                          "VideoFormatPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoPictureResourceInfoKHR ) == sizeof( VkVideoPictureResourceInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoPictureResourceInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoPictureResourceInfoKHR>::value,
                          "VideoPictureResourceInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoReferenceSlotInfoKHR ) == sizeof( VkVideoReferenceSlotInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoReferenceSlotInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoReferenceSlotInfoKHR>::value,
                          "VideoReferenceSlotInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoSessionMemoryRequirementsKHR ) == sizeof( VkVideoSessionMemoryRequirementsKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoSessionMemoryRequirementsKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoSessionMemoryRequirementsKHR>::value,
                          "VideoSessionMemoryRequirementsKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindVideoSessionMemoryInfoKHR ) == sizeof( VkBindVideoSessionMemoryInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindVideoSessionMemoryInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindVideoSessionMemoryInfoKHR>::value,
                          "BindVideoSessionMemoryInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR ) == sizeof( VkVideoSessionCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR>::value,
                          "VideoSessionCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR ) == sizeof( VkVideoSessionParametersCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR>::value,
                          "VideoSessionParametersCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoSessionParametersUpdateInfoKHR ) == sizeof( VkVideoSessionParametersUpdateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoSessionParametersUpdateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoSessionParametersUpdateInfoKHR>::value,
                          "VideoSessionParametersUpdateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoBeginCodingInfoKHR ) == sizeof( VkVideoBeginCodingInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoBeginCodingInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoBeginCodingInfoKHR>::value,
                          "VideoBeginCodingInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEndCodingInfoKHR ) == sizeof( VkVideoEndCodingInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEndCodingInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEndCodingInfoKHR>::value,
                          "VideoEndCodingInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoCodingControlInfoKHR ) == sizeof( VkVideoCodingControlInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoCodingControlInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoCodingControlInfoKHR>::value,
                          "VideoCodingControlInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_video_decode_queue ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeCapabilitiesKHR ) == sizeof( VkVideoDecodeCapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeCapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeCapabilitiesKHR>::value,
                          "VideoDecodeCapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeUsageInfoKHR ) == sizeof( VkVideoDecodeUsageInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeUsageInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeUsageInfoKHR>::value,
                          "VideoDecodeUsageInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeInfoKHR ) == sizeof( VkVideoDecodeInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeInfoKHR>::value,
                          "VideoDecodeInfoKHR is not nothrow_move_constructible!" );

//=== VK_NV_dedicated_allocation ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DedicatedAllocationImageCreateInfoNV ) == sizeof( VkDedicatedAllocationImageCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DedicatedAllocationImageCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DedicatedAllocationImageCreateInfoNV>::value,
                          "DedicatedAllocationImageCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DedicatedAllocationBufferCreateInfoNV ) == sizeof( VkDedicatedAllocationBufferCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DedicatedAllocationBufferCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DedicatedAllocationBufferCreateInfoNV>::value,
                          "DedicatedAllocationBufferCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DedicatedAllocationMemoryAllocateInfoNV ) == sizeof( VkDedicatedAllocationMemoryAllocateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DedicatedAllocationMemoryAllocateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DedicatedAllocationMemoryAllocateInfoNV>::value,
                          "DedicatedAllocationMemoryAllocateInfoNV is not nothrow_move_constructible!" );

//=== VK_EXT_transform_feedback ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackFeaturesEXT ) == sizeof( VkPhysicalDeviceTransformFeedbackFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackFeaturesEXT>::value,
                          "PhysicalDeviceTransformFeedbackFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceTransformFeedbackPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackPropertiesEXT>::value,
                          "PhysicalDeviceTransformFeedbackPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateInfoEXT ) ==
                            sizeof( VkPipelineRasterizationStateStreamCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateInfoEXT>::value,
                          "PipelineRasterizationStateStreamCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_NVX_binary_import ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CuModuleNVX ) == sizeof( VkCuModuleNVX ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CuModuleNVX>::value, "CuModuleNVX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CuFunctionNVX ) == sizeof( VkCuFunctionNVX ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CuFunctionNVX>::value, "CuFunctionNVX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX ) == sizeof( VkCuModuleCreateInfoNVX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX>::value,
                          "CuModuleCreateInfoNVX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX ) == sizeof( VkCuFunctionCreateInfoNVX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX>::value,
                          "CuFunctionCreateInfoNVX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CuLaunchInfoNVX ) == sizeof( VkCuLaunchInfoNVX ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CuLaunchInfoNVX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CuLaunchInfoNVX>::value,
                          "CuLaunchInfoNVX is not nothrow_move_constructible!" );

//=== VK_NVX_image_view_handle ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX ) == sizeof( VkImageViewHandleInfoNVX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX>::value,
                          "ImageViewHandleInfoNVX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX ) == sizeof( VkImageViewAddressPropertiesNVX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX>::value,
                          "ImageViewAddressPropertiesNVX is not nothrow_move_constructible!" );

//=== VK_KHR_video_encode_h264 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesKHR ) == sizeof( VkVideoEncodeH264CapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesKHR>::value,
                          "VideoEncodeH264CapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264QualityLevelPropertiesKHR ) == sizeof( VkVideoEncodeH264QualityLevelPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264QualityLevelPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264QualityLevelPropertiesKHR>::value,
                          "VideoEncodeH264QualityLevelPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionCreateInfoKHR ) == sizeof( VkVideoEncodeH264SessionCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionCreateInfoKHR>::value,
                          "VideoEncodeH264SessionCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoKHR ) ==
                            sizeof( VkVideoEncodeH264SessionParametersCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoKHR>::value,
                          "VideoEncodeH264SessionParametersCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoKHR ) == sizeof( VkVideoEncodeH264SessionParametersAddInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoKHR>::value,
                          "VideoEncodeH264SessionParametersAddInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersGetInfoKHR ) == sizeof( VkVideoEncodeH264SessionParametersGetInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersGetInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersGetInfoKHR>::value,
                          "VideoEncodeH264SessionParametersGetInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersFeedbackInfoKHR ) ==
                            sizeof( VkVideoEncodeH264SessionParametersFeedbackInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersFeedbackInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersFeedbackInfoKHR>::value,
                          "VideoEncodeH264SessionParametersFeedbackInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264PictureInfoKHR ) == sizeof( VkVideoEncodeH264PictureInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264PictureInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264PictureInfoKHR>::value,
                          "VideoEncodeH264PictureInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoKHR ) == sizeof( VkVideoEncodeH264DpbSlotInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoKHR>::value,
                          "VideoEncodeH264DpbSlotInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoKHR ) == sizeof( VkVideoEncodeH264NaluSliceInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoKHR>::value,
                          "VideoEncodeH264NaluSliceInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoKHR ) == sizeof( VkVideoEncodeH264ProfileInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoKHR>::value,
                          "VideoEncodeH264ProfileInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoKHR ) == sizeof( VkVideoEncodeH264RateControlInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoKHR>::value,
                          "VideoEncodeH264RateControlInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoKHR ) == sizeof( VkVideoEncodeH264RateControlLayerInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoKHR>::value,
                          "VideoEncodeH264RateControlLayerInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264QpKHR ) == sizeof( VkVideoEncodeH264QpKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264QpKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264QpKHR>::value,
                          "VideoEncodeH264QpKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeKHR ) == sizeof( VkVideoEncodeH264FrameSizeKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeKHR>::value,
                          "VideoEncodeH264FrameSizeKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH264GopRemainingFrameInfoKHR ) == sizeof( VkVideoEncodeH264GopRemainingFrameInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH264GopRemainingFrameInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH264GopRemainingFrameInfoKHR>::value,
                          "VideoEncodeH264GopRemainingFrameInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_video_encode_h265 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesKHR ) == sizeof( VkVideoEncodeH265CapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesKHR>::value,
                          "VideoEncodeH265CapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionCreateInfoKHR ) == sizeof( VkVideoEncodeH265SessionCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionCreateInfoKHR>::value,
                          "VideoEncodeH265SessionCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265QualityLevelPropertiesKHR ) == sizeof( VkVideoEncodeH265QualityLevelPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265QualityLevelPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265QualityLevelPropertiesKHR>::value,
                          "VideoEncodeH265QualityLevelPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoKHR ) ==
                            sizeof( VkVideoEncodeH265SessionParametersCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoKHR>::value,
                          "VideoEncodeH265SessionParametersCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoKHR ) == sizeof( VkVideoEncodeH265SessionParametersAddInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoKHR>::value,
                          "VideoEncodeH265SessionParametersAddInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersGetInfoKHR ) == sizeof( VkVideoEncodeH265SessionParametersGetInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersGetInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersGetInfoKHR>::value,
                          "VideoEncodeH265SessionParametersGetInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersFeedbackInfoKHR ) ==
                            sizeof( VkVideoEncodeH265SessionParametersFeedbackInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersFeedbackInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersFeedbackInfoKHR>::value,
                          "VideoEncodeH265SessionParametersFeedbackInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265PictureInfoKHR ) == sizeof( VkVideoEncodeH265PictureInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265PictureInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265PictureInfoKHR>::value,
                          "VideoEncodeH265PictureInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoKHR ) == sizeof( VkVideoEncodeH265DpbSlotInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoKHR>::value,
                          "VideoEncodeH265DpbSlotInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoKHR ) == sizeof( VkVideoEncodeH265NaluSliceSegmentInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoKHR>::value,
                          "VideoEncodeH265NaluSliceSegmentInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoKHR ) == sizeof( VkVideoEncodeH265ProfileInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoKHR>::value,
                          "VideoEncodeH265ProfileInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoKHR ) == sizeof( VkVideoEncodeH265RateControlInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoKHR>::value,
                          "VideoEncodeH265RateControlInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoKHR ) == sizeof( VkVideoEncodeH265RateControlLayerInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoKHR>::value,
                          "VideoEncodeH265RateControlLayerInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265QpKHR ) == sizeof( VkVideoEncodeH265QpKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265QpKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265QpKHR>::value,
                          "VideoEncodeH265QpKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeKHR ) == sizeof( VkVideoEncodeH265FrameSizeKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeKHR>::value,
                          "VideoEncodeH265FrameSizeKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeH265GopRemainingFrameInfoKHR ) == sizeof( VkVideoEncodeH265GopRemainingFrameInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeH265GopRemainingFrameInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeH265GopRemainingFrameInfoKHR>::value,
                          "VideoEncodeH265GopRemainingFrameInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_video_decode_h264 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH264ProfileInfoKHR ) == sizeof( VkVideoDecodeH264ProfileInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH264ProfileInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH264ProfileInfoKHR>::value,
                          "VideoDecodeH264ProfileInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH264CapabilitiesKHR ) == sizeof( VkVideoDecodeH264CapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH264CapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH264CapabilitiesKHR>::value,
                          "VideoDecodeH264CapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersCreateInfoKHR ) ==
                            sizeof( VkVideoDecodeH264SessionParametersCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersCreateInfoKHR>::value,
                          "VideoDecodeH264SessionParametersCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersAddInfoKHR ) == sizeof( VkVideoDecodeH264SessionParametersAddInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersAddInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersAddInfoKHR>::value,
                          "VideoDecodeH264SessionParametersAddInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureInfoKHR ) == sizeof( VkVideoDecodeH264PictureInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureInfoKHR>::value,
                          "VideoDecodeH264PictureInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH264DpbSlotInfoKHR ) == sizeof( VkVideoDecodeH264DpbSlotInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH264DpbSlotInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH264DpbSlotInfoKHR>::value,
                          "VideoDecodeH264DpbSlotInfoKHR is not nothrow_move_constructible!" );

//=== VK_AMD_texture_gather_bias_lod ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::TextureLODGatherFormatPropertiesAMD ) == sizeof( VkTextureLODGatherFormatPropertiesAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::TextureLODGatherFormatPropertiesAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::TextureLODGatherFormatPropertiesAMD>::value,
                          "TextureLODGatherFormatPropertiesAMD is not nothrow_move_constructible!" );

//=== VK_AMD_shader_info ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderResourceUsageAMD ) == sizeof( VkShaderResourceUsageAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ShaderResourceUsageAMD>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderResourceUsageAMD>::value,
                          "ShaderResourceUsageAMD is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderStatisticsInfoAMD ) == sizeof( VkShaderStatisticsInfoAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ShaderStatisticsInfoAMD>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderStatisticsInfoAMD>::value,
                          "ShaderStatisticsInfoAMD is not nothrow_move_constructible!" );

//=== VK_KHR_dynamic_rendering ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR ) ==
                            sizeof( VkRenderingFragmentShadingRateAttachmentInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR>::value,
                          "RenderingFragmentShadingRateAttachmentInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT ) ==
                            sizeof( VkRenderingFragmentDensityMapAttachmentInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT>::value,
                          "RenderingFragmentDensityMapAttachmentInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD ) == sizeof( VkAttachmentSampleCountInfoAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD>::value,
                          "AttachmentSampleCountInfoAMD is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX ) == sizeof( VkMultiviewPerViewAttributesInfoNVX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX>::value,
                          "MultiviewPerViewAttributesInfoNVX is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_GGP )
//=== VK_GGP_stream_descriptor_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP ) == sizeof( VkStreamDescriptorSurfaceCreateInfoGGP ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP>::value,
                          "StreamDescriptorSurfaceCreateInfoGGP is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_GGP*/

//=== VK_NV_corner_sampled_image ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCornerSampledImageFeaturesNV ) == sizeof( VkPhysicalDeviceCornerSampledImageFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCornerSampledImageFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCornerSampledImageFeaturesNV>::value,
                          "PhysicalDeviceCornerSampledImageFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_external_memory_capabilities ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV ) == sizeof( VkExternalImageFormatPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV>::value,
                          "ExternalImageFormatPropertiesNV is not nothrow_move_constructible!" );

//=== VK_NV_external_memory ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoNV ) == sizeof( VkExternalMemoryImageCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoNV>::value,
                          "ExternalMemoryImageCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoNV ) == sizeof( VkExportMemoryAllocateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoNV>::value,
                          "ExportMemoryAllocateInfoNV is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_NV_external_memory_win32 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoNV ) == sizeof( VkImportMemoryWin32HandleInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoNV>::value,
                          "ImportMemoryWin32HandleInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoNV ) == sizeof( VkExportMemoryWin32HandleInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoNV>::value,
                          "ExportMemoryWin32HandleInfoNV is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_NV_win32_keyed_mutex ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoNV ) == sizeof( VkWin32KeyedMutexAcquireReleaseInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoNV>::value,
                          "Win32KeyedMutexAcquireReleaseInfoNV is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

//=== VK_EXT_validation_flags ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ValidationFlagsEXT ) == sizeof( VkValidationFlagsEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ValidationFlagsEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ValidationFlagsEXT>::value,
                          "ValidationFlagsEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_VI_NN )
//=== VK_NN_vi_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN ) == sizeof( VkViSurfaceCreateInfoNN ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN>::value,
                          "ViSurfaceCreateInfoNN is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_VI_NN*/

//=== VK_EXT_astc_decode_mode ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewASTCDecodeModeEXT ) == sizeof( VkImageViewASTCDecodeModeEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewASTCDecodeModeEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewASTCDecodeModeEXT>::value,
                          "ImageViewASTCDecodeModeEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceASTCDecodeFeaturesEXT ) == sizeof( VkPhysicalDeviceASTCDecodeFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceASTCDecodeFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceASTCDecodeFeaturesEXT>::value,
                          "PhysicalDeviceASTCDecodeFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_pipeline_robustness ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeaturesEXT ) ==
                            sizeof( VkPhysicalDevicePipelineRobustnessFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeaturesEXT>::value,
                          "PhysicalDevicePipelineRobustnessFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessPropertiesEXT ) ==
                            sizeof( VkPhysicalDevicePipelineRobustnessPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessPropertiesEXT>::value,
                          "PhysicalDevicePipelineRobustnessPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfoEXT ) == sizeof( VkPipelineRobustnessCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfoEXT>::value,
                          "PipelineRobustnessCreateInfoEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_KHR_external_memory_win32 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoKHR ) == sizeof( VkImportMemoryWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoKHR>::value,
                          "ImportMemoryWin32HandleInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoKHR ) == sizeof( VkExportMemoryWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoKHR>::value,
                          "ExportMemoryWin32HandleInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR ) == sizeof( VkMemoryWin32HandlePropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR>::value,
                          "MemoryWin32HandlePropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryGetWin32HandleInfoKHR ) == sizeof( VkMemoryGetWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryGetWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryGetWin32HandleInfoKHR>::value,
                          "MemoryGetWin32HandleInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

//=== VK_KHR_external_memory_fd ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMemoryFdInfoKHR ) == sizeof( VkImportMemoryFdInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMemoryFdInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMemoryFdInfoKHR>::value,
                          "ImportMemoryFdInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR ) == sizeof( VkMemoryFdPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR>::value,
                          "MemoryFdPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryGetFdInfoKHR ) == sizeof( VkMemoryGetFdInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryGetFdInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryGetFdInfoKHR>::value,
                          "MemoryGetFdInfoKHR is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_KHR_win32_keyed_mutex ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoKHR ) == sizeof( VkWin32KeyedMutexAcquireReleaseInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoKHR>::value,
                          "Win32KeyedMutexAcquireReleaseInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_KHR_external_semaphore_win32 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportSemaphoreWin32HandleInfoKHR ) == sizeof( VkImportSemaphoreWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportSemaphoreWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportSemaphoreWin32HandleInfoKHR>::value,
                          "ImportSemaphoreWin32HandleInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportSemaphoreWin32HandleInfoKHR ) == sizeof( VkExportSemaphoreWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportSemaphoreWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportSemaphoreWin32HandleInfoKHR>::value,
                          "ExportSemaphoreWin32HandleInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::D3D12FenceSubmitInfoKHR ) == sizeof( VkD3D12FenceSubmitInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::D3D12FenceSubmitInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::D3D12FenceSubmitInfoKHR>::value,
                          "D3D12FenceSubmitInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreGetWin32HandleInfoKHR ) == sizeof( VkSemaphoreGetWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreGetWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreGetWin32HandleInfoKHR>::value,
                          "SemaphoreGetWin32HandleInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

//=== VK_KHR_external_semaphore_fd ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportSemaphoreFdInfoKHR ) == sizeof( VkImportSemaphoreFdInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportSemaphoreFdInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportSemaphoreFdInfoKHR>::value,
                          "ImportSemaphoreFdInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreGetFdInfoKHR ) == sizeof( VkSemaphoreGetFdInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreGetFdInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreGetFdInfoKHR>::value,
                          "SemaphoreGetFdInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_push_descriptor ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorPropertiesKHR ) == sizeof( VkPhysicalDevicePushDescriptorPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorPropertiesKHR>::value,
                          "PhysicalDevicePushDescriptorPropertiesKHR is not nothrow_move_constructible!" );

//=== VK_EXT_conditional_rendering ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ConditionalRenderingBeginInfoEXT ) == sizeof( VkConditionalRenderingBeginInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ConditionalRenderingBeginInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ConditionalRenderingBeginInfoEXT>::value,
                          "ConditionalRenderingBeginInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceConditionalRenderingFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceConditionalRenderingFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceConditionalRenderingFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceConditionalRenderingFeaturesEXT>::value,
                          "PhysicalDeviceConditionalRenderingFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceConditionalRenderingInfoEXT ) ==
                            sizeof( VkCommandBufferInheritanceConditionalRenderingInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceConditionalRenderingInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceConditionalRenderingInfoEXT>::value,
                          "CommandBufferInheritanceConditionalRenderingInfoEXT is not nothrow_move_constructible!" );

//=== VK_KHR_incremental_present ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PresentRegionsKHR ) == sizeof( VkPresentRegionsKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PresentRegionsKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PresentRegionsKHR>::value,
                          "PresentRegionsKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PresentRegionKHR ) == sizeof( VkPresentRegionKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PresentRegionKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PresentRegionKHR>::value,
                          "PresentRegionKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RectLayerKHR ) == sizeof( VkRectLayerKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RectLayerKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RectLayerKHR>::value, "RectLayerKHR is not nothrow_move_constructible!" );

//=== VK_NV_clip_space_w_scaling ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ViewportWScalingNV ) == sizeof( VkViewportWScalingNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ViewportWScalingNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ViewportWScalingNV>::value,
                          "ViewportWScalingNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineViewportWScalingStateCreateInfoNV ) == sizeof( VkPipelineViewportWScalingStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineViewportWScalingStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineViewportWScalingStateCreateInfoNV>::value,
                          "PipelineViewportWScalingStateCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_EXT_display_surface_counter ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT ) == sizeof( VkSurfaceCapabilities2EXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT>::value,
                          "SurfaceCapabilities2EXT is not nothrow_move_constructible!" );

//=== VK_EXT_display_control ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPowerInfoEXT ) == sizeof( VkDisplayPowerInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPowerInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPowerInfoEXT>::value,
                          "DisplayPowerInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT ) == sizeof( VkDeviceEventInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT>::value,
                          "DeviceEventInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT ) == sizeof( VkDisplayEventInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT>::value,
                          "DisplayEventInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainCounterCreateInfoEXT ) == sizeof( VkSwapchainCounterCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainCounterCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainCounterCreateInfoEXT>::value,
                          "SwapchainCounterCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_GOOGLE_display_timing ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE ) == sizeof( VkRefreshCycleDurationGOOGLE ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE>::value,
                          "RefreshCycleDurationGOOGLE is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE ) == sizeof( VkPastPresentationTimingGOOGLE ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE>::value,
                          "PastPresentationTimingGOOGLE is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PresentTimesInfoGOOGLE ) == sizeof( VkPresentTimesInfoGOOGLE ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PresentTimesInfoGOOGLE>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PresentTimesInfoGOOGLE>::value,
                          "PresentTimesInfoGOOGLE is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PresentTimeGOOGLE ) == sizeof( VkPresentTimeGOOGLE ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PresentTimeGOOGLE>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PresentTimeGOOGLE>::value,
                          "PresentTimeGOOGLE is not nothrow_move_constructible!" );

//=== VK_NVX_multiview_per_view_attributes ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX ) ==
                            sizeof( VkPhysicalDeviceMultiviewPerViewAttributesPropertiesNVX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX>::value,
                          "PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX is not nothrow_move_constructible!" );

//=== VK_NV_viewport_swizzle ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ViewportSwizzleNV ) == sizeof( VkViewportSwizzleNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ViewportSwizzleNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ViewportSwizzleNV>::value,
                          "ViewportSwizzleNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateInfoNV ) == sizeof( VkPipelineViewportSwizzleStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateInfoNV>::value,
                          "PipelineViewportSwizzleStateCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_EXT_discard_rectangles ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDiscardRectanglePropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceDiscardRectanglePropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiscardRectanglePropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiscardRectanglePropertiesEXT>::value,
                          "PhysicalDeviceDiscardRectanglePropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateInfoEXT ) == sizeof( VkPipelineDiscardRectangleStateCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateInfoEXT>::value,
                          "PipelineDiscardRectangleStateCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_conservative_rasterization ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceConservativeRasterizationPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceConservativeRasterizationPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceConservativeRasterizationPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceConservativeRasterizationPropertiesEXT>::value,
                          "PhysicalDeviceConservativeRasterizationPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateInfoEXT ) ==
                            sizeof( VkPipelineRasterizationConservativeStateCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateInfoEXT>::value,
                          "PipelineRasterizationConservativeStateCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_depth_clip_enable ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipEnableFeaturesEXT ) == sizeof( VkPhysicalDeviceDepthClipEnableFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipEnableFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipEnableFeaturesEXT>::value,
                          "PhysicalDeviceDepthClipEnableFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateInfoEXT ) ==
                            sizeof( VkPipelineRasterizationDepthClipStateCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateInfoEXT>::value,
                          "PipelineRasterizationDepthClipStateCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_hdr_metadata ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::HdrMetadataEXT ) == sizeof( VkHdrMetadataEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::HdrMetadataEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::HdrMetadataEXT>::value,
                          "HdrMetadataEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::XYColorEXT ) == sizeof( VkXYColorEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::XYColorEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::XYColorEXT>::value, "XYColorEXT is not nothrow_move_constructible!" );

//=== VK_IMG_relaxed_line_rasterization ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRelaxedLineRasterizationFeaturesIMG ) ==
                            sizeof( VkPhysicalDeviceRelaxedLineRasterizationFeaturesIMG ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRelaxedLineRasterizationFeaturesIMG>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRelaxedLineRasterizationFeaturesIMG>::value,
                          "PhysicalDeviceRelaxedLineRasterizationFeaturesIMG is not nothrow_move_constructible!" );

//=== VK_KHR_shared_presentable_image ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SharedPresentSurfaceCapabilitiesKHR ) == sizeof( VkSharedPresentSurfaceCapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SharedPresentSurfaceCapabilitiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SharedPresentSurfaceCapabilitiesKHR>::value,
                          "SharedPresentSurfaceCapabilitiesKHR is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_KHR_external_fence_win32 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportFenceWin32HandleInfoKHR ) == sizeof( VkImportFenceWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportFenceWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportFenceWin32HandleInfoKHR>::value,
                          "ImportFenceWin32HandleInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportFenceWin32HandleInfoKHR ) == sizeof( VkExportFenceWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportFenceWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportFenceWin32HandleInfoKHR>::value,
                          "ExportFenceWin32HandleInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FenceGetWin32HandleInfoKHR ) == sizeof( VkFenceGetWin32HandleInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FenceGetWin32HandleInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FenceGetWin32HandleInfoKHR>::value,
                          "FenceGetWin32HandleInfoKHR is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

//=== VK_KHR_external_fence_fd ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportFenceFdInfoKHR ) == sizeof( VkImportFenceFdInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportFenceFdInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportFenceFdInfoKHR>::value,
                          "ImportFenceFdInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FenceGetFdInfoKHR ) == sizeof( VkFenceGetFdInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FenceGetFdInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FenceGetFdInfoKHR>::value,
                          "FenceGetFdInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_performance_query ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryFeaturesKHR ) == sizeof( VkPhysicalDevicePerformanceQueryFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryFeaturesKHR>::value,
                          "PhysicalDevicePerformanceQueryFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryPropertiesKHR ) ==
                            sizeof( VkPhysicalDevicePerformanceQueryPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryPropertiesKHR>::value,
                          "PhysicalDevicePerformanceQueryPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceCounterKHR ) == sizeof( VkPerformanceCounterKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceCounterKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceCounterKHR>::value,
                          "PerformanceCounterKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionKHR ) == sizeof( VkPerformanceCounterDescriptionKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionKHR>::value,
                          "PerformanceCounterDescriptionKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueryPoolPerformanceCreateInfoKHR ) == sizeof( VkQueryPoolPerformanceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceCreateInfoKHR>::value,
                          "QueryPoolPerformanceCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceCounterResultKHR ) == sizeof( VkPerformanceCounterResultKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceCounterResultKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceCounterResultKHR>::value,
                          "PerformanceCounterResultKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AcquireProfilingLockInfoKHR ) == sizeof( VkAcquireProfilingLockInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AcquireProfilingLockInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AcquireProfilingLockInfoKHR>::value,
                          "AcquireProfilingLockInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceQuerySubmitInfoKHR ) == sizeof( VkPerformanceQuerySubmitInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceQuerySubmitInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceQuerySubmitInfoKHR>::value,
                          "PerformanceQuerySubmitInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_get_surface_capabilities2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSurfaceInfo2KHR ) == sizeof( VkPhysicalDeviceSurfaceInfo2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSurfaceInfo2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSurfaceInfo2KHR>::value,
                          "PhysicalDeviceSurfaceInfo2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR ) == sizeof( VkSurfaceCapabilities2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR>::value,
                          "SurfaceCapabilities2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR ) == sizeof( VkSurfaceFormat2KHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR>::value,
                          "SurfaceFormat2KHR is not nothrow_move_constructible!" );

//=== VK_KHR_get_display_properties2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayProperties2KHR ) == sizeof( VkDisplayProperties2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayProperties2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayProperties2KHR>::value,
                          "DisplayProperties2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR ) == sizeof( VkDisplayPlaneProperties2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR>::value,
                          "DisplayPlaneProperties2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR ) == sizeof( VkDisplayModeProperties2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR>::value,
                          "DisplayModeProperties2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPlaneInfo2KHR ) == sizeof( VkDisplayPlaneInfo2KHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPlaneInfo2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPlaneInfo2KHR>::value,
                          "DisplayPlaneInfo2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR ) == sizeof( VkDisplayPlaneCapabilities2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR>::value,
                          "DisplayPlaneCapabilities2KHR is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_IOS_MVK )
//=== VK_MVK_ios_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK ) == sizeof( VkIOSSurfaceCreateInfoMVK ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK>::value,
                          "IOSSurfaceCreateInfoMVK is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
//=== VK_MVK_macos_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK ) == sizeof( VkMacOSSurfaceCreateInfoMVK ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK>::value,
                          "MacOSSurfaceCreateInfoMVK is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

//=== VK_EXT_debug_utils ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugUtilsLabelEXT ) == sizeof( VkDebugUtilsLabelEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugUtilsLabelEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugUtilsLabelEXT>::value,
                          "DebugUtilsLabelEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataEXT ) == sizeof( VkDebugUtilsMessengerCallbackDataEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataEXT>::value,
                          "DebugUtilsMessengerCallbackDataEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT ) == sizeof( VkDebugUtilsMessengerCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT>::value,
                          "DebugUtilsMessengerCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT ) == sizeof( VkDebugUtilsMessengerEXT ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT>::value,
                          "DebugUtilsMessengerEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugUtilsObjectNameInfoEXT ) == sizeof( VkDebugUtilsObjectNameInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugUtilsObjectNameInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugUtilsObjectNameInfoEXT>::value,
                          "DebugUtilsObjectNameInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DebugUtilsObjectTagInfoEXT ) == sizeof( VkDebugUtilsObjectTagInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DebugUtilsObjectTagInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DebugUtilsObjectTagInfoEXT>::value,
                          "DebugUtilsObjectTagInfoEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
//=== VK_ANDROID_external_memory_android_hardware_buffer ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferUsageANDROID ) == sizeof( VkAndroidHardwareBufferUsageANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferUsageANDROID>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferUsageANDROID>::value,
                          "AndroidHardwareBufferUsageANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID ) == sizeof( VkAndroidHardwareBufferPropertiesANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID>::value,
                          "AndroidHardwareBufferPropertiesANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatPropertiesANDROID ) ==
                            sizeof( VkAndroidHardwareBufferFormatPropertiesANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatPropertiesANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatPropertiesANDROID>::value,
                          "AndroidHardwareBufferFormatPropertiesANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportAndroidHardwareBufferInfoANDROID ) == sizeof( VkImportAndroidHardwareBufferInfoANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportAndroidHardwareBufferInfoANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportAndroidHardwareBufferInfoANDROID>::value,
                          "ImportAndroidHardwareBufferInfoANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryGetAndroidHardwareBufferInfoANDROID ) == sizeof( VkMemoryGetAndroidHardwareBufferInfoANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryGetAndroidHardwareBufferInfoANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryGetAndroidHardwareBufferInfoANDROID>::value,
                          "MemoryGetAndroidHardwareBufferInfoANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalFormatANDROID ) == sizeof( VkExternalFormatANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalFormatANDROID>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalFormatANDROID>::value,
                          "ExternalFormatANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatProperties2ANDROID ) ==
                            sizeof( VkAndroidHardwareBufferFormatProperties2ANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatProperties2ANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatProperties2ANDROID>::value,
                          "AndroidHardwareBufferFormatProperties2ANDROID is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
//=== VK_AMDX_shader_enqueue ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueueFeaturesAMDX ) == sizeof( VkPhysicalDeviceShaderEnqueueFeaturesAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueueFeaturesAMDX>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueueFeaturesAMDX>::value,
                          "PhysicalDeviceShaderEnqueueFeaturesAMDX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueuePropertiesAMDX ) == sizeof( VkPhysicalDeviceShaderEnqueuePropertiesAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueuePropertiesAMDX>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueuePropertiesAMDX>::value,
                          "PhysicalDeviceShaderEnqueuePropertiesAMDX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineScratchSizeAMDX ) == sizeof( VkExecutionGraphPipelineScratchSizeAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineScratchSizeAMDX>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineScratchSizeAMDX>::value,
                          "ExecutionGraphPipelineScratchSizeAMDX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineCreateInfoAMDX ) == sizeof( VkExecutionGraphPipelineCreateInfoAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineCreateInfoAMDX>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineCreateInfoAMDX>::value,
                          "ExecutionGraphPipelineCreateInfoAMDX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DispatchGraphInfoAMDX ) == sizeof( VkDispatchGraphInfoAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DispatchGraphInfoAMDX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DispatchGraphInfoAMDX>::value,
                          "DispatchGraphInfoAMDX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DispatchGraphCountInfoAMDX ) == sizeof( VkDispatchGraphCountInfoAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DispatchGraphCountInfoAMDX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DispatchGraphCountInfoAMDX>::value,
                          "DispatchGraphCountInfoAMDX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineShaderStageNodeCreateInfoAMDX ) == sizeof( VkPipelineShaderStageNodeCreateInfoAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineShaderStageNodeCreateInfoAMDX>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineShaderStageNodeCreateInfoAMDX>::value,
                          "PipelineShaderStageNodeCreateInfoAMDX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstAMDX ) == sizeof( VkDeviceOrHostAddressConstAMDX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstAMDX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstAMDX>::value,
                          "DeviceOrHostAddressConstAMDX is not nothrow_move_constructible!" );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

//=== VK_EXT_sample_locations ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SampleLocationEXT ) == sizeof( VkSampleLocationEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SampleLocationEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SampleLocationEXT>::value,
                          "SampleLocationEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SampleLocationsInfoEXT ) == sizeof( VkSampleLocationsInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SampleLocationsInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SampleLocationsInfoEXT>::value,
                          "SampleLocationsInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AttachmentSampleLocationsEXT ) == sizeof( VkAttachmentSampleLocationsEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AttachmentSampleLocationsEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AttachmentSampleLocationsEXT>::value,
                          "AttachmentSampleLocationsEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassSampleLocationsEXT ) == sizeof( VkSubpassSampleLocationsEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassSampleLocationsEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassSampleLocationsEXT>::value,
                          "SubpassSampleLocationsEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassSampleLocationsBeginInfoEXT ) == sizeof( VkRenderPassSampleLocationsBeginInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassSampleLocationsBeginInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassSampleLocationsBeginInfoEXT>::value,
                          "RenderPassSampleLocationsBeginInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineSampleLocationsStateCreateInfoEXT ) == sizeof( VkPipelineSampleLocationsStateCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineSampleLocationsStateCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineSampleLocationsStateCreateInfoEXT>::value,
                          "PipelineSampleLocationsStateCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSampleLocationsPropertiesEXT ) == sizeof( VkPhysicalDeviceSampleLocationsPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSampleLocationsPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSampleLocationsPropertiesEXT>::value,
                          "PhysicalDeviceSampleLocationsPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT ) == sizeof( VkMultisamplePropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT>::value,
                          "MultisamplePropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_blend_operation_advanced ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedFeaturesEXT>::value,
                          "PhysicalDeviceBlendOperationAdvancedFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedPropertiesEXT>::value,
                          "PhysicalDeviceBlendOperationAdvancedPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineColorBlendAdvancedStateCreateInfoEXT ) ==
                            sizeof( VkPipelineColorBlendAdvancedStateCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineColorBlendAdvancedStateCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineColorBlendAdvancedStateCreateInfoEXT>::value,
                          "PipelineColorBlendAdvancedStateCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_NV_fragment_coverage_to_color ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateInfoNV ) == sizeof( VkPipelineCoverageToColorStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateInfoNV>::value,
                          "PipelineCoverageToColorStateCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_KHR_acceleration_structure ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceOrHostAddressKHR ) == sizeof( VkDeviceOrHostAddressKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressKHR>::value,
                          "DeviceOrHostAddressKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstKHR ) == sizeof( VkDeviceOrHostAddressConstKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstKHR>::value,
                          "DeviceOrHostAddressConstKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR ) == sizeof( VkAccelerationStructureBuildRangeInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR>::value,
                          "AccelerationStructureBuildRangeInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AabbPositionsKHR ) == sizeof( VkAabbPositionsKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AabbPositionsKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AabbPositionsKHR>::value,
                          "AabbPositionsKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryTrianglesDataKHR ) ==
                            sizeof( VkAccelerationStructureGeometryTrianglesDataKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryTrianglesDataKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryTrianglesDataKHR>::value,
                          "AccelerationStructureGeometryTrianglesDataKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::TransformMatrixKHR ) == sizeof( VkTransformMatrixKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::TransformMatrixKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::TransformMatrixKHR>::value,
                          "TransformMatrixKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR ) == sizeof( VkAccelerationStructureBuildGeometryInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR>::value,
                          "AccelerationStructureBuildGeometryInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryAabbsDataKHR ) == sizeof( VkAccelerationStructureGeometryAabbsDataKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryAabbsDataKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryAabbsDataKHR>::value,
                          "AccelerationStructureGeometryAabbsDataKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceKHR ) == sizeof( VkAccelerationStructureInstanceKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceKHR>::value,
                          "AccelerationStructureInstanceKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryInstancesDataKHR ) ==
                            sizeof( VkAccelerationStructureGeometryInstancesDataKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryInstancesDataKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryInstancesDataKHR>::value,
                          "AccelerationStructureGeometryInstancesDataKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryDataKHR ) == sizeof( VkAccelerationStructureGeometryDataKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryDataKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryDataKHR>::value,
                          "AccelerationStructureGeometryDataKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryKHR ) == sizeof( VkAccelerationStructureGeometryKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryKHR>::value,
                          "AccelerationStructureGeometryKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR ) == sizeof( VkAccelerationStructureCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR>::value,
                          "AccelerationStructureCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureKHR ) == sizeof( VkAccelerationStructureKHR ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureKHR>::value,
                          "AccelerationStructureKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureKHR ) == sizeof( VkWriteDescriptorSetAccelerationStructureKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureKHR>::value,
                          "WriteDescriptorSetAccelerationStructureKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructureFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceAccelerationStructureFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructureFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructureFeaturesKHR>::value,
                          "PhysicalDeviceAccelerationStructureFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructurePropertiesKHR ) ==
                            sizeof( VkPhysicalDeviceAccelerationStructurePropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructurePropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructurePropertiesKHR>::value,
                          "PhysicalDeviceAccelerationStructurePropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureDeviceAddressInfoKHR ) == sizeof( VkAccelerationStructureDeviceAddressInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureDeviceAddressInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureDeviceAddressInfoKHR>::value,
                          "AccelerationStructureDeviceAddressInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureVersionInfoKHR ) == sizeof( VkAccelerationStructureVersionInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureVersionInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureVersionInfoKHR>::value,
                          "AccelerationStructureVersionInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyAccelerationStructureToMemoryInfoKHR ) == sizeof( VkCopyAccelerationStructureToMemoryInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyAccelerationStructureToMemoryInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyAccelerationStructureToMemoryInfoKHR>::value,
                          "CopyAccelerationStructureToMemoryInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyMemoryToAccelerationStructureInfoKHR ) == sizeof( VkCopyMemoryToAccelerationStructureInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyMemoryToAccelerationStructureInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyMemoryToAccelerationStructureInfoKHR>::value,
                          "CopyMemoryToAccelerationStructureInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyAccelerationStructureInfoKHR ) == sizeof( VkCopyAccelerationStructureInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyAccelerationStructureInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyAccelerationStructureInfoKHR>::value,
                          "CopyAccelerationStructureInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR ) == sizeof( VkAccelerationStructureBuildSizesInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR>::value,
                          "AccelerationStructureBuildSizesInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_ray_tracing_pipeline ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoKHR ) == sizeof( VkRayTracingShaderGroupCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoKHR>::value,
                          "RayTracingShaderGroupCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR ) == sizeof( VkRayTracingPipelineCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR>::value,
                          "RayTracingPipelineCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelineFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceRayTracingPipelineFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelineFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelineFeaturesKHR>::value,
                          "PhysicalDeviceRayTracingPipelineFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelinePropertiesKHR ) ==
                            sizeof( VkPhysicalDeviceRayTracingPipelinePropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelinePropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelinePropertiesKHR>::value,
                          "PhysicalDeviceRayTracingPipelinePropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::StridedDeviceAddressRegionKHR ) == sizeof( VkStridedDeviceAddressRegionKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::StridedDeviceAddressRegionKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::StridedDeviceAddressRegionKHR>::value,
                          "StridedDeviceAddressRegionKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommandKHR ) == sizeof( VkTraceRaysIndirectCommandKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommandKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommandKHR>::value,
                          "TraceRaysIndirectCommandKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RayTracingPipelineInterfaceCreateInfoKHR ) == sizeof( VkRayTracingPipelineInterfaceCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RayTracingPipelineInterfaceCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RayTracingPipelineInterfaceCreateInfoKHR>::value,
                          "RayTracingPipelineInterfaceCreateInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_ray_query ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayQueryFeaturesKHR ) == sizeof( VkPhysicalDeviceRayQueryFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayQueryFeaturesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayQueryFeaturesKHR>::value,
                          "PhysicalDeviceRayQueryFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_NV_framebuffer_mixed_samples ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateInfoNV ) ==
                            sizeof( VkPipelineCoverageModulationStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateInfoNV>::value,
                          "PipelineCoverageModulationStateCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_NV_shader_sm_builtins ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsPropertiesNV ) == sizeof( VkPhysicalDeviceShaderSMBuiltinsPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsPropertiesNV>::value,
                          "PhysicalDeviceShaderSMBuiltinsPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsFeaturesNV ) == sizeof( VkPhysicalDeviceShaderSMBuiltinsFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsFeaturesNV>::value,
                          "PhysicalDeviceShaderSMBuiltinsFeaturesNV is not nothrow_move_constructible!" );

//=== VK_EXT_image_drm_format_modifier ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesListEXT ) == sizeof( VkDrmFormatModifierPropertiesListEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesListEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesListEXT>::value,
                          "DrmFormatModifierPropertiesListEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesEXT ) == sizeof( VkDrmFormatModifierPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesEXT>::value,
                          "DrmFormatModifierPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageDrmFormatModifierInfoEXT ) ==
                            sizeof( VkPhysicalDeviceImageDrmFormatModifierInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageDrmFormatModifierInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageDrmFormatModifierInfoEXT>::value,
                          "PhysicalDeviceImageDrmFormatModifierInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierListCreateInfoEXT ) == sizeof( VkImageDrmFormatModifierListCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierListCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierListCreateInfoEXT>::value,
                          "ImageDrmFormatModifierListCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierExplicitCreateInfoEXT ) ==
                            sizeof( VkImageDrmFormatModifierExplicitCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierExplicitCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierExplicitCreateInfoEXT>::value,
                          "ImageDrmFormatModifierExplicitCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT ) == sizeof( VkImageDrmFormatModifierPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT>::value,
                          "ImageDrmFormatModifierPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesList2EXT ) == sizeof( VkDrmFormatModifierPropertiesList2EXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesList2EXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesList2EXT>::value,
                          "DrmFormatModifierPropertiesList2EXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrmFormatModifierProperties2EXT ) == sizeof( VkDrmFormatModifierProperties2EXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrmFormatModifierProperties2EXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrmFormatModifierProperties2EXT>::value,
                          "DrmFormatModifierProperties2EXT is not nothrow_move_constructible!" );

//=== VK_EXT_validation_cache ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ValidationCacheEXT ) == sizeof( VkValidationCacheEXT ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ValidationCacheEXT>::value,
                          "ValidationCacheEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT ) == sizeof( VkValidationCacheCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT>::value,
                          "ValidationCacheCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderModuleValidationCacheCreateInfoEXT ) == sizeof( VkShaderModuleValidationCacheCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ShaderModuleValidationCacheCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderModuleValidationCacheCreateInfoEXT>::value,
                          "ShaderModuleValidationCacheCreateInfoEXT is not nothrow_move_constructible!" );

#if defined( VK_ENABLE_BETA_EXTENSIONS )
//=== VK_KHR_portability_subset ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetFeaturesKHR ) == sizeof( VkPhysicalDevicePortabilitySubsetFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetFeaturesKHR>::value,
                          "PhysicalDevicePortabilitySubsetFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetPropertiesKHR ) ==
                            sizeof( VkPhysicalDevicePortabilitySubsetPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetPropertiesKHR>::value,
                          "PhysicalDevicePortabilitySubsetPropertiesKHR is not nothrow_move_constructible!" );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

//=== VK_NV_shading_rate_image ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV ) == sizeof( VkShadingRatePaletteNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV>::value,
                          "ShadingRatePaletteNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineViewportShadingRateImageStateCreateInfoNV ) ==
                            sizeof( VkPipelineViewportShadingRateImageStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineViewportShadingRateImageStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineViewportShadingRateImageStateCreateInfoNV>::value,
                          "PipelineViewportShadingRateImageStateCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImageFeaturesNV ) == sizeof( VkPhysicalDeviceShadingRateImageFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImageFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImageFeaturesNV>::value,
                          "PhysicalDeviceShadingRateImageFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImagePropertiesNV ) == sizeof( VkPhysicalDeviceShadingRateImagePropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImagePropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImagePropertiesNV>::value,
                          "PhysicalDeviceShadingRateImagePropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CoarseSampleLocationNV ) == sizeof( VkCoarseSampleLocationNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CoarseSampleLocationNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CoarseSampleLocationNV>::value,
                          "CoarseSampleLocationNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV ) == sizeof( VkCoarseSampleOrderCustomNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV>::value,
                          "CoarseSampleOrderCustomNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineViewportCoarseSampleOrderStateCreateInfoNV ) ==
                            sizeof( VkPipelineViewportCoarseSampleOrderStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineViewportCoarseSampleOrderStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineViewportCoarseSampleOrderStateCreateInfoNV>::value,
                          "PipelineViewportCoarseSampleOrderStateCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_NV_ray_tracing ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoNV ) == sizeof( VkRayTracingShaderGroupCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoNV>::value,
                          "RayTracingShaderGroupCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV ) == sizeof( VkRayTracingPipelineCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV>::value,
                          "RayTracingPipelineCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GeometryTrianglesNV ) == sizeof( VkGeometryTrianglesNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GeometryTrianglesNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GeometryTrianglesNV>::value,
                          "GeometryTrianglesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GeometryAABBNV ) == sizeof( VkGeometryAABBNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GeometryAABBNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GeometryAABBNV>::value,
                          "GeometryAABBNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GeometryDataNV ) == sizeof( VkGeometryDataNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GeometryDataNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GeometryDataNV>::value,
                          "GeometryDataNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GeometryNV ) == sizeof( VkGeometryNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GeometryNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GeometryNV>::value, "GeometryNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureInfoNV ) == sizeof( VkAccelerationStructureInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureInfoNV>::value,
                          "AccelerationStructureInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV ) == sizeof( VkAccelerationStructureCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV>::value,
                          "AccelerationStructureCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureNV ) == sizeof( VkAccelerationStructureNV ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureNV>::value,
                          "AccelerationStructureNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV ) == sizeof( VkBindAccelerationStructureMemoryInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV>::value,
                          "BindAccelerationStructureMemoryInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureNV ) == sizeof( VkWriteDescriptorSetAccelerationStructureNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureNV>::value,
                          "WriteDescriptorSetAccelerationStructureNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsInfoNV ) ==
                            sizeof( VkAccelerationStructureMemoryRequirementsInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsInfoNV>::value,
                          "AccelerationStructureMemoryRequirementsInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPropertiesNV ) == sizeof( VkPhysicalDeviceRayTracingPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPropertiesNV>::value,
                          "PhysicalDeviceRayTracingPropertiesNV is not nothrow_move_constructible!" );

//=== VK_NV_representative_fragment_test ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRepresentativeFragmentTestFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRepresentativeFragmentTestFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRepresentativeFragmentTestFeaturesNV>::value,
                          "PhysicalDeviceRepresentativeFragmentTestFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRepresentativeFragmentTestStateCreateInfoNV ) ==
                            sizeof( VkPipelineRepresentativeFragmentTestStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRepresentativeFragmentTestStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRepresentativeFragmentTestStateCreateInfoNV>::value,
                          "PipelineRepresentativeFragmentTestStateCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_EXT_filter_cubic ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewImageFormatInfoEXT ) == sizeof( VkPhysicalDeviceImageViewImageFormatInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewImageFormatInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewImageFormatInfoEXT>::value,
                          "PhysicalDeviceImageViewImageFormatInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FilterCubicImageViewImageFormatPropertiesEXT ) ==
                            sizeof( VkFilterCubicImageViewImageFormatPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FilterCubicImageViewImageFormatPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FilterCubicImageViewImageFormatPropertiesEXT>::value,
                          "FilterCubicImageViewImageFormatPropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_external_memory_host ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMemoryHostPointerInfoEXT ) == sizeof( VkImportMemoryHostPointerInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMemoryHostPointerInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMemoryHostPointerInfoEXT>::value,
                          "ImportMemoryHostPointerInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT ) == sizeof( VkMemoryHostPointerPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT>::value,
                          "MemoryHostPointerPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryHostPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceExternalMemoryHostPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryHostPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryHostPropertiesEXT>::value,
                          "PhysicalDeviceExternalMemoryHostPropertiesEXT is not nothrow_move_constructible!" );

//=== VK_KHR_shader_clock ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderClockFeaturesKHR ) == sizeof( VkPhysicalDeviceShaderClockFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderClockFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderClockFeaturesKHR>::value,
                          "PhysicalDeviceShaderClockFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_AMD_pipeline_compiler_control ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCompilerControlCreateInfoAMD ) == sizeof( VkPipelineCompilerControlCreateInfoAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCompilerControlCreateInfoAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCompilerControlCreateInfoAMD>::value,
                          "PipelineCompilerControlCreateInfoAMD is not nothrow_move_constructible!" );

//=== VK_AMD_shader_core_properties ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD ) == sizeof( VkPhysicalDeviceShaderCorePropertiesAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD>::value,
                          "PhysicalDeviceShaderCorePropertiesAMD is not nothrow_move_constructible!" );

//=== VK_KHR_video_decode_h265 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoKHR ) == sizeof( VkVideoDecodeH265ProfileInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoKHR>::value,
                          "VideoDecodeH265ProfileInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesKHR ) == sizeof( VkVideoDecodeH265CapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesKHR>::value,
                          "VideoDecodeH265CapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoKHR ) ==
                            sizeof( VkVideoDecodeH265SessionParametersCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoKHR>::value,
                          "VideoDecodeH265SessionParametersCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoKHR ) == sizeof( VkVideoDecodeH265SessionParametersAddInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoKHR>::value,
                          "VideoDecodeH265SessionParametersAddInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoKHR ) == sizeof( VkVideoDecodeH265PictureInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoKHR>::value,
                          "VideoDecodeH265PictureInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoKHR ) == sizeof( VkVideoDecodeH265DpbSlotInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoKHR>::value,
                          "VideoDecodeH265DpbSlotInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_global_priority ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoKHR ) == sizeof( VkDeviceQueueGlobalPriorityCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoKHR>::value,
                          "DeviceQueueGlobalPriorityCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesKHR>::value,
                          "PhysicalDeviceGlobalPriorityQueryFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesKHR ) == sizeof( VkQueueFamilyGlobalPriorityPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesKHR>::value,
                          "QueueFamilyGlobalPriorityPropertiesKHR is not nothrow_move_constructible!" );

//=== VK_AMD_memory_overallocation_behavior ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD ) == sizeof( VkDeviceMemoryOverallocationCreateInfoAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD>::value,
                          "DeviceMemoryOverallocationCreateInfoAMD is not nothrow_move_constructible!" );

//=== VK_EXT_vertex_attribute_divisor ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT>::value,
                          "PhysicalDeviceVertexAttributeDivisorPropertiesEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_GGP )
//=== VK_GGP_frame_token ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP ) == sizeof( VkPresentFrameTokenGGP ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP>::value,
                          "PresentFrameTokenGGP is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_GGP*/

//=== VK_NV_compute_shader_derivatives ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceComputeShaderDerivativesFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesNV>::value,
                          "PhysicalDeviceComputeShaderDerivativesFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_mesh_shader ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesNV ) == sizeof( VkPhysicalDeviceMeshShaderFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesNV>::value,
                          "PhysicalDeviceMeshShaderFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesNV ) == sizeof( VkPhysicalDeviceMeshShaderPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesNV>::value,
                          "PhysicalDeviceMeshShaderPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandNV ) == sizeof( VkDrawMeshTasksIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandNV>::value,
                          "DrawMeshTasksIndirectCommandNV is not nothrow_move_constructible!" );

//=== VK_NV_shader_image_footprint ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageFootprintFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceShaderImageFootprintFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageFootprintFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageFootprintFeaturesNV>::value,
                          "PhysicalDeviceShaderImageFootprintFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_scissor_exclusive ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineViewportExclusiveScissorStateCreateInfoNV ) ==
                            sizeof( VkPipelineViewportExclusiveScissorStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineViewportExclusiveScissorStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineViewportExclusiveScissorStateCreateInfoNV>::value,
                          "PipelineViewportExclusiveScissorStateCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExclusiveScissorFeaturesNV ) == sizeof( VkPhysicalDeviceExclusiveScissorFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExclusiveScissorFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExclusiveScissorFeaturesNV>::value,
                          "PhysicalDeviceExclusiveScissorFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_device_diagnostic_checkpoints ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV ) == sizeof( VkQueueFamilyCheckpointPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV>::value,
                          "QueueFamilyCheckpointPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CheckpointDataNV ) == sizeof( VkCheckpointDataNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CheckpointDataNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CheckpointDataNV>::value,
                          "CheckpointDataNV is not nothrow_move_constructible!" );

//=== VK_INTEL_shader_integer_functions2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL ) ==
                            sizeof( VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL>::value,
                          "PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL is not nothrow_move_constructible!" );

//=== VK_INTEL_performance_query ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceValueDataINTEL ) == sizeof( VkPerformanceValueDataINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceValueDataINTEL>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceValueDataINTEL>::value,
                          "PerformanceValueDataINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceValueINTEL ) == sizeof( VkPerformanceValueINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceValueINTEL>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceValueINTEL>::value,
                          "PerformanceValueINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::InitializePerformanceApiInfoINTEL ) == sizeof( VkInitializePerformanceApiInfoINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::InitializePerformanceApiInfoINTEL>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::InitializePerformanceApiInfoINTEL>::value,
                          "InitializePerformanceApiInfoINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueryPoolPerformanceQueryCreateInfoINTEL ) == sizeof( VkQueryPoolPerformanceQueryCreateInfoINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceQueryCreateInfoINTEL>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceQueryCreateInfoINTEL>::value,
                          "QueryPoolPerformanceQueryCreateInfoINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceMarkerInfoINTEL ) == sizeof( VkPerformanceMarkerInfoINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceMarkerInfoINTEL>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceMarkerInfoINTEL>::value,
                          "PerformanceMarkerInfoINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceStreamMarkerInfoINTEL ) == sizeof( VkPerformanceStreamMarkerInfoINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceStreamMarkerInfoINTEL>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceStreamMarkerInfoINTEL>::value,
                          "PerformanceStreamMarkerInfoINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceOverrideInfoINTEL ) == sizeof( VkPerformanceOverrideInfoINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceOverrideInfoINTEL>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceOverrideInfoINTEL>::value,
                          "PerformanceOverrideInfoINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL ) == sizeof( VkPerformanceConfigurationAcquireInfoINTEL ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL>::value,
                          "PerformanceConfigurationAcquireInfoINTEL is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL ) == sizeof( VkPerformanceConfigurationINTEL ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL>::value,
                          "PerformanceConfigurationINTEL is not nothrow_move_constructible!" );

//=== VK_EXT_pci_bus_info ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePCIBusInfoPropertiesEXT ) == sizeof( VkPhysicalDevicePCIBusInfoPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePCIBusInfoPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePCIBusInfoPropertiesEXT>::value,
                          "PhysicalDevicePCIBusInfoPropertiesEXT is not nothrow_move_constructible!" );

//=== VK_AMD_display_native_hdr ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DisplayNativeHdrSurfaceCapabilitiesAMD ) == sizeof( VkDisplayNativeHdrSurfaceCapabilitiesAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DisplayNativeHdrSurfaceCapabilitiesAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DisplayNativeHdrSurfaceCapabilitiesAMD>::value,
                          "DisplayNativeHdrSurfaceCapabilitiesAMD is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainDisplayNativeHdrCreateInfoAMD ) == sizeof( VkSwapchainDisplayNativeHdrCreateInfoAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainDisplayNativeHdrCreateInfoAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainDisplayNativeHdrCreateInfoAMD>::value,
                          "SwapchainDisplayNativeHdrCreateInfoAMD is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_FUCHSIA )
//=== VK_FUCHSIA_imagepipe_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA ) == sizeof( VkImagePipeSurfaceCreateInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA>::value,
                          "ImagePipeSurfaceCreateInfoFUCHSIA is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
//=== VK_EXT_metal_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT ) == sizeof( VkMetalSurfaceCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT>::value,
                          "MetalSurfaceCreateInfoEXT is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_METAL_EXT*/

//=== VK_EXT_fragment_density_map ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceFragmentDensityMapFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapFeaturesEXT>::value,
                          "PhysicalDeviceFragmentDensityMapFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceFragmentDensityMapPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapPropertiesEXT>::value,
                          "PhysicalDeviceFragmentDensityMapPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT ) == sizeof( VkRenderPassFragmentDensityMapCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT>::value,
                          "RenderPassFragmentDensityMapCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_KHR_fragment_shading_rate ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR ) == sizeof( VkFragmentShadingRateAttachmentInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR>::value,
                          "FragmentShadingRateAttachmentInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR ) ==
                            sizeof( VkPipelineFragmentShadingRateStateCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR>::value,
                          "PipelineFragmentShadingRateStateCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceFragmentShadingRateFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR>::value,
                          "PhysicalDeviceFragmentShadingRateFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR ) ==
                            sizeof( VkPhysicalDeviceFragmentShadingRatePropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR>::value,
                          "PhysicalDeviceFragmentShadingRatePropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR ) == sizeof( VkPhysicalDeviceFragmentShadingRateKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR>::value,
                          "PhysicalDeviceFragmentShadingRateKHR is not nothrow_move_constructible!" );

//=== VK_AMD_shader_core_properties2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD ) == sizeof( VkPhysicalDeviceShaderCoreProperties2AMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD>::value,
                          "PhysicalDeviceShaderCoreProperties2AMD is not nothrow_move_constructible!" );

//=== VK_AMD_device_coherent_memory ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD ) == sizeof( VkPhysicalDeviceCoherentMemoryFeaturesAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD>::value,
                          "PhysicalDeviceCoherentMemoryFeaturesAMD is not nothrow_move_constructible!" );

//=== VK_KHR_dynamic_rendering_local_read ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>::value,
                          "PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderingAttachmentLocationInfoKHR ) == sizeof( VkRenderingAttachmentLocationInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderingAttachmentLocationInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderingAttachmentLocationInfoKHR>::value,
                          "RenderingAttachmentLocationInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderingInputAttachmentIndexInfoKHR ) == sizeof( VkRenderingInputAttachmentIndexInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderingInputAttachmentIndexInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderingInputAttachmentIndexInfoKHR>::value,
                          "RenderingInputAttachmentIndexInfoKHR is not nothrow_move_constructible!" );

//=== VK_EXT_shader_image_atomic_int64 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>::value,
                          "PhysicalDeviceShaderImageAtomicInt64FeaturesEXT is not nothrow_move_constructible!" );

//=== VK_KHR_shader_quad_control ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderQuadControlFeaturesKHR ) == sizeof( VkPhysicalDeviceShaderQuadControlFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderQuadControlFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderQuadControlFeaturesKHR>::value,
                          "PhysicalDeviceShaderQuadControlFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_EXT_memory_budget ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryBudgetPropertiesEXT ) == sizeof( VkPhysicalDeviceMemoryBudgetPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryBudgetPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryBudgetPropertiesEXT>::value,
                          "PhysicalDeviceMemoryBudgetPropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_memory_priority ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryPriorityFeaturesEXT ) == sizeof( VkPhysicalDeviceMemoryPriorityFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryPriorityFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryPriorityFeaturesEXT>::value,
                          "PhysicalDeviceMemoryPriorityFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryPriorityAllocateInfoEXT ) == sizeof( VkMemoryPriorityAllocateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryPriorityAllocateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryPriorityAllocateInfoEXT>::value,
                          "MemoryPriorityAllocateInfoEXT is not nothrow_move_constructible!" );

//=== VK_KHR_surface_protected_capabilities ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceProtectedCapabilitiesKHR ) == sizeof( VkSurfaceProtectedCapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceProtectedCapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceProtectedCapabilitiesKHR>::value,
                          "SurfaceProtectedCapabilitiesKHR is not nothrow_move_constructible!" );

//=== VK_NV_dedicated_allocation_image_aliasing ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV>::value,
                          "PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV is not nothrow_move_constructible!" );

//=== VK_EXT_buffer_device_address ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceBufferDeviceAddressFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesEXT>::value,
                          "PhysicalDeviceBufferDeviceAddressFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferDeviceAddressCreateInfoEXT ) == sizeof( VkBufferDeviceAddressCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferDeviceAddressCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferDeviceAddressCreateInfoEXT>::value,
                          "BufferDeviceAddressCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_validation_features ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ValidationFeaturesEXT ) == sizeof( VkValidationFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ValidationFeaturesEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ValidationFeaturesEXT>::value,
                          "ValidationFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_KHR_present_wait ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWaitFeaturesKHR ) == sizeof( VkPhysicalDevicePresentWaitFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWaitFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWaitFeaturesKHR>::value,
                          "PhysicalDevicePresentWaitFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_NV_cooperative_matrix ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV ) == sizeof( VkCooperativeMatrixPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV>::value,
                          "CooperativeMatrixPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesNV ) == sizeof( VkPhysicalDeviceCooperativeMatrixFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesNV>::value,
                          "PhysicalDeviceCooperativeMatrixFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesNV ) ==
                            sizeof( VkPhysicalDeviceCooperativeMatrixPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesNV>::value,
                          "PhysicalDeviceCooperativeMatrixPropertiesNV is not nothrow_move_constructible!" );

//=== VK_NV_coverage_reduction_mode ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCoverageReductionModeFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceCoverageReductionModeFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoverageReductionModeFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoverageReductionModeFeaturesNV>::value,
                          "PhysicalDeviceCoverageReductionModeFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateInfoNV ) == sizeof( VkPipelineCoverageReductionStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateInfoNV>::value,
                          "PipelineCoverageReductionStateCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV ) == sizeof( VkFramebufferMixedSamplesCombinationNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV>::value,
                          "FramebufferMixedSamplesCombinationNV is not nothrow_move_constructible!" );

//=== VK_EXT_fragment_shader_interlock ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderInterlockFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderInterlockFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderInterlockFeaturesEXT>::value,
                          "PhysicalDeviceFragmentShaderInterlockFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_ycbcr_image_arrays ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrImageArraysFeaturesEXT ) == sizeof( VkPhysicalDeviceYcbcrImageArraysFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrImageArraysFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrImageArraysFeaturesEXT>::value,
                          "PhysicalDeviceYcbcrImageArraysFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_provoking_vertex ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexFeaturesEXT ) == sizeof( VkPhysicalDeviceProvokingVertexFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexFeaturesEXT>::value,
                          "PhysicalDeviceProvokingVertexFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexPropertiesEXT ) == sizeof( VkPhysicalDeviceProvokingVertexPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexPropertiesEXT>::value,
                          "PhysicalDeviceProvokingVertexPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRasterizationProvokingVertexStateCreateInfoEXT ) ==
                            sizeof( VkPipelineRasterizationProvokingVertexStateCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRasterizationProvokingVertexStateCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRasterizationProvokingVertexStateCreateInfoEXT>::value,
                          "PipelineRasterizationProvokingVertexStateCreateInfoEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
//=== VK_EXT_full_screen_exclusive ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveInfoEXT ) == sizeof( VkSurfaceFullScreenExclusiveInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveInfoEXT>::value,
                          "SurfaceFullScreenExclusiveInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesFullScreenExclusiveEXT ) == sizeof( VkSurfaceCapabilitiesFullScreenExclusiveEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesFullScreenExclusiveEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesFullScreenExclusiveEXT>::value,
                          "SurfaceCapabilitiesFullScreenExclusiveEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveWin32InfoEXT ) == sizeof( VkSurfaceFullScreenExclusiveWin32InfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveWin32InfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveWin32InfoEXT>::value,
                          "SurfaceFullScreenExclusiveWin32InfoEXT is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

//=== VK_EXT_headless_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT ) == sizeof( VkHeadlessSurfaceCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT>::value,
                          "HeadlessSurfaceCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_shader_atomic_float ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT ) == sizeof( VkPhysicalDeviceShaderAtomicFloatFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT>::value,
                          "PhysicalDeviceShaderAtomicFloatFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_extended_dynamic_state ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicStateFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceExtendedDynamicStateFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicStateFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicStateFeaturesEXT>::value,
                          "PhysicalDeviceExtendedDynamicStateFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_KHR_deferred_host_operations ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeferredOperationKHR ) == sizeof( VkDeferredOperationKHR ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeferredOperationKHR>::value,
                          "DeferredOperationKHR is not nothrow_move_constructible!" );

//=== VK_KHR_pipeline_executable_properties ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR ) ==
                            sizeof( VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR>::value,
                          "PhysicalDevicePipelineExecutablePropertiesFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineInfoKHR ) == sizeof( VkPipelineInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineInfoKHR>::value,
                          "PipelineInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR ) == sizeof( VkPipelineExecutablePropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR>::value,
                          "PipelineExecutablePropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineExecutableInfoKHR ) == sizeof( VkPipelineExecutableInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineExecutableInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineExecutableInfoKHR>::value,
                          "PipelineExecutableInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticValueKHR ) == sizeof( VkPipelineExecutableStatisticValueKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticValueKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticValueKHR>::value,
                          "PipelineExecutableStatisticValueKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR ) == sizeof( VkPipelineExecutableStatisticKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR>::value,
                          "PipelineExecutableStatisticKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR ) ==
                            sizeof( VkPipelineExecutableInternalRepresentationKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR>::value,
                          "PipelineExecutableInternalRepresentationKHR is not nothrow_move_constructible!" );

//=== VK_EXT_host_image_copy ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyFeaturesEXT ) == sizeof( VkPhysicalDeviceHostImageCopyFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyFeaturesEXT>::value,
                          "PhysicalDeviceHostImageCopyFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyPropertiesEXT ) == sizeof( VkPhysicalDeviceHostImageCopyPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyPropertiesEXT>::value,
                          "PhysicalDeviceHostImageCopyPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryToImageCopyEXT ) == sizeof( VkMemoryToImageCopyEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryToImageCopyEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryToImageCopyEXT>::value,
                          "MemoryToImageCopyEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageToMemoryCopyEXT ) == sizeof( VkImageToMemoryCopyEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageToMemoryCopyEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageToMemoryCopyEXT>::value,
                          "ImageToMemoryCopyEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyMemoryToImageInfoEXT ) == sizeof( VkCopyMemoryToImageInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyMemoryToImageInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyMemoryToImageInfoEXT>::value,
                          "CopyMemoryToImageInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyImageToMemoryInfoEXT ) == sizeof( VkCopyImageToMemoryInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyImageToMemoryInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyImageToMemoryInfoEXT>::value,
                          "CopyImageToMemoryInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyImageToImageInfoEXT ) == sizeof( VkCopyImageToImageInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyImageToImageInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyImageToImageInfoEXT>::value,
                          "CopyImageToImageInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::HostImageLayoutTransitionInfoEXT ) == sizeof( VkHostImageLayoutTransitionInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::HostImageLayoutTransitionInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::HostImageLayoutTransitionInfoEXT>::value,
                          "HostImageLayoutTransitionInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubresourceHostMemcpySizeEXT ) == sizeof( VkSubresourceHostMemcpySizeEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubresourceHostMemcpySizeEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubresourceHostMemcpySizeEXT>::value,
                          "SubresourceHostMemcpySizeEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::HostImageCopyDevicePerformanceQueryEXT ) == sizeof( VkHostImageCopyDevicePerformanceQueryEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::HostImageCopyDevicePerformanceQueryEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::HostImageCopyDevicePerformanceQueryEXT>::value,
                          "HostImageCopyDevicePerformanceQueryEXT is not nothrow_move_constructible!" );

//=== VK_KHR_map_memory2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryMapInfoKHR ) == sizeof( VkMemoryMapInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryMapInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryMapInfoKHR>::value,
                          "MemoryMapInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryUnmapInfoKHR ) == sizeof( VkMemoryUnmapInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryUnmapInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryUnmapInfoKHR>::value,
                          "MemoryUnmapInfoKHR is not nothrow_move_constructible!" );

//=== VK_EXT_map_memory_placed ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedFeaturesEXT ) == sizeof( VkPhysicalDeviceMapMemoryPlacedFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedFeaturesEXT>::value,
                          "PhysicalDeviceMapMemoryPlacedFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedPropertiesEXT ) == sizeof( VkPhysicalDeviceMapMemoryPlacedPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedPropertiesEXT>::value,
                          "PhysicalDeviceMapMemoryPlacedPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryMapPlacedInfoEXT ) == sizeof( VkMemoryMapPlacedInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryMapPlacedInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryMapPlacedInfoEXT>::value,
                          "MemoryMapPlacedInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_shader_atomic_float2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>::value,
                          "PhysicalDeviceShaderAtomicFloat2FeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_surface_maintenance1 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfacePresentModeEXT ) == sizeof( VkSurfacePresentModeEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfacePresentModeEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfacePresentModeEXT>::value,
                          "SurfacePresentModeEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfacePresentScalingCapabilitiesEXT ) == sizeof( VkSurfacePresentScalingCapabilitiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfacePresentScalingCapabilitiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfacePresentScalingCapabilitiesEXT>::value,
                          "SurfacePresentScalingCapabilitiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfacePresentModeCompatibilityEXT ) == sizeof( VkSurfacePresentModeCompatibilityEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfacePresentModeCompatibilityEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfacePresentModeCompatibilityEXT>::value,
                          "SurfacePresentModeCompatibilityEXT is not nothrow_move_constructible!" );

//=== VK_EXT_swapchain_maintenance1 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSwapchainMaintenance1FeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>::value,
                          "PhysicalDeviceSwapchainMaintenance1FeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainPresentFenceInfoEXT ) == sizeof( VkSwapchainPresentFenceInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainPresentFenceInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainPresentFenceInfoEXT>::value,
                          "SwapchainPresentFenceInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainPresentModesCreateInfoEXT ) == sizeof( VkSwapchainPresentModesCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainPresentModesCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainPresentModesCreateInfoEXT>::value,
                          "SwapchainPresentModesCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainPresentModeInfoEXT ) == sizeof( VkSwapchainPresentModeInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainPresentModeInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainPresentModeInfoEXT>::value,
                          "SwapchainPresentModeInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainPresentScalingCreateInfoEXT ) == sizeof( VkSwapchainPresentScalingCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainPresentScalingCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainPresentScalingCreateInfoEXT>::value,
                          "SwapchainPresentScalingCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ReleaseSwapchainImagesInfoEXT ) == sizeof( VkReleaseSwapchainImagesInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ReleaseSwapchainImagesInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ReleaseSwapchainImagesInfoEXT>::value,
                          "ReleaseSwapchainImagesInfoEXT is not nothrow_move_constructible!" );

//=== VK_NV_device_generated_commands ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesNV ) ==
                            sizeof( VkPhysicalDeviceDeviceGeneratedCommandsPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesNV>::value,
                          "PhysicalDeviceDeviceGeneratedCommandsPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesNV>::value,
                          "PhysicalDeviceDeviceGeneratedCommandsFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GraphicsShaderGroupCreateInfoNV ) == sizeof( VkGraphicsShaderGroupCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GraphicsShaderGroupCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GraphicsShaderGroupCreateInfoNV>::value,
                          "GraphicsShaderGroupCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GraphicsPipelineShaderGroupsCreateInfoNV ) == sizeof( VkGraphicsPipelineShaderGroupsCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GraphicsPipelineShaderGroupsCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GraphicsPipelineShaderGroupsCreateInfoNV>::value,
                          "GraphicsPipelineShaderGroupsCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindShaderGroupIndirectCommandNV ) == sizeof( VkBindShaderGroupIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindShaderGroupIndirectCommandNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindShaderGroupIndirectCommandNV>::value,
                          "BindShaderGroupIndirectCommandNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandNV ) == sizeof( VkBindIndexBufferIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandNV>::value,
                          "BindIndexBufferIndirectCommandNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandNV ) == sizeof( VkBindVertexBufferIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandNV>::value,
                          "BindVertexBufferIndirectCommandNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SetStateFlagsIndirectCommandNV ) == sizeof( VkSetStateFlagsIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SetStateFlagsIndirectCommandNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SetStateFlagsIndirectCommandNV>::value,
                          "SetStateFlagsIndirectCommandNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV ) == sizeof( VkIndirectCommandsLayoutNV ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV>::value,
                          "IndirectCommandsLayoutNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::IndirectCommandsStreamNV ) == sizeof( VkIndirectCommandsStreamNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::IndirectCommandsStreamNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::IndirectCommandsStreamNV>::value,
                          "IndirectCommandsStreamNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenNV ) == sizeof( VkIndirectCommandsLayoutTokenNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenNV>::value,
                          "IndirectCommandsLayoutTokenNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV ) == sizeof( VkIndirectCommandsLayoutCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV>::value,
                          "IndirectCommandsLayoutCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoNV ) == sizeof( VkGeneratedCommandsInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoNV>::value,
                          "GeneratedCommandsInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoNV ) == sizeof( VkGeneratedCommandsMemoryRequirementsInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoNV>::value,
                          "GeneratedCommandsMemoryRequirementsInfoNV is not nothrow_move_constructible!" );

//=== VK_NV_inherited_viewport_scissor ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceInheritedViewportScissorFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceInheritedViewportScissorFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceInheritedViewportScissorFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceInheritedViewportScissorFeaturesNV>::value,
                          "PhysicalDeviceInheritedViewportScissorFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceViewportScissorInfoNV ) ==
                            sizeof( VkCommandBufferInheritanceViewportScissorInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceViewportScissorInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceViewportScissorInfoNV>::value,
                          "CommandBufferInheritanceViewportScissorInfoNV is not nothrow_move_constructible!" );

//=== VK_EXT_texel_buffer_alignment ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentFeaturesEXT>::value,
                          "PhysicalDeviceTexelBufferAlignmentFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_QCOM_render_pass_transform ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassTransformBeginInfoQCOM ) == sizeof( VkRenderPassTransformBeginInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassTransformBeginInfoQCOM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassTransformBeginInfoQCOM>::value,
                          "RenderPassTransformBeginInfoQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderPassTransformInfoQCOM ) ==
                            sizeof( VkCommandBufferInheritanceRenderPassTransformInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderPassTransformInfoQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderPassTransformInfoQCOM>::value,
                          "CommandBufferInheritanceRenderPassTransformInfoQCOM is not nothrow_move_constructible!" );

//=== VK_EXT_depth_bias_control ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthBiasControlFeaturesEXT ) == sizeof( VkPhysicalDeviceDepthBiasControlFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthBiasControlFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthBiasControlFeaturesEXT>::value,
                          "PhysicalDeviceDepthBiasControlFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DepthBiasInfoEXT ) == sizeof( VkDepthBiasInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DepthBiasInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DepthBiasInfoEXT>::value,
                          "DepthBiasInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DepthBiasRepresentationInfoEXT ) == sizeof( VkDepthBiasRepresentationInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DepthBiasRepresentationInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DepthBiasRepresentationInfoEXT>::value,
                          "DepthBiasRepresentationInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_device_memory_report ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceMemoryReportFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceDeviceMemoryReportFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceMemoryReportFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceMemoryReportFeaturesEXT>::value,
                          "PhysicalDeviceDeviceMemoryReportFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceDeviceMemoryReportCreateInfoEXT ) == sizeof( VkDeviceDeviceMemoryReportCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceDeviceMemoryReportCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceDeviceMemoryReportCreateInfoEXT>::value,
                          "DeviceDeviceMemoryReportCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceMemoryReportCallbackDataEXT ) == sizeof( VkDeviceMemoryReportCallbackDataEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceMemoryReportCallbackDataEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceMemoryReportCallbackDataEXT>::value,
                          "DeviceMemoryReportCallbackDataEXT is not nothrow_move_constructible!" );

//=== VK_EXT_robustness2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesEXT ) == sizeof( VkPhysicalDeviceRobustness2FeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesEXT>::value,
                          "PhysicalDeviceRobustness2FeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesEXT ) == sizeof( VkPhysicalDeviceRobustness2PropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesEXT>::value,
                          "PhysicalDeviceRobustness2PropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_custom_border_color ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerCustomBorderColorCreateInfoEXT ) == sizeof( VkSamplerCustomBorderColorCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerCustomBorderColorCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerCustomBorderColorCreateInfoEXT>::value,
                          "SamplerCustomBorderColorCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceCustomBorderColorPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorPropertiesEXT>::value,
                          "PhysicalDeviceCustomBorderColorPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorFeaturesEXT ) == sizeof( VkPhysicalDeviceCustomBorderColorFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorFeaturesEXT>::value,
                          "PhysicalDeviceCustomBorderColorFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_KHR_pipeline_library ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineLibraryCreateInfoKHR ) == sizeof( VkPipelineLibraryCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineLibraryCreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineLibraryCreateInfoKHR>::value,
                          "PipelineLibraryCreateInfoKHR is not nothrow_move_constructible!" );

//=== VK_NV_present_barrier ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePresentBarrierFeaturesNV ) == sizeof( VkPhysicalDevicePresentBarrierFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentBarrierFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentBarrierFeaturesNV>::value,
                          "PhysicalDevicePresentBarrierFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentBarrierNV ) == sizeof( VkSurfaceCapabilitiesPresentBarrierNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentBarrierNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentBarrierNV>::value,
                          "SurfaceCapabilitiesPresentBarrierNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainPresentBarrierCreateInfoNV ) == sizeof( VkSwapchainPresentBarrierCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainPresentBarrierCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainPresentBarrierCreateInfoNV>::value,
                          "SwapchainPresentBarrierCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_KHR_present_id ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PresentIdKHR ) == sizeof( VkPresentIdKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PresentIdKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PresentIdKHR>::value, "PresentIdKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePresentIdFeaturesKHR ) == sizeof( VkPhysicalDevicePresentIdFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentIdFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentIdFeaturesKHR>::value,
                          "PhysicalDevicePresentIdFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_KHR_video_encode_queue ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeInfoKHR ) == sizeof( VkVideoEncodeInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeInfoKHR>::value,
                          "VideoEncodeInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeCapabilitiesKHR ) == sizeof( VkVideoEncodeCapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeCapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeCapabilitiesKHR>::value,
                          "VideoEncodeCapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueryPoolVideoEncodeFeedbackCreateInfoKHR ) == sizeof( VkQueryPoolVideoEncodeFeedbackCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueryPoolVideoEncodeFeedbackCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueryPoolVideoEncodeFeedbackCreateInfoKHR>::value,
                          "QueryPoolVideoEncodeFeedbackCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeUsageInfoKHR ) == sizeof( VkVideoEncodeUsageInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeUsageInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeUsageInfoKHR>::value,
                          "VideoEncodeUsageInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeRateControlInfoKHR ) == sizeof( VkVideoEncodeRateControlInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlInfoKHR>::value,
                          "VideoEncodeRateControlInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeRateControlLayerInfoKHR ) == sizeof( VkVideoEncodeRateControlLayerInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlLayerInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlLayerInfoKHR>::value,
                          "VideoEncodeRateControlLayerInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeQualityLevelInfoKHR ) ==
                            sizeof( VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeQualityLevelInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeQualityLevelInfoKHR>::value,
                          "PhysicalDeviceVideoEncodeQualityLevelInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelPropertiesKHR ) == sizeof( VkVideoEncodeQualityLevelPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelPropertiesKHR>::value,
                          "VideoEncodeQualityLevelPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelInfoKHR ) == sizeof( VkVideoEncodeQualityLevelInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelInfoKHR>::value,
                          "VideoEncodeQualityLevelInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersGetInfoKHR ) == sizeof( VkVideoEncodeSessionParametersGetInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersGetInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersGetInfoKHR>::value,
                          "VideoEncodeSessionParametersGetInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersFeedbackInfoKHR ) ==
                            sizeof( VkVideoEncodeSessionParametersFeedbackInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersFeedbackInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersFeedbackInfoKHR>::value,
                          "VideoEncodeSessionParametersFeedbackInfoKHR is not nothrow_move_constructible!" );

//=== VK_NV_device_diagnostics_config ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV ) == sizeof( VkPhysicalDeviceDiagnosticsConfigFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV>::value,
                          "PhysicalDeviceDiagnosticsConfigFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV ) == sizeof( VkDeviceDiagnosticsConfigCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV>::value,
                          "DeviceDiagnosticsConfigCreateInfoNV is not nothrow_move_constructible!" );

#if defined( VK_ENABLE_BETA_EXTENSIONS )
//=== VK_NV_cuda_kernel_launch ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CudaModuleNV ) == sizeof( VkCudaModuleNV ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CudaModuleNV>::value, "CudaModuleNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CudaFunctionNV ) == sizeof( VkCudaFunctionNV ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CudaFunctionNV>::value,
                          "CudaFunctionNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CudaModuleCreateInfoNV ) == sizeof( VkCudaModuleCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CudaModuleCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CudaModuleCreateInfoNV>::value,
                          "CudaModuleCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CudaFunctionCreateInfoNV ) == sizeof( VkCudaFunctionCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CudaFunctionCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CudaFunctionCreateInfoNV>::value,
                          "CudaFunctionCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CudaLaunchInfoNV ) == sizeof( VkCudaLaunchInfoNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CudaLaunchInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CudaLaunchInfoNV>::value,
                          "CudaLaunchInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchFeaturesNV ) == sizeof( VkPhysicalDeviceCudaKernelLaunchFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchFeaturesNV>::value,
                          "PhysicalDeviceCudaKernelLaunchFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchPropertiesNV ) == sizeof( VkPhysicalDeviceCudaKernelLaunchPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchPropertiesNV>::value,
                          "PhysicalDeviceCudaKernelLaunchPropertiesNV is not nothrow_move_constructible!" );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

//=== VK_NV_low_latency ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueryLowLatencySupportNV ) == sizeof( VkQueryLowLatencySupportNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueryLowLatencySupportNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueryLowLatencySupportNV>::value,
                          "QueryLowLatencySupportNV is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_METAL_EXT )
//=== VK_EXT_metal_objects ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalObjectCreateInfoEXT ) == sizeof( VkExportMetalObjectCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalObjectCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalObjectCreateInfoEXT>::value,
                          "ExportMetalObjectCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalObjectsInfoEXT ) == sizeof( VkExportMetalObjectsInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalObjectsInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalObjectsInfoEXT>::value,
                          "ExportMetalObjectsInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalDeviceInfoEXT ) == sizeof( VkExportMetalDeviceInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalDeviceInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalDeviceInfoEXT>::value,
                          "ExportMetalDeviceInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalCommandQueueInfoEXT ) == sizeof( VkExportMetalCommandQueueInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalCommandQueueInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalCommandQueueInfoEXT>::value,
                          "ExportMetalCommandQueueInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalBufferInfoEXT ) == sizeof( VkExportMetalBufferInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalBufferInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalBufferInfoEXT>::value,
                          "ExportMetalBufferInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMetalBufferInfoEXT ) == sizeof( VkImportMetalBufferInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMetalBufferInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMetalBufferInfoEXT>::value,
                          "ImportMetalBufferInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalTextureInfoEXT ) == sizeof( VkExportMetalTextureInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalTextureInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalTextureInfoEXT>::value,
                          "ExportMetalTextureInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMetalTextureInfoEXT ) == sizeof( VkImportMetalTextureInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMetalTextureInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMetalTextureInfoEXT>::value,
                          "ImportMetalTextureInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalIOSurfaceInfoEXT ) == sizeof( VkExportMetalIOSurfaceInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalIOSurfaceInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalIOSurfaceInfoEXT>::value,
                          "ExportMetalIOSurfaceInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMetalIOSurfaceInfoEXT ) == sizeof( VkImportMetalIOSurfaceInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMetalIOSurfaceInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMetalIOSurfaceInfoEXT>::value,
                          "ImportMetalIOSurfaceInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExportMetalSharedEventInfoEXT ) == sizeof( VkExportMetalSharedEventInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExportMetalSharedEventInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExportMetalSharedEventInfoEXT>::value,
                          "ExportMetalSharedEventInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMetalSharedEventInfoEXT ) == sizeof( VkImportMetalSharedEventInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMetalSharedEventInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMetalSharedEventInfoEXT>::value,
                          "ImportMetalSharedEventInfoEXT is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_METAL_EXT*/

//=== VK_KHR_synchronization2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV ) == sizeof( VkQueueFamilyCheckpointProperties2NV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV>::value,
                          "QueueFamilyCheckpointProperties2NV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CheckpointData2NV ) == sizeof( VkCheckpointData2NV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CheckpointData2NV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CheckpointData2NV>::value,
                          "CheckpointData2NV is not nothrow_move_constructible!" );

//=== VK_EXT_descriptor_buffer ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceDescriptorBufferPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferPropertiesEXT>::value,
                          "PhysicalDeviceDescriptorBufferPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceDescriptorBufferDensityMapPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT>::value,
                          "PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferFeaturesEXT ) == sizeof( VkPhysicalDeviceDescriptorBufferFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferFeaturesEXT>::value,
                          "PhysicalDeviceDescriptorBufferFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorAddressInfoEXT ) == sizeof( VkDescriptorAddressInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorAddressInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorAddressInfoEXT>::value,
                          "DescriptorAddressInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorBufferBindingInfoEXT ) == sizeof( VkDescriptorBufferBindingInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorBufferBindingInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorBufferBindingInfoEXT>::value,
                          "DescriptorBufferBindingInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorBufferBindingPushDescriptorBufferHandleEXT ) ==
                            sizeof( VkDescriptorBufferBindingPushDescriptorBufferHandleEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorBufferBindingPushDescriptorBufferHandleEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorBufferBindingPushDescriptorBufferHandleEXT>::value,
                          "DescriptorBufferBindingPushDescriptorBufferHandleEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorDataEXT ) == sizeof( VkDescriptorDataEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorDataEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorDataEXT>::value,
                          "DescriptorDataEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorGetInfoEXT ) == sizeof( VkDescriptorGetInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorGetInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorGetInfoEXT>::value,
                          "DescriptorGetInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCaptureDescriptorDataInfoEXT ) == sizeof( VkBufferCaptureDescriptorDataInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCaptureDescriptorDataInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCaptureDescriptorDataInfoEXT>::value,
                          "BufferCaptureDescriptorDataInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageCaptureDescriptorDataInfoEXT ) == sizeof( VkImageCaptureDescriptorDataInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageCaptureDescriptorDataInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageCaptureDescriptorDataInfoEXT>::value,
                          "ImageCaptureDescriptorDataInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewCaptureDescriptorDataInfoEXT ) == sizeof( VkImageViewCaptureDescriptorDataInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewCaptureDescriptorDataInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewCaptureDescriptorDataInfoEXT>::value,
                          "ImageViewCaptureDescriptorDataInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerCaptureDescriptorDataInfoEXT ) == sizeof( VkSamplerCaptureDescriptorDataInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerCaptureDescriptorDataInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerCaptureDescriptorDataInfoEXT>::value,
                          "SamplerCaptureDescriptorDataInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OpaqueCaptureDescriptorDataCreateInfoEXT ) == sizeof( VkOpaqueCaptureDescriptorDataCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::OpaqueCaptureDescriptorDataCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OpaqueCaptureDescriptorDataCreateInfoEXT>::value,
                          "OpaqueCaptureDescriptorDataCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureCaptureDescriptorDataInfoEXT ) ==
                            sizeof( VkAccelerationStructureCaptureDescriptorDataInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureCaptureDescriptorDataInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureCaptureDescriptorDataInfoEXT>::value,
                          "AccelerationStructureCaptureDescriptorDataInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_graphics_pipeline_library ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>::value,
                          "PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT>::value,
                          "PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryCreateInfoEXT ) == sizeof( VkGraphicsPipelineLibraryCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryCreateInfoEXT>::value,
                          "GraphicsPipelineLibraryCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_AMD_shader_early_and_late_fragment_tests ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD ) ==
                            sizeof( VkPhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD>::value,
                          "PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD is not nothrow_move_constructible!" );

//=== VK_KHR_fragment_shader_barycentric ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>::value,
                          "PhysicalDeviceFragmentShaderBarycentricFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR ) ==
                            sizeof( VkPhysicalDeviceFragmentShaderBarycentricPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR>::value,
                          "PhysicalDeviceFragmentShaderBarycentricPropertiesKHR is not nothrow_move_constructible!" );

//=== VK_KHR_shader_subgroup_uniform_control_flow ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR>::value,
                          "PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_NV_fragment_shading_rate_enums ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceFragmentShadingRateEnumsFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsFeaturesNV>::value,
                          "PhysicalDeviceFragmentShadingRateEnumsFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsPropertiesNV ) ==
                            sizeof( VkPhysicalDeviceFragmentShadingRateEnumsPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsPropertiesNV>::value,
                          "PhysicalDeviceFragmentShadingRateEnumsPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateEnumStateCreateInfoNV ) ==
                            sizeof( VkPipelineFragmentShadingRateEnumStateCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateEnumStateCreateInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateEnumStateCreateInfoNV>::value,
                          "PipelineFragmentShadingRateEnumStateCreateInfoNV is not nothrow_move_constructible!" );

//=== VK_NV_ray_tracing_motion_blur ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryMotionTrianglesDataNV ) ==
                            sizeof( VkAccelerationStructureGeometryMotionTrianglesDataNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryMotionTrianglesDataNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryMotionTrianglesDataNV>::value,
                          "AccelerationStructureGeometryMotionTrianglesDataNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoNV ) == sizeof( VkAccelerationStructureMotionInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoNV>::value,
                          "AccelerationStructureMotionInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceNV ) == sizeof( VkAccelerationStructureMotionInstanceNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceNV>::value,
                          "AccelerationStructureMotionInstanceNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceDataNV ) == sizeof( VkAccelerationStructureMotionInstanceDataNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceDataNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceDataNV>::value,
                          "AccelerationStructureMotionInstanceDataNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureMatrixMotionInstanceNV ) ==
                            sizeof( VkAccelerationStructureMatrixMotionInstanceNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureMatrixMotionInstanceNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureMatrixMotionInstanceNV>::value,
                          "AccelerationStructureMatrixMotionInstanceNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureSRTMotionInstanceNV ) == sizeof( VkAccelerationStructureSRTMotionInstanceNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureSRTMotionInstanceNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureSRTMotionInstanceNV>::value,
                          "AccelerationStructureSRTMotionInstanceNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SRTDataNV ) == sizeof( VkSRTDataNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SRTDataNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SRTDataNV>::value, "SRTDataNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMotionBlurFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceRayTracingMotionBlurFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMotionBlurFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMotionBlurFeaturesNV>::value,
                          "PhysicalDeviceRayTracingMotionBlurFeaturesNV is not nothrow_move_constructible!" );

//=== VK_EXT_mesh_shader ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesEXT ) == sizeof( VkPhysicalDeviceMeshShaderFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesEXT>::value,
                          "PhysicalDeviceMeshShaderFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesEXT ) == sizeof( VkPhysicalDeviceMeshShaderPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesEXT>::value,
                          "PhysicalDeviceMeshShaderPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandEXT ) == sizeof( VkDrawMeshTasksIndirectCommandEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandEXT>::value,
                          "DrawMeshTasksIndirectCommandEXT is not nothrow_move_constructible!" );

//=== VK_EXT_ycbcr_2plane_444_formats ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT>::value,
                          "PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_fragment_density_map2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2FeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceFragmentDensityMap2FeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2FeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2FeaturesEXT>::value,
                          "PhysicalDeviceFragmentDensityMap2FeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2PropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceFragmentDensityMap2PropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2PropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2PropertiesEXT>::value,
                          "PhysicalDeviceFragmentDensityMap2PropertiesEXT is not nothrow_move_constructible!" );

//=== VK_QCOM_rotated_copy_commands ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyCommandTransformInfoQCOM ) == sizeof( VkCopyCommandTransformInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyCommandTransformInfoQCOM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyCommandTransformInfoQCOM>::value,
                          "CopyCommandTransformInfoQCOM is not nothrow_move_constructible!" );

//=== VK_KHR_workgroup_memory_explicit_layout ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>::value,
                          "PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_EXT_image_compression_control ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceImageCompressionControlFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlFeaturesEXT>::value,
                          "PhysicalDeviceImageCompressionControlFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageCompressionControlEXT ) == sizeof( VkImageCompressionControlEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageCompressionControlEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageCompressionControlEXT>::value,
                          "ImageCompressionControlEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageCompressionPropertiesEXT ) == sizeof( VkImageCompressionPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageCompressionPropertiesEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageCompressionPropertiesEXT>::value,
                          "ImageCompressionPropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_attachment_feedback_loop_layout ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>::value,
                          "PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_4444_formats ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevice4444FormatsFeaturesEXT ) == sizeof( VkPhysicalDevice4444FormatsFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevice4444FormatsFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevice4444FormatsFeaturesEXT>::value,
                          "PhysicalDevice4444FormatsFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_device_fault ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFaultFeaturesEXT ) == sizeof( VkPhysicalDeviceFaultFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFaultFeaturesEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFaultFeaturesEXT>::value,
                          "PhysicalDeviceFaultFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceFaultCountsEXT ) == sizeof( VkDeviceFaultCountsEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceFaultCountsEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceFaultCountsEXT>::value,
                          "DeviceFaultCountsEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceFaultInfoEXT ) == sizeof( VkDeviceFaultInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceFaultInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceFaultInfoEXT>::value,
                          "DeviceFaultInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceFaultAddressInfoEXT ) == sizeof( VkDeviceFaultAddressInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceFaultAddressInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceFaultAddressInfoEXT>::value,
                          "DeviceFaultAddressInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceFaultVendorInfoEXT ) == sizeof( VkDeviceFaultVendorInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceFaultVendorInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceFaultVendorInfoEXT>::value,
                          "DeviceFaultVendorInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionOneEXT ) == sizeof( VkDeviceFaultVendorBinaryHeaderVersionOneEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionOneEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionOneEXT>::value,
                          "DeviceFaultVendorBinaryHeaderVersionOneEXT is not nothrow_move_constructible!" );

//=== VK_EXT_rgba10x6_formats ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRGBA10X6FormatsFeaturesEXT ) == sizeof( VkPhysicalDeviceRGBA10X6FormatsFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRGBA10X6FormatsFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRGBA10X6FormatsFeaturesEXT>::value,
                          "PhysicalDeviceRGBA10X6FormatsFeaturesEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
//=== VK_EXT_directfb_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT ) == sizeof( VkDirectFBSurfaceCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT>::value,
                          "DirectFBSurfaceCreateInfoEXT is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

//=== VK_EXT_vertex_input_dynamic_state ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexInputDynamicStateFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexInputDynamicStateFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexInputDynamicStateFeaturesEXT>::value,
                          "PhysicalDeviceVertexInputDynamicStateFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT ) == sizeof( VkVertexInputBindingDescription2EXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT>::value,
                          "VertexInputBindingDescription2EXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT ) == sizeof( VkVertexInputAttributeDescription2EXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT>::value,
                          "VertexInputAttributeDescription2EXT is not nothrow_move_constructible!" );

//=== VK_EXT_physical_device_drm ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDrmPropertiesEXT ) == sizeof( VkPhysicalDeviceDrmPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDrmPropertiesEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDrmPropertiesEXT>::value,
                          "PhysicalDeviceDrmPropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_device_address_binding_report ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceAddressBindingReportFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceAddressBindingReportFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceAddressBindingReportFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceAddressBindingReportFeaturesEXT>::value,
                          "PhysicalDeviceAddressBindingReportFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceAddressBindingCallbackDataEXT ) == sizeof( VkDeviceAddressBindingCallbackDataEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceAddressBindingCallbackDataEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceAddressBindingCallbackDataEXT>::value,
                          "DeviceAddressBindingCallbackDataEXT is not nothrow_move_constructible!" );

//=== VK_EXT_depth_clip_control ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipControlFeaturesEXT ) == sizeof( VkPhysicalDeviceDepthClipControlFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipControlFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipControlFeaturesEXT>::value,
                          "PhysicalDeviceDepthClipControlFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineViewportDepthClipControlCreateInfoEXT ) ==
                            sizeof( VkPipelineViewportDepthClipControlCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineViewportDepthClipControlCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineViewportDepthClipControlCreateInfoEXT>::value,
                          "PipelineViewportDepthClipControlCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_primitive_topology_list_restart ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT ) ==
                            sizeof( VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>::value,
                          "PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_FUCHSIA )
//=== VK_FUCHSIA_external_memory ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMemoryZirconHandleInfoFUCHSIA ) == sizeof( VkImportMemoryZirconHandleInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMemoryZirconHandleInfoFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMemoryZirconHandleInfoFUCHSIA>::value,
                          "ImportMemoryZirconHandleInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA ) == sizeof( VkMemoryZirconHandlePropertiesFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA>::value,
                          "MemoryZirconHandlePropertiesFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryGetZirconHandleInfoFUCHSIA ) == sizeof( VkMemoryGetZirconHandleInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryGetZirconHandleInfoFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryGetZirconHandleInfoFUCHSIA>::value,
                          "MemoryGetZirconHandleInfoFUCHSIA is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
//=== VK_FUCHSIA_external_semaphore ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportSemaphoreZirconHandleInfoFUCHSIA ) == sizeof( VkImportSemaphoreZirconHandleInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportSemaphoreZirconHandleInfoFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportSemaphoreZirconHandleInfoFUCHSIA>::value,
                          "ImportSemaphoreZirconHandleInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SemaphoreGetZirconHandleInfoFUCHSIA ) == sizeof( VkSemaphoreGetZirconHandleInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SemaphoreGetZirconHandleInfoFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SemaphoreGetZirconHandleInfoFUCHSIA>::value,
                          "SemaphoreGetZirconHandleInfoFUCHSIA is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
//=== VK_FUCHSIA_buffer_collection ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCollectionFUCHSIA ) == sizeof( VkBufferCollectionFUCHSIA ),
                          "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCollectionFUCHSIA>::value,
                          "BufferCollectionFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCollectionCreateInfoFUCHSIA ) == sizeof( VkBufferCollectionCreateInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCollectionCreateInfoFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCollectionCreateInfoFUCHSIA>::value,
                          "BufferCollectionCreateInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportMemoryBufferCollectionFUCHSIA ) == sizeof( VkImportMemoryBufferCollectionFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportMemoryBufferCollectionFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportMemoryBufferCollectionFUCHSIA>::value,
                          "ImportMemoryBufferCollectionFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCollectionImageCreateInfoFUCHSIA ) == sizeof( VkBufferCollectionImageCreateInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCollectionImageCreateInfoFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCollectionImageCreateInfoFUCHSIA>::value,
                          "BufferCollectionImageCreateInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferConstraintsInfoFUCHSIA ) == sizeof( VkBufferConstraintsInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferConstraintsInfoFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferConstraintsInfoFUCHSIA>::value,
                          "BufferConstraintsInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCollectionBufferCreateInfoFUCHSIA ) == sizeof( VkBufferCollectionBufferCreateInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCollectionBufferCreateInfoFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCollectionBufferCreateInfoFUCHSIA>::value,
                          "BufferCollectionBufferCreateInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCollectionPropertiesFUCHSIA ) == sizeof( VkBufferCollectionPropertiesFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCollectionPropertiesFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCollectionPropertiesFUCHSIA>::value,
                          "BufferCollectionPropertiesFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SysmemColorSpaceFUCHSIA ) == sizeof( VkSysmemColorSpaceFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SysmemColorSpaceFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SysmemColorSpaceFUCHSIA>::value,
                          "SysmemColorSpaceFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFUCHSIA ) == sizeof( VkImageConstraintsInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFUCHSIA>::value,
                          "ImageConstraintsInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageFormatConstraintsInfoFUCHSIA ) == sizeof( VkImageFormatConstraintsInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageFormatConstraintsInfoFUCHSIA>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageFormatConstraintsInfoFUCHSIA>::value,
                          "ImageFormatConstraintsInfoFUCHSIA is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferCollectionConstraintsInfoFUCHSIA ) == sizeof( VkBufferCollectionConstraintsInfoFUCHSIA ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferCollectionConstraintsInfoFUCHSIA>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferCollectionConstraintsInfoFUCHSIA>::value,
                          "BufferCollectionConstraintsInfoFUCHSIA is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

//=== VK_HUAWEI_subpass_shading ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassShadingPipelineCreateInfoHUAWEI ) == sizeof( VkSubpassShadingPipelineCreateInfoHUAWEI ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassShadingPipelineCreateInfoHUAWEI>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassShadingPipelineCreateInfoHUAWEI>::value,
                          "SubpassShadingPipelineCreateInfoHUAWEI is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingFeaturesHUAWEI ) == sizeof( VkPhysicalDeviceSubpassShadingFeaturesHUAWEI ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingFeaturesHUAWEI>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingFeaturesHUAWEI>::value,
                          "PhysicalDeviceSubpassShadingFeaturesHUAWEI is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingPropertiesHUAWEI ) ==
                            sizeof( VkPhysicalDeviceSubpassShadingPropertiesHUAWEI ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingPropertiesHUAWEI>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingPropertiesHUAWEI>::value,
                          "PhysicalDeviceSubpassShadingPropertiesHUAWEI is not nothrow_move_constructible!" );

//=== VK_HUAWEI_invocation_mask ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceInvocationMaskFeaturesHUAWEI ) == sizeof( VkPhysicalDeviceInvocationMaskFeaturesHUAWEI ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceInvocationMaskFeaturesHUAWEI>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceInvocationMaskFeaturesHUAWEI>::value,
                          "PhysicalDeviceInvocationMaskFeaturesHUAWEI is not nothrow_move_constructible!" );

//=== VK_NV_external_memory_rdma ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MemoryGetRemoteAddressInfoNV ) == sizeof( VkMemoryGetRemoteAddressInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MemoryGetRemoteAddressInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MemoryGetRemoteAddressInfoNV>::value,
                          "MemoryGetRemoteAddressInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryRDMAFeaturesNV ) == sizeof( VkPhysicalDeviceExternalMemoryRDMAFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryRDMAFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryRDMAFeaturesNV>::value,
                          "PhysicalDeviceExternalMemoryRDMAFeaturesNV is not nothrow_move_constructible!" );

//=== VK_EXT_pipeline_properties ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelinePropertiesIdentifierEXT ) == sizeof( VkPipelinePropertiesIdentifierEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelinePropertiesIdentifierEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelinePropertiesIdentifierEXT>::value,
                          "PipelinePropertiesIdentifierEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelinePropertiesFeaturesEXT ) ==
                            sizeof( VkPhysicalDevicePipelinePropertiesFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelinePropertiesFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelinePropertiesFeaturesEXT>::value,
                          "PhysicalDevicePipelinePropertiesFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_frame_boundary ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFrameBoundaryFeaturesEXT ) == sizeof( VkPhysicalDeviceFrameBoundaryFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFrameBoundaryFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFrameBoundaryFeaturesEXT>::value,
                          "PhysicalDeviceFrameBoundaryFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::FrameBoundaryEXT ) == sizeof( VkFrameBoundaryEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::FrameBoundaryEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::FrameBoundaryEXT>::value,
                          "FrameBoundaryEXT is not nothrow_move_constructible!" );

//=== VK_EXT_multisampled_render_to_single_sampled ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT>::value,
                          "PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassResolvePerformanceQueryEXT ) == sizeof( VkSubpassResolvePerformanceQueryEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassResolvePerformanceQueryEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassResolvePerformanceQueryEXT>::value,
                          "SubpassResolvePerformanceQueryEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MultisampledRenderToSingleSampledInfoEXT ) == sizeof( VkMultisampledRenderToSingleSampledInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MultisampledRenderToSingleSampledInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MultisampledRenderToSingleSampledInfoEXT>::value,
                          "MultisampledRenderToSingleSampledInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_extended_dynamic_state2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState2FeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceExtendedDynamicState2FeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState2FeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState2FeaturesEXT>::value,
                          "PhysicalDeviceExtendedDynamicState2FeaturesEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
//=== VK_QNX_screen_surface ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX ) == sizeof( VkScreenSurfaceCreateInfoQNX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX>::value,
                          "ScreenSurfaceCreateInfoQNX is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

//=== VK_EXT_color_write_enable ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceColorWriteEnableFeaturesEXT ) == sizeof( VkPhysicalDeviceColorWriteEnableFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceColorWriteEnableFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceColorWriteEnableFeaturesEXT>::value,
                          "PhysicalDeviceColorWriteEnableFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineColorWriteCreateInfoEXT ) == sizeof( VkPipelineColorWriteCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineColorWriteCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineColorWriteCreateInfoEXT>::value,
                          "PipelineColorWriteCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_primitives_generated_query ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT ) ==
                            sizeof( VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT>::value,
                          "PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_KHR_ray_tracing_maintenance1 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMaintenance1FeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMaintenance1FeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMaintenance1FeaturesKHR>::value,
                          "PhysicalDeviceRayTracingMaintenance1FeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommand2KHR ) == sizeof( VkTraceRaysIndirectCommand2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommand2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommand2KHR>::value,
                          "TraceRaysIndirectCommand2KHR is not nothrow_move_constructible!" );

//=== VK_EXT_image_view_min_lod ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewMinLodFeaturesEXT ) == sizeof( VkPhysicalDeviceImageViewMinLodFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewMinLodFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewMinLodFeaturesEXT>::value,
                          "PhysicalDeviceImageViewMinLodFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewMinLodCreateInfoEXT ) == sizeof( VkImageViewMinLodCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewMinLodCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewMinLodCreateInfoEXT>::value,
                          "ImageViewMinLodCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_multi_draw ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawFeaturesEXT ) == sizeof( VkPhysicalDeviceMultiDrawFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawFeaturesEXT>::value,
                          "PhysicalDeviceMultiDrawFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawPropertiesEXT ) == sizeof( VkPhysicalDeviceMultiDrawPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawPropertiesEXT>::value,
                          "PhysicalDeviceMultiDrawPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT ) == sizeof( VkMultiDrawInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT>::value,
                          "MultiDrawInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT ) == sizeof( VkMultiDrawIndexedInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT>::value,
                          "MultiDrawIndexedInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_image_2d_view_of_3d ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImage2DViewOf3DFeaturesEXT ) == sizeof( VkPhysicalDeviceImage2DViewOf3DFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImage2DViewOf3DFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImage2DViewOf3DFeaturesEXT>::value,
                          "PhysicalDeviceImage2DViewOf3DFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_shader_tile_image ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImageFeaturesEXT ) == sizeof( VkPhysicalDeviceShaderTileImageFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImageFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImageFeaturesEXT>::value,
                          "PhysicalDeviceShaderTileImageFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImagePropertiesEXT ) == sizeof( VkPhysicalDeviceShaderTileImagePropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImagePropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImagePropertiesEXT>::value,
                          "PhysicalDeviceShaderTileImagePropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_opacity_micromap ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MicromapBuildInfoEXT ) == sizeof( VkMicromapBuildInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MicromapBuildInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MicromapBuildInfoEXT>::value,
                          "MicromapBuildInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MicromapUsageEXT ) == sizeof( VkMicromapUsageEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MicromapUsageEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MicromapUsageEXT>::value,
                          "MicromapUsageEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MicromapCreateInfoEXT ) == sizeof( VkMicromapCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MicromapCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MicromapCreateInfoEXT>::value,
                          "MicromapCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MicromapEXT ) == sizeof( VkMicromapEXT ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MicromapEXT>::value, "MicromapEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapFeaturesEXT ) == sizeof( VkPhysicalDeviceOpacityMicromapFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapFeaturesEXT>::value,
                          "PhysicalDeviceOpacityMicromapFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapPropertiesEXT ) == sizeof( VkPhysicalDeviceOpacityMicromapPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapPropertiesEXT>::value,
                          "PhysicalDeviceOpacityMicromapPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MicromapVersionInfoEXT ) == sizeof( VkMicromapVersionInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MicromapVersionInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MicromapVersionInfoEXT>::value,
                          "MicromapVersionInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyMicromapToMemoryInfoEXT ) == sizeof( VkCopyMicromapToMemoryInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyMicromapToMemoryInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyMicromapToMemoryInfoEXT>::value,
                          "CopyMicromapToMemoryInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyMemoryToMicromapInfoEXT ) == sizeof( VkCopyMemoryToMicromapInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyMemoryToMicromapInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyMemoryToMicromapInfoEXT>::value,
                          "CopyMemoryToMicromapInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyMicromapInfoEXT ) == sizeof( VkCopyMicromapInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyMicromapInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyMicromapInfoEXT>::value,
                          "CopyMicromapInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MicromapBuildSizesInfoEXT ) == sizeof( VkMicromapBuildSizesInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MicromapBuildSizesInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MicromapBuildSizesInfoEXT>::value,
                          "MicromapBuildSizesInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesOpacityMicromapEXT ) ==
                            sizeof( VkAccelerationStructureTrianglesOpacityMicromapEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesOpacityMicromapEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesOpacityMicromapEXT>::value,
                          "AccelerationStructureTrianglesOpacityMicromapEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MicromapTriangleEXT ) == sizeof( VkMicromapTriangleEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MicromapTriangleEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MicromapTriangleEXT>::value,
                          "MicromapTriangleEXT is not nothrow_move_constructible!" );

#if defined( VK_ENABLE_BETA_EXTENSIONS )
//=== VK_NV_displacement_micromap ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceDisplacementMicromapFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapFeaturesNV>::value,
                          "PhysicalDeviceDisplacementMicromapFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapPropertiesNV ) ==
                            sizeof( VkPhysicalDeviceDisplacementMicromapPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapPropertiesNV>::value,
                          "PhysicalDeviceDisplacementMicromapPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesDisplacementMicromapNV ) ==
                            sizeof( VkAccelerationStructureTrianglesDisplacementMicromapNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesDisplacementMicromapNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesDisplacementMicromapNV>::value,
                          "AccelerationStructureTrianglesDisplacementMicromapNV is not nothrow_move_constructible!" );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

//=== VK_HUAWEI_cluster_culling_shader ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderFeaturesHUAWEI ) ==
                            sizeof( VkPhysicalDeviceClusterCullingShaderFeaturesHUAWEI ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderFeaturesHUAWEI>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderFeaturesHUAWEI>::value,
                          "PhysicalDeviceClusterCullingShaderFeaturesHUAWEI is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderPropertiesHUAWEI ) ==
                            sizeof( VkPhysicalDeviceClusterCullingShaderPropertiesHUAWEI ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderPropertiesHUAWEI>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderPropertiesHUAWEI>::value,
                          "PhysicalDeviceClusterCullingShaderPropertiesHUAWEI is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI ) ==
                            sizeof( VkPhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI>::value,
                          "PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI is not nothrow_move_constructible!" );

//=== VK_EXT_border_color_swizzle ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceBorderColorSwizzleFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT>::value,
                          "PhysicalDeviceBorderColorSwizzleFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT ) ==
                            sizeof( VkSamplerBorderColorComponentMappingCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT>::value,
                          "SamplerBorderColorComponentMappingCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_pageable_device_local_memory ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT ) ==
                            sizeof( VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT>::value,
                          "PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_ARM_shader_core_properties ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesARM ) == sizeof( VkPhysicalDeviceShaderCorePropertiesARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesARM>::value,
                          "PhysicalDeviceShaderCorePropertiesARM is not nothrow_move_constructible!" );

//=== VK_KHR_shader_subgroup_rotate ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupRotateFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupRotateFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupRotateFeaturesKHR>::value,
                          "PhysicalDeviceShaderSubgroupRotateFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_ARM_scheduling_controls ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceQueueShaderCoreControlCreateInfoARM ) == sizeof( VkDeviceQueueShaderCoreControlCreateInfoARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceQueueShaderCoreControlCreateInfoARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceQueueShaderCoreControlCreateInfoARM>::value,
                          "DeviceQueueShaderCoreControlCreateInfoARM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsFeaturesARM ) ==
                            sizeof( VkPhysicalDeviceSchedulingControlsFeaturesARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsFeaturesARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsFeaturesARM>::value,
                          "PhysicalDeviceSchedulingControlsFeaturesARM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsPropertiesARM ) ==
                            sizeof( VkPhysicalDeviceSchedulingControlsPropertiesARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsPropertiesARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsPropertiesARM>::value,
                          "PhysicalDeviceSchedulingControlsPropertiesARM is not nothrow_move_constructible!" );

//=== VK_EXT_image_sliced_view_of_3d ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageSlicedViewOf3DFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceImageSlicedViewOf3DFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageSlicedViewOf3DFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageSlicedViewOf3DFeaturesEXT>::value,
                          "PhysicalDeviceImageSlicedViewOf3DFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewSlicedCreateInfoEXT ) == sizeof( VkImageViewSlicedCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewSlicedCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewSlicedCreateInfoEXT>::value,
                          "ImageViewSlicedCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_VALVE_descriptor_set_host_mapping ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE ) ==
                            sizeof( VkPhysicalDeviceDescriptorSetHostMappingFeaturesVALVE ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE>::value,
                          "PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE ) == sizeof( VkDescriptorSetBindingReferenceVALVE ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE>::value,
                          "DescriptorSetBindingReferenceVALVE is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE ) == sizeof( VkDescriptorSetLayoutHostMappingInfoVALVE ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE>::value,
                          "DescriptorSetLayoutHostMappingInfoVALVE is not nothrow_move_constructible!" );

//=== VK_EXT_depth_clamp_zero_one ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesEXT ) == sizeof( VkPhysicalDeviceDepthClampZeroOneFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesEXT>::value,
                          "PhysicalDeviceDepthClampZeroOneFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_non_seamless_cube_map ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT>::value,
                          "PhysicalDeviceNonSeamlessCubeMapFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_ARM_render_pass_striped ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedFeaturesARM ) == sizeof( VkPhysicalDeviceRenderPassStripedFeaturesARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedFeaturesARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedFeaturesARM>::value,
                          "PhysicalDeviceRenderPassStripedFeaturesARM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedPropertiesARM ) ==
                            sizeof( VkPhysicalDeviceRenderPassStripedPropertiesARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedPropertiesARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedPropertiesARM>::value,
                          "PhysicalDeviceRenderPassStripedPropertiesARM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassStripeBeginInfoARM ) == sizeof( VkRenderPassStripeBeginInfoARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassStripeBeginInfoARM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassStripeBeginInfoARM>::value,
                          "RenderPassStripeBeginInfoARM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassStripeInfoARM ) == sizeof( VkRenderPassStripeInfoARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassStripeInfoARM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassStripeInfoARM>::value,
                          "RenderPassStripeInfoARM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassStripeSubmitInfoARM ) == sizeof( VkRenderPassStripeSubmitInfoARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassStripeSubmitInfoARM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassStripeSubmitInfoARM>::value,
                          "RenderPassStripeSubmitInfoARM is not nothrow_move_constructible!" );

//=== VK_QCOM_fragment_density_map_offset ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM ) ==
                            sizeof( VkPhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM>::value,
                          "PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM ) ==
                            sizeof( VkPhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM>::value,
                          "PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubpassFragmentDensityMapOffsetEndInfoQCOM ) == sizeof( VkSubpassFragmentDensityMapOffsetEndInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubpassFragmentDensityMapOffsetEndInfoQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubpassFragmentDensityMapOffsetEndInfoQCOM>::value,
                          "SubpassFragmentDensityMapOffsetEndInfoQCOM is not nothrow_move_constructible!" );

//=== VK_NV_copy_memory_indirect ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyMemoryIndirectCommandNV ) == sizeof( VkCopyMemoryIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyMemoryIndirectCommandNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyMemoryIndirectCommandNV>::value,
                          "CopyMemoryIndirectCommandNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectCommandNV ) == sizeof( VkCopyMemoryToImageIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectCommandNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectCommandNV>::value,
                          "CopyMemoryToImageIndirectCommandNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesNV ) == sizeof( VkPhysicalDeviceCopyMemoryIndirectFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesNV>::value,
                          "PhysicalDeviceCopyMemoryIndirectFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectPropertiesNV ) ==
                            sizeof( VkPhysicalDeviceCopyMemoryIndirectPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectPropertiesNV>::value,
                          "PhysicalDeviceCopyMemoryIndirectPropertiesNV is not nothrow_move_constructible!" );

//=== VK_NV_memory_decompression ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DecompressMemoryRegionNV ) == sizeof( VkDecompressMemoryRegionNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DecompressMemoryRegionNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DecompressMemoryRegionNV>::value,
                          "DecompressMemoryRegionNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceMemoryDecompressionFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionFeaturesNV>::value,
                          "PhysicalDeviceMemoryDecompressionFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionPropertiesNV ) ==
                            sizeof( VkPhysicalDeviceMemoryDecompressionPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionPropertiesNV>::value,
                          "PhysicalDeviceMemoryDecompressionPropertiesNV is not nothrow_move_constructible!" );

//=== VK_NV_device_generated_commands_compute ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV>::value,
                          "PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ComputePipelineIndirectBufferInfoNV ) == sizeof( VkComputePipelineIndirectBufferInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ComputePipelineIndirectBufferInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ComputePipelineIndirectBufferInfoNV>::value,
                          "ComputePipelineIndirectBufferInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineIndirectDeviceAddressInfoNV ) == sizeof( VkPipelineIndirectDeviceAddressInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineIndirectDeviceAddressInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineIndirectDeviceAddressInfoNV>::value,
                          "PipelineIndirectDeviceAddressInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindPipelineIndirectCommandNV ) == sizeof( VkBindPipelineIndirectCommandNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindPipelineIndirectCommandNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindPipelineIndirectCommandNV>::value,
                          "BindPipelineIndirectCommandNV is not nothrow_move_constructible!" );

//=== VK_NV_linear_color_attachment ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceLinearColorAttachmentFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV>::value,
                          "PhysicalDeviceLinearColorAttachmentFeaturesNV is not nothrow_move_constructible!" );

//=== VK_KHR_shader_maximal_reconvergence ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR>::value,
                          "PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_EXT_image_compression_control_swapchain ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceImageCompressionControlSwapchainFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT>::value,
                          "PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_QCOM_image_processing ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM ) == sizeof( VkImageViewSampleWeightCreateInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM>::value,
                          "ImageViewSampleWeightCreateInfoQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM ) == sizeof( VkPhysicalDeviceImageProcessingFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM>::value,
                          "PhysicalDeviceImageProcessingFeaturesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM ) ==
                            sizeof( VkPhysicalDeviceImageProcessingPropertiesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM>::value,
                          "PhysicalDeviceImageProcessingPropertiesQCOM is not nothrow_move_constructible!" );

//=== VK_EXT_nested_command_buffer ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceNestedCommandBufferFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferFeaturesEXT>::value,
                          "PhysicalDeviceNestedCommandBufferFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceNestedCommandBufferPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferPropertiesEXT>::value,
                          "PhysicalDeviceNestedCommandBufferPropertiesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_external_memory_acquire_unmodified ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalMemoryAcquireUnmodifiedEXT ) == sizeof( VkExternalMemoryAcquireUnmodifiedEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalMemoryAcquireUnmodifiedEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalMemoryAcquireUnmodifiedEXT>::value,
                          "ExternalMemoryAcquireUnmodifiedEXT is not nothrow_move_constructible!" );

//=== VK_EXT_extended_dynamic_state3 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3FeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceExtendedDynamicState3FeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3FeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3FeaturesEXT>::value,
                          "PhysicalDeviceExtendedDynamicState3FeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3PropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceExtendedDynamicState3PropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3PropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3PropertiesEXT>::value,
                          "PhysicalDeviceExtendedDynamicState3PropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT ) == sizeof( VkColorBlendEquationEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT>::value,
                          "ColorBlendEquationEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT ) == sizeof( VkColorBlendAdvancedEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT>::value,
                          "ColorBlendAdvancedEXT is not nothrow_move_constructible!" );

//=== VK_EXT_subpass_merge_feedback ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassMergeFeedbackFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceSubpassMergeFeedbackFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassMergeFeedbackFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassMergeFeedbackFeaturesEXT>::value,
                          "PhysicalDeviceSubpassMergeFeedbackFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassCreationControlEXT ) == sizeof( VkRenderPassCreationControlEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassCreationControlEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassCreationControlEXT>::value,
                          "RenderPassCreationControlEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackInfoEXT ) == sizeof( VkRenderPassCreationFeedbackInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackInfoEXT>::value,
                          "RenderPassCreationFeedbackInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackCreateInfoEXT ) == sizeof( VkRenderPassCreationFeedbackCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackCreateInfoEXT>::value,
                          "RenderPassCreationFeedbackCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackInfoEXT ) == sizeof( VkRenderPassSubpassFeedbackInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackInfoEXT>::value,
                          "RenderPassSubpassFeedbackInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackCreateInfoEXT ) == sizeof( VkRenderPassSubpassFeedbackCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackCreateInfoEXT>::value,
                          "RenderPassSubpassFeedbackCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_LUNARG_direct_driver_loading ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DirectDriverLoadingInfoLUNARG ) == sizeof( VkDirectDriverLoadingInfoLUNARG ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DirectDriverLoadingInfoLUNARG>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DirectDriverLoadingInfoLUNARG>::value,
                          "DirectDriverLoadingInfoLUNARG is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DirectDriverLoadingListLUNARG ) == sizeof( VkDirectDriverLoadingListLUNARG ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DirectDriverLoadingListLUNARG>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DirectDriverLoadingListLUNARG>::value,
                          "DirectDriverLoadingListLUNARG is not nothrow_move_constructible!" );

//=== VK_EXT_shader_module_identifier ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceShaderModuleIdentifierFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierFeaturesEXT>::value,
                          "PhysicalDeviceShaderModuleIdentifierFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierPropertiesEXT ) ==
                            sizeof( VkPhysicalDeviceShaderModuleIdentifierPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierPropertiesEXT>::value,
                          "PhysicalDeviceShaderModuleIdentifierPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineShaderStageModuleIdentifierCreateInfoEXT ) ==
                            sizeof( VkPipelineShaderStageModuleIdentifierCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineShaderStageModuleIdentifierCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineShaderStageModuleIdentifierCreateInfoEXT>::value,
                          "PipelineShaderStageModuleIdentifierCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderModuleIdentifierEXT ) == sizeof( VkShaderModuleIdentifierEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ShaderModuleIdentifierEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderModuleIdentifierEXT>::value,
                          "ShaderModuleIdentifierEXT is not nothrow_move_constructible!" );

//=== VK_EXT_rasterization_order_attachment_access ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT>::value,
                          "PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_NV_optical_flow ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowFeaturesNV ) == sizeof( VkPhysicalDeviceOpticalFlowFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowFeaturesNV>::value,
                          "PhysicalDeviceOpticalFlowFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowPropertiesNV ) == sizeof( VkPhysicalDeviceOpticalFlowPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowPropertiesNV>::value,
                          "PhysicalDeviceOpticalFlowPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatInfoNV ) == sizeof( VkOpticalFlowImageFormatInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatInfoNV>::value,
                          "OpticalFlowImageFormatInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatPropertiesNV ) == sizeof( VkOpticalFlowImageFormatPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatPropertiesNV>::value,
                          "OpticalFlowImageFormatPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OpticalFlowSessionNV ) == sizeof( VkOpticalFlowSessionNV ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OpticalFlowSessionNV>::value,
                          "OpticalFlowSessionNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateInfoNV ) == sizeof( VkOpticalFlowSessionCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateInfoNV>::value,
                          "OpticalFlowSessionCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreatePrivateDataInfoNV ) == sizeof( VkOpticalFlowSessionCreatePrivateDataInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreatePrivateDataInfoNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreatePrivateDataInfoNV>::value,
                          "OpticalFlowSessionCreatePrivateDataInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OpticalFlowExecuteInfoNV ) == sizeof( VkOpticalFlowExecuteInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::OpticalFlowExecuteInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OpticalFlowExecuteInfoNV>::value,
                          "OpticalFlowExecuteInfoNV is not nothrow_move_constructible!" );

//=== VK_EXT_legacy_dithering ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyDitheringFeaturesEXT ) == sizeof( VkPhysicalDeviceLegacyDitheringFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyDitheringFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyDitheringFeaturesEXT>::value,
                          "PhysicalDeviceLegacyDitheringFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_pipeline_protected_access ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeaturesEXT ) ==
                            sizeof( VkPhysicalDevicePipelineProtectedAccessFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeaturesEXT>::value,
                          "PhysicalDevicePipelineProtectedAccessFeaturesEXT is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
//=== VK_ANDROID_external_format_resolve ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolveFeaturesANDROID ) ==
                            sizeof( VkPhysicalDeviceExternalFormatResolveFeaturesANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolveFeaturesANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolveFeaturesANDROID>::value,
                          "PhysicalDeviceExternalFormatResolveFeaturesANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolvePropertiesANDROID ) ==
                            sizeof( VkPhysicalDeviceExternalFormatResolvePropertiesANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolvePropertiesANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolvePropertiesANDROID>::value,
                          "PhysicalDeviceExternalFormatResolvePropertiesANDROID is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatResolvePropertiesANDROID ) ==
                            sizeof( VkAndroidHardwareBufferFormatResolvePropertiesANDROID ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatResolvePropertiesANDROID>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatResolvePropertiesANDROID>::value,
                          "AndroidHardwareBufferFormatResolvePropertiesANDROID is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

//=== VK_KHR_maintenance5 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5FeaturesKHR ) == sizeof( VkPhysicalDeviceMaintenance5FeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5FeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5FeaturesKHR>::value,
                          "PhysicalDeviceMaintenance5FeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5PropertiesKHR ) == sizeof( VkPhysicalDeviceMaintenance5PropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5PropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5PropertiesKHR>::value,
                          "PhysicalDeviceMaintenance5PropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::RenderingAreaInfoKHR ) == sizeof( VkRenderingAreaInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::RenderingAreaInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::RenderingAreaInfoKHR>::value,
                          "RenderingAreaInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::DeviceImageSubresourceInfoKHR ) == sizeof( VkDeviceImageSubresourceInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::DeviceImageSubresourceInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::DeviceImageSubresourceInfoKHR>::value,
                          "DeviceImageSubresourceInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImageSubresource2KHR ) == sizeof( VkImageSubresource2KHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImageSubresource2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImageSubresource2KHR>::value,
                          "ImageSubresource2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SubresourceLayout2KHR ) == sizeof( VkSubresourceLayout2KHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SubresourceLayout2KHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SubresourceLayout2KHR>::value,
                          "SubresourceLayout2KHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineCreateFlags2CreateInfoKHR ) == sizeof( VkPipelineCreateFlags2CreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineCreateFlags2CreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineCreateFlags2CreateInfoKHR>::value,
                          "PipelineCreateFlags2CreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BufferUsageFlags2CreateInfoKHR ) == sizeof( VkBufferUsageFlags2CreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BufferUsageFlags2CreateInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BufferUsageFlags2CreateInfoKHR>::value,
                          "BufferUsageFlags2CreateInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_ray_tracing_position_fetch ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPositionFetchFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPositionFetchFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPositionFetchFeaturesKHR>::value,
                          "PhysicalDeviceRayTracingPositionFetchFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_EXT_shader_object ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderEXT ) == sizeof( VkShaderEXT ), "handle and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderEXT>::value, "ShaderEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectFeaturesEXT ) == sizeof( VkPhysicalDeviceShaderObjectFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectFeaturesEXT>::value,
                          "PhysicalDeviceShaderObjectFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectPropertiesEXT ) == sizeof( VkPhysicalDeviceShaderObjectPropertiesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectPropertiesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectPropertiesEXT>::value,
                          "PhysicalDeviceShaderObjectPropertiesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ShaderCreateInfoEXT ) == sizeof( VkShaderCreateInfoEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ShaderCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ShaderCreateInfoEXT>::value,
                          "ShaderCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_QCOM_tile_properties ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM ) == sizeof( VkPhysicalDeviceTilePropertiesFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM>::value,
                          "PhysicalDeviceTilePropertiesFeaturesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::TilePropertiesQCOM ) == sizeof( VkTilePropertiesQCOM ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::TilePropertiesQCOM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::TilePropertiesQCOM>::value,
                          "TilePropertiesQCOM is not nothrow_move_constructible!" );

//=== VK_SEC_amigo_profiling ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC ) == sizeof( VkPhysicalDeviceAmigoProfilingFeaturesSEC ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC>::value,
                          "PhysicalDeviceAmigoProfilingFeaturesSEC is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC ) == sizeof( VkAmigoProfilingSubmitInfoSEC ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC>::value,
                          "AmigoProfilingSubmitInfoSEC is not nothrow_move_constructible!" );

//=== VK_QCOM_multiview_per_view_viewports ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM ) ==
                            sizeof( VkPhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM>::value,
                          "PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM is not nothrow_move_constructible!" );

//=== VK_NV_ray_tracing_invocation_reorder ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesNV ) ==
                            sizeof( VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesNV>::value,
                          "PhysicalDeviceRayTracingInvocationReorderPropertiesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesNV>::value,
                          "PhysicalDeviceRayTracingInvocationReorderFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_extended_sparse_address_space ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceExtendedSparseAddressSpaceFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV>::value,
                          "PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpacePropertiesNV ) ==
                            sizeof( VkPhysicalDeviceExtendedSparseAddressSpacePropertiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpacePropertiesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpacePropertiesNV>::value,
                          "PhysicalDeviceExtendedSparseAddressSpacePropertiesNV is not nothrow_move_constructible!" );

//=== VK_EXT_mutable_descriptor_type ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT>::value,
                          "PhysicalDeviceMutableDescriptorTypeFeaturesEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT ) == sizeof( VkMutableDescriptorTypeListEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT>::value,
                          "MutableDescriptorTypeListEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT ) == sizeof( VkMutableDescriptorTypeCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT>::value,
                          "MutableDescriptorTypeCreateInfoEXT is not nothrow_move_constructible!" );

//=== VK_EXT_layer_settings ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LayerSettingsCreateInfoEXT ) == sizeof( VkLayerSettingsCreateInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LayerSettingsCreateInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LayerSettingsCreateInfoEXT>::value,
                          "LayerSettingsCreateInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LayerSettingEXT ) == sizeof( VkLayerSettingEXT ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LayerSettingEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LayerSettingEXT>::value,
                          "LayerSettingEXT is not nothrow_move_constructible!" );

//=== VK_ARM_shader_core_builtins ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsFeaturesARM ) ==
                            sizeof( VkPhysicalDeviceShaderCoreBuiltinsFeaturesARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsFeaturesARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsFeaturesARM>::value,
                          "PhysicalDeviceShaderCoreBuiltinsFeaturesARM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsPropertiesARM ) ==
                            sizeof( VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsPropertiesARM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsPropertiesARM>::value,
                          "PhysicalDeviceShaderCoreBuiltinsPropertiesARM is not nothrow_move_constructible!" );

//=== VK_EXT_pipeline_library_group_handles ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT ) ==
                            sizeof( VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT>::value,
                          "PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_EXT_dynamic_rendering_unused_attachments ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT>::value,
                          "PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_NV_low_latency2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LatencySleepModeInfoNV ) == sizeof( VkLatencySleepModeInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LatencySleepModeInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LatencySleepModeInfoNV>::value,
                          "LatencySleepModeInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LatencySleepInfoNV ) == sizeof( VkLatencySleepInfoNV ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LatencySleepInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LatencySleepInfoNV>::value,
                          "LatencySleepInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SetLatencyMarkerInfoNV ) == sizeof( VkSetLatencyMarkerInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SetLatencyMarkerInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SetLatencyMarkerInfoNV>::value,
                          "SetLatencyMarkerInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::GetLatencyMarkerInfoNV ) == sizeof( VkGetLatencyMarkerInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::GetLatencyMarkerInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::GetLatencyMarkerInfoNV>::value,
                          "GetLatencyMarkerInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LatencyTimingsFrameReportNV ) == sizeof( VkLatencyTimingsFrameReportNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LatencyTimingsFrameReportNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LatencyTimingsFrameReportNV>::value,
                          "LatencyTimingsFrameReportNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LatencySubmissionPresentIdNV ) == sizeof( VkLatencySubmissionPresentIdNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LatencySubmissionPresentIdNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LatencySubmissionPresentIdNV>::value,
                          "LatencySubmissionPresentIdNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SwapchainLatencyCreateInfoNV ) == sizeof( VkSwapchainLatencyCreateInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SwapchainLatencyCreateInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SwapchainLatencyCreateInfoNV>::value,
                          "SwapchainLatencyCreateInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::OutOfBandQueueTypeInfoNV ) == sizeof( VkOutOfBandQueueTypeInfoNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::OutOfBandQueueTypeInfoNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::OutOfBandQueueTypeInfoNV>::value,
                          "OutOfBandQueueTypeInfoNV is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::LatencySurfaceCapabilitiesNV ) == sizeof( VkLatencySurfaceCapabilitiesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::LatencySurfaceCapabilitiesNV>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::LatencySurfaceCapabilitiesNV>::value,
                          "LatencySurfaceCapabilitiesNV is not nothrow_move_constructible!" );

//=== VK_KHR_cooperative_matrix ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesKHR ) == sizeof( VkCooperativeMatrixPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesKHR>::value,
                          "CooperativeMatrixPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesKHR ) == sizeof( VkPhysicalDeviceCooperativeMatrixFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesKHR>::value,
                          "PhysicalDeviceCooperativeMatrixFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesKHR ) ==
                            sizeof( VkPhysicalDeviceCooperativeMatrixPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesKHR>::value,
                          "PhysicalDeviceCooperativeMatrixPropertiesKHR is not nothrow_move_constructible!" );

//=== VK_QCOM_multiview_per_view_render_areas ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM ) ==
                            sizeof( VkPhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM>::value,
                          "PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM ) ==
                            sizeof( VkMultiviewPerViewRenderAreasRenderPassBeginInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM>::value,
                          "MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM is not nothrow_move_constructible!" );

//=== VK_KHR_video_decode_av1 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeAV1ProfileInfoKHR ) == sizeof( VkVideoDecodeAV1ProfileInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeAV1ProfileInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeAV1ProfileInfoKHR>::value,
                          "VideoDecodeAV1ProfileInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeAV1CapabilitiesKHR ) == sizeof( VkVideoDecodeAV1CapabilitiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeAV1CapabilitiesKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeAV1CapabilitiesKHR>::value,
                          "VideoDecodeAV1CapabilitiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeAV1SessionParametersCreateInfoKHR ) ==
                            sizeof( VkVideoDecodeAV1SessionParametersCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeAV1SessionParametersCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeAV1SessionParametersCreateInfoKHR>::value,
                          "VideoDecodeAV1SessionParametersCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeAV1PictureInfoKHR ) == sizeof( VkVideoDecodeAV1PictureInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeAV1PictureInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeAV1PictureInfoKHR>::value,
                          "VideoDecodeAV1PictureInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoDecodeAV1DpbSlotInfoKHR ) == sizeof( VkVideoDecodeAV1DpbSlotInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoDecodeAV1DpbSlotInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoDecodeAV1DpbSlotInfoKHR>::value,
                          "VideoDecodeAV1DpbSlotInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_video_maintenance1 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoMaintenance1FeaturesKHR ) == sizeof( VkPhysicalDeviceVideoMaintenance1FeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoMaintenance1FeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoMaintenance1FeaturesKHR>::value,
                          "PhysicalDeviceVideoMaintenance1FeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VideoInlineQueryInfoKHR ) == sizeof( VkVideoInlineQueryInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VideoInlineQueryInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VideoInlineQueryInfoKHR>::value,
                          "VideoInlineQueryInfoKHR is not nothrow_move_constructible!" );

//=== VK_NV_per_stage_descriptor_set ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDevicePerStageDescriptorSetFeaturesNV ) ==
                            sizeof( VkPhysicalDevicePerStageDescriptorSetFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDevicePerStageDescriptorSetFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDevicePerStageDescriptorSetFeaturesNV>::value,
                          "PhysicalDevicePerStageDescriptorSetFeaturesNV is not nothrow_move_constructible!" );

//=== VK_QCOM_image_processing2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2FeaturesQCOM ) == sizeof( VkPhysicalDeviceImageProcessing2FeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2FeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2FeaturesQCOM>::value,
                          "PhysicalDeviceImageProcessing2FeaturesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2PropertiesQCOM ) ==
                            sizeof( VkPhysicalDeviceImageProcessing2PropertiesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2PropertiesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2PropertiesQCOM>::value,
                          "PhysicalDeviceImageProcessing2PropertiesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerBlockMatchWindowCreateInfoQCOM ) == sizeof( VkSamplerBlockMatchWindowCreateInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerBlockMatchWindowCreateInfoQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerBlockMatchWindowCreateInfoQCOM>::value,
                          "SamplerBlockMatchWindowCreateInfoQCOM is not nothrow_move_constructible!" );

//=== VK_QCOM_filter_cubic_weights ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicWeightsFeaturesQCOM ) == sizeof( VkPhysicalDeviceCubicWeightsFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicWeightsFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicWeightsFeaturesQCOM>::value,
                          "PhysicalDeviceCubicWeightsFeaturesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerCubicWeightsCreateInfoQCOM ) == sizeof( VkSamplerCubicWeightsCreateInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerCubicWeightsCreateInfoQCOM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerCubicWeightsCreateInfoQCOM>::value,
                          "SamplerCubicWeightsCreateInfoQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BlitImageCubicWeightsInfoQCOM ) == sizeof( VkBlitImageCubicWeightsInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BlitImageCubicWeightsInfoQCOM>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BlitImageCubicWeightsInfoQCOM>::value,
                          "BlitImageCubicWeightsInfoQCOM is not nothrow_move_constructible!" );

//=== VK_QCOM_ycbcr_degamma ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrDegammaFeaturesQCOM ) == sizeof( VkPhysicalDeviceYcbcrDegammaFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrDegammaFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrDegammaFeaturesQCOM>::value,
                          "PhysicalDeviceYcbcrDegammaFeaturesQCOM is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM ) ==
                            sizeof( VkSamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM>::value,
                          "SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM is not nothrow_move_constructible!" );

//=== VK_QCOM_filter_cubic_clamp ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicClampFeaturesQCOM ) == sizeof( VkPhysicalDeviceCubicClampFeaturesQCOM ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicClampFeaturesQCOM>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicClampFeaturesQCOM>::value,
                          "PhysicalDeviceCubicClampFeaturesQCOM is not nothrow_move_constructible!" );

//=== VK_EXT_attachment_feedback_loop_dynamic_state ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT ) ==
                            sizeof( VkPhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>::value,
                          "PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT is not nothrow_move_constructible!" );

//=== VK_KHR_vertex_attribute_divisor ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesKHR ) ==
                            sizeof( VkPhysicalDeviceVertexAttributeDivisorPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesKHR>::value,
                          "PhysicalDeviceVertexAttributeDivisorPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionKHR ) == sizeof( VkVertexInputBindingDivisorDescriptionKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionKHR>::value,
                          "VertexInputBindingDivisorDescriptionKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoKHR ) ==
                            sizeof( VkPipelineVertexInputDivisorStateCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoKHR>::value,
                          "PipelineVertexInputDivisorStateCreateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesKHR>::value,
                          "PhysicalDeviceVertexAttributeDivisorFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_KHR_shader_float_controls2 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloatControls2FeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceShaderFloatControls2FeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloatControls2FeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloatControls2FeaturesKHR>::value,
                          "PhysicalDeviceShaderFloatControls2FeaturesKHR is not nothrow_move_constructible!" );

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
//=== VK_QNX_external_memory_screen_buffer ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ScreenBufferPropertiesQNX ) == sizeof( VkScreenBufferPropertiesQNX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ScreenBufferPropertiesQNX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ScreenBufferPropertiesQNX>::value,
                          "ScreenBufferPropertiesQNX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ScreenBufferFormatPropertiesQNX ) == sizeof( VkScreenBufferFormatPropertiesQNX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ScreenBufferFormatPropertiesQNX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ScreenBufferFormatPropertiesQNX>::value,
                          "ScreenBufferFormatPropertiesQNX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ImportScreenBufferInfoQNX ) == sizeof( VkImportScreenBufferInfoQNX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ImportScreenBufferInfoQNX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ImportScreenBufferInfoQNX>::value,
                          "ImportScreenBufferInfoQNX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::ExternalFormatQNX ) == sizeof( VkExternalFormatQNX ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::ExternalFormatQNX>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::ExternalFormatQNX>::value,
                          "ExternalFormatQNX is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX ) ==
                            sizeof( VkPhysicalDeviceExternalMemoryScreenBufferFeaturesQNX ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX>::value,
                          "PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX is not nothrow_move_constructible!" );
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

//=== VK_MSFT_layered_driver ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredDriverPropertiesMSFT ) == sizeof( VkPhysicalDeviceLayeredDriverPropertiesMSFT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredDriverPropertiesMSFT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredDriverPropertiesMSFT>::value,
                          "PhysicalDeviceLayeredDriverPropertiesMSFT is not nothrow_move_constructible!" );

//=== VK_KHR_index_type_uint8 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesKHR ) == sizeof( VkPhysicalDeviceIndexTypeUint8FeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesKHR>::value,
                          "PhysicalDeviceIndexTypeUint8FeaturesKHR is not nothrow_move_constructible!" );

//=== VK_KHR_line_rasterization ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesKHR ) == sizeof( VkPhysicalDeviceLineRasterizationFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesKHR>::value,
                          "PhysicalDeviceLineRasterizationFeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesKHR ) ==
                            sizeof( VkPhysicalDeviceLineRasterizationPropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesKHR>::value,
                          "PhysicalDeviceLineRasterizationPropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoKHR ) ==
                            sizeof( VkPipelineRasterizationLineStateCreateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoKHR>::value,
                          "PipelineRasterizationLineStateCreateInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_calibrated_timestamps ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoKHR ) == sizeof( VkCalibratedTimestampInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoKHR>::value,
                          "CalibratedTimestampInfoKHR is not nothrow_move_constructible!" );

//=== VK_KHR_shader_expect_assume ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderExpectAssumeFeaturesKHR ) ==
                            sizeof( VkPhysicalDeviceShaderExpectAssumeFeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderExpectAssumeFeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderExpectAssumeFeaturesKHR>::value,
                          "PhysicalDeviceShaderExpectAssumeFeaturesKHR is not nothrow_move_constructible!" );

//=== VK_KHR_maintenance6 ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6FeaturesKHR ) == sizeof( VkPhysicalDeviceMaintenance6FeaturesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6FeaturesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6FeaturesKHR>::value,
                          "PhysicalDeviceMaintenance6FeaturesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6PropertiesKHR ) == sizeof( VkPhysicalDeviceMaintenance6PropertiesKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6PropertiesKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6PropertiesKHR>::value,
                          "PhysicalDeviceMaintenance6PropertiesKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindMemoryStatusKHR ) == sizeof( VkBindMemoryStatusKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindMemoryStatusKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindMemoryStatusKHR>::value,
                          "BindMemoryStatusKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindDescriptorSetsInfoKHR ) == sizeof( VkBindDescriptorSetsInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindDescriptorSetsInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindDescriptorSetsInfoKHR>::value,
                          "BindDescriptorSetsInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PushConstantsInfoKHR ) == sizeof( VkPushConstantsInfoKHR ), "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PushConstantsInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PushConstantsInfoKHR>::value,
                          "PushConstantsInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PushDescriptorSetInfoKHR ) == sizeof( VkPushDescriptorSetInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PushDescriptorSetInfoKHR>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PushDescriptorSetInfoKHR>::value,
                          "PushDescriptorSetInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PushDescriptorSetWithTemplateInfoKHR ) == sizeof( VkPushDescriptorSetWithTemplateInfoKHR ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PushDescriptorSetWithTemplateInfoKHR>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PushDescriptorSetWithTemplateInfoKHR>::value,
                          "PushDescriptorSetWithTemplateInfoKHR is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::SetDescriptorBufferOffsetsInfoEXT ) == sizeof( VkSetDescriptorBufferOffsetsInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::SetDescriptorBufferOffsetsInfoEXT>::value, "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::SetDescriptorBufferOffsetsInfoEXT>::value,
                          "SetDescriptorBufferOffsetsInfoEXT is not nothrow_move_constructible!" );

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::BindDescriptorBufferEmbeddedSamplersInfoEXT ) ==
                            sizeof( VkBindDescriptorBufferEmbeddedSamplersInfoEXT ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::BindDescriptorBufferEmbeddedSamplersInfoEXT>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::BindDescriptorBufferEmbeddedSamplersInfoEXT>::value,
                          "BindDescriptorBufferEmbeddedSamplersInfoEXT is not nothrow_move_constructible!" );

//=== VK_NV_descriptor_pool_overallocation ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorPoolOverallocationFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceDescriptorPoolOverallocationFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorPoolOverallocationFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorPoolOverallocationFeaturesNV>::value,
                          "PhysicalDeviceDescriptorPoolOverallocationFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_raw_access_chains ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRawAccessChainsFeaturesNV ) == sizeof( VkPhysicalDeviceRawAccessChainsFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRawAccessChainsFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRawAccessChainsFeaturesNV>::value,
                          "PhysicalDeviceRawAccessChainsFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_shader_atomic_float16_vector ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceShaderAtomicFloat16VectorFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV>::value,
                          "PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV is not nothrow_move_constructible!" );

//=== VK_NV_ray_tracing_validation ===

VULKAN_HPP_STATIC_ASSERT( sizeof( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingValidationFeaturesNV ) ==
                            sizeof( VkPhysicalDeviceRayTracingValidationFeaturesNV ),
                          "struct and wrapper have different size!" );
VULKAN_HPP_STATIC_ASSERT( std::is_standard_layout<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingValidationFeaturesNV>::value,
                          "struct wrapper is not a standard layout!" );
VULKAN_HPP_STATIC_ASSERT( std::is_nothrow_move_constructible<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingValidationFeaturesNV>::value,
                          "PhysicalDeviceRayTracingValidationFeaturesNV is not nothrow_move_constructible!" );

#endif

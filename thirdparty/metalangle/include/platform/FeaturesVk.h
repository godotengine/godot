//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FeaturesVk.h: Optional features for the Vulkan renderer.
//

#ifndef ANGLE_PLATFORM_FEATURESVK_H_
#define ANGLE_PLATFORM_FEATURESVK_H_

#include "platform/Feature.h"

namespace angle
{

struct FeaturesVk : FeatureSetBase
{
    FeaturesVk();
    ~FeaturesVk();

    // Line segment rasterization must follow OpenGL rules. This means using an algorithm similar
    // to Bresenham's. Vulkan uses a different algorithm. This feature enables the use of pixel
    // shader patching to implement OpenGL basic line rasterization rules. This feature will
    // normally always be enabled. Exposing it as an option enables performance testing.
    Feature basicGLLineRasterization = {
        "basic_gl_line_rasterization", FeatureCategory::VulkanFeatures,
        "Enable the use of pixel shader patching to implement OpenGL basic line "
        "rasterization rules",
        &members};

    // Flips the viewport to render upside-down. This has the effect to render the same way as
    // OpenGL. If this feature gets enabled, we enable the KHR_MAINTENANCE_1 extension to allow
    // negative viewports. We inverse rendering to the backbuffer by reversing the height of the
    // viewport and increasing Y by the height. So if the viewport was (0,0,width,height), it
    // becomes (0, height, width, -height). Unfortunately, when we start doing this, we also need
    // to adjust a lot of places since the rendering now happens upside-down. Affected places so
    // far:
    // -readPixels
    // -copyTexImage
    // -framebuffer blit
    // -generating mipmaps
    // -Point sprites tests
    // -texStorage
    Feature flipViewportY = {"flip_viewport_y", FeatureCategory::VulkanFeatures,
                             "Flips the viewport to render upside-down", &members};

    // Add an extra copy region when using vkCmdCopyBuffer as the Windows Intel driver seems
    // to have a bug where the last region is ignored.
    Feature extraCopyBufferRegion = {
        "extra_copy_buffer_region", FeatureCategory::VulkanWorkarounds,
        "Some drivers seem to have a bug where the last copy region in vkCmdCopyBuffer is ignored",
        &members};

    // This flag is added for the sole purpose of end2end tests, to test the correctness
    // of various algorithms when a fallback format is used, such as using a packed format to
    // emulate a depth- or stencil-only format.
    Feature forceFallbackFormat = {"force_fallback_format", FeatureCategory::VulkanWorkarounds,
                                   "Force a fallback format for angle_end2end_tests", &members};

    // On some NVIDIA drivers the point size range reported from the API is inconsistent with the
    // actual behavior. Clamp the point size to the value from the API to fix this.
    // Tracked in http://anglebug.com/2970.
    Feature clampPointSize = {
        "clamp_point_size", FeatureCategory::VulkanWorkarounds,
        "The point size range reported from the API is inconsistent with the actual behavior",
        &members, "http://anglebug.com/2970"};

    // On some android devices, the memory barrier between the compute shader that converts vertex
    // attributes and the vertex shader that reads from it is ineffective.  Only known workaround is
    // to perform a flush after the conversion.  http://anglebug.com/3016
    Feature flushAfterVertexConversion = {
        "flush_after_vertex_conversion", FeatureCategory::VulkanWorkarounds,
        "The memory barrier between the compute shader that converts vertex attributes and the "
        "vertex shader that reads from it is ineffective",
        &members, "http://anglebug.com/3016"};

    // Whether the VkDevice supports the VK_KHR_incremental_present extension, on which the
    // EGL_KHR_swap_buffers_with_damage extension can be layered.
    Feature supportsIncrementalPresent = {
        "supports_incremental_present", FeatureCategory::VulkanFeatures,
        "VkDevice supports the VK_KHR_incremental_present extension", &members};

    // Whether texture copies on cube map targets should be done on GPU.  This is a workaround for
    // Intel drivers on windows that have an issue with creating single-layer views on cube map
    // textures.
    Feature forceCPUPathForCubeMapCopy = {
        "force_cpu_path_for_cube_map_copy", FeatureCategory::VulkanWorkarounds,
        "Some drivers have an issue with creating single-layer views on cube map textures",
        &members};

    // Whether the VkDevice supports the VK_ANDROID_external_memory_android_hardware_buffer
    // extension, on which the EGL_ANDROID_image_native_buffer extension can be layered.
    Feature supportsAndroidHardwareBuffer = {
        "supports_android_hardware_buffer", FeatureCategory::VulkanFeatures,
        "VkDevice supports the VK_ANDROID_external_memory_android_hardware_buffer extension",
        &members};

    // Whether the VkDevice supports the VK_KHR_external_memory_fd extension, on which the
    // GL_EXT_memory_object_fd extension can be layered.
    Feature supportsExternalMemoryFd = {
        "supports_external_memory_fd", FeatureCategory::VulkanFeatures,
        "VkDevice supports the VK_KHR_external_memory_fd extension", &members};

    // Whether the VkDevice supports the VK_KHR_external_semaphore_fd extension, on which the
    // GL_EXT_semaphore_fd extension can be layered.
    Feature supportsExternalSemaphoreFd = {
        "supports_external_semaphore_fd", FeatureCategory::VulkanFeatures,
        "VkDevice supports the VK_KHR_external_semaphore_fd extension", &members};

    // Whether the VkDevice supports the VK_EXT_shader_stencil_export extension, which is used to
    // perform multisampled resolve of stencil buffer.  A multi-step workaround is used instead if
    // this extension is not available.
    Feature supportsShaderStencilExport = {
        "supports_shader_stencil_export", FeatureCategory::VulkanFeatures,
        "VkDevice supports the VK_EXT_shader_stencil_export extension", &members};

    // Where VK_EXT_transform_feedback is not support, an emulation path is used.
    // http://anglebug.com/3205
    Feature emulateTransformFeedback = {
        "emulate_transform_feedback", FeatureCategory::VulkanFeatures,
        "Emulate transform feedback as the VK_EXT_transform_feedback is not present.", &members,
        "http://anglebug.com/3205"};

    // VK_PRESENT_MODE_FIFO_KHR causes random timeouts on Linux Intel. http://anglebug.com/3153
    Feature disableFifoPresentMode = {
        "disable_fifo_present_mode", FeatureCategory::VulkanWorkarounds,
        "VK_PRESENT_MODE_FIFO_KHR causes random timeouts", &members, "http://anglebug.com/3153"};

    // On Qualcomm, a bug is preventing us from using loadOp=Clear with inline commands in the
    // render pass.  http://anglebug.com/2361
    Feature restartRenderPassAfterLoadOpClear = {
        "restart_render_pass_after_load_op_clear", FeatureCategory::VulkanWorkarounds,
        "A bug is preventing us from using loadOp=Clear with inline commands in the render pass",
        &members, "http://anglebug.com/2361"};

    // On Qualcomm, gaps in bound descriptor set indices causes the post-gap sets to misbehave.
    // For example, binding only descriptor set 3 results in zero being read from a uniform buffer
    // object within that set.  This flag results in empty descriptor sets being bound for any
    // unused descriptor set to work around this issue.  http://anglebug.com/2727
    Feature bindEmptyForUnusedDescriptorSets = {
        "bind_empty_for_unused_descriptor_sets", FeatureCategory::VulkanWorkarounds,
        "Gaps in bound descriptor set indices causes the post-gap sets to misbehave", &members,
        "http://anglebug.com/2727"};

    // When the scissor is (0,0,0,0) on Windows Intel, the driver acts as if the scissor was
    // disabled.  Work-around this by setting the scissor to just outside of the render area
    // (e.g. (renderArea.x, renderArea.y, 1, 1)). http://anglebug.com/3407
    Feature forceNonZeroScissor = {
        "force_non_zero_scissor", FeatureCategory::VulkanWorkarounds,
        "When the scissor is (0,0,0,0), the driver acts as if the scissor was disabled", &members,
        "http://anglebug.com/3407"};

    // OES_depth_texture is a commonly expected feature on Android. However it
    // requires that D16_UNORM support texture filtering
    // (e.g. VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) and some devices
    // do not. Work-around this by setting saying D16_UNORM supports filtering
    // anyway.
    Feature forceD16TexFilter = {
        "force_D16_texture_filter", FeatureCategory::VulkanWorkarounds,
        "VK_FORMAT_D16_UNORM does not support VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT, "
        "which prevents OES_depth_texture from being supported.",
        &members, "http://anglebug.com/3452"};

    // On some android devices, vkCmdBlitImage with flipped coordinates blits incorrectly.  This
    // workaround makes sure this path is avoided.  http://anglebug.com/3498
    Feature disableFlippingBlitWithCommand = {
        "disable_flipping_blit_with_command", FeatureCategory::VulkanWorkarounds,
        "vkCmdBlitImage with flipped coordinates blits incorrectly.", &members,
        "http://anglebug.com/3498"};

    // On platform with Intel or AMD GPU, a window resizing would not trigger the vulkan driver to
    // return VK_ERROR_OUT_OF_DATE on swapchain present.  Work-around by query current window extent
    // every frame to detect a window resizing.
    // http://anglebug.com/3623, http://anglebug.com/3624, http://anglebug.com/3625
    Feature perFrameWindowSizeQuery = {
        "per_frame_window_size_query", FeatureCategory::VulkanWorkarounds,
        "Vulkan swapchain is not returning VK_ERROR_OUT_OF_DATE when window resizing", &members,
        "http://anglebug.com/3623, http://anglebug.com/3624, http://anglebug.com/3625"};

    // On Pixel1XL and Pixel2, reset a vkCommandBuffer seems to have side effects on binding
    // descriptor sets to it afterwards, Work-around by keep using transient vkCommandBuffer on
    // those devices.
    // http://b/135763283
    Feature transientCommandBuffer = {
        "transient_command_buffer", FeatureCategory::VulkanWorkarounds,
        "Keep using transient vkCommandBuffer to work around driver issue in reseting"
        "vkCommandBuffer",
        &members, "http://b/135763283"};

    // Seamful cube map emulation misbehaves on the AMD windows driver, so it's disallowed.
    Feature disallowSeamfulCubeMapEmulation = {
        "disallow_seamful_cube_map_emulation", FeatureCategory::VulkanWorkarounds,
        "Seamful cube map emulation misbehaves on some drivers, so it's disallowed", &members,
        "http://anglebug.com/3243"};

    // Qualcomm shader compiler doesn't support sampler arrays as parameters, so
    // revert to old RewriteStructSamplers behavior, which produces fewer.
    Feature forceOldRewriteStructSamplers = {
        "force_old_rewrite_struct_samplers", FeatureCategory::VulkanWorkarounds,
        "Some shader compilers don't support sampler arrays as parameters, so revert to old "
        "RewriteStructSamplers behavior, which produces fewer.",
        &members, "http://anglebug.com/2703"};

    // Whether the VkDevice supports the VK_EXT_swapchain_colorspace extension
    // http://anglebug.com/2514
    Feature supportsSwapchainColorspace = {
        "supports_swapchain_colorspace", FeatureCategory::VulkanFeatures,
        "VkDevice supports the VK_EXT_swapchain_colorspace extension", &members,
        "http://anglebug.com/2514"};
};

inline FeaturesVk::FeaturesVk()  = default;
inline FeaturesVk::~FeaturesVk() = default;

}  // namespace angle

#endif  // ANGLE_PLATFORM_FEATURESVK_H_

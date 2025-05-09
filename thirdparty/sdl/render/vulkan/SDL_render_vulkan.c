/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_VIDEO_RENDER_VULKAN

#define SDL_VULKAN_FRAME_QUEUE_DEPTH            2
#define SDL_VULKAN_NUM_VERTEX_BUFFERS           256
#define SDL_VULKAN_VERTEX_BUFFER_DEFAULT_SIZE   65536
#define SDL_VULKAN_CONSTANT_BUFFER_DEFAULT_SIZE 65536
#define SDL_VULKAN_NUM_UPLOAD_BUFFERS           32
#define SDL_VULKAN_MAX_DESCRIPTOR_SETS          4096

#define SDL_VULKAN_VALIDATION_LAYER_NAME        "VK_LAYER_KHRONOS_validation"

#define VK_NO_PROTOTYPES
#include "../../video/SDL_vulkan_internal.h"
#include "../../video/SDL_sysvideo.h"
#include "../SDL_sysrender.h"
#include "../SDL_d3dmath.h"
#include "../../video/SDL_pixels_c.h"
#include "SDL_shaders_vulkan.h"

#define SET_ERROR_CODE(message, rc)                                                                 \
    if (SDL_GetHintBoolean(SDL_HINT_RENDER_VULKAN_DEBUG, false)) {                                  \
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s: %s", message, SDL_Vulkan_GetResultString(rc)); \
        SDL_TriggerBreakpoint();                                                                    \
    }                                                                                               \
    SDL_SetError("%s: %s", message, SDL_Vulkan_GetResultString(rc))                                 \

#define SET_ERROR_MESSAGE(message)                                                                  \
    if (SDL_GetHintBoolean(SDL_HINT_RENDER_VULKAN_DEBUG, false)) {                                  \
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s", message);                                     \
        SDL_TriggerBreakpoint();                                                                    \
    }                                                                                               \
    SDL_SetError("%s", message)                                                                     \

#define VULKAN_FUNCTIONS()                                              \
    VULKAN_DEVICE_FUNCTION(vkAcquireNextImageKHR)                       \
    VULKAN_DEVICE_FUNCTION(vkAllocateCommandBuffers)                    \
    VULKAN_DEVICE_FUNCTION(vkAllocateDescriptorSets)                    \
    VULKAN_DEVICE_FUNCTION(vkAllocateMemory)                            \
    VULKAN_DEVICE_FUNCTION(vkBeginCommandBuffer)                        \
    VULKAN_DEVICE_FUNCTION(vkBindBufferMemory)                          \
    VULKAN_DEVICE_FUNCTION(vkBindImageMemory)                           \
    VULKAN_DEVICE_FUNCTION(vkCmdBeginRenderPass)                        \
    VULKAN_DEVICE_FUNCTION(vkCmdBindDescriptorSets)                     \
    VULKAN_DEVICE_FUNCTION(vkCmdBindPipeline)                           \
    VULKAN_DEVICE_FUNCTION(vkCmdBindVertexBuffers)                      \
    VULKAN_DEVICE_FUNCTION(vkCmdClearColorImage)                        \
    VULKAN_DEVICE_FUNCTION(vkCmdCopyBufferToImage)                      \
    VULKAN_DEVICE_FUNCTION(vkCmdCopyImageToBuffer)                      \
    VULKAN_DEVICE_FUNCTION(vkCmdDraw)                                   \
    VULKAN_DEVICE_FUNCTION(vkCmdEndRenderPass)                          \
    VULKAN_DEVICE_FUNCTION(vkCmdPipelineBarrier)                        \
    VULKAN_DEVICE_FUNCTION(vkCmdPushConstants)                          \
    VULKAN_DEVICE_FUNCTION(vkCmdSetScissor)                             \
    VULKAN_DEVICE_FUNCTION(vkCmdSetViewport)                            \
    VULKAN_DEVICE_FUNCTION(vkCreateBuffer)                              \
    VULKAN_DEVICE_FUNCTION(vkCreateCommandPool)                         \
    VULKAN_DEVICE_FUNCTION(vkCreateDescriptorPool)                      \
    VULKAN_DEVICE_FUNCTION(vkCreateDescriptorSetLayout)                 \
    VULKAN_DEVICE_FUNCTION(vkCreateFence)                               \
    VULKAN_DEVICE_FUNCTION(vkCreateFramebuffer)                         \
    VULKAN_DEVICE_FUNCTION(vkCreateGraphicsPipelines)                   \
    VULKAN_DEVICE_FUNCTION(vkCreateImage)                               \
    VULKAN_DEVICE_FUNCTION(vkCreateImageView)                           \
    VULKAN_DEVICE_FUNCTION(vkCreatePipelineLayout)                      \
    VULKAN_DEVICE_FUNCTION(vkCreateRenderPass)                          \
    VULKAN_DEVICE_FUNCTION(vkCreateSampler)                             \
    VULKAN_DEVICE_FUNCTION(vkCreateSemaphore)                           \
    VULKAN_DEVICE_FUNCTION(vkCreateShaderModule)                        \
    VULKAN_DEVICE_FUNCTION(vkCreateSwapchainKHR)                        \
    VULKAN_DEVICE_FUNCTION(vkDestroyBuffer)                             \
    VULKAN_DEVICE_FUNCTION(vkDestroyCommandPool)                        \
    VULKAN_DEVICE_FUNCTION(vkDestroyDevice)                             \
    VULKAN_DEVICE_FUNCTION(vkDestroyDescriptorPool)                     \
    VULKAN_DEVICE_FUNCTION(vkDestroyDescriptorSetLayout)                \
    VULKAN_DEVICE_FUNCTION(vkDestroyFence)                              \
    VULKAN_DEVICE_FUNCTION(vkDestroyFramebuffer)                        \
    VULKAN_DEVICE_FUNCTION(vkDestroyImage)                              \
    VULKAN_DEVICE_FUNCTION(vkDestroyImageView)                          \
    VULKAN_DEVICE_FUNCTION(vkDestroyPipeline)                           \
    VULKAN_DEVICE_FUNCTION(vkDestroyPipelineLayout)                     \
    VULKAN_DEVICE_FUNCTION(vkDestroyRenderPass)                         \
    VULKAN_DEVICE_FUNCTION(vkDestroySampler)                            \
    VULKAN_DEVICE_FUNCTION(vkDestroySemaphore)                          \
    VULKAN_DEVICE_FUNCTION(vkDestroyShaderModule)                       \
    VULKAN_DEVICE_FUNCTION(vkDestroySwapchainKHR)                       \
    VULKAN_DEVICE_FUNCTION(vkDeviceWaitIdle)                            \
    VULKAN_DEVICE_FUNCTION(vkEndCommandBuffer)                          \
    VULKAN_DEVICE_FUNCTION(vkFreeCommandBuffers)                        \
    VULKAN_DEVICE_FUNCTION(vkFreeMemory)                                \
    VULKAN_DEVICE_FUNCTION(vkGetBufferMemoryRequirements)               \
    VULKAN_DEVICE_FUNCTION(vkGetImageMemoryRequirements)                \
    VULKAN_DEVICE_FUNCTION(vkGetDeviceQueue)                            \
    VULKAN_DEVICE_FUNCTION(vkGetFenceStatus)                            \
    VULKAN_DEVICE_FUNCTION(vkGetSwapchainImagesKHR)                     \
    VULKAN_DEVICE_FUNCTION(vkMapMemory)                                 \
    VULKAN_DEVICE_FUNCTION(vkQueuePresentKHR)                           \
    VULKAN_DEVICE_FUNCTION(vkQueueSubmit)                               \
    VULKAN_DEVICE_FUNCTION(vkResetCommandBuffer)                        \
    VULKAN_DEVICE_FUNCTION(vkResetCommandPool)                          \
    VULKAN_DEVICE_FUNCTION(vkResetDescriptorPool)                       \
    VULKAN_DEVICE_FUNCTION(vkResetFences)                               \
    VULKAN_DEVICE_FUNCTION(vkUnmapMemory)                               \
    VULKAN_DEVICE_FUNCTION(vkUpdateDescriptorSets)                      \
    VULKAN_DEVICE_FUNCTION(vkWaitForFences)                             \
    VULKAN_GLOBAL_FUNCTION(vkCreateInstance)                            \
    VULKAN_GLOBAL_FUNCTION(vkEnumerateInstanceExtensionProperties)      \
    VULKAN_GLOBAL_FUNCTION(vkEnumerateInstanceLayerProperties)          \
    VULKAN_INSTANCE_FUNCTION(vkCreateDevice)                            \
    VULKAN_INSTANCE_FUNCTION(vkDestroyInstance)                         \
    VULKAN_INSTANCE_FUNCTION(vkDestroySurfaceKHR)                       \
    VULKAN_INSTANCE_FUNCTION(vkEnumerateDeviceExtensionProperties)      \
    VULKAN_INSTANCE_FUNCTION(vkEnumeratePhysicalDevices)                \
    VULKAN_INSTANCE_FUNCTION(vkGetDeviceProcAddr)                       \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceFeatures)               \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceProperties)             \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceMemoryProperties)       \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceQueueFamilyProperties)  \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceFormatsKHR)      \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceSupportKHR)      \
    VULKAN_INSTANCE_FUNCTION(vkQueueWaitIdle)                           \
    VULKAN_OPTIONAL_INSTANCE_FUNCTION(vkGetPhysicalDeviceFeatures2KHR)              \
    VULKAN_OPTIONAL_INSTANCE_FUNCTION(vkGetPhysicalDeviceFormatProperties2KHR)      \
    VULKAN_OPTIONAL_INSTANCE_FUNCTION(vkGetPhysicalDeviceImageFormatProperties2KHR) \
    VULKAN_OPTIONAL_INSTANCE_FUNCTION(vkGetPhysicalDeviceMemoryProperties2KHR)      \
    VULKAN_OPTIONAL_INSTANCE_FUNCTION(vkGetPhysicalDeviceProperties2KHR)            \
    VULKAN_OPTIONAL_DEVICE_FUNCTION(vkCreateSamplerYcbcrConversionKHR)              \
    VULKAN_OPTIONAL_DEVICE_FUNCTION(vkDestroySamplerYcbcrConversionKHR)             \

#define VULKAN_DEVICE_FUNCTION(name)            static PFN_##name name = NULL;
#define VULKAN_GLOBAL_FUNCTION(name)            static PFN_##name name = NULL;
#define VULKAN_INSTANCE_FUNCTION(name)          static PFN_##name name = NULL;
#define VULKAN_OPTIONAL_INSTANCE_FUNCTION(name) static PFN_##name name = NULL;
#define VULKAN_OPTIONAL_DEVICE_FUNCTION(name)   static PFN_##name name = NULL;
VULKAN_FUNCTIONS()
#undef VULKAN_DEVICE_FUNCTION
#undef VULKAN_GLOBAL_FUNCTION
#undef VULKAN_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_DEVICE_FUNCTION

// Renderpass types
typedef enum {
    VULKAN_RENDERPASS_LOAD = 0,
    VULKAN_RENDERPASS_CLEAR = 1,
    VULKAN_RENDERPASS_COUNT
} VULKAN_RenderPass;

// Vertex shader, common values
typedef struct
{
    Float4X4 model;
    Float4X4 projectionAndView;
} VULKAN_VertexShaderConstants;

// These should mirror the definitions in VULKAN_PixelShader_Common.hlsli
//static const float TONEMAP_NONE = 0;
//static const float TONEMAP_LINEAR = 1;
static const float TONEMAP_CHROME = 2;

static const float INPUTTYPE_UNSPECIFIED = 0;
static const float INPUTTYPE_SRGB = 1;
static const float INPUTTYPE_SCRGB = 2;
static const float INPUTTYPE_HDR10 = 3;

// Pixel shader constants, common values
typedef struct
{
    float scRGB_output;
    float input_type;
    float color_scale;
    float pixel_art;

    float texel_width;
    float texel_height;
    float texture_width;
    float texture_height;

    float tonemap_method;
    float tonemap_factor1;
    float tonemap_factor2;
    float sdr_white_point;
} VULKAN_PixelShaderConstants;

// Per-vertex data
typedef struct
{
    float pos[2];
    float tex[2];
    SDL_FColor color;
} VULKAN_VertexPositionColor;

// Vulkan Buffer
typedef struct
{
    VkDeviceMemory deviceMemory;
    VkBuffer buffer;
    VkDeviceSize size;
    void *mappedBufferPtr;

} VULKAN_Buffer;

// Vulkan image
typedef struct
{
    bool allocatedImage;
    VkImage image;
    VkImageView imageView;
    VkDeviceMemory deviceMemory;
    VkImageLayout imageLayout;
    VkFormat format;
} VULKAN_Image;

// Per-texture data
typedef struct
{
    VULKAN_Image mainImage;
    VkRenderPass mainRenderpasses[VULKAN_RENDERPASS_COUNT];
    VkFramebuffer mainFramebuffer;
    VULKAN_Buffer stagingBuffer;
    SDL_Rect lockedRect;
    int width;
    int height;
    VULKAN_Shader shader;

    // Object passed to VkImageView and VkSampler for doing Ycbcr -> RGB conversion
    VkSamplerYcbcrConversion samplerYcbcrConversion;
    // Sampler created with samplerYcbcrConversion, passed to PSO as immutable sampler
    VkSampler samplerYcbcr;
    // Descriptor set layout with samplerYcbcr baked as immutable sampler
    VkDescriptorSetLayout descriptorSetLayoutYcbcr;
    // Pipeline layout with immutable sampler descriptor set layout
    VkPipelineLayout pipelineLayoutYcbcr;

} VULKAN_TextureData;

// Pipeline State Object data
typedef struct
{
    VULKAN_Shader shader;
    VULKAN_PixelShaderConstants shader_constants;
    SDL_BlendMode blendMode;
    VkPrimitiveTopology topology;
    VkFormat format;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipeline pipeline;
} VULKAN_PipelineState;

typedef struct
{
    VkBuffer vertexBuffer;
} VULKAN_DrawStateCache;

// Private renderer data
typedef struct
{
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    VkInstance instance;
    bool instance_external;
    VkSurfaceKHR surface;
    bool surface_external;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties physicalDeviceProperties;
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    VkPhysicalDeviceFeatures physicalDeviceFeatures;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkDevice device;
    bool device_external;
    uint32_t graphicsQueueFamilyIndex;
    uint32_t presentQueueFamilyIndex;
    VkSwapchainKHR swapchain;
    VkCommandPool commandPool;
    VkCommandBuffer *commandBuffers;
    uint32_t currentCommandBufferIndex;
    VkCommandBuffer currentCommandBuffer;
    VkFence *fences;
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    VkSurfaceFormatKHR *surfaceFormats;
    bool recreateSwapchain;
    int vsync;
    SDL_PropertiesID create_props;

    VkFramebuffer *framebuffers;
    VkRenderPass renderPasses[VULKAN_RENDERPASS_COUNT];
    VkRenderPass currentRenderPass;

    VkShaderModule vertexShaderModules[NUM_SHADERS];
    VkShaderModule fragmentShaderModules[NUM_SHADERS];
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;

    // Vertex buffer data
    VULKAN_Buffer vertexBuffers[SDL_VULKAN_NUM_VERTEX_BUFFERS];
    VULKAN_VertexShaderConstants vertexShaderConstantsData;

    // Data for staging/allocating textures
    VULKAN_Buffer **uploadBuffers;
    int *currentUploadBuffer;

    // Data for updating constants
    VULKAN_Buffer **constantBuffers;
    uint32_t *numConstantBuffers;
    uint32_t currentConstantBufferIndex;
    int32_t currentConstantBufferOffset;

    VkSampler samplers[RENDER_SAMPLER_COUNT];
    VkDescriptorPool **descriptorPools;
    uint32_t *numDescriptorPools;
    uint32_t currentDescriptorPoolIndex;
    uint32_t currentDescriptorSetIndex;

    int pipelineStateCount;
    VULKAN_PipelineState *pipelineStates;
    VULKAN_PipelineState *currentPipelineState;

    bool supportsEXTSwapchainColorspace;
    bool supportsKHRGetPhysicalDeviceProperties2;
    bool supportsKHRSamplerYCbCrConversion;
    uint32_t surfaceFormatsAllocatedCount;
    uint32_t surfaceFormatsCount;
    uint32_t swapchainDesiredImageCount;
    VkSurfaceFormatKHR surfaceFormat;
    VkExtent2D swapchainSize;
    VkSurfaceTransformFlagBitsKHR swapChainPreTransform;
    uint32_t swapchainImageCount;
    VkImage *swapchainImages;
    VkImageView *swapchainImageViews;
    VkImageLayout *swapchainImageLayouts;
    VkSemaphore *imageAvailableSemaphores;
    VkSemaphore *renderingFinishedSemaphores;
    VkSemaphore currentImageAvailableSemaphore;
    uint32_t currentSwapchainImageIndex;

    VkPipelineStageFlags *waitDestStageMasks;
    VkSemaphore *waitRenderSemaphores;
    uint32_t waitRenderSemaphoreCount;
    uint32_t waitRenderSemaphoreMax;
    VkSemaphore *signalRenderSemaphores;
    uint32_t signalRenderSemaphoreCount;
    uint32_t signalRenderSemaphoreMax;

    // Cached renderer properties
    VULKAN_TextureData *textureRenderTarget;
    bool cliprectDirty;
    bool currentCliprectEnabled;
    SDL_Rect currentCliprect;
    SDL_Rect currentViewport;
    int currentViewportRotation;
    bool viewportDirty;
    Float4X4 identity;
    VkComponentMapping identitySwizzle;
    int currentVertexBuffer;
    bool issueBatch;
} VULKAN_RenderData;

static SDL_PixelFormat VULKAN_VkFormatToSDLPixelFormat(VkFormat vkFormat)
{
    switch (vkFormat) {
    case VK_FORMAT_B8G8R8A8_UNORM:
        return SDL_PIXELFORMAT_ARGB8888;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return SDL_PIXELFORMAT_ABGR8888;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        return SDL_PIXELFORMAT_ABGR2101010;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return SDL_PIXELFORMAT_RGBA64_FLOAT;
    default:
        return SDL_PIXELFORMAT_UNKNOWN;
    }
}

static int VULKAN_VkFormatGetNumPlanes(VkFormat vkFormat)
{
    switch (vkFormat) {
    case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
        return 3;
    case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
    case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
        return 2;
    default:
        return 1;
    }
}

static VkDeviceSize VULKAN_GetBytesPerPixel(VkFormat vkFormat)
{
    switch (vkFormat) {
    case VK_FORMAT_R8_UNORM:
        return 1;
    case VK_FORMAT_R8G8_UNORM:
        return 2;
    case VK_FORMAT_R16G16_UNORM:
        return 4;
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 8;
    default:
        return 4;
    }
}

static VkFormat SDLPixelFormatToVkTextureFormat(Uint32 format, Uint32 output_colorspace)
{
    switch (format) {
    case SDL_PIXELFORMAT_RGBA64_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case SDL_PIXELFORMAT_ABGR2101010:
        return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case SDL_PIXELFORMAT_ARGB8888:
        if (output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return VK_FORMAT_B8G8R8A8_SRGB;
        }
        return VK_FORMAT_B8G8R8A8_UNORM;
    case SDL_PIXELFORMAT_ABGR8888:
        if (output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            return VK_FORMAT_R8G8B8A8_SRGB;
        }
        return VK_FORMAT_R8G8B8A8_UNORM;
    case SDL_PIXELFORMAT_YUY2:
        return VK_FORMAT_G8B8G8R8_422_UNORM;
    case SDL_PIXELFORMAT_UYVY:
        return VK_FORMAT_B8G8R8G8_422_UNORM;
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
        return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
        return  VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    case SDL_PIXELFORMAT_P010:
        return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
    default:
        return VK_FORMAT_UNDEFINED;
    }
}

static void VULKAN_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture);
static void VULKAN_DestroyBuffer(VULKAN_RenderData *rendererData, VULKAN_Buffer *vulkanBuffer);
static void VULKAN_DestroyImage(VULKAN_RenderData *rendererData, VULKAN_Image *vulkanImage);
static void VULKAN_ResetCommandList(VULKAN_RenderData *rendererData);
static bool VULKAN_FindMemoryTypeIndex(VULKAN_RenderData *rendererData, uint32_t typeBits, VkMemoryPropertyFlags requiredFlags, VkMemoryPropertyFlags desiredFlags, uint32_t *memoryTypeIndexOut);
static VkResult VULKAN_CreateWindowSizeDependentResources(SDL_Renderer *renderer);
static VkDescriptorPool VULKAN_AllocateDescriptorPool(VULKAN_RenderData *rendererData);
static VkResult VULKAN_CreateDescriptorSetAndPipelineLayout(VULKAN_RenderData *rendererData, VkSampler samplerYcbcr, VkDescriptorSetLayout *descriptorSetLayoutOut, VkPipelineLayout *pipelineLayoutOut);
static VkSurfaceTransformFlagBitsKHR VULKAN_GetRotationForCurrentRenderTarget(VULKAN_RenderData *rendererData);
static bool VULKAN_IsDisplayRotated90Degrees(VkSurfaceTransformFlagBitsKHR rotation);

static void VULKAN_DestroyAll(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData;
    if (renderer == NULL) {
        return;
    }
    rendererData = (VULKAN_RenderData *)renderer->internal;
    if (rendererData == NULL) {
        return;
    }

    // Release all textures
    for (SDL_Texture *texture = renderer->textures; texture; texture = texture->next) {
        VULKAN_DestroyTexture(renderer, texture);
    }

    if (rendererData->waitDestStageMasks) {
        SDL_free(rendererData->waitDestStageMasks);
        rendererData->waitDestStageMasks = NULL;
    }
    if (rendererData->waitRenderSemaphores) {
        SDL_free(rendererData->waitRenderSemaphores);
        rendererData->waitRenderSemaphores = NULL;
    }
    if (rendererData->signalRenderSemaphores) {
        SDL_free(rendererData->signalRenderSemaphores);
        rendererData->signalRenderSemaphores = NULL;
    }
    if (rendererData->surfaceFormats != NULL) {
        SDL_free(rendererData->surfaceFormats);
        rendererData->surfaceFormats = NULL;
        rendererData->surfaceFormatsAllocatedCount = 0;
    }
    if (rendererData->swapchainImages != NULL) {
        SDL_free(rendererData->swapchainImages);
        rendererData->swapchainImages = NULL;
    }
    if (rendererData->swapchain) {
        vkDestroySwapchainKHR(rendererData->device, rendererData->swapchain, NULL);
        rendererData->swapchain = VK_NULL_HANDLE;
    }
    if (rendererData->fences != NULL) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            if (rendererData->fences[i] != VK_NULL_HANDLE) {
                vkDestroyFence(rendererData->device, rendererData->fences[i], NULL);
                rendererData->fences[i] = VK_NULL_HANDLE;
            }
        }
        SDL_free(rendererData->fences);
        rendererData->fences = NULL;
    }
    if (rendererData->swapchainImageViews) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            if (rendererData->swapchainImageViews[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(rendererData->device, rendererData->swapchainImageViews[i], NULL);
            }
        }
        SDL_free(rendererData->swapchainImageViews);
        rendererData->swapchainImageViews = NULL;
    }
    if (rendererData->swapchainImageLayouts) {
        SDL_free(rendererData->swapchainImageLayouts);
        rendererData->swapchainImageLayouts = NULL;
    }
    if (rendererData->framebuffers) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            if (rendererData->framebuffers[i] != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(rendererData->device, rendererData->framebuffers[i], NULL);
            }
        }
        SDL_free(rendererData->framebuffers);
        rendererData->framebuffers = NULL;
    }
    for (uint32_t i = 0; i < SDL_arraysize(rendererData->samplers); i++) {
        if (rendererData->samplers[i] != VK_NULL_HANDLE) {
            vkDestroySampler(rendererData->device, rendererData->samplers[i], NULL);
            rendererData->samplers[i] = VK_NULL_HANDLE;
        }
    }
    for (uint32_t i = 0; i < SDL_arraysize(rendererData->vertexBuffers); i++ ) {
        VULKAN_DestroyBuffer(rendererData, &rendererData->vertexBuffers[i]);
    }
    SDL_memset(rendererData->vertexBuffers, 0, sizeof(rendererData->vertexBuffers));
    for (uint32_t i = 0; i < VULKAN_RENDERPASS_COUNT; i++) {
        if (rendererData->renderPasses[i] != VK_NULL_HANDLE) {
            vkDestroyRenderPass(rendererData->device, rendererData->renderPasses[i], NULL);
            rendererData->renderPasses[i] = VK_NULL_HANDLE;
        }
    }
    if (rendererData->imageAvailableSemaphores) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; ++i) {
            if (rendererData->imageAvailableSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(rendererData->device, rendererData->imageAvailableSemaphores[i], NULL);
            }
        }
        SDL_free(rendererData->imageAvailableSemaphores);
        rendererData->imageAvailableSemaphores = NULL;
    }
    if (rendererData->renderingFinishedSemaphores) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; ++i) {
            if (rendererData->renderingFinishedSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(rendererData->device, rendererData->renderingFinishedSemaphores[i], NULL);
            }
        }
        SDL_free(rendererData->renderingFinishedSemaphores);
        rendererData->renderingFinishedSemaphores = NULL;
    }
    if (rendererData->commandBuffers) {
        vkFreeCommandBuffers(rendererData->device, rendererData->commandPool, rendererData->swapchainImageCount, rendererData->commandBuffers);
        SDL_free(rendererData->commandBuffers);
        rendererData->commandBuffers = NULL;
        rendererData->currentCommandBuffer = VK_NULL_HANDLE;
        rendererData->currentCommandBufferIndex = 0;
    }
    if (rendererData->commandPool) {
        vkDestroyCommandPool(rendererData->device, rendererData->commandPool, NULL);
        rendererData->commandPool = VK_NULL_HANDLE;
    }
    if (rendererData->descriptorPools) {
        SDL_assert(rendererData->numDescriptorPools);
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            for (uint32_t j = 0; j < rendererData->numDescriptorPools[i]; j++) {
                if (rendererData->descriptorPools[i][j] != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(rendererData->device, rendererData->descriptorPools[i][j], NULL);
                }
            }
            SDL_free(rendererData->descriptorPools[i]);
        }
        SDL_free(rendererData->descriptorPools);
        rendererData->descriptorPools = NULL;
        SDL_free(rendererData->numDescriptorPools);
        rendererData->numDescriptorPools = NULL;
    }
    for (uint32_t i = 0; i < NUM_SHADERS; i++) {
        if (rendererData->vertexShaderModules[i] != VK_NULL_HANDLE) {
            vkDestroyShaderModule(rendererData->device, rendererData->vertexShaderModules[i], NULL);
            rendererData->vertexShaderModules[i] = VK_NULL_HANDLE;
        }
        if (rendererData->fragmentShaderModules[i] != VK_NULL_HANDLE) {
            vkDestroyShaderModule(rendererData->device, rendererData->fragmentShaderModules[i], NULL);
            rendererData->fragmentShaderModules[i] = VK_NULL_HANDLE;
        }
    }
    if (rendererData->descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(rendererData->device, rendererData->descriptorSetLayout, NULL);
        rendererData->descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (rendererData->pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(rendererData->device, rendererData->pipelineLayout, NULL);
        rendererData->pipelineLayout = VK_NULL_HANDLE;
    }
    for (int i = 0; i < rendererData->pipelineStateCount; i++) {
        vkDestroyPipeline(rendererData->device, rendererData->pipelineStates[i].pipeline, NULL);
    }
    SDL_free(rendererData->pipelineStates);
    rendererData->pipelineStates = NULL;
    rendererData->pipelineStateCount = 0;

    if (rendererData->currentUploadBuffer) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; ++i) {
            for (int j = 0; j < rendererData->currentUploadBuffer[i]; ++j) {
                VULKAN_DestroyBuffer(rendererData, &rendererData->uploadBuffers[i][j]);
            }
            SDL_free(rendererData->uploadBuffers[i]);
        }
        SDL_free(rendererData->uploadBuffers);
        rendererData->uploadBuffers = NULL;
        SDL_free(rendererData->currentUploadBuffer);
        rendererData->currentUploadBuffer = NULL;
    }

    if (rendererData->constantBuffers) {
        SDL_assert(rendererData->numConstantBuffers);
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; ++i) {
            for (uint32_t j = 0; j < rendererData->numConstantBuffers[i]; j++) {
                VULKAN_DestroyBuffer(rendererData, &rendererData->constantBuffers[i][j]);
            }
            SDL_free(rendererData->constantBuffers[i]);
        }
        SDL_free(rendererData->constantBuffers);
        rendererData->constantBuffers = NULL;
        SDL_free(rendererData->numConstantBuffers);
        rendererData->numConstantBuffers = NULL;
    }

    if (rendererData->device != VK_NULL_HANDLE && !rendererData->device_external) {
        vkDestroyDevice(rendererData->device, NULL);
        rendererData->device = VK_NULL_HANDLE;
    }
    if (rendererData->surface != VK_NULL_HANDLE && !rendererData->surface_external) {
        vkDestroySurfaceKHR(rendererData->instance, rendererData->surface, NULL);
        rendererData->surface = VK_NULL_HANDLE;
    }
    if (rendererData->instance != VK_NULL_HANDLE && !rendererData->instance_external) {
        vkDestroyInstance(rendererData->instance, NULL);
        rendererData->instance = VK_NULL_HANDLE;
    }
}

static void VULKAN_DestroyBuffer(VULKAN_RenderData *rendererData, VULKAN_Buffer *vulkanBuffer)
{
    if (vulkanBuffer->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(rendererData->device, vulkanBuffer->buffer, NULL);
        vulkanBuffer->buffer = VK_NULL_HANDLE;
    }
    if (vulkanBuffer->deviceMemory != VK_NULL_HANDLE) {
        vkFreeMemory(rendererData->device, vulkanBuffer->deviceMemory, NULL);
        vulkanBuffer->deviceMemory = VK_NULL_HANDLE;
    }
    SDL_memset(vulkanBuffer, 0, sizeof(VULKAN_Buffer));
}

static VkResult VULKAN_AllocateBuffer(VULKAN_RenderData *rendererData, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags requiredMemoryProps, VkMemoryPropertyFlags desiredMemoryProps, VULKAN_Buffer *bufferOut)
{
    VkResult result;
    VkBufferCreateInfo bufferCreateInfo = { 0 };
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    result = vkCreateBuffer(rendererData->device, &bufferCreateInfo, NULL, &bufferOut->buffer);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreateBuffer()", result);
        return result;
    }

    VkMemoryRequirements memoryRequirements = { 0 };
    vkGetBufferMemoryRequirements(rendererData->device, bufferOut->buffer, &memoryRequirements);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyBuffer(rendererData, bufferOut);
        SET_ERROR_CODE("vkGetBufferMemoryRequirements()", result);
        return result;
    }

    uint32_t memoryTypeIndex = 0;
    if (!VULKAN_FindMemoryTypeIndex(rendererData, memoryRequirements.memoryTypeBits, requiredMemoryProps, desiredMemoryProps, &memoryTypeIndex)) {
        VULKAN_DestroyBuffer(rendererData, bufferOut);
        return VK_ERROR_UNKNOWN;
    }

    VkMemoryAllocateInfo memoryAllocateInfo = { 0 };
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;
    result = vkAllocateMemory(rendererData->device, &memoryAllocateInfo, NULL, &bufferOut->deviceMemory);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyBuffer(rendererData, bufferOut);
        SET_ERROR_CODE("vkAllocateMemory()", result);
        return result;
    }
    result = vkBindBufferMemory(rendererData->device, bufferOut->buffer, bufferOut->deviceMemory, 0);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyBuffer(rendererData, bufferOut);
        SET_ERROR_CODE("vkBindBufferMemory()", result);
        return result;
    }

    result = vkMapMemory(rendererData->device, bufferOut->deviceMemory, 0, size, 0, &bufferOut->mappedBufferPtr);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyBuffer(rendererData, bufferOut);
        SET_ERROR_CODE("vkMapMemory()", result);
        return result;
    }
    bufferOut->size = size;
    return result;
}

static void VULKAN_DestroyImage(VULKAN_RenderData *rendererData, VULKAN_Image *vulkanImage)
{
    if (vulkanImage->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(rendererData->device, vulkanImage->imageView, NULL);
        vulkanImage->imageView = VK_NULL_HANDLE;
    }
    if (vulkanImage->image != VK_NULL_HANDLE) {
        if (vulkanImage->allocatedImage) {
            vkDestroyImage(rendererData->device, vulkanImage->image, NULL);
        }
        vulkanImage->image = VK_NULL_HANDLE;
    }

    if (vulkanImage->deviceMemory != VK_NULL_HANDLE) {
        if (vulkanImage->allocatedImage) {
            vkFreeMemory(rendererData->device, vulkanImage->deviceMemory, NULL);
        }
        vulkanImage->deviceMemory = VK_NULL_HANDLE;
    }
    SDL_memset(vulkanImage, 0, sizeof(VULKAN_Image));
}

static VkResult VULKAN_AllocateImage(VULKAN_RenderData *rendererData, SDL_PropertiesID create_props, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags imageUsage, VkComponentMapping swizzle, VkSamplerYcbcrConversionKHR samplerYcbcrConversion, VULKAN_Image *imageOut)
{
    VkResult result;
    VkSamplerYcbcrConversionInfoKHR samplerYcbcrConversionInfo = { 0 };

    SDL_memset(imageOut, 0, sizeof(VULKAN_Image));
    imageOut->format = format;
    imageOut->image = (VkImage)SDL_GetNumberProperty(create_props, SDL_PROP_TEXTURE_CREATE_VULKAN_TEXTURE_NUMBER, 0);

    if (imageOut->image == VK_NULL_HANDLE) {
        imageOut->allocatedImage = VK_TRUE;

        VkImageCreateInfo imageCreateInfo = { 0 };
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.flags = 0;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent.width = width;
        imageCreateInfo.extent.height = height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = imageUsage;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.queueFamilyIndexCount = 0;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        result = vkCreateImage(rendererData->device, &imageCreateInfo, NULL, &imageOut->image);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyImage(rendererData, imageOut);
            SET_ERROR_CODE("vkCreateImage()", result);
            return result;
        }

        VkMemoryRequirements memoryRequirements = { 0 };
        vkGetImageMemoryRequirements(rendererData->device, imageOut->image, &memoryRequirements);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyImage(rendererData, imageOut);
            SET_ERROR_CODE("vkGetImageMemoryRequirements()", result);
            return result;
        }

        uint32_t memoryTypeIndex = 0;
        if (!VULKAN_FindMemoryTypeIndex(rendererData, memoryRequirements.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memoryTypeIndex)) {
            VULKAN_DestroyImage(rendererData, imageOut);
            return VK_ERROR_UNKNOWN;
        }

        VkMemoryAllocateInfo memoryAllocateInfo = { 0 };
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;
        result = vkAllocateMemory(rendererData->device, &memoryAllocateInfo, NULL, &imageOut->deviceMemory);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyImage(rendererData, imageOut);
            SET_ERROR_CODE("vkAllocateMemory()", result);
            return result;
        }
        result = vkBindImageMemory(rendererData->device, imageOut->image, imageOut->deviceMemory, 0);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyImage(rendererData, imageOut);
            SET_ERROR_CODE("vkBindImageMemory()", result);
            return result;
        }
    } else {
        imageOut->imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkImageViewCreateInfo imageViewCreateInfo = { 0 };
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = imageOut->image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.components = swizzle;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    // If it's a YCbCr image, we need to pass the conversion info to the VkImageView (and the VkSampler)
    if (samplerYcbcrConversion != VK_NULL_HANDLE) {
        samplerYcbcrConversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR;
        samplerYcbcrConversionInfo.conversion = samplerYcbcrConversion;
        imageViewCreateInfo.pNext = &samplerYcbcrConversionInfo;
    }

    result = vkCreateImageView(rendererData->device, &imageViewCreateInfo, NULL, &imageOut->imageView);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyImage(rendererData, imageOut);
        SET_ERROR_CODE("vkCreateImageView()", result);
        return result;
    }

    return result;
}


static void VULKAN_RecordPipelineImageBarrier(VULKAN_RenderData *rendererData, VkAccessFlags sourceAccessMask, VkAccessFlags destAccessMask,
    VkPipelineStageFlags srcStageFlags, VkPipelineStageFlags dstStageFlags, VkImageLayout destLayout, VkImage image, VkImageLayout *imageLayout)
{
    // Stop any outstanding renderpass if open
    if (rendererData->currentRenderPass != VK_NULL_HANDLE) {
        vkCmdEndRenderPass(rendererData->currentCommandBuffer);
        rendererData->currentRenderPass = VK_NULL_HANDLE;
    }

    VkImageMemoryBarrier barrier = { 0 };
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = sourceAccessMask;
    barrier.dstAccessMask = destAccessMask;
    barrier.oldLayout = *imageLayout;
    barrier.newLayout = destLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(rendererData->currentCommandBuffer, srcStageFlags, dstStageFlags, 0, 0, NULL, 0, NULL, 1, &barrier);
    *imageLayout = destLayout;
}

static VkResult VULKAN_AcquireNextSwapchainImage(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = ( VULKAN_RenderData * )renderer->internal;

    VkResult result;

    rendererData->currentImageAvailableSemaphore = VK_NULL_HANDLE;
    result = vkAcquireNextImageKHR(rendererData->device, rendererData->swapchain, UINT64_MAX,
        rendererData->imageAvailableSemaphores[rendererData->currentCommandBufferIndex], VK_NULL_HANDLE, &rendererData->currentSwapchainImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
        result = VULKAN_CreateWindowSizeDependentResources(renderer);
        return result;
    } else if(result == VK_SUBOPTIMAL_KHR) {
        // Suboptimal, but we can contiue
    } else if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkAcquireNextImageKHR()", result);
        return result;
    }
    rendererData->currentImageAvailableSemaphore = rendererData->imageAvailableSemaphores[rendererData->currentCommandBufferIndex];
    return result;
}

static void VULKAN_BeginRenderPass(VULKAN_RenderData *rendererData, VkAttachmentLoadOp loadOp, VkClearColorValue *clearColor)
{
    int width = rendererData->swapchainSize.width;
    int height = rendererData->swapchainSize.height;
    if (rendererData->textureRenderTarget) {
        width = rendererData->textureRenderTarget->width;
        height = rendererData->textureRenderTarget->height;
    }

    switch (loadOp) {
    case VK_ATTACHMENT_LOAD_OP_CLEAR:
        rendererData->currentRenderPass = rendererData->textureRenderTarget ?
            rendererData->textureRenderTarget->mainRenderpasses[VULKAN_RENDERPASS_CLEAR] :
            rendererData->renderPasses[VULKAN_RENDERPASS_CLEAR];
        break;

    case VK_ATTACHMENT_LOAD_OP_LOAD:
    default:
        rendererData->currentRenderPass = rendererData->textureRenderTarget ?
            rendererData->textureRenderTarget->mainRenderpasses[VULKAN_RENDERPASS_LOAD] :
            rendererData->renderPasses[VULKAN_RENDERPASS_LOAD];
        break;
    }

    VkFramebuffer framebuffer = rendererData->textureRenderTarget ?
        rendererData->textureRenderTarget->mainFramebuffer :
        rendererData->framebuffers[rendererData->currentSwapchainImageIndex];

    VkRenderPassBeginInfo renderPassBeginInfo = { 0 };
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.pNext = NULL;
    renderPassBeginInfo.renderPass = rendererData->currentRenderPass;
    renderPassBeginInfo.framebuffer = framebuffer;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = (clearColor == NULL) ? 0 : 1;
    VkClearValue clearValue;
    if (clearColor != NULL) {
        clearValue.color = *clearColor;
        renderPassBeginInfo.pClearValues = &clearValue;
    }
    vkCmdBeginRenderPass(rendererData->currentCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
}

static void VULKAN_EnsureCommandBuffer(VULKAN_RenderData *rendererData)
{
    if (rendererData->currentCommandBuffer == VK_NULL_HANDLE) {
        rendererData->currentCommandBuffer = rendererData->commandBuffers[rendererData->currentCommandBufferIndex];
        VULKAN_ResetCommandList(rendererData);

        // Ensure the swapchain is in the correct layout
        if (rendererData->swapchainImageLayouts[rendererData->currentSwapchainImageIndex] == VK_IMAGE_LAYOUT_UNDEFINED) {
            VULKAN_RecordPipelineImageBarrier(rendererData,
                0,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                rendererData->swapchainImages[rendererData->currentSwapchainImageIndex],
                &rendererData->swapchainImageLayouts[rendererData->currentSwapchainImageIndex]);
        }
        else if (rendererData->swapchainImageLayouts[rendererData->currentCommandBufferIndex] != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            VULKAN_RecordPipelineImageBarrier(rendererData,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                rendererData->swapchainImages[rendererData->currentSwapchainImageIndex],
                &rendererData->swapchainImageLayouts[rendererData->currentSwapchainImageIndex]);
        }
    }
}

static bool VULKAN_ActivateCommandBuffer(SDL_Renderer *renderer, VkAttachmentLoadOp loadOp, VkClearColorValue *clearColor, VULKAN_DrawStateCache *stateCache)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;

    VULKAN_EnsureCommandBuffer(rendererData);

    if (rendererData->currentRenderPass == VK_NULL_HANDLE || loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        if (rendererData->currentRenderPass != VK_NULL_HANDLE) {
            vkCmdEndRenderPass(rendererData->currentCommandBuffer);
            rendererData->currentRenderPass = VK_NULL_HANDLE;
        }
        VULKAN_BeginRenderPass(rendererData, loadOp, clearColor);
    }

    // Bind cached VB now
    if (stateCache->vertexBuffer != VK_NULL_HANDLE) {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(rendererData->currentCommandBuffer, 0, 1, &stateCache->vertexBuffer, &offset);
    }

    return true;
}

static void VULKAN_WaitForGPU(VULKAN_RenderData *rendererData)
{
    vkQueueWaitIdle(rendererData->graphicsQueue);
}


static void VULKAN_ResetCommandList(VULKAN_RenderData *rendererData)
{
    vkResetCommandBuffer(rendererData->currentCommandBuffer, 0);
    for (uint32_t i = 0; i < rendererData->numDescriptorPools[rendererData->currentCommandBufferIndex]; i++) {
        vkResetDescriptorPool(rendererData->device, rendererData->descriptorPools[rendererData->currentCommandBufferIndex][i], 0);
    }

    VkCommandBufferBeginInfo beginInfo = { 0 };
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    vkBeginCommandBuffer(rendererData->currentCommandBuffer, &beginInfo);

    rendererData->currentPipelineState = NULL;
    rendererData->currentVertexBuffer = 0;
    rendererData->issueBatch = false;
    rendererData->cliprectDirty = true;
    rendererData->currentDescriptorSetIndex = 0;
    rendererData->currentDescriptorPoolIndex = 0;
    rendererData->currentConstantBufferOffset = -1;
    rendererData->currentConstantBufferIndex = 0;

    // Release any upload buffers that were inflight
    for (int i = 0; i < rendererData->currentUploadBuffer[rendererData->currentCommandBufferIndex]; ++i) {
        VULKAN_DestroyBuffer(rendererData, &rendererData->uploadBuffers[rendererData->currentCommandBufferIndex][i]);
    }
    rendererData->currentUploadBuffer[rendererData->currentCommandBufferIndex] = 0;
}

static VkResult VULKAN_IssueBatch(VULKAN_RenderData *rendererData)
{
    VkResult result;
    if (rendererData->currentCommandBuffer == VK_NULL_HANDLE) {
        return VK_SUCCESS;
    }

    if (rendererData->currentRenderPass) {
        vkCmdEndRenderPass(rendererData->currentCommandBuffer);
        rendererData->currentRenderPass = VK_NULL_HANDLE;
    }

    rendererData->currentPipelineState = VK_NULL_HANDLE;
    rendererData->viewportDirty = true;

    vkEndCommandBuffer(rendererData->currentCommandBuffer);

    VkSubmitInfo submitInfo = { 0 };
    VkPipelineStageFlags waitDestStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &rendererData->currentCommandBuffer;
    if (rendererData->waitRenderSemaphoreCount > 0) {
        Uint32 additionalSemaphoreCount = (rendererData->currentImageAvailableSemaphore != VK_NULL_HANDLE) ? 1 : 0;
        submitInfo.waitSemaphoreCount = rendererData->waitRenderSemaphoreCount + additionalSemaphoreCount;
        if (additionalSemaphoreCount > 0) {
            rendererData->waitRenderSemaphores[rendererData->waitRenderSemaphoreCount] = rendererData->currentImageAvailableSemaphore;
            rendererData->waitDestStageMasks[rendererData->waitRenderSemaphoreCount] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        submitInfo.pWaitSemaphores = rendererData->waitRenderSemaphores;
        submitInfo.pWaitDstStageMask = rendererData->waitDestStageMasks;
        rendererData->waitRenderSemaphoreCount = 0;
    } else if (rendererData->currentImageAvailableSemaphore != VK_NULL_HANDLE) {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &rendererData->currentImageAvailableSemaphore;
        submitInfo.pWaitDstStageMask = &waitDestStageMask;
    }

    result = vkQueueSubmit(rendererData->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    rendererData->currentImageAvailableSemaphore = VK_NULL_HANDLE;

    VULKAN_WaitForGPU(rendererData);

    VULKAN_ResetCommandList(rendererData);

    return result;
}

static void VULKAN_DestroyRenderer(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    if (rendererData) {
        if (rendererData->device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(rendererData->device);
            VULKAN_DestroyAll(renderer);
        }
        SDL_free(rendererData);
    }
}

static VkBlendFactor GetBlendFactor(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return VK_BLEND_FACTOR_ZERO;
    case SDL_BLENDFACTOR_ONE:
        return VK_BLEND_FACTOR_ONE;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return VK_BLEND_FACTOR_SRC_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return VK_BLEND_FACTOR_SRC_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case SDL_BLENDFACTOR_DST_COLOR:
        return VK_BLEND_FACTOR_DST_COLOR;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return VK_BLEND_FACTOR_DST_ALPHA;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    default:
        return VK_BLEND_FACTOR_MAX_ENUM;
    }
}

static VkBlendOp GetBlendOp(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return VK_BLEND_OP_ADD;
    case SDL_BLENDOPERATION_SUBTRACT:
        return VK_BLEND_OP_SUBTRACT;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return VK_BLEND_OP_REVERSE_SUBTRACT;
    case SDL_BLENDOPERATION_MINIMUM:
        return VK_BLEND_OP_MIN;
    case SDL_BLENDOPERATION_MAXIMUM:
        return VK_BLEND_OP_MAX;
    default:
        return VK_BLEND_OP_MAX_ENUM;
    }
}


static VULKAN_PipelineState *VULKAN_CreatePipelineState(SDL_Renderer *renderer,
    VULKAN_Shader shader, VkPipelineLayout pipelineLayout, VkDescriptorSetLayout descriptorSetLayout, SDL_BlendMode blendMode, VkPrimitiveTopology topology, VkFormat format)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_PipelineState *pipelineStates;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = VK_SUCCESS;
    VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = { 0 };
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo = { 0 };
    VkVertexInputAttributeDescription attributeDescriptions[3];
    VkVertexInputBindingDescription bindingDescriptions[1];
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo[2];
    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = { 0 };
    VkPipelineViewportStateCreateInfo viewportStateCreateInfo = { 0 };
    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo = { 0 };
    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo = { 0 };
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo = { 0 };
    VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo = { 0 };

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = { 0 };
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.flags = 0;
    pipelineCreateInfo.pStages = shaderStageCreateInfo;
    pipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    pipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    pipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    pipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
    pipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    pipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;

    // Shaders
    const char *name = "main";
    for (uint32_t i = 0; i < 2; i++) {
        SDL_memset(&shaderStageCreateInfo[i], 0, sizeof(shaderStageCreateInfo[i]));
        shaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo[i].module = (i == 0) ? rendererData->vertexShaderModules[shader] : rendererData->fragmentShaderModules[shader];
        shaderStageCreateInfo[i].stage = (i == 0) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStageCreateInfo[i].pName = name;
    }
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = &shaderStageCreateInfo[0];


    // Vertex input
    vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputCreateInfo.vertexAttributeDescriptionCount = 3;
    vertexInputCreateInfo.pVertexAttributeDescriptions = &attributeDescriptions[0];
    vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
    vertexInputCreateInfo.pVertexBindingDescriptions = &bindingDescriptions[0];

    attributeDescriptions[ 0 ].binding = 0;
    attributeDescriptions[ 0 ].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[ 0 ].location = 0;
    attributeDescriptions[ 0 ].offset = 0;
    attributeDescriptions[ 1 ].binding = 0;
    attributeDescriptions[ 1 ].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[ 1 ].location = 1;
    attributeDescriptions[ 1 ].offset = 8;
    attributeDescriptions[ 2 ].binding = 0;
    attributeDescriptions[ 2 ].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[ 2 ].location = 2;
    attributeDescriptions[ 2 ].offset = 16;

    bindingDescriptions[ 0 ].binding = 0;
    bindingDescriptions[ 0 ].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescriptions[ 0 ].stride = 32;

    // Input assembly
    inputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyStateCreateInfo.topology = topology;
    inputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

    viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCreateInfo.scissorCount = 1;
    viewportStateCreateInfo.viewportCount = 1;

    // Dynamic states
    dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    VkDynamicState dynamicStates[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    dynamicStateCreateInfo.dynamicStateCount = SDL_arraysize(dynamicStates);
    dynamicStateCreateInfo.pDynamicStates = dynamicStates;

    // Rasterization state
    rasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
    rasterizationStateCreateInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizationStateCreateInfo.cullMode = VK_CULL_MODE_NONE;
    rasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationStateCreateInfo.depthBiasEnable = VK_FALSE;
    rasterizationStateCreateInfo.depthBiasConstantFactor = 0.0f;
    rasterizationStateCreateInfo.depthBiasClamp = 0.0f;
    rasterizationStateCreateInfo.depthBiasSlopeFactor = 0.0f;
    rasterizationStateCreateInfo.lineWidth = 1.0f;

    // MSAA state
    multisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    VkSampleMask multiSampleMask = 0xFFFFFFFF;
    multisampleStateCreateInfo.pSampleMask = &multiSampleMask;
    multisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth Stencil
    depthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    // Color blend
    VkPipelineColorBlendAttachmentState colorBlendAttachment = { 0 };
    colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendStateCreateInfo.attachmentCount = 1;
    colorBlendStateCreateInfo.pAttachments = &colorBlendAttachment;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = GetBlendFactor(SDL_GetBlendModeSrcColorFactor(blendMode));
    colorBlendAttachment.srcAlphaBlendFactor = GetBlendFactor(SDL_GetBlendModeSrcAlphaFactor(blendMode));
    colorBlendAttachment.colorBlendOp = GetBlendOp(SDL_GetBlendModeColorOperation(blendMode));
    colorBlendAttachment.dstColorBlendFactor = GetBlendFactor(SDL_GetBlendModeDstColorFactor(blendMode));
    colorBlendAttachment.dstAlphaBlendFactor = GetBlendFactor(SDL_GetBlendModeDstAlphaFactor(blendMode));
    colorBlendAttachment.alphaBlendOp = GetBlendOp(SDL_GetBlendModeAlphaOperation(blendMode));
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    // Renderpass / layout
    pipelineCreateInfo.renderPass = rendererData->currentRenderPass;
    pipelineCreateInfo.subpass = 0;
    pipelineCreateInfo.layout = pipelineLayout;

    result = vkCreateGraphicsPipelines(rendererData->device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreateGraphicsPipelines()", result);
        return NULL;
    }

    pipelineStates = (VULKAN_PipelineState *)SDL_realloc(rendererData->pipelineStates, (rendererData->pipelineStateCount + 1) * sizeof(*pipelineStates));
    if (!pipelineStates) {
        return NULL;
    }
    pipelineStates[rendererData->pipelineStateCount].shader = shader;
    pipelineStates[rendererData->pipelineStateCount].blendMode = blendMode;
    pipelineStates[rendererData->pipelineStateCount].topology = topology;
    pipelineStates[rendererData->pipelineStateCount].format = format;
    pipelineStates[rendererData->pipelineStateCount].pipeline = pipeline;
    pipelineStates[rendererData->pipelineStateCount].descriptorSetLayout = descriptorSetLayout;
    pipelineStates[rendererData->pipelineStateCount].pipelineLayout = pipelineCreateInfo.layout;
    rendererData->pipelineStates = pipelineStates;
    ++rendererData->pipelineStateCount;

    return &pipelineStates[rendererData->pipelineStateCount - 1];
}

static bool VULKAN_FindMemoryTypeIndex(VULKAN_RenderData *rendererData, uint32_t typeBits, VkMemoryPropertyFlags requiredFlags, VkMemoryPropertyFlags desiredFlags, uint32_t *memoryTypeIndexOut)
{
    uint32_t memoryTypeIndex = 0;
    bool foundExactMatch = false;

    // Desired flags must be a superset of required flags.
    desiredFlags |= requiredFlags;

    for (memoryTypeIndex = 0; memoryTypeIndex < rendererData->physicalDeviceMemoryProperties.memoryTypeCount; memoryTypeIndex++) {
        if (typeBits & (1 << memoryTypeIndex)) {
            if (rendererData->physicalDeviceMemoryProperties.memoryTypes[memoryTypeIndex].propertyFlags == desiredFlags) {
                foundExactMatch = true;
                break;
            }
        }
    }
    if (!foundExactMatch) {
        for (memoryTypeIndex = 0; memoryTypeIndex < rendererData->physicalDeviceMemoryProperties.memoryTypeCount; memoryTypeIndex++) {
            if (typeBits & (1 << memoryTypeIndex)) {
                if ((rendererData->physicalDeviceMemoryProperties.memoryTypes[memoryTypeIndex].propertyFlags & requiredFlags) == requiredFlags) {
                    break;
                }
            }
        }
    }

    if (memoryTypeIndex >= rendererData->physicalDeviceMemoryProperties.memoryTypeCount) {
        SET_ERROR_MESSAGE("Unable to find memory type for allocation");
        return false;
    }
    *memoryTypeIndexOut = memoryTypeIndex;
    return true;
}

static VkResult VULKAN_CreateVertexBuffer(VULKAN_RenderData *rendererData, size_t vbidx, size_t size)
{
    VkResult result;

    VULKAN_DestroyBuffer(rendererData, &rendererData->vertexBuffers[vbidx]);

    result = VULKAN_AllocateBuffer(rendererData, size,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &rendererData->vertexBuffers[vbidx]);
    if (result != VK_SUCCESS) {
        return result;
    }
    return result;
}

static bool VULKAN_LoadGlobalFunctions(VULKAN_RenderData *rendererData)
{
#define VULKAN_DEVICE_FUNCTION(name)
#define VULKAN_GLOBAL_FUNCTION(name)                                                        \
    name = (PFN_##name)rendererData->vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);          \
    if (!name) {                                                                            \
        SET_ERROR_MESSAGE("vkGetInstanceProcAddr(VK_NULL_HANDLE, \"" #name "\") failed");   \
        return false;                                                                       \
    }
#define VULKAN_INSTANCE_FUNCTION(name)
#define VULKAN_OPTIONAL_INSTANCE_FUNCTION(name)
#define VULKAN_OPTIONAL_DEVICE_FUNCTION(name)
    VULKAN_FUNCTIONS()
#undef VULKAN_DEVICE_FUNCTION
#undef VULKAN_GLOBAL_FUNCTION
#undef VULKAN_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_DEVICE_FUNCTION

    return true;
}

static bool VULKAN_LoadInstanceFunctions(VULKAN_RenderData *rendererData)
{
#define VULKAN_DEVICE_FUNCTION(name)
#define VULKAN_GLOBAL_FUNCTION(name)
#define VULKAN_INSTANCE_FUNCTION(name)                                                      \
    name = (PFN_##name)rendererData->vkGetInstanceProcAddr(rendererData->instance, #name);  \
    if (!name) {                                                                            \
        SET_ERROR_MESSAGE("vkGetInstanceProcAddr(instance, \"" #name "\") failed");         \
        return false;                                                                       \
    }
#define VULKAN_OPTIONAL_INSTANCE_FUNCTION(name)                                             \
    name = (PFN_##name)rendererData->vkGetInstanceProcAddr(rendererData->instance, #name);
#define VULKAN_OPTIONAL_DEVICE_FUNCTION(name)

    VULKAN_FUNCTIONS()
#undef VULKAN_DEVICE_FUNCTION
#undef VULKAN_GLOBAL_FUNCTION
#undef VULKAN_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_DEVICE_FUNCTION

    return true;
}

static bool VULKAN_LoadDeviceFunctions(VULKAN_RenderData *rendererData)
{
#define VULKAN_DEVICE_FUNCTION(name)                                            \
    name = (PFN_##name)vkGetDeviceProcAddr(rendererData->device, #name);        \
    if (!name) {                                                                \
        SET_ERROR_MESSAGE("vkGetDeviceProcAddr(device, \"" #name "\") failed"); \
        return false;                                                           \
    }
#define VULKAN_GLOBAL_FUNCTION(name)
#define VULKAN_OPTIONAL_DEVICE_FUNCTION(name)                                \
    name = (PFN_##name)vkGetDeviceProcAddr(rendererData->device, #name);
#define VULKAN_INSTANCE_FUNCTION(name)
#define VULKAN_OPTIONAL_INSTANCE_FUNCTION(name)
    VULKAN_FUNCTIONS()
#undef VULKAN_DEVICE_FUNCTION
#undef VULKAN_GLOBAL_FUNCTION
#undef VULKAN_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_INSTANCE_FUNCTION
#undef VULKAN_OPTIONAL_DEVICE_FUNCTION
    return true;
}

static VkResult VULKAN_FindPhysicalDevice(VULKAN_RenderData *rendererData)
{
    uint32_t physicalDeviceCount = 0;
    VkPhysicalDevice *physicalDevices;
    VkQueueFamilyProperties *queueFamiliesProperties = NULL;
    uint32_t queueFamiliesPropertiesAllocatedSize = 0;
    VkExtensionProperties *deviceExtensions = NULL;
    uint32_t deviceExtensionsAllocatedSize = 0;
    uint32_t physicalDeviceIndex;
    VkResult result;

    result = vkEnumeratePhysicalDevices(rendererData->instance, &physicalDeviceCount, NULL);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkEnumeratePhysicalDevices()", result);
        return result;
    }
    if (physicalDeviceCount == 0) {
        SET_ERROR_MESSAGE("vkEnumeratePhysicalDevices(): no physical devices");
        return VK_ERROR_UNKNOWN;
    }
    physicalDevices = (VkPhysicalDevice *)SDL_malloc(sizeof(VkPhysicalDevice) * physicalDeviceCount);
    result = vkEnumeratePhysicalDevices(rendererData->instance, &physicalDeviceCount, physicalDevices);
    if (result != VK_SUCCESS) {
        SDL_free(physicalDevices);
        SET_ERROR_CODE("vkEnumeratePhysicalDevices()", result);
        return result;
    }
    rendererData->physicalDevice = NULL;
    for (physicalDeviceIndex = 0; physicalDeviceIndex < physicalDeviceCount; physicalDeviceIndex++) {
        uint32_t queueFamiliesCount = 0;
        uint32_t queueFamilyIndex;
        uint32_t deviceExtensionCount = 0;
        bool hasSwapchainExtension = false;
        uint32_t i;

        VkPhysicalDevice physicalDevice = physicalDevices[physicalDeviceIndex];
        vkGetPhysicalDeviceProperties(physicalDevice, &rendererData->physicalDeviceProperties);
        if (VK_VERSION_MAJOR(rendererData->physicalDeviceProperties.apiVersion) < 1) {
            continue;
        }
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &rendererData->physicalDeviceMemoryProperties);
        vkGetPhysicalDeviceFeatures(physicalDevice, &rendererData->physicalDeviceFeatures);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamiliesCount, NULL);
        if (queueFamiliesCount == 0) {
            continue;
        }
        if (queueFamiliesPropertiesAllocatedSize < queueFamiliesCount) {
            SDL_free(queueFamiliesProperties);
            queueFamiliesPropertiesAllocatedSize = queueFamiliesCount;
            queueFamiliesProperties = (VkQueueFamilyProperties *)SDL_malloc(sizeof(VkQueueFamilyProperties) * queueFamiliesPropertiesAllocatedSize);
            if (!queueFamiliesProperties) {
                SDL_free(physicalDevices);
                SDL_free(deviceExtensions);
                return VK_ERROR_UNKNOWN;
            }
        }
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamiliesCount, queueFamiliesProperties);
        rendererData->graphicsQueueFamilyIndex = queueFamiliesCount;
        rendererData->presentQueueFamilyIndex = queueFamiliesCount;
        for (queueFamilyIndex = 0; queueFamilyIndex < queueFamiliesCount; queueFamilyIndex++) {
            VkBool32 supported = 0;

            if (queueFamiliesProperties[queueFamilyIndex].queueCount == 0) {
                continue;
            }

            if (queueFamiliesProperties[queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                rendererData->graphicsQueueFamilyIndex = queueFamilyIndex;
            }

            result = vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, rendererData->surface, &supported);
            if (result != VK_SUCCESS) {
                SDL_free(physicalDevices);
                SDL_free(queueFamiliesProperties);
                SDL_free(deviceExtensions);
                SET_ERROR_CODE("vkGetPhysicalDeviceSurfaceSupportKHR()", result);
                return VK_ERROR_UNKNOWN;
            }
            if (supported) {
                rendererData->presentQueueFamilyIndex = queueFamilyIndex;
                if (queueFamiliesProperties[queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    break; // use this queue because it can present and do graphics
                }
            }
        }

        if (rendererData->graphicsQueueFamilyIndex == queueFamiliesCount) { // no good queues found
            continue;
        }
        if (rendererData->presentQueueFamilyIndex == queueFamiliesCount) { // no good queues found
            continue;
        }
        result = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionCount, NULL);
        if (result != VK_SUCCESS) {
            SDL_free(physicalDevices);
            SDL_free(queueFamiliesProperties);
            SDL_free(deviceExtensions);
            SET_ERROR_CODE("vkEnumerateDeviceExtensionProperties()", result);
            return VK_ERROR_UNKNOWN;
        }
        if (deviceExtensionCount == 0) {
            continue;
        }
        if (deviceExtensionsAllocatedSize < deviceExtensionCount) {
            SDL_free(deviceExtensions);
            deviceExtensionsAllocatedSize = deviceExtensionCount;
            deviceExtensions = (VkExtensionProperties *)SDL_malloc(sizeof(VkExtensionProperties) * deviceExtensionsAllocatedSize);
            if (!deviceExtensions) {
                SDL_free(physicalDevices);
                SDL_free(queueFamiliesProperties);
                return VK_ERROR_UNKNOWN;
            }
        }
        result = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionCount, deviceExtensions);
        if (result != VK_SUCCESS) {
            SDL_free(physicalDevices);
            SDL_free(queueFamiliesProperties);
            SDL_free(deviceExtensions);
            SET_ERROR_CODE("vkEnumerateDeviceExtensionProperties()", result);
            return result;
        }
        for (i = 0; i < deviceExtensionCount; i++) {
            if (SDL_strcmp(deviceExtensions[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                hasSwapchainExtension = true;
                break;
            }
        }
        if (!hasSwapchainExtension) {
            continue;
        }
        rendererData->physicalDevice = physicalDevice;
        break;
    }
    SDL_free(physicalDevices);
    SDL_free(queueFamiliesProperties);
    SDL_free(deviceExtensions);
    if (!rendererData->physicalDevice) {
        SET_ERROR_MESSAGE("No viable physical devices found");
        return VK_ERROR_UNKNOWN;
    }
    return VK_SUCCESS;
}

static VkResult VULKAN_GetSurfaceFormats(VULKAN_RenderData *rendererData)
{
    VkResult result = vkGetPhysicalDeviceSurfaceFormatsKHR(rendererData->physicalDevice,
                                                           rendererData->surface,
                                                           &rendererData->surfaceFormatsCount,
                                                           NULL);
    if (result != VK_SUCCESS) {
        rendererData->surfaceFormatsCount = 0;
        SET_ERROR_CODE("vkGetPhysicalDeviceSurfaceFormatsKHR()", result);
        return result;
    }
    if (rendererData->surfaceFormatsCount > rendererData->surfaceFormatsAllocatedCount) {
        rendererData->surfaceFormatsAllocatedCount = rendererData->surfaceFormatsCount;
        SDL_free(rendererData->surfaceFormats);
        rendererData->surfaceFormats = (VkSurfaceFormatKHR *)SDL_malloc(sizeof(VkSurfaceFormatKHR) * rendererData->surfaceFormatsAllocatedCount);
    }
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(rendererData->physicalDevice,
                                                  rendererData->surface,
                                                  &rendererData->surfaceFormatsCount,
                                                  rendererData->surfaceFormats);
    if (result != VK_SUCCESS) {
        rendererData->surfaceFormatsCount = 0;
        SET_ERROR_CODE("vkGetPhysicalDeviceSurfaceFormatsKHR()", result);
        return result;
    }

    return VK_SUCCESS;
}

static VkSemaphore VULKAN_CreateSemaphore(VULKAN_RenderData *rendererData)
{
    VkResult result;
    VkSemaphore semaphore = VK_NULL_HANDLE;

    VkSemaphoreCreateInfo semaphoreCreateInfo = { 0 };
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    result = vkCreateSemaphore(rendererData->device, &semaphoreCreateInfo, NULL, &semaphore);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreateSemaphore()", result);
        return VK_NULL_HANDLE;
    }
    return semaphore;
}

static bool VULKAN_DeviceExtensionsFound(VULKAN_RenderData *rendererData, int extensionsToCheck, const char* const* extNames)
{
    uint32_t extensionCount;
    bool foundExtensions = true;
    VkResult result = vkEnumerateDeviceExtensionProperties(rendererData->physicalDevice, NULL, &extensionCount, NULL);
    if (result != VK_SUCCESS ) {
        SET_ERROR_CODE("vkEnumerateDeviceExtensionProperties()", result);
        return false;
    }
    if (extensionCount > 0 ) {
        VkExtensionProperties *extensionProperties = (VkExtensionProperties *)SDL_calloc(extensionCount, sizeof(VkExtensionProperties));
        result = vkEnumerateDeviceExtensionProperties(rendererData->physicalDevice, NULL, &extensionCount, extensionProperties);
        if (result != VK_SUCCESS ) {
            SET_ERROR_CODE("vkEnumerateDeviceExtensionProperties()", result);
            SDL_free(extensionProperties);
            return false;
        }
        for (int ext = 0; ext < extensionsToCheck && foundExtensions; ext++) {
            bool foundExtension = false;
            for (uint32_t i = 0; i < extensionCount; i++) {
                if (SDL_strcmp(extensionProperties[i].extensionName, extNames[ext]) == 0) {
                    foundExtension = true;
                    break;
                }
            }
            foundExtensions &= foundExtension;
        }

        SDL_free(extensionProperties);
    }

    return foundExtensions;
}

static bool VULKAN_InstanceExtensionFound(VULKAN_RenderData *rendererData, const char *extName)
{
    uint32_t extensionCount;
    VkResult result = vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
    if (result != VK_SUCCESS ) {
        SET_ERROR_CODE("vkEnumerateInstanceExtensionProperties()", result);
        return false;
    }
    if (extensionCount > 0 ) {
        VkExtensionProperties *extensionProperties = (VkExtensionProperties *)SDL_calloc(extensionCount, sizeof(VkExtensionProperties));
        result = vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties);
        if (result != VK_SUCCESS ) {
            SET_ERROR_CODE("vkEnumerateInstanceExtensionProperties()", result);
            SDL_free(extensionProperties);
            return false;
        }
        for (uint32_t i = 0; i< extensionCount; i++) {
            if (SDL_strcmp(extensionProperties[i].extensionName, extName) == 0) {
                SDL_free(extensionProperties);
                return true;
            }
        }
        SDL_free(extensionProperties);
    }

    return false;
}

static bool VULKAN_ValidationLayersFound(void)
{
    uint32_t instanceLayerCount = 0;
    uint32_t i;
    bool foundValidation = false;

    vkEnumerateInstanceLayerProperties(&instanceLayerCount, NULL);
    if (instanceLayerCount > 0) {
        VkLayerProperties *instanceLayers = (VkLayerProperties *)SDL_calloc(instanceLayerCount, sizeof(VkLayerProperties));
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers);
        for (i = 0; i < instanceLayerCount; i++) {
            if (!SDL_strcmp(SDL_VULKAN_VALIDATION_LAYER_NAME, instanceLayers[i].layerName)) {
                foundValidation = true;
                break;
            }
        }
        SDL_free(instanceLayers);
    }

    return foundValidation;
}

// Create resources that depend on the device.
static VkResult VULKAN_CreateDeviceResources(SDL_Renderer *renderer, SDL_PropertiesID create_props)
{
    static const char *const deviceExtensionNames[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        /* VK_KHR_sampler_ycbcr_conversion + dependent extensions.
           Note VULKAN_DeviceExtensionsFound() call below, if these get moved in this
           array, update that check too.
       */
        VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
        VK_KHR_MAINTENANCE1_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
    };
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    SDL_VideoDevice *device = SDL_GetVideoDevice();
    VkResult result = VK_SUCCESS;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = NULL;
    bool createDebug = SDL_GetHintBoolean(SDL_HINT_RENDER_VULKAN_DEBUG, false);
    const char *validationLayerName[] = { SDL_VULKAN_VALIDATION_LAYER_NAME };

    if (!SDL_Vulkan_LoadLibrary(NULL)) {
        SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "SDL_Vulkan_LoadLibrary failed" );
        return VK_ERROR_UNKNOWN;
    }
    vkGetInstanceProcAddr = device ? (PFN_vkGetInstanceProcAddr)device->vulkan_config.vkGetInstanceProcAddr : NULL;
    if(!vkGetInstanceProcAddr) {
        SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "vkGetInstanceProcAddr is NULL" );
        return VK_ERROR_UNKNOWN;
    }

    // Load global Vulkan functions
    rendererData->vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    if (!VULKAN_LoadGlobalFunctions(rendererData)) {
        return VK_ERROR_UNKNOWN;
    }

    // Check for colorspace extension
    rendererData->supportsEXTSwapchainColorspace = VK_FALSE;
    if (renderer->output_colorspace == SDL_COLORSPACE_SRGB_LINEAR ||
        renderer->output_colorspace == SDL_COLORSPACE_HDR10) {
        rendererData->supportsEXTSwapchainColorspace = VULKAN_InstanceExtensionFound(rendererData, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
        if (!rendererData->supportsEXTSwapchainColorspace) {
            SDL_SetError("Using HDR output but %s not supported", VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
            return VK_ERROR_UNKNOWN;
        }
    }

    // Check for VK_KHR_get_physical_device_properties2
    rendererData->supportsKHRGetPhysicalDeviceProperties2 = VULKAN_InstanceExtensionFound(rendererData, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    // Create VkInstance
    rendererData->instance = (VkInstance)SDL_GetPointerProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_INSTANCE_POINTER, NULL);
    if (rendererData->instance) {
        rendererData->instance_external = true;
    } else {
        VkInstanceCreateInfo instanceCreateInfo = { 0 };
        VkApplicationInfo appInfo = { 0 };
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_0;
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pApplicationInfo = &appInfo;
        char const *const *instanceExtensions = SDL_Vulkan_GetInstanceExtensions(&instanceCreateInfo.enabledExtensionCount);

        const char **instanceExtensionsCopy = (const char **)SDL_calloc(instanceCreateInfo.enabledExtensionCount + 2, sizeof(const char *));
        for (uint32_t i = 0; i < instanceCreateInfo.enabledExtensionCount; i++) {
            instanceExtensionsCopy[i] = instanceExtensions[i];
        }
        if (rendererData->supportsEXTSwapchainColorspace) {
            instanceExtensionsCopy[instanceCreateInfo.enabledExtensionCount] = VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME;
            instanceCreateInfo.enabledExtensionCount++;
        }
        if (rendererData->supportsKHRGetPhysicalDeviceProperties2) {
            instanceExtensionsCopy[instanceCreateInfo.enabledExtensionCount] = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
            instanceCreateInfo.enabledExtensionCount++;
        }
        instanceCreateInfo.ppEnabledExtensionNames = (const char *const *)instanceExtensionsCopy;
        if (createDebug && VULKAN_ValidationLayersFound()) {
            instanceCreateInfo.ppEnabledLayerNames = validationLayerName;
            instanceCreateInfo.enabledLayerCount = 1;
        }
        result = vkCreateInstance(&instanceCreateInfo, NULL, &rendererData->instance);
        SDL_free((void *)instanceExtensionsCopy);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkCreateInstance()", result);
            return result;
        }
    }

    // Load instance Vulkan functions
    if (!VULKAN_LoadInstanceFunctions(rendererData)) {
        VULKAN_DestroyAll(renderer);
        return VK_ERROR_UNKNOWN;
    }

    // Create Vulkan surface
    rendererData->surface = (VkSurfaceKHR)SDL_GetNumberProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_SURFACE_NUMBER, 0);
    if (rendererData->surface) {
        rendererData->surface_external = true;
    } else {
        if (!device->Vulkan_CreateSurface || !device->Vulkan_CreateSurface(device, renderer->window, rendererData->instance, NULL, &rendererData->surface)) {
            VULKAN_DestroyAll(renderer);
            SET_ERROR_MESSAGE("Vulkan_CreateSurface() failed");
            return VK_ERROR_UNKNOWN;
        }
    }

    // Choose Vulkan physical device
    rendererData->physicalDevice = (VkPhysicalDevice)SDL_GetPointerProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_PHYSICAL_DEVICE_POINTER, NULL);
    if (rendererData->physicalDevice) {
        vkGetPhysicalDeviceMemoryProperties(rendererData->physicalDevice, &rendererData->physicalDeviceMemoryProperties);
        vkGetPhysicalDeviceFeatures(rendererData->physicalDevice, &rendererData->physicalDeviceFeatures);
    } else {
        if (VULKAN_FindPhysicalDevice(rendererData) != VK_SUCCESS) {
            VULKAN_DestroyAll(renderer);
            return VK_ERROR_UNKNOWN;
        }
    }

    if (SDL_HasProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_GRAPHICS_QUEUE_FAMILY_INDEX_NUMBER)) {
        rendererData->graphicsQueueFamilyIndex = (uint32_t)SDL_GetNumberProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_GRAPHICS_QUEUE_FAMILY_INDEX_NUMBER, 0);
    }
    if (SDL_HasProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_PRESENT_QUEUE_FAMILY_INDEX_NUMBER)) {
        rendererData->presentQueueFamilyIndex = (uint32_t)SDL_GetNumberProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_PRESENT_QUEUE_FAMILY_INDEX_NUMBER, 0);
    }

    if (rendererData->supportsKHRGetPhysicalDeviceProperties2 &&
        VULKAN_DeviceExtensionsFound(rendererData, 4, &deviceExtensionNames[1])) {
        rendererData->supportsKHRSamplerYCbCrConversion = true;
    }

    // Create Vulkan device
    rendererData->device = (VkDevice)SDL_GetPointerProperty(create_props, SDL_PROP_RENDERER_CREATE_VULKAN_DEVICE_POINTER, NULL);
    if (rendererData->device) {
        rendererData->device_external = true;
    } else {
        VkPhysicalDeviceSamplerYcbcrConversionFeatures deviceSamplerYcbcrConversionFeatures = { 0 };
        VkDeviceQueueCreateInfo deviceQueueCreateInfo[2] = { { 0 }, { 0 } };
        static const float queuePriority[] = { 1.0f };

        VkDeviceCreateInfo deviceCreateInfo = { 0 };
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 0;
        deviceCreateInfo.pQueueCreateInfos = deviceQueueCreateInfo;
        deviceCreateInfo.pEnabledFeatures = NULL;
        deviceCreateInfo.enabledExtensionCount = (rendererData->supportsKHRSamplerYCbCrConversion) ? 5 : 1;
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionNames;

        deviceQueueCreateInfo[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        deviceQueueCreateInfo[0].queueFamilyIndex = rendererData->graphicsQueueFamilyIndex;
        deviceQueueCreateInfo[0].queueCount = 1;
        deviceQueueCreateInfo[0].pQueuePriorities = queuePriority;
        ++deviceCreateInfo.queueCreateInfoCount;

        if (rendererData->presentQueueFamilyIndex != rendererData->graphicsQueueFamilyIndex) {
            deviceQueueCreateInfo[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            deviceQueueCreateInfo[1].queueFamilyIndex = rendererData->presentQueueFamilyIndex;
            deviceQueueCreateInfo[1].queueCount = 1;
            deviceQueueCreateInfo[1].pQueuePriorities = queuePriority;
            ++deviceCreateInfo.queueCreateInfoCount;
        }

        if (rendererData->supportsKHRSamplerYCbCrConversion) {
            deviceSamplerYcbcrConversionFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
            deviceSamplerYcbcrConversionFeatures.samplerYcbcrConversion = VK_TRUE;
            deviceSamplerYcbcrConversionFeatures.pNext = (void *)deviceCreateInfo.pNext;
            deviceCreateInfo.pNext = &deviceSamplerYcbcrConversionFeatures;
        }

        result = vkCreateDevice(rendererData->physicalDevice, &deviceCreateInfo, NULL, &rendererData->device);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkCreateDevice()", result);
            VULKAN_DestroyAll(renderer);
            return result;
        }
    }

    if (!VULKAN_LoadDeviceFunctions(rendererData)) {
        VULKAN_DestroyAll(renderer);
        return VK_ERROR_UNKNOWN;
    }

    // Get graphics/present queues
    vkGetDeviceQueue(rendererData->device, rendererData->graphicsQueueFamilyIndex, 0, &rendererData->graphicsQueue);
    if (rendererData->graphicsQueueFamilyIndex != rendererData->presentQueueFamilyIndex) {
        vkGetDeviceQueue(rendererData->device, rendererData->presentQueueFamilyIndex, 0, &rendererData->presentQueue);
    } else {
        rendererData->presentQueue = rendererData->graphicsQueue;
    }

    // Create command pool/command buffers
    VkCommandPoolCreateInfo commandPoolCreateInfo = { 0 };
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = rendererData->graphicsQueueFamilyIndex;
    result = vkCreateCommandPool(rendererData->device, &commandPoolCreateInfo, NULL, &rendererData->commandPool);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyAll(renderer);
        SET_ERROR_CODE("vkCreateCommandPool()", result);
        return result;
    }

    if (VULKAN_GetSurfaceFormats(rendererData) != VK_SUCCESS) {
        VULKAN_DestroyAll(renderer);
        return result;
    }

    // Create shaders / layouts
    for (uint32_t i = 0; i < NUM_SHADERS; i++) {
        VULKAN_Shader shader = (VULKAN_Shader)i;
        VkShaderModuleCreateInfo shaderModuleCreateInfo = { 0 };
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        VULKAN_GetVertexShader(shader, &shaderModuleCreateInfo.pCode, &shaderModuleCreateInfo.codeSize);
        result = vkCreateShaderModule(rendererData->device, &shaderModuleCreateInfo, NULL, &rendererData->vertexShaderModules[i]);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyAll(renderer);
            SET_ERROR_CODE("vkCreateShaderModule()", result);
            return result;
        }
        VULKAN_GetPixelShader(shader, &shaderModuleCreateInfo.pCode, &shaderModuleCreateInfo.codeSize);
        result = vkCreateShaderModule(rendererData->device, &shaderModuleCreateInfo, NULL, &rendererData->fragmentShaderModules[i]);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyAll(renderer);
            SET_ERROR_CODE("vkCreateShaderModule()", result);
            return result;
        }
    }

    // Descriptor set layout / pipeline layout
    result = VULKAN_CreateDescriptorSetAndPipelineLayout(rendererData, VK_NULL_HANDLE, &rendererData->descriptorSetLayout, &rendererData->pipelineLayout);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyAll(renderer);
        return result;
    }

    // Create default vertex buffers
    for (uint32_t i = 0; i < SDL_VULKAN_NUM_VERTEX_BUFFERS; ++i) {
        VULKAN_CreateVertexBuffer(rendererData, i, SDL_VULKAN_VERTEX_BUFFER_DEFAULT_SIZE);
    }

    SDL_PropertiesID props = SDL_GetRendererProperties(renderer);
    SDL_SetPointerProperty(props, SDL_PROP_RENDERER_VULKAN_INSTANCE_POINTER, rendererData->instance);
    SDL_SetNumberProperty(props, SDL_PROP_RENDERER_VULKAN_SURFACE_NUMBER, (Sint64)rendererData->surface);
    SDL_SetPointerProperty(props, SDL_PROP_RENDERER_VULKAN_PHYSICAL_DEVICE_POINTER, rendererData->physicalDevice);
    SDL_SetPointerProperty(props, SDL_PROP_RENDERER_VULKAN_DEVICE_POINTER, rendererData->device);
    SDL_SetNumberProperty(props, SDL_PROP_RENDERER_VULKAN_GRAPHICS_QUEUE_FAMILY_INDEX_NUMBER, rendererData->graphicsQueueFamilyIndex);
    SDL_SetNumberProperty(props, SDL_PROP_RENDERER_VULKAN_PRESENT_QUEUE_FAMILY_INDEX_NUMBER, rendererData->presentQueueFamilyIndex);

    return VK_SUCCESS;
}

static VkResult VULKAN_CreateFramebuffersAndRenderPasses(SDL_Renderer *renderer, int w, int h,
    VkFormat format, int imageViewCount, VkImageView *imageViews, VkFramebuffer *framebuffers, VkRenderPass renderPasses[VULKAN_RENDERPASS_COUNT])
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *) renderer->internal;
    VkResult result;

    VkAttachmentDescription attachmentDescription = { 0 };
    attachmentDescription.format = format;
    attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescription.flags = 0;

    VkAttachmentReference colorAttachmentReference = { 0 };
    colorAttachmentReference.attachment = 0;
    colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription = { 0 };
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.flags = 0;
    subpassDescription.inputAttachmentCount = 0;
    subpassDescription.pInputAttachments = NULL;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorAttachmentReference;
    subpassDescription.pResolveAttachments = NULL;
    subpassDescription.pDepthStencilAttachment = NULL;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments = NULL;

    VkSubpassDependency subPassDependency = { 0 };
    subPassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    subPassDependency.dstSubpass = 0;
    subPassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subPassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subPassDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    subPassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    subPassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassCreateInfo = { 0 };
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.flags = 0;
    renderPassCreateInfo.attachmentCount = 1;
    renderPassCreateInfo.pAttachments = &attachmentDescription;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDescription;
    renderPassCreateInfo.dependencyCount = 1;
    renderPassCreateInfo.pDependencies = &subPassDependency;

    result = vkCreateRenderPass(rendererData->device, &renderPassCreateInfo, NULL, &renderPasses[VULKAN_RENDERPASS_LOAD]);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreateRenderPass()", result);
        return result;
    }

    attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    result = vkCreateRenderPass(rendererData->device, &renderPassCreateInfo, NULL, &renderPasses[VULKAN_RENDERPASS_CLEAR]);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreateRenderPass()", result);
        return result;
    }

    VkFramebufferCreateInfo framebufferCreateInfo = { 0 };
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.pNext = NULL;
    framebufferCreateInfo.renderPass = rendererData->renderPasses[VULKAN_RENDERPASS_LOAD];
    framebufferCreateInfo.attachmentCount = 1;
    framebufferCreateInfo.width = w;
    framebufferCreateInfo.height = h;
    framebufferCreateInfo.layers = 1;

    for (int i = 0; i < imageViewCount; i++) {
        framebufferCreateInfo.pAttachments = &imageViews[i];
        result = vkCreateFramebuffer(rendererData->device, &framebufferCreateInfo, NULL, &framebuffers[i]);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkCreateFramebuffer()", result);
            return result;
        }
    }

    return result;
}

static VkResult VULKAN_CreateSwapChain(SDL_Renderer *renderer, int w, int h)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(rendererData->physicalDevice, rendererData->surface, &rendererData->surfaceCapabilities);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR()", result);
        return result;
    }

    // clean up previous swapchain resources
    if (rendererData->swapchainImageViews) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            vkDestroyImageView(rendererData->device, rendererData->swapchainImageViews[i], NULL);
        }
        SDL_free(rendererData->swapchainImageViews);
        rendererData->swapchainImageViews = NULL;
    }
    if (rendererData->fences) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            if (rendererData->fences[i] != VK_NULL_HANDLE) {
                vkDestroyFence(rendererData->device, rendererData->fences[i], NULL);
            }
        }
        SDL_free(rendererData->fences);
        rendererData->fences = NULL;
    }
    if (rendererData->commandBuffers) {
        vkResetCommandPool(rendererData->device, rendererData->commandPool, 0);
        SDL_free(rendererData->commandBuffers);
        rendererData->commandBuffers = NULL;
        rendererData->currentCommandBuffer = VK_NULL_HANDLE;
        rendererData->currentCommandBufferIndex = 0;
    }
    if (rendererData->framebuffers) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            if (rendererData->framebuffers[i] != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(rendererData->device, rendererData->framebuffers[i], NULL);
            }
        }
        SDL_free(rendererData->framebuffers);
        rendererData->framebuffers = NULL;
    }
    if (rendererData->descriptorPools) {
        SDL_assert(rendererData->numDescriptorPools);
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            for (uint32_t j = 0; j < rendererData->numDescriptorPools[i]; j++) {
                if (rendererData->descriptorPools[i][j] != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(rendererData->device, rendererData->descriptorPools[i][j], NULL);
                }
            }
            SDL_free(rendererData->descriptorPools[i]);
        }
        SDL_free(rendererData->descriptorPools);
        rendererData->descriptorPools = NULL;
        SDL_free(rendererData->numDescriptorPools);
        rendererData->numDescriptorPools = NULL;
    }
    if (rendererData->imageAvailableSemaphores) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; ++i) {
            if (rendererData->imageAvailableSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(rendererData->device, rendererData->imageAvailableSemaphores[i], NULL);
            }
        }
        SDL_free(rendererData->imageAvailableSemaphores);
        rendererData->imageAvailableSemaphores = NULL;
    }
    if (rendererData->renderingFinishedSemaphores) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; ++i) {
            if (rendererData->renderingFinishedSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(rendererData->device, rendererData->renderingFinishedSemaphores[i], NULL);
            }
        }
        SDL_free(rendererData->renderingFinishedSemaphores);
        rendererData->renderingFinishedSemaphores = NULL;
    }
    if (rendererData->uploadBuffers) {
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            for (uint32_t j = 0; j < SDL_VULKAN_NUM_UPLOAD_BUFFERS; j++) {
                VULKAN_DestroyBuffer(rendererData, &rendererData->uploadBuffers[i][j]);
            }
            SDL_free(rendererData->uploadBuffers[i]);
        }
        SDL_free(rendererData->uploadBuffers);
        rendererData->uploadBuffers = NULL;
    }
    if (rendererData->constantBuffers) {
        SDL_assert(rendererData->numConstantBuffers);
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; ++i) {
            for (uint32_t j = 0; j < rendererData->numConstantBuffers[i]; j++) {
                VULKAN_DestroyBuffer(rendererData, &rendererData->constantBuffers[i][j]);
            }
            SDL_free(rendererData->constantBuffers[i]);
        }
        SDL_free(rendererData->constantBuffers);
        rendererData->constantBuffers = NULL;
        SDL_free(rendererData->numConstantBuffers);
        rendererData->numConstantBuffers = NULL;
    }

    // pick an image count
    rendererData->swapchainDesiredImageCount = rendererData->surfaceCapabilities.minImageCount + SDL_VULKAN_FRAME_QUEUE_DEPTH;
    if ((rendererData->swapchainDesiredImageCount > rendererData->surfaceCapabilities.maxImageCount) &&
        (rendererData->surfaceCapabilities.maxImageCount > 0)) {
        rendererData->swapchainDesiredImageCount = rendererData->surfaceCapabilities.maxImageCount;
    }

    VkFormat desiredFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR desiredColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    if (renderer->output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
        desiredFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        desiredColorSpace = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
    }
    else if (renderer->output_colorspace == SDL_COLORSPACE_HDR10) {
        desiredFormat = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        desiredColorSpace = VK_COLOR_SPACE_HDR10_ST2084_EXT;
    }

    if ((rendererData->surfaceFormatsCount == 1) &&
        (rendererData->surfaceFormats[0].format == VK_FORMAT_UNDEFINED)) {
        // aren't any preferred formats, so we pick
        rendererData->surfaceFormat.colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
        rendererData->surfaceFormat.format = desiredFormat;
    } else {
        rendererData->surfaceFormat = rendererData->surfaceFormats[0];
        rendererData->surfaceFormat.colorSpace = rendererData->surfaceFormats[0].colorSpace;
        for (uint32_t i = 0; i < rendererData->surfaceFormatsCount; i++) {
            if (rendererData->surfaceFormats[i].format == desiredFormat &&
                rendererData->surfaceFormats[i].colorSpace == desiredColorSpace) {
                rendererData->surfaceFormat.colorSpace = rendererData->surfaceFormats[i].colorSpace;
                rendererData->surfaceFormat = rendererData->surfaceFormats[i];
                break;
            }
        }
    }

    rendererData->swapchainSize.width = SDL_clamp((uint32_t)w,
                                          rendererData->surfaceCapabilities.minImageExtent.width,
                                          rendererData->surfaceCapabilities.maxImageExtent.width);

    rendererData->swapchainSize.height = SDL_clamp((uint32_t)h,
                                           rendererData->surfaceCapabilities.minImageExtent.height,
                                           rendererData->surfaceCapabilities.maxImageExtent.height);

    // Handle rotation
    rendererData->swapChainPreTransform = rendererData->surfaceCapabilities.currentTransform;
    if (rendererData->swapChainPreTransform == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
        rendererData->swapChainPreTransform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) {
        uint32_t tempWidth = rendererData->swapchainSize.width;
        rendererData->swapchainSize.width = rendererData->swapchainSize.height;
        rendererData->swapchainSize.height = tempWidth;
    }

    if (rendererData->swapchainSize.width == 0 && rendererData->swapchainSize.height == 0) {
        // Don't recreate the swapchain if size is (0,0), just fail and continue attempting creation
        return VK_ERROR_OUT_OF_DATE_KHR;
    }

    // Choose a present mode. If vsync is requested, then use VK_PRESENT_MODE_FIFO_KHR which is guaranteed to be supported
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (rendererData->vsync <= 0) {
        uint32_t presentModeCount = 0;
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(rendererData->physicalDevice, rendererData->surface, &presentModeCount, NULL);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkGetPhysicalDeviceSurfacePresentModesKHR()", result);
            return result;
        }
        if (presentModeCount > 0) {
            VkPresentModeKHR *presentModes = (VkPresentModeKHR *)SDL_calloc(presentModeCount, sizeof(VkPresentModeKHR));
            result = vkGetPhysicalDeviceSurfacePresentModesKHR(rendererData->physicalDevice, rendererData->surface, &presentModeCount, presentModes);
            if (result != VK_SUCCESS) {
                SET_ERROR_CODE("vkGetPhysicalDeviceSurfacePresentModesKHR()", result);
                SDL_free(presentModes);
                return result;
            }

            if (rendererData->vsync == 0) {
                /* If vsync is not requested, in favor these options in order:
                   VK_PRESENT_MODE_IMMEDIATE_KHR    - no v-sync with tearing
                   VK_PRESENT_MODE_MAILBOX_KHR      - no v-sync without tearing
                   VK_PRESENT_MODE_FIFO_RELAXED_KHR - no v-sync, may tear */
                for (uint32_t i = 0; i < presentModeCount; i++) {
                    if (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                        presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                        break;
                    }
                    else if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
                        presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                    }
                    else if ((presentMode != VK_PRESENT_MODE_MAILBOX_KHR) &&
                             (presentModes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR)) {
                        presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
                    }
                }
            } else if (rendererData->vsync == -1) {
                for (uint32_t i = 0; i < presentModeCount; i++) {
                    if (presentModes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
                        presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
                        break;
                    }
                }
            }
            SDL_free(presentModes);
        }
    }

    VkSwapchainCreateInfoKHR swapchainCreateInfo = { 0 };
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.surface = rendererData->surface;
    swapchainCreateInfo.minImageCount = rendererData->swapchainDesiredImageCount;
    swapchainCreateInfo.imageFormat = rendererData->surfaceFormat.format;
    swapchainCreateInfo.imageColorSpace = rendererData->surfaceFormat.colorSpace;
    swapchainCreateInfo.imageExtent = rendererData->swapchainSize;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfo.preTransform = rendererData->swapChainPreTransform;
    swapchainCreateInfo.compositeAlpha = (renderer->window->flags & SDL_WINDOW_TRANSPARENT) ? (VkCompositeAlphaFlagBitsKHR)0 : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfo.presentMode = presentMode;
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = rendererData->swapchain;
    result = vkCreateSwapchainKHR(rendererData->device, &swapchainCreateInfo, NULL, &rendererData->swapchain);

    if (swapchainCreateInfo.oldSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(rendererData->device, swapchainCreateInfo.oldSwapchain, NULL);
    }

    if (result != VK_SUCCESS) {
        rendererData->swapchain = VK_NULL_HANDLE;
        SET_ERROR_CODE("vkCreateSwapchainKHR()", result);
        return result;
    }

    SDL_free(rendererData->swapchainImages);
    rendererData->swapchainImages = NULL;
    result = vkGetSwapchainImagesKHR(rendererData->device, rendererData->swapchain, &rendererData->swapchainImageCount, NULL);
    if (result != VK_SUCCESS) {
        rendererData->swapchainImageCount = 0;
        SET_ERROR_CODE("vkGetSwapchainImagesKHR()", result);
        return result;
    }

    rendererData->swapchainImages = (VkImage *)SDL_malloc(sizeof(VkImage) * rendererData->swapchainImageCount);
    result = vkGetSwapchainImagesKHR(rendererData->device,
                                     rendererData->swapchain,
                                     &rendererData->swapchainImageCount,
                                     rendererData->swapchainImages);
    if (result != VK_SUCCESS) {
        SDL_free(rendererData->swapchainImages);
        rendererData->swapchainImages = NULL;
        rendererData->swapchainImageCount = 0;
        SET_ERROR_CODE("vkGetSwapchainImagesKHR()", result);
        return result;
    }

    // Create VkImageView's for swapchain images
    {
        VkImageViewCreateInfo imageViewCreateInfo = { 0 };
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.flags = 0;
        imageViewCreateInfo.format = rendererData->surfaceFormat.format;
        imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        rendererData->swapchainImageViews = (VkImageView *)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkImageView));
        SDL_free(rendererData->swapchainImageLayouts);
        rendererData->swapchainImageLayouts = (VkImageLayout *)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkImageLayout));
        for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
            imageViewCreateInfo.image = rendererData->swapchainImages[i];
            result = vkCreateImageView(rendererData->device, &imageViewCreateInfo, NULL, &rendererData->swapchainImageViews[i]);
            if (result != VK_SUCCESS) {
                VULKAN_DestroyAll(renderer);
                SET_ERROR_CODE("vkCreateImageView()", result);
                return result;
            }
            rendererData->swapchainImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        }

    }

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = { 0 };
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = rendererData->commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = rendererData->swapchainImageCount;
    rendererData->commandBuffers = (VkCommandBuffer *)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkCommandBuffer));
    result = vkAllocateCommandBuffers(rendererData->device, &commandBufferAllocateInfo, rendererData->commandBuffers);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyAll(renderer);
        SET_ERROR_CODE("vkAllocateCommandBuffers()", result);
        return result;
    }

    // Create fences
    rendererData->fences = (VkFence *)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkFence));
    for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
        VkFenceCreateInfo fenceCreateInfo = { 0 };
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        result = vkCreateFence(rendererData->device, &fenceCreateInfo, NULL, &rendererData->fences[i]);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyAll(renderer);
            SET_ERROR_CODE("vkCreateFence()", result);
            return result;
        }
    }

    // Create renderpasses and framebuffer
    for (uint32_t i = 0; i < SDL_arraysize(rendererData->renderPasses); i++) {
        if (rendererData->renderPasses[i] != VK_NULL_HANDLE) {
            vkDestroyRenderPass(rendererData->device, rendererData->renderPasses[i], NULL);
            rendererData->renderPasses[i] = VK_NULL_HANDLE;
        }
    }
    rendererData->framebuffers = (VkFramebuffer *)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkFramebuffer));
    result = VULKAN_CreateFramebuffersAndRenderPasses(renderer,
        rendererData->swapchainSize.width,
        rendererData->swapchainSize.height,
        rendererData->surfaceFormat.format,
        rendererData->swapchainImageCount,
        rendererData->swapchainImageViews,
        rendererData->framebuffers,
        rendererData->renderPasses);
    if (result != VK_SUCCESS) {
        VULKAN_DestroyAll(renderer);
        SET_ERROR_CODE("VULKAN_CreateFramebuffersAndRenderPasses()", result);
        return result;
    }

    // Create descriptor pools - start by allocating one per swapchain image, let it grow if more are needed
    rendererData->descriptorPools = (VkDescriptorPool **)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkDescriptorPool*));
    rendererData->numDescriptorPools = (uint32_t *)SDL_calloc(rendererData->swapchainImageCount, sizeof(uint32_t));
    for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
        // Start by just allocating one pool, it will grow if needed
        rendererData->numDescriptorPools[i] = 1;
        rendererData->descriptorPools[i] = (VkDescriptorPool *)SDL_calloc(1, sizeof(VkDescriptorPool));
        rendererData->descriptorPools[i][0] = VULKAN_AllocateDescriptorPool(rendererData);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyAll(renderer);
            return result;
        }
    }

    // Create semaphores
    rendererData->imageAvailableSemaphores = (VkSemaphore *)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkSemaphore));
    rendererData->renderingFinishedSemaphores = (VkSemaphore *)SDL_calloc(rendererData->swapchainImageCount, sizeof(VkSemaphore));
    for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
        rendererData->imageAvailableSemaphores[i] = VULKAN_CreateSemaphore(rendererData);
        if (rendererData->imageAvailableSemaphores[i] == VK_NULL_HANDLE) {
            VULKAN_DestroyAll(renderer);
            return VK_ERROR_UNKNOWN;
        }
        rendererData->renderingFinishedSemaphores[i] = VULKAN_CreateSemaphore(rendererData);
        if (rendererData->renderingFinishedSemaphores[i] == VK_NULL_HANDLE) {
            VULKAN_DestroyAll(renderer);
            return VK_ERROR_UNKNOWN;
        }
    }

    // Upload buffers
    rendererData->uploadBuffers = (VULKAN_Buffer **)SDL_calloc(rendererData->swapchainImageCount, sizeof(VULKAN_Buffer*));
    for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
        rendererData->uploadBuffers[i] = (VULKAN_Buffer *)SDL_calloc(SDL_VULKAN_NUM_UPLOAD_BUFFERS, sizeof(VULKAN_Buffer));
    }
    SDL_free(rendererData->currentUploadBuffer);
    rendererData->currentUploadBuffer = (int *)SDL_calloc(rendererData->swapchainImageCount, sizeof(int));

    // Constant buffers
    rendererData->constantBuffers = (VULKAN_Buffer **)SDL_calloc(rendererData->swapchainImageCount, sizeof(VULKAN_Buffer*));
    rendererData->numConstantBuffers = (uint32_t *)SDL_calloc(rendererData->swapchainImageCount, sizeof(uint32_t));
    for (uint32_t i = 0; i < rendererData->swapchainImageCount; i++) {
        // Start with just allocating one, will grow if needed
        rendererData->numConstantBuffers[i] = 1;
        rendererData->constantBuffers[i] = (VULKAN_Buffer *)SDL_calloc(1, sizeof(VULKAN_Buffer));
        result = VULKAN_AllocateBuffer(rendererData,
            SDL_VULKAN_CONSTANT_BUFFER_DEFAULT_SIZE,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &rendererData->constantBuffers[i][0]);
        if (result != VK_SUCCESS) {
            VULKAN_DestroyAll(renderer);
            return result;
        }
    }
    rendererData->currentConstantBufferOffset = -1;
    rendererData->currentConstantBufferIndex = 0;

    VULKAN_AcquireNextSwapchainImage(renderer);

    SDL_PropertiesID props = SDL_GetRendererProperties(renderer);
    SDL_SetNumberProperty(props, SDL_PROP_RENDERER_VULKAN_SWAPCHAIN_IMAGE_COUNT_NUMBER, rendererData->swapchainImageCount);

    return result;
}

// Initialize all resources that change when the window's size changes.
static VkResult VULKAN_CreateWindowSizeDependentResources(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VkResult result = VK_SUCCESS;
    int w, h;

    // Release resources in the current command list
    VULKAN_IssueBatch(rendererData);
    VULKAN_WaitForGPU(rendererData);

    /* The width and height of the swap chain must be based on the display's
     * non-rotated size.
     */
    SDL_GetWindowSizeInPixels(renderer->window, &w, &h);

    result = VULKAN_CreateSwapChain(renderer, w, h);
    if (result != VK_SUCCESS) {
        rendererData->recreateSwapchain = VK_TRUE;
    }

    rendererData->viewportDirty = true;

    return result;
}

static bool VULKAN_HandleDeviceLost(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    bool recovered = false;

    VULKAN_DestroyAll(renderer);

    if (VULKAN_CreateDeviceResources(renderer, rendererData->create_props) == VK_SUCCESS &&
        VULKAN_CreateWindowSizeDependentResources(renderer) == VK_SUCCESS) {
        recovered = true;
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Renderer couldn't recover from device lost: %s", SDL_GetError());
        VULKAN_DestroyAll(renderer);
    }

    // Let the application know that the device has been reset or lost
    SDL_Event event;
    SDL_zero(event);
    event.type = recovered ? SDL_EVENT_RENDER_DEVICE_RESET : SDL_EVENT_RENDER_DEVICE_LOST;
    event.render.windowID = SDL_GetWindowID(SDL_GetRenderWindow(renderer));
    SDL_PushEvent(&event);

    return recovered;
}

// This method is called when the window's size changes.
static VkResult VULKAN_UpdateForWindowSizeChange(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    // If the GPU has previous work, wait for it to be done first
    VULKAN_WaitForGPU(rendererData);

    return VULKAN_CreateWindowSizeDependentResources(renderer);
}

static void VULKAN_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;

    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        rendererData->recreateSwapchain = true;
    }
}

static bool VULKAN_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    if (GetBlendFactor(srcColorFactor) == VK_BLEND_FACTOR_MAX_ENUM ||
        GetBlendFactor(srcAlphaFactor)  == VK_BLEND_FACTOR_MAX_ENUM ||
        GetBlendOp(colorOperation) == VK_BLEND_OP_MAX_ENUM ||
        GetBlendFactor(dstColorFactor) == VK_BLEND_FACTOR_MAX_ENUM ||
        GetBlendFactor(dstAlphaFactor) == VK_BLEND_FACTOR_MAX_ENUM ||
        GetBlendOp(alphaOperation) == VK_BLEND_OP_MAX_ENUM) {
        return false;
    }
    return true;
}

static bool VULKAN_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData;
    VkResult result;
    VkFormat textureFormat = SDLPixelFormatToVkTextureFormat(texture->format, renderer->output_colorspace);
    uint32_t width = texture->w;
    uint32_t height = texture->h;
    VkComponentMapping imageViewSwizzle = rendererData->identitySwizzle;

    if (!rendererData->device) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    if (textureFormat == VK_FORMAT_UNDEFINED) {
        return SDL_SetError("%s, An unsupported SDL pixel format (0x%x) was specified", __FUNCTION__, texture->format);
    }

    textureData = (VULKAN_TextureData *)SDL_calloc(1, sizeof(*textureData));
    if (!textureData) {
        return false;
    }
    texture->internal = textureData;
    if (SDL_COLORSPACETRANSFER(texture->colorspace) == SDL_TRANSFER_CHARACTERISTICS_SRGB) {
        textureData->shader = SHADER_RGB;
    } else {
        textureData->shader = SHADER_ADVANCED;
    }

#ifdef SDL_HAVE_YUV
    // YUV textures must have even width and height.  Also create Ycbcr conversion
    if (texture->format == SDL_PIXELFORMAT_YV12 ||
        texture->format == SDL_PIXELFORMAT_IYUV ||
        texture->format == SDL_PIXELFORMAT_NV12 ||
        texture->format == SDL_PIXELFORMAT_NV21 ||
        texture->format == SDL_PIXELFORMAT_P010) {
        const uint32_t YUV_SD_THRESHOLD = 576;

        // Check that we have VK_KHR_sampler_ycbcr_conversion support
        if (!rendererData->supportsKHRSamplerYCbCrConversion) {
            return SDL_SetError("YUV textures require a Vulkan device that supports VK_KHR_sampler_ycbcr_conversion");
        }

        VkSamplerYcbcrConversionCreateInfoKHR samplerYcbcrConversionCreateInfo = { 0 };
        samplerYcbcrConversionCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR;

        // Pad width/height to multiple of 2
        width = (width + 1) & ~1;
        height = (height + 1) & ~1;

        // Create samplerYcbcrConversion which will be used on the VkImageView and VkSampler
        samplerYcbcrConversionCreateInfo.format = textureFormat;
        switch (SDL_COLORSPACEMATRIX(texture->colorspace)) {
        case SDL_MATRIX_COEFFICIENTS_BT470BG:
        case SDL_MATRIX_COEFFICIENTS_BT601:
            samplerYcbcrConversionCreateInfo.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601_KHR;
            break;
        case SDL_MATRIX_COEFFICIENTS_BT709:
            samplerYcbcrConversionCreateInfo.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709_KHR;
            break;
        case SDL_MATRIX_COEFFICIENTS_BT2020_NCL:
            samplerYcbcrConversionCreateInfo.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020_KHR;
            break;
        case SDL_MATRIX_COEFFICIENTS_UNSPECIFIED:
            if (texture->format == SDL_PIXELFORMAT_P010) {
                samplerYcbcrConversionCreateInfo.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020_KHR;
            } else if (height > YUV_SD_THRESHOLD) {
                samplerYcbcrConversionCreateInfo.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709_KHR;
            } else {
                samplerYcbcrConversionCreateInfo.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601_KHR;
            }
            break;
        default:
            return SDL_SetError("Unsupported Ycbcr colorspace: %d", SDL_COLORSPACEMATRIX(texture->colorspace));
        }
        samplerYcbcrConversionCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        samplerYcbcrConversionCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        samplerYcbcrConversionCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        samplerYcbcrConversionCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        if (texture->format == SDL_PIXELFORMAT_YV12 ||
            texture->format == SDL_PIXELFORMAT_NV21) {
            samplerYcbcrConversionCreateInfo.components.r = VK_COMPONENT_SWIZZLE_B;
            samplerYcbcrConversionCreateInfo.components.b = VK_COMPONENT_SWIZZLE_R;
        }

        switch (SDL_COLORSPACERANGE(texture->colorspace)) {
        case SDL_COLOR_RANGE_LIMITED:
            samplerYcbcrConversionCreateInfo.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW_KHR;
            break;
        case SDL_COLOR_RANGE_FULL:
        default:
            samplerYcbcrConversionCreateInfo.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL_KHR;
            break;
        }

        switch (SDL_COLORSPACECHROMA(texture->colorspace)) {
        case SDL_CHROMA_LOCATION_LEFT:
            samplerYcbcrConversionCreateInfo.xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN_KHR;
            samplerYcbcrConversionCreateInfo.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT_KHR;
            break;
        case SDL_CHROMA_LOCATION_TOPLEFT:
            samplerYcbcrConversionCreateInfo.xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN_KHR;
            samplerYcbcrConversionCreateInfo.yChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN_KHR;
            break;
        case SDL_CHROMA_LOCATION_NONE:
        case SDL_CHROMA_LOCATION_CENTER:
        default:
            samplerYcbcrConversionCreateInfo.xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT_KHR;
            samplerYcbcrConversionCreateInfo.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT_KHR;
            break;
        }
        samplerYcbcrConversionCreateInfo.chromaFilter = VK_FILTER_LINEAR;
        samplerYcbcrConversionCreateInfo.forceExplicitReconstruction = VK_FALSE;

        result = vkCreateSamplerYcbcrConversionKHR(rendererData->device, &samplerYcbcrConversionCreateInfo, NULL, &textureData->samplerYcbcrConversion);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkCreateSamplerYcbcrConversionKHR()", result);
            return false;
        }

        // Also create VkSampler object which we will need to pass to the PSO as an immutable sampler
        VkSamplerCreateInfo samplerCreateInfo = { 0 };
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 1000.0f;

        VkSamplerYcbcrConversionInfoKHR samplerYcbcrConversionInfo = { 0 };
        samplerYcbcrConversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR;
        samplerYcbcrConversionInfo.conversion = textureData->samplerYcbcrConversion;
        samplerCreateInfo.pNext = &samplerYcbcrConversionInfo;
        result = vkCreateSampler(rendererData->device, &samplerCreateInfo, NULL, &textureData->samplerYcbcr);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkCreateSampler()", result);
            return false;
        }

        // Allocate special descriptor set layout with samplerYcbcr baked as an immutable sampler
        result = VULKAN_CreateDescriptorSetAndPipelineLayout(rendererData, textureData->samplerYcbcr, &textureData->descriptorSetLayoutYcbcr, &textureData->pipelineLayoutYcbcr);
        if (result != VK_SUCCESS) {
            return false;
        }
    }
#endif
    textureData->width = width;
    textureData->height = height;

    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }

    result = VULKAN_AllocateImage(rendererData, create_props, width, height, textureFormat, usage, imageViewSwizzle, textureData->samplerYcbcrConversion, &textureData->mainImage);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("VULKAN_AllocateImage()", result);
        return false;
    }

    SDL_PropertiesID props = SDL_GetTextureProperties(texture);
    SDL_SetNumberProperty(props, SDL_PROP_TEXTURE_CREATE_VULKAN_TEXTURE_NUMBER, (Sint64)textureData->mainImage.image);

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        result = VULKAN_CreateFramebuffersAndRenderPasses(renderer,
            texture->w,
            texture->h,
            textureFormat,
            1,
            &textureData->mainImage.imageView,
            &textureData->mainFramebuffer,
            textureData->mainRenderpasses);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("VULKAN_CreateFramebuffersAndRenderPasses()", result);
            return false;
        }
    }
    return true;
}

static void VULKAN_DestroyTexture(SDL_Renderer *renderer,
                                 SDL_Texture *texture)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = (VULKAN_TextureData *)texture->internal;

    if (!textureData) {
        return;
    }

    /* Because SDL_DestroyTexture might be called while the data is in-flight, we need to issue the batch first
       Unfortunately, this means that deleting a lot of textures mid-frame will have poor performance. */
    VULKAN_IssueBatch(rendererData);
    VULKAN_WaitForGPU(rendererData);

    VULKAN_DestroyImage(rendererData, &textureData->mainImage);

#ifdef SDL_HAVE_YUV
    if (textureData->samplerYcbcrConversion != VK_NULL_HANDLE) {
        vkDestroySamplerYcbcrConversionKHR(rendererData->device, textureData->samplerYcbcrConversion, NULL);
        textureData->samplerYcbcrConversion = VK_NULL_HANDLE;
    }
    if (textureData->samplerYcbcr != VK_NULL_HANDLE) {
        vkDestroySampler(rendererData->device, textureData->samplerYcbcr, NULL);
        textureData->samplerYcbcr = VK_NULL_HANDLE;
    }
    if (textureData->pipelineLayoutYcbcr != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(rendererData->device, textureData->pipelineLayoutYcbcr, NULL);
        textureData->pipelineLayoutYcbcr = VK_NULL_HANDLE;
    }
    if (textureData->descriptorSetLayoutYcbcr != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(rendererData->device, textureData->descriptorSetLayoutYcbcr, NULL);
        textureData->descriptorSetLayoutYcbcr = VK_NULL_HANDLE;
    }
#endif

    VULKAN_DestroyBuffer(rendererData, &textureData->stagingBuffer);
    if (textureData->mainFramebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(rendererData->device, textureData->mainFramebuffer, NULL);
        textureData->mainFramebuffer = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < SDL_arraysize(textureData->mainRenderpasses); i++) {
        if (textureData->mainRenderpasses[i] != VK_NULL_HANDLE) {
            vkDestroyRenderPass(rendererData->device, textureData->mainRenderpasses[i], NULL);
            textureData->mainRenderpasses[i] = VK_NULL_HANDLE;
        }
    }

    SDL_free(textureData);
    texture->internal = NULL;
}

static bool VULKAN_UpdateTextureInternal(VULKAN_RenderData *rendererData, VkImage image, VkFormat format, int plane, int x, int y, int w, int h, const void *pixels, int pitch, VkImageLayout *imageLayout)
{
    VkDeviceSize pixelSize = VULKAN_GetBytesPerPixel(format);
    VkDeviceSize length = w * pixelSize;
    VkDeviceSize uploadBufferSize = length * h;
    const Uint8 *src;
    Uint8 *dst;
    VkResult rc;
    int planeCount = VULKAN_VkFormatGetNumPlanes(format);

    VULKAN_EnsureCommandBuffer(rendererData);

    int currentUploadBufferIndex = rendererData->currentUploadBuffer[rendererData->currentCommandBufferIndex];
    VULKAN_Buffer *uploadBuffer = &rendererData->uploadBuffers[rendererData->currentCommandBufferIndex][currentUploadBufferIndex];

    rc = VULKAN_AllocateBuffer(rendererData, uploadBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        uploadBuffer);
    if (rc != VK_SUCCESS) {
        return false;
    }

    src = (const Uint8 *)pixels;
    dst = (Uint8 *)uploadBuffer->mappedBufferPtr;
    if (length == (VkDeviceSize)pitch) {
        SDL_memcpy(dst, src, (size_t)length * h);
    } else {
        if (length > (VkDeviceSize)pitch) {
            length = pitch;
        }
        for (VkDeviceSize row = h; row--; ) {
            SDL_memcpy(dst, src, (size_t)length);
            src += pitch;
            dst += length;
        }
    }

    // Make sure the destination is in the correct resource state
    VULKAN_RecordPipelineImageBarrier(rendererData,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        image,
        imageLayout);

    VkBufferImageCopy region;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageSubresource.mipLevel = 0;
    if (planeCount <= 1) {
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    } else {
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT << plane;
    }
    region.imageOffset.x = x;
    region.imageOffset.y = y;
    region.imageOffset.z = 0;
    region.imageExtent.width = w;
    region.imageExtent.height = h;
    region.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(rendererData->currentCommandBuffer, uploadBuffer->buffer, image, *imageLayout, 1, &region);

    // Transition the texture to be shader accessible
    VULKAN_RecordPipelineImageBarrier(rendererData,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        image,
        imageLayout);

    rendererData->currentUploadBuffer[rendererData->currentCommandBufferIndex]++;

    // If we've used up all the upload buffers, we need to issue the batch
    if (rendererData->currentUploadBuffer[rendererData->currentCommandBufferIndex] == SDL_VULKAN_NUM_UPLOAD_BUFFERS) {
        VULKAN_IssueBatch(rendererData);
    }

    return true;
}


static bool VULKAN_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                               const SDL_Rect *rect, const void *srcPixels,
                               int srcPitch)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = (VULKAN_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, 0, rect->x, rect->y, rect->w, rect->h, srcPixels, srcPitch, &textureData->mainImage.imageLayout)) {
        return false;
    }
#ifdef SDL_HAVE_YUV
    Uint32 numPlanes = VULKAN_VkFormatGetNumPlanes(textureData->mainImage.format);
    // Skip to the correct offset into the next texture
    srcPixels = (const void *)((const Uint8 *)srcPixels + rect->h * srcPitch);
    // YUV data
    if (numPlanes == 3) {
        for (Uint32 plane = 1; plane < numPlanes; plane++) {
            if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, plane, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, srcPixels, (srcPitch + 1) / 2, &textureData->mainImage.imageLayout)) {
                return false;
            }

            // Skip to the correct offset into the next texture
            srcPixels = (const void *)((const Uint8 *)srcPixels + ((rect->h + 1) / 2) * ((srcPitch + 1) / 2));
        }
    }
    // NV12/NV21 data
    else if (numPlanes == 2)
    {
        if (texture->format == SDL_PIXELFORMAT_P010) {
            srcPitch = (srcPitch + 3) & ~3;
        } else {
            srcPitch = (srcPitch + 1) & ~1;
        }

        if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, 1, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, srcPixels, srcPitch, &textureData->mainImage.imageLayout)) {
            return false;
        }
    }
#endif
    return true;
}

#ifdef SDL_HAVE_YUV
static bool VULKAN_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect,
                                  const Uint8 *Yplane, int Ypitch,
                                  const Uint8 *Uplane, int Upitch,
                                  const Uint8 *Vplane, int Vpitch)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = (VULKAN_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, 0, rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch, &textureData->mainImage.imageLayout)) {
        return false;
    }
    if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, 1, rect->x / 2, rect->y / 2, rect->w / 2, rect->h / 2, Uplane, Upitch, &textureData->mainImage.imageLayout)) {
        return false;
    }
    if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, 2, rect->x / 2, rect->y / 2, rect->w / 2, rect->h / 2, Vplane, Vpitch, &textureData->mainImage.imageLayout)) {
        return false;
    }
    return true;
}

static bool VULKAN_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                 const SDL_Rect *rect,
                                 const Uint8 *Yplane, int Ypitch,
                                 const Uint8 *UVplane, int UVpitch)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = (VULKAN_TextureData *)texture->internal;

    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, 0, rect->x, rect->y, rect->w, rect->h, Yplane, Ypitch, &textureData->mainImage.imageLayout)) {
        return false;
    }

    if (!VULKAN_UpdateTextureInternal(rendererData, textureData->mainImage.image, textureData->mainImage.format, 1, rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2, UVplane, UVpitch, &textureData->mainImage.imageLayout)) {
        return false;
    }
    return true;
}
#endif

static bool VULKAN_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                             const SDL_Rect *rect, void **pixels, int *pitch)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = (VULKAN_TextureData *)texture->internal;
    VkResult rc;
    if (!textureData) {
        return SDL_SetError("Texture is not currently available");
    }

    if (textureData->stagingBuffer.buffer != VK_NULL_HANDLE) {
        return SDL_SetError("texture is already locked");
    }

    VkDeviceSize pixelSize = VULKAN_GetBytesPerPixel(textureData->mainImage.format);
    VkDeviceSize length = rect->w * pixelSize;
    VkDeviceSize stagingBufferSize = length * rect->h;
    rc = VULKAN_AllocateBuffer(rendererData,
        stagingBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &textureData->stagingBuffer);
    if (rc != VK_SUCCESS) {
        return false;
    }

    /* Make note of where the staging texture will be written to
     * (on a call to SDL_UnlockTexture):
     */
    textureData->lockedRect = *rect;

    /* Make sure the caller has information on the texture's pixel buffer,
     * then return:
     */
    *pixels = textureData->stagingBuffer.mappedBufferPtr;
    *pitch = (int)length;
    return true;

}

static void VULKAN_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = (VULKAN_TextureData *)texture->internal;

    if (!textureData) {
        return;
    }

    VULKAN_EnsureCommandBuffer(rendererData);

     // Make sure the destination is in the correct resource state
    VULKAN_RecordPipelineImageBarrier(rendererData,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        textureData->mainImage.image,
        &textureData->mainImage.imageLayout);

    VkBufferImageCopy region;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageOffset.x = textureData->lockedRect.x;
    region.imageOffset.y = textureData->lockedRect.y;
    region.imageOffset.z = 0;
    region.imageExtent.width = textureData->lockedRect.w;
    region.imageExtent.height = textureData->lockedRect.h;
    region.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(rendererData->currentCommandBuffer, textureData->stagingBuffer.buffer, textureData->mainImage.image, textureData->mainImage.imageLayout, 1, &region);

    // Transition the texture to be shader accessible
    VULKAN_RecordPipelineImageBarrier(rendererData,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        textureData->mainImage.image,
        &textureData->mainImage.imageLayout);

    // Execute the command list before releasing the staging buffer
    VULKAN_IssueBatch(rendererData);

    VULKAN_DestroyBuffer(rendererData, &textureData->stagingBuffer);
}

static bool VULKAN_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = NULL;

    VULKAN_EnsureCommandBuffer(rendererData);

    if (!texture) {
        if (rendererData->textureRenderTarget) {

            VULKAN_RecordPipelineImageBarrier(rendererData,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                rendererData->textureRenderTarget->mainImage.image,
                &rendererData->textureRenderTarget->mainImage.imageLayout);
        }
        rendererData->textureRenderTarget = NULL;
        return true;
    }

    textureData = (VULKAN_TextureData *)texture->internal;

    if (textureData->mainImage.imageView == VK_NULL_HANDLE) {
        return SDL_SetError("specified texture is not a render target");
    }

    rendererData->textureRenderTarget = textureData;
    VULKAN_RecordPipelineImageBarrier(rendererData,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                rendererData->textureRenderTarget->mainImage.image,
                &rendererData->textureRenderTarget->mainImage.imageLayout);

    return true;
}

static bool VULKAN_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static bool VULKAN_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    VULKAN_VertexPositionColor *verts = (VULKAN_VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(VULKAN_VertexPositionColor), 0, &cmd->data.draw.first);
    int i;
    bool convert_color = SDL_RenderingLinearSpace(renderer);

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    for (i = 0; i < count; i++) {
        verts->pos[0] = points[i].x + 0.5f;
        verts->pos[1] = points[i].y + 0.5f;
        verts->tex[0] = 0.0f;
        verts->tex[1] = 0.0f;
        verts->color = cmd->data.draw.color;
        if (convert_color) {
            SDL_ConvertToLinear(&verts->color);
        }
        verts++;
    }
    return true;
}

static bool VULKAN_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                               const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                               int num_vertices, const void *indices, int num_indices, int size_indices,
                               float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    VULKAN_VertexPositionColor *verts = (VULKAN_VertexPositionColor *)SDL_AllocateRenderVertices(renderer, count * sizeof(VULKAN_VertexPositionColor), 0, &cmd->data.draw.first);
    bool convert_color = SDL_RenderingLinearSpace(renderer);
    VULKAN_TextureData *textureData = texture ? (VULKAN_TextureData *)texture->internal : NULL;
    float u_scale = textureData ? (float)texture->w / textureData->width : 0.0f;
    float v_scale = textureData ? (float)texture->h / textureData->height : 0.0f;

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    for (i = 0; i < count; i++) {
        int j;
        float *xy_;
        if (size_indices == 4) {
            j = ((const Uint32 *)indices)[i];
        } else if (size_indices == 2) {
            j = ((const Uint16 *)indices)[i];
        } else if (size_indices == 1) {
            j = ((const Uint8 *)indices)[i];
        } else {
            j = i;
        }

        xy_ = (float *)((char *)xy + j * xy_stride);

        verts->pos[0] = xy_[0] * scale_x;
        verts->pos[1] = xy_[1] * scale_y;
        verts->color = *(SDL_FColor *)((char *)color + j * color_stride);
        if (convert_color) {
            SDL_ConvertToLinear(&verts->color);
        }

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            verts->tex[0] = uv_[0] * u_scale;
            verts->tex[1] = uv_[1] * v_scale;
        } else {
            verts->tex[0] = 0.0f;
            verts->tex[1] = 0.0f;
        }

        verts += 1;
    }
    return true;
}

static bool VULKAN_UpdateVertexBuffer(SDL_Renderer *renderer,
                                    const void *vertexData, size_t dataSizeInBytes, VULKAN_DrawStateCache *stateCache)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    const int vbidx = rendererData->currentVertexBuffer;
    VULKAN_Buffer *vertexBuffer;

    if (dataSizeInBytes == 0) {
        return true; // nothing to do.
    }

    if (rendererData->issueBatch) {
        if (VULKAN_IssueBatch(rendererData) != VK_SUCCESS) {
            return SDL_SetError("Failed to issue intermediate batch");
        }
    }
    // If the existing vertex buffer isn't big enough, we need to recreate a big enough one
    if (dataSizeInBytes > rendererData->vertexBuffers[vbidx].size) {
        VULKAN_IssueBatch(rendererData);
        VULKAN_WaitForGPU(rendererData);
        VULKAN_CreateVertexBuffer(rendererData, vbidx, dataSizeInBytes);
    }

    vertexBuffer = &rendererData->vertexBuffers[vbidx];
    SDL_memcpy(vertexBuffer->mappedBufferPtr, vertexData, dataSizeInBytes);

    stateCache->vertexBuffer = vertexBuffer->buffer;

    rendererData->currentVertexBuffer = vbidx + 1;
    if (rendererData->currentVertexBuffer >= SDL_VULKAN_NUM_VERTEX_BUFFERS) {
        rendererData->currentVertexBuffer = 0;
        rendererData->issueBatch = true;
    }

    return true;
}

static VkSurfaceTransformFlagBitsKHR VULKAN_GetRotationForCurrentRenderTarget(VULKAN_RenderData *rendererData)
{
    if (rendererData->textureRenderTarget) {
        return VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        return rendererData->swapChainPreTransform;
    }
}

static bool VULKAN_IsDisplayRotated90Degrees(VkSurfaceTransformFlagBitsKHR rotation)
{
    switch (rotation) {
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
            return true;
        default:
            return false;
    }
}

static bool VULKAN_UpdateViewport(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    const SDL_Rect *viewport = &rendererData->currentViewport;
    Float4X4 projection;
    Float4X4 view;
    VkSurfaceTransformFlagBitsKHR rotation = VULKAN_GetRotationForCurrentRenderTarget(rendererData);
    bool swapDimensions;

    if (viewport->w == 0 || viewport->h == 0) {
        /* If the viewport is empty, assume that it is because
         * SDL_CreateRenderer is calling it, and will call it again later
         * with a non-empty viewport.
         */
        // SDL_Log("%s, no viewport was set!", __FUNCTION__);
        return false;
    }

    switch (rotation) {
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
            projection = MatrixRotationZ(SDL_PI_F * 0.5f);
            break;
        case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
            projection = MatrixRotationZ(SDL_PI_F);
            break;
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
            projection = MatrixRotationZ(-SDL_PI_F * 0.5f);
            break;
        case VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR:
        default:
            projection = MatrixIdentity();
            break;

    }

    // Update the view matrix
    SDL_zero(view);
    view.m[0][0] = 2.0f / viewport->w;
    view.m[1][1] = -2.0f / viewport->h;
    view.m[2][2] = 1.0f;
    view.m[3][0] = -1.0f;
    view.m[3][1] = 1.0f;
    view.m[3][3] = 1.0f;

    rendererData->vertexShaderConstantsData.projectionAndView = MatrixMultiply(
        view,
        projection);

    VkViewport vkViewport;

    swapDimensions = VULKAN_IsDisplayRotated90Degrees(rotation);
    if (swapDimensions) {
        vkViewport.x = viewport->y;
        vkViewport.y = viewport->x;
        vkViewport.width = viewport->h;
        vkViewport.height = viewport->w;
    }
    else {
        vkViewport.x = viewport->x;
        vkViewport.y = viewport->y;
        vkViewport.width = viewport->w;
        vkViewport.height = viewport->h;
    }
    vkViewport.minDepth = 0.0f;
    vkViewport.maxDepth = 1.0f;
    vkCmdSetViewport(rendererData->currentCommandBuffer, 0, 1, &vkViewport);

    rendererData->viewportDirty = false;
    return true;
}

static bool VULKAN_UpdateClipRect(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    const SDL_Rect *viewport = &rendererData->currentViewport;
    VkSurfaceTransformFlagBitsKHR rotation = VULKAN_GetRotationForCurrentRenderTarget(rendererData);
    bool swapDimensions = VULKAN_IsDisplayRotated90Degrees(rotation);

    VkRect2D scissor;
    if (rendererData->currentCliprectEnabled) {
        scissor.offset.x = viewport->x + rendererData->currentCliprect.x;
        scissor.offset.y = viewport->y + rendererData->currentCliprect.y;
        scissor.extent.width = rendererData->currentCliprect.w;
        scissor.extent.height = rendererData->currentCliprect.h;
    } else {
        scissor.offset.x = viewport->x;
        scissor.offset.y = viewport->y;
        scissor.extent.width = viewport->w;
        scissor.extent.height = viewport->h;
    }
    if (swapDimensions) {
        VkRect2D scissorTemp = scissor;
        scissor.offset.x = scissorTemp.offset.y;
        scissor.offset.y = scissorTemp.offset.x;
        scissor.extent.width = scissorTemp.extent.height;
        scissor.extent.height = scissorTemp.extent.width;
    }
    vkCmdSetScissor(rendererData->currentCommandBuffer, 0, 1, &scissor);

    rendererData->cliprectDirty = false;
    return true;
}

static void VULKAN_SetupShaderConstants(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const SDL_Texture *texture, VULKAN_PixelShaderConstants *constants)
{
    float output_headroom;

    SDL_zerop(constants);

    constants->scRGB_output = (float)SDL_RenderingLinearSpace(renderer);
    constants->color_scale = cmd->data.draw.color_scale;

    if (texture) {
        switch (texture->format) {
        case SDL_PIXELFORMAT_YV12:
        case SDL_PIXELFORMAT_IYUV:
        case SDL_PIXELFORMAT_NV12:
        case SDL_PIXELFORMAT_NV21:
            constants->input_type = INPUTTYPE_SRGB;
            break;
        case SDL_PIXELFORMAT_P010:
            constants->input_type = INPUTTYPE_HDR10;
            break;
        default:
            if (texture->colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
                constants->input_type = INPUTTYPE_SCRGB;
            } else if (SDL_COLORSPACEPRIMARIES(texture->colorspace) == SDL_COLOR_PRIMARIES_BT2020 &&
                       SDL_COLORSPACETRANSFER(texture->colorspace) == SDL_TRANSFER_CHARACTERISTICS_PQ) {
                constants->input_type = INPUTTYPE_HDR10;
            } else {
                // The sampler will convert from sRGB to linear on load if working in linear colorspace
                constants->input_type = INPUTTYPE_UNSPECIFIED;
            }
            break;
        }

        if (cmd->data.draw.texture_scale_mode == SDL_SCALEMODE_PIXELART) {
            constants->pixel_art = 1.0f;
            constants->texture_width = texture->w;
            constants->texture_height = texture->h;
            constants->texel_width = 1.0f / constants->texture_width;
            constants->texel_height = 1.0f / constants->texture_height;
        }

        constants->sdr_white_point = texture->SDR_white_point;

        if (renderer->target) {
            output_headroom = renderer->target->HDR_headroom;
        } else {
            output_headroom = renderer->HDR_headroom;
        }

        if (texture->HDR_headroom > output_headroom) {
            constants->tonemap_method = TONEMAP_CHROME;
            constants->tonemap_factor1 = (output_headroom / (texture->HDR_headroom * texture->HDR_headroom));
            constants->tonemap_factor2 = (1.0f / output_headroom);
        }
    }
}

static VkDescriptorPool VULKAN_AllocateDescriptorPool(VULKAN_RenderData *rendererData)
{
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorPoolSize descriptorPoolSizes[3];
    VkResult result;
    descriptorPoolSizes[0].descriptorCount = SDL_VULKAN_MAX_DESCRIPTOR_SETS;
    descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLER;

    descriptorPoolSizes[1].descriptorCount = SDL_VULKAN_MAX_DESCRIPTOR_SETS;
    descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    descriptorPoolSizes[2].descriptorCount = SDL_VULKAN_MAX_DESCRIPTOR_SETS;
    descriptorPoolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { 0 };
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.poolSizeCount = SDL_arraysize(descriptorPoolSizes);
    descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes;
    descriptorPoolCreateInfo.maxSets = SDL_VULKAN_MAX_DESCRIPTOR_SETS;
    result = vkCreateDescriptorPool(rendererData->device, &descriptorPoolCreateInfo, NULL, &descriptorPool);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreateDescrptorPool()", result);
        return VK_NULL_HANDLE;
    }

    return descriptorPool;
}

static VkResult VULKAN_CreateDescriptorSetAndPipelineLayout(VULKAN_RenderData *rendererData, VkSampler samplerYcbcr, VkDescriptorSetLayout *descriptorSetLayoutOut,
    VkPipelineLayout *pipelineLayoutOut)
{
    VkResult result;

    // Descriptor set layout
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { 0 };
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.flags = 0;
    VkDescriptorSetLayoutBinding layoutBindings[2];
    // PixelShaderConstants
    layoutBindings[0].binding = 1;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    layoutBindings[0].pImmutableSamplers = NULL;

    // Combined image/sampler
    layoutBindings[1].binding = 0;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    layoutBindings[1].pImmutableSamplers = (samplerYcbcr != VK_NULL_HANDLE) ? &samplerYcbcr : NULL;

    descriptorSetLayoutCreateInfo.bindingCount = 2;
    descriptorSetLayoutCreateInfo.pBindings = layoutBindings;
    result = vkCreateDescriptorSetLayout(rendererData->device, &descriptorSetLayoutCreateInfo, NULL, descriptorSetLayoutOut);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreateDescriptorSetLayout()", result);
        return result;
    }

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { 0 };
    VkPushConstantRange pushConstantRange;
    pushConstantRange.size = sizeof( VULKAN_VertexShaderConstants );
    pushConstantRange.offset = 0;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayoutOut;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    result = vkCreatePipelineLayout(rendererData->device, &pipelineLayoutCreateInfo, NULL, pipelineLayoutOut);
    if (result != VK_SUCCESS) {
        SET_ERROR_CODE("vkCreatePipelineLayout()", result);
        return result;
    }

    return result;
}

static VkDescriptorSet VULKAN_AllocateDescriptorSet(SDL_Renderer *renderer, VULKAN_Shader shader, VkDescriptorSetLayout descriptorSetLayout,
    VkSampler sampler, VkBuffer constantBuffer, VkDeviceSize constantBufferOffset, VkImageView imageView)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    uint32_t currentDescriptorPoolIndex = rendererData->currentDescriptorPoolIndex;
    VkDescriptorPool descriptorPool = rendererData->descriptorPools[rendererData->currentCommandBufferIndex][currentDescriptorPoolIndex];

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { 0 };
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkResult result = (rendererData->currentDescriptorSetIndex >= SDL_VULKAN_MAX_DESCRIPTOR_SETS) ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
    if (result == VK_SUCCESS) {
        result = vkAllocateDescriptorSets(rendererData->device, &descriptorSetAllocateInfo, &descriptorSet);
    }
    if (result != VK_SUCCESS) {
        // Out of descriptor sets in this pool - see if we have more pools allocated
        currentDescriptorPoolIndex++;
        if (currentDescriptorPoolIndex < rendererData->numDescriptorPools[rendererData->currentCommandBufferIndex]) {
            descriptorPool = rendererData->descriptorPools[rendererData->currentCommandBufferIndex][currentDescriptorPoolIndex];
            descriptorSetAllocateInfo.descriptorPool = descriptorPool;
            result = vkAllocateDescriptorSets(rendererData->device, &descriptorSetAllocateInfo, &descriptorSet);
            if (result != VK_SUCCESS) {
                // This should not fail - we are allocating from the front of the descriptor set
                SDL_SetError("Unable to allocate descriptor set");
                return VK_NULL_HANDLE;
            }
            rendererData->currentDescriptorPoolIndex = currentDescriptorPoolIndex;
            rendererData->currentDescriptorSetIndex = 0;

        }
        // We are out of pools, create a new one
        else {
            descriptorPool = VULKAN_AllocateDescriptorPool(rendererData);
            if (descriptorPool == VK_NULL_HANDLE) {
                // SDL_SetError called in VULKAN_AllocateDescriptorPool if we failed to allocate a new pool
                return VK_NULL_HANDLE;
            }
            rendererData->numDescriptorPools[rendererData->currentCommandBufferIndex]++;
            VkDescriptorPool *descriptorPools = (VkDescriptorPool *)SDL_realloc(rendererData->descriptorPools[rendererData->currentCommandBufferIndex],
                                                            sizeof(VkDescriptorPool) * rendererData->numDescriptorPools[rendererData->currentCommandBufferIndex]);
            descriptorPools[rendererData->numDescriptorPools[rendererData->currentCommandBufferIndex] - 1] = descriptorPool;
            rendererData->descriptorPools[rendererData->currentCommandBufferIndex] = descriptorPools;
            rendererData->currentDescriptorPoolIndex = currentDescriptorPoolIndex;
            rendererData->currentDescriptorSetIndex = 0;

            // Call recursively to allocate from the new pool
            return VULKAN_AllocateDescriptorSet(renderer, shader, descriptorSetLayout, sampler, constantBuffer, constantBufferOffset, imageView);
        }
    }
    rendererData->currentDescriptorSetIndex++;
    VkDescriptorImageInfo combinedImageSamplerDescriptor = { 0 };
    VkDescriptorBufferInfo bufferDescriptor = { 0 };
    bufferDescriptor.buffer = constantBuffer;
    bufferDescriptor.offset = constantBufferOffset;
    bufferDescriptor.range = sizeof(VULKAN_PixelShaderConstants);

    VkWriteDescriptorSet descriptorWrites[2];
    SDL_memset(descriptorWrites, 0, sizeof(descriptorWrites));
    uint32_t descriptorCount = 1; // Always have the uniform buffer

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 1;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].pBufferInfo = &bufferDescriptor;

    if (sampler != VK_NULL_HANDLE && imageView != VK_NULL_HANDLE) {
        descriptorCount++;
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSet;
        descriptorWrites[1].dstBinding = 0;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].pImageInfo = &combinedImageSamplerDescriptor;

        // Ignore the sampler if we're using YcBcCr data since it will be baked in the descriptor set layout
        if (descriptorSetLayout == rendererData->descriptorSetLayout) {
            combinedImageSamplerDescriptor.sampler = sampler;
        }
        combinedImageSamplerDescriptor.imageView = imageView;
        combinedImageSamplerDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    vkUpdateDescriptorSets(rendererData->device, descriptorCount, descriptorWrites, 0, NULL);

    return descriptorSet;
}

static bool VULKAN_SetDrawState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, VULKAN_Shader shader, VkPipelineLayout pipelineLayout, VkDescriptorSetLayout descriptorSetLayout,
    const VULKAN_PixelShaderConstants *shader_constants, VkPrimitiveTopology topology, VkImageView imageView, VkSampler sampler, const Float4X4 *matrix, VULKAN_DrawStateCache *stateCache)

{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    const SDL_BlendMode blendMode = cmd->data.draw.blend;
    VkFormat format = rendererData->surfaceFormat.format;
    const Float4X4 *newmatrix = matrix ? matrix : &rendererData->identity;
    bool updateConstants = false;
    VULKAN_PixelShaderConstants solid_constants;
    VkDescriptorSet descriptorSet;
    VkBuffer constantBuffer;
    VkDeviceSize constantBufferOffset;
    int i;

    if (!VULKAN_ActivateCommandBuffer(renderer, VK_ATTACHMENT_LOAD_OP_LOAD, NULL, stateCache)) {
        return false;
    }

    // See if we need to change the pipeline state
    if (!rendererData->currentPipelineState ||
        rendererData->currentPipelineState->shader != shader ||
        rendererData->currentPipelineState->blendMode != blendMode ||
        rendererData->currentPipelineState->topology != topology ||
        rendererData->currentPipelineState->format != format ||
        rendererData->currentPipelineState->pipelineLayout != pipelineLayout ||
        rendererData->currentPipelineState->descriptorSetLayout != descriptorSetLayout) {

        rendererData->currentPipelineState = NULL;
        for (i = 0; i < rendererData->pipelineStateCount; ++i) {
            VULKAN_PipelineState *candidatePiplineState = &rendererData->pipelineStates[i];
            if (candidatePiplineState->shader == shader &&
                candidatePiplineState->blendMode == blendMode &&
                candidatePiplineState->topology == topology &&
                candidatePiplineState->format == format &&
                candidatePiplineState->pipelineLayout == pipelineLayout &&
                candidatePiplineState->descriptorSetLayout == descriptorSetLayout) {
                rendererData->currentPipelineState = candidatePiplineState;
                break;
            }
        }

        // If we didn't find a match, create a new one -- it must mean the blend mode is non-standard
        if (!rendererData->currentPipelineState) {
            rendererData->currentPipelineState = VULKAN_CreatePipelineState(renderer, shader, pipelineLayout, descriptorSetLayout, blendMode, topology, format);
        }

        if (!rendererData->currentPipelineState) {
            return SDL_SetError("Unable to create required pipeline state");
        }

        vkCmdBindPipeline(rendererData->currentCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rendererData->currentPipelineState->pipeline);
        updateConstants = true;
    }

    if (rendererData->viewportDirty) {
        if (VULKAN_UpdateViewport(renderer)) {
            // vertexShaderConstantsData.projectionAndView has changed
            updateConstants = true;
        }
    }

    if (rendererData->cliprectDirty) {
        VULKAN_UpdateClipRect(renderer);
    }

    if (updateConstants == true || SDL_memcmp(&rendererData->vertexShaderConstantsData.model, newmatrix, sizeof(*newmatrix)) != 0) {
        SDL_memcpy(&rendererData->vertexShaderConstantsData.model, newmatrix, sizeof(*newmatrix));
        vkCmdPushConstants(rendererData->currentCommandBuffer, rendererData->currentPipelineState->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
            sizeof(rendererData->vertexShaderConstantsData),
            &rendererData->vertexShaderConstantsData);
    }

    if (!shader_constants) {
        VULKAN_SetupShaderConstants(renderer, cmd, NULL, &solid_constants);
        shader_constants = &solid_constants;
    }

    constantBuffer = rendererData->constantBuffers[rendererData->currentCommandBufferIndex][rendererData->currentConstantBufferIndex].buffer;
    constantBufferOffset = (rendererData->currentConstantBufferOffset < 0) ? 0 : rendererData->currentConstantBufferOffset;
    if (updateConstants ||
        SDL_memcmp(shader_constants, &rendererData->currentPipelineState->shader_constants, sizeof(*shader_constants)) != 0) {

        if (rendererData->currentConstantBufferOffset == -1) {
            // First time, grab offset 0
            rendererData->currentConstantBufferOffset = 0;
            constantBufferOffset = 0;
        }
        else {
            // Align the next address to the minUniformBufferOffsetAlignment
            VkDeviceSize alignment = rendererData->physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
            SDL_assert(rendererData->currentConstantBufferOffset >= 0 );
            rendererData->currentConstantBufferOffset += (int32_t)(sizeof(VULKAN_PixelShaderConstants) + alignment - 1) & ~(alignment - 1);
            constantBufferOffset = rendererData->currentConstantBufferOffset;
        }

        // If we have run out of size in this constant buffer, create another if needed
        if (rendererData->currentConstantBufferOffset >= SDL_VULKAN_CONSTANT_BUFFER_DEFAULT_SIZE) {
            uint32_t newConstantBufferIndex = (rendererData->currentConstantBufferIndex + 1);
            // We need a new constant buffer
            if (newConstantBufferIndex >= rendererData->numConstantBuffers[rendererData->currentCommandBufferIndex]) {
                VULKAN_Buffer newConstantBuffer;
                VkResult result = VULKAN_AllocateBuffer(rendererData,
                    SDL_VULKAN_CONSTANT_BUFFER_DEFAULT_SIZE,
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    &newConstantBuffer);
                if (result != VK_SUCCESS) {
                    return false;
                }

                rendererData->numConstantBuffers[rendererData->currentCommandBufferIndex]++;
                VULKAN_Buffer *newConstantBuffers = (VULKAN_Buffer *)SDL_realloc(rendererData->constantBuffers[rendererData->currentCommandBufferIndex],
                                                                sizeof(VULKAN_Buffer) * rendererData->numConstantBuffers[rendererData->currentCommandBufferIndex]);
                newConstantBuffers[rendererData->numConstantBuffers[rendererData->currentCommandBufferIndex] - 1] = newConstantBuffer;
                rendererData->constantBuffers[rendererData->currentCommandBufferIndex] = newConstantBuffers;
            }
            rendererData->currentConstantBufferIndex = newConstantBufferIndex;
            rendererData->currentConstantBufferOffset = 0;
            constantBufferOffset = 0;
            constantBuffer = rendererData->constantBuffers[rendererData->currentCommandBufferIndex][rendererData->currentConstantBufferIndex].buffer;
        }

        SDL_memcpy(&rendererData->currentPipelineState->shader_constants, shader_constants, sizeof(*shader_constants));

        // Upload constants to persistently mapped buffer
        uint8_t *dst = (uint8_t *)rendererData->constantBuffers[rendererData->currentCommandBufferIndex][rendererData->currentConstantBufferIndex].mappedBufferPtr;
        dst += constantBufferOffset;
        SDL_memcpy(dst, &rendererData->currentPipelineState->shader_constants, sizeof(VULKAN_PixelShaderConstants));
    }

    // Allocate/update descriptor set with the bindings
    descriptorSet = VULKAN_AllocateDescriptorSet(renderer, shader, descriptorSetLayout, sampler, constantBuffer, constantBufferOffset, imageView);
    if (descriptorSet == VK_NULL_HANDLE) {
        return false;
    }

    // Bind the descriptor set with the sampler/UBO/image views
    vkCmdBindDescriptorSets(rendererData->currentCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rendererData->currentPipelineState->pipelineLayout,
            0, 1, &descriptorSet, 0, NULL);

    return true;
}

static VkSampler VULKAN_GetSampler(VULKAN_RenderData *data, SDL_ScaleMode scale_mode, SDL_TextureAddressMode address_u, SDL_TextureAddressMode address_v)
{
    Uint32 key = RENDER_SAMPLER_HASHKEY(scale_mode, address_u, address_v);
    SDL_assert(key < SDL_arraysize(data->samplers));
    if (!data->samplers[key]) {
        VkSamplerCreateInfo samplerCreateInfo = { 0 };
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 1000.0f;
        switch (scale_mode) {
        case SDL_SCALEMODE_NEAREST:
            samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
            samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
            break;
        case SDL_SCALEMODE_PIXELART:    // Uses linear sampling
        case SDL_SCALEMODE_LINEAR:
            samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
            samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
            break;
        default:
            SDL_SetError("Unknown scale mode: %d", scale_mode);
            return VK_NULL_HANDLE;
        }
        VkResult result = vkCreateSampler(data->device, &samplerCreateInfo, NULL, &data->samplers[key]);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkCreateSampler()", result);
            return false;
        }
    }
    return data->samplers[key];
}

static bool VULKAN_SetCopyState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const Float4X4 *matrix, VULKAN_DrawStateCache *stateCache)
{
    SDL_Texture *texture = cmd->data.draw.texture;
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VULKAN_TextureData *textureData = (VULKAN_TextureData *)texture->internal;
    VkSampler textureSampler = VK_NULL_HANDLE;
    VULKAN_PixelShaderConstants constants;
    VkDescriptorSetLayout descriptorSetLayout = (textureData->descriptorSetLayoutYcbcr != VK_NULL_HANDLE) ? textureData->descriptorSetLayoutYcbcr : rendererData->descriptorSetLayout;
    VkPipelineLayout pipelineLayout = (textureData->pipelineLayoutYcbcr != VK_NULL_HANDLE) ? textureData->pipelineLayoutYcbcr : rendererData->pipelineLayout;

    VULKAN_SetupShaderConstants(renderer, cmd, texture, &constants);

    textureSampler = VULKAN_GetSampler(rendererData, cmd->data.draw.texture_scale_mode, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);
    if (textureSampler == VK_NULL_HANDLE) {
        return false;
    }

    if (textureData->mainImage.imageLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        bool stoppedRenderPass = false;
        if (rendererData->currentRenderPass != VK_NULL_HANDLE) {
            vkCmdEndRenderPass(rendererData->currentCommandBuffer);
            rendererData->currentRenderPass = VK_NULL_HANDLE;
            stoppedRenderPass = true;
        }

        VULKAN_RecordPipelineImageBarrier(rendererData,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            textureData->mainImage.image,
            &textureData->mainImage.imageLayout);

        if (stoppedRenderPass) {
            VULKAN_BeginRenderPass(rendererData, VK_ATTACHMENT_LOAD_OP_LOAD, NULL);
        }
    }

    return VULKAN_SetDrawState(renderer, cmd, textureData->shader, pipelineLayout, descriptorSetLayout, &constants, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, textureData->mainImage.imageView, textureSampler, matrix, stateCache);
}

static void VULKAN_DrawPrimitives(SDL_Renderer *renderer, VkPrimitiveTopology primitiveTopology, const size_t vertexStart, const size_t vertexCount)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    vkCmdDraw(rendererData->currentCommandBuffer, (uint32_t)vertexCount, 1, (uint32_t)vertexStart, 0);
}

static void VULKAN_InvalidateCachedState(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    rendererData->currentPipelineState = NULL;
    rendererData->cliprectDirty = true;
}

static bool VULKAN_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VkSurfaceTransformFlagBitsKHR currentRotation = VULKAN_GetRotationForCurrentRenderTarget(rendererData);
    VULKAN_DrawStateCache stateCache;
    SDL_memset(&stateCache, 0, sizeof(stateCache));

    if (!rendererData->device) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    if(rendererData->currentViewportRotation != currentRotation) {
        rendererData->currentViewportRotation = currentRotation;
        rendererData->viewportDirty = true;
        rendererData->cliprectDirty = true;
    }

    if (rendererData->recreateSwapchain) {
        if (VULKAN_UpdateForWindowSizeChange(renderer) != VK_SUCCESS) {
            return false;
        }
        rendererData->recreateSwapchain = false;
    }

    if (!VULKAN_UpdateVertexBuffer(renderer, vertices, vertsize, &stateCache)) {
        return false;
    }

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            break; // this isn't currently used in this render backend.
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &rendererData->currentViewport;
            if (SDL_memcmp(viewport, &cmd->data.viewport.rect, sizeof(cmd->data.viewport.rect)) != 0) {
                SDL_copyp(viewport, &cmd->data.viewport.rect);
                rendererData->viewportDirty = true;
                rendererData->cliprectDirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            if (rendererData->currentCliprectEnabled != cmd->data.cliprect.enabled) {
                rendererData->currentCliprectEnabled = cmd->data.cliprect.enabled;
                rendererData->cliprectDirty = true;
            }
            if (SDL_memcmp(&rendererData->currentCliprect, rect, sizeof(*rect)) != 0) {
                SDL_copyp(&rendererData->currentCliprect, rect);
                rendererData->cliprectDirty = true;
            }
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            bool convert_color = SDL_RenderingLinearSpace(renderer);
            SDL_FColor color = cmd->data.color.color;
            if (convert_color) {
                SDL_ConvertToLinear(&color);
            }
            color.r *= cmd->data.color.color_scale;
            color.g *= cmd->data.color.color_scale;
            color.b *= cmd->data.color.color_scale;

            VkClearColorValue clearColor;
            clearColor.float32[0] = color.r;
            clearColor.float32[1] = color.g;
            clearColor.float32[2] = color.b;
            clearColor.float32[3] = color.a;
            VULKAN_ActivateCommandBuffer(renderer, VK_ATTACHMENT_LOAD_OP_CLEAR, &clearColor, &stateCache);
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(VULKAN_VertexPositionColor);
            VULKAN_SetDrawState(renderer, cmd, SHADER_SOLID, rendererData->pipelineLayout, rendererData->descriptorSetLayout, NULL, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_NULL_HANDLE, VK_NULL_HANDLE, NULL, &stateCache);
            VULKAN_DrawPrimitives(renderer, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, start, count);
            break;
        }

        case SDL_RENDERCMD_DRAW_LINES:
        {
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(VULKAN_VertexPositionColor);
            const VULKAN_VertexPositionColor *verts = (VULKAN_VertexPositionColor *)(((Uint8 *)vertices) + first);
            VULKAN_SetDrawState(renderer, cmd, SHADER_SOLID, rendererData->pipelineLayout, rendererData->descriptorSetLayout, NULL, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, VK_NULL_HANDLE, VK_NULL_HANDLE, NULL, &stateCache);
            VULKAN_DrawPrimitives(renderer, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, start, count);
            if (verts[0].pos[0] != verts[count - 1].pos[0] || verts[0].pos[1] != verts[count - 1].pos[1]) {
                VULKAN_SetDrawState(renderer, cmd, SHADER_SOLID, rendererData->pipelineLayout, rendererData->descriptorSetLayout, NULL, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_NULL_HANDLE, VK_NULL_HANDLE, NULL, &stateCache);
                VULKAN_DrawPrimitives(renderer, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, start + (count - 1), 1);
            }
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS: // unused
            break;

        case SDL_RENDERCMD_COPY: // unused
            break;

        case SDL_RENDERCMD_COPY_EX: // unused
            break;

        case SDL_RENDERCMD_GEOMETRY:
        {
            SDL_Texture *texture = cmd->data.draw.texture;
            const size_t count = cmd->data.draw.count;
            const size_t first = cmd->data.draw.first;
            const size_t start = first / sizeof(VULKAN_VertexPositionColor);

            if (texture) {
                VULKAN_SetCopyState(renderer, cmd, NULL, &stateCache);
            } else {
                VULKAN_SetDrawState(renderer, cmd, SHADER_SOLID, rendererData->pipelineLayout, rendererData->descriptorSetLayout, NULL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_NULL_HANDLE, VK_NULL_HANDLE, NULL, &stateCache);
            }

            VULKAN_DrawPrimitives(renderer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, start, count);
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }
    return true;
}

static SDL_Surface* VULKAN_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VkImage backBuffer;
    VkImageLayout *imageLayout;
    VULKAN_Buffer readbackBuffer;
    VkDeviceSize pixelSize;
    VkDeviceSize length;
    VkDeviceSize readbackBufferSize;
    VkFormat vkFormat;
    SDL_Surface *output;

    VULKAN_EnsureCommandBuffer(rendererData);

    // Stop any outstanding renderpass if open
    if (rendererData->currentRenderPass != VK_NULL_HANDLE) {
        vkCmdEndRenderPass(rendererData->currentCommandBuffer);
        rendererData->currentRenderPass = VK_NULL_HANDLE;
    }

    if (rendererData->textureRenderTarget) {
        backBuffer = rendererData->textureRenderTarget->mainImage.image;
        imageLayout = &rendererData->textureRenderTarget->mainImage.imageLayout;
        vkFormat = rendererData->textureRenderTarget->mainImage.format;
    } else {
        backBuffer = rendererData->swapchainImages[rendererData->currentSwapchainImageIndex];
        imageLayout = &rendererData->swapchainImageLayouts[rendererData->currentSwapchainImageIndex];
        vkFormat = rendererData->surfaceFormat.format;
    }

    pixelSize = VULKAN_GetBytesPerPixel(vkFormat);
    length = rect->w * pixelSize;
    readbackBufferSize = length * rect->h;
    if (VULKAN_AllocateBuffer(rendererData, readbackBufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &readbackBuffer) != VK_SUCCESS) {
        return NULL;
    }


    // Make sure the source is in the correct resource state
    VULKAN_RecordPipelineImageBarrier(rendererData,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        backBuffer,
        imageLayout);

    // Copy the image to the readback buffer
    VkBufferImageCopy region;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageOffset.x = rect->x;
    region.imageOffset.y = rect->y;
    region.imageOffset.z = 0;
    region.imageExtent.width = rect->w;
    region.imageExtent.height = rect->h;
    region.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(rendererData->currentCommandBuffer, backBuffer, *imageLayout, readbackBuffer.buffer, 1, &region);

    // We need to issue the command list for the copy to finish
    VULKAN_IssueBatch(rendererData);

    // Transition the render target back to a render target
     VULKAN_RecordPipelineImageBarrier(rendererData,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        backBuffer,
        imageLayout);

    output = SDL_DuplicatePixels(
        rect->w, rect->h,
        VULKAN_VkFormatToSDLPixelFormat(vkFormat),
        renderer->target ? renderer->target->colorspace : renderer->output_colorspace,
        readbackBuffer.mappedBufferPtr,
        (int)length);

    VULKAN_DestroyBuffer(rendererData, &readbackBuffer);

    return output;
}

static bool VULKAN_AddVulkanRenderSemaphores(SDL_Renderer *renderer, Uint32 wait_stage_mask, Sint64 wait_semaphore, Sint64 signal_semaphore)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;

    if (wait_semaphore) {
        if (rendererData->waitRenderSemaphoreCount == rendererData->waitRenderSemaphoreMax) {
            // Allocate an additional one at the end for the normal present wait
            VkPipelineStageFlags *waitDestStageMasks = (VkPipelineStageFlags *)SDL_realloc(rendererData->waitDestStageMasks, (rendererData->waitRenderSemaphoreMax + 2) * sizeof(*waitDestStageMasks));
            if (!waitDestStageMasks) {
                return false;
            }
            rendererData->waitDestStageMasks = waitDestStageMasks;

            VkSemaphore *semaphores = (VkSemaphore *)SDL_realloc(rendererData->waitRenderSemaphores, (rendererData->waitRenderSemaphoreMax + 2) * sizeof(*semaphores));
            if (!semaphores) {
                return false;
            }
            rendererData->waitRenderSemaphores = semaphores;
            ++rendererData->waitRenderSemaphoreMax;
        }
        rendererData->waitDestStageMasks[rendererData->waitRenderSemaphoreCount] = wait_stage_mask;
        rendererData->waitRenderSemaphores[rendererData->waitRenderSemaphoreCount] = (VkSemaphore)wait_semaphore;
        ++rendererData->waitRenderSemaphoreCount;
    }

    if (signal_semaphore) {
        if (rendererData->signalRenderSemaphoreCount == rendererData->signalRenderSemaphoreMax) {
            // Allocate an additional one at the end for the normal present signal
            VkSemaphore *semaphores = (VkSemaphore *)SDL_realloc(rendererData->signalRenderSemaphores, (rendererData->signalRenderSemaphoreMax + 2) * sizeof(*semaphores));
            if (!semaphores) {
                return false;
            }
            rendererData->signalRenderSemaphores = semaphores;
            ++rendererData->signalRenderSemaphoreMax;
        }
        rendererData->signalRenderSemaphores[rendererData->signalRenderSemaphoreCount] = (VkSemaphore)signal_semaphore;
        ++rendererData->signalRenderSemaphoreCount;
    }

    return true;
}

static bool VULKAN_RenderPresent(SDL_Renderer *renderer)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;
    VkResult result = VK_SUCCESS;

    if (!rendererData->device) {
        return SDL_SetError("Device lost and couldn't be recovered");
    }

    if (rendererData->currentCommandBuffer) {
        rendererData->currentPipelineState = VK_NULL_HANDLE;
        rendererData->viewportDirty = true;

        VULKAN_RecordPipelineImageBarrier(rendererData,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            rendererData->swapchainImages[rendererData->currentSwapchainImageIndex],
            &rendererData->swapchainImageLayouts[rendererData->currentSwapchainImageIndex]);

        vkEndCommandBuffer(rendererData->currentCommandBuffer);

        result = vkResetFences(rendererData->device, 1, &rendererData->fences[rendererData->currentCommandBufferIndex]);
        if (result != VK_SUCCESS) {
            SET_ERROR_CODE("vkResetFences()", result);
            return false;
        }

        VkPipelineStageFlags waitDestStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submitInfo = { 0 };
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        if (rendererData->waitRenderSemaphoreCount > 0) {
            Uint32 additionalSemaphoreCount = (rendererData->currentImageAvailableSemaphore != VK_NULL_HANDLE) ? 1 : 0;
            submitInfo.waitSemaphoreCount = rendererData->waitRenderSemaphoreCount + additionalSemaphoreCount;
            if (additionalSemaphoreCount > 0) {
                rendererData->waitRenderSemaphores[rendererData->waitRenderSemaphoreCount] = rendererData->currentImageAvailableSemaphore;
                rendererData->waitDestStageMasks[rendererData->waitRenderSemaphoreCount] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            }
            submitInfo.pWaitSemaphores = rendererData->waitRenderSemaphores;
            submitInfo.pWaitDstStageMask = rendererData->waitDestStageMasks;
            rendererData->waitRenderSemaphoreCount = 0;
        } else if (rendererData->currentImageAvailableSemaphore != VK_NULL_HANDLE) {
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &rendererData->currentImageAvailableSemaphore;
            submitInfo.pWaitDstStageMask = &waitDestStageMask;
        }
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &rendererData->currentCommandBuffer;
        if (rendererData->signalRenderSemaphoreCount > 0) {
            submitInfo.signalSemaphoreCount = rendererData->signalRenderSemaphoreCount + 1;
            rendererData->signalRenderSemaphores[rendererData->signalRenderSemaphoreCount] = rendererData->renderingFinishedSemaphores[rendererData->currentCommandBufferIndex];
            submitInfo.pSignalSemaphores = rendererData->signalRenderSemaphores;
            rendererData->signalRenderSemaphoreCount = 0;
        } else {
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &rendererData->renderingFinishedSemaphores[rendererData->currentCommandBufferIndex];
        }
        result = vkQueueSubmit(rendererData->graphicsQueue, 1, &submitInfo, rendererData->fences[rendererData->currentCommandBufferIndex]);
        if (result != VK_SUCCESS) {
            if (result == VK_ERROR_DEVICE_LOST) {
                if (VULKAN_HandleDeviceLost(renderer)) {
                    SDL_SetError("Present failed, device lost");
                } else {
                    // Recovering from device lost failed, error is already set
                }
            } else {
                SET_ERROR_CODE("vkQueueSubmit()", result);
            }
            return false;
        }
        rendererData->currentCommandBuffer = VK_NULL_HANDLE;
        rendererData->currentImageAvailableSemaphore = VK_NULL_HANDLE;

        VkPresentInfoKHR presentInfo = { 0 };
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &rendererData->renderingFinishedSemaphores[rendererData->currentCommandBufferIndex];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &rendererData->swapchain;
        presentInfo.pImageIndices = &rendererData->currentSwapchainImageIndex;
        result = vkQueuePresentKHR(rendererData->presentQueue, &presentInfo);
        if ((result != VK_SUCCESS) && (result != VK_ERROR_OUT_OF_DATE_KHR) && (result != VK_ERROR_SURFACE_LOST_KHR) && (result != VK_SUBOPTIMAL_KHR )) {
            SET_ERROR_CODE("vkQueuePresentKHR()", result);
            return false;
        }

        rendererData->currentCommandBufferIndex = ( rendererData->currentCommandBufferIndex + 1 ) % rendererData->swapchainImageCount;

        // Wait for previous time this command buffer was submitted, will be N frames ago
        result = vkWaitForFences(rendererData->device, 1, &rendererData->fences[rendererData->currentCommandBufferIndex], VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) {
            if (result == VK_ERROR_DEVICE_LOST) {
                if (VULKAN_HandleDeviceLost(renderer)) {
                    SDL_SetError("Present failed, device lost");
                } else {
                    // Recovering from device lost failed, error is already set
                }
            } else {
                SET_ERROR_CODE("vkWaitForFences()", result);
            }
            return false;
        }

        VULKAN_AcquireNextSwapchainImage(renderer);
    }

    return true;
}

static bool VULKAN_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    VULKAN_RenderData *rendererData = (VULKAN_RenderData *)renderer->internal;

    switch (vsync) {
    case -1:
    case 0:
    case 1:
        // Supported
        break;
    default:
        return SDL_Unsupported();
    }
    if (vsync != rendererData->vsync) {
        rendererData->vsync = vsync;
        rendererData->recreateSwapchain = true;
    }
    return true;
}

static bool VULKAN_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    VULKAN_RenderData *rendererData;

    SDL_SetupRendererColorspace(renderer, create_props);

    if (renderer->output_colorspace != SDL_COLORSPACE_SRGB &&
        renderer->output_colorspace != SDL_COLORSPACE_SRGB_LINEAR
        /*&& renderer->output_colorspace != SDL_COLORSPACE_HDR10*/) {
        return SDL_SetError("Unsupported output colorspace");
    }

    rendererData = (VULKAN_RenderData *)SDL_calloc(1, sizeof(*rendererData));
    if (!rendererData) {
        return false;
    }

    rendererData->identity = MatrixIdentity();
    rendererData->identitySwizzle.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    rendererData->identitySwizzle.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    rendererData->identitySwizzle.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    rendererData->identitySwizzle.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    // Save the create props in case we need to recreate on device lost
    rendererData->create_props = SDL_CreateProperties();
    if (!SDL_CopyProperties(create_props, rendererData->create_props)) {
        SDL_free(rendererData);
        return false;
    }

    renderer->WindowEvent = VULKAN_WindowEvent;
    renderer->SupportsBlendMode = VULKAN_SupportsBlendMode;
    renderer->CreateTexture = VULKAN_CreateTexture;
    renderer->UpdateTexture = VULKAN_UpdateTexture;
#ifdef SDL_HAVE_YUV
    renderer->UpdateTextureYUV = VULKAN_UpdateTextureYUV;
    renderer->UpdateTextureNV = VULKAN_UpdateTextureNV;
#endif
    renderer->LockTexture = VULKAN_LockTexture;
    renderer->UnlockTexture = VULKAN_UnlockTexture;
    renderer->SetRenderTarget = VULKAN_SetRenderTarget;
    renderer->QueueSetViewport = VULKAN_QueueNoOp;
    renderer->QueueSetDrawColor = VULKAN_QueueNoOp;
    renderer->QueueDrawPoints = VULKAN_QueueDrawPoints;
    renderer->QueueDrawLines = VULKAN_QueueDrawPoints; // lines and points queue vertices the same way.
    renderer->QueueGeometry = VULKAN_QueueGeometry;
    renderer->InvalidateCachedState = VULKAN_InvalidateCachedState;
    renderer->RunCommandQueue = VULKAN_RunCommandQueue;
    renderer->RenderReadPixels = VULKAN_RenderReadPixels;
    renderer->AddVulkanRenderSemaphores = VULKAN_AddVulkanRenderSemaphores;
    renderer->RenderPresent = VULKAN_RenderPresent;
    renderer->DestroyTexture = VULKAN_DestroyTexture;
    renderer->DestroyRenderer = VULKAN_DestroyRenderer;
    renderer->SetVSync = VULKAN_SetVSync;
    renderer->internal = rendererData;
    VULKAN_InvalidateCachedState(renderer);

    renderer->name = VULKAN_RenderDriver.name;
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR8888);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR2101010);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBA64_FLOAT);
    SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 16384);

    /* HACK: make sure the SDL_Renderer references the SDL_Window data now, in
     * order to give init functions access to the underlying window handle:
     */
    renderer->window = window;

    // Initialize Vulkan resources
    if (VULKAN_CreateDeviceResources(renderer, create_props) != VK_SUCCESS) {
        return false;
    }

    if (VULKAN_CreateWindowSizeDependentResources(renderer) != VK_SUCCESS) {
        return false;
    }

#ifdef SDL_HAVE_YUV
    if (rendererData->supportsKHRSamplerYCbCrConversion) {
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_YV12);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_IYUV);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV12);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV21);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_P010);
    }
#endif

    return true;
}

SDL_RenderDriver VULKAN_RenderDriver = {
    VULKAN_CreateRenderer, "vulkan"
};

#endif // SDL_VIDEO_RENDER_VULKAN

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
#include "SDL_sysgpu.h"

// FIXME: This could probably use SDL_ObjectValid
#define CHECK_DEVICE_MAGIC(device, retval)  \
    if (device == NULL) {                   \
        SDL_SetError("Invalid GPU device"); \
        return retval;                      \
    }

#define CHECK_COMMAND_BUFFER                                        \
    if (((CommandBufferCommonHeader *)command_buffer)->submitted) { \
        SDL_assert_release(!"Command buffer already submitted!");   \
        return;                                                     \
    }

#define CHECK_COMMAND_BUFFER_RETURN_FALSE                           \
    if (((CommandBufferCommonHeader *)command_buffer)->submitted) { \
        SDL_assert_release(!"Command buffer already submitted!");   \
        return false;                                               \
    }

#define CHECK_COMMAND_BUFFER_RETURN_NULL                            \
    if (((CommandBufferCommonHeader *)command_buffer)->submitted) { \
        SDL_assert_release(!"Command buffer already submitted!");   \
        return NULL;                                                \
    }

#define CHECK_ANY_PASS_IN_PROGRESS(msg, retval)                                 \
    if (                                                                        \
        ((CommandBufferCommonHeader *)command_buffer)->render_pass.in_progress ||  \
        ((CommandBufferCommonHeader *)command_buffer)->compute_pass.in_progress || \
        ((CommandBufferCommonHeader *)command_buffer)->copy_pass.in_progress) {    \
        SDL_assert_release(!msg);                                               \
        return retval;                                                          \
    }

#define CHECK_RENDERPASS                                     \
    if (!((RenderPass *)render_pass)->in_progress) {                 \
        SDL_assert_release(!"Render pass not in progress!"); \
        return;                                              \
    }

#define CHECK_SAMPLER_TEXTURES                                                                                                          \
    RenderPass *rp = (RenderPass *)render_pass;                                                                                         \
    for (Uint32 color_target_index = 0; color_target_index < rp->num_color_targets; color_target_index += 1) {                          \
        for (Uint32 texture_sampler_index = 0; texture_sampler_index < num_bindings; texture_sampler_index += 1) {                      \
            if (rp->color_targets[color_target_index] == texture_sampler_bindings[texture_sampler_index].texture) {                     \
                SDL_assert_release(!"Texture cannot be simultaneously bound as a color target and a sampler!");                         \
            }                                                                                                                           \
        }                                                                                                                               \
    }                                                                                                                                   \
                                                                                                                                        \
    for (Uint32 texture_sampler_index = 0; texture_sampler_index < num_bindings; texture_sampler_index += 1) {                          \
        if (rp->depth_stencil_target != NULL && rp->depth_stencil_target == texture_sampler_bindings[texture_sampler_index].texture) {  \
            SDL_assert_release(!"Texture cannot be simultaneously bound as a depth stencil target and a sampler!");                     \
        }                                                                                                                               \
    }

#define CHECK_STORAGE_TEXTURES                                                                                              \
    RenderPass *rp = (RenderPass *)render_pass;                                                                             \
    for (Uint32 color_target_index = 0; color_target_index < rp->num_color_targets; color_target_index += 1) {              \
        for (Uint32 texture_sampler_index = 0; texture_sampler_index < num_bindings; texture_sampler_index += 1) {          \
            if (rp->color_targets[color_target_index] == storage_textures[texture_sampler_index]) {                         \
                SDL_assert_release(!"Texture cannot be simultaneously bound as a color target and a storage texture!");     \
            }                                                                                                               \
        }                                                                                                                   \
    }                                                                                                                       \
                                                                                                                            \
    for (Uint32 texture_sampler_index = 0; texture_sampler_index < num_bindings; texture_sampler_index += 1) {              \
        if (rp->depth_stencil_target != NULL && rp->depth_stencil_target == storage_textures[texture_sampler_index]) {      \
            SDL_assert_release(!"Texture cannot be simultaneously bound as a depth stencil target and a storage texture!"); \
        }                                                                                                                   \
    }

#define CHECK_GRAPHICS_PIPELINE_BOUND                                                       \
    if (!((CommandBufferCommonHeader *)RENDERPASS_COMMAND_BUFFER)->graphics_pipeline_bound) { \
        SDL_assert_release(!"Graphics pipeline not bound!");                                \
        return;                                                                             \
    }

#define CHECK_COMPUTEPASS                                     \
    if (!((Pass *)compute_pass)->in_progress) {                 \
        SDL_assert_release(!"Compute pass not in progress!"); \
        return;                                               \
    }

#define CHECK_COMPUTE_PIPELINE_BOUND                                                        \
    if (!((CommandBufferCommonHeader *)COMPUTEPASS_COMMAND_BUFFER)->compute_pipeline_bound) { \
        SDL_assert_release(!"Compute pipeline not bound!");                                 \
        return;                                                                             \
    }

#define CHECK_COPYPASS                                     \
    if (!((Pass *)copy_pass)->in_progress) {                 \
        SDL_assert_release(!"Copy pass not in progress!"); \
        return;                                            \
    }

#define CHECK_TEXTUREFORMAT_ENUM_INVALID(enumval, retval)     \
    if (enumval <= SDL_GPU_TEXTUREFORMAT_INVALID || enumval >= SDL_GPU_TEXTUREFORMAT_MAX_ENUM_VALUE) {               \
        SDL_assert_release(!"Invalid texture format enum!"); \
        return retval;                                       \
    }

#define CHECK_VERTEXELEMENTFORMAT_ENUM_INVALID(enumval, retval)       \
    if (enumval <= SDL_GPU_VERTEXELEMENTFORMAT_INVALID || enumval >= SDL_GPU_VERTEXELEMENTFORMAT_MAX_ENUM_VALUE) {  \
        SDL_assert_release(!"Invalid vertex format enum!");          \
        return retval;                                               \
    }

#define CHECK_COMPAREOP_ENUM_INVALID(enumval, retval)                              \
    if (enumval <= SDL_GPU_COMPAREOP_INVALID || enumval >= SDL_GPU_COMPAREOP_MAX_ENUM_VALUE) { \
        SDL_assert_release(!"Invalid compare op enum!");                          \
        return retval;                                                            \
    }

#define CHECK_STENCILOP_ENUM_INVALID(enumval, retval)                                \
    if (enumval <= SDL_GPU_STENCILOP_INVALID || enumval >= SDL_GPU_STENCILOP_MAX_ENUM_VALUE) { \
        SDL_assert_release(!"Invalid stencil op enum!");                            \
        return retval;                                                              \
    }

#define CHECK_BLENDOP_ENUM_INVALID(enumval, retval)                              \
    if (enumval <= SDL_GPU_BLENDOP_INVALID || enumval >= SDL_GPU_BLENDOP_MAX_ENUM_VALUE) { \
        SDL_assert_release(!"Invalid blend op enum!");                          \
        return retval;                                                          \
    }

#define CHECK_BLENDFACTOR_ENUM_INVALID(enumval, retval)                                  \
    if (enumval <= SDL_GPU_BLENDFACTOR_INVALID || enumval >= SDL_GPU_BLENDFACTOR_MAX_ENUM_VALUE) { \
        SDL_assert_release(!"Invalid blend factor enum!");                              \
        return retval;                                                                  \
    }

#define CHECK_SWAPCHAINCOMPOSITION_ENUM_INVALID(enumval, retval)    \
    if (enumval < 0 || enumval >= SDL_GPU_SWAPCHAINCOMPOSITION_MAX_ENUM_VALUE) {              \
        SDL_assert_release(!"Invalid swapchain composition enum!"); \
        return retval;                                              \
    }

#define CHECK_PRESENTMODE_ENUM_INVALID(enumval, retval)    \
    if (enumval < 0 || enumval >= SDL_GPU_PRESENTMODE_MAX_ENUM_VALUE) {              \
        SDL_assert_release(!"Invalid present mode enum!"); \
        return retval;                                     \
    }

#define COMMAND_BUFFER_DEVICE \
    ((CommandBufferCommonHeader *)command_buffer)->device

#define RENDERPASS_COMMAND_BUFFER \
    ((RenderPass *)render_pass)->command_buffer

#define RENDERPASS_DEVICE \
    ((CommandBufferCommonHeader *)RENDERPASS_COMMAND_BUFFER)->device

#define COMPUTEPASS_COMMAND_BUFFER \
    ((Pass *)compute_pass)->command_buffer

#define COMPUTEPASS_DEVICE \
    ((CommandBufferCommonHeader *)COMPUTEPASS_COMMAND_BUFFER)->device

#define COPYPASS_COMMAND_BUFFER \
    ((Pass *)copy_pass)->command_buffer

#define COPYPASS_DEVICE \
    ((CommandBufferCommonHeader *)COPYPASS_COMMAND_BUFFER)->device

// Drivers

#ifndef SDL_GPU_DISABLED
static const SDL_GPUBootstrap *backends[] = {
#ifdef SDL_GPU_PRIVATE
    &PrivateGPUDriver,
#endif
#ifdef SDL_GPU_METAL
    &MetalDriver,
#endif
#ifdef SDL_GPU_VULKAN
    &VulkanDriver,
#endif
#ifdef SDL_GPU_D3D12
    &D3D12Driver,
#endif
    NULL
};
#endif // !SDL_GPU_DISABLED

// Internal Utility Functions

SDL_GPUGraphicsPipeline *SDL_GPU_FetchBlitPipeline(
    SDL_GPUDevice *device,
    SDL_GPUTextureType source_texture_type,
    SDL_GPUTextureFormat destination_format,
    SDL_GPUShader *blit_vertex_shader,
    SDL_GPUShader *blit_from_2d_shader,
    SDL_GPUShader *blit_from_2d_array_shader,
    SDL_GPUShader *blit_from_3d_shader,
    SDL_GPUShader *blit_from_cube_shader,
    SDL_GPUShader *blit_from_cube_array_shader,
    BlitPipelineCacheEntry **blit_pipelines,
    Uint32 *blit_pipeline_count,
    Uint32 *blit_pipeline_capacity)
{
    SDL_GPUGraphicsPipelineCreateInfo blit_pipeline_create_info;
    SDL_GPUColorTargetDescription color_target_desc;
    SDL_GPUGraphicsPipeline *pipeline;

    if (blit_pipeline_count == NULL) {
        // use pre-created, format-agnostic pipelines
        return (*blit_pipelines)[source_texture_type].pipeline;
    }

    for (Uint32 i = 0; i < *blit_pipeline_count; i += 1) {
        if ((*blit_pipelines)[i].type == source_texture_type && (*blit_pipelines)[i].format == destination_format) {
            return (*blit_pipelines)[i].pipeline;
        }
    }

    // No pipeline found, we'll need to make one!
    SDL_zero(blit_pipeline_create_info);

    SDL_zero(color_target_desc);
    color_target_desc.blend_state.color_write_mask = 0xF;
    color_target_desc.format = destination_format;

    blit_pipeline_create_info.target_info.color_target_descriptions = &color_target_desc;
    blit_pipeline_create_info.target_info.num_color_targets = 1;
    blit_pipeline_create_info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM; // arbitrary
    blit_pipeline_create_info.target_info.has_depth_stencil_target = false;

    blit_pipeline_create_info.vertex_shader = blit_vertex_shader;
    if (source_texture_type == SDL_GPU_TEXTURETYPE_CUBE) {
        blit_pipeline_create_info.fragment_shader = blit_from_cube_shader;
    } else if (source_texture_type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
        blit_pipeline_create_info.fragment_shader = blit_from_cube_array_shader;
    }  else if (source_texture_type == SDL_GPU_TEXTURETYPE_2D_ARRAY) {
        blit_pipeline_create_info.fragment_shader = blit_from_2d_array_shader;
    } else if (source_texture_type == SDL_GPU_TEXTURETYPE_3D) {
        blit_pipeline_create_info.fragment_shader = blit_from_3d_shader;
    } else {
        blit_pipeline_create_info.fragment_shader = blit_from_2d_shader;
    }

    blit_pipeline_create_info.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
    blit_pipeline_create_info.multisample_state.enable_mask = false;

    blit_pipeline_create_info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;

    pipeline = SDL_CreateGPUGraphicsPipeline(
        device,
        &blit_pipeline_create_info);

    if (pipeline == NULL) {
        SDL_SetError("Failed to create GPU pipeline for blit");
        return NULL;
    }

    // Cache the new pipeline
    EXPAND_ARRAY_IF_NEEDED(
        (*blit_pipelines),
        BlitPipelineCacheEntry,
        *blit_pipeline_count + 1,
        *blit_pipeline_capacity,
        *blit_pipeline_capacity * 2);

    (*blit_pipelines)[*blit_pipeline_count].pipeline = pipeline;
    (*blit_pipelines)[*blit_pipeline_count].type = source_texture_type;
    (*blit_pipelines)[*blit_pipeline_count].format = destination_format;
    *blit_pipeline_count += 1;

    return pipeline;
}

void SDL_GPU_BlitCommon(
    SDL_GPUCommandBuffer *command_buffer,
    const SDL_GPUBlitInfo *info,
    SDL_GPUSampler *blit_linear_sampler,
    SDL_GPUSampler *blit_nearest_sampler,
    SDL_GPUShader *blit_vertex_shader,
    SDL_GPUShader *blit_from_2d_shader,
    SDL_GPUShader *blit_from_2d_array_shader,
    SDL_GPUShader *blit_from_3d_shader,
    SDL_GPUShader *blit_from_cube_shader,
    SDL_GPUShader *blit_from_cube_array_shader,
    BlitPipelineCacheEntry **blit_pipelines,
    Uint32 *blit_pipeline_count,
    Uint32 *blit_pipeline_capacity)
{
    CommandBufferCommonHeader *cmdbufHeader = (CommandBufferCommonHeader *)command_buffer;
    SDL_GPURenderPass *render_pass;
    TextureCommonHeader *src_header = (TextureCommonHeader *)info->source.texture;
    TextureCommonHeader *dst_header = (TextureCommonHeader *)info->destination.texture;
    SDL_GPUGraphicsPipeline *blit_pipeline;
    SDL_GPUColorTargetInfo color_target_info;
    SDL_GPUViewport viewport;
    SDL_GPUTextureSamplerBinding texture_sampler_binding;
    BlitFragmentUniforms blit_fragment_uniforms;
    Uint32 layer_divisor;

    blit_pipeline = SDL_GPU_FetchBlitPipeline(
        cmdbufHeader->device,
        src_header->info.type,
        dst_header->info.format,
        blit_vertex_shader,
        blit_from_2d_shader,
        blit_from_2d_array_shader,
        blit_from_3d_shader,
        blit_from_cube_shader,
        blit_from_cube_array_shader,
        blit_pipelines,
        blit_pipeline_count,
        blit_pipeline_capacity);

    SDL_assert(blit_pipeline != NULL);

    color_target_info.load_op = info->load_op;
    color_target_info.clear_color = info->clear_color;
    color_target_info.store_op = SDL_GPU_STOREOP_STORE;

    color_target_info.texture = info->destination.texture;
    color_target_info.mip_level = info->destination.mip_level;
    color_target_info.layer_or_depth_plane = info->destination.layer_or_depth_plane;
    color_target_info.cycle = info->cycle;

    render_pass = SDL_BeginGPURenderPass(
        command_buffer,
        &color_target_info,
        1,
        NULL);

    viewport.x = (float)info->destination.x;
    viewport.y = (float)info->destination.y;
    viewport.w = (float)info->destination.w;
    viewport.h = (float)info->destination.h;
    viewport.min_depth = 0;
    viewport.max_depth = 1;

    SDL_SetGPUViewport(
        render_pass,
        &viewport);

    SDL_BindGPUGraphicsPipeline(
        render_pass,
        blit_pipeline);

    texture_sampler_binding.texture = info->source.texture;
    texture_sampler_binding.sampler =
        info->filter == SDL_GPU_FILTER_NEAREST ? blit_nearest_sampler : blit_linear_sampler;

    SDL_BindGPUFragmentSamplers(
        render_pass,
        0,
        &texture_sampler_binding,
        1);

    blit_fragment_uniforms.left = (float)info->source.x / (src_header->info.width >> info->source.mip_level);
    blit_fragment_uniforms.top = (float)info->source.y / (src_header->info.height >> info->source.mip_level);
    blit_fragment_uniforms.width = (float)info->source.w / (src_header->info.width >> info->source.mip_level);
    blit_fragment_uniforms.height = (float)info->source.h / (src_header->info.height >> info->source.mip_level);
    blit_fragment_uniforms.mip_level = info->source.mip_level;

    layer_divisor = (src_header->info.type == SDL_GPU_TEXTURETYPE_3D) ? src_header->info.layer_count_or_depth : 1;
    blit_fragment_uniforms.layer_or_depth = (float)info->source.layer_or_depth_plane / layer_divisor;

    if (info->flip_mode & SDL_FLIP_HORIZONTAL) {
        blit_fragment_uniforms.left += blit_fragment_uniforms.width;
        blit_fragment_uniforms.width *= -1;
    }

    if (info->flip_mode & SDL_FLIP_VERTICAL) {
        blit_fragment_uniforms.top += blit_fragment_uniforms.height;
        blit_fragment_uniforms.height *= -1;
    }

    SDL_PushGPUFragmentUniformData(
        command_buffer,
        0,
        &blit_fragment_uniforms,
        sizeof(blit_fragment_uniforms));

    SDL_DrawGPUPrimitives(render_pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(render_pass);
}

// Driver Functions

#ifndef SDL_GPU_DISABLED
static const SDL_GPUBootstrap * SDL_GPUSelectBackend(SDL_PropertiesID props)
{
    Uint32 i;
    SDL_GPUShaderFormat format_flags = 0;
    const char *gpudriver;
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    if (_this == NULL) {
        SDL_SetError("Video subsystem not initialized");
        return NULL;
    }

    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_PRIVATE_BOOLEAN, false)) {
        format_flags |= SDL_GPU_SHADERFORMAT_PRIVATE;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, false)) {
        format_flags |= SDL_GPU_SHADERFORMAT_SPIRV;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXBC_BOOLEAN, false)) {
        format_flags |= SDL_GPU_SHADERFORMAT_DXBC;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXIL_BOOLEAN, false)) {
        format_flags |= SDL_GPU_SHADERFORMAT_DXIL;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_MSL_BOOLEAN, false)) {
        format_flags |= SDL_GPU_SHADERFORMAT_MSL;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_METALLIB_BOOLEAN, false)) {
        format_flags |= SDL_GPU_SHADERFORMAT_METALLIB;
    }

    gpudriver = SDL_GetHint(SDL_HINT_GPU_DRIVER);
    if (gpudriver == NULL) {
        gpudriver = SDL_GetStringProperty(props, SDL_PROP_GPU_DEVICE_CREATE_NAME_STRING, NULL);
    }

    // Environment/Properties override...
    if (gpudriver != NULL) {
        for (i = 0; backends[i]; i += 1) {
            if (SDL_strcasecmp(gpudriver, backends[i]->name) == 0) {
                if (!(backends[i]->shader_formats & format_flags)) {
                    SDL_SetError("Required shader format for backend %s not provided!", gpudriver);
                    return NULL;
                }
                if (backends[i]->PrepareDriver(_this)) {
                    return backends[i];
                }
            }
        }

        SDL_SetError("SDL_HINT_GPU_DRIVER %s unsupported!", gpudriver);
        return NULL;
    }

    for (i = 0; backends[i]; i += 1) {
        if ((backends[i]->shader_formats & format_flags) == 0) {
            // Don't select a backend which doesn't support the app's shaders.
            continue;
        }
        if (backends[i]->PrepareDriver(_this)) {
            return backends[i];
        }
    }

    SDL_SetError("No supported SDL_GPU backend found!");
    return NULL;
}

static void SDL_GPU_FillProperties(
    SDL_PropertiesID props,
    SDL_GPUShaderFormat format_flags,
    bool debug_mode,
    const char *name)
{
    if (format_flags & SDL_GPU_SHADERFORMAT_PRIVATE) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_PRIVATE_BOOLEAN, true);
    }
    if (format_flags & SDL_GPU_SHADERFORMAT_SPIRV) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, true);
    }
    if (format_flags & SDL_GPU_SHADERFORMAT_DXBC) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXBC_BOOLEAN, true);
    }
    if (format_flags & SDL_GPU_SHADERFORMAT_DXIL) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXIL_BOOLEAN, true);
    }
    if (format_flags & SDL_GPU_SHADERFORMAT_MSL) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_MSL_BOOLEAN, true);
    }
    if (format_flags & SDL_GPU_SHADERFORMAT_METALLIB) {
        SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_METALLIB_BOOLEAN, true);
    }
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, debug_mode);
    SDL_SetStringProperty(props, SDL_PROP_GPU_DEVICE_CREATE_NAME_STRING, name);
}
#endif // SDL_GPU_DISABLED

bool SDL_GPUSupportsShaderFormats(
    SDL_GPUShaderFormat format_flags,
    const char *name)
{
#ifndef SDL_GPU_DISABLED
    bool result;
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_GPU_FillProperties(props, format_flags, false, name);
    result = SDL_GPUSupportsProperties(props);
    SDL_DestroyProperties(props);
    return result;
#else
    SDL_SetError("SDL not built with GPU support");
    return false;
#endif
}

bool SDL_GPUSupportsProperties(SDL_PropertiesID props)
{
#ifndef SDL_GPU_DISABLED
    return (SDL_GPUSelectBackend(props) != NULL);
#else
    SDL_SetError("SDL not built with GPU support");
    return false;
#endif
}

SDL_GPUDevice *SDL_CreateGPUDevice(
    SDL_GPUShaderFormat format_flags,
    bool debug_mode,
    const char *name)
{
#ifndef SDL_GPU_DISABLED
    SDL_GPUDevice *result;
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_GPU_FillProperties(props, format_flags, debug_mode, name);
    result = SDL_CreateGPUDeviceWithProperties(props);
    SDL_DestroyProperties(props);
    return result;
#else
    SDL_SetError("SDL not built with GPU support");
    return NULL;
#endif // SDL_GPU_DISABLED
}

SDL_GPUDevice *SDL_CreateGPUDeviceWithProperties(SDL_PropertiesID props)
{
#ifndef SDL_GPU_DISABLED
    bool debug_mode;
    bool preferLowPower;
    SDL_GPUDevice *result = NULL;
    const SDL_GPUBootstrap *selectedBackend;

    selectedBackend = SDL_GPUSelectBackend(props);
    if (selectedBackend != NULL) {
        debug_mode = SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, true);
        preferLowPower = SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN, false);

        result = selectedBackend->CreateDevice(debug_mode, preferLowPower, props);
        if (result != NULL) {
            result->backend = selectedBackend->name;
            result->shader_formats = selectedBackend->shader_formats;
            result->debug_mode = debug_mode;
        }
    }
    return result;
#else
    SDL_SetError("SDL not built with GPU support");
    return NULL;
#endif // SDL_GPU_DISABLED
}

void SDL_DestroyGPUDevice(SDL_GPUDevice *device)
{
    CHECK_DEVICE_MAGIC(device, );

    device->DestroyDevice(device);
}

int SDL_GetNumGPUDrivers(void)
{
#ifndef SDL_GPU_DISABLED
    return SDL_arraysize(backends) - 1;
#else
    return 0;
#endif
}

const char * SDL_GetGPUDriver(int index)
{
    if (index < 0 || index >= SDL_GetNumGPUDrivers()) {
        SDL_InvalidParamError("index");
        return NULL;
    }
#ifndef SDL_GPU_DISABLED
    return backends[index]->name;
#else
    return NULL;
#endif
}

const char * SDL_GetGPUDeviceDriver(SDL_GPUDevice *device)
{
    CHECK_DEVICE_MAGIC(device, NULL);

    return device->backend;
}

SDL_GPUShaderFormat SDL_GetGPUShaderFormats(SDL_GPUDevice *device)
{
    CHECK_DEVICE_MAGIC(device, SDL_GPU_SHADERFORMAT_INVALID);

    return device->shader_formats;
}

SDL_PropertiesID SDL_GetGPUDeviceProperties(SDL_GPUDevice *device)
{
    CHECK_DEVICE_MAGIC(device, 0);

    return device->GetDeviceProperties(device);
}

Uint32 SDL_GPUTextureFormatTexelBlockSize(
    SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM:
        return 8;
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
        return 16;
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
        return 1;
    case SDL_GPU_TEXTUREFORMAT_B5G6R5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B5G5R5A1_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
    case SDL_GPU_TEXTUREFORMAT_D16_UNORM:
        return 2;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32_INT:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
        return 4;
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return 5;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT:
        return 8;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
        return 16;
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_FLOAT:
        return 16;
    default:
        SDL_assert_release(!"Unrecognized TextureFormat!");
        return 0;
    }
}

bool SDL_GPUTextureSupportsFormat(
    SDL_GPUDevice *device,
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type,
    SDL_GPUTextureUsageFlags usage)
{
    CHECK_DEVICE_MAGIC(device, false);

    if (device->debug_mode) {
        CHECK_TEXTUREFORMAT_ENUM_INVALID(format, false)
    }

    return device->SupportsTextureFormat(
        device->driverData,
        format,
        type,
        usage);
}

bool SDL_GPUTextureSupportsSampleCount(
    SDL_GPUDevice *device,
    SDL_GPUTextureFormat format,
    SDL_GPUSampleCount sample_count)
{
    CHECK_DEVICE_MAGIC(device, 0);

    if (device->debug_mode) {
        CHECK_TEXTUREFORMAT_ENUM_INVALID(format, 0)
    }

    return device->SupportsSampleCount(
        device->driverData,
        format,
        sample_count);
}

// State Creation

SDL_GPUComputePipeline *SDL_CreateGPUComputePipeline(
    SDL_GPUDevice *device,
    const SDL_GPUComputePipelineCreateInfo *createinfo)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (createinfo == NULL) {
        SDL_InvalidParamError("createinfo");
        return NULL;
    }

    if (device->debug_mode) {
        if (createinfo->format == SDL_GPU_SHADERFORMAT_INVALID) {
            SDL_assert_release(!"Shader format cannot be INVALID!");
            return NULL;
        }
        if (!(createinfo->format & device->shader_formats)) {
            SDL_assert_release(!"Incompatible shader format for GPU backend");
            return NULL;
        }
        if (createinfo->num_readwrite_storage_textures > MAX_COMPUTE_WRITE_TEXTURES) {
            SDL_assert_release(!"Compute pipeline write-only texture count cannot be higher than 8!");
            return NULL;
        }
        if (createinfo->num_readwrite_storage_buffers > MAX_COMPUTE_WRITE_BUFFERS) {
            SDL_assert_release(!"Compute pipeline write-only buffer count cannot be higher than 8!");
            return NULL;
        }
        if (createinfo->threadcount_x == 0 ||
            createinfo->threadcount_y == 0 ||
            createinfo->threadcount_z == 0) {
            SDL_assert_release(!"Compute pipeline threadCount dimensions must be at least 1!");
            return NULL;
        }
    }

    return device->CreateComputePipeline(
        device->driverData,
        createinfo);
}

SDL_GPUGraphicsPipeline *SDL_CreateGPUGraphicsPipeline(
    SDL_GPUDevice *device,
    const SDL_GPUGraphicsPipelineCreateInfo *graphicsPipelineCreateInfo)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (graphicsPipelineCreateInfo == NULL) {
        SDL_InvalidParamError("graphicsPipelineCreateInfo");
        return NULL;
    }

    if (device->debug_mode) {
        if (graphicsPipelineCreateInfo->vertex_shader == NULL) {
            SDL_assert_release(!"Vertex shader cannot be NULL!");
            return NULL;
        }
        if (graphicsPipelineCreateInfo->fragment_shader == NULL) {
            SDL_assert_release(!"Fragment shader cannot be NULL!");
            return NULL;
        }
        if (graphicsPipelineCreateInfo->target_info.num_color_targets > 0 && graphicsPipelineCreateInfo->target_info.color_target_descriptions == NULL) {
            SDL_assert_release(!"Color target descriptions array pointer cannot be NULL if num_color_targets is greater than zero!");
            return NULL;
        }
        for (Uint32 i = 0; i < graphicsPipelineCreateInfo->target_info.num_color_targets; i += 1) {
            CHECK_TEXTUREFORMAT_ENUM_INVALID(graphicsPipelineCreateInfo->target_info.color_target_descriptions[i].format, NULL);
            if (IsDepthFormat(graphicsPipelineCreateInfo->target_info.color_target_descriptions[i].format)) {
                SDL_assert_release(!"Color target formats cannot be a depth format!");
                return NULL;
            }
            if (!SDL_GPUTextureSupportsFormat(device, graphicsPipelineCreateInfo->target_info.color_target_descriptions[i].format, SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
                SDL_assert_release(!"Format is not supported for color targets on this device!");
                return NULL;
            }
            if (graphicsPipelineCreateInfo->target_info.color_target_descriptions[i].blend_state.enable_blend) {
                const SDL_GPUColorTargetBlendState *blend_state = &graphicsPipelineCreateInfo->target_info.color_target_descriptions[i].blend_state;
                CHECK_BLENDFACTOR_ENUM_INVALID(blend_state->src_color_blendfactor, NULL)
                CHECK_BLENDFACTOR_ENUM_INVALID(blend_state->dst_color_blendfactor, NULL)
                CHECK_BLENDOP_ENUM_INVALID(blend_state->color_blend_op, NULL)
                CHECK_BLENDFACTOR_ENUM_INVALID(blend_state->src_alpha_blendfactor, NULL)
                CHECK_BLENDFACTOR_ENUM_INVALID(blend_state->dst_alpha_blendfactor, NULL)
                CHECK_BLENDOP_ENUM_INVALID(blend_state->alpha_blend_op, NULL)

                // TODO: validate that format support blending?
            }
        }
        if (graphicsPipelineCreateInfo->target_info.has_depth_stencil_target) {
            CHECK_TEXTUREFORMAT_ENUM_INVALID(graphicsPipelineCreateInfo->target_info.depth_stencil_format, NULL);
            if (!IsDepthFormat(graphicsPipelineCreateInfo->target_info.depth_stencil_format)) {
                SDL_assert_release(!"Depth-stencil target format must be a depth format!");
                return NULL;
            }
            if (!SDL_GPUTextureSupportsFormat(device, graphicsPipelineCreateInfo->target_info.depth_stencil_format, SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
                SDL_assert_release(!"Format is not supported for depth targets on this device!");
                return NULL;
            }
        }
        if (graphicsPipelineCreateInfo->multisample_state.enable_alpha_to_coverage) {
            if (graphicsPipelineCreateInfo->target_info.num_color_targets < 1) {
                SDL_assert_release(!"Alpha-to-coverage enabled but no color targets present!");
                return NULL;
            }
            if (!FormatHasAlpha(graphicsPipelineCreateInfo->target_info.color_target_descriptions[0].format)) {
                SDL_assert_release(!"Format is not compatible with alpha-to-coverage!");
                return NULL;
            }

            // TODO: validate that format supports belnding? This is only required on Metal.
        }
        if (graphicsPipelineCreateInfo->vertex_input_state.num_vertex_buffers > 0 && graphicsPipelineCreateInfo->vertex_input_state.vertex_buffer_descriptions == NULL) {
            SDL_assert_release(!"Vertex buffer descriptions array pointer cannot be NULL!");
            return NULL;
        }
        if (graphicsPipelineCreateInfo->vertex_input_state.num_vertex_buffers > MAX_VERTEX_BUFFERS) {
            SDL_assert_release(!"The number of vertex buffer descriptions in a vertex input state must not exceed 16!");
            return NULL;
        }
        if (graphicsPipelineCreateInfo->vertex_input_state.num_vertex_attributes > 0 && graphicsPipelineCreateInfo->vertex_input_state.vertex_attributes == NULL) {
            SDL_assert_release(!"Vertex attributes array pointer cannot be NULL!");
            return NULL;
        }
        if (graphicsPipelineCreateInfo->vertex_input_state.num_vertex_attributes > MAX_VERTEX_ATTRIBUTES) {
            SDL_assert_release(!"The number of vertex attributes in a vertex input state must not exceed 16!");
            return NULL;
        }
        for (Uint32 i = 0; i < graphicsPipelineCreateInfo->vertex_input_state.num_vertex_buffers; i += 1) {
            if (graphicsPipelineCreateInfo->vertex_input_state.vertex_buffer_descriptions[i].instance_step_rate != 0) {
                SDL_assert_release(!"For all vertex buffer descriptions, instance_step_rate must be 0!");
                return NULL;
            }
        }
        Uint32 locations[MAX_VERTEX_ATTRIBUTES];
        for (Uint32 i = 0; i < graphicsPipelineCreateInfo->vertex_input_state.num_vertex_attributes; i += 1) {
            CHECK_VERTEXELEMENTFORMAT_ENUM_INVALID(graphicsPipelineCreateInfo->vertex_input_state.vertex_attributes[i].format, NULL);

            locations[i] = graphicsPipelineCreateInfo->vertex_input_state.vertex_attributes[i].location;
            for (Uint32 j = 0; j < i; j += 1) {
                if (locations[j] == locations[i]) {
                    SDL_assert_release(!"Each vertex attribute location in a vertex input state must be unique!");
                    return NULL;
                }
            }
        }
        if (graphicsPipelineCreateInfo->multisample_state.enable_mask) {
            SDL_assert_release(!"For multisample states, enable_mask must be false!");
            return NULL;
        }
        if (graphicsPipelineCreateInfo->multisample_state.sample_mask != 0) {
            SDL_assert_release(!"For multisample states, sample_mask must be 0!");
            return NULL;
        }
        if (graphicsPipelineCreateInfo->depth_stencil_state.enable_depth_test) {
            CHECK_COMPAREOP_ENUM_INVALID(graphicsPipelineCreateInfo->depth_stencil_state.compare_op, NULL)
        }
        if (graphicsPipelineCreateInfo->depth_stencil_state.enable_stencil_test) {
            const SDL_GPUStencilOpState *stencil_state = &graphicsPipelineCreateInfo->depth_stencil_state.back_stencil_state;
            CHECK_COMPAREOP_ENUM_INVALID(stencil_state->compare_op, NULL)
            CHECK_STENCILOP_ENUM_INVALID(stencil_state->fail_op, NULL)
            CHECK_STENCILOP_ENUM_INVALID(stencil_state->pass_op, NULL)
            CHECK_STENCILOP_ENUM_INVALID(stencil_state->depth_fail_op, NULL)
        }
    }

    return device->CreateGraphicsPipeline(
        device->driverData,
        graphicsPipelineCreateInfo);
}

SDL_GPUSampler *SDL_CreateGPUSampler(
    SDL_GPUDevice *device,
    const SDL_GPUSamplerCreateInfo *createinfo)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (createinfo == NULL) {
        SDL_InvalidParamError("createinfo");
        return NULL;
    }

    return device->CreateSampler(
        device->driverData,
        createinfo);
}

SDL_GPUShader *SDL_CreateGPUShader(
    SDL_GPUDevice *device,
    const SDL_GPUShaderCreateInfo *createinfo)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (createinfo == NULL) {
        SDL_InvalidParamError("createinfo");
        return NULL;
    }

    if (device->debug_mode) {
        if (createinfo->format == SDL_GPU_SHADERFORMAT_INVALID) {
            SDL_assert_release(!"Shader format cannot be INVALID!");
            return NULL;
        }
        if (!(createinfo->format & device->shader_formats)) {
            SDL_assert_release(!"Incompatible shader format for GPU backend");
            return NULL;
        }
    }

    return device->CreateShader(
        device->driverData,
        createinfo);
}

SDL_GPUTexture *SDL_CreateGPUTexture(
    SDL_GPUDevice *device,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (createinfo == NULL) {
        SDL_InvalidParamError("createinfo");
        return NULL;
    }

    if (device->debug_mode) {
        bool failed = false;

        const Uint32 MAX_2D_DIMENSION = 16384;
        const Uint32 MAX_3D_DIMENSION = 2048;

        // Common checks for all texture types
        CHECK_TEXTUREFORMAT_ENUM_INVALID(createinfo->format, NULL)

        if (createinfo->width <= 0 || createinfo->height <= 0 || createinfo->layer_count_or_depth <= 0) {
            SDL_assert_release(!"For any texture: width, height, and layer_count_or_depth must be >= 1");
            failed = true;
        }
        if (createinfo->num_levels <= 0) {
            SDL_assert_release(!"For any texture: num_levels must be >= 1");
            failed = true;
        }
        if ((createinfo->usage & SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ) && (createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER)) {
            SDL_assert_release(!"For any texture: usage cannot contain both GRAPHICS_STORAGE_READ and SAMPLER");
            failed = true;
        }
        if (createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1 &&
            (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_SAMPLER |
                                  SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                                  SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                  SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE))) {
            SDL_assert_release(!"For multisample textures: usage cannot contain SAMPLER or STORAGE flags");
            failed = true;
        }
        if (IsDepthFormat(createinfo->format) && (createinfo->usage & ~(SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER))) {
            SDL_assert_release(!"For depth textures: usage cannot contain any flags except for DEPTH_STENCIL_TARGET and SAMPLER");
            failed = true;
        }
        if (IsIntegerFormat(createinfo->format) && (createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER)) {
            SDL_assert_release(!"For any texture: usage cannot contain SAMPLER for textures with an integer format");
            failed = true;
        }

        if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE) {
            // Cubemap validation
            if (createinfo->width != createinfo->height) {
                SDL_assert_release(!"For cube textures: width and height must be identical");
                failed = true;
            }
            if (createinfo->width > MAX_2D_DIMENSION || createinfo->height > MAX_2D_DIMENSION) {
                SDL_assert_release(!"For cube textures: width and height must be <= 16384");
                failed = true;
            }
            if (createinfo->layer_count_or_depth != 6) {
                SDL_assert_release(!"For cube textures: layer_count_or_depth must be 6");
                failed = true;
            }
            if (createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1) {
                SDL_assert_release(!"For cube textures: sample_count must be SDL_GPU_SAMPLECOUNT_1");
                failed = true;
            }
            if (!SDL_GPUTextureSupportsFormat(device, createinfo->format, SDL_GPU_TEXTURETYPE_CUBE, createinfo->usage)) {
                SDL_assert_release(!"For cube textures: the format is unsupported for the given usage");
                failed = true;
            }
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
            // Cubemap array validation
            if (createinfo->width != createinfo->height) {
                SDL_assert_release(!"For cube array textures: width and height must be identical");
                failed = true;
            }
            if (createinfo->width > MAX_2D_DIMENSION || createinfo->height > MAX_2D_DIMENSION) {
                SDL_assert_release(!"For cube array textures: width and height must be <= 16384");
                failed = true;
            }
            if (createinfo->layer_count_or_depth % 6 != 0) {
                SDL_assert_release(!"For cube array textures: layer_count_or_depth must be a multiple of 6");
                failed = true;
            }
            if (createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1) {
                SDL_assert_release(!"For cube array textures: sample_count must be SDL_GPU_SAMPLECOUNT_1");
                failed = true;
            }
            if (!SDL_GPUTextureSupportsFormat(device, createinfo->format, SDL_GPU_TEXTURETYPE_CUBE_ARRAY, createinfo->usage)) {
                SDL_assert_release(!"For cube array textures: the format is unsupported for the given usage");
                failed = true;
            }
        } else if (createinfo->type == SDL_GPU_TEXTURETYPE_3D) {
            // 3D Texture Validation
            if (createinfo->width > MAX_3D_DIMENSION || createinfo->height > MAX_3D_DIMENSION || createinfo->layer_count_or_depth > MAX_3D_DIMENSION) {
                SDL_assert_release(!"For 3D textures: width, height, and layer_count_or_depth must be <= 2048");
                failed = true;
            }
            if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
                SDL_assert_release(!"For 3D textures: usage must not contain DEPTH_STENCIL_TARGET");
                failed = true;
            }
            if (createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1) {
                SDL_assert_release(!"For 3D textures: sample_count must be SDL_GPU_SAMPLECOUNT_1");
                failed = true;
            }
            if (!SDL_GPUTextureSupportsFormat(device, createinfo->format, SDL_GPU_TEXTURETYPE_3D, createinfo->usage)) {
                SDL_assert_release(!"For 3D textures: the format is unsupported for the given usage");
                failed = true;
            }
        } else {
            if (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY) {
                // Array Texture Validation
                if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
                    SDL_assert_release(!"For array textures: usage must not contain DEPTH_STENCIL_TARGET");
                    failed = true;
                }
                if (createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1) {
                    SDL_assert_release(!"For array textures: sample_count must be SDL_GPU_SAMPLECOUNT_1");
                    failed = true;
                }
            }
            if (createinfo->sample_count > SDL_GPU_SAMPLECOUNT_1 && createinfo->num_levels > 1) {
                SDL_assert_release(!"For 2D multisample textures: num_levels must be 1");
                failed = true;
            }
            if (!SDL_GPUTextureSupportsFormat(device, createinfo->format, SDL_GPU_TEXTURETYPE_2D, createinfo->usage)) {
                SDL_assert_release(!"For 2D textures: the format is unsupported for the given usage");
                failed = true;
            }
        }

        if (failed) {
            return NULL;
        }
    }

    return device->CreateTexture(
        device->driverData,
        createinfo);
}

SDL_GPUBuffer *SDL_CreateGPUBuffer(
    SDL_GPUDevice *device,
    const SDL_GPUBufferCreateInfo *createinfo)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (createinfo == NULL) {
        SDL_InvalidParamError("createinfo");
        return NULL;
    }

    const char *debugName = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_BUFFER_CREATE_NAME_STRING, NULL);

    return device->CreateBuffer(
        device->driverData,
        createinfo->usage,
        createinfo->size,
        debugName);
}

SDL_GPUTransferBuffer *SDL_CreateGPUTransferBuffer(
    SDL_GPUDevice *device,
    const SDL_GPUTransferBufferCreateInfo *createinfo)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (createinfo == NULL) {
        SDL_InvalidParamError("createinfo");
        return NULL;
    }

    const char *debugName = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_TRANSFERBUFFER_CREATE_NAME_STRING, NULL);

    return device->CreateTransferBuffer(
        device->driverData,
        createinfo->usage,
        createinfo->size,
        debugName);
}

// Debug Naming

void SDL_SetGPUBufferName(
    SDL_GPUDevice *device,
    SDL_GPUBuffer *buffer,
    const char *text)
{
    CHECK_DEVICE_MAGIC(device, );
    if (buffer == NULL) {
        SDL_InvalidParamError("buffer");
        return;
    }
    if (text == NULL) {
        SDL_InvalidParamError("text");
    }

    device->SetBufferName(
        device->driverData,
        buffer,
        text);
}

void SDL_SetGPUTextureName(
    SDL_GPUDevice *device,
    SDL_GPUTexture *texture,
    const char *text)
{
    CHECK_DEVICE_MAGIC(device, );
    if (texture == NULL) {
        SDL_InvalidParamError("texture");
        return;
    }
    if (text == NULL) {
        SDL_InvalidParamError("text");
    }

    device->SetTextureName(
        device->driverData,
        texture,
        text);
}

void SDL_InsertGPUDebugLabel(
    SDL_GPUCommandBuffer *command_buffer,
    const char *text)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (text == NULL) {
        SDL_InvalidParamError("text");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
    }

    COMMAND_BUFFER_DEVICE->InsertDebugLabel(
        command_buffer,
        text);
}

void SDL_PushGPUDebugGroup(
    SDL_GPUCommandBuffer *command_buffer,
    const char *name)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (name == NULL) {
        SDL_InvalidParamError("name");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
    }

    COMMAND_BUFFER_DEVICE->PushDebugGroup(
        command_buffer,
        name);
}

void SDL_PopGPUDebugGroup(
    SDL_GPUCommandBuffer *command_buffer)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
    }

    COMMAND_BUFFER_DEVICE->PopDebugGroup(
        command_buffer);
}

// Disposal

void SDL_ReleaseGPUTexture(
    SDL_GPUDevice *device,
    SDL_GPUTexture *texture)
{
    CHECK_DEVICE_MAGIC(device, );
    if (texture == NULL) {
        return;
    }

    device->ReleaseTexture(
        device->driverData,
        texture);
}

void SDL_ReleaseGPUSampler(
    SDL_GPUDevice *device,
    SDL_GPUSampler *sampler)
{
    CHECK_DEVICE_MAGIC(device, );
    if (sampler == NULL) {
        return;
    }

    device->ReleaseSampler(
        device->driverData,
        sampler);
}

void SDL_ReleaseGPUBuffer(
    SDL_GPUDevice *device,
    SDL_GPUBuffer *buffer)
{
    CHECK_DEVICE_MAGIC(device, );
    if (buffer == NULL) {
        return;
    }

    device->ReleaseBuffer(
        device->driverData,
        buffer);
}

void SDL_ReleaseGPUTransferBuffer(
    SDL_GPUDevice *device,
    SDL_GPUTransferBuffer *transfer_buffer)
{
    CHECK_DEVICE_MAGIC(device, );
    if (transfer_buffer == NULL) {
        return;
    }

    device->ReleaseTransferBuffer(
        device->driverData,
        transfer_buffer);
}

void SDL_ReleaseGPUShader(
    SDL_GPUDevice *device,
    SDL_GPUShader *shader)
{
    CHECK_DEVICE_MAGIC(device, );
    if (shader == NULL) {
        return;
    }

    device->ReleaseShader(
        device->driverData,
        shader);
}

void SDL_ReleaseGPUComputePipeline(
    SDL_GPUDevice *device,
    SDL_GPUComputePipeline *compute_pipeline)
{
    CHECK_DEVICE_MAGIC(device, );
    if (compute_pipeline == NULL) {
        return;
    }

    device->ReleaseComputePipeline(
        device->driverData,
        compute_pipeline);
}

void SDL_ReleaseGPUGraphicsPipeline(
    SDL_GPUDevice *device,
    SDL_GPUGraphicsPipeline *graphics_pipeline)
{
    CHECK_DEVICE_MAGIC(device, );
    if (graphics_pipeline == NULL) {
        return;
    }

    device->ReleaseGraphicsPipeline(
        device->driverData,
        graphics_pipeline);
}

// Command Buffer

SDL_GPUCommandBuffer *SDL_AcquireGPUCommandBuffer(
    SDL_GPUDevice *device)
{
    SDL_GPUCommandBuffer *command_buffer;
    CommandBufferCommonHeader *commandBufferHeader;

    CHECK_DEVICE_MAGIC(device, NULL);

    command_buffer = device->AcquireCommandBuffer(
        device->driverData);

    if (command_buffer == NULL) {
        return NULL;
    }

    commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;
    commandBufferHeader->device = device;
    commandBufferHeader->render_pass.command_buffer = command_buffer;
    commandBufferHeader->render_pass.in_progress = false;
    commandBufferHeader->graphics_pipeline_bound = false;
    commandBufferHeader->compute_pass.command_buffer = command_buffer;
    commandBufferHeader->compute_pass.in_progress = false;
    commandBufferHeader->compute_pipeline_bound = false;
    commandBufferHeader->copy_pass.command_buffer = command_buffer;
    commandBufferHeader->copy_pass.in_progress = false;
    commandBufferHeader->swapchain_texture_acquired = false;
    commandBufferHeader->submitted = false;

    return command_buffer;
}

// Uniforms

void SDL_PushGPUVertexUniformData(
    SDL_GPUCommandBuffer *command_buffer,
    Uint32 slot_index,
    const void *data,
    Uint32 length)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (data == NULL) {
        SDL_InvalidParamError("data");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
    }

    COMMAND_BUFFER_DEVICE->PushVertexUniformData(
        command_buffer,
        slot_index,
        data,
        length);
}

void SDL_PushGPUFragmentUniformData(
    SDL_GPUCommandBuffer *command_buffer,
    Uint32 slot_index,
    const void *data,
    Uint32 length)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (data == NULL) {
        SDL_InvalidParamError("data");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
    }

    COMMAND_BUFFER_DEVICE->PushFragmentUniformData(
        command_buffer,
        slot_index,
        data,
        length);
}

void SDL_PushGPUComputeUniformData(
    SDL_GPUCommandBuffer *command_buffer,
    Uint32 slot_index,
    const void *data,
    Uint32 length)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (data == NULL) {
        SDL_InvalidParamError("data");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
    }

    COMMAND_BUFFER_DEVICE->PushComputeUniformData(
        command_buffer,
        slot_index,
        data,
        length);
}

// Render Pass

SDL_GPURenderPass *SDL_BeginGPURenderPass(
    SDL_GPUCommandBuffer *command_buffer,
    const SDL_GPUColorTargetInfo *color_target_infos,
    Uint32 num_color_targets,
    const SDL_GPUDepthStencilTargetInfo *depth_stencil_target_info)
{
    CommandBufferCommonHeader *commandBufferHeader;

    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return NULL;
    }
    if (color_target_infos == NULL && num_color_targets > 0) {
        SDL_InvalidParamError("color_target_infos");
        return NULL;
    }

    if (num_color_targets > MAX_COLOR_TARGET_BINDINGS) {
        SDL_SetError("num_color_targets exceeds MAX_COLOR_TARGET_BINDINGS");
        return NULL;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER_RETURN_NULL
        CHECK_ANY_PASS_IN_PROGRESS("Cannot begin render pass during another pass!", NULL)

        for (Uint32 i = 0; i < num_color_targets; i += 1) {
            TextureCommonHeader *textureHeader = (TextureCommonHeader *)color_target_infos[i].texture;

            if (color_target_infos[i].cycle && color_target_infos[i].load_op == SDL_GPU_LOADOP_LOAD) {
                SDL_assert_release(!"Cannot cycle color target when load op is LOAD!");
            }

            if (color_target_infos[i].store_op == SDL_GPU_STOREOP_RESOLVE || color_target_infos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
                if (color_target_infos[i].resolve_texture == NULL) {
                    SDL_assert_release(!"Store op is RESOLVE or RESOLVE_AND_STORE but resolve_texture is NULL!");
                } else {
                    TextureCommonHeader *resolveTextureHeader = (TextureCommonHeader *)color_target_infos[i].resolve_texture;
                    if (textureHeader->info.sample_count == SDL_GPU_SAMPLECOUNT_1) {
                        SDL_assert_release(!"Store op is RESOLVE or RESOLVE_AND_STORE but texture is not multisample!");
                    }
                    if (resolveTextureHeader->info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
                        SDL_assert_release(!"Resolve texture must have a sample count of 1!");
                    }
                    if (resolveTextureHeader->info.format != textureHeader->info.format) {
                        SDL_assert_release(!"Resolve texture must have the same format as its corresponding color target!");
                    }
                    if (resolveTextureHeader->info.type == SDL_GPU_TEXTURETYPE_3D) {
                        SDL_assert_release(!"Resolve texture must not be of TEXTURETYPE_3D!");
                    }
                    if (!(resolveTextureHeader->info.usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
                        SDL_assert_release(!"Resolve texture usage must include COLOR_TARGET!");
                    }
                }
            }
        }

        if (depth_stencil_target_info != NULL) {

            TextureCommonHeader *textureHeader = (TextureCommonHeader *)depth_stencil_target_info->texture;
            if (!(textureHeader->info.usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
                SDL_assert_release(!"Depth target must have been created with the DEPTH_STENCIL_TARGET usage flag!");
            }

            if (depth_stencil_target_info->cycle && (depth_stencil_target_info->load_op == SDL_GPU_LOADOP_LOAD || depth_stencil_target_info->stencil_load_op == SDL_GPU_LOADOP_LOAD)) {
                SDL_assert_release(!"Cannot cycle depth target when load op or stencil load op is LOAD!");
            }

            if (depth_stencil_target_info->store_op == SDL_GPU_STOREOP_RESOLVE ||
                depth_stencil_target_info->stencil_store_op == SDL_GPU_STOREOP_RESOLVE ||
                depth_stencil_target_info->store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE ||
                depth_stencil_target_info->stencil_store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
                SDL_assert_release(!"RESOLVE store ops are not supported for depth-stencil targets!");
            }
        }
    }

    COMMAND_BUFFER_DEVICE->BeginRenderPass(
        command_buffer,
        color_target_infos,
        num_color_targets,
        depth_stencil_target_info);

    commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;
    commandBufferHeader->render_pass.in_progress = true;
    for (Uint32 i = 0; i < num_color_targets; i += 1) {
        commandBufferHeader->render_pass.color_targets[i] = color_target_infos[i].texture;
    }
    commandBufferHeader->render_pass.num_color_targets = num_color_targets;
    if (depth_stencil_target_info != NULL) {
        commandBufferHeader->render_pass.depth_stencil_target = depth_stencil_target_info->texture;
    }
    return (SDL_GPURenderPass *)&(commandBufferHeader->render_pass);
}

void SDL_BindGPUGraphicsPipeline(
    SDL_GPURenderPass *render_pass,
    SDL_GPUGraphicsPipeline *graphics_pipeline)
{
    CommandBufferCommonHeader *commandBufferHeader;

    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (graphics_pipeline == NULL) {
        SDL_InvalidParamError("graphics_pipeline");
        return;
    }

    RENDERPASS_DEVICE->BindGraphicsPipeline(
        RENDERPASS_COMMAND_BUFFER,
        graphics_pipeline);

    commandBufferHeader = (CommandBufferCommonHeader *)RENDERPASS_COMMAND_BUFFER;
    commandBufferHeader->graphics_pipeline_bound = true;
}

void SDL_SetGPUViewport(
    SDL_GPURenderPass *render_pass,
    const SDL_GPUViewport *viewport)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (viewport == NULL) {
        SDL_InvalidParamError("viewport");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->SetViewport(
        RENDERPASS_COMMAND_BUFFER,
        viewport);
}

void SDL_SetGPUScissor(
    SDL_GPURenderPass *render_pass,
    const SDL_Rect *scissor)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (scissor == NULL) {
        SDL_InvalidParamError("scissor");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->SetScissor(
        RENDERPASS_COMMAND_BUFFER,
        scissor);
}

void SDL_SetGPUBlendConstants(
    SDL_GPURenderPass *render_pass,
    SDL_FColor blend_constants)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->SetBlendConstants(
        RENDERPASS_COMMAND_BUFFER,
        blend_constants);
}

void SDL_SetGPUStencilReference(
    SDL_GPURenderPass *render_pass,
    Uint8 reference)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->SetStencilReference(
        RENDERPASS_COMMAND_BUFFER,
        reference);
}

void SDL_BindGPUVertexBuffers(
    SDL_GPURenderPass *render_pass,
    Uint32 first_binding,
    const SDL_GPUBufferBinding *bindings,
    Uint32 num_bindings)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (bindings == NULL && num_bindings > 0) {
        SDL_InvalidParamError("bindings");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->BindVertexBuffers(
        RENDERPASS_COMMAND_BUFFER,
        first_binding,
        bindings,
        num_bindings);
}

void SDL_BindGPUIndexBuffer(
    SDL_GPURenderPass *render_pass,
    const SDL_GPUBufferBinding *binding,
    SDL_GPUIndexElementSize index_element_size)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (binding == NULL) {
        SDL_InvalidParamError("binding");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->BindIndexBuffer(
        RENDERPASS_COMMAND_BUFFER,
        binding,
        index_element_size);
}

void SDL_BindGPUVertexSamplers(
    SDL_GPURenderPass *render_pass,
    Uint32 first_slot,
    const SDL_GPUTextureSamplerBinding *texture_sampler_bindings,
    Uint32 num_bindings)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (texture_sampler_bindings == NULL && num_bindings > 0) {
        SDL_InvalidParamError("texture_sampler_bindings");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_SAMPLER_TEXTURES
    }

    RENDERPASS_DEVICE->BindVertexSamplers(
        RENDERPASS_COMMAND_BUFFER,
        first_slot,
        texture_sampler_bindings,
        num_bindings);
}

void SDL_BindGPUVertexStorageTextures(
    SDL_GPURenderPass *render_pass,
    Uint32 first_slot,
    SDL_GPUTexture *const *storage_textures,
    Uint32 num_bindings)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (storage_textures == NULL && num_bindings > 0) {
        SDL_InvalidParamError("storage_textures");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_STORAGE_TEXTURES
    }

    RENDERPASS_DEVICE->BindVertexStorageTextures(
        RENDERPASS_COMMAND_BUFFER,
        first_slot,
        storage_textures,
        num_bindings);
}

void SDL_BindGPUVertexStorageBuffers(
    SDL_GPURenderPass *render_pass,
    Uint32 first_slot,
    SDL_GPUBuffer *const *storage_buffers,
    Uint32 num_bindings)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (storage_buffers == NULL && num_bindings > 0) {
        SDL_InvalidParamError("storage_buffers");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->BindVertexStorageBuffers(
        RENDERPASS_COMMAND_BUFFER,
        first_slot,
        storage_buffers,
        num_bindings);
}

void SDL_BindGPUFragmentSamplers(
    SDL_GPURenderPass *render_pass,
    Uint32 first_slot,
    const SDL_GPUTextureSamplerBinding *texture_sampler_bindings,
    Uint32 num_bindings)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (texture_sampler_bindings == NULL && num_bindings > 0) {
        SDL_InvalidParamError("texture_sampler_bindings");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_SAMPLER_TEXTURES
    }

    RENDERPASS_DEVICE->BindFragmentSamplers(
        RENDERPASS_COMMAND_BUFFER,
        first_slot,
        texture_sampler_bindings,
        num_bindings);
}

void SDL_BindGPUFragmentStorageTextures(
    SDL_GPURenderPass *render_pass,
    Uint32 first_slot,
    SDL_GPUTexture *const *storage_textures,
    Uint32 num_bindings)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (storage_textures == NULL && num_bindings > 0) {
        SDL_InvalidParamError("storage_textures");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_STORAGE_TEXTURES
    }

    RENDERPASS_DEVICE->BindFragmentStorageTextures(
        RENDERPASS_COMMAND_BUFFER,
        first_slot,
        storage_textures,
        num_bindings);
}

void SDL_BindGPUFragmentStorageBuffers(
    SDL_GPURenderPass *render_pass,
    Uint32 first_slot,
    SDL_GPUBuffer *const *storage_buffers,
    Uint32 num_bindings)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (storage_buffers == NULL && num_bindings > 0) {
        SDL_InvalidParamError("storage_buffers");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->BindFragmentStorageBuffers(
        RENDERPASS_COMMAND_BUFFER,
        first_slot,
        storage_buffers,
        num_bindings);
}

void SDL_DrawGPUIndexedPrimitives(
    SDL_GPURenderPass *render_pass,
    Uint32 num_indices,
    Uint32 num_instances,
    Uint32 first_index,
    Sint32 vertex_offset,
    Uint32 first_instance)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_GRAPHICS_PIPELINE_BOUND
    }

    RENDERPASS_DEVICE->DrawIndexedPrimitives(
        RENDERPASS_COMMAND_BUFFER,
        num_indices,
        num_instances,
        first_index,
        vertex_offset,
        first_instance);
}

void SDL_DrawGPUPrimitives(
    SDL_GPURenderPass *render_pass,
    Uint32 num_vertices,
    Uint32 num_instances,
    Uint32 first_vertex,
    Uint32 first_instance)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_GRAPHICS_PIPELINE_BOUND
    }

    RENDERPASS_DEVICE->DrawPrimitives(
        RENDERPASS_COMMAND_BUFFER,
        num_vertices,
        num_instances,
        first_vertex,
        first_instance);
}

void SDL_DrawGPUPrimitivesIndirect(
    SDL_GPURenderPass *render_pass,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 draw_count)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (buffer == NULL) {
        SDL_InvalidParamError("buffer");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_GRAPHICS_PIPELINE_BOUND
    }

    RENDERPASS_DEVICE->DrawPrimitivesIndirect(
        RENDERPASS_COMMAND_BUFFER,
        buffer,
        offset,
        draw_count);
}

void SDL_DrawGPUIndexedPrimitivesIndirect(
    SDL_GPURenderPass *render_pass,
    SDL_GPUBuffer *buffer,
    Uint32 offset,
    Uint32 draw_count)
{
    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }
    if (buffer == NULL) {
        SDL_InvalidParamError("buffer");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
        CHECK_GRAPHICS_PIPELINE_BOUND
    }

    RENDERPASS_DEVICE->DrawIndexedPrimitivesIndirect(
        RENDERPASS_COMMAND_BUFFER,
        buffer,
        offset,
        draw_count);
}

void SDL_EndGPURenderPass(
    SDL_GPURenderPass *render_pass)
{
    CommandBufferCommonHeader *commandBufferCommonHeader;

    if (render_pass == NULL) {
        SDL_InvalidParamError("render_pass");
        return;
    }

    if (RENDERPASS_DEVICE->debug_mode) {
        CHECK_RENDERPASS
    }

    RENDERPASS_DEVICE->EndRenderPass(
        RENDERPASS_COMMAND_BUFFER);

    commandBufferCommonHeader = (CommandBufferCommonHeader *)RENDERPASS_COMMAND_BUFFER;
    commandBufferCommonHeader->render_pass.in_progress = false;
    for (Uint32 i = 0; i < MAX_COLOR_TARGET_BINDINGS; i += 1)
    {
        commandBufferCommonHeader->render_pass.color_targets[i] = NULL;
    }
    commandBufferCommonHeader->render_pass.num_color_targets = 0;
    commandBufferCommonHeader->render_pass.depth_stencil_target = NULL;
    commandBufferCommonHeader->graphics_pipeline_bound = false;
}

// Compute Pass

SDL_GPUComputePass *SDL_BeginGPUComputePass(
    SDL_GPUCommandBuffer *command_buffer,
    const SDL_GPUStorageTextureReadWriteBinding *storage_texture_bindings,
    Uint32 num_storage_texture_bindings,
    const SDL_GPUStorageBufferReadWriteBinding *storage_buffer_bindings,
    Uint32 num_storage_buffer_bindings)
{
    CommandBufferCommonHeader *commandBufferHeader;

    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return NULL;
    }
    if (storage_texture_bindings == NULL && num_storage_texture_bindings > 0) {
        SDL_InvalidParamError("storage_texture_bindings");
        return NULL;
    }
    if (storage_buffer_bindings == NULL && num_storage_buffer_bindings > 0) {
        SDL_InvalidParamError("storage_buffer_bindings");
        return NULL;
    }
    if (num_storage_texture_bindings > MAX_COMPUTE_WRITE_TEXTURES) {
        SDL_InvalidParamError("num_storage_texture_bindings");
        return NULL;
    }
    if (num_storage_buffer_bindings > MAX_COMPUTE_WRITE_BUFFERS) {
        SDL_InvalidParamError("num_storage_buffer_bindings");
        return NULL;
    }
    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER_RETURN_NULL
        CHECK_ANY_PASS_IN_PROGRESS("Cannot begin compute pass during another pass!", NULL)

        for (Uint32 i = 0; i < num_storage_texture_bindings; i += 1) {
            TextureCommonHeader *header = (TextureCommonHeader *)storage_texture_bindings[i].texture;
            if (!(header->info.usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) && !(header->info.usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
                SDL_assert_release(!"Texture must be created with COMPUTE_STORAGE_WRITE or COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE flag");
                return NULL;
            }
        }

        // TODO: validate buffer usage?
    }

    COMMAND_BUFFER_DEVICE->BeginComputePass(
        command_buffer,
        storage_texture_bindings,
        num_storage_texture_bindings,
        storage_buffer_bindings,
        num_storage_buffer_bindings);

    commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;
    commandBufferHeader->compute_pass.in_progress = true;
    return (SDL_GPUComputePass *)&(commandBufferHeader->compute_pass);
}

void SDL_BindGPUComputePipeline(
    SDL_GPUComputePass *compute_pass,
    SDL_GPUComputePipeline *compute_pipeline)
{
    CommandBufferCommonHeader *commandBufferHeader;

    if (compute_pass == NULL) {
        SDL_InvalidParamError("compute_pass");
        return;
    }
    if (compute_pipeline == NULL) {
        SDL_InvalidParamError("compute_pipeline");
        return;
    }

    if (COMPUTEPASS_DEVICE->debug_mode) {
        CHECK_COMPUTEPASS
    }

    COMPUTEPASS_DEVICE->BindComputePipeline(
        COMPUTEPASS_COMMAND_BUFFER,
        compute_pipeline);

    commandBufferHeader = (CommandBufferCommonHeader *)COMPUTEPASS_COMMAND_BUFFER;
    commandBufferHeader->compute_pipeline_bound = true;
}

void SDL_BindGPUComputeSamplers(
    SDL_GPUComputePass *compute_pass,
    Uint32 first_slot,
    const SDL_GPUTextureSamplerBinding *texture_sampler_bindings,
    Uint32 num_bindings)
{
    if (compute_pass == NULL) {
        SDL_InvalidParamError("compute_pass");
        return;
    }
    if (texture_sampler_bindings == NULL && num_bindings > 0) {
        SDL_InvalidParamError("texture_sampler_bindings");
        return;
    }

    if (COMPUTEPASS_DEVICE->debug_mode) {
        CHECK_COMPUTEPASS
    }

    COMPUTEPASS_DEVICE->BindComputeSamplers(
        COMPUTEPASS_COMMAND_BUFFER,
        first_slot,
        texture_sampler_bindings,
        num_bindings);
}

void SDL_BindGPUComputeStorageTextures(
    SDL_GPUComputePass *compute_pass,
    Uint32 first_slot,
    SDL_GPUTexture *const *storage_textures,
    Uint32 num_bindings)
{
    if (compute_pass == NULL) {
        SDL_InvalidParamError("compute_pass");
        return;
    }
    if (storage_textures == NULL && num_bindings > 0) {
        SDL_InvalidParamError("storage_textures");
        return;
    }

    if (COMPUTEPASS_DEVICE->debug_mode) {
        CHECK_COMPUTEPASS
    }

    COMPUTEPASS_DEVICE->BindComputeStorageTextures(
        COMPUTEPASS_COMMAND_BUFFER,
        first_slot,
        storage_textures,
        num_bindings);
}

void SDL_BindGPUComputeStorageBuffers(
    SDL_GPUComputePass *compute_pass,
    Uint32 first_slot,
    SDL_GPUBuffer *const *storage_buffers,
    Uint32 num_bindings)
{
    if (compute_pass == NULL) {
        SDL_InvalidParamError("compute_pass");
        return;
    }
    if (storage_buffers == NULL && num_bindings > 0) {
        SDL_InvalidParamError("storage_buffers");
        return;
    }

    if (COMPUTEPASS_DEVICE->debug_mode) {
        CHECK_COMPUTEPASS
    }

    COMPUTEPASS_DEVICE->BindComputeStorageBuffers(
        COMPUTEPASS_COMMAND_BUFFER,
        first_slot,
        storage_buffers,
        num_bindings);
}

void SDL_DispatchGPUCompute(
    SDL_GPUComputePass *compute_pass,
    Uint32 groupcount_x,
    Uint32 groupcount_y,
    Uint32 groupcount_z)
{
    if (compute_pass == NULL) {
        SDL_InvalidParamError("compute_pass");
        return;
    }

    if (COMPUTEPASS_DEVICE->debug_mode) {
        CHECK_COMPUTEPASS
        CHECK_COMPUTE_PIPELINE_BOUND
    }

    COMPUTEPASS_DEVICE->DispatchCompute(
        COMPUTEPASS_COMMAND_BUFFER,
        groupcount_x,
        groupcount_y,
        groupcount_z);
}

void SDL_DispatchGPUComputeIndirect(
    SDL_GPUComputePass *compute_pass,
    SDL_GPUBuffer *buffer,
    Uint32 offset)
{
    if (compute_pass == NULL) {
        SDL_InvalidParamError("compute_pass");
        return;
    }

    if (COMPUTEPASS_DEVICE->debug_mode) {
        CHECK_COMPUTEPASS
        CHECK_COMPUTE_PIPELINE_BOUND
    }

    COMPUTEPASS_DEVICE->DispatchComputeIndirect(
        COMPUTEPASS_COMMAND_BUFFER,
        buffer,
        offset);
}

void SDL_EndGPUComputePass(
    SDL_GPUComputePass *compute_pass)
{
    CommandBufferCommonHeader *commandBufferCommonHeader;

    if (compute_pass == NULL) {
        SDL_InvalidParamError("compute_pass");
        return;
    }

    if (COMPUTEPASS_DEVICE->debug_mode) {
        CHECK_COMPUTEPASS
    }

    COMPUTEPASS_DEVICE->EndComputePass(
        COMPUTEPASS_COMMAND_BUFFER);

    commandBufferCommonHeader = (CommandBufferCommonHeader *)COMPUTEPASS_COMMAND_BUFFER;
    commandBufferCommonHeader->compute_pass.in_progress = false;
    commandBufferCommonHeader->compute_pipeline_bound = false;
}

// TransferBuffer Data

void *SDL_MapGPUTransferBuffer(
    SDL_GPUDevice *device,
    SDL_GPUTransferBuffer *transfer_buffer,
    bool cycle)
{
    CHECK_DEVICE_MAGIC(device, NULL);
    if (transfer_buffer == NULL) {
        SDL_InvalidParamError("transfer_buffer");
        return NULL;
    }

    return device->MapTransferBuffer(
        device->driverData,
        transfer_buffer,
        cycle);
}

void SDL_UnmapGPUTransferBuffer(
    SDL_GPUDevice *device,
    SDL_GPUTransferBuffer *transfer_buffer)
{
    CHECK_DEVICE_MAGIC(device, );
    if (transfer_buffer == NULL) {
        SDL_InvalidParamError("transfer_buffer");
        return;
    }

    device->UnmapTransferBuffer(
        device->driverData,
        transfer_buffer);
}

// Copy Pass

SDL_GPUCopyPass *SDL_BeginGPUCopyPass(
    SDL_GPUCommandBuffer *command_buffer)
{
    CommandBufferCommonHeader *commandBufferHeader;

    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return NULL;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER_RETURN_NULL
        CHECK_ANY_PASS_IN_PROGRESS("Cannot begin copy pass during another pass!", NULL)
    }

    COMMAND_BUFFER_DEVICE->BeginCopyPass(
        command_buffer);

    commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;
    commandBufferHeader->copy_pass.in_progress = true;
    return (SDL_GPUCopyPass *)&(commandBufferHeader->copy_pass);
}

void SDL_UploadToGPUTexture(
    SDL_GPUCopyPass *copy_pass,
    const SDL_GPUTextureTransferInfo *source,
    const SDL_GPUTextureRegion *destination,
    bool cycle)
{
    if (copy_pass == NULL) {
        SDL_InvalidParamError("copy_pass");
        return;
    }
    if (source == NULL) {
        SDL_InvalidParamError("source");
        return;
    }
    if (destination == NULL) {
        SDL_InvalidParamError("destination");
        return;
    }

    if (COPYPASS_DEVICE->debug_mode) {
        CHECK_COPYPASS
        if (source->transfer_buffer == NULL) {
            SDL_assert_release(!"Source transfer buffer cannot be NULL!");
            return;
        }
        if (destination->texture == NULL) {
            SDL_assert_release(!"Destination texture cannot be NULL!");
            return;
        }
    }

    COPYPASS_DEVICE->UploadToTexture(
        COPYPASS_COMMAND_BUFFER,
        source,
        destination,
        cycle);
}

void SDL_UploadToGPUBuffer(
    SDL_GPUCopyPass *copy_pass,
    const SDL_GPUTransferBufferLocation *source,
    const SDL_GPUBufferRegion *destination,
    bool cycle)
{
    if (copy_pass == NULL) {
        SDL_InvalidParamError("copy_pass");
        return;
    }
    if (source == NULL) {
        SDL_InvalidParamError("source");
        return;
    }
    if (destination == NULL) {
        SDL_InvalidParamError("destination");
        return;
    }

    if (COPYPASS_DEVICE->debug_mode) {
        CHECK_COPYPASS
        if (source->transfer_buffer == NULL) {
            SDL_assert_release(!"Source transfer buffer cannot be NULL!");
            return;
        }
        if (destination->buffer == NULL) {
            SDL_assert_release(!"Destination buffer cannot be NULL!");
            return;
        }
    }

    COPYPASS_DEVICE->UploadToBuffer(
        COPYPASS_COMMAND_BUFFER,
        source,
        destination,
        cycle);
}

void SDL_CopyGPUTextureToTexture(
    SDL_GPUCopyPass *copy_pass,
    const SDL_GPUTextureLocation *source,
    const SDL_GPUTextureLocation *destination,
    Uint32 w,
    Uint32 h,
    Uint32 d,
    bool cycle)
{
    if (copy_pass == NULL) {
        SDL_InvalidParamError("copy_pass");
        return;
    }
    if (source == NULL) {
        SDL_InvalidParamError("source");
        return;
    }
    if (destination == NULL) {
        SDL_InvalidParamError("destination");
        return;
    }

    if (COPYPASS_DEVICE->debug_mode) {
        CHECK_COPYPASS
        if (source->texture == NULL) {
            SDL_assert_release(!"Source texture cannot be NULL!");
            return;
        }
        if (destination->texture == NULL) {
            SDL_assert_release(!"Destination texture cannot be NULL!");
            return;
        }

        TextureCommonHeader *srcHeader = (TextureCommonHeader *)source->texture;
        TextureCommonHeader *dstHeader = (TextureCommonHeader *)destination->texture;
        if (srcHeader->info.format != dstHeader->info.format) {
            SDL_assert_release(!"Source and destination textures must have the same format!");
            return;
        }
    }

    COPYPASS_DEVICE->CopyTextureToTexture(
        COPYPASS_COMMAND_BUFFER,
        source,
        destination,
        w,
        h,
        d,
        cycle);
}

void SDL_CopyGPUBufferToBuffer(
    SDL_GPUCopyPass *copy_pass,
    const SDL_GPUBufferLocation *source,
    const SDL_GPUBufferLocation *destination,
    Uint32 size,
    bool cycle)
{
    if (copy_pass == NULL) {
        SDL_InvalidParamError("copy_pass");
        return;
    }
    if (source == NULL) {
        SDL_InvalidParamError("source");
        return;
    }
    if (destination == NULL) {
        SDL_InvalidParamError("destination");
        return;
    }

    if (COPYPASS_DEVICE->debug_mode) {
        CHECK_COPYPASS
        if (source->buffer == NULL) {
            SDL_assert_release(!"Source buffer cannot be NULL!");
            return;
        }
        if (destination->buffer == NULL) {
            SDL_assert_release(!"Destination buffer cannot be NULL!");
            return;
        }
    }

    COPYPASS_DEVICE->CopyBufferToBuffer(
        COPYPASS_COMMAND_BUFFER,
        source,
        destination,
        size,
        cycle);
}

void SDL_DownloadFromGPUTexture(
    SDL_GPUCopyPass *copy_pass,
    const SDL_GPUTextureRegion *source,
    const SDL_GPUTextureTransferInfo *destination)
{
    if (copy_pass == NULL) {
        SDL_InvalidParamError("copy_pass");
        return;
    }
    if (source == NULL) {
        SDL_InvalidParamError("source");
        return;
    }
    if (destination == NULL) {
        SDL_InvalidParamError("destination");
        return;
    }

    if (COPYPASS_DEVICE->debug_mode) {
        CHECK_COPYPASS
        if (source->texture == NULL) {
            SDL_assert_release(!"Source texture cannot be NULL!");
            return;
        }
        if (destination->transfer_buffer == NULL) {
            SDL_assert_release(!"Destination transfer buffer cannot be NULL!");
            return;
        }
    }

    COPYPASS_DEVICE->DownloadFromTexture(
        COPYPASS_COMMAND_BUFFER,
        source,
        destination);
}

void SDL_DownloadFromGPUBuffer(
    SDL_GPUCopyPass *copy_pass,
    const SDL_GPUBufferRegion *source,
    const SDL_GPUTransferBufferLocation *destination)
{
    if (copy_pass == NULL) {
        SDL_InvalidParamError("copy_pass");
        return;
    }
    if (source == NULL) {
        SDL_InvalidParamError("source");
        return;
    }
    if (destination == NULL) {
        SDL_InvalidParamError("destination");
        return;
    }

    if (COPYPASS_DEVICE->debug_mode) {
        CHECK_COPYPASS
        if (source->buffer == NULL) {
            SDL_assert_release(!"Source buffer cannot be NULL!");
            return;
        }
        if (destination->transfer_buffer == NULL) {
            SDL_assert_release(!"Destination transfer buffer cannot be NULL!");
            return;
        }
    }

    COPYPASS_DEVICE->DownloadFromBuffer(
        COPYPASS_COMMAND_BUFFER,
        source,
        destination);
}

void SDL_EndGPUCopyPass(
    SDL_GPUCopyPass *copy_pass)
{
    if (copy_pass == NULL) {
        SDL_InvalidParamError("copy_pass");
        return;
    }

    if (COPYPASS_DEVICE->debug_mode) {
        CHECK_COPYPASS
    }

    COPYPASS_DEVICE->EndCopyPass(
        COPYPASS_COMMAND_BUFFER);

    ((CommandBufferCommonHeader *)COPYPASS_COMMAND_BUFFER)->copy_pass.in_progress = false;
}

void SDL_GenerateMipmapsForGPUTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_GPUTexture *texture)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (texture == NULL) {
        SDL_InvalidParamError("texture");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
        CHECK_ANY_PASS_IN_PROGRESS("Cannot generate mipmaps during a pass!", )

        TextureCommonHeader *header = (TextureCommonHeader *)texture;
        if (header->info.num_levels <= 1) {
            SDL_assert_release(!"Cannot generate mipmaps for texture with num_levels <= 1!");
            return;
        }

        if (!(header->info.usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) || !(header->info.usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
            SDL_assert_release(!"GenerateMipmaps texture must be created with SAMPLER and COLOR_TARGET usage flags!");
            return;
        }
    }

    COMMAND_BUFFER_DEVICE->GenerateMipmaps(
        command_buffer,
        texture);
}

void SDL_BlitGPUTexture(
    SDL_GPUCommandBuffer *command_buffer,
    const SDL_GPUBlitInfo *info)
{
    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return;
    }
    if (info == NULL) {
        SDL_InvalidParamError("info");
        return;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER
        CHECK_ANY_PASS_IN_PROGRESS("Cannot blit during a pass!", )

        // Validation
        bool failed = false;
        TextureCommonHeader *srcHeader = (TextureCommonHeader *)info->source.texture;
        TextureCommonHeader *dstHeader = (TextureCommonHeader *)info->destination.texture;

        if (srcHeader == NULL) {
            SDL_assert_release(!"Blit source texture must be non-NULL");
            return; // attempting to proceed will crash
        }
        if (dstHeader == NULL) {
            SDL_assert_release(!"Blit destination texture must be non-NULL");
            return; // attempting to proceed will crash
        }
        if (srcHeader->info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
            SDL_assert_release(!"Blit source texture must have a sample count of 1");
            failed = true;
        }
        if ((srcHeader->info.usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) == 0) {
            SDL_assert_release(!"Blit source texture must be created with the SAMPLER usage flag");
            failed = true;
        }
        if ((dstHeader->info.usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) == 0) {
            SDL_assert_release(!"Blit destination texture must be created with the COLOR_TARGET usage flag");
            failed = true;
        }
        if (IsDepthFormat(srcHeader->info.format)) {
            SDL_assert_release(!"Blit source texture cannot have a depth format");
            failed = true;
        }
        if (info->source.w == 0 || info->source.h == 0 || info->destination.w == 0 || info->destination.h == 0) {
            SDL_assert_release(!"Blit source/destination regions must have non-zero width, height, and depth");
            failed = true;
        }

        if (failed) {
            return;
        }
    }

    COMMAND_BUFFER_DEVICE->Blit(
        command_buffer,
        info);
}

// Submission/Presentation

bool SDL_WindowSupportsGPUSwapchainComposition(
    SDL_GPUDevice *device,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchain_composition)
{
    CHECK_DEVICE_MAGIC(device, false);
    if (window == NULL) {
        SDL_InvalidParamError("window");
        return false;
    }

    if (device->debug_mode) {
        CHECK_SWAPCHAINCOMPOSITION_ENUM_INVALID(swapchain_composition, false)
    }

    return device->SupportsSwapchainComposition(
        device->driverData,
        window,
        swapchain_composition);
}

bool SDL_WindowSupportsGPUPresentMode(
    SDL_GPUDevice *device,
    SDL_Window *window,
    SDL_GPUPresentMode present_mode)
{
    CHECK_DEVICE_MAGIC(device, false);
    if (window == NULL) {
        SDL_InvalidParamError("window");
        return false;
    }

    if (device->debug_mode) {
        CHECK_PRESENTMODE_ENUM_INVALID(present_mode, false)
    }

    return device->SupportsPresentMode(
        device->driverData,
        window,
        present_mode);
}

bool SDL_ClaimWindowForGPUDevice(
    SDL_GPUDevice *device,
    SDL_Window *window)
{
    CHECK_DEVICE_MAGIC(device, false);
    if (window == NULL) {
        return SDL_InvalidParamError("window");
    }

    if ((window->flags & SDL_WINDOW_TRANSPARENT) != 0) {
        return SDL_SetError("The GPU API doesn't support transparent windows");
    }

    return device->ClaimWindow(
        device->driverData,
        window);
}

void SDL_ReleaseWindowFromGPUDevice(
    SDL_GPUDevice *device,
    SDL_Window *window)
{
    CHECK_DEVICE_MAGIC(device, );
    if (window == NULL) {
        SDL_InvalidParamError("window");
        return;
    }

    device->ReleaseWindow(
        device->driverData,
        window);
}

bool SDL_SetGPUSwapchainParameters(
    SDL_GPUDevice *device,
    SDL_Window *window,
    SDL_GPUSwapchainComposition swapchain_composition,
    SDL_GPUPresentMode present_mode)
{
    CHECK_DEVICE_MAGIC(device, false);
    if (window == NULL) {
        SDL_InvalidParamError("window");
        return false;
    }

    if (device->debug_mode) {
        CHECK_SWAPCHAINCOMPOSITION_ENUM_INVALID(swapchain_composition, false)
        CHECK_PRESENTMODE_ENUM_INVALID(present_mode, false)
    }

    return device->SetSwapchainParameters(
        device->driverData,
        window,
        swapchain_composition,
        present_mode);
}

bool SDL_SetGPUAllowedFramesInFlight(
    SDL_GPUDevice *device,
    Uint32 allowed_frames_in_flight)
{
    CHECK_DEVICE_MAGIC(device, false);

    if (device->debug_mode) {
        if (allowed_frames_in_flight < 1 || allowed_frames_in_flight > 3)
        {
            SDL_assert_release(!"allowed_frames_in_flight value must be between 1 and 3!");
        }
    }

    allowed_frames_in_flight = SDL_clamp(allowed_frames_in_flight, 1, 3);
    return device->SetAllowedFramesInFlight(
        device->driverData,
        allowed_frames_in_flight);
}

SDL_GPUTextureFormat SDL_GetGPUSwapchainTextureFormat(
    SDL_GPUDevice *device,
    SDL_Window *window)
{
    CHECK_DEVICE_MAGIC(device, SDL_GPU_TEXTUREFORMAT_INVALID);
    if (window == NULL) {
        SDL_InvalidParamError("window");
        return SDL_GPU_TEXTUREFORMAT_INVALID;
    }

    return device->GetSwapchainTextureFormat(
        device->driverData,
        window);
}

bool SDL_AcquireGPUSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height)
{
    CommandBufferCommonHeader *commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;

    if (command_buffer == NULL) {
        return SDL_InvalidParamError("command_buffer");
    }
    if (window == NULL) {
        return SDL_InvalidParamError("window");
    }
    if (swapchain_texture == NULL) {
        return SDL_InvalidParamError("swapchain_texture");
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER_RETURN_FALSE
        CHECK_ANY_PASS_IN_PROGRESS("Cannot acquire a swapchain texture during a pass!", false)
    }

    bool result = COMMAND_BUFFER_DEVICE->AcquireSwapchainTexture(
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);

    if (*swapchain_texture != NULL){
        commandBufferHeader->swapchain_texture_acquired = true;
    }

    return result;
}

bool SDL_WaitForGPUSwapchain(
    SDL_GPUDevice *device,
    SDL_Window *window)
{
    CHECK_DEVICE_MAGIC(device, false);

    if (window == NULL) {
        return SDL_InvalidParamError("window");
    }

    return device->WaitForSwapchain(
        device->driverData,
        window);
}

bool SDL_WaitAndAcquireGPUSwapchainTexture(
    SDL_GPUCommandBuffer *command_buffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchain_texture,
    Uint32 *swapchain_texture_width,
    Uint32 *swapchain_texture_height)
{
    CommandBufferCommonHeader *commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;

    if (command_buffer == NULL) {
        return SDL_InvalidParamError("command_buffer");
    }
    if (window == NULL) {
        return SDL_InvalidParamError("window");
    }
    if (swapchain_texture == NULL) {
        return SDL_InvalidParamError("swapchain_texture");
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER_RETURN_FALSE
        CHECK_ANY_PASS_IN_PROGRESS("Cannot acquire a swapchain texture during a pass!", false)
    }

    bool result = COMMAND_BUFFER_DEVICE->WaitAndAcquireSwapchainTexture(
        command_buffer,
        window,
        swapchain_texture,
        swapchain_texture_width,
        swapchain_texture_height);

    if (*swapchain_texture != NULL){
        commandBufferHeader->swapchain_texture_acquired = true;
    }

    return result;
}

bool SDL_SubmitGPUCommandBuffer(
    SDL_GPUCommandBuffer *command_buffer)
{
    CommandBufferCommonHeader *commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;

    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return false;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER_RETURN_FALSE
        if (
            commandBufferHeader->render_pass.in_progress ||
            commandBufferHeader->compute_pass.in_progress ||
            commandBufferHeader->copy_pass.in_progress) {
            SDL_assert_release(!"Cannot submit command buffer while a pass is in progress!");
            return false;
        }
    }

    commandBufferHeader->submitted = true;

    return COMMAND_BUFFER_DEVICE->Submit(
        command_buffer);
}

SDL_GPUFence *SDL_SubmitGPUCommandBufferAndAcquireFence(
    SDL_GPUCommandBuffer *command_buffer)
{
    CommandBufferCommonHeader *commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;

    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return NULL;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        CHECK_COMMAND_BUFFER_RETURN_NULL
        if (
            commandBufferHeader->render_pass.in_progress ||
            commandBufferHeader->compute_pass.in_progress ||
            commandBufferHeader->copy_pass.in_progress) {
            SDL_assert_release(!"Cannot submit command buffer while a pass is in progress!");
            return NULL;
        }
    }

    commandBufferHeader->submitted = true;

    return COMMAND_BUFFER_DEVICE->SubmitAndAcquireFence(
        command_buffer);
}

bool SDL_CancelGPUCommandBuffer(
    SDL_GPUCommandBuffer *command_buffer)
{
    CommandBufferCommonHeader *commandBufferHeader = (CommandBufferCommonHeader *)command_buffer;

    if (command_buffer == NULL) {
        SDL_InvalidParamError("command_buffer");
        return false;
    }

    if (COMMAND_BUFFER_DEVICE->debug_mode) {
        if (commandBufferHeader->swapchain_texture_acquired) {
            SDL_assert_release(!"Cannot cancel command buffer after a swapchain texture has been acquired!");
            return false;
        }
    }

    return COMMAND_BUFFER_DEVICE->Cancel(
        command_buffer);
}

bool SDL_WaitForGPUIdle(
    SDL_GPUDevice *device)
{
    CHECK_DEVICE_MAGIC(device, false);

    return device->Wait(
        device->driverData);
}

bool SDL_WaitForGPUFences(
    SDL_GPUDevice *device,
    bool wait_all,
    SDL_GPUFence *const *fences,
    Uint32 num_fences)
{
    CHECK_DEVICE_MAGIC(device, false);
    if (fences == NULL && num_fences > 0) {
        SDL_InvalidParamError("fences");
        return false;
    }

    return device->WaitForFences(
        device->driverData,
        wait_all,
        fences,
        num_fences);
}

bool SDL_QueryGPUFence(
    SDL_GPUDevice *device,
    SDL_GPUFence *fence)
{
    CHECK_DEVICE_MAGIC(device, false);
    if (fence == NULL) {
        SDL_InvalidParamError("fence");
        return false;
    }

    return device->QueryFence(
        device->driverData,
        fence);
}

void SDL_ReleaseGPUFence(
    SDL_GPUDevice *device,
    SDL_GPUFence *fence)
{
    CHECK_DEVICE_MAGIC(device, );
    if (fence == NULL) {
        return;
    }

    device->ReleaseFence(
        device->driverData,
        fence);
}

Uint32 SDL_CalculateGPUTextureFormatSize(
    SDL_GPUTextureFormat format,
    Uint32 width,
    Uint32 height,
    Uint32 depth_or_layer_count)
{
    Uint32 blockWidth = SDL_max(Texture_GetBlockWidth(format), 1);
    Uint32 blockHeight = SDL_max(Texture_GetBlockHeight(format), 1);
    Uint32 blocksPerRow = (width + blockWidth - 1) / blockWidth;
    Uint32 blocksPerColumn = (height + blockHeight - 1) / blockHeight;
    return depth_or_layer_count * blocksPerRow * blocksPerColumn * SDL_GPUTextureFormatTexelBlockSize(format);
}

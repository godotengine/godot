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

#ifdef SDL_VIDEO_RENDER_GPU

#include "../../video/SDL_pixels_c.h"
#include "../SDL_d3dmath.h"
#include "../SDL_sysrender.h"
#include "SDL_gpu_util.h"
#include "SDL_pipeline_gpu.h"
#include "SDL_shaders_gpu.h"

typedef struct GPU_VertexShaderUniformData
{
    Float4X4 mvp;
    SDL_FColor color;
} GPU_VertexShaderUniformData;

typedef struct GPU_FragmentShaderUniformData
{
    float texel_width;
    float texel_height;
    float texture_width;
    float texture_height;
} GPU_FragmentShaderUniformData;

typedef struct GPU_RenderData
{
    SDL_GPUDevice *device;
    GPU_Shaders shaders;
    GPU_PipelineCache pipeline_cache;

    struct
    {
        SDL_GPUTexture *texture;
        SDL_GPUTextureFormat format;
        Uint32 width;
        Uint32 height;
    } backbuffer;

    struct
    {
        SDL_GPUSwapchainComposition composition;
        SDL_GPUPresentMode present_mode;
    } swapchain;

    struct
    {
        SDL_GPUTransferBuffer *transfer_buf;
        SDL_GPUBuffer *buffer;
        Uint32 buffer_size;
    } vertices;

    struct
    {
        SDL_GPURenderPass *render_pass;
        SDL_Texture *render_target;
        SDL_GPUCommandBuffer *command_buffer;
        SDL_GPUColorTargetInfo color_attachment;
        SDL_GPUViewport viewport;
        SDL_Rect scissor;
        SDL_FColor draw_color;
        bool scissor_enabled;
        bool scissor_was_enabled;
    } state;

    SDL_GPUSampler *samplers[RENDER_SAMPLER_COUNT];
} GPU_RenderData;

typedef struct GPU_TextureData
{
    SDL_GPUTexture *texture;
    SDL_GPUTextureFormat format;
    GPU_FragmentShaderID shader;
    void *pixels;
    int pitch;
    SDL_Rect locked_rect;
} GPU_TextureData;

static bool GPU_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    if (GPU_ConvertBlendFactor(srcColorFactor) == SDL_GPU_BLENDFACTOR_INVALID ||
        GPU_ConvertBlendFactor(srcAlphaFactor) == SDL_GPU_BLENDFACTOR_INVALID ||
        GPU_ConvertBlendOperation(colorOperation) == SDL_GPU_BLENDOP_INVALID ||
        GPU_ConvertBlendFactor(dstColorFactor) == SDL_GPU_BLENDFACTOR_INVALID ||
        GPU_ConvertBlendFactor(dstAlphaFactor) == SDL_GPU_BLENDFACTOR_INVALID ||
        GPU_ConvertBlendOperation(alphaOperation) == SDL_GPU_BLENDOP_INVALID) {
        return false;
    }

    return true;
}

static SDL_GPUTextureFormat PixFormatToTexFormat(SDL_PixelFormat pixel_format)
{
    switch (pixel_format) {
    case SDL_PIXELFORMAT_BGRA32:
    case SDL_PIXELFORMAT_BGRX32:
        return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    case SDL_PIXELFORMAT_RGBA32:
    case SDL_PIXELFORMAT_RGBX32:
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;

    // YUV TODO
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
    case SDL_PIXELFORMAT_UYVY:
    default:
        return SDL_GPU_TEXTUREFORMAT_INVALID;
    }
}

static SDL_PixelFormat TexFormatToPixFormat(SDL_GPUTextureFormat tex_format)
{
    switch (tex_format) {
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
        return SDL_PIXELFORMAT_RGBA32;
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
        return SDL_PIXELFORMAT_BGRA32;
    case SDL_GPU_TEXTUREFORMAT_B5G6R5_UNORM:
        return SDL_PIXELFORMAT_BGR565;
    case SDL_GPU_TEXTUREFORMAT_B5G5R5A1_UNORM:
        return SDL_PIXELFORMAT_BGRA5551;
    case SDL_GPU_TEXTUREFORMAT_B4G4R4A4_UNORM:
        return SDL_PIXELFORMAT_BGRA4444;
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
        return SDL_PIXELFORMAT_ABGR2101010;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
        return SDL_PIXELFORMAT_RGBA64;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
        return SDL_PIXELFORMAT_RGBA32;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
        return SDL_PIXELFORMAT_RGBA64_FLOAT;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
        return SDL_PIXELFORMAT_RGBA128_FLOAT;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
        return SDL_PIXELFORMAT_RGBA32;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
        return SDL_PIXELFORMAT_RGBA64;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
        return SDL_PIXELFORMAT_RGBA32;
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
        return SDL_PIXELFORMAT_BGRA32;
    default:
        return SDL_PIXELFORMAT_UNKNOWN;
    }
}

static bool GPU_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    GPU_RenderData *renderdata = (GPU_RenderData *)renderer->internal;
    GPU_TextureData *data;
    SDL_GPUTextureFormat format;
    SDL_GPUTextureUsageFlags usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;

    format = PixFormatToTexFormat(texture->format);

    if (format == SDL_GPU_TEXTUREFORMAT_INVALID) {
        return SDL_SetError("Texture format %s not supported by SDL_GPU",
                            SDL_GetPixelFormatName(texture->format));
    }

    data = (GPU_TextureData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }

    if (texture->access == SDL_TEXTUREACCESS_STREAMING) {
        size_t size;
        data->pitch = texture->w * SDL_BYTESPERPIXEL(texture->format);
        size = (size_t)texture->h * data->pitch;
        if (texture->format == SDL_PIXELFORMAT_YV12 ||
            texture->format == SDL_PIXELFORMAT_IYUV) {
            // Need to add size for the U and V planes
            size += 2 * ((texture->h + 1) / 2) * ((data->pitch + 1) / 2);
        }
        if (texture->format == SDL_PIXELFORMAT_NV12 ||
            texture->format == SDL_PIXELFORMAT_NV21) {
            // Need to add size for the U/V plane
            size += 2 * ((texture->h + 1) / 2) * ((data->pitch + 1) / 2);
        }
        data->pixels = SDL_calloc(1, size);
        if (!data->pixels) {
            SDL_free(data);
            return false;
        }

        // TODO allocate a persistent transfer buffer
    }

    if (texture->access == SDL_TEXTUREACCESS_TARGET) {
        usage |= SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    }

    texture->internal = data;
    SDL_GPUTextureCreateInfo tci;
    SDL_zero(tci);
    tci.format = format;
    tci.layer_count_or_depth = 1;
    tci.num_levels = 1;
    tci.usage = usage;
    tci.width = texture->w;
    tci.height = texture->h;
    tci.sample_count = SDL_GPU_SAMPLECOUNT_1;

    data->format = format;
    data->texture = SDL_CreateGPUTexture(renderdata->device, &tci);

    if (!data->texture) {
        return false;
    }

    if (texture->format == SDL_PIXELFORMAT_RGBA32 || texture->format == SDL_PIXELFORMAT_BGRA32) {
        data->shader = FRAG_SHADER_TEXTURE_RGBA;
    } else {
        data->shader = FRAG_SHADER_TEXTURE_RGB;
    }

    return true;
}

static bool GPU_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                              const SDL_Rect *rect, const void *pixels, int pitch)
{
    GPU_RenderData *renderdata = (GPU_RenderData *)renderer->internal;
    GPU_TextureData *data = (GPU_TextureData *)texture->internal;
    const Uint32 texturebpp = SDL_BYTESPERPIXEL(texture->format);

    size_t row_size, data_size;

    if (!SDL_size_mul_check_overflow(rect->w, texturebpp, &row_size) ||
        !SDL_size_mul_check_overflow(rect->h, row_size, &data_size)) {
        return SDL_SetError("update size overflow");
    }

    SDL_GPUTransferBufferCreateInfo tbci;
    SDL_zero(tbci);
    tbci.size = (Uint32)data_size;
    tbci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;

    SDL_GPUTransferBuffer *tbuf = SDL_CreateGPUTransferBuffer(renderdata->device, &tbci);

    if (tbuf == NULL) {
        return false;
    }

    Uint8 *output = SDL_MapGPUTransferBuffer(renderdata->device, tbuf, false);

    if ((size_t)pitch == row_size) {
        SDL_memcpy(output, pixels, data_size);
    } else {
        // FIXME is negative pitch supposed to work?
        // If not, maybe use SDL_GPUTextureTransferInfo::pixels_per_row instead of this
        const Uint8 *input = pixels;

        for (int i = 0; i < rect->h; ++i) {
            SDL_memcpy(output, input, row_size);
            output += row_size;
            input += pitch;
        }
    }

    SDL_UnmapGPUTransferBuffer(renderdata->device, tbuf);

    SDL_GPUCommandBuffer *cbuf = renderdata->state.command_buffer;
    SDL_GPUCopyPass *cpass = SDL_BeginGPUCopyPass(cbuf);

    SDL_GPUTextureTransferInfo tex_src;
    SDL_zero(tex_src);
    tex_src.transfer_buffer = tbuf;
    tex_src.rows_per_layer = rect->h;
    tex_src.pixels_per_row = rect->w;

    SDL_GPUTextureRegion tex_dst;
    SDL_zero(tex_dst);
    tex_dst.texture = data->texture;
    tex_dst.x = rect->x;
    tex_dst.y = rect->y;
    tex_dst.w = rect->w;
    tex_dst.h = rect->h;
    tex_dst.d = 1;

    SDL_UploadToGPUTexture(cpass, &tex_src, &tex_dst, false);
    SDL_EndGPUCopyPass(cpass);
    SDL_ReleaseGPUTransferBuffer(renderdata->device, tbuf);

    return true;
}

static bool GPU_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                            const SDL_Rect *rect, void **pixels, int *pitch)
{
    GPU_TextureData *data = (GPU_TextureData *)texture->internal;

    data->locked_rect = *rect;
    *pixels =
        (void *)((Uint8 *)data->pixels + rect->y * data->pitch +
                 rect->x * SDL_BYTESPERPIXEL(texture->format));
    *pitch = data->pitch;
    return true;
}

static void GPU_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GPU_TextureData *data = (GPU_TextureData *)texture->internal;
    const SDL_Rect *rect;
    void *pixels;

    rect = &data->locked_rect;
    pixels =
        (void *)((Uint8 *)data->pixels + rect->y * data->pitch +
                 rect->x * SDL_BYTESPERPIXEL(texture->format));
    GPU_UpdateTexture(renderer, texture, rect, pixels, data->pitch);
}

static bool GPU_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GPU_RenderData *data = (GPU_RenderData *)renderer->internal;

    data->state.render_target = texture;

    return true;
}

static bool GPU_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static SDL_FColor GetDrawCmdColor(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    SDL_FColor color = cmd->data.color.color;

    if (SDL_RenderingLinearSpace(renderer)) {
        SDL_ConvertToLinear(&color);
    }

    color.r *= cmd->data.color.color_scale;
    color.g *= cmd->data.color.color_scale;
    color.b *= cmd->data.color.color_scale;

    return color;
}

static bool GPU_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    float *verts = (float *)SDL_AllocateRenderVertices(renderer, count * 2 * sizeof(float), 0, &cmd->data.draw.first);

    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    for (int i = 0; i < count; i++) {
        *(verts++) = 0.5f + points[i].x;
        *(verts++) = 0.5f + points[i].y;
    }

    return true;
}

static bool GPU_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                              const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                              int num_vertices, const void *indices, int num_indices, int size_indices,
                              float scale_x, float scale_y)
{
    int i;
    int count = indices ? num_indices : num_vertices;
    float *verts;
    size_t sz = 2 * sizeof(float) + 4 * sizeof(float) + (texture ? 2 : 0) * sizeof(float);
    const float color_scale = cmd->data.draw.color_scale;
    bool convert_color = SDL_RenderingLinearSpace(renderer);

    verts = (float *)SDL_AllocateRenderVertices(renderer, count * sz, 0, &cmd->data.draw.first);
    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    for (i = 0; i < count; i++) {
        int j;
        float *xy_;
        SDL_FColor col_;
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

        *(verts++) = xy_[0] * scale_x;
        *(verts++) = xy_[1] * scale_y;

        col_ = *(SDL_FColor *)((char *)color + j * color_stride);
        if (convert_color) {
            SDL_ConvertToLinear(&col_);
        }

        // FIXME: The Vulkan backend doesn't multiply by color_scale. GL does. I'm not sure which one is wrong.
        // ANSWER: The color scale should be applied in linear space when using the scRGB colorspace. This is done in shaders in the Vulkan backend.
        *(verts++) = col_.r * color_scale;
        *(verts++) = col_.g * color_scale;
        *(verts++) = col_.b * color_scale;
        *(verts++) = col_.a;

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            *(verts++) = uv_[0];
            *(verts++) = uv_[1];
        }
    }
    return true;
}

static void GPU_InvalidateCachedState(SDL_Renderer *renderer)
{
    GPU_RenderData *data = (GPU_RenderData *)renderer->internal;

    data->state.scissor_enabled = false;
}

static SDL_GPURenderPass *RestartRenderPass(GPU_RenderData *data)
{
    if (data->state.render_pass) {
        SDL_EndGPURenderPass(data->state.render_pass);
    }

    data->state.render_pass = SDL_BeginGPURenderPass(
        data->state.command_buffer, &data->state.color_attachment, 1, NULL);

    // *** FIXME ***
    // This is busted. We should be able to know which load op to use.
    // LOAD is incorrect behavior most of the time, unless we had to break a render pass.
    // -cosmonaut
    data->state.color_attachment.load_op = SDL_GPU_LOADOP_LOAD;
    data->state.scissor_was_enabled = false;

    return data->state.render_pass;
}

static void PushVertexUniforms(GPU_RenderData *data, SDL_RenderCommand *cmd)
{
    GPU_VertexShaderUniformData uniforms;
    SDL_zero(uniforms);
    uniforms.mvp.m[0][0] = 2.0f / data->state.viewport.w;
    uniforms.mvp.m[1][1] = -2.0f / data->state.viewport.h;
    uniforms.mvp.m[2][2] = 1.0f;
    uniforms.mvp.m[3][0] = -1.0f;
    uniforms.mvp.m[3][1] = 1.0f;
    uniforms.mvp.m[3][3] = 1.0f;

    uniforms.color = data->state.draw_color;

    SDL_PushGPUVertexUniformData(data->state.command_buffer, 0, &uniforms, sizeof(uniforms));
}

static void PushFragmentUniforms(GPU_RenderData *data, SDL_RenderCommand *cmd)
{
    if (cmd->data.draw.texture &&
        cmd->data.draw.texture_scale_mode == SDL_SCALEMODE_PIXELART) {
        SDL_Texture *texture = cmd->data.draw.texture;
        GPU_FragmentShaderUniformData uniforms;
        SDL_zero(uniforms);
        uniforms.texture_width = texture->w;
        uniforms.texture_height = texture->h;
        uniforms.texel_width = 1.0f / uniforms.texture_width;
        uniforms.texel_height = 1.0f / uniforms.texture_height;
        SDL_PushGPUFragmentUniformData(data->state.command_buffer, 0, &uniforms, sizeof(uniforms));
    }
}

static void SetViewportAndScissor(GPU_RenderData *data)
{
    SDL_SetGPUViewport(data->state.render_pass, &data->state.viewport);

    if (data->state.scissor_enabled) {
        SDL_SetGPUScissor(data->state.render_pass, &data->state.scissor);
        data->state.scissor_was_enabled = true;
    } else if (data->state.scissor_was_enabled) {
        SDL_Rect r;
        r.x = (int)data->state.viewport.x;
        r.y = (int)data->state.viewport.y;
        r.w = (int)data->state.viewport.w;
        r.h = (int)data->state.viewport.h;
        SDL_SetGPUScissor(data->state.render_pass, &r);
        data->state.scissor_was_enabled = false;
    }
}

static SDL_GPUSampler *GetSampler(GPU_RenderData *data, SDL_ScaleMode scale_mode, SDL_TextureAddressMode address_u, SDL_TextureAddressMode address_v)
{
    Uint32 key = RENDER_SAMPLER_HASHKEY(scale_mode, address_u, address_v);
    SDL_assert(key < SDL_arraysize(data->samplers));
    if (!data->samplers[key]) {
        SDL_GPUSamplerCreateInfo sci;
        SDL_zero(sci);
        switch (scale_mode) {
        case SDL_SCALEMODE_NEAREST:
            sci.min_filter = SDL_GPU_FILTER_NEAREST;
            sci.mag_filter = SDL_GPU_FILTER_NEAREST;
            sci.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
            break;
        case SDL_SCALEMODE_PIXELART:    // Uses linear sampling
        case SDL_SCALEMODE_LINEAR:
            sci.min_filter = SDL_GPU_FILTER_LINEAR;
            sci.mag_filter = SDL_GPU_FILTER_LINEAR;
            sci.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
            break;
        default:
            SDL_SetError("Unknown scale mode: %d", scale_mode);
            return NULL;
        }
        switch (address_u) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            sci.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            sci.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_u);
            return NULL;
        }
        switch (address_v) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            sci.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            sci.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_v);
            return NULL;
        }
        sci.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;

        data->samplers[key] = SDL_CreateGPUSampler(data->device, &sci);
    }
    return data->samplers[key];
}

static void Draw(
    GPU_RenderData *data, SDL_RenderCommand *cmd,
    Uint32 num_verts,
    Uint32 offset,
    SDL_GPUPrimitiveType prim)
{
    if (!data->state.render_pass || data->state.color_attachment.load_op == SDL_GPU_LOADOP_CLEAR) {
        RestartRenderPass(data);
    }

    SDL_GPURenderPass *pass = data->state.render_pass;
    SDL_GPURenderState *custom_state = cmd->data.draw.gpu_render_state;
    SDL_GPUShader *custom_frag_shader = custom_state ? custom_state->fragment_shader : NULL;
    GPU_VertexShaderID v_shader;
    GPU_FragmentShaderID f_shader;

    if (prim == SDL_GPU_PRIMITIVETYPE_TRIANGLELIST) {
        SDL_Texture *texture = cmd->data.draw.texture;
        if (texture) {
            v_shader = VERT_SHADER_TRI_TEXTURE;
            if (texture->format == SDL_PIXELFORMAT_RGBA32 || texture->format == SDL_PIXELFORMAT_BGRA32) {
                if (cmd->data.draw.texture_scale_mode == SDL_SCALEMODE_PIXELART) {
                    f_shader = FRAG_SHADER_TEXTURE_RGBA_PIXELART;
                } else {
                    f_shader = FRAG_SHADER_TEXTURE_RGBA;
                }
            } else {
                if (cmd->data.draw.texture_scale_mode == SDL_SCALEMODE_PIXELART) {
                    f_shader = FRAG_SHADER_TEXTURE_RGB_PIXELART;
                } else {
                    f_shader = FRAG_SHADER_TEXTURE_RGB;
                }
            }
        } else {
            v_shader = VERT_SHADER_TRI_COLOR;
            f_shader = FRAG_SHADER_COLOR;
        }
    } else {
        v_shader = VERT_SHADER_LINEPOINT;
        f_shader = FRAG_SHADER_COLOR;
    }

    if (custom_frag_shader) {
        f_shader = FRAG_SHADER_TEXTURE_CUSTOM;
        data->shaders.frag_shaders[FRAG_SHADER_TEXTURE_CUSTOM] = custom_frag_shader;
    }

    GPU_PipelineParameters pipe_params;
    SDL_zero(pipe_params);
    pipe_params.blend_mode = cmd->data.draw.blend;
    pipe_params.vert_shader = v_shader;
    pipe_params.frag_shader = f_shader;
    pipe_params.primitive_type = prim;
    pipe_params.custom_frag_shader = custom_frag_shader;

    if (data->state.render_target) {
        pipe_params.attachment_format = ((GPU_TextureData *)data->state.render_target->internal)->format;
    } else {
        pipe_params.attachment_format = data->backbuffer.format;
    }

    SDL_GPUGraphicsPipeline *pipe = GPU_GetPipeline(&data->pipeline_cache, &data->shaders, data->device, &pipe_params);
    if (!pipe) {
        return;
    }

    SDL_BindGPUGraphicsPipeline(pass, pipe);

    Uint32 sampler_slot = 0;
    if (cmd->data.draw.texture) {
        GPU_TextureData *tdata = (GPU_TextureData *)cmd->data.draw.texture->internal;
        SDL_GPUTextureSamplerBinding sampler_bind;
        SDL_zero(sampler_bind);
        sampler_bind.sampler = GetSampler(data, cmd->data.draw.texture_scale_mode, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);
        sampler_bind.texture = tdata->texture;
        SDL_BindGPUFragmentSamplers(pass, sampler_slot++, &sampler_bind, 1);
    }
    if (custom_state) {
        if (custom_state->num_sampler_bindings > 0) {
            SDL_BindGPUFragmentSamplers(pass, sampler_slot, custom_state->sampler_bindings, custom_state->num_sampler_bindings);
        }
        if (custom_state->num_storage_textures > 0) {
            SDL_BindGPUFragmentStorageTextures(pass, 0, custom_state->storage_textures, custom_state->num_storage_textures);
        }
        if (custom_state->num_storage_buffers > 0) {
            SDL_BindGPUFragmentStorageBuffers(pass, 0, custom_state->storage_buffers, custom_state->num_storage_buffers);
        }
        if (custom_state->num_uniform_buffers > 0) {
            for (int i = 0; i < custom_state->num_uniform_buffers; i++) {
                SDL_GPURenderStateUniformBuffer *ub = &custom_state->uniform_buffers[i];
                SDL_PushGPUFragmentUniformData(data->state.command_buffer, ub->slot_index, ub->data, ub->length);
            }
        }
    } else {
        PushFragmentUniforms(data, cmd);
    }

    SDL_GPUBufferBinding buffer_bind;
    SDL_zero(buffer_bind);
    buffer_bind.buffer = data->vertices.buffer;
    buffer_bind.offset = offset;
    SDL_BindGPUVertexBuffers(pass, 0, &buffer_bind, 1);
    PushVertexUniforms(data, cmd);

    SetViewportAndScissor(data);

    SDL_DrawGPUPrimitives(pass, num_verts, 1, 0, 0);
}

static void ReleaseVertexBuffer(GPU_RenderData *data)
{
    if (data->vertices.buffer) {
        SDL_ReleaseGPUBuffer(data->device, data->vertices.buffer);
    }

    if (data->vertices.transfer_buf) {
        SDL_ReleaseGPUTransferBuffer(data->device, data->vertices.transfer_buf);
    }

    data->vertices.buffer_size = 0;
}

static bool InitVertexBuffer(GPU_RenderData *data, Uint32 size)
{
    SDL_GPUBufferCreateInfo bci;
    SDL_zero(bci);
    bci.size = size;
    bci.usage = SDL_GPU_BUFFERUSAGE_VERTEX;

    data->vertices.buffer = SDL_CreateGPUBuffer(data->device, &bci);

    if (!data->vertices.buffer) {
        return false;
    }

    SDL_GPUTransferBufferCreateInfo tbci;
    SDL_zero(tbci);
    tbci.size = size;
    tbci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;

    data->vertices.transfer_buf = SDL_CreateGPUTransferBuffer(data->device, &tbci);

    if (!data->vertices.transfer_buf) {
        return false;
    }

    return true;
}

static bool UploadVertices(GPU_RenderData *data, void *vertices, size_t vertsize)
{
    if (vertsize == 0) {
        return true;
    }

    if (vertsize > data->vertices.buffer_size) {
        ReleaseVertexBuffer(data);
        if (!InitVertexBuffer(data, (Uint32)vertsize)) {
            return false;
        }
    }

    void *staging_buf = SDL_MapGPUTransferBuffer(data->device, data->vertices.transfer_buf, true);
    SDL_memcpy(staging_buf, vertices, vertsize);
    SDL_UnmapGPUTransferBuffer(data->device, data->vertices.transfer_buf);

    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(data->state.command_buffer);

    if (!pass) {
        return false;
    }

    SDL_GPUTransferBufferLocation src;
    SDL_zero(src);
    src.transfer_buffer = data->vertices.transfer_buf;

    SDL_GPUBufferRegion dst;
    SDL_zero(dst);
    dst.buffer = data->vertices.buffer;
    dst.size = (Uint32)vertsize;

    SDL_UploadToGPUBuffer(pass, &src, &dst, true);
    SDL_EndGPUCopyPass(pass);

    return true;
}

// *** FIXME ***
// We might be able to run these data uploads on a separate command buffer
// which would allow us to avoid breaking render passes.
// Honestly I'm a little skeptical of this entire approach,
// we already have a command buffer structure
// so it feels weird to be deferring the operations manually.
// We could also fairly easily run the geometry transformations
// on compute shaders instead of the CPU, which would be a HUGE performance win.
// -cosmonaut
static bool GPU_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    GPU_RenderData *data = (GPU_RenderData *)renderer->internal;

    if (!UploadVertices(data, vertices, vertsize)) {
        return false;
    }

    data->state.color_attachment.load_op = SDL_GPU_LOADOP_LOAD;

    if (renderer->target) {
        GPU_TextureData *tdata = renderer->target->internal;
        data->state.color_attachment.texture = tdata->texture;
    } else {
        data->state.color_attachment.texture = data->backbuffer.texture;
    }

    if (!data->state.color_attachment.texture) {
        return SDL_SetError("Render target texture is NULL");
    }

    while (cmd) {
        switch (cmd->command) {
        case SDL_RENDERCMD_SETDRAWCOLOR:
        {
            data->state.draw_color = GetDrawCmdColor(renderer, cmd);
            break;
        }

        case SDL_RENDERCMD_SETVIEWPORT:
        {
            SDL_Rect *viewport = &cmd->data.viewport.rect;
            data->state.viewport.x = viewport->x;
            data->state.viewport.y = viewport->y;
            data->state.viewport.w = viewport->w;
            data->state.viewport.h = viewport->h;
            break;
        }

        case SDL_RENDERCMD_SETCLIPRECT:
        {
            const SDL_Rect *rect = &cmd->data.cliprect.rect;
            data->state.scissor.x = (int)data->state.viewport.x + rect->x;
            data->state.scissor.y = (int)data->state.viewport.y + rect->y;
            data->state.scissor.w = rect->w;
            data->state.scissor.h = rect->h;
            data->state.scissor_enabled = cmd->data.cliprect.enabled;
            break;
        }

        case SDL_RENDERCMD_CLEAR:
        {
            data->state.color_attachment.clear_color = GetDrawCmdColor(renderer, cmd);
            data->state.color_attachment.load_op = SDL_GPU_LOADOP_CLEAR;
            break;
        }

        case SDL_RENDERCMD_FILL_RECTS: // unused
            break;

        case SDL_RENDERCMD_COPY: // unused
            break;

        case SDL_RENDERCMD_COPY_EX: // unused
            break;

        case SDL_RENDERCMD_DRAW_LINES:
        {
            Uint32 count = (Uint32)cmd->data.draw.count;
            Uint32 offset = (Uint32)cmd->data.draw.first;

            if (count > 2) {
                // joined lines cannot be grouped
                Draw(data, cmd, count, offset, SDL_GPU_PRIMITIVETYPE_LINESTRIP);
            } else {
                // let's group non joined lines
                SDL_RenderCommand *finalcmd = cmd;
                SDL_RenderCommand *nextcmd = cmd->next;
                SDL_BlendMode thisblend = cmd->data.draw.blend;

                while (nextcmd) {
                    const SDL_RenderCommandType nextcmdtype = nextcmd->command;
                    if (nextcmdtype != SDL_RENDERCMD_DRAW_LINES) {
                        break; // can't go any further on this draw call, different render command up next.
                    } else if (nextcmd->data.draw.count != 2) {
                        break; // can't go any further on this draw call, those are joined lines
                    } else if (nextcmd->data.draw.blend != thisblend) {
                        break; // can't go any further on this draw call, different blendmode copy up next.
                    } else {
                        finalcmd = nextcmd; // we can combine copy operations here. Mark this one as the furthest okay command.
                        count += (Uint32)nextcmd->data.draw.count;
                    }
                    nextcmd = nextcmd->next;
                }

                Draw(data, cmd, count, offset, SDL_GPU_PRIMITIVETYPE_LINELIST);
                cmd = finalcmd; // skip any copy commands we just combined in here.
            }
            break;
        }

        case SDL_RENDERCMD_DRAW_POINTS:
        case SDL_RENDERCMD_GEOMETRY:
        {
            /* as long as we have the same copy command in a row, with the
               same texture, we can combine them all into a single draw call. */
            SDL_Texture *thistexture = cmd->data.draw.texture;
            SDL_BlendMode thisblend = cmd->data.draw.blend;
            SDL_ScaleMode thisscalemode = cmd->data.draw.texture_scale_mode;
            SDL_TextureAddressMode thisaddressmode_u = cmd->data.draw.texture_address_mode_u;
            SDL_TextureAddressMode thisaddressmode_v = cmd->data.draw.texture_address_mode_v;
            const SDL_RenderCommandType thiscmdtype = cmd->command;
            SDL_RenderCommand *finalcmd = cmd;
            SDL_RenderCommand *nextcmd = cmd->next;
            Uint32 count = (Uint32)cmd->data.draw.count;
            Uint32 offset = (Uint32)cmd->data.draw.first;

            while (nextcmd) {
                const SDL_RenderCommandType nextcmdtype = nextcmd->command;
                if (nextcmdtype != thiscmdtype) {
                    break; // can't go any further on this draw call, different render command up next.
                } else if (nextcmd->data.draw.texture != thistexture ||
                           nextcmd->data.draw.texture_scale_mode != thisscalemode ||
                           nextcmd->data.draw.texture_address_mode_u != thisaddressmode_u ||
                           nextcmd->data.draw.texture_address_mode_v != thisaddressmode_v ||
                           nextcmd->data.draw.blend != thisblend) {
                    // FIXME should we check address mode too?
                    break; // can't go any further on this draw call, different texture/blendmode copy up next.
                } else {
                    finalcmd = nextcmd; // we can combine copy operations here. Mark this one as the furthest okay command.
                    count += (Uint32)nextcmd->data.draw.count;
                }
                nextcmd = nextcmd->next;
            }

            SDL_GPUPrimitiveType prim = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST; // SDL_RENDERCMD_GEOMETRY
            if (thiscmdtype == SDL_RENDERCMD_DRAW_POINTS) {
                prim = SDL_GPU_PRIMITIVETYPE_POINTLIST;
            }

            Draw(data, cmd, count, offset, prim);

            cmd = finalcmd; // skip any copy commands we just combined in here.
            break;
        }

        case SDL_RENDERCMD_NO_OP:
            break;
        }

        cmd = cmd->next;
    }

    if (data->state.color_attachment.load_op == SDL_GPU_LOADOP_CLEAR) {
        RestartRenderPass(data);
    }

    if (data->state.render_pass) {
        SDL_EndGPURenderPass(data->state.render_pass);
        data->state.render_pass = NULL;
    }

    return true;
}

static SDL_Surface *GPU_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    GPU_RenderData *data = (GPU_RenderData *)renderer->internal;
    SDL_GPUTexture *gpu_tex;
    SDL_PixelFormat pixfmt;

    if (data->state.render_target) {
        SDL_Texture *texture = data->state.render_target;
        GPU_TextureData *texdata = texture->internal;
        gpu_tex = texdata->texture;
        pixfmt = texture->format;
    } else {
        gpu_tex = data->backbuffer.texture;
        pixfmt = TexFormatToPixFormat(data->backbuffer.format);

        if (pixfmt == SDL_PIXELFORMAT_UNKNOWN) {
            SDL_SetError("Unsupported backbuffer format");
            return NULL;
        }
    }

    Uint32 bpp = SDL_BYTESPERPIXEL(pixfmt);
    size_t row_size, image_size;

    if (!SDL_size_mul_check_overflow(rect->w, bpp, &row_size) ||
        !SDL_size_mul_check_overflow(rect->h, row_size, &image_size)) {
        SDL_SetError("read size overflow");
        return NULL;
    }

    SDL_Surface *surface = SDL_CreateSurface(rect->w, rect->h, pixfmt);

    if (!surface) {
        return NULL;
    }

    SDL_GPUTransferBufferCreateInfo tbci;
    SDL_zero(tbci);
    tbci.size = (Uint32)image_size;
    tbci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;

    SDL_GPUTransferBuffer *tbuf = SDL_CreateGPUTransferBuffer(data->device, &tbci);

    if (!tbuf) {
        return NULL;
    }

    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(data->state.command_buffer);

    SDL_GPUTextureRegion src;
    SDL_zero(src);
    src.texture = gpu_tex;
    src.x = rect->x;
    src.y = rect->y;
    src.w = rect->w;
    src.h = rect->h;
    src.d = 1;

    SDL_GPUTextureTransferInfo dst;
    SDL_zero(dst);
    dst.transfer_buffer = tbuf;
    dst.rows_per_layer = rect->h;
    dst.pixels_per_row = rect->w;

    SDL_DownloadFromGPUTexture(pass, &src, &dst);
    SDL_EndGPUCopyPass(pass);

    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(data->state.command_buffer);
    SDL_WaitForGPUFences(data->device, true, &fence, 1);
    SDL_ReleaseGPUFence(data->device, fence);
    data->state.command_buffer = SDL_AcquireGPUCommandBuffer(data->device);

    void *mapped_tbuf = SDL_MapGPUTransferBuffer(data->device, tbuf, false);

    if ((size_t)surface->pitch == row_size) {
        SDL_memcpy(surface->pixels, mapped_tbuf, image_size);
    } else {
        Uint8 *input = mapped_tbuf;
        Uint8 *output = surface->pixels;

        for (int row = 0; row < rect->h; ++row) {
            SDL_memcpy(output, input, row_size);
            output += surface->pitch;
            input += row_size;
        }
    }

    SDL_UnmapGPUTransferBuffer(data->device, tbuf);
    SDL_ReleaseGPUTransferBuffer(data->device, tbuf);

    return surface;
}

static bool CreateBackbuffer(GPU_RenderData *data, Uint32 w, Uint32 h, SDL_GPUTextureFormat fmt)
{
    SDL_GPUTextureCreateInfo tci;
    SDL_zero(tci);
    tci.width = w;
    tci.height = h;
    tci.format = fmt;
    tci.layer_count_or_depth = 1;
    tci.num_levels = 1;
    tci.sample_count = SDL_GPU_SAMPLECOUNT_1;
    tci.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;

    data->backbuffer.texture = SDL_CreateGPUTexture(data->device, &tci);
    data->backbuffer.width = w;
    data->backbuffer.height = h;
    data->backbuffer.format = fmt;

    if (!data->backbuffer.texture) {
        return false;
    }

    return true;
}

static bool GPU_RenderPresent(SDL_Renderer *renderer)
{
    GPU_RenderData *data = (GPU_RenderData *)renderer->internal;

    SDL_GPUTexture *swapchain;
    Uint32 swapchain_texture_width, swapchain_texture_height;
    bool result = SDL_WaitAndAcquireGPUSwapchainTexture(data->state.command_buffer, renderer->window, &swapchain, &swapchain_texture_width, &swapchain_texture_height);

    if (!result) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to acquire swapchain texture: %s", SDL_GetError());
    }

    if (swapchain != NULL) {
        SDL_GPUBlitInfo blit_info;
        SDL_zero(blit_info);

        blit_info.source.texture = data->backbuffer.texture;
        blit_info.source.w = data->backbuffer.width;
        blit_info.source.h = data->backbuffer.height;
        blit_info.destination.texture = swapchain;
        blit_info.destination.w = swapchain_texture_width;
        blit_info.destination.h = swapchain_texture_height;
        blit_info.load_op = SDL_GPU_LOADOP_DONT_CARE;
        blit_info.filter = SDL_GPU_FILTER_LINEAR;

        SDL_BlitGPUTexture(data->state.command_buffer, &blit_info);

        SDL_SubmitGPUCommandBuffer(data->state.command_buffer);

        if (swapchain_texture_width != data->backbuffer.width || swapchain_texture_height != data->backbuffer.height) {
            SDL_ReleaseGPUTexture(data->device, data->backbuffer.texture);
            CreateBackbuffer(data, swapchain_texture_width, swapchain_texture_height, SDL_GetGPUSwapchainTextureFormat(data->device, renderer->window));
        }
    } else {
        SDL_SubmitGPUCommandBuffer(data->state.command_buffer);
    }

    data->state.command_buffer = SDL_AcquireGPUCommandBuffer(data->device);

    return true;
}

static void GPU_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    GPU_RenderData *renderdata = (GPU_RenderData *)renderer->internal;
    GPU_TextureData *data = (GPU_TextureData *)texture->internal;

    if (renderdata->state.render_target == texture) {
        renderdata->state.render_target = NULL;
    }

    if (!data) {
        return;
    }

    SDL_ReleaseGPUTexture(renderdata->device, data->texture);
    SDL_free(data->pixels);
    SDL_free(data);
    texture->internal = NULL;
}

static void GPU_DestroyRenderer(SDL_Renderer *renderer)
{
    GPU_RenderData *data = (GPU_RenderData *)renderer->internal;

    if (!data) {
        return;
    }

    if (data->state.command_buffer) {
        SDL_SubmitGPUCommandBuffer(data->state.command_buffer);
        data->state.command_buffer = NULL;
    }

    for (Uint32 i = 0; i < SDL_arraysize(data->samplers); ++i) {
        if (data->samplers[i]) {
            SDL_ReleaseGPUSampler(data->device, data->samplers[i]);
        }
    }

    if (data->backbuffer.texture) {
        SDL_ReleaseGPUTexture(data->device, data->backbuffer.texture);
    }

    if (renderer->window && data->device) {
        SDL_ReleaseWindowFromGPUDevice(data->device, renderer->window);
    }

    ReleaseVertexBuffer(data);
    GPU_DestroyPipelineCache(&data->pipeline_cache);

    if (data->device) {
        GPU_ReleaseShaders(&data->shaders, data->device);
        SDL_DestroyGPUDevice(data->device);
    }

    SDL_free(data);
}

static bool ChoosePresentMode(SDL_GPUDevice *device, SDL_Window *window, const int vsync, SDL_GPUPresentMode *out_mode)
{
    SDL_GPUPresentMode mode;

    switch (vsync) {
    case 0:
        mode = SDL_GPU_PRESENTMODE_MAILBOX;

        if (!SDL_WindowSupportsGPUPresentMode(device, window, mode)) {
            mode = SDL_GPU_PRESENTMODE_IMMEDIATE;

            if (!SDL_WindowSupportsGPUPresentMode(device, window, mode)) {
                mode = SDL_GPU_PRESENTMODE_VSYNC;
            }
        }

        // FIXME should we return an error if both mailbox and immediate fail?
        break;

    case 1:
        mode = SDL_GPU_PRESENTMODE_VSYNC;
        break;

    default:
        return SDL_Unsupported();
    }

    *out_mode = mode;
    return true;
}

static bool GPU_SetVSync(SDL_Renderer *renderer, const int vsync)
{
    GPU_RenderData *data = (GPU_RenderData *)renderer->internal;
    SDL_GPUPresentMode mode = SDL_GPU_PRESENTMODE_VSYNC;

    if (!ChoosePresentMode(data->device, renderer->window, vsync, &mode)) {
        return false;
    }

    if (mode != data->swapchain.present_mode) {
        // XXX returns bool instead of SDL-style error code
        if (SDL_SetGPUSwapchainParameters(data->device, renderer->window, data->swapchain.composition, mode)) {
            data->swapchain.present_mode = mode;
            return true;
        } else {
            return false;
        }
    }

    return true;
}

static bool GPU_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    GPU_RenderData *data = NULL;

    SDL_SetupRendererColorspace(renderer, create_props);

    if (renderer->output_colorspace != SDL_COLORSPACE_SRGB) {
        // TODO
        return SDL_SetError("Unsupported output colorspace");
    }

    data = (GPU_RenderData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }

    renderer->SupportsBlendMode = GPU_SupportsBlendMode;
    renderer->CreateTexture = GPU_CreateTexture;
    renderer->UpdateTexture = GPU_UpdateTexture;
    renderer->LockTexture = GPU_LockTexture;
    renderer->UnlockTexture = GPU_UnlockTexture;
    renderer->SetRenderTarget = GPU_SetRenderTarget;
    renderer->QueueSetViewport = GPU_QueueNoOp;
    renderer->QueueSetDrawColor = GPU_QueueNoOp;
    renderer->QueueDrawPoints = GPU_QueueDrawPoints;
    renderer->QueueDrawLines = GPU_QueueDrawPoints; // lines and points queue vertices the same way.
    renderer->QueueGeometry = GPU_QueueGeometry;
    renderer->InvalidateCachedState = GPU_InvalidateCachedState;
    renderer->RunCommandQueue = GPU_RunCommandQueue;
    renderer->RenderReadPixels = GPU_RenderReadPixels;
    renderer->RenderPresent = GPU_RenderPresent;
    renderer->DestroyTexture = GPU_DestroyTexture;
    renderer->DestroyRenderer = GPU_DestroyRenderer;
    renderer->SetVSync = GPU_SetVSync;
    renderer->internal = data;
    renderer->window = window;
    renderer->name = GPU_RenderDriver.name;

    bool debug = SDL_GetBooleanProperty(create_props, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, false);
    bool lowpower = SDL_GetBooleanProperty(create_props, SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN, false);

    // Prefer environment variables/hints if they exist, otherwise defer to properties
    debug = SDL_GetHintBoolean(SDL_HINT_RENDER_GPU_DEBUG, debug);
    lowpower = SDL_GetHintBoolean(SDL_HINT_RENDER_GPU_LOW_POWER, lowpower);

    SDL_SetBooleanProperty(create_props, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, debug);
    SDL_SetBooleanProperty(create_props, SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN, lowpower);

    GPU_FillSupportedShaderFormats(create_props);
    data->device = SDL_CreateGPUDeviceWithProperties(create_props);

    if (!data->device) {
        return false;
    }

    if (!GPU_InitShaders(&data->shaders, data->device)) {
        return false;
    }

    if (!GPU_InitPipelineCache(&data->pipeline_cache, data->device)) {
        return false;
    }

    // XXX what's a good initial size?
    if (!InitVertexBuffer(data, 1 << 16)) {
        return false;
    }

    if (!SDL_ClaimWindowForGPUDevice(data->device, window)) {
        return false;
    }

    data->swapchain.composition = SDL_GPU_SWAPCHAINCOMPOSITION_SDR;
    data->swapchain.present_mode = SDL_GPU_PRESENTMODE_VSYNC;

    int vsync = (int)SDL_GetNumberProperty(create_props, SDL_PROP_RENDERER_CREATE_PRESENT_VSYNC_NUMBER, 0);
    ChoosePresentMode(data->device, window, vsync, &data->swapchain.present_mode);

    SDL_SetGPUSwapchainParameters(data->device, window, data->swapchain.composition, data->swapchain.present_mode);

    SDL_SetGPUAllowedFramesInFlight(data->device, 1);

    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_BGRA32);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBA32);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_BGRX32);
    SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBX32);

    SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 16384);

    data->state.draw_color.r = 1.0f;
    data->state.draw_color.g = 1.0f;
    data->state.draw_color.b = 1.0f;
    data->state.draw_color.a = 1.0f;
    data->state.viewport.min_depth = 0;
    data->state.viewport.max_depth = 1;
    data->state.command_buffer = SDL_AcquireGPUCommandBuffer(data->device);

    int w, h;
    SDL_GetWindowSizeInPixels(window, &w, &h);

    if (!CreateBackbuffer(data, w, h, SDL_GetGPUSwapchainTextureFormat(data->device, window))) {
        return false;
    }

    SDL_SetPointerProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_GPU_DEVICE_POINTER, data->device);

    return true;
}

SDL_RenderDriver GPU_RenderDriver = {
    GPU_CreateRenderer, "gpu"
};

#endif // SDL_VIDEO_RENDER_GPU

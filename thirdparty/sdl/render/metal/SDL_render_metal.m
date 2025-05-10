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

#ifdef SDL_VIDEO_RENDER_METAL

#include "../SDL_sysrender.h"
#include "../../video/SDL_pixels_c.h"

#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#ifdef SDL_VIDEO_DRIVER_COCOA
#import <AppKit/NSWindow.h>
#import <AppKit/NSView.h>
#endif
#ifdef SDL_VIDEO_DRIVER_UIKIT
#import <UIKit/UIKit.h>
#endif

// Regenerate these with build-metal-shaders.sh
#ifdef SDL_PLATFORM_MACOS
#include "SDL_shaders_metal_macos.h"
#elif defined(SDL_PLATFORM_TVOS)
#if TARGET_OS_SIMULATOR
#include "SDL_shaders_metal_tvsimulator.h"
#else
#include "SDL_shaders_metal_tvos.h"
#endif
#else
#if TARGET_OS_SIMULATOR
#include "SDL_shaders_metal_iphonesimulator.h"
#else
#include "SDL_shaders_metal_ios.h"
#endif
#endif

// Apple Metal renderer implementation

// macOS requires constants in a buffer to have a 256 byte alignment.
// Use native type alignments from https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
#if defined(SDL_PLATFORM_MACOS) || TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
#define CONSTANT_ALIGN(x) (256)
#else
#define CONSTANT_ALIGN(x) (x < 4 ? 4 : x)
#endif

#define DEVICE_ALIGN(x) (x < 4 ? 4 : x)

#define ALIGN_CONSTANTS(align, size) ((size + CONSTANT_ALIGN(align) - 1) & (~(CONSTANT_ALIGN(align) - 1)))

static const size_t CONSTANTS_OFFSET_INVALID = 0xFFFFFFFF;
static const size_t CONSTANTS_OFFSET_IDENTITY = 0;
static const size_t CONSTANTS_OFFSET_HALF_PIXEL_TRANSFORM = ALIGN_CONSTANTS(16, CONSTANTS_OFFSET_IDENTITY + sizeof(float) * 16);
static const size_t CONSTANTS_OFFSET_DECODE_BT601_LIMITED = ALIGN_CONSTANTS(16, CONSTANTS_OFFSET_HALF_PIXEL_TRANSFORM + sizeof(float) * 16);
static const size_t CONSTANTS_OFFSET_DECODE_BT601_FULL = ALIGN_CONSTANTS(16, CONSTANTS_OFFSET_DECODE_BT601_LIMITED + sizeof(float) * 4 * 4);
static const size_t CONSTANTS_OFFSET_DECODE_BT709_LIMITED = ALIGN_CONSTANTS(16, CONSTANTS_OFFSET_DECODE_BT601_FULL + sizeof(float) * 4 * 4);
static const size_t CONSTANTS_OFFSET_DECODE_BT709_FULL = ALIGN_CONSTANTS(16, CONSTANTS_OFFSET_DECODE_BT709_LIMITED + sizeof(float) * 4 * 4);
static const size_t CONSTANTS_OFFSET_DECODE_BT2020_LIMITED = ALIGN_CONSTANTS(16, CONSTANTS_OFFSET_DECODE_BT709_FULL + sizeof(float) * 4 * 4);
static const size_t CONSTANTS_OFFSET_DECODE_BT2020_FULL = ALIGN_CONSTANTS(16, CONSTANTS_OFFSET_DECODE_BT2020_LIMITED + sizeof(float) * 4 * 4);
static const size_t CONSTANTS_LENGTH = CONSTANTS_OFFSET_DECODE_BT2020_FULL + sizeof(float) * 4 * 4;

typedef enum SDL_MetalVertexFunction
{
    SDL_METAL_VERTEX_SOLID,
    SDL_METAL_VERTEX_COPY,
} SDL_MetalVertexFunction;

typedef enum SDL_MetalFragmentFunction
{
    SDL_METAL_FRAGMENT_SOLID = 0,
    SDL_METAL_FRAGMENT_COPY,
    SDL_METAL_FRAGMENT_YUV,
    SDL_METAL_FRAGMENT_NV12,
    SDL_METAL_FRAGMENT_COUNT,
} SDL_MetalFragmentFunction;

typedef struct METAL_PipelineState
{
    SDL_BlendMode blendMode;
    void *pipe;
} METAL_PipelineState;

typedef struct METAL_PipelineCache
{
    METAL_PipelineState *states;
    int count;
    SDL_MetalVertexFunction vertexFunction;
    SDL_MetalFragmentFunction fragmentFunction;
    MTLPixelFormat renderTargetFormat;
    const char *label;
} METAL_PipelineCache;

/* Each shader combination used by drawing functions has a separate pipeline
 * cache, and we have a separate list of caches for each render target pixel
 * format. This is more efficient than iterating over a global cache to find
 * the pipeline based on the specified shader combination and RT pixel format,
 * since we know what the RT pixel format is when we set the render target, and
 * we know what the shader combination is inside each drawing function's code. */
typedef struct METAL_ShaderPipelines
{
    MTLPixelFormat renderTargetFormat;
    METAL_PipelineCache caches[SDL_METAL_FRAGMENT_COUNT];
} METAL_ShaderPipelines;

@interface SDL3METAL_RenderData : NSObject
@property(nonatomic, retain) id<MTLDevice> mtldevice;
@property(nonatomic, retain) id<MTLCommandQueue> mtlcmdqueue;
@property(nonatomic, retain) id<MTLCommandBuffer> mtlcmdbuffer;
@property(nonatomic, retain) id<MTLRenderCommandEncoder> mtlcmdencoder;
@property(nonatomic, retain) id<MTLLibrary> mtllibrary;
@property(nonatomic, retain) id<CAMetalDrawable> mtlbackbuffer;
@property(nonatomic, retain) NSMutableDictionary<NSNumber *, id<MTLSamplerState>> *mtlsamplers;
@property(nonatomic, retain) id<MTLBuffer> mtlbufconstants;
@property(nonatomic, retain) id<MTLBuffer> mtlbufquadindices;
@property(nonatomic, assign) SDL_MetalView mtlview;
@property(nonatomic, retain) CAMetalLayer *mtllayer;
@property(nonatomic, retain) MTLRenderPassDescriptor *mtlpassdesc;
@property(nonatomic, assign) METAL_ShaderPipelines *activepipelines;
@property(nonatomic, assign) METAL_ShaderPipelines *allpipelines;
@property(nonatomic, assign) int pipelinescount;
@end

@implementation SDL3METAL_RenderData
@end

@interface SDL3METAL_TextureData : NSObject
@property(nonatomic, retain) id<MTLTexture> mtltexture;
@property(nonatomic, retain) id<MTLTexture> mtltextureUv;
@property(nonatomic, assign) SDL_MetalFragmentFunction fragmentFunction;
#ifdef SDL_HAVE_YUV
@property(nonatomic, assign) BOOL yuv;
@property(nonatomic, assign) BOOL nv12;
@property(nonatomic, assign) size_t conversionBufferOffset;
#endif
@property(nonatomic, assign) BOOL hasdata;
@property(nonatomic, retain) id<MTLBuffer> lockedbuffer;
@property(nonatomic, assign) SDL_Rect lockedrect;
@end

@implementation SDL3METAL_TextureData
@end

static const MTLBlendOperation invalidBlendOperation = (MTLBlendOperation)0xFFFFFFFF;
static const MTLBlendFactor invalidBlendFactor = (MTLBlendFactor)0xFFFFFFFF;

static MTLBlendOperation GetBlendOperation(SDL_BlendOperation operation)
{
    switch (operation) {
    case SDL_BLENDOPERATION_ADD:
        return MTLBlendOperationAdd;
    case SDL_BLENDOPERATION_SUBTRACT:
        return MTLBlendOperationSubtract;
    case SDL_BLENDOPERATION_REV_SUBTRACT:
        return MTLBlendOperationReverseSubtract;
    case SDL_BLENDOPERATION_MINIMUM:
        return MTLBlendOperationMin;
    case SDL_BLENDOPERATION_MAXIMUM:
        return MTLBlendOperationMax;
    default:
        return invalidBlendOperation;
    }
}

static MTLBlendFactor GetBlendFactor(SDL_BlendFactor factor)
{
    switch (factor) {
    case SDL_BLENDFACTOR_ZERO:
        return MTLBlendFactorZero;
    case SDL_BLENDFACTOR_ONE:
        return MTLBlendFactorOne;
    case SDL_BLENDFACTOR_SRC_COLOR:
        return MTLBlendFactorSourceColor;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return MTLBlendFactorOneMinusSourceColor;
    case SDL_BLENDFACTOR_SRC_ALPHA:
        return MTLBlendFactorSourceAlpha;
    case SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return MTLBlendFactorOneMinusSourceAlpha;
    case SDL_BLENDFACTOR_DST_COLOR:
        return MTLBlendFactorDestinationColor;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return MTLBlendFactorOneMinusDestinationColor;
    case SDL_BLENDFACTOR_DST_ALPHA:
        return MTLBlendFactorDestinationAlpha;
    case SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return MTLBlendFactorOneMinusDestinationAlpha;
    default:
        return invalidBlendFactor;
    }
}

static NSString *GetVertexFunctionName(SDL_MetalVertexFunction function)
{
    switch (function) {
    case SDL_METAL_VERTEX_SOLID:
        return @"SDL_Solid_vertex";
    case SDL_METAL_VERTEX_COPY:
        return @"SDL_Copy_vertex";
    default:
        return nil;
    }
}

static NSString *GetFragmentFunctionName(SDL_MetalFragmentFunction function)
{
    switch (function) {
    case SDL_METAL_FRAGMENT_SOLID:
        return @"SDL_Solid_fragment";
    case SDL_METAL_FRAGMENT_COPY:
        return @"SDL_Copy_fragment";
    case SDL_METAL_FRAGMENT_YUV:
        return @"SDL_YUV_fragment";
    case SDL_METAL_FRAGMENT_NV12:
        return @"SDL_NV12_fragment";
    default:
        return nil;
    }
}

static id<MTLRenderPipelineState> MakePipelineState(SDL3METAL_RenderData *data, METAL_PipelineCache *cache,
                                                    NSString *blendlabel, SDL_BlendMode blendmode)
{
    MTLRenderPipelineDescriptor *mtlpipedesc;
    MTLVertexDescriptor *vertdesc;
    MTLRenderPipelineColorAttachmentDescriptor *rtdesc;
    NSError *err = nil;
    id<MTLRenderPipelineState> state;
    METAL_PipelineState pipeline;
    METAL_PipelineState *states;

    id<MTLFunction> mtlvertfn = [data.mtllibrary newFunctionWithName:GetVertexFunctionName(cache->vertexFunction)];
    id<MTLFunction> mtlfragfn = [data.mtllibrary newFunctionWithName:GetFragmentFunctionName(cache->fragmentFunction)];
    SDL_assert(mtlvertfn != nil);
    SDL_assert(mtlfragfn != nil);

    mtlpipedesc = [[MTLRenderPipelineDescriptor alloc] init];
    mtlpipedesc.vertexFunction = mtlvertfn;
    mtlpipedesc.fragmentFunction = mtlfragfn;

    vertdesc = [MTLVertexDescriptor vertexDescriptor];

    switch (cache->vertexFunction) {
    case SDL_METAL_VERTEX_SOLID:
        // position (float2), color (float4)
        vertdesc.layouts[0].stride = sizeof(float) * 2 + sizeof(float) * 4;
        vertdesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

        vertdesc.attributes[0].format = MTLVertexFormatFloat2;
        vertdesc.attributes[0].offset = 0;
        vertdesc.attributes[0].bufferIndex = 0;

        vertdesc.attributes[1].format = MTLVertexFormatFloat4;
        vertdesc.attributes[1].offset = sizeof(float) * 2;
        vertdesc.attributes[1].bufferIndex = 0;

        break;
    case SDL_METAL_VERTEX_COPY:
        // position (float2), color (float4), texcoord (float2)
        vertdesc.layouts[0].stride = sizeof(float) * 2 + sizeof(float) * 4 + sizeof(float) * 2;
        vertdesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

        vertdesc.attributes[0].format = MTLVertexFormatFloat2;
        vertdesc.attributes[0].offset = 0;
        vertdesc.attributes[0].bufferIndex = 0;

        vertdesc.attributes[1].format = MTLVertexFormatFloat4;
        vertdesc.attributes[1].offset = sizeof(float) * 2;
        vertdesc.attributes[1].bufferIndex = 0;

        vertdesc.attributes[2].format = MTLVertexFormatFloat2;
        vertdesc.attributes[2].offset = sizeof(float) * 2 + sizeof(float) * 4;
        vertdesc.attributes[2].bufferIndex = 0;
        break;
    }

    mtlpipedesc.vertexDescriptor = vertdesc;

    rtdesc = mtlpipedesc.colorAttachments[0];
    rtdesc.pixelFormat = cache->renderTargetFormat;

    if (blendmode != SDL_BLENDMODE_NONE) {
        rtdesc.blendingEnabled = YES;
        rtdesc.sourceRGBBlendFactor = GetBlendFactor(SDL_GetBlendModeSrcColorFactor(blendmode));
        rtdesc.destinationRGBBlendFactor = GetBlendFactor(SDL_GetBlendModeDstColorFactor(blendmode));
        rtdesc.rgbBlendOperation = GetBlendOperation(SDL_GetBlendModeColorOperation(blendmode));
        rtdesc.sourceAlphaBlendFactor = GetBlendFactor(SDL_GetBlendModeSrcAlphaFactor(blendmode));
        rtdesc.destinationAlphaBlendFactor = GetBlendFactor(SDL_GetBlendModeDstAlphaFactor(blendmode));
        rtdesc.alphaBlendOperation = GetBlendOperation(SDL_GetBlendModeAlphaOperation(blendmode));
    } else {
        rtdesc.blendingEnabled = NO;
    }

    mtlpipedesc.label = [@(cache->label) stringByAppendingString:blendlabel];

    state = [data.mtldevice newRenderPipelineStateWithDescriptor:mtlpipedesc error:&err];
    SDL_assert(err == nil);

    pipeline.blendMode = blendmode;
    pipeline.pipe = (void *)CFBridgingRetain(state);

    states = SDL_realloc(cache->states, (cache->count + 1) * sizeof(pipeline));

    if (states) {
        states[cache->count++] = pipeline;
        cache->states = states;
        return (__bridge id<MTLRenderPipelineState>)pipeline.pipe;
    } else {
        CFBridgingRelease(pipeline.pipe);
        return NULL;
    }
}

static void MakePipelineCache(SDL3METAL_RenderData *data, METAL_PipelineCache *cache, const char *label,
                              MTLPixelFormat rtformat, SDL_MetalVertexFunction vertfn, SDL_MetalFragmentFunction fragfn)
{
    SDL_zerop(cache);

    cache->vertexFunction = vertfn;
    cache->fragmentFunction = fragfn;
    cache->renderTargetFormat = rtformat;
    cache->label = label;

    /* Create pipeline states for the default blend modes. Custom blend modes
     * will be added to the cache on-demand. */
    MakePipelineState(data, cache, @" (blend=none)", SDL_BLENDMODE_NONE);
    MakePipelineState(data, cache, @" (blend=blend)", SDL_BLENDMODE_BLEND);
    MakePipelineState(data, cache, @" (blend=add)", SDL_BLENDMODE_ADD);
    MakePipelineState(data, cache, @" (blend=mod)", SDL_BLENDMODE_MOD);
    MakePipelineState(data, cache, @" (blend=mul)", SDL_BLENDMODE_MUL);
}

static void DestroyPipelineCache(METAL_PipelineCache *cache)
{
    if (cache != NULL) {
        for (int i = 0; i < cache->count; i++) {
            CFBridgingRelease(cache->states[i].pipe);
        }

        SDL_free(cache->states);
    }
}

void MakeShaderPipelines(SDL3METAL_RenderData *data, METAL_ShaderPipelines *pipelines, MTLPixelFormat rtformat)
{
    SDL_zerop(pipelines);

    pipelines->renderTargetFormat = rtformat;

    MakePipelineCache(data, &pipelines->caches[SDL_METAL_FRAGMENT_SOLID], "SDL primitives pipeline", rtformat, SDL_METAL_VERTEX_SOLID, SDL_METAL_FRAGMENT_SOLID);
    MakePipelineCache(data, &pipelines->caches[SDL_METAL_FRAGMENT_COPY], "SDL copy pipeline", rtformat, SDL_METAL_VERTEX_COPY, SDL_METAL_FRAGMENT_COPY);
    MakePipelineCache(data, &pipelines->caches[SDL_METAL_FRAGMENT_YUV], "SDL YUV pipeline", rtformat, SDL_METAL_VERTEX_COPY, SDL_METAL_FRAGMENT_YUV);
    MakePipelineCache(data, &pipelines->caches[SDL_METAL_FRAGMENT_NV12], "SDL NV12 pipeline", rtformat, SDL_METAL_VERTEX_COPY, SDL_METAL_FRAGMENT_NV12);
}

static METAL_ShaderPipelines *ChooseShaderPipelines(SDL3METAL_RenderData *data, MTLPixelFormat rtformat)
{
    METAL_ShaderPipelines *allpipelines = data.allpipelines;
    int count = data.pipelinescount;

    for (int i = 0; i < count; i++) {
        if (allpipelines[i].renderTargetFormat == rtformat) {
            return &allpipelines[i];
        }
    }

    allpipelines = SDL_realloc(allpipelines, (count + 1) * sizeof(METAL_ShaderPipelines));

    if (allpipelines == NULL) {
        return NULL;
    }

    MakeShaderPipelines(data, &allpipelines[count], rtformat);

    data.allpipelines = allpipelines;
    data.pipelinescount = count + 1;

    return &data.allpipelines[count];
}

static void DestroyAllPipelines(METAL_ShaderPipelines *allpipelines, int count)
{
    if (allpipelines != NULL) {
        for (int i = 0; i < count; i++) {
            for (int cache = 0; cache < SDL_METAL_FRAGMENT_COUNT; cache++) {
                DestroyPipelineCache(&allpipelines[i].caches[cache]);
            }
        }

        SDL_free(allpipelines);
    }
}

static inline id<MTLRenderPipelineState> ChoosePipelineState(SDL3METAL_RenderData *data, METAL_ShaderPipelines *pipelines, SDL_MetalFragmentFunction fragfn, SDL_BlendMode blendmode)
{
    METAL_PipelineCache *cache = &pipelines->caches[fragfn];

    for (int i = 0; i < cache->count; i++) {
        if (cache->states[i].blendMode == blendmode) {
            return (__bridge id<MTLRenderPipelineState>)cache->states[i].pipe;
        }
    }

    return MakePipelineState(data, cache, [NSString stringWithFormat:@" (blend=custom 0x%x)", blendmode], blendmode);
}

static bool METAL_ActivateRenderCommandEncoder(SDL_Renderer *renderer, MTLLoadAction load, MTLClearColor *clear_color, id<MTLBuffer> vertex_buffer)
{
    SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;

    /* Our SetRenderTarget just signals that the next render operation should
     * set up a new render pass. This is where that work happens. */
    if (data.mtlcmdencoder == nil) {
        id<MTLTexture> mtltexture = nil;

        if (renderer->target != NULL) {
            SDL3METAL_TextureData *texdata = (__bridge SDL3METAL_TextureData *)renderer->target->internal;
            mtltexture = texdata.mtltexture;
        } else {
            if (data.mtlbackbuffer == nil) {
                /* The backbuffer's contents aren't guaranteed to persist after
                 * presenting, so we can leave it undefined when loading it. */
                data.mtlbackbuffer = [data.mtllayer nextDrawable];
                if (load == MTLLoadActionLoad) {
                    load = MTLLoadActionDontCare;
                }
            }
            if (data.mtlbackbuffer != nil) {
                mtltexture = data.mtlbackbuffer.texture;
            }
        }

        /* mtltexture can be nil here if macOS refused to give us a drawable,
           which apparently can happen for minimized windows, etc. */
        if (mtltexture == nil) {
            return false;
        }

        if (load == MTLLoadActionClear) {
            SDL_assert(clear_color != NULL);
            data.mtlpassdesc.colorAttachments[0].clearColor = *clear_color;
        }

        data.mtlpassdesc.colorAttachments[0].loadAction = load;
        data.mtlpassdesc.colorAttachments[0].texture = mtltexture;

        data.mtlcmdbuffer = [data.mtlcmdqueue commandBuffer];
        data.mtlcmdencoder = [data.mtlcmdbuffer renderCommandEncoderWithDescriptor:data.mtlpassdesc];

        if (data.mtlbackbuffer != nil && mtltexture == data.mtlbackbuffer.texture) {
            data.mtlcmdencoder.label = @"SDL metal renderer backbuffer";
        } else {
            data.mtlcmdencoder.label = @"SDL metal renderer render target";
        }

        /* Set up buffer bindings for positions, texcoords, and color once here,
         * the offsets are adjusted in the code that uses them. */
        if (vertex_buffer != nil) {
            [data.mtlcmdencoder setVertexBuffer:vertex_buffer offset:0 atIndex:0];
            [data.mtlcmdencoder setFragmentBuffer:vertex_buffer offset:0 atIndex:0];
        }

        data.activepipelines = ChooseShaderPipelines(data, mtltexture.pixelFormat);

        // make sure this has a definite place in the queue. This way it will
        //  execute reliably whether the app tries to make its own command buffers
        //  or whatever. This means we can _always_ batch rendering commands!
        [data.mtlcmdbuffer enqueue];
    }

    return true;
}

static void METAL_WindowEvent(SDL_Renderer *renderer, const SDL_WindowEvent *event)
{
}

static bool METAL_GetOutputSize(SDL_Renderer *renderer, int *w, int *h)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        if (w) {
            *w = (int)data.mtllayer.drawableSize.width;
        }
        if (h) {
            *h = (int)data.mtllayer.drawableSize.height;
        }
        return true;
    }
}

static bool METAL_SupportsBlendMode(SDL_Renderer *renderer, SDL_BlendMode blendMode)
{
    SDL_BlendFactor srcColorFactor = SDL_GetBlendModeSrcColorFactor(blendMode);
    SDL_BlendFactor srcAlphaFactor = SDL_GetBlendModeSrcAlphaFactor(blendMode);
    SDL_BlendOperation colorOperation = SDL_GetBlendModeColorOperation(blendMode);
    SDL_BlendFactor dstColorFactor = SDL_GetBlendModeDstColorFactor(blendMode);
    SDL_BlendFactor dstAlphaFactor = SDL_GetBlendModeDstAlphaFactor(blendMode);
    SDL_BlendOperation alphaOperation = SDL_GetBlendModeAlphaOperation(blendMode);

    if (GetBlendFactor(srcColorFactor) == invalidBlendFactor ||
        GetBlendFactor(srcAlphaFactor) == invalidBlendFactor ||
        GetBlendOperation(colorOperation) == invalidBlendOperation ||
        GetBlendFactor(dstColorFactor) == invalidBlendFactor ||
        GetBlendFactor(dstAlphaFactor) == invalidBlendFactor ||
        GetBlendOperation(alphaOperation) == invalidBlendOperation) {
        return false;
    }
    return true;
}

size_t GetBT601ConversionMatrix(SDL_Colorspace colorspace)
{
    switch (SDL_COLORSPACERANGE(colorspace)) {
    case SDL_COLOR_RANGE_LIMITED:
    case SDL_COLOR_RANGE_UNKNOWN:
        return CONSTANTS_OFFSET_DECODE_BT601_LIMITED;
    case SDL_COLOR_RANGE_FULL:
        return CONSTANTS_OFFSET_DECODE_BT601_FULL;
    default:
        break;
    }
    return 0;
}

size_t GetBT709ConversionMatrix(SDL_Colorspace colorspace)
{
    switch (SDL_COLORSPACERANGE(colorspace)) {
    case SDL_COLOR_RANGE_LIMITED:
    case SDL_COLOR_RANGE_UNKNOWN:
        return CONSTANTS_OFFSET_DECODE_BT709_LIMITED;
    case SDL_COLOR_RANGE_FULL:
        return CONSTANTS_OFFSET_DECODE_BT709_FULL;
    default:
        break;
    }
    return 0;
}

size_t GetBT2020ConversionMatrix(SDL_Colorspace colorspace)
{
    switch (SDL_COLORSPACERANGE(colorspace)) {
    case SDL_COLOR_RANGE_LIMITED:
    case SDL_COLOR_RANGE_UNKNOWN:
        return CONSTANTS_OFFSET_DECODE_BT2020_LIMITED;
    case SDL_COLOR_RANGE_FULL:
        return CONSTANTS_OFFSET_DECODE_BT2020_FULL;
    default:
        break;
    }
    return 0;
}

size_t GetYCbCRtoRGBConversionMatrix(SDL_Colorspace colorspace, int w, int h, int bits_per_pixel)
{
    const int YUV_SD_THRESHOLD = 576;

    switch (SDL_COLORSPACEMATRIX(colorspace)) {
    case SDL_MATRIX_COEFFICIENTS_BT470BG:
    case SDL_MATRIX_COEFFICIENTS_BT601:
        return GetBT601ConversionMatrix(colorspace);

    case SDL_MATRIX_COEFFICIENTS_BT709:
        return GetBT709ConversionMatrix(colorspace);

    case SDL_MATRIX_COEFFICIENTS_BT2020_NCL:
        return GetBT2020ConversionMatrix(colorspace);

    case SDL_MATRIX_COEFFICIENTS_UNSPECIFIED:
        switch (bits_per_pixel) {
        case 8:
            if (h <= YUV_SD_THRESHOLD) {
                return GetBT601ConversionMatrix(colorspace);
            } else {
                return GetBT709ConversionMatrix(colorspace);
            }
        case 10:
        case 16:
            return GetBT2020ConversionMatrix(colorspace);
        default:
            break;
        }
        break;
    default:
        break;
    }
    return 0;
}

static bool METAL_CreateTexture(SDL_Renderer *renderer, SDL_Texture *texture, SDL_PropertiesID create_props)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        MTLPixelFormat pixfmt;
        MTLTextureDescriptor *mtltexdesc;
        id<MTLTexture> mtltexture = nil, mtltextureUv = nil;
        SDL3METAL_TextureData *texturedata;
        CVPixelBufferRef pixelbuffer = nil;
        IOSurfaceRef surface = nil;

        pixelbuffer = SDL_GetPointerProperty(create_props, SDL_PROP_TEXTURE_CREATE_METAL_PIXELBUFFER_POINTER, nil);
        if (pixelbuffer) {
            surface = CVPixelBufferGetIOSurface(pixelbuffer);
            if (!surface) {
                return SDL_SetError("CVPixelBufferGetIOSurface() failed");
            }
        }

        switch (texture->format) {
        case SDL_PIXELFORMAT_ABGR8888:
            if (renderer->output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
                pixfmt = MTLPixelFormatRGBA8Unorm_sRGB;
            } else {
                pixfmt = MTLPixelFormatRGBA8Unorm;
            }
            break;
        case SDL_PIXELFORMAT_ARGB8888:
            if (renderer->output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
                pixfmt = MTLPixelFormatBGRA8Unorm_sRGB;
            } else {
                pixfmt = MTLPixelFormatBGRA8Unorm;
            }
            break;
        case SDL_PIXELFORMAT_ABGR2101010:
            pixfmt = MTLPixelFormatRGB10A2Unorm;
            break;
        case SDL_PIXELFORMAT_IYUV:
        case SDL_PIXELFORMAT_YV12:
        case SDL_PIXELFORMAT_NV12:
        case SDL_PIXELFORMAT_NV21:
            pixfmt = MTLPixelFormatR8Unorm;
            break;
        case SDL_PIXELFORMAT_P010:
            pixfmt = MTLPixelFormatR16Unorm;
            break;
        case SDL_PIXELFORMAT_RGBA64_FLOAT:
            pixfmt = MTLPixelFormatRGBA16Float;
            break;
        case SDL_PIXELFORMAT_RGBA128_FLOAT:
            pixfmt = MTLPixelFormatRGBA32Float;
            break;
        default:
            return SDL_SetError("Texture format %s not supported by Metal", SDL_GetPixelFormatName(texture->format));
        }

        mtltexdesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixfmt
                                                                        width:(NSUInteger)texture->w
                                                                       height:(NSUInteger)texture->h
                                                                    mipmapped:NO];

        if (texture->access == SDL_TEXTUREACCESS_TARGET) {
            mtltexdesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        } else {
            mtltexdesc.usage = MTLTextureUsageShaderRead;
        }

        if (surface) {
            mtltexture = [data.mtldevice newTextureWithDescriptor:mtltexdesc iosurface:surface plane:0];
        } else {
            mtltexture = [data.mtldevice newTextureWithDescriptor:mtltexdesc];
        }
        if (mtltexture == nil) {
            return SDL_SetError("Texture allocation failed");
        }

        mtltextureUv = nil;
#ifdef SDL_HAVE_YUV
        BOOL yuv = (texture->format == SDL_PIXELFORMAT_IYUV || texture->format == SDL_PIXELFORMAT_YV12);
        BOOL nv12 = (texture->format == SDL_PIXELFORMAT_NV12 || texture->format == SDL_PIXELFORMAT_NV21 || texture->format == SDL_PIXELFORMAT_P010);

        if (yuv) {
            mtltexdesc.pixelFormat = MTLPixelFormatR8Unorm;
            mtltexdesc.width = (texture->w + 1) / 2;
            mtltexdesc.height = (texture->h + 1) / 2;
            mtltexdesc.textureType = MTLTextureType2DArray;
            mtltexdesc.arrayLength = 2;
        } else if (texture->format == SDL_PIXELFORMAT_P010) {
            mtltexdesc.pixelFormat = MTLPixelFormatRG16Unorm;
            mtltexdesc.width = (texture->w + 1) / 2;
            mtltexdesc.height = (texture->h + 1) / 2;
        } else if (nv12) {
            mtltexdesc.pixelFormat = MTLPixelFormatRG8Unorm;
            mtltexdesc.width = (texture->w + 1) / 2;
            mtltexdesc.height = (texture->h + 1) / 2;
        }

        if (yuv || nv12) {
            if (surface) {
                mtltextureUv = [data.mtldevice newTextureWithDescriptor:mtltexdesc iosurface:surface plane:1];
            } else {
                mtltextureUv = [data.mtldevice newTextureWithDescriptor:mtltexdesc];
            }
            if (mtltextureUv == nil) {
                return SDL_SetError("Texture allocation failed");
            }
        }
#endif // SDL_HAVE_YUV
        texturedata = [[SDL3METAL_TextureData alloc] init];
#ifdef SDL_HAVE_YUV
        if (yuv) {
            texturedata.fragmentFunction = SDL_METAL_FRAGMENT_YUV;
        } else if (nv12) {
            texturedata.fragmentFunction = SDL_METAL_FRAGMENT_NV12;
        } else
#endif
        {
            texturedata.fragmentFunction = SDL_METAL_FRAGMENT_COPY;
        }
        texturedata.mtltexture = mtltexture;
        texturedata.mtltextureUv = mtltextureUv;
#ifdef SDL_HAVE_YUV
        texturedata.yuv = yuv;
        texturedata.nv12 = nv12;
        if (yuv || nv12) {
            size_t offset = GetYCbCRtoRGBConversionMatrix(texture->colorspace, texture->w, texture->h, 8);
            if (offset == 0) {
                return SDL_SetError("Unsupported YUV colorspace");
            }
            texturedata.conversionBufferOffset = offset;
        }
#endif
        texture->internal = (void *)CFBridgingRetain(texturedata);

        return true;
    }
}

static void METAL_UploadTextureData(id<MTLTexture> texture, SDL_Rect rect, int slice,
                                    const void *pixels, int pitch)
{
    [texture replaceRegion:MTLRegionMake2D(rect.x, rect.y, rect.w, rect.h)
               mipmapLevel:0
                     slice:slice
                 withBytes:pixels
               bytesPerRow:pitch
             bytesPerImage:0];
}

static MTLStorageMode METAL_GetStorageMode(id<MTLResource> resource)
{
    return resource.storageMode;
}

static bool METAL_UpdateTextureInternal(SDL_Renderer *renderer, SDL3METAL_TextureData *texturedata,
                                       id<MTLTexture> texture, SDL_Rect rect, int slice,
                                       const void *pixels, int pitch)
{
    SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
    SDL_Rect stagingrect = { 0, 0, rect.w, rect.h };
    MTLTextureDescriptor *desc;
    id<MTLTexture> stagingtex;
    id<MTLBlitCommandEncoder> blitcmd;

    /* If the texture is managed or shared and this is the first upload, we can
     * use replaceRegion to upload to it directly. Otherwise we upload the data
     * to a staging texture and copy that over. */
    if (!texturedata.hasdata && METAL_GetStorageMode(texture) != MTLStorageModePrivate) {
        METAL_UploadTextureData(texture, rect, slice, pixels, pitch);
        return true;
    }

    desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat
                                                              width:rect.w
                                                             height:rect.h
                                                          mipmapped:NO];

    if (desc == nil) {
        return SDL_OutOfMemory();
    }

    /* TODO: We could have a pool of textures or a MTLHeap we allocate from,
     * and release a staging texture back to the pool in the command buffer's
     * completion handler. */
    stagingtex = [data.mtldevice newTextureWithDescriptor:desc];
    if (stagingtex == nil) {
        return SDL_OutOfMemory();
    }

    METAL_UploadTextureData(stagingtex, stagingrect, 0, pixels, pitch);

    if (data.mtlcmdencoder != nil) {
        [data.mtlcmdencoder endEncoding];
        data.mtlcmdencoder = nil;
    }

    if (data.mtlcmdbuffer == nil) {
        data.mtlcmdbuffer = [data.mtlcmdqueue commandBuffer];
    }

    blitcmd = [data.mtlcmdbuffer blitCommandEncoder];

    [blitcmd copyFromTexture:stagingtex
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(rect.w, rect.h, 1)
                   toTexture:texture
            destinationSlice:slice
            destinationLevel:0
           destinationOrigin:MTLOriginMake(rect.x, rect.y, 0)];

    [blitcmd endEncoding];

    /* TODO: This isn't very efficient for the YUV formats, which call
     * UpdateTextureInternal multiple times in a row. */
    [data.mtlcmdbuffer commit];
    data.mtlcmdbuffer = nil;

    return true;
}

static bool METAL_UpdateTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                               const SDL_Rect *rect, const void *pixels, int pitch)
{
    @autoreleasepool {
        SDL3METAL_TextureData *texturedata = (__bridge SDL3METAL_TextureData *)texture->internal;

        if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltexture, *rect, 0, pixels, pitch)) {
            return false;
        }
#ifdef SDL_HAVE_YUV
        if (texturedata.yuv) {
            int Uslice = texture->format == SDL_PIXELFORMAT_YV12 ? 1 : 0;
            int Vslice = texture->format == SDL_PIXELFORMAT_YV12 ? 0 : 1;
            int UVpitch = (pitch + 1) / 2;
            SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };

            // Skip to the correct offset into the next texture
            pixels = (const void *)((const Uint8 *)pixels + rect->h * pitch);
            if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltextureUv, UVrect, Uslice, pixels, UVpitch)) {
                return false;
            }

            // Skip to the correct offset into the next texture
            pixels = (const void *)((const Uint8 *)pixels + UVrect.h * UVpitch);
            if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltextureUv, UVrect, Vslice, pixels, UVpitch)) {
                return false;
            }
        }

        if (texturedata.nv12) {
            SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };
            int UVpitch = 2 * ((pitch + 1) / 2);

            // Skip to the correct offset into the next texture
            pixels = (const void *)((const Uint8 *)pixels + rect->h * pitch);
            if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltextureUv, UVrect, 0, pixels, UVpitch)) {
                return false;
            }
        }
#endif
        texturedata.hasdata = YES;

        return true;
    }
}

#ifdef SDL_HAVE_YUV
static bool METAL_UpdateTextureYUV(SDL_Renderer *renderer, SDL_Texture *texture,
                                  const SDL_Rect *rect,
                                  const Uint8 *Yplane, int Ypitch,
                                  const Uint8 *Uplane, int Upitch,
                                  const Uint8 *Vplane, int Vpitch)
{
    @autoreleasepool {
        SDL3METAL_TextureData *texturedata = (__bridge SDL3METAL_TextureData *)texture->internal;
        const int Uslice = 0;
        const int Vslice = 1;
        SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };

        // Bail out if we're supposed to update an empty rectangle
        if (rect->w <= 0 || rect->h <= 0) {
            return true;
        }

        if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltexture, *rect, 0, Yplane, Ypitch)) {
            return false;
        }
        if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltextureUv, UVrect, Uslice, Uplane, Upitch)) {
            return false;
        }
        if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltextureUv, UVrect, Vslice, Vplane, Vpitch)) {
            return false;
        }

        texturedata.hasdata = YES;

        return true;
    }
}

static bool METAL_UpdateTextureNV(SDL_Renderer *renderer, SDL_Texture *texture,
                                 const SDL_Rect *rect,
                                 const Uint8 *Yplane, int Ypitch,
                                 const Uint8 *UVplane, int UVpitch)
{
    @autoreleasepool {
        SDL3METAL_TextureData *texturedata = (__bridge SDL3METAL_TextureData *)texture->internal;
        SDL_Rect UVrect = { rect->x / 2, rect->y / 2, (rect->w + 1) / 2, (rect->h + 1) / 2 };

        // Bail out if we're supposed to update an empty rectangle
        if (rect->w <= 0 || rect->h <= 0) {
            return true;
        }

        if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltexture, *rect, 0, Yplane, Ypitch)) {
            return false;
        }

        if (!METAL_UpdateTextureInternal(renderer, texturedata, texturedata.mtltextureUv, UVrect, 0, UVplane, UVpitch)) {
            return false;
        }

        texturedata.hasdata = YES;

        return true;
    }
}
#endif

static bool METAL_LockTexture(SDL_Renderer *renderer, SDL_Texture *texture,
                             const SDL_Rect *rect, void **pixels, int *pitch)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        SDL3METAL_TextureData *texturedata = (__bridge SDL3METAL_TextureData *)texture->internal;
        int buffersize = 0;
        id<MTLBuffer> lockedbuffer = nil;

        if (rect->w <= 0 || rect->h <= 0) {
            return SDL_SetError("Invalid rectangle dimensions for LockTexture.");
        }

        *pitch = SDL_BYTESPERPIXEL(texture->format) * rect->w;
#ifdef SDL_HAVE_YUV
        if (texturedata.yuv || texturedata.nv12) {
            buffersize = ((*pitch) * rect->h) + (2 * (*pitch + 1) / 2) * ((rect->h + 1) / 2);
        } else
#endif
        {
            buffersize = (*pitch) * rect->h;
        }

        lockedbuffer = [data.mtldevice newBufferWithLength:buffersize options:MTLResourceStorageModeShared];
        if (lockedbuffer == nil) {
            return SDL_OutOfMemory();
        }

        texturedata.lockedrect = *rect;
        texturedata.lockedbuffer = lockedbuffer;
        *pixels = [lockedbuffer contents];

        return true;
    }
}

static void METAL_UnlockTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        SDL3METAL_TextureData *texturedata = (__bridge SDL3METAL_TextureData *)texture->internal;
        id<MTLBlitCommandEncoder> blitcmd;
        SDL_Rect rect = texturedata.lockedrect;
        int pitch = SDL_BYTESPERPIXEL(texture->format) * rect.w;
#ifdef SDL_HAVE_YUV
        SDL_Rect UVrect = { rect.x / 2, rect.y / 2, (rect.w + 1) / 2, (rect.h + 1) / 2 };
#endif

        if (texturedata.lockedbuffer == nil) {
            return;
        }

        if (data.mtlcmdencoder != nil) {
            [data.mtlcmdencoder endEncoding];
            data.mtlcmdencoder = nil;
        }

        if (data.mtlcmdbuffer == nil) {
            data.mtlcmdbuffer = [data.mtlcmdqueue commandBuffer];
        }

        blitcmd = [data.mtlcmdbuffer blitCommandEncoder];

        [blitcmd copyFromBuffer:texturedata.lockedbuffer
                   sourceOffset:0
              sourceBytesPerRow:pitch
            sourceBytesPerImage:0
                     sourceSize:MTLSizeMake(rect.w, rect.h, 1)
                      toTexture:texturedata.mtltexture
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(rect.x, rect.y, 0)];
#ifdef SDL_HAVE_YUV
        if (texturedata.yuv) {
            int Uslice = texture->format == SDL_PIXELFORMAT_YV12 ? 1 : 0;
            int Vslice = texture->format == SDL_PIXELFORMAT_YV12 ? 0 : 1;
            int UVpitch = (pitch + 1) / 2;

            [blitcmd copyFromBuffer:texturedata.lockedbuffer
                       sourceOffset:rect.h * pitch
                  sourceBytesPerRow:UVpitch
                sourceBytesPerImage:UVpitch * UVrect.h
                         sourceSize:MTLSizeMake(UVrect.w, UVrect.h, 1)
                          toTexture:texturedata.mtltextureUv
                   destinationSlice:Uslice
                   destinationLevel:0
                  destinationOrigin:MTLOriginMake(UVrect.x, UVrect.y, 0)];

            [blitcmd copyFromBuffer:texturedata.lockedbuffer
                       sourceOffset:(rect.h * pitch) + UVrect.h * UVpitch
                  sourceBytesPerRow:UVpitch
                sourceBytesPerImage:UVpitch * UVrect.h
                         sourceSize:MTLSizeMake(UVrect.w, UVrect.h, 1)
                          toTexture:texturedata.mtltextureUv
                   destinationSlice:Vslice
                   destinationLevel:0
                  destinationOrigin:MTLOriginMake(UVrect.x, UVrect.y, 0)];
        }

        if (texturedata.nv12) {
            int UVpitch = 2 * ((pitch + 1) / 2);

            [blitcmd copyFromBuffer:texturedata.lockedbuffer
                       sourceOffset:rect.h * pitch
                  sourceBytesPerRow:UVpitch
                sourceBytesPerImage:0
                         sourceSize:MTLSizeMake(UVrect.w, UVrect.h, 1)
                          toTexture:texturedata.mtltextureUv
                   destinationSlice:0
                   destinationLevel:0
                  destinationOrigin:MTLOriginMake(UVrect.x, UVrect.y, 0)];
        }
#endif
        [blitcmd endEncoding];

        [data.mtlcmdbuffer commit];
        data.mtlcmdbuffer = nil;

        texturedata.lockedbuffer = nil; // Retained property, so it calls release.
        texturedata.hasdata = YES;
    }
}

static bool METAL_SetRenderTarget(SDL_Renderer *renderer, SDL_Texture *texture)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;

        if (data.mtlcmdencoder) {
            /* End encoding for the previous render target so we can set up a new
             * render pass for this one. */
            [data.mtlcmdencoder endEncoding];
            [data.mtlcmdbuffer commit];

            data.mtlcmdencoder = nil;
            data.mtlcmdbuffer = nil;
        }

        /* We don't begin a new render pass right away - we delay it until an actual
         * draw or clear happens. That way we can use hardware clears when possible,
         * which are only available when beginning a new render pass. */
        return true;
    }
}

static bool METAL_QueueSetViewport(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    float projection[4][4]; // Prepare an orthographic projection
    const int w = cmd->data.viewport.rect.w;
    const int h = cmd->data.viewport.rect.h;
    const size_t matrixlen = sizeof(projection);
    float *matrix = (float *)SDL_AllocateRenderVertices(renderer, matrixlen, CONSTANT_ALIGN(16), &cmd->data.viewport.first);
    if (!matrix) {
        return false;
    }

    SDL_memset(projection, '\0', matrixlen);
    if (w && h) {
        projection[0][0] = 2.0f / w;
        projection[1][1] = -2.0f / h;
        projection[3][0] = -1.0f;
        projection[3][1] = 1.0f;
        projection[3][3] = 1.0f;
    }
    SDL_memcpy(matrix, projection, matrixlen);

    return true;
}

static bool METAL_QueueNoOp(SDL_Renderer *renderer, SDL_RenderCommand *cmd)
{
    return true; // nothing to do in this backend.
}

static bool METAL_QueueDrawPoints(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    SDL_FColor color = cmd->data.draw.color;
    bool convert_color = SDL_RenderingLinearSpace(renderer);

    const size_t vertlen = (2 * sizeof(float) + 4 * sizeof(float)) * count;
    float *verts = (float *)SDL_AllocateRenderVertices(renderer, vertlen, DEVICE_ALIGN(8), &cmd->data.draw.first);
    if (!verts) {
        return false;
    }
    cmd->data.draw.count = count;

    if (convert_color) {
        SDL_ConvertToLinear(&color);
    }

    for (int i = 0; i < count; i++, points++) {
        *(verts++) = points->x;
        *(verts++) = points->y;
        *(verts++) = color.r;
        *(verts++) = color.g;
        *(verts++) = color.b;
        *(verts++) = color.a;
    }
    return true;
}

static bool METAL_QueueDrawLines(SDL_Renderer *renderer, SDL_RenderCommand *cmd, const SDL_FPoint *points, int count)
{
    SDL_FColor color = cmd->data.draw.color;
    bool convert_color = SDL_RenderingLinearSpace(renderer);
    size_t vertlen;
    float *verts;

    SDL_assert(count >= 2); // should have been checked at the higher level.

    vertlen = (2 * sizeof(float) + 4 * sizeof(float)) * count;
    verts = (float *)SDL_AllocateRenderVertices(renderer, vertlen, DEVICE_ALIGN(8), &cmd->data.draw.first);
    if (!verts) {
        return false;
    }
    cmd->data.draw.count = count;

    if (convert_color) {
        SDL_ConvertToLinear(&color);
    }

    for (int i = 0; i < count; i++, points++) {
        *(verts++) = points->x;
        *(verts++) = points->y;
        *(verts++) = color.r;
        *(verts++) = color.g;
        *(verts++) = color.b;
        *(verts++) = color.a;
    }

    /* If the line segment is completely horizontal or vertical,
       make it one pixel longer, to satisfy the diamond-exit rule.
       We should probably do this for diagonal lines too, but we'd have to
       do some trigonometry to figure out the correct pixel and generally
       when we have problems with pixel perfection, it's for straight lines
       that are missing a pixel that frames something and not arbitrary
       angles. Maybe !!! FIXME for later, though. */

    points -= 2; // update the last line.
    verts -= 2 + 1;

    {
        const float xstart = points[0].x;
        const float ystart = points[0].y;
        const float xend = points[1].x;
        const float yend = points[1].y;

        if (ystart == yend) { // horizontal line
            verts[0] += (xend > xstart) ? 1.0f : -1.0f;
        } else if (xstart == xend) { // vertical line
            verts[1] += (yend > ystart) ? 1.0f : -1.0f;
        }
    }

    return true;
}

static bool METAL_QueueGeometry(SDL_Renderer *renderer, SDL_RenderCommand *cmd, SDL_Texture *texture,
                               const float *xy, int xy_stride, const SDL_FColor *color, int color_stride, const float *uv, int uv_stride,
                               int num_vertices, const void *indices, int num_indices, int size_indices,
                               float scale_x, float scale_y)
{
    bool convert_color = SDL_RenderingLinearSpace(renderer);
    int count = indices ? num_indices : num_vertices;
    const size_t vertlen = (2 * sizeof(float) + 4 * sizeof(float) + (texture ? 2 : 0) * sizeof(float)) * count;
    float *verts = (float *)SDL_AllocateRenderVertices(renderer, vertlen, DEVICE_ALIGN(8), &cmd->data.draw.first);
    if (!verts) {
        return false;
    }

    cmd->data.draw.count = count;
    size_indices = indices ? size_indices : 0;

    for (int i = 0; i < count; i++) {
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

        *(verts++) = col_.r;
        *(verts++) = col_.g;
        *(verts++) = col_.b;
        *(verts++) = col_.a;

        if (texture) {
            float *uv_ = (float *)((char *)uv + j * uv_stride);
            *(verts++) = uv_[0];
            *(verts++) = uv_[1];
        }
    }

    return true;
}

// These should mirror the definitions in SDL_shaders_metal.metal
//static const float TONEMAP_NONE = 0;
//static const float TONEMAP_LINEAR = 1;
static const float TONEMAP_CHROME = 2;

//static const float TEXTURETYPE_NONE = 0;
static const float TEXTURETYPE_RGB = 1;
static const float TEXTURETYPE_RGB_PIXELART = 2;
static const float TEXTURETYPE_NV12 = 3;
static const float TEXTURETYPE_NV21 = 4;
static const float TEXTURETYPE_YUV = 5;

//static const float INPUTTYPE_UNSPECIFIED = 0;
static const float INPUTTYPE_SRGB = 1;
static const float INPUTTYPE_SCRGB = 2;
static const float INPUTTYPE_HDR10 = 3;

typedef struct
{
    float scRGB_output;
    float texture_type;
    float input_type;
    float color_scale;

    float texel_width;
    float texel_height;
    float texture_width;
    float texture_height;

    float tonemap_method;
    float tonemap_factor1;
    float tonemap_factor2;
    float sdr_white_point;
} PixelShaderConstants;

typedef struct
{
    __unsafe_unretained id<MTLRenderPipelineState> pipeline;
    __unsafe_unretained id<MTLBuffer> vertex_buffer;
    size_t constants_offset;
    SDL_Texture *texture;
    SDL_ScaleMode texture_scale_mode;
    SDL_TextureAddressMode texture_address_mode_u;
    SDL_TextureAddressMode texture_address_mode_v;
    bool cliprect_dirty;
    bool cliprect_enabled;
    SDL_Rect cliprect;
    bool viewport_dirty;
    SDL_Rect viewport;
    size_t projection_offset;
    bool shader_constants_dirty;
    PixelShaderConstants shader_constants;
} METAL_DrawStateCache;

static void SetupShaderConstants(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const SDL_Texture *texture, PixelShaderConstants *constants)
{
    float output_headroom;

    SDL_zerop(constants);

    constants->scRGB_output = (float)SDL_RenderingLinearSpace(renderer);
    constants->color_scale = cmd->data.draw.color_scale;

    if (texture) {
        switch (texture->format) {
        case SDL_PIXELFORMAT_YV12:
        case SDL_PIXELFORMAT_IYUV:
            constants->texture_type = TEXTURETYPE_YUV;
            break;
        case SDL_PIXELFORMAT_NV12:
            constants->texture_type = TEXTURETYPE_NV12;
            break;
        case SDL_PIXELFORMAT_NV21:
            constants->texture_type = TEXTURETYPE_NV21;
            break;
        case SDL_PIXELFORMAT_P010:
            constants->texture_type = TEXTURETYPE_NV12;
            break;
        default:
            if (cmd->data.draw.texture_scale_mode == SDL_SCALEMODE_PIXELART) {
                constants->texture_type = TEXTURETYPE_RGB_PIXELART;
                constants->texture_width = texture->w;
                constants->texture_height = texture->h;
                constants->texel_width = 1.0f / constants->texture_width;
                constants->texel_height = 1.0f / constants->texture_height;
            } else {
                constants->texture_type = TEXTURETYPE_RGB;
            }
        }

        switch (SDL_COLORSPACETRANSFER(texture->colorspace)) {
        case SDL_TRANSFER_CHARACTERISTICS_LINEAR:
            constants->input_type = INPUTTYPE_SCRGB;
            break;
        case SDL_TRANSFER_CHARACTERISTICS_PQ:
            constants->input_type = INPUTTYPE_HDR10;
            break;
        default:
            constants->input_type = INPUTTYPE_SRGB;
            break;
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

static bool SetDrawState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const SDL_MetalFragmentFunction shader, PixelShaderConstants *shader_constants, const size_t constants_offset, id<MTLBuffer> mtlbufvertex, METAL_DrawStateCache *statecache)
{
    SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
    const SDL_BlendMode blend = cmd->data.draw.blend;
    size_t first = cmd->data.draw.first;
    id<MTLRenderPipelineState> newpipeline;
    PixelShaderConstants solid_constants;

    if (!METAL_ActivateRenderCommandEncoder(renderer, MTLLoadActionLoad, NULL, statecache->vertex_buffer)) {
        return false;
    }

    if (statecache->viewport_dirty) {
        MTLViewport viewport;
        viewport.originX = statecache->viewport.x;
        viewport.originY = statecache->viewport.y;
        viewport.width = statecache->viewport.w;
        viewport.height = statecache->viewport.h;
        viewport.znear = 0.0;
        viewport.zfar = 1.0;
        [data.mtlcmdencoder setViewport:viewport];
        [data.mtlcmdencoder setVertexBuffer:mtlbufvertex offset:statecache->projection_offset atIndex:2]; // projection
        statecache->viewport_dirty = false;
    }

    if (statecache->cliprect_dirty) {
        SDL_Rect output;
        SDL_Rect clip;
        if (statecache->cliprect_enabled) {
            clip = statecache->cliprect;
            clip.x += statecache->viewport.x;
            clip.y += statecache->viewport.y;
        } else {
            clip = statecache->viewport;
        }

        // Set Scissor Rect Validation: w/h must be <= render pass
        SDL_zero(output);

        if (renderer->target) {
            output.w = renderer->target->w;
            output.h = renderer->target->h;
        } else {
            METAL_GetOutputSize(renderer, &output.w, &output.h);
        }

        if (SDL_GetRectIntersection(&output, &clip, &clip)) {
            MTLScissorRect mtlrect;
            mtlrect.x = clip.x;
            mtlrect.y = clip.y;
            mtlrect.width = clip.w;
            mtlrect.height = clip.h;
            [data.mtlcmdencoder setScissorRect:mtlrect];
        }

        statecache->cliprect_dirty = false;
    }

    newpipeline = ChoosePipelineState(data, data.activepipelines, shader, blend);
    if (newpipeline != statecache->pipeline) {
        [data.mtlcmdencoder setRenderPipelineState:newpipeline];
        statecache->pipeline = newpipeline;
    }

    if (!shader_constants) {
        SetupShaderConstants(renderer, cmd, NULL, &solid_constants);
        shader_constants = &solid_constants;
    }

    if (statecache->shader_constants_dirty ||
        SDL_memcmp(shader_constants, &statecache->shader_constants, sizeof(*shader_constants)) != 0) {
        id<MTLBuffer> mtlbufconstants = [data.mtldevice newBufferWithLength:sizeof(*shader_constants) options:MTLResourceStorageModeShared];
        mtlbufconstants.label = @"SDL shader constants data";
        SDL_memcpy([mtlbufconstants contents], shader_constants, sizeof(*shader_constants));
        [data.mtlcmdencoder setFragmentBuffer:mtlbufconstants offset:0 atIndex:0];

        SDL_memcpy(&statecache->shader_constants, shader_constants, sizeof(*shader_constants));
        statecache->shader_constants_dirty = false;
    }

    if (constants_offset != statecache->constants_offset) {
        if (constants_offset != CONSTANTS_OFFSET_INVALID) {
            [data.mtlcmdencoder setVertexBuffer:data.mtlbufconstants offset:constants_offset atIndex:3];
        }
        statecache->constants_offset = constants_offset;
    }

    [data.mtlcmdencoder setVertexBufferOffset:first atIndex:0]; // position/texcoords
    return true;
}

static id<MTLSamplerState> GetSampler(SDL3METAL_RenderData *data, SDL_ScaleMode scale_mode, SDL_TextureAddressMode address_u, SDL_TextureAddressMode address_v)
{
    NSNumber *key = [NSNumber numberWithInteger:RENDER_SAMPLER_HASHKEY(scale_mode, address_u, address_v)];
    id<MTLSamplerState> mtlsampler = data.mtlsamplers[key];
    if (mtlsampler == nil) {
        MTLSamplerDescriptor *samplerdesc;
        samplerdesc = [[MTLSamplerDescriptor alloc] init];
        switch (scale_mode) {
        case SDL_SCALEMODE_NEAREST:
            samplerdesc.minFilter = MTLSamplerMinMagFilterNearest;
            samplerdesc.magFilter = MTLSamplerMinMagFilterNearest;
            break;
        case SDL_SCALEMODE_PIXELART:    // Uses linear sampling
        case SDL_SCALEMODE_LINEAR:
            samplerdesc.minFilter = MTLSamplerMinMagFilterLinear;
            samplerdesc.magFilter = MTLSamplerMinMagFilterLinear;
            break;
        default:
            SDL_SetError("Unknown scale mode: %d", scale_mode);
            return nil;
        }
        switch (address_u) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            samplerdesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            samplerdesc.sAddressMode = MTLSamplerAddressModeRepeat;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_u);
            return nil;
        }
        switch (address_v) {
        case SDL_TEXTURE_ADDRESS_CLAMP:
            samplerdesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
            break;
        case SDL_TEXTURE_ADDRESS_WRAP:
            samplerdesc.tAddressMode = MTLSamplerAddressModeRepeat;
            break;
        default:
            SDL_SetError("Unknown texture address mode: %d", address_v);
            return nil;
        }
        mtlsampler = [data.mtldevice newSamplerStateWithDescriptor:samplerdesc];
        if (mtlsampler == nil) {
            SDL_SetError("Couldn't create sampler");
            return nil;
        }
        data.mtlsamplers[key] = mtlsampler;
    }
    return mtlsampler;
}

static bool SetCopyState(SDL_Renderer *renderer, const SDL_RenderCommand *cmd, const size_t constants_offset,
                             id<MTLBuffer> mtlbufvertex, METAL_DrawStateCache *statecache)
{
    SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
    SDL_Texture *texture = cmd->data.draw.texture;
    SDL3METAL_TextureData *texturedata = (__bridge SDL3METAL_TextureData *)texture->internal;
    PixelShaderConstants constants;

    SetupShaderConstants(renderer, cmd, texture, &constants);

    if (!SetDrawState(renderer, cmd, texturedata.fragmentFunction, &constants, constants_offset, mtlbufvertex, statecache)) {
        return false;
    }

    if (texture != statecache->texture) {
        [data.mtlcmdencoder setFragmentTexture:texturedata.mtltexture atIndex:0];
#ifdef SDL_HAVE_YUV
        if (texturedata.yuv || texturedata.nv12) {
            [data.mtlcmdencoder setFragmentTexture:texturedata.mtltextureUv atIndex:1];
            [data.mtlcmdencoder setFragmentBuffer:data.mtlbufconstants offset:texturedata.conversionBufferOffset atIndex:1];
        }
#endif
        statecache->texture = texture;
    }

    if (cmd->data.draw.texture_scale_mode != statecache->texture_scale_mode ||
        cmd->data.draw.texture_address_mode_u != statecache->texture_address_mode_u ||
        cmd->data.draw.texture_address_mode_v != statecache->texture_address_mode_v) {
        id<MTLSamplerState> mtlsampler = GetSampler(data, cmd->data.draw.texture_scale_mode, cmd->data.draw.texture_address_mode_u, cmd->data.draw.texture_address_mode_v);
        if (mtlsampler == nil) {
            return false;
        }
        [data.mtlcmdencoder setFragmentSamplerState:mtlsampler atIndex:0];

        statecache->texture_scale_mode = cmd->data.draw.texture_scale_mode;
        statecache->texture_address_mode_u = cmd->data.draw.texture_address_mode_u;
        statecache->texture_address_mode_v = cmd->data.draw.texture_address_mode_v;
    }
    return true;
}

static void METAL_InvalidateCachedState(SDL_Renderer *renderer)
{
    // METAL_DrawStateCache only exists during a run of METAL_RunCommandQueue, so there's nothing to invalidate!
}

static bool METAL_RunCommandQueue(SDL_Renderer *renderer, SDL_RenderCommand *cmd, void *vertices, size_t vertsize)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        id<MTLBuffer> mtlbufvertex = nil;
        METAL_DrawStateCache statecache;
        SDL_zero(statecache);

        statecache.pipeline = nil;
        statecache.vertex_buffer = nil;
        statecache.constants_offset = CONSTANTS_OFFSET_INVALID;
        statecache.texture = NULL;
        statecache.texture_scale_mode = SDL_SCALEMODE_INVALID;
        statecache.texture_address_mode_u = SDL_TEXTURE_ADDRESS_INVALID;
        statecache.texture_address_mode_v = SDL_TEXTURE_ADDRESS_INVALID;
        statecache.shader_constants_dirty = true;
        statecache.cliprect_dirty = true;
        statecache.viewport_dirty = true;
        statecache.projection_offset = 0;

        // !!! FIXME: have a ring of pre-made MTLBuffers we cycle through? How expensive is creation?
        if (vertsize > 0) {
            /* We can memcpy to a shared buffer from the CPU and read it from the GPU
             * without any extra copying. It's a bit slower on macOS to read shared
             * data from the GPU than to read managed/private data, but we avoid the
             * cost of copying the data and the code's simpler. Apple's best
             * practices guide recommends this approach for streamed vertex data.
             */
            mtlbufvertex = [data.mtldevice newBufferWithLength:vertsize options:MTLResourceStorageModeShared];
            mtlbufvertex.label = @"SDL vertex data";
            SDL_memcpy([mtlbufvertex contents], vertices, vertsize);

            statecache.vertex_buffer = mtlbufvertex;
        }

        // If there's a command buffer here unexpectedly (app requested one?). Commit it so we can start fresh.
        [data.mtlcmdencoder endEncoding];
        [data.mtlcmdbuffer commit];
        data.mtlcmdencoder = nil;
        data.mtlcmdbuffer = nil;

        while (cmd) {
            switch (cmd->command) {
            case SDL_RENDERCMD_SETVIEWPORT:
            {
                SDL_memcpy(&statecache.viewport, &cmd->data.viewport.rect, sizeof(statecache.viewport));
                statecache.projection_offset = cmd->data.viewport.first;
                statecache.viewport_dirty = true;
                statecache.cliprect_dirty = true;
                break;
            }

            case SDL_RENDERCMD_SETCLIPRECT:
            {
                SDL_memcpy(&statecache.cliprect, &cmd->data.cliprect.rect, sizeof(statecache.cliprect));
                statecache.cliprect_enabled = cmd->data.cliprect.enabled;
                statecache.cliprect_dirty = true;
                break;
            }

            case SDL_RENDERCMD_SETDRAWCOLOR:
            {
                break;
            }

            case SDL_RENDERCMD_CLEAR:
            {
                /* If we're already encoding a command buffer, dump it without committing it. We'd just
                    clear all its work anyhow, and starting a new encoder will let us use a hardware clear
                    operation via MTLLoadActionClear. */
                if (data.mtlcmdencoder != nil) {
                    [data.mtlcmdencoder endEncoding];

                    // !!! FIXME: have to commit, or an uncommitted but enqueued buffer will prevent the frame from finishing.
                    [data.mtlcmdbuffer commit];
                    data.mtlcmdencoder = nil;
                    data.mtlcmdbuffer = nil;
                }

                // force all this state to be reconfigured on next command buffer.
                statecache.pipeline = nil;
                statecache.constants_offset = CONSTANTS_OFFSET_INVALID;
                statecache.texture = NULL;
                statecache.shader_constants_dirty = true;
                statecache.cliprect_dirty = true;
                statecache.viewport_dirty = true;

                {
                    bool convert_color = SDL_RenderingLinearSpace(renderer);
                    SDL_FColor color = cmd->data.color.color;
                    if (convert_color) {
                        SDL_ConvertToLinear(&color);
                    }
                    color.r *= cmd->data.color.color_scale;
                    color.g *= cmd->data.color.color_scale;
                    color.b *= cmd->data.color.color_scale;
                    MTLClearColor mtlcolor = MTLClearColorMake(color.r, color.g, color.b, color.a);

                    // get new command encoder, set up with an initial clear operation.
                    // (this might fail, and future draw operations will notice.)
                    METAL_ActivateRenderCommandEncoder(renderer, MTLLoadActionClear, &mtlcolor, mtlbufvertex);
                }
                break;
            }

            case SDL_RENDERCMD_DRAW_POINTS:
            case SDL_RENDERCMD_DRAW_LINES:
            {
                const size_t count = cmd->data.draw.count;
                const MTLPrimitiveType primtype = (cmd->command == SDL_RENDERCMD_DRAW_POINTS) ? MTLPrimitiveTypePoint : MTLPrimitiveTypeLineStrip;
                if (SetDrawState(renderer, cmd, SDL_METAL_FRAGMENT_SOLID, NULL, CONSTANTS_OFFSET_HALF_PIXEL_TRANSFORM, mtlbufvertex, &statecache)) {
                    [data.mtlcmdencoder drawPrimitives:primtype vertexStart:0 vertexCount:count];
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
                const size_t count = cmd->data.draw.count;
                SDL_Texture *texture = cmd->data.draw.texture;

                if (texture) {
                    if (SetCopyState(renderer, cmd, CONSTANTS_OFFSET_IDENTITY, mtlbufvertex, &statecache)) {
                        [data.mtlcmdencoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:count];
                    }
                } else {
                    if (SetDrawState(renderer, cmd, SDL_METAL_FRAGMENT_SOLID, NULL, CONSTANTS_OFFSET_IDENTITY, mtlbufvertex, &statecache)) {
                        [data.mtlcmdencoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:count];
                    }
                }
                break;
            }

            case SDL_RENDERCMD_NO_OP:
                break;
            }
            cmd = cmd->next;
        }

        return true;
    }
}

static SDL_Surface *METAL_RenderReadPixels(SDL_Renderer *renderer, const SDL_Rect *rect)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        id<MTLTexture> mtltexture;
        MTLRegion mtlregion;
        Uint32 format;
        SDL_Surface *surface;

        if (!METAL_ActivateRenderCommandEncoder(renderer, MTLLoadActionLoad, NULL, nil)) {
            SDL_SetError("Failed to activate render command encoder (is your window in the background?");
            return NULL;
        }

        [data.mtlcmdencoder endEncoding];
        mtltexture = data.mtlpassdesc.colorAttachments[0].texture;

#ifdef SDL_PLATFORM_MACOS
        /* on macOS with managed-storage textures, we need to tell the driver to
         * update the CPU-side copy of the texture data.
         * NOTE: Currently all of our textures are managed on macOS. We'll need some
         * extra copying for any private textures. */
        if (METAL_GetStorageMode(mtltexture) == MTLStorageModeManaged) {
            id<MTLBlitCommandEncoder> blit = [data.mtlcmdbuffer blitCommandEncoder];
            [blit synchronizeResource:mtltexture];
            [blit endEncoding];
        }
#endif

        /* Commit the current command buffer and wait until it's completed, to make
         * sure the GPU has finished rendering to it by the time we read it. */
        [data.mtlcmdbuffer commit];
        [data.mtlcmdbuffer waitUntilCompleted];
        data.mtlcmdencoder = nil;
        data.mtlcmdbuffer = nil;

        mtlregion = MTLRegionMake2D(rect->x, rect->y, rect->w, rect->h);

        switch (mtltexture.pixelFormat) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
            format = SDL_PIXELFORMAT_ARGB8888;
            break;
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
            format = SDL_PIXELFORMAT_ABGR8888;
            break;
        case MTLPixelFormatRGB10A2Unorm:
            format = SDL_PIXELFORMAT_ABGR2101010;
            break;
        case MTLPixelFormatRGBA16Float:
            format = SDL_PIXELFORMAT_RGBA64_FLOAT;
            break;
        default:
            SDL_SetError("Unknown framebuffer pixel format");
            return NULL;
        }
        surface = SDL_CreateSurface(rect->w, rect->h, format);
        if (surface) {
            [mtltexture getBytes:surface->pixels bytesPerRow:surface->pitch fromRegion:mtlregion mipmapLevel:0];
        }
        return surface;
    }
}

static bool METAL_RenderPresent(SDL_Renderer *renderer)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        bool ready = true;

        // If we don't have a command buffer, we can't present, so activate to get one.
        if (data.mtlcmdencoder == nil) {
            // We haven't even gotten a backbuffer yet? Load and clear it. Otherwise, load the existing data.
            if (data.mtlbackbuffer == nil) {
                float alpha = (SDL_GetWindowFlags(renderer->window) & SDL_WINDOW_TRANSPARENT) ? 0.0f : 1.0f;
                MTLClearColor color = MTLClearColorMake(0.0f, 0.0f, 0.0f, alpha);
                ready = METAL_ActivateRenderCommandEncoder(renderer, MTLLoadActionClear, &color, nil);
            } else {
                ready = METAL_ActivateRenderCommandEncoder(renderer, MTLLoadActionLoad, NULL, nil);
            }
        }

        [data.mtlcmdencoder endEncoding];

        // If we don't have a drawable to present, don't try to present it.
        //  But we'll still try to commit the command buffer in case it was already enqueued.
        if (ready) {
            SDL_assert(data.mtlbackbuffer != nil);
            [data.mtlcmdbuffer presentDrawable:data.mtlbackbuffer];
        }

        [data.mtlcmdbuffer commit];

        data.mtlcmdencoder = nil;
        data.mtlcmdbuffer = nil;
        data.mtlbackbuffer = nil;

        if (renderer->hidden || !ready) {
            return false;
        }
        return true;
    }
}

static void METAL_DestroyTexture(SDL_Renderer *renderer, SDL_Texture *texture)
{
    @autoreleasepool {
        CFBridgingRelease(texture->internal);
        texture->internal = NULL;
    }
}

static void METAL_DestroyRenderer(SDL_Renderer *renderer)
{
    @autoreleasepool {
        if (renderer->internal) {
            SDL3METAL_RenderData *data = CFBridgingRelease(renderer->internal);

            if (data.mtlcmdencoder != nil) {
                [data.mtlcmdencoder endEncoding];
            }

            DestroyAllPipelines(data.allpipelines, data.pipelinescount);

            /* Release the metal view instead of destroying it,
               in case we want to use it later (recreating the renderer)
             */
            // SDL_Metal_DestroyView(data.mtlview);
            CFBridgingRelease(data.mtlview);
        }
    }
}

static void *METAL_GetMetalLayer(SDL_Renderer *renderer)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        return (__bridge void *)data.mtllayer;
    }
}

static void *METAL_GetMetalCommandEncoder(SDL_Renderer *renderer)
{
    @autoreleasepool {
        // note that data.mtlcmdencoder can be nil if METAL_ActivateRenderCommandEncoder fails.
        //  Before SDL 2.0.18, it might have returned a non-nil encoding that might not have been
        //  usable for presentation. Check your return values!
        SDL3METAL_RenderData *data;
        METAL_ActivateRenderCommandEncoder(renderer, MTLLoadActionLoad, NULL, nil);
        data = (__bridge SDL3METAL_RenderData *)renderer->internal;
        return (__bridge void *)data.mtlcmdencoder;
    }
}

static bool METAL_SetVSync(SDL_Renderer *renderer, const int vsync)
{
#if defined(SDL_PLATFORM_MACOS) || TARGET_OS_MACCATALYST
    SDL3METAL_RenderData *data = (__bridge SDL3METAL_RenderData *)renderer->internal;
    switch (vsync) {
    case 0:
        data.mtllayer.displaySyncEnabled = NO;
        break;
    case 1:
        data.mtllayer.displaySyncEnabled = YES;
        break;
    default:
        return SDL_Unsupported();
    }
    return true;
#else
    switch (vsync) {
    case 1:
        return true;
    default:
        return SDL_Unsupported();
    }
#endif
}

static SDL_MetalView GetWindowView(SDL_Window *window)
{
#ifdef SDL_VIDEO_DRIVER_COCOA
    NSWindow *nswindow = (__bridge NSWindow *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, NULL);
    NSInteger tag = (NSInteger)SDL_GetNumberProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_COCOA_METAL_VIEW_TAG_NUMBER, 0);
    if (nswindow && tag) {
        NSView *view = nswindow.contentView;
        if (view.subviews.count > 0) {
            view = view.subviews[0];
            if (view.tag == tag) {
                return (SDL_MetalView)CFBridgingRetain(view);
            }
        }
    }
#endif

#ifdef SDL_VIDEO_DRIVER_UIKIT
    UIWindow *uiwindow = (__bridge UIWindow *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, NULL);
    NSInteger tag = (NSInteger)SDL_GetNumberProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_UIKIT_METAL_VIEW_TAG_NUMBER, 0);
    if (uiwindow && tag) {
        UIView *view = uiwindow.rootViewController.view;
        if (view.tag == tag) {
            return (SDL_MetalView)CFBridgingRetain(view);
        }
    }
#endif

    return nil;
}

static bool METAL_CreateRenderer(SDL_Renderer *renderer, SDL_Window *window, SDL_PropertiesID create_props)
{
    @autoreleasepool {
        SDL3METAL_RenderData *data = NULL;
        id<MTLDevice> mtldevice = nil;
        SDL_MetalView view = NULL;
        CAMetalLayer *layer = nil;
        NSError *err = nil;
        dispatch_data_t mtllibdata;
        char *constantdata;
        int maxtexsize, quadcount = UINT16_MAX / 4;
        UInt16 *indexdata;
        size_t indicessize = sizeof(UInt16) * quadcount * 6;
        id<MTLCommandQueue> mtlcmdqueue;
        id<MTLLibrary> mtllibrary;
        id<MTLBuffer> mtlbufconstantstaging, mtlbufquadindicesstaging, mtlbufconstants, mtlbufquadindices;
        id<MTLCommandBuffer> cmdbuffer;
        id<MTLBlitCommandEncoder> blitcmd;
        bool scRGB_supported = false;

        // Note: matrices are column major.
        float identitytransform[16] = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        };

        float halfpixeltransform[16] = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.5f,
            0.5f,
            0.0f,
            1.0f,
        };

        const size_t YCbCr_shader_matrix_size = 4 * 4 * sizeof(float);

        SDL_SetupRendererColorspace(renderer, create_props);

#ifndef SDL_PLATFORM_TVOS
        if (@available(macos 10.11, iOS 16.0, *)) {
            scRGB_supported = true;
        }
#endif
        if (renderer->output_colorspace != SDL_COLORSPACE_SRGB) {
            if (renderer->output_colorspace == SDL_COLORSPACE_SRGB_LINEAR && scRGB_supported) {
                // This colorspace is supported
            } else {
                return SDL_SetError("Unsupported output colorspace");
            }
        }

#ifdef SDL_PLATFORM_MACOS
        if (SDL_GetHintBoolean(SDL_HINT_RENDER_METAL_PREFER_LOW_POWER_DEVICE, true)) {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();

            for (id<MTLDevice> device in devices) {
                if (device.isLowPower) {
                    mtldevice = device;
                    break;
                }
            }
        }
#endif

        if (mtldevice == nil) {
            mtldevice = MTLCreateSystemDefaultDevice();
        }

        if (mtldevice == nil) {
            return SDL_SetError("Failed to obtain Metal device");
        }

        view = GetWindowView(window);
        if (view == nil) {
            view = SDL_Metal_CreateView(window);
        }

        if (view == NULL) {
            return false;
        }

        // !!! FIXME: error checking on all of this.
        data = [[SDL3METAL_RenderData alloc] init];

        if (data == nil) {
            /* Release the metal view instead of destroying it,
               in case we want to use it later (recreating the renderer)
             */
            // SDL_Metal_DestroyView(view);
            CFBridgingRelease(view);
            return SDL_SetError("SDL3METAL_RenderData alloc/init failed");
        }

        renderer->internal = (void *)CFBridgingRetain(data);
        METAL_InvalidateCachedState(renderer);
        renderer->window = window;

        data.mtlview = view;

#ifdef SDL_PLATFORM_MACOS
        layer = (CAMetalLayer *)[(__bridge NSView *)view layer];
#else
        layer = (CAMetalLayer *)[(__bridge UIView *)view layer];
#endif

#ifndef SDL_PLATFORM_TVOS
        if (renderer->output_colorspace == SDL_COLORSPACE_SRGB_LINEAR) {
            if (@available(macos 10.11, iOS 16.0, *)) {
                layer.wantsExtendedDynamicRangeContent = YES;
            } else {
                SDL_assert(!"Logic error, scRGB is not actually supported");
            }
            layer.pixelFormat = MTLPixelFormatRGBA16Float;

            const CFStringRef name = kCGColorSpaceExtendedLinearSRGB;
            CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(name);
            layer.colorspace = colorspace;
            CGColorSpaceRelease(colorspace);
        }
#endif // !SDL_PLATFORM_TVOS

        layer.device = mtldevice;

        // Necessary for RenderReadPixels.
        layer.framebufferOnly = NO;

        data.mtldevice = layer.device;
        data.mtllayer = layer;
        mtlcmdqueue = [data.mtldevice newCommandQueue];
        data.mtlcmdqueue = mtlcmdqueue;
        data.mtlcmdqueue.label = @"SDL Metal Renderer";
        data.mtlpassdesc = [MTLRenderPassDescriptor renderPassDescriptor];

        // The compiled .metallib is embedded in a static array in a header file
        // but the original shader source code is in SDL_shaders_metal.metal.
        mtllibdata = dispatch_data_create(sdl_metallib, sdl_metallib_len, dispatch_get_global_queue(0, 0), ^{
                                                                          });
        mtllibrary = [data.mtldevice newLibraryWithData:mtllibdata error:&err];
        data.mtllibrary = mtllibrary;
        SDL_assert(err == nil);
        data.mtllibrary.label = @"SDL Metal renderer shader library";

        // Do some shader pipeline state loading up-front rather than on demand.
        data.pipelinescount = 0;
        data.allpipelines = NULL;
        ChooseShaderPipelines(data, MTLPixelFormatBGRA8Unorm);

        data.mtlsamplers = [[NSMutableDictionary<NSNumber *, id<MTLSamplerState>> alloc] init];

        mtlbufconstantstaging = [data.mtldevice newBufferWithLength:CONSTANTS_LENGTH options:MTLResourceStorageModeShared];

        constantdata = [mtlbufconstantstaging contents];
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_IDENTITY, identitytransform, sizeof(identitytransform));
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_HALF_PIXEL_TRANSFORM, halfpixeltransform, sizeof(halfpixeltransform));
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_DECODE_BT601_LIMITED, SDL_GetYCbCRtoRGBConversionMatrix(SDL_COLORSPACE_BT601_LIMITED, 0, 0, 8), YCbCr_shader_matrix_size);
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_DECODE_BT601_FULL, SDL_GetYCbCRtoRGBConversionMatrix(SDL_COLORSPACE_BT601_FULL, 0, 0, 8), YCbCr_shader_matrix_size);
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_DECODE_BT709_LIMITED, SDL_GetYCbCRtoRGBConversionMatrix(SDL_COLORSPACE_BT709_LIMITED, 0, 0, 8), YCbCr_shader_matrix_size);
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_DECODE_BT709_FULL, SDL_GetYCbCRtoRGBConversionMatrix(SDL_COLORSPACE_BT709_FULL, 0, 0, 8), YCbCr_shader_matrix_size);
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_DECODE_BT2020_LIMITED, SDL_GetYCbCRtoRGBConversionMatrix(SDL_COLORSPACE_BT2020_LIMITED, 0, 0, 10), YCbCr_shader_matrix_size);
        SDL_memcpy(constantdata + CONSTANTS_OFFSET_DECODE_BT2020_FULL, SDL_GetYCbCRtoRGBConversionMatrix(SDL_COLORSPACE_BT2020_FULL, 0, 0, 10), YCbCr_shader_matrix_size);

        mtlbufquadindicesstaging = [data.mtldevice newBufferWithLength:indicessize options:MTLResourceStorageModeShared];

        /* Quads in the following vertex order (matches the FillRects vertices):
         * 1---3
         * | \ |
         * 0---2
         */
        indexdata = [mtlbufquadindicesstaging contents];
        for (int i = 0; i < quadcount; i++) {
            indexdata[i * 6 + 0] = i * 4 + 0;
            indexdata[i * 6 + 1] = i * 4 + 1;
            indexdata[i * 6 + 2] = i * 4 + 2;

            indexdata[i * 6 + 3] = i * 4 + 2;
            indexdata[i * 6 + 4] = i * 4 + 1;
            indexdata[i * 6 + 5] = i * 4 + 3;
        }

        mtlbufconstants = [data.mtldevice newBufferWithLength:CONSTANTS_LENGTH options:MTLResourceStorageModePrivate];
        data.mtlbufconstants = mtlbufconstants;
        data.mtlbufconstants.label = @"SDL constant data";

        mtlbufquadindices = [data.mtldevice newBufferWithLength:indicessize options:MTLResourceStorageModePrivate];
        data.mtlbufquadindices = mtlbufquadindices;
        data.mtlbufquadindices.label = @"SDL quad index buffer";

        cmdbuffer = [data.mtlcmdqueue commandBuffer];
        blitcmd = [cmdbuffer blitCommandEncoder];

        [blitcmd copyFromBuffer:mtlbufconstantstaging sourceOffset:0 toBuffer:mtlbufconstants destinationOffset:0 size:CONSTANTS_LENGTH];
        [blitcmd copyFromBuffer:mtlbufquadindicesstaging sourceOffset:0 toBuffer:mtlbufquadindices destinationOffset:0 size:indicessize];

        [blitcmd endEncoding];
        [cmdbuffer commit];

        // !!! FIXME: force more clears here so all the drawables are sane to start, and our static buffers are definitely flushed.

        renderer->WindowEvent = METAL_WindowEvent;
        renderer->GetOutputSize = METAL_GetOutputSize;
        renderer->SupportsBlendMode = METAL_SupportsBlendMode;
        renderer->CreateTexture = METAL_CreateTexture;
        renderer->UpdateTexture = METAL_UpdateTexture;
#ifdef SDL_HAVE_YUV
        renderer->UpdateTextureYUV = METAL_UpdateTextureYUV;
        renderer->UpdateTextureNV = METAL_UpdateTextureNV;
#endif
        renderer->LockTexture = METAL_LockTexture;
        renderer->UnlockTexture = METAL_UnlockTexture;
        renderer->SetRenderTarget = METAL_SetRenderTarget;
        renderer->QueueSetViewport = METAL_QueueSetViewport;
        renderer->QueueSetDrawColor = METAL_QueueNoOp;
        renderer->QueueDrawPoints = METAL_QueueDrawPoints;
        renderer->QueueDrawLines = METAL_QueueDrawLines;
        renderer->QueueGeometry = METAL_QueueGeometry;
        renderer->InvalidateCachedState = METAL_InvalidateCachedState;
        renderer->RunCommandQueue = METAL_RunCommandQueue;
        renderer->RenderReadPixels = METAL_RenderReadPixels;
        renderer->RenderPresent = METAL_RenderPresent;
        renderer->DestroyTexture = METAL_DestroyTexture;
        renderer->DestroyRenderer = METAL_DestroyRenderer;
        renderer->SetVSync = METAL_SetVSync;
        renderer->GetMetalLayer = METAL_GetMetalLayer;
        renderer->GetMetalCommandEncoder = METAL_GetMetalCommandEncoder;

        renderer->name = METAL_RenderDriver.name;
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ARGB8888);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR8888);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_ABGR2101010);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBA64_FLOAT);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_RGBA128_FLOAT);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_YV12);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_IYUV);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV12);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_NV21);
        SDL_AddSupportedTextureFormat(renderer, SDL_PIXELFORMAT_P010);

#if defined(SDL_PLATFORM_MACOS) || TARGET_OS_MACCATALYST
        data.mtllayer.displaySyncEnabled = NO;
#endif

        // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        maxtexsize = 4096;
#if defined(SDL_PLATFORM_MACOS) || TARGET_OS_MACCATALYST
        maxtexsize = 16384;
#elif defined(SDL_PLATFORM_TVOS)
        maxtexsize = 8192;
        if ([mtldevice supportsFeatureSet:MTLFeatureSet_tvOS_GPUFamily2_v1]) {
            maxtexsize = 16384;
        }
#else
        if ([mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1]) {
            maxtexsize = 16384;
        } else if ([mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily3_v1]) {
            maxtexsize = 16384;
        } else if ([mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily2_v2] || [mtldevice supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily1_v2]) {
            maxtexsize = 8192;
        } else {
            maxtexsize = 4096;
        }
#endif

        SDL_SetNumberProperty(SDL_GetRendererProperties(renderer), SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, maxtexsize);

        return true;
    }
}

SDL_RenderDriver METAL_RenderDriver = {
    METAL_CreateRenderer, "metal"
};

#endif // SDL_VIDEO_RENDER_METAL

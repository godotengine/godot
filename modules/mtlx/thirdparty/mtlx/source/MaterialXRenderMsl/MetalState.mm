//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include "MetalState.h"
#import <Metal/Metal.h>

#include <MaterialXRenderMsl/MetalFramebuffer.h>

std::unique_ptr<MetalState> MetalState::singleton = nullptr;

MetalState::MetalState()
{
}

void MetalState::initialize(id<MTLDevice> mtlDevice, id<MTLCommandQueue> mtlCmdQueue)
{
    device = mtlDevice;
    cmdQueue = mtlCmdQueue;
    
#ifdef MAC_OS_VERSION_11_0
    if (@available(macOS 11.0, ios 14.0, *))
    {
        supportsTiledPipeline = [device supportsFamily:MTLGPUFamilyApple4];
    }
#else
    supportsTiledPipeline = false;
#endif
    
    MTLDepthStencilDescriptor* depthStencilDesc = [MTLDepthStencilDescriptor new];
    depthStencilDesc.depthWriteEnabled    = true;
    depthStencilDesc.depthCompareFunction = MTLCompareFunctionLess;
    opaqueDepthStencilState = [device newDepthStencilStateWithDescriptor:depthStencilDesc];
       
    depthStencilDesc.depthWriteEnabled    = false;
    depthStencilDesc.depthCompareFunction = MTLCompareFunctionLess;
    transparentDepthStencilState = [device newDepthStencilStateWithDescriptor:depthStencilDesc];
       
    depthStencilDesc.depthWriteEnabled    = true;
    depthStencilDesc.depthCompareFunction = MTLCompareFunctionAlways;
    envMapDepthStencilState = [device newDepthStencilStateWithDescriptor:depthStencilDesc];
    
    initLinearToSRGBKernel();
}

void MetalState::initLinearToSRGBKernel()
{
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
#ifdef MAC_OS_VERSION_11_0
    if (@available(macOS 11.0, ios 14.0, *))
        options.languageVersion = MTLLanguageVersion2_3;
    else
#endif
        options.languageVersion = MTLLanguageVersion2_0;
    options.fastMathEnabled = true;
    
#ifdef MAC_OS_VERSION_11_0
    bool useTiledPipeline = supportsTiledPipeline;
    if(useTiledPipeline)
    {
        if(@available(macOS 11.0, ios 14.0, *))
        {
            NSString* linearToSRGB_kernel =
            @"#include <metal_stdlib>                                                        \n"
            "#include <simd/simd.h>                                                          \n"
            "                                                                                \n"
            "using namespace metal;                                                          \n"
            "                                                                                \n"
            "struct RenderTarget {                                                           \n"
            "    half4 colorTarget [[color(0)]];                                             \n"
            "};                                                                              \n"
            "                                                                                \n"
            "                                                                                \n"
            "                                                                                \n"
            "half4 linearToSRGB(half4 color_linear)                                          \n"
            "{                                                                               \n"
            "    half4 color_srgb;                                                           \n"
            "    for(int i = 0; i < 3; ++i)                                                  \n"
            "        color_srgb[i] = (color_linear[i] < 0.0031308) ?                         \n"
            "            (12.92 * color_linear[i])                 :                         \n"
            "            (1.055 * pow(color_linear[i], 1.0h / 2.2h) - 0.055);                \n"
            "    color_srgb[3] = color_linear[3];                                            \n"
            "    return color_srgb;                                                          \n"
            "}                                                                               \n"
            "                                                                                \n"
            "kernel void LinearToSRGB_kernel(                                                \n"
            "    imageblock<RenderTarget,imageblock_layout_implicit> imageBlock,             \n"
            "    ushort2 tid                 [[ thread_position_in_threadgroup ]])           \n"
            "{                                                                               \n"
            "    RenderTarget linearValue = imageBlock.read(tid);                            \n"
            "    RenderTarget srgbValue;                                                     \n"
            "    srgbValue.colorTarget = linearToSRGB(linearValue.colorTarget);              \n"
            "    imageBlock.write(srgbValue, tid);                                           \n"
            "}                                                                               \n";
            
            id<MTLLibrary> library = [device newLibraryWithSource:linearToSRGB_kernel options:options error:&error];
            id<MTLFunction> function = [library newFunctionWithName:@"LinearToSRGB_kernel"];
            
            MTLTileRenderPipelineDescriptor* renderPipelineDescriptor = [MTLTileRenderPipelineDescriptor new];
            [renderPipelineDescriptor setRasterSampleCount:1];
            [[renderPipelineDescriptor colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm];
            [renderPipelineDescriptor setTileFunction:function];
            linearToSRGB_pso = [device newRenderPipelineStateWithTileDescriptor:renderPipelineDescriptor options:0 reflection:nil error:&error];
        }
        else
        {
            useTiledPipeline = false;
        }
    }
    
    if(!useTiledPipeline)
#endif
    {
        NSString* linearToSRGB_kernel =
        @"#include <metal_stdlib>                                       \n"
         "#include <simd/simd.h>                                        \n"
         "                                                              \n"
         "using namespace metal;                                        \n"
         "                                                              \n"
         "struct VSOutput                                               \n"
         "{                                                             \n"
         "    float4 position [[position]];                             \n"
         "};                                                            \n"
         "                                                              \n"
         "vertex VSOutput VertexMain(uint vertexId [[ vertex_id ]])     \n"
         "{                                                             \n"
         "    VSOutput vsOut;                                           \n"
         "                                                              \n"
         "    switch(vertexId)                                          \n"
         "    {                                                         \n"
         "    case 0: vsOut.position = float4(-1, -1, 0.5, 1); break;   \n"
         "    case 1: vsOut.position = float4(-1,  3, 0.5, 1); break;   \n"
         "    case 2: vsOut.position = float4( 3, -1, 0.5, 1); break;   \n"
         "    };                                                        \n"
         "                                                              \n"
         "    return vsOut;                                             \n"
         "}                                                             \n"
         "                                                              \n"
         "half4 linearToSRGB(half4 color_linear)                        \n"
         "{                                                             \n"
         "    half4 color_srgb;                                         \n"
         "    for(int i = 0; i < 3; ++i)                                \n"
         "        color_srgb[i] = (color_linear[i] < 0.0031308) ?       \n"
         "          (12.92 * color_linear[i])                   :       \n"
         "          (1.055 * pow(color_linear[i], 1.0h / 2.2h) - 0.055);\n"
         "    color_srgb[3] = color_linear[3];                          \n"
         "    return color_srgb;                                        \n"
         "}                                                             \n"
         "                                                              \n"
         "fragment half4 FragmentMain(                                  \n"
         "    texture2d<half>  inputTex  [[ texture(0) ]],              \n"
         "    float4           fragCoord [[ position ]]                 \n"
         ")                                                             \n"
         "{                                                             \n"
         "    constexpr sampler ss(                                     \n"
         "        coord::pixel,                                         \n"
         "        address::clamp_to_border,                             \n"
         "        filter::linear);                                      \n"
         "    return linearToSRGB(inputTex.sample(ss, fragCoord.xy));   \n"
         "}                                                             \n";

        id<MTLLibrary> library = [device newLibraryWithSource:linearToSRGB_kernel options:options error:&error];

        id<MTLFunction> vertexfunction = [library newFunctionWithName:@"VertexMain"];
        id<MTLFunction> Fragmentfunction = [library newFunctionWithName:@"FragmentMain"];
        
        MTLRenderPipelineDescriptor* renderPipelineDesc = [MTLRenderPipelineDescriptor new];
        [renderPipelineDesc setVertexFunction:vertexfunction];
        [renderPipelineDesc setFragmentFunction:Fragmentfunction];
        [[renderPipelineDesc colorAttachments][0] setPixelFormat:MTLPixelFormatBGRA8Unorm];
        [renderPipelineDesc setDepthAttachmentPixelFormat:MTLPixelFormatDepth32Float];
        linearToSRGB_pso = [device newRenderPipelineStateWithDescriptor:renderPipelineDesc error:&error];
    }
}

void MetalState::triggerProgrammaticCapture()
{
    MTLCaptureManager*    captureManager    = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor* captureDescriptor = [MTLCaptureDescriptor new];
   
    [captureDescriptor setCaptureObject:device];
    
    NSError* error = nil;
    if(![captureManager startCaptureWithDescriptor:captureDescriptor error:&error])
    {
        NSLog(@"Failed to start capture, error %@", error);
    }
}

void MetalState::stopProgrammaticCapture()
{
    MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
    [captureManager stopCapture];
}

void MetalState::beginCommandBuffer()
{
    cmdBuffer = [cmdQueue commandBuffer];
    inFlightCommandBuffers++;
}

void MetalState::beginEncoder(MTLRenderPassDescriptor* renderpassDesc)
{
    renderCmdEncoder = [cmdBuffer
                        renderCommandEncoderWithDescriptor:renderpassDesc];
}

void MetalState::endEncoder()
{
    [renderCmdEncoder endEncoding];
}

void MetalState::endCommandBuffer()
{
    endEncoder();
    [cmdBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
        inFlightCommandBuffers--;
        inFlightCV.notify_one();
    }];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}

void MetalState::waitForComplition()
{
    std::unique_lock<std::mutex> lock(inFlightMutex);
    while (inFlightCommandBuffers != 0){
        inFlightCV.wait(lock, [this]{ return inFlightCommandBuffers.load() == 0; });
    }
}

MaterialX::MetalFramebufferPtr MetalState::currentFramebuffer()
{
    return framebufferStack.top();
}

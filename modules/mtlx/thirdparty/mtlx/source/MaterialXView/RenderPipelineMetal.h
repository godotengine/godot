//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef RENDER_PIPELINE_METAL_H
#define RENDER_PIPELINE_METAL_H

#include <MaterialXView/RenderPipeline.h>

MATERIALX_NAMESPACE_BEGIN
using MetalFramebufferPtr = std::shared_ptr<class MetalFramebuffer>;
MATERIALX_NAMESPACE_END

#define SHADOWMAP_TEX_COUNT 2

class Viewer;
using MetalRenderPipelinePtr = std::shared_ptr<class MetalRenderPipeline>;

class MetalRenderPipeline : public RenderPipeline
{
  public:
    ~MetalRenderPipeline() { }
    
    static MetalRenderPipelinePtr create(Viewer* viewer)
    {
        return std::make_shared<MetalRenderPipeline>(viewer);
    }
    
    std::shared_ptr<void> createTextureBaker(unsigned int width,
                                             unsigned int height,
                                             mx::Image::BaseType baseType) override;
    
    void initialize(void* metal_device, void* metal_cmd_queue) override;
    
    void initFramebuffer(int width, int height,
                         void* color_texture) override;
    void resizeFramebuffer(int width, int height,
                         void* color_texture) override;
    mx::ImageHandlerPtr createImageHandler() override;
    mx::MaterialPtr     createMaterial() override;
    void updateAlbedoTable(int tableSize) override;
    void updatePrefilteredMap() override;
    void renderFrame(void* color_texture, int shadowMapSize, const char* dirLightNodeCat) override;
    void bakeTextures() override;
    mx::ImagePtr getFrameImage() override;
    
  public:
    MetalRenderPipeline(Viewer* viewerPtr);
    
  protected:
    mx::ImagePtr getShadowMap(int shadowMapSize) override;
    mx::MetalFramebufferPtr  _shadowMapFramebuffer;
    mx::MetalFramebufferPtr  _prefilterFramebuffer;
    mx::ImagePtr             _shadowMap[SHADOWMAP_TEX_COUNT];
};
    
#endif // RENDER_PIPELINE_METAL_H

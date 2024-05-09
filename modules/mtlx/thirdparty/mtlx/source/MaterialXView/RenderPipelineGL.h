//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef RENDER_PIPELINE_GL_H
#define RENDER_PIPELINE_GL_H

#include <MaterialXView/RenderPipeline.h>

class Viewer;
using GLRenderPipelinePtr = std::shared_ptr<class GLRenderPipeline>;

class GLRenderPipeline : public RenderPipeline
{
  public:
    ~GLRenderPipeline() { }
    
    static GLRenderPipelinePtr create(Viewer* viewer)
    {
        return std::make_shared<GLRenderPipeline>(viewer);
    }
    
    void initialize(void* metal_device, void* metal_cmd_queue) override;
    
    void initFramebuffer(int width, int height,
                         void* color_texture) override;
    void resizeFramebuffer(int width, int height,
                         void* color_texture) override;
    
    mx::ImageHandlerPtr createImageHandler() override;
    mx::MaterialPtr     createMaterial() override;
    void updateAlbedoTable(int tableSize) override;
    void updatePrefilteredMap() override;
    std::shared_ptr<void> createTextureBaker(unsigned int width,
                                             unsigned int height,
                                             mx::Image::BaseType baseType) override;
    void renderFrame(void* color_texture, int shadowMapSize, const char* dirLightNodeCat) override;
    void bakeTextures() override;
    mx::ImagePtr getFrameImage() override;
    
  public:
    GLRenderPipeline(Viewer* viewerPtr);
    
  protected:
    mx::ImagePtr getShadowMap(int shadowMapSize) override;
};
    
#endif // RENDER_PIPELINE_GL_H

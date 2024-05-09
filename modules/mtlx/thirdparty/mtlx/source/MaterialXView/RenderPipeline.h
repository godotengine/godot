//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef RENDER_PIPELINE_H
#define RENDER_PIPELINE_H

#include <MaterialXView/Editor.h>

#include <MaterialXRender/ShaderMaterial.h>
#include <MaterialXRender/Camera.h>
#include <MaterialXRender/GeometryHandler.h>
#include <MaterialXRender/LightHandler.h>
#include <MaterialXRender/ImageHandler.h>
#include <MaterialXRender/Image.h>

#include <MaterialXCore/Value.h>
#include <MaterialXCore/Unit.h>

MATERIALX_NAMESPACE_BEGIN
#ifdef MATERIALXVIEW_METAL_BACKEND
using TextureBakerPtr = shared_ptr<class TextureBakerMsl>;
#else
using TextureBakerPtr = shared_ptr<class TextureBakerGlsl>;
#endif
MATERIALX_NAMESPACE_END

#include <memory>

namespace mx = MaterialX;

class Viewer;
using RenderPipelinePtr = std::shared_ptr<class RenderPipeline>;

class RenderPipeline
{
  public:
    RenderPipeline() = delete;
    RenderPipeline(Viewer* viewer)
    {
        _viewer = viewer;
    }
    virtual ~RenderPipeline() { }

    virtual void initialize(void* device, void* command_queue) = 0;

    virtual mx::ImageHandlerPtr createImageHandler() = 0;
    virtual mx::MaterialPtr createMaterial() = 0;
    virtual void bakeTextures() = 0;

    virtual void updateAlbedoTable(int tableSize) = 0;
    virtual void updatePrefilteredMap() = 0;
    virtual std::shared_ptr<void> createTextureBaker(unsigned int width,
                                                     unsigned int height,
                                                     mx::Image::BaseType baseType) = 0;

    virtual void renderFrame(void* color_texture, int shadowMapSize, const char* dirLightNodeCat) = 0;

    virtual void initFramebuffer(int width, int height,
                                 void* color_texture) = 0;
    virtual void resizeFramebuffer(int width, int height,
                                   void* color_texture) = 0;

    virtual mx::ImagePtr getShadowMap(int shadowMapSize) = 0;

    virtual mx::ImagePtr getFrameImage() = 0;

  public:
    Viewer* _viewer;
};
#endif // RENDER_PIPELINE_H

//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXView/RenderPipelineMetal.h>
#include <MaterialXView/Viewer.h>
#include <MaterialXRenderMsl/MetalTextureHandler.h>

#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXRenderMsl/MetalState.h>
#include <MaterialXRenderMsl/TextureBaker.h>
#include <MaterialXRenderMsl/MetalFramebuffer.h>
#include <MaterialXRenderMsl/MslMaterial.h>

#include <nanogui/messagedialog.h>

namespace
{

const float PI = std::acos(-1.0f);

}

MetalRenderPipeline::MetalRenderPipeline(Viewer* viewerPtr) :
    RenderPipeline(viewerPtr)
{
}

void MetalRenderPipeline::initialize(void* metal_device, void* metal_cmd_queue)
{
    MTL(initialize((id<MTLDevice>)metal_device,
                   (id<MTLCommandQueue>)metal_cmd_queue));
}

mx::ImageHandlerPtr MetalRenderPipeline::createImageHandler()
{
    return mx::MetalTextureHandler::create(
        MTL(device),
        mx::StbImageLoader::create());
}

mx::MaterialPtr MetalRenderPipeline::createMaterial()
{
    return mx::MslMaterial::create();
}

std::shared_ptr<void> MetalRenderPipeline::createTextureBaker(unsigned int width,
                                                              unsigned int height,
                                                              mx::Image::BaseType baseType)
{
    return std::static_pointer_cast<void>(mx::TextureBakerMsl::create(width, height, baseType));
}

void MetalRenderPipeline::initFramebuffer(int width, int height,
                                          void* color_texture)
{
    MTL_PUSH_FRAMEBUFFER(mx::MetalFramebuffer::create(
                            MTL(device),
                            width * _viewer->m_pixel_ratio,
                            height * _viewer->m_pixel_ratio,
                            4, mx::Image::BaseType::UINT8,
                            MTL(supportsTiledPipeline) ?
                            (id<MTLTexture>)color_texture : nil,
                            false,  MTLPixelFormatBGRA8Unorm));
}

void MetalRenderPipeline::resizeFramebuffer(int width, int height,
                                            void* color_texture)
{
    MTL_POP_FRAMEBUFFER();
    initFramebuffer(width, height, color_texture);
}

void MetalRenderPipeline::updateAlbedoTable(int tableSize)
{
    auto& genContext    = _viewer->_genContext;
    auto& lightHandler  = _viewer->_lightHandler;
    auto& imageHandler  = _viewer->_imageHandler;
    
    if (lightHandler->getAlbedoTable())
    {
        return;
    }
    
    // Create framebuffer.
    mx::MetalFramebufferPtr framebuffer = mx::MetalFramebuffer::create(MTL(device),
                            tableSize, tableSize,
                            2,
                            mx::Image::BaseType::FLOAT);
        
        bool captureCommandBuffer = false;
        if(captureCommandBuffer)
            MTL_TRIGGER_CAPTURE;
    
        MTL_PUSH_FRAMEBUFFER(framebuffer);
        
        MTL(beginCommandBuffer());
        
        MTLRenderPassDescriptor* renderpassDesc = [MTLRenderPassDescriptor new];
        
        [renderpassDesc.colorAttachments[0] setTexture:framebuffer->getColorTexture()];
        [renderpassDesc.colorAttachments[0] setClearColor:MTLClearColorMake(0.0f, 0.0f, 0.0f, 0.0f)];
        [renderpassDesc.colorAttachments[0] setLoadAction:MTLLoadActionClear];
        [renderpassDesc.colorAttachments[0] setStoreAction:MTLStoreActionStore];
        
        [renderpassDesc.depthAttachment setTexture:framebuffer->getDepthTexture()];
        [renderpassDesc.depthAttachment setClearDepth:1.0];
        [renderpassDesc.depthAttachment setLoadAction:MTLLoadActionClear];
        [renderpassDesc.depthAttachment setStoreAction:MTLStoreActionStore];
        [renderpassDesc setStencilAttachment:nil];
        
        MTL(beginEncoder(renderpassDesc));

    // Create shader.
    mx::ShaderPtr hwShader = mx::createAlbedoTableShader(genContext, _viewer->_stdLib, "__ALBEDO_TABLE_SHADER__");
    mx::MslMaterialPtr material = mx::MslMaterial::create();
    try
    {
        material->generateShader(hwShader);
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(_viewer, ng::MessageDialog::Type::Warning, "Failed to generate albedo table shader", e.what());
        return;
    }

    // Render albedo table.
    material->bindShader();
    if (material->getProgram()->hasUniform(mx::HW::ALBEDO_TABLE_SIZE))
    {
        material->getProgram()->bindUniform(mx::HW::ALBEDO_TABLE_SIZE, mx::Value::createValue(tableSize));
    }
    material->getProgram()->prepareUsedResources(
                    MTL(renderCmdEncoder),
                    _viewer->_identityCamera,
                    nullptr,
                    imageHandler,
                    lightHandler);
    _viewer->renderScreenSpaceQuad(material);
    
    MTL(endCommandBuffer());
    
    MTL_POP_FRAMEBUFFER();
    
    if(captureCommandBuffer)
        MTL_STOP_CAPTURE;

    // Store albedo table image.
    imageHandler->releaseRenderResources(lightHandler->getAlbedoTable());
    lightHandler->setAlbedoTable(framebuffer->getColorImage(MTL(cmdQueue)));
    if (_viewer->_saveGeneratedLights)
    {
        imageHandler->saveImage("AlbedoTable.exr", lightHandler->getAlbedoTable());
    }
}

void MetalRenderPipeline::updatePrefilteredMap()
{
    auto& genContext    = _viewer->_genContext;
    auto& lightHandler  = _viewer->_lightHandler;
    auto& imageHandler  = _viewer->_imageHandler;

    if (lightHandler->getEnvPrefilteredMap())
    {
        return;
    }

    mx::ImagePtr srcTex = lightHandler->getEnvRadianceMap();
    int w = srcTex->getWidth();
    int h = srcTex->getHeight();

    mx::MetalTextureHandlerPtr mtlImageHandler = std::dynamic_pointer_cast<mx::MetalTextureHandler>(imageHandler);
    mx::ImagePtr outTex = mx::Image::create(w, h, 3, mx::Image::BaseType::HALF);
    mtlImageHandler->createRenderResources(outTex, true, true);
    id<MTLTexture> metalTex = mtlImageHandler->getAssociatedMetalTexture(outTex);

    // Create framebuffer.
    _prefilterFramebuffer = mx::MetalFramebuffer::create(
        MTL(device),
        w, h,
        4,
        mx::Image::BaseType::HALF,
        metalTex
    );

    MTL_PUSH_FRAMEBUFFER(_prefilterFramebuffer);

    // Create shader.
    mx::ShaderPtr hwShader = mx::createEnvPrefilterShader(genContext, _viewer->_stdLib, "__ENV_PREFILTER__");
    mx::MslMaterialPtr material = mx::MslMaterial::create();
    try
    {
        material->generateShader(hwShader);
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(_viewer, ng::MessageDialog::Type::Warning, "Failed to generate convolution shader", e.what());
    }

    int i = 0;

    while (w > 0 && h > 0)
    {
        MTL(beginCommandBuffer());
        MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor new];
        [desc.colorAttachments[0] setTexture:metalTex];
        [desc.colorAttachments[0] setLevel:i];
        [desc.colorAttachments[0] setLoadAction:MTLLoadActionDontCare];
        [desc.colorAttachments[0] setStoreAction:MTLStoreActionStore];
        [desc.depthAttachment setTexture:_prefilterFramebuffer->getDepthTexture()];
        [desc.depthAttachment setLoadAction:MTLLoadActionDontCare];
        [desc.depthAttachment setStoreAction:MTLStoreActionDontCare];
        [desc setStencilAttachment:nil];
        
        MTL(beginEncoder(desc));
        [MTL(renderCmdEncoder) setDepthStencilState:MTL_DEPTHSTENCIL_STATE(opaque)];

        _prefilterFramebuffer->bind(desc);
        material->bindShader();
        material->getProgram()->bindUniform(mx::HW::ENV_PREFILTER_MIP, mx::Value::createValue(i));

        bool prevValue = lightHandler->getUsePrefilteredMap();
        lightHandler->setUsePrefilteredMap(false);
        material->getProgram()->prepareUsedResources(
                        MTL(renderCmdEncoder),
                        _viewer->_identityCamera,
                        nullptr,
                        imageHandler,
                        lightHandler);
        lightHandler->setUsePrefilteredMap(prevValue);

        _viewer->renderScreenSpaceQuad(material);

        MTL(endCommandBuffer());
        [desc release];

        w /= 2;
        h /= 2;
        i++;
    }

    MTL_POP_FRAMEBUFFER();

    lightHandler->setEnvPrefilteredMap(outTex);
}

mx::ImagePtr MetalRenderPipeline::getShadowMap(int shadowMapSize)
{
    auto& genContext      = _viewer->_genContext;
    auto& lightHandler    = _viewer->_lightHandler;
    auto& imageHandler    = _viewer->_imageHandler;
    auto& shadowCamera    = _viewer->_shadowCamera;
    auto& stdLib          = _viewer->_stdLib;
    auto& geometryHandler = _viewer->_geometryHandler;
    auto& identityCamera  = _viewer->_identityCamera;
    
    mx::MetalTextureHandlerPtr mtlImageHandler =
        std::dynamic_pointer_cast<mx::MetalTextureHandler>(imageHandler);
    
    id<MTLTexture> shadowMapTex[SHADOWMAP_TEX_COUNT];
    for(int i = 0; i < SHADOWMAP_TEX_COUNT; ++i)
    {
        if(!_shadowMap[i] || _shadowMap[i]->getWidth() != shadowMapSize ||
           !mtlImageHandler->getAssociatedMetalTexture(_shadowMap[i]))
        {
            _shadowMap[i] = mx::Image::create(shadowMapSize, shadowMapSize, 2, mx::Image::BaseType::FLOAT);
            _viewer->_imageHandler->createRenderResources(_shadowMap[i], false, true);
        }
        
        shadowMapTex[i] =
            mtlImageHandler->getAssociatedMetalTexture(_shadowMap[i]);
    }
    
    if (!_viewer->_shadowMap)
    {
        // Create framebuffer.
        if(!_shadowMapFramebuffer)
        {
            _shadowMapFramebuffer = mx::MetalFramebuffer::create(
                                                       MTL(device),
                                                       shadowMapSize,
                                                       shadowMapSize,
                                                       2,
                                                       mx::Image::BaseType::FLOAT,
                                                       shadowMapTex[0]);
        }
        MTL_PUSH_FRAMEBUFFER(_shadowMapFramebuffer);
        
        // Generate shaders for shadow rendering.
        if (!_viewer->_shadowMaterial)
        {
            try
            {
                mx::ShaderPtr hwShader = mx::createDepthShader(genContext, stdLib, "__SHADOW_SHADER__");
                _viewer->_shadowMaterial = mx::MslMaterial::create();
                _viewer->_shadowMaterial->generateShader(hwShader);
            }
            catch (std::exception& e)
            {
                std::cerr << "Failed to generate shadow shader: " << e.what() << std::endl;
                _viewer->_shadowMaterial = nullptr;
            }
        }
        if (!_viewer->_shadowBlurMaterial)
        {
            try
            {
                mx::ShaderPtr hwShader = mx::createBlurShader(genContext, stdLib, "__SHADOW_BLUR_SHADER__", "gaussian", 1.0f);
                _viewer->_shadowBlurMaterial = mx::MslMaterial::create();
                _viewer->_shadowBlurMaterial->generateShader(hwShader);
            }
            catch (std::exception& e)
            {
                std::cerr << "Failed to generate shadow blur shader: " << e.what() << std::endl;
                _viewer->_shadowBlurMaterial = nullptr;
            }
        }

        if (_viewer->_shadowMaterial && _viewer->_shadowBlurMaterial)
        {
            bool captureShadowGeneration = false;
            if(captureShadowGeneration)
                MTL_TRIGGER_CAPTURE;
            
            MTL(beginCommandBuffer());
            MTLRenderPassDescriptor* renderpassDesc = [MTLRenderPassDescriptor new];
            _shadowMapFramebuffer->setColorTexture(shadowMapTex[0]);
            _shadowMapFramebuffer->bind(renderpassDesc);
            MTL(beginEncoder(renderpassDesc));
            [MTL(renderCmdEncoder) setDepthStencilState:MTL_DEPTHSTENCIL_STATE(opaque)];

            // Render shadow geometry.
            _viewer->_shadowMaterial->bindShader();
            for (auto mesh : _viewer->_geometryHandler->getMeshes())
            {
                _viewer->_shadowMaterial->bindMesh(mesh);
                _viewer->_shadowMaterial->bindViewInformation(shadowCamera);
                std::static_pointer_cast<mx::MslMaterial>
                (_viewer->_shadowMaterial)->prepareUsedResources(
                            shadowCamera,
                            geometryHandler,
                            imageHandler,
                            lightHandler);
                for (size_t i = 0; i < mesh->getPartitionCount(); i++)
                {
                    mx::MeshPartitionPtr geom = mesh->getPartition(i);
                    _viewer->_shadowMaterial->drawPartition(geom);
                }
            }
            
            MTL(endCommandBuffer());
            
            // Apply Gaussian blurring.
            mx::ImageSamplingProperties blurSamplingProperties;
            blurSamplingProperties.uaddressMode = mx::ImageSamplingProperties::AddressMode::CLAMP;
            blurSamplingProperties.vaddressMode = mx::ImageSamplingProperties::AddressMode::CLAMP;
            blurSamplingProperties.filterType = mx::ImageSamplingProperties::FilterType::CLOSEST;
            for (unsigned int i = 0; i < _viewer->_shadowSoftness; i++)
            {
                MTL(beginCommandBuffer());
                _shadowMapFramebuffer->setColorTexture(shadowMapTex[(i+1) % 2]);
                _shadowMapFramebuffer->bind(renderpassDesc);
                MTL(beginEncoder(renderpassDesc));
                _shadowMapFramebuffer->bind(renderpassDesc);
                _viewer->_shadowBlurMaterial->bindShader();
                std::static_pointer_cast<mx::MslMaterial>
                (_viewer->_shadowBlurMaterial)->getProgram()->bindTexture(
                    _viewer->_imageHandler,
                    "image_file_tex",
                    _shadowMap[i % 2],
                    blurSamplingProperties);
                std::static_pointer_cast<mx::MslMaterial>
                (_viewer->_shadowBlurMaterial)->prepareUsedResources(
                    identityCamera,
                    geometryHandler,
                    imageHandler,
                    lightHandler);
                _viewer->_shadowBlurMaterial->unbindGeometry();
                _viewer->renderScreenSpaceQuad(_viewer->_shadowBlurMaterial);
                MTL(endCommandBuffer());
            }
            
            MTL_POP_FRAMEBUFFER();
            if(captureShadowGeneration)
                MTL_STOP_CAPTURE;
            
            [renderpassDesc release];
        }
    }

    _viewer->_shadowMap = _shadowMap[_viewer->_shadowSoftness % 2];
    return _viewer->_shadowMap;
}

void MetalRenderPipeline::renderFrame(void* color_texture, int shadowMapSize, const char* dirLightNodeCat)
{
    auto& genContext    = _viewer->_genContext;
    auto& lightHandler  = _viewer->_lightHandler;
    auto& imageHandler  = _viewer->_imageHandler;
    auto& viewCamera    = _viewer->_viewCamera;
    auto& envCamera     = _viewer->_envCamera;
    auto& shadowCamera  = _viewer->_shadowCamera;
    float lightRotation = _viewer->_lightRotation;
    auto& searchPath    = _viewer->_searchPath;
    auto& geometryHandler    = _viewer->_geometryHandler;
    
    // Update prefiltered environment.
    if (lightHandler->getUsePrefilteredMap() && !_viewer->_materialAssignments.empty())
    {
        updatePrefilteredMap();
    }

    // Update lighting state.
    lightHandler->setLightTransform(mx::Matrix44::createRotationY(lightRotation / 180.0f * M_PI));

    // Update shadow state.
    mx::ShadowState shadowState;
    shadowState.ambientOcclusionGain = _viewer->_ambientOcclusionGain;
    mx::NodePtr dirLight = lightHandler->getFirstLightOfCategory(dirLightNodeCat);
    if (genContext.getOptions().hwShadowMap && dirLight)
    {
        mx::ImagePtr shadowMap = getShadowMap(shadowMapSize);
        if (shadowMap)
        {
            shadowState.shadowMap = shadowMap;
            shadowState.shadowMatrix = viewCamera->getWorldMatrix().getInverse() *
                shadowCamera->getWorldViewProjMatrix();
        }
        else
        {
            genContext.getOptions().hwShadowMap = false;
        }
    }
    
    bool captureFrame = false;
    if(captureFrame)
        MTL_TRIGGER_CAPTURE;
    
    bool useTiledPipeline;
    if(@available(macOS 11.0, ios 14.0, *))
    {
        useTiledPipeline = MTL(supportsTiledPipeline);
    }
    else
    {
        useTiledPipeline = false;
    }
    
    MTL(beginCommandBuffer());
    MTLRenderPassDescriptor* renderpassDesc = [MTLRenderPassDescriptor new];
    if(useTiledPipeline)
    {
        [renderpassDesc.colorAttachments[0] setTexture:(id<MTLTexture>)color_texture];
    }
    else
    {
        [renderpassDesc.colorAttachments[0] setTexture:MTL(currentFramebuffer())->getColorTexture()];
    }
    [renderpassDesc.colorAttachments[0] setClearColor:MTLClearColorMake(
                                        _viewer->m_background[0],
                                        _viewer->m_background[1],
                                        _viewer->m_background[2],
                                        _viewer->m_background[3])];
    [renderpassDesc.colorAttachments[0] setLoadAction:MTLLoadActionClear];
    [renderpassDesc.colorAttachments[0] setStoreAction:MTLStoreActionStore];
    
    [renderpassDesc.depthAttachment setTexture:MTL(currentFramebuffer())->getDepthTexture()];
    [renderpassDesc.depthAttachment setClearDepth:1.0];
    [renderpassDesc.depthAttachment setLoadAction:MTLLoadActionClear];
    [renderpassDesc.depthAttachment setStoreAction:MTLStoreActionStore];
    [renderpassDesc setStencilAttachment:nil];
        
    MTL(beginEncoder(renderpassDesc));
        
    [MTL(renderCmdEncoder) setFrontFacingWinding:MTLWindingClockwise];

    // Environment background
    if (_viewer->_drawEnvironment)
    {
        [MTL(renderCmdEncoder) setDepthStencilState:MTL_DEPTHSTENCIL_STATE(envMap)];
        mx::MslMaterialPtr envMaterial = std::static_pointer_cast<mx::MslMaterial>(_viewer->getEnvironmentMaterial());
        if (envMaterial)
        {
            const mx::MeshList& meshes = _viewer->_envGeometryHandler->getMeshes();
            mx::MeshPartitionPtr envPart = !meshes.empty() ? meshes[0]->getPartition(0) : nullptr;
            if (envPart)
            {
                // Apply rotation to the environment shader.
                float longitudeOffset = (lightRotation / 360.0f) + 0.5f;
                envMaterial->modifyUniform("longitude/in2", mx::Value::createValue(longitudeOffset));

                // Apply light intensity to the environment shader.
                envMaterial->modifyUniform("envImageAdjusted/in2", mx::Value::createValue(lightHandler->getEnvLightIntensity()));

                // Render the environment mesh.
                [MTL(renderCmdEncoder) setCullMode:MTLCullModeNone];
                envMaterial->bindShader();
                envMaterial->bindMesh(meshes[0]);
                envMaterial->bindViewInformation(envCamera);
                envMaterial->bindImages(imageHandler, searchPath, false);
                envMaterial->prepareUsedResources(envCamera,
                                        _viewer->_envGeometryHandler,
                                        imageHandler,
                                        lightHandler);
                envMaterial->drawPartition(envPart);
                [MTL(renderCmdEncoder) setCullMode:MTLCullModeNone];
            }
        }
        else
        {
            _viewer->_drawEnvironment = false;
        }
    }

    // Enable backface culling if requested.
    if (!_viewer->_renderDoubleSided)
    {
        [MTL(renderCmdEncoder) setCullMode:MTLCullModeBack];
    }

    // Opaque pass
    [MTL(renderCmdEncoder) setDepthStencilState:MTL_DEPTHSTENCIL_STATE(opaque)];
    for (const auto& assignment : _viewer->_materialAssignments)
    {
        mx::MeshPartitionPtr geom = assignment.first;
        mx::MslMaterialPtr material = std::static_pointer_cast<mx::MslMaterial>(assignment.second);
        shadowState.ambientOcclusionMap = _viewer->getAmbientOcclusionImage(material);
        if (!material)
        {
            continue;
        }

        material->bindShader();
        material->bindMesh(_viewer->_geometryHandler->findParentMesh(geom));
        if (material->getProgram()->hasUniform(mx::HW::ALPHA_THRESHOLD))
        {
            material->getProgram()->bindUniform(mx::HW::ALPHA_THRESHOLD, mx::Value::createValue(0.99f));
        }
        material->bindViewInformation(viewCamera);
        material->bindLighting(lightHandler, imageHandler, shadowState);
        material->bindImages(imageHandler, _viewer->_searchPath);
        material->prepareUsedResources(viewCamera,
                             geometryHandler,
                             imageHandler,
                             lightHandler);
        material->drawPartition(geom);
        material->unbindImages(imageHandler);
    }

    // Transparent pass
    if (_viewer->_renderTransparency)
    {
        [MTL(renderCmdEncoder) setDepthStencilState:MTL_DEPTHSTENCIL_STATE(transparent)];
        for (const auto& assignment : _viewer->_materialAssignments)
        {
            mx::MeshPartitionPtr geom = assignment.first;
            mx::MslMaterialPtr material = std::static_pointer_cast<mx::MslMaterial>(assignment.second);
            shadowState.ambientOcclusionMap = _viewer->getAmbientOcclusionImage(material);
            if (!material || !material->hasTransparency())
            {
                continue;
            }

            material->bindShader();
            material->bindMesh(geometryHandler->findParentMesh(geom));
            if (material->getProgram()->hasUniform(mx::HW::ALPHA_THRESHOLD))
            {
                material->getProgram()->bindUniform(mx::HW::ALPHA_THRESHOLD, mx::Value::createValue(0.001f));
            }
            material->bindViewInformation(viewCamera);
            material->bindLighting(lightHandler, imageHandler, shadowState);
            material->bindImages(imageHandler, searchPath);
            material->prepareUsedResources(viewCamera,
                                 geometryHandler,
                                 imageHandler,
                                 lightHandler);
            material->drawPartition(geom);
            material->unbindImages(imageHandler);
        }
    }

    if (!_viewer->_renderDoubleSided)
    {
        [MTL(renderCmdEncoder) setCullMode:MTLCullModeNone];
    }

    // Wireframe pass
    if (_viewer->_outlineSelection)
    {
        mx::MslMaterialPtr wireMaterial =
            std::static_pointer_cast<mx::MslMaterial>(_viewer->getWireframeMaterial());
        if (wireMaterial)
        {
            [MTL(renderCmdEncoder) setCullMode:MTLCullModeNone];
            [MTL(renderCmdEncoder) setTriangleFillMode:MTLTriangleFillModeLines];
            wireMaterial->bindShader();
            wireMaterial->bindMesh(geometryHandler->findParentMesh(_viewer->getSelectedGeometry()));
            wireMaterial->bindViewInformation(viewCamera);
            wireMaterial->prepareUsedResources(viewCamera,
                                 geometryHandler,
                                 imageHandler,
                                 lightHandler);
            wireMaterial->drawPartition(_viewer->getSelectedGeometry());
            [MTL(renderCmdEncoder) setTriangleFillMode:MTLTriangleFillModeFill];
            [MTL(renderCmdEncoder) setCullMode:MTLCullModeNone];
        }
        else
        {
            _viewer->_outlineSelection = false;
        }
    }
    
#ifdef MAC_OS_VERSION_11_0
    if(useTiledPipeline)
    {
        if(@available(macOS 11.0, ios 14.0, *))
        {
            [MTL(renderCmdEncoder) setRenderPipelineState:MTL(linearToSRGB_pso)];
            [MTL(renderCmdEncoder) dispatchThreadsPerTile:MTLSizeMake(
                                        MTL(renderCmdEncoder).tileWidth,
                                        MTL(renderCmdEncoder).tileHeight, 1)];
        }
    }
    
    if(!useTiledPipeline)
#endif
    {
        MTL(endEncoder());
        [renderpassDesc.colorAttachments[0] setTexture:(id<MTLTexture>)color_texture];
        MTL(beginEncoder(renderpassDesc));
        [MTL(renderCmdEncoder) setRenderPipelineState:MTL(linearToSRGB_pso)];
        [MTL(renderCmdEncoder)
            setFragmentTexture:MTL(currentFramebuffer())->getColorTexture()
            atIndex:0];
        [MTL(renderCmdEncoder) drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    
    MTL(endCommandBuffer());
    
    if(captureFrame)
        MTL_STOP_CAPTURE;
    
    [renderpassDesc release];
}

void MetalRenderPipeline::bakeTextures()
{
    auto& imageHandler = _viewer->_imageHandler;
    
    mx::MaterialPtr material = _viewer->getSelectedMaterial();
    mx::DocumentPtr doc = material ? material->getDocument() : nullptr;
    if (!doc)
    {
        return;
    }

    {
        // Construct a texture baker.
        mx::Image::BaseType baseType = _viewer->_bakeHdr ? mx::Image::BaseType::FLOAT : mx::Image::BaseType::UINT8;
        mx::UnsignedIntPair bakingRes = _viewer->computeBakingResolution(doc);
        mx::TextureBakerPtr baker = std::static_pointer_cast<mx::TextureBakerPtr::element_type>(createTextureBaker(bakingRes.first, bakingRes.second, baseType));
        baker->setupUnitSystem(_viewer->_stdLib);
        baker->setDistanceUnit(_viewer->_genContext.getOptions().targetDistanceUnit);
        baker->setAverageImages(_viewer->_bakeAverage);
        baker->setOptimizeConstants(_viewer->_bakeOptimize);
        baker->writeDocumentPerMaterial(_viewer->_bakeDocumentPerMaterial);

        // Assign our existing image handler, releasing any existing render resources for cached images.
        imageHandler->releaseRenderResources();
        baker->setImageHandler(imageHandler);

        // Extend the image search path to include material source folders.
        mx::FileSearchPath extendedSearchPath = _viewer->_searchPath;
        extendedSearchPath.append(_viewer->_materialSearchPath);

        // Bake all materials in the active document.
        try
        {
            baker->bakeAllMaterials(doc, extendedSearchPath, _viewer->_bakeFilename);
        }
        catch (std::exception& e)
        {
            std::cerr << "Error in texture baking: " << e.what() << std::endl;
        }

        // Release any render resources generated by the baking process.
        imageHandler->releaseRenderResources();
    }
}

mx::ImagePtr MetalRenderPipeline::getFrameImage()
{
    unsigned int width = MTL(currentFramebuffer())->getWidth();
    unsigned int height = MTL(currentFramebuffer())->getHeight();
    
    MTL(waitForComplition());
    mx::MetalFramebufferPtr framebuffer = mx::MetalFramebuffer::create(
                            MTL(device),
                            width, height, 4,
                            mx::Image::BaseType::UINT8,
                            MTL(supportsTiledPipeline) ?
                                (id<MTLTexture>)_viewer->_colorTexture :
                                MTL(currentFramebuffer())->getColorTexture(),
                            false, MTLPixelFormatBGRA8Unorm);
    mx::ImagePtr frame = framebuffer->getColorImage(MTL(cmdQueue));
    
    // Flips the captured image
    std::vector<unsigned char> tmp(frame->getRowStride());
    unsigned int half_height = height / 2;
    unsigned char* resourceBuffer = static_cast<unsigned char*>(frame->getResourceBuffer());
    for(unsigned int i = 0; i < half_height; ++i)
    {
        memcpy(tmp.data(),
               &resourceBuffer[i*frame->getRowStride()], frame->getRowStride());
        memcpy(&resourceBuffer[i*frame->getRowStride()],
               &resourceBuffer[(height - i - 1) * frame->getRowStride()], frame->getRowStride());
        memcpy(&resourceBuffer[(height - i - 1) * frame->getRowStride()],
               tmp.data(), frame->getRowStride());
    }
    
    framebuffer = nullptr;
    return frame;
}

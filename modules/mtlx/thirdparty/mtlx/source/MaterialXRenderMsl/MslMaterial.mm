//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderMsl/MslMaterial.h>

#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXRenderMsl/MetalTextureHandler.h>
#include <MaterialXRenderMsl/MslPipelineStateObject.h>
#include <MaterialXRender/Util.h>
#include <MaterialXFormat/Util.h>

#include <MaterialXRenderMsl/MetalState.h>

MATERIALX_NAMESPACE_BEGIN

const std::string DISTANCE_UNIT_TARGET_NAME = "u_distanceUnitTarget";

//
// Material methods
//

bool MslMaterial::loadSource(const FilePath& vertexShaderFile, const FilePath& pixelShaderFile, bool hasTransparency)
{
    _hasTransparency = hasTransparency;

    std::string vertexShader = readFile(vertexShaderFile);
    if (vertexShader.empty())
    {
        return false;
    }

    std::string pixelShader = readFile(pixelShaderFile);
    if (pixelShader.empty())
    {
        return false;
    }

    // TODO:
    // Here we set new source code on the _glProgram without rebuilding 
    // the _hwShader instance. So the _hwShader is not in sync with the
    // _glProgram after this operation.
    _glProgram = MslProgram::create();
    _glProgram->addStage(Stage::VERTEX, vertexShader);
    _glProgram->addStage(Stage::PIXEL, pixelShader);
    _glProgram->build(MTL(device), MTL(currentFramebuffer()));

    return true;
}

void MslMaterial::clearShader()
{
    _hwShader = nullptr;
    _glProgram = nullptr;
}

bool MslMaterial::generateShader(GenContext& context)
{
    if (!_elem)
    {
        return false;
    }

    _hasTransparency = isTransparentSurface(_elem, context.getShaderGenerator().getTarget());

    GenContext materialContext = context;
    materialContext.getOptions().hwTransparency = _hasTransparency;

    // Initialize in case creation fails and throws an exception
    clearShader();

    _hwShader = createShader("Shader", materialContext, _elem);
    if (!_hwShader)
    {
        return false;
    }

    _glProgram = MslProgram::create();
    _glProgram->setStages(_hwShader);
    _glProgram->build(MTL(device), MTL(currentFramebuffer()));

    return true;
}

bool MslMaterial::generateShader(ShaderPtr hwShader)
{
    _hwShader = hwShader;

    _glProgram = MslProgram::create();
    _glProgram->setStages(hwShader);
    _glProgram->build(MTL(device), MTL(currentFramebuffer()));

    return true;
}

bool MslMaterial::bindShader() const
{
    if (_glProgram)
    {
        _glProgram->bind(MTL(renderCmdEncoder));
        return true;
    }
    return false;
}

void MslMaterial::prepareUsedResources(CameraPtr cam,
                          GeometryHandlerPtr geometryHandler,
                          ImageHandlerPtr imageHandler,
                          LightHandlerPtr lightHandler)
{
    if (!_glProgram)
    {
        return;
    }
    
    _glProgram->prepareUsedResources(MTL(renderCmdEncoder),
                           cam, geometryHandler,
                           imageHandler,
                           lightHandler);
}

void MslMaterial::bindMesh(MeshPtr mesh)
{
    if (!mesh || !_glProgram)
    {
        return;
    }

    _glProgram->bind(MTL(renderCmdEncoder));
    if (_boundMesh && mesh != _boundMesh)
    {
        _glProgram->unbindGeometry();
    }
    _glProgram->bindMesh(MTL(renderCmdEncoder), mesh);
    _boundMesh = mesh;
}

bool MslMaterial::bindPartition(MeshPartitionPtr part) const
{
    if (!_glProgram)
    {
        return false;
    }

    _glProgram->bind(MTL(renderCmdEncoder));
    _glProgram->bindPartition(part);

    return true;
}

void MslMaterial::bindViewInformation(CameraPtr camera)
{
    if (!_glProgram)
    {
        return;
    }

    _glProgram->bindViewInformation(camera);
}

void MslMaterial::unbindImages(ImageHandlerPtr imageHandler)
{
    for (ImagePtr image : _boundImages)
    {
        imageHandler->unbindImage(image);
    }
}

void MslMaterial::bindImages(ImageHandlerPtr imageHandler, const FileSearchPath& searchPath, bool enableMipmaps)
{
    if (!_glProgram)
    {
        return;
    }

    _boundImages.clear();
    _glProgram->setEnableMipMaps(enableMipmaps);
    
    // Texture and Samplers being bound to the right texture and sampler in MslPipelineStateObject automatically.
}

ImagePtr MslMaterial::bindImage(const FilePath& filePath,
                                 const std::string& uniformName,
                                 ImageHandlerPtr imageHandler,
                                 const ImageSamplingProperties& samplingProperties)
{
    if (!_glProgram)
    {
        return nullptr;
    }

    // Create a filename resolver for geometric properties.
    StringResolverPtr resolver = StringResolver::create();
    if (!getUdim().empty())
    {
        resolver->setUdimString(getUdim());
    }
    imageHandler->setFilenameResolver(resolver);

    // Acquire the given image.
    return imageHandler->acquireImage(filePath, samplingProperties.defaultColor);
}

void MslMaterial::bindLighting(LightHandlerPtr lightHandler,
                            ImageHandlerPtr imageHandler,
                            const ShadowState& shadowState)
{
    if (!_glProgram)
    {
        return;
    }

    // Bind environment and local lighting.
    _glProgram->bindLighting(lightHandler, imageHandler);

    // Bind shadow map properties
    if (shadowState.shadowMap && _glProgram->hasUniform(TEXTURE_NAME(HW::SHADOW_MAP)))
    {
        ImageSamplingProperties samplingProperties;
        samplingProperties.uaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
        samplingProperties.vaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
        samplingProperties.filterType = ImageSamplingProperties::FilterType::LINEAR;

        // Bind the shadow map.
        _glProgram->bindTexture(imageHandler, TEXTURE_NAME(HW::SHADOW_MAP), shadowState.shadowMap, samplingProperties);
        _glProgram->bindUniform(HW::SHADOW_MATRIX, Value::createValue(shadowState.shadowMatrix));
    }

    // Bind ambient occlusion properties.
    if (shadowState.ambientOcclusionMap && _glProgram->hasUniform(TEXTURE_NAME(HW::AMB_OCC_MAP)))
    {
        ImageSamplingProperties samplingProperties;
        samplingProperties.uaddressMode = ImageSamplingProperties::AddressMode::PERIODIC;
        samplingProperties.vaddressMode = ImageSamplingProperties::AddressMode::PERIODIC;
        samplingProperties.filterType = ImageSamplingProperties::FilterType::LINEAR;

        // Bind the ambient occlusion map.
        _glProgram->bindTexture(imageHandler, TEXTURE_NAME(HW::AMB_OCC_MAP),
                                shadowState.ambientOcclusionMap,
                                samplingProperties);
        
        _glProgram->bindUniform(HW::AMB_OCC_GAIN, Value::createValue(shadowState.ambientOcclusionGain));
    }
}

void MslMaterial::drawPartition(MeshPartitionPtr part) const
{
    if (!part || !bindPartition(part))
    {
        return;
    }
    MeshIndexBuffer& indexData = part->getIndices();

    [MTL(renderCmdEncoder) drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                 indexCount:indexData.size()
                                  indexType:MTLIndexTypeUInt32
                                indexBuffer:_glProgram->getIndexBuffer(part)
                          indexBufferOffset:0];
}

void MslMaterial::unbindGeometry()
{
    if (_glProgram)
    {
        _glProgram->unbindGeometry();
    }
    _boundMesh = nullptr;
}

VariableBlock* MslMaterial::getPublicUniforms() const
{
    if (!_hwShader)
    {
        return nullptr;
    }

    ShaderStage& stage = _hwShader->getStage(Stage::PIXEL);
    VariableBlock& block = stage.getUniformBlock(HW::PUBLIC_UNIFORMS);

    return &block;
}

ShaderPort* MslMaterial::findUniform(const std::string& path) const
{
    ShaderPort* port = nullptr;
    VariableBlock* publicUniforms = getPublicUniforms();
    if (publicUniforms)
    {
        // Scan block based on path match predicate
        port = publicUniforms->find(
            [path](ShaderPort* port)
            {
                return (port && stringEndsWith(port->getPath(), path));
            });

        // Check if the uniform exists in the shader program
        if (port && !_glProgram->getUniformsList().count(
                publicUniforms->getInstance() + "." +
                port->getVariable()))
        {
            port = nullptr;
        }
    }
    return port;
}

void MslMaterial::modifyUniform(const std::string& path, ConstValuePtr value, std::string valueString)
{
    ShaderPort* uniform = findUniform(path);
    if (!uniform)
    {
        return;
    }

    _glProgram->bindUniform(uniform->getVariable(), value);

    if (valueString.empty())
    {
        valueString = value->getValueString();
    }
    uniform->setValue(Value::createValueFromStrings(valueString, uniform->getType()->getName()));
    if (_doc)
    {
        ElementPtr element = _doc->getDescendant(uniform->getPath());
        if (element)
        {
            ValueElementPtr valueElement = element->asA<ValueElement>();
            if (valueElement)
            {
                valueElement->setValueString(valueString);
            }
        }
    }
}

MATERIALX_NAMESPACE_END


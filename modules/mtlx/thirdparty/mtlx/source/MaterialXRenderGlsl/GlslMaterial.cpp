//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderGlsl/GlslMaterial.h>

#include <MaterialXRenderGlsl/External/Glad/glad.h>
#include <MaterialXRenderGlsl/GLTextureHandler.h>
#include <MaterialXRenderGlsl/GLUtil.h>

#include <MaterialXRender/Util.h>

#include <MaterialXFormat/Util.h>

MATERIALX_NAMESPACE_BEGIN

const std::string DISTANCE_UNIT_TARGET_NAME = "u_distanceUnitTarget";

//
// GlslMaterial methods
//

bool GlslMaterial::loadSource(const FilePath& vertexShaderFile, const FilePath& pixelShaderFile, bool hasTransparency)
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
    _glProgram = GlslProgram::create();
    _glProgram->addStage(Stage::VERTEX, vertexShader);
    _glProgram->addStage(Stage::PIXEL, pixelShader);

    return true;
}

void GlslMaterial::clearShader()
{
    _hwShader = nullptr;
    _glProgram = nullptr;
}

bool GlslMaterial::generateShader(GenContext& context)
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

    _glProgram = GlslProgram::create();
    _glProgram->setStages(_hwShader);

    return true;
}

bool GlslMaterial::generateShader(ShaderPtr hwShader)
{
    _hwShader = hwShader;

    _glProgram = GlslProgram::create();
    _glProgram->setStages(hwShader);

    return true;
}

bool GlslMaterial::bindShader() const
{
    if (!_glProgram)
    {
        return false;
    }

    if (!_glProgram->hasBuiltData())
    {
        _glProgram->build();
    }
    return _glProgram->bind();
}

void GlslMaterial::bindMesh(MeshPtr mesh)
{
    if (!mesh || !bindShader())
    {
        return;
    }

    if (mesh != _boundMesh)
    {
        _glProgram->unbindGeometry();
    }
    _glProgram->bindMesh(mesh);
    _boundMesh = mesh;
}

bool GlslMaterial::bindPartition(MeshPartitionPtr part) const
{
    if (!bindShader())
    {
        return false;
    }

    _glProgram->bindPartition(part);

    return true;
}

void GlslMaterial::bindViewInformation(CameraPtr camera)
{
    if (!_glProgram)
    {
        return;
    }

    _glProgram->bindViewInformation(camera);
}

void GlslMaterial::unbindImages(ImageHandlerPtr imageHandler)
{
    for (ImagePtr image : _boundImages)
    {
        imageHandler->unbindImage(image);
    }
}

void GlslMaterial::bindImages(ImageHandlerPtr imageHandler, const FileSearchPath& searchPath, bool enableMipmaps)
{
    if (!_glProgram)
    {
        return;
    }

    _boundImages.clear();

    const VariableBlock* publicUniforms = getPublicUniforms();
    if (!publicUniforms)
    {
        return;
    }
    for (const auto& uniform : publicUniforms->getVariableOrder())
    {
        if (uniform->getType() != Type::FILENAME)
        {
            continue;
        }
        const std::string& uniformVariable = uniform->getVariable();
        std::string filename;
        if (uniform->getValue())
        {
            filename = searchPath.find(uniform->getValue()->getValueString());
        }

        // Extract out sampling properties
        ImageSamplingProperties samplingProperties;
        samplingProperties.setProperties(uniformVariable, *publicUniforms);

        // Set the requested mipmap sampling property,
        samplingProperties.enableMipmaps = enableMipmaps;

        ImagePtr image = bindImage(filename, uniformVariable, imageHandler, samplingProperties);
        if (image)
        {
            _boundImages.push_back(image);
        }
    }
}

ImagePtr GlslMaterial::bindImage(const FilePath& filePath, const std::string& uniformName, ImageHandlerPtr imageHandler,
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
    ImagePtr image = imageHandler->acquireImage(filePath, samplingProperties.defaultColor);
    if (!image)
    {
        return nullptr;
    }

    // Bind the image and set its sampling properties.
    if (imageHandler->bindImage(image, samplingProperties))
    {
        GLTextureHandlerPtr textureHandler = std::static_pointer_cast<GLTextureHandler>(imageHandler);
        int textureLocation = textureHandler->getBoundTextureLocation(image->getResourceId());
        if (textureLocation >= 0)
        {
            _glProgram->bindUniform(uniformName, Value::createValue(textureLocation), false);
            return image;
        }
    }
    return nullptr;
}

void GlslMaterial::bindLighting(LightHandlerPtr lightHandler, ImageHandlerPtr imageHandler, const ShadowState& shadowState)
{
    if (!_glProgram)
    {
        return;
    }

    // Bind environment and local lighting.
    _glProgram->bindLighting(lightHandler, imageHandler);

    // Bind shadow map properties
    if (shadowState.shadowMap && _glProgram->hasUniform(HW::SHADOW_MAP))
    {
        ImageSamplingProperties samplingProperties;
        samplingProperties.uaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
        samplingProperties.vaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
        samplingProperties.filterType = ImageSamplingProperties::FilterType::LINEAR;

        // Bind the shadow map.
        if (imageHandler->bindImage(shadowState.shadowMap, samplingProperties))
        {
            GLTextureHandlerPtr textureHandler = std::static_pointer_cast<GLTextureHandler>(imageHandler);
            int textureLocation = textureHandler->getBoundTextureLocation(shadowState.shadowMap->getResourceId());
            if (textureLocation >= 0)
            {
                _glProgram->bindUniform(HW::SHADOW_MAP, Value::createValue(textureLocation));
            }
        }
        _glProgram->bindUniform(HW::SHADOW_MATRIX, Value::createValue(shadowState.shadowMatrix));
    }

    // Bind ambient occlusion properties.
    if (shadowState.ambientOcclusionMap && _glProgram->hasUniform(HW::AMB_OCC_MAP))
    {
        ImageSamplingProperties samplingProperties;
        samplingProperties.uaddressMode = ImageSamplingProperties::AddressMode::PERIODIC;
        samplingProperties.vaddressMode = ImageSamplingProperties::AddressMode::PERIODIC;
        samplingProperties.filterType = ImageSamplingProperties::FilterType::LINEAR;

        // Bind the ambient occlusion map.
        if (imageHandler->bindImage(shadowState.ambientOcclusionMap, samplingProperties))
        {
            GLTextureHandlerPtr textureHandler = std::static_pointer_cast<GLTextureHandler>(imageHandler);
            int textureLocation = textureHandler->getBoundTextureLocation(shadowState.ambientOcclusionMap->getResourceId());
            if (textureLocation >= 0)
            {
                _glProgram->bindUniform(HW::AMB_OCC_MAP, Value::createValue(textureLocation));
            }
        }
        _glProgram->bindUniform(HW::AMB_OCC_GAIN, Value::createValue(shadowState.ambientOcclusionGain));
    }
}

void GlslMaterial::drawPartition(MeshPartitionPtr part) const
{
    if (!part || !bindPartition(part))
    {
        return;
    }
    MeshIndexBuffer& indexData = part->getIndices();
    glDrawElements(GL_TRIANGLES, (GLsizei) indexData.size(), GL_UNSIGNED_INT, (void*) 0);
    checkGlErrors("after draw partition");
}

void GlslMaterial::unbindGeometry()
{
    if (!_boundMesh)
    {
        return;
    }

    if (bindShader())
    {
        _glProgram->unbindGeometry();
    }
    _boundMesh = nullptr;
}

VariableBlock* GlslMaterial::getPublicUniforms() const
{
    if (!_hwShader)
    {
        return nullptr;
    }

    ShaderStage& stage = _hwShader->getStage(Stage::PIXEL);
    VariableBlock& block = stage.getUniformBlock(HW::PUBLIC_UNIFORMS);

    return &block;
}

ShaderPort* GlslMaterial::findUniform(const std::string& path) const
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
        if (port && !_glProgram->getUniformsList().count(port->getVariable()))
        {
            port = nullptr;
        }
    }
    return port;
}

void GlslMaterial::modifyUniform(const std::string& path, ConstValuePtr value, std::string valueString)
{
    if (!bindShader())
    {
        return;
    }

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

//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderMsl/MslPipelineStateObject.h>
#include <MaterialXRenderMsl/MetalTextureHandler.h>
#include <MaterialXRenderMsl/MetalFramebuffer.h>

#include <MaterialXRender/LightHandler.h>
#include <MaterialXRender/ShaderRenderer.h>

#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenShader/Util.h>

#include <iostream>

#import <Metal/Metal.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const float PI = std::acos(-1.0f);

} // anonymous namespace

// Metal Constants
unsigned int MslProgram::UNDEFINED_METAL_RESOURCE_ID = 0;
int MslProgram::UNDEFINED_METAL_PROGRAM_LOCATION = -1;
int MslProgram::Input::INVALID_METAL_TYPE = -1;

//
// MslProgram methods
//

MslProgram::MslProgram() :
    _pso(nil),
    _psoReflection(nil),
    _shader(nullptr),
    _alphaBlendingEnabled(false)
{
}

MslProgram::~MslProgram()
{
    reset();
}

void MslProgram::setStages(ShaderPtr shader)
{
    if (!shader)
    {
        throw ExceptionRenderError("Cannot set stages using null hardware shader");
    }

    // Clear out any old data
    clearStages();

    // Extract out the shader code per stage
    _shader = shader;
    for (size_t i =0; i<shader->numStages(); ++i)
    {
        const ShaderStage& stage = shader->getStage(i);
        addStage(stage.getName(), stage.getSourceCode());
    }

    // A stage change invalidates any cached parsed inputs
    clearInputLists();
}

void MslProgram::addStage(const string& stage, const string& sourceCode)
{
    _stages[stage] = sourceCode;
}

const string& MslProgram::getStageSourceCode(const string& stage) const
{
    auto it = _stages.find(stage);
    if (it != _stages.end())
    {
        return it->second;
    }
    return EMPTY_STRING;
}

void MslProgram::clearStages()
{
    _stages.clear();

    // Clearing stages invalidates any cached inputs
    clearInputLists();
}

MTLVertexFormat GetMetalFormatFromMetalType(MTLDataType type)
{
    switch(type)
    {
        case MTLDataTypeFloat : return MTLVertexFormatFloat;
        case MTLDataTypeFloat2: return MTLVertexFormatFloat2;
        case MTLDataTypeFloat3: return MTLVertexFormatFloat3;
        case MTLDataTypeFloat4: return MTLVertexFormatFloat4;
        case MTLDataTypeInt:    return MTLVertexFormatInt;
        case MTLDataTypeInt2:   return MTLVertexFormatInt2;
        case MTLDataTypeInt3:   return MTLVertexFormatInt3;
        case MTLDataTypeInt4:   return MTLVertexFormatInt4;
        default:                return MTLVertexFormatInvalid;
    }
    return MTLVertexFormatInt;
}

int GetStrideOfMetalType(MTLDataType type)
{
    switch(type)
    {
        case MTLDataTypeInt:
        case MTLDataTypeFloat : return 1 * 4;
        case MTLDataTypeInt2:
        case MTLDataTypeFloat2: return 2 * 4;
        case MTLDataTypeInt3:
        case MTLDataTypeFloat3: return 3 * 4;
        case MTLDataTypeInt4:
        case MTLDataTypeFloat4: return 4 * 4;
        default:                return 0;
    }
    
    return 0;
}

id<MTLRenderPipelineState> MslProgram::build(id<MTLDevice> device, MetalFramebufferPtr framebuffer)
{
    StringVec errors;
    const string errorType("MSL program creation error.");

    NSError* error = nil;
    
    reset();
    
    _device = device;

    unsigned int stagesBuilt = 0;
    unsigned int desiredStages = 0;
    for (const auto& it : _stages)
    {
        if (it.second.length())
            desiredStages++;
    }
    
    MTLCompileOptions* options = [MTLCompileOptions new];
#ifdef MAC_OS_VERSION_11_0
    if (@available(macOS 11.0, ios 14.0, *))
        options.languageVersion = MTLLanguageVersion2_3;
    else
#endif
        options.languageVersion = MTLLanguageVersion2_0;
    options.fastMathEnabled = true;

    // Create vertex shader
    id<MTLFunction> vertexShaderId = nil;
    {
        string &vertexShaderSource = _stages[Stage::VERTEX];
        if (vertexShaderSource.length() < 1)
        {
            errors.push_back("Vertex Shader is empty.");
            return nil;
        }
        
        NSString* sourcCode = [NSString stringWithUTF8String:vertexShaderSource.c_str()];
        id<MTLLibrary> library = [device newLibraryWithSource:sourcCode
                                                      options:options
                                                        error:&error];
        
        if(library == nil)
        {
            errors.push_back("Error in compiling vertex shader:");
            if(error)
            {
                errors.push_back([[error localizedDescription] UTF8String]);
            }
            return nil;
        }
        
        vertexShaderId = [library newFunctionWithName:@"VertexMain"];
        
        if(vertexShaderId)
        {
            stagesBuilt++;
        }
    }

    // Create fragment shader
    string& fragmentShaderSource = _stages[Stage::PIXEL];
    if (fragmentShaderSource.length() < 1)
    {
        errors.push_back("Fragment Shader is empty.");
        return nil;
    }
    
    // Fragment shader compilation code
    id<MTLFunction> fragmentShaderId = nil;
    {
        NSString* sourcCode = [NSString stringWithUTF8String:fragmentShaderSource.c_str()];       
        id<MTLLibrary> library = [device newLibraryWithSource:sourcCode options:options error:&error];
        if(!library)
        {
            errors.push_back("Error in compiling fragment shader:");
            if(error)
            {
                std::string libCompilationError = [[error localizedDescription] UTF8String];
                printf("Compilation Errors:%s\n", libCompilationError.c_str());
                errors.push_back(libCompilationError);
                return nil;
            }
        }
        
        fragmentShaderId = [library newFunctionWithName:@"FragmentMain"];
        assert(fragmentShaderId);
        
        if(library)
        {
            stagesBuilt++;
        }
    }
    

    // Link stages to a programs
    if (stagesBuilt == desiredStages)
    {
        MTLRenderPipelineDescriptor* psoDesc = [MTLRenderPipelineDescriptor new];
        psoDesc.vertexFunction = vertexShaderId;
        psoDesc.fragmentFunction = fragmentShaderId;
        psoDesc.colorAttachments[0].pixelFormat =  framebuffer->getColorTexture().pixelFormat;
        psoDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        
        if (_shader->hasAttribute(HW::ATTR_TRANSPARENT))
        {
            psoDesc.colorAttachments[0].blendingEnabled = YES;
            psoDesc.colorAttachments[0].rgbBlendOperation           = MTLBlendOperationAdd;
            psoDesc.colorAttachments[0].alphaBlendOperation         = MTLBlendOperationAdd;
            psoDesc.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorSourceAlpha;
            psoDesc.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorSourceAlpha;
            psoDesc.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOneMinusSourceAlpha;
            psoDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            
            _alphaBlendingEnabled = true;
        }
        
        MTLVertexDescriptor *vd = [MTLVertexDescriptor new];
        
        for( int i = 0; i < vertexShaderId.vertexAttributes.count; ++i)
        {
            MTLVertexAttribute* vertexAttrib = vertexShaderId.vertexAttributes[i];
            
            vd.attributes[i].bufferIndex = i;
            vd.attributes[i].format = GetMetalFormatFromMetalType(vertexAttrib.attributeType);
            vd.attributes[i].offset = 0;

            InputPtr inputPtr = std::make_shared<Input>(vertexAttrib.attributeIndex, vertexAttrib.attributeType, GetStrideOfMetalType(vertexAttrib.attributeType), EMPTY_STRING);
            // Attempt to pull out the set number for specific attributes
            //
            string sattributeName(vertexAttrib.name.UTF8String);
            const string colorSet(HW::IN_COLOR + "_");
            const string uvSet(HW::IN_TEXCOORD + "_");
            if (string::npos != sattributeName.find(colorSet))
            {
                string setNumber = sattributeName.substr(colorSet.size(), sattributeName.size());
                inputPtr->value = Value::createValueFromStrings(setNumber, getTypeString<int>());
            }
            else if (string::npos != sattributeName.find(uvSet))
            {
                string setNumber = sattributeName.substr(uvSet.size(), sattributeName.size());
                inputPtr->value = Value::createValueFromStrings(setNumber, getTypeString<int>());
            }

            _attributeList[sattributeName] = inputPtr;
            
            vd.layouts[i].stride = GetStrideOfMetalType(vertexAttrib.attributeType);;
            vd.layouts[i].stepFunction = MTLVertexStepFunctionPerVertex;
            
        }
        
        psoDesc.vertexDescriptor = vd;
                
        _pso = [device newRenderPipelineStateWithDescriptor:psoDesc
                options:MTLPipelineOptionArgumentInfo|MTLPipelineOptionBufferTypeInfo
                reflection:&_psoReflection error:&error];
        
        [_pso retain];
        [_psoReflection retain];
        
        
        if(error)
        {
            errors.push_back("Error in linking program:");
            errors.push_back([[error localizedDescription] UTF8String]);
        }
    }
    else
    {
        errors.push_back("Failed to build all stages.");
        throw ExceptionRenderError(errorType, errors);
    }

    // If we encountered any errors while trying to create return list
    // of all errors. That is we collect all errors per stage plus any
    // errors during linking and throw one exception for them all so that
    // if there is a failure a complete set of issues is returned. We do
    // this after cleanup so keep //gl state clean.
    if (!errors.empty())
    {
        throw ExceptionRenderError(errorType, errors);
    }

    return _pso;
}

bool MslProgram::bind(id<MTLRenderCommandEncoder> renderCmdEncoder)
{
    if (_pso != nil)
    {
        [renderCmdEncoder setRenderPipelineState:_pso];
        return true;
    }
    return false;
}

void MslProgram::prepareUsedResources(id<MTLRenderCommandEncoder> renderCmdEncoder,
                                 CameraPtr cam,
                                 GeometryHandlerPtr geometryHandler,
                                 ImageHandlerPtr imageHandler,
                                 LightHandlerPtr lightHandler)
{
    // Bind the program to use
    if (!bind(renderCmdEncoder))
    {
        const string errorType("MSL bind inputs error.");
        StringVec errors;
        errors.push_back("Cannot bind inputs without a valid program");
        throw ExceptionRenderError(errorType, errors);
    }

    // Parse for uniforms and attributes
    getUniformsList();
    getAttributesList();
    
    // Bind based on inputs found
    bindViewInformation(cam);
    bindTimeAndFrame();
    bindLighting(lightHandler, imageHandler);
    bindTextures(renderCmdEncoder, lightHandler, imageHandler);
    bindUniformBuffers(renderCmdEncoder, lightHandler, cam);
}

void MslProgram::bindAttribute(id<MTLRenderCommandEncoder> renderCmdEncoder, const MslProgram::InputMap& inputs, MeshPtr mesh)
{
    const string errorType("MSL bind attribute error.");
    StringVec errors;

    if (!mesh)
    {
        errors.push_back("No geometry set to bind");
        throw ExceptionRenderError(errorType, errors);
    }

    const size_t FLOAT_SIZE = sizeof(float);

    for (const auto& input : inputs)
    {
        int location = input.second->location;
        unsigned int index = input.second->value ? input.second->value->asA<int>() : 0;

        unsigned int stride = 0;
        MeshStreamPtr stream = mesh->getStream(input.first);
        if (!stream)
        {
            errors.push_back("Geometry buffer could not be retrieved for binding: " + input.first + ". Index: " + std::to_string(index));
            throw ExceptionRenderError(errorType, errors);
        }
        MeshFloatBuffer& attributeData = stream->getData();
        stride = stream->getStride();

        if (attributeData.empty() || (stride == 0))
        {
            errors.push_back("Geometry buffer could not be retrieved for binding: " + input.first + ". Index: " + std::to_string(index));
            throw ExceptionRenderError(errorType, errors);
        }

        if (_attributeBufferIds.find(input.first) == _attributeBufferIds.end())
        {
            std::vector<unsigned char> restructuredData;
            const void* bufferData = nullptr;
            size_t bufferSize = 0;
            
            int shaderStride = input.second->size / FLOAT_SIZE;
            
            if(shaderStride == stride)
            {
                bufferData = &attributeData[0];
                bufferSize = attributeData.size()*FLOAT_SIZE;
            }
            else
            {
                size_t nElements = attributeData.size()/stride;
                bufferSize = nElements*shaderStride*FLOAT_SIZE;
                restructuredData.resize(bufferSize, 0.0f);
                size_t j = 0;
                for(int i = 0; i < nElements; ++i)
                {
                    memcpy(&restructuredData[j],
                           &attributeData[i*stride],
                           stride*FLOAT_SIZE);
                    j += shaderStride * FLOAT_SIZE;
                }
                bufferData = &restructuredData[0];
            }

            // Create a buffer based on attribute type.
            id<MTLBuffer> buffer = [_device newBufferWithBytes:bufferData length:bufferSize options:MTLResourceStorageModeShared];
            _attributeBufferIds[input.first] = buffer;
        }
        
        [renderCmdEncoder setVertexBuffer:_attributeBufferIds[input.first] offset:0 atIndex:location];
    }
}

void MslProgram::bindPartition(MeshPartitionPtr part)
{
    StringVec errors;
    const string errorType("MSL geometry bind error.");
    if (!part || part->getFaceCount() == 0)
    {
        errors.push_back("Cannot bind geometry partition");
        throw ExceptionRenderError(errorType, errors);
    }

    if (_indexBufferIds.find(part) == _indexBufferIds.end())
    {
        MeshIndexBuffer& indexData = part->getIndices();
        size_t indexBufferSize = indexData.size();
        id<MTLBuffer> indexBuffer = [_device newBufferWithBytes:&indexData[0] length:indexBufferSize * sizeof(uint32_t) options:MTLStorageModeShared];
        _indexBufferIds[part] = indexBuffer;
    }
}

void MslProgram::bindMesh(id<MTLRenderCommandEncoder> renderCmdEncoder, MeshPtr mesh)
{
    StringVec errors;
    const string errorType("MSL geometry bind error.");

    if (_pso == nil)
    {
        errors.push_back("Cannot bind geometry without a valid program");
        throw ExceptionRenderError(errorType, errors);
    }
    if (!mesh)
    {
        errors.push_back("No mesh to bind");
        throw ExceptionRenderError(errorType, errors);
    }

    if (_boundMesh && mesh != _boundMesh)
    {
        unbindGeometry();
    }
    _boundMesh = mesh;

    MslProgram::InputMap foundList;
    const MslProgram::InputMap& attributeList = getAttributesList();

    // Bind positions
    findInputs(HW::IN_POSITION, attributeList, foundList, true);
    if (foundList.size())
    {
        bindAttribute(renderCmdEncoder, foundList, mesh);
    }

    // Bind normals
    findInputs(HW::IN_NORMAL, attributeList, foundList, true);
    if (foundList.size())
    {
        bindAttribute(renderCmdEncoder, foundList, mesh);
    }

    // Bind tangents
    findInputs(HW::IN_TANGENT, attributeList, foundList, true);
    if (foundList.size())
    {
        bindAttribute(renderCmdEncoder, foundList, mesh);
    }

    // Bind colors
    // Search for anything that starts with the color prefix
    findInputs(HW::IN_COLOR + "_", attributeList, foundList, false);
    if (foundList.size())
    {
        bindAttribute(renderCmdEncoder, foundList, mesh);
    }

    // Bind texture coordinates
    // Search for anything that starts with the texcoord prefix
    findInputs(HW::IN_TEXCOORD + "_", attributeList, foundList, false);
    if (foundList.size())
    {
        bindAttribute(renderCmdEncoder, foundList, mesh);
    }

    // Bind any named varying geometric property information
    findInputs(HW::IN_GEOMPROP + "_", attributeList, foundList, false);
    if (foundList.size())
    {
        bindAttribute(renderCmdEncoder, foundList, mesh);
    }

    // Bind any named uniform geometric property information
    const MslProgram::InputMap& uniformList = getUniformsList();
    findInputs(HW::GEOMPROP + "_", uniformList, foundList, false);
}

void MslProgram::unbindGeometry()
{
    // Clean up buffers
    //
    for (const auto& attributeBufferId : _attributeBufferIds)
    {
        [attributeBufferId.second release];
    }
    _attributeBufferIds.clear();
    for (const auto& indexBufferId : _indexBufferIds)
    {
        [indexBufferId.second release];
    }
    _indexBufferIds.clear();
}

ImagePtr MslProgram::bindTexture(id<MTLRenderCommandEncoder> renderCmdEncoder,
                                 unsigned int uniformLocation,
                                 const FilePath& filePath,
                                 ImageSamplingProperties samplingProperties,
                                 ImageHandlerPtr imageHandler)
{
    // Acquire the image.
    string error;
    ImagePtr image = imageHandler->acquireImage(filePath, samplingProperties.defaultColor);
    imageHandler->bindImage(image, samplingProperties);
    return bindTexture(renderCmdEncoder, uniformLocation, image, imageHandler);
}

ImagePtr MslProgram::bindTexture(id<MTLRenderCommandEncoder> renderCmdEncoder,
                                 unsigned int uniformLocation,
                                 ImagePtr image,
                                 ImageHandlerPtr imageHandler)
{
    // Acquire the image.
    string error;
    if (static_cast<MetalTextureHandler*>(imageHandler.get())->bindImage(
            renderCmdEncoder, uniformLocation, image))
    {
        return image;
    }
    return nullptr;
}

MaterialX::ValuePtr MslProgram::findUniformValue(
                                    const string& uniformName,
                                    const MslProgram::InputMap& uniformList)
{
    auto uniform = uniformList.find(uniformName);
    if (uniform != uniformList.end())
    {
        int location = uniform->second->location;
        if (location >= 0)
        {
            return uniform->second->value;
        }
    }
    return nullptr;
}

void MslProgram::bindTextures(id<MTLRenderCommandEncoder> renderCmdEncoder,
                              LightHandlerPtr lightHandler,
                              ImageHandlerPtr imageHandler)
{
    const VariableBlock& publicUniforms = _shader->getStage(Stage::PIXEL).getUniformBlock(HW::PUBLIC_UNIFORMS);
    for (MTLArgument *arg in _psoReflection.fragmentArguments)
    {
        if (arg.type == MTLArgumentTypeTexture)
        {
            bool found = false;
            
            if(lightHandler)
            {
                // Bind environment lights.
                ImageMap envLights =
                {
                    { HW::ENV_RADIANCE,   lightHandler->getUsePrefilteredMap() ? lightHandler->getEnvPrefilteredMap() : lightHandler->getEnvRadianceMap() },
                    { HW::ENV_IRRADIANCE, lightHandler->getEnvIrradianceMap() }
                };
                for (const auto& env : envLights)
                {
                    std::string str(arg.name.UTF8String);
                    size_t loc = str.find(env.first);
                    if(loc != std::string::npos && env.second)
                    {
                        ImageSamplingProperties samplingProperties;
                        samplingProperties.uaddressMode = ImageSamplingProperties::AddressMode::PERIODIC;
                        samplingProperties.vaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
                        samplingProperties.filterType = ImageSamplingProperties::FilterType::LINEAR;
                        
                        static_cast<MaterialX::MetalTextureHandler*>
                        (imageHandler.get())->bindImage(env.second, samplingProperties);
                        bindTexture(renderCmdEncoder, (unsigned int)arg.index, env.second, imageHandler);
                        found = true;
                    }
                    
                }
            }
            
            if(!found)
            {
                ImagePtr image = nullptr;
                if(_explicitBoundImages.find(arg.name.UTF8String) != _explicitBoundImages.end())
                {
                    image = _explicitBoundImages[arg.name.UTF8String];
                }

                if(image && (image->getWidth() > 1 || image->getHeight() > 1))
                {
                    bindTexture(renderCmdEncoder, (unsigned int)arg.index, image, imageHandler);
                    found = true;
                }
            }
            
            if(!found)
            {
                auto uniform = _uniformList.find(arg.name.UTF8String);
                if(uniform != _uniformList.end())
                {
                    string fileName = uniform->second->value ? uniform->second->value->getValueString() : "";
                    ImageSamplingProperties samplingProperties;
                    string uniformNameWithoutPostfix = uniform->first;
                    {
                        size_t pos = uniformNameWithoutPostfix.find_last_of(IMAGE_PROPERTY_SEPARATOR);
                        if(pos != std::string::npos)
                            uniformNameWithoutPostfix = uniformNameWithoutPostfix.substr(0, pos);
                    }
                    samplingProperties.setProperties(uniformNameWithoutPostfix, publicUniforms);
                    samplingProperties.enableMipmaps = _enableMipMapping;
                    bindTexture(renderCmdEncoder, (unsigned int)arg.index, fileName, samplingProperties, imageHandler);
                }
            }
        }
    }
}

void MslProgram::bindTexture(ImageHandlerPtr imageHandler,
                             string shaderTextureName,
                             ImagePtr imagePtr,
                             ImageSamplingProperties samplingProperties)
{
    if(imageHandler->bindImage(imagePtr, samplingProperties))
    {
        _explicitBoundImages[shaderTextureName] = imagePtr;
    }
}

void MslProgram::bindLighting(LightHandlerPtr lightHandler, ImageHandlerPtr imageHandler)
{
    if (!lightHandler)
    {
        // Nothing to bind if a light handler is not used.
        // This is a valid condition for shaders that don't
        // need lighting so just ignore silently.
        return;
    }

    StringVec errors;

    if (_pso == nil)
    {
        const string errorType("MSL light binding error.");
        errors.push_back("Cannot bind without a valid program");
        throw ExceptionRenderError(errorType, errors);
    }

    const MslProgram::InputMap& uniformList = getUniformsList();

    // Set the number of active light sources
    size_t lightCount = lightHandler->getLightSources().size();
    auto input = uniformList.find(HW::NUM_ACTIVE_LIGHT_SOURCES);
    if (input == uniformList.end())
    {
        // No lighting information so nothing further to do
        lightCount = 0;
    }

    if (lightCount == 0 &&
        !lightHandler->getEnvRadianceMap() &&
        !lightHandler->getEnvIrradianceMap())
    {
        return;
    }

    // Bind environment lights.
    Matrix44 envRotation = Matrix44::createRotationY(PI) * lightHandler->getLightTransform().getTranspose();
    bindUniform(HW::ENV_MATRIX, Value::createValue(envRotation), false);
    bindUniform(HW::ENV_RADIANCE_SAMPLES, Value::createValue(lightHandler->getEnvSampleCount()), false);
    bindUniform(HW::ENV_LIGHT_INTENSITY, Value::createValue(lightHandler->getEnvLightIntensity()), false);
    ImageMap envLights =
    {
        { HW::ENV_RADIANCE, lightHandler->getEnvRadianceMap() },
        { HW::ENV_IRRADIANCE, lightHandler->getEnvIrradianceMap() }
    };
    for (const auto& env : envLights)
    {
        auto iblUniform = uniformList.find(TEXTURE_NAME(env.first));
        MslProgram::InputPtr inputPtr = iblUniform != uniformList.end() ? iblUniform->second : nullptr;
        if (inputPtr)
        {
            ImagePtr image;
            if (inputPtr->value)
            {
                string filename = inputPtr->value->getValueString();
                if (!filename.empty())
                {
                    image = imageHandler->acquireImage(filename);
                }
            }
            if (!image)
            {
                image = env.second;
            }

            if (image)
            {
                ImageSamplingProperties samplingProperties;
                samplingProperties.uaddressMode = ImageSamplingProperties::AddressMode::PERIODIC;
                samplingProperties.vaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
                samplingProperties.filterType = ImageSamplingProperties::FilterType::LINEAR;
                imageHandler->bindImage(image, samplingProperties);
            }
        }
    }

    // Bind direct lighting properties.
    if (hasUniform(HW::NUM_ACTIVE_LIGHT_SOURCES))
    {
        int lightCount = lightHandler->getDirectLighting() ? (int) lightHandler->getLightSources().size() : 0;
        bindUniform(HW::NUM_ACTIVE_LIGHT_SOURCES, Value::createValue(lightCount));
        LightIdMap idMap = lightHandler->computeLightIdMap(lightHandler->getLightSources());
        size_t index = 0;
        for (NodePtr light : lightHandler->getLightSources())
        {
            auto nodeDef = light->getNodeDef();
            if (!nodeDef)
            {
                continue;
            }

            const std::string prefix = HW::LIGHT_DATA_INSTANCE + "[" + std::to_string(index) + "]";

            // Set light type id
            std::string lightType(prefix + ".type");
            if (hasUniform(lightType))
            {
                unsigned int lightTypeValue = idMap[nodeDef->getName()];
                bindUniform(lightType, Value::createValue((int) lightTypeValue));
            }

            // Set all inputs
            for (const auto& input : light->getInputs())
            {
                // Make sure we have a value to set
                if (input->hasValue())
                {
                    std::string inputName(prefix + "." + input->getName());
                    if (hasUniform(inputName))
                    {
                        if (input->getName() == "direction" && input->hasValue() && input->getValue()->isA<Vector3>())
                        {
                            Vector3 dir = input->getValue()->asA<Vector3>();
                            dir = lightHandler->getLightTransform().transformVector(dir);
                            bindUniform(inputName, Value::createValue(dir));
                        }
                        else
                        {
                            bindUniform(inputName, input->getValue());
                        }
                    }
                }
            }

            ++index;
        }
    }

    // Bind the directional albedo table, if needed.
    ImagePtr albedoTable = lightHandler->getAlbedoTable();
    if (albedoTable && hasUniform(TEXTURE_NAME(HW::ALBEDO_TABLE)))
    {
        ImageSamplingProperties samplingProperties;
        samplingProperties.uaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
        samplingProperties.vaddressMode = ImageSamplingProperties::AddressMode::CLAMP;
        samplingProperties.filterType = ImageSamplingProperties::FilterType::LINEAR;
        bindTexture(imageHandler,
                    TEXTURE_NAME(HW::ALBEDO_TABLE),
                    albedoTable,
                    samplingProperties);
    }
}

bool MslProgram::hasUniform(const string& name)
{
    const MslProgram::InputMap& uniformList = getUniformsList();
    if(uniformList.find(name) != uniformList.end())
        return true;
    if(_globalUniformNameList.find(name) != _globalUniformNameList.end() && uniformList.find(_globalUniformNameList[name]) != uniformList.end())
        return true;
    return false;
}

void MslProgram::bindUniform(const string& name, ConstValuePtr value, bool errorIfMissing)
{
    const MslProgram::InputMap& uniformList = getUniformsList();
    auto input = uniformList.find(name);
    if (input != uniformList.end())
    {
        _uniformList[name]->value = value->copy();
    }
    else
    {
        auto globalNameMapping = _globalUniformNameList.find(name);
        if(globalNameMapping != _globalUniformNameList.end())
        {
            bindUniform(globalNameMapping->second, value, errorIfMissing);
        }
        else
        {
            if (errorIfMissing)
            {
                throw ExceptionRenderError("Unknown uniform: " + name);
            }
            return;
        }
    }
}

void MslProgram::bindViewInformation(CameraPtr camera)
{
    StringVec errors;
    const string errorType("MSL view input binding error.");

    if (_pso == nil)
    {
        errors.push_back("Cannot bind without a valid program");
        throw ExceptionRenderError(errorType, errors);
    }
    if (!camera)
    {
        errors.push_back("Cannot bind without a view handler");
        throw ExceptionRenderError(errorType, errors);
    }
}

void MslProgram::bindTimeAndFrame(float time, float frame)
{
    _time = time;
    _frame = frame;
}

void MslProgram::clearInputLists()
{
    _uniformList.clear();
    _globalUniformNameList.clear();
    _attributeList.clear();
    _attributeBufferIds.clear();
    _indexBufferIds.clear();
    _explicitBoundImages.clear();
}

const MslProgram::InputMap& MslProgram::getUniformsList()
{
    return updateUniformsList();
}

const MslProgram::InputMap& MslProgram::getAttributesList()
{
    return updateAttributesList();
}

const MslProgram::InputMap& MslProgram::updateUniformsList()
{
    StringVec errors;
    const string errorType("MSL uniform parsing error.");

    if (_uniformList.size() > 0)
    {
        return _uniformList;
    }

    if (_pso == nil)
    {
        errors.push_back("Cannot parse for uniforms without a valid program");
        throw ExceptionRenderError(errorType, errors);
    }
        
    for (MTLArgument *arg in _psoReflection.vertexArguments)
    {
        if (arg.bufferDataType == MTLDataTypeStruct)
        {
            for (MTLStructMember* member in arg.bufferStructType.members)
            {
                InputPtr inputPtr = std::make_shared<Input>(arg.index, member.dataType, arg.bufferDataSize, EMPTY_STRING);
                std::string memberName =  member.name.UTF8String;
                std::string uboDotMemberName = std::string(arg.name.UTF8String) + "." + member.name.UTF8String;
                _uniformList[uboDotMemberName] = inputPtr;
                _globalUniformNameList[member.name.UTF8String] = uboDotMemberName;
            }
        }
    }
    
    for (MTLArgument *arg in _psoReflection.fragmentArguments)
    {
        if (arg.type == MTLArgumentTypeBuffer && arg.bufferDataType == MTLDataTypeStruct)
        {
            for (MTLStructMember* member in arg.bufferStructType.members)
            {
                std::string uboObjectName = std::string(arg.name.UTF8String);
                std::string memberName =  member.name.UTF8String;
                std::string uboDotMemberName = uboObjectName + "." + memberName;
                
                InputPtr inputPtr = std::make_shared<Input>(arg.index, member.dataType, arg.bufferDataSize, EMPTY_STRING);
                _uniformList[uboDotMemberName] = inputPtr;
                _globalUniformNameList[memberName] = uboDotMemberName;
                
                if(MTLArrayType* arrayMember = member.arrayType)
                {
                    for(int i = 0; i < arrayMember.arrayLength; ++i)
                    {
                        for (MTLStructMember* ArrayOfStructMember in arrayMember.elementStructType.members)
                        {
                            std::string memberNameDotSubmember = memberName + "[" + std::to_string(i) + "]." + ArrayOfStructMember.name.UTF8String;
                            std::string uboDotMemberNameDotSubmemberName = uboObjectName + "." + memberNameDotSubmember;
                            
                            InputPtr inputPtr = std::make_shared<Input>(ArrayOfStructMember.argumentIndex, ArrayOfStructMember.dataType, ArrayOfStructMember.offset, EMPTY_STRING);
                            _uniformList[uboDotMemberNameDotSubmemberName] = inputPtr;
                            _globalUniformNameList[memberNameDotSubmember] = uboDotMemberNameDotSubmemberName;
                        }
                    }
                }
            }
        }
        
        if(arg.type == MTLArgumentTypeTexture)
        {
            if(HW::ENV_RADIANCE != arg.name.UTF8String && HW::ENV_IRRADIANCE != arg.name.UTF8String)
            {
                std::string texture_name = arg.name.UTF8String;
                InputPtr inputPtr = std::make_shared<Input>(arg.index, 58, -1, EMPTY_STRING);
                _uniformList[texture_name] = inputPtr;
            }
        }
    }

    if (_shader)
    {
        // Check for any type mismatches between the program and the h/w shader.
        // i.e the type indicated by the HwShader does not match what was generated.
        bool uniformTypeMismatchFound = false;

        const ShaderStage& ps = _shader->getStage(Stage::PIXEL);
        const ShaderStage& vs = _shader->getStage(Stage::VERTEX);

        // Process constants
        const VariableBlock& constants = ps.getConstantBlock();
        for (size_t i=0; i< constants.size(); ++i)
        {
            const ShaderPort* v = constants[i];
            // There is no way to match with an unnamed variable
            if (v->getVariable().empty())
            {
                continue;
            }

            // TODO: Shoud we really create new ones here each update?
            InputPtr inputPtr = std::make_shared<Input>(-1, -1, int(v->getType()->getSize()), EMPTY_STRING);
            _uniformList[v->getVariable()] = inputPtr;
            inputPtr->isConstant = true;
            inputPtr->value = v->getValue();
            inputPtr->typeString = v->getType()->getName();
            inputPtr->path = v->getPath();
        }

        // Process pixel stage uniforms
        for (const auto& uniformMap : ps.getUniformBlocks())
        {
            const VariableBlock& uniforms = *uniformMap.second;
            if (uniforms.getName() == HW::LIGHT_DATA)
            {
                // Need to go through LightHandler to match with uniforms
                continue;
            }

            for (size_t i = 0; i < uniforms.size(); ++i)
            {
                const ShaderPort* v = uniforms[i];
                MTLDataType resourceType = mapTypeToMetalType(v->getType());

                // There is no way to match with an unnamed variable
                if (v->getVariable().empty())
                {
                    continue;
                }

                // Ignore types which are unsupported in MSL.
                if (resourceType == MTLDataTypeNone)
                {
                    continue;
                }

                int tries = 0;
                auto inputIt = _uniformList.find(v->getVariable());
try_again:      if (inputIt != _uniformList.end())
                {
                    Input* input = inputIt->second.get();
                    input->path = v->getPath();
                    input->value = v->getValue();
                    if (input->resourceType == resourceType)
                    {
                        input->typeString = v->getType()->getName();
                    }
                    else
                    {
                        errors.push_back(
                            "Pixel shader uniform block type mismatch [" + uniforms.getName() + "]. "
                            + "Name: \"" + v->getVariable()
                            + "\". Type: \"" + v->getType()->getName()
                            + "\". Semantic: \"" + v->getSemantic()
                            + "\". Value: \"" + (v->getValue() ? v->getValue()->getValueString() : "<none>")
                            + "\". resourceType: " + std::to_string(mapTypeToMetalType(v->getType()))
                        );
                        uniformTypeMismatchFound = true;
                    }
                }
                else
                {
                    if(tries == 0)
                    {
                        ++tries;
                        if(v->getType() == Type::FILENAME)
                        {
                            inputIt = _uniformList.find(TEXTURE_NAME(v->getVariable()));
                        }
                        else
                        {
                            inputIt = _uniformList.find(uniforms.getInstance() + "." + v->getVariable());
                        }
                        goto try_again;
                    }
                }
            }
        }

        // Process vertex stage uniforms
        for (const auto& uniformMap : vs.getUniformBlocks())
        {
            const VariableBlock& uniforms = *uniformMap.second;
            for (size_t i = 0; i < uniforms.size(); ++i)
            {
                const ShaderPort* v = uniforms[i];
                auto inputIt = _uniformList.find(v->getVariable());
                if (inputIt != _uniformList.end())
                {
                    Input* input = inputIt->second.get();
                    if (input->resourceType == mapTypeToMetalType(v->getType()))
                    {
                        input->typeString = v->getType()->getName();
                        input->value = v->getValue();
                        input->path = v->getPath();
                        input->unit = v->getUnit();
                    }
                    else
                    {
                        errors.push_back(
                            "Vertex shader uniform block type mismatch [" + uniforms.getName() + "]. "
                            + "Name: \"" + v->getVariable()
                            + "\". Type: \"" + v->getType()->getName()
                            + "\". Semantic: \"" + v->getSemantic()
                            + "\". Value: \"" + (v->getValue() ? v->getValue()->getValueString() : "<none>")
                            + "\". Unit: \"" + (!v->getUnit().empty() ? v->getUnit() : "<none>")
                            + "\". resourceType: " + std::to_string(mapTypeToMetalType(v->getType()))
                        );
                        uniformTypeMismatchFound = true;
                    }
                }
            }
        }

        // Throw an error if any type mismatches were found
        if (uniformTypeMismatchFound)
        {
            throw ExceptionRenderError(errorType, errors);
        }
    }

    return _uniformList;
}

void MslProgram::bindUniformBuffers(id<MTLRenderCommandEncoder> renderCmdEncoder, LightHandlerPtr lightHandler, CameraPtr cam)
{
    auto setCommonUniform = [this](LightHandlerPtr lightHandler, CameraPtr cam, std::string uniformName, std::vector<unsigned char>& data, size_t offset) -> bool
    {
        // View position and direction
        if (uniformName == HW::VIEW_POSITION)
        {
            Matrix44 viewInverse = cam->getViewMatrix().getInverse();
            MaterialX::Vector3 viewPosition(viewInverse[3][0], viewInverse[3][1], viewInverse[3][2]);
            memcpy((void*)&data[offset], viewPosition.data(), 3 * sizeof(float));
            return true;
        }
        if (uniformName == HW::VIEW_DIRECTION)
        {
            memcpy((void*)&data[offset], cam->getViewPosition().data(), 3 * sizeof(float));
            return true;
        }
        
        // World matrix variants
        const Matrix44& world = cam->getWorldMatrix();
        Matrix44 invWorld = world.getInverse();
        Matrix44 invTransWorld = invWorld.getTranspose();
        if (uniformName == HW::WORLD_MATRIX)
        {
            memcpy((void*)&data[offset], world.data(), 4 * 4 * sizeof(float));
            return false;
        }
        if (uniformName == HW::WORLD_TRANSPOSE_MATRIX)
        {
            memcpy((void*)&data[offset], world.getTranspose().data(), 4 * 4 * sizeof(float));
            return true;
        }
        if (uniformName == HW::WORLD_INVERSE_MATRIX)
        {
            memcpy((void*)&data[offset], invWorld.getTranspose().data(), 4 * 4 * sizeof(float));
            return true;
        }
        if (uniformName == HW::WORLD_INVERSE_TRANSPOSE_MATRIX)
        {
            memcpy((void*)&data[offset], invTransWorld.getTranspose().data(), 4 * 4 * sizeof(float));
            return true;
        }
        if(uniformName == HW::FRAME)
        {
            memcpy((void*)&data[offset], (const void*)&_frame, sizeof(float));
            return true;
        }
        if(uniformName == HW::TIME)
        {
            memcpy((void*)&data[offset], (const void*)&_time, sizeof(float));
            return true;
        }


        // Projection matrix variants
        const Matrix44& proj = cam->getProjectionMatrix();
        if (uniformName == HW::PROJ_MATRIX)
        {
            memcpy((void*)&data[offset], proj.data(), 4 * 4 * sizeof(float));
            return true;
        }
        if (uniformName == HW::PROJ_TRANSPOSE_MATRIX)
        {
            memcpy((void*)&data[offset], proj.getTranspose().data(), 4 * 4 * sizeof(float));
            return true;
        }

        const Matrix44& projInverse= proj.getInverse();
        if (uniformName == HW::PROJ_INVERSE_MATRIX)
        {
            memcpy((void*)&data[offset], projInverse.data(), 4 * 4 * sizeof(float));
            return true;
        }
        if (uniformName == HW::PROJ_INVERSE_TRANSPOSE_MATRIX)
        {
            memcpy((void*)&data[offset], projInverse.getTranspose().data(), 4 * 4 * sizeof(float));
            return true;
        }

        // View matrix variants
        const Matrix44& view = cam->getViewMatrix();
        Matrix44 viewInverse = view.getInverse();
        if (uniformName == HW::VIEW_MATRIX)
        {
            memcpy((void*)&data[offset], view.data(), 4 * 4 * sizeof(float));
            return true;
        }
        if (uniformName == HW::VIEW_TRANSPOSE_MATRIX)
        {
            memcpy((void*)&data[offset], view.getTranspose().data(), 4 * 4 * sizeof(float));
            return true;
        }
        if (uniformName == HW::VIEW_INVERSE_MATRIX)
        {
            memcpy((void*)&data[offset], viewInverse.data(), 4 * 4 * sizeof(float));
            return true;
        }
        if (uniformName == HW::VIEW_INVERSE_TRANSPOSE_MATRIX)
        {
            memcpy((void*)&data[offset], viewInverse.getTranspose().data(), 4 * 4 * sizeof(float));
            return true;
        }
        
        // View-projection matrix
        const Matrix44& viewProj = view * proj;
        if (uniformName == HW::VIEW_PROJECTION_MATRIX)
        {
            memcpy((void*)&data[offset], viewProj.data(), 4 * 4 * sizeof(float));
            return true;
        }
        
        // View-projection-world matrix
        const Matrix44& viewProjWorld = viewProj * world;
        if (uniformName == HW::WORLD_VIEW_PROJECTION_MATRIX)
        {
            memcpy((void*)&data[offset], viewProjWorld.data(), 4 * 4 * sizeof(float));
            return true;
        }
        
        if (uniformName == HW::ENV_RADIANCE_MIPS)
        {
            unsigned int maxMipCount = lightHandler->getEnvRadianceMap()->getMaxMipCount();
            memcpy((void*)&data[offset], &maxMipCount, sizeof(maxMipCount));
            return true;
        }
        
        return false;
    };

    auto setValue = [](MaterialX::ValuePtr value, std::vector<unsigned char>& data, size_t offset)
    {
        if (value->getTypeString() == "float")
        {
            float v = value->asA<float>();
            memcpy((void*)&data[offset], &v, sizeof(float));
        }
        else if (value->getTypeString() == "integer")
        {
            int v = value->asA<int>();
            memcpy((void*)&data[offset], &v, sizeof(int));
        }
        else if (value->getTypeString() == "boolean")
        {
            bool v = value->asA<bool>();
            memcpy((void*)&data[offset], &v, sizeof(bool));
        }
        else if (value->getTypeString() == "color3")
        {
            Color3 v = value->asA<Color3>();
            memcpy((void*)&data[offset], &v, sizeof(Color3));
        }
        else if (value->getTypeString() == "color4")
        {
            Color4 v = value->asA<Color4>();
            memcpy((void*)&data[offset], &v, sizeof(Color4));
        }
        else if (value->getTypeString() == "vector2")
        {
            Vector2 v = value->asA<Vector2>();
            memcpy((void*)&data[offset], &v, sizeof(Vector2));
        }
        else if (value->getTypeString() == "vector3")
        {
            Vector3 v = value->asA<Vector3>();
            memcpy((void*)&data[offset], &v, sizeof(Vector3));
        }
        else if (value->getTypeString() == "vector4")
        {
            Vector4 v = value->asA<Vector4>();
            memcpy((void*)&data[offset], &v, sizeof(Vector4));
        }
        else if (value->getTypeString() == "matrix33")
        {
            Matrix33 m = value->asA<Matrix33>();
            float tmp[12] = {m[0][0], m[0][1], m[0][2], 0.0f,
                             m[1][0], m[1][1], m[1][2], 0.0f,
                             m[2][0], m[2][1], m[2][2], 0.0f};
            memcpy((void*)&data[offset], &tmp, sizeof(tmp));
        }
        else if (value->getTypeString() == "matrix44")
        {
            Matrix44 m = value->asA<Matrix44>();
            memcpy((void*)&data[offset], &m, sizeof(Matrix44));
        }
        else if (value->getTypeString() == "string")
        {
            // Bound differently. Ignored here!
        }
        else
        {
            throw ExceptionRenderError(
                "MSL input binding error.",
                { "Unsupported data type when setting uniform value" }
            );
        }
    };
    
    for (MTLArgument *arg in _psoReflection.vertexArguments)
    {
        if (arg.type == MTLArgumentTypeBuffer && arg.bufferDataType == MTLDataTypeStruct)
        {
            std::vector<unsigned char> uniformBufferData(arg.bufferDataSize);
            for (MTLStructMember* member in arg.bufferStructType.members)
            {
                if(!setCommonUniform(lightHandler, cam, member.name.UTF8String, uniformBufferData, member.offset))
                {
                    MaterialX::ValuePtr value = _uniformList[string(arg.name.UTF8String) + "." + member.name.UTF8String]->value;
                    if(value)
                    {
                        setValue(value, uniformBufferData, member.offset);
                    }
                }
            }
            
            if(arg.bufferStructType)
                [renderCmdEncoder setVertexBytes:(void*)uniformBufferData.data() length:arg.bufferDataSize atIndex:arg.index];
        }
    }
    
    for (MTLArgument *arg in _psoReflection.fragmentArguments)
    {
        if (arg.type == MTLArgumentTypeBuffer && arg.bufferDataType == MTLDataTypeStruct)
        {
            std::vector<unsigned char> uniformBufferData(arg.bufferDataSize);
    
            for (MTLStructMember* member in arg.bufferStructType.members)
            {
                string uniformName = string(arg.name.UTF8String) + "." + member.name.UTF8String;
                    
                if(!setCommonUniform(lightHandler, cam, member.name.UTF8String, uniformBufferData, member.offset))
                {
                    auto uniformInfo = _uniformList.find(uniformName);
                    if (uniformInfo != _uniformList.end())
                    {
                        MaterialX::ValuePtr value = uniformInfo->second->value;
                        if(value)
                        {
                            setValue(value, uniformBufferData, member.offset);
                        }
                    }
                    else
                    {
                        
                    }
                }
                    
                if(MTLArrayType* arrayMember = member.arrayType)
                {
                    for(int i = 0; i < arrayMember.arrayLength; ++i)
                    {
                        for (MTLStructMember* ArrayOfStructMember in arrayMember.elementStructType.members)
                        {
                            string uniformNameSubArray = uniformName + "[" + std::to_string(i) + "]." + ArrayOfStructMember.name.UTF8String;

                            auto uniformInfo = _uniformList.find(uniformNameSubArray);
                            if (uniformInfo != _uniformList.end())
                            {
                                MaterialX::ValuePtr value = uniformInfo->second->value;
                                if(value)
                                {
                                    setValue(value, uniformBufferData, member.offset + i * arrayMember.stride + ArrayOfStructMember.offset);
                                }
                            }
                        }
                    }
                }
            }
            
            if(arg.bufferStructType)
                [renderCmdEncoder setFragmentBytes:(void*)uniformBufferData.data() length:arg.bufferDataSize atIndex:arg.index];
        }
    }
}

void MslProgram::reset()
{
    if (_pso != nil)
    {
       [_pso release];
    }
    _pso = nil;
    
    if(_psoReflection != nil)
    {
       [_psoReflection release];
    }
    _psoReflection = nil;

    // Program deleted, so also clear cached input lists
    clearInputLists();
}

MTLDataType MslProgram::mapTypeToMetalType(const TypeDesc* type)
{
    if (type == Type::INTEGER)
        return MTLDataTypeInt;
    else if (type == Type::BOOLEAN)
        return MTLDataTypeBool;
    else if (type == Type::FLOAT)
        return MTLDataTypeFloat;
    else if (type->isFloat2())
        return MTLDataTypeFloat2;
    else if (type->isFloat3())
        return MTLDataTypeFloat3;
    else if (type->isFloat4())
        return MTLDataTypeFloat4;
    else if (type == Type::MATRIX33)
        return MTLDataTypeFloat3x3;
    else if (type == Type::MATRIX44)
        return MTLDataTypeFloat4x4;
    else if (type == Type::FILENAME)
    {
        // A "filename" is not indicative of type, so just return a 2d sampler.
        return MTLDataTypeTexture;
    }
    else if (type == Type::BSDF               ||
             type == Type::MATERIAL           ||
             type == Type::DISPLACEMENTSHADER ||
             type == Type::EDF                ||
             type == Type::VDF                ||
             type == Type::SURFACESHADER      ||
             type == Type::LIGHTSHADER        ||
             type == Type::VOLUMESHADER)
        return MTLDataTypeStruct;

    return MTLDataTypeNone;
}

const MslProgram::InputMap& MslProgram::updateAttributesList()
{
    StringVec errors;
    const string errorType("MSL attribute parsing error.");

    if (_pso == nil)
    {
        errors.push_back("Cannot parse for attributes without a valid program");
        throw ExceptionRenderError(errorType, errors);
    }

    if (_shader)
    {
        const ShaderStage& vs = _shader->getStage(Stage::VERTEX);

        bool uniformTypeMismatchFound = false;

        const VariableBlock& vertexInputs = vs.getInputBlock(HW::VERTEX_INPUTS);
        if (!vertexInputs.empty())
        {
            for (size_t i = 0; i < vertexInputs.size(); ++i)
            {
                const ShaderPort* v = vertexInputs[i];
                
                string variableName = v->getVariable();
                size_t dotPos = variableName.find('.');
                string variableMemberName = variableName.substr(dotPos + 1);
                
                auto inputIt = _attributeList.find(variableMemberName);
                if (inputIt != _attributeList.end())
                {
                    Input* input = inputIt->second.get();
                    input->value = v->getValue();
                    if (input->resourceType == mapTypeToMetalType(v->getType()))
                    {
                        input->typeString = v->getType()->getName();
                    }
                    else
                    {
                        errors.push_back(
                            "Vertex shader attribute type mismatch in block. Name: \"" + v->getVariable()
                            + "\". Type: \"" + v->getType()->getName()
                            + "\". Semantic: \"" + v->getSemantic()
                            + "\". Value: \"" + (v->getValue() ? v->getValue()->getValueString() : "<none>")
                            + "\". resourceType: " + std::to_string(mapTypeToMetalType(v->getType()))
                        );
                        uniformTypeMismatchFound = true;
                    }
                }
            }
        }

        // Throw an error if any type mismatches were found
        if (uniformTypeMismatchFound)
        {
            throw ExceptionRenderError(errorType, errors);
        }
    }

    return _attributeList;
}

void MslProgram::findInputs(const string& variable,
                             const InputMap& variableList,
                             InputMap& foundList,
                             bool exactMatch)
{
    foundList.clear();

    // Scan all attributes which match the attribute identifier completely or as a prefix
    //
    int ilocation = UNDEFINED_METAL_PROGRAM_LOCATION;
    auto input = variableList.find(variable);
    if (input != variableList.end())
    {
        ilocation = input->second->location;
        if (ilocation >= 0)
        {
            foundList[variable] = input->second;
        }
    }
    else if (!exactMatch)
    {
        for (input = variableList.begin(); input != variableList.end(); ++input)
        {
            const string& name = input->first;
            if (name.compare(0, variable.size(), variable) == 0)
            {
                ilocation = input->second->location;
                if (ilocation >= 0)
                {
                    foundList[input->first] = input->second;
                }
            }
        }
    }
}

void MslProgram::printUniforms(std::ostream& outputStream)
{
    updateUniformsList();
    for (const auto& input : _uniformList)
    {
        unsigned int resourceType = input.second->resourceType;
        int location = input.second->location;
        int size = input.second->size;
        string type = input.second->typeString;
        string value = input.second->value ? input.second->value->getValueString() : EMPTY_STRING;
        string unit = input.second->unit;
        string colorspace = input.second->colorspace;
        bool isConstant = input.second->isConstant;
        outputStream << "Program Uniform: \"" << input.first
            << "\". Location:" << location
            << ". ResourceType: " << std::hex << resourceType
            << ". Size: " << std::dec << size;
        if (!type.empty())
            outputStream << ". TypeString: \"" << type << "\"";
        if (!value.empty())
        {
            outputStream << ". Value: " << value;
            if (!unit.empty())
                outputStream << ". Unit: " << unit;
            if (!colorspace.empty())
                outputStream << ". Colorspace: " << colorspace;
        }
        outputStream << ". Is constant: " << isConstant;
        if (!input.second->path.empty())
            outputStream << ". Element Path: \"" << input.second->path << "\"";
        outputStream << "." << std::endl;
    }
}


void MslProgram::printAttributes(std::ostream& outputStream)
{
    updateAttributesList();
    for (const auto& input : _attributeList)
    {
        unsigned int resourceType = input.second->resourceType;
        int location = input.second->location;
        int size = input.second->size;
        string type = input.second->typeString;
        string value = input.second->value ? input.second->value->getValueString() : EMPTY_STRING;
        outputStream << "Program Attribute: \"" << input.first
            << "\". Location:" << location
            << ". ResourceType: " << std::hex << resourceType
            << ". Size: " << std::dec << size;
        if (!type.empty())
            outputStream << ". TypeString: \"" << type << "\"";
        if (!value.empty())
            outputStream << ". Value: " << value;
        outputStream << "." << std::endl;
    }
}

MATERIALX_NAMESPACE_END

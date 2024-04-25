//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MSLPROGRAM_H
#define MATERIALX_MSLPROGRAM_H

/// @file
/// MSL Program interfaces

#include <MaterialXRenderMsl/Export.h>

#include <MaterialXRender/Camera.h>
#include <MaterialXRender/GeometryHandler.h>
#include <MaterialXRender/ImageHandler.h>
#include <MaterialXRender/LightHandler.h>

#include <MaterialXGenShader/Shader.h>

#import <Metal/Metal.h>

MATERIALX_NAMESPACE_BEGIN

// Shared pointer to a MslProgram
using MslProgramPtr = std::shared_ptr<class MslProgram>;
using MetalFramebufferPtr = std::shared_ptr<class MetalFramebuffer>;

/// @class MslProgram
/// A class representing an executable MSL program.
///
/// There are two main interfaces which can be used.  One which takes in a HwShader and one which
/// allows for explicit setting of shader stage code.
class MX_RENDERMSL_API MslProgram
{
  public:
    /// Create a MSL program instance
    static MslProgramPtr create()
    {
        return MslProgramPtr(new MslProgram());
    }

    /// Destructor
    virtual ~MslProgram();

    /// @name Shader code setup
    /// @{

    /// Set up code stages to validate based on an input hardware shader.
    /// @param shader Hardware shader to use
    void setStages(ShaderPtr shader);

    /// Set the code stages based on a list of stage strings.
    /// Refer to the ordering of stages as defined by a HwShader.
    /// @param stage Name of the shader stage.
    /// @param sourceCode Source code of the shader stage.
    void addStage(const string& stage, const string& sourceCode);

    /// Get source code string for a given stage.
    /// @return Shader stage string. String is empty if not found.
    const string& getStageSourceCode(const string& stage) const;

    /// Clear out any existing stages
    void clearStages();

    /// Return the shader, if any, used to generate this program.
    ShaderPtr getShader() const
    {
        return _shader;
    }

    /// @}
    /// @name Program validation and introspection
    /// @{

    /// Create the pipeline state object from stages specified
    ///  @param device MetalDevice that  pipeline state object is being created on.
    ///  @param framebuffer specifies information about output frame buffer.
    /// An exception is thrown if the program cannot be created.
    /// The exception will contain a list of program creation errors.
    /// @return Pipeline State Object identifier.
    id<MTLRenderPipelineState> build(id<MTLDevice> device, MetalFramebufferPtr frameBuffer);

    /// Structure to hold information about program inputs.
    /// The structure is populated by directly scanning the program so may not contain
    /// some inputs listed on any associated HwShader as those inputs may have been
    /// optimized out if they are unused.
    struct MX_RENDERMSL_API Input
    {
        static int INVALID_METAL_TYPE;

        /// Program location. -1 means an invalid location
        int location;
        /// Metal type of the input. -1 means an invalid type
        int resourceType;
        /// Size.
        int size;
        /// Input type string. Will only be non-empty if initialized stages with a HwShader
        string typeString;
        /// Input value. Will only be non-empty if initialized stages with a HwShader and a value was set during
        /// shader generation.
        MaterialX::ValuePtr value;
        /// Is this a constant
        bool isConstant;
        /// Element path (if any)
        string path;
        /// Unit
        string unit;
        /// Colorspace 
        string colorspace;

        /// Program input constructor
        Input(int inputLocation, int inputType, int inputSize, const string& inputPath) :
            location(inputLocation),
            resourceType(inputType),
            size(inputSize),
            isConstant(false),
            path(inputPath)
        { }
    };
    /// Program input structure shared pointer type
    using InputPtr = std::shared_ptr<Input>;
    /// Program input shaded pointer map type
    using InputMap = std::unordered_map<string, InputPtr>;

    /// Get list of program input uniforms.
    /// The program must have been created successfully first.
    /// An exception is thrown if the parsing of the program for uniforms cannot be performed.
    /// @return Program uniforms list.
    const InputMap& getUniformsList();

    /// Get list of program input attributes.
    /// The program must have been created successfully first.
    /// An exception is thrown if the parsing of the program for attribute cannot be performed.
    /// @return Program attributes list.
    const InputMap& getAttributesList();

    /// Find the locations in the program which starts with a given variable name
    /// @param variable Variable to search for
    /// @param variableList List of program inputs to search
    /// @param foundList Returned list of found program inputs. Empty if none found.
    /// @param exactMatch Search for exact variable name match.
    void findInputs(const string& variable,
                    const InputMap& variableList,
                    InputMap& foundList,
                    bool exactMatch);

    /// @}
    /// @name Program activation
    /// @{

    /// Bind the pipeline state object to the command encoder.
    /// @param renderCmdEncoder encoder that binds the pipeline state object.
    /// @return False if failed
    bool bind(id<MTLRenderCommandEncoder> renderCmdEncoder);
    
    /// Bind inputs
    ///  @param renderCmdEncoder encoder that inputs will be bound to.
    ///  @param cam Camera object use to view the object
    ///  @param geometryHandler
    ///  @param imageHandler
    ///  @param lightHandler
    ///  @return void - No return value
    void prepareUsedResources(id<MTLRenderCommandEncoder> renderCmdEncoder,
                        CameraPtr cam,
                        GeometryHandlerPtr geometryHandler,
                        ImageHandlerPtr imageHandler,
                        LightHandlerPtr lightHandler);

    /// Return true if a uniform with the given name is present.
    bool hasUniform(const string& name);

    /// Bind a value to the uniform with the given name.
    void bindUniform(const string& name, ConstValuePtr value, bool errorIfMissing = true);

    /// Bind attribute buffers to attribute inputs.
    /// A hardware buffer of the given attribute type is created and bound to the program locations
    /// for the input attribute.
    /// @param renderCmdEncoder Metal Render Command Encoder that the attribute being bind to
    /// @param inputs Attribute inputs to bind to
    /// @param mesh Mesh containing streams to bind
    void bindAttribute(id<MTLRenderCommandEncoder> renderCmdEncoder,
                       const MslProgram::InputMap& inputs,
                       MeshPtr mesh);

    /// Bind input geometry partition (indexing)
    void bindPartition(MeshPartitionPtr partition);

    /// Bind input geometry streams
    void bindMesh(id<MTLRenderCommandEncoder> renderCmdEncoder, MeshPtr mesh);
    
    /// Queries the index buffer assinged to a mesh partition
    id<MTLBuffer> getIndexBuffer(MeshPartitionPtr mesh) {
        if(_indexBufferIds.find(mesh) != _indexBufferIds.end())
            return _indexBufferIds[mesh];
        return nil;
    }


    /// Unbind any bound geometry
    void unbindGeometry();

    /// Bind any input textures
    void bindTextures(id<MTLRenderCommandEncoder> renderCmdEncoder,
                      LightHandlerPtr lightHandler,
                      ImageHandlerPtr imageHandler);
    
    void bindTexture(ImageHandlerPtr imageHandler,
                     string shaderTextureName,
                     ImagePtr imagePtr,
                     ImageSamplingProperties samplingProperties);

    /// Bind lighting
    void bindLighting(LightHandlerPtr lightHandler, ImageHandlerPtr imageHandler);

    /// Bind view information
    void bindViewInformation(CameraPtr camera);
    
    /// Bind time and frame
    void bindTimeAndFrame(float time = 1.0f, float frame = 1.0f);

    /// @}
    /// @name Utilities
    /// @{
    
    /// Returns if alpha blending is enabled.
    bool isTransparent() const { return _alphaBlendingEnabled; }
    
    /// Specify textures bound to this program shouldn't be mip mapped.
    void setEnableMipMaps(bool enableMipMapping) { _enableMipMapping = enableMipMapping; }

    /// Print all uniforms to the given stream.
    void printUniforms(std::ostream& outputStream);

    /// Print all attributes to the given stream.
    void printAttributes(std::ostream& outputStream);

    /// @}

  public:
    static unsigned int UNDEFINED_METAL_RESOURCE_ID;
    static int UNDEFINED_METAL_PROGRAM_LOCATION;

  protected:
    MslProgram();

    // Update a list of program input uniforms
    const InputMap& updateUniformsList();

    // Update a list of program input attributes
    const InputMap& updateAttributesList();

    // Clear out any cached input lists
    void clearInputLists();

    // Utility to find a uniform value in an uniform list.
    // If uniform cannot be found a null pointer will be return.
    ValuePtr findUniformValue(const string& uniformName, const InputMap& uniformList);

    // Bind an individual texture to a program uniform location
    ImagePtr bindTexture(id<MTLRenderCommandEncoder> renderCmdEncoder,
                         unsigned int uniformLocation,
                         const FilePath& filePath,
                         ImageSamplingProperties samplingProperties,
                         ImageHandlerPtr imageHandler);
    
    // Bind an individual texture to a program uniform location
    ImagePtr bindTexture(id<MTLRenderCommandEncoder> renderCmdEncoder,
                         unsigned int uniformLocation,
                         ImagePtr imagePtr,
                         ImageHandlerPtr imageHandler);
        
    void bindUniformBuffers(id<MTLRenderCommandEncoder> renderCmdEncoder,
                            LightHandlerPtr lightHandler,
                            CameraPtr camera);

    // Delete any currently created pso
    void reset();

    // Utility to map a MaterialX type to an METAL type
    static MTLDataType mapTypeToMetalType(const TypeDesc* type);

  private:
    // Stages used to create program
    // Map of stage name and its source code
    StringMap _stages;

    // Generated pipeline state object. A non-zero number indicates a valid shader program.
    id<MTLRenderPipelineState> _pso = nil;
    MTLRenderPipelineReflection* _psoReflection = nil;

    // List of program input uniforms
    InputMap _uniformList;
    std::unordered_map<std::string, std::string> _globalUniformNameList;
    // List of program input attributes
    InputMap _attributeList;
    
    std::unordered_map<std::string, ImagePtr> _explicitBoundImages;

    // Hardware shader (if any) used for program creation
    ShaderPtr _shader;

    // Attribute buffer resource handles
    // for each attribute identifier in the program
    std::unordered_map<string, id<MTLBuffer>> _attributeBufferIds;

    // Attribute indexing buffer handle
    std::map<MeshPartitionPtr, id<MTLBuffer>> _indexBufferIds;

    // Program texture map
    std::unordered_map<string, unsigned int> _programTextures;
    
    // Metal Device Object
    id<MTLDevice> _device = nil;
    
    // Currently bound mesh
    MeshPtr _boundMesh = nullptr;

    bool _alphaBlendingEnabled = false;
    
    float _time = 0.0f;
    float _frame = 0.0f;
    
    bool _enableMipMapping = true;
};

MATERIALX_NAMESPACE_END

#endif

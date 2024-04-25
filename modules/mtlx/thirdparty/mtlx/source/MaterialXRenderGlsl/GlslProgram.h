//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GLSLPROGRAM_H
#define MATERIALX_GLSLPROGRAM_H

/// @file
/// GLSL Program interfaces

#include <MaterialXRenderGlsl/Export.h>

#include <MaterialXRender/Camera.h>
#include <MaterialXRender/GeometryHandler.h>
#include <MaterialXRender/ImageHandler.h>
#include <MaterialXRender/LightHandler.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

// Shared pointer to a GlslProgram
using GlslProgramPtr = std::shared_ptr<class GlslProgram>;

/// @class GlslProgram
/// A class representing an executable GLSL program.
///
/// There are two main interfaces which can be used.  One which takes in a HwShader and one which
/// allows for explicit setting of shader stage code.
class MX_RENDERGLSL_API GlslProgram
{
  public:
    /// Create a GLSL program instance
    static GlslProgramPtr create()
    {
        return GlslProgramPtr(new GlslProgram());
    }

    /// Destructor
    virtual ~GlslProgram();

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

    /// Return the shader, if any, used to generate this program.
    ShaderPtr getShader() const
    {
        return _shader;
    }

    /// @}
    /// @name Program building
    /// @{

    /// Build shader program data from the source code set for
    /// each shader stage.
    ///
    /// An exception is thrown if the program cannot be built.
    /// The exception will contain a list of compilation errors.
    void build();

    /// Return true if built shader program data is present.
    bool hasBuiltData();

    // Clear built shader program data, if any.
    void clearBuiltData();

    /// @}
    /// @name Program introspection
    /// @{

    /// Structure to hold information about program inputs.
    /// The structure is populated by directly scanning the program so may not contain
    /// some inputs listed on any associated HwShader as those inputs may have been
    /// optimized out if they are unused.
    struct MX_RENDERGLSL_API Input
    {
        static int INVALID_OPENGL_TYPE;

        /// Program location. -1 means an invalid location
        int location;
        /// OpenGL type of the input. -1 means an invalid type
        int gltype;
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
            gltype(inputType),
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

    /// Bind the program.
    /// @return False if failed
    bool bind();

    /// Return true if the program has active attributes.
    bool hasActiveAttributes() const;

    /// Return true if a uniform with the given name is present.
    bool hasUniform(const string& name);

    /// Bind a value to the uniform with the given name.
    void bindUniform(const string& name, ConstValuePtr value, bool errorIfMissing = true);

    /// Bind attribute buffers to attribute inputs.
    /// A hardware buffer of the given attribute type is created and bound to the program locations
    /// for the input attribute.
    /// @param inputs Attribute inputs to bind to
    /// @param mesh Mesh containing streams to bind
    void bindAttribute(const GlslProgram::InputMap& inputs, MeshPtr mesh);

    /// Bind input geometry partition (indexing)
    void bindPartition(MeshPartitionPtr partition);

    /// Bind input geometry streams
    void bindMesh(MeshPtr mesh);

    /// Unbind any bound geometry
    void unbindGeometry();

    /// Bind any input textures
    void bindTextures(ImageHandlerPtr imageHandler);

    /// Bind lighting
    void bindLighting(LightHandlerPtr lightHandler, ImageHandlerPtr imageHandler);

    /// Bind view information
    void bindViewInformation(CameraPtr camera);

    /// Bind time and frame
    void bindTimeAndFrame(float time = 1.0f, float frame = 1.0f);

    /// Unbind the program.  Equivalent to binding no program
    void unbind() const;

    /// @}
    /// @name Utilities
    /// @{

    /// Print all uniforms to the given stream.
    void printUniforms(std::ostream& outputStream);

    /// Print all attributes to the given stream.
    void printAttributes(std::ostream& outputStream);

    /// @}

  public:
    static unsigned int UNDEFINED_OPENGL_RESOURCE_ID;
    static int UNDEFINED_OPENGL_PROGRAM_LOCATION;

  protected:
    GlslProgram();

    // Update a list of program input uniforms
    const InputMap& updateUniformsList();

    // Update a list of program input attributes
    const InputMap& updateAttributesList();

    // Utility to find a uniform value in an uniform list.
    // If uniform cannot be found a null pointer will be return.
    ValuePtr findUniformValue(const string& uniformName, const InputMap& uniformList);

    // Bind an individual texture to a program uniform location
    ImagePtr bindTexture(unsigned int uniformType, int uniformLocation, const FilePath& filePath,
                         ImageHandlerPtr imageHandler, const ImageSamplingProperties& imageProperties);

    // Utility to map a MaterialX type to an OpenGL type
    static int mapTypeToOpenGLType(const TypeDesc* type);

    // Bind a value to the uniform at the given location.
    void bindUniformLocation(int location, ConstValuePtr value);

  private:
    // Stages used to create program
    // Map of stage name and its source code
    StringMap _stages;

    // Generated program. A non-zero number indicates a valid shader program.
    unsigned int _programId;

    // List of program input uniforms
    InputMap _uniformList;
    // List of program input attributes
    InputMap _attributeList;

    // Hardware shader (if any) used for program creation
    ShaderPtr _shader;

    // Attribute buffer resource handles
    // for each attribute identifier in the program
    std::unordered_map<string, unsigned int> _attributeBufferIds;

    // Attribute indexing buffer handle
    std::map<MeshPartitionPtr, unsigned int> _indexBufferIds;

    // Attribute vertex array handle
    unsigned int _vertexArray;

    // Currently bound mesh
    MeshPtr _boundMesh;

    // Program texture map
    std::unordered_map<string, unsigned int> _programTextures;

    // Enabled vertex stream program locations
    std::set<int> _enabledStreamLocations;
};

MATERIALX_NAMESPACE_END

#endif

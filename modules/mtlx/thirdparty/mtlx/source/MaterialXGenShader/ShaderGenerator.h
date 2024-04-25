//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SHADERGENERATOR_H
#define MATERIALX_SHADERGENERATOR_H

/// @file
/// Base shader generator class

#include <MaterialXGenShader/Export.h>

#include <MaterialXGenShader/ColorManagementSystem.h>
#include <MaterialXGenShader/Factory.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Syntax.h>

#include <MaterialXFormat/File.h>

#include <MaterialXCore/Exception.h>

MATERIALX_NAMESPACE_BEGIN

/// @class ShaderGenerator
/// Base class for shader generators
/// All third-party shader generators should derive from this class.
/// Derived classes should use DECLARE_SHADER_GENERATOR / DEFINE_SHADER_GENERATOR
/// in their declaration / definition, and register with the Registry class.
class MX_GENSHADER_API ShaderGenerator
{
  public:
    /// Destructor
    virtual ~ShaderGenerator() { }

    /// Return the name of the target this generator is for.
    virtual const string& getTarget() const
    {
        return EMPTY_STRING;
    }

    /// Generate a shader starting from the given element, translating
    /// the element and all dependencies upstream into shader code.
    virtual ShaderPtr generate(const string&, ElementPtr, GenContext&) const
    {
        return nullptr;
    }

    /// Start a new scope using the given bracket type.
    virtual void emitScopeBegin(ShaderStage& stage, Syntax::Punctuation punc = Syntax::CURLY_BRACKETS) const;

    /// End the current scope.
    virtual void emitScopeEnd(ShaderStage& stage, bool semicolon = false, bool newline = true) const;

    /// Start a new line.
    virtual void emitLineBegin(ShaderStage& stage) const;

    /// End the current line.
    virtual void emitLineEnd(ShaderStage& stage, bool semicolon = true) const;

    /// Add a line break.
    virtual void emitLineBreak(ShaderStage& stage) const;

    /// Add a string.
    virtual void emitString(const string& str, ShaderStage& stage) const;

    /// Add a single line of code, optionally appending a semicolon.
    virtual void emitLine(const string& str, ShaderStage& stage, bool semicolon = true) const;

    /// Add a single line code comment.
    virtual void emitComment(const string& str, ShaderStage& stage) const;

    /// Add a block of code.
    virtual void emitBlock(const string& str, const FilePath& sourceFilename, GenContext& context, ShaderStage& stage) const;

    /// Add the contents of a standard library include file if not already present.
    /// The library file prefix of the given context, if any, will be prepended
    /// to the given filename.
    virtual void emitLibraryInclude(const FilePath& filename, GenContext& context, ShaderStage& stage) const;

    /// Add a value.
    template <typename T>
    void emitValue(const T& value, ShaderStage& stage) const
    {
        stage.addValue<T>(value);
    }

    /// Add the function definition for a single node.
    virtual void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const;

    /// Add all function definitions for a graph.
    virtual void emitFunctionDefinitions(const ShaderGraph& graph, GenContext& context, ShaderStage& stage) const;

    /// Add the function call for a single node.
    virtual void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const;
    [[deprecated]] virtual void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage, bool checkScope) const;

    /// Add all function calls for a graph. If a classification mask is given only functions for
    /// nodes matching this classification will be emitted.
    virtual void emitFunctionCalls(const ShaderGraph& graph, GenContext& context, ShaderStage& stage, uint32_t classification = 0u) const;

    /// Add function calls for nodes connected directly upstream from the given node.
    /// If a classification mask is given only functions for nodes matching this classification
    /// will be emitted.
    virtual void emitDependentFunctionCalls(const ShaderNode& node, GenContext& context, ShaderStage& stage, uint32_t classification = 0u) const;

    /// Emit code for starting a new function body.
    virtual void emitFunctionBodyBegin(const ShaderNode& node, GenContext& context, ShaderStage& stage, Syntax::Punctuation punc = Syntax::CURLY_BRACKETS) const;

    /// Emit code for ending a function body.
    virtual void emitFunctionBodyEnd(const ShaderNode& node, GenContext& context, ShaderStage& stage) const;

    /// Emit type definitions for all data types that needs it.
    virtual void emitTypeDefinitions(GenContext& context, ShaderStage& stage) const;

    /// Emit the connected variable name for an input,
    /// or constant value if the port is not connected
    virtual void emitInput(const ShaderInput* input, GenContext& context, ShaderStage& stage) const;

    /// Emit the output variable name for an output, optionally including it's type
    /// and default value assignment.
    virtual void emitOutput(const ShaderOutput* output, bool includeType, bool assignValue, GenContext& context, ShaderStage& stage) const;

    /// Emit definitions for all shader variables in a block.
    /// @param block Block to emit.
    /// @param qualifier Optional qualifier to add before the variable declaration.
    /// @param separator Separator to use between the declarations.
    /// @param context Context for generation.
    /// @param stage The stage to emit code into.
    /// @param assignValue If true the variables are initialized with their value.
    virtual void emitVariableDeclarations(const VariableBlock& block, const string& qualifier, const string& separator, GenContext& context, ShaderStage& stage,
                                          bool assignValue = true) const;

    /// Emit definition of a single shader variable.
    /// @param variable Shader port representing the variable.
    /// @param qualifier Optional qualifier to add before the variable declaration.
    /// @param context Context for generation.
    /// @param stage The stage to emit code into.
    /// @param assignValue If true the variable is initialized with its value.
    virtual void emitVariableDeclaration(const ShaderPort* variable, const string& qualifier, GenContext& context, ShaderStage& stage,
                                         bool assignValue = true) const;

    /// Return the closure contexts defined for the given node.
    virtual void getClosureContexts(const ShaderNode& node, vector<ClosureContext*>& cct) const;

    /// Return the result of an upstream connection or value for an input.
    virtual string getUpstreamResult(const ShaderInput* input, GenContext& context) const;

    /// Return the syntax object for the language used by the code generator
    const Syntax& getSyntax() const { return *_syntax; }

    /// Register a shader node implementation for a given implementation element name
    void registerImplementation(const string& name, CreatorFunction<ShaderNodeImpl> creator);

    /// Register a shader node implementation for a given set of implementation element names
    void registerImplementation(const StringVec& nameVec, CreatorFunction<ShaderNodeImpl> creator);

    /// Determine if a shader node implementation has been registered for a given implementation element name
    bool implementationRegistered(const string& name) const;

    /// Return a registered shader node implementation for the given nodedef.
    virtual ShaderNodeImplPtr getImplementation(const NodeDef& nodedef, GenContext& context) const;

    /// Sets the color management system
    void setColorManagementSystem(ColorManagementSystemPtr colorManagementSystem)
    {
        _colorManagementSystem = colorManagementSystem;
    }

    /// Returns the color management system
    ColorManagementSystemPtr getColorManagementSystem() const
    {
        return _colorManagementSystem;
    }

    /// Sets the unit system
    void setUnitSystem(UnitSystemPtr unitSystem)
    {
        _unitSystem = unitSystem;
    }

    /// Returns the unit system
    UnitSystemPtr getUnitSystem() const
    {
        return _unitSystem;
    }

    /// Return the map of token substitutions used by the generator.
    const StringMap& getTokenSubstitutions() const
    {
        return _tokenSubstitutions;
    }

    /// Register metadata that should be exported to the generated shaders.
    /// Supported metadata includes standard UI attributes like "uiname", "uifolder",
    /// "uimin", "uimax", etc.
    /// But it is also extendable by defining custom attributes using AttributeDefs.
    /// Any AttributeDef in the given document with exportable="true" will be
    /// exported as shader metadata when found on nodes during shader generation.
    /// Derived shader generators may override this method to change the registration.
    /// Applications must explicitly call this method before shader generation to enable
    /// export of metadata.
    virtual void registerShaderMetadata(const DocumentPtr& doc, GenContext& context) const;

  protected:
    /// Protected constructor
    ShaderGenerator(SyntaxPtr syntax);

    /// Create a new stage in a shader.
    virtual ShaderStagePtr createStage(const string& name, Shader& shader) const;

    /// Set function name for a stage.
    void setFunctionName(const string& functionName, ShaderStage& stage) const
    {
        stage.setFunctionName(functionName);
    }

    /// Replace tokens with identifiers according to the given substitutions map.
    void replaceTokens(const StringMap& substitutions, ShaderStage& stage) const;

    /// Create shader variables (e.g. uniforms, inputs and outputs) for
    /// nodes that require input data from the application.
    void createVariables(ShaderGraphPtr graph, GenContext& context, Shader& shader) const;

  protected:
    static const string T_FILE_TRANSFORM_UV;

    SyntaxPtr _syntax;
    Factory<ShaderNodeImpl> _implFactory;
    ColorManagementSystemPtr _colorManagementSystem;
    UnitSystemPtr _unitSystem;
    mutable StringMap _tokenSubstitutions;

    friend ShaderGraph;
};

/// @class ExceptionShaderGenError
/// An exception that is thrown when shader generation fails.
class MX_GENSHADER_API ExceptionShaderGenError : public Exception
{
  public:
    using Exception::Exception;
};

MATERIALX_NAMESPACE_END

#endif // MATERIALX_SHADERGENERATOR_H

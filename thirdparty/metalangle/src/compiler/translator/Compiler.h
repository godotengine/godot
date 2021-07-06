//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_COMPILER_H_
#define COMPILER_TRANSLATOR_COMPILER_H_

//
// Machine independent part of the compiler private objects
// sent as ShHandle to the driver.
//
// This should not be included by driver code.
//

#include <GLSLANG/ShaderVars.h>

#include "compiler/translator/BuiltInFunctionEmulator.h"
#include "compiler/translator/CallDAG.h"
#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/ExtensionBehavior.h"
#include "compiler/translator/HashNames.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/Pragma.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/ValidateAST.h"
#include "third_party/compiler/ArrayBoundsClamper.h"

namespace sh
{

class TCompiler;
class TParseContext;
#ifdef ANGLE_ENABLE_HLSL
class TranslatorHLSL;
#endif  // ANGLE_ENABLE_HLSL

//
// Helper function to check if the shader type is GLSL.
//
bool IsGLSL130OrNewer(ShShaderOutput output);
bool IsGLSL420OrNewer(ShShaderOutput output);
bool IsGLSL410OrOlder(ShShaderOutput output);

//
// Helper function to check if the invariant qualifier can be removed.
//
bool RemoveInvariant(sh::GLenum shaderType,
                     int shaderVersion,
                     ShShaderOutput outputType,
                     ShCompileOptions compileOptions);

//
// The base class used to back handles returned to the driver.
//
class TShHandleBase
{
  public:
    TShHandleBase();
    virtual ~TShHandleBase();
    virtual TCompiler *getAsCompiler() { return 0; }
#ifdef ANGLE_ENABLE_HLSL
    virtual TranslatorHLSL *getAsTranslatorHLSL() { return 0; }
#endif  // ANGLE_ENABLE_HLSL

  protected:
    // Memory allocator. Allocates and tracks memory required by the compiler.
    // Deallocates all memory when compiler is destructed.
    angle::PoolAllocator allocator;
};

//
// The base class for the machine dependent compiler to derive from
// for managing object code from the compile.
//
class TCompiler : public TShHandleBase
{
  public:
    TCompiler(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output);
    ~TCompiler() override;
    TCompiler *getAsCompiler() override { return this; }

    bool Init(const ShBuiltInResources &resources);

    // compileTreeForTesting should be used only when tests require access to
    // the AST. Users of this function need to manually manage the global pool
    // allocator. Returns nullptr whenever there are compilation errors.
    TIntermBlock *compileTreeForTesting(const char *const shaderStrings[],
                                        size_t numStrings,
                                        ShCompileOptions compileOptions);

    bool compile(const char *const shaderStrings[],
                 size_t numStrings,
                 ShCompileOptions compileOptions);

    // Get results of the last compilation.
    int getShaderVersion() const { return mShaderVersion; }
    TInfoSink &getInfoSink() { return mInfoSink; }

    bool isComputeShaderLocalSizeDeclared() const { return mComputeShaderLocalSizeDeclared; }
    const sh::WorkGroupSize &getComputeShaderLocalSize() const { return mComputeShaderLocalSize; }
    int getNumViews() const { return mNumViews; }

    // Clears the results from the previous compilation.
    void clearResults();

    const std::vector<sh::ShaderVariable> &getAttributes() const { return mAttributes; }
    const std::vector<sh::ShaderVariable> &getOutputVariables() const { return mOutputVariables; }
    const std::vector<sh::ShaderVariable> &getUniforms() const { return mUniforms; }
    const std::vector<sh::ShaderVariable> &getInputVaryings() const { return mInputVaryings; }
    const std::vector<sh::ShaderVariable> &getOutputVaryings() const { return mOutputVaryings; }
    const std::vector<sh::InterfaceBlock> &getInterfaceBlocks() const { return mInterfaceBlocks; }
    const std::vector<sh::InterfaceBlock> &getUniformBlocks() const { return mUniformBlocks; }
    const std::vector<sh::InterfaceBlock> &getShaderStorageBlocks() const
    {
        return mShaderStorageBlocks;
    }
    const std::vector<sh::InterfaceBlock> &getInBlocks() const { return mInBlocks; }

    ShHashFunction64 getHashFunction() const { return mResources.HashFunction; }
    NameMap &getNameMap() { return mNameMap; }
    TSymbolTable &getSymbolTable() { return mSymbolTable; }
    ShShaderSpec getShaderSpec() const { return mShaderSpec; }
    ShShaderOutput getOutputType() const { return mOutputType; }
    const std::string &getBuiltInResourcesString() const { return mBuiltInResourcesString; }

    bool shouldRunLoopAndIndexingValidation(ShCompileOptions compileOptions) const;

    // Get the resources set by InitBuiltInSymbolTable
    const ShBuiltInResources &getResources() const;

    int getGeometryShaderMaxVertices() const { return mGeometryShaderMaxVertices; }
    int getGeometryShaderInvocations() const { return mGeometryShaderInvocations; }
    TLayoutPrimitiveType getGeometryShaderInputPrimitiveType() const
    {
        return mGeometryShaderInputPrimitiveType;
    }
    TLayoutPrimitiveType getGeometryShaderOutputPrimitiveType() const
    {
        return mGeometryShaderOutputPrimitiveType;
    }

    sh::GLenum getShaderType() const { return mShaderType; }

    bool validateAST(TIntermNode *root);

  protected:
    // Add emulated functions to the built-in function emulator.
    virtual void initBuiltInFunctionEmulator(BuiltInFunctionEmulator *emu,
                                             ShCompileOptions compileOptions)
    {}
    // Translate to object code. May generate performance warnings through the diagnostics.
    ANGLE_NO_DISCARD virtual bool translate(TIntermBlock *root,
                                            ShCompileOptions compileOptions,
                                            PerformanceDiagnostics *perfDiagnostics) = 0;
    // Get built-in extensions with default behavior.
    const TExtensionBehavior &getExtensionBehavior() const;
    const char *getSourcePath() const;
    const TPragma &getPragma() const { return mPragma; }
    void writePragma(ShCompileOptions compileOptions);
    // Relies on collectVariables having been called.
    bool isVaryingDefined(const char *varyingName);

    const ArrayBoundsClamper &getArrayBoundsClamper() const;
    ShArrayIndexClampingStrategy getArrayIndexClampingStrategy() const;
    const BuiltInFunctionEmulator &getBuiltInFunctionEmulator() const;

    virtual bool shouldFlattenPragmaStdglInvariantAll() = 0;
    virtual bool shouldCollectVariables(ShCompileOptions compileOptions);

    bool wereVariablesCollected() const;
    std::vector<sh::ShaderVariable> mAttributes;
    std::vector<sh::ShaderVariable> mOutputVariables;
    std::vector<sh::ShaderVariable> mUniforms;
    std::vector<sh::ShaderVariable> mInputVaryings;
    std::vector<sh::ShaderVariable> mOutputVaryings;
    std::vector<sh::InterfaceBlock> mInterfaceBlocks;
    std::vector<sh::InterfaceBlock> mUniformBlocks;
    std::vector<sh::InterfaceBlock> mShaderStorageBlocks;
    std::vector<sh::InterfaceBlock> mInBlocks;

  private:
    // Initialize symbol-table with built-in symbols.
    bool initBuiltInSymbolTable(const ShBuiltInResources &resources);
    // Compute the string representation of the built-in resources
    void setResourceString();
    // Return false if the call depth is exceeded.
    bool checkCallDepth();
    // Insert statements to reference all members in unused uniform blocks with standard and shared
    // layout. This is to work around a Mac driver that treats unused standard/shared
    // uniform blocks as inactive.
    ANGLE_NO_DISCARD bool useAllMembersInUnusedStandardAndSharedBlocks(TIntermBlock *root);
    // Insert statements to initialize output variables in the beginning of main().
    // This is to avoid undefined behaviors.
    ANGLE_NO_DISCARD bool initializeOutputVariables(TIntermBlock *root);
    // Insert gl_Position = vec4(0,0,0,0) to the beginning of main().
    // It is to work around a Linux driver bug where missing this causes compile failure
    // while spec says it is allowed.
    // This function should only be applied to vertex shaders.
    ANGLE_NO_DISCARD bool initializeGLPosition(TIntermBlock *root);
    // Return true if the maximum expression complexity is below the limit.
    bool limitExpressionComplexity(TIntermBlock *root);
    // Creates the function call DAG for further analysis, returning false if there is a recursion
    bool initCallDag(TIntermNode *root);
    // Return false if "main" doesn't exist
    bool tagUsedFunctions();
    void internalTagUsedFunction(size_t index);

    void collectInterfaceBlocks();

    bool mVariablesCollected;

    bool mGLPositionInitialized;

    // Removes unused function declarations and prototypes from the AST
    class UnusedPredicate;
    void pruneUnusedFunctions(TIntermBlock *root);

    TIntermBlock *compileTreeImpl(const char *const shaderStrings[],
                                  size_t numStrings,
                                  const ShCompileOptions compileOptions);

    // Fetches and stores shader metadata that is not stored within the AST itself, such as shader
    // version.
    void setASTMetadata(const TParseContext &parseContext);

    // Check if shader version meets the requirement.
    bool checkShaderVersion(TParseContext *parseContext);

    // Does checks that need to be run after parsing is complete and returns true if they pass.
    bool checkAndSimplifyAST(TIntermBlock *root,
                             const TParseContext &parseContext,
                             ShCompileOptions compileOptions);

    sh::GLenum mShaderType;
    ShShaderSpec mShaderSpec;
    ShShaderOutput mOutputType;

    struct FunctionMetadata
    {
        FunctionMetadata() : used(false) {}
        bool used;
    };

    CallDAG mCallDag;
    std::vector<FunctionMetadata> mFunctionMetadata;

    ShBuiltInResources mResources;
    std::string mBuiltInResourcesString;

    // Built-in symbol table for the given language, spec, and resources.
    // It is preserved from compile-to-compile.
    TSymbolTable mSymbolTable;
    // Built-in extensions with default behavior.
    TExtensionBehavior mExtensionBehavior;

    ArrayBoundsClamper mArrayBoundsClamper;
    BuiltInFunctionEmulator mBuiltInFunctionEmulator;

    // Results of compilation.
    int mShaderVersion;
    TInfoSink mInfoSink;  // Output sink.
    TDiagnostics mDiagnostics;
    const char *mSourcePath;  // Path of source file or NULL

    // compute shader local group size
    bool mComputeShaderLocalSizeDeclared;
    sh::WorkGroupSize mComputeShaderLocalSize;

    // GL_OVR_multiview num_views.
    int mNumViews;

    // geometry shader parameters.
    int mGeometryShaderMaxVertices;
    int mGeometryShaderInvocations;
    TLayoutPrimitiveType mGeometryShaderInputPrimitiveType;
    TLayoutPrimitiveType mGeometryShaderOutputPrimitiveType;

    // name hashing.
    NameMap mNameMap;

    TPragma mPragma;

    // Track what should be validated given passes currently applied.
    ValidateASTOptions mValidateASTOptions;

    ShCompileOptions mCompileOptions;
};

//
// This is the interface between the machine independent code
// and the machine dependent code.
//
// The machine dependent code should derive from the classes
// above. Then Construct*() and Delete*() will create and
// destroy the machine dependent objects, which contain the
// above machine independent information.
//
TCompiler *ConstructCompiler(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output);
void DeleteCompiler(TCompiler *);

void EmitWorkGroupSizeGLSL(const TCompiler &, TInfoSinkBase &sink);
void EmitMultiviewGLSL(const TCompiler &, const ShCompileOptions &, TBehavior, TInfoSinkBase &sink);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_COMPILER_H_

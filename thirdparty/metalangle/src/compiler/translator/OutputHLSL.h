//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_OUTPUTHLSL_H_
#define COMPILER_TRANSLATOR_OUTPUTHLSL_H_

#include <list>
#include <map>
#include <stack>

#include "angle_gl.h"
#include "compiler/translator/ASTMetadataHLSL.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/FlagStd140Structs.h"
#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/ShaderStorageBlockOutputHLSL.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

class BuiltInFunctionEmulator;

namespace sh
{
class AtomicCounterFunctionHLSL;
class ImageFunctionHLSL;
class ResourcesHLSL;
class StructureHLSL;
class TextureFunctionHLSL;
class TSymbolTable;
class TVariable;
class UnfoldShortCircuit;

using ReferencedVariables = std::map<int, const TVariable *>;

class OutputHLSL : public TIntermTraverser
{
  public:
    OutputHLSL(sh::GLenum shaderType,
               ShShaderSpec shaderSpec,
               int shaderVersion,
               const TExtensionBehavior &extensionBehavior,
               const char *sourcePath,
               ShShaderOutput outputType,
               int numRenderTargets,
               int maxDualSourceDrawBuffers,
               const std::vector<ShaderVariable> &uniforms,
               ShCompileOptions compileOptions,
               sh::WorkGroupSize workGroupSize,
               TSymbolTable *symbolTable,
               PerformanceDiagnostics *perfDiagnostics,
               const std::vector<InterfaceBlock> &shaderStorageBlocks);

    ~OutputHLSL();

    void output(TIntermNode *treeRoot, TInfoSinkBase &objSink);

    const std::map<std::string, unsigned int> &getShaderStorageBlockRegisterMap() const;
    const std::map<std::string, unsigned int> &getUniformBlockRegisterMap() const;
    const std::map<std::string, unsigned int> &getUniformRegisterMap() const;
    unsigned int getReadonlyImage2DRegisterIndex() const;
    unsigned int getImage2DRegisterIndex() const;
    const std::set<std::string> &getUsedImage2DFunctionNames() const;

    TInfoSinkBase &getInfoSink()
    {
        ASSERT(!mInfoSinkStack.empty());
        return *mInfoSinkStack.top();
    }

  protected:
    friend class ShaderStorageBlockOutputHLSL;

    TString zeroInitializer(const TType &type) const;

    void writeReferencedAttributes(TInfoSinkBase &out) const;
    void writeReferencedVaryings(TInfoSinkBase &out) const;
    void header(TInfoSinkBase &out,
                const std::vector<MappedStruct> &std140Structs,
                const BuiltInFunctionEmulator *builtInFunctionEmulator) const;

    void writeFloat(TInfoSinkBase &out, float f);
    void writeSingleConstant(TInfoSinkBase &out, const TConstantUnion *const constUnion);
    const TConstantUnion *writeConstantUnionArray(TInfoSinkBase &out,
                                                  const TConstantUnion *const constUnion,
                                                  const size_t size);

    // Visit AST nodes and output their code to the body stream
    void visitSymbol(TIntermSymbol *) override;
    void visitConstantUnion(TIntermConstantUnion *) override;
    bool visitSwizzle(Visit visit, TIntermSwizzle *node) override;
    bool visitBinary(Visit visit, TIntermBinary *) override;
    bool visitUnary(Visit visit, TIntermUnary *) override;
    bool visitTernary(Visit visit, TIntermTernary *) override;
    bool visitIfElse(Visit visit, TIntermIfElse *) override;
    bool visitSwitch(Visit visit, TIntermSwitch *) override;
    bool visitCase(Visit visit, TIntermCase *) override;
    void visitFunctionPrototype(TIntermFunctionPrototype *node) override;
    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *) override;
    bool visitBlock(Visit visit, TIntermBlock *node) override;
    bool visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node) override;
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;
    bool visitLoop(Visit visit, TIntermLoop *) override;
    bool visitBranch(Visit visit, TIntermBranch *) override;

    bool handleExcessiveLoop(TInfoSinkBase &out, TIntermLoop *node);

    // Emit one of three strings depending on traverse phase. Called with literal strings so using
    // const char* instead of TString.
    void outputTriplet(TInfoSinkBase &out,
                       Visit visit,
                       const char *preString,
                       const char *inString,
                       const char *postString);
    void outputLineDirective(TInfoSinkBase &out, int line);
    void writeParameter(const TVariable *param, TInfoSinkBase &out);

    void outputConstructor(TInfoSinkBase &out, Visit visit, TIntermAggregate *node);
    const TConstantUnion *writeConstantUnion(TInfoSinkBase &out,
                                             const TType &type,
                                             const TConstantUnion *constUnion);

    void outputEqual(Visit visit, const TType &type, TOperator op, TInfoSinkBase &out);
    void outputAssign(Visit visit, const TType &type, TInfoSinkBase &out);

    void writeEmulatedFunctionTriplet(TInfoSinkBase &out, Visit visit, TOperator op);

    // Returns true if it found a 'same symbol' initializer (initializer that references the
    // variable it's initting)
    bool writeSameSymbolInitializer(TInfoSinkBase &out,
                                    TIntermSymbol *symbolNode,
                                    TIntermTyped *expression);
    // Returns true if variable initializer could be written using literal {} notation.
    bool writeConstantInitialization(TInfoSinkBase &out,
                                     TIntermSymbol *symbolNode,
                                     TIntermTyped *expression);

    void writeIfElse(TInfoSinkBase &out, TIntermIfElse *node);

    // Returns the function name
    TString addStructEqualityFunction(const TStructure &structure);
    TString addArrayEqualityFunction(const TType &type);
    TString addArrayAssignmentFunction(const TType &type);
    TString addArrayConstructIntoFunction(const TType &type);

    // Ensures if the type is a struct, the struct is defined
    void ensureStructDefined(const TType &type);

    bool shaderNeedsGenerateOutput() const;
    const char *generateOutputCall() const;

    sh::GLenum mShaderType;
    ShShaderSpec mShaderSpec;
    int mShaderVersion;
    const TExtensionBehavior &mExtensionBehavior;
    const char *mSourcePath;
    const ShShaderOutput mOutputType;
    ShCompileOptions mCompileOptions;

    bool mInsideFunction;
    bool mInsideMain;

    // Output streams
    TInfoSinkBase mHeader;
    TInfoSinkBase mBody;
    TInfoSinkBase mFooter;

    // A stack is useful when we want to traverse in the header, or in helper functions, but not
    // always write to the body. Instead use an InfoSink stack to keep our current state intact.
    // TODO (jmadill): Just passing an InfoSink in function parameters would be simpler.
    std::stack<TInfoSinkBase *> mInfoSinkStack;

    ReferencedVariables mReferencedUniforms;

    // Indexed by block id, not instance id.
    ReferencedInterfaceBlocks mReferencedUniformBlocks;

    ReferencedVariables mReferencedAttributes;
    ReferencedVariables mReferencedVaryings;
    ReferencedVariables mReferencedOutputVariables;

    StructureHLSL *mStructureHLSL;
    ResourcesHLSL *mResourcesHLSL;
    TextureFunctionHLSL *mTextureFunctionHLSL;
    ImageFunctionHLSL *mImageFunctionHLSL;
    AtomicCounterFunctionHLSL *mAtomicCounterFunctionHLSL;

    // Parameters determining what goes in the header output
    bool mUsesFragColor;
    bool mUsesFragData;
    bool mUsesDepthRange;
    bool mUsesFragCoord;
    bool mUsesPointCoord;
    bool mUsesFrontFacing;
    bool mUsesPointSize;
    bool mUsesInstanceID;
    bool mHasMultiviewExtensionEnabled;
    bool mUsesViewID;
    bool mUsesVertexID;
    bool mUsesFragDepth;
    bool mUsesNumWorkGroups;
    bool mUsesWorkGroupID;
    bool mUsesLocalInvocationID;
    bool mUsesGlobalInvocationID;
    bool mUsesLocalInvocationIndex;
    bool mUsesXor;
    bool mUsesDiscardRewriting;
    bool mUsesNestedBreak;
    bool mRequiresIEEEStrictCompiling;
    mutable bool mUseZeroArray;
    bool mUsesSecondaryColor;

    int mNumRenderTargets;
    int mMaxDualSourceDrawBuffers;

    int mUniqueIndex;  // For creating unique names

    CallDAG mCallDag;
    MetadataList mASTMetadataList;
    ASTMetadataHLSL *mCurrentFunctionMetadata;
    bool mOutputLod0Function;
    bool mInsideDiscontinuousLoop;
    int mNestedLoopDepth;

    TIntermSymbol *mExcessiveLoopIndex;

    TString structInitializerString(int indent, const TType &type, const TString &name) const;

    struct HelperFunction
    {
        TString functionName;
        TString functionDefinition;

        virtual ~HelperFunction() {}
    };

    // A list of all equality comparison functions. It's important to preserve the order at
    // which we add the functions, since nested structures call each other recursively, and
    // structure equality functions may need to call array equality functions and vice versa.
    // The ownership of the pointers is maintained by the type-specific arrays.
    std::vector<HelperFunction *> mEqualityFunctions;

    struct StructEqualityFunction : public HelperFunction
    {
        const TStructure *structure;
    };
    std::vector<StructEqualityFunction *> mStructEqualityFunctions;

    struct ArrayHelperFunction : public HelperFunction
    {
        TType type;
    };
    std::vector<ArrayHelperFunction *> mArrayEqualityFunctions;

    std::vector<ArrayHelperFunction> mArrayAssignmentFunctions;

    // The construct-into functions are functions that fill an N-element array passed as an out
    // parameter with the other N parameters of the function. This is used to work around that
    // arrays can't be return values in HLSL.
    std::vector<ArrayHelperFunction> mArrayConstructIntoFunctions;

    sh::WorkGroupSize mWorkGroupSize;

    PerformanceDiagnostics *mPerfDiagnostics;

  private:
    TString generateStructMapping(const std::vector<MappedStruct> &std140Structs) const;
    ImmutableString samplerNamePrefixFromStruct(TIntermTyped *node);
    bool ancestorEvaluatesToSamplerInStruct();
    // We need to do struct mapping when pass the struct to a function or copy the struct via
    // assignment.
    bool needStructMapping(TIntermTyped *node);

    ShaderStorageBlockOutputHLSL *mSSBOOutputHLSL;
    bool mNeedStructMapping;
};
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_OUTPUTHLSL_H_

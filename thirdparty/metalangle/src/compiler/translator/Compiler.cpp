//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/Compiler.h"

#include <sstream>

#include "angle_gl.h"
#include "common/utilities.h"
#include "compiler/translator/CallDAG.h"
#include "compiler/translator/CollectVariables.h"
#include "compiler/translator/Initialize.h"
#include "compiler/translator/IsASTDepthBelowLimit.h"
#include "compiler/translator/OutputTree.h"
#include "compiler/translator/ParseContext.h"
#include "compiler/translator/ValidateLimitations.h"
#include "compiler/translator/ValidateMaxParameters.h"
#include "compiler/translator/ValidateOutputs.h"
#include "compiler/translator/ValidateVaryingLocations.h"
#include "compiler/translator/VariablePacker.h"
#include "compiler/translator/tree_ops/AddAndTrueToLoopCondition.h"
#include "compiler/translator/tree_ops/ClampFragDepth.h"
#include "compiler/translator/tree_ops/ClampPointSize.h"
#include "compiler/translator/tree_ops/DeclareAndInitBuiltinsForInstancedMultiview.h"
#include "compiler/translator/tree_ops/DeferGlobalInitializers.h"
#include "compiler/translator/tree_ops/EmulateGLFragColorBroadcast.h"
#include "compiler/translator/tree_ops/EmulateMultiDrawShaderBuiltins.h"
#include "compiler/translator/tree_ops/EmulatePrecision.h"
#include "compiler/translator/tree_ops/FoldExpressions.h"
#include "compiler/translator/tree_ops/InitializeVariables.h"
#include "compiler/translator/tree_ops/PruneEmptyCases.h"
#include "compiler/translator/tree_ops/PruneNoOps.h"
#include "compiler/translator/tree_ops/RegenerateStructNames.h"
#include "compiler/translator/tree_ops/RemoveArrayLengthMethod.h"
#include "compiler/translator/tree_ops/RemoveInvariantDeclaration.h"
#include "compiler/translator/tree_ops/RemovePow.h"
#include "compiler/translator/tree_ops/RemoveUnreferencedVariables.h"
#include "compiler/translator/tree_ops/RewriteDoWhile.h"
#include "compiler/translator/tree_ops/RewriteRepeatedAssignToSwizzled.h"
#include "compiler/translator/tree_ops/ScalarizeVecAndMatConstructorArgs.h"
#include "compiler/translator/tree_ops/SeparateDeclarations.h"
#include "compiler/translator/tree_ops/SimplifyLoopConditions.h"
#include "compiler/translator/tree_ops/SplitSequenceOperator.h"
#include "compiler/translator/tree_ops/UnfoldShortCircuitAST.h"
#include "compiler/translator/tree_ops/UseInterfaceBlockFields.h"
#include "compiler/translator/tree_ops/VectorizeVectorScalarArithmetic.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/IntermNodePatternMatcher.h"
#include "compiler/translator/tree_util/ReplaceShadowingVariables.h"
#include "compiler/translator/util.h"
#include "third_party/compiler/ArrayBoundsClamper.h"

namespace sh
{

namespace
{

#if defined(ANGLE_ENABLE_FUZZER_CORPUS_OUTPUT)
void DumpFuzzerCase(char const *const *shaderStrings,
                    size_t numStrings,
                    uint32_t type,
                    uint32_t spec,
                    uint32_t output,
                    uint64_t options)
{
    static int fileIndex = 0;

    std::ostringstream o = sh::InitializeStream<std::ostringstream>();
    o << "corpus/" << fileIndex++ << ".sample";
    std::string s = o.str();

    // Must match the input format of the fuzzer
    FILE *f = fopen(s.c_str(), "w");
    fwrite(&type, sizeof(type), 1, f);
    fwrite(&spec, sizeof(spec), 1, f);
    fwrite(&output, sizeof(output), 1, f);
    fwrite(&options, sizeof(options), 1, f);

    char zero[128 - 20] = {0};
    fwrite(&zero, 128 - 20, 1, f);

    for (size_t i = 0; i < numStrings; i++)
    {
        fwrite(shaderStrings[i], sizeof(char), strlen(shaderStrings[i]), f);
    }
    fwrite(&zero, 1, 1, f);

    fclose(f);
}
#endif  // defined(ANGLE_ENABLE_FUZZER_CORPUS_OUTPUT)
}  // anonymous namespace

bool IsGLSL130OrNewer(ShShaderOutput output)
{
    return (output == SH_GLSL_130_OUTPUT || output == SH_GLSL_140_OUTPUT ||
            output == SH_GLSL_150_CORE_OUTPUT || output == SH_GLSL_330_CORE_OUTPUT ||
            output == SH_GLSL_400_CORE_OUTPUT || output == SH_GLSL_410_CORE_OUTPUT ||
            output == SH_GLSL_420_CORE_OUTPUT || output == SH_GLSL_430_CORE_OUTPUT ||
            output == SH_GLSL_440_CORE_OUTPUT || output == SH_GLSL_450_CORE_OUTPUT);
}

bool IsGLSL420OrNewer(ShShaderOutput output)
{
    return (output == SH_GLSL_420_CORE_OUTPUT || output == SH_GLSL_430_CORE_OUTPUT ||
            output == SH_GLSL_440_CORE_OUTPUT || output == SH_GLSL_450_CORE_OUTPUT);
}

bool IsGLSL410OrOlder(ShShaderOutput output)
{
    return (output == SH_GLSL_130_OUTPUT || output == SH_GLSL_140_OUTPUT ||
            output == SH_GLSL_150_CORE_OUTPUT || output == SH_GLSL_330_CORE_OUTPUT ||
            output == SH_GLSL_400_CORE_OUTPUT || output == SH_GLSL_410_CORE_OUTPUT);
}

bool RemoveInvariant(sh::GLenum shaderType,
                     int shaderVersion,
                     ShShaderOutput outputType,
                     ShCompileOptions compileOptions)
{
    if (shaderType == GL_FRAGMENT_SHADER && IsGLSL420OrNewer(outputType))
        return true;

    if ((compileOptions & SH_REMOVE_INVARIANT_AND_CENTROID_FOR_ESSL3) != 0 &&
        shaderVersion >= 300 && shaderType == GL_VERTEX_SHADER)
        return true;

    return false;
}

size_t GetGlobalMaxTokenSize(ShShaderSpec spec)
{
    // WebGL defines a max token length of 256, while ES2 leaves max token
    // size undefined. ES3 defines a max size of 1024 characters.
    switch (spec)
    {
        case SH_WEBGL_SPEC:
            return 256;
        default:
            return 1024;
    }
}

int GetMaxUniformVectorsForShaderType(GLenum shaderType, const ShBuiltInResources &resources)
{
    switch (shaderType)
    {
        case GL_VERTEX_SHADER:
            return resources.MaxVertexUniformVectors;
        case GL_FRAGMENT_SHADER:
            return resources.MaxFragmentUniformVectors;

        // TODO (jiawei.shao@intel.com): check if we need finer-grained component counting
        case GL_COMPUTE_SHADER:
            return resources.MaxComputeUniformComponents / 4;
        case GL_GEOMETRY_SHADER_EXT:
            return resources.MaxGeometryUniformComponents / 4;
        default:
            UNREACHABLE();
            return -1;
    }
}

namespace
{

class TScopedPoolAllocator
{
  public:
    TScopedPoolAllocator(angle::PoolAllocator *allocator) : mAllocator(allocator)
    {
        mAllocator->push();
        SetGlobalPoolAllocator(mAllocator);
    }
    ~TScopedPoolAllocator()
    {
        SetGlobalPoolAllocator(nullptr);
        mAllocator->pop();
    }

  private:
    angle::PoolAllocator *mAllocator;
};

class TScopedSymbolTableLevel
{
  public:
    TScopedSymbolTableLevel(TSymbolTable *table) : mTable(table)
    {
        ASSERT(mTable->isEmpty());
        mTable->push();
    }
    ~TScopedSymbolTableLevel()
    {
        while (!mTable->isEmpty())
            mTable->pop();
    }

  private:
    TSymbolTable *mTable;
};

int GetMaxShaderVersionForSpec(ShShaderSpec spec)
{
    switch (spec)
    {
        case SH_GLES2_SPEC:
        case SH_WEBGL_SPEC:
            return 100;
        case SH_GLES3_SPEC:
        case SH_WEBGL2_SPEC:
            return 300;
        case SH_GLES3_1_SPEC:
        case SH_WEBGL3_SPEC:
            return 310;
        case SH_GL_CORE_SPEC:
        case SH_GL_COMPATIBILITY_SPEC:
            return 460;
        default:
            UNREACHABLE();
            return 0;
    }
}

bool ValidateFragColorAndFragData(GLenum shaderType,
                                  int shaderVersion,
                                  const TSymbolTable &symbolTable,
                                  TDiagnostics *diagnostics)
{
    if (shaderVersion > 100 || shaderType != GL_FRAGMENT_SHADER)
    {
        return true;
    }

    bool usesFragColor = false;
    bool usesFragData  = false;
    // This validation is a bit stricter than the spec - it's only an error to write to
    // both FragData and FragColor. But because it's better not to have reads from undefined
    // variables, we always return an error if they are both referenced, rather than only if they
    // are written.
    if (symbolTable.isStaticallyUsed(*BuiltInVariable::gl_FragColor()) ||
        symbolTable.isStaticallyUsed(*BuiltInVariable::gl_SecondaryFragColorEXT()))
    {
        usesFragColor = true;
    }
    // Extension variables may not always be initialized (saves some time at symbol table init).
    bool secondaryFragDataUsed =
        symbolTable.gl_SecondaryFragDataEXT() != nullptr &&
        symbolTable.isStaticallyUsed(*symbolTable.gl_SecondaryFragDataEXT());
    if (symbolTable.isStaticallyUsed(*symbolTable.gl_FragData()) || secondaryFragDataUsed)
    {
        usesFragData = true;
    }
    if (usesFragColor && usesFragData)
    {
        const char *errorMessage = "cannot use both gl_FragData and gl_FragColor";
        if (symbolTable.isStaticallyUsed(*BuiltInVariable::gl_SecondaryFragColorEXT()) ||
            secondaryFragDataUsed)
        {
            errorMessage =
                "cannot use both output variable sets (gl_FragData, gl_SecondaryFragDataEXT)"
                " and (gl_FragColor, gl_SecondaryFragColorEXT)";
        }
        diagnostics->globalError(errorMessage);
        return false;
    }
    return true;
}

}  // namespace

TShHandleBase::TShHandleBase()
{
    allocator.push();
    SetGlobalPoolAllocator(&allocator);
}

TShHandleBase::~TShHandleBase()
{
    SetGlobalPoolAllocator(nullptr);
    allocator.popAll();
}

TCompiler::TCompiler(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output)
    : mVariablesCollected(false),
      mGLPositionInitialized(false),
      mShaderType(type),
      mShaderSpec(spec),
      mOutputType(output),
      mBuiltInFunctionEmulator(),
      mDiagnostics(mInfoSink.info),
      mSourcePath(nullptr),
      mComputeShaderLocalSizeDeclared(false),
      mComputeShaderLocalSize(1),
      mGeometryShaderMaxVertices(-1),
      mGeometryShaderInvocations(0),
      mGeometryShaderInputPrimitiveType(EptUndefined),
      mGeometryShaderOutputPrimitiveType(EptUndefined),
      mCompileOptions(0)
{}

TCompiler::~TCompiler() {}

bool TCompiler::shouldRunLoopAndIndexingValidation(ShCompileOptions compileOptions) const
{
    // If compiling an ESSL 1.00 shader for WebGL, or if its been requested through the API,
    // validate loop and indexing as well (to verify that the shader only uses minimal functionality
    // of ESSL 1.00 as in Appendix A of the spec).
    return (IsWebGLBasedSpec(mShaderSpec) && mShaderVersion == 100) ||
           (compileOptions & SH_VALIDATE_LOOP_INDEXING);
}

bool TCompiler::Init(const ShBuiltInResources &resources)
{
    SetGlobalPoolAllocator(&allocator);

    // Generate built-in symbol table.
    if (!initBuiltInSymbolTable(resources))
        return false;

    mResources = resources;
    setResourceString();

    InitExtensionBehavior(resources, mExtensionBehavior);
    mArrayBoundsClamper.SetClampingStrategy(resources.ArrayIndexClampingStrategy);
    return true;
}

TIntermBlock *TCompiler::compileTreeForTesting(const char *const shaderStrings[],
                                               size_t numStrings,
                                               ShCompileOptions compileOptions)
{
    return compileTreeImpl(shaderStrings, numStrings, compileOptions);
}

TIntermBlock *TCompiler::compileTreeImpl(const char *const shaderStrings[],
                                         size_t numStrings,
                                         const ShCompileOptions compileOptions)
{
    // Remember the compile options for helper functions such as validateAST.
    mCompileOptions = compileOptions;

    clearResults();

    ASSERT(numStrings > 0);
    ASSERT(GetGlobalPoolAllocator());

    // Reset the extension behavior for each compilation unit.
    ResetExtensionBehavior(mExtensionBehavior);

    // If gl_DrawID is not supported, remove it from the available extensions
    // Currently we only allow emulation of gl_DrawID
    const bool glDrawIDSupported = (compileOptions & SH_EMULATE_GL_DRAW_ID) != 0u;
    if (!glDrawIDSupported)
    {
        auto it = mExtensionBehavior.find(TExtension::ANGLE_multi_draw);
        if (it != mExtensionBehavior.end())
        {
            mExtensionBehavior.erase(it);
        }
    }

    const bool glBaseVertexBaseInstanceSupported =
        (compileOptions & SH_EMULATE_GL_BASE_VERTEX_BASE_INSTANCE) != 0u;
    if (!glBaseVertexBaseInstanceSupported)
    {
        auto it = mExtensionBehavior.find(TExtension::ANGLE_base_vertex_base_instance);
        if (it != mExtensionBehavior.end())
        {
            mExtensionBehavior.erase(it);
        }
    }

    // First string is path of source file if flag is set. The actual source follows.
    size_t firstSource = 0;
    if (compileOptions & SH_SOURCE_PATH)
    {
        mSourcePath = shaderStrings[0];
        ++firstSource;
    }

    TParseContext parseContext(mSymbolTable, mExtensionBehavior, mShaderType, mShaderSpec,
                               compileOptions, !IsDesktopGLSpec(mShaderSpec), &mDiagnostics,
                               getResources(), getOutputType());

    parseContext.setFragmentPrecisionHighOnESSL1(mResources.FragmentPrecisionHigh == 1);

    // We preserve symbols at the built-in level from compile-to-compile.
    // Start pushing the user-defined symbols at global level.
    TScopedSymbolTableLevel globalLevel(&mSymbolTable);
    ASSERT(mSymbolTable.atGlobalLevel());

    // Parse shader.
    if (PaParseStrings(numStrings - firstSource, &shaderStrings[firstSource], nullptr,
                       &parseContext) != 0)
    {
        return nullptr;
    }

    if (parseContext.getTreeRoot() == nullptr)
    {
        return nullptr;
    }

    setASTMetadata(parseContext);

    if (!checkShaderVersion(&parseContext))
    {
        return nullptr;
    }

    TIntermBlock *root = parseContext.getTreeRoot();
    if (!checkAndSimplifyAST(root, parseContext, compileOptions))
    {
        return nullptr;
    }

    return root;
}

bool TCompiler::checkShaderVersion(TParseContext *parseContext)
{
    if (GetMaxShaderVersionForSpec(mShaderSpec) < mShaderVersion)
    {
        mDiagnostics.globalError("unsupported shader version");
        return false;
    }

    ASSERT(parseContext);
    switch (mShaderType)
    {
        case GL_COMPUTE_SHADER:
            if (mShaderVersion < 310)
            {
                mDiagnostics.globalError("Compute shader is not supported in this shader version.");
                return false;
            }
            break;

        case GL_GEOMETRY_SHADER_EXT:
            if (mShaderVersion < 310)
            {
                mDiagnostics.globalError(
                    "Geometry shader is not supported in this shader version.");
                return false;
            }
            else
            {
                ASSERT(mShaderVersion == 310);
                if (!parseContext->checkCanUseExtension(sh::TSourceLoc(),
                                                        TExtension::EXT_geometry_shader))
                {
                    return false;
                }
            }
            break;

        default:
            break;
    }

    return true;
}

void TCompiler::setASTMetadata(const TParseContext &parseContext)
{
    mShaderVersion = parseContext.getShaderVersion();

    mPragma = parseContext.pragma();
    mSymbolTable.setGlobalInvariant(mPragma.stdgl.invariantAll);

    mComputeShaderLocalSizeDeclared = parseContext.isComputeShaderLocalSizeDeclared();
    mComputeShaderLocalSize         = parseContext.getComputeShaderLocalSize();

    mNumViews = parseContext.getNumViews();

    if (mShaderType == GL_GEOMETRY_SHADER_EXT)
    {
        mGeometryShaderInputPrimitiveType  = parseContext.getGeometryShaderInputPrimitiveType();
        mGeometryShaderOutputPrimitiveType = parseContext.getGeometryShaderOutputPrimitiveType();
        mGeometryShaderMaxVertices         = parseContext.getGeometryShaderMaxVertices();
        mGeometryShaderInvocations         = parseContext.getGeometryShaderInvocations();
    }
}

bool TCompiler::validateAST(TIntermNode *root)
{
    if ((mCompileOptions & SH_VALIDATE_AST) != 0)
    {
        bool valid = ValidateAST(root, &mDiagnostics, mValidateASTOptions);

        // In debug, assert validation.  In release, validation errors will be returned back to the
        // application as internal ANGLE errors.
        ASSERT(valid);

        return valid;
    }
    return true;
}

bool TCompiler::checkAndSimplifyAST(TIntermBlock *root,
                                    const TParseContext &parseContext,
                                    ShCompileOptions compileOptions)
{
    // Disallow expressions deemed too complex.
    if ((compileOptions & SH_LIMIT_EXPRESSION_COMPLEXITY) && !limitExpressionComplexity(root))
    {
        return false;
    }

    if (shouldRunLoopAndIndexingValidation(compileOptions) &&
        !ValidateLimitations(root, mShaderType, &mSymbolTable, &mDiagnostics))
    {
        return false;
    }

    if (!ValidateFragColorAndFragData(mShaderType, mShaderVersion, mSymbolTable, &mDiagnostics))
    {
        return false;
    }

    // Fold expressions that could not be folded before validation that was done as a part of
    // parsing.
    if (!FoldExpressions(this, root, &mDiagnostics))
    {
        return false;
    }
    // Folding should only be able to generate warnings.
    ASSERT(mDiagnostics.numErrors() == 0);

    // We prune no-ops to work around driver bugs and to keep AST processing and output simple.
    // The following kinds of no-ops are pruned:
    //   1. Empty declarations "int;".
    //   2. Literal statements: "1.0;". The ESSL output doesn't define a default precision
    //      for float, so float literal statements would end up with no precision which is
    //      invalid ESSL.
    // After this empty declarations are not allowed in the AST.
    if (!PruneNoOps(this, root, &mSymbolTable))
    {
        return false;
    }

    // Create the function DAG and check there is no recursion
    if (!initCallDag(root))
    {
        return false;
    }

    if ((compileOptions & SH_LIMIT_CALL_STACK_DEPTH) && !checkCallDepth())
    {
        return false;
    }

    // Checks which functions are used and if "main" exists
    mFunctionMetadata.clear();
    mFunctionMetadata.resize(mCallDag.size());
    if (!tagUsedFunctions())
    {
        return false;
    }

    if (!(compileOptions & SH_DONT_PRUNE_UNUSED_FUNCTIONS))
    {
        pruneUnusedFunctions(root);
    }
    if (IsSpecWithFunctionBodyNewScope(mShaderSpec, mShaderVersion))
    {
        if (!ReplaceShadowingVariables(this, root, &mSymbolTable))
        {
            return false;
        }
    }

    if (mShaderVersion >= 310 && !ValidateVaryingLocations(root, &mDiagnostics, mShaderType))
    {
        return false;
    }

    if (mShaderVersion >= 300 && mShaderType == GL_FRAGMENT_SHADER &&
        !ValidateOutputs(root, getExtensionBehavior(), mResources.MaxDrawBuffers, &mDiagnostics))
    {
        return false;
    }

    // Fail compilation if precision emulation not supported.
    if (getResources().WEBGL_debug_shader_precision && getPragma().debugShaderPrecision &&
        !EmulatePrecision::SupportedInLanguage(mOutputType))
    {
        mDiagnostics.globalError("Precision emulation not supported for this output type.");
        return false;
    }

    // Clamping uniform array bounds needs to happen after validateLimitations pass.
    if (compileOptions & SH_CLAMP_INDIRECT_ARRAY_BOUNDS)
    {
        mArrayBoundsClamper.MarkIndirectArrayBoundsForClamping(root);
    }

    if ((compileOptions & SH_INITIALIZE_BUILTINS_FOR_INSTANCED_MULTIVIEW) &&
        (parseContext.isExtensionEnabled(TExtension::OVR_multiview2) ||
         parseContext.isExtensionEnabled(TExtension::OVR_multiview)) &&
        getShaderType() != GL_COMPUTE_SHADER)
    {
        if (!DeclareAndInitBuiltinsForInstancedMultiview(
                this, root, mNumViews, mShaderType, compileOptions, mOutputType, &mSymbolTable))
        {
            return false;
        }
    }

    // This pass might emit short circuits so keep it before the short circuit unfolding
    if (compileOptions & SH_REWRITE_DO_WHILE_LOOPS)
    {
        if (!RewriteDoWhile(this, root, &mSymbolTable))
        {
            return false;
        }
    }

    if (compileOptions & SH_ADD_AND_TRUE_TO_LOOP_CONDITION)
    {
        if (!AddAndTrueToLoopCondition(this, root))
        {
            return false;
        }
    }

    if (compileOptions & SH_UNFOLD_SHORT_CIRCUIT)
    {
        if (!UnfoldShortCircuitAST(this, root))
        {
            return false;
        }
    }

    if (compileOptions & SH_REMOVE_POW_WITH_CONSTANT_EXPONENT)
    {
        if (!RemovePow(this, root, &mSymbolTable))
        {
            return false;
        }
    }

    if (compileOptions & SH_REGENERATE_STRUCT_NAMES)
    {
        RegenerateStructNames gen(&mSymbolTable);
        root->traverse(&gen);
        if (!validateAST(root))
        {
            return false;
        }
    }

    if (mShaderType == GL_VERTEX_SHADER &&
        IsExtensionEnabled(mExtensionBehavior, TExtension::ANGLE_multi_draw))
    {
        if ((compileOptions & SH_EMULATE_GL_DRAW_ID) != 0u)
        {
            if (!EmulateGLDrawID(this, root, &mSymbolTable, &mUniforms,
                                 shouldCollectVariables(compileOptions)))
            {
                return false;
            }
        }
    }

    if (mShaderType == GL_VERTEX_SHADER &&
        IsExtensionEnabled(mExtensionBehavior, TExtension::ANGLE_base_vertex_base_instance))
    {
        if ((compileOptions & SH_EMULATE_GL_BASE_VERTEX_BASE_INSTANCE) != 0u)
        {
            if (!EmulateGLBaseVertexBaseInstance(this, root, &mSymbolTable, &mUniforms,
                                                 shouldCollectVariables(compileOptions),
                                                 compileOptions & SH_ADD_BASE_VERTEX_TO_VERTEX_ID))
            {
                return false;
            }
        }
    }

    if (mShaderType == GL_FRAGMENT_SHADER && mShaderVersion == 100 && mResources.EXT_draw_buffers &&
        mResources.MaxDrawBuffers > 1 &&
        IsExtensionEnabled(mExtensionBehavior, TExtension::EXT_draw_buffers))
    {
        if (!EmulateGLFragColorBroadcast(this, root, mResources.MaxDrawBuffers, &mOutputVariables,
                                         &mSymbolTable, mShaderVersion))
        {
            return false;
        }
    }

    int simplifyScalarized = (compileOptions & SH_SCALARIZE_VEC_AND_MAT_CONSTRUCTOR_ARGS)
                                 ? IntermNodePatternMatcher::kScalarizedVecOrMatConstructor
                                 : 0;

    // Split multi declarations and remove calls to array length().
    // Note that SimplifyLoopConditions needs to be run before any other AST transformations
    // that may need to generate new statements from loop conditions or loop expressions.
    if (!SimplifyLoopConditions(this, root,
                                IntermNodePatternMatcher::kMultiDeclaration |
                                    IntermNodePatternMatcher::kArrayLengthMethod |
                                    simplifyScalarized,
                                &getSymbolTable()))
    {
        return false;
    }

    // Note that separate declarations need to be run before other AST transformations that
    // generate new statements from expressions.
    if (!SeparateDeclarations(this, root))
    {
        return false;
    }
    mValidateASTOptions.validateMultiDeclarations = true;

    if (!SplitSequenceOperator(this, root,
                               IntermNodePatternMatcher::kArrayLengthMethod | simplifyScalarized,
                               &getSymbolTable()))
    {
        return false;
    }

    if (!RemoveArrayLengthMethod(this, root))
    {
        return false;
    }

    if (!RemoveUnreferencedVariables(this, root, &mSymbolTable))
    {
        return false;
    }

    // In case the last case inside a switch statement is a certain type of no-op, GLSL compilers in
    // drivers may not accept it. In this case we clean up the dead code from the end of switch
    // statements. This is also required because PruneNoOps or RemoveUnreferencedVariables may have
    // left switch statements that only contained an empty declaration inside the final case in an
    // invalid state. Relies on that PruneNoOps and RemoveUnreferencedVariables have already been
    // run.
    if (!PruneEmptyCases(this, root))
    {
        return false;
    }

    // Built-in function emulation needs to happen after validateLimitations pass.
    // TODO(jmadill): Remove global pool allocator.
    GetGlobalPoolAllocator()->lock();
    initBuiltInFunctionEmulator(&mBuiltInFunctionEmulator, compileOptions);
    GetGlobalPoolAllocator()->unlock();
    mBuiltInFunctionEmulator.markBuiltInFunctionsForEmulation(root);

    bool highPrecisionSupported = mShaderVersion > 100 || mShaderType != GL_FRAGMENT_SHADER ||
                                  mResources.FragmentPrecisionHigh == 1;
    if (compileOptions & SH_SCALARIZE_VEC_AND_MAT_CONSTRUCTOR_ARGS)
    {
        if (!ScalarizeVecAndMatConstructorArgs(this, root, mShaderType, highPrecisionSupported,
                                               &mSymbolTable))
        {
            return false;
        }
    }

    if (shouldCollectVariables(compileOptions))
    {
        ASSERT(!mVariablesCollected);
        CollectVariables(root, &mAttributes, &mOutputVariables, &mUniforms, &mInputVaryings,
                         &mOutputVaryings, &mUniformBlocks, &mShaderStorageBlocks, &mInBlocks,
                         mResources.HashFunction, &mSymbolTable, mShaderType, mExtensionBehavior);
        collectInterfaceBlocks();
        mVariablesCollected = true;
        if (compileOptions & SH_USE_UNUSED_STANDARD_SHARED_BLOCKS)
        {
            if (!useAllMembersInUnusedStandardAndSharedBlocks(root))
            {
                return false;
            }
        }
        if (compileOptions & SH_ENFORCE_PACKING_RESTRICTIONS)
        {
            int maxUniformVectors = GetMaxUniformVectorsForShaderType(mShaderType, mResources);
            // Returns true if, after applying the packing rules in the GLSL ES 1.00.17 spec
            // Appendix A, section 7, the shader does not use too many uniforms.
            if (!CheckVariablesInPackingLimits(maxUniformVectors, mUniforms))
            {
                mDiagnostics.globalError("too many uniforms");
                return false;
            }
        }
        if ((compileOptions & SH_INIT_OUTPUT_VARIABLES) && (mShaderType != GL_COMPUTE_SHADER))
        {
            if (!initializeOutputVariables(root))
            {
                return false;
            }
        }
    }

    // Removing invariant declarations must be done after collecting variables.
    // Otherwise, built-in invariant declarations don't apply.
    if (RemoveInvariant(mShaderType, mShaderVersion, mOutputType, compileOptions))
    {
        if (!RemoveInvariantDeclaration(this, root))
        {
            return false;
        }
    }

    // gl_Position is always written in compatibility output mode.
    // It may have been already initialized among other output variables, in that case we don't
    // need to initialize it twice.
    if (mShaderType == GL_VERTEX_SHADER && !mGLPositionInitialized &&
        ((compileOptions & SH_INIT_GL_POSITION) || (mOutputType == SH_GLSL_COMPATIBILITY_OUTPUT)))
    {
        if (!initializeGLPosition(root))
        {
            return false;
        }
        mGLPositionInitialized = true;
    }

    // DeferGlobalInitializers needs to be run before other AST transformations that generate new
    // statements from expressions. But it's fine to run DeferGlobalInitializers after the above
    // SplitSequenceOperator and RemoveArrayLengthMethod since they only have an effect on the AST
    // on ESSL >= 3.00, and the initializers that need to be deferred can only exist in ESSL < 3.00.
    bool initializeLocalsAndGlobals =
        (compileOptions & SH_INITIALIZE_UNINITIALIZED_LOCALS) && !IsOutputHLSL(getOutputType());
    bool canUseLoopsToInitialize = !(compileOptions & SH_DONT_USE_LOOPS_TO_INITIALIZE_VARIABLES);
    if (!DeferGlobalInitializers(this, root, initializeLocalsAndGlobals, canUseLoopsToInitialize,
                                 highPrecisionSupported, &mSymbolTable))
    {
        return false;
    }

    if (initializeLocalsAndGlobals)
    {
        // Initialize uninitialized local variables.
        // In some cases initializing can generate extra statements in the parent block, such as
        // when initializing nameless structs or initializing arrays in ESSL 1.00. In that case
        // we need to first simplify loop conditions. We've already separated declarations
        // earlier, which is also required. If we don't follow the Appendix A limitations, loop
        // init statements can declare arrays or nameless structs and have multiple
        // declarations.

        if (!shouldRunLoopAndIndexingValidation(compileOptions))
        {
            if (!SimplifyLoopConditions(this, root,
                                        IntermNodePatternMatcher::kArrayDeclaration |
                                            IntermNodePatternMatcher::kNamelessStructDeclaration,
                                        &getSymbolTable()))
            {
                return false;
            }
        }

        if (!InitializeUninitializedLocals(this, root, getShaderVersion(), canUseLoopsToInitialize,
                                           highPrecisionSupported, &getSymbolTable()))
        {
            return false;
        }
    }

    if (getShaderType() == GL_VERTEX_SHADER && (compileOptions & SH_CLAMP_POINT_SIZE))
    {
        if (!ClampPointSize(this, root, mResources.MaxPointSize, &getSymbolTable()))
        {
            return false;
        }
    }

    if (getShaderType() == GL_FRAGMENT_SHADER && (compileOptions & SH_CLAMP_FRAG_DEPTH))
    {
        if (!ClampFragDepth(this, root, &getSymbolTable()))
        {
            return false;
        }
    }

    if (compileOptions & SH_REWRITE_REPEATED_ASSIGN_TO_SWIZZLED)
    {
        if (!sh::RewriteRepeatedAssignToSwizzled(this, root))
        {
            return false;
        }
    }

    if (compileOptions & SH_REWRITE_VECTOR_SCALAR_ARITHMETIC)
    {
        if (!VectorizeVectorScalarArithmetic(this, root, &getSymbolTable()))
        {
            return false;
        }
    }

    return true;
}

bool TCompiler::compile(const char *const shaderStrings[],
                        size_t numStrings,
                        ShCompileOptions compileOptionsIn)
{
#if defined(ANGLE_ENABLE_FUZZER_CORPUS_OUTPUT)
    DumpFuzzerCase(shaderStrings, numStrings, mShaderType, mShaderSpec, mOutputType,
                   compileOptionsIn);
#endif  // defined(ANGLE_ENABLE_FUZZER_CORPUS_OUTPUT)

    if (numStrings == 0)
        return true;

    ShCompileOptions compileOptions = compileOptionsIn;

    // Apply key workarounds.
    if (shouldFlattenPragmaStdglInvariantAll())
    {
        // This should be harmless to do in all cases, but for the moment, do it only conditionally.
        compileOptions |= SH_FLATTEN_PRAGMA_STDGL_INVARIANT_ALL;
    }

    TScopedPoolAllocator scopedAlloc(&allocator);
    TIntermBlock *root = compileTreeImpl(shaderStrings, numStrings, compileOptions);

    if (root)
    {
        if (compileOptions & SH_INTERMEDIATE_TREE)
        {
            OutputTree(root, mInfoSink.info);
        }

        if (compileOptions & SH_OBJECT_CODE)
        {
            PerformanceDiagnostics perfDiagnostics(&mDiagnostics);
            if (!translate(root, compileOptions, &perfDiagnostics))
            {
                return false;
            }
        }

        if (mShaderType == GL_VERTEX_SHADER)
        {
            bool lookForDrawID =
                IsExtensionEnabled(mExtensionBehavior, TExtension::ANGLE_multi_draw) &&
                ((compileOptions & SH_EMULATE_GL_DRAW_ID) != 0u);
            bool lookForBaseVertexBaseInstance =
                IsExtensionEnabled(mExtensionBehavior,
                                   TExtension::ANGLE_base_vertex_base_instance) &&
                ((compileOptions & SH_EMULATE_GL_BASE_VERTEX_BASE_INSTANCE) != 0u);

            if (lookForDrawID || lookForBaseVertexBaseInstance)
            {
                for (auto &uniform : mUniforms)
                {
                    if (lookForDrawID && uniform.name == "angle_DrawID" &&
                        uniform.mappedName == "angle_DrawID")
                    {
                        uniform.name = "gl_DrawID";
                    }
                    else if (lookForBaseVertexBaseInstance && uniform.name == "angle_BaseVertex" &&
                             uniform.mappedName == "angle_BaseVertex")
                    {
                        uniform.name = "gl_BaseVertex";
                    }
                    else if (lookForBaseVertexBaseInstance &&
                             uniform.name == "angle_BaseInstance" &&
                             uniform.mappedName == "angle_BaseInstance")
                    {
                        uniform.name = "gl_BaseInstance";
                    }
                }
            }
        }

        // The IntermNode tree doesn't need to be deleted here, since the
        // memory will be freed in a big chunk by the PoolAllocator.
        return true;
    }
    return false;
}

bool TCompiler::initBuiltInSymbolTable(const ShBuiltInResources &resources)
{
    if (resources.MaxDrawBuffers < 1)
    {
        return false;
    }
    if (resources.EXT_blend_func_extended && resources.MaxDualSourceDrawBuffers < 1)
    {
        return false;
    }

    mSymbolTable.initializeBuiltIns(mShaderType, mShaderSpec, resources);

    return true;
}

void TCompiler::setResourceString()
{
    std::ostringstream strstream = sh::InitializeStream<std::ostringstream>();

    // clang-format off
    strstream << ":MaxVertexAttribs:" << mResources.MaxVertexAttribs
        << ":MaxVertexUniformVectors:" << mResources.MaxVertexUniformVectors
        << ":MaxVaryingVectors:" << mResources.MaxVaryingVectors
        << ":MaxVertexTextureImageUnits:" << mResources.MaxVertexTextureImageUnits
        << ":MaxCombinedTextureImageUnits:" << mResources.MaxCombinedTextureImageUnits
        << ":MaxTextureImageUnits:" << mResources.MaxTextureImageUnits
        << ":MaxFragmentUniformVectors:" << mResources.MaxFragmentUniformVectors
        << ":MaxDrawBuffers:" << mResources.MaxDrawBuffers
        << ":OES_standard_derivatives:" << mResources.OES_standard_derivatives
        << ":OES_EGL_image_external:" << mResources.OES_EGL_image_external
        << ":OES_EGL_image_external_essl3:" << mResources.OES_EGL_image_external_essl3
        << ":NV_EGL_stream_consumer_external:" << mResources.NV_EGL_stream_consumer_external
        << ":ARB_texture_rectangle:" << mResources.ARB_texture_rectangle
        << ":EXT_draw_buffers:" << mResources.EXT_draw_buffers
        << ":FragmentPrecisionHigh:" << mResources.FragmentPrecisionHigh
        << ":MaxExpressionComplexity:" << mResources.MaxExpressionComplexity
        << ":MaxCallStackDepth:" << mResources.MaxCallStackDepth
        << ":MaxFunctionParameters:" << mResources.MaxFunctionParameters
        << ":EXT_blend_func_extended:" << mResources.EXT_blend_func_extended
        << ":EXT_frag_depth:" << mResources.EXT_frag_depth
        << ":EXT_shader_texture_lod:" << mResources.EXT_shader_texture_lod
        << ":EXT_shader_framebuffer_fetch:" << mResources.EXT_shader_framebuffer_fetch
        << ":NV_shader_framebuffer_fetch:" << mResources.NV_shader_framebuffer_fetch
        << ":ARM_shader_framebuffer_fetch:" << mResources.ARM_shader_framebuffer_fetch
        << ":OVR_multiview2:" << mResources.OVR_multiview2
        << ":OVR_multiview:" << mResources.OVR_multiview
        << ":EXT_YUV_target:" << mResources.EXT_YUV_target
        << ":EXT_geometry_shader:" << mResources.EXT_geometry_shader
        << ":OES_texture_3D:" << mResources.OES_texture_3D
        << ":MaxVertexOutputVectors:" << mResources.MaxVertexOutputVectors
        << ":MaxFragmentInputVectors:" << mResources.MaxFragmentInputVectors
        << ":MinProgramTexelOffset:" << mResources.MinProgramTexelOffset
        << ":MaxProgramTexelOffset:" << mResources.MaxProgramTexelOffset
        << ":MaxDualSourceDrawBuffers:" << mResources.MaxDualSourceDrawBuffers
        << ":MaxViewsOVR:" << mResources.MaxViewsOVR
        << ":NV_draw_buffers:" << mResources.NV_draw_buffers
        << ":WEBGL_debug_shader_precision:" << mResources.WEBGL_debug_shader_precision
        << ":ANGLE_multi_draw:" << mResources.ANGLE_multi_draw
        << ":ANGLE_base_vertex_base_instance:" << mResources.ANGLE_base_vertex_base_instance
        << ":APPLE_clip_distance:" << mResources.APPLE_clip_distance
        << ":MinProgramTextureGatherOffset:" << mResources.MinProgramTextureGatherOffset
        << ":MaxProgramTextureGatherOffset:" << mResources.MaxProgramTextureGatherOffset
        << ":MaxImageUnits:" << mResources.MaxImageUnits
        << ":MaxVertexImageUniforms:" << mResources.MaxVertexImageUniforms
        << ":MaxFragmentImageUniforms:" << mResources.MaxFragmentImageUniforms
        << ":MaxComputeImageUniforms:" << mResources.MaxComputeImageUniforms
        << ":MaxCombinedImageUniforms:" << mResources.MaxCombinedImageUniforms
        << ":MaxCombinedShaderOutputResources:" << mResources.MaxCombinedShaderOutputResources
        << ":MaxComputeWorkGroupCountX:" << mResources.MaxComputeWorkGroupCount[0]
        << ":MaxComputeWorkGroupCountY:" << mResources.MaxComputeWorkGroupCount[1]
        << ":MaxComputeWorkGroupCountZ:" << mResources.MaxComputeWorkGroupCount[2]
        << ":MaxComputeWorkGroupSizeX:" << mResources.MaxComputeWorkGroupSize[0]
        << ":MaxComputeWorkGroupSizeY:" << mResources.MaxComputeWorkGroupSize[1]
        << ":MaxComputeWorkGroupSizeZ:" << mResources.MaxComputeWorkGroupSize[2]
        << ":MaxComputeUniformComponents:" << mResources.MaxComputeUniformComponents
        << ":MaxComputeTextureImageUnits:" << mResources.MaxComputeTextureImageUnits
        << ":MaxComputeAtomicCounters:" << mResources.MaxComputeAtomicCounters
        << ":MaxComputeAtomicCounterBuffers:" << mResources.MaxComputeAtomicCounterBuffers
        << ":MaxVertexAtomicCounters:" << mResources.MaxVertexAtomicCounters
        << ":MaxFragmentAtomicCounters:" << mResources.MaxFragmentAtomicCounters
        << ":MaxCombinedAtomicCounters:" << mResources.MaxCombinedAtomicCounters
        << ":MaxAtomicCounterBindings:" << mResources.MaxAtomicCounterBindings
        << ":MaxVertexAtomicCounterBuffers:" << mResources.MaxVertexAtomicCounterBuffers
        << ":MaxFragmentAtomicCounterBuffers:" << mResources.MaxFragmentAtomicCounterBuffers
        << ":MaxCombinedAtomicCounterBuffers:" << mResources.MaxCombinedAtomicCounterBuffers
        << ":MaxAtomicCounterBufferSize:" << mResources.MaxAtomicCounterBufferSize
        << ":MaxGeometryUniformComponents:" << mResources.MaxGeometryUniformComponents
        << ":MaxGeometryUniformBlocks:" << mResources.MaxGeometryUniformBlocks
        << ":MaxGeometryInputComponents:" << mResources.MaxGeometryInputComponents
        << ":MaxGeometryOutputComponents:" << mResources.MaxGeometryOutputComponents
        << ":MaxGeometryOutputVertices:" << mResources.MaxGeometryOutputVertices
        << ":MaxGeometryTotalOutputComponents:" << mResources.MaxGeometryTotalOutputComponents
        << ":MaxGeometryTextureImageUnits:" << mResources.MaxGeometryTextureImageUnits
        << ":MaxGeometryAtomicCounterBuffers:" << mResources.MaxGeometryAtomicCounterBuffers
        << ":MaxGeometryAtomicCounters:" << mResources.MaxGeometryAtomicCounters
        << ":MaxGeometryShaderStorageBlocks:" << mResources.MaxGeometryShaderStorageBlocks
        << ":MaxGeometryShaderInvocations:" << mResources.MaxGeometryShaderInvocations
        << ":MaxGeometryImageUniforms:" << mResources.MaxGeometryImageUniforms
        << ":MaxClipDistances" << mResources.MaxClipDistances;
    // clang-format on

    mBuiltInResourcesString = strstream.str();
}

void TCompiler::collectInterfaceBlocks()
{
    ASSERT(mInterfaceBlocks.empty());
    mInterfaceBlocks.reserve(mUniformBlocks.size() + mShaderStorageBlocks.size() +
                             mInBlocks.size());
    mInterfaceBlocks.insert(mInterfaceBlocks.end(), mUniformBlocks.begin(), mUniformBlocks.end());
    mInterfaceBlocks.insert(mInterfaceBlocks.end(), mShaderStorageBlocks.begin(),
                            mShaderStorageBlocks.end());
    mInterfaceBlocks.insert(mInterfaceBlocks.end(), mInBlocks.begin(), mInBlocks.end());
}

void TCompiler::clearResults()
{
    mArrayBoundsClamper.Cleanup();
    mInfoSink.info.erase();
    mInfoSink.obj.erase();
    mInfoSink.debug.erase();
    mDiagnostics.resetErrorCount();

    mAttributes.clear();
    mOutputVariables.clear();
    mUniforms.clear();
    mInputVaryings.clear();
    mOutputVaryings.clear();
    mInterfaceBlocks.clear();
    mUniformBlocks.clear();
    mShaderStorageBlocks.clear();
    mInBlocks.clear();
    mVariablesCollected    = false;
    mGLPositionInitialized = false;

    mNumViews = -1;

    mGeometryShaderInputPrimitiveType  = EptUndefined;
    mGeometryShaderOutputPrimitiveType = EptUndefined;
    mGeometryShaderInvocations         = 0;
    mGeometryShaderMaxVertices         = -1;

    mBuiltInFunctionEmulator.cleanup();

    mNameMap.clear();

    mSourcePath = nullptr;

    mSymbolTable.clearCompilationResults();
}

bool TCompiler::initCallDag(TIntermNode *root)
{
    mCallDag.clear();

    switch (mCallDag.init(root, &mDiagnostics))
    {
        case CallDAG::INITDAG_SUCCESS:
            return true;
        case CallDAG::INITDAG_RECURSION:
        case CallDAG::INITDAG_UNDEFINED:
            // Error message has already been written out.
            ASSERT(mDiagnostics.numErrors() > 0);
            return false;
    }

    UNREACHABLE();
    return true;
}

bool TCompiler::checkCallDepth()
{
    std::vector<int> depths(mCallDag.size());

    for (size_t i = 0; i < mCallDag.size(); i++)
    {
        int depth                     = 0;
        const CallDAG::Record &record = mCallDag.getRecordFromIndex(i);

        for (const int &calleeIndex : record.callees)
        {
            depth = std::max(depth, depths[calleeIndex] + 1);
        }

        depths[i] = depth;

        if (depth >= mResources.MaxCallStackDepth)
        {
            // Trace back the function chain to have a meaningful info log.
            std::stringstream errorStream = sh::InitializeStream<std::stringstream>();
            errorStream << "Call stack too deep (larger than " << mResources.MaxCallStackDepth
                        << ") with the following call chain: "
                        << record.node->getFunction()->name();

            int currentFunction = static_cast<int>(i);
            int currentDepth    = depth;

            while (currentFunction != -1)
            {
                errorStream
                    << " -> "
                    << mCallDag.getRecordFromIndex(currentFunction).node->getFunction()->name();

                int nextFunction = -1;
                for (const int &calleeIndex : mCallDag.getRecordFromIndex(currentFunction).callees)
                {
                    if (depths[calleeIndex] == currentDepth - 1)
                    {
                        currentDepth--;
                        nextFunction = calleeIndex;
                    }
                }

                currentFunction = nextFunction;
            }

            std::string errorStr = errorStream.str();
            mDiagnostics.globalError(errorStr.c_str());

            return false;
        }
    }

    return true;
}

bool TCompiler::tagUsedFunctions()
{
    // Search from main, starting from the end of the DAG as it usually is the root.
    for (size_t i = mCallDag.size(); i-- > 0;)
    {
        if (mCallDag.getRecordFromIndex(i).node->getFunction()->isMain())
        {
            internalTagUsedFunction(i);
            return true;
        }
    }

    mDiagnostics.globalError("Missing main()");
    return false;
}

void TCompiler::internalTagUsedFunction(size_t index)
{
    if (mFunctionMetadata[index].used)
    {
        return;
    }

    mFunctionMetadata[index].used = true;

    for (int calleeIndex : mCallDag.getRecordFromIndex(index).callees)
    {
        internalTagUsedFunction(calleeIndex);
    }
}

// A predicate for the stl that returns if a top-level node is unused
class TCompiler::UnusedPredicate
{
  public:
    UnusedPredicate(const CallDAG *callDag, const std::vector<FunctionMetadata> *metadatas)
        : mCallDag(callDag), mMetadatas(metadatas)
    {}

    bool operator()(TIntermNode *node)
    {
        const TIntermFunctionPrototype *asFunctionPrototype   = node->getAsFunctionPrototypeNode();
        const TIntermFunctionDefinition *asFunctionDefinition = node->getAsFunctionDefinition();

        const TFunction *func = nullptr;

        if (asFunctionDefinition)
        {
            func = asFunctionDefinition->getFunction();
        }
        else if (asFunctionPrototype)
        {
            func = asFunctionPrototype->getFunction();
        }
        if (func == nullptr)
        {
            return false;
        }

        size_t callDagIndex = mCallDag->findIndex(func->uniqueId());
        if (callDagIndex == CallDAG::InvalidIndex)
        {
            // This happens only for unimplemented prototypes which are thus unused
            ASSERT(asFunctionPrototype);
            return true;
        }

        ASSERT(callDagIndex < mMetadatas->size());
        return !(*mMetadatas)[callDagIndex].used;
    }

  private:
    const CallDAG *mCallDag;
    const std::vector<FunctionMetadata> *mMetadatas;
};

void TCompiler::pruneUnusedFunctions(TIntermBlock *root)
{
    UnusedPredicate isUnused(&mCallDag, &mFunctionMetadata);
    TIntermSequence *sequence = root->getSequence();

    if (!sequence->empty())
    {
        sequence->erase(std::remove_if(sequence->begin(), sequence->end(), isUnused),
                        sequence->end());
    }
}

bool TCompiler::limitExpressionComplexity(TIntermBlock *root)
{
    if (!IsASTDepthBelowLimit(root, mResources.MaxExpressionComplexity))
    {
        mDiagnostics.globalError("Expression too complex.");
        return false;
    }

    if (!ValidateMaxParameters(root, mResources.MaxFunctionParameters))
    {
        mDiagnostics.globalError("Function has too many parameters.");
        return false;
    }

    return true;
}

bool TCompiler::shouldCollectVariables(ShCompileOptions compileOptions)
{
    return (compileOptions & SH_VARIABLES) != 0;
}

bool TCompiler::wereVariablesCollected() const
{
    return mVariablesCollected;
}

bool TCompiler::initializeGLPosition(TIntermBlock *root)
{
    InitVariableList list;
    sh::ShaderVariable var(GL_FLOAT_VEC4);
    var.name = "gl_Position";
    list.push_back(var);
    return InitializeVariables(this, root, list, &mSymbolTable, mShaderVersion, mExtensionBehavior,
                               false, false);
}

bool TCompiler::useAllMembersInUnusedStandardAndSharedBlocks(TIntermBlock *root)
{
    sh::InterfaceBlockList list;

    for (const sh::InterfaceBlock &block : mUniformBlocks)
    {
        if (!block.staticUse &&
            (block.layout == sh::BLOCKLAYOUT_STD140 || block.layout == sh::BLOCKLAYOUT_SHARED))
        {
            list.push_back(block);
        }
    }

    return sh::UseInterfaceBlockFields(this, root, list, mSymbolTable);
}

bool TCompiler::initializeOutputVariables(TIntermBlock *root)
{
    InitVariableList list;
    if (mShaderType == GL_VERTEX_SHADER || mShaderType == GL_GEOMETRY_SHADER_EXT)
    {
        for (const sh::ShaderVariable &var : mOutputVaryings)
        {
            list.push_back(var);
            if (var.name == "gl_Position")
            {
                ASSERT(!mGLPositionInitialized);
                mGLPositionInitialized = true;
            }
        }
    }
    else
    {
        ASSERT(mShaderType == GL_FRAGMENT_SHADER);
        for (const sh::ShaderVariable &var : mOutputVariables)
        {
            list.push_back(var);
        }
    }
    return InitializeVariables(this, root, list, &mSymbolTable, mShaderVersion, mExtensionBehavior,
                               false, false);
}

const TExtensionBehavior &TCompiler::getExtensionBehavior() const
{
    return mExtensionBehavior;
}

const char *TCompiler::getSourcePath() const
{
    return mSourcePath;
}

const ShBuiltInResources &TCompiler::getResources() const
{
    return mResources;
}

const ArrayBoundsClamper &TCompiler::getArrayBoundsClamper() const
{
    return mArrayBoundsClamper;
}

ShArrayIndexClampingStrategy TCompiler::getArrayIndexClampingStrategy() const
{
    return mResources.ArrayIndexClampingStrategy;
}

const BuiltInFunctionEmulator &TCompiler::getBuiltInFunctionEmulator() const
{
    return mBuiltInFunctionEmulator;
}

void TCompiler::writePragma(ShCompileOptions compileOptions)
{
    if (!(compileOptions & SH_FLATTEN_PRAGMA_STDGL_INVARIANT_ALL))
    {
        TInfoSinkBase &sink = mInfoSink.obj;
        if (mPragma.stdgl.invariantAll)
            sink << "#pragma STDGL invariant(all)\n";
    }
}

bool TCompiler::isVaryingDefined(const char *varyingName)
{
    ASSERT(mVariablesCollected);
    for (size_t ii = 0; ii < mInputVaryings.size(); ++ii)
    {
        if (mInputVaryings[ii].name == varyingName)
        {
            return true;
        }
    }
    for (size_t ii = 0; ii < mOutputVaryings.size(); ++ii)
    {
        if (mOutputVaryings[ii].name == varyingName)
        {
            return true;
        }
    }

    return false;
}

void EmitWorkGroupSizeGLSL(const TCompiler &compiler, TInfoSinkBase &sink)
{
    if (compiler.isComputeShaderLocalSizeDeclared())
    {
        const sh::WorkGroupSize &localSize = compiler.getComputeShaderLocalSize();
        sink << "layout (local_size_x=" << localSize[0] << ", local_size_y=" << localSize[1]
             << ", local_size_z=" << localSize[2] << ") in;\n";
    }
}

void EmitMultiviewGLSL(const TCompiler &compiler,
                       const ShCompileOptions &compileOptions,
                       const TBehavior behavior,
                       TInfoSinkBase &sink)
{
    ASSERT(behavior != EBhUndefined);
    if (behavior == EBhDisable)
        return;

    const bool isVertexShader = (compiler.getShaderType() == GL_VERTEX_SHADER);
    if (compileOptions & SH_INITIALIZE_BUILTINS_FOR_INSTANCED_MULTIVIEW)
    {
        // Emit ARB_shader_viewport_layer_array/NV_viewport_array2 in a vertex shader if the
        // SH_SELECT_VIEW_IN_NV_GLSL_VERTEX_SHADER option is set and the
        // OVR_multiview(2) extension is requested.
        if (isVertexShader && (compileOptions & SH_SELECT_VIEW_IN_NV_GLSL_VERTEX_SHADER))
        {
            sink << "#if defined(GL_ARB_shader_viewport_layer_array)\n"
                 << "#extension GL_ARB_shader_viewport_layer_array : require\n"
                 << "#elif defined(GL_NV_viewport_array2)\n"
                 << "#extension GL_NV_viewport_array2 : require\n"
                 << "#endif\n";
        }
    }
    else
    {
        sink << "#extension GL_OVR_multiview2 : " << GetBehaviorString(behavior) << "\n";

        const auto &numViews = compiler.getNumViews();
        if (isVertexShader && numViews != -1)
        {
            sink << "layout(num_views=" << numViews << ") in;\n";
        }
    }
}

}  // namespace sh

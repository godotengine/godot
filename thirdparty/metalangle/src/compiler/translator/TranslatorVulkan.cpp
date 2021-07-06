//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// TranslatorVulkan:
//   A GLSL-based translator that outputs shaders that fit GL_KHR_vulkan_glsl.
//   The shaders are then fed into glslang to spit out SPIR-V (libANGLE-side).
//   See: https://www.khronos.org/registry/vulkan/specs/misc/GL_KHR_vulkan_glsl.txt
//

#include "compiler/translator/TranslatorVulkan.h"

#include "angle_gl.h"
#include "common/utilities.h"
#include "compiler/translator/BuiltinsWorkaroundGLSL.h"
#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/OutputVulkanGLSL.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/tree_ops/NameEmbeddedUniformStructs.h"
#include "compiler/translator/tree_ops/NameNamelessUniformBuffers.h"
#include "compiler/translator/tree_ops/RewriteAtomicCounters.h"
#include "compiler/translator/tree_ops/RewriteCubeMapSamplersAs2DArray.h"
#include "compiler/translator/tree_ops/RewriteDfdy.h"
#include "compiler/translator/tree_ops/RewriteRowMajorMatrices.h"
#include "compiler/translator/tree_ops/RewriteStructSamplers.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/FindFunction.h"
#include "compiler/translator/tree_util/FindMain.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/ReplaceClipDistanceVariable.h"
#include "compiler/translator/tree_util/ReplaceVariable.h"
#include "compiler/translator/tree_util/RunAtTheEndOfShader.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{
// This traverses nodes, find the struct ones and add their declarations to the sink. It also
// removes the nodes from the tree as it processes them.
class DeclareStructTypesTraverser : public TIntermTraverser
{
  public:
    explicit DeclareStructTypesTraverser(TOutputVulkanGLSL *outputVulkanGLSL)
        : TIntermTraverser(true, false, false), mOutputVulkanGLSL(outputVulkanGLSL)
    {}

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        ASSERT(visit == PreVisit);

        if (!mInGlobalScope)
        {
            return false;
        }

        const TIntermSequence &sequence = *(node->getSequence());
        TIntermTyped *declarator        = sequence.front()->getAsTyped();
        const TType &type               = declarator->getType();

        if (type.isStructSpecifier())
        {
            const TStructure *structure = type.getStruct();

            // Embedded structs should be parsed away by now.
            ASSERT(structure->symbolType() != SymbolType::Empty);
            mOutputVulkanGLSL->writeStructType(structure);

            TIntermSymbol *symbolNode = declarator->getAsSymbolNode();
            if (symbolNode && symbolNode->variable().symbolType() == SymbolType::Empty)
            {
                // Remove the struct specifier declaration from the tree so it isn't parsed again.
                TIntermSequence emptyReplacement;
                mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), node,
                                                emptyReplacement);
            }
        }

        return false;
    }

  private:
    TOutputVulkanGLSL *mOutputVulkanGLSL;
};

class DeclareDefaultUniformsTraverser : public TIntermTraverser
{
  public:
    DeclareDefaultUniformsTraverser(TInfoSinkBase *sink,
                                    ShHashFunction64 hashFunction,
                                    NameMap *nameMap)
        : TIntermTraverser(true, true, true),
          mSink(sink),
          mHashFunction(hashFunction),
          mNameMap(nameMap),
          mInDefaultUniform(false)
    {}

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        const TIntermSequence &sequence = *(node->getSequence());

        // TODO(jmadill): Compound declarations.
        ASSERT(sequence.size() == 1);

        TIntermTyped *variable = sequence.front()->getAsTyped();
        const TType &type      = variable->getType();
        bool isUniform         = type.getQualifier() == EvqUniform && !type.isInterfaceBlock() &&
                         !IsOpaqueType(type.getBasicType());

        if (visit == PreVisit)
        {
            if (isUniform)
            {
                (*mSink) << "    " << GetTypeName(type, mHashFunction, mNameMap) << " ";
                mInDefaultUniform = true;
            }
        }
        else if (visit == InVisit)
        {
            mInDefaultUniform = isUniform;
        }
        else if (visit == PostVisit)
        {
            if (isUniform)
            {
                (*mSink) << ";\n";

                // Remove the uniform declaration from the tree so it isn't parsed again.
                TIntermSequence emptyReplacement;
                mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), node,
                                                emptyReplacement);
            }

            mInDefaultUniform = false;
        }
        return true;
    }

    void visitSymbol(TIntermSymbol *symbol) override
    {
        if (mInDefaultUniform)
        {
            const ImmutableString &name = symbol->variable().name();
            ASSERT(!name.beginsWith("gl_"));
            (*mSink) << HashName(&symbol->variable(), mHashFunction, mNameMap)
                     << ArrayString(symbol->getType());
        }
    }

  private:
    TInfoSinkBase *mSink;
    ShHashFunction64 mHashFunction;
    NameMap *mNameMap;
    bool mInDefaultUniform;
};

constexpr ImmutableString kFlippedPointCoordName    = ImmutableString("flippedPointCoord");
constexpr ImmutableString kFlippedFragCoordName     = ImmutableString("flippedFragCoord");
constexpr ImmutableString kEmulatedDepthRangeParams = ImmutableString("ANGLEDepthRangeParams");
constexpr ImmutableString kUniformsBlockName        = ImmutableString("ANGLEUniformBlock");
constexpr ImmutableString kUniformsVarName          = ImmutableString("ANGLEUniforms");

constexpr const char kViewport[]             = "viewport";
constexpr const char kHalfRenderAreaHeight[] = "halfRenderAreaHeight";
constexpr const char kViewportYScale[]       = "viewportYScale";
constexpr const char kNegViewportYScale[]    = "negViewportYScale";
constexpr const char kXfbActiveUnpaused[]    = "xfbActiveUnpaused";
constexpr const char kXfbVerticesPerDraw[]   = "xfbVerticesPerDraw";
constexpr const char kXfbBufferOffsets[]     = "xfbBufferOffsets";
constexpr const char kAcbBufferOffsets[]     = "acbBufferOffsets";
constexpr const char kDepthRange[]           = "depthRange";
constexpr const char kClipDistancesEnabled[] = "clipDistancesEnabled";

constexpr size_t kNumGraphicsDriverUniforms                                                = 10;
constexpr std::array<const char *, kNumGraphicsDriverUniforms> kGraphicsDriverUniformNames = {
    {kViewport, kHalfRenderAreaHeight, kViewportYScale, kNegViewportYScale, kClipDistancesEnabled,
     kXfbActiveUnpaused, kXfbVerticesPerDraw, kXfbBufferOffsets, kAcbBufferOffsets, kDepthRange}};

constexpr size_t kNumComputeDriverUniforms                                               = 1;
constexpr std::array<const char *, kNumComputeDriverUniforms> kComputeDriverUniformNames = {
    {kAcbBufferOffsets}};

TIntermBinary *CreateDriverUniformRef(const TVariable *driverUniforms, const char *fieldName)
{
    size_t fieldIndex =
        FindFieldIndex(driverUniforms->getType().getInterfaceBlock()->fields(), fieldName);

    TIntermSymbol *angleUniformsRef = new TIntermSymbol(driverUniforms);
    TConstantUnion *uniformIndex    = new TConstantUnion;
    uniformIndex->setIConst(static_cast<int>(fieldIndex));
    TIntermConstantUnion *indexRef =
        new TIntermConstantUnion(uniformIndex, *StaticType::GetBasic<EbtInt>());
    return new TIntermBinary(EOpIndexDirectInterfaceBlock, angleUniformsRef, indexRef);
}

// Replaces a builtin variable with a version that corrects the Y coordinate.
ANGLE_NO_DISCARD bool FlipBuiltinVariable(TCompiler *compiler,
                                          TIntermBlock *root,
                                          TIntermSequence *insertSequence,
                                          TIntermTyped *viewportYScale,
                                          TSymbolTable *symbolTable,
                                          const TVariable *builtin,
                                          const ImmutableString &flippedVariableName,
                                          TIntermTyped *pivot)
{
    // Create a symbol reference to 'builtin'.
    TIntermSymbol *builtinRef = new TIntermSymbol(builtin);

    // Create a swizzle to "builtin.y"
    TVector<int> swizzleOffsetY = {1};
    TIntermSwizzle *builtinY    = new TIntermSwizzle(builtinRef, swizzleOffsetY);

    // Create a symbol reference to our new variable that will hold the modified builtin.
    const TType *type = StaticType::GetForVec<EbtFloat>(
        EvqGlobal, static_cast<unsigned char>(builtin->getType().getNominalSize()));
    TVariable *replacementVar =
        new TVariable(symbolTable, flippedVariableName, type, SymbolType::AngleInternal);
    DeclareGlobalVariable(root, replacementVar);
    TIntermSymbol *flippedBuiltinRef = new TIntermSymbol(replacementVar);

    // Use this new variable instead of 'builtin' everywhere.
    if (!ReplaceVariable(compiler, root, builtin, replacementVar))
    {
        return false;
    }

    // Create the expression "(builtin.y - pivot) * viewportYScale + pivot
    TIntermBinary *removePivot = new TIntermBinary(EOpSub, builtinY, pivot);
    TIntermBinary *inverseY    = new TIntermBinary(EOpMul, removePivot, viewportYScale);
    TIntermBinary *plusPivot   = new TIntermBinary(EOpAdd, inverseY, pivot->deepCopy());

    // Create the corrected variable and copy the value of the original builtin.
    TIntermSequence *sequence = new TIntermSequence();
    sequence->push_back(builtinRef->deepCopy());
    TIntermAggregate *aggregate = TIntermAggregate::CreateConstructor(builtin->getType(), sequence);
    TIntermBinary *assignment   = new TIntermBinary(EOpInitialize, flippedBuiltinRef, aggregate);

    // Create an assignment to the replaced variable's y.
    TIntermSwizzle *correctedY = new TIntermSwizzle(flippedBuiltinRef->deepCopy(), swizzleOffsetY);
    TIntermBinary *assignToY   = new TIntermBinary(EOpAssign, correctedY, plusPivot);

    // Add this assigment at the beginning of the main function
    insertSequence->insert(insertSequence->begin(), assignToY);
    insertSequence->insert(insertSequence->begin(), assignment);

    return compiler->validateAST(root);
}

TIntermSequence *GetMainSequence(TIntermBlock *root)
{
    TIntermFunctionDefinition *main = FindMain(root);
    return main->getBody()->getSequence();
}

// Declares a new variable to replace gl_DepthRange, its values are fed from a driver uniform.
ANGLE_NO_DISCARD bool ReplaceGLDepthRangeWithDriverUniform(TCompiler *compiler,
                                                           TIntermBlock *root,
                                                           const TVariable *driverUniforms,
                                                           TSymbolTable *symbolTable)
{
    // Create a symbol reference to "gl_DepthRange"
    const TVariable *depthRangeVar = static_cast<const TVariable *>(
        symbolTable->findBuiltIn(ImmutableString("gl_DepthRange"), 0));

    // ANGLEUniforms.depthRange
    TIntermBinary *angleEmulatedDepthRangeRef = CreateDriverUniformRef(driverUniforms, kDepthRange);

    // Use this variable instead of gl_DepthRange everywhere.
    return ReplaceVariableWithTyped(compiler, root, depthRangeVar, angleEmulatedDepthRangeRef);
}

// This operation performs the viewport depth translation needed by Vulkan. In GL the viewport
// transformation is slightly different - see the GL 2.0 spec section "2.12.1 Controlling the
// Viewport". In Vulkan the corresponding spec section is currently "23.4. Coordinate
// Transformations".
// The equations reduce to an expression:
//
//     z_vk = 0.5 * (w_gl + z_gl)
//
// where z_vk is the depth output of a Vulkan vertex shader and z_gl is the same for GL.
ANGLE_NO_DISCARD bool AppendVertexShaderDepthCorrectionToMain(TCompiler *compiler,
                                                              TIntermBlock *root,
                                                              TSymbolTable *symbolTable)
{
    // Create a symbol reference to "gl_Position"
    const TVariable *position  = BuiltInVariable::gl_Position();
    TIntermSymbol *positionRef = new TIntermSymbol(position);

    // Create a swizzle to "gl_Position.z"
    TVector<int> swizzleOffsetZ = {2};
    TIntermSwizzle *positionZ   = new TIntermSwizzle(positionRef, swizzleOffsetZ);

    // Create a constant "0.5"
    TIntermConstantUnion *oneHalf = CreateFloatNode(0.5f);

    // Create a swizzle to "gl_Position.w"
    TVector<int> swizzleOffsetW = {3};
    TIntermSwizzle *positionW   = new TIntermSwizzle(positionRef->deepCopy(), swizzleOffsetW);

    // Create the expression "(gl_Position.z + gl_Position.w) * 0.5".
    TIntermBinary *zPlusW = new TIntermBinary(EOpAdd, positionZ->deepCopy(), positionW->deepCopy());
    TIntermBinary *halfZPlusW = new TIntermBinary(EOpMul, zPlusW, oneHalf->deepCopy());

    // Create the assignment "gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5"
    TIntermTyped *positionZLHS = positionZ->deepCopy();
    TIntermBinary *assignment  = new TIntermBinary(TOperator::EOpAssign, positionZLHS, halfZPlusW);

    // Append the assignment as a statement at the end of the shader.
    return RunAtTheEndOfShader(compiler, root, assignment, symbolTable);
}

ANGLE_NO_DISCARD bool AppendVertexShaderTransformFeedbackOutputToMain(TCompiler *compiler,
                                                                      TIntermBlock *root,
                                                                      TSymbolTable *symbolTable)
{
    TVariable *xfbPlaceholder = new TVariable(symbolTable, ImmutableString("@@ XFB-OUT @@"),
                                              new TType(), SymbolType::AngleInternal);

    // Append the assignment as a statement at the end of the shader.
    return RunAtTheEndOfShader(compiler, root, new TIntermSymbol(xfbPlaceholder), symbolTable);
}

// The Add*DriverUniformsToShader operation adds an internal uniform block to a shader. The driver
// block is used to implement Vulkan-specific features and workarounds. Returns the driver uniforms
// variable.
//
// There are Graphics and Compute variations as they require different uniforms.
const TVariable *AddGraphicsDriverUniformsToShader(TIntermBlock *root,
                                                   TSymbolTable *symbolTable,
                                                   const std::vector<TField *> &additionalFields)
{
    // Init the depth range type.
    TFieldList *depthRangeParamsFields = new TFieldList();
    depthRangeParamsFields->push_back(new TField(new TType(EbtFloat, EbpHigh, EvqGlobal, 1, 1),
                                                 ImmutableString("near"), TSourceLoc(),
                                                 SymbolType::AngleInternal));
    depthRangeParamsFields->push_back(new TField(new TType(EbtFloat, EbpHigh, EvqGlobal, 1, 1),
                                                 ImmutableString("far"), TSourceLoc(),
                                                 SymbolType::AngleInternal));
    depthRangeParamsFields->push_back(new TField(new TType(EbtFloat, EbpHigh, EvqGlobal, 1, 1),
                                                 ImmutableString("diff"), TSourceLoc(),
                                                 SymbolType::AngleInternal));
    // This additional field might be used by subclass such as TranslatorMetal.
    depthRangeParamsFields->push_back(new TField(new TType(EbtFloat, EbpHigh, EvqGlobal, 1, 1),
                                                 ImmutableString("reserved"), TSourceLoc(),
                                                 SymbolType::AngleInternal));
    TStructure *emulatedDepthRangeParams = new TStructure(
        symbolTable, kEmulatedDepthRangeParams, depthRangeParamsFields, SymbolType::AngleInternal);
    TType *emulatedDepthRangeType = new TType(emulatedDepthRangeParams, false);

    // Declare a global depth range variable.
    TVariable *depthRangeVar =
        new TVariable(symbolTable->nextUniqueId(), kEmptyImmutableString, SymbolType::Empty,
                      TExtension::UNDEFINED, emulatedDepthRangeType);

    DeclareGlobalVariable(root, depthRangeVar);

    // This field list mirrors the structure of GraphicsDriverUniforms in ContextVk.cpp.
    TFieldList *driverFieldList = new TFieldList;

    const std::array<TType *, kNumGraphicsDriverUniforms> kDriverUniformTypes = {{
        new TType(EbtFloat, 4),
        new TType(EbtFloat),
        new TType(EbtFloat),
        new TType(EbtFloat),
        new TType(EbtUInt),  // uint clipDistancesEnabled;  // 32 bits for 32 clip distances max
        new TType(EbtUInt),
        new TType(EbtUInt),
        // NOTE: There's a vec2 gap here that can be used in the future
        new TType(EbtInt, 4),
        new TType(EbtUInt, 4),
        emulatedDepthRangeType,
    }};

    for (size_t uniformIndex = 0; uniformIndex < kNumGraphicsDriverUniforms; ++uniformIndex)
    {
        TField *driverUniformField =
            new TField(kDriverUniformTypes[uniformIndex],
                       ImmutableString(kGraphicsDriverUniformNames[uniformIndex]), TSourceLoc(),
                       SymbolType::AngleInternal);
        driverFieldList->push_back(driverUniformField);
    }

    // Back-end specific fields
    driverFieldList->insert(driverFieldList->end(), additionalFields.begin(),
                            additionalFields.end());

    // Define a driver uniform block "ANGLEUniformBlock" with instance name "ANGLEUniforms".
    return DeclareInterfaceBlock(root, symbolTable, driverFieldList, EvqUniform,
                                 TMemoryQualifier::Create(), 0, kUniformsBlockName,
                                 kUniformsVarName);
}

const TVariable *AddComputeDriverUniformsToShader(TIntermBlock *root, TSymbolTable *symbolTable)
{
    // This field list mirrors the structure of ComputeDriverUniforms in ContextVk.cpp.
    TFieldList *driverFieldList = new TFieldList;

    const std::array<TType *, kNumComputeDriverUniforms> kDriverUniformTypes = {{
        new TType(EbtUInt, 4),
    }};

    for (size_t uniformIndex = 0; uniformIndex < kNumComputeDriverUniforms; ++uniformIndex)
    {
        TField *driverUniformField =
            new TField(kDriverUniformTypes[uniformIndex],
                       ImmutableString(kComputeDriverUniformNames[uniformIndex]), TSourceLoc(),
                       SymbolType::AngleInternal);
        driverFieldList->push_back(driverUniformField);
    }

    // Define a driver uniform block "ANGLEUniformBlock" with instance name "ANGLEUniforms".
    return DeclareInterfaceBlock(root, symbolTable, driverFieldList, EvqUniform,
                                 TMemoryQualifier::Create(), 0, kUniformsBlockName,
                                 kUniformsVarName);
}

TIntermPreprocessorDirective *GenerateLineRasterIfDef()
{
    return new TIntermPreprocessorDirective(
        PreprocessorDirective::Ifdef, ImmutableString("ANGLE_ENABLE_LINE_SEGMENT_RASTERIZATION"));
}

TIntermPreprocessorDirective *GenerateEndIf()
{
    return new TIntermPreprocessorDirective(PreprocessorDirective::Endif, kEmptyImmutableString);
}

TVariable *AddANGLEPositionVaryingDeclaration(TIntermBlock *root,
                                              TSymbolTable *symbolTable,
                                              TQualifier qualifier)
{
    TIntermSequence *insertSequence = new TIntermSequence;

    insertSequence->push_back(GenerateLineRasterIfDef());

    // Define a driver varying vec2 "ANGLEPosition".
    TType *varyingType               = new TType(EbtFloat, EbpMedium, qualifier, 4);
    TVariable *varyingVar            = new TVariable(symbolTable, ImmutableString("ANGLEPosition"),
                                          varyingType, SymbolType::AngleInternal);
    TIntermSymbol *varyingDeclarator = new TIntermSymbol(varyingVar);
    TIntermDeclaration *varyingDecl  = new TIntermDeclaration;
    varyingDecl->appendDeclarator(varyingDeclarator);
    insertSequence->push_back(varyingDecl);

    insertSequence->push_back(GenerateEndIf());

    // Insert the declarations before Main.
    size_t mainIndex = FindMainIndex(root);
    root->insertChildNodes(mainIndex, *insertSequence);

    return varyingVar;
}

void AddANGLEPositionVarying(TIntermBlock *root, TSymbolTable *symbolTable)
{
    TVariable *anglePosition = AddANGLEPositionVaryingDeclaration(root, symbolTable, EvqVaryingOut);

    // Create an assignment "ANGLEPosition = gl_Position".
    const TVariable *position = BuiltInVariable::gl_Position();
    TIntermSymbol *varyingRef = new TIntermSymbol(anglePosition);
    TIntermBinary *assignment =
        new TIntermBinary(EOpAssign, varyingRef, new TIntermSymbol(position));

    // Ensure the assignment runs at the end of the main() function.
    TIntermFunctionDefinition *main = FindMain(root);
    TIntermBlock *mainBody          = main->getBody();
    mainBody->appendStatement(GenerateLineRasterIfDef());
    mainBody->appendStatement(assignment);
    mainBody->appendStatement(GenerateEndIf());
}

ANGLE_NO_DISCARD bool InsertFragCoordCorrection(TCompiler *compiler,
                                                TIntermBlock *root,
                                                TIntermSequence *insertSequence,
                                                TSymbolTable *symbolTable,
                                                const TVariable *driverUniforms)
{
    TIntermBinary *viewportYScale = CreateDriverUniformRef(driverUniforms, kViewportYScale);
    TIntermBinary *pivot          = CreateDriverUniformRef(driverUniforms, kHalfRenderAreaHeight);
    return FlipBuiltinVariable(compiler, root, insertSequence, viewportYScale, symbolTable,
                               BuiltInVariable::gl_FragCoord(), kFlippedFragCoordName, pivot);
}

// This block adds OpenGL line segment rasterization emulation behind #ifdef guards.
// OpenGL's simple rasterization algorithm is a strict subset of the pixels generated by the Vulkan
// algorithm. Thus we can implement a shader patch that rejects pixels if they would not be
// generated by the OpenGL algorithm. OpenGL's algorithm is similar to Bresenham's line algorithm.
// It is implemented for each pixel by testing if the line segment crosses a small diamond inside
// the pixel. See the OpenGL ES 2.0 spec section "3.4.1 Basic Line Segment Rasterization". Also
// see the Vulkan spec section "24.6.1. Basic Line Segment Rasterization":
// https://khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#primsrast-lines-basic
//
// Using trigonometric math and the fact that we know the size of the diamond we can derive a
// formula to test if the line segment crosses the pixel center. gl_FragCoord is used along with an
// internal position varying to determine the inputs to the formula.
//
// The implementation of the test code is similar to the following pseudocode:
//
// void main()
// {
//     vec2 b = (((position.xy / position.w) * 0.5) + 0.5) * gl_Viewport.zw + gl_Viewport.xy;
//     vec2 ba = abs(b - gl_FragCoord.xy);
//     vec2 ba2 = 2.0 * (ba * ba);
//     vec2 bp = ba2 + ba2.yx - ba;
//     if (bp.x > epsilon && bp.y > epsilon)
//         discard;
//     <otherwise run fragment shader main>
// }
ANGLE_NO_DISCARD bool AddLineSegmentRasterizationEmulation(TCompiler *compiler,
                                                           TInfoSinkBase &sink,
                                                           TIntermBlock *root,
                                                           TSymbolTable *symbolTable,
                                                           const TVariable *driverUniforms,
                                                           bool usesFragCoord)
{
    TVariable *anglePosition = AddANGLEPositionVaryingDeclaration(root, symbolTable, EvqVaryingIn);

    const TType *vec2Type = StaticType::GetBasic<EbtFloat, 2>();

    // Create a swizzle to "ANGLEUniforms.viewport.xy".
    TIntermBinary *viewportRef   = CreateDriverUniformRef(driverUniforms, kViewport);
    TVector<int> swizzleOffsetXY = {0, 1};
    TIntermSwizzle *viewportXY   = new TIntermSwizzle(viewportRef->deepCopy(), swizzleOffsetXY);

    // Create a swizzle to "ANGLEUniforms.viewport.zw".
    TVector<int> swizzleOffsetZW = {2, 3};
    TIntermSwizzle *viewportZW   = new TIntermSwizzle(viewportRef, swizzleOffsetZW);

    // ANGLEPosition.xy / ANGLEPosition.w
    TIntermSymbol *position     = new TIntermSymbol(anglePosition);
    TIntermSwizzle *positionXY  = new TIntermSwizzle(position, swizzleOffsetXY);
    TVector<int> swizzleOffsetW = {3};
    TIntermSwizzle *positionW   = new TIntermSwizzle(position->deepCopy(), swizzleOffsetW);
    TIntermBinary *positionNDC  = new TIntermBinary(EOpDiv, positionXY, positionW);

    // ANGLEPosition * 0.5
    TIntermConstantUnion *oneHalf = CreateFloatNode(0.5f);
    TIntermBinary *halfPosition   = new TIntermBinary(EOpVectorTimesScalar, positionNDC, oneHalf);

    // (ANGLEPosition * 0.5) + 0.5
    TIntermBinary *offsetHalfPosition =
        new TIntermBinary(EOpAdd, halfPosition, oneHalf->deepCopy());

    // ((ANGLEPosition * 0.5) + 0.5) * ANGLEUniforms.viewport.zw
    TIntermBinary *scaledPosition = new TIntermBinary(EOpMul, offsetHalfPosition, viewportZW);

    // ((ANGLEPosition * 0.5) + 0.5) * ANGLEUniforms.viewport + ANGLEUniforms.viewport.xy
    TIntermBinary *windowPosition = new TIntermBinary(EOpAdd, scaledPosition, viewportXY);

    // Assign to a temporary "b".
    TVariable *bTemp          = CreateTempVariable(symbolTable, vec2Type);
    TIntermDeclaration *bDecl = CreateTempInitDeclarationNode(bTemp, windowPosition);

    // gl_FragCoord.xy
    const TVariable *fragCoord  = BuiltInVariable::gl_FragCoord();
    TIntermSymbol *fragCoordRef = new TIntermSymbol(fragCoord);
    TIntermSwizzle *fragCoordXY = new TIntermSwizzle(fragCoordRef, swizzleOffsetXY);

    // b - gl_FragCoord.xy
    TIntermSymbol *bRef           = CreateTempSymbolNode(bTemp);
    TIntermBinary *differenceExpr = new TIntermBinary(EOpSub, bRef, fragCoordXY);

    // abs(b - gl_FragCoord.xy)
    TIntermUnary *baAbs = new TIntermUnary(EOpAbs, differenceExpr, nullptr);

    // Assign to a temporary "ba".
    TVariable *baTemp          = CreateTempVariable(symbolTable, vec2Type);
    TIntermDeclaration *baDecl = CreateTempInitDeclarationNode(baTemp, baAbs);
    TIntermSymbol *ba          = CreateTempSymbolNode(baTemp);

    // ba * ba
    TIntermBinary *baSq = new TIntermBinary(EOpMul, ba, ba->deepCopy());

    // 2.0 * ba * ba
    TIntermTyped *two      = CreateFloatNode(2.0f);
    TIntermBinary *twoBaSq = new TIntermBinary(EOpVectorTimesScalar, baSq, two);

    // Assign to a temporary "ba2".
    TVariable *ba2Temp          = CreateTempVariable(symbolTable, vec2Type);
    TIntermDeclaration *ba2Decl = CreateTempInitDeclarationNode(ba2Temp, twoBaSq);

    // Create a swizzle to "ba2.yx".
    TVector<int> swizzleOffsetYX = {1, 0};
    TIntermSymbol *ba2           = CreateTempSymbolNode(ba2Temp);
    TIntermSwizzle *ba2YX        = new TIntermSwizzle(ba2, swizzleOffsetYX);

    // ba2 + ba2.yx - ba
    TIntermBinary *ba2PlusBaYX2 = new TIntermBinary(EOpAdd, ba2->deepCopy(), ba2YX);
    TIntermBinary *bpInit       = new TIntermBinary(EOpSub, ba2PlusBaYX2, ba->deepCopy());

    // Assign to a temporary "bp".
    TVariable *bpTemp          = CreateTempVariable(symbolTable, vec2Type);
    TIntermDeclaration *bpDecl = CreateTempInitDeclarationNode(bpTemp, bpInit);
    TIntermSymbol *bp          = CreateTempSymbolNode(bpTemp);

    // Create a swizzle to "bp.x".
    TVector<int> swizzleOffsetX = {0};
    TIntermSwizzle *bpX         = new TIntermSwizzle(bp, swizzleOffsetX);

    // Using a small epsilon value ensures that we don't suffer from numerical instability when
    // lines are exactly vertical or horizontal.
    static constexpr float kEpisilon = 0.00001f;
    TIntermConstantUnion *epsilon    = CreateFloatNode(kEpisilon);

    // bp.x > epsilon
    TIntermBinary *checkX = new TIntermBinary(EOpGreaterThan, bpX, epsilon);

    // Create a swizzle to "bp.y".
    TVector<int> swizzleOffsetY = {1};
    TIntermSwizzle *bpY         = new TIntermSwizzle(bp->deepCopy(), swizzleOffsetY);

    // bp.y > epsilon
    TIntermBinary *checkY = new TIntermBinary(EOpGreaterThan, bpY, epsilon->deepCopy());

    // (bp.x > epsilon) && (bp.y > epsilon)
    TIntermBinary *checkXY = new TIntermBinary(EOpLogicalAnd, checkX, checkY);

    // discard
    TIntermBranch *discard     = new TIntermBranch(EOpKill, nullptr);
    TIntermBlock *discardBlock = new TIntermBlock;
    discardBlock->appendStatement(discard);

    // if ((bp.x > epsilon) && (bp.y > epsilon)) discard;
    TIntermIfElse *ifStatement = new TIntermIfElse(checkXY, discardBlock, nullptr);

    // Ensure the line raster code runs at the beginning of main().
    TIntermFunctionDefinition *main = FindMain(root);
    TIntermSequence *mainSequence   = main->getBody()->getSequence();
    ASSERT(mainSequence);

    std::array<TIntermNode *, 6> nodes = {
        {bDecl, baDecl, ba2Decl, bpDecl, ifStatement, GenerateEndIf()}};
    mainSequence->insert(mainSequence->begin(), nodes.begin(), nodes.end());

    // If the shader does not use frag coord, we should insert it inside the ifdef.
    if (!usesFragCoord)
    {
        if (!InsertFragCoordCorrection(compiler, root, mainSequence, symbolTable, driverUniforms))
        {
            return false;
        }
    }

    mainSequence->insert(mainSequence->begin(), GenerateLineRasterIfDef());

    return compiler->validateAST(root);
}

}  // anonymous namespace

TranslatorVulkan::TranslatorVulkan(sh::GLenum type, ShShaderSpec spec)
    : TCompiler(type, spec, SH_GLSL_450_CORE_OUTPUT)
{}

bool TranslatorVulkan::translateImpl(TIntermBlock *root,
                                     ShCompileOptions compileOptions,
                                     PerformanceDiagnostics * /*perfDiagnostics*/,
                                     const TVariable **driverUniformsOut,
                                     TOutputVulkanGLSL *outputGLSL)
{
    TInfoSinkBase &sink = getInfoSink().obj;

    if (getShaderType() == GL_VERTEX_SHADER)
    {
        if (!ShaderBuiltinsWorkaround(this, root, &getSymbolTable(), compileOptions))
        {
            return false;
        }
    }

    sink << "#version 450 core\n";

    // Write out default uniforms into a uniform block assigned to a specific set/binding.
    int defaultUniformCount           = 0;
    int aggregateTypesUsedForUniforms = 0;
    int atomicCounterCount            = 0;
    for (const auto &uniform : getUniforms())
    {
        if (!uniform.isBuiltIn() && uniform.staticUse && !gl::IsOpaqueType(uniform.type))
        {
            ++defaultUniformCount;
        }

        if (uniform.isStruct() || uniform.isArrayOfArrays())
        {
            ++aggregateTypesUsedForUniforms;
        }

        if (gl::IsAtomicCounterType(uniform.type))
        {
            ++atomicCounterCount;
        }
    }

    // TODO(lucferron): Refactor this function to do fewer tree traversals.
    // http://anglebug.com/2461
    if (aggregateTypesUsedForUniforms > 0)
    {
        if (!NameEmbeddedStructUniforms(this, root, &getSymbolTable()))
        {
            return false;
        }

        bool rewriteStructSamplersResult;
        int removedUniformsCount;

        if (compileOptions & SH_USE_OLD_REWRITE_STRUCT_SAMPLERS)
        {
            rewriteStructSamplersResult =
                RewriteStructSamplersOld(this, root, &getSymbolTable(), &removedUniformsCount);
        }
        else
        {
            rewriteStructSamplersResult =
                RewriteStructSamplers(this, root, &getSymbolTable(), &removedUniformsCount);
        }

        if (!rewriteStructSamplersResult)
        {
            return false;
        }
        defaultUniformCount -= removedUniformsCount;

        // We must declare the struct types before using them.
        DeclareStructTypesTraverser structTypesTraverser(outputGLSL);
        root->traverse(&structTypesTraverser);
        if (!structTypesTraverser.updateTree(this, root))
        {
            return false;
        }
    }

    // Rewrite samplerCubes as sampler2DArrays.  This must be done after rewriting struct samplers
    // as it doesn't expect that.
    if (compileOptions & SH_EMULATE_SEAMFUL_CUBE_MAP_SAMPLING)
    {
        if (!RewriteCubeMapSamplersAs2DArray(this, root, &getSymbolTable(),
                                             getShaderType() == GL_FRAGMENT_SHADER))
        {
            return false;
        }
    }

    if (getShaderVersion() >= 300)
    {
        // Make sure every uniform buffer variable has a name.  The following transformation relies
        // on this.
        if (!NameNamelessUniformBuffers(this, root, &getSymbolTable()))
        {
            return false;
        }

        // In GLES3+, matrices can be declared row- or column-major.  Transform all to column-major
        // as interface block field layout qualifiers are not allowed.  This should be done after
        // samplers are taken out of structs (as structs could be rewritten), but before uniforms
        // are collected in a uniform buffer as they are handled especially.
        if (!RewriteRowMajorMatrices(this, root, &getSymbolTable()))
        {
            return false;
        }
    }

    if (defaultUniformCount > 0)
    {
        sink << "\n@@ LAYOUT-defaultUniforms(std140) @@ uniform defaultUniforms\n{\n";

        DeclareDefaultUniformsTraverser defaultTraverser(&sink, getHashFunction(), &getNameMap());
        root->traverse(&defaultTraverser);
        if (!defaultTraverser.updateTree(this, root))
        {
            return false;
        }

        sink << "};\n";
    }

    const TVariable *driverUniforms;
    if (getShaderType() == GL_COMPUTE_SHADER)
    {
        driverUniforms = AddComputeDriverUniformsToShader(root, &getSymbolTable());
    }
    else
    {
        std::vector<TField *> additionalFields;
        createGraphicsDriverUniformAdditionFields(&additionalFields);
        driverUniforms =
            AddGraphicsDriverUniformsToShader(root, &getSymbolTable(), additionalFields);
    }

    if (atomicCounterCount > 0)
    {
        // ANGLEUniforms.acbBufferOffsets
        const TIntermBinary *acbBufferOffsets =
            CreateDriverUniformRef(driverUniforms, kAcbBufferOffsets);

        if (!RewriteAtomicCounters(this, root, &getSymbolTable(), acbBufferOffsets))
        {
            return false;
        }
    }

    if (getShaderType() != GL_COMPUTE_SHADER)
    {
        if (!ReplaceGLDepthRangeWithDriverUniform(this, root, driverUniforms, &getSymbolTable()))
        {
            return false;
        }
    }

    // Declare gl_FragColor and glFragData as webgl_FragColor and webgl_FragData
    // if it's core profile shaders and they are used.
    if (getShaderType() == GL_FRAGMENT_SHADER)
    {
        bool usesPointCoord = false;
        bool usesFragCoord  = false;

        // Search for the gl_PointCoord usage, if its used, we need to flip the y coordinate.
        for (const ShaderVariable &inputVarying : mInputVaryings)
        {
            if (!inputVarying.isBuiltIn())
            {
                continue;
            }

            if (inputVarying.name == "gl_PointCoord")
            {
                usesPointCoord = true;
                break;
            }

            if (inputVarying.name == "gl_FragCoord")
            {
                usesFragCoord = true;
                break;
            }
        }

        if (!AddLineSegmentRasterizationEmulation(this, sink, root, &getSymbolTable(),
                                                  driverUniforms, usesFragCoord))
        {
            return false;
        }

        bool hasGLFragColor = false;
        bool hasGLFragData  = false;

        for (const ShaderVariable &outputVar : mOutputVariables)
        {
            if (outputVar.name == "gl_FragColor")
            {
                ASSERT(!hasGLFragColor);
                hasGLFragColor = true;
                continue;
            }
            else if (outputVar.name == "gl_FragData")
            {
                ASSERT(!hasGLFragData);
                hasGLFragData = true;
                continue;
            }
        }
        ASSERT(!(hasGLFragColor && hasGLFragData));
        if (hasGLFragColor)
        {
            sink << "layout(location = 0) out vec4 webgl_FragColor;\n";
        }
        if (hasGLFragData)
        {
            sink << "layout(location = 0) out vec4 webgl_FragData[gl_MaxDrawBuffers];\n";
        }

        if (usesPointCoord)
        {
            TIntermBinary *viewportYScale =
                CreateDriverUniformRef(driverUniforms, kNegViewportYScale);
            TIntermConstantUnion *pivot = CreateFloatNode(0.5f);
            if (!FlipBuiltinVariable(this, root, GetMainSequence(root), viewportYScale,
                                     &getSymbolTable(), BuiltInVariable::gl_PointCoord(),
                                     kFlippedPointCoordName, pivot))
            {
                return false;
            }
        }

        if (usesFragCoord)
        {
            if (!InsertFragCoordCorrection(this, root, GetMainSequence(root), &getSymbolTable(),
                                           driverUniforms))
            {
                return false;
            }
        }

        {
            TIntermBinary *viewportYScale = CreateDriverUniformRef(driverUniforms, kViewportYScale);
            if (!RewriteDfdy(this, root, getSymbolTable(), getShaderVersion(), viewportYScale))
            {
                return false;
            }
        }
    }
    else if (getShaderType() == GL_VERTEX_SHADER)
    {
        AddANGLEPositionVarying(root, &getSymbolTable());

        // Add a macro to declare transform feedback buffers.
        sink << "@@ XFB-DECL @@\n\n";

        // Append a macro for transform feedback substitution prior to modifying depth.
        if (!AppendVertexShaderTransformFeedbackOutputToMain(this, root, &getSymbolTable()))
        {
            return false;
        }

        // Search for the gl_ClipDistance usage, if its used, we need to do some replacements.
        bool useClipDistance = false;
        for (const ShaderVariable &outputVarying : mOutputVaryings)
        {
            if (outputVarying.name == "gl_ClipDistance")
            {
                useClipDistance = true;
                break;
            }
        }
        if (useClipDistance && !ReplaceClipDistanceAssignments(
                                   this, root, &getSymbolTable(),
                                   CreateDriverUniformRef(driverUniforms, kClipDistancesEnabled)))
        {
            return false;
        }

        // Append depth range translation to main.
        if (!transformDepthBeforeCorrection(root, driverUniforms))
        {
            return false;
        }
        if (!AppendVertexShaderDepthCorrectionToMain(this, root, &getSymbolTable()))
        {
            return false;
        }
    }
    else if (getShaderType() == GL_GEOMETRY_SHADER)
    {
        AddANGLEPositionVarying(root, &getSymbolTable());

        WriteGeometryShaderLayoutQualifiers(
            sink, getGeometryShaderInputPrimitiveType(), getGeometryShaderInvocations(),
            getGeometryShaderOutputPrimitiveType(), getGeometryShaderMaxVertices());
    }
    else
    {
        ASSERT(getShaderType() == GL_COMPUTE_SHADER);
        EmitWorkGroupSizeGLSL(*this, sink);
    }

    if (!validateAST(root))
    {
        return false;
    }

    if (driverUniformsOut)
    {
        *driverUniformsOut = driverUniforms;
    }

    return true;
}

bool TranslatorVulkan::translate(TIntermBlock *root,
                                 ShCompileOptions compileOptions,
                                 PerformanceDiagnostics *perfDiagnostics)
{

    TInfoSinkBase &sink = getInfoSink().obj;
    TOutputVulkanGLSL outputGLSL(sink, getArrayIndexClampingStrategy(), getHashFunction(),
                                 getNameMap(), &getSymbolTable(), getShaderType(),
                                 getShaderVersion(), getOutputType(), compileOptions);

    if (!translateImpl(root, compileOptions, perfDiagnostics, nullptr, &outputGLSL))
    {
        return false;
    }

    // Write translated shader.
    root->traverse(&outputGLSL);

    return true;
}

bool TranslatorVulkan::shouldFlattenPragmaStdglInvariantAll()
{
    // Not necessary.
    return false;
}

TIntermBinary *TranslatorVulkan::getDriverUniformNegViewportYScaleRef(
    const TVariable *driverUniforms) const
{
    return CreateDriverUniformRef(driverUniforms, kNegViewportYScale);
}

TIntermBinary *TranslatorVulkan::getDriverUniformDepthRangeReservedFieldRef(
    const TVariable *driverUniforms) const
{
    TIntermBinary *depthRange = CreateDriverUniformRef(driverUniforms, kDepthRange);

    return new TIntermBinary(EOpIndexDirectStruct, depthRange, CreateIndexNode(3));
}

}  // namespace sh

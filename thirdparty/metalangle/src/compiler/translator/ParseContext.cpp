//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/ParseContext.h"

#include <stdarg.h>
#include <stdio.h>

#include "common/mathutil.h"
#include "compiler/preprocessor/SourceLocation.h"
#include "compiler/translator/Declarator.h"
#include "compiler/translator/ParseContext_interm.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/ValidateGlobalInitializer.h"
#include "compiler/translator/ValidateSwitch.h"
#include "compiler/translator/glslang.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/util.h"

namespace sh
{

///////////////////////////////////////////////////////////////////////
//
// Sub- vector and matrix fields
//
////////////////////////////////////////////////////////////////////////

namespace
{

const int kWebGLMaxStructNesting = 4;

bool ContainsSampler(const TStructure *structType);

bool ContainsSampler(const TType &type)
{
    if (IsSampler(type.getBasicType()))
    {
        return true;
    }
    if (type.getBasicType() == EbtStruct)
    {
        return ContainsSampler(type.getStruct());
    }

    return false;
}

bool ContainsSampler(const TStructure *structType)
{
    for (const auto &field : structType->fields())
    {
        if (ContainsSampler(*field->type()))
            return true;
    }
    return false;
}

// Get a token from an image argument to use as an error message token.
const char *GetImageArgumentToken(TIntermTyped *imageNode)
{
    ASSERT(IsImage(imageNode->getBasicType()));
    while (imageNode->getAsBinaryNode() &&
           (imageNode->getAsBinaryNode()->getOp() == EOpIndexIndirect ||
            imageNode->getAsBinaryNode()->getOp() == EOpIndexDirect))
    {
        imageNode = imageNode->getAsBinaryNode()->getLeft();
    }
    TIntermSymbol *imageSymbol = imageNode->getAsSymbolNode();
    if (imageSymbol)
    {
        return imageSymbol->getName().data();
    }
    return "image";
}

bool CanSetDefaultPrecisionOnType(const TPublicType &type)
{
    if (!SupportsPrecision(type.getBasicType()))
    {
        return false;
    }
    if (type.getBasicType() == EbtUInt)
    {
        // ESSL 3.00.4 section 4.5.4
        return false;
    }
    if (type.isAggregate())
    {
        // Not allowed to set for aggregate types
        return false;
    }
    return true;
}

// Map input primitive types to input array sizes in a geometry shader.
GLuint GetGeometryShaderInputArraySize(TLayoutPrimitiveType primitiveType)
{
    switch (primitiveType)
    {
        case EptPoints:
            return 1u;
        case EptLines:
            return 2u;
        case EptTriangles:
            return 3u;
        case EptLinesAdjacency:
            return 4u;
        case EptTrianglesAdjacency:
            return 6u;
        default:
            UNREACHABLE();
            return 0u;
    }
}

bool IsBufferOrSharedVariable(TIntermTyped *var)
{
    if (var->isInterfaceBlock() || var->getQualifier() == EvqBuffer ||
        var->getQualifier() == EvqShared)
    {
        return true;
    }
    return false;
}

}  // namespace

// This tracks each binding point's current default offset for inheritance of subsequent
// variables using the same binding, and keeps offsets unique and non overlapping.
// See GLSL ES 3.1, section 4.4.6.
class TParseContext::AtomicCounterBindingState
{
  public:
    AtomicCounterBindingState() : mDefaultOffset(0) {}
    // Inserts a new span and returns -1 if overlapping, else returns the starting offset of
    // newly inserted span.
    int insertSpan(int start, size_t length)
    {
        gl::RangeI newSpan(start, start + static_cast<int>(length));
        for (const auto &span : mSpans)
        {
            if (newSpan.intersects(span))
            {
                return -1;
            }
        }
        mSpans.push_back(newSpan);
        mDefaultOffset = newSpan.high();
        return start;
    }
    // Inserts a new span starting from the default offset.
    int appendSpan(size_t length) { return insertSpan(mDefaultOffset, length); }
    void setDefaultOffset(int offset) { mDefaultOffset = offset; }

  private:
    int mDefaultOffset;
    std::vector<gl::RangeI> mSpans;
};

TParseContext::TParseContext(TSymbolTable &symt,
                             TExtensionBehavior &ext,
                             sh::GLenum type,
                             ShShaderSpec spec,
                             ShCompileOptions options,
                             bool checksPrecErrors,
                             TDiagnostics *diagnostics,
                             const ShBuiltInResources &resources,
                             ShShaderOutput outputType)
    : symbolTable(symt),
      mDeferredNonEmptyDeclarationErrorCheck(false),
      mShaderType(type),
      mShaderSpec(spec),
      mCompileOptions(options),
      mShaderVersion(100),
      mTreeRoot(nullptr),
      mLoopNestingLevel(0),
      mStructNestingLevel(0),
      mSwitchNestingLevel(0),
      mCurrentFunctionType(nullptr),
      mFunctionReturnsValue(false),
      mChecksPrecisionErrors(checksPrecErrors),
      mFragmentPrecisionHighOnESSL1(false),
      mDefaultUniformMatrixPacking(EmpColumnMajor),
      mDefaultUniformBlockStorage(sh::IsWebGLBasedSpec(spec) ? EbsStd140 : EbsShared),
      mDefaultBufferMatrixPacking(EmpColumnMajor),
      mDefaultBufferBlockStorage(sh::IsWebGLBasedSpec(spec) ? EbsStd140 : EbsShared),
      mDiagnostics(diagnostics),
      mDirectiveHandler(ext,
                        *mDiagnostics,
                        mShaderVersion,
                        mShaderType,
                        resources.WEBGL_debug_shader_precision == 1),
      mPreprocessor(mDiagnostics, &mDirectiveHandler, angle::pp::PreprocessorSettings(spec)),
      mScanner(nullptr),
      mMinProgramTexelOffset(resources.MinProgramTexelOffset),
      mMaxProgramTexelOffset(resources.MaxProgramTexelOffset),
      mMinProgramTextureGatherOffset(resources.MinProgramTextureGatherOffset),
      mMaxProgramTextureGatherOffset(resources.MaxProgramTextureGatherOffset),
      mComputeShaderLocalSizeDeclared(false),
      mComputeShaderLocalSize(-1),
      mNumViews(-1),
      mMaxNumViews(resources.MaxViewsOVR),
      mMaxImageUnits(resources.MaxImageUnits),
      mMaxCombinedTextureImageUnits(resources.MaxCombinedTextureImageUnits),
      mMaxUniformLocations(resources.MaxUniformLocations),
      mMaxUniformBufferBindings(resources.MaxUniformBufferBindings),
      mMaxAtomicCounterBindings(resources.MaxAtomicCounterBindings),
      mMaxShaderStorageBufferBindings(resources.MaxShaderStorageBufferBindings),
      mDeclaringFunction(false),
      mGeometryShaderInputPrimitiveType(EptUndefined),
      mGeometryShaderOutputPrimitiveType(EptUndefined),
      mGeometryShaderInvocations(0),
      mGeometryShaderMaxVertices(-1),
      mMaxGeometryShaderInvocations(resources.MaxGeometryShaderInvocations),
      mMaxGeometryShaderMaxVertices(resources.MaxGeometryOutputVertices),
      mFunctionBodyNewScope(false),
      mOutputType(outputType)
{}

TParseContext::~TParseContext() {}

bool TParseContext::anyMultiviewExtensionAvailable()
{
    return isExtensionEnabled(TExtension::OVR_multiview) ||
           isExtensionEnabled(TExtension::OVR_multiview2);
}

bool TParseContext::parseVectorFields(const TSourceLoc &line,
                                      const ImmutableString &compString,
                                      int vecSize,
                                      TVector<int> *fieldOffsets)
{
    ASSERT(fieldOffsets);
    size_t fieldCount = compString.length();
    if (fieldCount > 4u)
    {
        error(line, "illegal vector field selection", compString);
        return false;
    }
    fieldOffsets->resize(fieldCount);

    enum
    {
        exyzw,
        ergba,
        estpq
    } fieldSet[4];

    for (unsigned int i = 0u; i < fieldOffsets->size(); ++i)
    {
        switch (compString[i])
        {
            case 'x':
                (*fieldOffsets)[i] = 0;
                fieldSet[i]        = exyzw;
                break;
            case 'r':
                (*fieldOffsets)[i] = 0;
                fieldSet[i]        = ergba;
                break;
            case 's':
                (*fieldOffsets)[i] = 0;
                fieldSet[i]        = estpq;
                break;
            case 'y':
                (*fieldOffsets)[i] = 1;
                fieldSet[i]        = exyzw;
                break;
            case 'g':
                (*fieldOffsets)[i] = 1;
                fieldSet[i]        = ergba;
                break;
            case 't':
                (*fieldOffsets)[i] = 1;
                fieldSet[i]        = estpq;
                break;
            case 'z':
                (*fieldOffsets)[i] = 2;
                fieldSet[i]        = exyzw;
                break;
            case 'b':
                (*fieldOffsets)[i] = 2;
                fieldSet[i]        = ergba;
                break;
            case 'p':
                (*fieldOffsets)[i] = 2;
                fieldSet[i]        = estpq;
                break;

            case 'w':
                (*fieldOffsets)[i] = 3;
                fieldSet[i]        = exyzw;
                break;
            case 'a':
                (*fieldOffsets)[i] = 3;
                fieldSet[i]        = ergba;
                break;
            case 'q':
                (*fieldOffsets)[i] = 3;
                fieldSet[i]        = estpq;
                break;
            default:
                error(line, "illegal vector field selection", compString);
                return false;
        }
    }

    for (unsigned int i = 0u; i < fieldOffsets->size(); ++i)
    {
        if ((*fieldOffsets)[i] >= vecSize)
        {
            error(line, "vector field selection out of range", compString);
            return false;
        }

        if (i > 0)
        {
            if (fieldSet[i] != fieldSet[i - 1])
            {
                error(line, "illegal - vector component fields not from the same set", compString);
                return false;
            }
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////
//
// Errors
//
////////////////////////////////////////////////////////////////////////

//
// Used by flex/bison to output all syntax and parsing errors.
//
void TParseContext::error(const TSourceLoc &loc, const char *reason, const char *token)
{
    mDiagnostics->error(loc, reason, token);
}

void TParseContext::error(const TSourceLoc &loc, const char *reason, const ImmutableString &token)
{
    mDiagnostics->error(loc, reason, token.data());
}

void TParseContext::warning(const TSourceLoc &loc, const char *reason, const char *token)
{
    mDiagnostics->warning(loc, reason, token);
}

void TParseContext::outOfRangeError(bool isError,
                                    const TSourceLoc &loc,
                                    const char *reason,
                                    const char *token)
{
    if (isError)
    {
        error(loc, reason, token);
    }
    else
    {
        warning(loc, reason, token);
    }
}

//
// Same error message for all places assignments don't work.
//
void TParseContext::assignError(const TSourceLoc &line,
                                const char *op,
                                const TType &left,
                                const TType &right)
{
    TInfoSinkBase reasonStream;
    reasonStream << "cannot convert from '" << right << "' to '" << left << "'";
    error(line, reasonStream.c_str(), op);
}

//
// Same error message for all places unary operations don't work.
//
void TParseContext::unaryOpError(const TSourceLoc &line, const char *op, const TType &operand)
{
    TInfoSinkBase reasonStream;
    reasonStream << "wrong operand type - no operation '" << op
                 << "' exists that takes an operand of type " << operand
                 << " (or there is no acceptable conversion)";
    error(line, reasonStream.c_str(), op);
}

//
// Same error message for all binary operations don't work.
//
void TParseContext::binaryOpError(const TSourceLoc &line,
                                  const char *op,
                                  const TType &left,
                                  const TType &right)
{
    TInfoSinkBase reasonStream;
    reasonStream << "wrong operand types - no operation '" << op
                 << "' exists that takes a left-hand operand of type '" << left
                 << "' and a right operand of type '" << right
                 << "' (or there is no acceptable conversion)";
    error(line, reasonStream.c_str(), op);
}

void TParseContext::checkPrecisionSpecified(const TSourceLoc &line,
                                            TPrecision precision,
                                            TBasicType type)
{
    if (!mChecksPrecisionErrors)
        return;

    if (precision != EbpUndefined && !SupportsPrecision(type))
    {
        error(line, "illegal type for precision qualifier", getBasicString(type));
    }

    if (precision == EbpUndefined)
    {
        switch (type)
        {
            case EbtFloat:
                error(line, "No precision specified for (float)", "");
                return;
            case EbtInt:
            case EbtUInt:
                UNREACHABLE();  // there's always a predeclared qualifier
                error(line, "No precision specified (int)", "");
                return;
            default:
                if (IsOpaqueType(type))
                {
                    error(line, "No precision specified", getBasicString(type));
                    return;
                }
        }
    }
}

void TParseContext::markStaticReadIfSymbol(TIntermNode *node)
{
    TIntermSwizzle *swizzleNode = node->getAsSwizzleNode();
    if (swizzleNode)
    {
        markStaticReadIfSymbol(swizzleNode->getOperand());
        return;
    }
    TIntermBinary *binaryNode = node->getAsBinaryNode();
    if (binaryNode)
    {
        switch (binaryNode->getOp())
        {
            case EOpIndexDirect:
            case EOpIndexIndirect:
            case EOpIndexDirectStruct:
            case EOpIndexDirectInterfaceBlock:
                markStaticReadIfSymbol(binaryNode->getLeft());
                return;
            default:
                return;
        }
    }
    TIntermSymbol *symbolNode = node->getAsSymbolNode();
    if (symbolNode)
    {
        symbolTable.markStaticRead(symbolNode->variable());
    }
}

// Both test and if necessary, spit out an error, to see if the node is really
// an l-value that can be operated on this way.
bool TParseContext::checkCanBeLValue(const TSourceLoc &line, const char *op, TIntermTyped *node)
{
    TIntermSwizzle *swizzleNode = node->getAsSwizzleNode();
    if (swizzleNode)
    {
        bool ok = checkCanBeLValue(line, op, swizzleNode->getOperand());
        if (ok && swizzleNode->hasDuplicateOffsets())
        {
            error(line, " l-value of swizzle cannot have duplicate components", op);
            return false;
        }
        return ok;
    }

    TIntermBinary *binaryNode = node->getAsBinaryNode();
    if (binaryNode)
    {
        switch (binaryNode->getOp())
        {
            case EOpIndexDirect:
            case EOpIndexIndirect:
            case EOpIndexDirectStruct:
            case EOpIndexDirectInterfaceBlock:
                if (node->getMemoryQualifier().readonly)
                {
                    error(line, "can't modify a readonly variable", op);
                    return false;
                }
                return checkCanBeLValue(line, op, binaryNode->getLeft());
            default:
                break;
        }
        error(line, " l-value required", op);
        return false;
    }

    std::string message;
    switch (node->getQualifier())
    {
        case EvqConst:
            message = "can't modify a const";
            break;
        case EvqConstReadOnly:
            message = "can't modify a const";
            break;
        case EvqAttribute:
            message = "can't modify an attribute";
            break;
        case EvqFragmentIn:
        case EvqVertexIn:
        case EvqGeometryIn:
        case EvqFlatIn:
        case EvqSmoothIn:
        case EvqCentroidIn:
            message = "can't modify an input";
            break;
        case EvqUniform:
            message = "can't modify a uniform";
            break;
        case EvqVaryingIn:
            message = "can't modify a varying";
            break;
        case EvqFragCoord:
            message = "can't modify gl_FragCoord";
            break;
        case EvqFrontFacing:
            message = "can't modify gl_FrontFacing";
            break;
        case EvqPointCoord:
            message = "can't modify gl_PointCoord";
            break;
        case EvqNumWorkGroups:
            message = "can't modify gl_NumWorkGroups";
            break;
        case EvqWorkGroupSize:
            message = "can't modify gl_WorkGroupSize";
            break;
        case EvqWorkGroupID:
            message = "can't modify gl_WorkGroupID";
            break;
        case EvqLocalInvocationID:
            message = "can't modify gl_LocalInvocationID";
            break;
        case EvqGlobalInvocationID:
            message = "can't modify gl_GlobalInvocationID";
            break;
        case EvqLocalInvocationIndex:
            message = "can't modify gl_LocalInvocationIndex";
            break;
        case EvqViewIDOVR:
            message = "can't modify gl_ViewID_OVR";
            break;
        case EvqComputeIn:
            message = "can't modify work group size variable";
            break;
        case EvqPerVertexIn:
            message = "can't modify any member in gl_in";
            break;
        case EvqPrimitiveIDIn:
            message = "can't modify gl_PrimitiveIDIn";
            break;
        case EvqInvocationID:
            message = "can't modify gl_InvocationID";
            break;
        case EvqPrimitiveID:
            if (mShaderType == GL_FRAGMENT_SHADER)
            {
                message = "can't modify gl_PrimitiveID in a fragment shader";
            }
            break;
        case EvqLayer:
            if (mShaderType == GL_FRAGMENT_SHADER)
            {
                message = "can't modify gl_Layer in a fragment shader";
            }
            break;
        default:
            //
            // Type that can't be written to?
            //
            if (node->getBasicType() == EbtVoid)
            {
                message = "can't modify void";
            }
            if (IsOpaqueType(node->getBasicType()))
            {
                message = "can't modify a variable with type ";
                message += getBasicString(node->getBasicType());
            }
            else if (node->getMemoryQualifier().readonly)
            {
                message = "can't modify a readonly variable";
            }
    }

    ASSERT(binaryNode == nullptr && swizzleNode == nullptr);
    TIntermSymbol *symNode = node->getAsSymbolNode();
    if (message.empty() && symNode != nullptr)
    {
        symbolTable.markStaticWrite(symNode->variable());
        return true;
    }

    std::stringstream reasonStream = sh::InitializeStream<std::stringstream>();
    reasonStream << "l-value required";
    if (!message.empty())
    {
        if (symNode)
        {
            // Symbol inside an expression can't be nameless.
            ASSERT(symNode->variable().symbolType() != SymbolType::Empty);
            const ImmutableString &symbol = symNode->getName();
            reasonStream << " (" << message << " \"" << symbol << "\")";
        }
        else
        {
            reasonStream << " (" << message << ")";
        }
    }
    std::string reason = reasonStream.str();
    error(line, reason.c_str(), op);

    return false;
}

// Both test, and if necessary spit out an error, to see if the node is really
// a constant.
void TParseContext::checkIsConst(TIntermTyped *node)
{
    if (node->getQualifier() != EvqConst)
    {
        error(node->getLine(), "constant expression required", "");
    }
}

// Both test, and if necessary spit out an error, to see if the node is really
// an integer.
void TParseContext::checkIsScalarInteger(TIntermTyped *node, const char *token)
{
    if (!node->isScalarInt())
    {
        error(node->getLine(), "integer expression required", token);
    }
}

// Both test, and if necessary spit out an error, to see if we are currently
// globally scoped.
bool TParseContext::checkIsAtGlobalLevel(const TSourceLoc &line, const char *token)
{
    if (!symbolTable.atGlobalLevel())
    {
        error(line, "only allowed at global scope", token);
        return false;
    }
    return true;
}

// ESSL 3.00.5 sections 3.8 and 3.9.
// If it starts "gl_" or contains two consecutive underscores, it's reserved.
// Also checks for "webgl_" and "_webgl_" reserved identifiers if parsing a webgl shader.
bool TParseContext::checkIsNotReserved(const TSourceLoc &line, const ImmutableString &identifier)
{
    static const char *reservedErrMsg = "reserved built-in name";
    if (identifier.beginsWith("gl_"))
    {
        error(line, reservedErrMsg, "gl_");
        return false;
    }
    if (sh::IsWebGLBasedSpec(mShaderSpec))
    {
        if (identifier.beginsWith("webgl_"))
        {
            error(line, reservedErrMsg, "webgl_");
            return false;
        }
        if (identifier.beginsWith("_webgl_"))
        {
            error(line, reservedErrMsg, "_webgl_");
            return false;
        }
    }
    if (identifier.contains("__"))
    {
        error(line,
              "identifiers containing two consecutive underscores (__) are reserved as "
              "possible future keywords",
              identifier);
        return false;
    }
    return true;
}

// Make sure the argument types are correct for constructing a specific type.
bool TParseContext::checkConstructorArguments(const TSourceLoc &line,
                                              const TIntermSequence &arguments,
                                              const TType &type)
{
    if (arguments.empty())
    {
        error(line, "constructor does not have any arguments", "constructor");
        return false;
    }

    for (TIntermNode *arg : arguments)
    {
        markStaticReadIfSymbol(arg);
        const TIntermTyped *argTyped = arg->getAsTyped();
        ASSERT(argTyped != nullptr);
        if (type.getBasicType() != EbtStruct && IsOpaqueType(argTyped->getBasicType()))
        {
            std::string reason("cannot convert a variable with type ");
            reason += getBasicString(argTyped->getBasicType());
            error(line, reason.c_str(), "constructor");
            return false;
        }
        else if (argTyped->getMemoryQualifier().writeonly)
        {
            error(line, "cannot convert a variable with writeonly", "constructor");
            return false;
        }
        if (argTyped->getBasicType() == EbtVoid)
        {
            error(line, "cannot convert a void", "constructor");
            return false;
        }
    }

    if (type.isArray())
    {
        // The size of an unsized constructor should already have been determined.
        ASSERT(!type.isUnsizedArray());
        if (static_cast<size_t>(type.getOutermostArraySize()) != arguments.size())
        {
            error(line, "array constructor needs one argument per array element", "constructor");
            return false;
        }
        // GLSL ES 3.00 section 5.4.4: Each argument must be the same type as the element type of
        // the array.
        for (TIntermNode *const &argNode : arguments)
        {
            const TType &argType = argNode->getAsTyped()->getType();
            if (mShaderVersion < 310 && argType.isArray())
            {
                error(line, "constructing from a non-dereferenced array", "constructor");
                return false;
            }
            if (!argType.isElementTypeOf(type))
            {
                error(line, "Array constructor argument has an incorrect type", "constructor");
                return false;
            }
        }
    }
    else if (type.getBasicType() == EbtStruct)
    {
        const TFieldList &fields = type.getStruct()->fields();
        if (fields.size() != arguments.size())
        {
            error(line,
                  "Number of constructor parameters does not match the number of structure fields",
                  "constructor");
            return false;
        }

        for (size_t i = 0; i < fields.size(); i++)
        {
            if (i >= arguments.size() ||
                arguments[i]->getAsTyped()->getType() != *fields[i]->type())
            {
                error(line, "Structure constructor arguments do not match structure fields",
                      "constructor");
                return false;
            }
        }
    }
    else
    {
        // We're constructing a scalar, vector, or matrix.

        // Note: It's okay to have too many components available, but not okay to have unused
        // arguments. 'full' will go to true when enough args have been seen. If we loop again,
        // there is an extra argument, so 'overFull' will become true.

        size_t size    = 0;
        bool full      = false;
        bool overFull  = false;
        bool matrixArg = false;
        for (TIntermNode *arg : arguments)
        {
            const TIntermTyped *argTyped = arg->getAsTyped();
            ASSERT(argTyped != nullptr);

            if (argTyped->getBasicType() == EbtStruct)
            {
                error(line, "a struct cannot be used as a constructor argument for this type",
                      "constructor");
                return false;
            }
            if (argTyped->getType().isArray())
            {
                error(line, "constructing from a non-dereferenced array", "constructor");
                return false;
            }
            if (argTyped->getType().isMatrix())
            {
                matrixArg = true;
            }

            size += argTyped->getType().getObjectSize();
            if (full)
            {
                overFull = true;
            }
            if (size >= type.getObjectSize())
            {
                full = true;
            }
        }

        if (type.isMatrix() && matrixArg)
        {
            if (arguments.size() != 1)
            {
                error(line, "constructing matrix from matrix can only take one argument",
                      "constructor");
                return false;
            }
        }
        else
        {
            if (size != 1 && size < type.getObjectSize())
            {
                error(line, "not enough data provided for construction", "constructor");
                return false;
            }
            if (overFull)
            {
                error(line, "too many arguments", "constructor");
                return false;
            }
        }
    }

    return true;
}

// This function checks to see if a void variable has been declared and raise an error message for
// such a case
//
// returns true in case of an error
//
bool TParseContext::checkIsNonVoid(const TSourceLoc &line,
                                   const ImmutableString &identifier,
                                   const TBasicType &type)
{
    if (type == EbtVoid)
    {
        error(line, "illegal use of type 'void'", identifier);
        return false;
    }

    return true;
}

// This function checks to see if the node (for the expression) contains a scalar boolean expression
// or not.
bool TParseContext::checkIsScalarBool(const TSourceLoc &line, const TIntermTyped *type)
{
    if (type->getBasicType() != EbtBool || !type->isScalar())
    {
        error(line, "boolean expression expected", "");
        return false;
    }
    return true;
}

// This function checks to see if the node (for the expression) contains a scalar boolean expression
// or not.
void TParseContext::checkIsScalarBool(const TSourceLoc &line, const TPublicType &pType)
{
    if (pType.getBasicType() != EbtBool || pType.isAggregate())
    {
        error(line, "boolean expression expected", "");
    }
}

bool TParseContext::checkIsNotOpaqueType(const TSourceLoc &line,
                                         const TTypeSpecifierNonArray &pType,
                                         const char *reason)
{
    if (pType.type == EbtStruct)
    {
        if (ContainsSampler(pType.userDef))
        {
            std::stringstream reasonStream = sh::InitializeStream<std::stringstream>();
            reasonStream << reason << " (structure contains a sampler)";
            std::string reasonStr = reasonStream.str();
            error(line, reasonStr.c_str(), getBasicString(pType.type));
            return false;
        }
        // only samplers need to be checked from structs, since other opaque types can't be struct
        // members.
        return true;
    }
    else if (IsOpaqueType(pType.type))
    {
        error(line, reason, getBasicString(pType.type));
        return false;
    }

    return true;
}

void TParseContext::checkDeclaratorLocationIsNotSpecified(const TSourceLoc &line,
                                                          const TPublicType &pType)
{
    if (pType.layoutQualifier.location != -1)
    {
        error(line, "location must only be specified for a single input or output variable",
              "location");
    }
}

void TParseContext::checkLocationIsNotSpecified(const TSourceLoc &location,
                                                const TLayoutQualifier &layoutQualifier)
{
    if (layoutQualifier.location != -1)
    {
        const char *errorMsg = "invalid layout qualifier: only valid on program inputs and outputs";
        if (mShaderVersion >= 310)
        {
            errorMsg =
                "invalid layout qualifier: only valid on shader inputs, outputs, and uniforms";
        }
        error(location, errorMsg, "location");
    }
}

void TParseContext::checkStd430IsForShaderStorageBlock(const TSourceLoc &location,
                                                       const TLayoutBlockStorage &blockStorage,
                                                       const TQualifier &qualifier)
{
    if (blockStorage == EbsStd430 && qualifier != EvqBuffer)
    {
        error(location, "The std430 layout is supported only for shader storage blocks.", "std430");
    }
}

void TParseContext::checkOutParameterIsNotOpaqueType(const TSourceLoc &line,
                                                     TQualifier qualifier,
                                                     const TType &type)
{
    ASSERT(qualifier == EvqOut || qualifier == EvqInOut);
    if (IsOpaqueType(type.getBasicType()))
    {
        error(line, "opaque types cannot be output parameters", type.getBasicString());
    }
}

// Do size checking for an array type's size.
unsigned int TParseContext::checkIsValidArraySize(const TSourceLoc &line, TIntermTyped *expr)
{
    TIntermConstantUnion *constant = expr->getAsConstantUnion();

    // ANGLE should be able to fold any EvqConst expressions resulting in an integer - but to be
    // safe against corner cases we still check for constant folding. Some interpretations of the
    // spec have allowed constant expressions with side effects - like array length() method on a
    // non-constant array.
    if (expr->getQualifier() != EvqConst || constant == nullptr || !constant->isScalarInt())
    {
        error(line, "array size must be a constant integer expression", "");
        return 1u;
    }

    unsigned int size = 0u;

    if (constant->getBasicType() == EbtUInt)
    {
        size = constant->getUConst(0);
    }
    else
    {
        int signedSize = constant->getIConst(0);

        if (signedSize < 0)
        {
            error(line, "array size must be non-negative", "");
            return 1u;
        }

        size = static_cast<unsigned int>(signedSize);
    }

    if (size == 0u)
    {
        error(line, "array size must be greater than zero", "");
        return 1u;
    }

    if (IsOutputHLSL(getOutputType()))
    {
        // The size of arrays is restricted here to prevent issues further down the
        // compiler/translator/driver stack. Shader Model 5 generation hardware is limited to
        // 4096 registers so this should be reasonable even for aggressively optimizable code.
        const unsigned int sizeLimit = 65536;

        if (size > sizeLimit)
        {
            error(line, "array size too large", "");
            return 1u;
        }
    }

    return size;
}

// See if this qualifier can be an array.
bool TParseContext::checkIsValidQualifierForArray(const TSourceLoc &line,
                                                  const TPublicType &elementQualifier)
{
    if ((elementQualifier.qualifier == EvqAttribute) ||
        (elementQualifier.qualifier == EvqVertexIn) ||
        (elementQualifier.qualifier == EvqConst && mShaderVersion < 300))
    {
        error(line, "cannot declare arrays of this qualifier",
              TType(elementQualifier).getQualifierString());
        return false;
    }

    return true;
}

// See if this element type can be formed into an array.
bool TParseContext::checkArrayElementIsNotArray(const TSourceLoc &line,
                                                const TPublicType &elementType)
{
    if (mShaderVersion < 310 && elementType.isArray())
    {
        TInfoSinkBase typeString;
        typeString << TType(elementType);
        error(line, "cannot declare arrays of arrays", typeString.c_str());
        return false;
    }
    return true;
}

// Check if this qualified element type can be formed into an array. This is only called when array
// brackets are associated with an identifier in a declaration, like this:
//   float a[2];
// Similar checks are done in addFullySpecifiedType for array declarations where the array brackets
// are associated with the type, like this:
//   float[2] a;
bool TParseContext::checkIsValidTypeAndQualifierForArray(const TSourceLoc &indexLocation,
                                                         const TPublicType &elementType)
{
    if (!checkArrayElementIsNotArray(indexLocation, elementType))
    {
        return false;
    }
    // In ESSL1.00 shaders, structs cannot be varying (section 4.3.5). This is checked elsewhere.
    // In ESSL3.00 shaders, struct inputs/outputs are allowed but not arrays of structs (section
    // 4.3.4).
    // Geometry shader requires each user-defined input be declared as arrays or inside input
    // blocks declared as arrays (GL_EXT_geometry_shader section 11.1gs.4.3). For the purposes of
    // interface matching, such variables and blocks are treated as though they were not declared
    // as arrays (GL_EXT_geometry_shader section 7.4.1).
    if (mShaderVersion >= 300 && elementType.getBasicType() == EbtStruct &&
        sh::IsVarying(elementType.qualifier) &&
        !IsGeometryShaderInput(mShaderType, elementType.qualifier))
    {
        TInfoSinkBase typeString;
        typeString << TType(elementType);
        error(indexLocation, "cannot declare arrays of structs of this qualifier",
              typeString.c_str());
        return false;
    }
    return checkIsValidQualifierForArray(indexLocation, elementType);
}

// Enforce non-initializer type/qualifier rules.
void TParseContext::checkCanBeDeclaredWithoutInitializer(const TSourceLoc &line,
                                                         const ImmutableString &identifier,
                                                         TType *type)
{
    ASSERT(type != nullptr);
    if (type->getQualifier() == EvqConst)
    {
        // Make the qualifier make sense.
        type->setQualifier(EvqTemporary);

        // Generate informative error messages for ESSL1.
        // In ESSL3 arrays and structures containing arrays can be constant.
        if (mShaderVersion < 300 && type->isStructureContainingArrays())
        {
            error(line,
                  "structures containing arrays may not be declared constant since they cannot be "
                  "initialized",
                  identifier);
        }
        else
        {
            error(line, "variables with qualifier 'const' must be initialized", identifier);
        }
    }
    // This will make the type sized if it isn't sized yet.
    checkIsNotUnsizedArray(line, "implicitly sized arrays need to be initialized", identifier,
                           type);
}

// Do some simple checks that are shared between all variable declarations,
// and update the symbol table.
//
// Returns true if declaring the variable succeeded.
//
bool TParseContext::declareVariable(const TSourceLoc &line,
                                    const ImmutableString &identifier,
                                    const TType *type,
                                    TVariable **variable)
{
    ASSERT((*variable) == nullptr);

    (*variable) = new TVariable(&symbolTable, identifier, type, SymbolType::UserDefined);

    ASSERT(type->getLayoutQualifier().index == -1 ||
           (isExtensionEnabled(TExtension::EXT_blend_func_extended) &&
            mShaderType == GL_FRAGMENT_SHADER && mShaderVersion >= 300));
    if (type->getQualifier() == EvqFragmentOut)
    {
        if (type->getLayoutQualifier().index != -1 && type->getLayoutQualifier().location == -1)
        {
            error(line,
                  "If index layout qualifier is specified for a fragment output, location must "
                  "also be specified.",
                  "index");
            return false;
        }
    }
    else
    {
        checkIndexIsNotSpecified(line, type->getLayoutQualifier().index);
    }

    checkBindingIsValid(line, *type);

    bool needsReservedCheck = true;

    // gl_LastFragData may be redeclared with a new precision qualifier
    if (type->isArray() && identifier.beginsWith("gl_LastFragData"))
    {
        const TVariable *maxDrawBuffers = static_cast<const TVariable *>(
            symbolTable.findBuiltIn(ImmutableString("gl_MaxDrawBuffers"), mShaderVersion));
        if (type->isArrayOfArrays())
        {
            error(line, "redeclaration of gl_LastFragData as an array of arrays", identifier);
            return false;
        }
        else if (static_cast<int>(type->getOutermostArraySize()) ==
                 maxDrawBuffers->getConstPointer()->getIConst())
        {
            if (const TSymbol *builtInSymbol = symbolTable.findBuiltIn(identifier, mShaderVersion))
            {
                needsReservedCheck = !checkCanUseExtension(line, builtInSymbol->extension());
            }
        }
        else
        {
            error(line, "redeclaration of gl_LastFragData with size != gl_MaxDrawBuffers",
                  identifier);
            return false;
        }
    }
    else if (type->isArray() && identifier == "gl_ClipDistance")
    {
        // gl_ClipDistance can be redeclared with smaller size than gl_MaxClipDistances
        const TVariable *maxClipDistances = static_cast<const TVariable *>(
            symbolTable.findBuiltIn(ImmutableString("gl_MaxClipDistances"), mShaderVersion));
        if (!maxClipDistances)
        {
            // Unsupported extension
            needsReservedCheck = true;
        }
        else if (type->isArrayOfArrays())
        {
            error(line, "redeclaration of gl_ClipDistance as an array of arrays", identifier);
            return false;
        }
        else if (static_cast<int>(type->getOutermostArraySize()) <=
                 maxClipDistances->getConstPointer()->getIConst())
        {
            if (const TSymbol *builtInSymbol = symbolTable.findBuiltIn(identifier, mShaderVersion))
            {
                needsReservedCheck = !checkCanUseExtension(line, builtInSymbol->extension());
            }
        }
        else
        {
            error(line, "redeclaration of gl_ClipDistance with size > gl_MaxClipDistances",
                  identifier);
            return false;
        }
    }

    if (needsReservedCheck && !checkIsNotReserved(line, identifier))
        return false;

    if (!symbolTable.declare(*variable))
    {
        error(line, "redefinition", identifier);
        return false;
    }

    if (!checkIsNonVoid(line, identifier, type->getBasicType()))
        return false;

    return true;
}

void TParseContext::checkIsParameterQualifierValid(
    const TSourceLoc &line,
    const TTypeQualifierBuilder &typeQualifierBuilder,
    TType *type)
{
    // The only parameter qualifiers a parameter can have are in, out, inout or const.
    TTypeQualifier typeQualifier = typeQualifierBuilder.getParameterTypeQualifier(mDiagnostics);

    if (typeQualifier.qualifier == EvqOut || typeQualifier.qualifier == EvqInOut)
    {
        checkOutParameterIsNotOpaqueType(line, typeQualifier.qualifier, *type);
    }

    if (!IsImage(type->getBasicType()))
    {
        checkMemoryQualifierIsNotSpecified(typeQualifier.memoryQualifier, line);
    }
    else
    {
        type->setMemoryQualifier(typeQualifier.memoryQualifier);
    }

    type->setQualifier(typeQualifier.qualifier);

    if (typeQualifier.precision != EbpUndefined)
    {
        type->setPrecision(typeQualifier.precision);
    }
}

template <size_t size>
bool TParseContext::checkCanUseOneOfExtensions(const TSourceLoc &line,
                                               const std::array<TExtension, size> &extensions)
{
    ASSERT(!extensions.empty());
    const TExtensionBehavior &extBehavior = extensionBehavior();

    bool canUseWithWarning    = false;
    bool canUseWithoutWarning = false;

    const char *errorMsgString   = "";
    TExtension errorMsgExtension = TExtension::UNDEFINED;

    for (TExtension extension : extensions)
    {
        auto extIter = extBehavior.find(extension);
        if (canUseWithWarning)
        {
            // We already have an extension that we can use, but with a warning.
            // See if we can use the alternative extension without a warning.
            if (extIter == extBehavior.end())
            {
                continue;
            }
            if (extIter->second == EBhEnable || extIter->second == EBhRequire)
            {
                canUseWithoutWarning = true;
                break;
            }
            continue;
        }
        if (extIter == extBehavior.end())
        {
            errorMsgString    = "extension is not supported";
            errorMsgExtension = extension;
        }
        else if (extIter->second == EBhUndefined || extIter->second == EBhDisable)
        {
            errorMsgString    = "extension is disabled";
            errorMsgExtension = extension;
        }
        else if (extIter->second == EBhWarn)
        {
            errorMsgExtension = extension;
            canUseWithWarning = true;
        }
        else
        {
            ASSERT(extIter->second == EBhEnable || extIter->second == EBhRequire);
            canUseWithoutWarning = true;
            break;
        }
    }

    if (canUseWithoutWarning)
    {
        return true;
    }
    if (canUseWithWarning)
    {
        warning(line, "extension is being used", GetExtensionNameString(errorMsgExtension));
        return true;
    }
    error(line, errorMsgString, GetExtensionNameString(errorMsgExtension));
    return false;
}

template bool TParseContext::checkCanUseOneOfExtensions(
    const TSourceLoc &line,
    const std::array<TExtension, 1> &extensions);
template bool TParseContext::checkCanUseOneOfExtensions(
    const TSourceLoc &line,
    const std::array<TExtension, 2> &extensions);
template bool TParseContext::checkCanUseOneOfExtensions(
    const TSourceLoc &line,
    const std::array<TExtension, 3> &extensions);

bool TParseContext::checkCanUseExtension(const TSourceLoc &line, TExtension extension)
{
    ASSERT(extension != TExtension::UNDEFINED);
    return checkCanUseOneOfExtensions(line, std::array<TExtension, 1u>{{extension}});
}

// ESSL 3.00.6 section 4.8 Empty Declarations: "The combinations of qualifiers that cause
// compile-time or link-time errors are the same whether or not the declaration is empty".
// This function implements all the checks that are done on qualifiers regardless of if the
// declaration is empty.
void TParseContext::declarationQualifierErrorCheck(const sh::TQualifier qualifier,
                                                   const sh::TLayoutQualifier &layoutQualifier,
                                                   const TSourceLoc &location)
{
    if (qualifier == EvqShared && !layoutQualifier.isEmpty())
    {
        error(location, "Shared memory declarations cannot have layout specified", "layout");
    }

    if (layoutQualifier.matrixPacking != EmpUnspecified)
    {
        error(location, "layout qualifier only valid for interface blocks",
              getMatrixPackingString(layoutQualifier.matrixPacking));
        return;
    }

    if (layoutQualifier.blockStorage != EbsUnspecified)
    {
        error(location, "layout qualifier only valid for interface blocks",
              getBlockStorageString(layoutQualifier.blockStorage));
        return;
    }

    if (qualifier == EvqFragmentOut)
    {
        if (layoutQualifier.location != -1 && layoutQualifier.yuv == true)
        {
            error(location, "invalid layout qualifier combination", "yuv");
            return;
        }
    }
    else
    {
        checkYuvIsNotSpecified(location, layoutQualifier.yuv);
    }

    // If multiview extension is enabled, "in" qualifier is allowed in the vertex shader in previous
    // parsing steps. So it needs to be checked here.
    if (anyMultiviewExtensionAvailable() && mShaderVersion < 300 && qualifier == EvqVertexIn)
    {
        error(location, "storage qualifier supported in GLSL ES 3.00 and above only", "in");
    }

    bool canHaveLocation = qualifier == EvqVertexIn || qualifier == EvqFragmentOut;
    if (mShaderVersion >= 310)
    {
        canHaveLocation = canHaveLocation || qualifier == EvqUniform || IsVarying(qualifier);
        // We're not checking whether the uniform location is in range here since that depends on
        // the type of the variable.
        // The type can only be fully determined for non-empty declarations.
    }
    if (!canHaveLocation)
    {
        checkLocationIsNotSpecified(location, layoutQualifier);
    }
}

void TParseContext::atomicCounterQualifierErrorCheck(const TPublicType &publicType,
                                                     const TSourceLoc &location)
{
    if (publicType.precision != EbpHigh)
    {
        error(location, "Can only be highp", "atomic counter");
    }
    // dEQP enforces compile error if location is specified. See uniform_location.test.
    if (publicType.layoutQualifier.location != -1)
    {
        error(location, "location must not be set for atomic_uint", "layout");
    }
    if (publicType.layoutQualifier.binding == -1)
    {
        error(location, "no binding specified", "atomic counter");
    }
}

void TParseContext::emptyDeclarationErrorCheck(const TType &type, const TSourceLoc &location)
{
    if (type.isUnsizedArray())
    {
        // ESSL3 spec section 4.1.9: Array declaration which leaves the size unspecified is an
        // error. It is assumed that this applies to empty declarations as well.
        error(location, "empty array declaration needs to specify a size", "");
    }

    if (type.getQualifier() != EvqFragmentOut)
    {
        checkIndexIsNotSpecified(location, type.getLayoutQualifier().index);
    }
}

// These checks are done for all declarations that are non-empty. They're done for non-empty
// declarations starting a declarator list, and declarators that follow an empty declaration.
void TParseContext::nonEmptyDeclarationErrorCheck(const TPublicType &publicType,
                                                  const TSourceLoc &identifierLocation)
{
    switch (publicType.qualifier)
    {
        case EvqVaryingIn:
        case EvqVaryingOut:
        case EvqAttribute:
        case EvqVertexIn:
        case EvqFragmentOut:
        case EvqComputeIn:
            if (publicType.getBasicType() == EbtStruct)
            {
                error(identifierLocation, "cannot be used with a structure",
                      getQualifierString(publicType.qualifier));
                return;
            }
            break;
        case EvqBuffer:
            if (publicType.getBasicType() != EbtInterfaceBlock)
            {
                error(identifierLocation,
                      "cannot declare buffer variables at global scope(outside a block)",
                      getQualifierString(publicType.qualifier));
                return;
            }
            break;
        default:
            break;
    }
    std::string reason(getBasicString(publicType.getBasicType()));
    reason += "s must be uniform";
    if (publicType.qualifier != EvqUniform &&
        !checkIsNotOpaqueType(identifierLocation, publicType.typeSpecifierNonArray, reason.c_str()))
    {
        return;
    }

    if ((publicType.qualifier != EvqTemporary && publicType.qualifier != EvqGlobal &&
         publicType.qualifier != EvqConst) &&
        publicType.getBasicType() == EbtYuvCscStandardEXT)
    {
        error(identifierLocation, "cannot be used with a yuvCscStandardEXT",
              getQualifierString(publicType.qualifier));
        return;
    }

    if (mShaderVersion >= 310 && publicType.qualifier == EvqUniform)
    {
        // Valid uniform declarations can't be unsized arrays since uniforms can't be initialized.
        // But invalid shaders may still reach here with an unsized array declaration.
        TType type(publicType);
        if (!type.isUnsizedArray())
        {
            checkUniformLocationInRange(identifierLocation, type.getLocationCount(),
                                        publicType.layoutQualifier);
        }
    }

    // check for layout qualifier issues
    const TLayoutQualifier layoutQualifier = publicType.layoutQualifier;

    if (IsImage(publicType.getBasicType()))
    {

        switch (layoutQualifier.imageInternalFormat)
        {
            case EiifRGBA32F:
            case EiifRGBA16F:
            case EiifR32F:
            case EiifRGBA8:
            case EiifRGBA8_SNORM:
                if (!IsFloatImage(publicType.getBasicType()))
                {
                    error(identifierLocation,
                          "internal image format requires a floating image type",
                          getBasicString(publicType.getBasicType()));
                    return;
                }
                break;
            case EiifRGBA32I:
            case EiifRGBA16I:
            case EiifRGBA8I:
            case EiifR32I:
                if (!IsIntegerImage(publicType.getBasicType()))
                {
                    error(identifierLocation,
                          "internal image format requires an integer image type",
                          getBasicString(publicType.getBasicType()));
                    return;
                }
                break;
            case EiifRGBA32UI:
            case EiifRGBA16UI:
            case EiifRGBA8UI:
            case EiifR32UI:
                if (!IsUnsignedImage(publicType.getBasicType()))
                {
                    error(identifierLocation,
                          "internal image format requires an unsigned image type",
                          getBasicString(publicType.getBasicType()));
                    return;
                }
                break;
            case EiifUnspecified:
                error(identifierLocation, "layout qualifier", "No image internal format specified");
                return;
            default:
                error(identifierLocation, "layout qualifier", "unrecognized token");
                return;
        }

        // GLSL ES 3.10 Revision 4, 4.9 Memory Access Qualifiers
        switch (layoutQualifier.imageInternalFormat)
        {
            case EiifR32F:
            case EiifR32I:
            case EiifR32UI:
                break;
            default:
                if (!publicType.memoryQualifier.readonly && !publicType.memoryQualifier.writeonly)
                {
                    error(identifierLocation, "layout qualifier",
                          "Except for images with the r32f, r32i and r32ui format qualifiers, "
                          "image variables must be qualified readonly and/or writeonly");
                    return;
                }
                break;
        }
    }
    else
    {
        checkInternalFormatIsNotSpecified(identifierLocation, layoutQualifier.imageInternalFormat);
        checkMemoryQualifierIsNotSpecified(publicType.memoryQualifier, identifierLocation);
    }

    if (IsAtomicCounter(publicType.getBasicType()))
    {
        atomicCounterQualifierErrorCheck(publicType, identifierLocation);
    }
    else
    {
        checkOffsetIsNotSpecified(identifierLocation, layoutQualifier.offset);
    }
}

void TParseContext::checkBindingIsValid(const TSourceLoc &identifierLocation, const TType &type)
{
    TLayoutQualifier layoutQualifier = type.getLayoutQualifier();
    // Note that the ESSL 3.10 section 4.4.5 is not particularly clear on how the binding qualifier
    // on arrays of arrays should be handled. We interpret the spec so that the binding value is
    // incremented for each element of the innermost nested arrays. This is in line with how arrays
    // of arrays of blocks are specified to behave in GLSL 4.50 and a conservative interpretation
    // when it comes to which shaders are accepted by the compiler.
    int arrayTotalElementCount = type.getArraySizeProduct();
    if (IsImage(type.getBasicType()))
    {
        checkImageBindingIsValid(identifierLocation, layoutQualifier.binding,
                                 arrayTotalElementCount);
    }
    else if (IsSampler(type.getBasicType()))
    {
        checkSamplerBindingIsValid(identifierLocation, layoutQualifier.binding,
                                   arrayTotalElementCount);
    }
    else if (IsAtomicCounter(type.getBasicType()))
    {
        checkAtomicCounterBindingIsValid(identifierLocation, layoutQualifier.binding);
    }
    else
    {
        ASSERT(!IsOpaqueType(type.getBasicType()));
        checkBindingIsNotSpecified(identifierLocation, layoutQualifier.binding);
    }
}

void TParseContext::checkLayoutQualifierSupported(const TSourceLoc &location,
                                                  const ImmutableString &layoutQualifierName,
                                                  int versionRequired)
{

    if (mShaderVersion < versionRequired)
    {
        error(location, "invalid layout qualifier: not supported", layoutQualifierName);
    }
}

bool TParseContext::checkWorkGroupSizeIsNotSpecified(const TSourceLoc &location,
                                                     const TLayoutQualifier &layoutQualifier)
{
    const sh::WorkGroupSize &localSize = layoutQualifier.localSize;
    for (size_t i = 0u; i < localSize.size(); ++i)
    {
        if (localSize[i] != -1)
        {
            error(location,
                  "invalid layout qualifier: only valid when used with 'in' in a compute shader "
                  "global layout declaration",
                  getWorkGroupSizeString(i));
            return false;
        }
    }

    return true;
}

void TParseContext::checkInternalFormatIsNotSpecified(const TSourceLoc &location,
                                                      TLayoutImageInternalFormat internalFormat)
{
    if (internalFormat != EiifUnspecified)
    {
        error(location, "invalid layout qualifier: only valid when used with images",
              getImageInternalFormatString(internalFormat));
    }
}

void TParseContext::checkIndexIsNotSpecified(const TSourceLoc &location, int index)
{
    if (index != -1)
    {
        error(location,
              "invalid layout qualifier: only valid when used with a fragment shader output in "
              "ESSL version >= 3.00 and EXT_blend_func_extended is enabled",
              "index");
    }
}

void TParseContext::checkBindingIsNotSpecified(const TSourceLoc &location, int binding)
{
    if (binding != -1)
    {
        error(location,
              "invalid layout qualifier: only valid when used with opaque types or blocks",
              "binding");
    }
}

void TParseContext::checkOffsetIsNotSpecified(const TSourceLoc &location, int offset)
{
    if (offset != -1)
    {
        error(location, "invalid layout qualifier: only valid when used with atomic counters",
              "offset");
    }
}

void TParseContext::checkImageBindingIsValid(const TSourceLoc &location,
                                             int binding,
                                             int arrayTotalElementCount)
{
    // Expects arraySize to be 1 when setting binding for only a single variable.
    if (binding >= 0 && binding + arrayTotalElementCount > mMaxImageUnits)
    {
        error(location, "image binding greater than gl_MaxImageUnits", "binding");
    }
}

void TParseContext::checkSamplerBindingIsValid(const TSourceLoc &location,
                                               int binding,
                                               int arrayTotalElementCount)
{
    // Expects arraySize to be 1 when setting binding for only a single variable.
    if (binding >= 0 && binding + arrayTotalElementCount > mMaxCombinedTextureImageUnits)
    {
        error(location, "sampler binding greater than maximum texture units", "binding");
    }
}

void TParseContext::checkBlockBindingIsValid(const TSourceLoc &location,
                                             const TQualifier &qualifier,
                                             int binding,
                                             int arraySize)
{
    int size = (arraySize == 0 ? 1 : arraySize);
    if (qualifier == EvqUniform)
    {
        if (binding + size > mMaxUniformBufferBindings)
        {
            error(location, "uniform block binding greater than MAX_UNIFORM_BUFFER_BINDINGS",
                  "binding");
        }
    }
    else if (qualifier == EvqBuffer)
    {
        if (binding + size > mMaxShaderStorageBufferBindings)
        {
            error(location,
                  "shader storage block binding greater than MAX_SHADER_STORAGE_BUFFER_BINDINGS",
                  "binding");
        }
    }
}
void TParseContext::checkAtomicCounterBindingIsValid(const TSourceLoc &location, int binding)
{
    if (binding >= mMaxAtomicCounterBindings)
    {
        error(location, "atomic counter binding greater than gl_MaxAtomicCounterBindings",
              "binding");
    }
}

void TParseContext::checkUniformLocationInRange(const TSourceLoc &location,
                                                int objectLocationCount,
                                                const TLayoutQualifier &layoutQualifier)
{
    int loc = layoutQualifier.location;
    if (loc >= 0 && loc + objectLocationCount > mMaxUniformLocations)
    {
        error(location, "Uniform location out of range", "location");
    }
}

void TParseContext::checkYuvIsNotSpecified(const TSourceLoc &location, bool yuv)
{
    if (yuv != false)
    {
        error(location, "invalid layout qualifier: only valid on program outputs", "yuv");
    }
}

void TParseContext::functionCallRValueLValueErrorCheck(const TFunction *fnCandidate,
                                                       TIntermAggregate *fnCall)
{
    for (size_t i = 0; i < fnCandidate->getParamCount(); ++i)
    {
        TQualifier qual        = fnCandidate->getParam(i)->getType().getQualifier();
        TIntermTyped *argument = (*(fnCall->getSequence()))[i]->getAsTyped();
        bool argumentIsRead = (IsQualifierUnspecified(qual) || qual == EvqIn || qual == EvqInOut ||
                               qual == EvqConstReadOnly);
        if (argumentIsRead)
        {
            markStaticReadIfSymbol(argument);
            if (!IsImage(argument->getBasicType()))
            {
                if (argument->getMemoryQualifier().writeonly)
                {
                    error(argument->getLine(),
                          "Writeonly value cannot be passed for 'in' or 'inout' parameters.",
                          fnCall->functionName());
                    return;
                }
            }
        }
        if (qual == EvqOut || qual == EvqInOut)
        {
            if (!checkCanBeLValue(argument->getLine(), "assign", argument))
            {
                error(argument->getLine(),
                      "Constant value cannot be passed for 'out' or 'inout' parameters.",
                      fnCall->functionName());
                return;
            }
        }
    }
}

void TParseContext::checkInvariantVariableQualifier(bool invariant,
                                                    const TQualifier qualifier,
                                                    const TSourceLoc &invariantLocation)
{
    if (!invariant)
        return;

    if (mShaderVersion < 300)
    {
        // input variables in the fragment shader can be also qualified as invariant
        if (!sh::CanBeInvariantESSL1(qualifier))
        {
            error(invariantLocation, "Cannot be qualified as invariant.", "invariant");
        }
    }
    else
    {
        if (!sh::CanBeInvariantESSL3OrGreater(qualifier))
        {
            error(invariantLocation, "Cannot be qualified as invariant.", "invariant");
        }
    }
}

bool TParseContext::isExtensionEnabled(TExtension extension) const
{
    return IsExtensionEnabled(extensionBehavior(), extension);
}

void TParseContext::handleExtensionDirective(const TSourceLoc &loc,
                                             const char *extName,
                                             const char *behavior)
{
    angle::pp::SourceLocation srcLoc;
    srcLoc.file = loc.first_file;
    srcLoc.line = loc.first_line;
    mDirectiveHandler.handleExtension(srcLoc, extName, behavior);
}

void TParseContext::handlePragmaDirective(const TSourceLoc &loc,
                                          const char *name,
                                          const char *value,
                                          bool stdgl)
{
    angle::pp::SourceLocation srcLoc;
    srcLoc.file = loc.first_file;
    srcLoc.line = loc.first_line;
    mDirectiveHandler.handlePragma(srcLoc, name, value, stdgl);
}

sh::WorkGroupSize TParseContext::getComputeShaderLocalSize() const
{
    sh::WorkGroupSize result(-1);
    for (size_t i = 0u; i < result.size(); ++i)
    {
        if (mComputeShaderLocalSizeDeclared && mComputeShaderLocalSize[i] == -1)
        {
            result[i] = 1;
        }
        else
        {
            result[i] = mComputeShaderLocalSize[i];
        }
    }
    return result;
}

TIntermConstantUnion *TParseContext::addScalarLiteral(const TConstantUnion *constantUnion,
                                                      const TSourceLoc &line)
{
    TIntermConstantUnion *node = new TIntermConstantUnion(
        constantUnion, TType(constantUnion->getType(), EbpUndefined, EvqConst));
    node->setLine(line);
    return node;
}

/////////////////////////////////////////////////////////////////////////////////
//
// Non-Errors.
//
/////////////////////////////////////////////////////////////////////////////////

const TVariable *TParseContext::getNamedVariable(const TSourceLoc &location,
                                                 const ImmutableString &name,
                                                 const TSymbol *symbol)
{
    if (!symbol)
    {
        error(location, "undeclared identifier", name);
        return nullptr;
    }

    if (!symbol->isVariable())
    {
        error(location, "variable expected", name);
        return nullptr;
    }

    const TVariable *variable = static_cast<const TVariable *>(symbol);

    if (variable->extension() != TExtension::UNDEFINED)
    {
        checkCanUseExtension(location, variable->extension());
    }

    // GLSL ES 3.1 Revision 4, 7.1.3 Compute Shader Special Variables
    if (getShaderType() == GL_COMPUTE_SHADER && !mComputeShaderLocalSizeDeclared &&
        variable->getType().getQualifier() == EvqWorkGroupSize)
    {
        error(location,
              "It is an error to use gl_WorkGroupSize before declaring the local group size",
              "gl_WorkGroupSize");
    }
    return variable;
}

TIntermTyped *TParseContext::parseVariableIdentifier(const TSourceLoc &location,
                                                     const ImmutableString &name,
                                                     const TSymbol *symbol)
{
    const TVariable *variable = getNamedVariable(location, name, symbol);

    if (!variable)
    {
        TIntermTyped *node = CreateZeroNode(TType(EbtFloat, EbpHigh, EvqConst));
        node->setLine(location);
        return node;
    }

    const TType &variableType = variable->getType();
    TIntermTyped *node        = nullptr;

    if (variable->getConstPointer() && variableType.canReplaceWithConstantUnion())
    {
        const TConstantUnion *constArray = variable->getConstPointer();
        node                             = new TIntermConstantUnion(constArray, variableType);
    }
    else if (variableType.getQualifier() == EvqWorkGroupSize && mComputeShaderLocalSizeDeclared)
    {
        // gl_WorkGroupSize can be used to size arrays according to the ESSL 3.10.4 spec, so it
        // needs to be added to the AST as a constant and not as a symbol.
        sh::WorkGroupSize workGroupSize = getComputeShaderLocalSize();
        TConstantUnion *constArray      = new TConstantUnion[3];
        for (size_t i = 0; i < 3; ++i)
        {
            constArray[i].setUConst(static_cast<unsigned int>(workGroupSize[i]));
        }

        ASSERT(variableType.getBasicType() == EbtUInt);
        ASSERT(variableType.getObjectSize() == 3);

        TType type(variableType);
        type.setQualifier(EvqConst);
        node = new TIntermConstantUnion(constArray, type);
    }
    else if ((mGeometryShaderInputPrimitiveType != EptUndefined) &&
             (variableType.getQualifier() == EvqPerVertexIn))
    {
        ASSERT(symbolTable.getGlInVariableWithArraySize() != nullptr);
        node = new TIntermSymbol(symbolTable.getGlInVariableWithArraySize());
    }
    else
    {
        node = new TIntermSymbol(variable);
    }
    ASSERT(node != nullptr);
    node->setLine(location);
    return node;
}

// Initializers show up in several places in the grammar.  Have one set of
// code to handle them here.
//
// Returns true on success.
bool TParseContext::executeInitializer(const TSourceLoc &line,
                                       const ImmutableString &identifier,
                                       TType *type,
                                       TIntermTyped *initializer,
                                       TIntermBinary **initNode)
{
    ASSERT(initNode != nullptr);
    ASSERT(*initNode == nullptr);

    if (type->isUnsizedArray())
    {
        // In case initializer is not an array or type has more dimensions than initializer, this
        // will default to setting array sizes to 1. We have not checked yet whether the initializer
        // actually is an array or not. Having a non-array initializer for an unsized array will
        // result in an error later, so we don't generate an error message here.
        auto *arraySizes = initializer->getType().getArraySizes();
        type->sizeUnsizedArrays(arraySizes);
    }

    const TQualifier qualifier = type->getQualifier();

    bool constError = false;
    if (qualifier == EvqConst)
    {
        if (EvqConst != initializer->getType().getQualifier())
        {
            TInfoSinkBase reasonStream;
            reasonStream << "assigning non-constant to '" << *type << "'";
            error(line, reasonStream.c_str(), "=");

            // We're still going to declare the variable to avoid extra error messages.
            type->setQualifier(EvqTemporary);
            constError = true;
        }
    }

    TVariable *variable = nullptr;
    if (!declareVariable(line, identifier, type, &variable))
    {
        return false;
    }

    if (constError)
    {
        return false;
    }

    bool globalInitWarning = false;
    if (symbolTable.atGlobalLevel() &&
        !ValidateGlobalInitializer(initializer, mShaderVersion, sh::IsWebGLBasedSpec(mShaderSpec),
                                   &globalInitWarning))
    {
        // Error message does not completely match behavior with ESSL 1.00, but
        // we want to steer developers towards only using constant expressions.
        error(line, "global variable initializers must be constant expressions", "=");
        return false;
    }
    if (globalInitWarning)
    {
        warning(
            line,
            "global variable initializers should be constant expressions "
            "(uniforms and globals are allowed in global initializers for legacy compatibility)",
            "=");
    }

    // identifier must be of type constant, a global, or a temporary
    if ((qualifier != EvqTemporary) && (qualifier != EvqGlobal) && (qualifier != EvqConst))
    {
        error(line, " cannot initialize this type of qualifier ",
              variable->getType().getQualifierString());
        return false;
    }

    TIntermSymbol *intermSymbol = new TIntermSymbol(variable);
    intermSymbol->setLine(line);

    if (!binaryOpCommonCheck(EOpInitialize, intermSymbol, initializer, line))
    {
        assignError(line, "=", variable->getType(), initializer->getType());
        return false;
    }

    if (qualifier == EvqConst)
    {
        // Save the constant folded value to the variable if possible.
        const TConstantUnion *constArray = initializer->getConstantValue();
        if (constArray)
        {
            variable->shareConstPointer(constArray);
            if (initializer->getType().canReplaceWithConstantUnion())
            {
                ASSERT(*initNode == nullptr);
                return true;
            }
        }
    }

    *initNode = new TIntermBinary(EOpInitialize, intermSymbol, initializer);
    markStaticReadIfSymbol(initializer);
    (*initNode)->setLine(line);
    return true;
}

TIntermNode *TParseContext::addConditionInitializer(const TPublicType &pType,
                                                    const ImmutableString &identifier,
                                                    TIntermTyped *initializer,
                                                    const TSourceLoc &loc)
{
    checkIsScalarBool(loc, pType);
    TIntermBinary *initNode = nullptr;
    TType *type             = new TType(pType);
    if (executeInitializer(loc, identifier, type, initializer, &initNode))
    {
        // The initializer is valid. The init condition needs to have a node - either the
        // initializer node, or a constant node in case the initialized variable is const and won't
        // be recorded in the AST.
        if (initNode == nullptr)
        {
            return initializer;
        }
        else
        {
            TIntermDeclaration *declaration = new TIntermDeclaration();
            declaration->appendDeclarator(initNode);
            return declaration;
        }
    }
    return nullptr;
}

TIntermNode *TParseContext::addLoop(TLoopType type,
                                    TIntermNode *init,
                                    TIntermNode *cond,
                                    TIntermTyped *expr,
                                    TIntermNode *body,
                                    const TSourceLoc &line)
{
    TIntermNode *node       = nullptr;
    TIntermTyped *typedCond = nullptr;
    if (cond)
    {
        markStaticReadIfSymbol(cond);
        typedCond = cond->getAsTyped();
    }
    if (expr)
    {
        markStaticReadIfSymbol(expr);
    }
    // In case the loop body was not parsed as a block and contains a statement that simply refers
    // to a variable, we need to mark it as statically used.
    if (body)
    {
        markStaticReadIfSymbol(body);
    }
    if (cond == nullptr || typedCond)
    {
        if (type == ELoopDoWhile)
        {
            checkIsScalarBool(line, typedCond);
        }
        // In the case of other loops, it was checked before that the condition is a scalar boolean.
        ASSERT(mDiagnostics->numErrors() > 0 || typedCond == nullptr ||
               (typedCond->getBasicType() == EbtBool && !typedCond->isArray() &&
                !typedCond->isVector()));

        node = new TIntermLoop(type, init, typedCond, expr, EnsureBlock(body));
        node->setLine(line);
        return node;
    }

    ASSERT(type != ELoopDoWhile);

    TIntermDeclaration *declaration = cond->getAsDeclarationNode();
    ASSERT(declaration);
    TIntermBinary *declarator = declaration->getSequence()->front()->getAsBinaryNode();
    ASSERT(declarator->getLeft()->getAsSymbolNode());

    // The condition is a declaration. In the AST representation we don't support declarations as
    // loop conditions. Wrap the loop to a block that declares the condition variable and contains
    // the loop.
    TIntermBlock *block = new TIntermBlock();

    TIntermDeclaration *declareCondition = new TIntermDeclaration();
    declareCondition->appendDeclarator(declarator->getLeft()->deepCopy());
    block->appendStatement(declareCondition);

    TIntermBinary *conditionInit = new TIntermBinary(EOpAssign, declarator->getLeft()->deepCopy(),
                                                     declarator->getRight()->deepCopy());
    TIntermLoop *loop = new TIntermLoop(type, init, conditionInit, expr, EnsureBlock(body));
    block->appendStatement(loop);
    loop->setLine(line);
    block->setLine(line);
    return block;
}

TIntermNode *TParseContext::addIfElse(TIntermTyped *cond,
                                      TIntermNodePair code,
                                      const TSourceLoc &loc)
{
    bool isScalarBool = checkIsScalarBool(loc, cond);
    // In case the conditional statements were not parsed as blocks and contain a statement that
    // simply refers to a variable, we need to mark them as statically used.
    if (code.node1)
    {
        markStaticReadIfSymbol(code.node1);
    }
    if (code.node2)
    {
        markStaticReadIfSymbol(code.node2);
    }

    // For compile time constant conditions, prune the code now.
    if (isScalarBool && cond->getAsConstantUnion())
    {
        if (cond->getAsConstantUnion()->getBConst(0) == true)
        {
            return EnsureBlock(code.node1);
        }
        else
        {
            return EnsureBlock(code.node2);
        }
    }

    TIntermIfElse *node = new TIntermIfElse(cond, EnsureBlock(code.node1), EnsureBlock(code.node2));
    markStaticReadIfSymbol(cond);
    node->setLine(loc);

    return node;
}

void TParseContext::addFullySpecifiedType(TPublicType *typeSpecifier)
{
    checkPrecisionSpecified(typeSpecifier->getLine(), typeSpecifier->precision,
                            typeSpecifier->getBasicType());

    if (mShaderVersion < 300 && typeSpecifier->isArray())
    {
        error(typeSpecifier->getLine(), "not supported", "first-class array");
        typeSpecifier->clearArrayness();
    }
}

TPublicType TParseContext::addFullySpecifiedType(const TTypeQualifierBuilder &typeQualifierBuilder,
                                                 const TPublicType &typeSpecifier)
{
    TTypeQualifier typeQualifier = typeQualifierBuilder.getVariableTypeQualifier(mDiagnostics);

    TPublicType returnType     = typeSpecifier;
    returnType.qualifier       = typeQualifier.qualifier;
    returnType.invariant       = typeQualifier.invariant;
    returnType.layoutQualifier = typeQualifier.layoutQualifier;
    returnType.memoryQualifier = typeQualifier.memoryQualifier;
    returnType.precision       = typeSpecifier.precision;

    if (typeQualifier.precision != EbpUndefined)
    {
        returnType.precision = typeQualifier.precision;
    }

    checkPrecisionSpecified(typeSpecifier.getLine(), returnType.precision,
                            typeSpecifier.getBasicType());

    checkInvariantVariableQualifier(returnType.invariant, returnType.qualifier,
                                    typeSpecifier.getLine());

    checkWorkGroupSizeIsNotSpecified(typeSpecifier.getLine(), returnType.layoutQualifier);

    if (mShaderVersion < 300)
    {
        if (typeSpecifier.isArray())
        {
            error(typeSpecifier.getLine(), "not supported", "first-class array");
            returnType.clearArrayness();
        }

        if (returnType.qualifier == EvqAttribute &&
            (typeSpecifier.getBasicType() == EbtBool || typeSpecifier.getBasicType() == EbtInt))
        {
            error(typeSpecifier.getLine(), "cannot be bool or int",
                  getQualifierString(returnType.qualifier));
        }

        if ((returnType.qualifier == EvqVaryingIn || returnType.qualifier == EvqVaryingOut) &&
            (typeSpecifier.getBasicType() == EbtBool || typeSpecifier.getBasicType() == EbtInt))
        {
            error(typeSpecifier.getLine(), "cannot be bool or int",
                  getQualifierString(returnType.qualifier));
        }
    }
    else
    {
        if (!returnType.layoutQualifier.isEmpty())
        {
            checkIsAtGlobalLevel(typeSpecifier.getLine(), "layout");
        }
        if (sh::IsVarying(returnType.qualifier) || returnType.qualifier == EvqVertexIn ||
            returnType.qualifier == EvqFragmentOut)
        {
            checkInputOutputTypeIsValidES3(returnType.qualifier, typeSpecifier,
                                           typeSpecifier.getLine());
        }
        if (returnType.qualifier == EvqComputeIn)
        {
            error(typeSpecifier.getLine(), "'in' can be only used to specify the local group size",
                  "in");
        }
    }

    return returnType;
}

void TParseContext::checkInputOutputTypeIsValidES3(const TQualifier qualifier,
                                                   const TPublicType &type,
                                                   const TSourceLoc &qualifierLocation)
{
    // An input/output variable can never be bool or a sampler. Samplers are checked elsewhere.
    if (type.getBasicType() == EbtBool)
    {
        error(qualifierLocation, "cannot be bool", getQualifierString(qualifier));
    }

    // Specific restrictions apply for vertex shader inputs and fragment shader outputs.
    switch (qualifier)
    {
        case EvqVertexIn:
            // ESSL 3.00 section 4.3.4
            if (type.isArray())
            {
                error(qualifierLocation, "cannot be array", getQualifierString(qualifier));
            }
            // Vertex inputs with a struct type are disallowed in nonEmptyDeclarationErrorCheck
            return;
        case EvqFragmentOut:
            // ESSL 3.00 section 4.3.6
            if (type.typeSpecifierNonArray.isMatrix())
            {
                error(qualifierLocation, "cannot be matrix", getQualifierString(qualifier));
            }
            // Fragment outputs with a struct type are disallowed in nonEmptyDeclarationErrorCheck
            return;
        default:
            break;
    }

    // Vertex shader outputs / fragment shader inputs have a different, slightly more lenient set of
    // restrictions.
    bool typeContainsIntegers =
        (type.getBasicType() == EbtInt || type.getBasicType() == EbtUInt ||
         type.isStructureContainingType(EbtInt) || type.isStructureContainingType(EbtUInt));
    if (typeContainsIntegers && qualifier != EvqFlatIn && qualifier != EvqFlatOut)
    {
        error(qualifierLocation, "must use 'flat' interpolation here",
              getQualifierString(qualifier));
    }

    if (type.getBasicType() == EbtStruct)
    {
        // ESSL 3.00 sections 4.3.4 and 4.3.6.
        // These restrictions are only implied by the ESSL 3.00 spec, but
        // the ESSL 3.10 spec lists these restrictions explicitly.
        if (type.isArray())
        {
            error(qualifierLocation, "cannot be an array of structures",
                  getQualifierString(qualifier));
        }
        if (type.isStructureContainingArrays())
        {
            error(qualifierLocation, "cannot be a structure containing an array",
                  getQualifierString(qualifier));
        }
        if (type.isStructureContainingType(EbtStruct))
        {
            error(qualifierLocation, "cannot be a structure containing a structure",
                  getQualifierString(qualifier));
        }
        if (type.isStructureContainingType(EbtBool))
        {
            error(qualifierLocation, "cannot be a structure containing a bool",
                  getQualifierString(qualifier));
        }
    }
}

void TParseContext::checkLocalVariableConstStorageQualifier(const TQualifierWrapperBase &qualifier)
{
    if (qualifier.getType() == QtStorage)
    {
        const TStorageQualifierWrapper &storageQualifier =
            static_cast<const TStorageQualifierWrapper &>(qualifier);
        if (!declaringFunction() && storageQualifier.getQualifier() != EvqConst &&
            !symbolTable.atGlobalLevel())
        {
            error(storageQualifier.getLine(),
                  "Local variables can only use the const storage qualifier.",
                  storageQualifier.getQualifierString());
        }
    }
}

void TParseContext::checkMemoryQualifierIsNotSpecified(const TMemoryQualifier &memoryQualifier,
                                                       const TSourceLoc &location)
{
    const std::string reason(
        "Only allowed with shader storage blocks, variables declared within shader storage blocks "
        "and variables declared as image types.");
    if (memoryQualifier.readonly)
    {
        error(location, reason.c_str(), "readonly");
    }
    if (memoryQualifier.writeonly)
    {
        error(location, reason.c_str(), "writeonly");
    }
    if (memoryQualifier.coherent)
    {
        error(location, reason.c_str(), "coherent");
    }
    if (memoryQualifier.restrictQualifier)
    {
        error(location, reason.c_str(), "restrict");
    }
    if (memoryQualifier.volatileQualifier)
    {
        error(location, reason.c_str(), "volatile");
    }
}

// Make sure there is no offset overlapping, and store the newly assigned offset to "type" in
// intermediate tree.
void TParseContext::checkAtomicCounterOffsetDoesNotOverlap(bool forceAppend,
                                                           const TSourceLoc &loc,
                                                           TType *type)
{
    const size_t size = type->isArray() ? kAtomicCounterArrayStride * type->getArraySizeProduct()
                                        : kAtomicCounterSize;
    TLayoutQualifier layoutQualifier = type->getLayoutQualifier();
    auto &bindingState               = mAtomicCounterBindingStates[layoutQualifier.binding];
    int offset;
    if (layoutQualifier.offset == -1 || forceAppend)
    {
        offset = bindingState.appendSpan(size);
    }
    else
    {
        offset = bindingState.insertSpan(layoutQualifier.offset, size);
    }
    if (offset == -1)
    {
        error(loc, "Offset overlapping", "atomic counter");
        return;
    }
    layoutQualifier.offset = offset;
    type->setLayoutQualifier(layoutQualifier);
}

void TParseContext::checkAtomicCounterOffsetAlignment(const TSourceLoc &location, const TType &type)
{
    TLayoutQualifier layoutQualifier = type.getLayoutQualifier();

    // OpenGL ES 3.1 Table 6.5, Atomic counter offset must be a multiple of 4
    if (layoutQualifier.offset % 4 != 0)
    {
        error(location, "Offset must be multiple of 4", "atomic counter");
    }
}

void TParseContext::checkGeometryShaderInputAndSetArraySize(const TSourceLoc &location,
                                                            const ImmutableString &token,
                                                            TType *type)
{
    if (IsGeometryShaderInput(mShaderType, type->getQualifier()))
    {
        if (type->isArray() && type->getOutermostArraySize() == 0u)
        {
            // Set size for the unsized geometry shader inputs if they are declared after a valid
            // input primitive declaration.
            if (mGeometryShaderInputPrimitiveType != EptUndefined)
            {
                ASSERT(symbolTable.getGlInVariableWithArraySize() != nullptr);
                type->sizeOutermostUnsizedArray(
                    symbolTable.getGlInVariableWithArraySize()->getType().getOutermostArraySize());
            }
            else
            {
                // [GLSL ES 3.2 SPEC Chapter 4.4.1.2]
                // An input can be declared without an array size if there is a previous layout
                // which specifies the size.
                error(location,
                      "Missing a valid input primitive declaration before declaring an unsized "
                      "array input",
                      token);
            }
        }
        else if (type->isArray())
        {
            setGeometryShaderInputArraySize(type->getOutermostArraySize(), location);
        }
        else
        {
            error(location, "Geometry shader input variable must be declared as an array", token);
        }
    }
}

TIntermDeclaration *TParseContext::parseSingleDeclaration(
    TPublicType &publicType,
    const TSourceLoc &identifierOrTypeLocation,
    const ImmutableString &identifier)
{
    TType *type = new TType(publicType);
    if ((mCompileOptions & SH_FLATTEN_PRAGMA_STDGL_INVARIANT_ALL) &&
        mDirectiveHandler.pragma().stdgl.invariantAll)
    {
        TQualifier qualifier = type->getQualifier();

        // The directive handler has already taken care of rejecting invalid uses of this pragma
        // (for example, in ESSL 3.00 fragment shaders), so at this point, flatten it into all
        // affected variable declarations:
        //
        // 1. Built-in special variables which are inputs to the fragment shader. (These are handled
        // elsewhere, in TranslatorGLSL.)
        //
        // 2. Outputs from vertex shaders in ESSL 1.00 and 3.00 (EvqVaryingOut and EvqVertexOut). It
        // is actually less likely that there will be bugs in the handling of ESSL 3.00 shaders, but
        // the way this is currently implemented we have to enable this compiler option before
        // parsing the shader and determining the shading language version it uses. If this were
        // implemented as a post-pass, the workaround could be more targeted.
        if (qualifier == EvqVaryingOut || qualifier == EvqVertexOut)
        {
            type->setInvariant(true);
        }
    }

    checkGeometryShaderInputAndSetArraySize(identifierOrTypeLocation, identifier, type);

    declarationQualifierErrorCheck(publicType.qualifier, publicType.layoutQualifier,
                                   identifierOrTypeLocation);

    bool emptyDeclaration                  = (identifier == "");
    mDeferredNonEmptyDeclarationErrorCheck = emptyDeclaration;

    TIntermSymbol *symbol = nullptr;
    if (emptyDeclaration)
    {
        emptyDeclarationErrorCheck(*type, identifierOrTypeLocation);
        // In most cases we don't need to create a symbol node for an empty declaration.
        // But if the empty declaration is declaring a struct type, the symbol node will store that.
        if (type->getBasicType() == EbtStruct)
        {
            TVariable *emptyVariable =
                new TVariable(&symbolTable, kEmptyImmutableString, type, SymbolType::Empty);
            symbol = new TIntermSymbol(emptyVariable);
        }
        else if (IsAtomicCounter(publicType.getBasicType()))
        {
            setAtomicCounterBindingDefaultOffset(publicType, identifierOrTypeLocation);
        }
    }
    else
    {
        nonEmptyDeclarationErrorCheck(publicType, identifierOrTypeLocation);

        checkCanBeDeclaredWithoutInitializer(identifierOrTypeLocation, identifier, type);

        if (IsAtomicCounter(type->getBasicType()))
        {
            checkAtomicCounterOffsetDoesNotOverlap(false, identifierOrTypeLocation, type);

            checkAtomicCounterOffsetAlignment(identifierOrTypeLocation, *type);
        }

        TVariable *variable = nullptr;
        if (declareVariable(identifierOrTypeLocation, identifier, type, &variable))
        {
            symbol = new TIntermSymbol(variable);
        }
    }

    TIntermDeclaration *declaration = new TIntermDeclaration();
    declaration->setLine(identifierOrTypeLocation);
    if (symbol)
    {
        symbol->setLine(identifierOrTypeLocation);
        declaration->appendDeclarator(symbol);
    }
    return declaration;
}

TIntermDeclaration *TParseContext::parseSingleArrayDeclaration(
    TPublicType &elementType,
    const TSourceLoc &identifierLocation,
    const ImmutableString &identifier,
    const TSourceLoc &indexLocation,
    const TVector<unsigned int> &arraySizes)
{
    mDeferredNonEmptyDeclarationErrorCheck = false;

    declarationQualifierErrorCheck(elementType.qualifier, elementType.layoutQualifier,
                                   identifierLocation);

    nonEmptyDeclarationErrorCheck(elementType, identifierLocation);

    checkIsValidTypeAndQualifierForArray(indexLocation, elementType);

    TType *arrayType = new TType(elementType);
    arrayType->makeArrays(arraySizes);

    checkGeometryShaderInputAndSetArraySize(indexLocation, identifier, arrayType);

    checkCanBeDeclaredWithoutInitializer(identifierLocation, identifier, arrayType);

    if (IsAtomicCounter(arrayType->getBasicType()))
    {
        checkAtomicCounterOffsetDoesNotOverlap(false, identifierLocation, arrayType);

        checkAtomicCounterOffsetAlignment(identifierLocation, *arrayType);
    }

    TIntermDeclaration *declaration = new TIntermDeclaration();
    declaration->setLine(identifierLocation);

    TVariable *variable = nullptr;
    if (declareVariable(identifierLocation, identifier, arrayType, &variable))
    {
        TIntermSymbol *symbol = new TIntermSymbol(variable);
        symbol->setLine(identifierLocation);
        declaration->appendDeclarator(symbol);
    }

    return declaration;
}

TIntermDeclaration *TParseContext::parseSingleInitDeclaration(const TPublicType &publicType,
                                                              const TSourceLoc &identifierLocation,
                                                              const ImmutableString &identifier,
                                                              const TSourceLoc &initLocation,
                                                              TIntermTyped *initializer)
{
    mDeferredNonEmptyDeclarationErrorCheck = false;

    declarationQualifierErrorCheck(publicType.qualifier, publicType.layoutQualifier,
                                   identifierLocation);

    nonEmptyDeclarationErrorCheck(publicType, identifierLocation);

    TIntermDeclaration *declaration = new TIntermDeclaration();
    declaration->setLine(identifierLocation);

    TIntermBinary *initNode = nullptr;
    TType *type             = new TType(publicType);
    if (executeInitializer(identifierLocation, identifier, type, initializer, &initNode))
    {
        if (initNode)
        {
            declaration->appendDeclarator(initNode);
        }
    }
    return declaration;
}

TIntermDeclaration *TParseContext::parseSingleArrayInitDeclaration(
    TPublicType &elementType,
    const TSourceLoc &identifierLocation,
    const ImmutableString &identifier,
    const TSourceLoc &indexLocation,
    const TVector<unsigned int> &arraySizes,
    const TSourceLoc &initLocation,
    TIntermTyped *initializer)
{
    mDeferredNonEmptyDeclarationErrorCheck = false;

    declarationQualifierErrorCheck(elementType.qualifier, elementType.layoutQualifier,
                                   identifierLocation);

    nonEmptyDeclarationErrorCheck(elementType, identifierLocation);

    checkIsValidTypeAndQualifierForArray(indexLocation, elementType);

    TType *arrayType = new TType(elementType);
    arrayType->makeArrays(arraySizes);

    TIntermDeclaration *declaration = new TIntermDeclaration();
    declaration->setLine(identifierLocation);

    // initNode will correspond to the whole of "type b[n] = initializer".
    TIntermBinary *initNode = nullptr;
    if (executeInitializer(identifierLocation, identifier, arrayType, initializer, &initNode))
    {
        if (initNode)
        {
            declaration->appendDeclarator(initNode);
        }
    }

    return declaration;
}

TIntermInvariantDeclaration *TParseContext::parseInvariantDeclaration(
    const TTypeQualifierBuilder &typeQualifierBuilder,
    const TSourceLoc &identifierLoc,
    const ImmutableString &identifier,
    const TSymbol *symbol)
{
    TTypeQualifier typeQualifier = typeQualifierBuilder.getVariableTypeQualifier(mDiagnostics);

    if (!typeQualifier.invariant)
    {
        error(identifierLoc, "Expected invariant", identifier);
        return nullptr;
    }
    if (!checkIsAtGlobalLevel(identifierLoc, "invariant varying"))
    {
        return nullptr;
    }
    if (!symbol)
    {
        error(identifierLoc, "undeclared identifier declared as invariant", identifier);
        return nullptr;
    }
    if (!IsQualifierUnspecified(typeQualifier.qualifier))
    {
        error(identifierLoc, "invariant declaration specifies qualifier",
              getQualifierString(typeQualifier.qualifier));
    }
    if (typeQualifier.precision != EbpUndefined)
    {
        error(identifierLoc, "invariant declaration specifies precision",
              getPrecisionString(typeQualifier.precision));
    }
    if (!typeQualifier.layoutQualifier.isEmpty())
    {
        error(identifierLoc, "invariant declaration specifies layout", "'layout'");
    }

    const TVariable *variable = getNamedVariable(identifierLoc, identifier, symbol);
    if (!variable)
    {
        return nullptr;
    }
    const TType &type = variable->getType();

    checkInvariantVariableQualifier(typeQualifier.invariant, type.getQualifier(),
                                    typeQualifier.line);
    checkMemoryQualifierIsNotSpecified(typeQualifier.memoryQualifier, typeQualifier.line);

    symbolTable.addInvariantVarying(*variable);

    TIntermSymbol *intermSymbol = new TIntermSymbol(variable);
    intermSymbol->setLine(identifierLoc);

    return new TIntermInvariantDeclaration(intermSymbol, identifierLoc);
}

void TParseContext::parseDeclarator(TPublicType &publicType,
                                    const TSourceLoc &identifierLocation,
                                    const ImmutableString &identifier,
                                    TIntermDeclaration *declarationOut)
{
    // If the declaration starting this declarator list was empty (example: int,), some checks were
    // not performed.
    if (mDeferredNonEmptyDeclarationErrorCheck)
    {
        nonEmptyDeclarationErrorCheck(publicType, identifierLocation);
        mDeferredNonEmptyDeclarationErrorCheck = false;
    }

    checkDeclaratorLocationIsNotSpecified(identifierLocation, publicType);

    TType *type = new TType(publicType);

    checkGeometryShaderInputAndSetArraySize(identifierLocation, identifier, type);

    checkCanBeDeclaredWithoutInitializer(identifierLocation, identifier, type);

    if (IsAtomicCounter(type->getBasicType()))
    {
        checkAtomicCounterOffsetDoesNotOverlap(true, identifierLocation, type);

        checkAtomicCounterOffsetAlignment(identifierLocation, *type);
    }

    TVariable *variable = nullptr;
    if (declareVariable(identifierLocation, identifier, type, &variable))
    {
        TIntermSymbol *symbol = new TIntermSymbol(variable);
        symbol->setLine(identifierLocation);
        declarationOut->appendDeclarator(symbol);
    }
}

void TParseContext::parseArrayDeclarator(TPublicType &elementType,
                                         const TSourceLoc &identifierLocation,
                                         const ImmutableString &identifier,
                                         const TSourceLoc &arrayLocation,
                                         const TVector<unsigned int> &arraySizes,
                                         TIntermDeclaration *declarationOut)
{
    // If the declaration starting this declarator list was empty (example: int,), some checks were
    // not performed.
    if (mDeferredNonEmptyDeclarationErrorCheck)
    {
        nonEmptyDeclarationErrorCheck(elementType, identifierLocation);
        mDeferredNonEmptyDeclarationErrorCheck = false;
    }

    checkDeclaratorLocationIsNotSpecified(identifierLocation, elementType);

    if (checkIsValidTypeAndQualifierForArray(arrayLocation, elementType))
    {
        TType *arrayType = new TType(elementType);
        arrayType->makeArrays(arraySizes);

        checkGeometryShaderInputAndSetArraySize(identifierLocation, identifier, arrayType);

        checkCanBeDeclaredWithoutInitializer(identifierLocation, identifier, arrayType);

        if (IsAtomicCounter(arrayType->getBasicType()))
        {
            checkAtomicCounterOffsetDoesNotOverlap(true, identifierLocation, arrayType);

            checkAtomicCounterOffsetAlignment(identifierLocation, *arrayType);
        }

        TVariable *variable = nullptr;
        if (declareVariable(identifierLocation, identifier, arrayType, &variable))
        {
            TIntermSymbol *symbol = new TIntermSymbol(variable);
            symbol->setLine(identifierLocation);
            declarationOut->appendDeclarator(symbol);
        }
    }
}

void TParseContext::parseInitDeclarator(const TPublicType &publicType,
                                        const TSourceLoc &identifierLocation,
                                        const ImmutableString &identifier,
                                        const TSourceLoc &initLocation,
                                        TIntermTyped *initializer,
                                        TIntermDeclaration *declarationOut)
{
    // If the declaration starting this declarator list was empty (example: int,), some checks were
    // not performed.
    if (mDeferredNonEmptyDeclarationErrorCheck)
    {
        nonEmptyDeclarationErrorCheck(publicType, identifierLocation);
        mDeferredNonEmptyDeclarationErrorCheck = false;
    }

    checkDeclaratorLocationIsNotSpecified(identifierLocation, publicType);

    TIntermBinary *initNode = nullptr;
    TType *type             = new TType(publicType);
    if (executeInitializer(identifierLocation, identifier, type, initializer, &initNode))
    {
        //
        // build the intermediate representation
        //
        if (initNode)
        {
            declarationOut->appendDeclarator(initNode);
        }
    }
}

void TParseContext::parseArrayInitDeclarator(const TPublicType &elementType,
                                             const TSourceLoc &identifierLocation,
                                             const ImmutableString &identifier,
                                             const TSourceLoc &indexLocation,
                                             const TVector<unsigned int> &arraySizes,
                                             const TSourceLoc &initLocation,
                                             TIntermTyped *initializer,
                                             TIntermDeclaration *declarationOut)
{
    // If the declaration starting this declarator list was empty (example: int,), some checks were
    // not performed.
    if (mDeferredNonEmptyDeclarationErrorCheck)
    {
        nonEmptyDeclarationErrorCheck(elementType, identifierLocation);
        mDeferredNonEmptyDeclarationErrorCheck = false;
    }

    checkDeclaratorLocationIsNotSpecified(identifierLocation, elementType);

    checkIsValidTypeAndQualifierForArray(indexLocation, elementType);

    TType *arrayType = new TType(elementType);
    arrayType->makeArrays(arraySizes);

    // initNode will correspond to the whole of "b[n] = initializer".
    TIntermBinary *initNode = nullptr;
    if (executeInitializer(identifierLocation, identifier, arrayType, initializer, &initNode))
    {
        if (initNode)
        {
            declarationOut->appendDeclarator(initNode);
        }
    }
}

TIntermNode *TParseContext::addEmptyStatement(const TSourceLoc &location)
{
    // It's simpler to parse an empty statement as a constant expression rather than having a
    // different type of node just for empty statements, that will be pruned from the AST anyway.
    TIntermNode *node = CreateZeroNode(TType(EbtInt, EbpMedium));
    node->setLine(location);
    return node;
}

void TParseContext::setAtomicCounterBindingDefaultOffset(const TPublicType &publicType,
                                                         const TSourceLoc &location)
{
    const TLayoutQualifier &layoutQualifier = publicType.layoutQualifier;
    checkAtomicCounterBindingIsValid(location, layoutQualifier.binding);
    if (layoutQualifier.binding == -1 || layoutQualifier.offset == -1)
    {
        error(location, "Requires both binding and offset", "layout");
        return;
    }
    mAtomicCounterBindingStates[layoutQualifier.binding].setDefaultOffset(layoutQualifier.offset);
}

void TParseContext::parseDefaultPrecisionQualifier(const TPrecision precision,
                                                   const TPublicType &type,
                                                   const TSourceLoc &loc)
{
    if ((precision == EbpHigh) && (getShaderType() == GL_FRAGMENT_SHADER) &&
        !getFragmentPrecisionHigh())
    {
        error(loc, "precision is not supported in fragment shader", "highp");
    }

    if (!CanSetDefaultPrecisionOnType(type))
    {
        error(loc, "illegal type argument for default precision qualifier",
              getBasicString(type.getBasicType()));
        return;
    }
    symbolTable.setDefaultPrecision(type.getBasicType(), precision);
}

bool TParseContext::checkPrimitiveTypeMatchesTypeQualifier(const TTypeQualifier &typeQualifier)
{
    switch (typeQualifier.layoutQualifier.primitiveType)
    {
        case EptLines:
        case EptLinesAdjacency:
        case EptTriangles:
        case EptTrianglesAdjacency:
            return typeQualifier.qualifier == EvqGeometryIn;

        case EptLineStrip:
        case EptTriangleStrip:
            return typeQualifier.qualifier == EvqGeometryOut;

        case EptPoints:
            return true;

        default:
            UNREACHABLE();
            return false;
    }
}

void TParseContext::setGeometryShaderInputArraySize(unsigned int inputArraySize,
                                                    const TSourceLoc &line)
{
    if (!symbolTable.setGlInArraySize(inputArraySize))
    {
        error(line,
              "Array size or input primitive declaration doesn't match the size of earlier sized "
              "array inputs.",
              "layout");
    }
}

bool TParseContext::parseGeometryShaderInputLayoutQualifier(const TTypeQualifier &typeQualifier)
{
    ASSERT(typeQualifier.qualifier == EvqGeometryIn);

    const TLayoutQualifier &layoutQualifier = typeQualifier.layoutQualifier;

    if (layoutQualifier.maxVertices != -1)
    {
        error(typeQualifier.line,
              "max_vertices can only be declared in 'out' layout in a geometry shader", "layout");
        return false;
    }

    // Set mGeometryInputPrimitiveType if exists
    if (layoutQualifier.primitiveType != EptUndefined)
    {
        if (!checkPrimitiveTypeMatchesTypeQualifier(typeQualifier))
        {
            error(typeQualifier.line, "invalid primitive type for 'in' layout", "layout");
            return false;
        }

        if (mGeometryShaderInputPrimitiveType == EptUndefined)
        {
            mGeometryShaderInputPrimitiveType = layoutQualifier.primitiveType;
            setGeometryShaderInputArraySize(
                GetGeometryShaderInputArraySize(mGeometryShaderInputPrimitiveType),
                typeQualifier.line);
        }
        else if (mGeometryShaderInputPrimitiveType != layoutQualifier.primitiveType)
        {
            error(typeQualifier.line, "primitive doesn't match earlier input primitive declaration",
                  "layout");
            return false;
        }
    }

    // Set mGeometryInvocations if exists
    if (layoutQualifier.invocations > 0)
    {
        if (mGeometryShaderInvocations == 0)
        {
            mGeometryShaderInvocations = layoutQualifier.invocations;
        }
        else if (mGeometryShaderInvocations != layoutQualifier.invocations)
        {
            error(typeQualifier.line, "invocations contradicts to the earlier declaration",
                  "layout");
            return false;
        }
    }

    return true;
}

bool TParseContext::parseGeometryShaderOutputLayoutQualifier(const TTypeQualifier &typeQualifier)
{
    ASSERT(typeQualifier.qualifier == EvqGeometryOut);

    const TLayoutQualifier &layoutQualifier = typeQualifier.layoutQualifier;

    if (layoutQualifier.invocations > 0)
    {
        error(typeQualifier.line,
              "invocations can only be declared in 'in' layout in a geometry shader", "layout");
        return false;
    }

    // Set mGeometryOutputPrimitiveType if exists
    if (layoutQualifier.primitiveType != EptUndefined)
    {
        if (!checkPrimitiveTypeMatchesTypeQualifier(typeQualifier))
        {
            error(typeQualifier.line, "invalid primitive type for 'out' layout", "layout");
            return false;
        }

        if (mGeometryShaderOutputPrimitiveType == EptUndefined)
        {
            mGeometryShaderOutputPrimitiveType = layoutQualifier.primitiveType;
        }
        else if (mGeometryShaderOutputPrimitiveType != layoutQualifier.primitiveType)
        {
            error(typeQualifier.line,
                  "primitive doesn't match earlier output primitive declaration", "layout");
            return false;
        }
    }

    // Set mGeometryMaxVertices if exists
    if (layoutQualifier.maxVertices > -1)
    {
        if (mGeometryShaderMaxVertices == -1)
        {
            mGeometryShaderMaxVertices = layoutQualifier.maxVertices;
        }
        else if (mGeometryShaderMaxVertices != layoutQualifier.maxVertices)
        {
            error(typeQualifier.line, "max_vertices contradicts to the earlier declaration",
                  "layout");
            return false;
        }
    }

    return true;
}

void TParseContext::parseGlobalLayoutQualifier(const TTypeQualifierBuilder &typeQualifierBuilder)
{
    TTypeQualifier typeQualifier = typeQualifierBuilder.getVariableTypeQualifier(mDiagnostics);
    const TLayoutQualifier layoutQualifier = typeQualifier.layoutQualifier;

    checkInvariantVariableQualifier(typeQualifier.invariant, typeQualifier.qualifier,
                                    typeQualifier.line);

    // It should never be the case, but some strange parser errors can send us here.
    if (layoutQualifier.isEmpty())
    {
        error(typeQualifier.line, "Error during layout qualifier parsing.", "?");
        return;
    }

    if (!layoutQualifier.isCombinationValid())
    {
        error(typeQualifier.line, "invalid layout qualifier combination", "layout");
        return;
    }

    checkIndexIsNotSpecified(typeQualifier.line, layoutQualifier.index);

    checkBindingIsNotSpecified(typeQualifier.line, layoutQualifier.binding);

    checkMemoryQualifierIsNotSpecified(typeQualifier.memoryQualifier, typeQualifier.line);

    checkInternalFormatIsNotSpecified(typeQualifier.line, layoutQualifier.imageInternalFormat);

    checkYuvIsNotSpecified(typeQualifier.line, layoutQualifier.yuv);

    checkOffsetIsNotSpecified(typeQualifier.line, layoutQualifier.offset);

    checkStd430IsForShaderStorageBlock(typeQualifier.line, layoutQualifier.blockStorage,
                                       typeQualifier.qualifier);

    if (typeQualifier.qualifier == EvqComputeIn)
    {
        if (mComputeShaderLocalSizeDeclared &&
            !layoutQualifier.isLocalSizeEqual(mComputeShaderLocalSize))
        {
            error(typeQualifier.line, "Work group size does not match the previous declaration",
                  "layout");
            return;
        }

        if (mShaderVersion < 310)
        {
            error(typeQualifier.line, "in type qualifier supported in GLSL ES 3.10 only", "layout");
            return;
        }

        if (!layoutQualifier.localSize.isAnyValueSet())
        {
            error(typeQualifier.line, "No local work group size specified", "layout");
            return;
        }

        const TVariable *maxComputeWorkGroupSize = static_cast<const TVariable *>(
            symbolTable.findBuiltIn(ImmutableString("gl_MaxComputeWorkGroupSize"), mShaderVersion));

        const TConstantUnion *maxComputeWorkGroupSizeData =
            maxComputeWorkGroupSize->getConstPointer();

        for (size_t i = 0u; i < layoutQualifier.localSize.size(); ++i)
        {
            if (layoutQualifier.localSize[i] != -1)
            {
                mComputeShaderLocalSize[i]             = layoutQualifier.localSize[i];
                const int maxComputeWorkGroupSizeValue = maxComputeWorkGroupSizeData[i].getIConst();
                if (mComputeShaderLocalSize[i] < 1 ||
                    mComputeShaderLocalSize[i] > maxComputeWorkGroupSizeValue)
                {
                    std::stringstream reasonStream = sh::InitializeStream<std::stringstream>();
                    reasonStream << "invalid value: Value must be at least 1 and no greater than "
                                 << maxComputeWorkGroupSizeValue;
                    const std::string &reason = reasonStream.str();

                    error(typeQualifier.line, reason.c_str(), getWorkGroupSizeString(i));
                    return;
                }
            }
        }

        mComputeShaderLocalSizeDeclared = true;
    }
    else if (typeQualifier.qualifier == EvqGeometryIn)
    {
        if (mShaderVersion < 310)
        {
            error(typeQualifier.line, "in type qualifier supported in GLSL ES 3.10 only", "layout");
            return;
        }

        if (!parseGeometryShaderInputLayoutQualifier(typeQualifier))
        {
            return;
        }
    }
    else if (typeQualifier.qualifier == EvqGeometryOut)
    {
        if (mShaderVersion < 310)
        {
            error(typeQualifier.line, "out type qualifier supported in GLSL ES 3.10 only",
                  "layout");
            return;
        }

        if (!parseGeometryShaderOutputLayoutQualifier(typeQualifier))
        {
            return;
        }
    }
    else if (anyMultiviewExtensionAvailable() && typeQualifier.qualifier == EvqVertexIn)
    {
        // This error is only specified in WebGL, but tightens unspecified behavior in the native
        // specification.
        if (mNumViews != -1 && layoutQualifier.numViews != mNumViews)
        {
            error(typeQualifier.line, "Number of views does not match the previous declaration",
                  "layout");
            return;
        }

        if (layoutQualifier.numViews == -1)
        {
            error(typeQualifier.line, "No num_views specified", "layout");
            return;
        }

        if (layoutQualifier.numViews > mMaxNumViews)
        {
            error(typeQualifier.line, "num_views greater than the value of GL_MAX_VIEWS_OVR",
                  "layout");
            return;
        }

        mNumViews = layoutQualifier.numViews;
    }
    else
    {
        if (!checkWorkGroupSizeIsNotSpecified(typeQualifier.line, layoutQualifier))
        {
            return;
        }

        if (typeQualifier.qualifier != EvqUniform && typeQualifier.qualifier != EvqBuffer)
        {
            error(typeQualifier.line, "invalid qualifier: global layout can only be set for blocks",
                  getQualifierString(typeQualifier.qualifier));
            return;
        }

        if (mShaderVersion < 300)
        {
            error(typeQualifier.line, "layout qualifiers supported in GLSL ES 3.00 and above",
                  "layout");
            return;
        }

        checkLocationIsNotSpecified(typeQualifier.line, layoutQualifier);

        if (layoutQualifier.matrixPacking != EmpUnspecified)
        {
            if (typeQualifier.qualifier == EvqUniform)
            {
                mDefaultUniformMatrixPacking = layoutQualifier.matrixPacking;
            }
            else if (typeQualifier.qualifier == EvqBuffer)
            {
                mDefaultBufferMatrixPacking = layoutQualifier.matrixPacking;
            }
        }

        if (layoutQualifier.blockStorage != EbsUnspecified)
        {
            if (typeQualifier.qualifier == EvqUniform)
            {
                mDefaultUniformBlockStorage = layoutQualifier.blockStorage;
            }
            else if (typeQualifier.qualifier == EvqBuffer)
            {
                mDefaultBufferBlockStorage = layoutQualifier.blockStorage;
            }
        }
    }
}

TIntermFunctionPrototype *TParseContext::createPrototypeNodeFromFunction(
    const TFunction &function,
    const TSourceLoc &location,
    bool insertParametersToSymbolTable)
{
    checkIsNotReserved(location, function.name());

    TIntermFunctionPrototype *prototype = new TIntermFunctionPrototype(&function);
    prototype->setLine(location);

    for (size_t i = 0; i < function.getParamCount(); i++)
    {
        const TVariable *param = function.getParam(i);

        // If the parameter has no name, it's not an error, just don't add it to symbol table (could
        // be used for unused args).
        if (param->symbolType() != SymbolType::Empty)
        {
            if (insertParametersToSymbolTable)
            {
                if (!symbolTable.declare(const_cast<TVariable *>(param)))
                {
                    error(location, "redefinition", param->name());
                }
            }
            // Unsized type of a named parameter should have already been checked and sanitized.
            ASSERT(!param->getType().isUnsizedArray());
        }
        else
        {
            if (param->getType().isUnsizedArray())
            {
                error(location, "function parameter array must be sized at compile time", "[]");
                // We don't need to size the arrays since the parameter is unnamed and hence
                // inaccessible.
            }
        }
    }
    return prototype;
}

TIntermFunctionPrototype *TParseContext::addFunctionPrototypeDeclaration(
    const TFunction &parsedFunction,
    const TSourceLoc &location)
{
    // Note: function found from the symbol table could be the same as parsedFunction if this is the
    // first declaration. Either way the instance in the symbol table is used to track whether the
    // function is declared multiple times.
    bool hadPrototypeDeclaration = false;
    const TFunction *function    = symbolTable.markFunctionHasPrototypeDeclaration(
        parsedFunction.getMangledName(), &hadPrototypeDeclaration);

    if (hadPrototypeDeclaration && mShaderVersion == 100)
    {
        // ESSL 1.00.17 section 4.2.7.
        // Doesn't apply to ESSL 3.00.4: see section 4.2.3.
        error(location, "duplicate function prototype declarations are not allowed", "function");
    }

    TIntermFunctionPrototype *prototype =
        createPrototypeNodeFromFunction(*function, location, false);

    symbolTable.pop();

    if (!symbolTable.atGlobalLevel())
    {
        // ESSL 3.00.4 section 4.2.4.
        error(location, "local function prototype declarations are not allowed", "function");
    }

    return prototype;
}

TIntermFunctionDefinition *TParseContext::addFunctionDefinition(
    TIntermFunctionPrototype *functionPrototype,
    TIntermBlock *functionBody,
    const TSourceLoc &location)
{
    // Undo push at end of parseFunctionDefinitionHeader() below for ESSL1.00 case
    if (mFunctionBodyNewScope)
    {
        mFunctionBodyNewScope = false;
        symbolTable.pop();
    }

    // Check that non-void functions have at least one return statement.
    if (mCurrentFunctionType->getBasicType() != EbtVoid && !mFunctionReturnsValue)
    {
        error(location,
              "function does not return a value:", functionPrototype->getFunction()->name());
    }

    if (functionBody == nullptr)
    {
        functionBody = new TIntermBlock();
        functionBody->setLine(location);
    }
    TIntermFunctionDefinition *functionNode =
        new TIntermFunctionDefinition(functionPrototype, functionBody);
    functionNode->setLine(location);

    symbolTable.pop();
    return functionNode;
}

void TParseContext::parseFunctionDefinitionHeader(const TSourceLoc &location,
                                                  const TFunction *function,
                                                  TIntermFunctionPrototype **prototypeOut)
{
    ASSERT(function);

    bool wasDefined = false;
    function        = symbolTable.setFunctionParameterNamesFromDefinition(function, &wasDefined);
    if (wasDefined)
    {
        error(location, "function already has a body", function->name());
    }

    // Remember the return type for later checking for return statements.
    mCurrentFunctionType  = &(function->getReturnType());
    mFunctionReturnsValue = false;

    *prototypeOut = createPrototypeNodeFromFunction(*function, location, true);
    setLoopNestingLevel(0);

    // ESSL 1.00 spec allows for variable in function body to redefine parameter
    if (IsSpecWithFunctionBodyNewScope(mShaderSpec, mShaderVersion))
    {
        mFunctionBodyNewScope = true;
        symbolTable.push();
    }
}

TFunction *TParseContext::parseFunctionDeclarator(const TSourceLoc &location, TFunction *function)
{
    //
    // We don't know at this point whether this is a function definition or a prototype.
    // The definition production code will check for redefinitions.
    // In the case of ESSL 1.00 the prototype production code will also check for redeclarations.
    //

    for (size_t i = 0u; i < function->getParamCount(); ++i)
    {
        const TVariable *param = function->getParam(i);
        if (param->getType().isStructSpecifier())
        {
            // ESSL 3.00.6 section 12.10.
            error(location, "Function parameter type cannot be a structure definition",
                  function->name());
        }
    }

    if (getShaderVersion() >= 300)
    {

        if (symbolTable.isUnmangledBuiltInName(function->name(), getShaderVersion(),
                                               extensionBehavior()))
        {
            // With ESSL 3.00 and above, names of built-in functions cannot be redeclared as
            // functions. Therefore overloading or redefining builtin functions is an error.
            error(location, "Name of a built-in function cannot be redeclared as function",
                  function->name());
        }
    }
    else
    {
        // ESSL 1.00.17 section 4.2.6: built-ins can be overloaded but not redefined. We assume that
        // this applies to redeclarations as well.
        const TSymbol *builtIn =
            symbolTable.findBuiltIn(function->getMangledName(), getShaderVersion());
        if (builtIn)
        {
            error(location, "built-in functions cannot be redefined", function->name());
        }
    }

    // Return types and parameter qualifiers must match in all redeclarations, so those are checked
    // here.
    const TFunction *prevDec =
        static_cast<const TFunction *>(symbolTable.findGlobal(function->getMangledName()));
    if (prevDec)
    {
        if (prevDec->getReturnType() != function->getReturnType())
        {
            error(location, "function must have the same return type in all of its declarations",
                  function->getReturnType().getBasicString());
        }
        for (size_t i = 0; i < prevDec->getParamCount(); ++i)
        {
            if (prevDec->getParam(i)->getType().getQualifier() !=
                function->getParam(i)->getType().getQualifier())
            {
                error(location,
                      "function must have the same parameter qualifiers in all of its declarations",
                      function->getParam(i)->getType().getQualifierString());
            }
        }
    }

    // Check for previously declared variables using the same name.
    const TSymbol *prevSym   = symbolTable.find(function->name(), getShaderVersion());
    bool insertUnmangledName = true;
    if (prevSym)
    {
        if (!prevSym->isFunction())
        {
            error(location, "redefinition of a function", function->name());
        }
        insertUnmangledName = false;
    }
    // Parsing is at the inner scope level of the function's arguments and body statement at this
    // point, but declareUserDefinedFunction takes care of declaring the function at the global
    // scope.
    symbolTable.declareUserDefinedFunction(function, insertUnmangledName);

    // Raise error message if main function takes any parameters or return anything other than void
    if (function->isMain())
    {
        if (function->getParamCount() > 0)
        {
            error(location, "function cannot take any parameter(s)", "main");
        }
        if (function->getReturnType().getBasicType() != EbtVoid)
        {
            error(location, "main function cannot return a value",
                  function->getReturnType().getBasicString());
        }
    }

    //
    // If this is a redeclaration, it could also be a definition, in which case, we want to use the
    // variable names from this one, and not the one that's
    // being redeclared.  So, pass back up this declaration, not the one in the symbol table.
    //
    return function;
}

TFunction *TParseContext::parseFunctionHeader(const TPublicType &type,
                                              const ImmutableString &name,
                                              const TSourceLoc &location)
{
    if (type.qualifier != EvqGlobal && type.qualifier != EvqTemporary)
    {
        error(location, "no qualifiers allowed for function return",
              getQualifierString(type.qualifier));
    }
    if (!type.layoutQualifier.isEmpty())
    {
        error(location, "no qualifiers allowed for function return", "layout");
    }
    // make sure an opaque type is not involved as well...
    std::string reason(getBasicString(type.getBasicType()));
    reason += "s can't be function return values";
    checkIsNotOpaqueType(location, type.typeSpecifierNonArray, reason.c_str());
    if (mShaderVersion < 300)
    {
        // Array return values are forbidden, but there's also no valid syntax for declaring array
        // return values in ESSL 1.00.
        ASSERT(!type.isArray() || mDiagnostics->numErrors() > 0);

        if (type.isStructureContainingArrays())
        {
            // ESSL 1.00.17 section 6.1 Function Definitions
            TInfoSinkBase typeString;
            typeString << TType(type);
            error(location, "structures containing arrays can't be function return values",
                  typeString.c_str());
        }
    }

    // Add the function as a prototype after parsing it (we do not support recursion)
    return new TFunction(&symbolTable, name, SymbolType::UserDefined, new TType(type), false);
}

TFunctionLookup *TParseContext::addNonConstructorFunc(const ImmutableString &name,
                                                      const TSymbol *symbol)
{
    return TFunctionLookup::CreateFunctionCall(name, symbol);
}

TFunctionLookup *TParseContext::addConstructorFunc(const TPublicType &publicType)
{
    if (mShaderVersion < 300 && publicType.isArray())
    {
        error(publicType.getLine(), "array constructor supported in GLSL ES 3.00 and above only",
              "[]");
    }
    if (publicType.isStructSpecifier())
    {
        error(publicType.getLine(), "constructor can't be a structure definition",
              getBasicString(publicType.getBasicType()));
    }

    TType *type = new TType(publicType);
    if (!type->canBeConstructed())
    {
        error(publicType.getLine(), "cannot construct this type",
              getBasicString(publicType.getBasicType()));
        type->setBasicType(EbtFloat);
    }
    return TFunctionLookup::CreateConstructor(type);
}

void TParseContext::checkIsNotUnsizedArray(const TSourceLoc &line,
                                           const char *errorMessage,
                                           const ImmutableString &token,
                                           TType *arrayType)
{
    if (arrayType->isUnsizedArray())
    {
        error(line, errorMessage, token);
        arrayType->sizeUnsizedArrays(nullptr);
    }
}

TParameter TParseContext::parseParameterDeclarator(TType *type,
                                                   const ImmutableString &name,
                                                   const TSourceLoc &nameLoc)
{
    ASSERT(type);
    checkIsNotUnsizedArray(nameLoc, "function parameter array must specify a size", name, type);
    if (type->getBasicType() == EbtVoid)
    {
        error(nameLoc, "illegal use of type 'void'", name);
    }
    checkIsNotReserved(nameLoc, name);
    TParameter param = {name.data(), type};
    return param;
}

TParameter TParseContext::parseParameterDeclarator(const TPublicType &publicType,
                                                   const ImmutableString &name,
                                                   const TSourceLoc &nameLoc)
{
    TType *type = new TType(publicType);
    return parseParameterDeclarator(type, name, nameLoc);
}

TParameter TParseContext::parseParameterArrayDeclarator(const ImmutableString &name,
                                                        const TSourceLoc &nameLoc,
                                                        const TVector<unsigned int> &arraySizes,
                                                        const TSourceLoc &arrayLoc,
                                                        TPublicType *elementType)
{
    checkArrayElementIsNotArray(arrayLoc, *elementType);
    TType *arrayType = new TType(*elementType);
    arrayType->makeArrays(arraySizes);
    return parseParameterDeclarator(arrayType, name, nameLoc);
}

bool TParseContext::checkUnsizedArrayConstructorArgumentDimensionality(
    const TIntermSequence &arguments,
    TType type,
    const TSourceLoc &line)
{
    if (arguments.empty())
    {
        error(line, "implicitly sized array constructor must have at least one argument", "[]");
        return false;
    }
    for (TIntermNode *arg : arguments)
    {
        const TIntermTyped *element = arg->getAsTyped();
        ASSERT(element);
        size_t dimensionalityFromElement = element->getType().getNumArraySizes() + 1u;
        if (dimensionalityFromElement > type.getNumArraySizes())
        {
            error(line, "constructing from a non-dereferenced array", "constructor");
            return false;
        }
        else if (dimensionalityFromElement < type.getNumArraySizes())
        {
            if (dimensionalityFromElement == 1u)
            {
                error(line, "implicitly sized array of arrays constructor argument is not an array",
                      "constructor");
            }
            else
            {
                error(line,
                      "implicitly sized array of arrays constructor argument dimensionality is too "
                      "low",
                      "constructor");
            }
            return false;
        }
    }
    return true;
}

// This function is used to test for the correctness of the parameters passed to various constructor
// functions and also convert them to the right datatype if it is allowed and required.
//
// Returns a node to add to the tree regardless of if an error was generated or not.
//
TIntermTyped *TParseContext::addConstructor(TFunctionLookup *fnCall, const TSourceLoc &line)
{
    TType type                 = fnCall->constructorType();
    TIntermSequence &arguments = fnCall->arguments();
    if (type.isUnsizedArray())
    {
        if (!checkUnsizedArrayConstructorArgumentDimensionality(arguments, type, line))
        {
            type.sizeUnsizedArrays(nullptr);
            return CreateZeroNode(type);
        }
        TIntermTyped *firstElement = arguments.at(0)->getAsTyped();
        ASSERT(firstElement);
        if (type.getOutermostArraySize() == 0u)
        {
            type.sizeOutermostUnsizedArray(static_cast<unsigned int>(arguments.size()));
        }
        for (size_t i = 0; i < firstElement->getType().getNumArraySizes(); ++i)
        {
            if ((*type.getArraySizes())[i] == 0u)
            {
                type.setArraySize(i, (*firstElement->getType().getArraySizes())[i]);
            }
        }
        ASSERT(!type.isUnsizedArray());
    }

    if (!checkConstructorArguments(line, arguments, type))
    {
        return CreateZeroNode(type);
    }

    TIntermAggregate *constructorNode = TIntermAggregate::CreateConstructor(type, &arguments);
    constructorNode->setLine(line);

    return constructorNode->fold(mDiagnostics);
}

//
// Interface/uniform blocks
// TODO(jiawei.shao@intel.com): implement GL_EXT_shader_io_blocks.
//
TIntermDeclaration *TParseContext::addInterfaceBlock(
    const TTypeQualifierBuilder &typeQualifierBuilder,
    const TSourceLoc &nameLine,
    const ImmutableString &blockName,
    TFieldList *fieldList,
    const ImmutableString &instanceName,
    const TSourceLoc &instanceLine,
    TIntermTyped *arrayIndex,
    const TSourceLoc &arrayIndexLine)
{
    checkIsNotReserved(nameLine, blockName);

    TTypeQualifier typeQualifier = typeQualifierBuilder.getVariableTypeQualifier(mDiagnostics);

    if (mShaderVersion < 310 && typeQualifier.qualifier != EvqUniform)
    {
        error(typeQualifier.line,
              "invalid qualifier: interface blocks must be uniform in version lower than GLSL ES "
              "3.10",
              getQualifierString(typeQualifier.qualifier));
    }
    else if (typeQualifier.qualifier != EvqUniform && typeQualifier.qualifier != EvqBuffer)
    {
        error(typeQualifier.line, "invalid qualifier: interface blocks must be uniform or buffer",
              getQualifierString(typeQualifier.qualifier));
    }

    if (typeQualifier.invariant)
    {
        error(typeQualifier.line, "invalid qualifier on interface block member", "invariant");
    }

    if (typeQualifier.qualifier != EvqBuffer)
    {
        checkMemoryQualifierIsNotSpecified(typeQualifier.memoryQualifier, typeQualifier.line);
    }

    // add array index
    unsigned int arraySize = 0;
    if (arrayIndex != nullptr)
    {
        arraySize = checkIsValidArraySize(arrayIndexLine, arrayIndex);
    }

    checkIndexIsNotSpecified(typeQualifier.line, typeQualifier.layoutQualifier.index);

    if (mShaderVersion < 310)
    {
        checkBindingIsNotSpecified(typeQualifier.line, typeQualifier.layoutQualifier.binding);
    }
    else
    {
        checkBlockBindingIsValid(typeQualifier.line, typeQualifier.qualifier,
                                 typeQualifier.layoutQualifier.binding, arraySize);
    }

    checkYuvIsNotSpecified(typeQualifier.line, typeQualifier.layoutQualifier.yuv);

    TLayoutQualifier blockLayoutQualifier = typeQualifier.layoutQualifier;
    checkLocationIsNotSpecified(typeQualifier.line, blockLayoutQualifier);
    checkStd430IsForShaderStorageBlock(typeQualifier.line, blockLayoutQualifier.blockStorage,
                                       typeQualifier.qualifier);

    if (blockLayoutQualifier.matrixPacking == EmpUnspecified)
    {
        if (typeQualifier.qualifier == EvqUniform)
        {
            blockLayoutQualifier.matrixPacking = mDefaultUniformMatrixPacking;
        }
        else if (typeQualifier.qualifier == EvqBuffer)
        {
            blockLayoutQualifier.matrixPacking = mDefaultBufferMatrixPacking;
        }
    }

    if (blockLayoutQualifier.blockStorage == EbsUnspecified)
    {
        if (typeQualifier.qualifier == EvqUniform)
        {
            blockLayoutQualifier.blockStorage = mDefaultUniformBlockStorage;
        }
        else if (typeQualifier.qualifier == EvqBuffer)
        {
            blockLayoutQualifier.blockStorage = mDefaultBufferBlockStorage;
        }
    }

    checkWorkGroupSizeIsNotSpecified(nameLine, blockLayoutQualifier);

    checkInternalFormatIsNotSpecified(nameLine, blockLayoutQualifier.imageInternalFormat);

    // check for sampler types and apply layout qualifiers
    for (size_t memberIndex = 0; memberIndex < fieldList->size(); ++memberIndex)
    {
        TField *field    = (*fieldList)[memberIndex];
        TType *fieldType = field->type();
        if (IsOpaqueType(fieldType->getBasicType()))
        {
            std::string reason("unsupported type - ");
            reason += fieldType->getBasicString();
            reason += " types are not allowed in interface blocks";
            error(field->line(), reason.c_str(), fieldType->getBasicString());
        }

        const TQualifier qualifier = fieldType->getQualifier();
        switch (qualifier)
        {
            case EvqGlobal:
                break;
            case EvqUniform:
                if (typeQualifier.qualifier == EvqBuffer)
                {
                    error(field->line(), "invalid qualifier on shader storage block member",
                          getQualifierString(qualifier));
                }
                break;
            case EvqBuffer:
                if (typeQualifier.qualifier == EvqUniform)
                {
                    error(field->line(), "invalid qualifier on uniform block member",
                          getQualifierString(qualifier));
                }
                break;
            default:
                error(field->line(), "invalid qualifier on interface block member",
                      getQualifierString(qualifier));
                break;
        }

        if (fieldType->isInvariant())
        {
            error(field->line(), "invalid qualifier on interface block member", "invariant");
        }

        // check layout qualifiers
        TLayoutQualifier fieldLayoutQualifier = fieldType->getLayoutQualifier();
        checkLocationIsNotSpecified(field->line(), fieldLayoutQualifier);
        checkIndexIsNotSpecified(field->line(), fieldLayoutQualifier.index);
        checkBindingIsNotSpecified(field->line(), fieldLayoutQualifier.binding);

        if (fieldLayoutQualifier.blockStorage != EbsUnspecified)
        {
            error(field->line(), "invalid layout qualifier: cannot be used here",
                  getBlockStorageString(fieldLayoutQualifier.blockStorage));
        }

        if (fieldLayoutQualifier.matrixPacking == EmpUnspecified)
        {
            fieldLayoutQualifier.matrixPacking = blockLayoutQualifier.matrixPacking;
        }
        else if (!fieldType->isMatrix() && fieldType->getBasicType() != EbtStruct)
        {
            warning(field->line(),
                    "extraneous layout qualifier: only has an effect on matrix types",
                    getMatrixPackingString(fieldLayoutQualifier.matrixPacking));
        }

        fieldType->setLayoutQualifier(fieldLayoutQualifier);

        if (mShaderVersion < 310 || memberIndex != fieldList->size() - 1u ||
            typeQualifier.qualifier != EvqBuffer)
        {
            // ESSL 3.10 spec section 4.1.9 allows for runtime-sized arrays.
            checkIsNotUnsizedArray(field->line(),
                                   "array members of interface blocks must specify a size",
                                   field->name(), field->type());
        }

        if (typeQualifier.qualifier == EvqBuffer)
        {
            // set memory qualifiers
            // GLSL ES 3.10 session 4.9 [Memory Access Qualifiers]. When a block declaration is
            // qualified with a memory qualifier, it is as if all of its members were declared with
            // the same memory qualifier.
            const TMemoryQualifier &blockMemoryQualifier = typeQualifier.memoryQualifier;
            TMemoryQualifier fieldMemoryQualifier        = fieldType->getMemoryQualifier();
            fieldMemoryQualifier.readonly |= blockMemoryQualifier.readonly;
            fieldMemoryQualifier.writeonly |= blockMemoryQualifier.writeonly;
            fieldMemoryQualifier.coherent |= blockMemoryQualifier.coherent;
            fieldMemoryQualifier.restrictQualifier |= blockMemoryQualifier.restrictQualifier;
            fieldMemoryQualifier.volatileQualifier |= blockMemoryQualifier.volatileQualifier;
            // TODO(jiajia.qin@intel.com): Decide whether if readonly and writeonly buffer variable
            // is legal. See bug https://github.com/KhronosGroup/OpenGL-API/issues/7
            fieldType->setMemoryQualifier(fieldMemoryQualifier);
        }
    }

    TInterfaceBlock *interfaceBlock = new TInterfaceBlock(
        &symbolTable, blockName, fieldList, blockLayoutQualifier, SymbolType::UserDefined);
    if (!symbolTable.declare(interfaceBlock))
    {
        error(nameLine, "redefinition of an interface block name", blockName);
    }

    TType *interfaceBlockType =
        new TType(interfaceBlock, typeQualifier.qualifier, blockLayoutQualifier);
    if (arrayIndex != nullptr)
    {
        interfaceBlockType->makeArray(arraySize);
    }

    // The instance variable gets created to refer to the interface block type from the AST
    // regardless of if there's an instance name. It's created as an empty symbol if there is no
    // instance name.
    TVariable *instanceVariable =
        new TVariable(&symbolTable, instanceName, interfaceBlockType,
                      instanceName.empty() ? SymbolType::Empty : SymbolType::UserDefined);

    if (instanceVariable->symbolType() == SymbolType::Empty)
    {
        // define symbols for the members of the interface block
        for (size_t memberIndex = 0; memberIndex < fieldList->size(); ++memberIndex)
        {
            TField *field    = (*fieldList)[memberIndex];
            TType *fieldType = new TType(*field->type());

            // set parent pointer of the field variable
            fieldType->setInterfaceBlock(interfaceBlock);

            fieldType->setQualifier(typeQualifier.qualifier);

            TVariable *fieldVariable =
                new TVariable(&symbolTable, field->name(), fieldType, SymbolType::UserDefined);
            if (!symbolTable.declare(fieldVariable))
            {
                error(field->line(), "redefinition of an interface block member name",
                      field->name());
            }
        }
    }
    else
    {
        checkIsNotReserved(instanceLine, instanceName);

        // add a symbol for this interface block
        if (!symbolTable.declare(instanceVariable))
        {
            error(instanceLine, "redefinition of an interface block instance name", instanceName);
        }
    }

    TIntermSymbol *blockSymbol = new TIntermSymbol(instanceVariable);
    blockSymbol->setLine(typeQualifier.line);
    TIntermDeclaration *declaration = new TIntermDeclaration();
    declaration->appendDeclarator(blockSymbol);
    declaration->setLine(nameLine);

    exitStructDeclaration();
    return declaration;
}

void TParseContext::enterStructDeclaration(const TSourceLoc &line,
                                           const ImmutableString &identifier)
{
    ++mStructNestingLevel;

    // Embedded structure definitions are not supported per GLSL ES spec.
    // ESSL 1.00.17 section 10.9. ESSL 3.00.6 section 12.11.
    if (mStructNestingLevel > 1)
    {
        error(line, "Embedded struct definitions are not allowed", "struct");
    }
}

void TParseContext::exitStructDeclaration()
{
    --mStructNestingLevel;
}

void TParseContext::checkIsBelowStructNestingLimit(const TSourceLoc &line, const TField &field)
{
    if (!sh::IsWebGLBasedSpec(mShaderSpec))
    {
        return;
    }

    if (field.type()->getBasicType() != EbtStruct)
    {
        return;
    }

    // We're already inside a structure definition at this point, so add
    // one to the field's struct nesting.
    if (1 + field.type()->getDeepestStructNesting() > kWebGLMaxStructNesting)
    {
        std::stringstream reasonStream = sh::InitializeStream<std::stringstream>();
        if (field.type()->getStruct()->symbolType() == SymbolType::Empty)
        {
            // This may happen in case there are nested struct definitions. While they are also
            // invalid GLSL, they don't cause a syntax error.
            reasonStream << "Struct nesting";
        }
        else
        {
            reasonStream << "Reference of struct type " << field.type()->getStruct()->name();
        }
        reasonStream << " exceeds maximum allowed nesting level of " << kWebGLMaxStructNesting;
        std::string reason = reasonStream.str();
        error(line, reason.c_str(), field.name());
        return;
    }
}

//
// Parse an array index expression
//
TIntermTyped *TParseContext::addIndexExpression(TIntermTyped *baseExpression,
                                                const TSourceLoc &location,
                                                TIntermTyped *indexExpression)
{
    if (!baseExpression->isArray() && !baseExpression->isMatrix() && !baseExpression->isVector())
    {
        if (baseExpression->getAsSymbolNode())
        {
            error(location, " left of '[' is not of type array, matrix, or vector ",
                  baseExpression->getAsSymbolNode()->getName());
        }
        else
        {
            error(location, " left of '[' is not of type array, matrix, or vector ", "expression");
        }

        return CreateZeroNode(TType(EbtFloat, EbpHigh, EvqConst));
    }

    if (baseExpression->getQualifier() == EvqPerVertexIn)
    {
        ASSERT(mShaderType == GL_GEOMETRY_SHADER_EXT);
        if (mGeometryShaderInputPrimitiveType == EptUndefined)
        {
            error(location, "missing input primitive declaration before indexing gl_in.", "[");
            return CreateZeroNode(TType(EbtFloat, EbpHigh, EvqConst));
        }
    }

    TIntermConstantUnion *indexConstantUnion = indexExpression->getAsConstantUnion();

    // ANGLE should be able to fold any constant expressions resulting in an integer - but to be
    // safe we don't treat "EvqConst" that's evaluated according to the spec as being sufficient
    // for constness. Some interpretations of the spec have allowed constant expressions with side
    // effects - like array length() method on a non-constant array.
    if (indexExpression->getQualifier() != EvqConst || indexConstantUnion == nullptr)
    {
        if (baseExpression->isInterfaceBlock())
        {
            // TODO(jiawei.shao@intel.com): implement GL_EXT_shader_io_blocks.
            switch (baseExpression->getQualifier())
            {
                case EvqPerVertexIn:
                    break;
                case EvqUniform:
                case EvqBuffer:
                    error(location,
                          "array indexes for uniform block arrays and shader storage block arrays "
                          "must be constant integral expressions",
                          "[");
                    break;
                default:
                    // We can reach here only in error cases.
                    ASSERT(mDiagnostics->numErrors() > 0);
                    break;
            }
        }
        else if (baseExpression->getQualifier() == EvqFragmentOut)
        {
            error(location,
                  "array indexes for fragment outputs must be constant integral expressions", "[");
        }
        else if (mShaderSpec == SH_WEBGL2_SPEC && baseExpression->getQualifier() == EvqFragData)
        {
            error(location, "array index for gl_FragData must be constant zero", "[");
        }
    }

    if (indexConstantUnion)
    {
        // If an out-of-range index is not qualified as constant, the behavior in the spec is
        // undefined. This applies even if ANGLE has been able to constant fold it (ANGLE may
        // constant fold expressions that are not constant expressions). The most compatible way to
        // handle this case is to report a warning instead of an error and force the index to be in
        // the correct range.
        bool outOfRangeIndexIsError = indexExpression->getQualifier() == EvqConst;
        int index                   = 0;
        if (indexConstantUnion->getBasicType() == EbtInt)
        {
            index = indexConstantUnion->getIConst(0);
        }
        else if (indexConstantUnion->getBasicType() == EbtUInt)
        {
            index = static_cast<int>(indexConstantUnion->getUConst(0));
        }

        int safeIndex = -1;

        if (index < 0)
        {
            outOfRangeError(outOfRangeIndexIsError, location, "index expression is negative", "[]");
            safeIndex = 0;
        }

        if (!baseExpression->getType().isUnsizedArray())
        {
            if (baseExpression->isArray())
            {
                if (baseExpression->getQualifier() == EvqFragData && index > 0)
                {
                    if (!isExtensionEnabled(TExtension::EXT_draw_buffers))
                    {
                        outOfRangeError(outOfRangeIndexIsError, location,
                                        "array index for gl_FragData must be zero when "
                                        "GL_EXT_draw_buffers is disabled",
                                        "[]");
                        safeIndex = 0;
                    }
                }
            }
            // Only do generic out-of-range check if similar error hasn't already been reported.
            if (safeIndex < 0)
            {
                if (baseExpression->isArray())
                {
                    safeIndex = checkIndexLessThan(outOfRangeIndexIsError, location, index,
                                                   baseExpression->getOutermostArraySize(),
                                                   "array index out of range");
                }
                else if (baseExpression->isMatrix())
                {
                    safeIndex = checkIndexLessThan(outOfRangeIndexIsError, location, index,
                                                   baseExpression->getType().getCols(),
                                                   "matrix field selection out of range");
                }
                else
                {
                    ASSERT(baseExpression->isVector());
                    safeIndex = checkIndexLessThan(outOfRangeIndexIsError, location, index,
                                                   baseExpression->getType().getNominalSize(),
                                                   "vector field selection out of range");
                }
            }

            ASSERT(safeIndex >= 0);
            // Data of constant unions can't be changed, because it may be shared with other
            // constant unions or even builtins, like gl_MaxDrawBuffers. Instead use a new
            // sanitized object.
            if (safeIndex != index || indexConstantUnion->getBasicType() != EbtInt)
            {
                TConstantUnion *safeConstantUnion = new TConstantUnion();
                safeConstantUnion->setIConst(safeIndex);
                indexExpression = new TIntermConstantUnion(
                    safeConstantUnion, TType(EbtInt, indexExpression->getPrecision(),
                                             indexExpression->getQualifier()));
            }

            TIntermBinary *node =
                new TIntermBinary(EOpIndexDirect, baseExpression, indexExpression);
            node->setLine(location);
            return expressionOrFoldedResult(node);
        }
    }

    markStaticReadIfSymbol(indexExpression);
    TIntermBinary *node = new TIntermBinary(EOpIndexIndirect, baseExpression, indexExpression);
    node->setLine(location);
    // Indirect indexing can never be constant folded.
    return node;
}

int TParseContext::checkIndexLessThan(bool outOfRangeIndexIsError,
                                      const TSourceLoc &location,
                                      int index,
                                      int arraySize,
                                      const char *reason)
{
    // Should not reach here with an unsized / runtime-sized array.
    ASSERT(arraySize > 0);
    // A negative index should already have been checked.
    ASSERT(index >= 0);
    if (index >= arraySize)
    {
        std::stringstream reasonStream = sh::InitializeStream<std::stringstream>();
        reasonStream << reason << " '" << index << "'";
        std::string token = reasonStream.str();
        outOfRangeError(outOfRangeIndexIsError, location, reason, "[]");
        return arraySize - 1;
    }
    return index;
}

TIntermTyped *TParseContext::addFieldSelectionExpression(TIntermTyped *baseExpression,
                                                         const TSourceLoc &dotLocation,
                                                         const ImmutableString &fieldString,
                                                         const TSourceLoc &fieldLocation)
{
    if (baseExpression->isArray())
    {
        error(fieldLocation, "cannot apply dot operator to an array", ".");
        return baseExpression;
    }

    if (baseExpression->isVector())
    {
        TVector<int> fieldOffsets;
        if (!parseVectorFields(fieldLocation, fieldString, baseExpression->getNominalSize(),
                               &fieldOffsets))
        {
            fieldOffsets.resize(1);
            fieldOffsets[0] = 0;
        }
        TIntermSwizzle *node = new TIntermSwizzle(baseExpression, fieldOffsets);
        node->setLine(dotLocation);

        return node->fold(mDiagnostics);
    }
    else if (baseExpression->getBasicType() == EbtStruct)
    {
        const TFieldList &fields = baseExpression->getType().getStruct()->fields();
        if (fields.empty())
        {
            error(dotLocation, "structure has no fields", "Internal Error");
            return baseExpression;
        }
        else
        {
            bool fieldFound = false;
            unsigned int i;
            for (i = 0; i < fields.size(); ++i)
            {
                if (fields[i]->name() == fieldString)
                {
                    fieldFound = true;
                    break;
                }
            }
            if (fieldFound)
            {
                TIntermTyped *index = CreateIndexNode(i);
                index->setLine(fieldLocation);
                TIntermBinary *node =
                    new TIntermBinary(EOpIndexDirectStruct, baseExpression, index);
                node->setLine(dotLocation);
                return expressionOrFoldedResult(node);
            }
            else
            {
                error(dotLocation, " no such field in structure", fieldString);
                return baseExpression;
            }
        }
    }
    else if (baseExpression->isInterfaceBlock())
    {
        const TFieldList &fields = baseExpression->getType().getInterfaceBlock()->fields();
        if (fields.empty())
        {
            error(dotLocation, "interface block has no fields", "Internal Error");
            return baseExpression;
        }
        else
        {
            bool fieldFound = false;
            unsigned int i;
            for (i = 0; i < fields.size(); ++i)
            {
                if (fields[i]->name() == fieldString)
                {
                    fieldFound = true;
                    break;
                }
            }
            if (fieldFound)
            {
                TIntermTyped *index = CreateIndexNode(i);
                index->setLine(fieldLocation);
                TIntermBinary *node =
                    new TIntermBinary(EOpIndexDirectInterfaceBlock, baseExpression, index);
                node->setLine(dotLocation);
                // Indexing interface blocks can never be constant folded.
                return node;
            }
            else
            {
                error(dotLocation, " no such field in interface block", fieldString);
                return baseExpression;
            }
        }
    }
    else
    {
        if (mShaderVersion < 300)
        {
            error(dotLocation, " field selection requires structure or vector on left hand side",
                  fieldString);
        }
        else
        {
            error(dotLocation,
                  " field selection requires structure, vector, or interface block on left hand "
                  "side",
                  fieldString);
        }
        return baseExpression;
    }
}

TLayoutQualifier TParseContext::parseLayoutQualifier(const ImmutableString &qualifierType,
                                                     const TSourceLoc &qualifierTypeLine)
{
    TLayoutQualifier qualifier = TLayoutQualifier::Create();

    if (qualifierType == "shared")
    {
        if (sh::IsWebGLBasedSpec(mShaderSpec))
        {
            error(qualifierTypeLine, "Only std140 layout is allowed in WebGL", "shared");
        }
        qualifier.blockStorage = EbsShared;
    }
    else if (qualifierType == "packed")
    {
        if (sh::IsWebGLBasedSpec(mShaderSpec))
        {
            error(qualifierTypeLine, "Only std140 layout is allowed in WebGL", "packed");
        }
        qualifier.blockStorage = EbsPacked;
    }
    else if (qualifierType == "std430")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.blockStorage = EbsStd430;
    }
    else if (qualifierType == "std140")
    {
        qualifier.blockStorage = EbsStd140;
    }
    else if (qualifierType == "row_major")
    {
        qualifier.matrixPacking = EmpRowMajor;
    }
    else if (qualifierType == "column_major")
    {
        qualifier.matrixPacking = EmpColumnMajor;
    }
    else if (qualifierType == "location")
    {
        error(qualifierTypeLine, "invalid layout qualifier: location requires an argument",
              qualifierType);
    }
    else if (qualifierType == "yuv" && mShaderType == GL_FRAGMENT_SHADER)
    {
        if (checkCanUseExtension(qualifierTypeLine, TExtension::EXT_YUV_target))
        {
            qualifier.yuv = true;
        }
    }
    else if (qualifierType == "rgba32f")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA32F;
    }
    else if (qualifierType == "rgba16f")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA16F;
    }
    else if (qualifierType == "r32f")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifR32F;
    }
    else if (qualifierType == "rgba8")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA8;
    }
    else if (qualifierType == "rgba8_snorm")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA8_SNORM;
    }
    else if (qualifierType == "rgba32i")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA32I;
    }
    else if (qualifierType == "rgba16i")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA16I;
    }
    else if (qualifierType == "rgba8i")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA8I;
    }
    else if (qualifierType == "r32i")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifR32I;
    }
    else if (qualifierType == "rgba32ui")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA32UI;
    }
    else if (qualifierType == "rgba16ui")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA16UI;
    }
    else if (qualifierType == "rgba8ui")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifRGBA8UI;
    }
    else if (qualifierType == "r32ui")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.imageInternalFormat = EiifR32UI;
    }
    else if (qualifierType == "points" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.primitiveType = EptPoints;
    }
    else if (qualifierType == "lines" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.primitiveType = EptLines;
    }
    else if (qualifierType == "lines_adjacency" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.primitiveType = EptLinesAdjacency;
    }
    else if (qualifierType == "triangles" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.primitiveType = EptTriangles;
    }
    else if (qualifierType == "triangles_adjacency" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.primitiveType = EptTrianglesAdjacency;
    }
    else if (qualifierType == "line_strip" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.primitiveType = EptLineStrip;
    }
    else if (qualifierType == "triangle_strip" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        qualifier.primitiveType = EptTriangleStrip;
    }

    else
    {
        error(qualifierTypeLine, "invalid layout qualifier", qualifierType);
    }

    return qualifier;
}

void TParseContext::parseLocalSize(const ImmutableString &qualifierType,
                                   const TSourceLoc &qualifierTypeLine,
                                   int intValue,
                                   const TSourceLoc &intValueLine,
                                   const std::string &intValueString,
                                   size_t index,
                                   sh::WorkGroupSize *localSize)
{
    checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
    if (intValue < 1)
    {
        std::stringstream reasonStream = sh::InitializeStream<std::stringstream>();
        reasonStream << "out of range: " << getWorkGroupSizeString(index) << " must be positive";
        std::string reason = reasonStream.str();
        error(intValueLine, reason.c_str(), intValueString.c_str());
    }
    (*localSize)[index] = intValue;
}

void TParseContext::parseNumViews(int intValue,
                                  const TSourceLoc &intValueLine,
                                  const std::string &intValueString,
                                  int *numViews)
{
    // This error is only specified in WebGL, but tightens unspecified behavior in the native
    // specification.
    if (intValue < 1)
    {
        error(intValueLine, "out of range: num_views must be positive", intValueString.c_str());
    }
    *numViews = intValue;
}

void TParseContext::parseInvocations(int intValue,
                                     const TSourceLoc &intValueLine,
                                     const std::string &intValueString,
                                     int *numInvocations)
{
    // Although SPEC isn't clear whether invocations can be less than 1, we add this limit because
    // it doesn't make sense to accept invocations <= 0.
    if (intValue < 1 || intValue > mMaxGeometryShaderInvocations)
    {
        error(intValueLine,
              "out of range: invocations must be in the range of [1, "
              "MAX_GEOMETRY_SHADER_INVOCATIONS_OES]",
              intValueString.c_str());
    }
    else
    {
        *numInvocations = intValue;
    }
}

void TParseContext::parseMaxVertices(int intValue,
                                     const TSourceLoc &intValueLine,
                                     const std::string &intValueString,
                                     int *maxVertices)
{
    // Although SPEC isn't clear whether max_vertices can be less than 0, we add this limit because
    // it doesn't make sense to accept max_vertices < 0.
    if (intValue < 0 || intValue > mMaxGeometryShaderMaxVertices)
    {
        error(
            intValueLine,
            "out of range: max_vertices must be in the range of [0, gl_MaxGeometryOutputVertices]",
            intValueString.c_str());
    }
    else
    {
        *maxVertices = intValue;
    }
}

void TParseContext::parseIndexLayoutQualifier(int intValue,
                                              const TSourceLoc &intValueLine,
                                              const std::string &intValueString,
                                              int *index)
{
    // EXT_blend_func_extended specifies that most validation should happen at link time, but since
    // we're validating output variable locations at compile time, it makes sense to validate that
    // index is 0 or 1 also at compile time. Also since we use "-1" as a placeholder for unspecified
    // index, we can't accept it here.
    if (intValue < 0 || intValue > 1)
    {
        error(intValueLine, "out of range: index layout qualifier can only be 0 or 1",
              intValueString.c_str());
    }
    else
    {
        *index = intValue;
    }
}

TLayoutQualifier TParseContext::parseLayoutQualifier(const ImmutableString &qualifierType,
                                                     const TSourceLoc &qualifierTypeLine,
                                                     int intValue,
                                                     const TSourceLoc &intValueLine)
{
    TLayoutQualifier qualifier = TLayoutQualifier::Create();

    std::string intValueString = Str(intValue);

    if (qualifierType == "location")
    {
        // must check that location is non-negative
        if (intValue < 0)
        {
            error(intValueLine, "out of range: location must be non-negative",
                  intValueString.c_str());
        }
        else
        {
            qualifier.location           = intValue;
            qualifier.locationsSpecified = 1;
        }
    }
    else if (qualifierType == "binding")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        if (intValue < 0)
        {
            error(intValueLine, "out of range: binding must be non-negative",
                  intValueString.c_str());
        }
        else
        {
            qualifier.binding = intValue;
        }
    }
    else if (qualifierType == "offset")
    {
        checkLayoutQualifierSupported(qualifierTypeLine, qualifierType, 310);
        if (intValue < 0)
        {
            error(intValueLine, "out of range: offset must be non-negative",
                  intValueString.c_str());
        }
        else
        {
            qualifier.offset = intValue;
        }
    }
    else if (qualifierType == "local_size_x")
    {
        parseLocalSize(qualifierType, qualifierTypeLine, intValue, intValueLine, intValueString, 0u,
                       &qualifier.localSize);
    }
    else if (qualifierType == "local_size_y")
    {
        parseLocalSize(qualifierType, qualifierTypeLine, intValue, intValueLine, intValueString, 1u,
                       &qualifier.localSize);
    }
    else if (qualifierType == "local_size_z")
    {
        parseLocalSize(qualifierType, qualifierTypeLine, intValue, intValueLine, intValueString, 2u,
                       &qualifier.localSize);
    }
    else if (qualifierType == "num_views" && mShaderType == GL_VERTEX_SHADER)
    {
        if (checkCanUseOneOfExtensions(
                qualifierTypeLine, std::array<TExtension, 2u>{
                                       {TExtension::OVR_multiview, TExtension::OVR_multiview2}}))
        {
            parseNumViews(intValue, intValueLine, intValueString, &qualifier.numViews);
        }
    }
    else if (qualifierType == "invocations" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        parseInvocations(intValue, intValueLine, intValueString, &qualifier.invocations);
    }
    else if (qualifierType == "max_vertices" && mShaderType == GL_GEOMETRY_SHADER_EXT &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_geometry_shader))
    {
        parseMaxVertices(intValue, intValueLine, intValueString, &qualifier.maxVertices);
    }
    else if (qualifierType == "index" && mShaderType == GL_FRAGMENT_SHADER &&
             checkCanUseExtension(qualifierTypeLine, TExtension::EXT_blend_func_extended))
    {
        parseIndexLayoutQualifier(intValue, intValueLine, intValueString, &qualifier.index);
    }
    else
    {
        error(qualifierTypeLine, "invalid layout qualifier", qualifierType);
    }

    return qualifier;
}

TTypeQualifierBuilder *TParseContext::createTypeQualifierBuilder(const TSourceLoc &loc)
{
    return new TTypeQualifierBuilder(
        new TStorageQualifierWrapper(symbolTable.atGlobalLevel() ? EvqGlobal : EvqTemporary, loc),
        mShaderVersion);
}

TStorageQualifierWrapper *TParseContext::parseGlobalStorageQualifier(TQualifier qualifier,
                                                                     const TSourceLoc &loc)
{
    checkIsAtGlobalLevel(loc, getQualifierString(qualifier));
    return new TStorageQualifierWrapper(qualifier, loc);
}

TStorageQualifierWrapper *TParseContext::parseVaryingQualifier(const TSourceLoc &loc)
{
    if (getShaderType() == GL_VERTEX_SHADER)
    {
        return parseGlobalStorageQualifier(EvqVaryingOut, loc);
    }
    return parseGlobalStorageQualifier(EvqVaryingIn, loc);
}

TStorageQualifierWrapper *TParseContext::parseInQualifier(const TSourceLoc &loc)
{
    if (declaringFunction())
    {
        return new TStorageQualifierWrapper(EvqIn, loc);
    }

    switch (getShaderType())
    {
        case GL_VERTEX_SHADER:
        {
            if (mShaderVersion < 300 && !anyMultiviewExtensionAvailable() &&
                !IsDesktopGLSpec(mShaderSpec))
            {
                error(loc, "storage qualifier supported in GLSL ES 3.00 and above only", "in");
            }
            return new TStorageQualifierWrapper(EvqVertexIn, loc);
        }
        case GL_FRAGMENT_SHADER:
        {
            if (mShaderVersion < 300 && !IsDesktopGLSpec(mShaderSpec))
            {
                error(loc, "storage qualifier supported in GLSL ES 3.00 and above only", "in");
            }
            return new TStorageQualifierWrapper(EvqFragmentIn, loc);
        }
        case GL_COMPUTE_SHADER:
        {
            return new TStorageQualifierWrapper(EvqComputeIn, loc);
        }
        case GL_GEOMETRY_SHADER_EXT:
        {
            return new TStorageQualifierWrapper(EvqGeometryIn, loc);
        }
        default:
        {
            UNREACHABLE();
            return new TStorageQualifierWrapper(EvqLast, loc);
        }
    }
}

TStorageQualifierWrapper *TParseContext::parseOutQualifier(const TSourceLoc &loc)
{
    if (declaringFunction())
    {
        return new TStorageQualifierWrapper(EvqOut, loc);
    }
    switch (getShaderType())
    {
        case GL_VERTEX_SHADER:
        {
            if (mShaderVersion < 300 && !IsDesktopGLSpec(mShaderSpec))
            {
                error(loc, "storage qualifier supported in GLSL ES 3.00 and above only", "out");
            }
            return new TStorageQualifierWrapper(EvqVertexOut, loc);
        }
        case GL_FRAGMENT_SHADER:
        {
            if (mShaderVersion < 300 && !IsDesktopGLSpec(mShaderSpec))
            {
                error(loc, "storage qualifier supported in GLSL ES 3.00 and above only", "out");
            }
            return new TStorageQualifierWrapper(EvqFragmentOut, loc);
        }
        case GL_COMPUTE_SHADER:
        {
            error(loc, "storage qualifier isn't supported in compute shaders", "out");
            return new TStorageQualifierWrapper(EvqLast, loc);
        }
        case GL_GEOMETRY_SHADER_EXT:
        {
            return new TStorageQualifierWrapper(EvqGeometryOut, loc);
        }
        default:
        {
            UNREACHABLE();
            return new TStorageQualifierWrapper(EvqLast, loc);
        }
    }
}

TStorageQualifierWrapper *TParseContext::parseInOutQualifier(const TSourceLoc &loc)
{
    if (!declaringFunction())
    {
        error(loc, "invalid qualifier: can be only used with function parameters", "inout");
    }
    return new TStorageQualifierWrapper(EvqInOut, loc);
}

TLayoutQualifier TParseContext::joinLayoutQualifiers(TLayoutQualifier leftQualifier,
                                                     TLayoutQualifier rightQualifier,
                                                     const TSourceLoc &rightQualifierLocation)
{
    return sh::JoinLayoutQualifiers(leftQualifier, rightQualifier, rightQualifierLocation,
                                    mDiagnostics);
}

TDeclarator *TParseContext::parseStructDeclarator(const ImmutableString &identifier,
                                                  const TSourceLoc &loc)
{
    checkIsNotReserved(loc, identifier);
    return new TDeclarator(identifier, loc);
}

TDeclarator *TParseContext::parseStructArrayDeclarator(const ImmutableString &identifier,
                                                       const TSourceLoc &loc,
                                                       const TVector<unsigned int> *arraySizes)
{
    checkIsNotReserved(loc, identifier);
    return new TDeclarator(identifier, arraySizes, loc);
}

void TParseContext::checkDoesNotHaveDuplicateFieldName(const TFieldList::const_iterator begin,
                                                       const TFieldList::const_iterator end,
                                                       const ImmutableString &name,
                                                       const TSourceLoc &location)
{
    for (auto fieldIter = begin; fieldIter != end; ++fieldIter)
    {
        if ((*fieldIter)->name() == name)
        {
            error(location, "duplicate field name in structure", name);
        }
    }
}

TFieldList *TParseContext::addStructFieldList(TFieldList *fields, const TSourceLoc &location)
{
    for (TFieldList::const_iterator fieldIter = fields->begin(); fieldIter != fields->end();
         ++fieldIter)
    {
        checkDoesNotHaveDuplicateFieldName(fields->begin(), fieldIter, (*fieldIter)->name(),
                                           location);
    }
    return fields;
}

TFieldList *TParseContext::combineStructFieldLists(TFieldList *processedFields,
                                                   const TFieldList *newlyAddedFields,
                                                   const TSourceLoc &location)
{
    for (TField *field : *newlyAddedFields)
    {
        checkDoesNotHaveDuplicateFieldName(processedFields->begin(), processedFields->end(),
                                           field->name(), location);
        processedFields->push_back(field);
    }
    return processedFields;
}

TFieldList *TParseContext::addStructDeclaratorListWithQualifiers(
    const TTypeQualifierBuilder &typeQualifierBuilder,
    TPublicType *typeSpecifier,
    const TDeclaratorList *declaratorList)
{
    TTypeQualifier typeQualifier = typeQualifierBuilder.getVariableTypeQualifier(mDiagnostics);

    typeSpecifier->qualifier       = typeQualifier.qualifier;
    typeSpecifier->layoutQualifier = typeQualifier.layoutQualifier;
    typeSpecifier->memoryQualifier = typeQualifier.memoryQualifier;
    typeSpecifier->invariant       = typeQualifier.invariant;
    if (typeQualifier.precision != EbpUndefined)
    {
        typeSpecifier->precision = typeQualifier.precision;
    }
    return addStructDeclaratorList(*typeSpecifier, declaratorList);
}

TFieldList *TParseContext::addStructDeclaratorList(const TPublicType &typeSpecifier,
                                                   const TDeclaratorList *declaratorList)
{
    checkPrecisionSpecified(typeSpecifier.getLine(), typeSpecifier.precision,
                            typeSpecifier.getBasicType());

    checkIsNonVoid(typeSpecifier.getLine(), (*declaratorList)[0]->name(),
                   typeSpecifier.getBasicType());

    checkWorkGroupSizeIsNotSpecified(typeSpecifier.getLine(), typeSpecifier.layoutQualifier);

    TFieldList *fieldList = new TFieldList();

    for (const TDeclarator *declarator : *declaratorList)
    {
        TType *type = new TType(typeSpecifier);
        if (declarator->isArray())
        {
            // Don't allow arrays of arrays in ESSL < 3.10.
            checkArrayElementIsNotArray(typeSpecifier.getLine(), typeSpecifier);
            type->makeArrays(*declarator->arraySizes());
        }

        TField *field =
            new TField(type, declarator->name(), declarator->line(), SymbolType::UserDefined);
        checkIsBelowStructNestingLimit(typeSpecifier.getLine(), *field);
        fieldList->push_back(field);
    }

    return fieldList;
}

TTypeSpecifierNonArray TParseContext::addStructure(const TSourceLoc &structLine,
                                                   const TSourceLoc &nameLine,
                                                   const ImmutableString &structName,
                                                   TFieldList *fieldList)
{
    SymbolType structSymbolType = SymbolType::UserDefined;
    if (structName.empty())
    {
        structSymbolType = SymbolType::Empty;
    }
    TStructure *structure = new TStructure(&symbolTable, structName, fieldList, structSymbolType);

    // Store a bool in the struct if we're at global scope, to allow us to
    // skip the local struct scoping workaround in HLSL.
    structure->setAtGlobalScope(symbolTable.atGlobalLevel());

    if (structSymbolType != SymbolType::Empty)
    {
        checkIsNotReserved(nameLine, structName);
        if (!symbolTable.declare(structure))
        {
            error(nameLine, "redefinition of a struct", structName);
        }
    }

    // ensure we do not specify any storage qualifiers on the struct members
    for (unsigned int typeListIndex = 0; typeListIndex < fieldList->size(); typeListIndex++)
    {
        TField &field              = *(*fieldList)[typeListIndex];
        const TQualifier qualifier = field.type()->getQualifier();
        switch (qualifier)
        {
            case EvqGlobal:
            case EvqTemporary:
                break;
            default:
                error(field.line(), "invalid qualifier on struct member",
                      getQualifierString(qualifier));
                break;
        }
        if (field.type()->isInvariant())
        {
            error(field.line(), "invalid qualifier on struct member", "invariant");
        }
        // ESSL 3.10 section 4.1.8 -- atomic_uint or images are not allowed as structure member.
        if (IsImage(field.type()->getBasicType()) || IsAtomicCounter(field.type()->getBasicType()))
        {
            error(field.line(), "disallowed type in struct", field.type()->getBasicString());
        }

        checkIsNotUnsizedArray(field.line(), "array members of structs must specify a size",
                               field.name(), field.type());

        checkMemoryQualifierIsNotSpecified(field.type()->getMemoryQualifier(), field.line());

        checkIndexIsNotSpecified(field.line(), field.type()->getLayoutQualifier().index);

        checkBindingIsNotSpecified(field.line(), field.type()->getLayoutQualifier().binding);

        checkLocationIsNotSpecified(field.line(), field.type()->getLayoutQualifier());
    }

    TTypeSpecifierNonArray typeSpecifierNonArray;
    typeSpecifierNonArray.initializeStruct(structure, true, structLine);
    exitStructDeclaration();

    return typeSpecifierNonArray;
}

TIntermSwitch *TParseContext::addSwitch(TIntermTyped *init,
                                        TIntermBlock *statementList,
                                        const TSourceLoc &loc)
{
    TBasicType switchType = init->getBasicType();
    if ((switchType != EbtInt && switchType != EbtUInt) || init->isMatrix() || init->isArray() ||
        init->isVector())
    {
        error(init->getLine(), "init-expression in a switch statement must be a scalar integer",
              "switch");
        return nullptr;
    }

    ASSERT(statementList);
    if (!ValidateSwitchStatementList(switchType, mDiagnostics, statementList, loc))
    {
        ASSERT(mDiagnostics->numErrors() > 0);
        return nullptr;
    }

    markStaticReadIfSymbol(init);
    TIntermSwitch *node = new TIntermSwitch(init, statementList);
    node->setLine(loc);
    return node;
}

TIntermCase *TParseContext::addCase(TIntermTyped *condition, const TSourceLoc &loc)
{
    if (mSwitchNestingLevel == 0)
    {
        error(loc, "case labels need to be inside switch statements", "case");
        return nullptr;
    }
    if (condition == nullptr)
    {
        error(loc, "case label must have a condition", "case");
        return nullptr;
    }
    if ((condition->getBasicType() != EbtInt && condition->getBasicType() != EbtUInt) ||
        condition->isMatrix() || condition->isArray() || condition->isVector())
    {
        error(condition->getLine(), "case label must be a scalar integer", "case");
    }
    TIntermConstantUnion *conditionConst = condition->getAsConstantUnion();
    // ANGLE should be able to fold any EvqConst expressions resulting in an integer - but to be
    // safe against corner cases we still check for conditionConst. Some interpretations of the
    // spec have allowed constant expressions with side effects - like array length() method on a
    // non-constant array.
    if (condition->getQualifier() != EvqConst || conditionConst == nullptr)
    {
        error(condition->getLine(), "case label must be constant", "case");
    }
    TIntermCase *node = new TIntermCase(condition);
    node->setLine(loc);
    return node;
}

TIntermCase *TParseContext::addDefault(const TSourceLoc &loc)
{
    if (mSwitchNestingLevel == 0)
    {
        error(loc, "default labels need to be inside switch statements", "default");
        return nullptr;
    }
    TIntermCase *node = new TIntermCase(nullptr);
    node->setLine(loc);
    return node;
}

TIntermTyped *TParseContext::createUnaryMath(TOperator op,
                                             TIntermTyped *child,
                                             const TSourceLoc &loc,
                                             const TFunction *func)
{
    ASSERT(child != nullptr);

    switch (op)
    {
        case EOpLogicalNot:
            if (child->getBasicType() != EbtBool || child->isMatrix() || child->isArray() ||
                child->isVector())
            {
                unaryOpError(loc, GetOperatorString(op), child->getType());
                return nullptr;
            }
            break;
        case EOpBitwiseNot:
            if ((child->getBasicType() != EbtInt && child->getBasicType() != EbtUInt) ||
                child->isMatrix() || child->isArray())
            {
                unaryOpError(loc, GetOperatorString(op), child->getType());
                return nullptr;
            }
            break;
        case EOpPostIncrement:
        case EOpPreIncrement:
        case EOpPostDecrement:
        case EOpPreDecrement:
        case EOpNegative:
        case EOpPositive:
            if (child->getBasicType() == EbtStruct || child->isInterfaceBlock() ||
                child->getBasicType() == EbtBool || child->isArray() ||
                child->getBasicType() == EbtVoid || IsOpaqueType(child->getBasicType()))
            {
                unaryOpError(loc, GetOperatorString(op), child->getType());
                return nullptr;
            }
            break;
        // Operators for built-ins are already type checked against their prototype.
        default:
            break;
    }

    if (child->getMemoryQualifier().writeonly)
    {
        unaryOpError(loc, GetOperatorString(op), child->getType());
        return nullptr;
    }

    markStaticReadIfSymbol(child);
    TIntermUnary *node = new TIntermUnary(op, child, func);
    node->setLine(loc);

    return node->fold(mDiagnostics);
}

TIntermTyped *TParseContext::addUnaryMath(TOperator op, TIntermTyped *child, const TSourceLoc &loc)
{
    ASSERT(op != EOpNull);
    TIntermTyped *node = createUnaryMath(op, child, loc, nullptr);
    if (node == nullptr)
    {
        return child;
    }
    return node;
}

TIntermTyped *TParseContext::addUnaryMathLValue(TOperator op,
                                                TIntermTyped *child,
                                                const TSourceLoc &loc)
{
    checkCanBeLValue(loc, GetOperatorString(op), child);
    return addUnaryMath(op, child, loc);
}

TIntermTyped *TParseContext::expressionOrFoldedResult(TIntermTyped *expression)
{
    // If we can, we should return the folded version of the expression for subsequent parsing. This
    // enables folding the containing expression during parsing as well, instead of the separate
    // FoldExpressions() step where folding nested expressions requires multiple full AST
    // traversals.

    // Even if folding fails the fold() functions return some node representing the expression,
    // typically the original node. So "folded" can be assumed to be non-null.
    TIntermTyped *folded = expression->fold(mDiagnostics);
    ASSERT(folded != nullptr);
    if (folded->getQualifier() == expression->getQualifier())
    {
        // We need this expression to have the correct qualifier when validating the consuming
        // expression. So we can only return the folded node from here in case it has the same
        // qualifier as the original expression. In this kind of a cases the qualifier of the folded
        // node is EvqConst, whereas the qualifier of the expression is EvqTemporary:
        //  1. (true ? 1.0 : non_constant)
        //  2. (non_constant, 1.0)
        return folded;
    }
    return expression;
}

bool TParseContext::binaryOpCommonCheck(TOperator op,
                                        TIntermTyped *left,
                                        TIntermTyped *right,
                                        const TSourceLoc &loc)
{
    // Check opaque types are not allowed to be operands in expressions other than array indexing
    // and structure member selection.
    if (IsOpaqueType(left->getBasicType()) || IsOpaqueType(right->getBasicType()))
    {
        switch (op)
        {
            case EOpIndexDirect:
            case EOpIndexIndirect:
                break;

            default:
                ASSERT(op != EOpIndexDirectStruct);
                error(loc, "Invalid operation for variables with an opaque type",
                      GetOperatorString(op));
                return false;
        }
    }

    if (right->getMemoryQualifier().writeonly)
    {
        error(loc, "Invalid operation for variables with writeonly", GetOperatorString(op));
        return false;
    }

    if (left->getMemoryQualifier().writeonly)
    {
        switch (op)
        {
            case EOpAssign:
            case EOpInitialize:
            case EOpIndexDirect:
            case EOpIndexIndirect:
            case EOpIndexDirectStruct:
            case EOpIndexDirectInterfaceBlock:
                break;
            default:
                error(loc, "Invalid operation for variables with writeonly", GetOperatorString(op));
                return false;
        }
    }

    if (left->getType().getStruct() || right->getType().getStruct())
    {
        switch (op)
        {
            case EOpIndexDirectStruct:
                ASSERT(left->getType().getStruct());
                break;
            case EOpEqual:
            case EOpNotEqual:
            case EOpAssign:
            case EOpInitialize:
                if (left->getType() != right->getType())
                {
                    return false;
                }
                break;
            default:
                error(loc, "Invalid operation for structs", GetOperatorString(op));
                return false;
        }
    }

    if (left->isInterfaceBlock() || right->isInterfaceBlock())
    {
        switch (op)
        {
            case EOpIndexDirectInterfaceBlock:
                ASSERT(left->getType().getInterfaceBlock());
                break;
            default:
                error(loc, "Invalid operation for interface blocks", GetOperatorString(op));
                return false;
        }
    }

    if (left->isArray() != right->isArray())
    {
        error(loc, "array / non-array mismatch", GetOperatorString(op));
        return false;
    }

    if (left->isArray())
    {
        ASSERT(right->isArray());
        if (mShaderVersion < 300)
        {
            error(loc, "Invalid operation for arrays", GetOperatorString(op));
            return false;
        }

        switch (op)
        {
            case EOpEqual:
            case EOpNotEqual:
            case EOpAssign:
            case EOpInitialize:
                break;
            default:
                error(loc, "Invalid operation for arrays", GetOperatorString(op));
                return false;
        }
        // At this point, size of implicitly sized arrays should be resolved.
        if (*left->getType().getArraySizes() != *right->getType().getArraySizes())
        {
            error(loc, "array size mismatch", GetOperatorString(op));
            return false;
        }
    }

    // Check ops which require integer / ivec parameters
    bool isBitShift = false;
    switch (op)
    {
        case EOpBitShiftLeft:
        case EOpBitShiftRight:
        case EOpBitShiftLeftAssign:
        case EOpBitShiftRightAssign:
            // Unsigned can be bit-shifted by signed and vice versa, but we need to
            // check that the basic type is an integer type.
            isBitShift = true;
            if (!IsInteger(left->getBasicType()) || !IsInteger(right->getBasicType()))
            {
                return false;
            }
            break;
        case EOpBitwiseAnd:
        case EOpBitwiseXor:
        case EOpBitwiseOr:
        case EOpBitwiseAndAssign:
        case EOpBitwiseXorAssign:
        case EOpBitwiseOrAssign:
            // It is enough to check the type of only one operand, since later it
            // is checked that the operand types match.
            if (!IsInteger(left->getBasicType()))
            {
                return false;
            }
            break;
        default:
            break;
    }

    ImplicitTypeConversion conversion = GetConversion(left->getBasicType(), right->getBasicType());

    // Implicit type casting only supported for GL shaders
    if (!isBitShift && conversion != ImplicitTypeConversion::Same &&
        (!IsDesktopGLSpec(mShaderSpec) || !IsValidImplicitConversion(conversion, op)))
    {
        return false;
    }

    // Check that:
    // 1. Type sizes match exactly on ops that require that.
    // 2. Restrictions for structs that contain arrays or samplers are respected.
    // 3. Arithmetic op type dimensionality restrictions for ops other than multiply are respected.
    switch (op)
    {
        case EOpAssign:
        case EOpInitialize:
        case EOpEqual:
        case EOpNotEqual:
            // ESSL 1.00 sections 5.7, 5.8, 5.9
            if (mShaderVersion < 300 && left->getType().isStructureContainingArrays())
            {
                error(loc, "undefined operation for structs containing arrays",
                      GetOperatorString(op));
                return false;
            }
            // Samplers as l-values are disallowed also in ESSL 3.00, see section 4.1.7,
            // we interpret the spec so that this extends to structs containing samplers,
            // similarly to ESSL 1.00 spec.
            if ((mShaderVersion < 300 || op == EOpAssign || op == EOpInitialize) &&
                left->getType().isStructureContainingSamplers())
            {
                error(loc, "undefined operation for structs containing samplers",
                      GetOperatorString(op));
                return false;
            }

            if ((left->getNominalSize() != right->getNominalSize()) ||
                (left->getSecondarySize() != right->getSecondarySize()))
            {
                error(loc, "dimension mismatch", GetOperatorString(op));
                return false;
            }
            break;
        case EOpLessThan:
        case EOpGreaterThan:
        case EOpLessThanEqual:
        case EOpGreaterThanEqual:
            if (!left->isScalar() || !right->isScalar())
            {
                error(loc, "comparison operator only defined for scalars", GetOperatorString(op));
                return false;
            }
            break;
        case EOpAdd:
        case EOpSub:
        case EOpDiv:
        case EOpIMod:
        case EOpBitShiftLeft:
        case EOpBitShiftRight:
        case EOpBitwiseAnd:
        case EOpBitwiseXor:
        case EOpBitwiseOr:
        case EOpAddAssign:
        case EOpSubAssign:
        case EOpDivAssign:
        case EOpIModAssign:
        case EOpBitShiftLeftAssign:
        case EOpBitShiftRightAssign:
        case EOpBitwiseAndAssign:
        case EOpBitwiseXorAssign:
        case EOpBitwiseOrAssign:
            if ((left->isMatrix() && right->isVector()) || (left->isVector() && right->isMatrix()))
            {
                return false;
            }

            // Are the sizes compatible?
            if (left->getNominalSize() != right->getNominalSize() ||
                left->getSecondarySize() != right->getSecondarySize())
            {
                // If the nominal sizes of operands do not match:
                // One of them must be a scalar.
                if (!left->isScalar() && !right->isScalar())
                    return false;

                // In the case of compound assignment other than multiply-assign,
                // the right side needs to be a scalar. Otherwise a vector/matrix
                // would be assigned to a scalar. A scalar can't be shifted by a
                // vector either.
                if (!right->isScalar() &&
                    (IsAssignment(op) || op == EOpBitShiftLeft || op == EOpBitShiftRight))
                    return false;
            }
            break;
        default:
            break;
    }

    return true;
}

bool TParseContext::isMultiplicationTypeCombinationValid(TOperator op,
                                                         const TType &left,
                                                         const TType &right)
{
    switch (op)
    {
        case EOpMul:
        case EOpMulAssign:
            return left.getNominalSize() == right.getNominalSize() &&
                   left.getSecondarySize() == right.getSecondarySize();
        case EOpVectorTimesScalar:
            return true;
        case EOpVectorTimesScalarAssign:
            ASSERT(!left.isMatrix() && !right.isMatrix());
            return left.isVector() && !right.isVector();
        case EOpVectorTimesMatrix:
            return left.getNominalSize() == right.getRows();
        case EOpVectorTimesMatrixAssign:
            ASSERT(!left.isMatrix() && right.isMatrix());
            return left.isVector() && left.getNominalSize() == right.getRows() &&
                   left.getNominalSize() == right.getCols();
        case EOpMatrixTimesVector:
            return left.getCols() == right.getNominalSize();
        case EOpMatrixTimesScalar:
            return true;
        case EOpMatrixTimesScalarAssign:
            ASSERT(left.isMatrix() && !right.isMatrix());
            return !right.isVector();
        case EOpMatrixTimesMatrix:
            return left.getCols() == right.getRows();
        case EOpMatrixTimesMatrixAssign:
            ASSERT(left.isMatrix() && right.isMatrix());
            // We need to check two things:
            // 1. The matrix multiplication step is valid.
            // 2. The result will have the same number of columns as the lvalue.
            return left.getCols() == right.getRows() && left.getCols() == right.getCols();

        default:
            UNREACHABLE();
            return false;
    }
}

TIntermTyped *TParseContext::addBinaryMathInternal(TOperator op,
                                                   TIntermTyped *left,
                                                   TIntermTyped *right,
                                                   const TSourceLoc &loc)
{
    if (!binaryOpCommonCheck(op, left, right, loc))
        return nullptr;

    switch (op)
    {
        case EOpEqual:
        case EOpNotEqual:
        case EOpLessThan:
        case EOpGreaterThan:
        case EOpLessThanEqual:
        case EOpGreaterThanEqual:
            break;
        case EOpLogicalOr:
        case EOpLogicalXor:
        case EOpLogicalAnd:
            ASSERT(!left->isArray() && !right->isArray() && !left->getType().getStruct() &&
                   !right->getType().getStruct());
            if (left->getBasicType() != EbtBool || !left->isScalar() || !right->isScalar())
            {
                return nullptr;
            }
            // Basic types matching should have been already checked.
            ASSERT(right->getBasicType() == EbtBool);
            break;
        case EOpAdd:
        case EOpSub:
        case EOpDiv:
        case EOpMul:
            ASSERT(!left->isArray() && !right->isArray() && !left->getType().getStruct() &&
                   !right->getType().getStruct());
            if (left->getBasicType() == EbtBool)
            {
                return nullptr;
            }
            break;
        case EOpIMod:
            ASSERT(!left->isArray() && !right->isArray() && !left->getType().getStruct() &&
                   !right->getType().getStruct());
            // Note that this is only for the % operator, not for mod()
            if (left->getBasicType() == EbtBool || left->getBasicType() == EbtFloat)
            {
                return nullptr;
            }
            break;
        default:
            break;
    }

    if (op == EOpMul)
    {
        op = TIntermBinary::GetMulOpBasedOnOperands(left->getType(), right->getType());
        if (!isMultiplicationTypeCombinationValid(op, left->getType(), right->getType()))
        {
            return nullptr;
        }
    }

    TIntermBinary *node = new TIntermBinary(op, left, right);
    ASSERT(op != EOpAssign);
    markStaticReadIfSymbol(left);
    markStaticReadIfSymbol(right);
    node->setLine(loc);
    return expressionOrFoldedResult(node);
}

TIntermTyped *TParseContext::addBinaryMath(TOperator op,
                                           TIntermTyped *left,
                                           TIntermTyped *right,
                                           const TSourceLoc &loc)
{
    TIntermTyped *node = addBinaryMathInternal(op, left, right, loc);
    if (node == 0)
    {
        binaryOpError(loc, GetOperatorString(op), left->getType(), right->getType());
        return left;
    }
    return node;
}

TIntermTyped *TParseContext::addBinaryMathBooleanResult(TOperator op,
                                                        TIntermTyped *left,
                                                        TIntermTyped *right,
                                                        const TSourceLoc &loc)
{
    TIntermTyped *node = addBinaryMathInternal(op, left, right, loc);
    if (node == nullptr)
    {
        binaryOpError(loc, GetOperatorString(op), left->getType(), right->getType());
        node = CreateBoolNode(false);
        node->setLine(loc);
    }
    return node;
}

TIntermTyped *TParseContext::addAssign(TOperator op,
                                       TIntermTyped *left,
                                       TIntermTyped *right,
                                       const TSourceLoc &loc)
{
    checkCanBeLValue(loc, "assign", left);
    TIntermBinary *node = nullptr;
    if (binaryOpCommonCheck(op, left, right, loc))
    {
        if (op == EOpMulAssign)
        {
            op = TIntermBinary::GetMulAssignOpBasedOnOperands(left->getType(), right->getType());
            if (isMultiplicationTypeCombinationValid(op, left->getType(), right->getType()))
            {
                node = new TIntermBinary(op, left, right);
            }
        }
        else
        {
            node = new TIntermBinary(op, left, right);
        }
    }
    if (node == nullptr)
    {
        assignError(loc, "assign", left->getType(), right->getType());
        return left;
    }
    if (op != EOpAssign)
    {
        markStaticReadIfSymbol(left);
    }
    markStaticReadIfSymbol(right);
    node->setLine(loc);
    return node;
}

TIntermTyped *TParseContext::addComma(TIntermTyped *left,
                                      TIntermTyped *right,
                                      const TSourceLoc &loc)
{
    // WebGL2 section 5.26, the following results in an error:
    // "Sequence operator applied to void, arrays, or structs containing arrays"
    if (mShaderSpec == SH_WEBGL2_SPEC &&
        (left->isArray() || left->getBasicType() == EbtVoid ||
         left->getType().isStructureContainingArrays() || right->isArray() ||
         right->getBasicType() == EbtVoid || right->getType().isStructureContainingArrays()))
    {
        error(loc,
              "sequence operator is not allowed for void, arrays, or structs containing arrays",
              ",");
    }

    TIntermBinary *commaNode = TIntermBinary::CreateComma(left, right, mShaderVersion);
    markStaticReadIfSymbol(left);
    markStaticReadIfSymbol(right);
    commaNode->setLine(loc);

    return expressionOrFoldedResult(commaNode);
}

TIntermBranch *TParseContext::addBranch(TOperator op, const TSourceLoc &loc)
{
    switch (op)
    {
        case EOpContinue:
            if (mLoopNestingLevel <= 0)
            {
                error(loc, "continue statement only allowed in loops", "");
            }
            break;
        case EOpBreak:
            if (mLoopNestingLevel <= 0 && mSwitchNestingLevel <= 0)
            {
                error(loc, "break statement only allowed in loops and switch statements", "");
            }
            break;
        case EOpReturn:
            if (mCurrentFunctionType->getBasicType() != EbtVoid)
            {
                error(loc, "non-void function must return a value", "return");
            }
            break;
        case EOpKill:
            if (mShaderType != GL_FRAGMENT_SHADER)
            {
                error(loc, "discard supported in fragment shaders only", "discard");
            }
            break;
        default:
            UNREACHABLE();
            break;
    }
    return addBranch(op, nullptr, loc);
}

TIntermBranch *TParseContext::addBranch(TOperator op,
                                        TIntermTyped *expression,
                                        const TSourceLoc &loc)
{
    if (expression != nullptr)
    {
        markStaticReadIfSymbol(expression);
        ASSERT(op == EOpReturn);
        mFunctionReturnsValue = true;
        if (mCurrentFunctionType->getBasicType() == EbtVoid)
        {
            error(loc, "void function cannot return a value", "return");
        }
        else if (*mCurrentFunctionType != expression->getType())
        {
            error(loc, "function return is not matching type:", "return");
        }
    }
    TIntermBranch *node = new TIntermBranch(op, expression);
    node->setLine(loc);
    return node;
}

void TParseContext::appendStatement(TIntermBlock *block, TIntermNode *statement)
{
    if (statement != nullptr)
    {
        markStaticReadIfSymbol(statement);
        block->appendStatement(statement);
    }
}

void TParseContext::checkTextureGather(TIntermAggregate *functionCall)
{
    ASSERT(functionCall->getOp() == EOpCallBuiltInFunction);
    const TFunction *func = functionCall->getFunction();
    if (BuiltInGroup::isTextureGather(func))
    {
        bool isTextureGatherOffset = BuiltInGroup::isTextureGatherOffset(func);
        TIntermNode *componentNode = nullptr;
        TIntermSequence *arguments = functionCall->getSequence();
        ASSERT(arguments->size() >= 2u && arguments->size() <= 4u);
        const TIntermTyped *sampler = arguments->front()->getAsTyped();
        ASSERT(sampler != nullptr);
        switch (sampler->getBasicType())
        {
            case EbtSampler2D:
            case EbtISampler2D:
            case EbtUSampler2D:
            case EbtSampler2DArray:
            case EbtISampler2DArray:
            case EbtUSampler2DArray:
                if ((!isTextureGatherOffset && arguments->size() == 3u) ||
                    (isTextureGatherOffset && arguments->size() == 4u))
                {
                    componentNode = arguments->back();
                }
                break;
            case EbtSamplerCube:
            case EbtISamplerCube:
            case EbtUSamplerCube:
                ASSERT(!isTextureGatherOffset);
                if (arguments->size() == 3u)
                {
                    componentNode = arguments->back();
                }
                break;
            case EbtSampler2DShadow:
            case EbtSampler2DArrayShadow:
            case EbtSamplerCubeShadow:
                break;
            default:
                UNREACHABLE();
                break;
        }
        if (componentNode)
        {
            const TIntermConstantUnion *componentConstantUnion =
                componentNode->getAsConstantUnion();
            if (componentNode->getAsTyped()->getQualifier() != EvqConst || !componentConstantUnion)
            {
                error(functionCall->getLine(), "Texture component must be a constant expression",
                      func->name());
            }
            else
            {
                int component = componentConstantUnion->getIConst(0);
                if (component < 0 || component > 3)
                {
                    error(functionCall->getLine(), "Component must be in the range [0;3]",
                          func->name());
                }
            }
        }
    }
}

void TParseContext::checkTextureOffsetConst(TIntermAggregate *functionCall)
{
    ASSERT(functionCall->getOp() == EOpCallBuiltInFunction);
    const TFunction *func                  = functionCall->getFunction();
    TIntermNode *offset                    = nullptr;
    TIntermSequence *arguments             = functionCall->getSequence();
    bool useTextureGatherOffsetConstraints = false;
    if (BuiltInGroup::isTextureOffsetNoBias(func))
    {
        offset = arguments->back();
    }
    else if (BuiltInGroup::isTextureOffsetBias(func))
    {
        // A bias parameter follows the offset parameter.
        ASSERT(arguments->size() >= 3);
        offset = (*arguments)[2];
    }
    else if (BuiltInGroup::isTextureGatherOffset(func))
    {
        ASSERT(arguments->size() >= 3u);
        const TIntermTyped *sampler = arguments->front()->getAsTyped();
        ASSERT(sampler != nullptr);
        switch (sampler->getBasicType())
        {
            case EbtSampler2D:
            case EbtISampler2D:
            case EbtUSampler2D:
            case EbtSampler2DArray:
            case EbtISampler2DArray:
            case EbtUSampler2DArray:
                offset = (*arguments)[2];
                break;
            case EbtSampler2DShadow:
            case EbtSampler2DArrayShadow:
                offset = (*arguments)[3];
                break;
            default:
                UNREACHABLE();
                break;
        }
        useTextureGatherOffsetConstraints = true;
    }
    if (offset != nullptr)
    {
        TIntermConstantUnion *offsetConstantUnion = offset->getAsConstantUnion();
        if (offset->getAsTyped()->getQualifier() != EvqConst || !offsetConstantUnion)
        {
            error(functionCall->getLine(), "Texture offset must be a constant expression",
                  func->name());
        }
        else
        {
            ASSERT(offsetConstantUnion->getBasicType() == EbtInt);
            size_t size                  = offsetConstantUnion->getType().getObjectSize();
            const TConstantUnion *values = offsetConstantUnion->getConstantValue();
            int minOffsetValue = useTextureGatherOffsetConstraints ? mMinProgramTextureGatherOffset
                                                                   : mMinProgramTexelOffset;
            int maxOffsetValue = useTextureGatherOffsetConstraints ? mMaxProgramTextureGatherOffset
                                                                   : mMaxProgramTexelOffset;
            for (size_t i = 0u; i < size; ++i)
            {
                int offsetValue = values[i].getIConst();
                if (offsetValue > maxOffsetValue || offsetValue < minOffsetValue)
                {
                    std::stringstream tokenStream = sh::InitializeStream<std::stringstream>();
                    tokenStream << offsetValue;
                    std::string token = tokenStream.str();
                    error(offset->getLine(), "Texture offset value out of valid range",
                          token.c_str());
                }
            }
        }
    }
}

void TParseContext::checkAtomicMemoryBuiltinFunctions(TIntermAggregate *functionCall)
{
    const TFunction *func = functionCall->getFunction();
    if (BuiltInGroup::isAtomicMemory(func))
    {
        ASSERT(IsAtomicFunction(functionCall->getOp()));
        TIntermSequence *arguments = functionCall->getSequence();
        TIntermTyped *memNode      = (*arguments)[0]->getAsTyped();

        if (IsBufferOrSharedVariable(memNode))
        {
            return;
        }

        while (memNode->getAsBinaryNode())
        {
            memNode = memNode->getAsBinaryNode()->getLeft();
            if (IsBufferOrSharedVariable(memNode))
            {
                return;
            }
        }

        error(memNode->getLine(),
              "The value passed to the mem argument of an atomic memory function does not "
              "correspond to a buffer or shared variable.",
              func->name());
    }
}

// GLSL ES 3.10 Revision 4, 4.9 Memory Access Qualifiers
void TParseContext::checkImageMemoryAccessForBuiltinFunctions(TIntermAggregate *functionCall)
{
    ASSERT(functionCall->getOp() == EOpCallBuiltInFunction);

    const TFunction *func = functionCall->getFunction();

    if (BuiltInGroup::isImage(func))
    {
        TIntermSequence *arguments = functionCall->getSequence();
        TIntermTyped *imageNode    = (*arguments)[0]->getAsTyped();

        const TMemoryQualifier &memoryQualifier = imageNode->getMemoryQualifier();

        if (BuiltInGroup::isImageStore(func))
        {
            if (memoryQualifier.readonly)
            {
                error(imageNode->getLine(),
                      "'imageStore' cannot be used with images qualified as 'readonly'",
                      GetImageArgumentToken(imageNode));
            }
        }
        else if (BuiltInGroup::isImageLoad(func))
        {
            if (memoryQualifier.writeonly)
            {
                error(imageNode->getLine(),
                      "'imageLoad' cannot be used with images qualified as 'writeonly'",
                      GetImageArgumentToken(imageNode));
            }
        }
    }
}

// GLSL ES 3.10 Revision 4, 13.51 Matching of Memory Qualifiers in Function Parameters
void TParseContext::checkImageMemoryAccessForUserDefinedFunctions(
    const TFunction *functionDefinition,
    const TIntermAggregate *functionCall)
{
    ASSERT(functionCall->getOp() == EOpCallFunctionInAST);

    const TIntermSequence &arguments = *functionCall->getSequence();

    ASSERT(functionDefinition->getParamCount() == arguments.size());

    for (size_t i = 0; i < arguments.size(); ++i)
    {
        TIntermTyped *typedArgument        = arguments[i]->getAsTyped();
        const TType &functionArgumentType  = typedArgument->getType();
        const TType &functionParameterType = functionDefinition->getParam(i)->getType();
        ASSERT(functionArgumentType.getBasicType() == functionParameterType.getBasicType());

        if (IsImage(functionArgumentType.getBasicType()))
        {
            const TMemoryQualifier &functionArgumentMemoryQualifier =
                functionArgumentType.getMemoryQualifier();
            const TMemoryQualifier &functionParameterMemoryQualifier =
                functionParameterType.getMemoryQualifier();
            if (functionArgumentMemoryQualifier.readonly &&
                !functionParameterMemoryQualifier.readonly)
            {
                error(functionCall->getLine(),
                      "Function call discards the 'readonly' qualifier from image",
                      GetImageArgumentToken(typedArgument));
            }

            if (functionArgumentMemoryQualifier.writeonly &&
                !functionParameterMemoryQualifier.writeonly)
            {
                error(functionCall->getLine(),
                      "Function call discards the 'writeonly' qualifier from image",
                      GetImageArgumentToken(typedArgument));
            }

            if (functionArgumentMemoryQualifier.coherent &&
                !functionParameterMemoryQualifier.coherent)
            {
                error(functionCall->getLine(),
                      "Function call discards the 'coherent' qualifier from image",
                      GetImageArgumentToken(typedArgument));
            }

            if (functionArgumentMemoryQualifier.volatileQualifier &&
                !functionParameterMemoryQualifier.volatileQualifier)
            {
                error(functionCall->getLine(),
                      "Function call discards the 'volatile' qualifier from image",
                      GetImageArgumentToken(typedArgument));
            }
        }
    }
}

TIntermTyped *TParseContext::addFunctionCallOrMethod(TFunctionLookup *fnCall, const TSourceLoc &loc)
{
    if (fnCall->thisNode() != nullptr)
    {
        return addMethod(fnCall, loc);
    }
    if (fnCall->isConstructor())
    {
        return addConstructor(fnCall, loc);
    }
    return addNonConstructorFunctionCall(fnCall, loc);
}

TIntermTyped *TParseContext::addMethod(TFunctionLookup *fnCall, const TSourceLoc &loc)
{
    TIntermTyped *thisNode = fnCall->thisNode();
    // It's possible for the name pointer in the TFunction to be null in case it gets parsed as
    // a constructor. But such a TFunction can't reach here, since the lexer goes into FIELDS
    // mode after a dot, which makes type identifiers to be parsed as FIELD_SELECTION instead.
    // So accessing fnCall->name() below is safe.
    if (fnCall->name() != "length")
    {
        error(loc, "invalid method", fnCall->name());
    }
    else if (!fnCall->arguments().empty())
    {
        error(loc, "method takes no parameters", "length");
    }
    else if (!thisNode->isArray())
    {
        error(loc, "length can only be called on arrays", "length");
    }
    else if (thisNode->getQualifier() == EvqPerVertexIn &&
             mGeometryShaderInputPrimitiveType == EptUndefined)
    {
        ASSERT(mShaderType == GL_GEOMETRY_SHADER_EXT);
        error(loc, "missing input primitive declaration before calling length on gl_in", "length");
    }
    else
    {
        TIntermUnary *node = new TIntermUnary(EOpArrayLength, thisNode, nullptr);
        markStaticReadIfSymbol(thisNode);
        node->setLine(loc);
        return node->fold(mDiagnostics);
    }
    return CreateZeroNode(TType(EbtInt, EbpUndefined, EvqConst));
}

TIntermTyped *TParseContext::addNonConstructorFunctionCall(TFunctionLookup *fnCall,
                                                           const TSourceLoc &loc)
{
    // First check whether the function has been hidden by a variable name or struct typename by
    // using the symbol looked up in the lexical phase. If the function is not hidden, look for one
    // with a matching argument list.
    if (fnCall->symbol() != nullptr && !fnCall->symbol()->isFunction())
    {
        error(loc, "function name expected", fnCall->name());
    }
    else
    {
        // There are no inner functions, so it's enough to look for user-defined functions in the
        // global scope.
        const TSymbol *symbol = symbolTable.findGlobal(fnCall->getMangledName());

        if (symbol == nullptr && IsDesktopGLSpec(mShaderSpec))
        {
            // If using Desktop GL spec, need to check for implicit conversion
            symbol = symbolTable.findGlobalWithConversion(
                fnCall->getMangledNamesForImplicitConversions());
        }

        if (symbol != nullptr)
        {
            // A user-defined function - could be an overloaded built-in as well.
            ASSERT(symbol->symbolType() == SymbolType::UserDefined);
            const TFunction *fnCandidate = static_cast<const TFunction *>(symbol);
            TIntermAggregate *callNode =
                TIntermAggregate::CreateFunctionCall(*fnCandidate, &fnCall->arguments());
            callNode->setLine(loc);
            checkImageMemoryAccessForUserDefinedFunctions(fnCandidate, callNode);
            functionCallRValueLValueErrorCheck(fnCandidate, callNode);
            return callNode;
        }

        symbol = symbolTable.findBuiltIn(fnCall->getMangledName(), mShaderVersion);

        if (symbol == nullptr && IsDesktopGLSpec(mShaderSpec))
        {
            // If using Desktop GL spec, need to check for implicit conversion
            symbol = symbolTable.findBuiltInWithConversion(
                fnCall->getMangledNamesForImplicitConversions(), mShaderVersion);
        }

        if (symbol != nullptr)
        {
            // A built-in function.
            ASSERT(symbol->symbolType() == SymbolType::BuiltIn);
            const TFunction *fnCandidate = static_cast<const TFunction *>(symbol);

            if (fnCandidate->extension() != TExtension::UNDEFINED)
            {
                checkCanUseExtension(loc, fnCandidate->extension());
            }
            TOperator op = fnCandidate->getBuiltInOp();
            if (op != EOpCallBuiltInFunction)
            {
                // A function call mapped to a built-in operation.
                if (fnCandidate->getParamCount() == 1)
                {
                    // Treat it like a built-in unary operator.
                    TIntermNode *unaryParamNode = fnCall->arguments().front();
                    TIntermTyped *callNode =
                        createUnaryMath(op, unaryParamNode->getAsTyped(), loc, fnCandidate);
                    ASSERT(callNode != nullptr);
                    return callNode;
                }

                TIntermAggregate *callNode =
                    TIntermAggregate::CreateBuiltInFunctionCall(*fnCandidate, &fnCall->arguments());
                callNode->setLine(loc);

                checkAtomicMemoryBuiltinFunctions(callNode);

                // Some built-in functions have out parameters too.
                functionCallRValueLValueErrorCheck(fnCandidate, callNode);

                // See if we can constant fold a built-in. Note that this may be possible
                // even if it is not const-qualified.
                return callNode->fold(mDiagnostics);
            }

            // This is a built-in function with no op associated with it.
            TIntermAggregate *callNode =
                TIntermAggregate::CreateBuiltInFunctionCall(*fnCandidate, &fnCall->arguments());
            callNode->setLine(loc);
            checkTextureOffsetConst(callNode);
            checkTextureGather(callNode);
            checkImageMemoryAccessForBuiltinFunctions(callNode);
            functionCallRValueLValueErrorCheck(fnCandidate, callNode);
            return callNode;
        }
        else
        {
            error(loc, "no matching overloaded function found", fnCall->name());
        }
    }

    // Error message was already written. Put on a dummy node for error recovery.
    return CreateZeroNode(TType(EbtFloat, EbpMedium, EvqConst));
}

TIntermTyped *TParseContext::addTernarySelection(TIntermTyped *cond,
                                                 TIntermTyped *trueExpression,
                                                 TIntermTyped *falseExpression,
                                                 const TSourceLoc &loc)
{
    if (!checkIsScalarBool(loc, cond))
    {
        return falseExpression;
    }

    if (trueExpression->getType() != falseExpression->getType())
    {
        TInfoSinkBase reasonStream;
        reasonStream << "mismatching ternary operator operand types '" << trueExpression->getType()
                     << " and '" << falseExpression->getType() << "'";
        error(loc, reasonStream.c_str(), "?:");
        return falseExpression;
    }
    if (IsOpaqueType(trueExpression->getBasicType()))
    {
        // ESSL 1.00 section 4.1.7
        // ESSL 3.00.6 section 4.1.7
        // Opaque/sampler types are not allowed in most types of expressions, including ternary.
        // Note that structs containing opaque types don't need to be checked as structs are
        // forbidden below.
        error(loc, "ternary operator is not allowed for opaque types", "?:");
        return falseExpression;
    }

    if (cond->getMemoryQualifier().writeonly || trueExpression->getMemoryQualifier().writeonly ||
        falseExpression->getMemoryQualifier().writeonly)
    {
        error(loc, "ternary operator is not allowed for variables with writeonly", "?:");
        return falseExpression;
    }

    // ESSL 1.00.17 sections 5.2 and 5.7:
    // Ternary operator is not among the operators allowed for structures/arrays.
    // ESSL 3.00.6 section 5.7:
    // Ternary operator support is optional for arrays. No certainty that it works across all
    // devices with struct either, so we err on the side of caution here. TODO (oetuaho@nvidia.com):
    // Would be nice to make the spec and implementation agree completely here.
    if (trueExpression->isArray() || trueExpression->getBasicType() == EbtStruct)
    {
        error(loc, "ternary operator is not allowed for structures or arrays", "?:");
        return falseExpression;
    }
    if (trueExpression->getBasicType() == EbtInterfaceBlock)
    {
        error(loc, "ternary operator is not allowed for interface blocks", "?:");
        return falseExpression;
    }

    // WebGL2 section 5.26, the following results in an error:
    // "Ternary operator applied to void, arrays, or structs containing arrays"
    if (mShaderSpec == SH_WEBGL2_SPEC && trueExpression->getBasicType() == EbtVoid)
    {
        error(loc, "ternary operator is not allowed for void", "?:");
        return falseExpression;
    }

    TIntermTernary *node = new TIntermTernary(cond, trueExpression, falseExpression);
    markStaticReadIfSymbol(cond);
    markStaticReadIfSymbol(trueExpression);
    markStaticReadIfSymbol(falseExpression);
    node->setLine(loc);
    return expressionOrFoldedResult(node);
}

//
// Parse an array of strings using yyparse.
//
// Returns 0 for success.
//
int PaParseStrings(size_t count,
                   const char *const string[],
                   const int length[],
                   TParseContext *context)
{
    if ((count == 0) || (string == nullptr))
        return 1;

    if (glslang_initialize(context))
        return 1;

    int error = glslang_scan(count, string, length, context);
    if (!error)
        error = glslang_parse(context);

    glslang_finalize(context);

    return (error == 0) && (context->numErrors() == 0) ? 0 : 1;
}

}  // namespace sh

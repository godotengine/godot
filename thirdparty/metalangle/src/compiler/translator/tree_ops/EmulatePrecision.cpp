//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_ops/EmulatePrecision.h"

#include "compiler/translator/FunctionLookup.h"

#include <memory>

namespace sh
{

namespace
{

constexpr const ImmutableString kParamXName("x");
constexpr const ImmutableString kParamYName("y");
constexpr const ImmutableString kAngleFrmString("angle_frm");
constexpr const ImmutableString kAngleFrlString("angle_frl");

class RoundingHelperWriter : angle::NonCopyable
{
  public:
    static RoundingHelperWriter *createHelperWriter(const ShShaderOutput outputLanguage);

    void writeCommonRoundingHelpers(TInfoSinkBase &sink, const int shaderVersion);
    void writeCompoundAssignmentHelper(TInfoSinkBase &sink,
                                       const char *lType,
                                       const char *rType,
                                       const char *opStr,
                                       const char *opNameStr);

    virtual ~RoundingHelperWriter() {}

  protected:
    RoundingHelperWriter(const ShShaderOutput outputLanguage) : mOutputLanguage(outputLanguage) {}
    RoundingHelperWriter() = delete;

    const ShShaderOutput mOutputLanguage;

  private:
    virtual std::string getTypeString(const char *glslType)                               = 0;
    virtual void writeFloatRoundingHelpers(TInfoSinkBase &sink)                           = 0;
    virtual void writeVectorRoundingHelpers(TInfoSinkBase &sink, const unsigned int size) = 0;
    virtual void writeMatrixRoundingHelper(TInfoSinkBase &sink,
                                           const unsigned int columns,
                                           const unsigned int rows,
                                           const char *functionName)                      = 0;
};

class RoundingHelperWriterGLSL : public RoundingHelperWriter
{
  public:
    RoundingHelperWriterGLSL(const ShShaderOutput outputLanguage)
        : RoundingHelperWriter(outputLanguage)
    {}

  private:
    std::string getTypeString(const char *glslType) override;
    void writeFloatRoundingHelpers(TInfoSinkBase &sink) override;
    void writeVectorRoundingHelpers(TInfoSinkBase &sink, const unsigned int size) override;
    void writeMatrixRoundingHelper(TInfoSinkBase &sink,
                                   const unsigned int columns,
                                   const unsigned int rows,
                                   const char *functionName) override;
};

class RoundingHelperWriterESSL : public RoundingHelperWriterGLSL
{
  public:
    RoundingHelperWriterESSL(const ShShaderOutput outputLanguage)
        : RoundingHelperWriterGLSL(outputLanguage)
    {}

  private:
    std::string getTypeString(const char *glslType) override;
};

class RoundingHelperWriterHLSL : public RoundingHelperWriter
{
  public:
    RoundingHelperWriterHLSL(const ShShaderOutput outputLanguage)
        : RoundingHelperWriter(outputLanguage)
    {}

  private:
    std::string getTypeString(const char *glslType) override;
    void writeFloatRoundingHelpers(TInfoSinkBase &sink) override;
    void writeVectorRoundingHelpers(TInfoSinkBase &sink, const unsigned int size) override;
    void writeMatrixRoundingHelper(TInfoSinkBase &sink,
                                   const unsigned int columns,
                                   const unsigned int rows,
                                   const char *functionName) override;
};

RoundingHelperWriter *RoundingHelperWriter::createHelperWriter(const ShShaderOutput outputLanguage)
{
    ASSERT(EmulatePrecision::SupportedInLanguage(outputLanguage));
    switch (outputLanguage)
    {
        case SH_HLSL_4_1_OUTPUT:
            return new RoundingHelperWriterHLSL(outputLanguage);
        case SH_ESSL_OUTPUT:
            return new RoundingHelperWriterESSL(outputLanguage);
        default:
            return new RoundingHelperWriterGLSL(outputLanguage);
    }
}

void RoundingHelperWriter::writeCommonRoundingHelpers(TInfoSinkBase &sink, const int shaderVersion)
{
    // Write the angle_frm functions that round floating point numbers to
    // half precision, and angle_frl functions that round them to minimum lowp
    // precision.

    writeFloatRoundingHelpers(sink);
    writeVectorRoundingHelpers(sink, 2);
    writeVectorRoundingHelpers(sink, 3);
    writeVectorRoundingHelpers(sink, 4);
    if (shaderVersion > 100)
    {
        for (unsigned int columns = 2; columns <= 4; ++columns)
        {
            for (unsigned int rows = 2; rows <= 4; ++rows)
            {
                writeMatrixRoundingHelper(sink, columns, rows, "angle_frm");
                writeMatrixRoundingHelper(sink, columns, rows, "angle_frl");
            }
        }
    }
    else
    {
        for (unsigned int size = 2; size <= 4; ++size)
        {
            writeMatrixRoundingHelper(sink, size, size, "angle_frm");
            writeMatrixRoundingHelper(sink, size, size, "angle_frl");
        }
    }
}

void RoundingHelperWriter::writeCompoundAssignmentHelper(TInfoSinkBase &sink,
                                                         const char *lType,
                                                         const char *rType,
                                                         const char *opStr,
                                                         const char *opNameStr)
{
    std::string lTypeStr = getTypeString(lType);
    std::string rTypeStr = getTypeString(rType);

    // Note that y should be passed through angle_frm at the function call site,
    // but x can't be passed through angle_frm there since it is an inout parameter.
    // So only pass x and the result through angle_frm here.
    // clang-format off
    sink <<
        lTypeStr << " angle_compound_" << opNameStr << "_frm(inout " << lTypeStr << " x, in " << rTypeStr << " y) {\n"
        "    x = angle_frm(angle_frm(x) " << opStr << " y);\n"
        "    return x;\n"
        "}\n";
    sink <<
        lTypeStr << " angle_compound_" << opNameStr << "_frl(inout " << lTypeStr << " x, in " << rTypeStr << " y) {\n"
        "    x = angle_frl(angle_frl(x) " << opStr << " y);\n"
        "    return x;\n"
        "}\n";
    // clang-format on
}

std::string RoundingHelperWriterGLSL::getTypeString(const char *glslType)
{
    return glslType;
}

std::string RoundingHelperWriterESSL::getTypeString(const char *glslType)
{
    std::stringstream typeStrStr = sh::InitializeStream<std::stringstream>();
    typeStrStr << "highp " << glslType;
    return typeStrStr.str();
}

void RoundingHelperWriterGLSL::writeFloatRoundingHelpers(TInfoSinkBase &sink)
{
    // Unoptimized version of angle_frm for single floats:
    //
    // int webgl_maxNormalExponent(in int exponentBits)
    // {
    //     int possibleExponents = int(exp2(float(exponentBits)));
    //     int exponentBias = possibleExponents / 2 - 1;
    //     int allExponentBitsOne = possibleExponents - 1;
    //     return (allExponentBitsOne - 1) - exponentBias;
    // }
    //
    // float angle_frm(in float x)
    // {
    //     int mantissaBits = 10;
    //     int exponentBits = 5;
    //     float possibleMantissas = exp2(float(mantissaBits));
    //     float mantissaMax = 2.0 - 1.0 / possibleMantissas;
    //     int maxNE = webgl_maxNormalExponent(exponentBits);
    //     float max = exp2(float(maxNE)) * mantissaMax;
    //     if (x > max)
    //     {
    //         return max;
    //     }
    //     if (x < -max)
    //     {
    //         return -max;
    //     }
    //     float exponent = floor(log2(abs(x)));
    //     if (abs(x) == 0.0 || exponent < -float(maxNE))
    //     {
    //         return 0.0 * sign(x)
    //     }
    //     x = x * exp2(-(exponent - float(mantissaBits)));
    //     x = sign(x) * floor(abs(x));
    //     return x * exp2(exponent - float(mantissaBits));
    // }

    // All numbers with a magnitude less than 2^-15 are subnormal, and are
    // flushed to zero.

    // Note the constant numbers below:
    // a) 65504 is the maximum possible mantissa (1.1111111111 in binary) times
    //    2^15, the maximum normal exponent.
    // b) 10.0 is the number of mantissa bits.
    // c) -25.0 is the minimum normal half-float exponent -15.0 minus the number
    //    of mantissa bits.
    // d) + 1e-30 is to make sure the argument of log2() won't be zero. It can
    //    only affect the result of log2 on x where abs(x) < 1e-22. Since these
    //    numbers will be flushed to zero either way (2^-15 is the smallest
    //    normal positive number), this does not introduce any error.

    std::string floatType = getTypeString("float");

    // clang-format off
    sink <<
        floatType << " angle_frm(in " << floatType << " x) {\n"
        "    x = clamp(x, -65504.0, 65504.0);\n"
        "    " << floatType << " exponent = floor(log2(abs(x) + 1e-30)) - 10.0;\n"
        "    bool isNonZero = (exponent >= -25.0);\n"
        "    x = x * exp2(-exponent);\n"
        "    x = sign(x) * floor(abs(x));\n"
        "    return x * exp2(exponent) * float(isNonZero);\n"
        "}\n";

    sink <<
        floatType << " angle_frl(in " << floatType << " x) {\n"
        "    x = clamp(x, -2.0, 2.0);\n"
        "    x = x * 256.0;\n"
        "    x = sign(x) * floor(abs(x));\n"
        "    return x * 0.00390625;\n"
        "}\n";
    // clang-format on
}

void RoundingHelperWriterGLSL::writeVectorRoundingHelpers(TInfoSinkBase &sink,
                                                          const unsigned int size)
{
    std::stringstream vecTypeStrStr = sh::InitializeStream<std::stringstream>();
    vecTypeStrStr << "vec" << size;
    std::string vecType = getTypeString(vecTypeStrStr.str().c_str());

    // clang-format off
    sink <<
        vecType << " angle_frm(in " << vecType << " v) {\n"
        "    v = clamp(v, -65504.0, 65504.0);\n"
        "    " << vecType << " exponent = floor(log2(abs(v) + 1e-30)) - 10.0;\n"
        "    bvec" << size << " isNonZero = greaterThanEqual(exponent, vec" << size << "(-25.0));\n"
        "    v = v * exp2(-exponent);\n"
        "    v = sign(v) * floor(abs(v));\n"
        "    return v * exp2(exponent) * vec" << size << "(isNonZero);\n"
        "}\n";

    sink <<
        vecType << " angle_frl(in " << vecType << " v) {\n"
        "    v = clamp(v, -2.0, 2.0);\n"
        "    v = v * 256.0;\n"
        "    v = sign(v) * floor(abs(v));\n"
        "    return v * 0.00390625;\n"
        "}\n";
    // clang-format on
}

void RoundingHelperWriterGLSL::writeMatrixRoundingHelper(TInfoSinkBase &sink,
                                                         const unsigned int columns,
                                                         const unsigned int rows,
                                                         const char *functionName)
{
    std::stringstream matTypeStrStr = sh::InitializeStream<std::stringstream>();
    matTypeStrStr << "mat" << columns;
    if (rows != columns)
    {
        matTypeStrStr << "x" << rows;
    }
    std::string matType = getTypeString(matTypeStrStr.str().c_str());

    sink << matType << " " << functionName << "(in " << matType << " m) {\n"
         << "    " << matType << " rounded;\n";

    for (unsigned int i = 0; i < columns; ++i)
    {
        sink << "    rounded[" << i << "] = " << functionName << "(m[" << i << "]);\n";
    }

    sink << "    return rounded;\n"
            "}\n";
}

static const char *GetHLSLTypeStr(const char *floatTypeStr)
{
    if (strcmp(floatTypeStr, "float") == 0)
    {
        return "float";
    }
    if (strcmp(floatTypeStr, "vec2") == 0)
    {
        return "float2";
    }
    if (strcmp(floatTypeStr, "vec3") == 0)
    {
        return "float3";
    }
    if (strcmp(floatTypeStr, "vec4") == 0)
    {
        return "float4";
    }
    if (strcmp(floatTypeStr, "mat2") == 0)
    {
        return "float2x2";
    }
    if (strcmp(floatTypeStr, "mat3") == 0)
    {
        return "float3x3";
    }
    if (strcmp(floatTypeStr, "mat4") == 0)
    {
        return "float4x4";
    }
    if (strcmp(floatTypeStr, "mat2x3") == 0)
    {
        return "float2x3";
    }
    if (strcmp(floatTypeStr, "mat2x4") == 0)
    {
        return "float2x4";
    }
    if (strcmp(floatTypeStr, "mat3x2") == 0)
    {
        return "float3x2";
    }
    if (strcmp(floatTypeStr, "mat3x4") == 0)
    {
        return "float3x4";
    }
    if (strcmp(floatTypeStr, "mat4x2") == 0)
    {
        return "float4x2";
    }
    if (strcmp(floatTypeStr, "mat4x3") == 0)
    {
        return "float4x3";
    }
    UNREACHABLE();
    return nullptr;
}

std::string RoundingHelperWriterHLSL::getTypeString(const char *glslType)
{
    return GetHLSLTypeStr(glslType);
}

void RoundingHelperWriterHLSL::writeFloatRoundingHelpers(TInfoSinkBase &sink)
{
    // In HLSL scalars are the same as 1-vectors.
    writeVectorRoundingHelpers(sink, 1);
}

void RoundingHelperWriterHLSL::writeVectorRoundingHelpers(TInfoSinkBase &sink,
                                                          const unsigned int size)
{
    std::stringstream vecTypeStrStr = sh::InitializeStream<std::stringstream>();
    vecTypeStrStr << "float" << size;
    std::string vecType = vecTypeStrStr.str();

    // clang-format off
    sink <<
        vecType << " angle_frm(" << vecType << " v) {\n"
        "    v = clamp(v, -65504.0, 65504.0);\n"
        "    " << vecType << " exponent = floor(log2(abs(v) + 1e-30)) - 10.0;\n"
        "    bool" << size << " isNonZero = exponent < -25.0;\n"
        "    v = v * exp2(-exponent);\n"
        "    v = sign(v) * floor(abs(v));\n"
        "    return v * exp2(exponent) * (float" << size << ")(isNonZero);\n"
        "}\n";

    sink <<
        vecType << " angle_frl(" << vecType << " v) {\n"
        "    v = clamp(v, -2.0, 2.0);\n"
        "    v = v * 256.0;\n"
        "    v = sign(v) * floor(abs(v));\n"
        "    return v * 0.00390625;\n"
        "}\n";
    // clang-format on
}

void RoundingHelperWriterHLSL::writeMatrixRoundingHelper(TInfoSinkBase &sink,
                                                         const unsigned int columns,
                                                         const unsigned int rows,
                                                         const char *functionName)
{
    std::stringstream matTypeStrStr = sh::InitializeStream<std::stringstream>();
    matTypeStrStr << "float" << columns << "x" << rows;
    std::string matType = matTypeStrStr.str();

    sink << matType << " " << functionName << "(" << matType << " m) {\n"
         << "    " << matType << " rounded;\n";

    for (unsigned int i = 0; i < columns; ++i)
    {
        sink << "    rounded[" << i << "] = " << functionName << "(m[" << i << "]);\n";
    }

    sink << "    return rounded;\n"
            "}\n";
}

bool canRoundFloat(const TType &type)
{
    return type.getBasicType() == EbtFloat && !type.isArray() &&
           (type.getPrecision() == EbpLow || type.getPrecision() == EbpMedium);
}

bool ParentUsesResult(TIntermNode *parent, TIntermTyped *node)
{
    if (!parent)
    {
        return false;
    }

    TIntermBlock *blockParent = parent->getAsBlock();
    // If the parent is a block, the result is not assigned anywhere,
    // so rounding it is not needed. In particular, this can avoid a lot of
    // unnecessary rounding of unused return values of assignment.
    if (blockParent)
    {
        return false;
    }
    TIntermBinary *binaryParent = parent->getAsBinaryNode();
    if (binaryParent && binaryParent->getOp() == EOpComma && (binaryParent->getRight() != node))
    {
        return false;
    }
    return true;
}

bool ParentConstructorTakesCareOfRounding(TIntermNode *parent, TIntermTyped *node)
{
    if (!parent)
    {
        return false;
    }
    TIntermAggregate *parentConstructor = parent->getAsAggregate();
    if (!parentConstructor || parentConstructor->getOp() != EOpConstruct)
    {
        return false;
    }
    if (parentConstructor->getPrecision() != node->getPrecision())
    {
        return false;
    }
    return canRoundFloat(parentConstructor->getType());
}

}  // namespace

EmulatePrecision::EmulatePrecision(TSymbolTable *symbolTable)
    : TLValueTrackingTraverser(true, true, true, symbolTable), mDeclaringVariables(false)
{}

void EmulatePrecision::visitSymbol(TIntermSymbol *node)
{
    TIntermNode *parent = getParentNode();
    if (canRoundFloat(node->getType()) && ParentUsesResult(parent, node) &&
        !ParentConstructorTakesCareOfRounding(parent, node) && !mDeclaringVariables &&
        !isLValueRequiredHere())
    {
        TIntermNode *replacement = createRoundingFunctionCallNode(node);
        queueReplacement(replacement, OriginalNode::BECOMES_CHILD);
    }
}

bool EmulatePrecision::visitBinary(Visit visit, TIntermBinary *node)
{
    bool visitChildren = true;

    TOperator op = node->getOp();

    // RHS of initialize is not being declared.
    if (op == EOpInitialize && visit == InVisit)
        mDeclaringVariables = false;

    if ((op == EOpIndexDirectStruct) && visit == InVisit)
        visitChildren = false;

    if (visit != PreVisit)
        return visitChildren;

    const TType &type = node->getType();
    bool roundFloat   = canRoundFloat(type);

    if (roundFloat)
    {
        switch (op)
        {
            // Math operators that can result in a float may need to apply rounding to the return
            // value. Note that in the case of assignment, the rounding is applied to its return
            // value here, not the value being assigned.
            case EOpAssign:
            case EOpAdd:
            case EOpSub:
            case EOpMul:
            case EOpDiv:
            case EOpVectorTimesScalar:
            case EOpVectorTimesMatrix:
            case EOpMatrixTimesVector:
            case EOpMatrixTimesScalar:
            case EOpMatrixTimesMatrix:
            {
                TIntermNode *parent = getParentNode();
                if (!ParentUsesResult(parent, node) ||
                    ParentConstructorTakesCareOfRounding(parent, node))
                {
                    break;
                }
                TIntermNode *replacement = createRoundingFunctionCallNode(node);
                queueReplacement(replacement, OriginalNode::BECOMES_CHILD);
                break;
            }

            // Compound assignment cases need to replace the operator with a function call.
            case EOpAddAssign:
            {
                mEmulateCompoundAdd.insert(
                    TypePair(type.getBuiltInTypeNameString(),
                             node->getRight()->getType().getBuiltInTypeNameString()));
                TIntermNode *replacement = createCompoundAssignmentFunctionCallNode(
                    node->getLeft(), node->getRight(), "add");
                queueReplacement(replacement, OriginalNode::IS_DROPPED);
                break;
            }
            case EOpSubAssign:
            {
                mEmulateCompoundSub.insert(
                    TypePair(type.getBuiltInTypeNameString(),
                             node->getRight()->getType().getBuiltInTypeNameString()));
                TIntermNode *replacement = createCompoundAssignmentFunctionCallNode(
                    node->getLeft(), node->getRight(), "sub");
                queueReplacement(replacement, OriginalNode::IS_DROPPED);
                break;
            }
            case EOpMulAssign:
            case EOpVectorTimesMatrixAssign:
            case EOpVectorTimesScalarAssign:
            case EOpMatrixTimesScalarAssign:
            case EOpMatrixTimesMatrixAssign:
            {
                mEmulateCompoundMul.insert(
                    TypePair(type.getBuiltInTypeNameString(),
                             node->getRight()->getType().getBuiltInTypeNameString()));
                TIntermNode *replacement = createCompoundAssignmentFunctionCallNode(
                    node->getLeft(), node->getRight(), "mul");
                queueReplacement(replacement, OriginalNode::IS_DROPPED);
                break;
            }
            case EOpDivAssign:
            {
                mEmulateCompoundDiv.insert(
                    TypePair(type.getBuiltInTypeNameString(),
                             node->getRight()->getType().getBuiltInTypeNameString()));
                TIntermNode *replacement = createCompoundAssignmentFunctionCallNode(
                    node->getLeft(), node->getRight(), "div");
                queueReplacement(replacement, OriginalNode::IS_DROPPED);
                break;
            }
            default:
                // The rest of the binary operations should not need precision emulation.
                break;
        }
    }
    return visitChildren;
}

bool EmulatePrecision::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    // Variable or interface block declaration.
    if (visit == PreVisit)
    {
        mDeclaringVariables = true;
    }
    else if (visit == InVisit)
    {
        mDeclaringVariables = true;
    }
    else
    {
        mDeclaringVariables = false;
    }
    return true;
}

bool EmulatePrecision::visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node)
{
    return false;
}

bool EmulatePrecision::visitAggregate(Visit visit, TIntermAggregate *node)
{
    if (visit != PreVisit)
        return true;

    // User-defined function return values are not rounded. The calculations that produced
    // the value inside the function definition should have been rounded.
    TOperator op = node->getOp();
    if (op == EOpCallInternalRawFunction || op == EOpCallFunctionInAST ||
        (op == EOpConstruct && node->getBasicType() == EbtStruct))
    {
        return true;
    }

    TIntermNode *parent = getParentNode();
    if (canRoundFloat(node->getType()) && ParentUsesResult(parent, node) &&
        !ParentConstructorTakesCareOfRounding(parent, node))
    {
        TIntermNode *replacement = createRoundingFunctionCallNode(node);
        queueReplacement(replacement, OriginalNode::BECOMES_CHILD);
    }
    return true;
}

bool EmulatePrecision::visitUnary(Visit visit, TIntermUnary *node)
{
    switch (node->getOp())
    {
        case EOpNegative:
        case EOpLogicalNot:
        case EOpPostIncrement:
        case EOpPostDecrement:
        case EOpPreIncrement:
        case EOpPreDecrement:
        case EOpLogicalNotComponentWise:
            break;
        default:
            if (canRoundFloat(node->getType()) && visit == PreVisit)
            {
                TIntermNode *replacement = createRoundingFunctionCallNode(node);
                queueReplacement(replacement, OriginalNode::BECOMES_CHILD);
            }
            break;
    }

    return true;
}

void EmulatePrecision::writeEmulationHelpers(TInfoSinkBase &sink,
                                             const int shaderVersion,
                                             const ShShaderOutput outputLanguage)
{
    std::unique_ptr<RoundingHelperWriter> roundingHelperWriter(
        RoundingHelperWriter::createHelperWriter(outputLanguage));

    roundingHelperWriter->writeCommonRoundingHelpers(sink, shaderVersion);

    EmulationSet::const_iterator it;
    for (it = mEmulateCompoundAdd.begin(); it != mEmulateCompoundAdd.end(); it++)
        roundingHelperWriter->writeCompoundAssignmentHelper(sink, it->lType, it->rType, "+", "add");
    for (it = mEmulateCompoundSub.begin(); it != mEmulateCompoundSub.end(); it++)
        roundingHelperWriter->writeCompoundAssignmentHelper(sink, it->lType, it->rType, "-", "sub");
    for (it = mEmulateCompoundDiv.begin(); it != mEmulateCompoundDiv.end(); it++)
        roundingHelperWriter->writeCompoundAssignmentHelper(sink, it->lType, it->rType, "/", "div");
    for (it = mEmulateCompoundMul.begin(); it != mEmulateCompoundMul.end(); it++)
        roundingHelperWriter->writeCompoundAssignmentHelper(sink, it->lType, it->rType, "*", "mul");
}

// static
bool EmulatePrecision::SupportedInLanguage(const ShShaderOutput outputLanguage)
{
    switch (outputLanguage)
    {
        case SH_HLSL_4_1_OUTPUT:
        case SH_ESSL_OUTPUT:
            return true;
        default:
            // Other languages not yet supported
            return (outputLanguage == SH_GLSL_COMPATIBILITY_OUTPUT ||
                    sh::IsGLSL130OrNewer(outputLanguage));
    }
}

const TFunction *EmulatePrecision::getInternalFunction(const ImmutableString &functionName,
                                                       const TType &returnType,
                                                       TIntermSequence *arguments,
                                                       const TVector<const TVariable *> &parameters,
                                                       bool knownToNotHaveSideEffects)
{
    ImmutableString mangledName = TFunctionLookup::GetMangledName(functionName.data(), *arguments);
    if (mInternalFunctions.find(mangledName) == mInternalFunctions.end())
    {
        TFunction *func = new TFunction(mSymbolTable, functionName, SymbolType::AngleInternal,
                                        new TType(returnType), knownToNotHaveSideEffects);
        ASSERT(parameters.size() == arguments->size());
        for (size_t i = 0; i < parameters.size(); ++i)
        {
            func->addParameter(parameters[i]);
        }
        mInternalFunctions[mangledName] = func;
    }
    return mInternalFunctions[mangledName];
}

TIntermAggregate *EmulatePrecision::createRoundingFunctionCallNode(TIntermTyped *roundedChild)
{
    const ImmutableString *roundFunctionName = &kAngleFrmString;
    if (roundedChild->getPrecision() == EbpLow)
        roundFunctionName = &kAngleFrlString;
    TIntermSequence *arguments = new TIntermSequence();
    arguments->push_back(roundedChild);

    TVector<const TVariable *> parameters;
    TType *paramType = new TType(roundedChild->getType());
    paramType->setPrecision(EbpHigh);
    paramType->setQualifier(EvqIn);
    parameters.push_back(new TVariable(mSymbolTable, kParamXName,
                                       static_cast<const TType *>(paramType),
                                       SymbolType::AngleInternal));

    return TIntermAggregate::CreateRawFunctionCall(
        *getInternalFunction(*roundFunctionName, roundedChild->getType(), arguments, parameters,
                             true),
        arguments);
}

TIntermAggregate *EmulatePrecision::createCompoundAssignmentFunctionCallNode(TIntermTyped *left,
                                                                             TIntermTyped *right,
                                                                             const char *opNameStr)
{
    std::stringstream strstr = sh::InitializeStream<std::stringstream>();
    if (left->getPrecision() == EbpMedium)
        strstr << "angle_compound_" << opNameStr << "_frm";
    else
        strstr << "angle_compound_" << opNameStr << "_frl";
    ImmutableString functionName = ImmutableString(strstr.str());
    TIntermSequence *arguments   = new TIntermSequence();
    arguments->push_back(left);
    arguments->push_back(right);

    TVector<const TVariable *> parameters;
    TType *leftParamType = new TType(left->getType());
    leftParamType->setPrecision(EbpHigh);
    leftParamType->setQualifier(EvqOut);
    parameters.push_back(new TVariable(mSymbolTable, kParamXName,
                                       static_cast<const TType *>(leftParamType),
                                       SymbolType::AngleInternal));
    TType *rightParamType = new TType(right->getType());
    rightParamType->setPrecision(EbpHigh);
    rightParamType->setQualifier(EvqIn);
    parameters.push_back(new TVariable(mSymbolTable, kParamYName,
                                       static_cast<const TType *>(rightParamType),
                                       SymbolType::AngleInternal));

    return TIntermAggregate::CreateRawFunctionCall(
        *getInternalFunction(functionName, left->getType(), arguments, parameters, false),
        arguments);
}

}  // namespace sh

//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/OutputGLSLBase.h"

#include "angle_gl.h"
#include "common/debug.h"
#include "common/mathutil.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/util.h"

#include <cfloat>

namespace sh
{

namespace
{

bool isSingleStatement(TIntermNode *node)
{
    if (node->getAsFunctionDefinition())
    {
        return false;
    }
    else if (node->getAsBlock())
    {
        return false;
    }
    else if (node->getAsIfElseNode())
    {
        return false;
    }
    else if (node->getAsLoopNode())
    {
        return false;
    }
    else if (node->getAsSwitchNode())
    {
        return false;
    }
    else if (node->getAsCaseNode())
    {
        return false;
    }
    else if (node->getAsPreprocessorDirective())
    {
        return false;
    }
    return true;
}

class CommaSeparatedListItemPrefixGenerator
{
  public:
    CommaSeparatedListItemPrefixGenerator() : mFirst(true) {}

  private:
    bool mFirst;

    template <typename Stream>
    friend Stream &operator<<(Stream &out, CommaSeparatedListItemPrefixGenerator &gen);
};

template <typename Stream>
Stream &operator<<(Stream &out, CommaSeparatedListItemPrefixGenerator &gen)
{
    if (gen.mFirst)
    {
        gen.mFirst = false;
    }
    else
    {
        out << ", ";
    }
    return out;
}

}  // namespace

TOutputGLSLBase::TOutputGLSLBase(TInfoSinkBase &objSink,
                                 ShArrayIndexClampingStrategy clampingStrategy,
                                 ShHashFunction64 hashFunction,
                                 NameMap &nameMap,
                                 TSymbolTable *symbolTable,
                                 sh::GLenum shaderType,
                                 int shaderVersion,
                                 ShShaderOutput output,
                                 ShCompileOptions compileOptions)
    : TIntermTraverser(true, true, true, symbolTable),
      mObjSink(objSink),
      mDeclaringVariable(false),
      mClampingStrategy(clampingStrategy),
      mHashFunction(hashFunction),
      mNameMap(nameMap),
      mShaderType(shaderType),
      mShaderVersion(shaderVersion),
      mOutput(output),
      mCompileOptions(compileOptions)
{}

void TOutputGLSLBase::writeInvariantQualifier(const TType &type)
{
    if (!sh::RemoveInvariant(mShaderType, mShaderVersion, mOutput, mCompileOptions))
    {
        TInfoSinkBase &out = objSink();
        out << "invariant ";
    }
}

void TOutputGLSLBase::writeFloat(TInfoSinkBase &out, float f)
{
    if ((gl::isInf(f) || gl::isNaN(f)) && mShaderVersion >= 300)
    {
        out << "uintBitsToFloat(" << gl::bitCast<uint32_t>(f) << "u)";
    }
    else
    {
        out << std::min(FLT_MAX, std::max(-FLT_MAX, f));
    }
}

void TOutputGLSLBase::writeTriplet(Visit visit,
                                   const char *preStr,
                                   const char *inStr,
                                   const char *postStr)
{
    TInfoSinkBase &out = objSink();
    if (visit == PreVisit && preStr)
        out << preStr;
    else if (visit == InVisit && inStr)
        out << inStr;
    else if (visit == PostVisit && postStr)
        out << postStr;
}

void TOutputGLSLBase::writeBuiltInFunctionTriplet(Visit visit,
                                                  TOperator op,
                                                  bool useEmulatedFunction)
{
    TInfoSinkBase &out = objSink();
    if (visit == PreVisit)
    {
        const char *opStr(GetOperatorString(op));
        if (useEmulatedFunction)
        {
            BuiltInFunctionEmulator::WriteEmulatedFunctionName(out, opStr);
        }
        else
        {
            out << opStr;
        }
        out << "(";
    }
    else
    {
        writeTriplet(visit, nullptr, ", ", ")");
    }
}

// Outputs what goes inside layout(), except for location and binding qualifiers, as they are
// handled differently between GL GLSL and Vulkan GLSL.
std::string TOutputGLSLBase::getCommonLayoutQualifiers(TIntermTyped *variable)
{
    std::ostringstream out;
    CommaSeparatedListItemPrefixGenerator listItemPrefix;

    const TType &type                       = variable->getType();
    const TLayoutQualifier &layoutQualifier = type.getLayoutQualifier();

    if (type.getQualifier() == EvqFragmentOut || type.getQualifier() == EvqVertexIn ||
        IsVarying(type.getQualifier()))
    {
        if (type.getQualifier() == EvqFragmentOut && layoutQualifier.index >= 0)
        {
            out << listItemPrefix << "index = " << layoutQualifier.index;
        }
    }

    if (type.getQualifier() == EvqFragmentOut)
    {
        if (layoutQualifier.yuv == true)
        {
            out << listItemPrefix << "yuv";
        }
    }

    if (IsImage(type.getBasicType()))
    {
        if (layoutQualifier.imageInternalFormat != EiifUnspecified)
        {
            ASSERT(type.getQualifier() == EvqTemporary || type.getQualifier() == EvqUniform);
            out << listItemPrefix
                << getImageInternalFormatString(layoutQualifier.imageInternalFormat);
        }
    }

    if (IsAtomicCounter(type.getBasicType()))
    {
        out << listItemPrefix << "offset = " << layoutQualifier.offset;
    }

    return out.str();
}

// Outputs what comes after in/out/uniform/buffer storage qualifier.
std::string TOutputGLSLBase::getMemoryQualifiers(const TType &type)
{
    std::ostringstream out;

    const TMemoryQualifier &memoryQualifier = type.getMemoryQualifier();
    if (memoryQualifier.readonly)
    {
        ASSERT(IsImage(type.getBasicType()) || IsStorageBuffer(type.getQualifier()));
        out << "readonly ";
    }

    if (memoryQualifier.writeonly)
    {
        ASSERT(IsImage(type.getBasicType()) || IsStorageBuffer(type.getQualifier()));
        out << "writeonly ";
    }

    if (memoryQualifier.coherent)
    {
        ASSERT(IsImage(type.getBasicType()) || IsStorageBuffer(type.getQualifier()));
        out << "coherent ";
    }

    if (memoryQualifier.restrictQualifier)
    {
        ASSERT(IsImage(type.getBasicType()) || IsStorageBuffer(type.getQualifier()));
        out << "restrict ";
    }

    if (memoryQualifier.volatileQualifier)
    {
        ASSERT(IsImage(type.getBasicType()) || IsStorageBuffer(type.getQualifier()));
        out << "volatile ";
    }

    return out.str();
}

void TOutputGLSLBase::writeLayoutQualifier(TIntermTyped *variable)
{
    const TType &type = variable->getType();

    if (!NeedsToWriteLayoutQualifier(type))
    {
        return;
    }

    if (type.getBasicType() == EbtInterfaceBlock)
    {
        const TInterfaceBlock *interfaceBlock = type.getInterfaceBlock();
        declareInterfaceBlockLayout(interfaceBlock);
        return;
    }

    TInfoSinkBase &out                      = objSink();
    const TLayoutQualifier &layoutQualifier = type.getLayoutQualifier();
    out << "layout(";

    CommaSeparatedListItemPrefixGenerator listItemPrefix;

    if (type.getQualifier() == EvqFragmentOut || type.getQualifier() == EvqVertexIn ||
        IsVarying(type.getQualifier()))
    {
        if (layoutQualifier.location >= 0)
        {
            out << listItemPrefix << "location = " << layoutQualifier.location;
        }
    }

    if (IsOpaqueType(type.getBasicType()))
    {
        if (layoutQualifier.binding >= 0)
        {
            out << listItemPrefix << "binding = " << layoutQualifier.binding;
        }
    }

    std::string otherQualifiers = getCommonLayoutQualifiers(variable);
    if (!otherQualifiers.empty())
    {
        out << listItemPrefix << otherQualifiers;
    }

    out << ") ";
}

void TOutputGLSLBase::writeFieldLayoutQualifier(const TField *field)
{
    if (!field->type()->isMatrix() && !field->type()->isStructureContainingMatrices())
    {
        return;
    }

    TInfoSinkBase &out = objSink();

    out << "layout(";
    switch (field->type()->getLayoutQualifier().matrixPacking)
    {
        case EmpUnspecified:
        case EmpColumnMajor:
            // Default matrix packing is column major.
            out << "column_major";
            break;

        case EmpRowMajor:
            out << "row_major";
            break;

        default:
            UNREACHABLE();
            break;
    }
    out << ") ";
}

void TOutputGLSLBase::writeQualifier(TQualifier qualifier, const TType &type, const TSymbol *symbol)
{
    const char *result = mapQualifierToString(qualifier);
    if (result && result[0] != '\0')
    {
        objSink() << result << " ";
    }

    objSink() << getMemoryQualifiers(type);
}

const char *TOutputGLSLBase::mapQualifierToString(TQualifier qualifier)
{
    if (sh::IsGLSL410OrOlder(mOutput) && mShaderVersion >= 300 &&
        (mCompileOptions & SH_REMOVE_INVARIANT_AND_CENTROID_FOR_ESSL3) != 0)
    {
        switch (qualifier)
        {
            // The return string is consistent with sh::getQualifierString() from
            // BaseTypes.h minus the "centroid" keyword.
            case EvqCentroid:
                return "";
            case EvqCentroidIn:
                return "smooth in";
            case EvqCentroidOut:
                return "smooth out";
            default:
                break;
        }
    }
    if (sh::IsGLSL130OrNewer(mOutput))
    {
        switch (qualifier)
        {
            case EvqAttribute:
                return "in";
            case EvqVaryingIn:
                return "in";
            case EvqVaryingOut:
                return "out";
            default:
                break;
        }
    }
    return sh::getQualifierString(qualifier);
}

void TOutputGLSLBase::writeVariableType(const TType &type, const TSymbol *symbol)
{
    TQualifier qualifier = type.getQualifier();
    TInfoSinkBase &out   = objSink();
    if (type.isInvariant())
    {
        writeInvariantQualifier(type);
    }
    if (qualifier != EvqTemporary && qualifier != EvqGlobal)
    {
        writeQualifier(qualifier, type, symbol);
    }

    // Declare the struct if we have not done so already.
    if (type.getBasicType() == EbtStruct && !structDeclared(type.getStruct()))
    {
        const TStructure *structure = type.getStruct();

        declareStruct(structure);
    }
    else if (type.getBasicType() == EbtInterfaceBlock)
    {
        const TInterfaceBlock *interfaceBlock = type.getInterfaceBlock();
        declareInterfaceBlock(interfaceBlock);
    }
    else
    {
        if (writeVariablePrecision(type.getPrecision()))
            out << " ";
        out << getTypeName(type);
    }
}

void TOutputGLSLBase::writeFunctionParameters(const TFunction *func)
{
    TInfoSinkBase &out = objSink();
    size_t paramCount  = func->getParamCount();
    for (size_t i = 0; i < paramCount; ++i)
    {
        const TVariable *param = func->getParam(i);
        const TType &type      = param->getType();
        writeVariableType(type, param);

        if (param->symbolType() != SymbolType::Empty)
            out << " " << hashName(param);
        if (type.isArray())
            out << ArrayString(type);

        // Put a comma if this is not the last argument.
        if (i != paramCount - 1)
            out << ", ";
    }
}

const TConstantUnion *TOutputGLSLBase::writeConstantUnion(const TType &type,
                                                          const TConstantUnion *pConstUnion)
{
    TInfoSinkBase &out = objSink();

    if (type.getBasicType() == EbtStruct)
    {
        const TStructure *structure = type.getStruct();
        out << hashName(structure) << "(";

        const TFieldList &fields = structure->fields();
        for (size_t i = 0; i < fields.size(); ++i)
        {
            const TType *fieldType = fields[i]->type();
            ASSERT(fieldType != nullptr);
            pConstUnion = writeConstantUnion(*fieldType, pConstUnion);
            if (i != fields.size() - 1)
                out << ", ";
        }
        out << ")";
    }
    else
    {
        size_t size    = type.getObjectSize();
        bool writeType = size > 1;
        if (writeType)
            out << getTypeName(type) << "(";
        for (size_t i = 0; i < size; ++i, ++pConstUnion)
        {
            switch (pConstUnion->getType())
            {
                case EbtFloat:
                    writeFloat(out, pConstUnion->getFConst());
                    break;
                case EbtInt:
                    out << pConstUnion->getIConst();
                    break;
                case EbtUInt:
                    out << pConstUnion->getUConst() << "u";
                    break;
                case EbtBool:
                    out << pConstUnion->getBConst();
                    break;
                case EbtYuvCscStandardEXT:
                    out << getYuvCscStandardEXTString(pConstUnion->getYuvCscStandardEXTConst());
                    break;
                default:
                    UNREACHABLE();
            }
            if (i != size - 1)
                out << ", ";
        }
        if (writeType)
            out << ")";
    }
    return pConstUnion;
}

void TOutputGLSLBase::writeConstructorTriplet(Visit visit, const TType &type)
{
    TInfoSinkBase &out = objSink();
    if (visit == PreVisit)
    {
        if (type.isArray())
        {
            out << getTypeName(type);
            out << ArrayString(type);
            out << "(";
        }
        else
        {
            out << getTypeName(type) << "(";
        }
    }
    else
    {
        writeTriplet(visit, nullptr, ", ", ")");
    }
}

void TOutputGLSLBase::visitSymbol(TIntermSymbol *node)
{
    TInfoSinkBase &out = objSink();
    out << hashName(&node->variable());

    if (mDeclaringVariable && node->getType().isArray())
        out << ArrayString(node->getType());
}

void TOutputGLSLBase::visitConstantUnion(TIntermConstantUnion *node)
{
    writeConstantUnion(node->getType(), node->getConstantValue());
}

bool TOutputGLSLBase::visitSwizzle(Visit visit, TIntermSwizzle *node)
{
    TInfoSinkBase &out = objSink();
    if (visit == PostVisit)
    {
        out << ".";
        node->writeOffsetsAsXYZW(&out);
    }
    return true;
}

bool TOutputGLSLBase::visitBinary(Visit visit, TIntermBinary *node)
{
    bool visitChildren = true;
    TInfoSinkBase &out = objSink();
    switch (node->getOp())
    {
        case EOpComma:
            writeTriplet(visit, "(", ", ", ")");
            break;
        case EOpInitialize:
            if (visit == InVisit)
            {
                out << " = ";
                // RHS of initialize is not being declared.
                mDeclaringVariable = false;
            }
            break;
        case EOpAssign:
            writeTriplet(visit, "(", " = ", ")");
            break;
        case EOpAddAssign:
            writeTriplet(visit, "(", " += ", ")");
            break;
        case EOpSubAssign:
            writeTriplet(visit, "(", " -= ", ")");
            break;
        case EOpDivAssign:
            writeTriplet(visit, "(", " /= ", ")");
            break;
        case EOpIModAssign:
            writeTriplet(visit, "(", " %= ", ")");
            break;
        // Notice the fall-through.
        case EOpMulAssign:
        case EOpVectorTimesMatrixAssign:
        case EOpVectorTimesScalarAssign:
        case EOpMatrixTimesScalarAssign:
        case EOpMatrixTimesMatrixAssign:
            writeTriplet(visit, "(", " *= ", ")");
            break;
        case EOpBitShiftLeftAssign:
            writeTriplet(visit, "(", " <<= ", ")");
            break;
        case EOpBitShiftRightAssign:
            writeTriplet(visit, "(", " >>= ", ")");
            break;
        case EOpBitwiseAndAssign:
            writeTriplet(visit, "(", " &= ", ")");
            break;
        case EOpBitwiseXorAssign:
            writeTriplet(visit, "(", " ^= ", ")");
            break;
        case EOpBitwiseOrAssign:
            writeTriplet(visit, "(", " |= ", ")");
            break;

        case EOpIndexDirect:
            writeTriplet(visit, nullptr, "[", "]");
            break;
        case EOpIndexIndirect:
            if (node->getAddIndexClamp())
            {
                if (visit == InVisit)
                {
                    if (mClampingStrategy == SH_CLAMP_WITH_CLAMP_INTRINSIC)
                        out << "[int(clamp(float(";
                    else
                        out << "[webgl_int_clamp(";
                }
                else if (visit == PostVisit)
                {
                    TIntermTyped *left = node->getLeft();
                    TType leftType     = left->getType();

                    if (mClampingStrategy == SH_CLAMP_WITH_CLAMP_INTRINSIC)
                        out << "), 0.0, float(";
                    else
                        out << ", 0, ";

                    if (leftType.isUnsizedArray())
                    {
                        // For runtime-sized arrays in ESSL 3.10 we need to call the length method
                        // to get the length to clamp against. See ESSL 3.10 section 4.1.9. Note
                        // that a runtime-sized array expression is guaranteed not to have side
                        // effects, so it's fine to add the expression to the output twice.
                        ASSERT(mShaderVersion >= 310);
                        ASSERT(!left->hasSideEffects());
                        left->traverse(this);
                        out << ".length() - 1";
                    }
                    else
                    {
                        int maxSize;
                        if (leftType.isArray())
                        {
                            maxSize = static_cast<int>(leftType.getOutermostArraySize()) - 1;
                        }
                        else
                        {
                            maxSize = leftType.getNominalSize() - 1;
                        }
                        out << maxSize;
                    }
                    if (mClampingStrategy == SH_CLAMP_WITH_CLAMP_INTRINSIC)
                        out << ")))]";
                    else
                        out << ")]";
                }
            }
            else
            {
                writeTriplet(visit, nullptr, "[", "]");
            }
            break;
        case EOpIndexDirectStruct:
            if (visit == InVisit)
            {
                // Here we are writing out "foo.bar", where "foo" is struct
                // and "bar" is field. In AST, it is represented as a binary
                // node, where left child represents "foo" and right child "bar".
                // The node itself represents ".". The struct field "bar" is
                // actually stored as an index into TStructure::fields.
                out << ".";
                const TStructure *structure       = node->getLeft()->getType().getStruct();
                const TIntermConstantUnion *index = node->getRight()->getAsConstantUnion();
                const TField *field               = structure->fields()[index->getIConst(0)];

                out << hashFieldName(field);
                visitChildren = false;
            }
            break;
        case EOpIndexDirectInterfaceBlock:
            if (visit == InVisit)
            {
                out << ".";
                const TInterfaceBlock *interfaceBlock =
                    node->getLeft()->getType().getInterfaceBlock();
                const TIntermConstantUnion *index = node->getRight()->getAsConstantUnion();
                const TField *field               = interfaceBlock->fields()[index->getIConst(0)];
                out << hashFieldName(field);
                visitChildren = false;
            }
            break;

        case EOpAdd:
            writeTriplet(visit, "(", " + ", ")");
            break;
        case EOpSub:
            writeTriplet(visit, "(", " - ", ")");
            break;
        case EOpMul:
            writeTriplet(visit, "(", " * ", ")");
            break;
        case EOpDiv:
            writeTriplet(visit, "(", " / ", ")");
            break;
        case EOpIMod:
            writeTriplet(visit, "(", " % ", ")");
            break;
        case EOpBitShiftLeft:
            writeTriplet(visit, "(", " << ", ")");
            break;
        case EOpBitShiftRight:
            writeTriplet(visit, "(", " >> ", ")");
            break;
        case EOpBitwiseAnd:
            writeTriplet(visit, "(", " & ", ")");
            break;
        case EOpBitwiseXor:
            writeTriplet(visit, "(", " ^ ", ")");
            break;
        case EOpBitwiseOr:
            writeTriplet(visit, "(", " | ", ")");
            break;

        case EOpEqual:
            writeTriplet(visit, "(", " == ", ")");
            break;
        case EOpNotEqual:
            writeTriplet(visit, "(", " != ", ")");
            break;
        case EOpLessThan:
            writeTriplet(visit, "(", " < ", ")");
            break;
        case EOpGreaterThan:
            writeTriplet(visit, "(", " > ", ")");
            break;
        case EOpLessThanEqual:
            writeTriplet(visit, "(", " <= ", ")");
            break;
        case EOpGreaterThanEqual:
            writeTriplet(visit, "(", " >= ", ")");
            break;

        // Notice the fall-through.
        case EOpVectorTimesScalar:
        case EOpVectorTimesMatrix:
        case EOpMatrixTimesVector:
        case EOpMatrixTimesScalar:
        case EOpMatrixTimesMatrix:
            writeTriplet(visit, "(", " * ", ")");
            break;

        case EOpLogicalOr:
            writeTriplet(visit, "(", " || ", ")");
            break;
        case EOpLogicalXor:
            writeTriplet(visit, "(", " ^^ ", ")");
            break;
        case EOpLogicalAnd:
            writeTriplet(visit, "(", " && ", ")");
            break;
        default:
            UNREACHABLE();
    }

    return visitChildren;
}

bool TOutputGLSLBase::visitUnary(Visit visit, TIntermUnary *node)
{
    const char *preString  = "";
    const char *postString = ")";

    switch (node->getOp())
    {
        case EOpNegative:
            preString = "(-";
            break;
        case EOpPositive:
            preString = "(+";
            break;
        case EOpLogicalNot:
            preString = "(!";
            break;
        case EOpBitwiseNot:
            preString = "(~";
            break;

        case EOpPostIncrement:
            preString  = "(";
            postString = "++)";
            break;
        case EOpPostDecrement:
            preString  = "(";
            postString = "--)";
            break;
        case EOpPreIncrement:
            preString = "(++";
            break;
        case EOpPreDecrement:
            preString = "(--";
            break;
        case EOpArrayLength:
            preString  = "((";
            postString = ").length())";
            break;

        case EOpRadians:
        case EOpDegrees:
        case EOpSin:
        case EOpCos:
        case EOpTan:
        case EOpAsin:
        case EOpAcos:
        case EOpAtan:
        case EOpSinh:
        case EOpCosh:
        case EOpTanh:
        case EOpAsinh:
        case EOpAcosh:
        case EOpAtanh:
        case EOpExp:
        case EOpLog:
        case EOpExp2:
        case EOpLog2:
        case EOpSqrt:
        case EOpInversesqrt:
        case EOpAbs:
        case EOpSign:
        case EOpFloor:
        case EOpTrunc:
        case EOpRound:
        case EOpRoundEven:
        case EOpCeil:
        case EOpFract:
        case EOpIsnan:
        case EOpIsinf:
        case EOpFloatBitsToInt:
        case EOpFloatBitsToUint:
        case EOpIntBitsToFloat:
        case EOpUintBitsToFloat:
        case EOpPackSnorm2x16:
        case EOpPackUnorm2x16:
        case EOpPackHalf2x16:
        case EOpUnpackSnorm2x16:
        case EOpUnpackUnorm2x16:
        case EOpUnpackHalf2x16:
        case EOpPackUnorm4x8:
        case EOpPackSnorm4x8:
        case EOpUnpackUnorm4x8:
        case EOpUnpackSnorm4x8:
        case EOpLength:
        case EOpNormalize:
        case EOpDFdx:
        case EOpDFdy:
        case EOpFwidth:
        case EOpTranspose:
        case EOpDeterminant:
        case EOpInverse:
        case EOpAny:
        case EOpAll:
        case EOpLogicalNotComponentWise:
        case EOpBitfieldReverse:
        case EOpBitCount:
        case EOpFindLSB:
        case EOpFindMSB:
            writeBuiltInFunctionTriplet(visit, node->getOp(), node->getUseEmulatedFunction());
            return true;
        default:
            UNREACHABLE();
    }

    writeTriplet(visit, preString, nullptr, postString);

    return true;
}

bool TOutputGLSLBase::visitTernary(Visit visit, TIntermTernary *node)
{
    TInfoSinkBase &out = objSink();
    // Notice two brackets at the beginning and end. The outer ones
    // encapsulate the whole ternary expression. This preserves the
    // order of precedence when ternary expressions are used in a
    // compound expression, i.e., c = 2 * (a < b ? 1 : 2).
    out << "((";
    node->getCondition()->traverse(this);
    out << ") ? (";
    node->getTrueExpression()->traverse(this);
    out << ") : (";
    node->getFalseExpression()->traverse(this);
    out << "))";
    return false;
}

bool TOutputGLSLBase::visitIfElse(Visit visit, TIntermIfElse *node)
{
    TInfoSinkBase &out = objSink();

    out << "if (";
    node->getCondition()->traverse(this);
    out << ")\n";

    visitCodeBlock(node->getTrueBlock());

    if (node->getFalseBlock())
    {
        out << "else\n";
        visitCodeBlock(node->getFalseBlock());
    }
    return false;
}

bool TOutputGLSLBase::visitSwitch(Visit visit, TIntermSwitch *node)
{
    ASSERT(node->getStatementList());
    writeTriplet(visit, "switch (", ") ", nullptr);
    // The curly braces get written when visiting the statementList aggregate
    return true;
}

bool TOutputGLSLBase::visitCase(Visit visit, TIntermCase *node)
{
    if (node->hasCondition())
    {
        writeTriplet(visit, "case (", nullptr, "):\n");
        return true;
    }
    else
    {
        TInfoSinkBase &out = objSink();
        out << "default:\n";
        return false;
    }
}

bool TOutputGLSLBase::visitBlock(Visit visit, TIntermBlock *node)
{
    TInfoSinkBase &out = objSink();
    // Scope the blocks except when at the global scope.
    if (getCurrentTraversalDepth() > 0)
    {
        out << "{\n";
    }

    for (TIntermSequence::const_iterator iter = node->getSequence()->begin();
         iter != node->getSequence()->end(); ++iter)
    {
        TIntermNode *curNode = *iter;
        ASSERT(curNode != nullptr);
        curNode->traverse(this);

        if (isSingleStatement(curNode))
            out << ";\n";
    }

    // Scope the blocks except when at the global scope.
    if (getCurrentTraversalDepth() > 0)
    {
        out << "}\n";
    }
    return false;
}

bool TOutputGLSLBase::visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node)
{
    TIntermFunctionPrototype *prototype = node->getFunctionPrototype();
    prototype->traverse(this);
    visitCodeBlock(node->getBody());

    // Fully processed; no need to visit children.
    return false;
}

bool TOutputGLSLBase::visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node)
{
    TInfoSinkBase &out = objSink();
    ASSERT(visit == PreVisit);
    const TIntermSymbol *symbol = node->getSymbol();
    out << "invariant " << hashName(&symbol->variable());
    return false;
}

void TOutputGLSLBase::visitFunctionPrototype(TIntermFunctionPrototype *node)
{
    TInfoSinkBase &out = objSink();

    const TType &type = node->getType();
    writeVariableType(type, node->getFunction());
    if (type.isArray())
        out << ArrayString(type);

    out << " " << hashFunctionNameIfNeeded(node->getFunction());

    out << "(";
    writeFunctionParameters(node->getFunction());
    out << ")";
}

bool TOutputGLSLBase::visitAggregate(Visit visit, TIntermAggregate *node)
{
    bool visitChildren = true;
    TInfoSinkBase &out = objSink();
    switch (node->getOp())
    {
        case EOpCallFunctionInAST:
        case EOpCallInternalRawFunction:
        case EOpCallBuiltInFunction:
            // Function call.
            if (visit == PreVisit)
            {
                if (node->getOp() == EOpCallBuiltInFunction)
                {
                    out << translateTextureFunction(node->getFunction()->name());
                }
                else
                {
                    out << hashFunctionNameIfNeeded(node->getFunction());
                }
                out << "(";
            }
            else if (visit == InVisit)
                out << ", ";
            else
                out << ")";
            break;
        case EOpConstruct:
            writeConstructorTriplet(visit, node->getType());
            break;

        case EOpEqualComponentWise:
        case EOpNotEqualComponentWise:
        case EOpLessThanComponentWise:
        case EOpGreaterThanComponentWise:
        case EOpLessThanEqualComponentWise:
        case EOpGreaterThanEqualComponentWise:
        case EOpMod:
        case EOpModf:
        case EOpPow:
        case EOpAtan:
        case EOpMin:
        case EOpMax:
        case EOpClamp:
        case EOpMix:
        case EOpStep:
        case EOpSmoothstep:
        case EOpFrexp:
        case EOpLdexp:
        case EOpDistance:
        case EOpDot:
        case EOpCross:
        case EOpFaceforward:
        case EOpReflect:
        case EOpRefract:
        case EOpMulMatrixComponentWise:
        case EOpOuterProduct:
        case EOpBitfieldExtract:
        case EOpBitfieldInsert:
        case EOpUaddCarry:
        case EOpUsubBorrow:
        case EOpUmulExtended:
        case EOpImulExtended:
        case EOpBarrier:
        case EOpMemoryBarrier:
        case EOpMemoryBarrierAtomicCounter:
        case EOpMemoryBarrierBuffer:
        case EOpMemoryBarrierImage:
        case EOpMemoryBarrierShared:
        case EOpGroupMemoryBarrier:
        case EOpAtomicAdd:
        case EOpAtomicMin:
        case EOpAtomicMax:
        case EOpAtomicAnd:
        case EOpAtomicOr:
        case EOpAtomicXor:
        case EOpAtomicExchange:
        case EOpAtomicCompSwap:
        case EOpEmitVertex:
        case EOpEndPrimitive:
            writeBuiltInFunctionTriplet(visit, node->getOp(), node->getUseEmulatedFunction());
            break;
        default:
            UNREACHABLE();
    }
    return visitChildren;
}

bool TOutputGLSLBase::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    TInfoSinkBase &out = objSink();

    // Variable declaration.
    if (visit == PreVisit)
    {
        const TIntermSequence &sequence = *(node->getSequence());
        TIntermTyped *variable          = sequence.front()->getAsTyped();
        TIntermSymbol *symbolNode = variable->getAsSymbolNode();
        if (!symbolNode || symbolNode->getName() != "gl_ClipDistance")
        {
            // gl_ClipDistance re-declaration doesn't need layout.
            writeLayoutQualifier(variable);
        }
        writeVariableType(variable->getType(), symbolNode ? &symbolNode->variable() : nullptr);
        if (variable->getAsSymbolNode() == nullptr ||
            variable->getAsSymbolNode()->variable().symbolType() != SymbolType::Empty)
        {
            out << " ";
        }
        mDeclaringVariable = true;
    }
    else if (visit == InVisit)
    {
        UNREACHABLE();
    }
    else
    {
        mDeclaringVariable = false;
    }
    return true;
}

bool TOutputGLSLBase::visitLoop(Visit visit, TIntermLoop *node)
{
    TInfoSinkBase &out = objSink();

    TLoopType loopType = node->getType();

    if (loopType == ELoopFor)  // for loop
    {
        out << "for (";
        if (node->getInit())
            node->getInit()->traverse(this);
        out << "; ";

        if (node->getCondition())
            node->getCondition()->traverse(this);
        out << "; ";

        if (node->getExpression())
            node->getExpression()->traverse(this);
        out << ")\n";

        visitCodeBlock(node->getBody());
    }
    else if (loopType == ELoopWhile)  // while loop
    {
        out << "while (";
        ASSERT(node->getCondition() != nullptr);
        node->getCondition()->traverse(this);
        out << ")\n";

        visitCodeBlock(node->getBody());
    }
    else  // do-while loop
    {
        ASSERT(loopType == ELoopDoWhile);
        out << "do\n";

        visitCodeBlock(node->getBody());

        out << "while (";
        ASSERT(node->getCondition() != nullptr);
        node->getCondition()->traverse(this);
        out << ");\n";
    }

    // No need to visit children. They have been already processed in
    // this function.
    return false;
}

bool TOutputGLSLBase::visitBranch(Visit visit, TIntermBranch *node)
{
    switch (node->getFlowOp())
    {
        case EOpKill:
            writeTriplet(visit, "discard", nullptr, nullptr);
            break;
        case EOpBreak:
            writeTriplet(visit, "break", nullptr, nullptr);
            break;
        case EOpContinue:
            writeTriplet(visit, "continue", nullptr, nullptr);
            break;
        case EOpReturn:
            writeTriplet(visit, "return ", nullptr, nullptr);
            break;
        default:
            UNREACHABLE();
    }

    return true;
}

void TOutputGLSLBase::visitCodeBlock(TIntermBlock *node)
{
    TInfoSinkBase &out = objSink();
    if (node != nullptr)
    {
        node->traverse(this);
        // Single statements not part of a sequence need to be terminated
        // with semi-colon.
        if (isSingleStatement(node))
            out << ";\n";
    }
    else
    {
        out << "{\n}\n";  // Empty code block.
    }
}

void TOutputGLSLBase::visitPreprocessorDirective(TIntermPreprocessorDirective *node)
{
    TInfoSinkBase &out = objSink();

    out << "\n";

    switch (node->getDirective())
    {
        case PreprocessorDirective::Define:
            out << "#define";
            break;
        case PreprocessorDirective::Endif:
            out << "#endif";
            break;
        case PreprocessorDirective::If:
            out << "#if";
            break;
        case PreprocessorDirective::Ifdef:
            out << "#ifdef";
            break;

        default:
            UNREACHABLE();
            break;
    }

    if (!node->getCommand().empty())
    {
        out << " " << node->getCommand();
    }

    out << "\n";
}

ImmutableString TOutputGLSLBase::getTypeName(const TType &type)
{
    return GetTypeName(type, mHashFunction, &mNameMap);
}

ImmutableString TOutputGLSLBase::hashName(const TSymbol *symbol)
{
    return HashName(symbol, mHashFunction, &mNameMap);
}

ImmutableString TOutputGLSLBase::hashFieldName(const TField *field)
{
    ASSERT(field->symbolType() != SymbolType::Empty);
    if (field->symbolType() == SymbolType::UserDefined)
    {
        return HashName(field->name(), mHashFunction, &mNameMap);
    }

    return field->name();
}

ImmutableString TOutputGLSLBase::hashFunctionNameIfNeeded(const TFunction *func)
{
    if (func->isMain())
    {
        return func->name();
    }
    else
    {
        return hashName(func);
    }
}

bool TOutputGLSLBase::structDeclared(const TStructure *structure) const
{
    ASSERT(structure);
    if (structure->symbolType() == SymbolType::Empty)
    {
        return false;
    }

    return (mDeclaredStructs.count(structure->uniqueId().get()) > 0);
}

void TOutputGLSLBase::declareStruct(const TStructure *structure)
{
    TInfoSinkBase &out = objSink();

    out << "struct ";

    if (structure->symbolType() != SymbolType::Empty)
    {
        out << hashName(structure) << " ";
    }
    out << "{\n";
    const TFieldList &fields = structure->fields();
    for (size_t i = 0; i < fields.size(); ++i)
    {
        const TField *field = fields[i];
        if (writeVariablePrecision(field->type()->getPrecision()))
            out << " ";
        out << getTypeName(*field->type()) << " " << hashFieldName(field);
        if (field->type()->isArray())
            out << ArrayString(*field->type());
        out << ";\n";
    }
    out << "}";

    if (structure->symbolType() != SymbolType::Empty)
    {
        mDeclaredStructs.insert(structure->uniqueId().get());
    }
}

void TOutputGLSLBase::declareInterfaceBlockLayout(const TInterfaceBlock *interfaceBlock)
{
    TInfoSinkBase &out = objSink();

    out << "layout(";

    switch (interfaceBlock->blockStorage())
    {
        case EbsUnspecified:
        case EbsShared:
            // Default block storage is shared.
            out << "shared";
            break;

        case EbsPacked:
            out << "packed";
            break;

        case EbsStd140:
            out << "std140";
            break;

        case EbsStd430:
            out << "std430";
            break;

        default:
            UNREACHABLE();
            break;
    }

    if (interfaceBlock->blockBinding() >= 0)
    {
        out << ", ";
        out << "binding = " << interfaceBlock->blockBinding();
    }

    out << ") ";
}

void TOutputGLSLBase::declareInterfaceBlock(const TInterfaceBlock *interfaceBlock)
{
    TInfoSinkBase &out = objSink();

    out << hashName(interfaceBlock) << "{\n";
    const TFieldList &fields = interfaceBlock->fields();
    for (const TField *field : fields)
    {
        writeFieldLayoutQualifier(field);

        if (writeVariablePrecision(field->type()->getPrecision()))
            out << " ";
        out << getTypeName(*field->type()) << " " << hashFieldName(field);
        if (field->type()->isArray())
            out << ArrayString(*field->type());
        out << ";\n";
    }
    out << "}";
}

void WriteGeometryShaderLayoutQualifiers(TInfoSinkBase &out,
                                         sh::TLayoutPrimitiveType inputPrimitive,
                                         int invocations,
                                         sh::TLayoutPrimitiveType outputPrimitive,
                                         int maxVertices)
{
    // Omit 'invocations = 1'
    if (inputPrimitive != EptUndefined || invocations > 1)
    {
        out << "layout (";

        if (inputPrimitive != EptUndefined)
        {
            out << getGeometryShaderPrimitiveTypeString(inputPrimitive);
        }

        if (invocations > 1)
        {
            if (inputPrimitive != EptUndefined)
            {
                out << ", ";
            }
            out << "invocations = " << invocations;
        }
        out << ") in;\n";
    }

    if (outputPrimitive != EptUndefined || maxVertices != -1)
    {
        out << "layout (";

        if (outputPrimitive != EptUndefined)
        {
            out << getGeometryShaderPrimitiveTypeString(outputPrimitive);
        }

        if (maxVertices != -1)
        {
            if (outputPrimitive != EptUndefined)
            {
                out << ", ";
            }
            out << "max_vertices = " << maxVertices;
        }
        out << ") out;\n";
    }
}

// If SH_SCALARIZE_VEC_AND_MAT_CONSTRUCTOR_ARGS is enabled, layout qualifiers are spilled whenever
// variables with specified layout qualifiers are copied. Additional checks are needed against the
// type and storage qualifier of the variable to verify that layout qualifiers have to be outputted.
// TODO (mradev): Fix layout qualifier spilling in ScalarizeVecAndMatConstructorArgs and remove
// NeedsToWriteLayoutQualifier.
bool NeedsToWriteLayoutQualifier(const TType &type)
{
    if (type.getBasicType() == EbtInterfaceBlock)
    {
        return true;
    }

    const TLayoutQualifier &layoutQualifier = type.getLayoutQualifier();

    if ((type.getQualifier() == EvqFragmentOut || type.getQualifier() == EvqVertexIn ||
         IsVarying(type.getQualifier())) &&
        layoutQualifier.location >= 0)
    {
        return true;
    }

    if (type.getQualifier() == EvqFragmentOut && layoutQualifier.yuv == true)
    {
        return true;
    }

    if (IsOpaqueType(type.getBasicType()) && layoutQualifier.binding != -1)
    {
        return true;
    }

    if (IsImage(type.getBasicType()) && layoutQualifier.imageInternalFormat != EiifUnspecified)
    {
        return true;
    }
    return false;
}

}  // namespace sh

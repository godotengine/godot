//
// Copyright 2010 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/util.h"

#include <limits>

#include "common/utilities.h"
#include "compiler/preprocessor/numeric_lex.h"
#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/SymbolTable.h"

bool atoi_clamp(const char *str, unsigned int *value)
{
    bool success = angle::pp::numeric_lex_int(str, value);
    if (!success)
        *value = std::numeric_limits<unsigned int>::max();
    return success;
}

namespace sh
{

namespace
{

bool IsInterpolationIn(TQualifier qualifier)
{
    switch (qualifier)
    {
        case EvqSmoothIn:
        case EvqFlatIn:
        case EvqCentroidIn:
            return true;
        default:
            return false;
    }
}

}  // anonymous namespace

float NumericLexFloat32OutOfRangeToInfinity(const std::string &str)
{
    // Parses a decimal string using scientific notation into a floating point number.
    // Out-of-range values are converted to infinity. Values that are too small to be
    // represented are converted to zero.

    // The mantissa in decimal scientific notation. The magnitude of the mantissa integer does not
    // matter.
    unsigned int decimalMantissa = 0;
    size_t i                     = 0;
    bool decimalPointSeen        = false;
    bool nonZeroSeenInMantissa   = false;

    // The exponent offset reflects the position of the decimal point.
    int exponentOffset = -1;

    // This is just a counter for how many decimal digits are written to decimalMantissa.
    int mantissaDecimalDigits = 0;

    while (i < str.length())
    {
        const char c = str[i];
        if (c == 'e' || c == 'E')
        {
            break;
        }
        if (c == '.')
        {
            decimalPointSeen = true;
            ++i;
            continue;
        }

        unsigned int digit = static_cast<unsigned int>(c - '0');
        ASSERT(digit < 10u);
        if (digit != 0u)
        {
            nonZeroSeenInMantissa = true;
        }
        if (nonZeroSeenInMantissa)
        {
            // Add bits to the mantissa until space runs out in 32-bit int. This should be
            // enough precision to make the resulting binary mantissa accurate to 1 ULP.
            if (decimalMantissa <= (std::numeric_limits<unsigned int>::max() - 9u) / 10u)
            {
                decimalMantissa = decimalMantissa * 10u + digit;
                ++mantissaDecimalDigits;
            }
            if (!decimalPointSeen)
            {
                ++exponentOffset;
            }
        }
        else if (decimalPointSeen)
        {
            --exponentOffset;
        }
        ++i;
    }
    if (decimalMantissa == 0)
    {
        return 0.0f;
    }
    int exponent = 0;
    if (i < str.length())
    {
        ASSERT(str[i] == 'e' || str[i] == 'E');
        ++i;
        bool exponentOutOfRange = false;
        bool negativeExponent   = false;
        if (str[i] == '-')
        {
            negativeExponent = true;
            ++i;
        }
        else if (str[i] == '+')
        {
            ++i;
        }
        while (i < str.length())
        {
            const char c       = str[i];
            unsigned int digit = static_cast<unsigned int>(c - '0');
            ASSERT(digit < 10u);
            if (exponent <= (std::numeric_limits<int>::max() - 9) / 10)
            {
                exponent = exponent * 10 + digit;
            }
            else
            {
                exponentOutOfRange = true;
            }
            ++i;
        }
        if (negativeExponent)
        {
            exponent = -exponent;
        }
        if (exponentOutOfRange)
        {
            if (negativeExponent)
            {
                return 0.0f;
            }
            else
            {
                return std::numeric_limits<float>::infinity();
            }
        }
    }
    // Do the calculation in 64-bit to avoid overflow.
    long long exponentLong =
        static_cast<long long>(exponent) + static_cast<long long>(exponentOffset);
    if (exponentLong > std::numeric_limits<float>::max_exponent10)
    {
        return std::numeric_limits<float>::infinity();
    }
    else if (exponentLong < std::numeric_limits<float>::min_exponent10)
    {
        return 0.0f;
    }
    // The exponent is in range, so we need to actually evaluate the float.
    exponent     = static_cast<int>(exponentLong);
    double value = decimalMantissa;

    // Calculate the exponent offset to normalize the mantissa.
    int normalizationExponentOffset = 1 - mantissaDecimalDigits;
    // Apply the exponent.
    value *= std::pow(10.0, static_cast<double>(exponent + normalizationExponentOffset));
    if (value > static_cast<double>(std::numeric_limits<float>::max()))
    {
        return std::numeric_limits<float>::infinity();
    }
    if (value < static_cast<double>(std::numeric_limits<float>::min()))
    {
        return 0.0f;
    }
    return static_cast<float>(value);
}

bool strtof_clamp(const std::string &str, float *value)
{
    // Custom float parsing that can handle the following corner cases:
    //   1. The decimal mantissa is very small but the exponent is very large, putting the resulting
    //   number inside the float range.
    //   2. The decimal mantissa is very large but the exponent is very small, putting the resulting
    //   number inside the float range.
    //   3. The value is out-of-range and should be evaluated as infinity.
    //   4. The value is too small and should be evaluated as zero.
    // See ESSL 3.00.6 section 4.1.4 for the relevant specification.
    *value = NumericLexFloat32OutOfRangeToInfinity(str);
    return !gl::isInf(*value);
}

GLenum GLVariableType(const TType &type)
{
    if (type.getBasicType() == EbtFloat)
    {
        if (type.isVector())
        {
            switch (type.getNominalSize())
            {
                case 2:
                    return GL_FLOAT_VEC2;
                case 3:
                    return GL_FLOAT_VEC3;
                case 4:
                    return GL_FLOAT_VEC4;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return GL_NONE;
#endif
            }
        }
        else if (type.isMatrix())
        {
            switch (type.getCols())
            {
                case 2:
                    switch (type.getRows())
                    {
                        case 2:
                            return GL_FLOAT_MAT2;
                        case 3:
                            return GL_FLOAT_MAT2x3;
                        case 4:
                            return GL_FLOAT_MAT2x4;
                        default:
                            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                            return GL_NONE;
#endif
                    }

                case 3:
                    switch (type.getRows())
                    {
                        case 2:
                            return GL_FLOAT_MAT3x2;
                        case 3:
                            return GL_FLOAT_MAT3;
                        case 4:
                            return GL_FLOAT_MAT3x4;
                        default:
                            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                            return GL_NONE;
#endif
                    }

                case 4:
                    switch (type.getRows())
                    {
                        case 2:
                            return GL_FLOAT_MAT4x2;
                        case 3:
                            return GL_FLOAT_MAT4x3;
                        case 4:
                            return GL_FLOAT_MAT4;
                        default:
                            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                            return GL_NONE;
#endif
                    }

                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return GL_NONE;
#endif
            }
        }
        else
        {
            return GL_FLOAT;
        }
    }
    else if (type.getBasicType() == EbtInt)
    {
        if (type.isVector())
        {
            switch (type.getNominalSize())
            {
                case 2:
                    return GL_INT_VEC2;
                case 3:
                    return GL_INT_VEC3;
                case 4:
                    return GL_INT_VEC4;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return GL_NONE;
#endif
            }
        }
        else
        {
            ASSERT(!type.isMatrix());
            return GL_INT;
        }
    }
    else if (type.getBasicType() == EbtUInt)
    {
        if (type.isVector())
        {
            switch (type.getNominalSize())
            {
                case 2:
                    return GL_UNSIGNED_INT_VEC2;
                case 3:
                    return GL_UNSIGNED_INT_VEC3;
                case 4:
                    return GL_UNSIGNED_INT_VEC4;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return GL_NONE;
#endif
            }
        }
        else
        {
            ASSERT(!type.isMatrix());
            return GL_UNSIGNED_INT;
        }
    }
    else if (type.getBasicType() == EbtBool)
    {
        if (type.isVector())
        {
            switch (type.getNominalSize())
            {
                case 2:
                    return GL_BOOL_VEC2;
                case 3:
                    return GL_BOOL_VEC3;
                case 4:
                    return GL_BOOL_VEC4;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return GL_NONE;
#endif
            }
        }
        else
        {
            ASSERT(!type.isMatrix());
            return GL_BOOL;
        }
    }

    switch (type.getBasicType())
    {
        case EbtSampler2D:
            return GL_SAMPLER_2D;
        case EbtSampler3D:
            return GL_SAMPLER_3D;
        case EbtSamplerCube:
            return GL_SAMPLER_CUBE;
        case EbtSamplerExternalOES:
            return GL_SAMPLER_EXTERNAL_OES;
        case EbtSamplerExternal2DY2YEXT:
            return GL_SAMPLER_EXTERNAL_2D_Y2Y_EXT;
        case EbtSampler2DRect:
            return GL_SAMPLER_2D_RECT_ANGLE;
        case EbtSampler2DArray:
            return GL_SAMPLER_2D_ARRAY;
        case EbtSampler2DMS:
            return GL_SAMPLER_2D_MULTISAMPLE;
        case EbtSampler2DMSArray:
            return GL_SAMPLER_2D_MULTISAMPLE_ARRAY;
        case EbtISampler2D:
            return GL_INT_SAMPLER_2D;
        case EbtISampler3D:
            return GL_INT_SAMPLER_3D;
        case EbtISamplerCube:
            return GL_INT_SAMPLER_CUBE;
        case EbtISampler2DArray:
            return GL_INT_SAMPLER_2D_ARRAY;
        case EbtISampler2DMS:
            return GL_INT_SAMPLER_2D_MULTISAMPLE;
        case EbtISampler2DMSArray:
            return GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY;
        case EbtUSampler2D:
            return GL_UNSIGNED_INT_SAMPLER_2D;
        case EbtUSampler3D:
            return GL_UNSIGNED_INT_SAMPLER_3D;
        case EbtUSamplerCube:
            return GL_UNSIGNED_INT_SAMPLER_CUBE;
        case EbtUSampler2DArray:
            return GL_UNSIGNED_INT_SAMPLER_2D_ARRAY;
        case EbtUSampler2DMS:
            return GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE;
        case EbtUSampler2DMSArray:
            return GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY;
        case EbtSampler2DShadow:
            return GL_SAMPLER_2D_SHADOW;
        case EbtSamplerCubeShadow:
            return GL_SAMPLER_CUBE_SHADOW;
        case EbtSampler2DArrayShadow:
            return GL_SAMPLER_2D_ARRAY_SHADOW;
        case EbtImage2D:
            return GL_IMAGE_2D;
        case EbtIImage2D:
            return GL_INT_IMAGE_2D;
        case EbtUImage2D:
            return GL_UNSIGNED_INT_IMAGE_2D;
        case EbtImage2DArray:
            return GL_IMAGE_2D_ARRAY;
        case EbtIImage2DArray:
            return GL_INT_IMAGE_2D_ARRAY;
        case EbtUImage2DArray:
            return GL_UNSIGNED_INT_IMAGE_2D_ARRAY;
        case EbtImage3D:
            return GL_IMAGE_3D;
        case EbtIImage3D:
            return GL_INT_IMAGE_3D;
        case EbtUImage3D:
            return GL_UNSIGNED_INT_IMAGE_3D;
        case EbtImageCube:
            return GL_IMAGE_CUBE;
        case EbtIImageCube:
            return GL_INT_IMAGE_CUBE;
        case EbtUImageCube:
            return GL_UNSIGNED_INT_IMAGE_CUBE;
        case EbtAtomicCounter:
            return GL_UNSIGNED_INT_ATOMIC_COUNTER;
        default:
            UNREACHABLE();
    }

    return GL_NONE;
}

GLenum GLVariablePrecision(const TType &type)
{
    if (type.getBasicType() == EbtFloat)
    {
        switch (type.getPrecision())
        {
            case EbpHigh:
                return GL_HIGH_FLOAT;
            case EbpMedium:
                return GL_MEDIUM_FLOAT;
            case EbpLow:
                return GL_LOW_FLOAT;
            case EbpUndefined:
                // Desktop specs do not use precision
                return GL_NONE;
            default:
                UNREACHABLE();
        }
    }
    else if (type.getBasicType() == EbtInt || type.getBasicType() == EbtUInt)
    {
        switch (type.getPrecision())
        {
            case EbpHigh:
                return GL_HIGH_INT;
            case EbpMedium:
                return GL_MEDIUM_INT;
            case EbpLow:
                return GL_LOW_INT;
            case EbpUndefined:
                // Desktop specs do not use precision
                return GL_NONE;
            default:
                UNREACHABLE();
        }
    }

    // Other types (boolean, sampler) don't have a precision
    return GL_NONE;
}

ImmutableString ArrayString(const TType &type)
{
    if (!type.isArray())
        return ImmutableString("");

    const TVector<unsigned int> &arraySizes         = *type.getArraySizes();
    constexpr const size_t kMaxDecimalDigitsPerSize = 10u;
    ImmutableStringBuilder arrayString(arraySizes.size() * (kMaxDecimalDigitsPerSize + 2u));
    for (auto arraySizeIter = arraySizes.rbegin(); arraySizeIter != arraySizes.rend();
         ++arraySizeIter)
    {
        arrayString << "[";
        if (*arraySizeIter > 0)
        {
            arrayString.appendDecimal(*arraySizeIter);
        }
        arrayString << "]";
    }
    return arrayString;
}

ImmutableString GetTypeName(const TType &type, ShHashFunction64 hashFunction, NameMap *nameMap)
{
    if (type.getBasicType() == EbtStruct)
        return HashName(type.getStruct(), hashFunction, nameMap);
    else
        return ImmutableString(type.getBuiltInTypeNameString());
}

bool IsVaryingOut(TQualifier qualifier)
{
    switch (qualifier)
    {
        case EvqVaryingOut:
        case EvqSmoothOut:
        case EvqFlatOut:
        case EvqCentroidOut:
        case EvqVertexOut:
        case EvqGeometryOut:
            return true;

        default:
            break;
    }

    return false;
}

bool IsVaryingIn(TQualifier qualifier)
{
    switch (qualifier)
    {
        case EvqVaryingIn:
        case EvqSmoothIn:
        case EvqFlatIn:
        case EvqCentroidIn:
        case EvqFragmentIn:
        case EvqGeometryIn:
            return true;

        default:
            break;
    }

    return false;
}

bool IsVarying(TQualifier qualifier)
{
    return IsVaryingIn(qualifier) || IsVaryingOut(qualifier);
}

bool IsGeometryShaderInput(GLenum shaderType, TQualifier qualifier)
{
    return (qualifier == EvqGeometryIn) ||
           ((shaderType == GL_GEOMETRY_SHADER_EXT) && IsInterpolationIn(qualifier));
}

InterpolationType GetInterpolationType(TQualifier qualifier)
{
    switch (qualifier)
    {
        case EvqFlatIn:
        case EvqFlatOut:
            return INTERPOLATION_FLAT;

        case EvqSmoothIn:
        case EvqSmoothOut:
        case EvqVertexOut:
        case EvqFragmentIn:
        case EvqVaryingIn:
        case EvqVaryingOut:
        case EvqGeometryIn:
        case EvqGeometryOut:
            return INTERPOLATION_SMOOTH;

        case EvqCentroidIn:
        case EvqCentroidOut:
            return INTERPOLATION_CENTROID;

        default:
            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
            return INTERPOLATION_SMOOTH;
#endif
    }
}

TType GetShaderVariableBasicType(const sh::ShaderVariable &var)
{
    switch (var.type)
    {
        case GL_BOOL:
            return TType(EbtBool);
        case GL_BOOL_VEC2:
            return TType(EbtBool, 2);
        case GL_BOOL_VEC3:
            return TType(EbtBool, 3);
        case GL_BOOL_VEC4:
            return TType(EbtBool, 4);
        case GL_FLOAT:
            return TType(EbtFloat);
        case GL_FLOAT_VEC2:
            return TType(EbtFloat, 2);
        case GL_FLOAT_VEC3:
            return TType(EbtFloat, 3);
        case GL_FLOAT_VEC4:
            return TType(EbtFloat, 4);
        case GL_FLOAT_MAT2:
            return TType(EbtFloat, 2, 2);
        case GL_FLOAT_MAT3:
            return TType(EbtFloat, 3, 3);
        case GL_FLOAT_MAT4:
            return TType(EbtFloat, 4, 4);
        case GL_FLOAT_MAT2x3:
            return TType(EbtFloat, 2, 3);
        case GL_FLOAT_MAT2x4:
            return TType(EbtFloat, 2, 4);
        case GL_FLOAT_MAT3x2:
            return TType(EbtFloat, 3, 2);
        case GL_FLOAT_MAT3x4:
            return TType(EbtFloat, 3, 4);
        case GL_FLOAT_MAT4x2:
            return TType(EbtFloat, 4, 2);
        case GL_FLOAT_MAT4x3:
            return TType(EbtFloat, 4, 3);
        case GL_INT:
            return TType(EbtInt);
        case GL_INT_VEC2:
            return TType(EbtInt, 2);
        case GL_INT_VEC3:
            return TType(EbtInt, 3);
        case GL_INT_VEC4:
            return TType(EbtInt, 4);
        case GL_UNSIGNED_INT:
            return TType(EbtUInt);
        case GL_UNSIGNED_INT_VEC2:
            return TType(EbtUInt, 2);
        case GL_UNSIGNED_INT_VEC3:
            return TType(EbtUInt, 3);
        case GL_UNSIGNED_INT_VEC4:
            return TType(EbtUInt, 4);
        default:
            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
            return TType();
#endif
    }
}

void DeclareGlobalVariable(TIntermBlock *root, const TVariable *variable)
{
    TIntermDeclaration *declaration = new TIntermDeclaration();
    declaration->appendDeclarator(new TIntermSymbol(variable));

    TIntermSequence *globalSequence = root->getSequence();
    globalSequence->insert(globalSequence->begin(), declaration);
}

// GLSL ES 1.0.17 4.6.1 The Invariant Qualifier
bool CanBeInvariantESSL1(TQualifier qualifier)
{
    return IsVaryingIn(qualifier) || IsVaryingOut(qualifier) ||
           IsBuiltinOutputVariable(qualifier) ||
           (IsBuiltinFragmentInputVariable(qualifier) && qualifier != EvqFrontFacing);
}

// GLSL ES 3.00 Revision 6, 4.6.1 The Invariant Qualifier
// GLSL ES 3.10 Revision 4, 4.8.1 The Invariant Qualifier
bool CanBeInvariantESSL3OrGreater(TQualifier qualifier)
{
    return IsVaryingOut(qualifier) || qualifier == EvqFragmentOut ||
           IsBuiltinOutputVariable(qualifier);
}

bool IsBuiltinOutputVariable(TQualifier qualifier)
{
    switch (qualifier)
    {
        case EvqPosition:
        case EvqPointSize:
        case EvqFragDepth:
        case EvqFragDepthEXT:
        case EvqFragColor:
        case EvqSecondaryFragColorEXT:
        case EvqFragData:
        case EvqSecondaryFragDataEXT:
        case EvqClipDistance:
            return true;
        default:
            break;
    }
    return false;
}

bool IsBuiltinFragmentInputVariable(TQualifier qualifier)
{
    switch (qualifier)
    {
        case EvqFragCoord:
        case EvqPointCoord:
        case EvqFrontFacing:
            return true;
        default:
            break;
    }
    return false;
}

bool IsShaderOutput(TQualifier qualifier)
{
    return IsVaryingOut(qualifier) || IsBuiltinOutputVariable(qualifier);
}

bool IsOutputESSL(ShShaderOutput output)
{
    return output == SH_ESSL_OUTPUT;
}

bool IsOutputGLSL(ShShaderOutput output)
{
    switch (output)
    {
        case SH_GLSL_130_OUTPUT:
        case SH_GLSL_140_OUTPUT:
        case SH_GLSL_150_CORE_OUTPUT:
        case SH_GLSL_330_CORE_OUTPUT:
        case SH_GLSL_400_CORE_OUTPUT:
        case SH_GLSL_410_CORE_OUTPUT:
        case SH_GLSL_420_CORE_OUTPUT:
        case SH_GLSL_430_CORE_OUTPUT:
        case SH_GLSL_440_CORE_OUTPUT:
        case SH_GLSL_450_CORE_OUTPUT:
        case SH_GLSL_COMPATIBILITY_OUTPUT:
            return true;
        default:
            break;
    }
    return false;
}
bool IsOutputHLSL(ShShaderOutput output)
{
    switch (output)
    {
        case SH_HLSL_3_0_OUTPUT:
        case SH_HLSL_4_1_OUTPUT:
        case SH_HLSL_4_0_FL9_3_OUTPUT:
            return true;
        default:
            break;
    }
    return false;
}
bool IsOutputVulkan(ShShaderOutput output)
{
    return output == SH_GLSL_VULKAN_OUTPUT;
}
bool IsOutputMetal(ShShaderOutput output)
{
    return output == SH_GLSL_METAL_OUTPUT;
}

bool IsInShaderStorageBlock(TIntermTyped *node)
{
    TIntermSwizzle *swizzleNode = node->getAsSwizzleNode();
    if (swizzleNode)
    {
        return IsInShaderStorageBlock(swizzleNode->getOperand());
    }

    TIntermBinary *binaryNode = node->getAsBinaryNode();
    if (binaryNode)
    {
        switch (binaryNode->getOp())
        {
            case EOpIndexDirectInterfaceBlock:
            case EOpIndexIndirect:
            case EOpIndexDirect:
            case EOpIndexDirectStruct:
                return IsInShaderStorageBlock(binaryNode->getLeft());
            default:
                return false;
        }
    }

    const TType &type = node->getType();
    return type.getQualifier() == EvqBuffer;
}

GLenum GetImageInternalFormatType(TLayoutImageInternalFormat iifq)
{
    switch (iifq)
    {
        case EiifRGBA32F:
            return GL_RGBA32F;
        case EiifRGBA16F:
            return GL_RGBA16F;
        case EiifR32F:
            return GL_R32F;
        case EiifRGBA32UI:
            return GL_RGBA32UI;
        case EiifRGBA16UI:
            return GL_RGBA16UI;
        case EiifRGBA8UI:
            return GL_RGBA8UI;
        case EiifR32UI:
            return GL_R32UI;
        case EiifRGBA32I:
            return GL_RGBA32I;
        case EiifRGBA16I:
            return GL_RGBA16I;
        case EiifRGBA8I:
            return GL_RGBA8I;
        case EiifR32I:
            return GL_R32I;
        case EiifRGBA8:
            return GL_RGBA8;
        case EiifRGBA8_SNORM:
            return GL_RGBA8_SNORM;
        default:
            return GL_NONE;
    }
}

bool IsSpecWithFunctionBodyNewScope(ShShaderSpec shaderSpec, int shaderVersion)
{
    return (shaderVersion == 100 && !sh::IsWebGLBasedSpec(shaderSpec));
}

ImplicitTypeConversion GetConversion(TBasicType t1, TBasicType t2)
{
    if (t1 == t2)
        return ImplicitTypeConversion::Same;

    switch (t1)
    {
        case EbtInt:
            switch (t2)
            {
                case EbtInt:
                    UNREACHABLE();
                    break;
                case EbtUInt:
                    return ImplicitTypeConversion::Invalid;
                case EbtFloat:
                    return ImplicitTypeConversion::Left;
                default:
                    return ImplicitTypeConversion::Invalid;
            }
            break;
        case EbtUInt:
            switch (t2)
            {
                case EbtInt:
                    return ImplicitTypeConversion::Invalid;
                case EbtUInt:
                    UNREACHABLE();
                    break;
                case EbtFloat:
                    return ImplicitTypeConversion::Left;
                default:
                    return ImplicitTypeConversion::Invalid;
            }
            break;
        case EbtFloat:
            switch (t2)
            {
                case EbtInt:
                case EbtUInt:
                    return ImplicitTypeConversion::Right;
                case EbtFloat:
                    UNREACHABLE();
                    break;
                default:
                    return ImplicitTypeConversion::Invalid;
            }
            break;
        default:
            return ImplicitTypeConversion::Invalid;
    }
    return ImplicitTypeConversion::Invalid;
}

bool IsValidImplicitConversion(sh::ImplicitTypeConversion conversion, TOperator op)
{
    switch (conversion)
    {
        case sh::ImplicitTypeConversion::Same:
            return true;
        case sh::ImplicitTypeConversion::Left:
            switch (op)
            {
                case EOpEqual:
                case EOpNotEqual:
                case EOpLessThan:
                case EOpGreaterThan:
                case EOpLessThanEqual:
                case EOpGreaterThanEqual:
                case EOpAdd:
                case EOpSub:
                case EOpMul:
                case EOpDiv:
                    return true;
                default:
                    break;
            }
            break;
        case sh::ImplicitTypeConversion::Right:
            switch (op)
            {
                case EOpAssign:
                case EOpInitialize:
                case EOpEqual:
                case EOpNotEqual:
                case EOpLessThan:
                case EOpGreaterThan:
                case EOpLessThanEqual:
                case EOpGreaterThanEqual:
                case EOpAdd:
                case EOpSub:
                case EOpMul:
                case EOpDiv:
                case EOpAddAssign:
                case EOpSubAssign:
                case EOpMulAssign:
                case EOpDivAssign:
                    return true;
                default:
                    break;
            }
            break;
        case sh::ImplicitTypeConversion::Invalid:
            break;
    }
    return false;
}

size_t FindFieldIndex(const TFieldList &fieldList, const char *fieldName)
{
    for (size_t fieldIndex = 0; fieldIndex < fieldList.size(); ++fieldIndex)
    {
        if (strcmp(fieldList[fieldIndex]->name().data(), fieldName) == 0)
        {
            return fieldIndex;
        }
    }
    UNREACHABLE();
    return 0;
}

}  // namespace sh

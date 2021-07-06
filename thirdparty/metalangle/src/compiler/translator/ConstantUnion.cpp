//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ConstantUnion: Constant folding helper class.

#include "compiler/translator/ConstantUnion.h"

#include "common/mathutil.h"
#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

float CheckedSum(float lhs, float rhs, TDiagnostics *diag, const TSourceLoc &line)
{
    float result = lhs + rhs;
    if (gl::isNaN(result) && !gl::isNaN(lhs) && !gl::isNaN(rhs))
    {
        diag->warning(line, "Constant folded undefined addition generated NaN", "+");
    }
    else if (gl::isInf(result) && !gl::isInf(lhs) && !gl::isInf(rhs))
    {
        diag->warning(line, "Constant folded addition overflowed to infinity", "+");
    }
    return result;
}

float CheckedDiff(float lhs, float rhs, TDiagnostics *diag, const TSourceLoc &line)
{
    float result = lhs - rhs;
    if (gl::isNaN(result) && !gl::isNaN(lhs) && !gl::isNaN(rhs))
    {
        diag->warning(line, "Constant folded undefined subtraction generated NaN", "-");
    }
    else if (gl::isInf(result) && !gl::isInf(lhs) && !gl::isInf(rhs))
    {
        diag->warning(line, "Constant folded subtraction overflowed to infinity", "-");
    }
    return result;
}

float CheckedMul(float lhs, float rhs, TDiagnostics *diag, const TSourceLoc &line)
{
    float result = lhs * rhs;
    if (gl::isNaN(result) && !gl::isNaN(lhs) && !gl::isNaN(rhs))
    {
        diag->warning(line, "Constant folded undefined multiplication generated NaN", "*");
    }
    else if (gl::isInf(result) && !gl::isInf(lhs) && !gl::isInf(rhs))
    {
        diag->warning(line, "Constant folded multiplication overflowed to infinity", "*");
    }
    return result;
}

bool IsValidShiftOffset(const TConstantUnion &rhs)
{
    return (rhs.getType() == EbtInt && (rhs.getIConst() >= 0 && rhs.getIConst() <= 31)) ||
           (rhs.getType() == EbtUInt && rhs.getUConst() <= 31u);
}

}  // anonymous namespace

TConstantUnion::TConstantUnion()
{
    iConst = 0;
    type   = EbtVoid;
}

int TConstantUnion::getIConst() const
{
    ASSERT(type == EbtInt);
    return iConst;
}

unsigned int TConstantUnion::getUConst() const
{
    ASSERT(type == EbtUInt);
    return uConst;
}

float TConstantUnion::getFConst() const
{
    switch (type)
    {
        case EbtInt:
            return static_cast<float>(iConst);
        case EbtUInt:
            return static_cast<float>(uConst);
        default:
            ASSERT(type == EbtFloat);
            return fConst;
    }
}

bool TConstantUnion::getBConst() const
{
    ASSERT(type == EbtBool);
    return bConst;
}

bool TConstantUnion::isZero() const
{
    switch (type)
    {
        case EbtInt:
            return getIConst() == 0;
        case EbtUInt:
            return getUConst() == 0;
        case EbtFloat:
            return getFConst() == 0.0f;
        case EbtBool:
            return getBConst() == false;
        default:
            return false;
    }
}

TYuvCscStandardEXT TConstantUnion::getYuvCscStandardEXTConst() const
{
    ASSERT(type == EbtYuvCscStandardEXT);
    return yuvCscStandardEXTConst;
}

bool TConstantUnion::cast(TBasicType newType, const TConstantUnion &constant)
{
    switch (newType)
    {
        case EbtFloat:
            switch (constant.type)
            {
                case EbtInt:
                    setFConst(static_cast<float>(constant.getIConst()));
                    break;
                case EbtUInt:
                    setFConst(static_cast<float>(constant.getUConst()));
                    break;
                case EbtBool:
                    setFConst(static_cast<float>(constant.getBConst()));
                    break;
                case EbtFloat:
                    setFConst(static_cast<float>(constant.getFConst()));
                    break;
                default:
                    return false;
            }
            break;
        case EbtInt:
            switch (constant.type)
            {
                case EbtInt:
                    setIConst(static_cast<int>(constant.getIConst()));
                    break;
                case EbtUInt:
                    setIConst(static_cast<int>(constant.getUConst()));
                    break;
                case EbtBool:
                    setIConst(static_cast<int>(constant.getBConst()));
                    break;
                case EbtFloat:
                    setIConst(static_cast<int>(constant.getFConst()));
                    break;
                default:
                    return false;
            }
            break;
        case EbtUInt:
            switch (constant.type)
            {
                case EbtInt:
                    setUConst(static_cast<unsigned int>(constant.getIConst()));
                    break;
                case EbtUInt:
                    setUConst(static_cast<unsigned int>(constant.getUConst()));
                    break;
                case EbtBool:
                    setUConst(static_cast<unsigned int>(constant.getBConst()));
                    break;
                case EbtFloat:
                    if (constant.getFConst() < 0.0f)
                    {
                        // Avoid undefined behavior in C++ by first casting to signed int.
                        setUConst(
                            static_cast<unsigned int>(static_cast<int>(constant.getFConst())));
                    }
                    else
                    {
                        setUConst(static_cast<unsigned int>(constant.getFConst()));
                    }
                    break;
                default:
                    return false;
            }
            break;
        case EbtBool:
            switch (constant.type)
            {
                case EbtInt:
                    setBConst(constant.getIConst() != 0);
                    break;
                case EbtUInt:
                    setBConst(constant.getUConst() != 0);
                    break;
                case EbtBool:
                    setBConst(constant.getBConst());
                    break;
                case EbtFloat:
                    setBConst(constant.getFConst() != 0.0f);
                    break;
                default:
                    return false;
            }
            break;
        case EbtStruct:  // Struct fields don't get cast
            switch (constant.type)
            {
                case EbtInt:
                    setIConst(constant.getIConst());
                    break;
                case EbtUInt:
                    setUConst(constant.getUConst());
                    break;
                case EbtBool:
                    setBConst(constant.getBConst());
                    break;
                case EbtFloat:
                    setFConst(constant.getFConst());
                    break;
                default:
                    return false;
            }
            break;
        default:
            return false;
    }

    return true;
}

bool TConstantUnion::operator==(const int i) const
{
    switch (type)
    {
        case EbtFloat:
            return static_cast<float>(i) == fConst;
        default:
            return i == iConst;
    }
}

bool TConstantUnion::operator==(const unsigned int u) const
{
    switch (type)
    {
        case EbtFloat:
            return static_cast<float>(u) == fConst;
        default:
            return u == uConst;
    }
}

bool TConstantUnion::operator==(const float f) const
{
    switch (type)
    {
        case EbtInt:
            return f == static_cast<float>(iConst);
        case EbtUInt:
            return f == static_cast<float>(uConst);
        default:
            return f == fConst;
    }
}

bool TConstantUnion::operator==(const bool b) const
{
    return b == bConst;
}

bool TConstantUnion::operator==(const TYuvCscStandardEXT s) const
{
    return s == yuvCscStandardEXTConst;
}

bool TConstantUnion::operator==(const TConstantUnion &constant) const
{
    ImplicitTypeConversion conversion = GetConversion(constant.type, type);
    if (conversion == ImplicitTypeConversion::Same)
    {
        switch (type)
        {
            case EbtInt:
                return constant.iConst == iConst;
            case EbtUInt:
                return constant.uConst == uConst;
            case EbtFloat:
                return constant.fConst == fConst;
            case EbtBool:
                return constant.bConst == bConst;
            case EbtYuvCscStandardEXT:
                return constant.yuvCscStandardEXTConst == yuvCscStandardEXTConst;
            default:
                return false;
        }
    }
    else if (conversion == ImplicitTypeConversion::Invalid)
    {
        return false;
    }
    else
    {
        return constant.getFConst() == getFConst();
    }
}

bool TConstantUnion::operator!=(const int i) const
{
    return !operator==(i);
}

bool TConstantUnion::operator!=(const unsigned int u) const
{
    return !operator==(u);
}

bool TConstantUnion::operator!=(const float f) const
{
    return !operator==(f);
}

bool TConstantUnion::operator!=(const bool b) const
{
    return !operator==(b);
}

bool TConstantUnion::operator!=(const TYuvCscStandardEXT s) const
{
    return !operator==(s);
}

bool TConstantUnion::operator!=(const TConstantUnion &constant) const
{
    return !operator==(constant);
}

bool TConstantUnion::operator>(const TConstantUnion &constant) const
{

    ImplicitTypeConversion conversion = GetConversion(constant.type, type);
    if (conversion == ImplicitTypeConversion::Same)
    {
        switch (type)
        {
            case EbtInt:
                return iConst > constant.iConst;
            case EbtUInt:
                return uConst > constant.uConst;
            case EbtFloat:
                return fConst > constant.fConst;
            default:
                return false;  // Invalid operation, handled at semantic analysis
        }
    }
    else
    {
        ASSERT(conversion != ImplicitTypeConversion::Invalid);
        return getFConst() > constant.getFConst();
    }
}

bool TConstantUnion::operator<(const TConstantUnion &constant) const
{
    ImplicitTypeConversion conversion = GetConversion(constant.type, type);
    if (conversion == ImplicitTypeConversion::Same)
    {
        switch (type)
        {
            case EbtInt:
                return iConst < constant.iConst;
            case EbtUInt:
                return uConst < constant.uConst;
            case EbtFloat:
                return fConst < constant.fConst;
            default:
                return false;  // Invalid operation, handled at semantic analysis
        }
    }
    else
    {
        ASSERT(conversion != ImplicitTypeConversion::Invalid);
        return getFConst() < constant.getFConst();
    }
}

// static
TConstantUnion TConstantUnion::add(const TConstantUnion &lhs,
                                   const TConstantUnion &rhs,
                                   TDiagnostics *diag,
                                   const TSourceLoc &line)
{
    TConstantUnion returnValue;

    ImplicitTypeConversion conversion = GetConversion(lhs.type, rhs.type);
    if (conversion == ImplicitTypeConversion::Same)
    {
        switch (lhs.type)
        {
            case EbtInt:
                returnValue.setIConst(gl::WrappingSum<int>(lhs.iConst, rhs.iConst));
                break;
            case EbtUInt:
                returnValue.setUConst(gl::WrappingSum<unsigned int>(lhs.uConst, rhs.uConst));
                break;
            case EbtFloat:
                returnValue.setFConst(CheckedSum(lhs.fConst, rhs.fConst, diag, line));
                break;
            default:
                UNREACHABLE();
        }
    }
    else
    {
        ASSERT(conversion != ImplicitTypeConversion::Invalid);
        returnValue.setFConst(CheckedSum(lhs.getFConst(), rhs.getFConst(), diag, line));
    }

    return returnValue;
}

// static
TConstantUnion TConstantUnion::sub(const TConstantUnion &lhs,
                                   const TConstantUnion &rhs,
                                   TDiagnostics *diag,
                                   const TSourceLoc &line)
{
    TConstantUnion returnValue;

    ImplicitTypeConversion conversion = GetConversion(lhs.type, rhs.type);
    if (conversion == ImplicitTypeConversion::Same)
    {
        switch (lhs.type)
        {
            case EbtInt:
                returnValue.setIConst(gl::WrappingDiff<int>(lhs.iConst, rhs.iConst));
                break;
            case EbtUInt:
                returnValue.setUConst(gl::WrappingDiff<unsigned int>(lhs.uConst, rhs.uConst));
                break;
            case EbtFloat:
                returnValue.setFConst(CheckedDiff(lhs.fConst, rhs.fConst, diag, line));
                break;
            default:
                UNREACHABLE();
        }
    }
    else
    {
        ASSERT(conversion != ImplicitTypeConversion::Invalid);
        returnValue.setFConst(CheckedDiff(lhs.getFConst(), rhs.getFConst(), diag, line));
    }

    return returnValue;
}

// static
TConstantUnion TConstantUnion::mul(const TConstantUnion &lhs,
                                   const TConstantUnion &rhs,
                                   TDiagnostics *diag,
                                   const TSourceLoc &line)
{
    TConstantUnion returnValue;

    ImplicitTypeConversion conversion = GetConversion(lhs.type, rhs.type);
    if (conversion == ImplicitTypeConversion::Same)
    {
        switch (lhs.type)
        {
            case EbtInt:
                returnValue.setIConst(gl::WrappingMul(lhs.iConst, rhs.iConst));
                break;
            case EbtUInt:
                // Unsigned integer math in C++ is defined to be done in modulo 2^n, so we rely
                // on that to implement wrapping multiplication.
                returnValue.setUConst(lhs.uConst * rhs.uConst);
                break;
            case EbtFloat:
                returnValue.setFConst(CheckedMul(lhs.fConst, rhs.fConst, diag, line));
                break;
            default:
                UNREACHABLE();
        }
    }
    else
    {
        ASSERT(conversion != ImplicitTypeConversion::Invalid);
        returnValue.setFConst(CheckedMul(lhs.getFConst(), rhs.getFConst(), diag, line));
    }

    return returnValue;
}

TConstantUnion TConstantUnion::operator%(const TConstantUnion &constant) const
{
    TConstantUnion returnValue;
    ASSERT(type == constant.type);
    switch (type)
    {
        case EbtInt:
            returnValue.setIConst(iConst % constant.iConst);
            break;
        case EbtUInt:
            returnValue.setUConst(uConst % constant.uConst);
            break;
        default:
            UNREACHABLE();
    }

    return returnValue;
}

// static
TConstantUnion TConstantUnion::rshift(const TConstantUnion &lhs,
                                      const TConstantUnion &rhs,
                                      TDiagnostics *diag,
                                      const TSourceLoc &line)
{
    TConstantUnion returnValue;
    ASSERT(lhs.type == EbtInt || lhs.type == EbtUInt);
    ASSERT(rhs.type == EbtInt || rhs.type == EbtUInt);
    if (!IsValidShiftOffset(rhs))
    {
        diag->warning(line, "Undefined shift (operand out of range)", ">>");
        switch (lhs.type)
        {
            case EbtInt:
                returnValue.setIConst(0);
                break;
            case EbtUInt:
                returnValue.setUConst(0u);
                break;
            default:
                UNREACHABLE();
        }
        return returnValue;
    }

    switch (lhs.type)
    {
        case EbtInt:
        {
            unsigned int shiftOffset = 0;
            switch (rhs.type)
            {
                case EbtInt:
                    shiftOffset = static_cast<unsigned int>(rhs.iConst);
                    break;
                case EbtUInt:
                    shiftOffset = rhs.uConst;
                    break;
                default:
                    UNREACHABLE();
            }
            if (shiftOffset > 0)
            {
                // ESSL 3.00.6 section 5.9: "If E1 is a signed integer, the right-shift will extend
                // the sign bit." In C++ shifting negative integers is undefined, so we implement
                // extending the sign bit manually.
                int lhsSafe = lhs.iConst;
                if (lhsSafe == std::numeric_limits<int>::min())
                {
                    // The min integer needs special treatment because only bit it has set is the
                    // sign bit, which we clear later to implement safe right shift of negative
                    // numbers.
                    lhsSafe = -0x40000000;
                    --shiftOffset;
                }
                if (shiftOffset > 0)
                {
                    bool extendSignBit = false;
                    if (lhsSafe < 0)
                    {
                        extendSignBit = true;
                        // Clear the sign bit so that bitshift right is defined in C++.
                        lhsSafe &= 0x7fffffff;
                        ASSERT(lhsSafe > 0);
                    }
                    returnValue.setIConst(lhsSafe >> shiftOffset);

                    // Manually fill in the extended sign bit if necessary.
                    if (extendSignBit)
                    {
                        int extendedSignBit = static_cast<int>(0xffffffffu << (31 - shiftOffset));
                        returnValue.setIConst(returnValue.getIConst() | extendedSignBit);
                    }
                }
                else
                {
                    returnValue.setIConst(lhsSafe);
                }
            }
            else
            {
                returnValue.setIConst(lhs.iConst);
            }
            break;
        }
        case EbtUInt:
            switch (rhs.type)
            {
                case EbtInt:
                    returnValue.setUConst(lhs.uConst >> rhs.iConst);
                    break;
                case EbtUInt:
                    returnValue.setUConst(lhs.uConst >> rhs.uConst);
                    break;
                default:
                    UNREACHABLE();
            }
            break;

        default:
            UNREACHABLE();
    }
    return returnValue;
}

// static
TConstantUnion TConstantUnion::lshift(const TConstantUnion &lhs,
                                      const TConstantUnion &rhs,
                                      TDiagnostics *diag,
                                      const TSourceLoc &line)
{
    TConstantUnion returnValue;
    ASSERT(lhs.type == EbtInt || lhs.type == EbtUInt);
    ASSERT(rhs.type == EbtInt || rhs.type == EbtUInt);
    if (!IsValidShiftOffset(rhs))
    {
        diag->warning(line, "Undefined shift (operand out of range)", "<<");
        switch (lhs.type)
        {
            case EbtInt:
                returnValue.setIConst(0);
                break;
            case EbtUInt:
                returnValue.setUConst(0u);
                break;
            default:
                UNREACHABLE();
        }
        return returnValue;
    }

    switch (lhs.type)
    {
        case EbtInt:
            switch (rhs.type)
            {
                // Cast to unsigned integer before shifting, since ESSL 3.00.6 section 5.9 says that
                // lhs is "interpreted as a bit pattern". This also avoids the possibility of signed
                // integer overflow or undefined shift of a negative integer.
                case EbtInt:
                    returnValue.setIConst(
                        static_cast<int>(static_cast<uint32_t>(lhs.iConst) << rhs.iConst));
                    break;
                case EbtUInt:
                    returnValue.setIConst(
                        static_cast<int>(static_cast<uint32_t>(lhs.iConst) << rhs.uConst));
                    break;
                default:
                    UNREACHABLE();
            }
            break;

        case EbtUInt:
            switch (rhs.type)
            {
                case EbtInt:
                    returnValue.setUConst(lhs.uConst << rhs.iConst);
                    break;
                case EbtUInt:
                    returnValue.setUConst(lhs.uConst << rhs.uConst);
                    break;
                default:
                    UNREACHABLE();
            }
            break;

        default:
            UNREACHABLE();
    }
    return returnValue;
}

TConstantUnion TConstantUnion::operator&(const TConstantUnion &constant) const
{
    TConstantUnion returnValue;
    ASSERT(constant.type == EbtInt || constant.type == EbtUInt);
    switch (type)
    {
        case EbtInt:
            returnValue.setIConst(iConst & constant.iConst);
            break;
        case EbtUInt:
            returnValue.setUConst(uConst & constant.uConst);
            break;
        default:
            UNREACHABLE();
    }

    return returnValue;
}

TConstantUnion TConstantUnion::operator|(const TConstantUnion &constant) const
{
    TConstantUnion returnValue;
    ASSERT(type == constant.type);
    switch (type)
    {
        case EbtInt:
            returnValue.setIConst(iConst | constant.iConst);
            break;
        case EbtUInt:
            returnValue.setUConst(uConst | constant.uConst);
            break;
        default:
            UNREACHABLE();
    }

    return returnValue;
}

TConstantUnion TConstantUnion::operator^(const TConstantUnion &constant) const
{
    TConstantUnion returnValue;
    ASSERT(type == constant.type);
    switch (type)
    {
        case EbtInt:
            returnValue.setIConst(iConst ^ constant.iConst);
            break;
        case EbtUInt:
            returnValue.setUConst(uConst ^ constant.uConst);
            break;
        default:
            UNREACHABLE();
    }

    return returnValue;
}

TConstantUnion TConstantUnion::operator&&(const TConstantUnion &constant) const
{
    TConstantUnion returnValue;
    ASSERT(type == constant.type);
    switch (type)
    {
        case EbtBool:
            returnValue.setBConst(bConst && constant.bConst);
            break;
        default:
            UNREACHABLE();
    }

    return returnValue;
}

TConstantUnion TConstantUnion::operator||(const TConstantUnion &constant) const
{
    TConstantUnion returnValue;
    ASSERT(type == constant.type);
    switch (type)
    {
        case EbtBool:
            returnValue.setBConst(bConst || constant.bConst);
            break;
        default:
            UNREACHABLE();
    }

    return returnValue;
}

}  // namespace sh

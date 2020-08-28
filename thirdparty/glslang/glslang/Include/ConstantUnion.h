//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _CONSTANT_UNION_INCLUDED_
#define _CONSTANT_UNION_INCLUDED_

#include "../Include/Common.h"
#include "../Include/BaseTypes.h"

namespace glslang {

class TConstUnion {
public:
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    TConstUnion() : iConst(0), type(EbtInt) { }

    void setI8Const(signed char i)
    {
       i8Const = i;
       type = EbtInt8;
    }

    void setU8Const(unsigned char u)
    {
       u8Const = u;
       type = EbtUint8;
    }

    void setI16Const(signed short i)
    {
       i16Const = i;
       type = EbtInt16;
    }

    void setU16Const(unsigned short u)
    {
       u16Const = u;
       type = EbtUint16;
    }

    void setIConst(int i)
    {
        iConst = i;
        type = EbtInt;
    }

    void setUConst(unsigned int u)
    {
        uConst = u;
        type = EbtUint;
    }

    void setI64Const(long long i64)
    {
        i64Const = i64;
        type = EbtInt64;
    }

    void setU64Const(unsigned long long u64)
    {
        u64Const = u64;
        type = EbtUint64;
    }

    void setDConst(double d)
    {
        dConst = d;
        type = EbtDouble;
    }

    void setBConst(bool b)
    {
        bConst = b;
        type = EbtBool;
    }

    void setSConst(const TString* s)
    {
        sConst = s;
        type = EbtString;
    }

    signed char        getI8Const() const  { return i8Const; }
    unsigned char      getU8Const() const  { return u8Const; }
    signed short       getI16Const() const { return i16Const; }
    unsigned short     getU16Const() const { return u16Const; }
    int                getIConst() const   { return iConst; }
    unsigned int       getUConst() const   { return uConst; }
    long long          getI64Const() const { return i64Const; }
    unsigned long long getU64Const() const { return u64Const; }
    double             getDConst() const   { return dConst; }
    bool               getBConst() const   { return bConst; }
    const TString*     getSConst() const   { return sConst; }

    bool operator==(const signed char i) const
    {
        if (i == i8Const)
            return true;

        return false;
    }

    bool operator==(const unsigned char u) const
    {
        if (u == u8Const)
            return true;

        return false;
    }

   bool operator==(const signed short i) const
    {
        if (i == i16Const)
            return true;

        return false;
    }

    bool operator==(const unsigned short u) const
    {
        if (u == u16Const)
            return true;

        return false;
    }

    bool operator==(const int i) const
    {
        if (i == iConst)
            return true;

        return false;
    }

    bool operator==(const unsigned int u) const
    {
        if (u == uConst)
            return true;

        return false;
    }

    bool operator==(const long long i64) const
    {
        if (i64 == i64Const)
            return true;

        return false;
    }

    bool operator==(const unsigned long long u64) const
    {
        if (u64 == u64Const)
            return true;

        return false;
    }

    bool operator==(const double d) const
    {
        if (d == dConst)
            return true;

        return false;
    }

    bool operator==(const bool b) const
    {
        if (b == bConst)
            return true;

        return false;
    }

    bool operator==(const TConstUnion& constant) const
    {
        if (constant.type != type)
            return false;

        switch (type) {
        case EbtInt:
            if (constant.iConst == iConst)
                return true;

            break;
        case EbtUint:
            if (constant.uConst == uConst)
                return true;

            break;
        case EbtBool:
            if (constant.bConst == bConst)
                return true;

            break;
        case EbtDouble:
            if (constant.dConst == dConst)
                return true;

            break;

#ifndef GLSLANG_WEB
        case EbtInt16:
            if (constant.i16Const == i16Const)
                return true;

            break;
         case EbtUint16:
            if (constant.u16Const == u16Const)
                return true;

            break;
        case EbtInt8:
            if (constant.i8Const == i8Const)
                return true;

            break;
         case EbtUint8:
            if (constant.u8Const == u8Const)
                return true;

            break;
        case EbtInt64:
            if (constant.i64Const == i64Const)
                return true;

            break;
        case EbtUint64:
            if (constant.u64Const == u64Const)
                return true;

            break;
#endif
        default:
            assert(false && "Default missing");
        }

        return false;
    }

    bool operator!=(const signed char i) const
    {
        return !operator==(i);
    }

    bool operator!=(const unsigned char u) const
    {
        return !operator==(u);
    }

    bool operator!=(const signed short i) const
    {
        return !operator==(i);
    }

    bool operator!=(const unsigned short u) const
    {
        return !operator==(u);
    }

    bool operator!=(const int i) const
    {
        return !operator==(i);
    }

    bool operator!=(const unsigned int u) const
    {
        return !operator==(u);
    }

    bool operator!=(const long long i) const
    {
        return !operator==(i);
    }

    bool operator!=(const unsigned long long u) const
    {
        return !operator==(u);
    }

    bool operator!=(const float f) const
    {
        return !operator==(f);
    }

    bool operator!=(const bool b) const
    {
        return !operator==(b);
    }

    bool operator!=(const TConstUnion& constant) const
    {
        return !operator==(constant);
    }

    bool operator>(const TConstUnion& constant) const
    {
        assert(type == constant.type);
        switch (type) {
        case EbtInt:
            if (iConst > constant.iConst)
                return true;

            return false;
        case EbtUint:
            if (uConst > constant.uConst)
                return true;

            return false;
        case EbtDouble:
            if (dConst > constant.dConst)
                return true;

            return false;
#ifndef GLSLANG_WEB
        case EbtInt8:
            if (i8Const > constant.i8Const)
                return true;

            return false;
        case EbtUint8:
            if (u8Const > constant.u8Const)
                return true;

            return false;
        case EbtInt16:
            if (i16Const > constant.i16Const)
                return true;

            return false;
        case EbtUint16:
            if (u16Const > constant.u16Const)
                return true;

            return false;
        case EbtInt64:
            if (i64Const > constant.i64Const)
                return true;

            return false;
        case EbtUint64:
            if (u64Const > constant.u64Const)
                return true;

            return false;
#endif
        default:
            assert(false && "Default missing");
            return false;
        }
    }

    bool operator<(const TConstUnion& constant) const
    {
        assert(type == constant.type);
        switch (type) {
#ifndef GLSLANG_WEB
        case EbtInt8:
            if (i8Const < constant.i8Const)
                return true;

            return false;
        case EbtUint8:
            if (u8Const < constant.u8Const)
                return true;

            return false;
        case EbtInt16:
            if (i16Const < constant.i16Const)
                return true;

            return false;
        case EbtUint16:
            if (u16Const < constant.u16Const)
                return true;
            return false;
        case EbtInt64:
            if (i64Const < constant.i64Const)
                return true;

            return false;
        case EbtUint64:
            if (u64Const < constant.u64Const)
                return true;

            return false;
#endif
        case EbtDouble:
            if (dConst < constant.dConst)
                return true;

            return false;
        case EbtInt:
            if (iConst < constant.iConst)
                return true;

            return false;
        case EbtUint:
            if (uConst < constant.uConst)
                return true;

            return false;
        default:
            assert(false && "Default missing");
            return false;
        }
    }

    TConstUnion operator+(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtInt:    returnValue.setIConst(iConst + constant.iConst); break;
        case EbtUint:   returnValue.setUConst(uConst + constant.uConst); break;
        case EbtDouble: returnValue.setDConst(dConst + constant.dConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(i8Const + constant.i8Const); break;
        case EbtInt16:  returnValue.setI16Const(i16Const + constant.i16Const); break;
        case EbtInt64:  returnValue.setI64Const(i64Const + constant.i64Const); break;
        case EbtUint8:  returnValue.setU8Const(u8Const + constant.u8Const); break;
        case EbtUint16: returnValue.setU16Const(u16Const + constant.u16Const); break;
        case EbtUint64: returnValue.setU64Const(u64Const + constant.u64Const); break;
#endif
        default: assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator-(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtInt:    returnValue.setIConst(iConst - constant.iConst); break;
        case EbtUint:   returnValue.setUConst(uConst - constant.uConst); break;
        case EbtDouble: returnValue.setDConst(dConst - constant.dConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(i8Const - constant.i8Const); break;
        case EbtInt16:  returnValue.setI16Const(i16Const - constant.i16Const); break;
        case EbtInt64:  returnValue.setI64Const(i64Const - constant.i64Const); break;
        case EbtUint8:  returnValue.setU8Const(u8Const - constant.u8Const); break;
        case EbtUint16: returnValue.setU16Const(u16Const - constant.u16Const); break;
        case EbtUint64: returnValue.setU64Const(u64Const - constant.u64Const); break;
#endif
        default: assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator*(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtInt:    returnValue.setIConst(iConst * constant.iConst); break;
        case EbtUint:   returnValue.setUConst(uConst * constant.uConst); break;
        case EbtDouble: returnValue.setDConst(dConst * constant.dConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(i8Const * constant.i8Const); break;
        case EbtInt16:  returnValue.setI16Const(i16Const * constant.i16Const); break;
        case EbtInt64:  returnValue.setI64Const(i64Const * constant.i64Const); break;
        case EbtUint8:  returnValue.setU8Const(u8Const * constant.u8Const); break;
        case EbtUint16: returnValue.setU16Const(u16Const * constant.u16Const); break;
        case EbtUint64: returnValue.setU64Const(u64Const * constant.u64Const); break;
#endif
        default: assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator%(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtInt:    returnValue.setIConst(iConst % constant.iConst); break;
        case EbtUint:   returnValue.setUConst(uConst % constant.uConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(i8Const % constant.i8Const); break;
        case EbtInt16:  returnValue.setI8Const(i8Const % constant.i16Const); break;
        case EbtInt64:  returnValue.setI64Const(i64Const % constant.i64Const); break;
        case EbtUint8:  returnValue.setU8Const(u8Const % constant.u8Const); break;
        case EbtUint16: returnValue.setU16Const(u16Const % constant.u16Const); break;
        case EbtUint64: returnValue.setU64Const(u64Const % constant.u64Const); break;
#endif
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator>>(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        switch (type) {
#ifndef GLSLANG_WEB
        case EbtInt8:
            switch (constant.type) {
            case EbtInt8:   returnValue.setI8Const(i8Const >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setI8Const(i8Const >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setI8Const(i8Const >> constant.i16Const); break;
            case EbtUint16: returnValue.setI8Const(i8Const >> constant.u16Const); break;
            case EbtInt:    returnValue.setI8Const(i8Const >> constant.iConst);   break;
            case EbtUint:   returnValue.setI8Const(i8Const >> constant.uConst);   break;
            case EbtInt64:  returnValue.setI8Const(i8Const >> constant.i64Const); break;
            case EbtUint64: returnValue.setI8Const(i8Const >> constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint8:
            switch (constant.type) {
            case EbtInt8:   returnValue.setU8Const(u8Const >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setU8Const(u8Const >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setU8Const(u8Const >> constant.i16Const); break;
            case EbtUint16: returnValue.setU8Const(u8Const >> constant.u16Const); break;
            case EbtInt:    returnValue.setU8Const(u8Const >> constant.iConst);   break;
            case EbtUint:   returnValue.setU8Const(u8Const >> constant.uConst);   break;
            case EbtInt64:  returnValue.setU8Const(u8Const >> constant.i64Const); break;
            case EbtUint64: returnValue.setU8Const(u8Const >> constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtInt16:
            switch (constant.type) {
            case EbtInt8:   returnValue.setI16Const(i16Const >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setI16Const(i16Const >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setI16Const(i16Const >> constant.i16Const); break;
            case EbtUint16: returnValue.setI16Const(i16Const >> constant.u16Const); break;
            case EbtInt:    returnValue.setI16Const(i16Const >> constant.iConst);   break;
            case EbtUint:   returnValue.setI16Const(i16Const >> constant.uConst);   break;
            case EbtInt64:  returnValue.setI16Const(i16Const >> constant.i64Const); break;
            case EbtUint64: returnValue.setI16Const(i16Const >> constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint16:
            switch (constant.type) {
            case EbtInt8:   returnValue.setU16Const(u16Const >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setU16Const(u16Const >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setU16Const(u16Const >> constant.i16Const); break;
            case EbtUint16: returnValue.setU16Const(u16Const >> constant.u16Const); break;
            case EbtInt:    returnValue.setU16Const(u16Const >> constant.iConst);   break;
            case EbtUint:   returnValue.setU16Const(u16Const >> constant.uConst);   break;
            case EbtInt64:  returnValue.setU16Const(u16Const >> constant.i64Const); break;
            case EbtUint64: returnValue.setU16Const(u16Const >> constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
#endif
        case EbtInt:
            switch (constant.type) {
            case EbtInt:    returnValue.setIConst(iConst >> constant.iConst);   break;
            case EbtUint:   returnValue.setIConst(iConst >> constant.uConst);   break;
#ifndef GLSLANG_WEB
            case EbtInt8:   returnValue.setIConst(iConst >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setIConst(iConst >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setIConst(iConst >> constant.i16Const); break;
            case EbtUint16: returnValue.setIConst(iConst >> constant.u16Const); break;
            case EbtInt64:  returnValue.setIConst(iConst >> constant.i64Const); break;
            case EbtUint64: returnValue.setIConst(iConst >> constant.u64Const); break;
#endif
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint:
            switch (constant.type) {
            case EbtInt:    returnValue.setUConst(uConst >> constant.iConst);   break;
            case EbtUint:   returnValue.setUConst(uConst >> constant.uConst);   break;
#ifndef GLSLANG_WEB
            case EbtInt8:   returnValue.setUConst(uConst >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setUConst(uConst >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setUConst(uConst >> constant.i16Const); break;
            case EbtUint16: returnValue.setUConst(uConst >> constant.u16Const); break;
            case EbtInt64:  returnValue.setUConst(uConst >> constant.i64Const); break;
            case EbtUint64: returnValue.setUConst(uConst >> constant.u64Const); break;
#endif
            default:       assert(false && "Default missing");
            }
            break;
#ifndef GLSLANG_WEB
         case EbtInt64:
            switch (constant.type) {
            case EbtInt8:   returnValue.setI64Const(i64Const >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setI64Const(i64Const >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setI64Const(i64Const >> constant.i16Const); break;
            case EbtUint16: returnValue.setI64Const(i64Const >> constant.u16Const); break;
            case EbtInt:    returnValue.setI64Const(i64Const >> constant.iConst);   break;
            case EbtUint:   returnValue.setI64Const(i64Const >> constant.uConst);   break;
            case EbtInt64:  returnValue.setI64Const(i64Const >> constant.i64Const); break;
            case EbtUint64: returnValue.setI64Const(i64Const >> constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint64:
            switch (constant.type) {
            case EbtInt8:   returnValue.setU64Const(u64Const >> constant.i8Const);  break;
            case EbtUint8:  returnValue.setU64Const(u64Const >> constant.u8Const);  break;
            case EbtInt16:  returnValue.setU64Const(u64Const >> constant.i16Const); break;
            case EbtUint16: returnValue.setU64Const(u64Const >> constant.u16Const); break;
            case EbtInt:    returnValue.setU64Const(u64Const >> constant.iConst);   break;
            case EbtUint:   returnValue.setU64Const(u64Const >> constant.uConst);   break;
            case EbtInt64:  returnValue.setU64Const(u64Const >> constant.i64Const); break;
            case EbtUint64: returnValue.setU64Const(u64Const >> constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
#endif
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator<<(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        switch (type) {
#ifndef GLSLANG_WEB
        case EbtInt8:
            switch (constant.type) {
            case EbtInt8:   returnValue.setI8Const(i8Const << constant.i8Const);  break;
            case EbtUint8:  returnValue.setI8Const(i8Const << constant.u8Const);  break;
            case EbtInt16:  returnValue.setI8Const(i8Const << constant.i16Const); break;
            case EbtUint16: returnValue.setI8Const(i8Const << constant.u16Const); break;
            case EbtInt:    returnValue.setI8Const(i8Const << constant.iConst);   break;
            case EbtUint:   returnValue.setI8Const(i8Const << constant.uConst);   break;
            case EbtInt64:  returnValue.setI8Const(i8Const << constant.i64Const); break;
            case EbtUint64: returnValue.setI8Const(i8Const << constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint8:
            switch (constant.type) {
            case EbtInt8:   returnValue.setU8Const(u8Const << constant.i8Const);  break;
            case EbtUint8:  returnValue.setU8Const(u8Const << constant.u8Const);  break;
            case EbtInt16:  returnValue.setU8Const(u8Const << constant.i16Const); break;
            case EbtUint16: returnValue.setU8Const(u8Const << constant.u16Const); break;
            case EbtInt:    returnValue.setU8Const(u8Const << constant.iConst);   break;
            case EbtUint:   returnValue.setU8Const(u8Const << constant.uConst);   break;
            case EbtInt64:  returnValue.setU8Const(u8Const << constant.i64Const); break;
            case EbtUint64: returnValue.setU8Const(u8Const << constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtInt16:
            switch (constant.type) {
            case EbtInt8:   returnValue.setI16Const(i16Const << constant.i8Const);  break;
            case EbtUint8:  returnValue.setI16Const(i16Const << constant.u8Const);  break;
            case EbtInt16:  returnValue.setI16Const(i16Const << constant.i16Const); break;
            case EbtUint16: returnValue.setI16Const(i16Const << constant.u16Const); break;
            case EbtInt:    returnValue.setI16Const(i16Const << constant.iConst);   break;
            case EbtUint:   returnValue.setI16Const(i16Const << constant.uConst);   break;
            case EbtInt64:  returnValue.setI16Const(i16Const << constant.i64Const); break;
            case EbtUint64: returnValue.setI16Const(i16Const << constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint16:
            switch (constant.type) {
            case EbtInt8:   returnValue.setU16Const(u16Const << constant.i8Const);  break;
            case EbtUint8:  returnValue.setU16Const(u16Const << constant.u8Const);  break;
            case EbtInt16:  returnValue.setU16Const(u16Const << constant.i16Const); break;
            case EbtUint16: returnValue.setU16Const(u16Const << constant.u16Const); break;
            case EbtInt:    returnValue.setU16Const(u16Const << constant.iConst);   break;
            case EbtUint:   returnValue.setU16Const(u16Const << constant.uConst);   break;
            case EbtInt64:  returnValue.setU16Const(u16Const << constant.i64Const); break;
            case EbtUint64: returnValue.setU16Const(u16Const << constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
         case EbtInt64:
            switch (constant.type) {
            case EbtInt8:   returnValue.setI64Const(i64Const << constant.i8Const);  break;
            case EbtUint8:  returnValue.setI64Const(i64Const << constant.u8Const);  break;
            case EbtInt16:  returnValue.setI64Const(i64Const << constant.i16Const); break;
            case EbtUint16: returnValue.setI64Const(i64Const << constant.u16Const); break;
            case EbtInt:    returnValue.setI64Const(i64Const << constant.iConst);   break;
            case EbtUint:   returnValue.setI64Const(i64Const << constant.uConst);   break;
            case EbtInt64:  returnValue.setI64Const(i64Const << constant.i64Const); break;
            case EbtUint64: returnValue.setI64Const(i64Const << constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint64:
            switch (constant.type) {
            case EbtInt8:   returnValue.setU64Const(u64Const << constant.i8Const);  break;
            case EbtUint8:  returnValue.setU64Const(u64Const << constant.u8Const);  break;
            case EbtInt16:  returnValue.setU64Const(u64Const << constant.i16Const); break;
            case EbtUint16: returnValue.setU64Const(u64Const << constant.u16Const); break;
            case EbtInt:    returnValue.setU64Const(u64Const << constant.iConst);   break;
            case EbtUint:   returnValue.setU64Const(u64Const << constant.uConst);   break;
            case EbtInt64:  returnValue.setU64Const(u64Const << constant.i64Const); break;
            case EbtUint64: returnValue.setU64Const(u64Const << constant.u64Const); break;
            default:       assert(false && "Default missing");
            }
            break;
#endif
        case EbtInt:
            switch (constant.type) {
            case EbtInt:    returnValue.setIConst(iConst << constant.iConst);   break;
            case EbtUint:   returnValue.setIConst(iConst << constant.uConst);   break;
#ifndef GLSLANG_WEB
            case EbtInt8:   returnValue.setIConst(iConst << constant.i8Const);  break;
            case EbtUint8:  returnValue.setIConst(iConst << constant.u8Const);  break;
            case EbtInt16:  returnValue.setIConst(iConst << constant.i16Const); break;
            case EbtUint16: returnValue.setIConst(iConst << constant.u16Const); break;
            case EbtInt64:  returnValue.setIConst(iConst << constant.i64Const); break;
            case EbtUint64: returnValue.setIConst(iConst << constant.u64Const); break;
#endif
            default:       assert(false && "Default missing");
            }
            break;
        case EbtUint:
            switch (constant.type) {
            case EbtInt:    returnValue.setUConst(uConst << constant.iConst);   break;
            case EbtUint:   returnValue.setUConst(uConst << constant.uConst);   break;
#ifndef GLSLANG_WEB
            case EbtInt8:   returnValue.setUConst(uConst << constant.i8Const);  break;
            case EbtUint8:  returnValue.setUConst(uConst << constant.u8Const);  break;
            case EbtInt16:  returnValue.setUConst(uConst << constant.i16Const); break;
            case EbtUint16: returnValue.setUConst(uConst << constant.u16Const); break;
            case EbtInt64:  returnValue.setUConst(uConst << constant.i64Const); break;
            case EbtUint64: returnValue.setUConst(uConst << constant.u64Const); break;
#endif
            default:       assert(false && "Default missing");
            }
            break;
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator&(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtInt:    returnValue.setIConst(iConst & constant.iConst); break;
        case EbtUint:   returnValue.setUConst(uConst & constant.uConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(i8Const & constant.i8Const); break;
        case EbtUint8:  returnValue.setU8Const(u8Const & constant.u8Const); break;
        case EbtInt16:  returnValue.setI16Const(i16Const & constant.i16Const); break;
        case EbtUint16: returnValue.setU16Const(u16Const & constant.u16Const); break;
        case EbtInt64:  returnValue.setI64Const(i64Const & constant.i64Const); break;
        case EbtUint64: returnValue.setU64Const(u64Const & constant.u64Const); break;
#endif
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator|(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtInt:    returnValue.setIConst(iConst | constant.iConst); break;
        case EbtUint:   returnValue.setUConst(uConst | constant.uConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(i8Const | constant.i8Const); break;
        case EbtUint8:  returnValue.setU8Const(u8Const | constant.u8Const); break;
        case EbtInt16:  returnValue.setI16Const(i16Const | constant.i16Const); break;
        case EbtUint16: returnValue.setU16Const(u16Const | constant.u16Const); break;
        case EbtInt64:  returnValue.setI64Const(i64Const | constant.i64Const); break;
        case EbtUint64: returnValue.setU64Const(u64Const | constant.u64Const); break;
#endif
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator^(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtInt:    returnValue.setIConst(iConst ^ constant.iConst); break;
        case EbtUint:   returnValue.setUConst(uConst ^ constant.uConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(i8Const ^ constant.i8Const); break;
        case EbtUint8:  returnValue.setU8Const(u8Const ^ constant.u8Const); break;
        case EbtInt16:  returnValue.setI16Const(i16Const ^ constant.i16Const); break;
        case EbtUint16: returnValue.setU16Const(u16Const ^ constant.u16Const); break;
        case EbtInt64:  returnValue.setI64Const(i64Const ^ constant.i64Const); break;
        case EbtUint64: returnValue.setU64Const(u64Const ^ constant.u64Const); break;
#endif
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator~() const
    {
        TConstUnion returnValue;
        switch (type) {
        case EbtInt:    returnValue.setIConst(~iConst); break;
        case EbtUint:   returnValue.setUConst(~uConst); break;
#ifndef GLSLANG_WEB
        case EbtInt8:   returnValue.setI8Const(~i8Const); break;
        case EbtUint8:  returnValue.setU8Const(~u8Const); break;
        case EbtInt16:  returnValue.setI16Const(~i16Const); break;
        case EbtUint16: returnValue.setU16Const(~u16Const); break;
        case EbtInt64:  returnValue.setI64Const(~i64Const); break;
        case EbtUint64: returnValue.setU64Const(~u64Const); break;
#endif
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator&&(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtBool: returnValue.setBConst(bConst && constant.bConst); break;
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TConstUnion operator||(const TConstUnion& constant) const
    {
        TConstUnion returnValue;
        assert(type == constant.type);
        switch (type) {
        case EbtBool: returnValue.setBConst(bConst || constant.bConst); break;
        default:     assert(false && "Default missing");
        }

        return returnValue;
    }

    TBasicType getType() const { return type; }

private:
    union  {
        signed char        i8Const;     // used for i8vec, scalar int8s
        unsigned char      u8Const;     // used for u8vec, scalar uint8s
        signed short       i16Const;    // used for i16vec, scalar int16s
        unsigned short     u16Const;    // used for u16vec, scalar uint16s
        int                iConst;      // used for ivec, scalar ints
        unsigned int       uConst;      // used for uvec, scalar uints
        long long          i64Const;    // used for i64vec, scalar int64s
        unsigned long long u64Const;    // used for u64vec, scalar uint64s
        bool               bConst;      // used for bvec, scalar bools
        double             dConst;      // used for vec, dvec, mat, dmat, scalar floats and doubles
        const TString*     sConst;      // string constant
    };

    TBasicType type;
};

// Encapsulate having a pointer to an array of TConstUnion,
// which only needs to be allocated if its size is going to be
// bigger than 0.
//
// One convenience is being able to use [] to go inside the array, instead
// of C++ assuming it as an array of pointers to vectors.
//
// General usage is that the size is known up front, and it is
// created once with the proper size.
//
class TConstUnionArray {
public:
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    TConstUnionArray() : unionArray(nullptr) { }
    virtual ~TConstUnionArray() { }

    explicit TConstUnionArray(int size)
    {
        if (size == 0)
            unionArray = nullptr;
        else
            unionArray =  new TConstUnionVector(size);
    }
    TConstUnionArray(const TConstUnionArray& a) : unionArray(a.unionArray) { }
    TConstUnionArray(const TConstUnionArray& a, int start, int size)
    {
        unionArray = new TConstUnionVector(size);
        for (int i = 0; i < size; ++i)
            (*unionArray)[i] = a[start + i];
    }

    // Use this constructor for a smear operation
    TConstUnionArray(int size, const TConstUnion& val)
    {
        unionArray = new TConstUnionVector(size, val);
    }

    int size() const { return unionArray ? (int)unionArray->size() : 0; }
    TConstUnion& operator[](size_t index) { return (*unionArray)[index]; }
    const TConstUnion& operator[](size_t index) const { return (*unionArray)[index]; }
    bool operator==(const TConstUnionArray& rhs) const
    {
        // this includes the case that both are unallocated
        if (unionArray == rhs.unionArray)
            return true;

        if (! unionArray || ! rhs.unionArray)
            return false;

        return *unionArray == *rhs.unionArray;
    }
    bool operator!=(const TConstUnionArray& rhs) const { return ! operator==(rhs); }

    double dot(const TConstUnionArray& rhs)
    {
        assert(rhs.unionArray->size() == unionArray->size());
        double sum = 0.0;

        for (size_t comp = 0; comp < unionArray->size(); ++comp)
            sum += (*this)[comp].getDConst() * rhs[comp].getDConst();

        return sum;
    }

    bool empty() const { return unionArray == nullptr; }

protected:
    typedef TVector<TConstUnion> TConstUnionVector;
    TConstUnionVector* unionArray;
};

} // end namespace glslang

#endif // _CONSTANT_UNION_INCLUDED_

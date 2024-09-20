#pragma once

namespace math
{
namespace meta
{
    enum
    {
        COMP_N =    0x0,
        COMP_X =    0x1,
        COMP_Y =    0x2,
        COMP_Z =    0x3,
        COMP_W =    0x4,
        COMP_0 =    0x5,
        COMP_U =    0x6,
        COMP_A =    0x7
    };

#   define SWZ(x, y, z, w)      ((x)|((y) << 4)|((z) << 8)|((w) << 12))
#   define MSK(x, y, z, w)      (((x) ? 0xf : 0)|((y) ? 0xf0 : 0)|((z) ? 0xf00 : 0)|((w) ? 0xf000 : 0))

#   undef MAX   // some platforms define this in os headers
#   define MAX(a, b)            ((a)>(b) ? (a) : (b))


#   define COMP(swz, i)         (((swz) >> 4*(i)) & 0xf)
#   define MASK(swz)            ((((swz) & 0xf) ? 0xf : 0)|(((swz) & 0xf0) ? 0xf0 : 0)|(((swz) & 0xf00) ? 0xf00 : 0)|(((swz) & 0xf000) ? 0xf000 : 0))
#   define USED(swz)            ((swz)==(int) SWZ_ANY ? MSK_XYZW : ((((1 << 4*COMP(swz, 0))|(1 << 4*COMP(swz, 1))|(1 << 4*COMP(swz, 2))|(1 << 4*COMP(swz, 3))) >> 4)*0xf))
#   define BC(swz, n)           (((swz) & meta::MSK_XYZW)==meta::SWZ_ANY ? meta::SWZ_ANY : ((COMP(swz, 0)*0x1111) & ((1 << 4*n)-1)))
#   define FFS(swz)             (COMP(swz, 0) ? 0 : (COMP(swz, 1) ? 1 : (COMP(swz, 2) ? 2 : (COMP(swz, 3) ? 3 : 0))))
#   define MATCH(swz, c)        MSK(COMP(swz, 0)==c, COMP(swz, 1)==c, COMP(swz, 2)==c, COMP(swz, 3)==c)
#   define POP(swz)             ((COMP(swz, 0) ? 1 : 0)+(COMP(swz, 1) ? 1 : 0)+(COMP(swz, 2) ? 1 : 0)+(COMP(swz, 3) ? 1 : 0))
#   define ISDUP(swz)           (COMP(swz, FFS(swz))==COMP_A)
#   define ISSCAL(swz)          (POP(USED(swz))==1 ? COMP(swz, FFS(swz)) : 0)
#   define UNSWZ(swz, T)        (ISDUP(swz) ? SWZ_ANY : (MASK(swz) & SWZ_XYZW))
#   define IGNORED(swz)         MATCH(swz, COMP_U)
#   define ROTATE1(swz)         ((((swz) & meta::MSK_XYZ)<<4)|((swz)>>12))
#   define ROTATE2(swz)         ((((swz) & meta::MSK_XY)<<8)|((swz)>>8))
#   define ROTATE3(swz)         ((((swz) & meta::MSK_X)<<12)|((swz)>>4))

#   define SLEN(swz)            MAX(MAX(COMP(swz, 0), COMP(swz, 1)), MAX(COMP(swz, 2), COMP(swz, 3)))
#   define DLEN(swz)            (COMP(swz, 3) ? 4 : (COMP(swz, 2) ? 3 : (COMP(swz, 1) ? 2 : (COMP(swz, 0) ? 1 : 0))))

#   define LEN(swz)             ((swz)==meta::SWZ_ANY ? 1 : POP(swz))

#   define SELECT2(s0, c0, s1, c1) \
        ((c1)==(c0) ? (s1)<(s0) : (c1)<(c0))

#   define SELECT3(s0, c0, s1, c1, s2, c2) ( \
        (c2)==(c0) ? \
            ((c2)==(c1) ? ((s2)<(s1) && (s2)<(s0) ? 2 : (s1)<(s0)) : \
            (c2)<(c1) ? ((s2)<(s0) ? 2 : 0) : \
            1) : \
        (c2)<(c0) ? SELECT2(s1, c1, s2, c2)+1 : \
        SELECT2(s0, c0, s1, c1))

    enum
    {
        SWZ_X =     SWZ(COMP_X, 0, 0, 0),
        SWZ_Y =     SWZ(COMP_Y, 0, 0, 0),
        SWZ_Z =     SWZ(COMP_Z, 0, 0, 0),
        SWZ_W =     SWZ(COMP_W, 0, 0, 0),
        SWZ_XY =    SWZ(COMP_X, COMP_Y, 0, 0),
        SWZ_YZ =    SWZ(COMP_Y, COMP_Z, 0, 0),
        SWZ_ZW =    SWZ(COMP_Z, COMP_W, 0, 0),
        SWZ_XZ =    SWZ(COMP_X, COMP_Z, 0, 0),
        SWZ_YW =    SWZ(COMP_Y, COMP_W, 0, 0),
        SWZ_XYZ =   SWZ(COMP_X, COMP_Y, COMP_Z, 0),
        SWZ_YZX =   SWZ(COMP_Y, COMP_Z, COMP_X, 0),
        SWZ_ZXY =   SWZ(COMP_Z, COMP_X, COMP_Y, 0),
        SWZ_XYZW =  SWZ(COMP_X, COMP_Y, COMP_Z, COMP_W),
        SWZ_XXXX =  SWZ(COMP_X, COMP_X, COMP_X, COMP_X),
        SWZ_YYYY =  SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_Y),
        SWZ_ZZZZ =  SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_Z),
        SWZ_WWWW =  SWZ(COMP_W, COMP_W, COMP_W, COMP_W),
        SWZ_WZYX =  SWZ(COMP_W, COMP_Z, COMP_Y, COMP_X),
        SWZ_YZXY =  SWZ(COMP_Y, COMP_Z, COMP_X, COMP_Y),
        SWZ_ZXYW =  SWZ(COMP_Z, COMP_X, COMP_Y, COMP_W),
        SWZ_XXYY =  SWZ(COMP_X, COMP_X, COMP_Y, COMP_Y),
        SWZ_ZZWW =  SWZ(COMP_Z, COMP_Z, COMP_W, COMP_W),
        SWZ_XYXY =  SWZ(COMP_X, COMP_Y, COMP_X, COMP_Y),
        SWZ_ZWZW =  SWZ(COMP_Z, COMP_W, COMP_Z, COMP_W),
        SWZ_XXZZ =  SWZ(COMP_X, COMP_X, COMP_Z, COMP_Z),
        SWZ_YYWW =  SWZ(COMP_Y, COMP_Y, COMP_W, COMP_W),
        SWZ_WWZZ =  SWZ(COMP_W, COMP_W, COMP_Z, COMP_Z),
        SWZ_WWWZ =  SWZ(COMP_W, COMP_W, COMP_W, COMP_Z),
        SWZ_YZWX =  SWZ(COMP_Y, COMP_Z, COMP_W, COMP_X),
        SWZ_ZWXY =  SWZ(COMP_Z, COMP_W, COMP_X, COMP_Y),
        SWZ_ANY =   SWZ(COMP_A, COMP_A, COMP_A, COMP_A),
        SWZ_ALL =   0x10000,

        MSK_X =     MSK(1, 0, 0, 0),
        MSK_Y =     MSK(0, 1, 0, 0),
        MSK_XY =    MSK(1, 1, 0, 0),
        MSK_Z =     MSK(0, 0, 1, 0),
        MSK_W =     MSK(0, 0, 0, 1),
        MSK_ZW =    MSK(0, 0, 1, 1),
        MSK_XZ =    MSK(1, 0, 1, 0),
        MSK_YW =    MSK(0, 1, 0, 1),
        MSK_XYZ =   MSK(1, 1, 1, 0),
        MSK_XYZW =  MSK(1, 1, 1, 1)
    };

    template<int SWZ> struct Swizzle
    {
        enum
        {
            DUP = ISDUP(SWZ),
            I = COMP(SWZ, 0) ? 0 : COMP(SWZ, 1) ? 1 : COMP(SWZ, 2) ? 2 : COMP(SWZ, 3) ? 3 : 4,
            J = COMP(SWZ, I + 1) ? I + 1 : COMP(SWZ, I + 2) ? I + 2 : COMP(SWZ, I + 3) ? I + 3 : 4,
            K = COMP(SWZ, J + 1) ? J + 1 : COMP(SWZ, J + 2) ? J + 2 : 4,
            L = COMP(SWZ, K + 1) ? K + 1 : 4,
            N = DUP ? 1 : (I < 4) + (J < 4) + (K < 4) + (L < 4),
            C0 = DUP ? 0 : (I < 4 ? COMP(SWZ, I) - COMP_X : -1),
            C1 = DUP ? 0 : (J < 4 ? COMP(SWZ, J) - COMP_X : -1),
            C2 = DUP ? 0 : (K < 4 ? COMP(SWZ, K) - COMP_X : -1),
            C3 = DUP ? 0 : (L < 4 ? COMP(SWZ, L) - COMP_X : -1)
        };
    };

    template<int SRC, int SWZ> struct Compose
    {
        enum
        {
            S0 = COMP(SWZ, 0),
            S1 = COMP(SWZ, 1),
            S2 = COMP(SWZ, 2),
            S3 = COMP(SWZ, 3),
            D0 = SRC == SWZ_ANY ? COMP_X : (S0 ? COMP(SRC, MAX(S0 - COMP_X, 0)) : 0),
            D1 = SRC == SWZ_ANY ? COMP_Y : (S1 ? COMP(SRC, MAX(S1 - COMP_X, 0)) : 0),
            D2 = SRC == SWZ_ANY ? COMP_Z : (S2 ? COMP(SRC, MAX(S2 - COMP_X, 0)) : 0),
            D3 = SRC == SWZ_ANY ? COMP_W : (S3 ? COMP(SRC, MAX(S3 - COMP_X, 0)) : 0),
            RES = SWZ(D0, D1, D2, D3)
        };
    };

    template<int SRC, int DST> struct Inverse
    {
        enum
        {
            MSK_D = MASK(DST),
            USE_S = USED(SRC),
            POP_S = ISDUP(SRC) ? 1 : POP(USE_S),
            FFS = FFS(USED(SRC)),
            SCL = POP_S == 1 ? (ISDUP(SRC) ? SWZ_XYZW : SWZ(FFS + COMP_X, FFS + COMP_X, FFS + COMP_X, FFS + COMP_X)) : 0,
            S = SCL ? (SCL & MSK_D) : SRC,
            MSK_S = MASK(S),

            S0 = COMP(S, 0),
            S1 = COMP(S, 1),
            S2 = COMP(S, 2),
            S3 = COMP(S, 3),

            D0 = COMP(DST, 0),
            D1 = COMP(DST, 1),
            D2 = COMP(DST, 2),
            D3 = COMP(DST, 3),

            INV =
                (MSK_S == MSK_D) &&
                (D0 == 0 || D0 != D1 || S0 == S1) &&
                (D0 == 0 || D0 != D2 || S0 == S2) &&
                (D0 == 0 || D0 != D3 || S0 == S3) &&
                (D1 == 0 || D1 != D2 || S1 == S2) &&
                (D1 == 0 || D1 != D3 || S1 == S3) &&
                (D2 == 0 || D2 != D3 || S2 == S3),

            C0 = (D0 == COMP_X ? S0 : (D1 == COMP_X ? S1 : (D2 == COMP_X ? S2 : (D3 == COMP_X ? S3 : 0)))),
            C1 = (D0 == COMP_Y ? S0 : (D1 == COMP_Y ? S1 : (D2 == COMP_Y ? S2 : (D3 == COMP_Y ? S3 : 0)))),
            C2 = (D0 == COMP_Z ? S0 : (D1 == COMP_Z ? S1 : (D2 == COMP_Z ? S2 : (D3 == COMP_Z ? S3 : 0)))),
            C3 = (D0 == COMP_W ? S0 : (D1 == COMP_W ? S1 : (D2 == COMP_W ? S2 : (D3 == COMP_W ? S3 : 0)))),

            SWZ = (SRC & MSK_S) == (SWZ_ANY & MSK_S) ? (SRC & MSK_XYZW) : ((DST & MSK_D) == (SWZ_ANY & MSK_D) ? SCL : (INV ? SWZ(C0, C1, C2, C3) : 0))
        };
    };

    template<typename T, int RHS, int HINT = 0> struct UnOp
    {
        enum
        {
            ANY = ISDUP(RHS) ? SWZ_ANY : 0,
            MSK = MASK(RHS),

#       ifdef META_PEEPHOLE
            R = ANY | (MSK & SWZ_XYZW),
            S = ANY | RHS,
            COST = 0,
#       else
            R = ANY | RHS,
            S = ANY | (MSK & SWZ_XYZW)
#       endif
        };
    };

    template<typename T, int LHS, int RHS, int HINT = 0> struct BinOp
    {
        enum
        {
            LMSK = MASK(LHS),
            RMSK = MASK(RHS),

            ANY = (ISDUP(LHS) && ISDUP(RHS)) ? SWZ_ANY : 0,
            MSK = (ISDUP(LHS) ? RMSK : LMSK) | (ISDUP(RHS) ? LMSK : RMSK) | (ANY ? MSK_XYZW : 0),

#       ifdef META_PEEPHOLE
            // case0: conform lhs and rhs to destination
            RES0 = MSK & SWZ_XYZW,
            LHS0 = LHS,
            RHS0 = RHS,
            COST0 = T::template SWIZ<LHS0>::COST + T::template SWIZ<RHS0>::COST - (MASK(RES0) == HINT),
            // case1: conform rhs to lhs
            LHS1 = LHS == SWZ_ANY ? LHS : (SWZ_XYZW & USED(LHS)),
            RHS1 = Inverse<RHS, LHS>::SWZ,
            RES1 = RHS1 != 0 ? LHS : 0,
            COST1 = RES1 ? T::template SWIZ<RHS1>::COST - (MASK(RES1) == HINT) : (COST0 + 1),
            // case2: conform lhs to rhs
            RHS2 = RHS == SWZ_ANY ? RHS : (SWZ_XYZW & USED(RHS)),
            LHS2 = Inverse<LHS, RHS>::SWZ,
            RES2 = LHS2 != 0 ? RHS : 0,
            COST2 = RES2 ? T::template SWIZ<LHS2>::COST - (MASK(RES2) == HINT) : (COST0 + 1),

            SEL = SELECT3(RES0, COST0, RES1, COST1, RES2, COST2),

            L = SEL == 0 ? LHS0 : (SEL == 1 ? LHS1 : LHS2),
            R = SEL == 0 ? RHS0 : (SEL == 1 ? RHS1 : RHS2),
            S = ANY | (SEL == 0 ? RES0 : (SEL == 1 ? RES1 : RES2)),
            COST = SEL == 0 ? COST0 : (SEL == 1 ? COST1 : COST2),
#       else
            USE_L = (RHS == SWZ_ANY) | (LHS == RHS) ? USED(LHS) : 0,
            USE_R = (LHS == SWZ_ANY) | (RHS == LHS) ? USED(RHS) : 0,

            L = LHS == SWZ_ANY ? SWZ_ANY : (USE_L ? (USE_L & SWZ_XYZW) : LHS),
            R = RHS == SWZ_ANY ? SWZ_ANY : (USE_R ? (USE_R & SWZ_XYZW) : RHS),
            S = ANY | (USE_L ? LHS : (USE_R ? RHS : MSK & SWZ_XYZW))
#       endif
        };
    };

    template<typename T, int LHS, int CHS, int RHS, int HINT = 0> struct TernOp
    {
        enum
        {
            LMSK = MASK(LHS),
            CMSK = MASK(CHS),
            RMSK = MASK(RHS),

            ANY = (ISDUP(LHS) && ISDUP(CHS) && ISDUP(RHS)) ? SWZ_ANY : 0,
            MSK = (!ISDUP(LHS) ? LMSK : MSK_X) | (!ISDUP(CHS) ? CMSK : MSK_X) | (!ISDUP(RHS) ? RMSK : MSK_X) | (ANY ? MSK_XYZW : 0),

#       ifdef META_PEEPHOLE
            L0 = BinOp<T, LHS, CHS, HINT>::L,
            C0 = BinOp<T, LHS, CHS, HINT>::R,
            S0 = BinOp<T, LHS, CHS, HINT>::S,
            LC = BinOp<T, S0, RHS, HINT>::L,
            L = Compose<L0, LC>::RES,
            C = Compose<C0, LC>::RES,
            R = BinOp<T, S0, RHS, HINT>::R,
            S = BinOp<T, S0, RHS, HINT>::S,
            COST = BinOp<T, LHS, CHS, HINT>::COST + BinOp<T, S0, RHS, HINT>::COST
#       else
            U = USED(LHS) & SWZ_XYZW,
            L0 = LHS == SWZ_ANY ? SWZ_ANY : (LHS == CHS && LHS == RHS ? U : LHS),
            C0 = CHS == SWZ_ANY ? SWZ_ANY : (CHS == LHS && CHS == RHS ? U : CHS),
            S0 = (ANY | MSK) & SWZ_XYZW,
            LC = (ANY | MSK) & SWZ_XYZW,
            L = Compose<L0, LC>::RES,
            C = Compose<C0, LC>::RES,
            R = RHS == SWZ_ANY ? SWZ_ANY : (RHS == LHS && RHS == CHS ? U : RHS),
            S = (ANY | MSK) & SWZ_XYZW
#       endif
        };
    };

    template<int RAW> struct SwizBase
    {
        enum
        {
            IGN = (COMP(RAW, 0) == COMP_U ? 0xf : 0) | (COMP(RAW, 1) == COMP_U ? 0xf0 : 0) | (COMP(RAW, 2) == COMP_U ? 0xf00 : 0) | (COMP(RAW, 3) == COMP_U ? 0xf000 : 0),
            SWZ = RAW & ~IGN,
            MSK = MASK(SWZ),
            USE = USED(SWZ)
        };
    };

#   define MATCH_SWIZ(SCL, SWZ, REF, MSK, IGN)      ((SCL && (MSK|IGN)==MASK(REF)) || (((SWZ & ~IGN)|(IGN & REF)) & MSK)==(REF & MSK))

#   if defined(META_PEEPHOLE)
#       define META_COST(expr)  enum expr;
#   else
#       define META_COST(expr)
#   endif
}
}

#pragma once

#include "config.h"
#include "vec-swz.h"

namespace math
{
namespace meta
{
    template<typename T, typename S, int N> struct v : S
    {
        typedef typename S::packed packed;

        static MATH_FORCEINLINE v pack(const packed &p)
        {
            v r; r.p = p;
            return r;
        }
    };
    template<typename T, typename S> struct v<T, S, 1> : S
    {
        typedef typename S::packed packed;
        typedef typename T::type type;

        MATH_FORCEINLINE operator type() const
        {
            return T::template GET<S::SWZ>::f(this->p);
        }

        static MATH_FORCEINLINE v pack(const packed &p)
        {
            v r; r.p = p;
            return r;
        }
    };

    template<typename T, typename S, int N> struct lv : v<T, S, N>
    {
        enum
        {
            SWZ = S::SWZ, SEL = USED(SWZ)
        };
        typedef typename S::packed packed;

        static MATH_FORCEINLINE lv pack(const packed &p)
        {
            lv r; r.p = p;
            return r;
        }

        MATH_FORCEINLINE const typename T::type *operator&() const
        {
            return &((const typename T::type *) this)[FFS(SEL)];
        }

        MATH_FORCEINLINE typename T::type *operator&()
        {
            return &((typename T::type *) this)[FFS(SEL)];
        }

        // must be const to avoid bitwize copy
        MATH_FORCEINLINE vec_attr lv &operator=(const v<T, S, N> &rhs) vec_attr
        {
            this->p = T::template MASK<SWZ_XYZW & ~SEL, SWZ_XYZW & SEL, SEL>::f(this->p, rhs.p);
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr lv &operator=(const v<T, RHS, 1> &rhs) vec_attr
        {
            this->p = T::template MASK<SWZ_XYZW & ~SEL, Inverse<BC(RHS::SWZ, N), SWZ>::SWZ, SEL>::f(this->p, rhs.p);
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr lv &operator=(const v<T, RHS, N> &rhs) vec_attr
        {
            this->p = T::template MASK<SWZ_XYZW & ~SEL, Inverse<RHS::SWZ, SWZ>::SWZ, SEL>::f(this->p, rhs.p);
            return *this;
        }

        MATH_FORCEINLINE vec_attr lv &operator=(typename T::type rhs) vec_attr
        {
            this->p = T::template MASK<SWZ_XYZW & ~SEL, SWZ_ANY, SEL>::f(this->p, T::CTOR(rhs));
            return *this;
        }
    };
    template<typename T, typename S> struct lv<T, S, 1> : v<T, S, 1>
    {
        enum
        {
            SWZ = S::SWZ, SEL = USED(SWZ)
        };
        typedef typename S::packed packed;
        typedef typename T::type type;

        static MATH_FORCEINLINE lv pack(const packed &p)
        {
            lv r; r.p = p;
            return r;
        }

        MATH_FORCEINLINE operator type() const
        {
            return T::template GET<S::SWZ>::f(this->p);
        }

        MATH_FORCEINLINE const typename T::type *operator&() const
        {
            return &((const typename T::type *) this)[FFS(SEL)];
        }

        MATH_FORCEINLINE typename T::type *operator&()
        {
            return &((typename T::type *) this)[FFS(SEL)];
        }

        MATH_FORCEINLINE vec_attr lv &operator=(const v<T, S, 1> &rhs) vec_attr // must be const to avoid bitwize copy
        {
            this->p = T::template MASK<SWZ_XYZW & ~SEL, SWZ_XYZW & SEL, SEL>::f(this->p, rhs.p);
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr lv &operator=(const v<T, RHS, 1> &rhs) vec_attr // must be const to avoid bitwize copy
        {
            this->p = T::template MASK<SWZ_XYZW & ~SEL, Inverse<RHS::SWZ, SWZ>::SWZ, SEL>::f(this->p, rhs.p);
            return *this;
        }

        MATH_FORCEINLINE vec_attr lv &operator=(typename T::type rhs) vec_attr // must be const to avoid bitwize copy
        {
            this->p = T::template SET<SWZ>::f(this->p, rhs);
            return *this;
        }
    };

    template<typename T, int S, int N> struct rp
    {
        enum
        {
            SWZ = S
        };
        typedef typename T::packed packed;

        mutable packed p;
    };

    template<typename T, int S, int N> struct sp
    {
        enum
        {
            SWZ = S
        };
        typedef typename T::template packed<SWZ>::type packed;
        mutable packed p;
#   if defined(MATH_USE_DTOR)
        MATH_EMPTYINLINE ~sp()
        {
        }

#   endif
    };

    template<typename T, int S> struct sp<T, S, 1>
    {
        enum
        {
            SWZ = S
        };
        typedef typename T::packed packed;
        union
        {
            mutable packed p;
            lv<T, rp<T, COMP(SWZ, 0), 1>, 1> x;
        };

#   if defined(__GNUC__)
        sp() {}
        sp(const sp<T, S, 1>& v) : p(v.p) {}   // Necessary for correct optimized GCC codegen
#   endif

#   if defined(MATH_USE_DTOR)
        MATH_EMPTYINLINE ~sp()
        {
        }

#   endif
    };

    template<typename T, int S> struct sp<T, S, 2>
    {
        enum
        {
            SWZ = S
        };
        typedef typename T::packed packed;
        union
        {
            mutable packed p;
            lv<T, rp<T, COMP(SWZ, 0), 1>, 1> x, lo, even;
            lv<T, rp<T, COMP(SWZ, 1), 1>, 1> y, hi, odd;
            lv<T, rp<T, SWZ & MSK_XY, 2>, 2> xy;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), 0, 0), 2>, 2> yx;
            v<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 1)), 4>, 4> xxyy;
            v<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 1)), 4>, 4> xyxy;
        };
#   if defined(__GNUC__)
        sp() {}
        sp(const sp<T, S, 2>& v) : p(v.p) {}   // Necessary for correct optimized GCC codegen
#   endif
#   if defined(MATH_USE_DTOR)
        MATH_EMPTYINLINE ~sp()
        {
        }

#   endif
    };

    template<typename T, int S> struct sp<T, S, 3>
    {
        enum
        {
            SWZ = S
        };
        typedef typename T::packed packed;
        union
        {
            mutable packed p;
            lv<T, rp<T, COMP(SWZ, 0), 1>, 1> x;
            lv<T, rp<T, COMP(SWZ, 1), 1>, 1> y;
            lv<T, rp<T, COMP(SWZ, 2), 1>, 1> z;
            lv<T, rp<T, SWZ & MSK_XY, 2>, 2> xy;
            lv<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 2), 0, 0), 2>, 2> xz;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), 0, 0), 2>, 2> yx;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), 0, 0), 2>, 2> yz;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 0), 0, 0), 2>, 2> zx;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 1), 0, 0), 2>, 2> zy;
            lv<T, rp<T, SWZ & MSK_XYZ, 3>, 3> xyz;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 0), 0), 3>, 3> yxx;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 0), 0), 3>, 3> yzx;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 0), COMP(SWZ, 1), 0), 3>, 3> zxy;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 2), COMP(SWZ, 1), 0), 3>, 3> zzy;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 1), COMP(SWZ, 0), 0), 3>, 3> zyx;

            lv<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 2), COMP(SWZ, 1), 0), 3>, 3> xzy;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 2), 0), 3>, 3> yxz;

            lv<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 0)), 4>, 4> xyzx;
        };
#   if defined(__GNUC__)
        sp() {}
        sp(const sp<T, S, 3>& v) : p(v.p) {}   // Necessary for correct optimized GCC codegen
#   endif
#   if defined(MATH_USE_DTOR)
        MATH_EMPTYINLINE ~sp()
        {
        }

#   endif
    };

    template<typename T, int S> struct sp<T, S, 4>
    {
        enum
        {
            SWZ = S
        };
        typedef typename T::packed packed;
        union
        {
            mutable packed p;
            lv<T, rp<T, COMP(SWZ, 0), 1>, 1> x;
            lv<T, rp<T, COMP(SWZ, 1), 1>, 1> y;
            lv<T, rp<T, COMP(SWZ, 2), 1>, 1> z;
            lv<T, rp<T, COMP(SWZ, 3), 1>, 1> w;
            lv<T, rp<T, SWZ & MSK_XY, 2>, 2> xy, lo;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), 0, 0), 2>, 2> yx;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), 0, 0), 2>, 2> yz;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 0), 0, 0), 2>, 2> zx;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 1), 0, 0), 2>, 2> zy;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 3), 0, 0), 2>, 2> zw, hi;
            lv<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 2), 0, 0), 2>, 2> xz, even;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 3), 0, 0), 2>, 2> yw, odd;
            lv<T, rp<T, SWZ & MSK_XYZ, 3>, 3> xyz;
            lv<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 2), COMP(SWZ, 1), 0), 3>, 3> xzy;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 3), 0), 3>, 3> yzw;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 0), 0), 3>, 3> yzx;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 0), COMP(SWZ, 1), 0), 3>, 3> zxy;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 1), COMP(SWZ, 0), 0), 3>, 3> zyx;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 3), 0), 3>, 3> yxw;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 3), COMP(SWZ, 0), 0), 3>, 3> zwx;
            lv<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 2), COMP(SWZ, 1), 0), 3>, 3> wzy;
            lv<T, rp<T, SWZ, 4>, 4> xyzw;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 3), COMP(SWZ, 0)), 4>, 4> yzwx;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 3), COMP(SWZ, 0), COMP(SWZ, 1)), 4>, 4> zwxy;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 3)), 4>, 4> zyxw;
            lv<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 2)), 4>, 4> wxyz;
            lv<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 2), COMP(SWZ, 1), COMP(SWZ, 0)), 4>, 4> wzyx;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 0), COMP(SWZ, 3)), 4>, 4> yzxw;
            lv<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 3)), 4>, 4> zxyw;
            lv<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 2), COMP(SWZ, 1), COMP(SWZ, 3)), 4>, 4> xzyw;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 2), COMP(SWZ, 3)), 4>, 4> yxzw;
            lv<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 3), COMP(SWZ, 1), COMP(SWZ, 2)), 4>, 4> xwyz;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 3), COMP(SWZ, 2), COMP(SWZ, 0)), 4>, 4> ywzx;
            lv<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 2), COMP(SWZ, 0), COMP(SWZ, 1)), 4>, 4> wzxy;
            lv<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 3), COMP(SWZ, 2)), 4>, 4> yxwz;
            v<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 0), COMP(SWZ, 2), COMP(SWZ, 2)), 4>, 4> xxzz;
            v<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 1)), 4>, 4> xyxy;
            v<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 2)), 4>, 4> xyzz;
            v<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 1), COMP(SWZ, 3), COMP(SWZ, 3)), 4>, 4> yyww;
            v<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 2), COMP(SWZ, 0)), 4>, 4> yzzx;
            v<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 0), COMP(SWZ, 1), COMP(SWZ, 1)), 4>, 4> xxyy;
            v<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 0), COMP(SWZ, 2), COMP(SWZ, 0)), 4>, 4> zxzx;
            v<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 2), COMP(SWZ, 3), COMP(SWZ, 3)), 4>, 4> zzww;
            v<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 3), COMP(SWZ, 2), COMP(SWZ, 3)), 4>, 4> zwzw;
            v<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 2), COMP(SWZ, 0), COMP(SWZ, 1)), 4>, 4> yzxy;
            v<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 3), COMP(SWZ, 3), COMP(SWZ, 0)), 4>, 4> wwwx;
            v<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 0), COMP(SWZ, 0), COMP(SWZ, 3)), 4>, 4> yxxw;
            v<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 2), COMP(SWZ, 1), COMP(SWZ, 3)), 4>, 4> zzyw;
            v<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 3), COMP(SWZ, 3), COMP(SWZ, 2)), 4>, 4> wwwz;
            v<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 3), COMP(SWZ, 2), COMP(SWZ, 2)), 4>, 4> wwzz;
            v<T, rp<T, SWZ(COMP(SWZ, 0), COMP(SWZ, 0), COMP(SWZ, 0), COMP(SWZ, 0)), 4>, 4> xxxx;
            v<T, rp<T, SWZ(COMP(SWZ, 1), COMP(SWZ, 1), COMP(SWZ, 1), COMP(SWZ, 1)), 4>, 4> yyyy;
            v<T, rp<T, SWZ(COMP(SWZ, 2), COMP(SWZ, 2), COMP(SWZ, 2), COMP(SWZ, 2)), 4>, 4> zzzz;
            v<T, rp<T, SWZ(COMP(SWZ, 3), COMP(SWZ, 3), COMP(SWZ, 3), COMP(SWZ, 3)), 4>, 4> wwww;
        };
#   if defined(__GNUC__)
        sp() {}
        sp(const sp<T, S, 4>& v) : p(v.p) {}   // Necessary for correct optimized GCC codegen
#   endif
#   if defined(MATH_USE_DTOR)
        MATH_EMPTYINLINE ~sp()
        {
        }

#   endif
    };
}
}

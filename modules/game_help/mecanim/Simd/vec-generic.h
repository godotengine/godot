#pragma once

#include "config.h"

namespace math
{
namespace meta
{
    // l-values
    template<typename T> struct swz_xy
    {
        scalar_attr T x, y;
    };
    template<typename T> struct swz_yx
    {
        scalar_attr T y, x;
    };

    template<typename T> struct swz_xyz
    {
        scalar_attr T x, y, z;
    };
    template<typename T> struct swz_xzy
    {
        scalar_attr T x, z, y;
    };
    template<typename T> struct swz_yzx
    {
        scalar_attr T z, x, y;
    };
    template<typename T> struct swz_zxy
    {
        scalar_attr T y, z, x;
    };
    template<typename T> struct swz_zyx
    {
        scalar_attr T z, y, x;
    };
    template<typename T> struct swz_yxz
    {
        scalar_attr T y, x, z;
    };

    template<typename T> struct swz_yxx
    {
        union
        {
            T y, z;
        };
        T x;
    private:
        T _v1;
    };

    template<typename T> struct swz_zzy
    {
    private:
        T _v0;
    public:
        T z;
        union
        {
            T x, y;
        };
    };

    template<typename T> struct swz_xz
    {
    public:
        scalar_attr T x;
    private:
        scalar_attr T _v1;
    public:
        scalar_attr T y;
    };

    template<typename T> struct swz_yz
    {
    private:
        scalar_attr T _v0;
    public:
        scalar_attr T x;
    public:
        scalar_attr T y;
    };

    template<typename T> struct swz_yw
    {
    private:
        scalar_attr T _v0;
    public:
        scalar_attr T x;
    private:
        scalar_attr T _v2;
    public:
        scalar_attr T y;
    };

    template<typename T> struct swz_zx
    {
    public:
        scalar_attr T y;
    private:
        scalar_attr T _v1;
    public:
        scalar_attr T x;
    };

    template<typename T> struct swz_zy
    {
    private:
        scalar_attr T _v0;
    public:
        scalar_attr T y;
    public:
        scalar_attr T x;
    };

    template<typename T> struct swz_xyzw
    {
        scalar_attr T x, y, z, w;
    };

    template<typename T> struct swz_yzxw
    {
        scalar_attr T z, x, y, w;
    };

    template<typename T> struct swz_zxyw
    {
        scalar_attr T y, z, x, w;
    };


    template<typename T> struct swz_xzyw
    {
        scalar_attr T x, z, y, w;
    };

    template<typename T> struct swz_wzyx
    {
        scalar_attr T w, z, y, x;
    };

    template<typename T> struct swz_ywzx
    {
        scalar_attr T w, x, z, y;
    };

    template<typename T> struct swz_xwyz
    {
        scalar_attr T x, z, w, y;
    };

    template<typename T> struct swz_yxzw
    {
        scalar_attr T y, x, z, w;
    };

    template<typename T> struct swz_wxyz
    {
        scalar_attr T y, z, w, x;
    };

    template<typename T> struct swz_wzxy
    {
        scalar_attr T z, w, y, x;
    };

    template<typename T> struct swz_zwxy
    {
        scalar_attr T z, w, x, y;
    };

    template<typename T> struct swz_zyxw
    {
        scalar_attr T z, y, x, w;
    };

    template<typename T> struct swz_yxwz
    {
        scalar_attr T y, x, w, z;
    };

    template<typename T> struct swz_yzwx
    {
        scalar_attr T w, x, y, z;
    };

    template<typename T> struct swz_yzw
    {
    private:
        scalar_attr T _v0;
    public:
        scalar_attr T x, y, z;
    };

    template<typename T> struct swz_zw
    {
    private:
        scalar_attr T _v0, _v1;
    public:
        scalar_attr T x, y;
    };

    // r-values
    template<typename T> struct swz_zwzw
    {
    private:
        T _v0, _v1;
    public:
        union
        {
            T x, z;
        };
    public:
        union
        {
            T y, w;
        };
    };

    template<typename T> struct swz_zzww
    {
    private:
        T _v0, _v1;
    public:
        union
        {
            T x, y;
        };
    public:
        union
        {
            T z, w;
        };
    };

    template<typename T> struct swz_xxzz
    {
    public:
        union
        {
            T x, y;
        };
    private:
        T _v1;
    public:
        union
        {
            T z, w;
        };
    private:
        T _v3;
    };

    template<typename T> struct swz_yyww
    {
    private:
        T _v0;
    public:
        union
        {
            T x, y;
        };
    private:
        T _v1;
    public:
        union
        {
            T z, w;
        };
    };

    template<typename T> struct swz_yzzx
    {
    public:
        T w, x;
        union
        {
            T y, z;
        };
    private:
        T _v3;
    };

    template<typename T> struct swz_yxxw
    {
        union
        {
            T y, z;
        };
        T x;
    private:
        T _v1;
    public:
        T w;
    };

    template<typename T> struct swz_xxyy
    {
        union
        {
            T x, y;
        };
        union
        {
            T z, w;
        };
    };

    template<typename T> struct swz_xyxy
    {
    public:
        union
        {
            T x, z;
        };
        union
        {
            T y, w;
        };
    };

    template<typename T> struct swz_xyzz
    {
    public:
        T x, y;
        union
        {
            T z, w;
        };
    private:
        T _v3;
    };

    template<typename T> struct swz_yzxy
    {
        T  z;
        union
        {
            T x, w;
        };
        T y;
    };

    template<typename T> struct swz_xyzx
    {
        union
        {
            T x, w;
        };
        T y;
        T z;
    };

    template<typename T> struct swz_zxzx
    {
    public:
        union
        {
            T y, w;
        };
    private:
        T _v1;
    public:
        union
        {
            T x, z;
        };
    private:
        T _v3;
    };

    template<typename T> struct swz_zzyw
    {
    private:
        T _v0;
    public:
        T z;
        union
        {
            T x, y;
        };
        T w;
    };

    template<typename T> struct swz_www
    {
    private:
        T _v0, _v1, _v2;
    public:
        union
        {
            T x, y, z;
        };
    };

    template<typename T> struct swz_wwwx
    {
    public:
        T w;
    private:
        T _v1, _v2;
    public:
        union
        {
            T x, y, z;
        };
    };

    template<typename T> struct swz_wwwz
    {
    private:
        T _v0, _v1;
    public:
        T w;
    public:
        union
        {
            T x, y, z;
        };
    };

    template<typename T> struct swz_wwzz
    {
    private:
        T _v0, _v1;
    public:
        union
        {
            T z, w;
        };
    public:
        union
        {
            T x, y;
        };
    };

    template<typename T> struct swz_xxxx
    {
    public:
        union
        {
            T x, y, z, w;
        };
    };

    template<typename T> struct swz_yyyy
    {
    private:
        T _v0;
    public:
        union
        {
            T x, y, z, w;
        };
    };

    template<typename T> struct swz_zzzz
    {
    private:
        T _v0, _v1;
    public:
        union
        {
            T x, y, z, w;
        };
    };

    template<typename T> struct swz_wwww
    {
    private:
        T _v0, _v1, _v2;
    public:
        union
        {
            T x, y, z, w;
        };
    };

    // rvalues
    template<typename T, typename P, unsigned N> struct rv : P
    {
        typedef P       base;
    };

    // lvalues
    template<typename T, typename P, unsigned N> struct lv : rv<T, P, N>
    {
        typedef rv<T, P, N> base;
    };

    template<typename T, typename P> struct lv<T, P, 1U> : rv<T, P, 1U>
    {
        typedef rv<T, P, 1U>    base;
        MATH_FORCEINLINE vec_attr lv &set(T x) vec_attr
        {
            this->x = x;
            return *this;
        }

        MATH_FORCEINLINE vec_attr lv &operator=(T rhs) vec_attr
        {
            this->x = rhs;
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr lv &operator=(const rv<T, RHS, 1U> &rhs) vec_attr
        {
            return set((T)rhs.x);
        }

        /*
            MATH_FORCEINLINE const T *operator&() const
            {
                return &this->x;
            }
            MATH_FORCEINLINE T *operator&() vec_attr
            {
                return (T *) &this->x;
            }
        */
    };

    template<typename T, typename P> struct lv<T, P, 2U> : rv<T, P, 2U>
    {
        typedef rv<T, P, 2U>    base;
        MATH_FORCEINLINE vec_attr lv &set(T x, T y) vec_attr
        {
            this->x = x; this->y = y;
            return *this;
        }

        MATH_FORCEINLINE vec_attr lv &operator=(T rhs) vec_attr
        {
            this->x = this->y = rhs;
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr lv &operator=(const rv<T, RHS, 2U> &rhs) vec_attr
        {
            return set((T)rhs.x, (T)rhs.y);
        }
    };

    template<typename T, typename P> struct lv<T, P, 3U> : rv<T, P, 3U>
    {
        typedef rv<T, P, 3U>    base;
        MATH_FORCEINLINE vec_attr lv &set(T x, T y, T z) vec_attr
        {
            this->x = x; this->y = y; this->z = z;
            return *this;
        }

        MATH_FORCEINLINE vec_attr lv &operator=(T rhs) vec_attr
        {
            this->x = this->y = this->z = rhs;
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr lv &operator=(const rv<T, RHS, 3U> &rhs) vec_attr
        {
            return set((T)rhs.x, (T)rhs.y, (T)rhs.z);
        }
    };

    template<typename T, typename P> struct lv<T, P, 4U> : rv<T, P, 4U>
    {
        typedef rv<T, P, 4U>    base;
        MATH_FORCEINLINE vec_attr lv &set(T x, T y, T z, T w) vec_attr
        {
            this->x = x; this->y = y; this->z = z; this->w = w;
            return *this;
        }

        MATH_FORCEINLINE vec_attr lv &operator=(T rhs) vec_attr
        {
            this->x = this->y = this->z = this->w = rhs;
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr lv &operator=(const rv<T, RHS, 4U> &rhs) vec_attr
        {
            return set((T)rhs.x, (T)rhs.y, (T)rhs.z, (T)rhs.w);
        }
    };

    template<typename T> struct vec2_xy
    {
        union
        {
            lhs_attr lv<T, swz_xy<T>, 2U>       xy;
            lhs_attr lv<T, swz_yx<T>, 2U>       yx;
            rhs_attr rv<T, swz_xxyy<T>, 4U>     xxyy;
            rhs_attr rv<T, swz_xyxy<T>, 4U>     xyxy;
            // xx
            // yy
            struct
            {
                scalar_attr T x, y;
            };
        };
    };
    template<typename T> struct vec3_xyz
    {
        union
        {
            lhs_attr lv<T, swz_xy<T>, 2U>       xy;
            lhs_attr lv<T, swz_xz<T>, 2U>       xz;
            lhs_attr lv<T, swz_yx<T>, 2U>       yx;
            lhs_attr lv<T, swz_yz<T>, 2U>       yz;
            lhs_attr lv<T, swz_zx<T>, 2U>       zx;
            lhs_attr lv<T, swz_zy<T>, 2U>       zy;
            // xx
            // yy
            // zz
            lhs_attr lv<T, swz_xyz<T>, 3U>      xyz;
            lhs_attr lv<T, swz_yzx<T>, 3U>      yzx;
            lhs_attr lv<T, swz_yxx<T>, 3U>      yxx;
            lhs_attr lv<T, swz_zxy<T>, 3U>      zxy;
            lhs_attr lv<T, swz_zyx<T>, 3U>      zyx;
            lhs_attr lv<T, swz_zzy<T>, 3U>      zzy;

            lhs_attr lv<T, swz_xzy<T>, 3U>      xzy;
            lhs_attr lv<T, swz_yxz<T>, 3U>      yxz;

            rhs_attr rv<T, swz_xyzx<T>, 4U>     xyzx;
            struct
            {
                scalar_attr T x, y, z;
            };
        };
    };

    template<typename T> struct vec4_xyzw
    {
        union
        {
            lhs_attr lv<T, swz_xy<T>, 2U>   xy, lo;
            lhs_attr lv<T, swz_yx<T>, 2U>   yx;
            lhs_attr lv<T, swz_zw<T>, 2U>   zw, hi;
            lhs_attr lv<T, swz_xz<T>, 2U>   xz, even;
            lhs_attr lv<T, swz_yw<T>, 2U>   yw, odd;
            lhs_attr lv<T, swz_yz<T>, 2U>   yz;
            lhs_attr lv<T, swz_zx<T>, 2U>   zx;
            lhs_attr lv<T, swz_zy<T>, 2U>   zy;
            lhs_attr lv<T, swz_xyz<T>, 3U>  xyz;
            lhs_attr lv<T, swz_xzy<T>, 3U>  xzy;
            lhs_attr lv<T, swz_yzx<T>, 3U>  yzx;
            lhs_attr lv<T, swz_zxy<T>, 3U>  zxy;
            lhs_attr lv<T, swz_zyx<T>, 3U>  zyx;
            lhs_attr lv<T, swz_yzw<T>, 3U>  yzw;
            rhs_attr rv<T, swz_www<T>, 3U>  www;
            lhs_attr lv<T, swz_yxwz<T>, 3U> yxw;
            lhs_attr lv<T, swz_zwxy<T>, 3U> zwx;
            lhs_attr lv<T, swz_wzyx<T>, 3U> wzy;
            lhs_attr lv<T, swz_xyzw<T>, 4U> xyzw;
            lhs_attr lv<T, swz_yzxw<T>, 4U> yzxw;
            lhs_attr lv<T, swz_zxyw<T>, 4U> zxyw;
            lhs_attr lv<T, swz_wzyx<T>, 4U> wzyx;
            lhs_attr lv<T, swz_ywzx<T>, 4U> ywzx;
            lhs_attr lv<T, swz_xwyz<T>, 4U> xwyz;
            lhs_attr lv<T, swz_yxzw<T>, 4U> yxzw;
            lhs_attr lv<T, swz_wxyz<T>, 4U> wxyz;
            lhs_attr lv<T, swz_xzyw<T>, 4U> xzyw;
            lhs_attr lv<T, swz_wzxy<T>, 4U> wzxy;
            lhs_attr lv<T, swz_zwxy<T>, 4U> zwxy;
            lhs_attr lv<T, swz_zyxw<T>, 4U> zyxw;
            lhs_attr lv<T, swz_yxwz<T>, 4U> yxwz;
            lhs_attr lv<T, swz_yzwx<T>, 4U> yzwx;
            rhs_attr rv<T, swz_zwzw<T>, 4U> zwzw;
            rhs_attr rv<T, swz_zzww<T>, 4U> zzww;
            rhs_attr rv<T, swz_xxzz<T>, 4U> xxzz;
            rhs_attr rv<T, swz_yyww<T>, 4U> yyww;
            rhs_attr rv<T, swz_yzzx<T>, 4U> yzzx;
            rhs_attr rv<T, swz_yxxw<T>, 4U> yxxw;
            rhs_attr rv<T, swz_xxyy<T>, 4U> xxyy;
            rhs_attr rv<T, swz_xyxy<T>, 4U> xyxy;
            rhs_attr rv<T, swz_xyzz<T>, 4U> xyzz;
            rhs_attr rv<T, swz_yzxy<T>, 4U> yzxy;
            rhs_attr rv<T, swz_zxzx<T>, 4U> zxzx;
            rhs_attr rv<T, swz_zzyw<T>, 4U> zzyw;
            rhs_attr rv<T, swz_wwwx<T>, 4U> wwwx;
            rhs_attr rv<T, swz_wwwz<T>, 4U> wwwz;
            rhs_attr rv<T, swz_wwzz<T>, 4U> wwzz;
            rhs_attr rv<T, swz_xxxx<T>, 4U> xxxx;
            rhs_attr rv<T, swz_yyyy<T>, 4U> yyyy;
            rhs_attr rv<T, swz_zzzz<T>, 4U> zzzz;
            rhs_attr rv<T, swz_wwww<T>, 4U> wwww;
            struct
            {
                scalar_attr T x, y, z, w;
            };
        };
    };

    template<typename T> struct uninitialized : T
    {
        MATH_EMPTYINLINE uninitialized() {}
    };

    template<typename T, unsigned N> struct vec
    {
    };

    template<typename T> struct vec<T, 2U> : meta::uninitialized<meta::lv<T, meta::vec2_xy<T>, 2U> >
    {
        // implicit constructors
        MATH_EMPTYINLINE vec() {}
        MATH_FORCEINLINE vec(const vec &v)
        {
            this->xy = v.xy;
        }

        template<typename P> MATH_FORCEINLINE vec(const meta::rv<T, P, 2U> &v)
        {
            this->xy = v;
        }

        MATH_FORCEINLINE vec(T x)
        {
            this->xy = x;
        }

        // explicit constructors
        explicit MATH_FORCEINLINE vec(T x, T y)
        {
            this->x = x; this->y = y;
        }

        template<typename U, typename P> explicit MATH_FORCEINLINE vec(const meta::rv<U, P, 2U> &v)
        {
            this->xy.set((T)v.x, (T)v.y);
        }

        // assignment operators
        MATH_FORCEINLINE vec_attr vec &operator=(const vec &v) vec_attr
        {
            this->xy = v.xy;
            return *this;
        }

        MATH_FORCEINLINE vec_attr vec &operator=(T x) vec_attr
        {
            this->xy.set(x, x);
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr vec &operator=(const meta::rv<T, RHS, 2U> &rhs) vec_attr
        {
            this->xy = rhs;
            return *this;
        }

        template<typename R, typename RHS> MATH_FORCEINLINE vec_attr vec &operator=(const meta::rv<R, RHS, 2U> &rhs) vec_attr
        {
            this->xy.set((T)rhs.x, (T)rhs.y);
            return *this;
        }

        MATH_FORCEINLINE operator const T *() const
        {
            return &this->x;
        }

        MATH_FORCEINLINE operator T *() vec_attr
        {
            return (T*)&this->x;
        }
    };
    template<typename T> struct vec<T, 3U> : meta::uninitialized<meta::lv<T, meta::vec3_xyz<T>, 3U> >
    {
        // implicit constructors
        MATH_EMPTYINLINE vec() {}
        MATH_FORCEINLINE vec(const vec &u)
        {
            this->xyz = u.xyz;
        }

        template<typename P> MATH_FORCEINLINE vec(const meta::rv<T, P, 3U> &u)
        {
            this->xyz = u;
        }

        MATH_FORCEINLINE vec(T x)
        {
            this->xyz = x;
        }

        // explicit constructors
        explicit MATH_FORCEINLINE vec(T x, T y, T z)
        {
            this->x = x; this->y = y; this->z = z;
        }

        template<typename U, typename P> explicit MATH_FORCEINLINE vec(const meta::rv<U, P, 3U> &u)
        {
            this->xyz.set((T)u.x, (T)u.y, (T)u.z);
        }

        template<typename P> explicit MATH_FORCEINLINE vec(const meta::rv<T, P, 2U> &u, T z)
        {
            this->xy = u; this->z = z;
        }

        template<typename U, typename P> explicit MATH_FORCEINLINE vec(const meta::rv<U, P, 2U> &u, T z)
        {
            this->xyz.set((T)u.x, (T)u.y, z);
        }

        // assignment operators
        MATH_FORCEINLINE vec_attr vec &operator=(const vec &v) vec_attr
        {
            this->xyz = v.xyz;
            return *this;
        }

        MATH_FORCEINLINE vec_attr vec &operator=(T x) vec_attr
        {
            this->xyz.set(x, x, x);
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr vec &operator=(const meta::rv<T, RHS, 3U> &rhs) vec_attr
        {
            this->xyz = rhs;
            return *this;
        }

        template<typename R, typename RHS> MATH_FORCEINLINE vec_attr vec &operator=(const meta::rv<R, RHS, 3U> &rhs) vec_attr
        {
            this->xyz.set((T)rhs.x, (T)rhs.y, (T)rhs.z);
            return *this;
        }

        MATH_FORCEINLINE operator const T *() const
        {
            return &this->x;
        }

        MATH_FORCEINLINE operator T *() vec_attr
        {
            return (T*)&this->x;
        }
    };
    template<typename T> struct vec<T, 4U> : meta::uninitialized<meta::lv<T, meta::vec4_xyzw<T>, 4U> >
    {
        // implicit constructors
        MATH_EMPTYINLINE vec() {}
        MATH_FORCEINLINE vec(const vec &u)
        {
            this->xyzw = u.xyzw;
        }

        template<typename P> MATH_FORCEINLINE vec(const meta::rv<T, P, 4U> &u)
        {
            this->xyzw = u;
        }

        MATH_FORCEINLINE vec(T x)
        {
            this->xyzw = x;
        }

        // explicit constructors
        explicit MATH_FORCEINLINE vec(T x, T y, T z, T w)
        {
            this->x = x; this->y = y; this->z = z; this->w = w;
        }

        template<typename U, typename P> explicit MATH_FORCEINLINE vec(const meta::rv<U, P, 4U> &u)
        {
            this->xyzw.set((T)u.x, (T)u.y, (T)u.z, (T)u.w);
        }

        template<typename P> explicit MATH_FORCEINLINE vec(const meta::rv<T, P, 3U> &u, T w)
        {
            this->xyz = u; this->w = w;
        }

        template<typename U, typename P> explicit MATH_FORCEINLINE vec(const meta::rv<U, P, 3U> &u, T w)
        {
            this->xyzw.set((T)u.x, (T)u.y, (T)u.z, w);
        }

        template<typename P> explicit MATH_FORCEINLINE vec(const meta::rv<T, P, 2U> &v, T z, T w)
        {
            this->xy = v; this->z = z; this->w = w;
        }

        template<typename U, typename P> explicit MATH_FORCEINLINE vec(const meta::rv<U, P, 2U> &v, T z, T w)
        {
            this->xyzw.set((T)v.x, (T)v.y, z, w);
        }

        template<typename P, typename Q> explicit MATH_FORCEINLINE vec(const meta::rv<T, P, 2U> &u, const meta::rv<T, Q, 2U> &v)
        {
            this->xy = u; this->zw = v;
        }

        template<typename U, typename P, typename V, typename Q> explicit MATH_FORCEINLINE vec(const meta::rv<U, P, 2U> &u, const meta::rv<V, Q, 2U> &v)
        {
            this->xyzw.set((T)u.x, (T)u.y, (T)v.x, (T)v.y);
        }

        // assignment operators
        MATH_FORCEINLINE vec_attr vec &operator=(const vec &rhs) vec_attr
        {
            this->xyzw = rhs.xyzw;
            return *this;
        }

        MATH_FORCEINLINE vec_attr vec &operator=(T x) vec_attr
        {
            this->xyzw.set(x, x, x, x);
            return *this;
        }

        template<typename RHS> MATH_FORCEINLINE vec_attr vec &operator=(const meta::rv<T, RHS, 4U> &rhs) vec_attr
        {
            this->xyzw = rhs;
            return *this;
        }

        template<typename R, typename RHS> MATH_FORCEINLINE vec_attr vec &operator=(const meta::rv<R, RHS, 4U> &rhs) vec_attr
        {
            this->xyzw.set((T)rhs.x, (T)rhs.y, (T)rhs.z, (T)rhs.w);
            return *this;
        }

        MATH_FORCEINLINE operator const T *() const
        {
            return &this->x;
        }

        MATH_FORCEINLINE operator T *() vec_attr
        {
            return (T*)&this->x;
        }
    };

#   define DECL_VEC2(name, type) \
        typedef struct _##name : meta::vec<type, 2U> \
        { \
            typedef meta::vec<type, 2U>   base; \
            MATH_EMPTYINLINE _##name() : base() {} \
            MATH_FORCEINLINE _##name(const _##name &u) : base(u) {} \
            MATH_FORCEINLINE _##name(type x) : base(x) {} \
            MATH_FORCEINLINE _##name(zero_t) : base(type(0)) {} \
            template<typename P> MATH_FORCEINLINE _##name(const meta::rv<type, P, 2U> &u) : base(u) {} \
            explicit MATH_FORCEINLINE _##name(type x, type y) : base(x, y) {} \
            template<typename U, typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<U, P, 2U> &u) : base(u) {} \
            MATH_FORCEINLINE vec_attr _##name &operator=(const _##name &rhs) vec_attr { base::operator=(rhs); return *this; } \
            MATH_FORCEINLINE vec_attr _##name &operator=(type x) vec_attr { base::operator=(x); return *this; } \
            MATH_FORCEINLINE _##name *operator&() const { return (_##name *) this; } \
        } vec_attr name; \
    //template<> struct is_pod<name> { enum { value = true }; };

#   define DECL_VEC3(name, type) \
        typedef struct _##name : meta::vec<type, 3U> \
        { \
            typedef meta::vec<type, 3U>   base; \
            MATH_EMPTYINLINE _##name() : base() {} \
            MATH_FORCEINLINE _##name(const _##name &u) : base(u) {} \
            MATH_FORCEINLINE _##name(type x) : base(x) {} \
            MATH_FORCEINLINE _##name(zero_t) : base(type(0)) {} \
            template<typename P> MATH_FORCEINLINE _##name(const meta::rv<type, P, 3U> &u) : base(u) {} \
            explicit MATH_FORCEINLINE _##name(type x, type y, type z) : base(x, y, z) {} \
            template<typename U, typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<U, P, 3U> &u) : base(u) {} \
            template<typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<type, P, 2U> &u, type z) : base(u, z) {} \
            template<typename U, typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<U, P, 2U> &u, type z) : base(u, z) {} \
            MATH_FORCEINLINE vec_attr _##name &operator=(const _##name &rhs) vec_attr { base::operator=(rhs); return *this; } \
            MATH_FORCEINLINE vec_attr _##name &operator=(type x) vec_attr { base::operator=(x); return *this; } \
            MATH_FORCEINLINE _##name *operator&() const { return (_##name *) this; } \
        } vec_attr name; \
    //template<> struct is_pod<name> { enum { value = true }; };

#   define DECL_VEC4(name, type) \
        typedef struct _##name : meta::vec<type, 4U> \
        { \
            typedef meta::vec<type, 4U>   base; \
            MATH_EMPTYINLINE _##name() : base() {} \
            MATH_FORCEINLINE _##name(const _##name &u) : base(u) {} \
            MATH_FORCEINLINE _##name(type x) : base(x) {} \
            MATH_FORCEINLINE _##name(zero_t) : base(type(0)) {} \
            template<typename P> MATH_FORCEINLINE _##name(const meta::rv<type, P, 4U> &u) : base(u) {} \
            explicit MATH_FORCEINLINE _##name(type x, type y, type z, type w) : base(x, y, z, w) {} \
            template<typename U, typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<U, P, 4U> &u) : base(u) {} \
            template<typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<type, P, 3U> &u, type w) : base(u, w) {} \
            template<typename U, typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<U, P, 3U> &u, type w) : base(u, w) {} \
            template<typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<type, P, 2U> &u, type z, type w) : base(u, z, w) {} \
            template<typename U, typename P> explicit MATH_FORCEINLINE _##name(const meta::rv<U, P, 2U> &u, type z, type w) : base(u, z, w) {} \
            template<typename P, typename Q> explicit MATH_FORCEINLINE _##name(const meta::rv<type, P, 2U> &u, const meta::rv<type, Q, 2U> &v) : base(u, v) {} \
            template<typename U, typename P, typename V, typename Q> explicit MATH_FORCEINLINE _##name(const meta::rv<U, P, 2U> &u, const meta::rv<V, Q, 2U> &v) : base(u, v) {} \
            MATH_FORCEINLINE vec_attr _##name &operator=(const _##name &rhs) vec_attr { base::operator=(rhs); return *this; } \
            MATH_FORCEINLINE vec_attr _##name &operator=(type x) vec_attr { base::operator=(x); return *this; } \
            MATH_FORCEINLINE _##name *operator&() const { return (_##name *) this; } \
        } vec_attr name; \
    //template<> struct is_pod<name> { enum { value = true }; };
}
}

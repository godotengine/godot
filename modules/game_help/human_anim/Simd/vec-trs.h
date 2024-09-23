#pragma once

#include "./vec-types.h"
#include "./vec-quat.h"
#include "./vec-matrix.h"
#include "core/math/transform_3d.h"
#include "core/variant/dictionary.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

namespace math
{
    struct trsX
    {
        MATH_EMPTYINLINE trsX() {}
        MATH_FORCEINLINE trsX(trsX const& x)
        { t = x.t; q = x.q; s = x.s; }

        MATH_FORCEINLINE trsX &operator=(trsX const& x)
        { t = x.t; q = x.q; s = x.s; return *this; }

        MATH_FORCEINLINE trsX(float3 const& t, float4 const& q, float3 const& s)
        { this->t = t; this->q = q; this->s = s; }

        float3 t;
        float4 q;
        float3 s;
        MATH_FORCEINLINE Transform3D toTransform() const {
            Basis basis = Basis(Quaternion(q.x, q.y, q.z, q.w), Vector3(s.x, s.y, s.z));
            return Transform3D(basis, Vector3(t.x,t.y,t.z));
        }

        static MATH_FORCEINLINE trsX fromTransform(Transform3D const& t)
        {
            trsX r;
            Vector3 origin = t.origin;
            r.t = float3(origin.x, origin.y, origin.z);
            Quaternion q = t.basis.get_rotation_quaternion();
            r.q = float4(q.x, q.y, q.z, q.w);
            Vector3 s = t.basis.get_scale();
            r.s = float3(s.x, s.y, s.z);
            return r;
        }

        void load(const Dictionary & aDict) {
            Vector3 v3 = aDict["t"];
            t = float3(v3.x, v3.y, v3.z);
            Quaternion tq = aDict["q"];
            q = float4(tq.x, tq.y, tq.z, tq.w);
            v3 = aDict["s"];
            s = float3(v3.x, v3.y, v3.z);
        }

        void save(Dictionary & aDict) const {
            Vector3 v3 = Vector3(t.x,t.y,t.z);
            aDict["t"] = v3;
            Quaternion tq = Quaternion(q.x, q.y, q.z, q.w);
            aDict["q"] = tq;
            v3 = Vector3(s.x,s.y,s.z);
            aDict["s"] = v3;
        }
    };

    static MATH_FORCEINLINE bool operator==(trsX const& l, trsX const& r)
    {
        return all(l.t == r.t) && all(l.q == r.q) && all(l.s == r.s);
    }

    static MATH_FORCEINLINE trsX trsIdentity()
    {
        return trsX(float3(ZERO), quatIdentity(), float3(1.f));
    }

    static MATH_FORCEINLINE float3 mul(trsX const& x, float3 const& v)
    {
        return x.t + quatMulVec(x.q, v * x.s);
    }

    static MATH_FORCEINLINE float3 invMul(trsX const& x, float3 const& v)
    {
        return quatMulVec(quatConj(x.q), v - x.t) * inverseScale(x.s);
    }

// trsX is used to represent local transform. It should not be used by mul, invMul or mulInv. use affineX or rigid depending on scale type
// @TODO remove
    static MATH_FORCEINLINE trsX mul(trsX const& a, trsX const& b)
    {
        return trsX(mul(a, b.t), quatMul(a.q, b.q), a.s * b.s);
    }

// trsX is used to represent local transform. It should not be used by mul, invMul or mulInv. use affineX or rigid depending on scale type
// @TODO remove
    static MATH_FORCEINLINE trsX invMul(trsX const& a, trsX const& b)
    {
        return trsX(invMul(a, b.t), quatMul(quatConj(a.q), b.q), b.s * inverseScale(a.s));
    }

// trsX is used to represent local transform. It should not be used by mul, invMul or mulInv. use affineX or rigidX depending on scale type
// @TODO remove
    static MATH_FORCEINLINE trsX mulInv(trsX const& a, trsX const& b)
    {
        const float4 qinv = quatConj(b.q);
        const float3 sinv = inverseScale(a.s);

        return trsX(mul(a, quatMulVec(qinv, -b.t) * sinv), quatMul(a.q, qinv), a.s * sinv);
    }

// trsX is used to represent local transform. NS versions of mul and invMul should use rigidX instead
// @TODO remove
    static MATH_FORCEINLINE float3 trsMulVecNS(trsX const& x, float3 const& v)
    {
        return x.t + quatMulVec(x.q, v);
    }

// trsX is used to represent local transform. NS versions of mul and invMul should use rigidX instead
// @TODO remove
    static MATH_FORCEINLINE float3 trsInvMulVecNS(trsX const& x, float3 const& v)
    {
        return quatMulVec(quatConj(x.q), v - x.t);
    }

// trsX is used to represent local transform. NS versions of mul and invMul should use rigidX instead
// @TODO remove
    static MATH_FORCEINLINE trsX trsMulNS(trsX const& a, trsX const& b)
    {
        return trsX(trsMulVecNS(a, b.t), quatMul(a.q, b.q), float3(1.f));
    }

// trsX is used to represent local transform. NS versions of mul and invMul should use rigidX instead
// @TODO remove
    static MATH_FORCEINLINE trsX trsInvMulNS(trsX const& a, trsX const& b)
    {
        return trsX(trsInvMulVecNS(a, b.t), quatMul(quatConj(a.q), b.q), float3(1.f));
    }

    static MATH_FORCEINLINE trsX trsBlend(trsX const &a, trsX const &b, float1 const& w)
    {
        return trsX(lerp(a.t, b.t, w), quatLerp(a.q, b.q, w), lerp(a.s, b.s, w));
    }

    static MATH_FORCEINLINE trsX trsWeightNS(trsX const& x, float1 const& w)
    {
        return trsX(x.t * w, quatWeight(x.q, w), float3(1.f));
    }

    static MATH_FORCEINLINE trsX mirrorX(trsX const& x)
    {
        trsX ret = x;
        ret.t = mirrorX(ret.t);
        ret.q *= float4(1.f, -1.f, -1.f, 1.f);
        return ret;
    }
}

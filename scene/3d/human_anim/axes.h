#pragma once

#include "./defs.h"
#include "./types.h"
#include "./Simd/vec-math.h"
#include "./Simd/vec-quat.h"
#include "core/variant/dictionary.h"


namespace math
{
    struct Limit
    {

        float3 m_Min; // m_Min > 0 -> free, m_Min == 0 -> lock, m_Min < 0 -> limit
        float3 m_Max; // m_Max < 0 -> free, m_Max == 0 -> lock, m_Max > 0 -> limit

        inline Limit() : m_Min(1.f), m_Max(-1.f) {}
        inline Limit(float3 const& aMin, float3 const& aMax, float aRange) { m_Min = aMin; m_Max = aMax; }

        Limit& operator=(Limit const& other)
        {
            m_Min = other.m_Min;
            m_Max = other.m_Max;
            return *this;
        }

    };

    enum AxesType { kFull, kZYRoll, kRollZY, kEulerXYZ };

    struct SetupAxesInfo
    {
        float m_PreQ[4];
        float m_MainAxis[4];
        float m_Min[4];
        float m_Max[4];
        float m_Sgn[4];
        AxesType m_Type;
        human_anim::int32_t m_ForceAxis;
    };

    struct Axes
    {

        float4 m_PreQ;
        float4 m_PostQ;
        float3 m_Sgn;
        Limit m_Limit;
        float m_Length;
        human_anim::uint32_t m_Type; // AxesType

        inline Axes() : m_PreQ(quatIdentity()), m_PostQ(quatIdentity()), m_Sgn(1.f), m_Length(1.f), m_Type(kEulerXYZ) {}
        inline Axes(float4 const& aPreQ, float4 const& aPostQ, float3 const& aSgn, float const &aLength, AxesType const& aType) { m_PreQ = aPreQ; m_PostQ = aPostQ; m_Sgn = aSgn; m_Length = aLength; m_Type = aType; }

        Axes& operator=(Axes const& other)
        {
            m_PreQ = other.m_PreQ;
            m_PostQ = other.m_PostQ;
            m_Sgn = other.m_Sgn;
            m_Limit = other.m_Limit;
            m_Length = other.m_Length;
            m_Type = other.m_Type;
            return *this;
        }
        void load(const Dictionary & aDict) {
            Quaternion q = aDict["preQ"];
            m_PreQ = float4(q.x, q.y, q.z, q.w);

            q = aDict["postQ"];
            m_PostQ = float4(q.x, q.y, q.z, q.w);

            Vector3 sgn = aDict["sgn"];
            m_Sgn = float3(sgn.x, sgn.y, sgn.z);

            Vector3 lim = aDict["limit_min"];
            m_Limit.m_Min = float3(lim.x, lim.y, lim.z);

            lim = aDict["limit_max"];
            m_Limit.m_Max = float3(lim.x, lim.y, lim.z);

            m_Length = aDict["length"];
            m_Type = (AxesType)(int)aDict["type"];
        }

        void save(Dictionary & aDict) {
            aDict["preQ"] = Quaternion(m_PreQ.x, m_PreQ.y, m_PreQ.z, m_PreQ.w);
            aDict["postQ"] = Quaternion(m_PostQ.x, m_PostQ.y, m_PostQ.z, m_PostQ.w) ;
            aDict["sgn"] = Vector3(m_Sgn.x, m_Sgn.y, m_Sgn.z);
            aDict["limit_min"] = Vector3(m_Limit.m_Min.x, m_Limit.m_Min.y, m_Limit.m_Min.z);
            aDict["limit_max"] = Vector3(m_Limit.m_Max.x, m_Limit.m_Max.y, m_Limit.m_Max.z);
            aDict["length"] = m_Length;
            aDict["type"] = (int)m_Type;
        }

    };

    static inline float3 doubleAtan(const float3& v)
    {
        // using std atan since atan approximation is not precise enough for the retargeter
        // doubleAtan is only used by retargeter at import
        return float1(2.0f) * float3(std::atan(v.x), std::atan(v.y), std::atan(v.z));
    }

    inline float halfTan(const float& a)
    {
        return tan(clamp(0.5f * chgsign(fmod(abs(a) + math::pi(), 2.0f * math::pi()) - math::pi(), a), -math::pi_over_two() + epsilon_radian(), math::pi_over_two() - epsilon_radian()));
    }

    inline float3 halfTan(const float3& a)
    {
        return tan(clamp(0.5f * chgsign(fmod(abs(a) + math::pi(), 2.0f * math::pi()) - math::pi(), a), -math::pi_over_two() + epsilon_radian(), math::pi_over_two() - epsilon_radian()));
    }

    inline float LimitProject(float min, float max, float v)
    {
        float i = min < 0 ? -v / min : min > 0 ? v : 0;
        float a = max > 0 ? +v / max : max < 0 ? v : 0;
        return v < 0 ? i : a;
    }

    inline float LimitUnproject(float min, float max, float v)
    {
        float i = min < 0 ? -v * min : min > 0 ? v : 0;
        float a = max > 0 ? +v * max : max < 0 ? v : 0;
        return v < 0 ?  i : a;
    }

    inline float3 LimitProject(Limit const& l, float3 const& v)
    {
        const float3 min = select(select(float3(ZERO), v, l.m_Min > float1(ZERO)), -v / l.m_Min, l.m_Min < float1(ZERO));
        const float3 max = select(select(float3(ZERO), v, l.m_Max < float1(ZERO)), +v / l.m_Max, l.m_Max > float1(ZERO));
        return select(max, min, v < float3(ZERO));
    }

    inline float3 LimitUnproject(Limit const& l, float3 const& v)
    {
        const float3 min = select(select(float3(ZERO), v, l.m_Min > float1(ZERO)), -v * l.m_Min, l.m_Min < float1(ZERO));
        const float3 max = select(select(float3(ZERO), v, l.m_Max < float1(ZERO)), +v * l.m_Max, l.m_Max > float1(ZERO));
        return select(max, min, v < float3(ZERO));
    }

    inline float4 AxesProject(Axes const& a, float4 const& q)
    {
        return normalize(quatMul(quatConj(a.m_PreQ), quatMul(q, a.m_PostQ)));
    }

    inline float4 AxesUnproject(Axes const& a, float4 const& q)
    {
        return normalize(quatMul(a.m_PreQ, quatMul(q, quatConj(a.m_PostQ))));
    }

    inline float3 ToAxes(Axes const& a, float4 const& q)
    {
        const float4 qp = AxesProject(a, q);
        float3 xyz;
        switch (a.m_Type)
        {
            case kEulerXYZ: xyz = LimitProject(a.m_Limit, quatToEuler(qp)); break;
            case kZYRoll: xyz = LimitProject(a.m_Limit, doubleAtan(chgsign(quat2ZYRoll(qp), a.m_Sgn))); break;
            case kRollZY: xyz = LimitProject(a.m_Limit, doubleAtan(chgsign(quat2RollZY(qp), a.m_Sgn))); break;
            default:      xyz = LimitProject(a.m_Limit, doubleAtan(chgsign(quat2Qtan(qp), a.m_Sgn))); break;
        }
        return xyz;
    }

    inline float4 FromAxes(Axes const& a, float3 const& uvw)
    {
        float4 q;
        switch (a.m_Type)
        {
            case kEulerXYZ: q = eulerToQuat(uvw); break;
            case kZYRoll:   q = ZYRoll2Quat(chgsign(halfTan(LimitUnproject(a.m_Limit, uvw)), a.m_Sgn)); break;
            case kRollZY:   q = RollZY2Quat(chgsign(halfTan(LimitUnproject(a.m_Limit, uvw)), a.m_Sgn)); break;
            default:        q = qtan2Quat(chgsign(halfTan(LimitUnproject(a.m_Limit, uvw)), a.m_Sgn)); break;
        }

        return AxesUnproject(a, q);
    }
}

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
        // 使用标准的 atan 函数，因为近似的 atan 精度不足以满足重定向器的需求
        // doubleAtan 仅在导入时由重定向器使用
        return float1(2.0f) * float3(std::atan(v.x), std::atan(v.y), std::atan(v.z));
    }

    inline float halfTan(const float& a)
    {
        // 计算输入值的半切线，限制在[-π/2, π/2]范围内
        return tan(clamp(0.5f * chgsign(fmod(abs(a) + math::pi(), 2.0f * math::pi()) - math::pi(), a), 
                        -math::pi_over_two() + epsilon_radian(), 
                        math::pi_over_two() - epsilon_radian()));
    }

    inline float3 halfTan(const float3& a)
    {
        // 计算向量的每个分量的半切线，限制在[-π/2, π/2]范围内
        return tan(clamp(0.5f * chgsign(fmod(abs(a) + math::pi(), 2.0f * math::pi()) - math::pi(), a), 
                        -math::pi_over_two() + epsilon_radian(), 
                        math::pi_over_two() - epsilon_radian()));
    }

    inline float LimitProject(float min, float max, float v)
    {
        // 将值v限制在[min, max]范围内，返回适当的比例
        float i = min < 0 ? -v / min : min > 0 ? v : 0;
        float a = max > 0 ? +v / max : max < 0 ? v : 0;
        return v < 0 ? i : a;
    }

    inline float LimitUnproject(float min, float max, float v)
    {
        // 根据最小值和最大值反向限制v，返回适当的值
        float i = min < 0 ? -v * min : min > 0 ? v : 0;
        float a = max > 0 ? +v * max : max < 0 ? v : 0;
        return v < 0 ?  i : a;
    }

    inline float3 LimitProject(Limit const& l, float3 const& v)
    {
        // 对向量v进行限制投影，依据给定的限制条件
        const float3 min = select(select(float3(ZERO), v, l.m_Min > float1(ZERO)), -v / l.m_Min, l.m_Min < float1(ZERO));
        const float3 max = select(select(float3(ZERO), v, l.m_Max < float1(ZERO)), +v / l.m_Max, l.m_Max > float1(ZERO));
        return select(max, min, v < float3(ZERO));
    }

    inline float3 LimitUnproject(Limit const& l, float3 const& v)
    {
        // 对向量v进行限制反投影，依据给定的限制条件
        const float3 min = select(select(float3(ZERO), v, l.m_Min > float1(ZERO)), -v * l.m_Min, l.m_Min < float1(ZERO));
        const float3 max = select(select(float3(ZERO), v, l.m_Max < float1(ZERO)), +v * l.m_Max, l.m_Max > float1(ZERO));
        return select(max, min, v < float3(ZERO));
    }

    inline float4 AxesProject(Axes const& a, float4 const& q)
    {
        // 将四元数q投影到给定的轴上
        return normalize(quatMul(quatConj(a.m_PreQ), quatMul(q, a.m_PostQ)));
    }

    inline float4 AxesUnproject(Axes const& a, float4 const& q)
    {
        // 将四元数q从给定的轴上反投影
        return normalize(quatMul(a.m_PreQ, quatMul(q, quatConj(a.m_PostQ))));
    }

    inline float3 ToAxes(Axes const& a, float4 const& q)
    {
        // 将四元数q转换为轴表示，并限制在轴的范围内
        const float4 qp = AxesProject(a, q);
        float3 xyz;
        switch (a.m_Type)
        {
            case kEulerXYZ: xyz = LimitProject(a.m_Limit, quatToEuler(qp)); break; // 欧拉角
            case kZYRoll:   xyz = LimitProject(a.m_Limit, doubleAtan(chgsign(quat2ZYRoll(qp), a.m_Sgn))); break; // ZY滚动
            case kRollZY:   xyz = LimitProject(a.m_Limit, doubleAtan(chgsign(quat2RollZY(qp), a.m_Sgn))); break; // 滚动ZY
            default:        xyz = LimitProject(a.m_Limit, doubleAtan(chgsign(quat2Qtan(qp), a.m_Sgn))); break; // 默认四元数切线
        }
        return xyz;
    }

    inline float4 FromAxes(Axes const& a, float3 const& uvw)
    {
        // 从轴表示uv为四元数q
        float4 q;
        switch (a.m_Type)
        {
            case kEulerXYZ: q = eulerToQuat(uvw); break; // 欧拉角转四元数
            case kZYRoll:   q = ZYRoll2Quat(chgsign(halfTan(LimitUnproject(a.m_Limit, uvw)), a.m_Sgn)); break; // ZY滚动转四元数
            case kRollZY:   q = RollZY2Quat(chgsign(halfTan(LimitUnproject(a.m_Limit, uvw)), a.m_Sgn)); break; // 滚动ZY转四元数
            default:        q = qtan2Quat(chgsign(halfTan(LimitUnproject(a.m_Limit, uvw)), a.m_Sgn)); break; // 默认切线转四元数
        }

        return AxesUnproject(a, q); // 从给定的轴上反投影四元数q
}

}

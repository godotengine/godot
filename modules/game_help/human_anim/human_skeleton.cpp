#include "./human_skeleton.h"
#include "./axes.h"

namespace human_anim
{
namespace skeleton
{
    HumanSkeleton*     CreateSkeleton(int32_t aNodeCount, int32_t aAxesCount, RuntimeBaseAllocator& arAlloc)
    {
        HumanSkeleton* skeleton = memnew(HumanSkeleton);

        skeleton->m_Node.resize(aNodeCount);

		skeleton->m_AxesArray.resize(aAxesCount);

        return skeleton;
    }

    template<typename transformType> transformType Identity();

    template<> math::trsX Identity<math::trsX>()
    {
        return math::trsIdentity();
    }

    template<> math::affineX Identity<math::affineX>()
    {
        return math::affineIdentity();
    }




    template<typename transformType>
    SkeletonPoseT<transformType> *CreateSkeletonPose(HumanSkeleton const* apSkeleton, RuntimeBaseAllocator& arAlloc)
    {
        SkeletonPoseT<transformType>* skeletonPose = memnew(SkeletonPoseT<transformType>);

        skeletonPose->m_Count = apSkeleton->m_Node.size();
        skeletonPose->m_X.resize(apSkeleton->m_Node.size());

        return skeletonPose;
    }

    void DestroySkeleton(HumanSkeleton* apSkeleton, RuntimeBaseAllocator& arAlloc)
    {
        if (apSkeleton)
        {
			memdelete(apSkeleton);
        }
    }

    template<typename transformType>
    void DestroySkeletonPose(SkeletonPoseT<transformType>* apSkeletonPose, RuntimeBaseAllocator& arAlloc)
    {
        if (apSkeletonPose)
        {
			memdelete(apSkeletonPose);
        }
    }

    SkeletonMask* CreateSkeletonMask(uint32_t aNodeCount, SkeletonMaskElement const* elements, RuntimeBaseAllocator& arAlloc)
    {
        SkeletonMask* skeletonMask = memnew(SkeletonMask);

        skeletonMask->m_Count = aNodeCount;
        skeletonMask->m_Data.resize(aNodeCount);

        memcpy(skeletonMask->m_Data.ptr(), elements, sizeof(SkeletonMaskElement) * aNodeCount);

        return skeletonMask;
    }

    void DestroySkeletonMask(SkeletonMask* skeletonMask, RuntimeBaseAllocator& arAlloc)
    {
        if (skeletonMask)
        {
			memdelete(skeletonMask);
        }
    }

    void SkeletonCopy(HumanSkeleton const* apSrc, HumanSkeleton* apDst)
    {
        for (uint32_t nodeIter = 0; nodeIter < apDst->m_Node.size(); nodeIter++)
        {
            apDst->m_Node[nodeIter] = apSrc->m_Node[nodeIter];
        }

        for (uint32_t axesIter = 0; axesIter < apDst->m_AxesArray.size(); axesIter++)
        {
            apDst->m_AxesArray[axesIter] = apSrc->m_AxesArray[axesIter];
        }
    }

    template<typename transformTypeFrom, typename transformTypeTo>
    void SkeletonPoseCopy(SkeletonPoseT<transformTypeFrom> const* apFromPose, SkeletonPoseT<transformTypeTo>* apToPose)
    {
        uint32_t nodeCount = std::min(apFromPose->m_Count, apToPose->m_Count);
        transformTypeFrom const *from = apFromPose->m_X.ptr();
        transformTypeTo *to = apToPose->m_X.ptr();

        for (uint32_t nodeIter = 0; nodeIter < nodeCount; nodeIter++)
        {
            math::convert(from[nodeIter], to[nodeIter]);
        }
    }

    void SkeletonBuildReverseIndexArray(int32_t *reverseIndexArray, int32_t const*indexArray, HumanSkeleton const* apSrcSkeleton, HumanSkeleton const* apDstSkeleton)
    {
        for (uint32_t dstIter = 0; dstIter < apDstSkeleton->m_Node.size(); dstIter++)
        {
            reverseIndexArray[dstIter] = -1;
        }

        for (uint32_t srcIter = 0; srcIter < apSrcSkeleton->m_Node.size(); srcIter++)
        {
            if (indexArray[srcIter] != -1)
            {
                reverseIndexArray[indexArray[srcIter]] = srcIter;
            }
        }
    }

    void SkeletonPoseSetDirty(HumanSkeleton const* apSkeleton, uint32_t* apSkeletonPoseMask, int aIndex, int aStopIndex, uint32_t aMask)
    {
        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        if (parentIndex != -1)
        {
            if (aIndex != aStopIndex)
            {
                SkeletonPoseSetDirty(apSkeleton, apSkeletonPoseMask, parentIndex, aStopIndex, aMask);
            }
        }

        apSkeletonPoseMask[aIndex] = apSkeletonPoseMask[aIndex] | aMask;
    }

    template<typename transformType>
    void SkeletonPoseComputeGlobal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apLocalPose, SkeletonPoseT<transformType>* apGlobalPose)
    {
        Node const *node = apSkeleton->m_Node.ptr();
        transformType const *local = apLocalPose->m_X.ptr();
        transformType *global = apGlobalPose->m_X.ptr();

        global[0] = local[0];

        for (uint32_t nodeIter = 1; nodeIter < apSkeleton->m_Node.size(); nodeIter++)
        {
            global[nodeIter] = math::mul(global[node[nodeIter].m_ParentId], local[nodeIter]);
        }
    }

    // computes a global pose from a local pose for part of the skeleton starting at aIndex (child) to aStopIndex (ancestor)
    template<typename transformType>
    void SkeletonPoseComputeGlobal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apLocalPose, SkeletonPoseT<transformType>* apGlobalPose, int aIndex, int aStopIndex)
    {
        transformType const *local = apLocalPose->m_X.ptr();
        transformType *global = apGlobalPose->m_X.ptr();

        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        if (parentIndex != -1)
        {
            if (aIndex != aStopIndex)
            {
                SkeletonPoseComputeGlobal(apSkeleton, apLocalPose, apGlobalPose, parentIndex, aStopIndex);
            }

            global[aIndex] = math::mul(global[parentIndex], local[aIndex]);
        }
        else
        {
            global[aIndex] = local[aIndex];
        }
    }

    void SkeletonPoseComputeGlobalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseLocal, SkeletonPose* apSkeletonPoseGlobal)
    {
        apSkeletonPoseGlobal->m_X[0].q = apSkeletonPoseLocal->m_X[0].q;

        uint32_t i;
        for (i = 1; i < apSkeleton->m_Node.size(); i++)
        {
            apSkeletonPoseGlobal->m_X[i].q = math::normalize(math::quatMul(apSkeletonPoseGlobal->m_X[apSkeleton->m_Node[i].m_ParentId].q, apSkeletonPoseLocal->m_X[i].q));
        }
    }

    void SkeletonPoseComputeGlobalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseLocal, SkeletonPose* apSkeletonPoseGlobal, int aIndex, int aStopIndex)
    {
        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        if (parentIndex != -1)
        {
            if (aIndex != aStopIndex)
            {
                SkeletonPoseComputeGlobalQ(apSkeleton, apSkeletonPoseLocal, apSkeletonPoseGlobal, parentIndex, aStopIndex);
            }

            apSkeletonPoseGlobal->m_X[aIndex].q = math::normalize(math::quatMul(apSkeletonPoseGlobal->m_X[parentIndex].q, apSkeletonPoseLocal->m_X[aIndex].q));
        }
        else
        {
            apSkeletonPoseGlobal->m_X[aIndex].q = apSkeletonPoseLocal->m_X[aIndex].q;
        }
    }

    template<typename transformType>
    void SkeletonPoseComputeLocal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apGlobalPose, SkeletonPoseT<transformType>* apLocalPose)
    {
        Node const *node = apSkeleton->m_Node.ptr();
        transformType const *global = apGlobalPose->m_X.ptr();
        transformType *local = apLocalPose->m_X.ptr();

        for (uint32_t nodeIter = 1; nodeIter < apSkeleton->m_Node.size(); nodeIter++)
        {
            local[nodeIter] = math::invMul(global[node[nodeIter].m_ParentId], global[nodeIter]);
        }

        local[0] = global[0];
    }

    template<typename transformType>
    void SkeletonPoseComputeLocal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apGlobalPose, SkeletonPoseT<transformType>* apLocalPose, int aIndex, int aStopIndex)
    {
        transformType const *global = apGlobalPose->m_X.ptr();
        transformType *local = apLocalPose->m_X.ptr();

        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        if (parentIndex != -1)
        {
            local[aIndex] = math::invMul(global[parentIndex], global[aIndex]);

            if (aIndex != aStopIndex)
            {
                SkeletonPoseComputeLocal(apSkeleton, apGlobalPose, apLocalPose, parentIndex, aStopIndex);
            }
        }
        else
        {
            local[aIndex] = global[aIndex];
        }
    }

    void SkeletonPoseComputeLocalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseGlobal, SkeletonPose* apSkeletonPoseLocal)
    {
        uint32_t i;
        for (i = apSkeleton->m_Node.size() - 1; i > 0; i--)
        {
            apSkeletonPoseLocal->m_X[i].q = math::normalize(math::quatMul(math::quatConj(apSkeletonPoseGlobal->m_X[apSkeleton->m_Node[i].m_ParentId].q), apSkeletonPoseGlobal->m_X[i].q));
        }

        apSkeletonPoseLocal->m_X[0].q = apSkeletonPoseGlobal->m_X[0].q;
    }

    void SkeletonPoseComputeLocalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseGlobal, SkeletonPose* apSkeletonPoseLocal, int aIndex, int aStopIndex)
    {
        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        if (parentIndex != -1)
        {
            apSkeletonPoseLocal->m_X[aIndex].q = math::normalize(math::quatMul(math::quatConj(apSkeletonPoseGlobal->m_X[parentIndex].q), apSkeletonPoseGlobal->m_X[aIndex].q));

            if (aIndex != aStopIndex)
            {
                SkeletonPoseComputeLocalQ(apSkeleton, apSkeletonPoseGlobal, apSkeletonPoseLocal, parentIndex, aStopIndex);
            }
        }
        else
        {
            apSkeletonPoseLocal->m_X[aIndex].q = apSkeletonPoseGlobal->m_X[aIndex].q;
        }
    }

    math::trsX SkeletonGetGlobalX(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex)
    {
        math::trsX globalX = apSkeletonPose->m_X[aIndex];
        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        while (parentIndex >= 0)
        {
            globalX = math::mul(apSkeletonPose->m_X[parentIndex], globalX);
            parentIndex = apSkeleton->m_Node[parentIndex].m_ParentId;
        }

        return globalX;
    }

    math::float3 SkeletonGetGlobalPosition(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex)
    {
        math::float3 globalT = apSkeletonPose->m_X[aIndex].t;
        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        while (parentIndex >= 0)
        {
            globalT = math::mul(apSkeletonPose->m_X[parentIndex], globalT);
            parentIndex = apSkeleton->m_Node[parentIndex].m_ParentId;
        }

        return globalT;
    }

    math::float4 SkeletonGetGlobalRotation(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex)
    {
        math::float4 globalR = apSkeletonPose->m_X[aIndex].q;
        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        while (parentIndex >= 0)
        {
            globalR = math::scaleMulQuat(apSkeletonPose->m_X[parentIndex].s, globalR);
            globalR = math::quatMul(apSkeletonPose->m_X[parentIndex].q, globalR);

            parentIndex = apSkeleton->m_Node[parentIndex].m_ParentId;
        }

        return globalR;
    }

    void SkeletonGetGlobalPositionAndRotation(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex, math::float3 &position, math::float4 &rotation)
    {
        position = apSkeletonPose->m_X[aIndex].t;
        rotation = apSkeletonPose->m_X[aIndex].q;
        int parentIndex = apSkeleton->m_Node[aIndex].m_ParentId;

        while (parentIndex >= 0)
        {
            position = math::mul(apSkeletonPose->m_X[parentIndex], position);
            rotation = math::scaleMulQuat(apSkeletonPose->m_X[parentIndex].s, rotation);
            rotation = math::quatMul(apSkeletonPose->m_X[parentIndex].q, rotation);

            parentIndex = apSkeleton->m_Node[parentIndex].m_ParentId;
        }
    }

    void SkeletonInverseTransformPosition(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex, math::float3& p)
    {
        if (aIndex > 0)
        {
            SkeletonInverseTransformPosition(apSkeleton, apSkeletonPose, apSkeleton->m_Node[aIndex].m_ParentId, p);
        }

        p = math::invMul(apSkeletonPose->m_X[aIndex], p);
    }

    void SkeletonSetGlobalPosition(HumanSkeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float3& gt)
    {
        math::float3 lt = gt;

        if (aIndex > 0)
        {
            SkeletonInverseTransformPosition(apSkeleton, apSkeletonPose, apSkeleton->m_Node[aIndex].m_ParentId, lt);
        }

        apSkeletonPose->m_X[aIndex].t = lt;
    }

    void SkeletonInverseTransformRotation(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex, math::float4& r)
    {
        if (aIndex > 0)
        {
            SkeletonInverseTransformRotation(apSkeleton, apSkeletonPose, apSkeleton->m_Node[aIndex].m_ParentId, r);
        }

        r = math::quatMul(math::quatConj(apSkeletonPose->m_X[aIndex].q), r);
        r = math::scaleMulQuat(apSkeletonPose->m_X[aIndex].s, r);
    }

    void SkeletonSetGlobalRotation(HumanSkeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float4& gr)
    {
        math::float4 lr = gr;

        if (aIndex > 0)
        {
            SkeletonInverseTransformRotation(apSkeleton, apSkeletonPose, apSkeleton->m_Node[aIndex].m_ParentId, lr);
        }

        apSkeletonPose->m_X[aIndex].q = lr;
    }

    void SkeletonInverseTransformScale(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex, math::float3& s)
    {
        if (aIndex > 0)
        {
            SkeletonInverseTransformScale(apSkeleton, apSkeletonPose, apSkeleton->m_Node[aIndex].m_ParentId, s);
        }

        s *= math::inverseScale(apSkeletonPose->m_X[aIndex].s);
    }

    void SkeletonSetGlobalScale(HumanSkeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float3& gs)
    {
        math::float3 ls = gs;

        if (aIndex > 0)
        {
            SkeletonInverseTransformScale(apSkeleton, apSkeletonPose, apSkeleton->m_Node[aIndex].m_ParentId, ls);
        }

        apSkeletonPose->m_X[aIndex].s = ls;
    }

    math::float3 SkeletonGetDoF(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex)
    {
        const int32_t axesIndex = apSkeleton->m_Node[aIndex].m_AxesId;

        return math::select(math::quat2Qtan(apSkeletonPose->m_X[aIndex].q), math::ToAxes(apSkeleton->m_AxesArray[axesIndex], apSkeletonPose->m_X[aIndex].q), math::int3(-(axesIndex != -1)));
    }

    void SkeletonSetDoF(HumanSkeleton const* apSkeleton, SkeletonPose * apSkeletonPose, math::float3 const& aDoF, int32_t aIndex)
    {
        const int32_t axesIndex = apSkeleton->m_Node[aIndex].m_AxesId;
        apSkeletonPose->m_X[aIndex].q = math::select(math::qtan2Quat(aDoF), math::FromAxes(apSkeleton->m_AxesArray[axesIndex], aDoF), math::int4(-(axesIndex != -1)));
    }

    math::float3 SkeletonNodeEndPoint(HumanSkeleton const *apSkeleton, int32_t aIndex, SkeletonPose const *apSkeletonPose)
    {
        return math::mul(apSkeletonPose->m_X[aIndex], math::quatXcos(apSkeleton->m_AxesArray[aIndex].m_PostQ) * math::float1(apSkeleton->m_AxesArray[aIndex].m_Length));
    }

    void SkeletonAlign(skeleton::HumanSkeleton const *apSkeleton, math::float4 const &arRefQ, math::float4 & arQ, int32_t aIndex)
    {
        const int32_t axesIndex = apSkeleton->m_Node[aIndex].m_AxesId;

        if (axesIndex != -1)
        {
            math::Axes axes = apSkeleton->m_AxesArray[axesIndex];

            math::float3 refV = math::quatXcos(math::normalize(math::quatMul(arRefQ, axes.m_PostQ)));
            math::float3 v = math::quatXcos(math::normalize(math::quatMul(arQ, axes.m_PostQ)));
            math::float4 dq = math::quatArcRotate(v, refV);

            arQ = math::quatMul(dq, arQ);
        }
    }

    void SkeletonAlign(skeleton::HumanSkeleton const *apSkeleton, skeleton::SkeletonPose const*apSkeletonPoseRef, skeleton::SkeletonPose *apSkeletonPose, int32_t aIndex)
    {
        SkeletonAlign(apSkeleton, apSkeletonPoseRef->m_X[aIndex].q, apSkeletonPose->m_X[aIndex].q, aIndex);
    }

    void Skeleton2BoneAdjustLength(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, math::float1 const& aRatio, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace)
    {
        math::float3 vAB = apSkeletonPoseWorkspace->m_X[aIndexB].t - apSkeletonPoseWorkspace->m_X[aIndexA].t;
        math::float3 vBC = apSkeletonPoseWorkspace->m_X[aIndexC].t - apSkeletonPoseWorkspace->m_X[aIndexB].t;
        math::float3 vAD = aTarget - apSkeletonPoseWorkspace->m_X[aIndexA].t;

        math::float1 lenABC = math::length(vAB) + math::length(vBC);
        math::float1 lenAD = math::length(vAD);
        math::float1 ratio = lenAD / lenABC;
        math::float1 invARatio = math::float1(1.f) - aRatio;

        if (ratio > invARatio)
        {
            ratio = (ratio - (invARatio)) / (math::float1(2) * aRatio);
            ratio = math::saturate(ratio);
            math::float1 r = math::float1(1.f) + aRatio * ratio * ratio;

            apSkeletonPose->m_X[aIndexB].t *= r;
            apSkeletonPose->m_X[aIndexC].t *= r;
        }
    }

    static inline math::float1 triangleAngle(const math::float1& aLen, const math::float1& aLen1, const math::float1& aLen2)
    {
        math::float1 c = math::clamp((aLen1 * aLen1 + aLen2 * aLen2 - aLen * aLen) / (aLen1 * aLen2) / math::float1(2.f), math::float1(-1.f) , math::float1(1.f));
        return math::acos(c);
    }

    void Skeleton2BoneIK(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, float aWeight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace)
    {
        math::float4 qA0 = apSkeletonPose->m_X[aIndexA].q;
        math::float4 qB0 = apSkeletonPose->m_X[aIndexB].q;

        math::float3 dof = skeleton::SkeletonGetDoF(apSkeleton, apSkeletonPose, aIndexB) * math::float3(1.0f, 0, 0.9f);
        skeleton::SkeletonSetDoF(apSkeleton, apSkeletonPose, dof, aIndexB);
        skeleton::SkeletonPoseComputeGlobal(apSkeleton, apSkeletonPose, apSkeletonPoseWorkspace, aIndexC, aIndexB);

        math::float3 vAB = apSkeletonPoseWorkspace->m_X[aIndexB].t - apSkeletonPoseWorkspace->m_X[aIndexA].t;
        math::float3 vBC = apSkeletonPoseWorkspace->m_X[aIndexC].t - apSkeletonPoseWorkspace->m_X[aIndexB].t;
        math::float3 vAC = apSkeletonPoseWorkspace->m_X[aIndexC].t - apSkeletonPoseWorkspace->m_X[aIndexA].t;
        math::float3 vAD = aTarget - apSkeletonPoseWorkspace->m_X[aIndexA].t;

        math::float1 lenAB = math::length(vAB);
        math::float1 lenBC = math::length(vBC);
        math::float1 lenAC = math::length(vAC);
        math::float1 lenAD = math::length(vAD);

        math::float1 angleAC = triangleAngle(lenAC, lenAB, lenBC);
        math::float1 angleAD = triangleAngle(lenAD, lenAB, lenBC);

        math::float3 axis = math::normalize(math::cross(vAB, vBC));

        math::float1 a = math::float1(0.5f) * (angleAC - angleAD);
        math::float4 q = math::float4(axis, 1.f) * math::sincos_sssc(a);
        apSkeletonPoseWorkspace->m_X[aIndexB].q = math::normalize(math::quatMul(q, apSkeletonPoseWorkspace->m_X[aIndexB].q));

        skeleton::SkeletonPoseComputeLocal(apSkeleton, apSkeletonPoseWorkspace, apSkeletonPose, aIndexB, aIndexB);
        apSkeletonPose->m_X[aIndexB].q = math::quatLerp(qB0, apSkeletonPose->m_X[aIndexB].q, math::float1(aWeight));
        skeleton::SkeletonPoseComputeGlobal(apSkeleton, apSkeletonPose, apSkeletonPoseWorkspace, aIndexC, aIndexB);

        vAC = apSkeletonPoseWorkspace->m_X[aIndexC].t - apSkeletonPoseWorkspace->m_X[aIndexA].t;

        q = math::quatArcRotate(vAC, vAD);
        apSkeletonPoseWorkspace->m_X[aIndexA].q = math::quatMul(q, apSkeletonPoseWorkspace->m_X[aIndexA].q);
        skeleton::SkeletonPoseComputeLocal(apSkeleton, apSkeletonPoseWorkspace, apSkeletonPose, aIndexA, aIndexA);
        apSkeletonPose->m_X[aIndexA].q = math::quatLerp(apSkeletonPose->m_X[aIndexA].q, qA0, math::powr(math::float1(1.f - aWeight), math::float1(4.f)));
    }

    void Skeleton3BoneIK(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, float weight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace)
    {
        if (weight > 0)
        {
            math::float4 qA = apSkeletonPose->m_X[aIndexA].q;
            math::float4 qB = apSkeletonPose->m_X[aIndexB].q;
            math::float4 qC = apSkeletonPose->m_X[aIndexC].q;

            math::float1 fingerLen  = math::float1(apSkeleton->m_AxesArray[aIndexA].m_Length + apSkeleton->m_AxesArray[aIndexB].m_Length + apSkeleton->m_AxesArray[aIndexC].m_Length);
            math::float1 targetDist = math::length(apSkeletonPoseWorkspace->m_X[aIndexA].t - aTarget);
            math::float1 fact = math::powr(math::saturate(math::float1(targetDist / fingerLen)), math::float1(4.0f));
            math::float3 dof = math::float3(0.f, 0.f, 2.0f * float(fact) - 1.0f);

            skeleton::SkeletonSetDoF(apSkeleton, apSkeletonPose, dof, aIndexB);
            skeleton::SkeletonSetDoF(apSkeleton, apSkeletonPose, dof, aIndexC);
            skeleton::SkeletonPoseComputeGlobal(apSkeleton, apSkeletonPose, apSkeletonPoseWorkspace, aIndexC, aIndexB);

            math::float3 endT = SkeletonNodeEndPoint(apSkeleton, aIndexC, apSkeletonPoseWorkspace);
            math::float3 endV = endT - apSkeletonPoseWorkspace->m_X[aIndexA].t;
            math::float3 targetV = aTarget - apSkeletonPoseWorkspace->m_X[aIndexA].t;

            math::float4 q = math::quatArcRotate(endV, targetV);
            apSkeletonPoseWorkspace->m_X[aIndexA].q = math::quatMul(q, apSkeletonPoseWorkspace->m_X[aIndexA].q);
            skeleton::SkeletonPoseComputeLocal(apSkeleton, apSkeletonPoseWorkspace, apSkeletonPose, aIndexA, aIndexA);

            math::float1 w(weight);

            apSkeletonPose->m_X[aIndexA].q = math::quatLerp(qA, apSkeletonPose->m_X[aIndexA].q, w);
            apSkeletonPose->m_X[aIndexB].q = math::quatLerp(qB, apSkeletonPose->m_X[aIndexB].q, w);
            apSkeletonPose->m_X[aIndexC].q = math::quatLerp(qC, apSkeletonPose->m_X[aIndexC].q, w);
        }
    }

    void Skeleton2BoneAdjustHint(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aHint, float weight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace, float minLenRatio)
    {
        math::float3 A = apSkeletonPoseWorkspace->m_X[aIndexA].t;
        math::float3 B = apSkeletonPoseWorkspace->m_X[aIndexB].t;
        math::float3 C = apSkeletonPoseWorkspace->m_X[aIndexC].t;

        math::float3 vAC = C - A;

        float vACLen = math::length(vAC);

        if (vACLen > 0.0f)
        {
            math::float3 vACn = vAC / vACLen;

            math::float3 vAB = B - A;
            math::float3 vAH = aHint - A;

            math::float3 vABProj = vACn * math::dot(vAB, vACn);
            math::float3 vAHProj = vACn * math::dot(vAH, vACn);

            math::float3 vABPerp = vAB - vABProj;
            math::float3 vAHPerp = vAH - vAHProj;

            float vABPrepLen = math::length(vABPerp);
            float vAHPrepLen = math::length(vAHPerp);

            if (vABPrepLen > 0.0f && vAHPrepLen > 0.0f)
            {
                math::float3 vBC = C - B;

                float vABLen = math::length(vAB);
                float vBCLen = math::length(vBC);
                float minLen = minLenRatio * (vABLen + vBCLen);

                if (vABPrepLen > minLen)
                {
                    float safeLen = 2.0f * minLen;
                    float safeWeight = math::select(1.0f - (safeLen - vABPrepLen) / (safeLen - minLen), 1.0f, -int(vABPrepLen > safeLen));

                    math::float4 q = math::quatArcRotate(vABPerp, vAHPerp);
                    q.xyz *= weight * safeWeight;
                    apSkeletonPoseWorkspace->m_X[aIndexA].q = math::normalize(math::quatMul(q, apSkeletonPoseWorkspace->m_X[aIndexA].q));
                    skeleton::SkeletonPoseComputeLocal(apSkeleton, apSkeletonPoseWorkspace, apSkeletonPose, aIndexA, aIndexA);
                }
            }
        }
    }

    void SetupAxes(skeleton::HumanSkeleton *apSkeleton, skeleton::SkeletonPose const *apSkeletonPoseGlobal, math::SetupAxesInfo const& apSetupAxesInfo, int32_t aIndex, int32_t aAxisIndex, bool aLeft, float aLen)
    {
        skeleton::Node &node = apSkeleton->m_Node[aIndex];
        int32_t parentIndex = node.m_ParentId;

        if (node.m_AxesId != -1)
        {
            math::Axes &axes = apSkeleton->m_AxesArray[node.m_AxesId];

            math::trsX boneX = apSkeletonPoseGlobal->m_X[aIndex];

            axes.m_Limit.m_Min = math::radians(math::float3(apSetupAxesInfo.m_Min[0], apSetupAxesInfo.m_Min[1], apSetupAxesInfo.m_Min[2]));
            axes.m_Limit.m_Max = math::radians(math::float3(apSetupAxesInfo.m_Max[0], apSetupAxesInfo.m_Max[1], apSetupAxesInfo.m_Max[2]));
            axes.m_Sgn = math::float3(apSetupAxesInfo.m_Sgn[0], apSetupAxesInfo.m_Sgn[1], apSetupAxesInfo.m_Sgn[2]) * (aLeft ? math::float3(1.f) : math::float3(-1.f, 1.f, -1.f));

            math::float3 mainAxis = math::float3(apSetupAxesInfo.m_MainAxis[0], apSetupAxesInfo.m_MainAxis[1], apSetupAxesInfo.m_MainAxis[2]);
            math::float4 zeroQ = math::float4(apSetupAxesInfo.m_PreQ[0], apSetupAxesInfo.m_PreQ[1], apSetupAxesInfo.m_PreQ[2], apSetupAxesInfo.m_PreQ[3]) * (aLeft ? math::float4(1.f) : math::float4(1.f, 1.f, 1.f, -1.f));

            math::float3 u = math::float3(1.f, 0.f, 0.f);
            math::float3 w = math::float3(0.f, 1.f, 0.f);
            math::float3 v = math::float3(0.f, 0.f, 1.f);

            axes.m_Type = apSetupAxesInfo.m_Type;

            axes.m_Length = 1.0f;

            if (aAxisIndex != -1)
            {
                math::trsX axisX = apSkeletonPoseGlobal->m_X[aAxisIndex];

                math::float3 uDir = (axisX.t - boneX.t) * math::float1(aLen);
                float uLen = math::length(uDir);
                math::float3 vDir = math::cross(mainAxis, uDir);
                float vLen = math::length(vDir);

                if (uLen > math::epsilon_normal_sqrt() && vLen > math::epsilon_normal_sqrt())
                {
                    u = uDir / uLen;
                    v = vDir / vLen;
                    w = math::cross(u, v);

                    axes.m_Length = uLen;
                }
            }

            if (apSetupAxesInfo.m_ForceAxis)
            {
                math::float3 uDir;

                switch (apSetupAxesInfo.m_ForceAxis)
                {
                    case +1: uDir = math::float3(+1.f, 0.f, 0.f); break;
                    case -1: uDir = math::float3(-1.f, 0.f, 0.f); break;
                    case +2: uDir = math::float3(0.f, +1.f, 0.f); break;
                    case -2: uDir = math::float3(0.f, -1.f, 0.f); break;
                    case +3: uDir = math::float3(0.f, 0.f, +1.f); break;
                    default: uDir = math::float3(0.f, 0.f, -1.f); break;
                }

                math::float3 vDir = math::cross(mainAxis, uDir);
                float vLen = math::length(vDir);

                if (vLen > math::epsilon_normal_sqrt())
                {
                    u = uDir;
                    v = vDir / vLen;
                    w = math::cross(u, v);
                }
            }

            axes.m_Length *= fabs(aLen);

            math::float4 parentQ = math::select(math::quatIdentity(), apSkeletonPoseGlobal->m_X[parentIndex].q, math::int4(-(parentIndex != -1)));

            axes.m_PreQ = math::matrixToQuat(u.xyz, v.xyz, w.xyz);
            axes.m_PostQ = math::normalize(math::quatMul(math::quatConj(boneX.q), axes.m_PreQ));
            axes.m_PreQ = math::normalize(math::quatMul(math::quatConj(parentQ), math::quatMul(zeroQ, axes.m_PreQ)));
        }
    }


    // explicit template instantiations...


    template SkeletonPoseT<math::affineX> *CreateSkeletonPose<math::affineX>(HumanSkeleton const* apSkeleton, RuntimeBaseAllocator& arAlloc);
    template SkeletonPoseT<math::trsX> *CreateSkeletonPose<math::trsX>(HumanSkeleton const* apSkeleton, RuntimeBaseAllocator& arAlloc);

    template void DestroySkeletonPose<math::affineX>(SkeletonPoseT<math::affineX>* apSkeletonPose, RuntimeBaseAllocator& arAlloc);
    template void DestroySkeletonPose<math::trsX>(SkeletonPoseT<math::trsX>* apSkeletonPose, RuntimeBaseAllocator& arAlloc);

    template void SkeletonPoseCopy<math::trsX, math::trsX>(SkeletonPoseT<math::trsX> const* apFromPose, SkeletonPoseT<math::trsX>* apToPose);
    template void SkeletonPoseCopy<math::trsX, math::affineX>(SkeletonPoseT<math::trsX> const* apFromPose, SkeletonPoseT<math::affineX>* apToPose);


    template void SkeletonPoseComputeGlobal<math::trsX>(HumanSkeleton const* apSkeleton, SkeletonPoseT<math::trsX> const* apLocalPose, SkeletonPoseT<math::trsX>* apGlobalPose);
    template void SkeletonPoseComputeGlobal<math::affineX>(HumanSkeleton const* apSkeleton, SkeletonPoseT<math::affineX> const* apLocalPose, SkeletonPoseT<math::affineX>* apGlobalPose);

    template void SkeletonPoseComputeGlobal<math::trsX>(HumanSkeleton const* apSkeleton, SkeletonPoseT<math::trsX> const* apLocalPose, SkeletonPoseT<math::trsX>* apGlobalPose, int aIndex, int aStopIndex);
    template void SkeletonPoseComputeGlobal<math::affineX>(HumanSkeleton const* apSkeleton, SkeletonPoseT<math::affineX> const* apLocalPose, SkeletonPoseT<math::affineX>* apGlobalPose, int aIndex, int aStopIndex);

    template void SkeletonPoseComputeLocal<math::trsX>(HumanSkeleton const* apSkeleton, SkeletonPoseT<math::trsX> const* apGlobalPose, SkeletonPoseT<math::trsX>* apLocalPose);

    template void SkeletonPoseComputeLocal<math::trsX>(HumanSkeleton const* apSkeleton, SkeletonPoseT<math::trsX> const* apGlobalPose, SkeletonPoseT<math::trsX>* apLocalPose, int aIndex, int aStopIndex);
}
}

#pragma once

#include "./memory.h"
#include "./types.h"
//@TODO: Remove vec-transform.h
#include "./Simd/vec-transform.h"


namespace math
{
    struct SetupAxesInfo;
    struct Axes;
}

namespace mecanim
{
namespace skeleton
{
    struct Node
    {

        Node() : m_ParentId(-1), m_AxesId(-1) {}
        int32_t m_ParentId;
        int32_t m_AxesId;

    };

    struct Skeleton
    {

        Skeleton() : m_Count(0), m_AxesCount(0) {}

        uint32_t            m_Count;
        OffsetPtr<Node>     m_Node;
        OffsetPtr<uint32_t> m_ID;       // CRC(path)

        uint32_t                m_AxesCount;
        OffsetPtr<math::Axes>   m_AxesArray;

    };

    template<typename transformType>
    struct SkeletonPoseT
    {

        SkeletonPoseT() : m_Count(0) {}

        uint32_t                    m_Count;
        OffsetPtr<transformType>    m_X;

    };

    typedef SkeletonPoseT<math::trsX> SkeletonPose;

    struct SkeletonMaskElement
    {

        SkeletonMaskElement() : m_PathHash(0), m_Weight(0.f) {}

        uint32_t    m_PathHash;
        float       m_Weight;
    };

    struct SkeletonMask
    {

        SkeletonMask() : m_Count(0) {}

        uint32_t                        m_Count;
        OffsetPtr<SkeletonMaskElement>  m_Data;
    };

    Skeleton* CreateSkeleton(int32_t aNodeCount, int32_t aAxesCount, RuntimeBaseAllocator& arAlloc);
    void DestroySkeleton(Skeleton* apSkeleton, RuntimeBaseAllocator& arAlloc);


    template<typename transformType>
    size_t CalculateSkeletonPoseSize(Skeleton const* apSkeleton, size_t baseAddress, RuntimeBaseAllocator& arAlloc);

    template<typename transformType>
    SkeletonPoseT<transformType> *CreateSkeletonPose(Skeleton const* apSkeleton, RuntimeBaseAllocator& arAlloc);

    template<typename transformType>
    void DestroySkeletonPose(SkeletonPoseT<transformType>* apSkeletonPose, RuntimeBaseAllocator& arAlloc);

    SkeletonMask* CreateSkeletonMask(uint32_t aNodeCount, SkeletonMaskElement const* elements, RuntimeBaseAllocator& arAlloc);
    void DestroySkeletonMask(SkeletonMask* skeletonMask, RuntimeBaseAllocator& arAlloc);

    // copy skeleton
    void SkeletonCopy(Skeleton const* apSrc, Skeleton* apDst);

    // copy pose
    template<typename transformTypeFrom, typename transformTypeTo>
    void SkeletonPoseCopy(SkeletonPoseT<transformTypeFrom> const* apFromPose, SkeletonPoseT<transformTypeTo>* apToPose);

    template<typename transformTypeFrom, typename transformTypeTo>
    void SkeletonPoseCopy(SkeletonPoseT<transformTypeFrom> const* apFromPose, SkeletonPoseT<transformTypeTo>* apToPose, uint32_t aIndexCount, int32_t const *apIndexArray);

    template<typename transformTypeFrom, typename transformTypeTo>
    void SkeletonPoseCopy(Skeleton const* apFromSkeleton, SkeletonPoseT<transformTypeFrom> const* apFromPose, Skeleton const* apToSkeleton, SkeletonPoseT<transformTypeTo>* apToPose);

    // Find & Copy pose based on name binding
    int32_t SkeletonFindNodeUp(Skeleton const *apSkeleton, int32_t aIndex, uint32_t aID);
    int32_t SkeletonFindNode(Skeleton const *apSkeleton, uint32_t aID);
    void SkeletonBuildIndexArray(int32_t *indexArray, Skeleton const* apSrcSkeleton, Skeleton const* apDstSkeleton);
    void SkeletonBuildReverseIndexArray(int32_t *reverseIndexArray, int32_t const*indexArray, Skeleton const* apSrcSkeleton, Skeleton const* apDstSkeleton);

    // set mask for a skeleton mask array
    void SkeletonPoseSetDirty(Skeleton const* apSkeleton, uint32_t* apSkeletonPoseMask, int aIndex, int aStopIndex, uint32_t aMask);

    // those functions work in place
    // computes a global pose from a local pose
    template<typename transformType>
    void SkeletonPoseComputeGlobal(Skeleton const* apSkeleton, SkeletonPoseT<transformType> const* apLocalPose, SkeletonPoseT<transformType>* apGlobalPose);

    // computes a global pose from a local pose for part of the skeleton starting at aIndex (child) to aStopIndex (ancestor)
    template<typename transformType>
    void SkeletonPoseComputeGlobal(Skeleton const* apSkeleton, SkeletonPoseT<transformType> const* apLocalPose, SkeletonPoseT<transformType>* apGlobalPose, int aIndex, int aStopIndex);

    template<typename transformType>
    void SkeletonPoseComputeLocal(Skeleton const* apSkeleton, SkeletonPoseT<transformType> const* apGlobalPose, SkeletonPoseT<transformType>* apLocalPose);

    template<typename transformType>
    void SkeletonPoseComputeLocal(Skeleton const* apSkeleton, SkeletonPoseT<transformType> const* apGlobalPose, SkeletonPoseT<transformType>* apLocalPose, int aIndex, int aStopIndex);

    // computes a global Q pose from a local Q pose
    void SkeletonPoseComputeGlobalQ(Skeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseLocal, SkeletonPose* apSkeletonPoseGlobal);
    // computes a global Q pose from a local Q pose for part of the skeleton starting at aIndex (child) to aStopIndex (ancestor)
    void SkeletonPoseComputeGlobalQ(Skeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseLocal, SkeletonPose* apSkeletonPoseGlobal, int aIndex, int aStopIndex);
    // computes a local Q pose from a global Q pose
    void SkeletonPoseComputeLocalQ(Skeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseGlobal, SkeletonPose* apSkeletonPoseLocal);
    // computes a local Q pose from a global Q pose for part of the skeleton starting at aIndex (child) to aStopIndex (ancestor)
    void SkeletonPoseComputeLocalQ(Skeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseGlobal, SkeletonPose* apSkeletonPoseLocal, int aIndex, int aStopIndex);


    // get global trs
    math::trsX SkeletonGetGlobalX(Skeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // get global position
    math::float3 SkeletonGetGlobalPosition(Skeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // get global "scaled" rotation
    math::float4 SkeletonGetGlobalRotation(Skeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // get global position and "scaled" rotation
    void SkeletonGetGlobalPositionAndRotation(Skeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex, math::float3 &position, math::float4 &rotation);

    // set global position
    void SkeletonSetGlobalPosition(Skeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float3& gt);
    // set global "scaled" rotation
    void SkeletonSetGlobalRotation(Skeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float4& gr);
    // set global scale
    void SkeletonSetGlobalScale(Skeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float3& gs);

    // get dof for bone index in pose
    math::float3 SkeletonGetDoF(Skeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // set dof for bone index in pose
    void SkeletonSetDoF(Skeleton const* apSkeleton, SkeletonPose * apSkeletonPose, math::float3 const& aDoF, int32_t aIndex);
    // algin x axis of skeleton node quaternion to ref node quaternion
    void SkeletonAlign(skeleton::Skeleton const *apSkeleton, math::float4 const &arRefQ, math::float4 & arQ, int32_t aIndex);
    // algin x axis of skeleton pose node to ref pose node
    void SkeletonAlign(skeleton::Skeleton const *apSkeleton, skeleton::SkeletonPose const*apSkeletonPoseRef, skeleton::SkeletonPose *apSkeletonPose, int32_t aIndex);

    // ik
    // compute end point of a node which is x * xcos * lenght.
    math::float3 SkeletonNodeEndPoint(Skeleton const *apSkeleton, int32_t aIndex, SkeletonPose const*apSkeletonPose);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton2BoneAdjustLength(Skeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, math::float1 const& aRatio, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton2BoneIK(Skeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, float aWeight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton3BoneIK(Skeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, float weight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton2BoneAdjustHint(Skeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aHint, float weight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace, float minLenRatio = 0.05f);

    void SetupAxes(skeleton::Skeleton *apSkeleton, skeleton::SkeletonPose const *apSkeletonPoseGlobal, math::SetupAxesInfo const& apSetupAxesInfo, int32_t aIndex, int32_t aAxisIndex, bool aLeft, float aLen = 1.0f);
}
}

#pragma once

#include "./memory.h"
#include "./types.h"
//@TODO: Remove vec-transform.h
#include "./Simd/vec-transform.h"
#include "core/templates/local_vector.h"


namespace math
{
    struct SetupAxesInfo;
    struct Axes;
}

namespace human_anim
{
namespace skeleton
{
    struct Node
    {

        Node() : m_ParentId(-1), m_AxesId(-1) {}
        int32_t m_ParentId;
        int32_t m_AxesId;

    };

    struct HumanSkeleton
    {

        HumanSkeleton() : m_Count(0), m_AxesCount(0) {}

        uint32_t            m_Count;
        LocalVector<Node>     m_Node;
        LocalVector<uint32_t> m_ID;       // CRC(path)

        uint32_t                m_AxesCount;
        LocalVector<math::Axes>   m_AxesArray;

    };

    template<typename transformType>
    struct SkeletonPoseT
    {

        SkeletonPoseT() : m_Count(0) {}

        uint32_t                    m_Count;
        LocalVector<transformType>    m_X;

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
        LocalVector<SkeletonMaskElement>  m_Data;
    };

    HumanSkeleton* CreateSkeleton(int32_t aNodeCount, int32_t aAxesCount, RuntimeBaseAllocator& arAlloc);
    void DestroySkeleton(HumanSkeleton* apSkeleton, RuntimeBaseAllocator& arAlloc);



    template<typename transformType>
    SkeletonPoseT<math::affineX> *CreateSkeletonPose(HumanSkeleton const* apSkeleton, RuntimeBaseAllocator& arAlloc);

    template<typename transformType>
    void DestroySkeletonPose(SkeletonPoseT<transformType>* apSkeletonPose, RuntimeBaseAllocator& arAlloc);

    SkeletonMask* CreateSkeletonMask(uint32_t aNodeCount, SkeletonMaskElement const* elements, RuntimeBaseAllocator& arAlloc);
    void DestroySkeletonMask(SkeletonMask* skeletonMask, RuntimeBaseAllocator& arAlloc);

    // copy skeleton
    void SkeletonCopy(HumanSkeleton const* apSrc, HumanSkeleton* apDst);

    // copy pose
    template<typename transformTypeFrom, typename transformTypeTo>
    void SkeletonPoseCopy(SkeletonPoseT<transformTypeFrom> const* apFromPose, SkeletonPoseT<transformTypeTo>* apToPose);

    template<typename transformTypeFrom, typename transformTypeTo>
    void SkeletonPoseCopy(SkeletonPoseT<transformTypeFrom> const* apFromPose, SkeletonPoseT<transformTypeTo>* apToPose, uint32_t aIndexCount, int32_t const *apIndexArray);

    template<typename transformTypeFrom, typename transformTypeTo>
    void SkeletonPoseCopy(HumanSkeleton const* apFromSkeleton, SkeletonPoseT<transformTypeFrom> const* apFromPose, HumanSkeleton const* apToSkeleton, SkeletonPoseT<transformTypeTo>* apToPose);

    // Find & Copy pose based on name binding
    int32_t SkeletonFindNodeUp(HumanSkeleton const *apSkeleton, int32_t aIndex, uint32_t aID);
    int32_t SkeletonFindNode(HumanSkeleton const *apSkeleton, uint32_t aID);
    void SkeletonBuildIndexArray(int32_t *indexArray, HumanSkeleton const* apSrcSkeleton, HumanSkeleton const* apDstSkeleton);
    void SkeletonBuildReverseIndexArray(int32_t *reverseIndexArray, int32_t const*indexArray, HumanSkeleton const* apSrcSkeleton, HumanSkeleton const* apDstSkeleton);

    // set mask for a skeleton mask array
    void SkeletonPoseSetDirty(HumanSkeleton const* apSkeleton, uint32_t* apSkeletonPoseMask, int aIndex, int aStopIndex, uint32_t aMask);

    // those functions work in place
    // computes a global pose from a local pose
    template<typename transformType>
    void SkeletonPoseComputeGlobal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apLocalPose, SkeletonPoseT<transformType>* apGlobalPose);

    // computes a global pose from a local pose for part of the skeleton starting at aIndex (child) to aStopIndex (ancestor)
    template<typename transformType>
    void SkeletonPoseComputeGlobal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apLocalPose, SkeletonPoseT<transformType>* apGlobalPose, int aIndex, int aStopIndex);

    template<typename transformType>
    void SkeletonPoseComputeLocal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apGlobalPose, SkeletonPoseT<transformType>* apLocalPose);

    template<typename transformType>
    void SkeletonPoseComputeLocal(HumanSkeleton const* apSkeleton, SkeletonPoseT<transformType> const* apGlobalPose, SkeletonPoseT<transformType>* apLocalPose, int aIndex, int aStopIndex);

    // computes a global Q pose from a local Q pose
    void SkeletonPoseComputeGlobalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseLocal, SkeletonPose* apSkeletonPoseGlobal);
    // computes a global Q pose from a local Q pose for part of the skeleton starting at aIndex (child) to aStopIndex (ancestor)
    void SkeletonPoseComputeGlobalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseLocal, SkeletonPose* apSkeletonPoseGlobal, int aIndex, int aStopIndex);
    // computes a local Q pose from a global Q pose
    void SkeletonPoseComputeLocalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseGlobal, SkeletonPose* apSkeletonPoseLocal);
    // computes a local Q pose from a global Q pose for part of the skeleton starting at aIndex (child) to aStopIndex (ancestor)
    void SkeletonPoseComputeLocalQ(HumanSkeleton const* apSkeleton, SkeletonPose const* apSkeletonPoseGlobal, SkeletonPose* apSkeletonPoseLocal, int aIndex, int aStopIndex);


    // get global trs
    math::trsX SkeletonGetGlobalX(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // get global position
    math::float3 SkeletonGetGlobalPosition(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // get global "scaled" rotation
    math::float4 SkeletonGetGlobalRotation(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // get global position and "scaled" rotation
    void SkeletonGetGlobalPositionAndRotation(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex, math::float3 &position, math::float4 &rotation);

    // set global position
    void SkeletonSetGlobalPosition(HumanSkeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float3& gt);
    // set global "scaled" rotation
    void SkeletonSetGlobalRotation(HumanSkeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float4& gr);
    // set global scale
    void SkeletonSetGlobalScale(HumanSkeleton const* apSkeleton, SkeletonPose *apSkeletonPose, int32_t aIndex, const math::float3& gs);

    // get dof for bone index in pose
    math::float3 SkeletonGetDoF(HumanSkeleton const* apSkeleton, SkeletonPose const *apSkeletonPose, int32_t aIndex);
    // set dof for bone index in pose
    void SkeletonSetDoF(HumanSkeleton const* apSkeleton, SkeletonPose * apSkeletonPose, math::float3 const& aDoF, int32_t aIndex);
    // algin x axis of skeleton node quaternion to ref node quaternion
    void SkeletonAlign(skeleton::HumanSkeleton const *apSkeleton, math::float4 const &arRefQ, math::float4 & arQ, int32_t aIndex);
    // algin x axis of skeleton pose node to ref pose node
    void SkeletonAlign(skeleton::HumanSkeleton const *apSkeleton, skeleton::SkeletonPose const*apSkeletonPoseRef, skeleton::SkeletonPose *apSkeletonPose, int32_t aIndex);

    // ik
    // compute end point of a node which is x * xcos * lenght.
    math::float3 SkeletonNodeEndPoint(HumanSkeleton const *apSkeleton, int32_t aIndex, SkeletonPose const*apSkeletonPose);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton2BoneAdjustLength(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, math::float1 const& aRatio, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton2BoneIK(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, float aWeight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton3BoneIK(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aTarget, float weight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace);
    // The apSkeletonPoseWorkspace parameter has to be a valid global pose, otherwise unexpected result may occur
    void Skeleton2BoneAdjustHint(HumanSkeleton const *apSkeleton, int32_t aIndexA, int32_t aIndexB, int32_t aIndexC, math::float3 const &aHint, float weight, SkeletonPose *apSkeletonPose, SkeletonPose *apSkeletonPoseWorkspace, float minLenRatio = 0.05f);

    void SetupAxes(skeleton::HumanSkeleton *apSkeleton, skeleton::SkeletonPose const *apSkeletonPoseGlobal, math::SetupAxesInfo const& apSetupAxesInfo, int32_t aIndex, int32_t aAxisIndex, bool aLeft, float aLen = 1.0f);
}
}

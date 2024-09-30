#pragma once

#include "./memory.h"
#include "./types.h"
//@TODO: Remove vec-transform.h
#include "./Simd/vec-transform.h"
#include "core/templates/local_vector.h"
#include "./Simd/vec-trs.h"
#include "./axes.h"

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
        int32_t m_bone_index;
        int32_t m_ParentId;
        int32_t m_AxesId;

		void load(const Dictionary& aDict) {

            m_bone_index = aDict["bone_index"];
            m_ParentId = aDict["ParentId"];
            m_AxesId = aDict["AxesId"];
		}

        void save(Dictionary& aDict) {
            aDict["bone_index"] = m_bone_index;
            aDict["ParentId"] = m_ParentId;
            aDict["AxesId"] = m_AxesId;
        }

    };

    struct HumanSkeleton
    {

        HumanSkeleton() {

        }

		void load(const Dictionary& aDict) {
            int node_count = aDict["node_count"];
            int axes_count = aDict["axes_count"];
            m_Node.resize(node_count);
            m_AxesArray.resize(axes_count);
            Array nodes = aDict["nodes"];
            Array axes = aDict["axes"];
            for (int i = 0; i < node_count; i++) {
                Dictionary node_dict = nodes[i];
                m_Node[i].load(node_dict);
            }

            for (int i = 0; i < axes_count; i++) {
                Dictionary axes_dict = axes[i];
                m_AxesArray[i].load(axes_dict);
            }
        }
        void save(Dictionary& aDict) {

            aDict["node_count"] = m_Node.size();
            aDict["axes_count"] = m_AxesArray.size();

            Array nodes;
            Array axes;
            for (uint32_t i = 0; i < m_Node.size(); i++) {
                Dictionary node_dict;
                m_Node[i].save(node_dict);
                nodes.push_back(node_dict);
            }
            for (uint32_t i = 0; i < m_AxesArray.size(); i++) {
                Dictionary axes_dict;
                m_AxesArray[i].save(axes_dict);
                axes.push_back(axes_dict);
            }
            aDict["nodes"] = nodes;
            aDict["axes"] = axes;
        }

        LocalVector<Node>     m_Node;
        LocalVector<math::Axes>   m_AxesArray;

    };

    template<typename transformType>
    struct SkeletonPoseT
    {

        SkeletonPoseT() {}

        //uint32_t                    m_Count;
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


    // Find & Copy pose based on name binding
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

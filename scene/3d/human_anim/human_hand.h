#pragma once

#include "./defs.h"
#include "./memory.h"
#include "./types.h"
#include "./Simd/vec-trs.h"
#include "core/string/ustring.h"



namespace math
{
    struct SetupAxesInfo;
}

namespace human_anim
{
namespace skeleton
{
    struct HumanSkeleton;
    struct SetupAxesInfo;

    template<typename transformType> struct SkeletonPoseT;
    typedef SkeletonPoseT<math::trsX> SkeletonPose;
}

namespace hand
{
	// 指頭，大拇指 食指 中指等
    enum Fingers
    {
        kThumb = 0 ,
        kIndex,
        kMiddle,
        kRing,
        kLittle,
        kLastFinger
    };
	//指骨 每根指頭一共3個關節
    enum Phalanges
    {
        kProximal = 0,
        kIntermediate, //1
        kDistal, 
        kLastPhalange //3
    };
	//
    enum FingerDoF
    {
        kProximalDownUp = 0,	//  近端上下
        kProximalInOut,			// 1 //近端裏外
        kIntermediateCloseOpen, // 2 // 中間開關
        kDistalCloseOpen,		// 3 // 遠端開關
        kLastFingerDoF			// 4 
    };

    const int32_t s_BoneCount = kLastFinger * kLastPhalange; //  指頭數量乘以指骨數量 5*3=15
    const int32_t s_DoFCount = kLastFinger * kLastFingerDoF; //  指頭數量乘以指骨DOF 5*4 =20//但只手的手指的全部dof

    inline int32_t GetBoneIndex(int32_t fingerIndex, int32_t phalangeIndex) { return fingerIndex * kLastPhalange + phalangeIndex; }
    inline int32_t GetFingerIndex(int32_t boneIndex) { return boneIndex / kLastPhalange; }
    inline int32_t GetPhalangeIndex(int32_t boneIndex) { return boneIndex % kLastPhalange; }
    inline int32_t GetDoFIndex(int32_t fingerIndex, int32_t phalangeDoFIndex) { return fingerIndex * kLastFingerDoF + phalangeDoFIndex; }

    const char* FingerName(uint32_t finger);
    String FingerDoFName(uint32_t finger);
    const char* PhalangeName(uint32_t finger);

    struct Hand
    {

        Hand();
        int32_t     m_HandBoneIndex[s_BoneCount];

    };

    struct HandPose
    {
        HandPose();

        math::trsX m_GrabX;
        float m_DoFArray[s_DoFCount];
        float m_Override;
        float m_CloseOpen;
        float m_InOut;
        float m_Grab;

    };

    int32_t MuscleFromBone(int32_t aBoneIndex, int32_t aDoFIndex);
    int32_t BoneFromMuscle(int32_t aDoFIndex);

    Hand* CreateHand(RuntimeBaseAllocator& alloc);
    void DestroyHand(Hand *hand, RuntimeBaseAllocator& alloc);

    void HandSetupAxes(Hand const *hand, skeleton::SkeletonPose const *skeletonPose, skeleton::HumanSkeleton *skeleton, bool aLeft);
    void HandCopyAxes(Hand const *srcHand, skeleton::HumanSkeleton const *srcSkeleton, Hand const *hand, skeleton::HumanSkeleton *skeleton);
    void HandPoseCopy(HandPose const *handPoseSrc, HandPose *handPoseDst);

    // Retargeting function set
    void HandPoseSolve(HandPose const* handPose, HandPose* handPoseOut);
    void Hand2SkeletonPose(Hand const *hand, skeleton::HumanSkeleton const *skeleton, HandPose const *handPose, skeleton::SkeletonPose *skeletonPose);
    void Skeleton2HandPose(Hand const *hand, skeleton::HumanSkeleton const *skeleton, skeleton::SkeletonPose const *skeletonPose, HandPose *handPose, float offset = 0.0f);
    // IK
    void FingerLengths(Hand const *hand, float *lengthArray);
    //  Fingers IK related functions will be re-activated soon (early 2016) for work done on real-time mocap
    //  void FingerBaseFromPose(Hand const *hand,skeleton::SkeletonPose const *skeletonPose,math::float4 *positionArray);
    //  void FingerTipsFromPose(Hand const *hand,skeleton::HumanSkeleton const *skeleton, skeleton::SkeletonPose const *skeletonPose,math::float4 *positionArray);
    //  void FingersIKSolve(Hand const *hand, skeleton::HumanSkeleton const *skeleton,math::float4 const *positionArray, float *apWeightArray, skeleton::SkeletonPose *skeletonPose, skeleton::SkeletonPose *skeletonPoseWorkspace);

    math::SetupAxesInfo const& GetAxeInfo(uint32_t index);
}// namespace hand
}

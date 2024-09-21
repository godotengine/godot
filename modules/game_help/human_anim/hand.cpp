
#include "./skeleton.h"
#include "./hand.h"
#include "./axes.h"

namespace human_anim
{
namespace hand
{
    const static int32_t Phalange2DoF[hand::kLastPhalange][3] =
    {
        { kProximalDownUp, kProximalInOut, -1 },    // kProximal
        { kIntermediateCloseOpen, -1, -1 },         // kIntermediate
        { kDistalCloseOpen, -1, -1 }                // kDistal
    };

    const static int32_t DoF2Bone[s_DoFCount] =
    {
        kThumb * kLastPhalange + kProximal,
        kThumb * kLastPhalange + kProximal,
        kThumb * kLastPhalange + kIntermediate,
        kThumb * kLastPhalange + kDistal,

        kIndex * kLastPhalange + kProximal,
        kIndex * kLastPhalange + kProximal,
        kIndex * kLastPhalange + kIntermediate,
        kIndex * kLastPhalange + kDistal,

        kMiddle * kLastPhalange + kProximal,
        kMiddle * kLastPhalange + kProximal,
        kMiddle * kLastPhalange + kIntermediate,
        kMiddle * kLastPhalange + kDistal,

        kRing * kLastPhalange + kProximal,
        kRing * kLastPhalange + kProximal,
        kRing * kLastPhalange + kIntermediate,
        kRing * kLastPhalange + kDistal,

        kLittle * kLastPhalange + kProximal,
        kLittle * kLastPhalange + kProximal,
        kLittle * kLastPhalange + kIntermediate,
        kLittle * kLastPhalange + kDistal
    };


    math::SetupAxesInfo const& GetAxeInfo(uint32_t index)
    {
        const static math::SetupAxesInfo setupAxesInfoArray[s_BoneCount] =
        {
            {{0, 0.125, 0.125f, 1}, {0, 1, 0, 0}, {0, -25, -20}, {0, 25, 20}, {-1, -1, 1, -1}, math::kZYRoll, 0},      // kThumb.kProximal
            {{0, -0.2f, 0, 1}, {0, 1, 0, 0}, {0, 0, -40}, {0, 0, 35}, {-1, 1, 1, -1}, math::kZYRoll, 0},                   // kThumb.kIntermediate
            {{0, -0.2f, 0, 1}, {0, 1, 0, 0}, {0, 0, -40}, {0, 0, 35}, {-1, 1, 1, -1}, math::kZYRoll, 0},                   // kThumb.kDistal
            {{0, 0.08f, 0.3f, 1}, {0, 0, 1, 0}, {0, -20, -50}, {0, 20, 50}, {-1, -1, -1, -1}, math::kZYRoll, 0},       // kIndex.kProximal
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0},              // kIndex.kIntermediate
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0},              // kIndex.kDistal
            {{0, 0.04f, 0.3f, 1}, {0, 0, 1, 0}, {0, -7.5f, -50}, {0, 7.5f, 50}, {-1, -1, -1, -1}, math::kZYRoll, 0},   // kMiddle.kProximal
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0},              // kMiddle.kIntermediate
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0},              // kMiddle.kDistal
            {{0, -0.04f, 0.3f, 1}, {0, 0, 1, 0}, {0, -7.5f, -50}, {0, 7.5f, 50}, {-1, 1, -1, -1}, math::kZYRoll, 0},       // kRing.kProximal
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0},              // kRing.kIntermediate
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0},              // kRing.kDistal
            {{0, -0.08f, 0.3f, 1}, {0, 0, 1, 0}, {0, -20, -50}, {0, 20, 50}, {-1, 1, -1, -1}, math::kZYRoll, 0},           // kLittle.kProximal
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0},              // kLittle.kIntermediate
            {{0, 0, 0.33f, 1}, {0, 0, 1, 0}, {0, 0, -45}, {0, 0, 45}, {-1, 1, -1, -1}, math::kZYRoll, 0}                   // kLittle.kDistal
        };

        return setupAxesInfoArray[index];
    }

    const char* FingerName(uint32_t aFinger)
    {
        const static char* fingerName[kLastFinger] =
        {
            "Thumb",
            "Index",
            "Middle",
            "Ring",
            "Little"
        };

        return fingerName[aFinger];
    }

    const char* PhalangeName(uint32_t aFinger)
    {
        const static char* phalangeName[kLastPhalange] =
        {
            "Proximal",
            "Intermediate",
            "Distal"
        };
        return phalangeName[aFinger];
    }

    const char* FingerDoFName(uint32_t aFinger)
    {
        const static char* fingerDoFName[kLastFingerDoF] =
        {
            "1 Stretched",
            "Spread",
            "2 Stretched",
            "3 Stretched"
        };
        return fingerDoFName[aFinger];
    }

    Hand::Hand()
    {
        memset(m_HandBoneIndex, -1, sizeof(int32_t) * s_BoneCount);
    }

    int32_t MuscleFromBone(int32_t aBoneIndex, int32_t aDoFIndex)
    {
        int32_t ret = -1;
        int32_t findex = GetFingerIndex(aBoneIndex);
        int32_t pindex = GetPhalangeIndex(aBoneIndex);

        if (Phalange2DoF[pindex][2 - aDoFIndex] != -1)
        {
            ret = findex * kLastFingerDoF + Phalange2DoF[pindex][2 - aDoFIndex];
        }

        return ret;
    }

    int32_t BoneFromMuscle(int32_t aDoFIndex)
    {
        return DoF2Bone[aDoFIndex];
    }

    Hand* CreateHand(RuntimeBaseAllocator& arAlloc)
    {
        Hand* hand = memnew(Hand);
        memset(hand->m_HandBoneIndex, -1, sizeof(int32_t) * kLastFinger * kLastPhalange);

        return hand;
    }

    void DestroyHand(Hand *apHand, RuntimeBaseAllocator& arAlloc)
    {
        if (apHand)
        {
            memdelete(apHand);
        }
    }

    HandPose::HandPose() : m_GrabX(math::trsIdentity())
    {
        int32_t i;

        for (i = 0; i < s_DoFCount; i++)
        {
            m_DoFArray[i] = 0;
        }

        m_Override = 0;
        m_CloseOpen = 0;
        m_InOut = 0;
        m_Grab = 0;
    }

    void HandPoseCopy(HandPose const *apHandPoseSrc, HandPose *apHandPoseDst)
    {
        int32_t i;

        for (i = 0; i < s_DoFCount; i++)
        {
            apHandPoseDst->m_DoFArray[i] = apHandPoseSrc->m_DoFArray[i];
        }

        apHandPoseDst->m_Override = apHandPoseDst->m_Override;
        apHandPoseDst->m_CloseOpen = apHandPoseDst->m_CloseOpen;
        apHandPoseDst->m_InOut = apHandPoseDst->m_InOut;
        apHandPoseDst->m_Grab = apHandPoseDst->m_Grab;
        apHandPoseDst->m_GrabX = apHandPoseDst->m_GrabX;
    }

    void HandSetupAxes(Hand const *apHand, skeleton::SkeletonPose const *apSkeletonPose, skeleton::Skeleton *apSkeleton, bool aLeft)
    {
        int32_t f, p, b;

        for (f = 0; f < kLastFinger; f++)
        {
            for (p = 0; p < kLastPhalange; p++)
            {
                float len = 1.0f;
                int32_t skAxisBoneId = -1;

                b = GetBoneIndex(f, p);
                int32_t skBoneIndex = apHand->m_HandBoneIndex[b];

                if (p < kLastPhalange - 1 && apHand->m_HandBoneIndex[GetBoneIndex(f, p + 1)] >= 0)
                {
                    skAxisBoneId = apHand->m_HandBoneIndex[GetBoneIndex(f, p + 1)];
                }
                else if (p > 0 && apHand->m_HandBoneIndex[GetBoneIndex(f, p - 1)] >= 0)
                {
                    skAxisBoneId = apHand->m_HandBoneIndex[GetBoneIndex(f, p - 1)];
                    len = -0.75f;
                }

                if (skBoneIndex >= 0)
                {
                    skeleton::SetupAxes(apSkeleton, apSkeletonPose, GetAxeInfo(b), skBoneIndex, skAxisBoneId, aLeft, len);
                }
            }
        }
    }

    void HandCopyAxes(Hand const *apSrcHand, skeleton::Skeleton const *apSrcSkeleton, Hand const *apHand, skeleton::Skeleton *apSkeleton)
    {
        int32_t i;

        for (i = 0; i < s_BoneCount; i++)
        {
            skeleton::Node const *srcNode = apSrcHand->m_HandBoneIndex[i] >= 0 ? &apSrcSkeleton->m_Node[apSrcHand->m_HandBoneIndex[i]] : 0;
            skeleton::Node const *node = apHand->m_HandBoneIndex[i] >= 0 ? &apSkeleton->m_Node[apHand->m_HandBoneIndex[i]] : 0;

            if (srcNode != 0 && node != 0 && srcNode->m_AxesId != -1 && node->m_AxesId != -1)
            {
                apSkeleton->m_AxesArray[node->m_AxesId] = apSrcSkeleton->m_AxesArray[srcNode->m_AxesId];
            }
        }
    }

    void HandPoseSolve(HandPose const* apHandPoseIn, HandPose* apHandPoseOut)
    {
        int32_t f;

        for (f = 0; f < kLastFinger; f++)
        {
            int32_t i = f * kLastFingerDoF;
            apHandPoseOut->m_DoFArray[i + kProximalDownUp] = (1 - apHandPoseIn->m_Override) * apHandPoseIn->m_DoFArray[i + kProximalDownUp] + apHandPoseIn->m_CloseOpen;
            apHandPoseOut->m_DoFArray[i + kProximalInOut] = (1 - apHandPoseIn->m_Override) * apHandPoseIn->m_DoFArray[i + kProximalInOut] + apHandPoseIn->m_InOut;
            apHandPoseOut->m_DoFArray[i + kIntermediateCloseOpen] = (1 - apHandPoseIn->m_Override) * apHandPoseIn->m_DoFArray[i + kIntermediateCloseOpen] + apHandPoseIn->m_CloseOpen;
            apHandPoseOut->m_DoFArray[i + kDistalCloseOpen] = (1 - apHandPoseIn->m_Override) * apHandPoseIn->m_DoFArray[i + kDistalCloseOpen] + apHandPoseIn->m_CloseOpen;
        }
    }

    void Hand2SkeletonPose(Hand const *apHand, skeleton::Skeleton const *apSkeleton, HandPose const *apHandPose, skeleton::SkeletonPose *apSkeletonPose)
    {
        int32_t f, p;

        for (f = 0; f < kLastFinger; f++)
        {
            for (p = 0; p < kLastPhalange; p++)
            {
                int32_t i = GetBoneIndex(f, p);

                if (apHand->m_HandBoneIndex[i] >= 0)
                {
                    math::int3 mask = math::int3(-(Phalange2DoF[p][2] != -1), -(Phalange2DoF[p][1] != -1), -(Phalange2DoF[p][0] != -1));
                    math::float3 xyz = math::select(math::float3(math::ZERO),
                        math::float3(apHandPose->m_DoFArray[GetDoFIndex(f, Phalange2DoF[p][2])], apHandPose->m_DoFArray[GetDoFIndex(f, Phalange2DoF[p][1])], apHandPose->m_DoFArray[GetDoFIndex(f, Phalange2DoF[p][0])]),
                        mask);

                    skeleton::SkeletonSetDoF(apSkeleton, apSkeletonPose, xyz, apHand->m_HandBoneIndex[i]);
                }
            }
        }
    }

    void Skeleton2HandPose(Hand const *apHand, skeleton::Skeleton const *apSkeleton, skeleton::SkeletonPose const *apSkeletonPose, HandPose *apHandPose, float aOffset)
    {
        int32_t f, p;

        for (f = 0; f < kLastFinger; f++)
        {
            for (p = 0; p < kLastPhalange; p++)
            {
                int32_t i = GetBoneIndex(f, p);

                if (apHand->m_HandBoneIndex[i] >= 0)
                {
                    const math::float3 xyz = skeleton::SkeletonGetDoF(apSkeleton, apSkeletonPose, apHand->m_HandBoneIndex[i]);

                    if (Phalange2DoF[p][2] != -1)
                        apHandPose->m_DoFArray[GetDoFIndex(f, Phalange2DoF[p][2])] = xyz.x + aOffset;
                    if (Phalange2DoF[p][1] != -1)
                        apHandPose->m_DoFArray[GetDoFIndex(f, Phalange2DoF[p][1])] = xyz.y + aOffset;
                    if (Phalange2DoF[p][0] != -1)
                        apHandPose->m_DoFArray[GetDoFIndex(f, Phalange2DoF[p][0])] = xyz.z + aOffset;
                }
            }
        }
    }

    void FingerLengths(Hand const *apHand, skeleton::Skeleton const *apSkeleton, float *apLengthArray)
    {
        int32_t f, p;

        for (f = 0; f < kLastFinger; f++)
        {
            apLengthArray[f] = 0.0f;

            for (p = 0; p < kLastPhalange; p++)
            {
                int32_t i = GetBoneIndex(f, p);

                if (apHand->m_HandBoneIndex[i] >= 0)
                {
                    apLengthArray[f] += apSkeleton->m_AxesArray[apHand->m_HandBoneIndex[i]].m_Length;
                }
            }
        }
    }

    /*
    void FingerBaseFromPose(Hand const *apHand,skeleton::SkeletonPose const *apSkeletonPose,math::float4 *apPositionArray)
    {
        int32_t f;

        for(f = 0; f < kLastFinger; f++)
        {
            int32_t i = GetBoneIndex(f,kProximal);

            if(apHand->m_HandBoneIndex[i] >= 0)
            {
                apPositionArray[f] = apSkeletonPose->m_X[apHand->m_HandBoneIndex[i]].t;
            }
        }
    }

    void FingerTipsFromPose(Hand const *apHand,skeleton::Skeleton const *apSkeleton, skeleton::SkeletonPose const *apSkeletonPose,math::float4 *apPositionArray)
    {
        int32_t f;

        for(f = 0; f < kLastFinger; f++)
        {
            int32_t i = GetBoneIndex(f,kDistal);

            if(apHand->m_HandBoneIndex[i] >= 0)
            {
                apPositionArray[f] = skeleton::SkeletonNodeEndPoint(apSkeleton,apHand->m_HandBoneIndex[i],apSkeletonPose);
            }
        }
    }

    void FingersIKSolve(Hand const *apHand, skeleton::Skeleton const *apSkeleton,math::float4 const *apPositionArray, float *apWeightArray, skeleton::SkeletonPose *apSkeletonPose, skeleton::SkeletonPose *apSkeletonPoseWorkspace)
    {
        int32_t f;

        for(f = 0; f < kLastFinger; f++)
        {
            int32_t topIndex = apHand->m_HandBoneIndex[hand::GetBoneIndex(f,hand::kProximal)];
            int32_t midIndex = apHand->m_HandBoneIndex[hand::GetBoneIndex(f,hand::kIntermediate)];
            int32_t endIndex = apHand->m_HandBoneIndex[hand::GetBoneIndex(f,hand::kDistal)];

            if(topIndex >= 0 && midIndex >= 0 && endIndex >= 0)
            {
                skeleton::Skeleton3BoneIK(apSkeleton,topIndex,midIndex,endIndex,apPositionArray[f],apWeightArray[f],apSkeletonPose,apSkeletonPoseWorkspace);
            }
        }
    }
    */
}
}

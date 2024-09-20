#if 0
#include "UnityPrefix.h"
#include "ImportAnimationUtility.h"

#include "Modules/Animation/human_anim/animation/animationset.h"
#include "Modules/Animation/AnimationUtility.h"
#include "Modules/Animation/Animation.h"
#include "Modules/Animation/Animator.h"
#include "Modules/Animation/AnimatorGenericBindings.h"
#include "Runtime/Math/AnimationCurveUtility.h"
#include "Modules/Animation/human_anim/math/axes.h"
#include "Runtime/Utilities/CRC32.h"
#include "Modules/Animation/MecanimClipBuilder.h"
#include "Runtime/Jobs/Jobs.h"
#include "Runtime/Jobs/BlockRangeJob.h"

#include "Editor/Src/Animation/KeyframeReducer.h"

PROFILER_INFORMATION(gRemovedMaskedCurve, "RemovedMaskedCurve", kProfilerAnimation);
PROFILER_INFORMATION(gAddAdditionnalCurve, "AddAdditionnalCurve", kProfilerAnimation);
PROFILER_INFORMATION(gUnrollMuscles, "UnrollMuscles", kProfilerAnimation);
PROFILER_INFORMATION(gOffsetMuscles, "OffsetMuscles", kProfilerAnimation);
PROFILER_INFORMATION(gGenerateMecanimClipsCurves, "GenerateMecanimClipsCurves", kProfilerAnimation);
PROFILER_INFORMATION(gGenerateMecanimClipCurveJob, "GenerateMecanimClipCurveJob", kProfilerAnimation);
PROFILER_INFORMATION(gGenerateMecanimClipCurveCombineJob, "GenerateMecanimClipCurveCombineJob", kProfilerAnimation);
PROFILER_INFORMATION(gOptimizeCurves, "OptimizeCurves", kProfilerAnimation);

template<class T>
struct HasAnimatedMuscleTransform
{
    core::string m_Path;
    float m_Epsilon;
    HasAnimatedMuscleTransform(core::string const & path, float epsilon) : m_Path(path), m_Epsilon(epsilon) {}
    bool operator()(T& curve)
    {
        return curve.path == m_Path && !IsConstantCurve(curve.curve, m_Epsilon);
    }
};


template<class T>
struct HasPathPredicate
{
    core::string mPath;
    HasPathPredicate(core::string const & path) : mPath(path) {}
    bool operator()(T& curve) {return curve.path == mPath; }
};

template<>
struct HasPathPredicate<TransformMaskElement>
{
    core::string mPath;
    HasPathPredicate(core::string const & path) : mPath(path) {}
    bool operator()(TransformMaskElement const& element) {return element.m_Path == mPath && element.m_Weight > 0.f; }
};

template<typename CURVETYPE, typename CURVEVECTOR> void RemoveMaskedCurve(TransformMaskElementList const& mask, CURVEVECTOR& curves)
{
    CURVEVECTOR removeCurves;
    typename CURVEVECTOR::iterator it;
    for (it = curves.begin(); it != curves.end(); ++it)
    {
        TransformMaskElementList::const_iterator maskIt = std::find_if(mask.begin(), mask.end(), HasPathPredicate<TransformMaskElementList::value_type>(it->path));
        if (maskIt == mask.end())
            removeCurves.push_back(*it);
    }

    for (it = removeCurves.begin(); it != removeCurves.end(); ++it)
    {
        typename CURVEVECTOR::iterator end = std::remove_if(curves.begin(), curves.end(), HasPathPredicate<CURVETYPE>(it->path));
        curves.erase(end, curves.end());
    }
}

void RemovedMaskedCurve(AnimationClip& clip, const ClipAnimationInfo& clipInfo, bool isClipOlderOr42)
{
    PROFILER_AUTO(gRemovedMaskedCurve);

    // - Clear muscle curves not in mask
    human_anim::human::HumanPoseMask humanMask = HumanPoseMaskFromBodyMask(clipInfo.bodyMask);

    for (int muscleIter = 0; muscleIter < human_anim::animation::s_ClipMuscleCurveCount; muscleIter++)
    {
        if (!human_anim::animation::GetMuscleCurveInMask(humanMask, muscleIter))
        {
            AnimationClip::FloatCurves &floatCurves = clip.GetFloatCurves();

            bool curveFound = false;

            for (AnimationClip::FloatCurves::iterator curveIter = floatCurves.begin(); !curveFound && curveIter !=  floatCurves.end(); curveIter++)
            {
                AnimationClip::FloatCurve& floatCurve = *curveIter;

                if (floatCurve.type == TypeOf<Animator>())
                {
                    if (floatCurve.attribute == human_anim::animation::GetMuscleCurveName(muscleIter).c_str())
                    {
                        floatCurves.erase(curveIter);
                        curveFound = true;
                    }
                }
            }
        }
    }

    AnimationClip::Vector3Curves& positionCurves = clip.GetPositionCurves();
    AnimationClip::QuaternionCurves& quaternionCurves = clip.GetQuaternionCurves();
    AnimationClip::Vector3Curves& eulerCurves = clip.GetEulerCurves();
    AnimationClip::Vector3Curves& scaleCurves = clip.GetScaleCurves();

    // by default transform mask is created empty,
    // for human clip the default behavior is to remove all curve except muscle curve
    // clip from 4.2 or previous version was importing everything
    if (clip.IsHumanMotion() && clipInfo.transformMask.size() == 0 && !isClipOlderOr42)
    {
        positionCurves.clear();
        quaternionCurves.clear();
        scaleCurves.clear();
        eulerCurves.clear();
    }
    else if (clipInfo.transformMask.size() > 0)
    {
        RemoveMaskedCurve<AnimationClip::Vector3Curve, AnimationClip::Vector3Curves>(clipInfo.transformMask, positionCurves);
        RemoveMaskedCurve<AnimationClip::Vector3Curve, AnimationClip::Vector3Curves>(clipInfo.transformMask, eulerCurves);
        RemoveMaskedCurve<AnimationClip::QuaternionCurve, AnimationClip::QuaternionCurves>(clipInfo.transformMask, quaternionCurves);
        RemoveMaskedCurve<AnimationClip::Vector3Curve, AnimationClip::Vector3Curves>(clipInfo.transformMask, scaleCurves);
    }
}

void AddAdditionalCurves(AnimationClip& clip, const ClipAnimationInfo& clipInfo)
{
    PROFILER_AUTO(gAddAdditionnalCurve);

    AnimationClip::FloatCurves &floatCurves = clip.GetFloatCurves();

    float stopTime = clip.GetAnimationClipSettings().m_StopTime;
    float startTime = clip.GetAnimationClipSettings().m_StartTime;

    const float kMinLength = 1 / clip.GetSampleRate();
    float length = std::max(stopTime - startTime, kMinLength);

    for (int curveIter = 0; curveIter < clipInfo.curves.size(); curveIter++)
    {
        floatCurves.push_back(AnimationClip::FloatCurve());

        AnimationClip::FloatCurve& curve = floatCurves.back();


        curve.type = TypeOf<Animator>();
        curve.attribute = clipInfo.curves[curveIter].name;
        curve.curve = clipInfo.curves[curveIter].curve;

        bool hasErrors = false;
        for (int keyIter = 0; keyIter < curve.curve.GetKeyCount(); keyIter++)
        {
            AnimationCurve::Keyframe& key = curve.curve.GetKey(keyIter);

            key.time *= length;
            key.time += startTime;
            key.inSlope /= length;
            key.outSlope /= length;

            if (!AnimationCurve::Keyframe::IsValid(key))
            {
                ErrorStringObject(Format("Key with index %i in curve %s of clip %s is invalid. This curve will not be used.", keyIter, curve.attribute.c_str(), clip.GetName()), &clip);
                hasErrors = true;
                continue;
            }
        }
        if (hasErrors)
        {
            floatCurves.pop_back();
        }
    }
}

AvatarType AnimationTypeToAvatarType(ModelImporter::AnimationType type)
{
    switch (type)
    {
        case ModelImporter::kHumanoid:  return kHumanoid;
        default: return kGeneric;
    }
}

bool IsZeroCurve(AnimationCurve& curve, float zeroTol = 1e-4f)
{
    bool ret = true;

    int keyCount = curve.GetKeyCount();

    for (int keyIter = 0; ret && keyIter < keyCount; keyIter++)
    {
        ret = math::abs(curve.GetKey(keyIter).value) < zeroTol;
    }

    return ret;
}

bool ShouldAddCurve(AnimationCurve& curve, int curveIndex)
{
    bool ret = true;

    if (curveIndex >= human_anim::animation::s_ClipMuscleCurveTDoFBegin && curveIndex < human_anim::animation::s_ClipMuscleCurveTDoFEnd)
    {
        ret = !IsZeroCurve(curve);
    }

    return ret;
}

void UnrollMuscles(AnimationCurve& curve, int curveIndex, const human_anim::animation::AvatarConstant& avatarConstant)
{
    PROFILER_AUTO(gUnrollMuscles);

    if (curveIndex >= human_anim::animation::s_ClipMuscleCurveBodyDoFBegin && curveIndex < human_anim::animation::s_ClipMuscleCurveBodyDoFEnd)
    {
        human_anim::int32_t muscleIndex = curveIndex - human_anim::animation::s_ClipMuscleCurveBodyDoFBegin;
        float min = math::radians(-180.0f);
        float max = math::radians(+180.0f);
        human_anim::human::GetMuscleRange(avatarConstant.m_Human.Get(), muscleIndex, min, max);

        int keyCount = curve.GetKeyCount();

        int minKeyIndex = 0;
        float minKeyValue = 1e6;

        for (int keyIter = 0; keyIter < keyCount; keyIter++)
        {
            curve.GetKey(keyIter).value = math::LimitUnproject(min, max, curve.GetKey(keyIter).value);

            float absValue = std::abs(curve.GetKey(keyIter).value);

            if (absValue < minKeyValue)
            {
                minKeyIndex = keyIter;
                minKeyValue = absValue;
            }
        }

        float prevKeyValue = curve.GetKey(minKeyIndex).value;

        for (int keyIter = minKeyIndex + 1; keyIter < keyCount; keyIter++)
        {
            float keyValue = curve.GetKey(keyIter).value;

            keyValue += math::round((prevKeyValue - keyValue) / math::radians(360.f)) * math::radians(360.f);

            curve.GetKey(keyIter).value = keyValue;

            prevKeyValue = keyValue;
        }

        prevKeyValue = curve.GetKey(minKeyIndex).value;

        for (int keyIter = minKeyIndex - 1; keyIter >= 0; keyIter--)
        {
            float keyValue = curve.GetKey(keyIter).value;

            keyValue += math::round((prevKeyValue - keyValue) / math::radians(360.f)) * math::radians(360.f);

            curve.GetKey(keyIter).value = keyValue;

            prevKeyValue = keyValue;
        }

        for (int keyIter = 0; keyIter < keyCount; keyIter++)
        {
            curve.GetKey(keyIter).value = math::LimitProject(min, max, curve.GetKey(keyIter).value);
        }

        if (curve.GetKeyCount() >= 3)
        {
            Assert(IsFinite(curve.GetKey(0).value));
            Assert(IsFinite(curve.GetKey(1).value));
            Assert(IsFinite(curve.GetKey(2).value));
        }
    }
}

void OffsetMuscles(AnimationClip& clip, const human_anim::animation::AvatarConstant& avatarConstant)
{
    PROFILER_AUTO(gOffsetMuscles);

    AnimationClip::FloatCurves& curves = clip.GetFloatCurves();

    int curveCount =  curves.size();

    for (int curveIter = 0; curveIter < curveCount; curveIter++)
    {
        if (curves[curveIter].type == TypeOf<Animator>())
        {
            human_anim::uint32_t muscleId = ComputeCRC32(curves[curveIter].attribute.c_str());
            human_anim::int32_t curveIndex = human_anim::animation::FindMuscleIndex(muscleId);

            if (curveIndex >= human_anim::animation::s_ClipMuscleCurveBodyDoFBegin && curveIndex < human_anim::animation::s_ClipMuscleCurveBodyDoFEnd)
            {
                human_anim::int32_t muscleIndex = curveIndex - human_anim::animation::s_ClipMuscleCurveBodyDoFBegin;
                float min = math::radians(-180.0f);
                float max = math::radians(+180.0f);
                human_anim::human::GetMuscleRange(avatarConstant.m_Human.Get(), muscleIndex, min, max);

                AnimationCurve &curve = curves[curveIter].curve;

                int keyCount = curve.GetKeyCount();

                int minKeyIndex = -1;
                float minKeyValue = 1e6;

                for (int keyIter = 0; keyIter < keyCount; keyIter++)
                {
                    float absValue = std::abs(curve.GetKey(keyIter).value);

                    if (absValue < minKeyValue)
                    {
                        minKeyIndex = keyIter;
                        minKeyValue = absValue;
                    }
                }

                if (minKeyIndex != -1)
                {
                    float value = curve.GetKey(minKeyIndex).value;
                    value = math::LimitUnproject(min, max, value);
                    float offset = math::round(value / math::radians(360.f)) * math::radians(360.f);
                    value -= offset;
                    value = math::LimitProject(min, max, value);
                    offset = value - curve.GetKey(minKeyIndex).value;

                    for (int keyIter = 0; keyIter < keyCount; keyIter++)
                    {
                        curve.GetKey(keyIter).value = curve.GetKey(keyIter).value + offset;
                    }
                }
            }
        }
    }
}

static int positionBoneList[] = { human_anim::human::kLeftLowerLeg, human_anim::human::kLeftFoot, human_anim::human::kRightLowerLeg, human_anim::human::kRightFoot, human_anim::human::kLeftUpperArm, human_anim::human::kLeftLowerArm, human_anim::human::kLeftHand, human_anim::human::kRightUpperArm, human_anim::human::kRightLowerArm, human_anim::human::kRightHand, human_anim::human::kHead };
static int orientationBoneList[] = { human_anim::human::kLeftFoot, human_anim::human::kRightFoot, human_anim::human::kLeftHand, human_anim::human::kRightHand, human_anim::human::kHead };

const int positionBoneCount = sizeof(positionBoneList) / sizeof(int);
const int orientationBoneCount = sizeof(orientationBoneList) / sizeof(int);

math::float1 RetargetPositionError(const human_anim::human::Human &human, int humanIndex, human_anim::skeleton::SkeletonPose const &poseA, const human_anim::skeleton::SkeletonPose &poseB) // in normalized meter
{
    math::trsX xa = poseA.m_X[human.m_HumanBoneIndex[humanIndex]];
    math::trsX xb = poseB.m_X[human.m_HumanBoneIndex[humanIndex]];

    return math::length(xb.t - xa.t) / math::float1(human.m_Scale);
}

math::float1 RetargetOrientationError(const human_anim::human::Human &human, int humanIndex, human_anim::skeleton::SkeletonPose const &poseA, const human_anim::skeleton::SkeletonPose &poseB) // in degrees
{
    math::trsX xa = poseA.m_X[human.m_HumanBoneIndex[humanIndex]];
    math::trsX xb = poseB.m_X[human.m_HumanBoneIndex[humanIndex]];

    math::float4 qDiff = math::normalize(math::quatMul(math::quatConj(xb.q), xa.q));

    qDiff.w = math::float1(math::ZERO);

    return math::degrees(math::float1(2.0f * asin(math::length(qDiff))));
}

namespace
{
    struct GenerateMecanimClipsCurvesJobData
    {
        GenerateMecanimClipsCurvesJobData(
            AnimationClips const& clips,
            human_anim::animation::AvatarConstant const& avatarConstant,
            bool isHuman,
            HumanDescription const& humanDescription,
            AvatarBuilder::NamedTransforms const& namedTransform,
            bool doRetargetingQuality,
            int humanoidOversampling,
            human_anim::int32_t motionTransformIndex)
            : alloc(kMemTempJobAlloc)
            , clips(clips)
            , avatarConstant(avatarConstant)
            , isHuman(isHuman)
            , humanDescription(humanDescription)
            , namedTransform(namedTransform)
            , doRetargetingQuality(doRetargetingQuality)
            , humanoidOversampling(humanoidOversampling)
            , motionTransformIndex(motionTransformIndex)
            , clipBindings(NULL)
            , tqsMap(NULL)
        {
            clipBindings = UnityEngine::Animation::CreateAnimationSetBindings(clips, alloc);
            tqsMap = UnityEngine::Animation::CreateAvatarTQSMap(&avatarConstant, *clipBindings, alloc);

            invalidAnimationWarning.resize_initialized(clips.size());
            retargetingQualityMessages.resize_initialized(clips.size());
        }

        ~GenerateMecanimClipsCurvesJobData()
        {
            DestroyAnimationSetBindings(clipBindings, alloc);
            alloc.Deallocate(tqsMap);
        }

        BlockRange blocks[kMaximumBlockRangeCount];

        AnimationClips const& clips;
        human_anim::animation::AvatarConstant const& avatarConstant;
        HumanDescription const& humanDescription;
        AvatarBuilder::NamedTransforms const& namedTransform;
        int humanoidOversampling;
        bool isHuman;
        bool doRetargetingQuality;

        human_anim::memory::HeapAllocator alloc;
        human_anim::int32_t motionTransformIndex;

        UnityEngine::Animation::AnimationSetBindings* clipBindings;
        human_anim::animation::SkeletonTQSMap* tqsMap;

        math::float3 tDoFBaseArray[human_anim::human::kLastTDoF];

        dynamic_array<core::string> invalidAnimationWarning;
        dynamic_array<core::string> retargetingQualityMessages;

        core::string outInvalidAnimationWarning;
        core::string outRetargetingQualityMessages;
    };

    void GenerateMecanimClipCurveJob(GenerateMecanimClipsCurvesJobData* jobData, unsigned blockIndex)
    {
        PROFILER_AUTO(gGenerateMecanimClipCurveJob);

        int begin = jobData->blocks[blockIndex].startIndex;
        int end = begin + jobData->blocks[blockIndex].rangeSize;

        math::trsX motionX(math::trsIdentity());
        human_anim::human::HumanPose pose;

        human_anim::ValueArray*     valuesDefault = CreateValueArray(jobData->clipBindings->animationSet->m_DynamicFullValuesConstant, jobData->alloc);
        human_anim::ValueArray*     values = CreateValueArray(jobData->clipBindings->animationSet->m_DynamicFullValuesConstant, jobData->alloc);
        human_anim::ValueArrayMask* valuesMask = CreateValueArrayMask(jobData->clipBindings->animationSet->m_DynamicFullValuesConstant, jobData->alloc);

        human_anim::skeleton::SkeletonPose* humanLclPose = NULL;
        human_anim::skeleton::SkeletonPose* humanPoseA = NULL;
        human_anim::skeleton::SkeletonPose* humanPoseB = NULL;
        human_anim::skeleton::SkeletonPose* humanPoseC = NULL;
        human_anim::skeleton::SkeletonPose* humanPoseD = NULL;

        human_anim::skeleton::SkeletonPose* avatarLclPose = human_anim::skeleton::CreateSkeletonPose<math::trsX>(jobData->avatarConstant.m_AvatarSkeleton.Get(), jobData->alloc);
        human_anim::skeleton::SkeletonPose* avatarGblPose = human_anim::skeleton::CreateSkeletonPose<math::trsX>(jobData->avatarConstant.m_AvatarSkeleton.Get(), jobData->alloc);

        ValueFromSkeletonPose(*jobData->avatarConstant.m_AvatarSkeleton, *jobData->avatarConstant.m_DefaultPose, jobData->tqsMap, *valuesDefault);

        if (jobData->isHuman)
        {
            humanLclPose = human_anim::skeleton::CreateSkeletonPose<math::trsX>(jobData->avatarConstant.m_Human->m_Skeleton.Get(), jobData->alloc);
            humanPoseA = human_anim::skeleton::CreateSkeletonPose<math::trsX>(jobData->avatarConstant.m_Human->m_Skeleton.Get(), jobData->alloc);
            humanPoseB = human_anim::skeleton::CreateSkeletonPose<math::trsX>(jobData->avatarConstant.m_Human->m_Skeleton.Get(), jobData->alloc);
            humanPoseC = human_anim::skeleton::CreateSkeletonPose<math::trsX>(jobData->avatarConstant.m_Human->m_Skeleton.Get(), jobData->alloc);
            humanPoseD = human_anim::skeleton::CreateSkeletonPose<math::trsX>(jobData->avatarConstant.m_Human->m_Skeleton.Get(), jobData->alloc);

            if (jobData->avatarConstant.m_Human->m_HasTDoF)
            {
                human_anim::human::RetargetFromTDoFBase(jobData->avatarConstant.m_Human.Get(), humanPoseA, &jobData->tDoFBaseArray[0]);
            }
        }

        int rootNodeIndex = jobData->avatarConstant.m_RootMotionBoneIndex;

        for (int i = begin; i != end; ++i)
        {
            AnimationClip* animationClip = jobData->clips[i];
            core::string clipName = core::string(animationClip->GetName()).substr(11);
            core::string clipWarning = "";
            core::string clipRetargetWarning = "";

            human_anim::animation::ClipMuscleConstant *muscleClip = animationClip->GetRuntimeAsset();

            human_anim::animation::ClipInput in;
            human_anim::animation::ClipOutput *out = human_anim::animation::CreateClipOutput(muscleClip->m_Clip.Get(), jobData->alloc);
            human_anim::animation::ClipMemory *mem = human_anim::animation::CreateClipMemory(muscleClip->m_Clip.Get(), jobData->alloc);

            // compute frame count
            float period = 1.0f / animationClip->GetSampleRate() / float(jobData->humanoidOversampling);

            float startTime = animationClip->GetRange().first;
            float stopTime = animationClip->GetRange().second;

            if (stopTime > startTime)
            {
                AnimationCurve curveArray[human_anim::animation::s_ClipMuscleCurveCount];

                int curveStart = jobData->motionTransformIndex != -1 ? human_anim::animation::s_ClipMuscleCurveMotionBegin : human_anim::animation::s_ClipMuscleCurveRootBegin;

                int curveCount = 0;

                if (jobData->isHuman)
                {
                    curveCount = jobData->avatarConstant.m_Human.Get()->m_HasTDoF ? human_anim::animation::s_ClipMuscleCurveCount : human_anim::animation::s_ClipMuscleCurveCount - human_anim::animation::s_ClipMuscleCurveTDoFCount;
                }
                else
                {
                    curveCount += human_anim::animation::s_ClipMuscleCurveMotionCount;

                    if (rootNodeIndex != -1)
                    {
                        curveCount += human_anim::animation::s_ClipMuscleCurveRootCount;
                    }
                }

                int keyCount = ceil((stopTime - startTime) / period) + 1;

                for (int curveIter = curveStart; curveIter < curveCount; curveIter++)
                {
                    curveArray[curveIter].ResizeUninitialized(keyCount);
                }

                float positionAverageError[positionBoneCount];
                float positionMaximumError[positionBoneCount];
                float positionMaximumErrorTime[positionBoneCount];

                float orientationAverageError[orientationBoneCount];
                float orientationMaximumError[orientationBoneCount];
                float orientationMaximumErrorTime[orientationBoneCount];

                if (jobData->doRetargetingQuality)
                {
                    for (int boneIter = 0; boneIter < positionBoneCount; boneIter++)
                    {
                        positionAverageError[boneIter] = 0;
                        positionMaximumError[boneIter] = 0;
                        positionMaximumErrorTime[boneIter] = 0;
                    }

                    for (int boneIter = 0; boneIter < orientationBoneCount; boneIter++)
                    {
                        orientationAverageError[boneIter] = 0;
                        orientationMaximumError[boneIter] = 0;
                        orientationMaximumErrorTime[boneIter] = 0;
                    }
                }

                for (int keyIter = 0; keyIter < keyCount; keyIter++)
                {
                    float time = startTime + float(keyIter) * period;

                    in.m_Time = time;
                    EvaluateClip(muscleClip->m_Clip.Get(), &in, mem, out);

                    human_anim::SetValueMask<false>(valuesMask, false);
                    human_anim::animation::ValuesFromClip<false>(*valuesDefault, *out, jobData->clipBindings->animationSet->m_ClipConstant[i].m_Bindings, jobData->clipBindings->animationSet->m_IntegerRemapStride, *values, *valuesMask);
                    SkeletonPoseFromValue(*jobData->avatarConstant.m_AvatarSkeleton, *jobData->avatarConstant.m_DefaultPose, *values, jobData->tqsMap, *avatarLclPose, 0, false);

                    if (jobData->motionTransformIndex != -1)
                    {
                        human_anim::skeleton::SkeletonPoseComputeGlobal(jobData->avatarConstant.m_AvatarSkeleton.Get(), avatarLclPose, avatarGblPose, jobData->motionTransformIndex, 0);
                        motionX = avatarGblPose->m_X[jobData->motionTransformIndex];
                    }

                    if (jobData->isHuman)
                    {
                        if (jobData->motionTransformIndex != -1)
                        {
                            motionX.t /= jobData->avatarConstant.m_Human->m_Scale;
                        }

                        // [case 493451] Attached scene does not play animation correctly, must bake all animation for node between root and hips transform
                        human_anim::int32_t rootIndex = jobData->avatarConstant.m_HumanSkeletonIndexArray[0];

                        human_anim::skeleton::SkeletonPoseComputeGlobal(jobData->avatarConstant.m_AvatarSkeleton.Get(), avatarLclPose, avatarGblPose, rootIndex, 0);
                        human_anim::skeleton::SkeletonPoseCopy(jobData->avatarConstant.m_AvatarSkeleton.Get(), avatarLclPose, jobData->avatarConstant.m_Human->m_Skeleton.Get(), humanLclPose);

                        humanLclPose->m_X[0] = avatarGblPose->m_X[rootIndex];

                        human_anim::human::RetargetFrom(jobData->avatarConstant.m_Human.Get(), humanLclPose, &pose, humanPoseA, humanPoseB, humanPoseC, humanPoseD, &jobData->tDoFBaseArray[0]);

                        if (jobData->doRetargetingQuality)
                        {
                            human_anim::skeleton::SkeletonPoseComputeGlobal(jobData->avatarConstant.m_AvatarSkeleton.Get(), avatarLclPose, avatarGblPose);
                            human_anim::skeleton::SkeletonPoseCopy(jobData->avatarConstant.m_AvatarSkeleton.Get(), avatarGblPose, jobData->avatarConstant.m_Human->m_Skeleton.Get(), humanPoseA);

                            human_anim::human::HumanPose poseOut;
                            human_anim::human::RetargetTo(jobData->avatarConstant.m_Human.Get(), &pose, 0, math::trsIdentity(), &poseOut, humanLclPose, humanPoseB);
                            human_anim::skeleton::SkeletonPoseComputeGlobal(jobData->avatarConstant.m_Human->m_Skeleton.Get(), humanLclPose, humanPoseB);

                            for (int boneIter = 0; boneIter < positionBoneCount; boneIter++)
                            {
                                float positionError = RetargetPositionError(*jobData->avatarConstant.m_Human, positionBoneList[boneIter], *humanPoseA, *humanPoseB);

                                positionAverageError[boneIter] += positionError;

                                if (positionError > positionMaximumError[boneIter])
                                {
                                    positionMaximumError[boneIter] = positionError;
                                    positionMaximumErrorTime[boneIter] = time;
                                }
                            }

                            for (int boneIter = 0; boneIter < orientationBoneCount; boneIter++)
                            {
                                float orientationError = RetargetOrientationError(*jobData->avatarConstant.m_Human, orientationBoneList[boneIter], *humanPoseA, *humanPoseB);

                                orientationAverageError[boneIter] += orientationError;

                                if (orientationError > orientationMaximumError[boneIter])
                                {
                                    orientationMaximumError[boneIter] = orientationError;
                                    orientationMaximumErrorTime[boneIter] = time;
                                }
                            }
                        }
                    }
                    else if (rootNodeIndex != -1)
                    {
                        human_anim::skeleton::SkeletonPoseComputeGlobal(jobData->avatarConstant.m_AvatarSkeleton.Get(), avatarLclPose, avatarGblPose);

                        pose.m_RootX = avatarGblPose->m_X[rootNodeIndex];
                        pose.m_RootX.q = math::quatMul(pose.m_RootX.q, math::quatConj(jobData->avatarConstant.m_RootMotionBoneX.q));
                    }

                    for (int curveIter = curveStart; curveIter < curveCount; curveIter++)
                    {
                        AnimationCurve::Keyframe* keyFrame = &curveArray[curveIter].GetKey(keyIter);

                        keyFrame->value = human_anim::animation::GetMuscleCurveValue(pose, motionX, curveIter);
                        keyFrame->time = time;
                        keyFrame->inSlope = 0;
                        keyFrame->outSlope = 0;
                        keyFrame->inWeight = DefaultWeight<float>();
                        keyFrame->outWeight = DefaultWeight<float>();
                        keyFrame->weightedMode = kNotWeighted;
                        keyFrame->tangentMode = 0;

                        // unroll quaternions
                        if (keyIter > 0)
                        {
                            int tqIndex = human_anim::animation::GetMuscleCurveTQIndex(curveIter);
                            if (tqIndex == 6)
                            {
                                math::float4 qprev, q;

                                qprev.x = curveArray[curveIter - 3].GetKey(keyIter - 1).value;
                                qprev.y = curveArray[curveIter - 2].GetKey(keyIter - 1).value;
                                qprev.z = curveArray[curveIter - 1].GetKey(keyIter - 1).value;
                                qprev.w = curveArray[curveIter - 0].GetKey(keyIter - 1).value;

                                q.x = curveArray[curveIter - 3].GetKey(keyIter).value;
                                q.y = curveArray[curveIter - 2].GetKey(keyIter).value;
                                q.z = curveArray[curveIter - 1].GetKey(keyIter).value;
                                q.w = curveArray[curveIter - 0].GetKey(keyIter).value;

                                q = math::chgsign(q, math::dot(qprev, q));

                                curveArray[curveIter - 3].GetKey(keyIter).value = (float)q.x;
                                curveArray[curveIter - 2].GetKey(keyIter).value = (float)q.y;
                                curveArray[curveIter - 1].GetKey(keyIter).value = (float)q.z;
                                curveArray[curveIter - 0].GetKey(keyIter).value = (float)q.w;
                            }
                        }
                    }
                }

                if (jobData->doRetargetingQuality)
                {
                    float maxPositionError(0.001f); // in normalized meter
                    float maxOrientationError(0.5f); // in degrees

                    bool retargetWarning = false;

                    for (int boneIter = 0; boneIter < positionBoneCount; boneIter++)
                    {
                        positionAverageError[boneIter] /= float(keyCount);

                        if (positionMaximumError[boneIter] > maxPositionError)
                        {
                            retargetWarning = true;
                        }
                    }

                    for (int boneIter = 0; boneIter < orientationBoneCount; boneIter++)
                    {
                        orientationAverageError[boneIter] /= float(keyCount);

                        if (orientationMaximumError[boneIter] > maxOrientationError)
                        {
                            retargetWarning = true;
                        }
                    }

                    if (retargetWarning)
                    {
                        for (int boneIter = 0; boneIter < positionBoneCount; boneIter++)
                        {
                            if (positionMaximumError[boneIter] > maxPositionError)
                            {
                                float time = positionMaximumErrorTime[boneIter];
                                int seconds = int(time);
                                int frame = int((time - seconds) * animationClip->GetSampleRate());
                                int fullFrame = int(float(seconds) * animationClip->GetSampleRate()) + frame;
                                float percent = 100 * (time - startTime) / (stopTime - startTime);

                                clipRetargetWarning += Format("\t%s average position error %.1f mm and maximum position error %.1f mm at time %i:%2i (%3.1f%%) Frame %i\n", human_anim::human::BoneName(positionBoneList[boneIter]), positionAverageError[boneIter] * 1000, positionMaximumError[boneIter] * 1000, seconds, frame, percent, fullFrame).c_str();
                            }
                        }

                        for (int boneIter = 0; boneIter < orientationBoneCount; boneIter++)
                        {
                            if (orientationMaximumError[boneIter] > maxOrientationError)
                            {
                                float time = orientationMaximumErrorTime[boneIter];
                                int seconds = int(time);
                                int frame = int((time - seconds) * animationClip->GetSampleRate());
                                int fullFrame = int(float(seconds) * animationClip->GetSampleRate()) + frame;
                                float percent = 100 * (time - startTime) / (stopTime - startTime);

                                clipRetargetWarning += Format("\t%s average orientation error %.1f deg and maximum orientation error %.1f deg at time %i:%2i (%3.1f%%) Frame %i\n", human_anim::human::BoneName(orientationBoneList[boneIter]), orientationAverageError[boneIter], orientationMaximumError[boneIter], seconds, frame, percent, fullFrame).c_str();
                            }
                        }
                    }
                }

                if (jobData->isHuman)
                {
                    human_anim::int32_t rootIndex = jobData->avatarConstant.m_HumanSkeletonIndexArray[0];

                    bool startConversionWarning = false;

                    // Remove animations that are handled by muscle clip
                    for (int j = 0; j < jobData->namedTransform.size(); j++)
                    {
                        core::string transformPath = jobData->namedTransform[j].path;
                        core::string transformName = jobData->namedTransform[j].name;

                        human_anim::uint32_t pathHash = ComputeCRC32(transformPath.c_str());

                        AnimationClip::Vector3Curves& positionCurves = animationClip->GetPositionCurves();
                        AnimationClip::QuaternionCurves& quaternionCurves = animationClip->GetQuaternionCurves();
                        AnimationClip::Vector3Curves& eulerCurves = animationClip->GetEulerCurves();
                        AnimationClip::Vector3Curves& scaleCurves = animationClip->GetScaleCurves();

                        int nodeIndex = human_anim::skeleton::SkeletonFindNode(jobData->avatarConstant.m_Human->m_Skeleton.Get(), pathHash);

                        if (nodeIndex != -1 || human_anim::skeleton::SkeletonFindNodeUp(jobData->avatarConstant.m_AvatarSkeleton.Get(), rootIndex, pathHash) != -1)
                        {
                            HumanBoneList::const_iterator boneIt = std::find_if(jobData->humanDescription.m_Human.begin(), jobData->humanDescription.m_Human.end(), FindBoneName(jobData->namedTransform[j].name));

                            bool isHips = boneIt == jobData->humanDescription.m_Human.begin();
                            bool isInBetweenHumanBones = boneIt == jobData->humanDescription.m_Human.end();

                            if (isHips)
                            {
                                startConversionWarning = true;
                            }

                            if (startConversionWarning)
                            {
                                if (!isHips)
                                {
                                    if (std::find_if(positionCurves.begin(), positionCurves.end(), HasAnimatedMuscleTransform<AnimationClip::Vector3Curve>(transformPath, jobData->avatarConstant.m_Human->m_Scale / 1000.0f)) != positionCurves.end())
                                    {
                                        bool hasTDOF = false;

                                        for (int dofIter = 0; !hasTDOF && dofIter < human_anim::human::kLastTDoF; dofIter++)
                                        {
                                            int boneIndex = jobData->avatarConstant.m_Human->m_HumanBoneIndex[human_anim::human::BoneFromTDoF(dofIter)];
                                            if (boneIndex != -1)
                                            {
                                                hasTDOF = (boneIndex == nodeIndex);
                                            }
                                        }

                                        if (!(jobData->avatarConstant.m_Human->m_HasTDoF && hasTDOF))
                                        {
                                            clipWarning += Format("\t'%s' has translation animation that will be discarded.\n", transformName.c_str());
                                        }
                                    }
                                }

                                if (isInBetweenHumanBones)
                                {
                                    if (std::find_if(quaternionCurves.begin(), quaternionCurves.end(), HasAnimatedMuscleTransform<AnimationClip::QuaternionCurve>(transformPath, 0.0001f)) != quaternionCurves.end())
                                    {
                                        clipWarning += Format("\t'%s' is inbetween humanoid transforms and has rotation animation that will be discarded.\n", transformName.c_str());
                                    }
                                }

                                if (std::find_if(scaleCurves.begin(), scaleCurves.end(), HasAnimatedMuscleTransform<AnimationClip::Vector3Curve>(transformPath, 0.0001f)) != scaleCurves.end())
                                {
                                    clipWarning += Format("\t'%s' has scale animation that will be discarded.\n", transformName.c_str());
                                }
                            }

                            AnimationClip::Vector3Curves::iterator posEnd = std::remove_if(positionCurves.begin(), positionCurves.end(), HasPathPredicate<AnimationClip::Vector3Curve>(transformPath));
                            positionCurves.erase(posEnd, positionCurves.end());

                            AnimationClip::QuaternionCurves::iterator rotEnd = std::remove_if(quaternionCurves.begin(), quaternionCurves.end(), HasPathPredicate<AnimationClip::QuaternionCurve>(transformPath));
                            quaternionCurves.erase(rotEnd, quaternionCurves.end());

                            AnimationClip::Vector3Curves::iterator eulerEnd = std::remove_if(eulerCurves.begin(), eulerCurves.end(), HasPathPredicate<AnimationClip::Vector3Curve>(transformPath));
                            eulerCurves.erase(eulerEnd, eulerCurves.end());

                            AnimationClip::Vector3Curves::iterator scaleEnd = std::remove_if(scaleCurves.begin(), scaleCurves.end(), HasPathPredicate<AnimationClip::Vector3Curve>(transformPath));
                            scaleCurves.erase(scaleEnd, scaleCurves.end());
                        }
                    }
                }

                AnimationClip::FloatCurves& curves = animationClip->GetFloatCurves();

                for (int curveIter = curveStart; curveIter < curveCount; curveIter++)
                {
                    UnrollMuscles(curveArray[curveIter], curveIter, jobData->avatarConstant);

                    RecalculateSplineSlope(curveArray[curveIter]);

                    if (ShouldAddCurve(curveArray[curveIter], curveIter))
                    {
                        curves.push_back(AnimationClip::FloatCurve());
                        curves.back().type = TypeOf<Animator>();
                        curves.back().attribute = human_anim::animation::GetMuscleCurveName(curveIter).c_str();
                        curves.back().curve = curveArray[curveIter];
                    }
                }

                if (jobData->isHuman)
                {
                    if (clipWarning.length() > 0)
                    {
                        jobData->invalidAnimationWarning[i] += Format("\nClip '%s' has import animation warnings that might lower retargeting quality:\n", clipName.c_str());

                        if (!jobData->avatarConstant.m_Human->m_HasTDoF)
                        {
                            jobData->invalidAnimationWarning[i] += Format("Note: Activate translation DOF on avatar to improve retargeting quality.\n");
                        }

                        jobData->invalidAnimationWarning[i] += clipWarning;
                    }

                    if (jobData->doRetargetingQuality)
                    {
                        jobData->retargetingQualityMessages[i] += Format("\nRetargeting quality report for clip '%s':\n", clipName.c_str()).c_str();

                        if (clipRetargetWarning.length() > 0)
                        {
                            jobData->retargetingQualityMessages[i] += clipRetargetWarning.c_str();
                        }
                        else
                        {
                            jobData->retargetingQualityMessages[i] += "\tRetargeting quality is good. No significant differences with original animation were found.\n";
                        }
                    }
                }
            }

            human_anim::animation::DestroyClipOutput(out, jobData->alloc);
            human_anim::animation::DestroyClipMemory(mem, jobData->alloc);
        }

        human_anim::skeleton::DestroySkeletonPose(avatarLclPose, jobData->alloc);
        human_anim::skeleton::DestroySkeletonPose(avatarGblPose, jobData->alloc);

        if (jobData->isHuman)
        {
            human_anim::skeleton::DestroySkeletonPose(humanLclPose, jobData->alloc);
            human_anim::skeleton::DestroySkeletonPose(humanPoseA, jobData->alloc);
            human_anim::skeleton::DestroySkeletonPose(humanPoseB, jobData->alloc);
            human_anim::skeleton::DestroySkeletonPose(humanPoseC, jobData->alloc);
            human_anim::skeleton::DestroySkeletonPose(humanPoseD, jobData->alloc);
        }

        DestroyValueArray(valuesDefault, jobData->alloc);
        DestroyValueArray(values, jobData->alloc);
        DestroyValueArrayMask(valuesMask, jobData->alloc);
    }

    void GenerateMecanimClipCurveCombineJob(GenerateMecanimClipsCurvesJobData* jobData)
    {
        PROFILER_AUTO(gGenerateMecanimClipCurveCombineJob);

        for (dynamic_array<core::string>::const_iterator it = jobData->invalidAnimationWarning.begin(); it != jobData->invalidAnimationWarning.end(); ++it)
        {
            jobData->outInvalidAnimationWarning += *it;
        }

        for (dynamic_array<core::string>::const_iterator it = jobData->retargetingQualityMessages.begin(); it != jobData->retargetingQualityMessages.end(); ++it)
        {
            jobData->outRetargetingQualityMessages += *it;
        }
    }
} // namespace

core::string GenerateMecanimClipsCurves(AnimationClips const& clips,
    human_anim::animation::AvatarConstant const& avatarConstant,
    bool isHuman,
    HumanDescription const& humanDescription,
    GameObject& rootGameObject,
    AvatarBuilder::NamedTransforms const& namedTransform,
    core::string const& motionNodeName,
    bool doRetargetingQuality,
    core::string &outRetargetingQualityMessages,
    int humanoidOversampling)
{
    PROFILER_AUTO(gGenerateMecanimClipsCurves);

    if (clips.empty())
    {
        return "";
    }

    human_anim::int32_t motionTransformIndex = motionNodeName != "" ? motionNodeName == "<Root Transform>" ? 0 : human_anim::skeleton::SkeletonFindNode(avatarConstant.m_AvatarSkeleton.Get(), ComputeCRC32(motionNodeName.c_str())) : -1;

    GenerateMecanimClipsCurvesJobData jobData(clips, avatarConstant, isHuman, humanDescription, namedTransform, doRetargetingQuality, humanoidOversampling, motionTransformIndex);
    JobFence fence;

    // each iteration of the loop that we want to parallelize is quite heavy, so we just want to spread the load as much as possible over many threads
    const int minIterationsPerJob = 1;
    int jobCount = ConfigureBlockRangesWithMinIndicesPerJob(jobData.blocks, clips.size(), minIterationsPerJob);

    ScheduleJobForEach(fence, GenerateMecanimClipCurveJob, &jobData, jobCount, GenerateMecanimClipCurveCombineJob, kHighJobPriority);
    SyncFence(fence);

    outRetargetingQualityMessages = jobData.outRetargetingQualityMessages;

    return jobData.outInvalidAnimationWarning;
}

void OptimizeCurves(const AnimationClips& clips, int animationCompression, float animationRotationError, float animationPositionError, float animationScaleError)
{
    PROFILER_AUTO(gOptimizeCurves);

    if (clips.empty() || clips[0]->IsLegacy())
    {
        return;
    }

    dynamic_array<JobFence> fences(kMemTempAlloc);
    fences.resize_initialized(clips.size());

    for (size_t i = 0; i < clips.size(); ++i)
    {
        AnimationClip& clip = *clips[i];
        clip.SetUseHighQualityCurve(animationCompression != ModelImporter::kAnimationCompressionOptimal);

        if (animationCompression >= ModelImporter::kAnimationCompressionKeyframeReduction)
            fences[i] = ReduceKeyframes(clip, animationRotationError, animationPositionError, animationScaleError, animationPositionError);
    }

    SyncFences(fences.data(), fences.size());

    if (animationCompression >= ModelImporter::kAnimationCompressionKeyframeReductionAndCompression)
    {
        for (size_t i = 0; i < clips.size(); ++i)
        {
            AnimationClip& clip = *clips[i];
            clip.SetCompressionEnabled(true);
        }
    }
}
#endif
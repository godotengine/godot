#pragma once

#include "Modules/Animation/AvatarBuilder.h"
#include "ModelImporter.h"
#if 0
class AnimationClip;
struct ClipAnimationInfo;
struct ImportFloatAnimation;

void RemovedMaskedCurve(AnimationClip& clip, const ClipAnimationInfo& clipInfo, bool isClipOlderOr42 = false);
void AddAdditionalCurves(AnimationClip& clip, const ClipAnimationInfo& clipInfo);

AvatarType AnimationTypeToAvatarType(ModelImporter::AnimationType type);

core::string GenerateMecanimClipsCurves(AnimationClips const& clips, mecanim::animation::AvatarConstant const& avatarConstant, bool isHuman, HumanDescription const& humanDescription, GameObject& rootGameObject, AvatarBuilder::NamedTransforms const& namedTransform,
    core::string const& motionNodeName, bool doRetargetingQuality, core::string &retargetingQualityMessages, int humanoidOversampling);

void OffsetMuscles(AnimationClip& clip, mecanim::animation::AvatarConstant const& avatarConstant);

void OptimizeCurves(const AnimationClips& clips, int animationCompression, float animationRotationError, float animationPositionError, float animationScaleError);
#endif

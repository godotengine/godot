
#include "./skeleton.h"
#include "./human.h"
#include "./axes.h"

namespace human_anim
{
    // anonymous namespace to hide data in local file scope
namespace
{
    using namespace human;

    static const int32_t Bone2DoF[kLastBone][3] =
    {
        { -1, -1, -1 },                                                                                                         // kHips
        { kLeftLegDoFStart + kUpperLegFrontBack, kLeftLegDoFStart + kUpperLegInOut, kLeftLegDoFStart + kUpperLegRollInOut },          // kLeftUpperLeg
        { kRightLegDoFStart + kUpperLegFrontBack, kRightLegDoFStart + kUpperLegInOut, kRightLegDoFStart + kUpperLegRollInOut },       // kRightUpperLeg
        { kLeftLegDoFStart + kLegCloseOpen, -1, kLeftLegDoFStart + kLegRollInOut },                                                 // kLeftLeg
        { kRightLegDoFStart + kLegCloseOpen, -1, kRightLegDoFStart + kLegRollInOut },                                               // kRightLeg
        { kLeftLegDoFStart + kFootCloseOpen, kLeftLegDoFStart + kFootInOut, -1 },                                                   // kLeftFoot
        { kRightLegDoFStart + kFootCloseOpen, kRightLegDoFStart + kFootInOut, -1 },                                                 // kRightFoot
        { kBodyDoFStart + kSpineFrontBack, kBodyDoFStart + kSpineLeftRight, kBodyDoFStart + kSpineRollLeftRight },                    // kSpine
        { kBodyDoFStart + kChestFrontBack, kBodyDoFStart + kChestLeftRight, kBodyDoFStart + kChestRollLeftRight },                    // kChest
        { kBodyDoFStart + kUpperChestFrontBack, kBodyDoFStart + kUpperChestLeftRight, kBodyDoFStart + kUpperChestRollLeftRight },                 // kUpperChest
        { kHeadDoFStart + kNeckFrontBack, kHeadDoFStart + kNeckLeftRight, kHeadDoFStart + kNeckRollLeftRight },                       // kNeck
        { kHeadDoFStart + kHeadFrontBack, kHeadDoFStart + kHeadLeftRight, kHeadDoFStart + kHeadRollLeftRight },                       // kHead
        { kLeftArmDoFStart + kShoulderDownUp, kLeftArmDoFStart + kShoulderFrontBack, -1 },                                          // kLeftShoulder
        { kRightArmDoFStart + kShoulderDownUp, kRightArmDoFStart + kShoulderFrontBack, -1 },                                        // kRightShoulder
        { kLeftArmDoFStart + kArmDownUp, kLeftArmDoFStart + kArmFrontBack, kLeftArmDoFStart + kArmRollInOut },                        // kLeftArm
        { kRightArmDoFStart + kArmDownUp, kRightArmDoFStart + kArmFrontBack, kRightArmDoFStart + kArmRollInOut },                     // kRightArm
        { kLeftArmDoFStart + kForeArmCloseOpen, -1, kLeftArmDoFStart + kForeArmRollInOut },                                         // kLeftForeArm
        { kRightArmDoFStart + kForeArmCloseOpen, -1, kRightArmDoFStart + kForeArmRollInOut },                                       // kRightForeArm
        { kLeftArmDoFStart + kHandDownUp, kLeftArmDoFStart + kHandInOut, -1 },                                                       // kLeftHand
        { kRightArmDoFStart + kHandDownUp, kRightArmDoFStart + kHandInOut, -1 },                                                     // kRightHand
        { kLeftLegDoFStart + kToesUpDown, -1, -1},                                                                                // kLeftToes
        { kRightLegDoFStart + kToesUpDown, -1, -1 },                                                                              // kRightToes
        { kHeadDoFStart + kLeftEyeDownUp, kHeadDoFStart + kLeftEyeLeftRight, -1 },                                                   // LeftEye
        { kHeadDoFStart + kRightEyeDownUp, kHeadDoFStart + kRightEyeLeftRight, -1 },                                                 // RightEye
        { kHeadDoFStart + kJawDownUp, kHeadDoFStart + kJawLeftRight, -1 }                                                            // Jaw
    };

    static const float HumanBoneDefaultMass[kLastBone] =
    {
        12.0f,      // kHips
        10.0f,      // kLeftUpperLeg
        10.0f,      // kRightUpperLeg
        4.0f,       // kLeftLowerLeg
        4.0f,       // kRightLowerLeg
        0.8f,       // kLeftFoot
        0.8f,       // kRightFoot
        2.5f,       // kSpine
        12.0f,      // kChest
        12.0f,      // kUpperChest
        1.0f,       // kNeck
        4.0f,       // kHead
        0.5f,       // kLeftShoulder
        0.5f,       // kRightShoulder
        2.0f,       // kLeftUpperArm
        2.0f,       // kRightUpperArm
        1.5f,       // kLeftLowerArm
        1.5f,       // kRightLowerArm
        0.5f,       // kLeftHand
        0.5f,       // kRightHand
        0.2f,       // kLeftToes
        0.2f,       // kRightToes
        0.0f,       // LeftEye
        0.0f,       // RightEye
        0.0f        // Jaw
    };

    static const float BodyDoFMirror[kLastBodyDoF] =
    {
        +1.0f,  // kSpineFrontBack = 0,
        -1.0f,  // kSpineLeftRight,
        -1.0f,  // kSpineRollLeftRight,
        +1.0f,  // kChestFrontBack,
        -1.0f,  // kChestLeftRight,
        -1.0f,  // kChestRollLeftRight,
        +1.0f,  // kUpperChestFrontBack,
        -1.0f,  // kUpperChestLeftRight,
        -1.0f   // kUpperChestRollLeftRight,
    };

    static const float HeadDoFMirror[kLastHeadDoF] =
    {
        +1.0f,  // kNeckFrontBack = 0,
        -1.0f,  // kNeckLeftRight,
        -1.0f,  // kNeckRollLeftRight,
        +1.0f,  // kHeadFrontBack,
        -1.0f,  // kHeadLeftRight,
        -1.0f,  // kHeadRollLeftRight,
        +1.0f,  // kLeftEyeDownUp,
        -1.0f,  // kLeftEyeLeftRight,
        +1.0f,  // kRightEyeDownUp,
        -1.0f,  // kRightEyeLeftRight,
        +1.0f,  // kJawDownUp,
        -1.0f   // kJawLeftRight,
    };

    static const int32_t BoneChildren[kLastBone][4] =
    {
        { 3, kLeftUpperLeg, kRightUpperLeg, kSpine },// kHips
        { 1, kLeftLowerLeg },                                // kLeftUpperLeg
        { 1, kRightLowerLeg },                           // kRightUpperLeg
        { 1, kLeftFoot },                            // kLeftLowerLeg
        { 1, kRightFoot },                           // kRightLowerLeg
        { 1, kLeftToes },                            // kLeftFoot
        { 1, kRightToes },                           // kRightFoot
        { 1, kChest },                               // kSpine
        { 1, kUpperChest },                              // kChest
        { 3, kNeck, kLeftShoulder, kRightShoulder }, // kUpperChest
        { 1, kHead },                                // kNeck
        { 3, kLeftEye, kRightEye, kJaw },              // kHead
        { 1, kLeftUpperArm },                                // kLeftShoulder
        { 1, kRightUpperArm },                           // kRightShoulder
        { 1, kLeftLowerArm },                            // kLeftUpperArm
        { 1, kRightLowerArm },                       // kRightUpperArm
        { 1, kLeftHand },                            // kLeftLowerArm
        { 1, kRightHand },                           // kRightLowerArm
        { 0 },                                      // kLeftHand
        { 0 },                                      // kRightHand
        { 0 },                                      // kLeftToes
        { 0 },                                      // kRightToes
        { 0 },                                      // kLeftEye
        { 0 },                                      // kRightEye
        { 0 }                                       // kJaw
    };

    static const int32_t DoF2Bone[human::kLastDoF] =
    {
        kSpine,
        kSpine,
        kSpine,
        kChest,
        kChest,
        kChest,
        kUpperChest,
        kUpperChest,
        kUpperChest,
        kNeck,
        kNeck,
        kNeck,
        kHead,
        kHead,
        kHead,
        kLeftEye,
        kLeftEye,
        kRightEye,
        kRightEye,
        kJaw,
        kJaw,
        kLeftUpperLeg,
        kLeftUpperLeg,
        kLeftUpperLeg,
        kLeftLowerLeg,
        kLeftLowerLeg,
        kLeftFoot,
        kLeftFoot,
        kLeftToes,
        kRightUpperLeg,
        kRightUpperLeg,
        kRightUpperLeg,
        kRightLowerLeg,
        kRightLowerLeg,
        kRightFoot,
        kRightFoot,
        kRightToes,
        kLeftShoulder,
        kLeftShoulder,
        kLeftUpperArm,
        kLeftUpperArm,
        kLeftUpperArm,
        kLeftLowerArm,
        kLeftLowerArm,
        kLeftHand,
        kLeftHand,
        kRightShoulder,
        kRightShoulder,
        kRightUpperArm,
        kRightUpperArm,
        kRightUpperArm,
        kRightLowerArm,
        kRightLowerArm,
        kRightHand,
        kRightHand
    };

    static const int32_t DoF2BoneDoFIndex[human::kLastDoF] =
    {
        2, // kSpine,
        1, // kSpine,
        0, // kSpine,
        2, // kChest,
        1, // kChest,
        0, // kChest,
        2, // kUpperChest,
        1, // kUpperChest,
        0, // kUpperChest,
        2, // kNeck,
        1, // kNeck,
        0, // kNeck,
        2, // kHead,
        1, // kHead,
        0, // kHead,
        2, // kLeftEye,
        1, // kLeftEye,
        2, // kRightEye,
        1, // kRightEye,
        2, // kJaw,
        1, // kJaw,
        2, // kLeftUpperLeg,
        1, // kLeftUpperLeg,
        0, // kLeftUpperLeg,
        2, // kLeftLowerLeg,
        0, // kLeftLowerLeg,
        2, // kLeftFoot,
        1, // kLeftFoot,
        2, // kLeftToes,
        2, // kRightUpperLeg,
        1, // kRightUpperLeg,
        0, // kRightUpperLeg,
        2, // kRightLowerLeg,
        0, // kRightLowerLeg,
        2, // kRightFoot,
        1, // kRightFoot,
        2, // kRightToes,
        2, // kLeftShoulder,
        1, // kLeftShoulder,
        2, // kLeftUpperArm,
        1, // kLeftUpperArm,
        0, // kLeftUpperArm,
        2, // kLeftLowerArm,
        0, // kLeftLowerArm,
        2, // kLeftHand,
        1, // kLeftHand,
        2, // kRightShoulder,
        1, // kRightShoulder,
        2, // kRightUpperArm,
        1, // kRightUpperArm,
        0, // kRightUpperArm,
        2, // kRightLowerArm,
        0, // kRightLowerArm,
        2, // kRightHand,
        1, // kRightHand
    };

    static const int32_t TDoF2Bone[human::kLastTDoF] =
    {
        kSpine,             // kBodyTDoFStart + kSpineTDoF
        kChest,             // kBodyTDoFStart + kChestTDoF
        kUpperChest,        // kBodyTDoFStart + kUpperChestTDoF
        kNeck,              // kHeadTDoFStart + kNeckTDoF
        kHead,              // kHeadTDosStart + kHeadTDoF
        kLeftUpperLeg,      // kLeftLegTDoFStart + kUpperLegTDoF
        kLeftLowerLeg,      // kLeftLegTDoFStart + kLowerLegTDoF
        kLeftFoot,          // kLeftLegTDoFStart + kFootTDoF
        kLeftToes,          // kLeftLegTDoFStart + kToesTDoF
        kRightUpperLeg,     // kRightLegTDoFStart + kUpperLegTDoF
        kRightLowerLeg,     // kRightLegTDoFStart + kLowerLegTDoF
        kRightFoot,         // kRightLegTDoFStart + kFootTDoF
        kRightToes,         // kRightLegTDoFStart + kToesTDoF
        kLeftShoulder,      // kLeftArmTDoFStart + kShoulderTDoF
        kLeftUpperArm,      // kLeftArmTDoFStart + kUpperArmTDoF
        kLeftLowerArm,      // kLeftArmTDoFStart + kLowerArmTDoF
        kLeftHand,          // kLeftArmTDoFStart + kHandTDoF
        kRightShoulder,     // kRightArmTDoFStart + kShoulderTDoF
        kRightUpperArm,     // kRightArmTDoFStart + kUpperArmTDoF
        kRightLowerArm,     // kRightArmTDoFStart + kLowerArmTDoF
        kRightHand,         // kRightArmTDoFStart + kHandTDoF
    };

    static const int32_t BoneParent[kLastHumanBone] =
    {
        -1,
        kBodyBoneStart + kHips,
        kBodyBoneStart + kHips,
        kBodyBoneStart + kLeftUpperLeg,
        kBodyBoneStart + kRightUpperLeg,
        kBodyBoneStart + kLeftLowerLeg,
        kBodyBoneStart + kRightLowerLeg,
        kBodyBoneStart + kHips,
        kBodyBoneStart + kSpine,
        kBodyBoneStart + kChest,
        kBodyBoneStart + kUpperChest,
        kBodyBoneStart + kNeck,
        kBodyBoneStart + kUpperChest,
        kBodyBoneStart + kUpperChest,
        kBodyBoneStart + kLeftShoulder,
        kBodyBoneStart + kRightShoulder,
        kBodyBoneStart + kLeftUpperArm,
        kBodyBoneStart + kRightUpperArm,
        kBodyBoneStart + kLeftLowerArm,
        kBodyBoneStart + kRightLowerArm,
        kBodyBoneStart + kLeftFoot,
        kBodyBoneStart + kRightFoot,
        kBodyBoneStart + kHead,
        kBodyBoneStart + kHead,
        kBodyBoneStart + kHead,
        kBodyBoneStart + kLeftHand,
        kLeftThumbStart + hand::kProximal,
        kLeftThumbStart + hand::kIntermediate,
        kBodyBoneStart + kLeftHand,
        kLeftIndexStart + hand::kProximal,
        kLeftIndexStart + hand::kIntermediate,
        kBodyBoneStart + kLeftHand,
        kLeftMiddleStart + hand::kProximal,
        kLeftMiddleStart + hand::kIntermediate,
        kBodyBoneStart + kLeftHand,
        kLeftRingStart + hand::kProximal,
        kLeftRingStart + hand::kIntermediate,
        kBodyBoneStart + kLeftHand,
        kLeftLittleStart + hand::kProximal,
        kLeftLittleStart + hand::kIntermediate,
        kBodyBoneStart + kRightHand,
        kRightThumbStart + hand::kProximal,
        kRightThumbStart + hand::kIntermediate,
        kBodyBoneStart + kRightHand,
        kRightIndexStart + hand::kProximal,
        kRightIndexStart + hand::kIntermediate,
        kBodyBoneStart + kRightHand,
        kRightMiddleStart + hand::kProximal,
        kRightMiddleStart + hand::kIntermediate,
        kBodyBoneStart + kRightHand,
        kRightRingStart + hand::kProximal,
        kRightRingStart + hand::kIntermediate,
        kBodyBoneStart + kRightHand,
        kRightLittleStart + hand::kProximal,
        kRightLittleStart + hand::kIntermediate,
    };
}

namespace human
{
    bool RequiredBone(uint32_t aBoneIndex)
    {
        static bool requiredBone[kLastBone] =
        {
            true,   //kHips
            true,   //kLeftUpperLeg
            true,   //kRightUpperLeg
            true,   //kLeftLowerLeg
            true,   //kRightLowerLeg
            true,   //kLeftFoot
            true,   //kRightFoot
            true,   //kSpine
            false,  //kChest
            false,  //kUpperChest
            false,  //kNeck
            true,   //kHead
            false,  //kLeftShoulder
            false,  //kRightShoulder
            true,   //kLeftUpperArm
            true,   //kRightUpperArm
            true,   //kLeftLowerArm
            true,   //kRightLowerArm
            true,   //kLeftHand
            true,   //kRightHand
            false,  //kLeftToes
            false,  //kRightToes
            false,  //kLeftEye,
            false,  //kRightEye,
            false   //kJaw,
        };

        if (aBoneIndex < kLastBone)
            return requiredBone[aBoneIndex];

        return false;
    }

    const char* BoneName(uint32_t aBoneIndex)
    {
        static const char* boneName[kLastBone] =
        {
            "Hips",
            "LeftUpperLeg",
            "RightUpperLeg",
            "LeftLowerLeg",
            "RightLowerLeg",
            "LeftFoot",
            "RightFoot",
            "Spine",
            "Chest",
            "UpperChest",
            "Neck",
            "Head",
            "LeftShoulder",
            "RightShoulder",
            "LeftUpperArm",
            "RightUpperArm",
            "LeftLowerArm",
            "RightLowerArm",
            "LeftHand",
            "RightHand",
            "LeftToes",
            "RightToes",
            "LeftEye",
            "RightEye",
            "Jaw"
        };

        return boneName[aBoneIndex];
    }

    const char* MuscleName(uint32_t aBoneIndex)
    {
        static const char* muscleName[human::kLastDoF] =
        {
            "Spine Front-Back",
            "Spine Left-Right",
            "Spine Twist Left-Right",

            "Chest Front-Back",
            "Chest Left-Right",
            "Chest Twist Left-Right",

            "UpperChest Front-Back",
            "UpperChest Left-Right",
            "UpperChest Twist Left-Right",

            "Neck Nod Down-Up",
            "Neck Tilt Left-Right",
            "Neck Turn Left-Right",

            "Head Nod Down-Up",
            "Head Tilt Left-Right",
            "Head Turn Left-Right",

            "Left Eye Down-Up",
            "Left Eye In-Out",
            
            "Right Eye Down-Up",
            "Right Eye In-Out",

            "Jaw Close",
            "Jaw Left-Right",

            "Left Upper Leg Front-Back",
            "Left Upper Leg In-Out",
            "Left Upper Leg Twist In-Out",
            "Left Lower Leg Stretch",
            "Left Lower Leg Twist In-Out",
            "Left Foot Up-Down",
            "Left Foot Twist In-Out",
            "Left Toes Up-Down",

            "Right Upper Leg Front-Back",
            "Right Upper Leg In-Out",
            "Right Upper Leg Twist In-Out",
            "Right Lower Leg Stretch",
            "Right Lower Leg Twist In-Out",
            "Right Foot Up-Down",
            "Right Foot Twist In-Out",
            "Right Toes Up-Down",

            "Left Shoulder Down-Up",
            "Left Shoulder Front-Back",
            "Left Arm Down-Up",
            "Left Arm Front-Back",
            "Left Arm Twist In-Out",
            "Left Forearm Stretch",
            "Left Forearm Twist In-Out",
            "Left Hand Down-Up",
            "Left Hand In-Out",

            "Right Shoulder Down-Up",
            "Right Shoulder Front-Back",
            "Right Arm Down-Up",
            "Right Arm Front-Back",
            "Right Arm Twist In-Out",
            "Right Forearm Stretch",
            "Right Forearm Twist In-Out",
            "Right Hand Down-Up",
            "Right Hand In-Out"
        };

        return muscleName[aBoneIndex];
    }

    bool MaskHasLeftFootGoal(const HumanPoseMask& mask)
    {
        return mask.test(human_anim::human::kMaskGoalStartIndex + human_anim::human::kLeftFootGoal);
    }

    bool MaskHasRightFootGoal(const HumanPoseMask& mask)
    {
        return mask.test(human_anim::human::kMaskGoalStartIndex + human_anim::human::kRightFootGoal);
    }

    int32_t MuscleFromBone(int32_t aBoneIndex, int32_t aDoFIndex)
    {
        if (aBoneIndex < 0 || aBoneIndex >= kLastBone)
            return -1;

        if (aDoFIndex < 0 || aDoFIndex > 2)
            return -1;

        return Bone2DoF[aBoneIndex][2 - aDoFIndex];
    }

    int32_t BoneFromMuscle(int32_t aDoFIndex)
    {
        if (aDoFIndex >= human::kLastDoF)
            return -1;

        return DoF2Bone[aDoFIndex];
    }

    int32_t BoneFromTDoF(int32_t aTDoFIndex)
    {
        if (aTDoFIndex < 0 || aTDoFIndex >= human::kLastTDoF)
            return -1;

        return TDoF2Bone[aTDoFIndex];
    }

    int32_t BoneParentIndex(int32_t humanIndex)
    {
        return BoneParent[humanIndex];
    }

    int32_t GetValidBoneParentIndex(Human const *human, int32_t humanIndex)
    {
        int skBoneParentIndex = -1;

        humanIndex = BoneParentIndex(humanIndex);
        for (; humanIndex != -1; humanIndex = BoneParentIndex(humanIndex))
        {
            skBoneParentIndex = human->m_HumanBoneIndex[humanIndex];
            if (skBoneParentIndex != -1)
                break;
        }

        return humanIndex;
    }

    HumanPoseMask FullBodyMask()
    {
        return HumanPoseMask(~HumanPoseMask::type(0));
    }

    math::SetupAxesInfo const& GetAxeInfo(uint32_t index)
    {
        const static math::SetupAxesInfo setupAxesInfoArray[kLastBone] =
        {
            {{0, 0, 0, 1}, {-1, 0, 0, 0}, {-40, -40, -40}, {40, 40, 40}, {1, 1, 1, 1}, math::kZYRoll, 0},                      // kHips,
            {{-0.268f, 0, 0, 1}, {1, 0, 0, 0}, {-60, -60, -90}, {60, 60, 50}, {1, 1, 1, 1}, math::kZYRoll, 0},                 // kLeftUpperLeg,
            {{-0.268f, 0, 0, 1}, {1, 0, 0, 0}, {-60, -60, -90}, {60, 60, 50}, {-1, -1, 1, 1}, math::kZYRoll, 0},               // kRightUpperLeg,
            {{0.839f, 0, 0, 1}, {1, 0, 0, 0}, {-90, 0, -80}, {90, 0, 80}, {1, 1, -1, 1}, math::kZYRoll, 0},                    // kLeftLeg,
            {{0.839f, 0, 0, 1}, {1, 0, 0, 0}, {-90, 0, -80}, {90, 0, 80}, {-1, 1, -1, 1}, math::kZYRoll, 0},                   // kRightLeg,
            {{0, 0, 0, 1}, {1, 0, 0, 0}, {0, -30, -50}, {0, 30, 50}, {1, 1, 1, 1}, math::kZYRoll, -2},                         // kLeftFoot,
            {{0, 0, 0, 1}, {1, 0, 0, 0}, {0, -30, -50}, {0, 30, 50}, {1, -1, 1, 1}, math::kZYRoll, -2},                        // kRightFoot,
            {{0, 0, 0, 1}, {-1, 0, 0, 0}, {-40, -40, -40}, {40, 40, 40}, {1, 1, 1, 1}, math::kZYRoll, 0},                      // kSpine,
            {{0, 0, 0, 1}, {-1, 0, 0, 0}, {-40, -40, -40}, {40, 40, 40}, {1, 1, 1, 1}, math::kZYRoll, 0},                      // kChest,
            {{0, 0, 0, 1}, {-1, 0, 0, 0}, {-20, -20, -20}, {20, 20, 20}, {1, 1, 1, 1}, math::kZYRoll, 0},                      // kUpperChest,
            {{0, 0, 0, 1}, {-1, 0, 0, 0}, {-40, -40, -40}, {40, 40, 40}, {1, 1, 1, 1}, math::kZYRoll, 0},                      // kNeck,
            {{0, 0, 0, 1}, {-1, 0, 0, 0}, {-40, -40, -40}, {40, 40, 40}, {1, 1, 1, 1}, math::kZYRoll, 2},                      // kHead,
            {{0, 0, 0, 1}, {0, 0, 1, 0}, {0, -15, -15}, {0, 15, 30}, {1, 1, -1, 1}, math::kZYRoll, 0},                         // kLeftShoulder,
            {{0, 0, 0, 1}, {0, 0, 1, 0}, {0, -15, -15}, {0, 15, 30}, {1, 1, 1, 1}, math::kZYRoll, 0},                          // kRightShoulder,
            {{0, 0.268f, 0.364f, 1}, {0, 0, 1, 0}, {-90, -100, -60}, {90, 100, 100}, {1, 1, -1, 1}, math::kZYRoll, 0},         // kLeftArm,
            {{0, -0.268f, -0.364f, 1}, {0, 0, 1, 0}, {-90, -100, -60}, {90, 100, 100}, {-1, 1, 1, 1}, math::kZYRoll, 0},       // kRightArm,
            {{0, 0.839f, 0, 1}, {0, 1, 0, 0}, {-90, 0, -80}, {90, 0, 80}, {1, 1, -1, 1}, math::kZYRoll, 0},                    // kLeftForeArm,
            {{0, -0.839f, 0, 1}, {0, 1, 0, 0}, {-90, 0, -80}, {90, 0, 80}, {-1, 1, 1, 1}, math::kZYRoll, 0},                   // kRightForeArm,
            {{0, 0, 0, 1}, {0, 0, 1, 0}, {0, -40, -80}, {0, 40, 80}, {1, 1, -1, 1}, math::kZYRoll, 0},                         // kLeftHand,
            {{0, 0, 0, 1}, {0, 0, 1, 0}, {0, -40, -80}, {0, 40, 80}, {1, 1, 1, 1}, math::kZYRoll, 0},                          // kRightHand,
            {{0, 0, 0, 1}, {1, 0, 0, 0}, {0, 0, -50}, {0, 0, 50}, {1, 1, 1, 1}, math::kZYRoll, 3},                             // kLeftToes,
            {{0, 0, 0, 1}, {1, 0, 0, 0}, {0, 0, -50}, {0, 0, 50}, {1, 1, 1, 1}, math::kZYRoll, 3},                             // kRightToes,
            {{0, 0, 0, 1}, {1, 0, 0, 0}, {0, -20, -10}, {0, 20, 15}, {1, 1, -1, 1}, math::kZYRoll, 3},                         // kLeftEye,
            {{0, 0, 0, 1}, {1, 0, 0, 0}, {0, -20, -10}, {0, 20, 15}, {1, -1, -1, 1}, math::kZYRoll, 3},                        // kRightEye,
            {{0.09f, 0, 0, 1}, {1, 0, 0, 0}, {0, -10, -10}, {0, 10, 10}, {1, 1, -1, 1}, math::kZYRoll, 3}                      // kJaw,
        };

        return setupAxesInfoArray[index];
    }

    Human::Human() :
        m_RootX(math::trsIdentity()),
        m_Scale(1),
        m_ArmTwist(0.5f),
        m_ForeArmTwist(0.5f),
        m_UpperLegTwist(0.5f),
        m_LegTwist(0.5f),
        m_ArmStretch(0.05f),
        m_LegStretch(0.05f),
        m_FeetSpacing(0.0f),
        m_HasLeftHand(false),
        m_HasRightHand(false),
        m_HasTDoF(false)
    {
        int32_t i;

        float mass = 0;

        for (i = 0; i < kLastBone; i++)
        {
            m_HumanBoneIndex[i] = -1;
            m_HumanBoneMass[i] = HumanBoneDefaultMass[i];
            mass += m_HumanBoneMass[i];
        }

        for (i = 0; i < kLastBone; i++)
        {
            m_HumanBoneMass[i] /= mass;
        }
    }

    Human* CreateHuman(skeleton::HumanSkeleton *apSkeleton, skeleton::SkeletonPose *apSkeletonPose, RuntimeBaseAllocator& arAlloc)
    {
        Human* human = memnew(Human);

        human->m_Skeleton = apSkeleton;
        human->m_SkeletonPose = apSkeletonPose->m_X;

        memset(human->m_HumanBoneIndex, -1, sizeof(int32_t) * kLastBone);

        human->m_HasLeftHand = false;
        human->m_HasRightHand = false;

        human->m_HasTDoF = false;

        human->m_Scale = 1;

        return human;
    }

    void DestroyHuman(Human *apHuman, RuntimeBaseAllocator& arAlloc)
    {
        if (apHuman)
        {
            human_anim::skeleton::DestroySkeleton(apHuman->m_Skeleton, arAlloc);

            human_anim::hand::DestroyHand(apHuman->m_LeftHand, arAlloc);
            human_anim::hand::DestroyHand(apHuman->m_RightHand, arAlloc);

            memdelete(apHuman);
        }
    }

    HumanPose::HumanPose() : m_RootX(math::trsIdentity())
    {
        int32_t i;

        for (i = 0; i < kLastDoF; i++)
        {
            m_DoFArray[i] = 0;
        }

        for (i = 0; i < kLastTDoF; i++)
        {
            m_TDoFArray[i] = math::float3(math::ZERO);
        }

        m_LookAtPosition = math::float3(math::ZERO);
        m_LookAtWeight = math::float4(math::ZERO);
    }

    void HumanAdjustMass(Human *apHuman)
    {
        if (apHuman->m_HumanBoneIndex[kNeck] < 0)
        {
            apHuman->m_HumanBoneMass[kUpperChest] += apHuman->m_HumanBoneMass[kNeck];
            apHuman->m_HumanBoneMass[kNeck] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kLeftShoulder] < 0)
        {
            apHuman->m_HumanBoneMass[kUpperChest] += apHuman->m_HumanBoneMass[kLeftShoulder];
            apHuman->m_HumanBoneMass[kLeftShoulder] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kRightShoulder] < 0)
        {
            apHuman->m_HumanBoneMass[kUpperChest] += apHuman->m_HumanBoneMass[kRightShoulder];
            apHuman->m_HumanBoneMass[kRightShoulder] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kUpperChest] < 0)
        {
            apHuman->m_HumanBoneMass[kChest] += apHuman->m_HumanBoneMass[kUpperChest];
            apHuman->m_HumanBoneMass[kUpperChest] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kChest] < 0)
        {
            apHuman->m_HumanBoneMass[kSpine] += apHuman->m_HumanBoneMass[kChest];
            apHuman->m_HumanBoneMass[kChest] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kLeftToes] < 0)
        {
            apHuman->m_HumanBoneMass[kLeftFoot] += apHuman->m_HumanBoneMass[kLeftToes];
            apHuman->m_HumanBoneMass[kLeftToes] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kRightToes] < 0)
        {
            apHuman->m_HumanBoneMass[kRightFoot] += apHuman->m_HumanBoneMass[kRightToes];
            apHuman->m_HumanBoneMass[kRightToes] = 0;
        }
    }

    math::Axes GetAxes(Human const *apHuman, int32_t aBoneIndex)
    {
        math::Axes ret;

        int32_t skIndex = apHuman->m_HumanBoneIndex[aBoneIndex];

        if (skIndex >= 0)
        {
            int32_t axesIndex = apHuman->m_Skeleton->m_Node[skIndex].m_AxesId;

            if (axesIndex >= 0)
            {
                ret = apHuman->m_Skeleton->m_AxesArray[axesIndex];
            }
        }

        return ret;
    }

	void SkeletonPoseComputeGlobal(skeleton::HumanSkeleton const* apSkeleton, math::trsX* apLocalPose, skeleton::SkeletonPoseT<math::trsX>* apGlobalPose, int aIndex, int aStopIndex)
	{
		math::trsX const* local = apLocalPose;
		math::trsX* global = apGlobalPose->m_X.ptr();

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
	void SkeletonPoseComputeLocal(skeleton::HumanSkeleton const* apSkeleton, skeleton::SkeletonPoseT<math::trsX> const* apGlobalPose, math::trsX* apLocalPose)
	{
		uint32_t nodeCount = apSkeleton->m_Count;
		skeleton::Node const* node = apSkeleton->m_Node.ptr();
		math::trsX const* global = apGlobalPose->m_X.ptr();
		math::trsX* local = apLocalPose;

		for (uint32_t nodeIter = 1; nodeIter < nodeCount; nodeIter++)
		{
			local[nodeIter] = math::invMul(global[node[nodeIter].m_ParentId], global[nodeIter]);
		}

		local[0] = global[0];
	}

	void SkeletonPoseComputeLocal(skeleton::HumanSkeleton const* apSkeleton, skeleton::SkeletonPoseT<math::trsX> const* apGlobalPose, math::trsX* apLocalPose, int aIndex, int aStopIndex)
	{
		math::trsX const* global = apGlobalPose->m_X.ptr();
		math::trsX* local = apLocalPose;

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


	void SkeletonPoseComputeGlobal(skeleton::HumanSkeleton const* apSkeleton, math::trsX const* apLocalPose, skeleton::SkeletonPoseT<math::trsX>* apGlobalPose)
	{
		uint32_t nodeCount = apSkeleton->m_Count;
		skeleton::Node const* node = apSkeleton->m_Node.ptr();
		math::trsX const* local = apLocalPose;
		math::trsX* global = apGlobalPose->m_X.ptr();

		global[0] = local[0];

		for (uint32_t nodeIter = 1; nodeIter < nodeCount; nodeIter++)
		{
			global[nodeIter] = math::mul(global[node[nodeIter].m_ParentId], local[nodeIter]);
		}
	}

    void HumanSetupAxes(Human *apHuman, skeleton::SkeletonPose const *apSkeletonPoseGlobal)
    {
        apHuman->m_RootX = math::trsIdentity();
        apHuman->m_RootX = HumanComputeRootXform(apHuman, apSkeletonPoseGlobal);
        apHuman->m_Scale = apHuman->m_RootX.t.y;

        SkeletonPoseComputeLocal(apHuman->m_Skeleton, apSkeletonPoseGlobal, apHuman->m_SkeletonPose.ptr());

        int32_t i;

        for (i = 0; i < kLastBone; i++)
        {
            int32_t skBoneIndex = apHuman->m_HumanBoneIndex[i];

            int32_t skAxisBoneId = -1;
            float len = 1.0f;

            switch (i)
            {
                case kLeftEye:
                case kRightEye:
                case kJaw:
                    len = 0.1f;
                    break;

                case kHead:
                    if (apHuman->m_HumanBoneIndex[kNeck] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kNeck];
                        len = -1.0f;
                    }
                    else if (apHuman->m_HumanBoneIndex[kChest] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kChest];
                        len = -0.5f;
                    }
                    else
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kSpine];
                        len = -0.25f;
                    }
                    break;

                case kLeftFoot:
                    len = -apSkeletonPoseGlobal->m_X[skBoneIndex].t.y;
                    break;

                case kRightFoot:
                    len = -apSkeletonPoseGlobal->m_X[skBoneIndex].t.y;
                    break;

                case kLeftHand:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kLeftLowerArm];
                    len = -0.5f;
                    break;

                case kRightHand:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kRightLowerArm];
                    len = -0.5f;
                    break;

                case kLeftToes:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kLeftFoot];
                    len = 0.5f;
                    break;

                case kRightToes:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kRightFoot];
                    len = 0.5f;
                    break;

                case kHips:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kSpine];
                    break;

                case kLeftUpperLeg:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kLeftLowerLeg];
                    break;

                case kRightUpperLeg:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kRightLowerLeg];
                    break;

                case kLeftLowerLeg:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kLeftFoot];
                    break;

                case kRightLowerLeg:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kRightFoot];
                    break;

                case kSpine:
                    if (apHuman->m_HumanBoneIndex[kChest] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kChest];
                    }
                    else if (apHuman->m_HumanBoneIndex[kUpperChest] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kUpperChest];
                    }
                    else if (apHuman->m_HumanBoneIndex[kNeck] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kNeck];
                    }
                    else
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kHead];
                    }
                    break;

                case kChest:
                    if (apHuman->m_HumanBoneIndex[kUpperChest] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kUpperChest];
                    }
                    else if (apHuman->m_HumanBoneIndex[kNeck] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kNeck];
                    }
                    else
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kHead];
                    }
                    break;

                case kUpperChest:
                    if (apHuman->m_HumanBoneIndex[kNeck] >= 0)
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kNeck];
                    }
                    else
                    {
                        skAxisBoneId = apHuman->m_HumanBoneIndex[kHead];
                    }
                    break;

                case kNeck:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kHead];
                    break;

                case kLeftShoulder:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kLeftUpperArm];
                    break;

                case kRightShoulder:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kRightUpperArm];
                    break;

                case kLeftUpperArm:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kLeftLowerArm];
                    break;

                case kRightUpperArm:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kRightLowerArm];
                    break;

                case kLeftLowerArm:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kLeftHand];
                    break;

                case kRightLowerArm:
                    skAxisBoneId = apHuman->m_HumanBoneIndex[kRightHand];
                    break;
            }

            if (skBoneIndex >= 0)
            {
                skeleton::SetupAxes(apHuman->m_Skeleton, apSkeletonPoseGlobal, GetAxeInfo(i), skBoneIndex, skAxisBoneId, true, len);
            }
        }
    }

    void HumanCopyAxes(Human const *apSrcHuman, Human *apHuman)
    {
        int32_t i;

        for (i = 0; i < kLastBone; i++)
        {
            skeleton::Node const * srcNode = apSrcHuman->m_HumanBoneIndex[i] >= 0 ? &apSrcHuman->m_Skeleton->m_Node[apSrcHuman->m_HumanBoneIndex[i]] : 0;
            skeleton::Node const * node = apHuman->m_HumanBoneIndex[i] >= 0 ? &apHuman->m_Skeleton->m_Node[apHuman->m_HumanBoneIndex[i]] : 0;

            if (srcNode != 0 && node != 0 && srcNode->m_AxesId != -1 && node->m_AxesId != -1)
            {
                apHuman->m_Skeleton->m_AxesArray[node->m_AxesId] = apSrcHuman->m_Skeleton->m_AxesArray[srcNode->m_AxesId];
            }
        }
    }

    void GetMuscleRange(Human const *apHuman, int32_t aDoFIndex, float &aMin, float &aMax)
    {
        math::Axes const& axes = GetAxes(apHuman, DoF2Bone[aDoFIndex]);

        switch (DoF2BoneDoFIndex[aDoFIndex])
        {
            case 0: aMin = axes.m_Limit.m_Min.x; aMax = axes.m_Limit.m_Max.x; break;
            case 1: aMin = axes.m_Limit.m_Min.y; aMax = axes.m_Limit.m_Max.y; break;
            case 2: aMin = axes.m_Limit.m_Min.z; aMax = axes.m_Limit.m_Max.z; break;
        }
    }

    math::float4 AddAxis(Human const *apHuman, int32_t aIndex, math::float4 const &arQ)
    {
        math::Axes cAxes = apHuman->m_Skeleton->m_AxesArray[apHuman->m_Skeleton->m_Node[aIndex].m_AxesId];
        return math::normalize(math::quatMul(arQ, cAxes.m_PostQ));
    }

    math::float4 RemoveAxis(Human const *apHuman, int32_t aIndex, const math::float4 &arQ)
    {
        math::Axes cAxes = apHuman->m_Skeleton->m_AxesArray[apHuman->m_Skeleton->m_Node[aIndex].m_AxesId];
        return math::normalize(math::quatMul(arQ, math::quatConj(cAxes.m_PostQ)));
    }

    void HumanPoseAdjustForMissingBones(Human const *apHuman, HumanPose *apHumanPose)
    {
        if (apHuman->m_HumanBoneIndex[kNeck] < 0)
        {
            apHumanPose->m_DoFArray[kHeadDoFStart + kHeadFrontBack] += apHumanPose->m_DoFArray[kHeadDoFStart + kNeckFrontBack];
            apHumanPose->m_DoFArray[kHeadDoFStart + kNeckFrontBack] = 0;

            apHumanPose->m_DoFArray[kHeadDoFStart + kHeadLeftRight] += apHumanPose->m_DoFArray[kHeadDoFStart + kNeckLeftRight];
            apHumanPose->m_DoFArray[kHeadDoFStart + kNeckLeftRight] = 0;

            apHumanPose->m_DoFArray[kHeadDoFStart + kHeadRollLeftRight] += apHumanPose->m_DoFArray[kHeadDoFStart + kNeckRollLeftRight];
            apHumanPose->m_DoFArray[kHeadDoFStart + kNeckRollLeftRight] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kUpperChest] < 0)
        {
            apHumanPose->m_DoFArray[kBodyDoFStart + kChestFrontBack] += (20.0f / 40.0f) * apHumanPose->m_DoFArray[kBodyDoFStart + kUpperChestFrontBack];
            apHumanPose->m_DoFArray[kBodyDoFStart + kUpperChestFrontBack] = 0;

            apHumanPose->m_DoFArray[kBodyDoFStart + kChestLeftRight] += (20.0f / 40.0f) * apHumanPose->m_DoFArray[kBodyDoFStart + kUpperChestLeftRight];
            apHumanPose->m_DoFArray[kBodyDoFStart + kUpperChestLeftRight] = 0;

            apHumanPose->m_DoFArray[kBodyDoFStart + kChestRollLeftRight] += (20.0f / 40.0f) * apHumanPose->m_DoFArray[kBodyDoFStart + kUpperChestRollLeftRight];
            apHumanPose->m_DoFArray[kBodyDoFStart + kUpperChestRollLeftRight] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kChest] < 0)
        {
            apHumanPose->m_DoFArray[kBodyDoFStart + kSpineFrontBack] += apHumanPose->m_DoFArray[kBodyDoFStart + kChestFrontBack];
            apHumanPose->m_DoFArray[kBodyDoFStart + kChestFrontBack] = 0;

            apHumanPose->m_DoFArray[kBodyDoFStart + kSpineLeftRight] += apHumanPose->m_DoFArray[kBodyDoFStart + kChestLeftRight];
            apHumanPose->m_DoFArray[kBodyDoFStart + kChestLeftRight] = 0;

            apHumanPose->m_DoFArray[kBodyDoFStart + kSpineRollLeftRight] += apHumanPose->m_DoFArray[kBodyDoFStart + kChestRollLeftRight];
            apHumanPose->m_DoFArray[kBodyDoFStart + kChestRollLeftRight] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kLeftShoulder] < 0)
        {
            apHumanPose->m_DoFArray[kLeftArmDoFStart + kArmDownUp] += (30.0f / 200.0f) * apHumanPose->m_DoFArray[kLeftArmDoFStart + kShoulderDownUp];
            apHumanPose->m_DoFArray[kLeftArmDoFStart + kShoulderDownUp] = 0;

            apHumanPose->m_DoFArray[kLeftArmDoFStart + kArmFrontBack] += (45.0f / 160.0f) * apHumanPose->m_DoFArray[kLeftArmDoFStart + kShoulderFrontBack];
            apHumanPose->m_DoFArray[kLeftArmDoFStart + kShoulderFrontBack] = 0;
        }

        if (apHuman->m_HumanBoneIndex[kRightShoulder] < 0)
        {
            apHumanPose->m_DoFArray[kRightArmDoFStart + kArmDownUp] += (30.0f / 200.0f) * apHumanPose->m_DoFArray[kRightArmDoFStart + kShoulderDownUp];
            apHumanPose->m_DoFArray[kRightArmDoFStart + kShoulderDownUp] = 0;

            apHumanPose->m_DoFArray[kRightArmDoFStart + kArmFrontBack] += (45.0f / 160.0f) * apHumanPose->m_DoFArray[kRightArmDoFStart + kShoulderFrontBack];
            apHumanPose->m_DoFArray[kRightArmDoFStart + kShoulderFrontBack] = 0;
        }
    }

    void Human2SkeletonPose(Human const *apHuman, HumanPose const *apHumanPose, skeleton::SkeletonPose *apSkeletonPose, int32_t i)
    {
        if (apHuman->m_HumanBoneIndex[i] != -1)
        {
            math::int3 mask = math::int3(-(Bone2DoF[i][2] != -1), -(Bone2DoF[i][1] != -1), -(Bone2DoF[i][0] != -1));
            math::float3 xyz = math::select(math::float3(math::ZERO),
                math::float3(apHumanPose->m_DoFArray[Bone2DoF[i][2]], apHumanPose->m_DoFArray[Bone2DoF[i][1]], apHumanPose->m_DoFArray[Bone2DoF[i][0]]),
                mask);

            skeleton::SkeletonSetDoF(apHuman->m_Skeleton, apSkeletonPose, xyz, apHuman->m_HumanBoneIndex[i]);
        }
    }

    void Human2SkeletonPose(Human const *apHuman, HumanPose const *apHumanPose, skeleton::SkeletonPose *apSkeletonPose)
    {
        int32_t i;
        for (i = 1; i < kLastBone; i++)
        {
            Human2SkeletonPose(apHuman, apHumanPose, apSkeletonPose, i);
        }

        if (apHuman->m_HasLeftHand)
        {
            hand::Hand2SkeletonPose(apHuman->m_LeftHand, apHuman->m_Skeleton, &apHumanPose->m_LeftHandPose, apSkeletonPose);
        }

        if (apHuman->m_HasRightHand)
        {
            hand::Hand2SkeletonPose(apHuman->m_RightHand, apHuman->m_Skeleton, &apHumanPose->m_RightHandPose, apSkeletonPose);
        }
    }

    void Skeleton2HumanPose(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPose, HumanPose *apHumanPose, int32_t i)
    {
        if (apHuman->m_HumanBoneIndex[i] != -1)
        {
            const math::float3 xyz = skeleton::SkeletonGetDoF(apHuman->m_Skeleton, apSkeletonPose, apHuman->m_HumanBoneIndex[i]);

            if (Bone2DoF[i][2] != -1)
                apHumanPose->m_DoFArray[Bone2DoF[i][2]] = xyz.x;
            if (Bone2DoF[i][1] != -1)
                apHumanPose->m_DoFArray[Bone2DoF[i][1]] = xyz.y;
            if (Bone2DoF[i][0] != -1)
                apHumanPose->m_DoFArray[Bone2DoF[i][0]] = xyz.z;
        }
    }

    void Skeleton2HumanPose(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPose, HumanPose *apHumanPose)
    {
        int32_t i;

        for (i = 1; i < kLastBone; i++)
        {
            Skeleton2HumanPose(apHuman, apSkeletonPose, apHumanPose, i);
        }

        if (apHuman->m_HasLeftHand)
        {
            hand::Skeleton2HandPose(apHuman->m_LeftHand, apHuman->m_Skeleton, apSkeletonPose, &apHumanPose->m_LeftHandPose);
        }

        if (apHuman->m_HasRightHand)
        {
            hand::Skeleton2HandPose(apHuman->m_RightHand, apHuman->m_Skeleton, apSkeletonPose, &apHumanPose->m_RightHandPose);
        }
    }

    math::float3 HumanComputeBoneMassCenter(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPose, int32_t aBoneIndex)
    {
        math::float3 ret(math::ZERO);

        switch (aBoneIndex)
        {
            case kHips:
                ret = math::float1(1.0f / 3.0f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperLeg]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightUpperLeg]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kSpine]].t);
                break;

            case kSpine:
                if (apHuman->m_HumanBoneIndex[kChest] >= 0)
                {
                    ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kSpine]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kChest]].t);
                }
                else
                {
                    ret = math::float1(0.1f) * apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kSpine]].t + math::float1(0.9f * 0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperArm]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightUpperArm]].t);
                }
                break;

            case kChest:
                if (apHuman->m_HumanBoneIndex[kUpperChest] >= 0)
                {
                    ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kChest]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kUpperChest]].t);
                }
                else if (apHuman->m_HumanBoneIndex[kNeck] >= 0 && apHuman->m_HumanBoneIndex[kLeftShoulder] >= 0 && apHuman->m_HumanBoneIndex[kRightShoulder] >= 0)
                {
                    ret = math::float1(0.25f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kChest]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kNeck]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftShoulder]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightShoulder]].t);
                }
                else
                {
                    ret = math::float1(1.0f / 3.0f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kChest]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperArm]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightUpperArm]].t);
                }
                break;

            case kUpperChest:
                if (apHuman->m_HumanBoneIndex[kNeck] >= 0 && apHuman->m_HumanBoneIndex[kLeftShoulder] >= 0 && apHuman->m_HumanBoneIndex[kRightShoulder] >= 0)
                {
                    ret = math::float1(0.25f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kUpperChest]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kNeck]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftShoulder]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightShoulder]].t);
                }
                else
                {
                    ret = math::float1(1.0f / 3.0f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kUpperChest]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperArm]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightUpperArm]].t);
                }
                break;

            case kNeck:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kNeck]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kHead]].t);
                break;

            case kLeftUpperLeg:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperLeg]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftLowerLeg]].t);
                break;

            case kLeftLowerLeg:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftLowerLeg]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftFoot]].t);
                break;

            case kLeftShoulder:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftShoulder]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperArm]].t);
                break;

            case kLeftUpperArm:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperArm]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftLowerArm]].t);
                break;

            case kLeftLowerArm:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftUpperArm]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kLeftHand]].t);
                break;

            case kRightUpperLeg:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightUpperLeg]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightLowerLeg]].t);
                break;

            case kRightLowerLeg:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightLowerLeg]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightFoot]].t);
                break;

            case kRightShoulder:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightShoulder]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightUpperArm]].t);
                break;

            case kRightUpperArm:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightUpperArm]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightLowerArm]].t);
                break;

            case kRightLowerArm:
                ret = math::float1(0.5f) * (apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightLowerArm]].t + apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[kRightHand]].t);
                break;

            default:
                ret = apSkeletonPose->m_X[apHuman->m_HumanBoneIndex[aBoneIndex]].t;
                break;
        }

        return ret;
    }

    math::float3 HumanComputeMassCenter(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPoseGlobal)
    {
        math::float3 ret(math::ZERO);

        int32_t i;

        float mass = 0;

        for (i = 0; i < kLastBone; i++)
        {
            int32_t index = apHuman->m_HumanBoneIndex[i];

            if (index >= 0)
            {
                float boneMass = apHuman->m_HumanBoneMass[i];
                ret += HumanComputeBoneMassCenter(apHuman, apSkeletonPoseGlobal, i) * math::float1(boneMass);
                mass += boneMass;
            }
        }

        return ret / math::float1(mass);
    }

    float HumanComputeMomentumOfInertia(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPoseGlobal)
    {
        float ret = 0;

        math::float3 mc = HumanComputeMassCenter(apHuman, apSkeletonPoseGlobal);

        int32_t i;

        for (i = 0; i < kLastBone; i++)
        {
            int32_t index = apHuman->m_HumanBoneIndex[i];

            if (index >= 0)
            {
                float r = math::length(HumanComputeBoneMassCenter(apHuman, apSkeletonPoseGlobal, index) - mc);
                ret += apHuman->m_HumanBoneMass[i] * r * r;
            }
        }

        return ret;
    }

    math::float4 HumanComputeOrientation(Human const* apHuman, skeleton::SkeletonPose const* apPoseGlobal)
    {
        int32_t llIndex = apHuman->m_HumanBoneIndex[kLeftUpperLeg];
        int32_t rlIndex = apHuman->m_HumanBoneIndex[kRightUpperLeg];

        int32_t laIndex = apHuman->m_HumanBoneIndex[kLeftUpperArm];
        int32_t raIndex = apHuman->m_HumanBoneIndex[kRightUpperArm];

        math::float3 legMC = math::float1(0.5f) * (apPoseGlobal->m_X[llIndex].t + apPoseGlobal->m_X[rlIndex].t);
        math::float3 armMC = math::float1(0.5f) * (apPoseGlobal->m_X[laIndex].t + apPoseGlobal->m_X[raIndex].t);

        math::float3 upV = math::normalize(armMC - legMC);

        math::float3 legV = apPoseGlobal->m_X[rlIndex].t - apPoseGlobal->m_X[llIndex].t;
        math::float3 armV = apPoseGlobal->m_X[raIndex].t - apPoseGlobal->m_X[laIndex].t;

        math::float3 rightV = math::normalize(legV + armV);
        math::float3 frontV = math::cross(rightV, upV);

        rightV = math::cross(upV, frontV);

        return math::normalize(math::quatMul(math::matrixToQuat(rightV, upV, frontV), math::quatConj(apHuman->m_RootX.q)));
    }

    math::trsX HumanComputeRootXform(Human const* apHuman, skeleton::SkeletonPose const* apPoseGlobal)
    {
        return math::trsX(HumanComputeMassCenter(apHuman, apPoseGlobal), HumanComputeOrientation(apHuman, apPoseGlobal), math::float3(1.0f));
    }

    float HumanGetFootHeight(Human const* apHuman, bool aLeft)
    {
        return apHuman->m_Skeleton->m_AxesArray[apHuman->m_Skeleton->m_Node[apHuman->m_HumanBoneIndex[aLeft ? kLeftFoot : kRightFoot]].m_AxesId].m_Length;
    }

    math::float3 HumanGetFootBottom(Human const* apHuman, bool aLeft)
    {
        return math::float3(HumanGetFootHeight(apHuman, aLeft), 0, 0);
    }

    math::float4 HumanGetGoalOrientationOffset(Goal goalIndex)
    {
        const math::float4 goalOrientationOffsetArray[kLastGoal] = { math::float4(0.5f, -0.5f, 0.5f, 0.5f), math::float4(0.5f, -0.5f, 0.5f, 0.5f), math::float4(0.707107f, 0, 0.707107f, 0), math::float4(0, 0.707107f, 0, 0.707107f)};
        return goalOrientationOffsetArray[goalIndex];
    }

    math::float3 HumanGetHintPosition(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPoseGlobal, Goal goalIndex)
    {
        int midIndex = apHuman->m_HumanBoneIndex[s_HumanGoalInfo[goalIndex].m_MidIndex];

        if (goalIndex <= kRightFootGoal)
        {
            int endIndex = apHuman->m_HumanBoneIndex[s_HumanGoalInfo[goalIndex].m_EndIndex];
            math::float3 footFrontU = math::quatYcos(AddAxis(apHuman, endIndex, apSkeletonPoseGlobal->m_X[endIndex].q));
            return apSkeletonPoseGlobal->m_X[midIndex].t - math::float1(0.25f) * apHuman->m_Scale * footFrontU;
        }
        else
        {
            return apSkeletonPoseGlobal->m_X[midIndex].t;
        }
    }

    math::trsX HumanGetGoalXform(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPoseGlobal, Goal goalIndex)
    {
        math::trsX goalX;
        int32_t index = apHuman->m_HumanBoneIndex[s_HumanGoalInfo[goalIndex].m_Index];
        goalX.t = apSkeletonPoseGlobal->m_X[index].t;
        goalX.q = AddAxis(apHuman, index, apSkeletonPoseGlobal->m_X[index].q);
        goalX.s = math::float3(1.f);

        if (goalIndex < 2)
            goalX.t = math::mul(goalX, human::HumanGetFootBottom(apHuman, goalIndex == 0));

        return goalX;
    }

    math::float3 HumanGetGoalPosition(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPoseLocal, Goal goalIndex)
    {
        int32_t index = apHuman->m_HumanBoneIndex[s_HumanGoalInfo[goalIndex].m_Index];

        math::trsX goalX = skeleton::SkeletonGetGlobalX(apHuman->m_Skeleton, apSkeletonPoseLocal, index);
        goalX.q = AddAxis(apHuman, index, goalX.q);
        goalX.s = math::float3(1.f);

        return goalX.t;
    }

    math::float4 HumanGetGoalRotation(Human const *apHuman, skeleton::SkeletonPose const *apSkeletonPoseLocal, Goal goalIndex)
    {
        int32_t index = apHuman->m_HumanBoneIndex[s_HumanGoalInfo[goalIndex].m_Index];
        return AddAxis(apHuman, index, skeleton::SkeletonGetGlobalRotation(apHuman->m_Skeleton, apSkeletonPoseLocal, index));
    }

    void HumanPoseClear(HumanPose& arPose)
    {
        uint32_t i;

        arPose.m_RootX = math::trsIdentity();
        arPose.m_LookAtPosition = math::float3(math::ZERO);
        arPose.m_LookAtWeight = math::float4(math::ZERO);

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X = math::trsIdentity();
        }

        for (i = 0; i < kLastDoF; i++)
        {
            arPose.m_DoFArray[i] = 0;
        }

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            arPose.m_LeftHandPose.m_DoFArray[i] = 0;
            arPose.m_RightHandPose.m_DoFArray[i] = 0;
        }

        for (i = 0; i < kLastTDoF; i++)
        {
            arPose.m_TDoFArray[i] = math::float3(math::ZERO);
        }
    }

    void HumanPoseCopy(HumanPose &poseDst, HumanPose const &poseSrc, bool aDoFOnly)
    {
        uint32_t i;

        if (!aDoFOnly)
        {
            poseDst.m_RootX = poseSrc.m_RootX;
            poseDst.m_LookAtPosition = poseSrc.m_LookAtPosition;
            poseDst.m_LookAtWeight = poseSrc.m_LookAtWeight;

            for (i = 0; i < kLastGoal; i++)
            {
                poseDst.m_GoalArray[i].m_X = poseSrc.m_GoalArray[i].m_X;
                poseDst.m_GoalArray[i].m_WeightT = poseSrc.m_GoalArray[i].m_WeightT;
                poseDst.m_GoalArray[i].m_WeightR = poseSrc.m_GoalArray[i].m_WeightR;
                poseDst.m_GoalArray[i].m_HintT = poseSrc.m_GoalArray[i].m_HintT;
                poseDst.m_GoalArray[i].m_HintWeightT = poseSrc.m_GoalArray[i].m_HintWeightT;
            }
        }

        for (i = 0; i < kLastDoF; i++)
        {
            poseDst.m_DoFArray[i] = poseSrc.m_DoFArray[i];
        }

        hand::HandPoseCopy(&poseSrc.m_LeftHandPose, &poseDst.m_LeftHandPose);
        hand::HandPoseCopy(&poseSrc.m_RightHandPose, &poseDst.m_RightHandPose);

        for (i = 0; i < kLastTDoF; i++)
        {
            poseDst.m_TDoFArray[i] = poseSrc.m_TDoFArray[i];
        }
    }

    void HumanPoseCopy(HumanPose &poseDst, HumanPose const &poseSrc, HumanPoseMask const &arHumanPoseMask)
    {
        if (arHumanPoseMask == FullBodyMask())
        {
            HumanPoseCopy(poseDst, poseSrc);
        }
        else
        {
            int32_t i;
            for (i = 0; i < kLastDoF; i++)
            {
                if (arHumanPoseMask.test(kMaskDoFStartIndex + i))
                {
                    poseDst.m_DoFArray[i] = poseSrc.m_DoFArray[i];
                }
                else
                {
                    poseDst.m_DoFArray[i] = 0;
                }
            }

            if (arHumanPoseMask.test(kMaskLeftHand))
            {
                for (i = 0; i < hand::s_DoFCount; i++)
                {
                    poseDst.m_LeftHandPose.m_DoFArray[i] = poseSrc.m_LeftHandPose.m_DoFArray[i];
                }
            }
            else
            {
                for (i = 0; i < hand::s_DoFCount; i++)
                {
                    poseDst.m_LeftHandPose.m_DoFArray[i] = 0;
                }
            }

            if (arHumanPoseMask.test(kMaskRightHand))
            {
                for (i = 0; i < hand::s_DoFCount; i++)
                {
                    poseDst.m_RightHandPose.m_DoFArray[i] = poseSrc.m_RightHandPose.m_DoFArray[i];
                }
            }
            else
            {
                for (i = 0; i < hand::s_DoFCount; i++)
                {
                    poseDst.m_RightHandPose.m_DoFArray[i] = 0;
                }
            }

            for (i = 0; i < kLastGoal; i++)
            {
                if (arHumanPoseMask.test(kMaskGoalStartIndex + i))
                {
                    poseDst.m_GoalArray[i].m_X = poseSrc.m_GoalArray[i].m_X;
                }
                else
                {
                    poseDst.m_GoalArray[i].m_X = math::trsIdentity();
                }
            }

            if (arHumanPoseMask.test(kMaskRootIndex))
            {
                poseDst.m_RootX = poseSrc.m_RootX;
            }
            else
            {
                poseDst.m_RootX = math::trsIdentity();
            }

            for (i = 0; i < kLastTDoF; i++)
            {
                if (arHumanPoseMask.test(kMaskTDoFStartIndex + i))
                {
                    poseDst.m_TDoFArray[i] = poseSrc.m_TDoFArray[i];
                }
                else
                {
                    poseDst.m_TDoFArray[i] = math::float3(math::ZERO);
                }
            }
        }
    }

    void HumanPoseAdd(HumanPose &arPose, HumanPose const &arPoseA, HumanPose const &arPoseB)
    {
        uint32_t i;

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X = math::mul(arPoseA.m_GoalArray[i].m_X, arPoseB.m_GoalArray[i].m_X);
        }

        for (i = 0; i < kLastDoF; i++)
        {
            arPose.m_DoFArray[i] = arPoseA.m_DoFArray[i] + arPoseB.m_DoFArray[i];
        }

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            arPose.m_LeftHandPose.m_DoFArray[i] = arPoseA.m_LeftHandPose.m_DoFArray[i] + arPoseB.m_LeftHandPose.m_DoFArray[i];
            arPose.m_RightHandPose.m_DoFArray[i] = arPoseA.m_RightHandPose.m_DoFArray[i] + arPoseB.m_RightHandPose.m_DoFArray[i];
        }

        arPose.m_RootX = math::mul(arPoseA.m_RootX, arPoseB.m_RootX);

        for (i = 0; i < kLastTDoF; i++)
        {
            arPose.m_TDoFArray[i] = arPoseA.m_TDoFArray[i] + arPoseB.m_TDoFArray[i];
        }
    }

    void HumanPoseSub(HumanPose &arPose, HumanPose const &arPoseA, HumanPose const &arPoseB)
    {
        uint32_t i;

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X = math::trsInvMulNS(arPoseB.m_GoalArray[i].m_X, arPoseA.m_GoalArray[i].m_X);
        }

        for (i = 0; i < kLastDoF; i++)
        {
            arPose.m_DoFArray[i] = arPoseA.m_DoFArray[i] - arPoseB.m_DoFArray[i];
        }

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            arPose.m_LeftHandPose.m_DoFArray[i] = arPoseA.m_LeftHandPose.m_DoFArray[i] - arPoseB.m_LeftHandPose.m_DoFArray[i];
            arPose.m_RightHandPose.m_DoFArray[i] = arPoseA.m_RightHandPose.m_DoFArray[i] - arPoseB.m_RightHandPose.m_DoFArray[i];
        }

        arPose.m_RootX = math::trsInvMulNS(arPoseB.m_RootX, arPoseA.m_RootX);

        for (i = 0; i < kLastTDoF; i++)
        {
            arPose.m_TDoFArray[i] = arPoseA.m_TDoFArray[i] - arPoseB.m_TDoFArray[i];
        }
    }

    void HumanPoseWeight(HumanPose &arPose, HumanPose const &arPoseA, float aWeight)
    {
        uint32_t i;

        math::float1 w(aWeight);

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X = math::trsWeightNS(arPoseA.m_GoalArray[i].m_X, w);
        }

        for (i = 0; i < kLastDoF; i++)
        {
            arPose.m_DoFArray[i] = arPoseA.m_DoFArray[i] * aWeight;
        }

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            arPose.m_LeftHandPose.m_DoFArray[i] = arPoseA.m_LeftHandPose.m_DoFArray[i] * aWeight;
            arPose.m_RightHandPose.m_DoFArray[i] = arPoseA.m_RightHandPose.m_DoFArray[i] * aWeight;
        }

        arPose.m_RootX = math::trsWeightNS(arPoseA.m_RootX, w);

        for (i = 0; i < kLastTDoF; i++)
        {
            arPose.m_TDoFArray[i] = arPoseA.m_TDoFArray[i] * w;
        }
    }

    void HumanPoseMirror(HumanPose &arPose, HumanPose const &arPoseA)
    {
        uint32_t i;

        for (i = 0; i < kLastBodyDoF; i++)
        {
            arPose.m_DoFArray[kBodyDoFStart + i] *= BodyDoFMirror[i];
        }

        for (i = 0; i < kLastHeadDoF; i++)
        {
            arPose.m_DoFArray[kHeadDoFStart + i] *= HeadDoFMirror[i];
        }

        for (i = 0; i < kLastArmDoF; i++)
        {
            float dof = arPose.m_DoFArray[kLeftArmDoFStart + i];
            arPose.m_DoFArray[kLeftArmDoFStart + i] = arPose.m_DoFArray[kRightArmDoFStart + i];
            arPose.m_DoFArray[kRightArmDoFStart + i] = dof;
        }

        for (i = 0; i < kLastLegDoF; i++)
        {
            float dof = arPose.m_DoFArray[kLeftLegDoFStart + i];
            arPose.m_DoFArray[kLeftLegDoFStart + i] = arPose.m_DoFArray[kRightLegDoFStart + i];
            arPose.m_DoFArray[kRightLegDoFStart + i] = dof;
        }

        math::trsX x = arPose.m_GoalArray[kLeftFootGoal].m_X;
        arPose.m_GoalArray[kLeftFootGoal].m_X = arPose.m_GoalArray[kRightFootGoal].m_X;
        arPose.m_GoalArray[kRightFootGoal].m_X = x;

        x = arPose.m_GoalArray[kLeftHandGoal].m_X;
        arPose.m_GoalArray[kLeftHandGoal].m_X = arPose.m_GoalArray[kRightHandGoal].m_X;
        arPose.m_GoalArray[kRightHandGoal].m_X = x;

        const math::float4 offsetQY = math::float4(0.f, 1.f, 0.f, 0.f);
        const math::float4 offsetQZ = math::float4(0.f, 0.f, 1.f, 0.f);

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X = math::mirrorX(arPose.m_GoalArray[i].m_X);
        }

        arPose.m_GoalArray[kLeftFootGoal].m_X.q = math::normalize(math::quatMul(arPose.m_GoalArray[kLeftFootGoal].m_X.q, offsetQY));
        arPose.m_GoalArray[kRightFootGoal].m_X.q = math::normalize(math::quatMul(arPose.m_GoalArray[kRightFootGoal].m_X.q, offsetQY));
        arPose.m_GoalArray[kLeftHandGoal].m_X.q = math::normalize(math::quatMul(arPose.m_GoalArray[kLeftHandGoal].m_X.q, offsetQZ));
        arPose.m_GoalArray[kRightHandGoal].m_X.q = math::normalize(math::quatMul(arPose.m_GoalArray[kRightHandGoal].m_X.q, offsetQZ));

        arPose.m_RootX = math::mirrorX(arPose.m_RootX);

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            float leftdof = arPose.m_LeftHandPose.m_DoFArray[i];
            arPose.m_LeftHandPose.m_DoFArray[i] = arPose.m_RightHandPose.m_DoFArray[i];
            arPose.m_RightHandPose.m_DoFArray[i] = leftdof;
        }

        for (i = 0; i < kLastTDoF; i++)
        {
            arPose.m_TDoFArray[kBodyTDoFStart + i] = math::mirrorX(arPose.m_TDoFArray[kBodyTDoFStart + i]);
        }

        for (i = 0; i < kLastArmTDoF; i++)
        {
            math::float3 tdof = arPose.m_TDoFArray[kLeftArmTDoFStart + i];
            arPose.m_TDoFArray[kLeftArmTDoFStart + i] = arPose.m_TDoFArray[kRightArmTDoFStart + i];
            arPose.m_TDoFArray[kRightArmTDoFStart + i] = tdof;
        }

        for (i = 0; i < kLastLegTDoF; i++)
        {
            math::float3 tdof = arPose.m_TDoFArray[kLeftLegTDoFStart + i];
            arPose.m_TDoFArray[kLeftLegTDoFStart + i] = arPose.m_TDoFArray[kRightLegTDoFStart + i];
            arPose.m_TDoFArray[kRightLegTDoFStart + i] = tdof;
        }
    }

    void HumanPoseBlendBegin(HumanPose &arPose)
    {
        uint32_t i;

        for (i = 0; i < kLastDoF; i++)
        {
            arPose.m_DoFArray[i] = 0;
        }

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            arPose.m_LeftHandPose.m_DoFArray[i] = 0;
            arPose.m_RightHandPose.m_DoFArray[i] = 0;
        }

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X.t = math::float3(math::ZERO);
            arPose.m_GoalArray[i].m_X.q = math::float4(math::ZERO);
            arPose.m_GoalArray[i].m_X.s = math::float3(1.f);
        }

        arPose.m_RootX.t = math::float3(math::ZERO);
        arPose.m_RootX.q = math::float4(math::ZERO);
        arPose.m_RootX.s = math::float3(1.f);

        for (i = 0; i < kLastTDoF; i++)
        {
            arPose.m_TDoFArray[i] = math::float3(math::ZERO);
        }
    }

    void HumanPoseBlendNode(HumanPose &arPose, HumanPose *apNodePose, float aWeight)
    {
        uint32_t i;

        float w = aWeight;
        math::float1 w1(w);

        for (i = 0; i < kLastDoF; i++)
        {
            arPose.m_DoFArray[i] += apNodePose->m_DoFArray[i] * w;
        }

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            arPose.m_LeftHandPose.m_DoFArray[i] += apNodePose->m_LeftHandPose.m_DoFArray[i] * w;
            arPose.m_RightHandPose.m_DoFArray[i] += apNodePose->m_RightHandPose.m_DoFArray[i] * w;
        }

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X.t += apNodePose->m_GoalArray[i].m_X.t * w1;
            arPose.m_GoalArray[i].m_X.q += math::chgsign(apNodePose->m_GoalArray[i].m_X.q * w1, math::dot(arPose.m_GoalArray[i].m_X.q, apNodePose->m_GoalArray[i].m_X.q));
        }

        arPose.m_RootX.t += apNodePose->m_RootX.t * w1;
        arPose.m_RootX.q += math::chgsign(apNodePose->m_RootX.q * w1, math::dot(arPose.m_RootX.q, apNodePose->m_RootX.q));

        for (i = 0; i < kLastTDoF; i++)
        {
            arPose.m_TDoFArray[i] += apNodePose->m_TDoFArray[i] * w1;
        }
    }

    void HumanPoseBlendEnd(HumanPose &arPose, float &weightSum)
    {
        uint32_t i;

        math::float4 q = math::float4(0.f, 0.f, 0.f, math::saturate(1.0f - weightSum));

        for (i = 0; i < kLastGoal; i++)
        {
            arPose.m_GoalArray[i].m_X.q = math::normalize(arPose.m_GoalArray[i].m_X.q + q);
        }

        arPose.m_RootX.q = math::normalize(arPose.m_RootX.q + q);
    }

    void HumanPoseClear(HumanPose& arPose, HumanPoseMask const &arHumanPoseMask)
    {
        uint32_t i;

        if (!arHumanPoseMask.test(kMaskRootIndex))
        {
            arPose.m_RootX = math::trsIdentity();
        }

        for (i = 0; i < kLastGoal; i++)
        {
            if (!arHumanPoseMask.test(kMaskGoalStartIndex + i))
            {
                arPose.m_GoalArray[i].m_X = math::trsIdentity();
            }
        }

        for (i = 0; i < kLastDoF; i++)
        {
            if (!arHumanPoseMask.test(kMaskDoFStartIndex + i))
            {
                arPose.m_DoFArray[i] = 0;
            }
        }

        for (i = 0; i < (uint32_t)hand::s_DoFCount; i++)
        {
            if (!arHumanPoseMask.test(kMaskLeftHand))
            {
                arPose.m_LeftHandPose.m_DoFArray[i] = 0;
            }

            if (!arHumanPoseMask.test(kMaskRightHand))
            {
                arPose.m_RightHandPose.m_DoFArray[i] = 0;
            }
        }

        for (i = 0; i < kLastTDoF; i++)
        {
            if (!arHumanPoseMask.test(kMaskTDoFStartIndex + i))
            {
                arPose.m_TDoFArray[i] = math::float3(math::ZERO);
            }
        }
    }

    void HumanFixTwist(Human const *apHuman, skeleton::SkeletonPose *apSkeletonPose, skeleton::SkeletonPose *apSkeletonPoseWs, int32_t aPIndex, int32_t aCIndex, const math::float1& aTwist)
    {
        int32_t pNodeIndex = apHuman->m_HumanBoneIndex[aPIndex];
        int32_t cNodeIndex = apHuman->m_HumanBoneIndex[aCIndex];
        int32_t aNodeIndex = apHuman->m_Skeleton->m_Node[pNodeIndex].m_ParentId;

        math::Axes pAxes = apHuman->m_Skeleton->m_AxesArray[apHuman->m_Skeleton->m_Node[pNodeIndex].m_AxesId];

        apSkeletonPoseWs->m_X[aNodeIndex].q = math::quatIdentity();
        skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, pNodeIndex);

        math::float4 pq = apSkeletonPose->m_X[pNodeIndex].q;
        math::float4 cqg = apSkeletonPoseWs->m_X[cNodeIndex].q;

        math::float3 pxyz = math::ToAxes(pAxes, apSkeletonPose->m_X[pNodeIndex].q);
        pxyz.x *= aTwist;

        apSkeletonPose->m_X[pNodeIndex].q = math::FromAxes(pAxes, pxyz);

        skeleton::SkeletonAlign(apHuman->m_Skeleton, pq, apSkeletonPose->m_X[pNodeIndex].q, pNodeIndex);

        skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, pNodeIndex);

        apSkeletonPoseWs->m_X[cNodeIndex].q = cqg;

        skeleton::SkeletonPoseComputeLocalQ(apHuman->m_Skeleton, apSkeletonPoseWs, apSkeletonPose, cNodeIndex, cNodeIndex);
    }

    void HumanFixEndPointsSkeletonPose(Human const *apHuman, skeleton::SkeletonPose const*apSkeletonPoseRef, HumanPose *apHumanPose, skeleton::SkeletonPose *apSkeletonPoseGbl, skeleton::SkeletonPose *apSkeletonPoseLcl, skeleton::SkeletonPose *apSkeletonPoseWs, int32_t cIndex)
    {
        apSkeletonPoseGbl->m_X[apHuman->m_HumanBoneIndex[cIndex]].q = apSkeletonPoseRef->m_X[apHuman->m_HumanBoneIndex[cIndex]].q;
        skeleton::SkeletonPoseComputeLocalQ(apHuman->m_Skeleton, apSkeletonPoseGbl, apSkeletonPoseLcl, apHuman->m_HumanBoneIndex[cIndex], apHuman->m_HumanBoneIndex[cIndex]);
    }

    void HumanAlignSkeletonPose(Human const *apHuman, skeleton::SkeletonPose const*apSkeletonPoseRef, HumanPose *apHumanPose, skeleton::SkeletonPose *apSkeletonPoseGbl, skeleton::SkeletonPose *apSkeletonPoseLcl, int32_t cIndex, int32_t pIndex)
    {
        Skeleton2HumanPose(apHuman, apSkeletonPoseLcl, apHumanPose, cIndex);
        Human2SkeletonPose(apHuman, apHumanPose, apSkeletonPoseLcl, cIndex);

        skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPoseLcl, apSkeletonPoseGbl, apHuman->m_HumanBoneIndex[cIndex], apHuman->m_HumanBoneIndex[pIndex]);
        skeleton::SkeletonAlign(apHuman->m_Skeleton, apSkeletonPoseRef, apSkeletonPoseGbl, apHuman->m_HumanBoneIndex[cIndex]);
        skeleton::SkeletonPoseComputeLocalQ(apHuman->m_Skeleton, apSkeletonPoseGbl, apSkeletonPoseLcl, apHuman->m_HumanBoneIndex[cIndex], apHuman->m_HumanBoneIndex[pIndex]);
    }

    void Human2LimbAlign(Human const *apHuman, skeleton::SkeletonPose const*apSkeletonPoseRef, skeleton::SkeletonPose *apSkeletonPoseGbl, skeleton::SkeletonPose *apSkeletonPoseLcl, int32_t eIndex, int32_t pIndex)
    {
        math::float3 src = apSkeletonPoseGbl->m_X[apHuman->m_HumanBoneIndex[pIndex]].t;
        math::float3 end = apSkeletonPoseGbl->m_X[apHuman->m_HumanBoneIndex[eIndex]].t;
        math::float3 goal = apSkeletonPoseRef->m_X[apHuman->m_HumanBoneIndex[eIndex]].t;
        math::float4 q = math::quatArcRotate(end - src, goal - src);

        apSkeletonPoseGbl->m_X[apHuman->m_HumanBoneIndex[pIndex]].q = math::quatMul(q, apSkeletonPoseGbl->m_X[apHuman->m_HumanBoneIndex[pIndex]].q);
        skeleton::SkeletonPoseComputeLocal(apHuman->m_Skeleton, apSkeletonPoseGbl, apSkeletonPoseLcl, apHuman->m_HumanBoneIndex[pIndex], apHuman->m_HumanBoneIndex[pIndex]);
        skeleton::SkeletonPoseComputeGlobal(apHuman->m_Skeleton, apSkeletonPoseLcl, apSkeletonPoseGbl, apHuman->m_HumanBoneIndex[eIndex], apHuman->m_HumanBoneIndex[pIndex]);
    }

    void HumanFixMidDoF(Human const *apHuman, 
        skeleton::SkeletonPose *apSkeletonPose, 
        skeleton::SkeletonPose *apSkeletonPoseWs, 
        int32_t aPIndex, 
        int32_t aCIndex, 
        float maxError = 0.1f, 
        int maxIter = 10)
    {
        // 获取父骨骼和子骨骼在骨骼索引中的位置
        int32_t pNodeIndex = apHuman->m_HumanBoneIndex[aPIndex];
        int32_t cNodeIndex = apHuman->m_HumanBoneIndex[aCIndex];
        
        // 获取父骨骼的父节点，即祖父节点在骨骼索引中的位置
        int32_t aNodeIndex = apHuman->m_Skeleton->m_Node[pNodeIndex].m_ParentId;

        // 获取子骨骼的轴向信息，用于约束和投影旋转
        math::Axes cAxes = apHuman->m_Skeleton->m_AxesArray[apHuman->m_Skeleton->m_Node[cNodeIndex].m_AxesId];

        // 将祖父节点的旋转初始化为单位四元数（无旋转）
        apSkeletonPoseWs->m_X[aNodeIndex].q = math::quatIdentity();

        // 计算子骨骼节点的全局旋转四元数（从局部姿态到全局姿态）
        skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, pNodeIndex);

        // 获取父骨骼的当前局部旋转四元数和子骨骼的全局旋转四元数
        math::float4 pq = apSkeletonPose->m_X[pNodeIndex].q;
        math::float4 cqg = apSkeletonPoseWs->m_X[cNodeIndex].q;

        // 设置初始误差为360度，并初始化迭代次数
        float prevError = 360.0f;
        int iter = 0;
        bool exit = false;

        // 循环修正，直到误差低于最大允许误差或达到最大迭代次数
        while (!exit && iter < maxIter)
        {
            // 将子骨骼的当前局部旋转投影到轴向上
            math::float4 cql = AxesProject(cAxes, apSkeletonPose->m_X[cNodeIndex].q);
            
            // 将投影后的四元数转换为ZYRoll旋转角
            math::float3 xyz = math::quat2ZYRoll(cql);

            // 根据轴向符号获取旋转的度量，并计算局部误差
            math::float3 dof = math::doubleAtan(math::chgsign(xyz, cAxes.m_Sgn));
            float localError = math::degrees(float(math::abs(dof.y)));

            // 检查是否达到了误差阈值或误差是否开始增加
            if (localError < maxError || (localError < 10.0f * maxError && localError > prevError))
            {
                exit = true;
            }
            else
            {
                // 将Y轴旋转分量设置为0，即消除Y轴旋转
                xyz.y = 0;

                // 重新计算子骨骼的局部四元数旋转
                cql = math::ZYRoll2Quat(xyz);

                // 将修正后的四元数从轴向投影回局部空间
                apSkeletonPose->m_X[cNodeIndex].q = math::AxesUnproject(cAxes, cql);

                // 重新计算子骨骼和父骨骼的全局旋转四元数
                skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, cNodeIndex);

                // 计算旋转差异并应用于父骨骼，使父骨骼对齐子骨骼
                math::float4 qdiff = math::quatMul(cqg, math::quatConj(apSkeletonPoseWs->m_X[cNodeIndex].q));
                apSkeletonPose->m_X[pNodeIndex].q = math::normalize(math::quatMul(qdiff, apSkeletonPose->m_X[pNodeIndex].q));

                // 确保父骨骼的旋转四元数与初始旋转对齐
                skeleton::SkeletonAlign(apHuman->m_Skeleton, pq, apSkeletonPose->m_X[pNodeIndex].q, pNodeIndex);

                // 再次计算全局姿态并保持子骨骼的全局旋转不变
                skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, pNodeIndex);
                apSkeletonPoseWs->m_X[cNodeIndex].q = cqg;

                // 计算子骨骼和父骨骼的局部旋转
                skeleton::SkeletonPoseComputeLocalQ(apHuman->m_Skeleton, apSkeletonPoseWs, apSkeletonPose, cNodeIndex, cNodeIndex);

                // 记录当前误差并进入下一次迭代
                prevError = localError;
                iter++;
            }
        }
    }


    void HumanFixEndDoF(Human const *apHuman, 
        skeleton::SkeletonPose *apSkeletonPose, 
        skeleton::SkeletonPose *apSkeletonPoseWs, 
        int32_t aPIndex, 
        int32_t aCIndex, 
        float maxError = 0.1f, 
        int maxIter = 5)
    {
        // 获取父骨骼和子骨骼在骨骼索引中的位置
        int32_t pNodeIndex = apHuman->m_HumanBoneIndex[aPIndex];
        int32_t cNodeIndex = apHuman->m_HumanBoneIndex[aCIndex];
        
        // 获取父骨骼的父节点，即祖父节点在骨骼索引中的位置
        int32_t aNodeIndex = apHuman->m_Skeleton->m_Node[pNodeIndex].m_ParentId;

        // 获取父骨骼和子骨骼的轴向信息，用于约束和投影旋转
        math::Axes pAxes = apHuman->m_Skeleton->m_AxesArray[apHuman->m_Skeleton->m_Node[pNodeIndex].m_AxesId];
        math::Axes cAxes = apHuman->m_Skeleton->m_AxesArray[apHuman->m_Skeleton->m_Node[cNodeIndex].m_AxesId];

        // 将祖父节点的旋转初始化为单位四元数（无旋转）
        apSkeletonPoseWs->m_X[aNodeIndex].q = math::quatIdentity();

        // 计算子骨骼节点的全局旋转四元数（从局部姿态到全局姿态）
        skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, pNodeIndex);

        // 获取父骨骼的当前局部旋转四元数和子骨骼的全局旋转四元数
        math::float4 pq = apSkeletonPose->m_X[pNodeIndex].q;
        math::float4 cqg = apSkeletonPoseWs->m_X[cNodeIndex].q;

        // 将父骨骼和子骨骼的轴向初始旋转设为零旋转
        math::float4 pq0 = FromAxes(pAxes, math::float3(math::ZERO));
        math::float4 cql0 = FromAxes(cAxes, math::float3(math::ZERO));

        // 设置初始误差为360度，并初始化迭代次数
        float prevError = 360.0f;
        int iter = 0;
        bool exit = false;

        // 使用迭代来修正旋转误差，直到误差低于最大允许误差或达到最大迭代次数
        while (!exit && iter < maxIter)
        {
            // 将子骨骼的当前局部旋转投影到轴向上
            math::float4 cql = AxesProject(cAxes, apSkeletonPose->m_X[cNodeIndex].q);
            
            // 将投影后的四元数转换为ZYRoll旋转角
            math::float3 xyz = math::quat2ZYRoll(cql);

            // 计算局部误差，主要考虑X轴的旋转角度
            float localError = math::abs(math::degrees(float(math::doubleAtan(xyz).x)));

            // 检查误差是否低于阈值或误差是否开始增加
            if (localError < maxError || (iter > 1 && localError > prevError))
            {
                exit = true;
            }
            else
            {
                if (iter == 0) // 初次迭代时重置父骨骼和子骨骼的旋转，解决类似手腕扭转的问题
                {
                    apSkeletonPose->m_X[pNodeIndex].q = pq0;
                    apSkeletonPose->m_X[cNodeIndex].q = cql0;

                    // 重新计算子骨骼的全局旋转
                    skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, pNodeIndex);
                }
                else
                {
                    // 将X轴旋转分量设置为0
                    xyz.x = 0.f;

                    // 重新计算子骨骼的局部四元数旋转
                    cql = math::ZYRoll2Quat(xyz);

                    // 将修正后的四元数从轴向投影回局部空间
                    apSkeletonPose->m_X[cNodeIndex].q = math::AxesUnproject(cAxes, cql);

                    // 重新计算子骨骼的全局旋转
                    skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, cNodeIndex);
                }

                // 计算旋转差异并应用于父骨骼，使父骨骼对齐子骨骼
                math::float4 qdiff = math::quatMul(cqg, math::quatConj(apSkeletonPoseWs->m_X[cNodeIndex].q));
                apSkeletonPose->m_X[pNodeIndex].q = math::normalize(math::quatMul(qdiff, apSkeletonPose->m_X[pNodeIndex].q));

                // 确保父骨骼的旋转四元数与初始旋转对齐
                skeleton::SkeletonAlign(apHuman->m_Skeleton, pq, apSkeletonPose->m_X[pNodeIndex].q, pNodeIndex);

                // 再次计算全局姿态并保持子骨骼的全局旋转不变
                skeleton::SkeletonPoseComputeGlobalQ(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs, cNodeIndex, pNodeIndex);
                apSkeletonPoseWs->m_X[cNodeIndex].q = cqg;

                // 计算子骨骼的局部旋转
                skeleton::SkeletonPoseComputeLocalQ(apHuman->m_Skeleton, apSkeletonPoseWs, apSkeletonPose, cNodeIndex, cNodeIndex);

                // 记录当前误差并进入下一次迭代
                prevError = localError;
                iter++;
            }
        }
    }


    void ReachGoalRotation(Human const *apHuman, math::float4 const &arEndQ, int32_t aGoalIndex, skeleton::SkeletonPose *apSkeletonPose, skeleton::SkeletonPose *apSkeletonPoseGbl, skeleton::SkeletonPose *apSkeletonPoseWorkspace)
    {
        int32_t index = apHuman->m_HumanBoneIndex[s_HumanGoalInfo[aGoalIndex].m_Index];
        int32_t parentIndex = apHuman->m_Skeleton->m_Node[index].m_ParentId;
        apSkeletonPose->m_X[index].q = math::normalize(math::quatMul(math::quatConj(apSkeletonPoseGbl->m_X[parentIndex].q), arEndQ));

        HumanFixEndDoF(apHuman, apSkeletonPose, apSkeletonPoseWorkspace, s_HumanGoalInfo[aGoalIndex].m_MidIndex, s_HumanGoalInfo[aGoalIndex].m_EndIndex, 0.05f, 1);
    }

    void RetargetFromTDoFBase(Human const *apHuman,
        skeleton::SkeletonPose *apSkeletonPoseGbl,
        math::float3 *apTDoFBase)
    {
        // 获取 Human 的骨架结构
        skeleton::HumanSkeleton const *skeleton = apHuman->m_Skeleton;

        // 计算全局骨架姿态，基于 Human 中的局部姿态，存储在 apSkeletonPoseGbl
		SkeletonPoseComputeGlobal(skeleton, apHuman->m_SkeletonPose.ptr(), apSkeletonPoseGbl);

        // 遍历每一个 TDoF（自由度），初始化 apTDoFBase
        for (int tDoFIter = 0; tDoFIter < kLastTDoF; tDoFIter++)
        {
            // 将当前 TDoF 的基准值初始化为零向量
            apTDoFBase[tDoFIter] = math::float3(math::ZERO);

            // 根据 TDoF 索引获取对应的骨骼索引
            int boneIndex = BoneFromTDoF(tDoFIter);
            // 获取该骨骼的有效父骨骼索引
            int boneParentIndex = GetValidBoneParentIndex(apHuman, boneIndex);

            // 根据骨骼索引获取骨架中的实际索引
            int skBoneIndex = apHuman->m_HumanBoneIndex[boneIndex];
            int skBoneParentIndex = apHuman->m_HumanBoneIndex[boneParentIndex];

            // 检查骨骼索引是否有效（不等于 -1）
            if (skBoneIndex != -1 && skBoneParentIndex != -1)
            {
                // 获取父骨骼的全局变换矩阵 pgx
                math::trsX pgx = apSkeletonPoseGbl->m_X[skBoneParentIndex];
                // 对父骨骼应用额外的轴变换
                pgx.q = AddAxis(apHuman, skBoneParentIndex, pgx.q);
                // 使用父骨骼的全局变换逆矩阵与子骨骼的全局位置相乘，计算 TDoF 基准位置
                apTDoFBase[tDoFIter] = math::invMul(pgx, apSkeletonPoseGbl->m_X[skBoneIndex].t);
            }
        }
    }


    void RetargetFromTDoF(int32_t dofIndex,
        Human const *apHuman,                       // 人体骨架对象
        skeleton::SkeletonPose const *apSkeletonPoseRef, // 参考骨骼姿态（全局姿态）
        math::float3 const *apTDoFBase,             // 自由度基础数组
        HumanPose *apHumanPose,                     // 输出的人体姿态
        skeleton::SkeletonPose *apSkeletonPoseLcl,  // 局部骨骼姿态
        skeleton::SkeletonPose *apSkeletonPoseGbl)  // 全局骨骼姿态
    {
        // 初始化该自由度为零
        apHumanPose->m_TDoFArray[dofIndex] = math::float3(math::ZERO);

        // 获取该自由度的对应骨骼索引和其父骨骼索引
        int boneIndex = BoneFromTDoF(dofIndex);
        int boneParentIndex = GetValidBoneParentIndex(apHuman, boneIndex);

        // 获取骨骼在 skeleton 中的索引
        int skBoneIndex = apHuman->m_HumanBoneIndex[boneIndex];
        int skBoneParentIndex = apHuman->m_HumanBoneIndex[boneParentIndex];

        // 如果骨骼和其父骨骼在 skeleton 中存在
        if (skBoneIndex != -1 && skBoneParentIndex != -1)
        {
            // 获取骨架
            skeleton::HumanSkeleton const *skeleton = apHuman->m_Skeleton;

            // 计算父骨骼的全局姿态
            int skParentIndex = skeleton->m_Node[skBoneIndex].m_ParentId;
            skeleton::SkeletonPoseComputeGlobal(skeleton, apSkeletonPoseLcl, apSkeletonPoseGbl, skParentIndex, -1);
            
            // 计算当前骨骼相对于父骨骼的位移
            apSkeletonPoseLcl->m_X[skBoneIndex].t = math::invMul(apSkeletonPoseGbl->m_X[skParentIndex], apSkeletonPoseRef->m_X[skBoneIndex].t);

            // 获取父骨骼的全局姿态
            math::trsX pgx = apSkeletonPoseGbl->m_X[skBoneParentIndex];
            pgx.q = AddAxis(apHuman, skBoneParentIndex, pgx.q);

            // 计算自由度的变换值并调整缩放
            apHumanPose->m_TDoFArray[dofIndex] = math::invMul(pgx, apSkeletonPoseRef->m_X[skBoneIndex].t) - apTDoFBase[dofIndex];
            apHumanPose->m_TDoFArray[dofIndex] /= apHuman->m_Scale;
        }
    }


    void RetargetToTDoF(Human const *apHuman,
        HumanPose *apHumanPoseOut,                  // 输出的人体姿态
        const LocalVector< math::trsX>& apSkeletonPoseRef, // 参考骨骼姿态（全局）
        skeleton::SkeletonPose *apSkeletonPose,     // 输出的局部骨骼姿态
        skeleton::SkeletonPose *apSkeletonPoseWs)   // 输出的全局骨骼姿态
    {
        // 遍历所有自由度
        for (int tDoFIter = 0; tDoFIter < kLastTDoF; tDoFIter++)
        {
            int boneIndex = BoneFromTDoF(tDoFIter); // 获取当前自由度对应的骨骼索引
            int boneParentIndex = GetValidBoneParentIndex(apHuman, boneIndex); // 获取父骨骼索引

            int skIndex = apHuman->m_HumanBoneIndex[boneIndex]; // 骨骼索引
            int skParentIndex = apHuman->m_HumanBoneIndex[boneParentIndex]; // 父骨骼索引

            // 确认骨骼和父骨骼存在
            if (skIndex != -1 && skParentIndex != -1)
            {
                skeleton::HumanSkeleton const *skeleton = apHuman->m_Skeleton;

                // 将父骨骼的全局姿态初始化为单位矩阵
                apSkeletonPoseWs->m_X[skeleton->m_Node[skParentIndex].m_ParentId] = math::trsIdentity();
                
                // 将当前骨骼的位移设置为参考骨骼姿态
                apSkeletonPose->m_X[skIndex].t = apSkeletonPoseRef[skIndex].t;

                // 计算全局姿态
                skeleton::SkeletonPoseComputeGlobal(skeleton, apSkeletonPose, apSkeletonPoseWs, skIndex, skParentIndex);

                // 获取父骨骼的姿态，并在父骨骼上应用自由度旋转
                math::trsX pgx = apSkeletonPoseWs->m_X[skParentIndex];
                pgx.q = AddAxis(apHuman, skParentIndex, pgx.q);

                // 计算当前自由度的变换，并将其应用到全局姿态
                math::float3 tdof = apHumanPoseOut->m_TDoFArray[tDoFIter] * apHuman->m_Scale;
                apSkeletonPoseWs->m_X[skIndex].t += math::quatMulVec(pgx.q, tdof * pgx.s);

                // 重新计算局部姿态
                skeleton::SkeletonPoseComputeLocal(skeleton, apSkeletonPoseWs, apSkeletonPose, skIndex, skIndex);
            }
        }
    }


    void RetargetFrom(Human *apHuman,
        skeleton::SkeletonPose const *apSkeletonPose,
        HumanPose *apHumanPose,
        skeleton::SkeletonPose *apSkeletonPoseRef,
        skeleton::SkeletonPose *apSkeletonPoseGbl,
        skeleton::SkeletonPose *apSkeletonPoseLcl,
        skeleton::SkeletonPose *apSkeletonPoseWs,
        math::float3 const *tDoFBase)
    {
        // 获取 hips 骨骼的索引
        const int32_t hipsIndex = apHuman->m_HumanBoneIndex[human::kHips];
        const math::float1 scale(apHuman->m_Scale);

        // 计算全局骨骼姿态
        skeleton::SkeletonPoseComputeGlobal(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseRef);
        // 拷贝骨骼姿态
        skeleton::SkeletonPoseCopy(apSkeletonPoseRef, apSkeletonPoseGbl);
        skeleton::SkeletonPoseCopy(apSkeletonPose, apSkeletonPoseLcl);

        // 强制将虚拟骨骼设为默认旋转
        int32_t nodeIter;
        for (nodeIter = 1; nodeIter < (int)apHuman->m_Skeleton->m_Count; nodeIter++)
        {
            if (apHuman->m_Skeleton->m_Node[nodeIter].m_AxesId == -1)
            {
                apSkeletonPoseGbl->m_X[nodeIter].q = math::quatMul(apSkeletonPoseGbl->m_X[apHuman->m_Skeleton->m_Node[nodeIter].m_ParentId].q, apHuman->m_SkeletonPose[nodeIter].q);
            }
        }

        // 计算局部旋转
        skeleton::SkeletonPoseComputeLocalQ(apHuman->m_Skeleton, apSkeletonPoseGbl, apSkeletonPoseLcl);

        // 如果有额外的自由度（DoF），处理自由度调整
        if (apHuman->m_HasTDoF && tDoFBase)
        {
            // 将 hips 骨骼的姿态设置为参考姿态
            apSkeletonPoseLcl->m_X[0] = math::trsIdentity();
            apSkeletonPoseLcl->m_X[hipsIndex] = apSkeletonPoseRef->m_X[hipsIndex];

            // 处理腿部、躯干和手臂的自由度调整
            RetargetFromTDoF(kLeftLegTDoFStart + kUpperLegTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kRightLegTDoFStart + kUpperLegTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kBodyTDoFStart + kSpineTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kBodyTDoFStart + kChestTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kBodyTDoFStart + kUpperChestTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kLeftArmTDoFStart + kShoulderTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kRightArmTDoFStart + kShoulderTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
        }

        // 对齐肩膀
        if (apHuman->m_HumanBoneIndex[kLeftShoulder] != -1)
            HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kLeftShoulder, kLeftShoulder);
        if (apHuman->m_HumanBoneIndex[kRightShoulder] != -1)
            HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kRightShoulder, kRightShoulder);

        // 继续处理自由度
        if (apHuman->m_HasTDoF && tDoFBase)
        {
            RetargetFromTDoF(kLeftArmTDoFStart + kUpperArmTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kRightArmTDoFStart + kUpperArmTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
        }

        // 对齐四肢（上肢、下肢）
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kLeftUpperArm, apHuman->m_HumanBoneIndex[kLeftShoulder] != -1 ? kLeftShoulder : kLeftUpperArm);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kRightUpperArm, apHuman->m_HumanBoneIndex[kRightShoulder] != -1 ? kRightShoulder : kRightUpperArm);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kLeftUpperLeg, kLeftUpperLeg);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kRightUpperLeg, kRightUpperLeg);

        // 处理下肢、上臂自由度
        if (apHuman->m_HasTDoF && tDoFBase)
        {
            RetargetFromTDoF(kLeftLegTDoFStart + kLowerLegTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kRightLegTDoFStart + kLowerLegTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kLeftArmTDoFStart + kLowerArmTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kRightArmTDoFStart + kLowerArmTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
        }

        // 对齐并修正下肢
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kLeftLowerArm, kLeftUpperArm);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kRightLowerArm, kRightUpperArm);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kLeftLowerLeg, kLeftUpperLeg);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kRightLowerLeg, kRightUpperLeg);

        // 修正自由度的中间部分
        HumanFixMidDoF(apHuman, apSkeletonPoseLcl, apSkeletonPoseWs, kLeftUpperArm, kLeftLowerArm);
        HumanFixMidDoF(apHuman, apSkeletonPoseLcl, apSkeletonPoseWs, kRightUpperArm, kRightLowerArm);
        HumanFixMidDoF(apHuman, apSkeletonPoseLcl, apSkeletonPoseWs, kLeftUpperLeg, kLeftLowerLeg);
        HumanFixMidDoF(apHuman, apSkeletonPoseLcl, apSkeletonPoseWs, kRightUpperLeg, kRightLowerLeg);

        // 强制将 human pose 反映到 skeleton 中
        Skeleton2HumanPose(apHuman, apSkeletonPoseLcl, apHumanPose);
        Human2SkeletonPose(apHuman, apHumanPose, apSkeletonPoseLcl);

        // 处理额外的自由度
        if (apHuman->m_HasTDoF && tDoFBase)
        {
            RetargetFromTDoF(kLeftLegTDoFStart + kFootTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kRightLegTDoFStart + kFootTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
        }

        // 修正旋转
        if (apHuman->m_HasTDoF && tDoFBase)
        {
            RetargetFromTDoF(kLeftArmTDoFStart + kHandTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
            RetargetFromTDoF(kRightArmTDoFStart + kHandTDoF, apHuman, apSkeletonPoseRef, &tDoFBase[0], apHumanPose, apSkeletonPoseLcl, apSkeletonPoseGbl);
        }

        // 对齐并修正手和脚
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kLeftHand, kLeftLowerArm);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kRightHand, kRightLowerArm);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kLeftFoot, kLeftLowerLeg);
        HumanAlignSkeletonPose(apHuman, apSkeletonPoseRef, apHumanPose, apSkeletonPoseGbl, apSkeletonPoseLcl, kRightFoot, kRightLowerLeg);

        // 更新骨骼姿态
        skeleton::SkeletonPoseCopy(apSkeletonPoseLcl, apSkeletonPoseWs);
    }


    void RetargetTo(Human *apHuman,
        HumanPose const *apHumanPoseBase,     // 基础的人体姿态
        HumanPose const *apHumanPose,         // 输入的人体姿态（可选）
        const math::trsX &arX,                // 变换矩阵（平移、旋转和缩放）
        HumanPose *apHumanPoseOut,            // 输出的人体姿态
        skeleton::SkeletonPose *apSkeletonPose, // 输出的骨架姿态（局部坐标）
        skeleton::SkeletonPose *apSkeletonPoseWs, // 输出的骨架姿态（全局坐标）
        bool adjustMissingBones)              // 是否调整缺失的骨骼
    {
        const int32_t rootIndex = 0;            // 根骨骼索引
        const int32_t hipsIndex = apHuman->m_HumanBoneIndex[human::kHips]; // hips 骨骼索引
        const math::float1 scale(apHuman->m_Scale);  // 人体的缩放比例

        // 复制基础人体姿态到输出姿态
        human::HumanPoseCopy(*apHumanPoseOut, *apHumanPoseBase);

        // 调整根骨骼的平移并进行缩放
        apHumanPoseOut->m_RootX.t *= scale;
        // 根据输入变换矩阵对根骨骼进行变换
        apHumanPoseOut->m_RootX = math::mul(arX, apHumanPoseOut->m_RootX);

        // 处理所有的目标（例如手脚的目标位置）
        int32_t goalIter;
        for (goalIter = 0; goalIter < kLastGoal; goalIter++)
        {
            // 使用输入的人体姿态中的目标姿态或基础姿态
            apHumanPoseOut->m_GoalArray[goalIter].m_X = apHumanPose ? apHumanPose->m_GoalArray[goalIter].m_X : apHumanPoseBase->m_GoalArray[goalIter].m_X;
            apHumanPoseOut->m_GoalArray[goalIter].m_X.t *= scale;
            apHumanPoseOut->m_GoalArray[goalIter].m_X = math::mul(arX, apHumanPoseOut->m_GoalArray[goalIter].m_X);

            // 如果是脚部（goalIter < 2），进行额外的底部调整
            if (goalIter < 2)
                apHumanPoseOut->m_GoalArray[goalIter].m_X.t = math::mul(apHumanPoseOut->m_GoalArray[goalIter].m_X, -human::HumanGetFootBottom(apHuman, goalIter == 0));
        }

        //////////////////////////////////////////////////
        //
        // 转换肌肉空间到基础姿态
        //
		apHuman->m_SkeletonPose = apSkeletonPose->m_X;
        if (adjustMissingBones)
            HumanPoseAdjustForMissingBones(apHuman, apHumanPoseOut);
        Human2SkeletonPose(apHuman, apHumanPoseOut, apSkeletonPose);

        // 如果有额外的自由度（TDoF），处理这些自由度
        if (apHuman->m_HasTDoF)
        {
            RetargetToTDoF(apHuman, apHumanPoseOut, apHuman->m_SkeletonPose, apSkeletonPose, apSkeletonPoseWs);
        }

        // 计算全局骨骼姿态
        skeleton::SkeletonPoseComputeGlobal(apHuman->m_Skeleton, apSkeletonPose, apSkeletonPoseWs);

        ///////////////////////////////////////////////////////
        //
        // 调整 hips 的局部姿态
        //
        math::trsX rootX = HumanComputeRootXform(apHuman, apSkeletonPoseWs);
        apSkeletonPose->m_X[hipsIndex] = math::trsInvMulNS(rootX, apSkeletonPoseWs->m_X[hipsIndex]);
        apSkeletonPose->m_X[hipsIndex].s = apSkeletonPoseWs->m_X[hipsIndex].s;

        ////////////////////////////////////////////////////////
        //
        // 转换肌肉空间到输出姿态
        //
        if (apHumanPose)
        {
            // 复制输入的人体姿态到输出姿态
            human::HumanPoseCopy(*apHumanPoseOut, *apHumanPose, true);
            if (adjustMissingBones)
                HumanPoseAdjustForMissingBones(apHuman, apHumanPoseOut);
            Human2SkeletonPose(apHuman, apHumanPoseOut, apSkeletonPose);

            // 处理额外的自由度
            if (apHuman->m_HasTDoF)
            {
                RetargetToTDoF(apHuman, apHumanPoseOut, apHuman->m_SkeletonPose, apSkeletonPose, apSkeletonPoseWs);
            }
        }

        //////////////////////////////////////////////////
        //
        // 设置根骨骼的姿态
        //
        apSkeletonPose->m_X[rootIndex] = apHumanPoseOut->m_RootX;
    }

    math::float4 GetLookAtDeltaQ(math::float3 const &pivot, math::float3 const &eyesT, math::float4 const &eyesQ, math::float3 const &eyesDir, math::float3 const &target, math::float1 const &weight)
    {
        math::float1 len = math::length(target - eyesT);
        math::float3 dstV = target - pivot;
        math::float3 srcV = eyesT - math::quatMulVec(eyesQ, eyesDir * math::float1(len)) - pivot;

        return math::quatWeight(math::quatArcRotate(srcV, dstV), weight);
    }

    float ComputeHierarchicMass(int32_t aBoneIndex, float *apMassArray)
    {
        apMassArray[aBoneIndex] = HumanBoneDefaultMass[aBoneIndex];

        for (int childIter = 0; childIter < BoneChildren[aBoneIndex][0]; childIter++)
        {
            apMassArray[aBoneIndex] += ComputeHierarchicMass(BoneChildren[aBoneIndex][1 + childIter], apMassArray);
        }

        return apMassArray[aBoneIndex];
    }

    float DeltaPoseQuality(HumanPose &arDeltaPose, float aTol)
    {
        float massArray[kLastBone];
        ComputeHierarchicMass(0, massArray);

        float q = 0;
        float sumW = 0;

        for (int dofIter = 0; dofIter < kLastDoF; dofIter++)
        {
            int32_t boneIndex = DoF2Bone[dofIter];

            float v = math::saturate((aTol - math::abs(arDeltaPose.m_DoFArray[dofIter])) / aTol);

            q += v * massArray[boneIndex];

            sumW += massArray[boneIndex];
        }

        return q / sumW;
    }
} // namespace human
}


#ifndef BODY_MAIN_H
#define BODY_MAIN_H
#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "body_part.h"
#include "animation_help.h"


#include "modules/limboai/bt/bt_player.h"

// 身体的插槽信息
class BodySocket
{
    Transform3D localPose;
    Transform3D globalPose;

    void on_bone_pose_update(Skeleton3D *p_skeleton, int p_bone_index)
    {
        globalPose = p_skeleton->get_bone_global_pose(p_bone_index) * localPose;
    }

};
// 身体主要部件部分
class CharacterBodyMain : public CharacterBody3D {
    GDCLASS(CharacterBodyMain, CharacterBody3D);
    static void _bind_methods();

public:
    // 初始化身體
    void init_main_body(String p_skeleton_file_path,StringName p_animation_group);
    void clear_all();

public:
	void set_behavior_tree(const Ref<BehaviorTree> &p_tree)
    {
        get_bt_player()->set_behavior_tree(p_tree);
    }
	Ref<BehaviorTree> get_behavior_tree()  { return get_bt_player()->get_behavior_tree(); };

	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan)
    {
        get_bt_player()->set_blackboard_plan(p_plan);
    }
	Ref<BlackboardPlan> get_blackboard_plan() { return get_bt_player()->get_blackboard_plan(); }

	void set_update_mode(int p_mode)
    {
        get_bt_player()->set_update_mode((BTPlayer::UpdateMode)p_mode);
    }
	int get_update_mode() { return (int)(get_bt_player()->get_update_mode()); }

	Ref<Blackboard> get_blackboard()  { return get_bt_player()->get_blackboard(); }
	void set_blackboard(const Ref<Blackboard> &p_blackboard) { get_bt_player()->set_blackboard(p_blackboard); }

	void restart()
    {
        get_bt_player()->restart();
    }
	int get_last_status() { return get_bt_player()->get_last_status(); }

	Ref<BTTask> get_tree_instance() { return get_bt_player()->get_tree_instance(); }

protected:
    void load_skeleton();
    void load_mesh(const StringName& part_name,String p_mesh_file_path);
    BTPlayer * get_bt_player();
protected:
    void behavior_tree_finished(int last_status);
    void behavior_tree_update(int last_status);


protected:
    Skeleton3D *skeleton = nullptr;
    AnimationPlayer *player = nullptr;
    AnimationTree *tree = nullptr;
    mutable BTPlayer *btPlayer = nullptr;
    // 骨架配置文件
    String skeleton_res;
    // 动画组配置
    String animation_group;
    // 身體部件列表
    PackedStringArray   partList;
    // 插槽信息
    HashMap<StringName,BodySocket> socket;
    // 身体部件信息
    HashMap<StringName,CharacterBodyPartInstane> bodyPart;
    


};
struct CharacterAnimationInstance
{    
	Vector<real_t> track_weights;
    Vector<float> m_WeightArray;
    Vector<StringName>    m_ChildAnimationArray;
    Vector<AnimationMixer::PlaybackInfo> m_ChildAnimationPlaybackArray;
    float time;
    float delta;
};

// 动画遮罩
class CharacterAnimatorMask : public Resource
{
    GDCLASS(CharacterAnimatorMask, Resource);
};
class CharacterAnimatorNodeBase : public Resource
{
    GDCLASS(CharacterAnimatorNodeBase, Resource);

    enum PlayState
    {
        PS_None,
        PS_FadeIn,
        PS_Play,
        PS_FadeOut,
    };
    PlayState m_PlayState = PS_None;
public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard,const StringName & property_name)
    {

    }
public:
    void _blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,CharacterAnimationInstance *p_playback_info,float total_weight,const Vector<float> &weight_array);

    
    struct MotionNeighborList
    {

        MotionNeighborList() : m_Count(0)
        {
        }

        uint32_t m_Count;
        Vector<uint32_t> m_NeighborArray;
    };
    struct Blend1dDataConstant
    {

        Blend1dDataConstant() : m_ChildCount(0)
        {
        }

        uint32_t            m_ChildCount;
        Vector<float>    m_ChildThresholdArray;

    };
    struct Blend2dDataConstant
    {

        Blend2dDataConstant()
        {
        }

        uint32_t                m_ChildCount;
        Vector<Vector2>     m_ChildPositionArray;

        uint32_t                m_ChildMagnitudeCount;
        Vector<float>        m_ChildMagnitudeArray; // Used by type 2
        uint32_t                m_ChildPairVectorCount;
        Vector<Vector2>     m_ChildPairVectorArray; // Used by type 2, (3 TODO)
        uint32_t                m_ChildPairAvgMagInvCount;
        Vector<float>        m_ChildPairAvgMagInvArray; // Used by type 2
        uint32_t                        m_ChildNeighborListCount;
        Vector<MotionNeighborList>   m_ChildNeighborListArray; // Used by type 2, (3 TODO)

    };
        // Constant data for direct blend node types - parameters
    struct BlendDirectDataConstant
    {

        BlendDirectDataConstant() : m_ChildCount(0), m_NormalizedBlendValues(0)
        {
        }

        uint32_t            m_ChildCount;
        Vector<uint32_t> m_ChildBlendEventIDArray;
        bool                m_NormalizedBlendValues;
    };



    static float WeightForIndex(const float* thresholdArray, uint32_t count, uint32_t index, float blend)
    {
        if (blend >= thresholdArray[index])
        {
            if (index + 1 == count)
            {
                return 1.0f;
            }
            else if (thresholdArray[index + 1] < blend)
            {
                return 0.0f;
            }
            else
            {
                if (thresholdArray[index] - thresholdArray[index + 1] != 0)
                {
                    return (blend - thresholdArray[index + 1]) / (thresholdArray[index] - thresholdArray[index + 1]);
                }
                else
                {
                    return 1.0f;
                }
            }
        }
        else
        {
            if (index == 0)
            {
                return 1.0f;
            }
            else if (thresholdArray[index - 1] > blend)
            {
                return 0.0f;
            }
            else
            {
                if ((thresholdArray[index] - thresholdArray[index - 1]) != 0)
                {
                    return (blend - thresholdArray[index - 1]) / (thresholdArray[index] - thresholdArray[index - 1]);
                }
                else
                {
                    return 1.0f;
                }
            }
        }
    }

       static void GetWeightsSimpleDirectional(const Blend2dDataConstant& blendConstant,
        float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
        float blendValueX, float blendValueY, bool preCompute = false)
    {
        // Get constants
        const Vector2* positionArray = blendConstant.m_ChildPositionArray.ptr();
        uint32_t count = blendConstant.m_ChildCount;

        if (weightArray == NULL || positionArray == NULL)
            return;

        // Initialize all weights to 0
        for (uint32_t i = 0; i < count; i++)
            weightArray[i] = 0;

        // Handle fallback
        if (count < 2)
        {
            if (count == 1)
                weightArray[0] = 1;
            return;
        }

        Vector2 blendPosition = Vector2(blendValueX, blendValueY);

        // Handle special case when sampled ecactly in the middle
        if (blendPosition == Vector2(0,0))
        {
            // If we have a center motion, give that one all the weight
            for (uint32_t i = 0; i < count; i++)
            {
                if (positionArray[i] == Vector2(0, 0))
                {
                    weightArray[i] = 1;
                    return;
                }
            }

            // Otherwise divide weight evenly
            float sharedWeight = 1.0f / count;
            for (uint32_t i = 0; i < count; i++)
                weightArray[i] = sharedWeight;
            return;
        }

        int indexA = -1;
        int indexB = -1;
        int indexCenter = -1;
        float maxDotForNegCross = -100000.0f;
        float maxDotForPosCross = -100000.0f;
        for (uint32_t i = 0; i < count; i++)
        {
            if (positionArray[i] == Vector2(0, 0))
            {
                if (indexCenter >= 0)
                    return;
                indexCenter = i;
                continue;
            }
            Vector2 posNormalized = positionArray[i];
            posNormalized.normalize();
            float dot = posNormalized.dot(blendPosition);
            float cross = posNormalized.x * blendPosition.y - posNormalized.y * blendPosition.x;
            if (cross > 0)
            {
                if (dot > maxDotForPosCross)
                {
                    maxDotForPosCross = dot;
                    indexA = i;
                }
            }
            else
            {
                if (dot > maxDotForNegCross)
                {
                    maxDotForNegCross = dot;
                    indexB = i;
                }
            }
        }

        float centerWeight = 0;

        if (indexA < 0 || indexB < 0)
        {
            // Fallback if sampling point is not inside a triangle
            centerWeight = 1;
        }
        else
        {
            Vector2 a = positionArray[indexA];
            Vector2 b = positionArray[indexB];

            // Calculate weights using barycentric coordinates
            // (formulas from http://en.wikipedia.org/wiki/Barycentric_coordinate_system_%28mathematics%29 )
            float det = b.y * a.x - b.x * a.y;        // Simplified from: (b.y-0)*(a.x-0) + (0-b.x)*(a.y-0);
            float wA = (b.y * blendValueX - b.x * blendValueY) / det; // Simplified from: ((b.y-0)*(l.x-0) + (0-b.x)*(l.y-0)) / det;
            float wB = (a.x * blendValueY - a.y * blendValueX) / det; // Simplified from: ((0-a.y)*(l.x-0) + (a.x-0)*(l.y-0)) / det;
            centerWeight = 1 - wA - wB;

            // Clamp to be inside triangle
            if (centerWeight < 0)
            {
                centerWeight = 0;
                float sum = wA + wB;
                wA /= sum;
                wB /= sum;
            }
            else if (centerWeight > 1)
            {
                centerWeight = 1;
                wA = 0;
                wB = 0;
            }

            // Give weight to the two vertices on the periphery that are closest
            weightArray[indexA] = wA;
            weightArray[indexB] = wB;
        }

        if (indexCenter >= 0)
        {
            weightArray[indexCenter] = centerWeight;
        }
        else
        {
            // Give weight to all children when input is in the center
            float sharedWeight = 1.0f / count;
            for (uint32_t i = 0; i < count; i++)
                weightArray[i] += sharedWeight * centerWeight;
        }
    }

   static  float GetWeightFreeformDirectional(const Blend2dDataConstant& blendConstant, Vector2* workspaceBlendVectors, int i, int j, Vector2 blendPosition)
    {
        int pairIndex = i + j * blendConstant.m_ChildCount;
        Vector2 vecIJ = blendConstant.m_ChildPairVectorArray[pairIndex];
        Vector2 vecIO = workspaceBlendVectors[i];
        vecIO.y *= blendConstant.m_ChildPairAvgMagInvArray[pairIndex];

        if (blendConstant.m_ChildPositionArray[i] == Vector2(0, 0))
            vecIJ.x = workspaceBlendVectors[j].x;
        else if (blendConstant.m_ChildPositionArray[j] == Vector2(0, 0))
            vecIJ.x = workspaceBlendVectors[i].x;
        else if (vecIJ.x == 0 || blendPosition == Vector2(0, 0))
            vecIO.x = vecIJ.x;

        return 1 - vecIJ.dot(vecIO) / vecIJ.length_squared();
    }

    static void GetWeightsFreeformDirectional(const Blend2dDataConstant& blendConstant,
        float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
        float blendValueX, float blendValueY, bool preCompute = false)
    {
        // Get constants
        const Vector2* positionArray = blendConstant.m_ChildPositionArray.ptr();
        uint32_t count = blendConstant.m_ChildCount;
        const float* constantMagnitudes = blendConstant.m_ChildMagnitudeArray.ptr();
        const MotionNeighborList* constantChildNeighborLists = blendConstant.m_ChildNeighborListArray.ptr();

        Vector2 blendPosition = Vector2(blendValueX, blendValueY);
        float magO = blendPosition.length();

        if (blendPosition == Vector2(0, 0))
        {
            for (uint32_t i = 0; i < count; i++)
                workspaceBlendVectors[i] = Vector2(0, magO - constantMagnitudes[i]);
        }
        else
        {
            for (uint32_t i = 0; i < count; i++)
            {
                if (positionArray[i] == Vector2(0, 0))
                    workspaceBlendVectors[i] = Vector2(0, magO - constantMagnitudes[i]);
                else
                {
                    float angle = positionArray[i].angle_to( blendPosition);
                    if (positionArray[i].x * blendPosition.y - positionArray[i].y * blendPosition.x < 0)
                        angle = -angle;
                    workspaceBlendVectors[i] = Vector2(angle, magO - constantMagnitudes[i]);
                }
            }
        }

        const float kInversePI = 1 / Math_PI;
        if (preCompute)
        {
            for (uint32_t i = 0; i < count; i++)
            {
                // Fade out over 180 degrees away from example
                float value = 1 - Math::abs(workspaceBlendVectors[i].x) * kInversePI;
                cropArray[i] = -1;
                for (uint32_t j = 0; j < count; j++)
                {
                    if (i == j)
                        continue;

                    float newValue = GetWeightFreeformDirectional(blendConstant, workspaceBlendVectors, i, j, blendPosition);

                    if (newValue <= 0)
                    {
                        value = 0;
                        cropArray[i] = -1;
                        break;
                    }
                    // Used for determining neighbors
                    if (newValue < value)
                        cropArray[i] = j;
                    value = MIN(value, newValue);
                }
            }
            return;
        }

        for (uint32_t i = 0; i < count; i++)
        {
            // Fade out over 180 degrees away from example
            float value = 1 - Math::abs(workspaceBlendVectors[i].x) * kInversePI;
            for (uint32_t jIndex = 0; jIndex < constantChildNeighborLists[i].m_Count; jIndex++)
            {
                int j = constantChildNeighborLists[i].m_NeighborArray[jIndex];
                float newValue = GetWeightFreeformDirectional(blendConstant, workspaceBlendVectors, i, j, blendPosition);
                if (newValue <= 0)
                {
                    value = 0;
                    break;
                }
                value = MIN(value, newValue);
            }
            weightArray[i] = value;
        }

        // Normalize weights
        float summedWeight = 0;
        for (uint32_t i = 0; i < count; i++)
            summedWeight += weightArray[i];

        if (summedWeight > 0)
        {
            summedWeight = 1.0f / summedWeight; // Do division once instead of for every sample
            for (uint32_t i = 0; i < count; i++)
                weightArray[i] *= summedWeight;
        }
        else
        {
            // Give weight to all children as fallback when no children have any weight.
            // This happens when sampling in the center if no center motion is provided.
            float evenWeight = 1.0f / count;
            for (uint32_t i = 0; i < count; i++)
                weightArray[i] = evenWeight;
        }
    }

    static void GetWeightsFreeformCartesian(const Blend2dDataConstant& blendConstant,
        float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
        float blendValueX, float blendValueY, bool preCompute = false)
    {
        // Get constants
        const Vector2* positionArray = blendConstant.m_ChildPositionArray.ptr();
        uint32_t count = blendConstant.m_ChildCount;
        const MotionNeighborList* constantChildNeighborLists = blendConstant.m_ChildNeighborListArray.ptr();

        Vector2 blendPosition = Vector2(blendValueX, blendValueY);
        for (uint32_t i = 0; i < count; i++)
            workspaceBlendVectors[i] = blendPosition - positionArray[i];

        if (preCompute)
        {
            for (uint32_t i = 0; i < count; i++)
            {
                cropArray[i] = -1;
                Vector2 vecIO = workspaceBlendVectors[i];
                float value = 1;
                for (uint32_t j = 0; j < count; j++)
                {
                    if (i == j)
                        continue;

                    int pairIndex = i + j * blendConstant.m_ChildCount;
                    Vector2 vecIJ = blendConstant.m_ChildPairVectorArray[pairIndex];
                    float newValue = 1 - vecIJ.dot( vecIO) * blendConstant.m_ChildPairAvgMagInvArray[pairIndex];
                    if (newValue <= 0)
                    {
                        value = 0;
                        cropArray[i] = -1;
                        break;
                    }
                    // Used for determining neighbors
                    if (newValue < value)
                        cropArray[i] = j;
                    value = MIN(value, newValue);
                }
            }
            return;
        }

        for (uint32_t i = 0; i < count; i++)
        {
            Vector2 vecIO = workspaceBlendVectors[i];
            float value = 1;
            for (uint32_t jIndex = 0; jIndex < constantChildNeighborLists[i].m_Count; jIndex++)
            {
                uint32_t j = constantChildNeighborLists[i].m_NeighborArray[jIndex];
                if (i == j)
                    continue;

                int pairIndex = i + j * blendConstant.m_ChildCount;
                Vector2 vecIJ = blendConstant.m_ChildPairVectorArray[pairIndex];
                float newValue = 1 - vecIJ.dot( vecIO) * blendConstant.m_ChildPairAvgMagInvArray[pairIndex];
                if (newValue < 0)
                {
                    value = 0;
                    break;
                }
                value = MIN(value, newValue);
            }
            weightArray[i] = value;
        }

        // Normalize weights
        float summedWeight = 0;
        for (uint32_t i = 0; i < count; i++)
            summedWeight += weightArray[i];
        summedWeight = 1.0f / summedWeight; // Do division once instead of for every sample
        for (uint32_t i = 0; i < count; i++)
            weightArray[i] *= summedWeight;
    }

    static void GetWeights1d(const Blend1dDataConstant& blendConstant, float* weightArray, float blendValue)
    {
        blendValue = CLAMP(blendValue, blendConstant.m_ChildThresholdArray[0], blendConstant.m_ChildThresholdArray[blendConstant.m_ChildCount - 1]);
        for (uint32_t j = 0; j < blendConstant.m_ChildCount; j++)
            weightArray[j] = WeightForIndex(blendConstant.m_ChildThresholdArray.ptr(), blendConstant.m_ChildCount, j, blendValue);
    }

    // void GetWeights(const BlendTreeNodeConstant& nodeConstant, BlendTreeWorkspace &workspace, float* weightArray, float blendValueX, float blendValueY)
    // {
    //     if (nodeConstant.m_BlendType == Simple1D)
    //         GetWeights1d(*nodeConstant.m_Blend1dData.Get(), weightArray, blendValueX);
    //     else if (nodeConstant.m_BlendType == SimpleDirectionnal2D)
    //         GetWeightsSimpleDirectional(*nodeConstant.m_Blend2dData.Get(), weightArray, workspace.m_TempCropArray, workspace.m_ChildInputVectorArray, blendValueX, blendValueY);
    //     else if (nodeConstant.m_BlendType == FreeformDirectionnal2D)
    //         GetWeightsFreeformDirectional(*nodeConstant.m_Blend2dData.Get(), weightArray, workspace.m_TempCropArray, workspace.m_ChildInputVectorArray, blendValueX, blendValueY);
    //     else if (nodeConstant.m_BlendType == FreeformCartesian2D)
    //         GetWeightsFreeformCartesian(*nodeConstant.m_Blend2dData.Get(), weightArray, workspace.m_TempCropArray, workspace.m_ChildInputVectorArray, blendValueX, blendValueY);
    //     else if (nodeConstant.m_BlendType == Direct)
    //         GetWeightsDirect(*nodeConstant.m_BlendDirectData.Get(), weightArray);
    // }


};
class CharacterAnimatorNode1D : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorNode1D, CharacterAnimatorNodeBase);
public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard,const StringName & property_name) override;
protected:
    Blend1dDataConstant m_BlendData;
};
class CharacterAnimatorNode2D : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorNode2D, CharacterAnimatorNodeBase);
public:
    enum BlendType
    {
        SimpleDirectionnal2D = 1,
        FreeformDirectionnal2D = 2,
        FreeformCartesian2D = 3,
    };
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard,const StringName & property_name) override;
protected:
    BlendType m_BlendType;
    Blend2dDataConstant m_BlendData;
};
// 动画分层
class CharacterAnimatorLayer: public AnimationMixer
{
    GDCLASS(CharacterAnimatorLayer, AnimationMixer);

    List<Ref<CharacterAnimatorNodeBase>> play_list;
public:

    Ref<CharacterAnimatorMask> mask;

    enum BlendState
    {
        // 混合
        BS_Blend,
        // 覆盖
        BS_Override,
    };
    Vector<Vector2> m_ChildInputVectorArray;
    Vector<int> m_TempCropArray;
};

// 人物动画器
class CharacterAnimator : public Resource
{
    GDCLASS(CharacterAnimator, Resource);

    static void _bind_methods();


};

#endif
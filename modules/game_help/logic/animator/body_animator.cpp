#include "body_animator.h"
#include "../body_main.h"
#include "modules/realtime_retarget/src/retarget_utility.h"



void CharacterBoneMap::init_skeleton_bone_map()
{
    if(!is_by_sekeleton_file)
    {
        return;
    }
    if(!FileAccess::exists(ref_skeleton_file_path))
    {
        return;
    }
    Ref<PackedScene> scene = ResourceLoader::load(ref_skeleton_file_path);

    if(!scene.is_valid())
    {
        return;
    }
    Node* node = scene->instantiate();
    if(node == nullptr)
    {
        return;
    }
    Skeleton3D* skele = Object::cast_to<Skeleton3D>( node->get_node(NodePath("Seleton3D")));
    if(skele == nullptr)
    {
        return;
    }
    bone_map = skele->get_human_bone_mapping();
    node->queue_free();
    is_init_skeleton = true;

}

void CharacterAnimationItem::bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_animation_name", "animation_name"), &CharacterAnimationItem::set_animation_name);
    ClassDB::bind_method(D_METHOD("get_animation_name"), &CharacterAnimationItem::get_animation_name);

    ClassDB::bind_method(D_METHOD("set_animation_path", "animation_path"), &CharacterAnimationItem::set_animation_path);
    ClassDB::bind_method(D_METHOD("get_animation_path"), &CharacterAnimationItem::get_animation_path);

    ClassDB::bind_method(D_METHOD("set_bone_map_path", "bone_map_path"), &CharacterAnimationItem::set_bone_map_path);
    ClassDB::bind_method(D_METHOD("get_bone_map_path"), &CharacterAnimationItem::get_bone_map_path);

    ClassDB::bind_method(D_METHOD("set_speed", "speed"), &CharacterAnimationItem::set_speed);
    ClassDB::bind_method(D_METHOD("get_speed"), &CharacterAnimationItem::get_speed);

    ClassDB::bind_method(D_METHOD("set_is_clip", "is_clip"), &CharacterAnimationItem::set_is_clip);
    ClassDB::bind_method(D_METHOD("get_is_clip"), &CharacterAnimationItem::get_is_clip);

    ClassDB::bind_method(D_METHOD("set_child_node", "child_node"), &CharacterAnimationItem::set_child_node);
    ClassDB::bind_method(D_METHOD("get_child_node"), &CharacterAnimationItem::get_child_node);

    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "animation_name"), "set_animation_name", "get_animation_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "animation_path"), "set_animation_path", "get_animation_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "bone_map_path"), "set_bone_map_path", "get_bone_map_path");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed"), "set_speed", "get_speed");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_clip"), "set_is_clip", "get_is_clip");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "child_node"), "set_child_node", "get_child_node");

}
void CharacterAnimationItem::set_child_node(const Ref<CharacterAnimatorNodeBase>& p_child_node) 
{ 
    child_node = p_child_node; 
}
Ref<CharacterAnimatorNodeBase> CharacterAnimationItem::get_child_node() 
{
        return child_node; 
}
Ref<Animation> CharacterAnimationItem::get_animation()
{
    if(animation.is_null())
    {
        if(FileAccess::exists(animation_path))
        {
            animation = ResourceLoader::load(animation_path);
        }
    }
    return animation;
}
Ref<CharacterBoneMap> CharacterAnimationItem::get_bone_map()
{
    if(bone_map.is_null())
    {
        if(FileAccess::exists(bone_map_path))
        {
            bone_map = ResourceLoader::load(bone_map_path);
        }
    }
    return bone_map;
}

void CharacterAnimationItem::_init()
{
    if(is_init)
    {
        return;
    }
    if(is_clip)
    {
        if(FileAccess::exists(animation_path))
        {
        animation = ResourceLoader::load(animation_path);
        }
        if(FileAccess::exists(bone_map_path))
        {
            bone_map = ResourceLoader::load(bone_map_path);
        }
    }
    else
    {
        if(child_node.is_valid())
        {
            child_node->_init();
        }
    }
    is_init = true;
}
float CharacterAnimationItem::_get_animation_length()
{
    _init();
    if(is_clip)
    {
        if(animation.is_valid())
            return ABS(animation->get_length() * speed);
    }
    else
    {
        return child_node->_get_animation_length();
    }
    return 0.0f;
}
void CharacterAnimationItem::_set_animation_scale_by_length(float p_length)
{

}



void CharacterAnimatorNodeBase::bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_animation_arrays", "animation_arrays"), &CharacterAnimatorNodeBase::set_animation_arrays);
    ClassDB::bind_method(D_METHOD("get_animation_arrays"), &CharacterAnimatorNodeBase::get_animation_arrays);

    ClassDB::bind_method(D_METHOD("set_black_board_property", "black_board_property"), &CharacterAnimatorNodeBase::set_black_board_property);
    ClassDB::bind_method(D_METHOD("get_black_board_property"), &CharacterAnimatorNodeBase::get_black_board_property);

    ClassDB::bind_method(D_METHOD("set_black_board_property_y", "black_board_property"), &CharacterAnimatorNodeBase::set_black_board_property_y);
    ClassDB::bind_method(D_METHOD("get_black_board_property_y"), &CharacterAnimatorNodeBase::get_black_board_property_y);

    ClassDB::bind_method(D_METHOD("set_fade_out_time", "fade_out_time"), &CharacterAnimatorNodeBase::set_fade_out_time);
    ClassDB::bind_method(D_METHOD("get_fade_out_time"), &CharacterAnimatorNodeBase::get_fade_out_time);

    ClassDB::bind_method(D_METHOD("set_loop", "loop"), &CharacterAnimatorNodeBase::set_loop);
    ClassDB::bind_method(D_METHOD("get_loop"), &CharacterAnimatorNodeBase::get_loop);

    ClassDB::bind_method(D_METHOD("set_loop_count", "loop_count"), &CharacterAnimatorNodeBase::set_loop_count);
    ClassDB::bind_method(D_METHOD("get_loop_count"), &CharacterAnimatorNodeBase::get_loop_count);


    ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "animation_arrays"), "set_animation_arrays", "get_animation_arrays");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "black_board_property"), "set_black_board_property", "get_black_board_property");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "black_board_property_y"), "set_black_board_property_y", "get_black_board_property_y");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fade_out_time"), "set_fade_out_time", "get_fade_out_time");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "loop", PROPERTY_HINT_ENUM, "Once,Clamp Count,PingPong Once,PingPong Count"), "set_loop", "get_loop");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_count"), "set_loop_count", "get_loop_count");
    
    BIND_ENUM_CONSTANT(LOOP_Once);
    BIND_ENUM_CONSTANT(LOOP_ClampCount);
    BIND_ENUM_CONSTANT(LOOP_PingPongOnce);
    BIND_ENUM_CONSTANT(LOOP_PingPongCount);
}    
void CharacterAnimatorNodeBase::_init()
{
    for(int i = 0; i < animation_arrays.size(); ++i)
    {
        Ref<CharacterAnimationItem> item = animation_arrays[i];
        if(item.is_valid())
        {
            item->_init();
        }
    }
}

void CharacterAnimatorNodeBase::_blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,CharacterAnimationInstance *p_playback_info,float total_weight,const Vector<float> &weight_array,const Ref<Blackboard> &p_blackboard)
{
    touch();
    AnimationMixer::PlaybackInfo * p_playback_info_ptr = p_playback_info->m_ChildAnimationPlaybackArray.ptrw();
    for (int32_t i = 0; i < child_count; i++)
    {
        float w = weight_array[i] * total_weight;
        if(w > 0.01f)
        {	  
            Ref<CharacterAnimationItem> item = animation_arrays[i];
            if(item->is_clip){
                p_playback_info_ptr[i].weight = w;
                p_playback_info_ptr[i].delta = p_playback_info->delta * item->get_speed();
                p_playback_info_ptr[i].time += p_playback_info_ptr[i].delta ;
                p_playback_info_ptr[i].disable_path = p_playback_info->disable_path;
                Ref<Animation> animation = item->get_animation();
                if(animation.is_valid())
                {
                    Ref<CharacterBoneMap> bone_map = item->get_bone_map();
                    Dictionary bp;
                    if(bone_map.is_valid())
                    {
                        bp = bone_map->bone_map;
                    }
                    p_layer->make_animation_instance_anim(item->animation, p_playback_info_ptr[i],bp);
                }
                else
                {
                    p_layer->make_animation_instance(item->animation_name, p_playback_info_ptr[i]);                    
                }
            }
            else if(item->child_node.is_valid())
            {
                // 动画节点递归处理
                item->child_node->process_animation(p_layer,p_playback_info,w,p_blackboard);
            }
        }
    }
}
void CharacterAnimatorNodeBase::_normal_animation_length()
{
    float length = _get_animation_length();
    if(length > 0.0f)
    {
        _set_animation_scale_by_length(length);
    }
}
void CharacterAnimatorNodeBase::_set_animation_scale_by_length(float p_length)
{

}
float CharacterAnimatorNodeBase::_get_animation_length()
{
    float length = 0.0f;
    for(int i = 0; i < animation_arrays.size(); ++i)
    {
        Ref<CharacterAnimationItem> item = animation_arrays[i];
        if(item.is_valid())
        {
            length = MAX(length,item->_get_animation_length());
        }
    }
    return length;
}

float CharacterAnimatorNodeBase::weight_for_index(const float* thresholdArray, uint32_t count, uint32_t index, float blend)
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

void CharacterAnimatorNodeBase::get_weights_simple_directional(const Blend2dDataConstant& blendConstant,
    float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
    float blendValueX, float blendValueY, bool preCompute)
{
    // Get constants
    const Vector2* positionArray = blendConstant.position_array.ptr();
    uint32_t count = blendConstant.position_count;

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

float CharacterAnimatorNodeBase::get_weight_freeform_directional(const Blend2dDataConstant& blendConstant, Vector2* workspaceBlendVectors, int i, int j, Vector2 blendPosition)
{
    int pairIndex = i + j * blendConstant.position_count;
    Vector2 vecIJ = blendConstant.m_ChildPairVectorArray[pairIndex];
    Vector2 vecIO = workspaceBlendVectors[i];
    vecIO.y *= blendConstant.m_ChildPairAvgMagInvArray[pairIndex];

    if (blendConstant.position_array[i] == Vector2(0, 0))
        vecIJ.x = workspaceBlendVectors[j].x;
    else if (blendConstant.position_array[j] == Vector2(0, 0))
        vecIJ.x = workspaceBlendVectors[i].x;
    else if (vecIJ.x == 0 || blendPosition == Vector2(0, 0))
        vecIO.x = vecIJ.x;

    return 1 - vecIJ.dot(vecIO) / vecIJ.length_squared();
}

void CharacterAnimatorNodeBase::get_weights_freeform_directional(const Blend2dDataConstant& blendConstant,
    float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
    float blendValueX, float blendValueY, bool preCompute )
{
    // Get constants
    const Vector2* positionArray = blendConstant.position_array.ptr();
    uint32_t count = blendConstant.position_count;
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

                float newValue = get_weight_freeform_directional(blendConstant, workspaceBlendVectors, i, j, blendPosition);

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
            float newValue = get_weight_freeform_directional(blendConstant, workspaceBlendVectors, i, j, blendPosition);
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

void CharacterAnimatorNodeBase::get_weights_freeform_cartesian(const Blend2dDataConstant& blendConstant,
    float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
    float blendValueX, float blendValueY, bool preCompute )
{
    // Get constants
    const Vector2* positionArray = blendConstant.position_array.ptr();
    uint32_t count = blendConstant.position_count;
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

                int pairIndex = i + j * blendConstant.position_count;
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

            int pairIndex = i + j * blendConstant.position_count;
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

void CharacterAnimatorNodeBase::get_weights1d(const Blend1dDataConstant& blendConstant, float* weightArray, float blendValue)
{
    blendValue = CLAMP(blendValue, blendConstant.position_array[0], blendConstant.position_array[blendConstant.position_count - 1]);
    for (uint32_t j = 0; j < blendConstant.position_count; j++)
        weightArray[j] = weight_for_index(blendConstant.position_array.ptr(), blendConstant.position_count, j, blendValue);
}

void CharacterAnimatorNode1D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
{
    if(!p_blackboard->has_var(black_board_property))
    {
        return;
    }
    float v = p_blackboard->get_var(black_board_property,0);
    if(p_playback_info->m_WeightArray.size() != blend_data.position_count)
    {
        p_playback_info->m_WeightArray.resize(blend_data.position_count);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(blend_data.position_count);
    }
    get_weights1d(blend_data, p_playback_info->m_WeightArray.ptrw(), v);
    _blend_anmation(p_layer,blend_data.position_count, p_playback_info, total_weight,p_playback_info->m_WeightArray,p_blackboard);

}
void CharacterAnimatorNode2D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
{
    if(!p_blackboard->has_var(black_board_property))
    {
        return;
    }
    Vector2 v = p_blackboard->get_var(black_board_property,0);
    if(p_playback_info->m_WeightArray.size() != blend_data.position_count)
    {
        p_playback_info->m_WeightArray.resize(blend_data.position_count);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(blend_data.position_count);
    }
    if(p_layer->m_TempCropArray.size() < blend_data.position_count)
    {
        p_layer->m_TempCropArray.resize(blend_data.position_count);
        p_layer->m_ChildInputVectorArray.resize(blend_data.position_count);
    }
    if (blend_type == SimpleDirectionnal2D)
        get_weights_simple_directional(blend_data, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else if (blend_type == FreeformDirectionnal2D)
        get_weights_freeform_directional(blend_data, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else if (blend_type == FreeformCartesian2D)
        get_weights_freeform_cartesian(blend_data, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else 
        return;

    _blend_anmation(p_layer, blend_data.position_count,p_playback_info, total_weight,p_playback_info->m_WeightArray,p_blackboard);

}
void CharacterAnimatorLayer::_process_logic(const Ref<Blackboard>& p_playback_info, double p_delta, bool is_first)
{
    if(logic_context.animation_logic.is_null())
    {
        return;
    }
    if(logic_context.curr_name == StringName() )
    {
        logic_context.curr_name = logic_context.animation_logic->get_default_state_name();
    }
    Ref<CharacterAnimationLogicNode>   curr_logic = logic_context.curr_logic;


    bool change_state = false;
    if(curr_logic.is_null() || logic_context.curr_name != logic_context.last_name)
    {
        change_state = true;
    }

    else if(curr_logic.is_valid())
    {
        if(curr_logic->check_stop(this,*p_playback_info))
        {
            change_state = true;
        }
    }

    if(change_state)
    {
        curr_logic = logic_context.animation_logic->process_logic(logic_context.curr_name, *p_playback_info);
    }
    if(curr_logic.is_null())
    {
        return;
    }
    if(curr_logic != logic_context.curr_logic)
    {
        logic_context.curr_logic->process_stop(this,*p_playback_info);
        logic_context.curr_logic = curr_logic;
        logic_context.curr_logic->process_start(this,*p_playback_info);
    }
    logic_context.last_name = logic_context.curr_name;
    curr_logic->process(this,*p_playback_info,p_delta);
}
// 处理动画
void CharacterAnimatorLayer::_process_animation(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first)
{
    // 处理逻辑节点请求播放的动作
    if(logic_context.curr_animation.is_valid())
    {
        auto anim = logic_context.curr_animation->get_node();
        if(play_animation(anim))
        {
            logic_context.curr_animation_time_length = anim->_get_animation_length();
            logic_context.curr_animation_play_time = 0.0f;
        }
    }
    logic_context.curr_animation_play_time += p_delta;
    
    for(auto& anim : m_AnimationInstances)
    {

        anim.delta = p_delta;
        anim.time += p_delta;
        if(anim.m_PlayState == CharacterAnimationInstance::PS_FadeOut)
        {
            anim.fadeTotalTime += p_delta;
        }
    }
    auto it = m_AnimationInstances.begin();
    float total_weight = 0.0f;
    while(it != m_AnimationInstances.end())
    {
        if(it->get_weight() == 0.0f)
        {
            it = m_AnimationInstances.erase(it);
        }
        else
        {
            total_weight += it->get_weight();
            ++it;
        }
    }
    
    if(m_TotalAnimationWeight.size() < m_AnimationInstances.size())
    {
        m_TotalAnimationWeight.resize(m_AnimationInstances.size());
    }
    for(auto& anim : m_AnimationInstances)
    {
        anim.node->process_animation(this, &anim, it->get_weight() / total_weight, p_playback_info);
    }


	_blend_init();

	cb_begin_animation.call(this, p_delta, false);

	if (_blend_pre_process(p_delta, track_count, track_map)) {
		_blend_capture(p_delta);
		_blend_calc_total_weight();
		_blend_process(p_delta, false);
        // 混合
		layer_blend_apply();
		_blend_post_process();

		cb_end_animation.call(this, p_delta, false);
	};

    
	clear_animation_instances();
}

void CharacterAnimatorLayer::layer_blend_apply() {
	// Finally, set the tracks.
	for (const KeyValue<Animation::TypeHash, TrackCache *> &K : track_cache) {
		TrackCache *track = K.value;
		if (!deterministic && Math::is_zero_approx(track->total_weight)) {
			continue;
		}
		switch (track->type) {
			case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
				TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

				if (t->root_motion) {
					root_motion_position = root_motion_cache.loc;
					root_motion_rotation = root_motion_cache.rot;
					root_motion_scale = root_motion_cache.scale - Vector3(1, 1, 1);
					root_motion_position_accumulator = t->loc;
					root_motion_rotation_accumulator = t->rot;
					root_motion_scale_accumulator = t->scale;
				} else if (t->skeleton_id.is_valid() && t->bone_idx >= 0) {
					Skeleton3D *t_skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(t->skeleton_id));
					if (!t_skeleton) {
						return;
					}
					if (t->loc_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
						    t_skeleton->set_bone_pose_position(t->bone_idx, t_skeleton->get_bone_pose_position(t->bone_idx).lerp(t->loc,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_position(t->bone_idx, t->loc);
                        }
					}
					if (t->rot_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_skeleton->set_bone_pose_rotation(t->bone_idx, t_skeleton->get_bone_pose_rotation(t->bone_idx).slerp(t->rot,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_rotation(t->bone_idx, t->rot);
                        }                        
					}
					if (t->scale_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_skeleton->set_bone_pose_scale(t->bone_idx, t_skeleton->get_bone_pose_scale(t->bone_idx).lerp(t->scale,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_scale(t->bone_idx, t->scale);
                        }
					}

				} else if (!t->skeleton_id.is_valid()) {
					Node3D *t_node_3d = Object::cast_to<Node3D>(ObjectDB::get_instance(t->object_id));
					if (!t_node_3d) {
						return;
					}
					if (t->loc_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_node_3d->set_position(t_node_3d->get_position().lerp(t->loc,blend_weight));
                        }
                        else
                        {
                            t_node_3d->set_position(t->loc);
                        }
					}
					if (t->rot_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_node_3d->set_rotation(t_node_3d->get_rotation().slerp(t->rot.get_euler(),blend_weight));
                        }
                        else
                        {
                            t_node_3d->set_rotation(t->rot.get_euler());
                        }
					}
					if (t->scale_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_node_3d->set_scale(t_node_3d->get_scale().lerp(t->scale,blend_weight));
                        }
                        else
                        {                            
                            t_node_3d->set_scale(t->scale);
                        }
					}
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
				TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);

				MeshInstance3D *t_mesh_3d = Object::cast_to<MeshInstance3D>(ObjectDB::get_instance(t->object_id));
				if (t_mesh_3d) {
                    if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                    {
                        t_mesh_3d->set_blend_shape_value(t->shape_index, Math::lerp( t_mesh_3d->get_blend_shape_value(t->shape_index),t->value,blend_weight));
                    }
                    else
                    {
                        t_mesh_3d->set_blend_shape_value(t->shape_index, t->value);
                    }
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_VALUE: {
				TrackCacheValue *t = static_cast<TrackCacheValue *>(track);

				if (!t->is_variant_interpolatable || (callback_mode_discrete == ANIMATION_CALLBACK_MODE_DISCRETE_DOMINANT && t->use_discrete)) {
					break; // Don't overwrite the value set by UPDATE_DISCRETE.
				}

				// Trim unused elements if init array/string is not blended.
				if (t->value.is_array()) {
					int actual_blended_size = (int)Math::round(Math::abs(t->element_size.operator real_t()));
					if (actual_blended_size < (t->value.operator Array()).size()) {
						real_t abs_weight = Math::abs(track->total_weight);
						if (abs_weight >= 1.0) {
							(t->value.operator Array()).resize(actual_blended_size);
						} else if (t->init_value.is_string()) {
							(t->value.operator Array()).resize(Animation::interpolate_variant((t->init_value.operator String()).length(), actual_blended_size, abs_weight));
						}
					}
				}

				Object *t_obj = ObjectDB::get_instance(t->object_id);
				if (t_obj) {
					t_obj->set_indexed(t->subpath, Animation::cast_from_blendwise(t->value, t->init_value.get_type()));
				}

			} break;
			case Animation::TYPE_AUDIO: {
				TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);

				// Audio ending process.
				LocalVector<ObjectID> erase_maps;
				for (KeyValue<ObjectID, PlayingAudioTrackInfo> &L : t->playing_streams) {
					PlayingAudioTrackInfo &track_info = L.value;
					float db = Math::linear_to_db(track_info.use_blend ? track_info.volume : 1.0);
					LocalVector<int> erase_streams;
					HashMap<int, PlayingAudioStreamInfo> &map = track_info.stream_info;
					for (const KeyValue<int, PlayingAudioStreamInfo> &M : map) {
						PlayingAudioStreamInfo pasi = M.value;

						bool stop = false;
						if (!t->audio_stream_playback->is_stream_playing(pasi.index)) {
							stop = true;
						}
						if (!track_info.loop) {
							if (!track_info.backward) {
								if (track_info.time < pasi.start) {
									stop = true;
								}
							} else if (track_info.backward) {
								if (track_info.time > pasi.start) {
									stop = true;
								}
							}
						}
						if (pasi.len > 0) {
							double len = 0.0;
							if (!track_info.backward) {
								len = pasi.start > track_info.time ? (track_info.length - pasi.start) + track_info.time : track_info.time - pasi.start;
							} else {
								len = pasi.start < track_info.time ? (track_info.length - track_info.time) + pasi.start : pasi.start - track_info.time;
							}
							if (len > pasi.len) {
								stop = true;
							}
						}
						if (stop) {
							// Time to stop.
							t->audio_stream_playback->stop_stream(pasi.index);
							erase_streams.push_back(M.key);
						} else {
							t->audio_stream_playback->set_stream_volume(pasi.index, db);
						}
					}
					for (uint32_t erase_idx = 0; erase_idx < erase_streams.size(); erase_idx++) {
						map.erase(erase_streams[erase_idx]);
					}
					if (map.size() == 0) {
						erase_maps.push_back(L.key);
					}
				}
				for (uint32_t erase_idx = 0; erase_idx < erase_maps.size(); erase_idx++) {
					t->playing_streams.erase(erase_maps[erase_idx]);
				}
			} break;
			default: {
			} // The rest don't matter.
		}
	}
}

bool CharacterAnimatorLayer::play_animation(Ref<CharacterAnimatorNodeBase> p_node)
{
    if(p_node.is_null())
    {
        return false;
    }
    if(m_AnimationInstances.size() > 0 && m_AnimationInstances.back()->get().node == p_node)
    {
        return false;
    }
    p_node->_init();
    for(auto& anim : m_AnimationInstances)
    {
        anim.m_PlayState = CharacterAnimationInstance::PS_FadeOut;
    }
    CharacterAnimationInstance ins;
    ins.node = p_node;
    ins.m_PlayState = CharacterAnimationInstance::PS_Play;
    if(config.is_valid() && config->get_mask().is_valid())
    {
        ins.disable_path = config->get_mask()->disable_path;
    }
    m_AnimationInstances.push_back(ins);
    return true;
}


void CharacterAnimatorLayer::play_animation(const StringName& p_node_name)
{
    if(m_Animator == nullptr)
    {
        return;
    }
    logic_context.curr_animation = m_Animator->get_animation_by_name(p_node_name);
}

CharacterAnimatorLayer::~CharacterAnimatorLayer()
{
    if(m_Animator != nullptr)
        m_Animator->on_layer_delete(this);
}

void CharacterAnimator::set_body(class CharacterBodyMain* p_body)
{
     m_Body = p_body; 
     create_layers();
}

void CharacterAnimator::add_layer(const Ref<CharacterAnimatorLayerConfig>& _mask)
{
    if(_mask.is_null())
    {
        return ;
    }
    animation_layer_arrays.push_back(_mask);
    if(m_Body == nullptr )
    {
        return;
    }

    CharacterAnimatorLayer* layer = memnew(CharacterAnimatorLayer);
    layer->config = _mask;
    layer->set_name(_mask->get_layer_name());
    layer->m_Animator = this;

    m_Body->add_child(layer);
    layer->set_owner(m_Body);
    m_LayerList.push_back(layer);
}
void CharacterAnimator::create_layers()
{
    clear_layer();
     if(m_Body)
     {
        for(int i = 0; i < animation_layer_arrays.size(); ++i)
        {
            Ref<CharacterAnimatorLayerConfig> _mask = animation_layer_arrays[i];
            CharacterAnimatorLayer* layer = memnew(CharacterAnimatorLayer);
            layer->config = _mask;
            layer->set_name(_mask->get_layer_name());
            layer->m_Animator = this;

            m_Body->add_child(layer);
            layer->set_owner(m_Body);
            m_LayerList.push_back(layer);

        }
     }

}
void CharacterAnimator::clear_layer()
{
    auto it = m_LayerList.begin();
    while(it != m_LayerList.end())
    {
        CharacterAnimatorLayer* layer = *it;
        layer->queue_free();

    }
    m_LayerList.clear();
}
void CharacterAnimator::update_animation(float delta)
{
    auto it = m_LayerList.begin();
    bool is_first = true;
    while(it!=m_LayerList.end())
    {
        CharacterAnimatorLayer* layer = *it;
        if(layer->is_active())
        {
            layer->_process_animation(m_Body->get_blackboard(),delta,is_first);
            is_first = false;
        }
        ++it;
    }
}
Ref<CharacterAnimationLibrary::AnimationItem> CharacterAnimator::get_animation_by_name(const StringName& p_name)
{
    if(m_Body == nullptr)
    {
        return Ref<CharacterAnimationLibrary::AnimationItem>();
    }
    auto anim_lib = m_Body->get_animation_library();
    if(anim_lib.is_valid())
    {
        return anim_lib->get_animation_by_name(p_name);
    }

    return Ref<CharacterAnimationLibrary::AnimationItem>();
}


void CharacterAnimator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_animation_layer_arrays", "animation_layer_arrays"), &CharacterAnimator::set_animation_layer_arrays);
    ClassDB::bind_method(D_METHOD("get_animation_layer_arrays"), &CharacterAnimator::get_animation_layer_arrays);

    ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "animation_layer_arrays"), "set_animation_layer_arrays", "get_animation_layer_arrays");
}


//////////////////////////////////////////////// CharacterAnimationLogicNode /////////////////////////////////////////
void CharacterAnimationLogicNode::process_start(CharacterAnimatorLayer* animator,Blackboard* blackboard)
{
    // 播放动作
    animator->play_animation(player_animation_name);
    if (GDVIRTUAL_IS_OVERRIDDEN(_animation_process_start)) {
        GDVIRTUAL_CALL(_animation_process_start, animator,blackboard);
        return ;
    }
}

void CharacterAnimationLogicNode::process(CharacterAnimatorLayer* animator,Blackboard* blackboard, double delta)
{
    if (GDVIRTUAL_IS_OVERRIDDEN(_animation_process)) {
        GDVIRTUAL_CALL(_animation_process, animator,blackboard, delta);
        return ;
    }

}

void CharacterAnimationLogicNode::process_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard)
{
    if (GDVIRTUAL_IS_OVERRIDDEN(_animation_process_stop)) {
        GDVIRTUAL_CALL(_animation_process_stop, animator,blackboard);
        return ;
    }
}
bool CharacterAnimationLogicNode::check_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard)
{
    auto context = animator->_get_logic_context();
    if(context->time < check_stop_delay_time)
    {
        return false;
    }
    if (GDVIRTUAL_IS_OVERRIDDEN(_check_stop)) {
        bool is_stop = false;
        GDVIRTUAL_CALL(_check_stop, animator,blackboard, is_stop);
        return is_stop;
    }
    if(stop_check_type == Life)
    {
        return (life_time >= context->time);
    }
    else if(stop_check_type == AnimationLengthScale)
    {
        return (context->curr_animation_play_time / context->curr_animation_time_length >= anmation_scale );
    }
    else 
    {
        if(stop_check_condtion.is_valid())
        {
            return stop_check_condtion->is_enable(blackboard);
        }
    }
    return true;
}
    
void CharacterAnimationLogicNode::init_blackboard(Ref<BlackboardPlan> p_blackboard_plan)
{
    Ref<BlackboardPlan> blackboard_plan = p_blackboard_plan;
    if(blackboard_plan.is_null())
    {
        return ;
    }
    if(!blackboard_plan->has_var("OldForward"))
        blackboard_plan->add_var("OldForward",BBVariable(Variant::VECTOR3,Vector3()));

    if(!blackboard_plan->has_var("CurrForward"))
        blackboard_plan->add_var("CurrForward",BBVariable(Variant::VECTOR3,Vector3()));

    if(!blackboard_plan->has_var("MoveTarget"))
        blackboard_plan->add_var("MoveTarget",BBVariable(Variant::VECTOR3,Vector3()));

    if(!blackboard_plan->has_var("CurrState"))
        blackboard_plan->add_var("CurrState",BBVariable(Variant::STRING_NAME,StringName()));

    if(!blackboard_plan->has_var("HorizontalMovement"))
        blackboard_plan->add_var("HorizontalMovement",BBVariable(Variant::FLOAT,0.0f));

    if(!blackboard_plan->has_var("VerticalMovement"))
        blackboard_plan->add_var("VerticalMovement",BBVariable(Variant::FLOAT,0.0f));
    
    if(!blackboard_plan->has_var("Pitch"))
        blackboard_plan->add_var("Pitch",BBVariable(Variant::FLOAT,0.0f));
    if(!blackboard_plan->has_var("Yaw"))
        blackboard_plan->add_var("Yaw",BBVariable(Variant::FLOAT,0.0f));
    if(!blackboard_plan->has_var("Speed"))
        blackboard_plan->add_var("Speed",BBVariable(Variant::FLOAT,0.0f));
    
    // 是否使用能力
    if(!blackboard_plan->has_var("IsAbility"))
        blackboard_plan->add_var("IsAbility",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("AbilityIndex"))
        blackboard_plan->add_var("AbilityIndex",BBVariable(Variant::INT,0));
    if(!blackboard_plan->has_var("AbilityIntData"))
        blackboard_plan->add_var("AbilityIntData",BBVariable(Variant::INT,0));
    if(!blackboard_plan->has_var("AbilityFloatData"))
        blackboard_plan->add_var("AbilityFloatData",BBVariable(Variant::FLOAT,0.0f));
    
    if(!blackboard_plan->has_var("IsGround"))
        blackboard_plan->add_var("IsGround",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsMoving"))
        blackboard_plan->add_var("IsMoving",BBVariable(Variant::BOOL,false));
    
    if(!blackboard_plan->has_var("IsJump"))
        blackboard_plan->add_var("IsJump",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsCrouch"))
        blackboard_plan->add_var("IsCrouch",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsAttack"))
        blackboard_plan->add_var("IsAttack",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsDead"))
        blackboard_plan->add_var("IsDead",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("LegIndex"))
        blackboard_plan->add_var("LegIndex",BBVariable(Variant::INT,0));
    
    // AI 大腦更新頻率
    if(!blackboard_plan->has_var("AI_BrainUpdate_Rate"))
        blackboard_plan->add_var("AI_BrainUpdate_Rate",BBVariable(Variant::FLOAT,1.0f));
    
}
void CharacterAnimationLogicNode::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_blackboard_plan", "blackboard_plan"), &CharacterAnimationLogicNode::set_blackboard_plan);
    ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &CharacterAnimationLogicNode::get_blackboard_plan);

    ClassDB::bind_method(D_METHOD("set_priority", "priority"), &CharacterAnimationLogicNode::set_priority);
    ClassDB::bind_method(D_METHOD("get_priority"), &CharacterAnimationLogicNode::get_priority);

    ClassDB::bind_method(D_METHOD("set_player_animation_name", "player_animation_name"), &CharacterAnimationLogicNode::set_player_animation_name);
    ClassDB::bind_method(D_METHOD("get_player_animation_name"), &CharacterAnimationLogicNode::get_player_animation_name);

    ClassDB::bind_method(D_METHOD("set_enter_condtion", "enter_condtion"), &CharacterAnimationLogicNode::set_enter_condtion);
    ClassDB::bind_method(D_METHOD("get_enter_condtion"), &CharacterAnimationLogicNode::get_enter_condtion);

    ClassDB::bind_method(D_METHOD("set_check_stop_delay_time", "check_stop_delay_time"), &CharacterAnimationLogicNode::set_check_stop_delay_time);
    ClassDB::bind_method(D_METHOD("get_check_stop_delay_time"), &CharacterAnimationLogicNode::get_check_stop_delay_time);

    ClassDB::bind_method(D_METHOD("set_life_time", "life_time"), &CharacterAnimationLogicNode::set_life_time);
    ClassDB::bind_method(D_METHOD("get_life_time"), &CharacterAnimationLogicNode::get_life_time);

    ClassDB::bind_method(D_METHOD("set_stop_check_type", "stop_check_type"), &CharacterAnimationLogicNode::set_stop_check_type);
    ClassDB::bind_method(D_METHOD("get_stop_check_type"), &CharacterAnimationLogicNode::get_stop_check_type);

    ClassDB::bind_method(D_METHOD("set_stop_check_condtion", "stop_check_condtion"), &CharacterAnimationLogicNode::set_stop_check_condtion);
    ClassDB::bind_method(D_METHOD("get_stop_check_condtion"), &CharacterAnimationLogicNode::get_stop_check_condtion);

    ClassDB::bind_method(D_METHOD("set_stop_check_anmation_length_scale", "stop_check_anmation_length_scale"), &CharacterAnimationLogicNode::set_stop_check_anmation_length_scale);
    ClassDB::bind_method(D_METHOD("get_stop_check_anmation_length_scale"), &CharacterAnimationLogicNode::get_stop_check_anmation_length_scale);


    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan",PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan"), "set_blackboard_plan", "get_blackboard_plan");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "priority"), "set_priority", "get_priority");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "player_animation_name"), "set_player_animation_name", "get_player_animation_name");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "enter_condtion", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorCondition"), "set_enter_condtion", "get_enter_condtion");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "check_stop_delay_time"), "set_check_stop_delay_time", "get_check_stop_delay_time");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "life_time"), "set_life_time", "get_life_time");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "stop_check_type",PROPERTY_HINT_ENUM,"Life,PlayCount,Condition,Script"), "set_stop_check_type", "get_stop_check_type");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stop_check_condtion", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorCondition"), "set_stop_check_condtion", "get_stop_check_condtion");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stop_check_anmation_length_scale"), "set_stop_check_anmation_length_scale", "get_stop_check_anmation_length_scale");


    GDVIRTUAL_BIND(_animation_process_start,"_layer","_blackboard");
    GDVIRTUAL_BIND(_animation_process_stop,"_layer","_blackboard");
    GDVIRTUAL_BIND(_animation_process,"_layer","_blackboard", "_delta");
    GDVIRTUAL_BIND(_check_stop,"_layer","_blackboard");

    BIND_ENUM_CONSTANT(Life);
    BIND_ENUM_CONSTANT(AnimationLengthScale);
    BIND_ENUM_CONSTANT(Condition);
    BIND_ENUM_CONSTANT(Script);

}

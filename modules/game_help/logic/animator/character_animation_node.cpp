
#include "core/io/json.h"

#include "body_animator.h"
#include "../data_table_manager.h"
#include "../../unity/unity_animation_import.h"


void UnityAnimation::load_form_unity_asset()
{
	Error err;
	Ref<FileAccess> f = FileAccess::open(unity_asset_path, FileAccess::READ, &err);
	if(f.is_null())
	{
		return ;
	}
	clear();
	String yaml_anim = f->get_as_text();

	Ref<JSON> json = DataTableManager::get_singleton()->parse_yaml(yaml_anim);

	Dictionary dict = json->get_data();
	Callable on_load_animation =  DataTableManager::get_singleton()->get_animation_load_cb();
	Dictionary clip = dict["AnimationClip"];
	on_load_animation.call(clip,false,this);
    Ref<UnityAnimation> anim = this;
	UnityAnimationImport::ImportAnimation(clip,false,anim);
	optimize();
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

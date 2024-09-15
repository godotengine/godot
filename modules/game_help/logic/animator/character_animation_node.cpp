
#include "core/io/json.h"

#include "body_animator.h"
#include "../data_table_manager.h"
#include "../../unity/unity_animation_import.h"


void CharacterAnimationItem::_bind_methods()
{

    ClassDB::bind_method(D_METHOD("set_speed", "speed"), &CharacterAnimationItem::set_speed);
    ClassDB::bind_method(D_METHOD("get_speed"), &CharacterAnimationItem::get_speed);

    ClassDB::bind_method(D_METHOD("set_is_clip", "is_clip"), &CharacterAnimationItem::set_is_clip);
    ClassDB::bind_method(D_METHOD("get_is_clip"), &CharacterAnimationItem::get_is_clip);

    ClassDB::bind_method(D_METHOD("set_child_node", "child_node"), &CharacterAnimationItem::set_child_node);
    ClassDB::bind_method(D_METHOD("get_child_node"), &CharacterAnimationItem::get_child_node);

    ClassDB::bind_method(D_METHOD("set_animation", "animation"), &CharacterAnimationItem::set_animation);
    ClassDB::bind_method(D_METHOD("get_animation"), &CharacterAnimationItem::get_animation);


    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed"), "set_speed", "get_speed");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_clip"), "set_is_clip", "get_is_clip");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "child_node"), "set_child_node", "get_child_node");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation"), "set_animation", "get_animation");

}
void CharacterAnimationItem::set_child_node(const Ref<CharacterAnimatorNodeBase>& p_child_node) 
{ 
    child_node = p_child_node; 
}
Ref<CharacterAnimatorNodeBase> CharacterAnimationItem::get_child_node() 
{
        return child_node; 
}

void CharacterAnimationItem::_init()
{
    if(is_init)
    {
        return;
    }
	if (is_clip)
	{
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

void CharacterAnimatorNodeBase::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("_get_blackbord_propertys"), &CharacterAnimatorNodeBase::_get_blackbord_propertys);

    
    ClassDB::bind_method(D_METHOD("set_animation_arrays", "animation_arrays"), &CharacterAnimatorNodeBase::set_animation_arrays);
    ClassDB::bind_method(D_METHOD("get_animation_arrays"), &CharacterAnimatorNodeBase::get_animation_arrays);


    ClassDB::bind_method(D_METHOD("set_fade_out_time", "fade_out_time"), &CharacterAnimatorNodeBase::set_fade_out_time);
    ClassDB::bind_method(D_METHOD("get_fade_out_time"), &CharacterAnimatorNodeBase::get_fade_out_time);

    ClassDB::bind_method(D_METHOD("set_loop", "loop"), &CharacterAnimatorNodeBase::set_loop);
    ClassDB::bind_method(D_METHOD("get_loop"), &CharacterAnimatorNodeBase::get_loop);

    ClassDB::bind_method(D_METHOD("set_loop_count", "loop_count"), &CharacterAnimatorNodeBase::set_loop_count);
    ClassDB::bind_method(D_METHOD("get_loop_count"), &CharacterAnimatorNodeBase::get_loop_count);





    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "animation_arrays", PROPERTY_HINT_ARRAY_TYPE, "CharacterAnimationItem"), "set_animation_arrays", "get_animation_arrays");
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
void CharacterAnimatorNodeBase::update_animation_time(struct CharacterAnimationInstance* p_playback_info) {
	p_playback_info->animation_time_pos += p_playback_info->delta;
	while (p_playback_info->animation_time_pos > _get_animation_length())
	{
		p_playback_info->animation_time_pos -= _get_animation_length();
		++p_playback_info->play_count;
	}

}

void CharacterAnimatorNodeBase::_blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,CharacterAnimationInstance *p_playback_info,float total_weight,const LocalVector<float> &weight_array,const Ref<Blackboard> &p_blackboard)
{
    touch();
    AnimationMixer::PlaybackInfo * p_playback_info_ptr = p_playback_info->m_ChildAnimationPlaybackArray.ptr();
    for (int32_t i = 0; i < child_count; i++)
    {
        float w = weight_array[i] * total_weight;
        if(w > 0.01f)
        {	  
            Ref<CharacterAnimationItem> item = animation_arrays[i];
            if(item->is_clip){
                p_playback_info_ptr[i].weight = w;
                p_playback_info_ptr[i].delta = p_playback_info->delta / ABS(item->get_speed());
                p_playback_info_ptr[i].time = p_playback_info->animation_time_pos / ABS(item->get_speed());
                p_playback_info_ptr[i].disable_path = p_playback_info->disable_path;

				if (get_loop() == LOOP_Once)
				{
					double length = item->animation->get_length();
					double time = p_playback_info_ptr[i].time;
					while (p_playback_info_ptr[i].time > length)
					{
						p_playback_info_ptr[i].time -= length;
					}
				}
				if (p_playback_info_ptr[i].invert)
				{
					p_playback_info_ptr[i].delta = -p_playback_info_ptr[i].delta;
					p_playback_info_ptr[i].time = -p_playback_info_ptr[i].time;
				}
			
                Ref<Animation> animation = item->get_animation();
                Ref<CharacterBoneMap> bone_map = item->animation->get_bone_map();
                Dictionary bp;
                if(bone_map.is_valid())
                {
                    bp = bone_map->bone_map;
                }
                p_layer->play_animationm(item->animation, p_playback_info_ptr[i],bp);
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
    uint32_t count = blendConstant.position_array.size();

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
    int pairIndex = i + j * blendConstant.position_array.size();
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
void CharacterAnimatorNodeBase::Blend2dDataConstant::precompute_freeform(BlendType type)
{
	Vector2* positionArray = position_array.ptr();
	uint32_t count = position_array.size();
	float* constantMagnitudes = m_ChildMagnitudeArray.ptr();
	Vector2* constantChildPairVectors = m_ChildPairVectorArray.ptr();
	float* constantChildPairAvgMagInv = m_ChildPairAvgMagInvArray.ptr();
	MotionNeighborList* constantChildNeighborLists = m_ChildNeighborListArray.ptr();

	if (type == FreeformDirectionnal2D)
	{
		for (uint32_t i = 0; i < count; i++)
			constantMagnitudes[i] = positionArray[i].length();

		for (uint32_t i = 0; i < count; i++)
		{
			for (uint32_t j = 0; j < count; j++)
			{
				int pairIndex = i + j * count;

				// Calc avg magnitude for pair
				float magSum = constantMagnitudes[j] + constantMagnitudes[i];
				if (magSum > 0)
					constantChildPairAvgMagInv[pairIndex] = 2.0f / magSum;
				else
					constantChildPairAvgMagInv[pairIndex] = 2.0f / magSum;

				// Calc mag of vector and divide by avg magnitude
				float mag = (constantMagnitudes[j] - constantMagnitudes[i]) * constantChildPairAvgMagInv[pairIndex];

				if (constantMagnitudes[j] == 0 || constantMagnitudes[i] == 0)
					constantChildPairVectors[pairIndex] = Vector2(0, mag);
				else
				{
					float angle = positionArray[i].angle_to(positionArray[j]);
					if (positionArray[i].x * positionArray[j].y - positionArray[i].y * positionArray[j].x < 0)
						angle = -angle;
					constantChildPairVectors[pairIndex] = Vector2(angle, mag);
				}
			}
		}
	}
	else if (type == FreeformCartesian2D)
	{
		for (uint32_t i = 0; i < count; i++)
		{
			for (uint32_t j = 0; j < count; j++)
			{
				int pairIndex = i + j * count;
				constantChildPairAvgMagInv[pairIndex] = 1 / (positionArray[j] - positionArray[i]).length_squared();
				constantChildPairVectors[pairIndex] = positionArray[j] - positionArray[i];
			}
		}
	}

	float* weightArray = (float*)alloca(sizeof(float) * count);

	int* cropArray = (int*)alloca(sizeof(int) * count);

	Vector2* workspaceBlendVectors = (Vector2*)alloca(sizeof(Vector2) * count);

	bool* neighborArray; (bool*)alloca(sizeof(Vector2) * count * count);
	for (uint32_t c = 0; c < count * count; c++)
		neighborArray[c] = false;

	float minX = 10000.0f;
	float maxX = -10000.0f;
	float minY = 10000.0f;
	float maxY = -10000.0f;
	for (uint32_t c = 0; c < count; c++)
	{
		minX = std::min(minX, positionArray[c].x);
		maxX = std::max(maxX, positionArray[c].x);
		minY = std::min(minY, positionArray[c].y);
		maxY = std::max(maxY, positionArray[c].y);
	}
	float xRange = (maxX - minX) * 0.5f;
	float yRange = (maxY - minY) * 0.5f;
	minX -= xRange;
	maxX += xRange;
	minY -= yRange;
	maxY += yRange;

	for (uint32_t i = 0; i <= 100; i++)
	{
		for (uint32_t j = 0; j <= 100; j++)
		{
			float x = i * 0.01f;
			float y = j * 0.01f;
			if (type == FreeformDirectionnal2D)
				get_weights_freeform_directional(*this, weightArray, cropArray, workspaceBlendVectors, minX * (1 - x) + maxX * x, minY * (1 - y) + maxY * y, true);
			else if (type == FreeformCartesian2D)
				get_weights_freeform_cartesian(*this, weightArray, cropArray, workspaceBlendVectors, minX * (1 - x) + maxX * x, minY * (1 - y) + maxY * y, true);
			for (uint32_t c = 0; c < count; c++)
				if (cropArray[c] >= 0)
					neighborArray[c * count + cropArray[c]] = true;
		}
	}
	for (uint32_t c = 0; c < count; c++)
	{
		LocalVector<int> nList;
		for (uint32_t d = 0; d < count; d++)
			if (neighborArray[c * count + d])
				nList.push_back(d);

		constantChildNeighborLists[c].m_Count = nList.size();
		constantChildNeighborLists[c].m_NeighborArray.resize(nList.size());

		for (uint32_t d = 0; d < nList.size(); d++)
			constantChildNeighborLists[c].m_NeighborArray[d] = nList[d];
	}
}
void CharacterAnimatorNodeBase::get_weights_freeform_directional(const Blend2dDataConstant& blendConstant,
    float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
    float blendValueX, float blendValueY, bool preCompute )
{
    // Get constants
    const Vector2* positionArray = blendConstant.position_array.ptr();
    uint32_t count = blendConstant.position_array.size();
    if (count < 2)
    {
        if (count == 1)
            weightArray[0] = 1;
        return;
    }
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
    uint32_t count = blendConstant.position_array.size();
    if (count < 2)
    {
        if (count == 1)
            weightArray[0] = 1;
        return;
    }
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

                int pairIndex = i + j * blendConstant.position_array.size();
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

            int pairIndex = i + j * blendConstant.position_array.size();
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
    if (blendConstant.position_array.size() < 2)
    {
        if (blendConstant.position_array.size() == 1)
            weightArray[0] = 1;
        return;
    }
    blendValue = CLAMP(blendValue, blendConstant.position_array[0], blendConstant.position_array[blendConstant.position_array.size() - 1]);
    for (uint32_t j = 0; j < blendConstant.position_array.size(); j++)
        weightArray[j] = weight_for_index(blendConstant.position_array.ptr(), blendConstant.position_array.size(), j, blendValue);
}

void CharacterAnimatorNode1D::add_animation(const Ref<Animation> & p_anim,float p_pos)
{
    Ref<CharacterAnimationItem> item;
    item.instantiate();
    item->set_animation(p_anim);
	int count = blend_data.position_array.size() + 1;
    animation_arrays.resize(count);

	blend_data.position_array.resize(count);

    blend_data.position_array[count - 1] = p_pos;
    animation_arrays[count - 1] = item;

}
void CharacterAnimatorNode1D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
{
    float v = p_blackboard->get_var(black_board_property,0);
    if(p_playback_info->m_WeightArray.size() != blend_data.position_array.size())
    {
        p_playback_info->m_WeightArray.resize(blend_data.position_array.size());
        p_playback_info->m_ChildAnimationPlaybackArray.resize(blend_data.position_array.size());
    }
    get_weights1d(blend_data, p_playback_info->m_WeightArray.ptr(), v);
    _blend_anmation(p_layer,blend_data.position_array.size(), p_playback_info, total_weight,p_playback_info->m_WeightArray,p_blackboard);

}
void CharacterAnimatorLoopLast::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
{
        float w = total_weight;
        if(w > 0.01f)
        {	  
            Ref<CharacterAnimationItem> item = animation_arrays[p_playback_info->play_index];


			AnimationMixer::PlaybackInfo* p_playback_info_ptr = p_playback_info->m_ChildAnimationPlaybackArray.ptr();
            AnimationMixer::PlaybackInfo&  playback_info = p_playback_info_ptr[0];

			playback_info.delta = ABS(playback_info.delta);
			playback_info.time = ABS(playback_info.time);
			playback_info.weight = w;
			playback_info.delta = p_playback_info->delta * ABS(item->get_speed());
			playback_info.time += playback_info.delta ;
			playback_info.disable_path = p_playback_info->disable_path;
			// 循环播放
			double length = item->animation->get_length();
            if(p_playback_info->play_index < animation_arrays.size() - 1)
            {
                if(playback_info.time >= length)
                {
					playback_info.time -= length;
					p_playback_info->play_index = p_playback_info->play_index + 1;
                }
            }
			else
			{
				double time = playback_info.time;
				while (playback_info.time > length)
				{
					playback_info.time -= length;
				}
			}
			if (playback_info.invert)
			{
				playback_info.delta = -playback_info.delta;
				playback_info.time = -playback_info.time;
			}
            item = animation_arrays[p_playback_info->play_index];
                
            if(item->is_clip){
                Ref<Animation> animation = item->get_animation();
                Ref<CharacterBoneMap> bone_map = item->animation->get_bone_map();
                Dictionary bp;
                if(bone_map.is_valid())
                {
                    bp = bone_map->bone_map;
                }
                p_layer->make_animation_instance_anim(item->animation, playback_info,bp);
            }
            else if(item->child_node.is_valid())
            {
                // 动画节点递归处理
                item->child_node->process_animation(p_layer,p_playback_info,w,p_blackboard);
            }
        }

}
float CharacterAnimatorLoopLast::_get_animation_length()
{
    float length = 0.0f;
    for(int i = 0; i < animation_arrays.size(); ++i)
    {
        Ref<CharacterAnimationItem> item = animation_arrays[i];
        if(item.is_valid())
        {
            length += item->_get_animation_length();
        }
    }
    return length;
}

void CharacterAnimatorNode2D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
{
    Vector2 v = p_blackboard->get_var(black_board_property,0);
    if(p_playback_info->m_WeightArray.size() != blend_data.position_array.size())
    {
        p_playback_info->m_WeightArray.resize(blend_data.position_array.size());
        p_playback_info->m_ChildAnimationPlaybackArray.resize(blend_data.position_array.size());
    }
    if(p_layer->m_TempCropArray.size() < blend_data.position_array.size())
    {
        p_layer->m_TempCropArray.resize(blend_data.position_array.size());
        p_layer->m_ChildInputVectorArray.resize(blend_data.position_array.size());
    }
	blend_data.reset();
	blend_data.precompute_freeform(blend_type);

    if (blend_type == SimpleDirectionnal2D)
        get_weights_simple_directional(blend_data, (float*)p_playback_info->m_WeightArray.ptr(), (int*)p_layer->m_TempCropArray.ptr(), (Vector2*)p_layer->m_ChildInputVectorArray.ptr(), v.x, v.y);
    else if (blend_type == FreeformDirectionnal2D)
        get_weights_freeform_directional(blend_data, (float*)p_playback_info->m_WeightArray.ptr(), (int*)p_layer->m_TempCropArray.ptr(), (Vector2*)p_layer->m_ChildInputVectorArray.ptr(), v.x, v.y);
    else if (blend_type == FreeformCartesian2D)
        get_weights_freeform_cartesian(blend_data, (float*)p_playback_info->m_WeightArray.ptr(), (int*)p_layer->m_TempCropArray.ptr(), (Vector2*)p_layer->m_ChildInputVectorArray.ptr(), v.x, v.y);
    else 
        return;

    _blend_anmation(p_layer, blend_data.position_array.size(),p_playback_info, total_weight,p_playback_info->m_WeightArray,p_blackboard);

}

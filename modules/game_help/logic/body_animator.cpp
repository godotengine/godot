#include "body_animator.h"
#include "body_main.h"

void CharacterAnimatorNodeBase::_blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,CharacterAnimationInstance *p_playback_info,float total_weight,const Vector<float> &weight_array,Blackboard *p_blackboard)
{
    AnimationMixer::PlaybackInfo * p_playback_info_ptr = p_playback_info->m_ChildAnimationPlaybackArray.ptrw();
    for (uint32_t i = 0; i < child_count; i++)
    {
        float w = weight_array[i] * total_weight;
        if(w > 0.01f)
        {	  
            if(m_ChildAnimationArray[i].isClip){
                p_playback_info_ptr[i].weight = w;
                p_playback_info_ptr[i].time = p_playback_info->time;
                p_playback_info_ptr[i].delta = p_playback_info->delta;
                p_playback_info_ptr[i].disable_path = p_playback_info->disable_path;
                p_layer->make_animation_instance(m_ChildAnimationArray[i].m_Name, p_playback_info_ptr[i]);
            }
            else if(m_ChildAnimationArray[i].m_animation_node.is_valid())
            {
                // 动画节点递归处理
                m_ChildAnimationArray[i].m_animation_node->process_animation(p_layer,p_playback_info,w,p_blackboard);
            }
        }
    }
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

float CharacterAnimatorNodeBase::get_weight_freeform_directional(const Blend2dDataConstant& blendConstant, Vector2* workspaceBlendVectors, int i, int j, Vector2 blendPosition)
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

void CharacterAnimatorNodeBase::get_weights_freeform_directional(const Blend2dDataConstant& blendConstant,
    float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
    float blendValueX, float blendValueY, bool preCompute )
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

void CharacterAnimatorNodeBase::get_weights1d(const Blend1dDataConstant& blendConstant, float* weightArray, float blendValue)
{
    blendValue = CLAMP(blendValue, blendConstant.m_ChildThresholdArray[0], blendConstant.m_ChildThresholdArray[blendConstant.m_ChildCount - 1]);
    for (uint32_t j = 0; j < blendConstant.m_ChildCount; j++)
        weightArray[j] = weight_for_index(blendConstant.m_ChildThresholdArray.ptr(), blendConstant.m_ChildCount, j, blendValue);
}

void CharacterAnimatorNode1D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard)
{
    if(!p_blackboard->has_var(m_PropertyName))
    {
        return;
    }
    float v = p_blackboard->get_var(m_PropertyName,0);
    if(p_playback_info->m_WeightArray.size() != m_BlendData.m_ChildCount)
    {
        p_playback_info->m_WeightArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(m_BlendData.m_ChildCount);
    }
    get_weights1d(m_BlendData, p_playback_info->m_WeightArray.ptrw(), v);
    _blend_anmation(p_layer,m_BlendData.m_ChildCount, p_playback_info, total_weight,p_playback_info->m_WeightArray,p_blackboard);

}
void CharacterAnimatorNode2D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard)
{
    if(!p_blackboard->has_var(m_PropertyName))
    {
        return;
    }
    Vector2 v = p_blackboard->get_var(m_PropertyName,0);
    if(p_playback_info->m_WeightArray.size() != m_BlendData.m_ChildCount)
    {
        p_playback_info->m_WeightArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(m_BlendData.m_ChildCount);
    }
    if(p_layer->m_TempCropArray.size() < m_BlendData.m_ChildCount)
    {
        p_layer->m_TempCropArray.resize(m_BlendData.m_ChildCount);
        p_layer->m_ChildInputVectorArray.resize(m_BlendData.m_ChildCount);
    }
    if (m_BlendType == SimpleDirectionnal2D)
        get_weights_simple_directional(m_BlendData, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else if (m_BlendType == FreeformDirectionnal2D)
        get_weights_freeform_directional(m_BlendData, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else if (m_BlendType == FreeformCartesian2D)
        get_weights_freeform_cartesian(m_BlendData, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else 
        return;

    _blend_anmation(p_layer, m_BlendData.m_ChildCount,p_playback_info, total_weight,p_playback_info->m_WeightArray,p_blackboard);

}
// 处理动画
void CharacterAnimatorLayer::_process_animation(Blackboard *p_playback_info,double p_delta,bool is_first)
{
    
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

void CharacterAnimatorLayer::play_animation(Ref<CharacterAnimatorNodeBase> p_node)
{
    
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
}


CharacterAnimatorLayer::~CharacterAnimatorLayer()
{
    if(m_Animator != nullptr)
        m_Animator->on_layer_delete(this);
}


void CharacterAnimator::add_layer(const StringName& name,const Ref<CharacterAnimatorLayerConfig>& _mask)
{
    if(m_Body == nullptr || _mask.is_null())
    {
        return ;
    }

    CharacterAnimatorLayer* layer = memnew(CharacterAnimatorLayer);
    layer->config = _mask;
    layer->set_name(name.str());
    layer->m_Animator = this;

    m_Body->add_child(layer);
    layer->set_owner(m_Body);
    m_LayerList.push_back(layer);
}
void CharacterAnimator::clear_layer()
{
    while(m_LayerList.size()>0)
    {
        CharacterAnimatorLayer* layer = *m_LayerList.begin();
        memdelete(layer);
    }
}



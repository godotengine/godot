#pragma once

#include "scene/resources/animation.h"
#include "scene/3d/skeleton_3d.h"


namespace HumanAnim
{
    
 // 骨骼配置
    struct HumanSkeleton {
        
        AHashMap<StringName, Quaternion> real_local_pose; 
        AHashMap<StringName, Transform3D> real_pose; 

        AHashMap<StringName,Node3D*> bones;

        AHashMap<StringName, Vector3> bone_lookat;
		HashMap<StringName, Vector3> root_position;



		HashMap<StringName, Basis> root_global_rotation;
        HashMap<StringName, Vector3> root_global_move_add;
        HashMap<StringName, Basis> root_global_rotation_add;

        void rest(HumanConfig& p_config) {
			clear();
            for(auto& it : p_config.virtual_pose) {
                real_local_pose[it.key] = it.value.rotation;
				Transform3D& trans = real_pose[it.key];
				trans.basis = Basis(it.value.rotation);
				trans.origin = it.value.position;
            } 

            for(auto& it : p_config.root_bone) {
                root_position[it] = Vector3();
                root_global_rotation[it] = Quaternion();
                root_global_move_add[it] = Vector3();
                root_global_rotation_add[it] = Quaternion();
            }

        }

        void clear() { 
            real_local_pose.clear();
            real_pose.clear();
            bone_lookat.clear(); 
            for(auto& it : bones) {
                memdelete(it.value);
            }
            bones.clear();
        }

        void set_human_lookat(StringName p_bone,const Vector3& p_lookat) {
            String name = p_bone.substr(3);
            bone_lookat[name] = p_lookat;
        }
        static Basis retarget_root_direction(const Vector3& p_start_direction,const Vector3& p_end_direction) {
            Basis basis;
            const Vector3 axis = p_start_direction.cross(p_end_direction).normalized();
            if (axis.length_squared() != 0) {
                real_t dot = p_start_direction.dot(p_end_direction);
                dot = CLAMP(dot, -1.0f, 1.0f);
                const real_t angle_rads = Math::acos(dot);
                basis = Basis(axis, angle_rads);
            }
            return basis;
            
        }

        static Basis compute_lookat_rotation_add(Ref<Animation> p_animation,int track_index , double time_start, double time_end) {
				Vector3 loc,loc2;
				Error err = p_animation->try_position_track_interpolate(track_index, time_start, &loc);
				err = p_animation->try_position_track_interpolate(track_index, time_start, &loc2);
				return retarget_root_direction(loc,loc2);
            
        }

        static Vector3 compute_lookat_position_add(Ref<Animation> p_animation,int track_index , double time_start, double time_end) {
				Vector3 loc,loc2;
				Error err = p_animation->try_position_track_interpolate(track_index, time_start, &loc);
				err = p_animation->try_position_track_interpolate(track_index, time_start, &loc2);
				return loc2 - loc;
            
        }

        void set_root_lookat(Ref<Animation> p_animation,StringName p_bone, int track_index ,double time,double delta) {
            if(delta == 0) return;

            Basis q;

            double last_time = time - delta;

            if(delta >= 0) {
                if(last_time < 0) {
                    q = compute_lookat_rotation_add(p_animation,track_index, 0, time) * compute_lookat_rotation_add(p_animation,track_index, p_animation->get_length() + last_time, p_animation->get_length());
                }
                else {
                    q = compute_lookat_rotation_add(p_animation,track_index, last_time, time);
                }
            } else {
                if(last_time > p_animation->get_length()) {
                    q = compute_lookat_rotation_add(p_animation,track_index, time, p_animation->get_length()) * compute_lookat_rotation_add(p_animation,track_index, last_time - p_animation->get_length(), 0);
                }
                else {
                    q = compute_lookat_rotation_add(p_animation,track_index, last_time , time);
                }

            }
            String name = p_bone.substr(5);
			root_global_rotation_add[name] = q;
            Vector3 loc;
            p_animation->try_position_track_interpolate(track_index, time, &loc);
			bone_lookat[name] = loc;
        }

        void set_root_position_add(Ref<Animation> p_animation, StringName p_bone, int track_index ,double time,double delta) {
            if(delta == 0) return;

            Vector3 q;

            double last_time = time - delta;

            if(delta >= 0) {
                if(last_time < 0) {
                    q = compute_lookat_position_add(p_animation,track_index, 0, time) + compute_lookat_position_add(p_animation,track_index, p_animation->get_length() + last_time, p_animation->get_length());
                }
                else {
                    q = compute_lookat_position_add(p_animation,track_index, last_time, time);
                }
            } else {
                if(last_time > p_animation->get_length()) {
                    q = compute_lookat_position_add(p_animation,track_index, time, p_animation->get_length()) + compute_lookat_position_add(p_animation,track_index, last_time - p_animation->get_length(), 0);
                }
                else {
                    q = compute_lookat_position_add(p_animation,track_index, last_time , time);
                }

            }
            String name = p_bone.substr(5);
			root_global_move_add[name] = q;
            Vector3 loc;
            p_animation->try_position_track_interpolate(track_index, time, &loc);
			root_position[name] = loc;
            
        }

        void blend(HumanSkeleton& p_other,float p_weight) {
            for(auto& it : p_other.real_local_pose) {
                if(real_local_pose.has(it.key)) {
                    Quaternion& q = real_local_pose[it.key];
                    q = q.slerp(it.value, p_weight);
                }
            }
            for(auto& it : p_other.root_position) {
                if(root_global_move_add.has(it.key)) {
                    Vector3& v = root_position[it.key];
                    v = v.lerp(it.value, p_weight);
                }
            }
            for(auto& it : p_other.root_global_rotation) {
                if(root_global_move_add.has(it.key)) {
					Basis& v = root_global_rotation[it.key];
                    v = v.lerp(it.value, p_weight);
                }
            }

            for(auto& it : p_other.root_global_move_add) {
                if(root_global_move_add.has(it.key)) {
                    Vector3& v = root_global_move_add[it.key];
                    v = v.lerp(it.value, p_weight);
                }
            }
            for(auto& it : p_other.root_global_rotation_add) {
                if(root_global_rotation_add.has(it.key)) {
                    Basis& q = root_global_rotation_add[it.key];
                    q = q.slerp(it.value, p_weight);
                }
            }
            
        }

		void apply_root_motion(Node3D* node) {
			
			Transform3D curr_trans = node->get_transform();

			Transform3D add_trans;
			if (root_global_rotation_add.size() > 0) {
				add_trans.basis = root_global_rotation_add.begin()->value;
			}

			if (root_global_move_add.size() > 0) {
				add_trans.origin = add_trans.basis.xform(root_global_move_add.begin()->value);
			}

			node->set_transform(add_trans * curr_trans);
		}

        void apply(Skeleton3D *p_skeleton,float p_weight) {
            for(auto& it : real_local_pose) {
                int bone_index = p_skeleton->find_bone(it.key);
				if (bone_index >= 0) {
					p_skeleton->set_bone_pose_rotation(bone_index, p_skeleton->get_bone_pose_rotation(bone_index).slerp( it.value,p_weight));
				}
            }
        }

        void apply_root_motion(Vector3& p_position,Quaternion& p_rotation,Vector3& p_position_add,Quaternion & p_rotation_add,float p_weight) {

            if(root_global_rotation.size() > 0) {
                p_rotation = p_rotation.slerp(root_global_rotation.begin()->value.get_rotation_quaternion(),p_weight);
            }
            
			if (root_global_rotation_add.size() > 0) {
				p_rotation_add = p_rotation.slerp(root_global_rotation_add.begin()->value.get_rotation_quaternion(),p_weight);
			}

            if(root_position.size() > 0) {
                p_position = p_position.lerp(root_position.begin()->value,p_weight);
            }

            if(root_global_move_add.size() > 0) {
                p_position_add = p_position_add.lerp(root_global_move_add.begin()->value,p_weight);
            }
        }

    };

    class HumanAnimmation {
     public:


        /** 
        提取动画姿势:
        把动画文件的骨骼旋转应用到虚拟骨骼,获取到世界位置,河道世界位置
        

        虚拟骨骼:
        长度为1的骨骼,姿势保持和原始骨骼一致
        
        
        应用动画姿势:
        用当前角色的虚拟骨骼从父节点到子节点进行世界空间的Lookat计算
        用虚拟骨骼的世界变换得到局部旋转
        应用虚拟骨骼的局部旋转到真实骨骼
        Hip骨骼的世界朝向应用到角色的朝向,Hip的世界位置应用到角色的位置
        */
        // 构建虚拟姿势
        static void build_virtual_pose(Skeleton3D *p_skeleton,HumanConfig& p_config,HashMap<String, String>& p_human_bone_label) {
            Vector<int> root_bones = p_skeleton->get_root_bones();
            p_skeleton->_update_bones_nested_set();
			p_skeleton->force_update_all_bone_transforms(false);

			auto rbs = p_skeleton->get_root_bones();


            for(int root_bone = 0; root_bone < rbs.size(); ++root_bone){

                StringName bone_name = p_skeleton->get_bone_name(root_bone);
                BonePose pose;
                Transform3D trans = p_skeleton->get_bone_global_pose(root_bone);
                float height = 1.0;
                Vector<int> children = p_skeleton->get_bone_children(root_bone);
                for(int j=0;j<children.size();j++) {
                    pose.child_bones.push_back(p_skeleton->get_bone_name(children[j]));
                }
				pose.child_bones.sort_custom<SortStringName>();


                pose.position = trans.origin;
                pose.rotation = trans.basis.get_rotation_quaternion();
                float inv_height = 1.0 / height;
                pose.scale = Vector3(inv_height,inv_height,inv_height);
                pose.length = height;
                pose.bone_index = root_bone;
                p_config.virtual_pose[bone_name] = pose;

                // 构建所有子骨骼的姿势
                for(int j=0;j<children.size();j++) {
                    build_virtual_pose(p_config,p_skeleton, trans, children[j], p_human_bone_label);
                }
				p_config.root_bone.push_back(bone_name);
            }

            // 根据骨骼的高度计算虚拟姿势
            for(int i=0;i<p_skeleton->get_bone_count();i++) {
				String bone_name = p_skeleton->get_bone_name(i);
				if (!p_config.virtual_pose.has(bone_name)) {
					continue;
				}
                int parent = p_skeleton->get_bone_parent(i);
				String parent_bone_name = p_skeleton->get_bone_name(parent);
				if (!p_config.virtual_pose.has(parent_bone_name)) {
					continue;
				}
                BonePose& child_pose = p_config.virtual_pose[bone_name];
                child_pose.position *= child_pose.scale;
                
            }

            
        }
        static int get_bone_human_index(Skeleton3D* p_skeleton, Dictionary& p_bone_map,const NodePath& path) {
            if (path.get_subname_count() == 1) {
                // 获取骨骼映射
                StringName bone_name = path.get_subname(0);
				if (p_bone_map.has(bone_name)) {
					bone_name = p_bone_map[bone_name];
				}
                return p_skeleton->find_bone(bone_name);
            }
            return -1;

        }

        static Ref<Animation> build_human_animation(Skeleton3D* p_skeleton,HumanConfig& p_config,Ref<Animation> p_animation,Dictionary & p_bone_map) {            
            int key_count = p_animation->get_length() * 100 + 1;
            Vector3 loc,scale;
            Quaternion rot;
            HumanSkeleton skeleton_config;
            Vector<AHashMap<StringName, Vector3>> animation_lookat;

            //  根节点的位置
            Vector<HashMap<StringName, Vector3>> animation_root_position;
            Vector<HashMap<StringName, Vector3>> animation_root_lookat;
            animation_lookat.resize(key_count);
            animation_root_position.resize(key_count);
		    Vector<Animation::Track*> tracks = p_animation->get_tracks();

            // 获取非人型骨骼的轨迹
            List<Animation::Track*> other_tracks;
            for (int j = 0; j < tracks.size(); j++) {
                Animation::Track* track = tracks[j];
                if (track->type == Animation::TYPE_POSITION_3D) {
                    Animation::PositionTrack* track_cache = static_cast<Animation::PositionTrack*>(track);
                    NodePath path = track_cache->path;
                    StringName bone_name;
                    if (path.get_subname_count() == 1) {
                        // 获取骨骼映射
                        bone_name = path.get_subname(0);
						if (p_bone_map.has(bone_name)) {
							bone_name = p_bone_map[bone_name];
						}
                    }
                    if (bone_name != "Root" && !p_config.virtual_pose.has(bone_name)) {
                        other_tracks.push_back(track);
                        continue;
                    }
                }
                else if (track->type == Animation::TYPE_ROTATION_3D) {
                    Animation::RotationTrack* track_cache = static_cast<Animation::RotationTrack*>(track);

                    NodePath path = track_cache->path;
                    StringName bone_name;
                    if (path.get_subname_count() == 1) {
                        // 获取骨骼映射
                        bone_name = path.get_subname(0);
						if (p_bone_map.has(bone_name)) {
							bone_name = p_bone_map[bone_name];
						}
                    }
					if (bone_name != "Root" && !p_config.virtual_pose.has(bone_name)) {
                        other_tracks.push_back(track);
                        continue;
                    }
                }
                else if (track->type == Animation::TYPE_SCALE_3D) {
                    Animation::ScaleTrack* track_cache = static_cast<Animation::ScaleTrack*>(track);

                    NodePath path = track_cache->path;
                    StringName bone_name;
                    if (path.get_subname_count() == 1) {
                        // 获取骨骼映射
                        bone_name = path.get_subname(0);
						if (p_bone_map.has(bone_name)) {
							bone_name = p_bone_map[bone_name];
						}
                    }
					if (bone_name != "Root" && !p_config.virtual_pose.has(bone_name)) {
                        other_tracks.push_back(track);
                        continue;
                    }
                }
                else
                {
                    other_tracks.push_back(track);
                }
            }



            for(int i = 0; i < key_count; i++) {
                double time = double(i) / 100.0;
                for(int j = 0; j < tracks.size(); j++) {
                    Animation::Track* track = tracks[j];
                    if(track->type == Animation::TYPE_POSITION_3D) {
                        Animation::PositionTrack* track_cache = static_cast<Animation::PositionTrack*>(track);
                        int bone_index = get_bone_human_index(p_skeleton, p_bone_map,track_cache->path);
                        if(bone_index < 0) {
                            continue;
                        }
                        Error err = p_animation->try_position_track_interpolate(j, time, &loc);
                        p_skeleton->set_bone_pose_position(bone_index, loc);
                    }
                    else if(track->type == Animation::TYPE_ROTATION_3D) {
                        Animation::RotationTrack* track_cache = static_cast<Animation::RotationTrack*>(track);
                        int bone_index = get_bone_human_index(p_skeleton, p_bone_map, track_cache->path);
                        if(bone_index < 0) {
                            continue;
                        }
                        Error err = p_animation->try_rotation_track_interpolate(j, time, &rot);
                        p_skeleton->set_bone_pose_rotation(bone_index, rot);
                    }
                    else if(track->type == Animation::TYPE_SCALE_3D) {
                        Animation::ScaleTrack* track_cache = static_cast<Animation::ScaleTrack*>(track);
                        int bone_index = get_bone_human_index(p_skeleton, p_bone_map, track_cache->path);
                        if(bone_index < 0) {
                            continue;
                        }
                        Error err = p_animation->try_scale_track_interpolate(j, time, &scale);
                        p_skeleton->set_bone_pose_scale(bone_index, scale);
                    }
                }
                // 转换骨骼姿势到动画
                build_skeleton_pose(p_skeleton,p_config,skeleton_config);
                // 存储动画
                animation_lookat.set(i,skeleton_config.bone_lookat);
                animation_root_position.set(i,skeleton_config.root_position);

            }

            Ref<Animation> out_anim;
            out_anim.instantiate();
            out_anim->set_is_human_animation(true);
            if(animation_lookat.size() > 0) {

                auto& keys = animation_lookat[0];

                for(auto& it : keys) {
                    int track_index = out_anim->add_track(Animation::TYPE_POSITION_3D);
                    Animation::PositionTrack* track = static_cast<Animation::PositionTrack*>(out_anim->get_track(track_index));
                    track->path = String("hm.") + it.key;
                    track->interpolation = Animation::INTERPOLATION_LINEAR;
                    track->positions.resize(animation_lookat.size());
                    for(int i = 0;i < animation_lookat.size();i++) {
                        double time = double(i) / 100.0;
						Animation::TKey<Vector3> key;
						key.time = time;
						key.value = animation_lookat[i][it.key];
                        track->positions.set(i,key);

                    }
                }

                auto& root_keys = animation_root_position[0];
                for(auto& it : root_keys) {
                    int track_index = out_anim->add_track(Animation::TYPE_POSITION_3D);
                    Animation::PositionTrack* track = static_cast<Animation::PositionTrack*>(out_anim->get_track(track_index));
                    track->path = String("hm.p.") + it.key;
                    track->interpolation = Animation::INTERPOLATION_LINEAR;
                    track->positions.resize(animation_root_position.size());
                    for(int i = 0;i < animation_root_position.size();i++) {
                        double time = double(i) / 100.0;
                        Animation::TKey<Vector3> key;
                        key.time = time;
                        key.value = animation_root_position[i][it.key];
                        track->positions.set(i,key);
                    }
                }
                
            }
            // 拷贝轨迹
            for(auto& it : other_tracks) {
                out_anim->add_track_ins(it->duplicate());
            }

            // 
            return out_anim;


        }


        // 重定向根骨骼朝向
        static void retarget_root_motion(HumanConfig& p_config,HumanSkeleton& p_skeleton_config) {
            for(auto& it : p_config.root_bone) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                Transform3D local_trans;
				local_trans.basis = trans.basis;

				Transform3D child_trans = trans * p_skeleton_config.real_pose[pose.child_bones[0]];
				Vector3 forward = (child_trans.origin - trans.origin).normalized();
                
                trans.basis.rotate_to_align(forward, p_skeleton_config.bone_lookat[it] - trans.origin);
                p_skeleton_config.real_local_pose[it] = trans.basis.get_rotation_quaternion();


				p_skeleton_config.root_global_rotation[it] = (local_trans.basis.inverse() * trans.basis).get_rotation_quaternion();

            }
            
        }

        // 重定向骨骼
        static void retarget(HumanConfig& p_config,HumanSkeleton& p_skeleton_config) {
            for(auto& it : p_config.root_bone) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                Transform3D local_trans;
				local_trans.basis = trans.basis;

				Transform3D child_trans = trans * p_skeleton_config.real_pose[pose.child_bones[0]];
				Vector3 forward = (child_trans.origin - trans.origin).normalized();
                
                trans.basis.rotate_to_align(forward, p_skeleton_config.bone_lookat[it] - trans.origin);
                p_skeleton_config.real_local_pose[it] = trans.basis.get_rotation_quaternion();


				p_skeleton_config.root_global_rotation[it] = (local_trans.basis.inverse() * trans.basis).get_rotation_quaternion();
                

                retarget(p_config, pose, local_trans,p_skeleton_config);

                
            }
            
        }


        static const HashMap<String, String>& get_bone_label() {
            static HashMap<String, String> label_map = {
                {"Hips",L"臀部"},

                {"LeftUpperLeg",L"左上腿"},
                {"RightUpperLeg",L"右上腿"},

                {"LeftLowerLeg",L"左下腿"},
                {"RightLowerLeg",L"右下腿"},

                {"LeftFoot",L"左脚"},
                {"RightFoot",L"右脚"},

                {"Spine",L"脊柱"},
                {"Chest",L"颈部"},
                {"UpperChest",L"上胸部"},
                {"Neck",L"颈部"},
                {"Head",L"头部"},

                {"LeftShoulder",L"左肩"},
                {"RightShoulder",L"右肩"},

                {"LeftUpperArm",L"左上臂"},
                {"RightUpperArm",L"右上臂"},

                {"LeftLowerArm",L"左下臂"},
                {"RightLowerArm",L"右下臂"},

                {"LeftHand",L"左手"},
                {"RightHand",L"右手"},

                {"LeftToes",L"左足"},
                {"RightToes",L"右足"},

                {"LeftEye",L"左眼"},
                {"RightEye",L"右眼"},

                {"Jaw",L"下巴"},

                {"LeftThumbMetacarpal",L"左拇指"},
                {"LeftThumbProximal",L"左拇指近端"},
                {"LeftThumbDistal",L"左拇指远端"},

                {"LeftIndexProximal",L"左食指近端"},
                {"LeftIndexIntermediate",L"左食指中间"},
                {"LeftIndexDistal",L"左食指远端"},

                {"LeftMiddleProximal",L"左中指近端"},
                {"LeftMiddleIntermediate",L"左中指中间"},
                {"LeftMiddleDistal",L"左中指远端"},

                {"LeftRingProximal",L"左无名指近端"},
                {"LeftRingIntermediate",L"左无名指中间"},
                {"LeftRingDistal",L"左无名指远端"},

                {"LeftLittleProximal",L"左小拇指近端"},
                {"LeftLittleIntermediate",L"左小拇指中间"},
                {"LeftLittleDistal",L"左小拇指远端"},

                {"RightThumbMetacarpal",L"右拇指"},
                {"RightThumbProximal",L"右拇指近端"},
                {"RightThumbDistal",L"右拇指远端"},

                {"RightIndexProximal",L"右食指近端"},
                {"RightIndexIntermediate",L"右食指中间"},
                {"RightIndexDistal",L"右食指远端"},

                {"RightMiddleProximal",L"右中指近端"},
                {"RightMiddleIntermediate",L"右中指中间"},
                {"RightMiddleDistal",L"右中指远端"},

                {"RightRingProximal",L"右无名指近端"},
                {"RightRingIntermediate",L"右无名指中间"},
                {"RightRingDistal",L"右无名指远端"},

                {"RightLittleProximal",L"右小拇指近端"},
                {"RightLittleIntermediate",L"右小拇指中间"},
                {"RightLittleDistal",L"右小拇指远端"},

            };
            return label_map;
        }

     private:
     
        struct SortStringName {
            bool operator()(const StringName &l, const StringName &r) const {
                return l.str() > r.str();
            }
        };

        static void build_virtual_pose(HumanConfig& p_config,Skeleton3D* p_skeleton,Transform3D parent_trans,  int bone_index,HashMap<String, String>& p_human_bone_label) {
            
            //Vector<int> child_bones = p_skeleton->get_bone_children(bone_index);
            //for(int i=0; i < child_bones.size(); i++)
			{
                String bone_name = p_skeleton->get_bone_name(bone_index);
                if(!p_human_bone_label.has(bone_name)) {
                    return;
                }
                BonePose & pose = p_config.virtual_pose[bone_name];
                Transform3D trans = p_skeleton->get_bone_global_pose(bone_index);
                float height = 1.0;
                Vector<int> children = p_skeleton->get_bone_children(bone_index);

				for (int j = 0; j < children.size(); j++) {
					pose.child_bones.push_back(p_skeleton->get_bone_name(children[j]));
				}
				pose.child_bones.sort_custom<SortStringName>();

                Vector3 bone_foreard = trans.origin - parent_trans.origin;
				height = bone_foreard.length();

                Transform3D local_trans = p_skeleton->get_bone_pose(bone_index);
                pose.position = local_trans.origin;
                pose.rotation = local_trans.basis.get_rotation_quaternion();
                float inv_height = 1.0 / height;
                pose.scale = Vector3(inv_height,inv_height,inv_height);
                pose.length = height;
                pose.bone_index = bone_index;
                for(int j=0;j<children.size();j++) {
					build_virtual_pose(p_config, p_skeleton, trans, children[j], p_human_bone_label);
                }
            }
            
        }
		// 构建真实姿势
		static void build_skeleton_pose(Skeleton3D* p_skeleton, HumanConfig& p_config, HumanSkeleton& p_skeleton_config) {
			p_skeleton->_update_bones_nested_set();
			p_skeleton->force_update_all_bone_transforms(false);

			Transform3D local_trans;
			for (auto& it : p_config.root_bone) {
				BonePose& pose = p_config.virtual_pose[it];
				Transform3D& trans = p_skeleton_config.real_pose[it];
				trans = p_skeleton->get_bone_global_pose(pose.bone_index);

				local_trans.basis = Basis(pose.rotation);
				build_skeleton_local_pose(p_skeleton, p_config, pose, local_trans,p_skeleton_config);
			}

			for (auto& it : p_config.root_bone) {
				Transform3D& trans = p_skeleton_config.real_pose[it];
				Vector3 bone_foreard = Vector3(0, 0, 1);

				BonePose& pose = p_config.virtual_pose[it];
				Transform3D local_trans;
				local_trans.basis = Basis(pose.rotation);
				{
					Transform3D& child_trans = p_skeleton_config.real_pose[pose.child_bones[0]];
					bone_foreard = child_trans.origin - trans.origin;
				}
				p_skeleton_config.bone_lookat[it] = bone_foreard.normalized();
				p_skeleton_config.root_position[it] = trans.origin - pose.position;
				build_skeleton_global_lookat(p_config, pose,local_trans, p_skeleton_config);
			}
		}
        static void build_skeleton_local_pose(Skeleton3D* p_skeleton,HumanConfig& p_config,BonePose& parent_pose, Transform3D& parent_trans,HumanSkeleton& p_skeleton_config) {
            for(auto& it : parent_pose.child_bones) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                trans.basis = Basis(p_skeleton->get_bone_pose_rotation(pose.bone_index));
				trans.origin = pose.position;
				trans = parent_trans * trans;

                build_skeleton_local_pose(p_skeleton,p_config, pose, trans,p_skeleton_config);

            }
        }

        static void build_skeleton_global_lookat(HumanConfig& p_config, BonePose& bone_pose,Transform3D& parent_pose,HumanSkeleton& p_skeleton_config) {

            for(auto& it : bone_pose.child_bones) {
				BonePose& child_pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
				Vector3 forward;
				if (child_pose.child_bones.size() > 0) {
					Transform3D& child_trans = p_skeleton_config.real_pose[child_pose.child_bones[0]];
					forward = child_trans.origin - trans.origin;
				}
				else {
					forward = trans.origin - parent_pose.origin;
				}
                p_skeleton_config.bone_lookat[it] = trans.origin + forward.normalized();
				build_skeleton_global_lookat(p_config, child_pose,trans, p_skeleton_config);
            }
            
        }
        static void retarget(HumanConfig& p_config,BonePose& pose,Transform3D& parent_trans,HumanSkeleton& p_skeleton_config) {

            // 重定向骨骼的世界坐标
            Quaternion rot;

            for(auto& it : pose.child_bones) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                trans = parent_trans * trans;
				Vector3 forward;
				if (pose.child_bones.size() > 0) {
					Transform3D  child_trans = p_skeleton_config.real_pose[pose.child_bones[0]];
					child_trans = trans * child_trans;
					forward = child_trans.origin - trans.origin;
				}
				else {
					forward = trans.origin - parent_trans.origin;
				}
				float length = (p_skeleton_config.bone_lookat[it] - trans.origin).length();
                trans.basis.rotate_to_align(forward, p_skeleton_config.bone_lookat[it] - trans.origin);

                Transform3D local_trans = parent_trans.inverse() * trans;
                p_skeleton_config.real_local_pose[it] = local_trans.basis.get_rotation_quaternion();
                retarget(p_config,pose,trans,p_skeleton_config);
            }
            
        }
    

    };

    
}


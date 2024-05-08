

#ifndef ANIMATION_HELP_H
#define ANIMATION_HELP_H
#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "../csv/CSV_EditorImportPlugin.h"
#include "../commom/common.h"



class AnimationManager : public Object
{
    GDCLASS(AnimationManager, Object);
    static void _bind_methods();
    static AnimationManager *singleton;
    double lastTickTime = 0;
    NodePath animationPlayer = NodePath("AnimationPlayer");
    bool is_init = false;



    struct AnimationInfo
    {
        String animationPath;
        Ref<Animation> animation;
        bool is_error = false;
        double last_using_time = 0;
        AnimationInfo()
        {

        }
        AnimationInfo(String path)
        {
            animationPath = path;
        }
        void tick(double time)
        {
            if(time - last_using_time > 60.0)
            {
                animation.unref();
                last_using_time = time;
            }
        }
    };
    static Ref<Animation> getNullAnimation();
    // 处理开始播放动画
    static void on_animation_player_begin(AnimationPlayer* player,double p_delta, bool p_update_only)
    {
        if(singleton == nullptr || player == nullptr)
        {
            return;
        }
        if(player->get_animation_group() == StringName())
        {
            return;
        }
        singleton->animationGroup[player->get_animation_group()];
        player->reset_all_animation(singleton->getNullAnimation());
    }
    static void on_animation_player_end(AnimationPlayer* player,double p_delta, bool p_update_only)
    {
        if(singleton == nullptr || player == nullptr)
        {
            return;
        }
        if(player->get_animation_group() == StringName())
        {
            return;
        }
        Node3D* parent = Object::cast_to<Node3D>(player->get_parent());
        Transform3D t = parent->get_global_transform();
        t.basis.rotate(player->get_root_motion_rotation());


        parent->set_global_transform(t);

    }
    static void on_get_animation(AnimationPlayer* player,StringName p_animation)
    {
        if(singleton == nullptr || player == nullptr)
        {
            return;
        }
        if(player->get_animation_group() == StringName())
        {
            return;
        }
        if(singleton->animationGroup.has(p_animation))
        {
            auto anim = singleton->animationGroup[p_animation].get_animation(p_animation,OS::get_singleton()->get_unix_time());
            player->change_animation(p_animation,anim);
        }
        singleton->tick();

    }
    void tick()
    {
        double tm = OS::get_singleton()->get_unix_time();
        if(tm - lastTickTime < 5)
        {
            return;
        }
    }
    struct AnimationGroup
    {
        HashMap<StringName,AnimationInfo> animation;
        StringName group;
        // 动画树的路径
        String animationTreePath;
        String initAnimationLibraryPath;
        Ref<AnimationLibrary> animationLibrary;
        bool is_error = false;
        AnimationGroup()
        {

        }
        AnimationGroup(StringName _group,String anim_tree_path,String _animationLibraryPath)
        {
            group = _group;
            animationTreePath = anim_tree_path;
            initAnimationLibraryPath = _animationLibraryPath;
        }
        void add_animation(StringName anim_name,String animation_path)
        {
            if(animation.has(anim_name))
            {
                ERR_FAIL_MSG(vformat("Animation exits : \"%s\".", anim_name));
            }
            animation[anim_name] = AnimationInfo(animation_path);
        }
        Ref<Animation> get_animation(StringName p_animation,double time)
        {
            if(is_error)
            {
                return Ref<Animation>();
            }
            if(animation.has(p_animation))
            {
                auto& anim = animation[p_animation];
                anim.last_using_time = time;
                return anim.animation;
            }
            return AnimationManager::getNullAnimation();
        }
        // 配置动画树
        void setup_animation_player(Node* node,AnimationPlayer* player)
        {
            if(is_error)
            {
                return;
            }
            if(animationLibrary.is_null())
            {
                animationLibrary = ResourceLoader::load(initAnimationLibraryPath, NodePath("AnimationLibrary"));
                if(animationLibrary.is_null())
                {
                    
                    is_error = true;
                    ERR_FAIL_MSG(vformat("AnimationLibrary not fund : \"%s\".", initAnimationLibraryPath));
                    return;
                }
            }
            player->clear_all_animation();

            Ref<PackedScene> scene = ResourceLoader::load(animationTreePath);
            if(!scene.is_valid())
            {
                is_error = true;
                ERR_FAIL_MSG(vformat("AnimationTree not fund : \"%s\".", animationTreePath));
                return;
            }
            AnimationTree* tree = Object::cast_to<AnimationTree>(scene->instantiate());
            if(tree == nullptr)
            {
                is_error = true;
                ERR_FAIL_MSG(vformat("AnimationTree not fund : \"%s\".", animationTreePath));
                return;
            }      
            tree->set_owner(node);      
            player->add_animation_library(animationLibrary->get_name(),animationLibrary);
            player->set_begin_animation_cb(callable_mp_static(on_animation_player_begin));
            //player->set_end_animation_cb(callable_mp_static(on_animation_player_end));
            player->set_get_animation_cb(callable_mp_static(on_get_animation));
            player->set_animation_group(group);
            player->get_parent()->add_child(tree);
            return;
        }
        
        void tick(double time)
        {
            if(is_error)
            {
                return;
            }
            if(animationLibrary.is_null())
            {
                return;
            }
            for(auto& anim : animation)
            {
                anim.value.tick(time);
            }
        }
    };
    // 动画组信息
    HashMap<StringName,AnimationGroup> animationGroup;

    // 自动卸载时间
    float AutoUnlodTime = 50;
public:
    static void setup_animation_tree(Node* node,StringName group)
    {
        if(singleton == nullptr || node == nullptr)
        {
            return;
        }
        singleton->load_animation_tree(node,group); 
    }
    String animation_config_path = "res://animation_config.tres";
    void init()
    {
        if(is_init)
        {
            return;
        }
        animationGroup.clear();


        is_init = true;
    }
    // 增加一个动画分组
    void add_animation_group(StringName group_name,String animation_tree_path,String animation_library_path)
    {
        if(animationGroup.has(group_name))
        {            
	        ERR_FAIL_MSG(vformat("Animation group exits : \"%s\".", group_name));
        }
        animationGroup[group_name] = AnimationGroup(group_name,animation_tree_path,animation_library_path);
    }
    // 增加动画
    void add_animation(StringName group_name,StringName anim_name,String animation_path)
    {
        if(!animationGroup.has(group_name))
        {            
	        ERR_FAIL_MSG(vformat("Animation not fund : \"%s\".", group_name));
        }
        animationGroup[group_name].add_animation(anim_name,animation_path);
    }
    // 获取动画节点列表
    TypedArray<Animation> get_animation_nodes(String path,bool is_single_valid, bool change_animName_to_sceneName)
    {

        Ref<PackedScene> scene = ResourceLoader::load(path);
        if(!scene.is_valid())
        {
            return TypedArray<Animation>();
        }
        Ref<SceneState> ss =  scene->get_state();
        Node * node = ss->instantiate(SceneState::GEN_EDIT_STATE_DISABLED);
        AnimationPlayer* player = Object::cast_to<AnimationPlayer>(node->get_node(NodePath("AnimationPlayer")));


        if(player == nullptr)
        {
            return TypedArray<Animation>();
        }
        List<StringName> list;
        TypedArray<Animation> rs;
        player->get_animation_list(&list);
		for (const StringName& E : list)
        {
            auto anim = player->get_animation(E);
            if(anim.is_valid())
            {                
                if(is_single_valid)
                {
                    if(anim->get_length() > 0)
                    {
                        if(change_animName_to_sceneName)
                        {
                            anim->set_name(scene->get_name());
                        }
                        rs.push_back(anim);
                        memdelete(node);
                        return rs;
                    }
                }
                else
                {
                    if(anim->get_length() > 0)
                    {
                        if(rs.size() == 0)
                        {
                            anim->set_name(scene->get_name());
                        }
                        else
                        {
                            
                            anim->set_name(scene->get_name() + "_" + itos(rs.size() + 1));
                        }
                        rs.push_back(anim);
                    }
                }

            }
        }
        memdelete(node);
        return rs;
    }
    // 保存动画树
    void save_animation_tree(AnimationTree * tree,String path)    
    {
        Ref<PackedScene> scene = Ref<PackedScene>(memnew(PackedScene));
        scene->pack(tree);
        ResourceSaver::save(scene,path);
    }
    // 保存动画到动画库
    void save_animation_library(AnimationPlayer * tree,String path)    
    {
        Ref<AnimationLibrary> library = Ref<AnimationLibrary>(memnew(AnimationLibrary));
        List<StringName> list;
        List<StringName> anim_list;
        tree->get_animation_library_list(&list);
		for (const StringName& E : list)
        {
            auto lab = tree->get_animation_library(E);
            if(lab.is_valid())
            {
                anim_list.clear();
                lab->get_animation_list(&anim_list);
				for (const StringName& AE : anim_list)
                {
                    auto anim = lab->get_animation(AE);
                    if(anim.is_valid())
                    {
                        library->add_animation(AE,anim);
                    }
                }
            }
        }
        ResourceSaver::save(library,path);
        
    }
    // 保存动画到动画库
    void create_animation_library(Dictionary animation,String name,String path)    
    {
        Ref<AnimationLibrary> library = Ref<AnimationLibrary>(memnew(AnimationLibrary));
        List<StringName> list;

        auto anim_name_list = animation.keys();
        auto anim_list = animation.values();
        for(int i = 0; i < anim_name_list.size(); i++)
        {
            library->add_animation(anim_name_list[i],anim_list[i]);
        }
        library->set_name(name);
        ResourceSaver::save(library,path);
        
    }
    // 加载动画树
    void load_animation_tree(Node* node,StringName group)
    {
        AnimationPlayer* player = Object::cast_to<AnimationPlayer>(node->get_node(NodePath("AnimationPlayer")));
        if(player == nullptr)
        {
            ERR_FAIL_MSG("AnimationPlayer not found.");
            return;
        }
        if(!animationGroup.has(group))
        {
            ERR_FAIL_MSG("AnimationGroup not found.");
            return;
        }
        player->set_owner(node);
        animationGroup[group].setup_animation_player(node,player);

    }

};
#endif

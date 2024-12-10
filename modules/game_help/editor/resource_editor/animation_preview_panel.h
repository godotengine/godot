#include "../../game_gui/item_box.h"
#include "scene/gui/box_container.h"
#include "scene/gui/tab_bar.h"
#include "scene/gui/separator.h"
#include "scene/gui/button.h"
#include "scene/resources/animation.h"
#include "scene/gui/check_box.h"
#include "core/io/json.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include  "../../logic/body_main.h"
#include "resource_editor_tool_item.h"
#include "../animator_node_preview.h"
#include "../../logic/character_manager.h"


class AnimationPreviewPanel : public VBoxContainer {
    GDCLASS(AnimationPreviewPanel, VBoxContainer);
public:
    AnimationPreviewPanel();
protected:
    void _notification(int what);
    void _on_revert_show_toggled(bool pressed) ;
    void _on_tab_changed(int p_tab_idx);
    void _on_tag_pressed(bool value,String animation_tag) ;

    void _on_update_animation_resource();
    void parse_animation_path(String path) ;
	struct AnimationInfo : RefCounted {
		Ref<Animation> animation;
		String animation_path;
		StringName animation_group;
		StringName animation_tag;
        bool is_visible = false;
		bool load(Dictionary& p_dict) {
			animation_path = p_dict["animation_path"];
			animation_group = p_dict["animation_group"];
			animation_tag = p_dict["animation_tag"];
			return FileAccess::exists(animation_path);
		}
		void save(Dictionary& p_dict) {
			p_dict["animation_path"] = animation_path;
			p_dict["animation_group"] = animation_group;
			p_dict["animation_tag"] = animation_tag;
		}
        bool is_show(const StringName& group, const Dictionary& p_select_tag,bool is_revert_show) {            
            bool _is_show = animation_group == group && p_select_tag.get(animation_tag,false);
            if(_is_show && !is_revert_show) {
                return true;
            }
            return false;            
        }
	};
    void load_animation_config() ;
    void refresh_animation_list(bool update_ui = true) ;
    void save_animation_config();

protected:
    void on_item_visible_state_change(ItemBoxItem* item,bool p_visible);
    
    AnimationNodePreview* get_item_preview() ;

    bool update_preview() ;

    
protected:
    String last_select_group = "";
    Dictionary last_select_tag;
    bool last_inv_select = false;
    Array animation_group_list ;
    Array animation_tags_list;
    CheckBox* revert_show = nullptr;
    Button* update_animation_resource = nullptr;
    TabBar* animation_group_tab= nullptr;
    VFlowContainer* animation_tag_list= nullptr;
    ItemBox* animation_list= nullptr;
    LocalVector<String> animation_list_paths = {"res://Assets/public/animation/", "res://Assets/public/human_animation/"};
    List<Ref<AnimationInfo>> animations;
    List<Ref<AnimationInfo>> curr_show_animations;
    HashMap<ItemBoxItem*,AnimationNodePreview*> animation_preview_list;
    List<AnimationNodePreview*> unuse_preview_list;
    float last_update_time = 0;
    bool is_dirty = false;
};

class AnimationPreviewPanelItem : public ResourceEditorToolItem {
    GDCLASS(AnimationPreviewPanelItem, ResourceEditorToolItem);
    static void _bind_methods() {}

    virtual String get_name() const { return String(L"动画浏览"); }
    virtual Control *get_control() {

		return Object::cast_to< AnimationPreviewPanel>(ObjectDB::get_instance(control_id));
	}
	AnimationPreviewPanelItem() {
		AnimationPreviewPanel* c = memnew(AnimationPreviewPanel);
		control_id = c->get_instance_id();
	}
	~AnimationPreviewPanelItem() {
		Control* c = get_control();
		if (c != nullptr) {
			c->queue_free();
		}
	}
	ObjectID control_id;
};

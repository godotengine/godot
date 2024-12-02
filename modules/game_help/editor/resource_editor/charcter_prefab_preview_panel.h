#pragma once

#include "scene/gui/box_container.h"
#include "editor/editor_properties.h"
#include "editor/editor_resource_picker.h"
#include "../../logic/character_manager.h"
#include "resource_editor_tool_item.h"
#include "scene_view_panel.h"
#include "scene/gui/separator.h"
#include "../../game_gui/item_box.h"

class CharcterPrefabPreviewPanel : public VBoxContainer
{
    GDCLASS(CharcterPrefabPreviewPanel, VBoxContainer);

public:
    CharcterPrefabPreviewPanel(){
        set_h_size_flags(SIZE_EXPAND_FILL);
        set_v_size_flags(SIZE_EXPAND_FILL);

        update_animation_resource = memnew(Button);
        update_animation_resource->set_text(TTR("Update Animation"));
        update_animation_resource->set_h_size_flags(SIZE_EXPAND_FILL);
        update_animation_resource->connect("pressed", callable_mp(this, &AnimationPreviewPanel::_on_update_animation_resource));
        add_child(update_animation_resource);

        revert_show = memnew(CheckBox);
        revert_show->set_text(L"反向选择");
        revert_show->set_tooltip_text(L"反向选择,用来定位哪些没有分类的动画");
        revert_show->set_h_size_flags(SIZE_EXPAND_FILL);
        revert_show->connect("toggled", callable_mp(this, &AnimationPreviewPanel::_on_revert_show_toggled));
        add_child(revert_show);
        
		animation_group_tab = memnew(TabBar);
		animation_group_tab->set_self_modulate(Color(0.958148, 0.603324, 0.533511, 1));
		animation_group_tab->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
		animation_group_tab->set_max_tab_width(80);
		animation_group_tab->connect("tab_changed", callable_mp(this, &AnimationPreviewPanel::_on_tab_changed));
        add_child(animation_group_tab);

        HSeparator* separator = memnew(HSeparator);
        separator->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
        add_child(separator);

        HBoxContainer* hbc = memnew(HBoxContainer);
        hbc->set_v_size_flags(SIZE_EXPAND_FILL);
        add_child(hbc);


        VSeparator* vs = memnew(VSeparator);
        vs->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
        hbc->add_child(vs);

        animation_list = memnew(ItemBox);
        hbc->add_child(animation_list);

    }
    ~CharcterPrefabPreviewPanel() override;

protected:
    CheckBox* revert_show = nullptr;
    
    Button* update_animation_resource = nullptr;
    TabBar* animation_group_tab= nullptr;

    
    ItemBox* animation_list= nullptr;
};
#pragma once
#include "scene/gui/control.h"
#include "scene/gui/box_container.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/button.h"
#include "resource_editor_tool_item.h"
class ResourceEditorTool : public VBoxContainer
{
    GDCLASS(ResourceEditorTool, VBoxContainer)
    static void _bind_methods() {}
public:
    void select_tool(String tool) {
        curr_tool = tool;
        for(int i = 0; i < tool_bar->get_child_count(); i++) {
            Button* child = Object::cast_to<Button>(tool_bar->get_child(i));
            if(child->get_text() == tool) {
                child->set_pressed(true);
                child->set_modulate(Color(1, 0.355482, 0.26278, 1));
            }
            else {
                child->set_pressed(false);
                child->set_modulate(Color(1, 1, 1, 1));
            }
        }
        for(auto &item : items) {
            item.value->get_control()->set_visible(item.key == tool);
        }
    }

    ResourceEditorTool( ) {
        set_h_size_flags(SIZE_EXPAND_FILL);
        set_v_size_flags(SIZE_EXPAND_FILL);
		List<StringName> inheriters;
		ClassDB::get_inheriters_from_class(ResourceEditorToolItem::get_class_static(), &inheriters);

        for(const StringName &S : inheriters) {
            curr_tool = S;
            Ref<ResourceEditorToolItem> item = Ref<ResourceEditorToolItem>(ClassDB::instantiate(S));
            items[item->get_name()] = item;
        }
        tool_bar = memnew(FlowContainer);
        tool_bar->set_alignment(FlowContainer::ALIGNMENT_CENTER);
        tool_bar->set_reverse_fill(true);
        add_child(tool_bar);

        HSeparator* separator = memnew(HSeparator);
        separator->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
        add_child(separator);


        tool_content = memnew(VBoxContainer);
        tool_content->set_v_size_flags(SIZE_EXPAND_FILL);
        add_child(tool_content);

        if(items.size() > 0) {
            for(auto &item : items) {
                Button* button = memnew(Button);
                button->set_text(item.value->get_name());
                button->connect("pressed", callable_mp(this, &ResourceEditorTool::select_tool).bind(item.value->get_name()));
                tool_bar->add_child(button);
                item.value->get_control()->set_visible(false);

                Control* control = item.value->get_control();
                control->set_h_size_flags(SIZE_EXPAND_FILL);
                tool_content->add_child(control);
            }

            select_tool(items.begin()->value->get_name());
        }

        // List<StringName> script_inheriters;
        // ScriptServer::get_inheriters_list(ResourceEditorToolItem::get_class_static(), &script_inheriters);
    }
    virtual ~ResourceEditorTool() {}
    protected:
    FlowContainer* tool_bar = nullptr;
    VBoxContainer* tool_content = nullptr;
    HashMap<String, Ref<ResourceEditorToolItem>> items;
    String curr_tool;
};

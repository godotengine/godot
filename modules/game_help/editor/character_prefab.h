#include "body_main_editor.h"
#include "../logic/character_shape/character_body_prefab.h"
#include "editor/plugins/mesh_editor_plugin.h"


class CharacterPrefabSection : public LogicSectionBase {
    GDCLASS(CharacterPrefabSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override
    {
		set_category_name(L"身体部位列表");

    }
    virtual void create_child_list(VBoxContainer *hb) override {

    }
	virtual void update_state() override {
        prefab = Object::cast_to<CharacterBodyPrefab>(object);
        part = prefab->get_parts();
        arr = part.keys();
        for(int i=0; i<arr.size(); i++) {
            HBoxContainer* hbox = memnew(HBoxContainer);
            tasks_container->add_child(hbox);
            
            Label* label = memnew(Label);
            label->set_text(arr[i]);
            label->set_h_size_flags(SIZE_EXPAND_FILL);
            hbox->add_child(label);

            CheckButton* button = memnew(CheckButton);
            button->set_pressed(part[arr[i]]);
            button->connect("toggled", callable_mp(this, &CharacterPrefabSection::on_part_toggled).bind(i));
            hbox->add_child(button);
        }
    }
    void on_part_toggled(bool p_pressed, int index) {
        part[arr[index]] = p_pressed;
        prefab->set_parts(part);        
    }
    virtual String get_section_unfolded() const override{ return "parts Condition Section"; }

    Ref<CharacterBodyPrefab> prefab;
    Dictionary part;
    Array arr;
};

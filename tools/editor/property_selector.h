#ifndef PROPERTYSELECTOR_H
#define PROPERTYSELECTOR_H

#include "tools/editor/property_editor.h"
#include "scene/gui/rich_text_label.h"
#include "editor_help.h"

class PropertySelector : public ConfirmationDialog {
	GDCLASS(PropertySelector,ConfirmationDialog )


	LineEdit *search_box;
	Tree *search_options;

	void _update_search();

	void _sbox_input(const InputEvent& p_ie);

	void _confirmed();
	void _text_changed(const String& p_newtext);

	EditorHelpBit *help_bit;

	bool properties;
	String selected;
	Variant::Type type;
	InputEvent::Type event_type;
	String base_type;
	ObjectID script;
	Object *instance;

	void _item_selected();
protected:
	void _notification(int p_what);
	static void _bind_methods();



public:


	void select_method_from_base_type(const String& p_base,const String& p_current="");
	void select_method_from_script(const Ref<Script>& p_script,const String& p_current="");
	void select_method_from_basic_type(Variant::Type p_type,const String& p_current="");
	void select_method_from_instance(Object* p_instance, const String &p_current="");

	void select_property_from_base_type(const String& p_base,const String& p_current="");
	void select_property_from_script(const Ref<Script>& p_script,const String& p_current="");
	void select_property_from_basic_type(Variant::Type p_type,InputEvent::Type p_event_type,const String& p_current="");
	void select_property_from_instance(Object* p_instance, const String &p_current="");

	PropertySelector();
};

#endif // PROPERTYSELECTOR_H

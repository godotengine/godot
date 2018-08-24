#ifndef EDITOR_SECTIONED_INSPECTOR_H
#define EDITOR_SECTIONED_INSPECTOR_H

#include "editor/editor_inspector.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"

class SectionedInspectorFilter;

class SectionedInspector : public HSplitContainer {

	GDCLASS(SectionedInspector, HSplitContainer);

	ObjectID obj;

	Tree *sections;
	SectionedInspectorFilter *filter;

	Map<String, TreeItem *> section_map;
	EditorInspector *inspector;
	LineEdit *search_box;

	static void _bind_methods();
	void _section_selected();

	void _search_changed(const String &p_what);

public:
	void register_search_box(LineEdit *p_box);
	EditorInspector *get_inspector();
	void edit(Object *p_object);
	String get_full_item_path(const String &p_item);

	void set_current_section(const String &p_section);
	String get_current_section() const;

	void update_category_list();

	SectionedInspector();
	~SectionedInspector();
};
#endif // EDITOR_SECTIONED_INSPECTOR_H

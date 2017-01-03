#include "gi_probe_editor_plugin.h"


void GIProbeEditorPlugin::_bake() {

	if (gi_probe) {
		gi_probe->bake();
	}
}


void GIProbeEditorPlugin::edit(Object *p_object) {

	GIProbe * s = p_object->cast_to<GIProbe>();
	if (!s)
		return;

	gi_probe=s;
}

bool GIProbeEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("GIProbe");
}

void GIProbeEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		bake->show();
	} else {

		bake->hide();
	}

}

void GIProbeEditorPlugin::_bind_methods() {

	ClassDB::bind_method("_bake",&GIProbeEditorPlugin::_bake);
}

GIProbeEditorPlugin::GIProbeEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	bake = memnew( Button );
	bake->set_icon(editor->get_gui_base()->get_icon("BakedLight","EditorIcons"));
	bake->hide();;
	bake->connect("pressed",this,"_bake");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU,bake);
	gi_probe=NULL;
}


GIProbeEditorPlugin::~GIProbeEditorPlugin() {

	memdelete(bake);
}

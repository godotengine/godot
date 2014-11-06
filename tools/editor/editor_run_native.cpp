/*************************************************************************/
/*  editor_run_native.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "editor_run_native.h"
#include "editor_import_export.h"


void EditorRunNative::_notification(int p_what) {


	if (p_what==NOTIFICATION_ENTER_TREE) {

		List<StringName> ep;
		EditorImportExport::get_singleton()->get_export_platforms(&ep);
		ep.sort_custom<StringName::AlphCompare>();
		for(List<StringName>::Element *E=ep.front();E;E=E->next()) {

			Ref<EditorExportPlatform> eep = EditorImportExport::get_singleton()->get_export_platform(E->get());
			if (eep.is_null())
				continue;
			Ref<ImageTexture> icon = eep->get_logo();
			if (!icon.is_null()) {
				Image im = icon->get_data();
				im.clear_mipmaps();
				if (!im.empty()) {

					im.resize(16,16);

					Ref<ImageTexture> small_icon = memnew( ImageTexture);
					small_icon->create_from_image(im);
					MenuButton *mb = memnew( MenuButton );
					mb->get_popup()->connect("item_pressed",this,"_run_native",varray(E->get()));
					mb->set_icon(small_icon);
					add_child(mb);
					menus[E->get()]=mb;
				}
			}
		}
	}

	if (p_what==NOTIFICATION_PROCESS) {


		bool changed = EditorImportExport::get_singleton()->poll_export_platforms() || first;

		if (changed) {

			for(Map<StringName,MenuButton*>::Element *E=menus.front();E;E=E->next()) {

				Ref<EditorExportPlatform> eep = EditorImportExport::get_singleton()->get_export_platform(E->key());
				MenuButton *mb = E->get();
				int dc = eep->get_device_count();

				if (dc==0) {
					mb->hide();
				} else {

					mb->get_popup()->clear();
					mb->show();
					for(int i=0;i<dc;i++) {

						mb->get_popup()->add_icon_item(get_icon("Play","EditorIcons"),eep->get_device_name(i));
						mb->get_popup()->set_item_tooltip(mb->get_popup()->get_item_count() -1,eep->get_device_info(i));
					}
				}
			}

			first=false;
		}
	}

}


void EditorRunNative::_run_native(int p_idx,const String& p_platform) {

	Ref<EditorExportPlatform> eep = EditorImportExport::get_singleton()->get_export_platform(p_platform);
	ERR_FAIL_COND(eep.is_null());
	eep->run(p_idx,deploy_dumb);
}

void EditorRunNative::_bind_methods() {

	ObjectTypeDB::bind_method("_run_native",&EditorRunNative::_run_native);
}

void EditorRunNative::set_deploy_dumb(bool p_enabled) {

	deploy_dumb=p_enabled;
}

bool EditorRunNative::is_deploy_dumb_enabled() const{

	return deploy_dumb;
}


EditorRunNative::EditorRunNative()
{
	set_process(true);
	first=true;
	deploy_dumb=false;
}

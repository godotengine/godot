/*************************************************************************/
/*  property_editor.cpp                                                  */
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
#include "property_editor.h"
#include "scene/gui/label.h"
#include "io/resource_loader.h"
#include "io/image_loader.h"
#include "object_type_db.h"
#include "print_string.h"
#include "globals.h"
#include "scene/resources/font.h"
#include "pair.h"
#include "scene/scene_string_names.h"
#include "editor_settings.h"
#include "editor_import_export.h"
#include "editor_node.h"

void CustomPropertyEditor::_notification(int p_what) {
	

	if (p_what==NOTIFICATION_DRAW) {
		
		RID ci = get_canvas_item();
		get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));		
		/*
		if (v.get_type()==Variant::COLOR) {
			
			VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2( 10,10,60, get_size().height-20 ), v );
		}*/
	}
}


void CustomPropertyEditor::_menu_option(int p_which) {


	switch(type) {

		case Variant::INT: {

			if (hint==PROPERTY_HINT_FLAGS) {

				int val = v;

				if (val&(1<<p_which)) {

					val&=~(1<<p_which);
				} else {
					val|=(1<<p_which);
				}

				v=val;
				emit_signal("variant_changed");
			}
		} break;
		case Variant::OBJECT: {

			switch(p_which) {
				case OBJ_MENU_LOAD: {

					file->set_mode(FileDialog::MODE_OPEN_FILE);
					List<String> extensions;
					String type=(hint==PROPERTY_HINT_RESOURCE_TYPE)?hint_text:String();

					ResourceLoader::get_recognized_extensions_for_type(type,&extensions);
					file->clear_filters();
					for (List<String>::Element *E=extensions.front();E;E=E->next()) {

						file->add_filter("*."+E->get()+" ; "+E->get().to_upper() );

					}

					file->popup_centered_ratio();
				} break;

				case OBJ_MENU_EDIT: {

					RefPtr RefPtr=v;

					if (!RefPtr.is_null()) {

						emit_signal("resource_edit_request");
						hide();
					}
				} break;
				case OBJ_MENU_CLEAR: {


					v=Variant();
					emit_signal("variant_changed");
					hide();
				} break;

				case OBJ_MENU_MAKE_UNIQUE: {


					RefPtr RefPtr=v;
					Ref<Resource> res_orig = RefPtr;
					if (res_orig.is_null())
						return;

					List<PropertyInfo> property_list;
					res_orig->get_property_list(&property_list);
					List< Pair<String,Variant> > propvalues;

					for(List<PropertyInfo>::Element *E=property_list.front();E;E=E->next()) {

						Pair<String,Variant> p;
						PropertyInfo &pi = E->get();
						if (pi.usage&PROPERTY_USAGE_STORAGE) {

							p.first=pi.name;
							p.second=res_orig->get(pi.name);
						}

						propvalues.push_back(p);
					}

					String orig_type = res_orig->get_type();

					Object *inst = ObjectTypeDB::instance( orig_type );

					Ref<Resource> res = Ref<Resource>( inst->cast_to<Resource>() );

					ERR_FAIL_COND(res.is_null());

					for(List< Pair<String,Variant> >::Element *E=propvalues.front();E;E=E->next()) {

						Pair<String,Variant> &p=E->get();
						res->set(p.first,p.second);
					}

					v=res.get_ref_ptr();
					emit_signal("variant_changed");
					hide();
				} break;

				case OBJ_MENU_COPY: {

					EditorSettings::get_singleton()->set_resource_clipboard(v);

				} break;
				case OBJ_MENU_PASTE: {

					v=EditorSettings::get_singleton()->get_resource_clipboard();
					emit_signal("variant_changed");

				} break;
				case OBJ_MENU_REIMPORT: {

					RES r=v;
					if (r.is_valid() && r->get_import_metadata().is_valid()) {
						Ref<ResourceImportMetadata> rimd = r->get_import_metadata();
						Ref<EditorImportPlugin> eip = EditorImportExport::get_singleton()->get_import_plugin_by_name(rimd->get_editor());
						if (eip.is_valid()) {
							eip->import_dialog(r->get_path());
						}
					}
				} break;
				default: {


					ERR_FAIL_COND( inheritors_array.empty() );


					String intype=inheritors_array[p_which-TYPE_BASE_ID];

					Object *obj = ObjectTypeDB::instance(intype);
					ERR_BREAK( !obj );
					Resource *res=obj->cast_to<Resource>();
					ERR_BREAK( !res );

					v=Ref<Resource>(res).get_ref_ptr();
					emit_signal("variant_changed");

				} break;
			}


		} break;
		default:{}
	}


}

Variant CustomPropertyEditor::get_variant() const {
	
	return v;	
}
String CustomPropertyEditor::get_name() const {
	
	return name;
}

bool CustomPropertyEditor::edit(Object* p_owner,const String& p_name,Variant::Type p_type, const Variant& p_variant,int p_hint,String p_hint_text) {

	owner=p_owner;
	updating=true;
	name=p_name;
	v=p_variant;
	hint=p_hint;
	hint_text=p_hint_text;
	type_button->hide();
	color_picker->hide();
	texture_preview->hide();
	inheritors_array.clear();
	text_edit->hide();
	easing_draw->hide();
	
	for (int i=0;i<MAX_VALUE_EDITORS;i++) {
		
		value_editor[i]->hide();
		value_label[i]->hide();
		if (i<4)
			scroll[i]->hide();
	}

	for (int i=0;i<MAX_ACTION_BUTTONS;i++) {
	
		action_buttons[i]->hide();	
	}

	for(int i=0;i<20;i++)
		checks20[i]->hide();

	type = (p_variant.get_type()!=Variant::NIL && p_variant.get_type()!=Variant::_RID && p_type!=Variant::OBJECT)? p_variant.get_type() : p_type;


	switch(type) {
	
		case Variant::INT:
		case Variant::REAL: {

			if (hint==PROPERTY_HINT_ALL_FLAGS) {

				uint32_t flgs = v;
				for(int i=0;i<2;i++) {

					Point2 ofs(4,4);
					ofs.y+=22*i;
					for(int j=0;j<10;j++) {

						Button *c=checks20[i*10+j];
						Point2 o=ofs;
						o.x+=j*22;
						if (j>=5)
							o.x+=4;
						c->set_pos(o);
						c->set_pressed( flgs & (1<<(i*10+j)) );
						c->show();
					}


				}

				set_size(checks20[19]->get_pos()+Size2(20,25));


			} else if (hint==PROPERTY_HINT_EXP_EASING) {

				easing_draw->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,5);
				easing_draw->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,5);
				easing_draw->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,5);
				easing_draw->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,30);
				type_button->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,3);
				type_button->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,3);
				type_button->set_anchor_and_margin(MARGIN_TOP,ANCHOR_END,25);
				type_button->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,7);
				type_button->set_text("Preset..");
				type_button->get_popup()->clear();
				type_button->get_popup()->add_item("Linear",EASING_LINEAR);
				type_button->get_popup()->add_item("Ease In",EASING_EASE_IN);
				type_button->get_popup()->add_item("Ease Out",EASING_EASE_OUT);
				if (hint_text!="attenuation") {
					type_button->get_popup()->add_item("Zero",EASING_ZERO);
					type_button->get_popup()->add_item("Easing In-Out",EASING_IN_OUT);
					type_button->get_popup()->add_item("Easing Out-In",EASING_OUT_IN);
				}

				type_button->show();
				easing_draw->show();
				set_size(Size2(200,150));
			} else if (hint==PROPERTY_HINT_FLAGS) {
				menu->clear();
				Vector<String> flags = hint_text.split(",");
				for(int i=0;i<flags.size();i++) {
					String flag = flags[i];
					if (flag=="")
						continue;
					menu->add_check_item(flag,i);
					int f = v;
					if (f&(1<<i))
						menu->set_item_checked(menu->get_item_index(i),true);
				}
				menu->set_pos(get_pos());
				menu->popup();
				hide();
				updating=false;
				return false;

			} else {
				List<String> names;
				names.push_back("value:");
				config_value_editors(1,1,50,names);
				Vector3 vec=v;
				value_editor[0]->set_text( String::num(v) );
			}

		} break;
		case Variant::STRING: {

			if (hint==PROPERTY_HINT_FILE || hint==PROPERTY_HINT_GLOBAL_FILE) {

				List<String> names;
				names.push_back("File..");
				names.push_back("Clear");
				config_action_buttons(names);

			} else if (hint==PROPERTY_HINT_DIR || hint==PROPERTY_HINT_GLOBAL_DIR) {

				List<String> names;
				names.push_back("Dir..");
				names.push_back("Clear");
				config_action_buttons(names);
			} else if (hint==PROPERTY_HINT_ENUM) {



			} else if (hint==PROPERTY_HINT_MULTILINE_TEXT) {

				text_edit->show();
				text_edit->set_text(v);

				//action_buttons[0];

				int button_margin = get_constant("button_margin","Dialogs");
				int margin = get_constant("margin","Dialogs");

				action_buttons[0]->set_anchor( MARGIN_LEFT, ANCHOR_END );
				action_buttons[0]->set_anchor( MARGIN_TOP, ANCHOR_END );
				action_buttons[0]->set_anchor( MARGIN_RIGHT, ANCHOR_END );
				action_buttons[0]->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
				action_buttons[0]->set_begin( Point2( 70, button_margin-5 ) );
				action_buttons[0]->set_end( Point2( margin, margin ) );
				action_buttons[0]->set_text("Close");
				action_buttons[0]->show();

			} else {
				List<String> names;
				names.push_back("string:");
				config_value_editors(1,1,50,names);
				Vector3 vec=v;
				value_editor[0]->set_text( v );
			}

		} break;
		case Variant::VECTOR2: {

			List<String> names;
			names.push_back("x");
			names.push_back("y");
			config_value_editors(2,2,10,names);
			Vector2 vec=v;
			value_editor[0]->set_text( String::num( vec.x) );
			value_editor[1]->set_text( String::num( vec.y) );
		} break;
		case Variant::RECT2: {

			List<String> names;
			names.push_back("x");
			names.push_back("y");
			names.push_back("w");
			names.push_back("h");
			config_value_editors(4,4,10,names);
			Rect2 r=v;
			value_editor[0]->set_text( String::num( r.pos.x) );
			value_editor[1]->set_text( String::num( r.pos.y) );
			value_editor[2]->set_text( String::num( r.size.x) );
			value_editor[3]->set_text( String::num( r.size.y) );
		} break;
		case Variant::VECTOR3: {
			
			List<String> names;
			names.push_back("x");
			names.push_back("y");
			names.push_back("z");
			config_value_editors(3,3,10,names);
			Vector3 vec=v;
			value_editor[0]->set_text( String::num( vec.x) );
			value_editor[1]->set_text( String::num( vec.y) );
			value_editor[2]->set_text( String::num( vec.z) );
		} break;
		case Variant::PLANE: {
			
			List<String> names;
			names.push_back("x");
			names.push_back("y");
			names.push_back("z");
			names.push_back("d");
			config_value_editors(4,4,10,names);
			Plane plane=v;
			value_editor[0]->set_text( String::num( plane.normal.x ) );
			value_editor[1]->set_text( String::num( plane.normal.y ) );
			value_editor[2]->set_text( String::num( plane.normal.z ) );
			value_editor[3]->set_text( String::num( plane.d ) );
			
		} break;
		case Variant::QUAT: {
			
			List<String> names;
			names.push_back("x");
			names.push_back("y");
			names.push_back("z");
			names.push_back("w");
			config_value_editors(4,4,10,names);
			Quat q=v;
			value_editor[0]->set_text( String::num( q.x ) );
			value_editor[1]->set_text( String::num( q.y ) );
			value_editor[2]->set_text( String::num( q.z ) );
			value_editor[3]->set_text( String::num( q.w ) );

		} break;
		case Variant::_AABB: {
			
			List<String> names;
			names.push_back("px");
			names.push_back("py");
			names.push_back("pz");
			names.push_back("sx");
			names.push_back("sy");
			names.push_back("sz");
			config_value_editors(6,3,16,names);
			
			AABB aabb=v;
			value_editor[0]->set_text( String::num( aabb.pos.x ) );
			value_editor[1]->set_text( String::num( aabb.pos.y ) );
			value_editor[2]->set_text( String::num( aabb.pos.z ) );
			value_editor[3]->set_text( String::num( aabb.size.x ) );
			value_editor[4]->set_text( String::num( aabb.size.y ) );
			value_editor[5]->set_text( String::num( aabb.size.z ) );
			
		} break;
		case Variant::MATRIX32: {

			List<String> names;
			names.push_back("xx");
			names.push_back("xy");
			names.push_back("yx");
			names.push_back("yy");
			names.push_back("ox");
			names.push_back("oy");
			config_value_editors(6,2,16,names);

			Matrix32 basis=v;
			for(int i=0;i<6;i++) {

				value_editor[i]->set_text( String::num( basis.elements[i/2][i%2] ) );
			}

		} break;
		case Variant::MATRIX3: {
			
			List<String> names;
			names.push_back("xx");
			names.push_back("xy");
			names.push_back("xz");
			names.push_back("yx");
			names.push_back("yy");
			names.push_back("yz");
			names.push_back("zx");
			names.push_back("zy");
			names.push_back("zz");
			config_value_editors(9,3,16,names);
			
			Matrix3 basis=v;
			for(int i=0;i<9;i++) {
				
				value_editor[i]->set_text( String::num( basis.elements[i/3][i%3] ) );
			}
			
		} break;
		case Variant::TRANSFORM: {
			
			
			List<String> names;
			names.push_back("xx");
			names.push_back("xy");
			names.push_back("xz");
			names.push_back("xo");
			names.push_back("yx");
			names.push_back("yy");
			names.push_back("yz");
			names.push_back("yo");
			names.push_back("zx");
			names.push_back("zy");
			names.push_back("zz");
			names.push_back("zo");
			config_value_editors(12,4,16,names);
			
			Transform tr=v;
			for(int i=0;i<9;i++) {
				
				value_editor[(i/3)*4+i%3]->set_text( String::num( tr.basis.elements[i/3][i%3] ) );
			}
			
			value_editor[3]->set_text( String::num( tr.origin.x ) );
			value_editor[7]->set_text( String::num( tr.origin.y ) );
			value_editor[11]->set_text( String::num( tr.origin.z ) );
			
		} break;
		case Variant::COLOR: {
			

			color_picker->show();
			color_picker->set_edit_alpha(hint!=PROPERTY_HINT_COLOR_NO_ALPHA);
			color_picker->set_color(v);
			set_size( Size2(350, color_picker->get_combined_minimum_size().height+10));
			/*
			int ofs=80;
			int m=10;
			int h=20;
			Color c=v;
			float values[4]={c.r,c.g,c.b,c.a};
			for (int i=0;i<4;i++) {
				int y=m+i*h;
				
				value_editor[i]->show();
				value_label[i]->show();				
				value_label[i]->set_pos(Point2(ofs,y));
				scroll[i]->set_min(0);
				scroll[i]->set_max(1.0);
				scroll[i]->set_page(0);		
				scroll[i]->set_pos(Point2(ofs+15,y+Math::floor((h-scroll[i]->get_minimum_size().height)/2.0)));
				scroll[i]->set_val(values[i]);
				scroll[i]->set_size(Size2(120,1));
				scroll[i]->show();
				value_editor[i]->set_pos(Point2(ofs+140,y));
				value_editor[i]->set_size(Size2(40,h));
				value_editor[i]->set_text( String::num(values[i],2 ));
				
			}
			
			value_label[0]->set_text("R");
			value_label[1]->set_text("G");
			value_label[2]->set_text("B");
			value_label[3]->set_text("A");
			
			Size2 new_size = value_editor[3]->get_pos() + value_editor[3]->get_size() + Point2(10,10);
			set_size( new_size );
			*/
			
		} break;
		case Variant::IMAGE: {
			
			List<String> names;
			names.push_back("New");
			names.push_back("Load");
			names.push_back("Clear");
			config_action_buttons(names);
			
		} break;
		case Variant::NODE_PATH: {
			
			List<String> names;
			names.push_back("Assign");
			names.push_back("Clear");
			config_action_buttons(names);

		} break;
		case Variant::OBJECT: {
			
			if (hint!=PROPERTY_HINT_RESOURCE_TYPE)
				break;


			menu->clear();
			menu->set_size(Size2(1,1));


			if (hint_text!="") {
				int idx=0;

				for(int i=0;i<hint_text.get_slice_count(",");i++) {



					String base=hint_text.get_slice(",",i);

					Set<String> valid_inheritors;
					valid_inheritors.insert(base);
					List<String> inheritors;
					ObjectTypeDB::get_inheriters_from(base.strip_edges(),&inheritors);
					List<String>::Element *E=inheritors.front();
					while(E) {
						valid_inheritors.insert(E->get());
						E=E->next();
					}

					for(Set<String>::Element *E=valid_inheritors.front();E;E=E->next()) {
						String t = E->get();
						if (!ObjectTypeDB::can_instance(t))
							continue;
						inheritors_array.push_back(t);

						int id = TYPE_BASE_ID+idx;
						if (has_icon(t,"EditorIcons")) {

							menu->add_icon_item(get_icon(t,"EditorIcons"),"New "+t,id);
						} else {

							menu->add_item("New "+t,id);
						}

						idx++;
					}


				}

				if (menu->get_item_count())
					menu->add_separator();
			}

			menu->add_icon_item(get_icon("Load","EditorIcons"),"Load",OBJ_MENU_LOAD);

			if (!RES(v).is_null()) {

				menu->add_icon_item(get_icon("EditResource","EditorIcons"),"Edit",OBJ_MENU_EDIT);
				menu->add_icon_item(get_icon("Del","EditorIcons"),"Clear",OBJ_MENU_CLEAR);
				menu->add_icon_item(get_icon("Duplicate","EditorIcons"),"Make Unique",OBJ_MENU_MAKE_UNIQUE);
				RES r = v;
				if (r.is_valid() && r->get_path().is_resource_file() && r->get_import_metadata().is_valid()) {
					menu->add_separator();
					menu->add_icon_item(get_icon("Reload","EditorIcons"),"Re-Import",OBJ_MENU_REIMPORT);
				}
			}


			RES cb=EditorSettings::get_singleton()->get_resource_clipboard();
			bool paste_valid=cb.is_valid() && (hint_text=="" || ObjectTypeDB::is_type(cb->get_type(),hint_text));

			if (!RES(v).is_null() || paste_valid) {
				menu->add_separator();


				if (!RES(v).is_null()) {

					menu->add_item("Copy",OBJ_MENU_COPY);
				}

				if (paste_valid) {

					menu->add_item("Paste",OBJ_MENU_PASTE);
				}
			}



			menu->set_pos(get_pos());
			menu->popup();
			hide();
			updating=false;
			return false;


		} break;
		case Variant::INPUT_EVENT: {
			
			
		} break;
		case Variant::DICTIONARY: {
			
			
		} break;
		case Variant::RAW_ARRAY: {
			
			
		} break;
		case Variant::INT_ARRAY: {
			
			
		} break;
		case Variant::REAL_ARRAY: {
			
			
		} break;
		case Variant::STRING_ARRAY: {
			
			
		} break;
		case Variant::VECTOR3_ARRAY: {
			
			
		} break;
		case Variant::COLOR_ARRAY: {
			
			
		} break;
		default: {}
	}		
	
	updating=false;
	return true;
}

void CustomPropertyEditor::_file_selected(String p_file) {

	switch(type) {
		
		case Variant::STRING: {

			if (hint==PROPERTY_HINT_FILE || hint==PROPERTY_HINT_DIR) {

				v=Globals::get_singleton()->localize_path(p_file);
				emit_signal("variant_changed");
				hide();
			}

			if (hint==PROPERTY_HINT_GLOBAL_FILE || hint==PROPERTY_HINT_GLOBAL_DIR) {

				v=p_file;
				emit_signal("variant_changed");
				hide();
			}

		} break;
		case Variant::OBJECT: {

			String type=(hint==PROPERTY_HINT_RESOURCE_TYPE)?hint_text:String();

			RES res = ResourceLoader::load(p_file,type);
			if (res.is_null()) {
				error->set_text("Error loading file: Not a resource!");
				error->popup_centered(Size2(300,80));
				break;
			}
			v=res.get_ref_ptr();
			emit_signal("variant_changed");
			hide();
		} break;
		case Variant::IMAGE: {

			Image image;
			Error err = ImageLoader::load_image(p_file,&image);
			ERR_EXPLAIN("Couldn't load image");
			ERR_FAIL_COND(err);
			v=image;
			emit_signal("variant_changed");
			hide();
		} break;
		default: {}
	}
}

void CustomPropertyEditor::_type_create_selected(int p_idx) {


	if (type==Variant::INT || type==Variant::REAL) {


		float newval=0;
		switch(p_idx) {

			case EASING_LINEAR: {

				newval=1;
			} break;
			case EASING_EASE_IN: {

				newval=2.0;
			} break;
			case EASING_EASE_OUT: {
				newval=0.5;
			} break;
			case EASING_ZERO: {

				newval=0;
			} break;
			case EASING_IN_OUT: {

				newval=-0.5;
			} break;
			case EASING_OUT_IN: {
				newval=-2.0;
			} break;
		}

		v=newval;
		emit_signal("variant_changed");
		easing_draw->update();

	} else if (type==Variant::OBJECT) {

		ERR_FAIL_INDEX(p_idx,inheritors_array.size());

		//List<String> inheritors;
		//ObjectTypeDB::get_inheriters_from(hint_text,&inheritors);
		//inheritors.push_front(hint_text);

		//ERR_FAIL_INDEX( p_idx, inheritors.size() );
		String intype=inheritors_array[p_idx];

		Object *obj = ObjectTypeDB::instance(intype);

		ERR_FAIL_COND( !obj );


		Resource *res=obj->cast_to<Resource>();
		ERR_FAIL_COND( !res );

		v=Ref<Resource>(res).get_ref_ptr();
		emit_signal("variant_changed");
		hide();
	}

}


void CustomPropertyEditor::_color_changed(const Color& p_color) {

	v=p_color;
	emit_signal("variant_changed");

}

void CustomPropertyEditor::_node_path_selected(NodePath p_path) {

	if (owner && owner->is_type("Node")) {

		Node *node = owner->cast_to<Node>();
		Node *tonode=node->get_node(p_path);
		if (tonode) {

			p_path=node->get_path_to(tonode);
		}

	}

	v=p_path;
	emit_signal("variant_changed");

}

void CustomPropertyEditor::_action_pressed(int p_which) {


	if (updating) 
		return;
	
	switch(type) {
		case Variant::INT: {

			if (hint==PROPERTY_HINT_ALL_FLAGS) {

				uint32_t f = v;
				if (checks20[p_which]->is_pressed())
					f|=(1<<p_which);
				else
					f&=~(1<<p_which);

				v=f;
				emit_signal("variant_changed");
			}

		} break;
		case Variant::STRING: {

			if (hint==PROPERTY_HINT_MULTILINE_TEXT) {

				hide();

			} else if (hint==PROPERTY_HINT_FILE || hint==PROPERTY_HINT_GLOBAL_FILE) {
				if (p_which==0) {

					if (hint==PROPERTY_HINT_FILE)
						file->set_access(FileDialog::ACCESS_RESOURCES);
					else
						file->set_access(FileDialog::ACCESS_FILESYSTEM);

					file->set_mode(FileDialog::MODE_OPEN_FILE);
					file->clear_filters();

					file->clear_filters();

					if (hint_text!="") {
						Vector<String> extensions=hint_text.split(",");
						for(int i=0;i<extensions.size();i++) {

							String filter = extensions[i];
							if (filter.begins_with("."))
								filter="*"+extensions[i];
							else if (!filter.begins_with("*"))
								filter="*."+extensions[i];


							file->add_filter(filter+" ; "+extensions[i].to_upper() );

						}
					}
					file->popup_centered_ratio();
				} else {

					v="";
					emit_signal("variant_changed");
					hide();

				}

			} else if (hint==PROPERTY_HINT_DIR  || hint==PROPERTY_HINT_GLOBAL_DIR) {

				if (p_which==0) {

					if (hint==PROPERTY_HINT_DIR)
						file->set_access(FileDialog::ACCESS_RESOURCES);
					else
						file->set_access(FileDialog::ACCESS_FILESYSTEM);
					file->set_mode(FileDialog::MODE_OPEN_DIR);
					file->clear_filters();
					file->popup_centered_ratio();
				} else {

					v="";
					emit_signal("variant_changed");
					hide();

				}

			}

		} break;
		case Variant::NODE_PATH: {
				
			if (p_which==0) {


				scene_tree->popup_centered_ratio();

			} else if (p_which==1) {


				v=NodePath();
				emit_signal("variant_changed");
			}				
		} break;
		case Variant::OBJECT: {
		
			if (p_which==0) {
			

				ERR_FAIL_COND( inheritors_array.empty() );

				String intype=inheritors_array[0];

				
				if (hint==PROPERTY_HINT_RESOURCE_TYPE) {
				
					Object *obj = ObjectTypeDB::instance(intype);
					ERR_BREAK( !obj );
					Resource *res=obj->cast_to<Resource>();
					ERR_BREAK( !res );
					
					v=Ref<Resource>(res).get_ref_ptr();
					emit_signal("variant_changed");
					hide();
						
				}
			} else if (p_which==1) {
			
				file->set_access(FileDialog::ACCESS_RESOURCES);
				file->set_mode(FileDialog::MODE_OPEN_FILE);
				List<String> extensions;
				String type=(hint==PROPERTY_HINT_RESOURCE_TYPE)?hint_text:String();
				
				ResourceLoader::get_recognized_extensions_for_type(type,&extensions);
				file->clear_filters();
				for (List<String>::Element *E=extensions.front();E;E=E->next()) {
				
					file->add_filter("*."+E->get()+" ; "+E->get().to_upper() );
				
				}
				
				file->popup_centered_ratio();
			
			} else if (p_which==2) {
			
				RefPtr RefPtr=v;

				if (!RefPtr.is_null()) {

					emit_signal("resource_edit_request");
					hide();	
				}
			
			} else if (p_which==3) {
			
			
				v=Variant();
				emit_signal("variant_changed");
				hide();
			} else if (p_which==4) {


				RefPtr RefPtr=v;
				Ref<Resource> res_orig = RefPtr;
				if (res_orig.is_null())
					return;

				List<PropertyInfo> property_list;
				res_orig->get_property_list(&property_list);
				List< Pair<String,Variant> > propvalues;

				for(List<PropertyInfo>::Element *E=property_list.front();E;E=E->next()) {

					Pair<String,Variant> p;
					PropertyInfo &pi = E->get();
					if (pi.usage&PROPERTY_USAGE_STORAGE) {

						p.first=pi.name;
						p.second=res_orig->get(pi.name);
					}

					propvalues.push_back(p);
				}

				Ref<Resource> res = Ref<Resource>( ObjectTypeDB::instance( res_orig->get_type() ));

				ERR_FAIL_COND(res.is_null());

				for(List< Pair<String,Variant> >::Element *E=propvalues.front();E;E=E->next()) {

					Pair<String,Variant> &p=E->get();
					res->set(p.first,p.second);
				}

				v=res.get_ref_ptr();
				emit_signal("variant_changed");
				hide();
			}
				
		} break;
		case Variant::IMAGE: {

			if (p_which==0) {
				//new image too difficult
				ERR_PRINT("New Image Unimplemented");

			} else if (p_which==1) {

				file->set_access(FileDialog::ACCESS_RESOURCES);
				file->set_mode(FileDialog::MODE_OPEN_FILE);
				List<String> extensions;
				ImageLoader::get_recognized_extensions(&extensions);

				file->clear_filters();

				for (List<String>::Element *E=extensions.front();E;E=E->next()) {

					file->add_filter("*."+E->get()+" ; "+E->get().to_upper() );

				}

				file->popup_centered_ratio();

			} else if (p_which==2) {

				v=Image();
				emit_signal("variant_changed");
				hide();
			}

		} break;
		default: {};
	}

}

void CustomPropertyEditor::_scroll_modified(double p_value) {
	
	if (updating) 
		return;
	/*
	switch(type) {
		
		case Variant::COLOR: {
			
			for (int i=0;i<4;i++) {
				
				value_editor[i]->set_text( String::num(scroll[i]->get_val(),2) );
			}
			Color c;
			c.r=scroll[0]->get_val();
			c.g=scroll[1]->get_val();
			c.b=scroll[2]->get_val();
			c.a=scroll[3]->get_val();
			v=c;
			update();
			emit_signal("variant_changed");
		} break;
		default: {}
	}
	*/
}


void CustomPropertyEditor::_drag_easing(const InputEvent& p_ev) {


	if (p_ev.type==InputEvent::MOUSE_MOTION && p_ev.mouse_motion.button_mask&BUTTON_MASK_LEFT) {

		float rel = p_ev.mouse_motion.relative_x;
		if (rel==0)
		    return;

		bool flip=hint_text=="attenuation";

		if (flip)
			rel=-rel;

		float val = v;
		if (val==0)
			return;
		bool sg = val < 0;
		val = Math::absf(val);

		val = Math::log(val)/Math::log(2);
		//logspace
		val+=rel*0.05;
		//

		val = Math::pow(2,val);
		if (sg)
			val=-val;

		v=val;
		easing_draw->update();
		//emit_signal("variant_changed");
		emit_signal("variant_changed");
	}
	if (p_ev.type==InputEvent::MOUSE_BUTTON && p_ev.mouse_button.button_index==BUTTON_LEFT) {


	}

}


void CustomPropertyEditor::_draw_easing() {


	RID ci = easing_draw->get_canvas_item();

	Size2 s = easing_draw->get_size();
	Rect2 r(Point2(),s);
	r=r.grow(3);
	get_stylebox("normal","LineEdit")->draw(ci,r);
	//VisualServer::get_singleton()->canvas_item_add


	int points = 48;

	float prev=1.0;
	float exp=v;
	bool flip=hint_text=="attenuation";

	Ref<Font> f = get_font("font","Label");
	Color color = get_color("font_color","Label");

	for(int i=1;i<=points;i++) {

		float ifl = i/float(points);
		float iflp = (i-1)/float(points);

		float h = 1.0-Math::ease(ifl,exp);

		if (flip) {
			ifl=1.0-ifl;
			iflp=1.0-iflp;
		}

		VisualServer::get_singleton()->canvas_item_add_line(ci,Point2(iflp*s.width,prev*s.height),Point2(ifl*s.width,h*s.height),color);
		prev=h;
	}

	f->draw(ci,Point2(10,10+f->get_ascent()),String::num(exp,2),color);
}

void CustomPropertyEditor::_text_edit_changed() {

	v=text_edit->get_text();
	emit_signal("variant_changed");

}

void CustomPropertyEditor::_modified(String p_string) {
	
	if (updating)
		return;
	updating=true;
	switch(type) {
		case Variant::REAL: {

			if (hint!=PROPERTY_HINT_EXP_EASING) {
				v=value_editor[0]->get_text().to_double();
				emit_signal("variant_changed");

			}

		} break;
		case Variant::STRING: {

			v=value_editor[0]->get_text();
			emit_signal("variant_changed");
		} break;
		case Variant::VECTOR2: {

			Vector2 vec;
			vec.x=value_editor[0]->get_text().to_double();
			vec.y=value_editor[1]->get_text().to_double();
			v=vec;
			emit_signal("variant_changed");

		} break;
		case Variant::RECT2: {

			Rect2 r2;
			r2.pos.x=value_editor[0]->get_text().to_double();
			r2.pos.y=value_editor[1]->get_text().to_double();
			r2.size.x=value_editor[2]->get_text().to_double();
			r2.size.y=value_editor[3]->get_text().to_double();
			v=r2;
			emit_signal("variant_changed");

		} break;

		case Variant::VECTOR3: {
			
			Vector3 vec;
			vec.x=value_editor[0]->get_text().to_double();
			vec.y=value_editor[1]->get_text().to_double();
			vec.z=value_editor[2]->get_text().to_double();
			v=vec;
			emit_signal("variant_changed");
			
		} break;
		case Variant::PLANE: {
			
			Plane pl;
			pl.normal.x=value_editor[0]->get_text().to_double();
			pl.normal.y=value_editor[1]->get_text().to_double();
			pl.normal.z=value_editor[2]->get_text().to_double();
			pl.d=value_editor[3]->get_text().to_double();
			v=pl;
			emit_signal("variant_changed");
			
		} break;
		case Variant::QUAT: {
			
			Quat q;
			q.x=value_editor[0]->get_text().to_double();
			q.y=value_editor[1]->get_text().to_double();
			q.z=value_editor[2]->get_text().to_double();
			q.w=value_editor[3]->get_text().to_double();
			v=q;
			emit_signal("variant_changed");
			
		} break;
		case Variant::_AABB: {
			
			Vector3 pos;
			pos.x=value_editor[0]->get_text().to_double();
			pos.y=value_editor[1]->get_text().to_double();
			pos.z=value_editor[2]->get_text().to_double();
			Vector3 size;
			size.x=value_editor[3]->get_text().to_double();
			size.y=value_editor[4]->get_text().to_double();
			size.z=value_editor[5]->get_text().to_double();
			
			v=AABB(pos,size);
			emit_signal("variant_changed");
			
		} break;
		case Variant::MATRIX32: {

			Matrix3 m;
			for(int i=0;i<6;i++) {

				m.elements[i/2][i%2]=value_editor[i]->get_text().to_double();
			}

			v=m;
			emit_signal("variant_changed");

		} break;
		case Variant::MATRIX3: {
			
			Matrix3 m;
			for(int i=0;i<9;i++) {
					
				m.elements[i/3][i%3]=value_editor[i]->get_text().to_double();
			}
			
			v=m;
			emit_signal("variant_changed");
			
		} break;
		case Variant::TRANSFORM: {
			
			Matrix3 basis;
			for(int i=0;i<9;i++) {
				
				basis.elements[i/3][i%3]=value_editor[(i/3)*4+i%3]->get_text().to_double();
			}
			
			Vector3 origin;
			origin.x=value_editor[3]->get_text().to_double();
			origin.y=value_editor[7]->get_text().to_double();
			origin.z=value_editor[11]->get_text().to_double();
			
			v=Transform(basis,origin);
			emit_signal("variant_changed");
			
			
		} break;
		case Variant::COLOR: {
			/*
			for (int i=0;i<4;i++) {
				
				scroll[i]->set_val( value_editor[i]->get_text().to_double() );
			}
			Color c;
			c.r=value_editor[0]->get_text().to_double();
			c.g=value_editor[1]->get_text().to_double();
			c.b=value_editor[2]->get_text().to_double();
			c.a=value_editor[3]->get_text().to_double();
			v=c;
			update();
			emit_signal("variant_changed");
			*/
		} break;
		case Variant::IMAGE: {
			
			
		} break;
		case Variant::NODE_PATH: {
			
			
		} break;		
		case Variant::INPUT_EVENT: {
			
			
		} break;
		case Variant::DICTIONARY: {
			
			
		} break;
		case Variant::RAW_ARRAY: {
			
			
		} break;
		case Variant::INT_ARRAY: {
			
			
		} break;
		case Variant::REAL_ARRAY: {
			
			
		} break;
		case Variant::STRING_ARRAY: {
			
			
		} break;
		case Variant::VECTOR3_ARRAY: {
			
			
		} break;
		case Variant::COLOR_ARRAY: {
			
			
		} break;
		default: {}
	}		
	
	updating=false;
}

void CustomPropertyEditor::_focus_enter() {
	switch(type) {
		case Variant::REAL:
		case Variant::STRING:
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::VECTOR3:
		case Variant::PLANE:
		case Variant::QUAT:
		case Variant::_AABB:
		case Variant::MATRIX32:
		case Variant::MATRIX3:
		case Variant::TRANSFORM: {
			for (int i=0;i<MAX_VALUE_EDITORS;++i) {
				if (value_editor[i]->has_focus()) {
					value_editor[i]->select_all();
					break;
				}
			}
		} break;
		default: {}
	}

}

void CustomPropertyEditor::_focus_exit() {
	switch(type) {
		case Variant::REAL:
		case Variant::STRING:
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::VECTOR3:
		case Variant::PLANE:
		case Variant::QUAT:
		case Variant::_AABB:
		case Variant::MATRIX32:
		case Variant::MATRIX3:
		case Variant::TRANSFORM: {
			for (int i=0;i<MAX_VALUE_EDITORS;++i) {
				value_editor[i]->select(0, 0);
			}
		} break;
		default: {}
	}

}

void CustomPropertyEditor::config_action_buttons(const List<String>& p_strings) {

	int w=100;
	int h=18;
	int m=5;

	set_size( Size2( w, m*2+(h+m)*p_strings.size() ) );

	for (int i=0;i<MAX_ACTION_BUTTONS;i++) {
				
		if (i<p_strings.size()) {
			action_buttons[i]->show();
			action_buttons[i]->set_text(p_strings[i]);			
			action_buttons[i]->set_pos( Point2( m, m+i*(h+m) ));
			action_buttons[i]->set_size( Size2( w-m*2, h ) );
			action_buttons[i]->set_flat(true);
		} else {
			action_buttons[i]->hide();
		}
	}
	
	
}

void CustomPropertyEditor::config_value_editors(int p_amount, int p_columns,int p_label_w,const List<String>& p_strings) {
	
	int w=80;
	int h=20;
	int m=10;
	
	int rows=((p_amount-1)/p_columns)+1;
	
	set_size( Size2( m*(1+p_columns)+(w+p_label_w)*p_columns, m*(1+rows)+h*rows ) );
		
	for (int i=0;i<MAX_VALUE_EDITORS;i++) {
		
		int c=i%p_columns;
		int r=i/p_columns;
		
		if (i<p_amount) {
			value_editor[i]->show();
			value_label[i]->show();
			value_label[i]->set_text(i<p_strings.size()?p_strings[i]:String(""));			
			value_editor[i]->set_pos( Point2( m+p_label_w+c*(w+m+p_label_w), m+r*(h+m) ));
			value_editor[i]->set_size( Size2( w, h ) );
			value_label[i]->set_pos( Point2( m+c*(w+m+p_label_w), m+r*(h+m) ) );
			value_editor[i]->set_editable(!read_only);
		} else {
			value_editor[i]->hide();
			value_label[i]->hide();
		}
	}
	
	
	
}

void CustomPropertyEditor::_bind_methods() {
	
	ObjectTypeDB::bind_method("_focus_enter", &CustomPropertyEditor::_focus_enter);
	ObjectTypeDB::bind_method("_focus_exit", &CustomPropertyEditor::_focus_exit);
	ObjectTypeDB::bind_method("_modified",&CustomPropertyEditor::_modified);
	ObjectTypeDB::bind_method("_scroll_modified",&CustomPropertyEditor::_scroll_modified);
	ObjectTypeDB::bind_method("_action_pressed",&CustomPropertyEditor::_action_pressed);
	ObjectTypeDB::bind_method("_file_selected",&CustomPropertyEditor::_file_selected);
	ObjectTypeDB::bind_method("_type_create_selected",&CustomPropertyEditor::_type_create_selected);
	ObjectTypeDB::bind_method("_node_path_selected",&CustomPropertyEditor::_node_path_selected);
	ObjectTypeDB::bind_method("_color_changed",&CustomPropertyEditor::_color_changed);
	ObjectTypeDB::bind_method("_draw_easing",&CustomPropertyEditor::_draw_easing);
	ObjectTypeDB::bind_method("_drag_easing",&CustomPropertyEditor::_drag_easing);
	ObjectTypeDB::bind_method( "_text_edit_changed",&CustomPropertyEditor::_text_edit_changed);
	ObjectTypeDB::bind_method( "_menu_option",&CustomPropertyEditor::_menu_option);


	ADD_SIGNAL( MethodInfo("variant_changed") );
	ADD_SIGNAL( MethodInfo("resource_edit_request") );
}
CustomPropertyEditor::CustomPropertyEditor() {
	
	
	read_only=false;
	updating=false;
	
	for (int i=0;i<MAX_VALUE_EDITORS;i++) {
		
		value_editor[i]=memnew( LineEdit );
		add_child( value_editor[i] );
		value_label[i]=memnew( Label );
		add_child( value_label[i] );
		value_editor[i]->hide();
		value_label[i]->hide();
		value_editor[i]->connect("text_entered", this,"_modified");
		value_editor[i]->connect("focus_enter", this, "_focus_enter");
		value_editor[i]->connect("focus_exit", this, "_focus_exit");
	}
	
	for(int i=0;i<4;i++) {
	
		scroll[i] = memnew( HScrollBar );
		scroll[i]->hide();
		scroll[i]->set_min(0);
		scroll[i]->set_max(1.0);
		scroll[i]->set_step(0.01);		
		add_child(scroll[i]);
		scroll[i]->connect("value_changed", this,"_scroll_modified");
		
	}

	for(int i=0;i<20;i++) {
		checks20[i]=memnew( Button );
		checks20[i]->set_toggle_mode(true);
		checks20[i]->set_focus_mode(FOCUS_NONE);
		add_child(checks20[i]);
		checks20[i]->hide();
		checks20[i]->connect("pressed",this,"_action_pressed",make_binds(i));
		checks20[i]->set_tooltip("Bit "+itos(i)+", val "+itos(1<<i)+".");
	}

	text_edit = memnew( TextEdit );
	add_child(text_edit);
	text_edit->set_area_as_parent_rect();
	for(int i=0;i<4;i++)
		text_edit->set_margin((Margin)i,5);
	text_edit->set_margin(MARGIN_BOTTOM,30);

	text_edit->hide();
	text_edit->connect("text_changed",this,"_text_edit_changed");

	for (int i=0;i<MAX_ACTION_BUTTONS;i++) {
	
		action_buttons[i]=memnew(Button);
		action_buttons[i]->hide();	
		add_child(action_buttons[i]);
		Vector<Variant> binds;
		binds.push_back(i);
		action_buttons[i]->connect("pressed", this,"_action_pressed",binds);
	}
	
	color_picker = memnew( ColorPicker );
	add_child(color_picker);
	color_picker->hide();
	color_picker->set_area_as_parent_rect();
	for(int i=0;i<4;i++)
		color_picker->set_margin((Margin)i,5);
	color_picker->connect("color_changed",this,"_color_changed");

	set_as_toplevel(true);
	file = memnew ( FileDialog );
	add_child(file);
	file->hide();
	
	file->connect("file_selected", this,"_file_selected");
	file->connect("dir_selected", this,"_file_selected");

	error = memnew( ConfirmationDialog );
	error->set_title("Error!");
	add_child(error);
	//error->get_cancel()->hide();
	
	type_button = memnew( MenuButton );
	add_child(type_button);
	type_button->hide();
	type_button->get_popup()->connect("item_pressed", this,"_type_create_selected");
	

	scene_tree = memnew( SceneTreeDialog );
	add_child(scene_tree);
	scene_tree->connect("selected", this,"_node_path_selected");
	scene_tree->get_tree()->set_show_enabled_subscene(true);

	texture_preview = memnew( TextureFrame );
	add_child( texture_preview);
	texture_preview->hide();

	easing_draw=memnew( Control );
	add_child(easing_draw);
	easing_draw->hide();
	easing_draw->connect("draw",this,"_draw_easing");
	easing_draw->connect("input_event",this,"_drag_easing");
	//easing_draw->emit_signal(SceneStringNames::get_singleton()->input_event,InputEvent());
	easing_draw->set_default_cursor_shape(Control::CURSOR_MOVE);

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->connect("item_pressed",this,"_menu_option");
}



Node *PropertyEditor::get_instanced_node() {

	//this sucks badly
	if (!obj)
		return NULL;

	Node *node = obj->cast_to<Node>();
	if (!node)
		return NULL;

	if (node->get_filename()=="")
		return NULL;

	if (!node->get_owner())
		return NULL; //scene root i guess

	return node;
}

TreeItem *PropertyEditor::find_item(TreeItem *p_item,const String& p_name) {
	
	
	if (!p_item)
		return NULL;
	
	String name = p_item->get_metadata(1);
	
	if (name==p_name) {
		
		return p_item;
	}
	
	TreeItem *c=p_item->get_children();
	
	while (c) {
		
		TreeItem *found = find_item(c,p_name);
		if (found)
			return found;
		c=c->get_next();
	}
	
	return NULL;
}


void PropertyEditor::_changed_callback(Object *p_changed,const char * p_prop) {


	_changed_callbacks(p_changed,p_prop);
}

void PropertyEditor::_changed_callbacks(Object *p_changed,const String& p_prop) {


	if (p_changed!=obj)
		return;

	if (changing)
		return;

	if (p_prop==String())
		update_tree_pending=true;
	else {
		
		pending[p_prop]=p_prop;
			
	}
}

void PropertyEditor::update_property(const String& p_prop) {

	if (obj)
		_changed_callbacks(obj,p_prop);
}


void PropertyEditor::set_item_text(TreeItem *p_item, int p_type, const String& p_name, int p_hint, const String& p_hint_text) {
	
	switch( p_type ) {

		case Variant::BOOL: {

			p_item->set_checked( 1, obj->get( p_name ) );
		} break;
		case Variant::REAL:
		case Variant::INT: {

			if (p_hint==PROPERTY_HINT_ALL_FLAGS) {
				tree->update();
				break;
			}

			if (p_hint==PROPERTY_HINT_FLAGS) {
				Vector<String> values = p_hint_text.split(",");
				String flags;
				int val = obj->get(p_name);
				for(int i=0;i<values.size();i++) {

					String v = values[i];
					if (v=="")
						continue;
					if (!(val&(1<<i)))
						continue;

					if (flags!="")
						flags+=", ";
					flags+=v;
				}
				p_item->set_text(1, flags );
				break;
			}

			if (p_hint==PROPERTY_HINT_EXP_EASING) {

				p_item->set_text(1, String::num(obj->get( p_name ),2) );
				break;
			}


			//p_item->set_cell_mode( 1, TreeItem::CELL_MODE_RANGE );

			if (p_type==Variant::REAL) {
				p_item->set_range(1, obj->get( p_name ) );

			} else {
				p_item->set_range(1, obj->get( p_name ) );
			}


			p_item->set_editable(1,!read_only);


		} break;
		case Variant::STRING:

			if (p_hint==PROPERTY_HINT_ENUM) {

				Vector<String> strings = p_hint_text.split(",");
				String current = obj->get(p_name);
				int idx=0;
				for(int x=0;x<strings.size();x++) {
					if (strings[x]==current) {
						idx=x;
						break;
					}
				}
				p_item->set_text(1, p_hint_text);
				p_item->set_range(1,idx);
				break;
			}

		case Variant::VECTOR3:
		case Variant::QUAT:
		case Variant::VECTOR2:
		case Variant::_AABB:
		case Variant::RECT2:
		case Variant::MATRIX32:
		case Variant::MATRIX3:
		case Variant::TRANSFORM: {

			p_item->set_text(1,obj->get(p_name));

		} break;
		case Variant::COLOR: {

			p_item->set_custom_bg_color(1,obj->get(p_name));
			//p_item->set_text(1,obj->get(p_name));

		} break;
		case Variant::IMAGE: {
		
			Image img = obj->get( p_name );
			if (img.empty())
				p_item->set_text(1,"[Image (empty)]");
			else
				p_item->set_text(1,"[Image "+itos(img.get_width())+"x"+itos(img.get_height())+"]");

		} break;
		case Variant::NODE_PATH: {

			p_item->set_text(1,obj->get(p_name));
		} break;
		case Variant::OBJECT: {


			if (obj->get( p_name ).get_type() == Variant::NIL || obj->get( p_name ).operator RefPtr().is_null()) {
				p_item->set_text(1,"<null>");

				Dictionary d = p_item->get_metadata(0);
				int hint=d.has("hint")?d["hint"].operator int():-1;
				String hint_text=d.has("hint_text")?d["hint_text"]:"";
				if (hint==PROPERTY_HINT_RESOURCE_TYPE && hint_text == "Texture") {
					p_item->set_icon(1,NULL);
				}

			} else {
				RES res = obj->get( p_name ).operator RefPtr();
				if (res->is_type("Texture")) {
					int tw = EditorSettings::get_singleton()->get("property_editor/texture_preview_width");
					p_item->set_icon_max_width(1,tw);
					p_item->set_icon(1,res);
					p_item->set_text(1,"");

				} else if (res->get_name() != "") {

					p_item->set_text(1, res->get_name());
				} else if (res->get_path()!="" && !res->get_path().begins_with("local://")) {
					p_item->set_text(1, res->get_path().get_file());
				} else {
					p_item->set_text(1,"<"+res->get_type()+">");
				};
			}

		} break;
		default: {};
	}
	
}

void PropertyEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		get_tree()->connect("node_removed",this,"_node_removed");
	}
	if (p_what==NOTIFICATION_EXIT_TREE) {

		edit(NULL);
	}

	
	if (p_what==NOTIFICATION_FIXED_PROCESS) {
		
		
		if (refresh_countdown>0) {
			refresh_countdown-=get_fixed_process_delta_time();
			if (refresh_countdown<=0) {
				TreeItem *root = tree->get_root();
				_refresh_item(root);
			}
		}

		changing=true;
		 
		if (update_tree_pending) {

			update_tree();
			update_tree_pending=false;
			
		} else {
			
			const String *k=NULL;	
			while ((k=pending.next(k))) {
				
				TreeItem * item = find_item(tree->get_root(),*k);
				if (!item)
					continue;
				
				if (get_instanced_node()) {

					Dictionary d = get_instanced_node()->get_instance_state();
					if (d.has(*k)) {
						Variant v = obj->get(*k);
						Variant vorig = d[*k];

						int found=-1;
						for(int i=0;i<item->get_button_count(1);i++) {

							if (item->get_button_id(1,i)==3) {
								found=i;
								break;
							}
						}

						bool changed = ! (v==vorig);

						if ((found!=-1)!=changed) {

							if (changed) {

								item->add_button(1,get_icon("Reload","EditorIcons"),3);
							} else {

								item->erase_button(1,found);
							}

						}

					}

				}
				Dictionary d=item->get_metadata(0);	
				set_item_text(item,d["type"],d["name"],d["hint"],d["hint_text"]);
			}				
		}
			
		pending.clear();
			
		changing=false;
			
	}
}

TreeItem *PropertyEditor::get_parent_node(String p_path,HashMap<String,TreeItem*>& item_paths,TreeItem *root) {

	TreeItem *item=NULL;

	if (p_path=="") {

		item = root;
	} else if (item_paths.has(p_path)) {

		item = item_paths.get(p_path);
	} else {

//		printf("path %s parent path %s - item name %s\n",p_path.ascii().get_data(),p_path.left( p_path.find_last("/") ).ascii().get_data(),p_path.right( p_path.find_last("/") ).ascii().get_data() );
		TreeItem *parent = get_parent_node( p_path.left( p_path.find_last("/") ),item_paths,root );
		item = tree->create_item( parent );

		String name = (p_path.find("/")!=-1)?p_path.right( p_path.find_last("/")+1 ):p_path;
		if (capitalize_paths)
			item->set_text(0, name.capitalize() );
		else
			item->set_text(0, name );
		item->set_tooltip(0, p_path);
		if (item->get_parent()!=root) {
			item->set_icon( 0, get_icon("Folder","EditorIcons"));
		}

		item->set_editable(0,false);
		item->set_selectable(0,false);
		item->set_editable(1,false);
		item->set_selectable(1,false);

		if (item->get_parent()==root) {

			item->set_custom_bg_color(0,get_color("prop_subsection","Editor"));
			item->set_custom_bg_color(1,get_color("prop_subsection","Editor"));
		}

		item_paths[p_path]=item;
	}

	return item;

}


void PropertyEditor::_refresh_item(TreeItem *p_item) {

	if (!p_item)
		return;

	String name = p_item->get_metadata(1);

	if (name!=String()) {

		if (get_instanced_node()) {

			Dictionary d = get_instanced_node()->get_instance_state();
			if (d.has(name)) {
				Variant v = obj->get(name);
				Variant vorig = d[name];

				int found=-1;
				for(int i=0;i<p_item->get_button_count(1);i++) {

					if (p_item->get_button_id(1,i)==3) {
						found=i;
						break;
					}
				}

				bool changed = ! (v==vorig);

				if ((found!=-1)!=changed) {

					if (changed) {

						p_item->add_button(1,get_icon("Reload","EditorIcons"),3);
					} else {

						p_item->erase_button(1,found);
					}

				}

			}

		}

		Dictionary d=p_item->get_metadata(0);
		set_item_text(p_item,d["type"],d["name"],d["hint"],d["hint_text"]);
	}

	TreeItem *c=p_item->get_children();

	while (c) {

		_refresh_item(c);

		c=c->get_next();
	}

}

void PropertyEditor::refresh() {

	if (refresh_countdown>0)
		return;
	refresh_countdown=EditorSettings::get_singleton()->get("property_editor/auto_refresh_interval");

}

void PropertyEditor::update_tree() {


	tree->clear();

	if (!obj)
		return;

	HashMap<String,TreeItem*> item_path;

	TreeItem *root = tree->create_item(NULL);
	tree->set_hide_root(true);

	/*
	TreeItem *title = tree->create_item(root);

	title->set_custom_bg_color(0,get_color("prop_section","Editor"));
	title->set_text(0,"Property"); // todo, fetch name if ID exists in database
	title->set_editable(0,false);
	title->set_selectable(0,false);
	title->set_custom_bg_color(1,get_color("prop_section","Editor"));
	title->set_text(1,"Value"); // todo, fetch name if ID exists in database
	title->set_editable(1,false);
	title->set_selectable(1,false);
*/

	/*
	if (obj->cast_to<Node>() || obj->cast_to<Resource>()) {
		TreeItem *type = tree->create_item(root);

		type->set_text(0,"Type"); // todo, fetch name if ID exists in database
		type->set_text(1,obj->get_type()); // todo, fetch name if ID exists in database
		if (has_icon(obj->get_type(),"EditorIcons"))
			type->set_icon(1,get_icon(obj->get_type(),"EditorIcons") );
		else
			type->set_icon(1,get_icon("Object","EditorIcons") );

		type->set_selectable(0,false);
		type->set_selectable(1,false);


		TreeItem *name = tree->create_item(root);

		name->set_text(0,"Name"); // todo, fetch name if ID exists in database
		if (obj->is_type("Resource"))
			name->set_text(1,obj->cast_to<Resource>()->get_name());
		else if (obj->is_type("Node"))
			name->set_text(1,obj->cast_to<Node>()->get_name());
		name->set_selectable(0,false);
		name->set_selectable(1,false);
	}
*/
	List<PropertyInfo> plist;
	obj->get_property_list(&plist,true);

	bool draw_red=false;

	{
		Node *nod = obj->cast_to<Node>();
		Node *es = EditorNode::get_singleton()->get_edited_scene();
		if (nod && es!=nod && nod->get_owner()!=es) {
			draw_red=true;
		}
	}


	Color sscolor=get_color("prop_subsection","Editor");

	TreeItem * current_category=NULL;

	for (List<PropertyInfo>::Element *I=plist.front() ; I ; I=I->next()) {

		PropertyInfo& p = I->get();

		//make sure the property can be edited

		if (p.usage&PROPERTY_USAGE_CATEGORY) {

			if (!show_categories)
				continue;

			List<PropertyInfo>::Element *N=I->next();
			bool valid=true;
			//if no properties in category, skip
			while(N) {
				if (N->get().usage&PROPERTY_USAGE_EDITOR)
					break;
				if (N->get().usage&PROPERTY_USAGE_CATEGORY) {
					valid=false;
					break;
				}
				N=N->next();
			}
			if (!valid)
				continue; //empty, ignore
			TreeItem * sep = tree->create_item(root);
			current_category=sep;
			String type=p.name;
			/*if (has_icon(type,"EditorIcons"))
				sep->set_icon(0,get_icon(type,"EditorIcons") );
			else
				sep->set_icon(0,get_icon("Object","EditorIcons") );
			print_line("CATEGORY: "+type);
			*/
			sep->set_text(0,type);
			sep->set_selectable(0,false);
			sep->set_selectable(1,false);
			sep->set_custom_bg_color(0,get_color("prop_category","Editor"));
			sep->set_custom_bg_color(1,get_color("prop_category","Editor"));
			//sep->set_custom_color(0,Color(1,1,1));


			continue;
		} else  if ( ! (p.usage&PROPERTY_USAGE_EDITOR ) )
			continue;

		String path=p.name.left( p.name.find_last("/") ) ;
		//printf("property %s\n",p.name.ascii().get_data());
		TreeItem * parent = get_parent_node(path,item_path,current_category?current_category:root );
		//if (parent->get_parent()==root)
		//	parent=root;
		int level = 0;
		if (parent!=root) {
			level++;
			TreeItem *parent_lev=parent;
			while(parent_lev->get_parent()!=root) {
				parent_lev=parent_lev->get_parent();
				level++;
			}
		}
		if (level>4)
			level=4;

		Color col = sscolor;
		col.a=(level/4.0)*0.7;

		TreeItem * item = tree->create_item( parent );

		String name = (p.name.find("/")!=-1)?p.name.right( p.name.find_last("/")+1 ):p.name;

		if (level>0) {
			item->set_custom_bg_color(0,col);
			//item->set_custom_bg_color(1,col);
		}
		item->set_editable(0,false);
		item->set_selectable(0,false);

		if (p.usage&PROPERTY_USAGE_CHECKABLE) {

			item->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
			item->set_selectable(0,true);
			item->set_editable(0,true);
			item->set_checked(0,p.usage&PROPERTY_USAGE_CHECKED);
		}

		if (capitalize_paths)
			item->set_text( 0, name.capitalize() );
		else
			item->set_text( 0, name );

		item->set_tooltip(0, p.name);

		Dictionary d;
		d["name"]=p.name;
		d["type"]=(int)p.type;
		d["hint"]=(int)p.hint;
		d["hint_text"]=p.hint_string;
					
		item->set_metadata( 0, d );
		item->set_metadata( 1, p.name );

		if (draw_red)
			item->set_custom_color(0,Color(0.8,0.4,0.20));

		
		if (p.name==selected_property) {

			item->select(1);
		}

		
		//printf("property %s type %i\n",p.name.ascii().get_data(),p.type);
		switch( p.type ) {

			case Variant::BOOL: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CHECK );
				item->set_text(1,"On");
				item->set_checked( 1, obj->get( p.name ) );
				item->set_icon( 0, get_icon("Bool","EditorIcons") );
				item->set_editable(1,!read_only);

			} break;
			case Variant::REAL:
			case Variant::INT: {

				if (p.hint==PROPERTY_HINT_EXP_EASING) {

					item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
					item->set_text(1, String::num(obj->get( p.name ),2) );
					item->set_editable(1,!read_only);
					item->set_icon( 0, get_icon("Curve","EditorIcons"));

					break;

				}

				if (p.hint==PROPERTY_HINT_ALL_FLAGS) {

					item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
					item->set_editable(1,!read_only);
					item->set_custom_draw(1,this,"_draw_flags");
					break;
				}

				if (p.hint==PROPERTY_HINT_FLAGS) {


					item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
					item->set_editable(1,!read_only);
					//item->set_icon( 0, get_icon("Curve","EditorIcons"));

					Vector<String> values = p.hint_string.split(",");
					String flags;
					int val = obj->get(p.name);
					for(int i=0;i<values.size();i++) {

						String v = values[i];
						if (v=="")
							continue;
						if (!(val&(1<<i)))
							continue;

						if (flags!="")
							flags+=", ";
						flags+=v;
					}
					item->set_text(1, flags );
					break;

				}


				item->set_cell_mode( 1, TreeItem::CELL_MODE_RANGE );

				if (p.hint==PROPERTY_HINT_RANGE || p.hint==PROPERTY_HINT_EXP_RANGE) {

					int c = p.hint_string.get_slice_count(",");
					float min=0,max=100,step=1;
					if (c>=1) {

						min=p.hint_string.get_slice(",",0).to_double();
					}
					if (c>=2) {

						max=p.hint_string.get_slice(",",1).to_double();
					}

					if (p.type==Variant::REAL && c>=3) {

						step= p.hint_string.get_slice(",",2).to_double();
					} 
					

					item->set_range_config(1,min,max,step,p.hint==PROPERTY_HINT_EXP_RANGE);
				} else if (p.hint==PROPERTY_HINT_ENUM) {

//					int c = p.hint_string.get_slice_count(",");
					item->set_text(1,p.hint_string);
					item->set_icon( 0,get_icon("Enum","EditorIcons") );
					item->set_range(1, obj->get( p.name ) );
					item->set_editable(1,!read_only);
					break;

				} else {
					if (p.type == Variant::REAL) {

						item->set_range_config(1, -65536, 65535, 0.01);
					} else {

						item->set_range_config(1, -65536, 65535, 1);
					}
				};

				if (p.type==Variant::REAL) {
					item->set_icon( 0, get_icon("Real","EditorIcons"));
					item->set_range(1, obj->get( p.name ) );

				} else {
					item->set_icon( 0,get_icon("Integer","EditorIcons") );
					item->set_range(1, obj->get( p.name ) );
				}


				item->set_editable(1,!read_only);


			} break;
			case Variant::STRING: {

				switch (p.hint) {

					case PROPERTY_HINT_DIR:
					case PROPERTY_HINT_FILE:
					case PROPERTY_HINT_GLOBAL_DIR:
					case PROPERTY_HINT_GLOBAL_FILE: {

						item->set_cell_mode( 1, TreeItem::CELL_MODE_STRING );
						item->set_editable(1,!read_only);
						item->set_icon( 0, get_icon("File","EditorIcons") );
						item->set_text(1,obj->get(p.name));
						item->add_button(1,get_icon("Folder","EditorIcons"));

					} break;
					case PROPERTY_HINT_ENUM: {

						item->set_cell_mode( 1, TreeItem::CELL_MODE_RANGE );
						Vector<String> strings = p.hint_string.split(",");
						String current = obj->get(p.name);
						int idx=0;
						for(int x=0;x<strings.size();x++) {
							if (strings[x]==current) {
								idx=x;
								break;
							}
						}
						item->set_text(1, p.hint_string);
						item->set_range(1,idx);
						item->set_editable( 1, !read_only );
						item->set_icon( 0,get_icon("Enum","EditorIcons") );


					} break;
					default: {

						item->set_cell_mode( 1, TreeItem::CELL_MODE_STRING );
						item->set_editable(1,!read_only);
						item->set_icon( 0, get_icon("String","EditorIcons") );
						item->set_text(1,obj->get(p.name));
						if (p.hint==PROPERTY_HINT_MULTILINE_TEXT)
							item->add_button(1,get_icon("MultiLine","EditorIcons") );

					} break;
				}

			} break;
			case Variant::INT_ARRAY: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
				item->set_text(1,"[IntArray]");
				item->set_icon( 0, get_icon("ArrayInt","EditorIcons") );


			} break;
			case Variant::REAL_ARRAY: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
				item->set_text(1,"[RealArray]");
				item->set_icon( 0, get_icon("ArrayReal","EditorIcons") );

			} break;
			case Variant::STRING_ARRAY: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
				item->set_text(1,"[StringArray]");
				item->set_icon( 0, get_icon("ArrayString","EditorIcons") );

			} break;
			case Variant::RAW_ARRAY: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
				item->set_text(1,"[Raw Data]");
				item->set_icon( 0, get_icon("ArrayData","EditorIcons") );

			} break;
			case Variant::VECTOR2: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, true );
				item->set_text(1,obj->get(p.name));
				item->set_icon( 0,get_icon("Vector2","EditorIcons") );

			} break;
			case Variant::RECT2: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, true );
				item->set_text(1,obj->get(p.name));
				item->set_icon( 0,get_icon("Rect2","EditorIcons") );

			} break;
			case Variant::VECTOR3: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, true );
				item->set_text(1,obj->get(p.name));
				item->set_icon( 0,get_icon("Vector","EditorIcons") );

			} break;
			case Variant::TRANSFORM: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, true );
				item->set_text(1,obj->get(p.name));
				item->set_icon( 0,get_icon("Matrix","EditorIcons") );

			} break;
			case Variant::PLANE: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, true );
				item->set_text(1,obj->get(p.name));
				item->set_icon( 0,get_icon("Plane","EditorIcons") );

			} break;
			case Variant::_AABB: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, true );
				item->set_text(1,"AABB");
				item->set_icon( 0,get_icon("Rect3","EditorIcons") );
			} break;

			case Variant::QUAT: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, true );
				item->set_text(1,obj->get(p.name));
				item->set_icon( 0,get_icon("Quat","EditorIcons") );

			} break;
			case Variant::COLOR: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
//				item->set_text(1,obj->get(p.name));
				item->set_custom_bg_color(1,obj->get(p.name));
				item->set_icon( 0,get_icon("Color","EditorIcons") );

			} break;
			case Variant::IMAGE: {
			
				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
				Image img = obj->get( p.name );
				if (img.empty())
					item->set_text(1,"[Image (empty)]");
				else
					item->set_text(1,"[Image "+itos(img.get_width())+"x"+itos(img.get_height())+"]");
				item->set_icon( 0,get_icon("Image","EditorIcons") );
			
			} break;
			case Variant::NODE_PATH: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
				item->set_text(1,obj->get(p.name));

			} break;
			case Variant::OBJECT: {

				item->set_cell_mode( 1, TreeItem::CELL_MODE_CUSTOM );
				item->set_editable( 1, !read_only );
				item->add_button(1,get_icon("EditResource","EditorIcons"));
				String type;
				bool notnil=false;
				if (obj->get( p.name ).get_type() == Variant::NIL || obj->get( p.name ).operator RefPtr().is_null()) {
					item->set_text(1,"<null>");

				} else {
					RES res = obj->get( p.name ).operator RefPtr();

					if (res->is_type("Texture")) {
						int tw = EditorSettings::get_singleton()->get("property_editor/texture_preview_width");
						item->set_icon_max_width(1,tw);
						item->set_icon(1,res);
						item->set_text(1,"");

					} else if (res->get_name() != "") {

						item->set_text(1, res->get_name());
					} else if (res->get_path()!="" && !res->get_path().begins_with("local://")) {
						item->set_text(1, res->get_path().get_file());

					} else {
						item->set_text(1,"<"+res->get_type()+">");
					};
					notnil=true;

				}

				if (p.hint==PROPERTY_HINT_RESOURCE_TYPE) {
					//printf("prop %s , type %s\n",p.name.ascii().get_data(),p.hint_string.ascii().get_data());
					if (has_icon(p.hint_string,"EditorIcons"))
						item->set_icon( 0, get_icon(p.hint_string,"EditorIcons") );
					else
						item->set_icon( 0, get_icon("Object","EditorIcons") );
				}

//				item->double_click_signal.connect( Method1<int>( Method2<int,String>( this, &Editoritem_obj_edited ), p.name ) );

			} break;
			default: {};
		}

		if (keying) {

			item->add_button(1,get_icon("Key","EditorIcons"),2);
		}

		if (get_instanced_node()) {

			Dictionary d = get_instanced_node()->get_instance_state();
			if (d.has(p.name)) {
				Variant v = obj->get(p.name);
				Variant vorig = d[p.name];
				if (! (v==vorig)) {

					item->add_button(1,get_icon("Reload","EditorIcons"),3);
				}
			}

		}

	}
}


void PropertyEditor::_item_selected() {


	TreeItem *item = tree->get_selected();
	ERR_FAIL_COND(!item);
	selected_property=item->get_metadata(1);

}


void PropertyEditor::_edit_set(const String& p_name, const Variant& p_value) {

	if (autoclear) {
		TreeItem *item = tree->get_selected();
		if (item && item->get_cell_mode(0)==TreeItem::CELL_MODE_CHECK) {

			item->set_checked(0,true);
		}
	}

	if (!undo_redo) {

		obj->set(p_name,p_value);
		_changed_callbacks(obj,p_name);
		emit_signal(_prop_edited,p_name);


	} else {


		undo_redo->create_action("Set "+p_name,true);
		undo_redo->add_do_property(obj,p_name,p_value);
		undo_redo->add_undo_property(obj,p_name,obj->get(p_name));
		undo_redo->add_do_method(this,"_changed_callback",obj,p_name);
		undo_redo->add_undo_method(this,"_changed_callback",obj,p_name);
		undo_redo->add_undo_method(this,"_changed_callback",obj,p_name);
		Resource *r = obj->cast_to<Resource>();
		if (r) {
			if (!r->is_edited() && String(p_name)!="resource/edited") {
				undo_redo->add_do_method(r,"set_edited",true);
				undo_redo->add_undo_method(r,"set_edited",false);
			}
		}
		_prop_edited_name[0]=p_name;
		undo_redo->add_do_method(this,"emit_signal",_prop_edited,_prop_edited_name);
		undo_redo->commit_action();
	}
}


void PropertyEditor::_item_edited() {

	
	TreeItem * item = tree->get_edited();		
	Dictionary d = item->get_metadata(0);
	
	String name=d["name"];

	if (tree->get_edited_column()==0) {
		//property checked
		if (autoclear) {
			if (!item->is_checked(0)) {
				obj->set(name,Variant());
				update_property(name);
			} else {
				Variant::CallError ce;
				obj->set(name,Variant::construct(Variant::Type(int(d["type"])),NULL,0,ce));
			}
		} else {
			emit_signal("property_toggled",name,item->is_checked(0));
		}
		return;
	}

	if (autoclear && item->get_cell_mode(0)==TreeItem::CELL_MODE_CHECK && item->get_cell_mode(1)!=TreeItem::CELL_MODE_CUSTOM) {
		item->set_checked(0,true);

	}


	int type=d["type"];
	int hint= d["hint"];
	String hint_text=d["hint_text"];
	switch(type) {
		
		case Variant::NIL: {
			
		} break;			
		case Variant::BOOL: {
			
			_edit_set(name,item->is_checked(1));
		} break;
		case Variant::INT:
		case Variant::REAL: {

			if (hint==PROPERTY_HINT_ALL_FLAGS)
				break;
			if (hint==PROPERTY_HINT_EXP_EASING)
				break;
			if (hint==PROPERTY_HINT_FLAGS)
				break;

			if (type==Variant::INT)
				_edit_set(name,int(item->get_range(1)));
			else
				_edit_set(name,item->get_range(1));
		} break;
		case Variant::STRING: {
			
			if (hint==PROPERTY_HINT_ENUM) {

				int idx= item->get_range(1);

				Vector<String> strings = hint_text.split(",");
				String txt;
				if (idx>=0 && idx<strings.size()) {

					txt=strings[idx];
				}

				_edit_set(name,txt);
			} else {
				_edit_set(name,item->get_text(1));
			}
		} break;
			// math types
			
		case Variant::VECTOR3: {
			
		} break;
		case Variant::PLANE: {
			
		} break;
		case Variant::QUAT: {
			
		} break;
		case Variant::_AABB: {
			
		} break;
		case Variant::MATRIX3: {
			
		} break;
		case Variant::TRANSFORM: {
			
		} break;
			
		case Variant::COLOR: {
			//_edit_set(name,item->get_custom_bg_color(0));
		} break;
		case Variant::IMAGE: {
			
		} break;
		case Variant::NODE_PATH: {
			
		} break;

		case Variant::INPUT_EVENT: {
			
		} break;
		case Variant::DICTIONARY: {
			
		} break;
			
			// arrays
		case Variant::RAW_ARRAY: {
			
		} break;
		case Variant::INT_ARRAY: {
			
		} break;
		case Variant::REAL_ARRAY: {
			
		} break;
		case Variant::STRING_ARRAY: {
			
		} break;
		case Variant::VECTOR3_ARRAY: {
			
		} break;
		case Variant::COLOR_ARRAY: {
			
		} break;
			
		
	};
}


void PropertyEditor::_resource_edit_request() {

	RES res=custom_editor->get_variant();
	if (res.is_null())
		return;

	String name=custom_editor->get_name();


	emit_signal("resource_selected",res.get_ref_ptr(),name);
}

void PropertyEditor::_custom_editor_edited() {


	if (!obj)
		return;


	_edit_set(custom_editor->get_name(), custom_editor->get_variant());
}

void PropertyEditor::_custom_editor_request(bool p_arrow) {

	TreeItem * item = tree->get_edited();
	ERR_FAIL_COND(!item);
	Dictionary d = item->get_metadata(0);

	//int type=d["type"];
	String name=d["name"];
	Variant::Type type=Variant::NIL;
	if (d.has("type"))
		type=(Variant::Type)((int)(d["type"]));

	Variant v=obj->get(name);
	int hint=d.has("hint")?d["hint"].operator int():-1;
	String hint_text=d.has("hint_text")?d["hint_text"]:"";
	Rect2 where=tree->get_custom_popup_rect();
	custom_editor->set_pos(where.pos);

	if (custom_editor->edit(obj,name,type,v,hint,hint_text)) {
		custom_editor->popup();
	}
}

void PropertyEditor::edit(Object* p_object) {
	

	if (obj==p_object)
		return;
	if (obj) {
		
		obj->remove_change_receptor(this);
	}

	obj=p_object;
	update_tree();
	
	if (obj) {
		
		obj->add_change_receptor(this);
	}
	
	
}

void PropertyEditor::_edit_button(Object *p_item, int p_column, int p_button) {
	TreeItem *ti = p_item->cast_to<TreeItem>();
	ERR_FAIL_COND(!ti);

	Dictionary d = ti->get_metadata(0);

	if (p_button==2) {

		if (!d.has("name"))
			return;
		String prop=d["name"];
		emit_signal("property_keyed",prop,obj->get(prop));
	} else if (p_button==3) {

		if (!get_instanced_node())
			return;
		if (!d.has("name"))
			return;

		String prop=d["name"];

		Dictionary d2 = get_instanced_node()->get_instance_state();
		if (d2.has(prop)) {

			_edit_set(prop,d2[prop]);
		}

	} else {

		Dictionary d = ti->get_metadata(0);
		if (!d.has("type"))
			return;
		if (!d.has("hint"))
			return;
		if (!d.has("name"))
			return;
		if (!d.has("hint_text"))
			return;

		int t = d["type"];
		int h = d["hint"];
		String n = d["name"];
		String ht = d["hint_text"];

		if (t==Variant::STRING) {


			Variant v = obj->get(n);
			custom_editor->edit(obj,n,(Variant::Type)t,v,h,ht);
			//Rect2 where=tree->get_custom_popup_rect();
			if (h==PROPERTY_HINT_FILE || h==PROPERTY_HINT_DIR || h==PROPERTY_HINT_GLOBAL_DIR || h==PROPERTY_HINT_GLOBAL_FILE) {

				Rect2 where=tree->get_item_rect(ti,1);
				where.pos-=tree->get_scroll();
				where.pos+=tree->get_global_pos();
				custom_editor->set_pos(where.pos);
				custom_editor->popup();
			} else {
				custom_editor->popup_centered_ratio();
			}

		} else if (t==Variant::OBJECT) {

			RES r = obj->get(n);
			if (r.is_valid()) {

				emit_signal("resource_selected",r,n);
			}
		}
	}
}


void PropertyEditor::_node_removed(Node *p_node) {
	
	if (p_node==obj) {
		
		edit(NULL);
	}
}


void PropertyEditor::set_keying(bool p_active) {

	if (keying==p_active)
		return;

	keying=p_active;
	update_tree();
}

void PropertyEditor::_draw_flags(Object *t,const Rect2& p_rect) {

	TreeItem *ti = t->cast_to<TreeItem>();
	if (!ti)
		return;

	Dictionary d = ti->get_metadata(0);

	if (!d.has("name"))
		return;

	uint32_t f = obj->get(d["name"]);

	int bsize = (p_rect.size.height*80/100)/2;

	int h = bsize*2+1;
	int vofs = (p_rect.size.height-h)/2;

	for(int i=0;i<2;i++) {

		Point2 ofs(4,vofs);
		if (i==1)
			ofs.y+=bsize+1;

		ofs+=p_rect.pos;
		for(int j=0;j<10;j++) {

			Point2 o = ofs+Point2(j*(bsize+1),0);
			if (j>=5)
				o.x+=1;

			uint32_t idx=i*10+j;
			bool on=f&(1<<idx);
			tree->draw_rect(Rect2(o,Size2(bsize,bsize)),Color(0,0,0,on?0.8:0.3));
		}
	}


}

void PropertyEditor::_bind_methods() {

	ObjectTypeDB::bind_method( "_item_edited",&PropertyEditor::_item_edited);
	ObjectTypeDB::bind_method( "_item_selected",&PropertyEditor::_item_selected);
	ObjectTypeDB::bind_method( "_custom_editor_request",&PropertyEditor::_custom_editor_request);
	ObjectTypeDB::bind_method( "_custom_editor_edited",&PropertyEditor::_custom_editor_edited);	
	ObjectTypeDB::bind_method( "_resource_edit_request",&PropertyEditor::_resource_edit_request);	
	ObjectTypeDB::bind_method( "_node_removed",&PropertyEditor::_node_removed);		
	ObjectTypeDB::bind_method( "_edit_button",&PropertyEditor::_edit_button);
	ObjectTypeDB::bind_method( "_changed_callback",&PropertyEditor::_changed_callbacks);
	ObjectTypeDB::bind_method( "_draw_flags",&PropertyEditor::_draw_flags);

	ADD_SIGNAL( MethodInfo("property_toggled",PropertyInfo( Variant::STRING, "property"),PropertyInfo( Variant::BOOL, "value")));
	ADD_SIGNAL( MethodInfo("resource_selected", PropertyInfo( Variant::OBJECT, "res"),PropertyInfo( Variant::STRING, "prop") ) );
	ADD_SIGNAL( MethodInfo("property_keyed",PropertyInfo( Variant::STRING, "property")));
	ADD_SIGNAL( MethodInfo("property_edited",PropertyInfo( Variant::STRING, "property")));
}

Tree *PropertyEditor::get_scene_tree() {

	return tree;
}

Label* PropertyEditor::get_top_label() {

	return top_label;
}

void PropertyEditor::hide_top_label() {

	top_label->hide();
	tree->set_begin( Point2(0,0 ));
}

String PropertyEditor::get_selected_path() const {


	TreeItem *ti = tree->get_selected();
	if (!ti)
		return "";

	Dictionary d = ti->get_metadata(0);

	if (d.has("name"))
		return d["name"];
	else
		return "";
}

void PropertyEditor::set_capitalize_paths(bool p_capitalize) {

	capitalize_paths=p_capitalize;
}

void PropertyEditor::set_autoclear(bool p_enable) {

	autoclear=p_enable;
}

void PropertyEditor::set_show_categories(bool p_show) {

	show_categories=p_show;
	update_tree();
}

PropertyEditor::PropertyEditor() {
	
	_prop_edited="property_edited";
	_prop_edited_name.push_back(String());
	undo_redo=NULL;
	obj=NULL;
	changing=false;
	update_tree_pending=false;
	
	top_label = memnew( Label );
	top_label->set_text("Properties:");
	top_label->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	top_label->set_begin( Point2( 10,0) );
	top_label->set_end( Point2( 0,12) );		
	
	add_child(top_label);	

	
	tree = memnew( Tree );
	tree->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	tree->set_anchor( MARGIN_BOTTOM, ANCHOR_END );	
	tree->set_begin( Point2(0,19 ));
	tree->set_end( Point2(0,0 ));
	
	tree->set_columns(2);
	tree->set_column_expand(0,true);
	tree->set_column_min_width(0,30);
	tree->set_column_expand(1,true);
	tree->set_column_min_width(1,18);

	//tree->set_hide_root(true);
	add_child( tree );	
	
	tree->connect("item_edited", this,"_item_edited",varray(),CONNECT_DEFERRED);
	tree->connect("cell_selected", this,"_item_selected");
	
	set_fixed_process(true);
	
	custom_editor = memnew( CustomPropertyEditor );
	add_child(custom_editor);
	
	tree->connect("custom_popup_edited", this,"_custom_editor_request");
	tree->connect("button_pressed", this,"_edit_button");
	custom_editor->connect("variant_changed", this,"_custom_editor_edited");
	custom_editor->connect("resource_edit_request", this,"_resource_edit_request",make_binds(),CONNECT_DEFERRED);
	
	capitalize_paths=true;
	autoclear=false;
	tree->set_column_title(0,"Property");
	tree->set_column_title(1,"Value");
	tree->set_column_titles_visible(true);

	keying=false;
	read_only=false;
	show_categories=false;
	refresh_countdown=0;
	
}


PropertyEditor::~PropertyEditor()
{
}



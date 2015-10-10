#include "scene_format_text.h"

#include "globals.h"
#include "version.h"
#include "os/dir_access.h"

#define FORMAT_VERSION 1

void ResourceFormatSaverTextInstance::write_property(const String& p_name,const Variant& p_property,bool *r_ok) {

	if (r_ok)
		*r_ok=false;

	if (p_name!=String()) {
		f->store_string(p_name+" = ");
	}

	switch( p_property.get_type() ) {

		case Variant::NIL: {
			f->store_string("null");
		} break;
		case Variant::BOOL: {

			f->store_string(p_property.operator bool() ? "true":"false" );
		} break;
		case Variant::INT: {

			f->store_string( itos(p_property.operator int()) );
		} break;
		case Variant::REAL: {

			f->store_string( rtoss(p_property.operator real_t()) );
		} break;
		case Variant::STRING: {

			String str=p_property;

			str="\""+str.c_escape()+"\"";
			f->store_string( str );
		} break;
		case Variant::VECTOR2: {

			Vector2 v = p_property;
			f->store_string("Vector2( "+rtoss(v.x) +", "+rtoss(v.y)+" )" );
		} break;
		case Variant::RECT2: {

			Rect2 aabb = p_property;
			f->store_string("Rect2( "+rtoss(aabb.pos.x) +", "+rtoss(aabb.pos.y) +", "+rtoss(aabb.size.x) +", "+rtoss(aabb.size.y)+" )" );

		} break;
		case Variant::VECTOR3: {

			Vector3 v = p_property;
			f->store_string("Vector3( "+rtoss(v.x) +", "+rtoss(v.y)+", "+rtoss(v.z)+" )");
		} break;
		case Variant::PLANE: {

			Plane p = p_property;
			f->store_string("Plane( "+rtoss(p.normal.x) +", "+rtoss(p.normal.y)+", "+rtoss(p.normal.z)+", "+rtoss(p.d)+" )" );

		} break;
		case Variant::_AABB: {

			AABB aabb = p_property;
			f->store_string("AABB( "+rtoss(aabb.pos.x) +", "+rtoss(aabb.pos.y) +", "+rtoss(aabb.pos.z) +", "+rtoss(aabb.size.x) +", "+rtoss(aabb.size.y) +", "+rtoss(aabb.size.z)+" )"  );

		} break;
		case Variant::QUAT: {

			Quat quat = p_property;
			f->store_string("Quat( "+rtoss(quat.x)+", "+rtoss(quat.y)+", "+rtoss(quat.z)+", "+rtoss(quat.w)+" )");

		} break;
		case Variant::MATRIX32: {

			String s="Matrix32( ";
			Matrix32 m3 = p_property;
			for (int i=0;i<3;i++) {
				for (int j=0;j<2;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}

			f->store_string(s+" )");

		} break;
		case Variant::MATRIX3: {

			String s="Matrix3( ";
			Matrix3 m3 = p_property;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}

			f->store_string(s+" )");

		} break;
		case Variant::TRANSFORM: {

			String s="Transform( ";
			Transform t = p_property;
			Matrix3 &m3 = t.basis;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}

			s=s+", "+rtoss(t.origin.x) +", "+rtoss(t.origin.y)+", "+rtoss(t.origin.z);

			f->store_string(s+" )");
		} break;

			// misc types
		case Variant::COLOR: {

			Color c = p_property;
			f->store_string("Color( "+rtoss(c.r) +", "+rtoss(c.g)+", "+rtoss(c.b)+", "+rtoss(c.a)+" )");

		} break;
		case Variant::IMAGE: {


			Image img=p_property;

			if (img.empty()) {
				f->store_string("RawImage()");
				break;
			}

			String imgstr="RawImage( ";
			imgstr+=itos(img.get_width());
			imgstr+=", "+itos(img.get_height());
			imgstr+=", "+itos(img.get_mipmaps());
			imgstr+=", ";

			switch(img.get_format()) {

				case Image::FORMAT_GRAYSCALE: imgstr+="GRAYSCALE"; break;
				case Image::FORMAT_INTENSITY: imgstr+="INTENSITY"; break;
				case Image::FORMAT_GRAYSCALE_ALPHA: imgstr+="GRAYSCALE_ALPHA"; break;
				case Image::FORMAT_RGB: imgstr+="RGB"; break;
				case Image::FORMAT_RGBA: imgstr+="RGBA"; break;
				case Image::FORMAT_INDEXED : imgstr+="INDEXED"; break;
				case Image::FORMAT_INDEXED_ALPHA: imgstr+="INDEXED_ALPHA"; break;
				case Image::FORMAT_BC1: imgstr+="BC1"; break;
				case Image::FORMAT_BC2: imgstr+="BC2"; break;
				case Image::FORMAT_BC3: imgstr+="BC3"; break;
				case Image::FORMAT_BC4: imgstr+="BC4"; break;
				case Image::FORMAT_BC5: imgstr+="BC5"; break;
				case Image::FORMAT_PVRTC2: imgstr+="PVRTC2"; break;
				case Image::FORMAT_PVRTC2_ALPHA: imgstr+="PVRTC2_ALPHA"; break;
				case Image::FORMAT_PVRTC4: imgstr+="PVRTC4"; break;
				case Image::FORMAT_PVRTC4_ALPHA: imgstr+="PVRTC4_ALPHA"; break;
				case Image::FORMAT_ETC: imgstr+="ETC"; break;
				case Image::FORMAT_ATC: imgstr+="ATC"; break;
				case Image::FORMAT_ATC_ALPHA_EXPLICIT: imgstr+="ATC_ALPHA_EXPLICIT"; break;
				case Image::FORMAT_ATC_ALPHA_INTERPOLATED: imgstr+="ATC_ALPHA_INTERPOLATED"; break;
				case Image::FORMAT_CUSTOM: imgstr+="CUSTOM"; break;
				default: {}
			}


			String s;

			DVector<uint8_t> data = img.get_data();
			int len = data.size();
			DVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();;
			for (int i=0;i<len;i++) {

				uint8_t byte = ptr[i];
				const char  hex[16]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
				char str[3]={ hex[byte>>4], hex[byte&0xF], 0};
				s+=str;
			}

			imgstr+=", ";
			f->store_string(imgstr);
			f->store_string(s);
			f->store_string(" )");
		} break;
		case Variant::NODE_PATH: {

			String str=p_property;

			str="NodePath(\""+str.c_escape()+"\")";
			f->store_string(str);

		} break;

		case Variant::OBJECT: {

			RES res = p_property;
			if (res.is_null()) {
				f->store_string("null");
				if (r_ok)
					*r_ok=true;

				break; // don't save it
			}

			if (external_resources.has(res)) {

				f->store_string("ExtResource( "+itos(external_resources[res]+1)+" )");
			} else {

				if (internal_resources.has(res)) {
					f->store_string("SubResource( "+itos(internal_resources[res])+" )");
				} else 	if (res->get_path().length() && res->get_path().find("::")==-1) {

					//external resource
					String path=relative_paths?local_path.path_to_file(res->get_path()):res->get_path();
					f->store_string("Resource( \""+path+"\" )");
				} else {
					f->store_string("null");
					ERR_EXPLAIN("Resource was not pre cached for the resource section, bug?");
					ERR_BREAK(true);
					//internal resource
				}
			}

		} break;
		case Variant::INPUT_EVENT: {

			f->store_string("InputEvent()"); //will be added later
		} break;
		case Variant::DICTIONARY: {

			Dictionary dict = p_property;

			List<Variant> keys;
			dict.get_key_list(&keys);
			keys.sort();

			f->store_string("{ ");
			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				//if (!_check_type(dict[E->get()]))
				//	continue;
				bool ok;
				write_property("",E->get(),&ok);
				ERR_CONTINUE(!ok);

				f->store_string(":");
				write_property("",dict[E->get()],&ok);
				if (!ok)
					write_property("",Variant()); //at least make the file consistent..
				if (E->next())
					f->store_string(", ");
			}


			f->store_string(" }");


		} break;
		case Variant::ARRAY: {

			f->store_string("[ ");
			Array array = p_property;
			int len=array.size();
			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");
				write_property("",array[i]);


			}
			f->store_string(" ]");

		} break;

		case Variant::RAW_ARRAY: {

			f->store_string("RawArray( ");
			String s;
			DVector<uint8_t> data = p_property;
			int len = data.size();
			DVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();;
			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");
				uint8_t byte = ptr[i];
				const char  hex[16]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
				char str[3]={ hex[byte>>4], hex[byte&0xF], 0};
				f->store_string(str);

			}

			f->store_string(" )");

		} break;
		case Variant::INT_ARRAY: {

			f->store_string("IntArray( ");
			DVector<int> data = p_property;
			int len = data.size();
			DVector<int>::Read r = data.read();
			const int *ptr=r.ptr();;

			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");

				f->store_string(itos(ptr[i]));
			}


			f->store_string(" )");

		} break;
		case Variant::REAL_ARRAY: {

			f->store_string("FloatArray( ");
			DVector<real_t> data = p_property;
			int len = data.size();
			DVector<real_t>::Read r = data.read();
			const real_t *ptr=r.ptr();;

			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");
				f->store_string(rtoss(ptr[i]));
			}

			f->store_string(" )");

		} break;
		case Variant::STRING_ARRAY: {

			f->store_string("StringArray( ");
			DVector<String> data = p_property;
			int len = data.size();
			DVector<String>::Read r = data.read();
			const String *ptr=r.ptr();;
			String s;
			//write_string("\n");



			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");
				String str=ptr[i];
				f->store_string(""+str.c_escape()+"\"");
			}

			f->store_string(" )");

		} break;
		case Variant::VECTOR2_ARRAY: {

			f->store_string("Vector2Array( ");
			DVector<Vector2> data = p_property;
			int len = data.size();
			DVector<Vector2>::Read r = data.read();
			const Vector2 *ptr=r.ptr();;

			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");
				f->store_string(rtoss(ptr[i].x)+", "+rtoss(ptr[i].y) );
			}

			f->store_string(" )");

		} break;
		case Variant::VECTOR3_ARRAY: {

			f->store_string("Vector3Array( ");
			DVector<Vector3> data = p_property;
			int len = data.size();
			DVector<Vector3>::Read r = data.read();
			const Vector3 *ptr=r.ptr();;

			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");
				f->store_string(rtoss(ptr[i].x)+", "+rtoss(ptr[i].y)+", "+rtoss(ptr[i].z) );
			}

			f->store_string(" )");

		} break;
		case Variant::COLOR_ARRAY: {

			f->store_string("ColorArray( ");

			DVector<Color> data = p_property;
			int len = data.size();
			DVector<Color>::Read r = data.read();
			const Color *ptr=r.ptr();;

			for (int i=0;i<len;i++) {

				if (i>0)
					f->store_string(", ");

				f->store_string(rtoss(ptr[i].r)+", "+rtoss(ptr[i].g)+", "+rtoss(ptr[i].b)+", "+rtoss(ptr[i].a) );

			}
			f->store_string(" )");

		} break;
		default: {}

	}

	if (r_ok)
		*r_ok=true;

}


void ResourceFormatSaverTextInstance::_find_resources(const Variant& p_variant,bool p_main) {


	switch(p_variant.get_type()) {
		case Variant::OBJECT: {


			RES res = p_variant.operator RefPtr();

			if (res.is_null() || external_resources.has(res))
				return;

			if (!p_main && (!bundle_resources ) && res->get_path().length() && res->get_path().find("::") == -1 ) {
				int index = external_resources.size();
				external_resources[res]=index;
				return;
			}

			if (resource_set.has(res))
				return;

			List<PropertyInfo> property_list;

			res->get_property_list( &property_list );
			property_list.sort();

			List<PropertyInfo>::Element *I=property_list.front();

			while(I) {

				PropertyInfo pi=I->get();

				if (pi.usage&PROPERTY_USAGE_STORAGE || (bundle_resources && pi.usage&PROPERTY_USAGE_BUNDLE)) {

					Variant v=res->get(I->get().name);
					_find_resources(v);
				}

				I=I->next();
			}

			resource_set.insert( res ); //saved after, so the childs it needs are available when loaded
			saved_resources.push_back(res);

		} break;
		case Variant::ARRAY: {

			Array varray=p_variant;
			int len=varray.size();
			for(int i=0;i<len;i++) {

				Variant v=varray.get(i);
				_find_resources(v);
			}

		} break;
		case Variant::DICTIONARY: {

			Dictionary d=p_variant;
			List<Variant> keys;
			d.get_key_list(&keys);
			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				Variant v = d[E->get()];
				_find_resources(v);
			}
		} break;
		default: {}
	}

}



Error ResourceFormatSaverTextInstance::save(const String &p_path,const RES& p_resource,uint32_t p_flags) {

	if (p_path.ends_with(".tscn")) {
		packed_scene=p_resource;
	}

	Error err;
	f = FileAccess::open(p_path, FileAccess::WRITE,&err);
	ERR_FAIL_COND_V( err, ERR_CANT_OPEN );
	FileAccessRef _fref(f);

	local_path = Globals::get_singleton()->localize_path(p_path);

	relative_paths=p_flags&ResourceSaver::FLAG_RELATIVE_PATHS;
	skip_editor=p_flags&ResourceSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	bundle_resources=p_flags&ResourceSaver::FLAG_BUNDLE_RESOURCES;
	takeover_paths=p_flags&ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;
	if (!p_path.begins_with("res://")) {
		takeover_paths=false;
	}

	// save resources
	_find_resources(p_resource,true);

	if (packed_scene.is_valid()) {
		//add instances to external resources if saving a packed scene
		for(int i=0;i<packed_scene->get_state()->get_node_count();i++) {
			Ref<PackedScene> instance=packed_scene->get_state()->get_node_instance(i);
			if (instance.is_valid() && !external_resources.has(instance)) {
				int index = external_resources.size();
				external_resources[instance]=index;
			}
		}
	}


	ERR_FAIL_COND_V(err!=OK,err);

	{
		String title=packed_scene.is_valid()?"[gd_scene ":"[gd_resource ";
		if (packed_scene.is_null())
			title+="type=\""+p_resource->get_type()+"\" ";
		int load_steps=saved_resources.size()+external_resources.size();
		//if (packed_scene.is_valid()) {
		//	load_steps+=packed_scene->get_node_count();
		//}
		//no, better to not use load steps from nodes, no point to that

		if (load_steps>1) {
			title+="load_steps="+itos(load_steps)+" ";
		}
		title+="format="+itos(FORMAT_VERSION)+"";
		//title+="engine_version=\""+itos(VERSION_MAJOR)+"."+itos(VERSION_MINOR)+"\"";

		f->store_string(title);
		f->store_line("]\n"); //one empty line
	}


	for(Map<RES,int>::Element *E=external_resources.front();E;E=E->next()) {

		String p = E->key()->get_path();

		f->store_string("[ext_resource path=\""+p+"\" type=\""+E->key()->get_save_type()+"\" id="+itos(E->get()+1)+"]\n"); //bundled
	}

	if (external_resources.size())
		f->store_line(String()); //separate

	Set<int> used_indices;

	for(List<RES>::Element *E=saved_resources.front();E;E=E->next()) {

		RES res = E->get();
		if (E->next() && (res->get_path()=="" || res->get_path().find("::") != -1 )) {

			if (res->get_subindex()!=0) {
				if (used_indices.has(res->get_subindex())) {
					res->set_subindex(0); //repeated
				} else {
					used_indices.insert(res->get_subindex());
				}
			}
		}
	}

	for(List<RES>::Element *E=saved_resources.front();E;E=E->next()) {

		RES res = E->get();
		ERR_CONTINUE(!resource_set.has(res));
		bool main = (E->next()==NULL);

		if (main && packed_scene.is_valid())
			break; //save as a scene

		if (main) {
			f->store_line("[resource]\n");
		} else {
			String line="[sub_resource ";
			if (res->get_subindex()==0) {
				int new_subindex=1;
				if (used_indices.size()) {
					new_subindex=used_indices.back()->get()+1;
				}

				res->set_subindex(new_subindex);
				used_indices.insert(new_subindex);
			}

			int idx = res->get_subindex();
			line+="type=\""+res->get_type()+"\" id="+itos(idx);
			f->store_line(line+"]\n");
			if (takeover_paths) {
				res->set_path(p_path+"::"+itos(idx),true);
			}

			internal_resources[res]=idx;

		}


		List<PropertyInfo> property_list;
		res->get_property_list(&property_list);
//		property_list.sort();
		for(List<PropertyInfo>::Element *PE = property_list.front();PE;PE=PE->next()) {


			if (skip_editor && PE->get().name.begins_with("__editor"))
				continue;

			if (PE->get().usage&PROPERTY_USAGE_STORAGE || (bundle_resources && PE->get().usage&PROPERTY_USAGE_BUNDLE)) {

				String name = PE->get().name;
				Variant value = res->get(name);


				if ((PE->get().usage&PROPERTY_USAGE_STORE_IF_NONZERO && value.is_zero())||(PE->get().usage&PROPERTY_USAGE_STORE_IF_NONONE && value.is_one()) )
					continue;

				if (PE->get().type==Variant::OBJECT && value.is_zero())
					continue;

				write_property(name,value);
				f->store_string("\n");
			}


		}

		f->store_string("\n");

	}

	if (packed_scene.is_valid()) {
		//if this is a scene, save nodes and connections!
		Ref<SceneState> state = packed_scene->get_state();
		for(int i=0;i<state->get_node_count();i++) {

			StringName type = state->get_node_type(i);
			StringName name = state->get_node_name(i);
			NodePath path = state->get_node_path(i,true);
			NodePath owner = state->get_node_owner_path(i);
			Ref<PackedScene> instance = state->get_node_instance(i);
			Vector<StringName> groups = state->get_node_groups(i);

			String header="[node";
			header+=" name=\""+String(name)+"\"";
			if (type!=StringName()) {
				header+=" type=\""+String(type)+"\"";
			}
			if (path!=NodePath()) {
				header+=" parent=\""+String(path.simplified())+"\"";
			}
			if (owner!=NodePath() && owner!=NodePath(".")) {
				header+=" owner=\""+String(owner.simplified())+"\"";
			}

			if (groups.size()) {
				String sgroups=" groups=[ ";
				for(int j=0;j<groups.size();j++) {
					if (j>0)
						sgroups+=", ";
					sgroups+="\""+groups[i].operator String().c_escape()+"\"";
				}
				sgroups+=" ]";
				header+=sgroups;
			}

			f->store_string(header);

			if (instance.is_valid()) {
				f->store_string(" instance=");
				write_property("",instance);
			}

			f->store_line("]\n");

			for(int j=0;j<state->get_node_property_count(i);j++) {

				write_property(state->get_node_property_name(i,j),state->get_node_property_value(i,j));
				f->store_line(String());

			}

			if (state->get_node_property_count(i)) {
				//add space
				f->store_line(String());
			}

		}

		for(int i=0;i<state->get_connection_count();i++) {

			String connstr="[connection";
			connstr+=" signal=\""+String(state->get_connection_signal(i))+"\"";
			connstr+=" from=\""+String(state->get_connection_source(i).simplified())+"\"";
			connstr+=" to=\""+String(state->get_connection_target(i).simplified())+"\"";
			connstr+=" method=\""+String(state->get_connection_method(i))+"\"";
			int flags = state->get_connection_flags(i);
			if (flags!=Object::CONNECT_PERSIST) {
				connstr+=" flags="+itos(flags);
			}

			Array binds=state->get_connection_binds(i);
			f->store_string(connstr);
			if (binds.size()) {
				f->store_string(" binds=");
				write_property("",binds);
			}

			f->store_line("]\n");
		}

		f->store_line(String());

		Vector<NodePath> editable_instances = state->get_editable_instances();
		for(int i=0;i<editable_instances.size();i++) {
			f->store_line("[editable path=\""+editable_instances[i].operator String()+"\"]");
		}
	}

	if (f->get_error()!=OK && f->get_error()!=ERR_FILE_EOF) {
		f->close();
		return ERR_CANT_CREATE;
	}

	f->close();
	//memdelete(f);

	return OK;
}



Error ResourceFormatSaverText::save(const String &p_path,const RES& p_resource,uint32_t p_flags) {

	if (p_path.ends_with(".sct") && p_resource->get_type()!="PackedScene") {
		return ERR_FILE_UNRECOGNIZED;
	}

	ResourceFormatSaverTextInstance saver;
	return saver.save(p_path,p_resource,p_flags);

}

bool ResourceFormatSaverText::recognize(const RES& p_resource) const {


	return true; // all recognized!
}
void ResourceFormatSaverText::get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const {

	p_extensions->push_back("tres"); //text resource
	if (p_resource->get_type()=="PackedScene")
		p_extensions->push_back("tscn"); //text scene

}

ResourceFormatSaverText* ResourceFormatSaverText::singleton=NULL;
ResourceFormatSaverText::ResourceFormatSaverText() {
	singleton=this;
}

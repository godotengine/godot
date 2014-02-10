/*************************************************************************/
/*  object_format_xml.cpp                                                */
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
#ifdef XML_ENABLED
#ifdef OLD_SCENE_FORMAT_ENABLED
#include "object_format_xml.h"
#include "resource.h"
#include "io/resource_loader.h"
#include "print_string.h"
#include "object_type_db.h"
#include "globals.h"
#include "os/os.h"
#include "version.h"

void ObjectFormatSaverXML::escape(String& p_str) {
	
	p_str=p_str.replace("&","&amp;");
	p_str=p_str.replace("<","&gt;");
	p_str=p_str.replace(">","&lt;");
	p_str=p_str.replace("'","&apos;");
	p_str=p_str.replace("\"","&quot;");
	for (int i=1;i<32;i++) {
		
		char chr[2]={i,0};
		p_str=p_str.replace(chr,"&#"+String::num(i)+";");
	}

	
}
void ObjectFormatSaverXML::write_tabs(int p_diff) {

	for (int i=0;i<depth+p_diff;i++) {

		f->store_8('\t');
	}
}

void ObjectFormatSaverXML::write_string(String p_str,bool p_escape) {
	
	/* write an UTF8 string */
	if (p_escape)
		escape(p_str);
	
	f->store_string(p_str);;
	/*
	CharString cs=p_str.utf8();
	const char *data=cs.get_data();
		
	while (*data) {
		f->store_8(*data);
		data++;
	}*/
	
	
}	

void ObjectFormatSaverXML::enter_tag(const String& p_section,const String& p_args) {
	
	if (p_args.length())
		write_string("<"+p_section+" "+p_args+">",false);
	else
		write_string("<"+p_section+">",false);
	depth++;
}
void ObjectFormatSaverXML::exit_tag(const String& p_section) {
	
	depth--;	
	write_string("</"+p_section+">",false);

}

/*
static bool _check_type(const Variant& p_property) {

	if (p_property.get_type()==Variant::_RID)
		return false;
	if (p_property.get_type()==Variant::OBJECT) {
		RES res = p_property;
		if (res.is_null())
			return false;
	}

	return true;
}*/

void ObjectFormatSaverXML::write_property(const String& p_name,const Variant& p_property,bool *r_ok) {

	if (r_ok)
		*r_ok=false;
	
	String type;
	String params;
	bool oneliner=true;
	
	switch( p_property.get_type() ) {
			
		case Variant::NIL: 		type="nil"; break;
		case Variant::BOOL:		type="bool"; break;
		case Variant::INT: 		type="int"; break;
		case Variant::REAL:		type="real"; break;
		case Variant::STRING:		type="string"; break;
		case Variant::VECTOR2:		type="vector2"; break;
		case Variant::RECT2:		type="rect2"; break;
		case Variant::VECTOR3:		type="vector3"; break;
		case Variant::PLANE:		type="plane"; break;
		case Variant::_AABB:		type="aabb"; break;
		case Variant::QUAT:		type="quaternion"; break;
		case Variant::MATRIX32:		type="matrix32"; break;
		case Variant::MATRIX3:		type="matrix3"; break;
		case Variant::TRANSFORM:		type="transform"; break;
		case Variant::COLOR:		type="color"; break;
		case Variant::IMAGE: {			
			type="image"; 
			Image img=p_property;
			if (img.empty()) {
				enter_tag(type,"name=\""+p_name+"\"");
				exit_tag(type);
				if (r_ok)
					*r_ok=true;
				return;
			}
			params+="encoding=\"raw\"";
			params+=" width=\""+itos(img.get_width())+"\"";
			params+=" height=\""+itos(img.get_height())+"\"";
			params+=" mipmaps=\""+itos(img.get_mipmaps())+"\"";

			switch(img.get_format()) {
				
				case Image::FORMAT_GRAYSCALE: params+=" format=\"grayscale\""; break;
				case Image::FORMAT_INTENSITY: params+=" format=\"intensity\""; break;
				case Image::FORMAT_GRAYSCALE_ALPHA: params+=" format=\"grayscale_alpha\""; break;
				case Image::FORMAT_RGB: params+=" format=\"rgb\""; break;
				case Image::FORMAT_RGBA: params+=" format=\"rgba\""; break;
				case Image::FORMAT_INDEXED : params+=" format=\"indexed\""; break;
				case Image::FORMAT_INDEXED_ALPHA: params+=" format=\"indexed_alpha\""; break;
				case Image::FORMAT_BC1: params+=" format=\"bc1\""; break;
				case Image::FORMAT_BC2: params+=" format=\"bc2\""; break;
				case Image::FORMAT_BC3: params+=" format=\"bc3\""; break;
				case Image::FORMAT_BC4: params+=" format=\"bc4\""; break;
				case Image::FORMAT_BC5: params+=" format=\"bc5\""; break;
				case Image::FORMAT_CUSTOM: params+=" format=\"custom\" custom_size=\""+itos(img.get_data().size())+"\""; break;
				default: {}
			}
		} break;
		case Variant::NODE_PATH:		type="node_path"; break;			
		case Variant::OBJECT:	{
			type="resource"; 
			RES res = p_property;
			if (res.is_null()) {
				enter_tag(type,"name=\""+p_name+"\"");
				exit_tag(type);
				if (r_ok)
					*r_ok=true;

				return; // don't save it
			}
				
			params="resource_type=\""+res->get_type()+"\"";

			if (res->get_path().length() && res->get_path().find("::")==-1) {
				//external resource
				String path=relative_paths?local_path.path_to_file(res->get_path()):res->get_path();
				escape(path);
				params+=" path=\""+path+"\"";
			} else {

				//internal resource
				ERR_EXPLAIN("Resource was not pre cached for the resource section, bug?");
				ERR_FAIL_COND(!resource_map.has(res));

				params+=" path=\"local://"+itos(resource_map[res])+"\"";
			}
			
		} break;
		case Variant::INPUT_EVENT:	type="input_event"; break;
		case Variant::DICTIONARY:	type="dictionary" ; oneliner=false; break;
		case Variant::ARRAY:		type="array"; params="len=\""+itos(p_property.operator Array().size())+"\""; oneliner=false; break;
		
		case Variant::RAW_ARRAY:		type="raw_array"; params="len=\""+itos(p_property.operator DVector < uint8_t >().size())+"\""; break;
		case Variant::INT_ARRAY:		type="int_array"; params="len=\""+itos(p_property.operator DVector < int >().size())+"\""; break;
		case Variant::REAL_ARRAY:	type="real_array"; params="len=\""+itos(p_property.operator DVector < real_t >().size())+"\""; break;
		case Variant::STRING_ARRAY:	type="string_array"; params="len=\""+itos(p_property.operator DVector < String >().size())+"\""; break;
		case Variant::VECTOR2_ARRAY:	type="vector2_array"; params="len=\""+itos(p_property.operator DVector < Vector2 >().size())+"\""; break;
		case Variant::VECTOR3_ARRAY:	type="vector3_array"; params="len=\""+itos(p_property.operator DVector < Vector3 >().size())+"\""; break;
		case Variant::COLOR_ARRAY:	type="color_array"; params="len=\""+itos(p_property.operator DVector < Color >().size())+"\""; break;
		default: {
		
			ERR_PRINT("Unknown Variant type.");
			ERR_FAIL();
		}
			
	}

	write_tabs();

	if (p_name!="") {
		if (params.length())
			enter_tag(type,"name=\""+p_name+"\" "+params);
		else
			enter_tag(type,"name=\""+p_name+"\"");
	} else {
		if (params.length())
			enter_tag(type," "+params);
		else
			enter_tag(type,"");
	}

	if (!oneliner)
		write_string("\n",false);
	else
		write_string(" ",false);
	

	switch( p_property.get_type() ) {

		case Variant::NIL: {
			
		} break;
		case Variant::BOOL: {
			
			write_string( p_property.operator bool() ? "True":"False" );
		} break;
		case Variant::INT: {
			
			write_string( itos(p_property.operator int()) );
		} break;
		case Variant::REAL: {
			
			write_string( rtos(p_property.operator real_t()) );
		} break;
		case Variant::STRING: {
			
			String str=p_property;
			escape(str);
			str="\""+str+"\"";
			write_string( str,false );
		} break;
		case Variant::VECTOR2: {
			
			Vector2 v = p_property;
			write_string( rtoss(v.x) +", "+rtoss(v.y) );
		} break;
		case Variant::RECT2: {
		
			Rect2 aabb = p_property;
			write_string( rtoss(aabb.pos.x) +", "+rtoss(aabb.pos.y) +", "+rtoss(aabb.size.x) +", "+rtoss(aabb.size.y) );
		
		} break;
		case Variant::VECTOR3: {
			
			Vector3 v = p_property;
			write_string( rtoss(v.x) +", "+rtoss(v.y)+", "+rtoss(v.z) );
		} break;
		case Variant::PLANE: {
			
			Plane p = p_property;
			write_string( rtoss(p.normal.x) +", "+rtoss(p.normal.y)+", "+rtoss(p.normal.z)+", "+rtoss(p.d) );
			
		} break;
		case Variant::_AABB: {
			
			AABB aabb = p_property;
			write_string( rtoss(aabb.pos.x) +", "+rtoss(aabb.pos.y) +", "+rtoss(aabb.pos.z) +", "+rtoss(aabb.size.x) +", "+rtoss(aabb.size.y) +", "+rtoss(aabb.size.z)  );
			
		} break; 
		case Variant::QUAT: {
			
			Quat quat = p_property;
			write_string( rtoss(quat.x)+", "+rtoss(quat.y)+", "+rtoss(quat.z)+", "+rtoss(quat.w)+", ");
			
		} break; 
		case Variant::MATRIX32: {

			String s;
			Matrix32 m3 = p_property;
			for (int i=0;i<3;i++) {
				for (int j=0;j<2;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}

			write_string(s);

		} break;
		case Variant::MATRIX3: {
			
			String s;
			Matrix3 m3 = p_property;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {
					
					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}
					
			write_string(s);
			     
		} break;
		case Variant::TRANSFORM: {
			
			String s;
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
			
			write_string(s);
		} break;
			
			// misc types		
		case Variant::COLOR: {
			
			Color c = p_property;
			write_string( rtoss(c.r) +", "+rtoss(c.g)+", "+rtoss(c.b)+", "+rtoss(c.a) );
			
		} break;
		case Variant::IMAGE: {
			
			String s;
			Image img = p_property;
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
			
			write_string(s);
		} break;
		case Variant::NODE_PATH: {
			
			String str=p_property;
			escape(str);
			str="\""+str+"\"";
			write_string( str,false);
			
		} break;

		case Variant::OBJECT: {
			/* this saver does not save resources in here
			RES res = p_property;
			
			if (!res.is_null()) {
				
				String path=res->get_path();				
				if (!res->is_shared() || !path.length()) {
					// if no path, or path is from inside a scene
					write_object( *res );
				}
				
			}
			*/
								
		} break;
		case Variant::INPUT_EVENT: {
			
			write_string( p_property.operator String() );
		} break;
		case Variant::DICTIONARY: {
			
			Dictionary dict = p_property;
			

			List<Variant> keys;
			dict.get_key_list(&keys);
			
			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				//if (!_check_type(dict[E->get()]))
				//	continue;

				bool ok;
				write_property("",E->get(),&ok);
				ERR_CONTINUE(!ok);

				write_property("",dict[E->get()],&ok);
				if (!ok)
					write_property("",Variant()); //at least make the file consistent..
			}
			
			
			
			
		} break;
		case Variant::ARRAY: {
			
			Array array = p_property;
			int len=array.size();
			for (int i=0;i<len;i++) {
				
				write_property("",array[i]);
				
			}
			
		} break;
			
		case Variant::RAW_ARRAY: {
			
			String s;
			DVector<uint8_t> data = p_property;
			int len = data.size();
			DVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();;
			for (int i=0;i<len;i++) {
				
				uint8_t byte = ptr[i];
				const char  hex[16]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
				char str[3]={ hex[byte>>4], hex[byte&0xF], 0};
				s+=str;				
			}
			
			write_string(s,false);
			
		} break;
		case Variant::INT_ARRAY: {
			
			DVector<int> data = p_property;
			int len = data.size();
			DVector<int>::Read r = data.read();
			const int *ptr=r.ptr();;
			write_tabs();
			
			for (int i=0;i<len;i++) {
				
				if (i>0)
					write_string(", ",false);

				write_string(itos(ptr[i]),false);
			}
			


		} break;
		case Variant::REAL_ARRAY: {
			
			DVector<real_t> data = p_property;
			int len = data.size();
			DVector<real_t>::Read r = data.read();
			const real_t *ptr=r.ptr();;
			write_tabs();

			for (int i=0;i<len;i++) {
				
				if (i>0)
					write_string(", ",false);
				write_string(rtoss(ptr[i]),false);
			}


		} break;
		case Variant::STRING_ARRAY: {
			
			DVector<String> data = p_property;
			int len = data.size();
			DVector<String>::Read r = data.read();
			const String *ptr=r.ptr();;
			String s;
			
			for (int i=0;i<len;i++) {
			
				if (i>0)
					s+=", ";
				String str=ptr[i];
				escape(str);
				s=s+"\""+str+"\"";
			}
			
			write_string(s,false);
			
		} break;
		case Variant::VECTOR2_ARRAY: {

			DVector<Vector2> data = p_property;
			int len = data.size();
			DVector<Vector2>::Read r = data.read();
			const Vector2 *ptr=r.ptr();;
			write_tabs();

			for (int i=0;i<len;i++) {

				if (i>0)
					write_string(", ",false);
				write_string(rtoss(ptr[i].x),false);
				write_string(", "+rtoss(ptr[i].y),false);

			}


		} break;
		case Variant::VECTOR3_ARRAY: {
			
			DVector<Vector3> data = p_property;
			int len = data.size();
			DVector<Vector3>::Read r = data.read();
			const Vector3 *ptr=r.ptr();;
			write_tabs();

			for (int i=0;i<len;i++) {
				
				if (i>0)
					write_string(", ",false);
				write_string(rtoss(ptr[i].x),false);
				write_string(", "+rtoss(ptr[i].y),false);
				write_string(", "+rtoss(ptr[i].z),false);

			}
			

		} break;
		case Variant::COLOR_ARRAY: {
			
			DVector<Color> data = p_property;
			int len = data.size();
			DVector<Color>::Read r = data.read();
			const Color *ptr=r.ptr();;
			write_tabs();
			
			for (int i=0;i<len;i++) {
				
				if (i>0)
					write_string(", ",false);

				write_string(rtoss(ptr[i].r),false);
				write_string(", "+rtoss(ptr[i].g),false);
				write_string(", "+rtoss(ptr[i].b),false);
				write_string(", "+rtoss(ptr[i].a),false);

			}
			
		} break;
		default: {}
		
	}
	if (oneliner)
		write_string(" ");
	else
		write_tabs(-1);
	exit_tag(type);

	write_string("\n",false);

	if (r_ok)
		*r_ok=true;

}


void ObjectFormatSaverXML::_find_resources(const Variant& p_variant) {


	switch(p_variant.get_type()) {
		case Variant::OBJECT: {



			RES res = p_variant.operator RefPtr();

			if (res.is_null())
				return;

			if (!bundle_resources && res->get_path().length() && res->get_path().find("::") == -1 )
				return;

			if (resource_map.has(res))
				return;

			List<PropertyInfo> property_list;

			res->get_property_list( &property_list );

			List<PropertyInfo>::Element *I=property_list.front();

			while(I) {

				PropertyInfo pi=I->get();

				if (pi.usage&PROPERTY_USAGE_STORAGE || (bundle_resources && pi.usage&PROPERTY_USAGE_BUNDLE)) {

					Variant v=res->get(I->get().name);
					_find_resources(v);
				}

				I=I->next();
			}

			resource_map[ res ] = resource_map.size(); //saved after, so the childs it needs are available when loaded
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



Error ObjectFormatSaverXML::save(const Object *p_object,const Variant &p_meta) {

	ERR_FAIL_COND_V(!f,ERR_UNCONFIGURED);
	ERR_EXPLAIN("write_object should supply either an object, a meta, or both");
	ERR_FAIL_COND_V(!p_object && p_meta.get_type()==Variant::NIL, ERR_INVALID_PARAMETER);

	SavedObject *so = memnew( SavedObject );

	if (p_object)
		so->type=p_object->get_type();

	_find_resources(p_meta);
	so->meta=p_meta;

	if (p_object) {


		if (optimizer.is_valid()) {
			//use optimizer

			List<OptimizedSaver::Property> props;
			optimizer->get_property_list(p_object,&props);

			for(List<OptimizedSaver::Property>::Element *E=props.front();E;E=E->next()) {

				if (skip_editor && String(E->get().name).begins_with("__editor"))
					continue;
				_find_resources(E->get().value);
				SavedObject::SavedProperty sp;
				sp.name=E->get().name;
				sp.value=E->get().value;
				so->properties.push_back(sp);
			}

		} else {
			//use classic way
			List<PropertyInfo> property_list;
			p_object->get_property_list( &property_list );

			for(List<PropertyInfo>::Element *E=property_list.front();E;E=E->next()) {

				if (skip_editor && E->get().name.begins_with("__editor"))
					continue;
				if (E->get().usage&PROPERTY_USAGE_STORAGE || (bundle_resources && E->get().usage&PROPERTY_USAGE_BUNDLE)) {

					SavedObject::SavedProperty sp;
					sp.name=E->get().name;
					sp.value = p_object->get(E->get().name);
					_find_resources(sp.value);
					so->properties.push_back(sp);
				}
			}
		}

	}

	saved_objects.push_back(so);

	return OK;
}

ObjectFormatSaverXML::ObjectFormatSaverXML(FileAccess *p_file,const String& p_magic,const String& p_local_path,uint32_t p_flags,const Ref<OptimizedSaver>& p_optimizer) {

	optimizer=p_optimizer;
	relative_paths=p_flags&ObjectSaver::FLAG_RELATIVE_PATHS;
	skip_editor=p_flags&ObjectSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	bundle_resources=p_flags&ObjectSaver::FLAG_BUNDLE_RESOURCES;
	f=p_file; // should be already opened
	depth=0;	
	local_path=p_local_path;
	magic=p_magic;
}
ObjectFormatSaverXML::~ObjectFormatSaverXML() {
	
	write_string("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>",false); //no escape
	write_string("\n",false);
	enter_tag("object_file","magic=\""+magic+"\" "+"version=\""+itos(VERSION_MAJOR)+"."+itos(VERSION_MINOR)+"\" version_name=\""+VERSION_FULL_NAME+"\"");
	write_string("\n",false);

	// save resources

	for(List<RES>::Element *E=saved_resources.front();E;E=E->next()) {

		RES res = E->get();
		ERR_CONTINUE(!resource_map.has(res));

		write_tabs();
		if (res->get_path().length() && res->get_path().find("::") == -1 )
			enter_tag("resource","type=\""+res->get_type()+"\" path=\""+res->get_path()+"\""); //bundled
		else
			enter_tag("resource","type=\""+res->get_type()+"\" path=\"local://"+itos(resource_map[res])+"\"");

		if (optimizer.is_valid()) {

			List<OptimizedSaver::Property> props;
			optimizer->get_property_list(res.ptr(),&props);

			for(List<OptimizedSaver::Property>::Element *E=props.front();E;E=E->next()) {

				if (skip_editor && String(E->get().name).begins_with("__editor"))
					continue;

				write_property(E->get().name,E->get().value);
			}


		} else {

			List<PropertyInfo> property_list;
			res->get_property_list(&property_list);
			for(List<PropertyInfo>::Element *PE = property_list.front();PE;PE=PE->next()) {


				if (skip_editor && PE->get().name.begins_with("__editor"))
					continue;

				if (PE->get().usage&PROPERTY_USAGE_STORAGE || (bundle_resources && PE->get().usage&PROPERTY_USAGE_BUNDLE)) {

					String name = PE->get().name;
					Variant value = res->get(name);
					write_property(name,value);
				}


			}

		}
		write_tabs(-1);
		exit_tag("resource");
		write_string("\n",false);
	}

	if (!saved_objects.empty()) {


		for(List<SavedObject*>::Element *E=saved_objects.front();E;E=E->next()) {

			SavedObject *so = E->get();



			write_tabs();
			if (so->type!="")
				enter_tag("object","type=\""+so->type+"\"");
			else
				enter_tag("object");
			write_string("\n",false);

			if (so->meta.get_type()!=Variant::NIL) {

				write_property("__xml_meta__",so->meta);

			}

			List<SavedObject::SavedProperty>::Element *SE = so->properties.front();

			while(SE) {

				write_property(SE->get().name,SE->get().value);
				SE=SE->next();
			}


			write_tabs(-1);
			exit_tag("object");
			write_string("\n",false);
			memdelete(so); //no longer needed
		}


	} else {

		WARN_PRINT("File contains no saved objects.");
	}

	exit_tag("object_file");
	f->close();
	memdelete(f);
}


ObjectFormatSaver* ObjectFormatSaverInstancerXML::instance(const String& p_file,const String& p_magic,uint32_t p_flags,const Ref<OptimizedSaver>& p_optimizer) {
	
	Error err;
	FileAccess *f = FileAccess::open(p_file, FileAccess::WRITE,&err);

	ERR_FAIL_COND_V( err, NULL );
	String local_path = Globals::get_singleton()->localize_path(p_file);		

	return memnew( ObjectFormatSaverXML( f, p_magic,local_path,p_flags,p_optimizer ) );
}	

void ObjectFormatSaverInstancerXML::get_recognized_extensions(List<String> *p_extensions) const {
	
	p_extensions->push_back("xml");
}


ObjectFormatSaverInstancerXML::~ObjectFormatSaverInstancerXML() {
	
	
}

/************************************************/
/************************************************/
/************************************************/
/************************************************/
/************************************************/



#ifdef OPTIMIZED_XML_LOADER

#define IS_FLOAT_CHAR(m_c) \
	((m_c>='0' && m_c<='9') || m_c=='e' || m_c=='-' || m_c=='+' || m_c=='.')

#define XML_FAIL(m_cond,m_err) \
	if (m_cond) {\
		ERR_EXPLAIN(local_path+":"+itos(parser->get_current_line())+": "+String(m_err));\
		ERR_FAIL_COND_V( m_cond, ERR_FILE_CORRUPT );\
	}


Error ObjectFormatLoaderXML::_parse_property(Variant& r_v,String& r_name) {

	XML_FAIL( parser->is_empty(), "unexpected empty tag");

	String type=parser->get_node_name();
	String name=parser->get_attribute_value_safe("name");

	r_v=Variant();
	r_name=name;

	if (type=="dictionary") {

		Dictionary d;
		int reading=0;
		Variant key;
		while(parser->read()==OK) {

			if (parser->get_node_type()==XMLParser::NODE_ELEMENT) {
				Error err;
				String tagname;

				if (reading==0) {

					err=_parse_property(key,tagname);
					XML_FAIL( err,"error parsing dictionary key: "+name);
					reading++;
				} else {

					reading=0;
					Variant value;
					err=_parse_property(value,tagname);
					XML_FAIL( err,"error parsing dictionary value: "+name);
					d[key]=value;
				}

			} else if (parser->get_node_type()==XMLParser::NODE_ELEMENT_END && parser->get_node_name()=="dictionary") {
				r_v=d;
				return OK;
			}
		}


		XML_FAIL( true, "unexpected end of file while reading dictionary: "+name);

	} else if (type=="array") {

		XML_FAIL( !parser->has_attribute("len"), "array missing 'len' attribute");

		int len=parser->get_attribute_value("len").to_int();

		Array array;
		array.resize(len);


		Variant v;
		String tagname;
		int idx=0;


		while(parser->read()==OK) {

			if (parser->get_node_type()==XMLParser::NODE_ELEMENT) {

				XML_FAIL( idx >= len, "array size mismatch (too many elements)");
				Error err;
				String tagname;
				Variant key;

				err=_parse_property(key,tagname);
				XML_FAIL( err,"error parsing element of array: "+name);
				array[idx]=key;
				idx++;


			} else if (parser->get_node_type()==XMLParser::NODE_ELEMENT_END && parser->get_node_name()=="array") {

				XML_FAIL( idx != len, "array size mismatch (not "+itos(len)+"):"+name);
				r_v=array;
				return OK;
			}
		}

		XML_FAIL( true, "unexpected end of file while reading dictionary: "+name);

	} else if (type=="resource") {


		XML_FAIL(!parser->has_attribute("path"),"resource property has no 'path' set (embedding not supported).")

		String path=parser->get_attribute_value("path");
		String hint = parser->get_attribute_value_safe("resource_type");

		if (path.begins_with("local://"))
			path=path.replace("local://",local_path+"::");
		else if (path.find("://")==-1 && path.is_rel_path()) {
			// path is relative to file being loaded, so convert to a resource path
			path=Globals::get_singleton()->localize_path(local_path.get_base_dir()+"/"+path);

		}

		//take advantage of the resource loader cache. The resource is cached on it, even if
		RES res=ResourceLoader::load(path,hint);


		if (res.is_null()) {

			WARN_PRINT(String("Couldn't load resource: "+path).ascii().get_data());
		}

		r_v=res.get_ref_ptr();

	} else if (type=="image") {

		if (parser->has_attribute("encoding")) { //there is image data

			String encoding=parser->get_attribute_value("encoding");

			if (encoding=="raw") {

				//raw image (bytes)

				XML_FAIL( !parser->has_attribute("width"), "missing attribute in raw encoding: 'width'.");
				XML_FAIL( !parser->has_attribute("height"), "missing attribute in raw encoding: 'height'.");
				XML_FAIL( !parser->has_attribute("format"), "missing attribute in raw encoding: 'format'.");

				String format = parser->get_attribute_value("format");
				String width = parser->get_attribute_value("width");
				String height = parser->get_attribute_value("height");

				Image::Format imgformat;
				int chans=0;
				int pal=0;

				if (format=="grayscale") {
					imgformat=Image::FORMAT_GRAYSCALE;
					chans=1;
				} else if (format=="intensity") {
					imgformat=Image::FORMAT_INTENSITY;
					chans=1;
				} else if (format=="grayscale_alpha") {
					imgformat=Image::FORMAT_GRAYSCALE_ALPHA;
					chans=2;
				} else if (format=="rgb") {
					imgformat=Image::FORMAT_RGB;
					chans=3;
				} else if (format=="rgba") {
					imgformat=Image::FORMAT_RGBA;
					chans=4;
				} else if (format=="indexed") {
					imgformat=Image::FORMAT_INDEXED;
					chans=1;
					pal=256*3;
				} else if (format=="indexed_alpha") {
					imgformat=Image::FORMAT_INDEXED_ALPHA;
					chans=1;
					pal=256*4;
				} else {

					XML_FAIL(true, "invalid format for image: "+format);

				}

				XML_FAIL( chans==0, "invalid number of color channels in image (0).");

				int w=width.to_int();
				int h=height.to_int();

				if (w == 0 && w == 0) { //epmty, don't even bother
					//r_v = Image(w, h, imgformat);
					r_v=Image();
					return OK;
				} else {

					//decode hexa

					DVector<uint8_t> pixels;
					pixels.resize(chans*w*h+pal);
					int pixels_size=pixels.size();
					XML_FAIL( pixels_size==0, "corrupt");

					ERR_FAIL_COND_V(pixels_size==0,ERR_FILE_CORRUPT);
					DVector<uint8_t>::Write wr=pixels.write();
					uint8_t *bytes=wr.ptr();

					XML_FAIL( parser->read()!=OK, "error reading" );
					XML_FAIL( parser->get_node_type()!=XMLParser::NODE_TEXT, "expected text!");

					String text = parser->get_node_data().strip_edges();
					XML_FAIL( text.length()/2 != pixels_size, "unexpected image data size" );

					for(int i=0;i<pixels_size*2;i++) {

						uint8_t byte;
						CharType c=text[i];

						if ( (c>='0' && c<='9') || (c>='A' && c<='F') || (c>='a' && c<='f') ) {

							if (i&1) {

								byte|=HEX2CHR(c);
								bytes[i>>1]=byte;
							} else {

								byte=HEX2CHR(c)<<4;
							}


						}
					}

					wr=DVector<uint8_t>::Write();
					r_v=Image(w,h,imgformat,pixels);
				}
			}

		} else {
			r_v=Image(); // empty image, since no encoding defined
		}

	} else if (type=="raw_array") {

		XML_FAIL( !parser->has_attribute("len"), "array missing 'len' attribute");

		int len=parser->get_attribute_value("len").to_int();
		if (len>0) {

			XML_FAIL( parser->read()!=OK, "error reading" );
			XML_FAIL( parser->get_node_type()!=XMLParser::NODE_TEXT, "expected text!");
			String text = parser->get_node_data();

			XML_FAIL( text.length() != len*2, "raw array length mismatch" );

			DVector<uint8_t> bytes;
			bytes.resize(len);
			DVector<uint8_t>::Write w=bytes.write();
			uint8_t *bytesptr=w.ptr();


			for(int i=0;i<len*2;i++) {

				uint8_t byte;
				CharType c=text[i];

				if ( (c>='0' && c<='9') || (c>='A' && c<='F') || (c>='a' && c<='f') ) {

					if (i&1) {

						byte|=HEX2CHR(c);
						bytesptr[i>>1]=byte;
					} else {

						byte=HEX2CHR(c)<<4;
					}
				}
			}

			w=DVector<uint8_t>::Write();
			r_v=bytes;
		}

	} else if (type=="int_array") {

		int len=parser->get_attribute_value("len").to_int();

		if (len>0) {

			XML_FAIL( parser->read()!=OK, "error reading" );
			XML_FAIL( parser->get_node_type()!=XMLParser::NODE_TEXT, "expected text!");
			String text = parser->get_node_data();

			const CharType *c=text.c_str();
			DVector<int> varray;
			varray.resize(len);
			DVector<int>::Write w = varray.write();

			int idx=0;
			const CharType *from=c-1;

			while(*c) {

				bool ischar = (*c >='0' && *c<='9') || *c=='+' || *c=='-';
				if (!ischar) {

					if (int64_t(c-from)>1) {

						int i = String::to_int(from+1,int64_t(c-from));
						w[idx++]=i;
					}

					from=c;
				} else {

					XML_FAIL( idx >= len, "array too big");
				}

				c++;
			}

			XML_FAIL( idx != len, "array size mismatch");

			w = varray.write();
			r_v=varray;
		}



	} else if (type=="real_array") {

		int len=parser->get_attribute_value("len").to_int();

		if (len>0) {

			XML_FAIL( parser->read()!=OK, "error reading" );
			XML_FAIL( parser->get_node_type()!=XMLParser::NODE_TEXT, "expected text!");
			String text = parser->get_node_data();

			const CharType *c=text.c_str();
			DVector<real_t> varray;
			varray.resize(len);
			DVector<real_t>::Write w = varray.write();

			int idx=0;
			const CharType *from=c-1;

			while(*c) {

				bool ischar = IS_FLOAT_CHAR((*c));
				if (!ischar) {

					if (int64_t(c-from)>1) {

						real_t f = String::to_double(from+1,int64_t(c-from));
						w[idx++]=f;
					}

					from=c;
				} else {

					XML_FAIL( idx >= len, "array too big");
				}

				c++;
			}

			XML_FAIL( idx != len, "array size mismatch");

			w = varray.write();
			r_v=varray;
		}

	} else if (type=="string_array") {


		// this is invalid xml, and will have to be fixed at some point..

		int len=parser->get_attribute_value("len").to_int();

		if (len>0) {

			XML_FAIL( parser->read()!=OK, "error reading" );
			XML_FAIL( parser->get_node_type()!=XMLParser::NODE_TEXT, "expected text!");
			String text = parser->get_node_data();

			const CharType *c=text.c_str();
			DVector<String> sarray;
			sarray.resize(len);
			DVector<String>::Write w = sarray.write();


			bool inside=false;
			const CharType *from=c;
			int idx=0;

			while(*c) {

				if (inside) {

					if (*c == '"') {
						inside=false;
						String s = String(from,int64_t(c-from));
						w[idx]=s;
						idx++;
					}
				} else {

					if (*c == '"') {
						inside=true;
						from=c+1;
						XML_FAIL( idx>=len, "string array is too big!!: "+name);
					}
				}

				c++;
			}

			XML_FAIL( inside, "unterminated string array: "+name);
			XML_FAIL( len != idx, "string array size mismatch: "+name);

			w = DVector<String>::Write();

			r_v=sarray;

		}
	} else if (type=="vector3_array") {

		int len=parser->get_attribute_value("len").to_int();

		if (len>0) {

			XML_FAIL( parser->read()!=OK, "error reading" );
			XML_FAIL( parser->get_node_type()!=XMLParser::NODE_TEXT, "expected text!");
			String text = parser->get_node_data();

			const CharType *c=text.c_str();
			DVector<Vector3> varray;
			varray.resize(len);
			DVector<Vector3>::Write w = varray.write();

			int idx=0;
			int sidx=0;
			Vector3 v;
			const CharType *from=c-1;

			while(*c) {

				bool ischar = IS_FLOAT_CHAR((*c));
				if (!ischar) {

					if (int64_t(c-from)>1) {

						real_t f = String::to_double(from+1,int64_t(c-from));
						v[sidx++]=f;
						if (sidx==3) {
							w[idx++]=v;
							sidx=0;

						}
					}

					from=c;
				} else {

					XML_FAIL( idx >= len, "array too big");
				}

				c++;
			}

			XML_FAIL( idx != len, "array size mismatch");

			w = varray.write();
			r_v=varray;
		}

	} else if (type=="color_array") {

		int len=parser->get_attribute_value("len").to_int();

		if (len>0) {

			XML_FAIL( parser->read()!=OK, "error reading" );
			XML_FAIL( parser->get_node_type()!=XMLParser::NODE_TEXT, "expected text!");
			String text = parser->get_node_data();

			const CharType *c=text.c_str();
			DVector<Color> carray;
			carray.resize(len);
			DVector<Color>::Write w = carray.write();

			int idx=0;
			int sidx=0;
			Color v;
			const CharType *from=c-1;

			while(*c) {

				bool ischar = IS_FLOAT_CHAR((*c));
				if (!ischar) {

					if (int64_t(c-from)>1) {

						real_t f = String::to_double(from+1,int64_t(c-from));
						v[sidx++]=f;
						if (sidx==4) {
							w[idx++]=v;
							sidx=0;

						}
					}

					from=c;
				} else {

					XML_FAIL( idx >= len, "array too big");
				}

				c++;
			}

			XML_FAIL( idx != len, "array size mismatch");

			w = carray.write();
			r_v=carray;
		}
	} else {
		// simple string parsing code
		XML_FAIL( parser->read()!=OK, "can't read data" );

		String data=parser->get_node_data();
		data=data.strip_edges();

		if (type=="nil") {
			// uh do nothing

		} else if (type=="bool") {
			// uh do nothing
			if (data.nocasecmp_to("true")==0 || data.to_int()!=0)
				r_v=true;
			else
				r_v=false;

		} else if (type=="int") {

			r_v=data.to_int();
		} else if (type=="real") {

			r_v=data.to_double();
		} else if (type=="string") {

			String str=data;
			str=str.substr(1,str.length()-2);
			r_v=str;
		} else if (type=="vector3") {

			r_v=Vector3(
					data.get_slice(",",0).to_double(),
					data.get_slice(",",1).to_double(),
					data.get_slice(",",2).to_double()
				   );

		} else if (type=="vector2") {


			r_v=Vector2(
					data.get_slice(",",0).to_double(),
					data.get_slice(",",1).to_double()
				   );

		} else if (type=="plane") {

			r_v=Plane(
					data.get_slice(",",0).to_double(),
					data.get_slice(",",1).to_double(),
					data.get_slice(",",2).to_double(),
					data.get_slice(",",3).to_double()
				 );

		} else if (type=="quaternion") {

			r_v=Quat(
					data.get_slice(",",0).to_double(),
					data.get_slice(",",1).to_double(),
					data.get_slice(",",2).to_double(),
					data.get_slice(",",3).to_double()
				 );

		} else if (type=="rect2") {

			r_v=Rect2(
				Vector2(
					data.get_slice(",",0).to_double(),
					data.get_slice(",",1).to_double()
				),
				Vector2(
					data.get_slice(",",2).to_double(),
					data.get_slice(",",3).to_double()
				)
			);


		} else if (type=="aabb") {

			r_v=AABB(
				Vector3(
					data.get_slice(",",0).to_double(),
					data.get_slice(",",1).to_double(),
					data.get_slice(",",2).to_double()
				),
				Vector3(
					data.get_slice(",",3).to_double(),
					data.get_slice(",",4).to_double(),
					data.get_slice(",",5).to_double()
				)
			);


		} else if (type=="matrix3") {

			Matrix3 m3;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {
					m3.elements[i][j]=data.get_slice(",",i*3+j).to_double();
				}
			}
			r_v=m3;

		} else if (type=="transform") {

			Transform tr;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {
					tr.basis.elements[i][j]=data.get_slice(",",i*3+j).to_double();
				}

			}
			tr.origin=Vector3(
				     data.get_slice(",",9).to_double(),
				     data.get_slice(",",10).to_double(),
				     data.get_slice(",",11).to_double()
				   );
			r_v=tr;

		} else if (type=="color") {

			r_v=Color(
				   data.get_slice(",",0).to_double(),
				   data.get_slice(",",1).to_double(),
				   data.get_slice(",",2).to_double(),
				   data.get_slice(",",3).to_double()
				 );

		} else if (type=="node_path") {

			String str=data;
			str=str.substr(1,str.length()-2);
			r_v=NodePath( str );

		} else if (type=="input_event") {

			// ?
		} else {

			XML_FAIL(true,"unrecognized property tag: "+type);
		}
	}

	_close_tag(type);

	return OK;
}




Error ObjectFormatLoaderXML::_close_tag(const String& p_tag) {

	int c=1;

	while(parser->read()==OK) {


		if (parser->get_node_type()==XMLParser::NODE_ELEMENT && parser->get_node_name()==p_tag) {
			c++;
		} else if (parser->get_node_type()==XMLParser::NODE_ELEMENT_END && parser->get_node_name()==p_tag) {
			c--;

			if (c==0)
				return OK;
		}

	}

	return ERR_FILE_CORRUPT;
}

Error ObjectFormatLoaderXML::load(Object **p_object,Variant &p_meta)  {

	*p_object=NULL;
	p_meta=Variant();

	while(parser->read()==OK) {

		if (parser->get_node_type()==XMLParser::NODE_ELEMENT) {

			String name = parser->get_node_name();


			XML_FAIL( !parser->has_attribute("type"), "'type' attribute missing." );
			String type = parser->get_attribute_value("type");


			Object *obj=NULL;
			Ref<Resource> resource;
			if (name=="resource") {

				XML_FAIL( !parser->has_attribute("path"), "'path' attribute missing." );
				String path = parser->get_attribute_value("path");

				XML_FAIL(!path.begins_with("local://"),"path does not begin with 'local://'");


				path=path.replace("local://",local_path+"::");

				if (ResourceCache::has(path)) {
					Error err = _close_tag(name);
					XML_FAIL( err, "error skipping resource.");
					continue; //it's a resource, and it's already loaded

				}

				obj = ObjectTypeDB::instance(type);
				XML_FAIL(!obj,"couldn't instance object of type: '"+type+"'");

				Resource *r = obj->cast_to<Resource>();
				XML_FAIL(!obj,"object isn't of type Resource: '"+type+"'");

				resource = RES( r );
				r->set_path(path);


			} else if (name=="object") {


				if (ObjectTypeDB::can_instance(type)) {
					obj = ObjectTypeDB::instance(type);
					XML_FAIL(!obj,"couldn't instance object of type: '"+type+"'");
				} else {

					_close_tag(name);
					return ERR_SKIP;
				};
			} else {
				XML_FAIL(true,"Unknown main tag: "+parser->get_node_name());
			}

			//load properties

			while (parser->read()==OK) {

				if (parser->get_node_type()==XMLParser::NODE_ELEMENT_END && parser->get_node_name()==name)
					break;
				else if (parser->get_node_type()==XMLParser::NODE_ELEMENT) {

					String name;
					Variant v;
					Error err;
					err = _parse_property(v,name);
					XML_FAIL(err,"Error parsing property: "+name);

					if (resource.is_null() && name=="__xml_meta__") {

						p_meta=v;
						continue;
					} else {

						XML_FAIL( !obj, "Normal property found in meta object");

					}

					obj->set(name,v);


				}
			}


			if (!obj) {
				*p_object=NULL;
				return OK; // it was a meta object
			}

			if (resource.is_null()) {
				//regular object
				*p_object=obj;
				return OK;
			} else {

				resource_cache.push_back(resource); //keep it in mem until finished loading and load next
			}


		} else if (parser->get_node_type()==XMLParser::NODE_ELEMENT_END && parser->get_node_name()=="object_file")
			return ERR_FILE_EOF;
	}

	return OK; //never reach anyway
}

ObjectFormatLoaderXML* ObjectFormatLoaderInstancerXML::instance(const String& p_file,const String& p_magic) {

	Ref<XMLParser> parser = memnew( XMLParser );

	Error err = parser->open(p_file);
	ERR_FAIL_COND_V(err,NULL);

	ObjectFormatLoaderXML *loader = memnew( ObjectFormatLoaderXML );

	loader->parser=parser;
	loader->local_path = Globals::get_singleton()->localize_path(p_file);

	while(parser->read()==OK) {

		if (parser->get_node_type()==XMLParser::NODE_ELEMENT && parser->get_node_name()=="object_file") {

			ERR_FAIL_COND_V( parser->is_empty(), NULL );

			String version = parser->get_attribute_value_safe("version");
			String magic = parser->get_attribute_value_safe("MAGIC");

			if (version.get_slice_count(".")!=2) {

				ERR_EXPLAIN("Invalid Version String '"+version+"'' in file: "+p_file);
				ERR_FAIL_V(NULL);
			}

			int major = version.get_slice(".",0).to_int();
			int minor = version.get_slice(".",1).to_int();

			if (major>VERSION_MAJOR || (major==VERSION_MAJOR && minor>VERSION_MINOR)) {

				ERR_EXPLAIN("File Format '"+version+"' is too new! Please upgrade to a a new engine version: "+p_file);
				ERR_FAIL_V(NULL);

			}

			return loader;
		}

	}

	ERR_EXPLAIN("No data found in file!");
	ERR_FAIL_V(NULL);
}

void ObjectFormatLoaderInstancerXML::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("xml");
}



#else

ObjectFormatLoaderXML::Tag* ObjectFormatLoaderXML::parse_tag(bool *r_exit) {
	
	
	while(get_char()!='<' && !f->eof_reached()) {}
	if (f->eof_reached())
		return NULL;
	
	Tag tag;
	bool exit=false;
	if (r_exit)
		*r_exit=false;
	
	bool complete=false;
	while(!f->eof_reached()) {
		
		CharType c=get_char();
		if (c<33 && tag.name.length() && !exit) {
			break;
		} else if (c=='>') {
			complete=true;
			break;
		} else if (c=='/') {
			exit=true;
		} else {
			tag.name+=c;
		}
	}

	if (f->eof_reached())
		return NULL;
	
	if (exit) {
		if (!tag_stack.size()) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Unmatched exit tag </"+tag.name+">");
			ERR_FAIL_COND_V(!tag_stack.size(),NULL);
		}

		if (tag_stack.back()->get().name!=tag.name) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Mismatched exit tag. Got </"+tag.name+">, expected </"+tag_stack.back()->get().name+">");
			ERR_FAIL_COND_V(tag_stack.back()->get().name!=tag.name,NULL);
		}
		                 
		if (!complete) {
			while(get_char()!='>' && !f->eof_reached()) {}
			if (f->eof_reached())
				return NULL;			
		}

		if (r_exit)
			*r_exit=true;

		tag_stack.pop_back();
		return NULL;
		                 
	}
	
	if (!complete) {
		String name;
		String value;
		bool reading_value=false;
		
		while(!f->eof_reached()) {
			
			CharType c=get_char();
			if (c=='>') {
				if (value.length()) {
					
					tag.args[name]=value;
				}
				break;
				
			} else if ( ((!reading_value && (c<33)) || c=='=' || c=='"') && tag.name.length()) {
				
				if (!reading_value && name.length()) {
					
					reading_value=true;
				} else if (reading_value && value.length()) {
					
					tag.args[name]=value;
					name="";
					value="";
					reading_value=false;
				}
				
			} else if (reading_value) {
				
				value+=c;
			} else {
				
				name+=c;
			}
		}
		
		if (f->eof_reached())
			return NULL;
	}	
	
	tag_stack.push_back(tag);
		
	return &tag_stack.back()->get();
}


Error ObjectFormatLoaderXML::close_tag(const String& p_name) {

	int level=0;
	bool inside_tag=false;

	while(true) {
	
		if (f->eof_reached()) {

			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": EOF found while attempting to find  </"+p_name+">");
			ERR_FAIL_COND_V( f->eof_reached(), ERR_FILE_CORRUPT );
		}
		
		uint8_t c = get_char();
		
		if (c == '<') {

			if (inside_tag) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Malformed XML. Already inside Tag.");
				ERR_FAIL_COND_V(inside_tag,ERR_FILE_CORRUPT);
			}
			inside_tag=true;
			c = get_char();
			if (c == '/') {

				--level;
			} else {

				++level;
			};
		} else if (c == '>') {

			if (!inside_tag) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Malformed XML. Already outside Tag");
				ERR_FAIL_COND_V(!inside_tag,ERR_FILE_CORRUPT);
			}
			inside_tag=false;
			if (level == -1) {
				tag_stack.pop_back();
				return OK;
			};
		};
	}
	
	return OK;
}

void ObjectFormatLoaderXML::unquote(String& p_str) {
	
	p_str=p_str.strip_edges();
	p_str=p_str.replace("\"","");
	p_str=p_str.replace("&gt;","<");
	p_str=p_str.replace("&lt;",">");
	p_str=p_str.replace("&apos;","'");
	p_str=p_str.replace("&quot;","\"");
	for (int i=1;i<32;i++) {
		
		char chr[2]={i,0};
		p_str=p_str.replace("&#"+String::num(i)+";",chr);
	}
	p_str=p_str.replace("&amp;","&");
	
	//p_str.parse_utf8( p_str.ascii(true).get_data() );
		
}

Error ObjectFormatLoaderXML::goto_end_of_tag() {

	uint8_t c;
	while(true) {

		c=get_char();
		if (c=='>') //closetag
			break;
		if (f->eof_reached()) {

			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": EOF found while attempting to find close tag.");
			ERR_FAIL_COND_V( f->eof_reached(), ERR_FILE_CORRUPT );
		}

	}
	tag_stack.pop_back();

	return OK;
}


Error ObjectFormatLoaderXML::parse_property_data(String &r_data) {
	
	r_data="";
	CharString cs;
	while(true) {
		
		CharType c=get_char();
		if (c=='<')
			break;
		ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
		cs.push_back(c);
	}

	cs.push_back(0);

	r_data.parse_utf8(cs.get_data());

	while(get_char()!='>' && !f->eof_reached()) {}
	if (f->eof_reached()) {

		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Malformed XML.");
		ERR_FAIL_COND_V( f->eof_reached(), ERR_FILE_CORRUPT );
	}

	r_data=r_data.strip_edges();
	tag_stack.pop_back();
	
	return OK;	
}


Error ObjectFormatLoaderXML::_parse_array_element(Vector<char> &buff,bool p_number_only,FileAccess *f,bool *end) {

	if (buff.empty())
		buff.resize(32); // optimize

	int buff_max=buff.size();
	int buff_size=0;
	*end=false;
	char *buffptr=&buff[0];
	bool found=false;
	bool quoted=false;

	while(true) {

		char c=get_char();

		if (c==0) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": File corrupt (zero found).");
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		} else if (c=='"') {
			quoted=!quoted;
		} else if ((!quoted && ((p_number_only && c<33) || c==',')) || c=='<') {


			if (c=='<') {
				*end=true;
				break;
			}
			if (c<32 && f->eof_reached()) {
				*end=true;
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": File corrupt (unexpected EOF).");
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}

			if (found)
				break;

		} else {

			found=true;
			if (buff_size>=buff_max) {

				buff_max++;
				buff.resize(buff_max);

			}

			buffptr[buff_size]=c;
			buff_size++;
		}
	}

	if (buff_size>=buff_max) {

		buff_max++;
		buff.resize(buff_max);

	}

	buff[buff_size]=0;
	buff_size++;

	return OK;
}

Error ObjectFormatLoaderXML::parse_property(Variant& r_v, String &r_name)  {

	bool exit;
	Tag *tag = parse_tag(&exit);
	
	if (!tag) {
		if (exit) // shouldn't have exited
			return ERR_FILE_EOF;
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": File corrupt (No Property Tag).");
		ERR_FAIL_V(ERR_FILE_CORRUPT);
	}
	
	r_v=Variant();
	r_name="";


	//ERR_FAIL_COND_V(tag->name!="property",ERR_FILE_CORRUPT);
	//ERR_FAIL_COND_V(!tag->args.has("name"),ERR_FILE_CORRUPT);
//	ERR_FAIL_COND_V(!tag->args.has("type"),ERR_FILE_CORRUPT);
	
	//String name=tag->args["name"];
	//ERR_FAIL_COND_V(name=="",ERR_FILE_CORRUPT);
	String type=tag->name;
	String name=tag->args["name"];

	if (type=="") {
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": 'type' field is empty.");
		ERR_FAIL_COND_V(type=="",ERR_FILE_CORRUPT);
	}

	if (type=="dictionary") {
		
		Dictionary d;
		
		while(true) {
			
			Error err;
			String tagname;
			Variant key;

			int dictline = get_current_line();


			err=parse_property(key,tagname);

			if (err && err!=ERR_FILE_EOF) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error parsing dictionary: "+name+" (from line "+itos(dictline)+")");
				ERR_FAIL_COND_V(err && err!=ERR_FILE_EOF,err);
			}
			//ERR_FAIL_COND_V(tagname!="key",ERR_FILE_CORRUPT);
			if (err)
				break;
			Variant value;
			err=parse_property(value,tagname);
			if (err) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error parsing dictionary: "+name+" (from line "+itos(dictline)+")");
			}

			ERR_FAIL_COND_V(err,err);
			//ERR_FAIL_COND_V(tagname!="value",ERR_FILE_CORRUPT);

			d[key]=value;
		}
		

		//err=parse_property_data(name); // skip the rest
		//ERR_FAIL_COND_V(err,err);
		
		r_name=name;
		r_v=d;
		return OK;
		
	} else if (type=="array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}


		int len=tag->args["len"].to_int();
		
		Array array;
		array.resize(len);
		
		Error err;
		Variant v;
		String tagname;
		int idx=0;
		while( (err=parse_property(v,tagname))==OK ) {
			
			ERR_CONTINUE( idx <0 || idx >=len );
			
			array.set(idx,v);	
			idx++;
		}

		if (idx!=len) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error loading array (size mismatch): "+name);
			ERR_FAIL_COND_V(idx!=len,err);
		}

		if (err!=ERR_FILE_EOF) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error loading array: "+name);
			ERR_FAIL_COND_V(err!=ERR_FILE_EOF,err);
		}

		//err=parse_property_data(name); // skip the rest
		//ERR_FAIL_COND_V(err,err);
		
		r_name=name;
		r_v=array;
		return OK;
		
	} else if (type=="resource") {
		
		if (tag->args.has("path")) {
			
			String path=tag->args["path"];
			String hint;
			if (tag->args.has("resource_type"))
				hint=tag->args["resource_type"];
			
			if (path.begins_with("local://"))
				path=path.replace("local://",local_path+"::");
			else if (path.find("://")==-1 && path.is_rel_path()) {
				// path is relative to file being loaded, so convert to a resource path
				path=Globals::get_singleton()->localize_path(local_path.get_base_dir()+"/"+path);

			}

			//take advantage of the resource loader cache. The resource is cached on it, even if
			RES res=ResourceLoader::load(path,hint);
					

			if (res.is_null()) {
			
				WARN_PRINT(String("Couldn't load resource: "+path).ascii().get_data());
			}
			
			r_v=res.get_ref_ptr();
		}
				


		Error err=goto_end_of_tag();
		if (err) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error closing <resource> tag.");
			ERR_FAIL_COND_V(err,err);
		}


		r_name=name;

		return OK;
		
	} else if (type=="image") {
		
		if (!tag->args.has("encoding")) {
			//empty image
			r_v=Image();
			String sdfsdfg;
			Error err=parse_property_data(sdfsdfg);
			return OK;
		}

		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'encoding' field.");
		ERR_FAIL_COND_V( !tag->args.has("encoding"), ERR_FILE_CORRUPT );
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'width' field.");
		ERR_FAIL_COND_V( !tag->args.has("width"), ERR_FILE_CORRUPT );
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'height' field.");
		ERR_FAIL_COND_V( !tag->args.has("height"), ERR_FILE_CORRUPT );
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'format' field.");
		ERR_FAIL_COND_V( !tag->args.has("format"), ERR_FILE_CORRUPT );
		
		String encoding=tag->args["encoding"];
		
		if (encoding=="raw") {
			String width=tag->args["width"];
			String height=tag->args["height"];
			String format=tag->args["format"];
			int mipmaps=tag->args.has("mipmaps")?int(tag->args["mipmaps"].to_int()):int(0);
			int custom_size = tag->args.has("custom_size")?int(tag->args["custom_size"].to_int()):int(0);

			r_name=name;

			Image::Format imgformat;

			
			if (format=="grayscale") {
				imgformat=Image::FORMAT_GRAYSCALE;
			} else if (format=="intensity") {
				imgformat=Image::FORMAT_INTENSITY;
			} else if (format=="grayscale_alpha") {
				imgformat=Image::FORMAT_GRAYSCALE_ALPHA;
			} else if (format=="rgb") {
				imgformat=Image::FORMAT_RGB;
			} else if (format=="rgba") {
				imgformat=Image::FORMAT_RGBA;
			} else if (format=="indexed") {
				imgformat=Image::FORMAT_INDEXED;
			} else if (format=="indexed_alpha") {
				imgformat=Image::FORMAT_INDEXED_ALPHA;
			} else if (format=="bc1") {
				imgformat=Image::FORMAT_BC1;
			} else if (format=="bc2") {
				imgformat=Image::FORMAT_BC2;
			} else if (format=="bc3") {
				imgformat=Image::FORMAT_BC3;
			} else if (format=="bc4") {
				imgformat=Image::FORMAT_BC4;
			} else if (format=="bc5") {
				imgformat=Image::FORMAT_BC5;
			} else if (format=="custom") {
				imgformat=Image::FORMAT_CUSTOM;
			} else {
				
				ERR_FAIL_V( ERR_FILE_CORRUPT );
			}


			int datasize;
			int w=width.to_int();
			int h=height.to_int();

			if (w == 0 && w == 0) {
				//r_v = Image(w, h, imgformat);
				r_v=Image();
				String sdfsdfg;
				Error err=parse_property_data(sdfsdfg);
				return OK;
			};

			if (imgformat==Image::FORMAT_CUSTOM) {

				datasize=custom_size;
			} else {

				datasize = Image::get_image_data_size(h,w,imgformat,mipmaps);
			}

			if (datasize==0) {
				//r_v = Image(w, h, imgformat);
				r_v=Image();
				String sdfsdfg;
				Error err=parse_property_data(sdfsdfg);
				return OK;
			};

			DVector<uint8_t> pixels;
			pixels.resize(datasize);
			DVector<uint8_t>::Write wb = pixels.write();
			
			int idx=0;
			uint8_t byte;
			while( idx<datasize*2) {
				
				CharType c=get_char();
				
				ERR_FAIL_COND_V(c=='<',ERR_FILE_CORRUPT);

				if ( (c>='0' && c<='9') || (c>='A' && c<='F') || (c>='a' && c<='f') ) {

					if (idx&1) {

						byte|=HEX2CHR(c);
						wb[idx>>1]=byte;
					} else {

						byte=HEX2CHR(c)<<4;
					}

					idx++;
				}

			}
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);

			wb=DVector<uint8_t>::Write();

			r_v=Image(w,h,mipmaps,imgformat,pixels);
			String sdfsdfg;
			Error err=parse_property_data(sdfsdfg);
			ERR_FAIL_COND_V(err,err);
			
			return OK;
		}
		
		ERR_FAIL_V(ERR_FILE_CORRUPT);
		
	} else if (type=="raw_array") {
		
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": RawArray missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();
		
		DVector<uint8_t> bytes;
		bytes.resize(len);
		DVector<uint8_t>::Write w=bytes.write();
		uint8_t *bytesptr=w.ptr();		
		int idx=0;			
		uint8_t byte;
		while( idx<len*2) {
			
			CharType c=get_char();
			
			if (idx&1) {
				
				byte|=HEX2CHR(c);
				bytesptr[idx>>1]=byte;
			} else {
				
				byte=HEX2CHR(c)<<4;
			}

			idx++;
		}

		ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);

		w=DVector<uint8_t>::Write();
		r_v=bytes;
		String sdfsdfg;
		Error err=parse_property_data(sdfsdfg);
		ERR_FAIL_COND_V(err,err);
		r_name=name;
		
		return OK;
		
	} else if (type=="int_array") {
		
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();
		
		DVector<int> ints;
		ints.resize(len);
		DVector<int>::Write w=ints.write();
		int *intsptr=w.ptr();		
		int idx=0;			
		String str;
#if 0
		while( idx<len ) {
			
			
			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
						
			if (c<33 || c==',' || c=='<') {
								
				if (str.length()) {
					
					intsptr[idx]=str.to_int();
					str="";
					idx++;
				}
				
				if (c=='<') {
					
					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}
				
			} else {
				
				str+=c;
			}
		}

#else

		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);

			intsptr[idx]=String::to_int(&tmpdata[0]);
			idx++;
			if (end)
				break;

		}

#endif
		w=DVector<int>::Write();

		r_v=ints;		
		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;
		
		return OK;
	} else if (type=="real_array") {
		
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;
		
		DVector<real_t> reals;
		reals.resize(len);
		DVector<real_t>::Write w=reals.write();
		real_t *realsptr=w.ptr();		
		int idx=0;			
		String str;


#if 0
		while( idx<len ) {
			
			
			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
			
			
			if (c<33 || c==',' || c=='<') {
				
				if (str.length()) {
					
					realsptr[idx]=str.to_double();
					str="";
					idx++;
				}
				
				if (c=='<') {
					
					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}
				
			} else {
				
				str+=c;
			}
		}

#else



		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);

			realsptr[idx]=String::to_double(&tmpdata[0]);
			idx++;

			if (end)
				break;
		}

#endif

		w=DVector<real_t>::Write();
		r_v=reals;

		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;
		
		return OK;
	} else if (type=="string_array") {
		
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();
		
		DVector<String> strings;
		strings.resize(len);
		DVector<String>::Write w=strings.write();
		String *stringsptr=w.ptr();		
		int idx=0;			
		String str;
		
		bool inside_str=false;
		CharString cs;
		while( idx<len ) {
			
			
			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
			
			
			if (c=='"') {
				if (inside_str) {
					
					cs.push_back(0);
					String str;
					str.parse_utf8(cs.get_data());
					unquote(str);
					stringsptr[idx]=str;
					cs.clear();
					idx++;
					inside_str=false;
				} else {
					inside_str=true;
				}
			} else if (c=='<') {
					
				while(get_char()!='>' && !f->eof_reached()) {}
				ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
				break;

				
			} else if (inside_str){

				cs.push_back(c);
			}
		}
		w=DVector<String>::Write();
		r_v=strings;
		String sdfsdfg;
		Error err=parse_property_data(sdfsdfg);
		ERR_FAIL_COND_V(err,err);
		
		r_name=name;
		
		return OK;
	} else if (type=="vector3_array") {
		
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;
		
		DVector<Vector3> vectors;
		vectors.resize(len);
		DVector<Vector3>::Write w=vectors.write();
		Vector3 *vectorsptr=w.ptr();		
		int idx=0;
		int subidx=0;
		Vector3 auxvec;
		String str;

//		uint64_t tbegin = OS::get_singleton()->get_ticks_usec();
#if 0
		while( idx<len ) {
			
			
			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
			
			
			if (c<33 || c==',' || c=='<') {
				
				if (str.length()) {
					
					auxvec[subidx]=str.to_double();
					subidx++;
					str="";
					if (subidx==3) {
						vectorsptr[idx]=auxvec;

						idx++;
						subidx=0;
					}
				}
				
				if (c=='<') {
					
					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}
				
			} else {
				
				str+=c;
			}
		}
#else

		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;			
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);


			auxvec[subidx]=String::to_double(&tmpdata[0]);
			subidx++;
			if (subidx==3) {
				vectorsptr[idx]=auxvec;

				idx++;
				subidx=0;
			}

			if (end)
				break;
		}



#endif
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Premature end of vector3 array");
		ERR_FAIL_COND_V(idx<len,ERR_FILE_CORRUPT);
//		double time_taken = (OS::get_singleton()->get_ticks_usec() - tbegin)/1000000.0;


		w=DVector<Vector3>::Write();
		r_v=vectors;
		String sdfsdfg;
		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;
		
		return OK;

	} else if (type=="vector2_array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;

		DVector<Vector2> vectors;
		vectors.resize(len);
		DVector<Vector2>::Write w=vectors.write();
		Vector2 *vectorsptr=w.ptr();
		int idx=0;
		int subidx=0;
		Vector2 auxvec;
		String str;

//		uint64_t tbegin = OS::get_singleton()->get_ticks_usec();
#if 0
		while( idx<len ) {


			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);


			if (c<22 || c==',' || c=='<') {

				if (str.length()) {

					auxvec[subidx]=str.to_double();
					subidx++;
					str="";
					if (subidx==2) {
						vectorsptr[idx]=auxvec;

						idx++;
						subidx=0;
					}
				}

				if (c=='<') {

					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}

			} else {

				str+=c;
			}
		}
#else

		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);


			auxvec[subidx]=String::to_double(&tmpdata[0]);
			subidx++;
			if (subidx==2) {
				vectorsptr[idx]=auxvec;

				idx++;
				subidx=0;
			}

			if (end)
				break;
		}



#endif
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Premature end of vector2 array");
		ERR_FAIL_COND_V(idx<len,ERR_FILE_CORRUPT);
//		double time_taken = (OS::get_singleton()->get_ticks_usec() - tbegin)/1000000.0;


		w=DVector<Vector2>::Write();
		r_v=vectors;
		String sdfsdfg;
		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;

		return OK;

	} else if (type=="color_array") {
		
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;
		
		DVector<Color> colors;
		colors.resize(len);
		DVector<Color>::Write w=colors.write();
		Color *colorsptr=w.ptr();		
		int idx=0;
		int subidx=0;
		Color auxcol;
		String str;
		
		while( idx<len ) {
			
			
			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
			
			
			if (c<33 || c==',' || c=='<') {
				
				if (str.length()) {
					
					auxcol[subidx]=str.to_double();
					subidx++;
					str="";
					if (subidx==4) {
						colorsptr[idx]=auxcol;
						idx++;
						subidx=0;
					}
				}
				
				if (c=='<') {
					
					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}
				
			} else {
				
				str+=c;
			}
		}
		w=DVector<Color>::Write();
		r_v=colors;
		String sdfsdfg;
		Error err=parse_property_data(sdfsdfg);
		ERR_FAIL_COND_V(err,err);
		r_name=name;
		
		return OK;
	}
	
	
	String data;
	Error err = parse_property_data(data);
	ERR_FAIL_COND_V(err!=OK,err);
	
	if (type=="nil") {
		// uh do nothing
		
	} else if (type=="bool") {
		// uh do nothing
		if (data.nocasecmp_to("true")==0 || data.to_int()!=0)
			r_v=true;
		else
			r_v=false;
	} else if (type=="int") {
		
		r_v=data.to_int();
	} else if (type=="real") {
		
		r_v=data.to_double();
	} else if (type=="string") {
		
		String str=data;
		unquote(str);
		r_v=str;
	} else if (type=="vector3") {
		
		
		r_v=Vector3( 
				data.get_slice(",",0).to_double(),
				data.get_slice(",",1).to_double(),
				data.get_slice(",",2).to_double()
		           );
		             
	} else if (type=="vector2") {
		
		
		r_v=Vector2( 
				data.get_slice(",",0).to_double(),
				data.get_slice(",",1).to_double()
		           );
		             
	} else if (type=="plane") {
		
		r_v=Plane( 
				data.get_slice(",",0).to_double(),
				data.get_slice(",",1).to_double(),
				data.get_slice(",",2).to_double(),
				data.get_slice(",",3).to_double()
		         );
		
	} else if (type=="quaternion") {
		
		r_v=Quat( 
				data.get_slice(",",0).to_double(),
				data.get_slice(",",1).to_double(),
				data.get_slice(",",2).to_double(),
				data.get_slice(",",3).to_double()
		         );
		
	} else if (type=="rect2") {
		
		r_v=Rect2(
			Vector2( 
				data.get_slice(",",0).to_double(),
				data.get_slice(",",1).to_double()
			),
			Vector2( 
				data.get_slice(",",2).to_double(),
				data.get_slice(",",3).to_double()
			)
		);
		          
		
	} else if (type=="aabb") {
		
		r_v=AABB(
			Vector3( 
				data.get_slice(",",0).to_double(),
				data.get_slice(",",1).to_double(),
				data.get_slice(",",2).to_double()
			),
			Vector3( 
				data.get_slice(",",3).to_double(),
				data.get_slice(",",4).to_double(),
				data.get_slice(",",5).to_double()
			)
		);
		          
	} else if (type=="matrix32") {

		Matrix32 m3;
		for (int i=0;i<3;i++) {
			for (int j=0;j<2;j++) {
				m3.elements[i][j]=data.get_slice(",",i*2+j).to_double();
			}
		}
		r_v=m3;

	} else if (type=="matrix3") {
		
		Matrix3 m3;
		for (int i=0;i<3;i++) {
			for (int j=0;j<3;j++) {
				m3.elements[i][j]=data.get_slice(",",i*3+j).to_double();
			}
		}
		r_v=m3;		
		
	} else if (type=="transform") {
		
		Transform tr;
		for (int i=0;i<3;i++) {
			for (int j=0;j<3;j++) {
				tr.basis.elements[i][j]=data.get_slice(",",i*3+j).to_double();			
			}
		
		}
		tr.origin=Vector3( 
		             data.get_slice(",",9).to_double(),
		             data.get_slice(",",10).to_double(),
		             data.get_slice(",",11).to_double()
		           );
		r_v=tr;
		
	} else if (type=="color") {
		
		r_v=Color( 
		           data.get_slice(",",0).to_double(),
		           data.get_slice(",",1).to_double(),
		           data.get_slice(",",2).to_double(),
		           data.get_slice(",",3).to_double()
		         );
		
	} else if (type=="node_path") {
		
		String str=data;
		unquote(str);
		r_v=NodePath( str );
	} else if (type=="input_event") {
		
		// ?	
	} else {
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Unrecognized tag in file: "+type);
		ERR_FAIL_V(ERR_FILE_CORRUPT);
	}
	r_name=name;
	return OK;
}


Error ObjectFormatLoaderXML::load(Object **p_object,Variant &p_meta)  {
	
	*p_object=NULL;
	p_meta=Variant();
	


	while(true) {


		bool exit;
		Tag *tag = parse_tag(&exit);


		if (!tag) {
			if (!exit) // shouldn't have exited
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			*p_object=NULL;
			return ERR_FILE_EOF;
		}

		RES resource;
		Object *obj=NULL;

		if (tag->name=="resource") {
			//loading resource

			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <resource> missing 'len' field.");
			ERR_FAIL_COND_V(!tag->args.has("path"),ERR_FILE_CORRUPT);
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <resource> missing 'type' field.");
			ERR_FAIL_COND_V(!tag->args.has("type"),ERR_FILE_CORRUPT);
			String path=tag->args["path"];

			if (path.begins_with("local://")) {
				//built-in resource (but really external)
				path=path.replace("local://",local_path+"::");
			}


			if (ResourceCache::has(path)) {
				Error err = close_tag(tag->name);
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Unable to close <resource> tag.");
				ERR_FAIL_COND_V( err, err );
				continue; //it's a resource, and it's already loaded

			}

			String type = tag->args["type"];

			obj = ObjectTypeDB::instance(type);
			if (!obj) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Object of unrecognized type in file: "+type);
			}
			ERR_FAIL_COND_V(!obj,ERR_FILE_CORRUPT);

			Resource *r = obj->cast_to<Resource>();
			if (!r) {
				memdelete(obj); //bye
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Object type in resource field not a resource, type is: "+obj->get_type());
				ERR_FAIL_COND_V(!obj->cast_to<Resource>(),ERR_FILE_CORRUPT);
			}

			resource = RES( r );
			r->set_path(path);



		} else if (tag->name=="object") {

			if ( tag->args.has("type") ) {

				ERR_FAIL_COND_V(!ObjectTypeDB::type_exists(tag->args["type"]), ERR_FILE_CORRUPT);

				if (ObjectTypeDB::can_instance(tag->args["type"])) {
					obj = ObjectTypeDB::instance(tag->args["type"]);
					if (!obj) {
						ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Object of unrecognized type in file: "+tag->args["type"]);
					}
					ERR_FAIL_COND_V(!obj,ERR_FILE_CORRUPT);
				} else {

					close_tag(tag->name);
					return ERR_SKIP;
				};
			} else {
				//otherwise it's a meta object
			}

		} else {

			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Unknown main tag: "+tag->name);
			ERR_FAIL_V( ERR_FILE_CORRUPT );
		}

		//load properties

		while(true) {

			String name;
			Variant v;
			Error err;
			err = parse_property(v,name);
			if (err==ERR_FILE_EOF) //tag closed
				break;
			if (err!=OK) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": XML Parsing aborted.");
				ERR_FAIL_COND_V(err!=OK,ERR_FILE_CORRUPT);
			}
			if (resource.is_null() && name=="__xml_meta__") {

				p_meta=v;
				continue;
			} else if (!obj) {

				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Normal property found in meta object.");
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}

			obj->set(name,v);
		}

		if (!obj) {
			*p_object=NULL;
			return OK; // it was a meta object			
		}

		if (resource.is_null()) {

			//regular object
			*p_object=obj;
			return OK;
		} else {


			resource_cache.push_back(resource); //keep it in mem until finished loading
		}

		// a resource.. continue!

	}



	return OK; //never reach anyway
	
}

int ObjectFormatLoaderXML::get_current_line() const {

	return lines;
}


uint8_t ObjectFormatLoaderXML::get_char() const {

	uint8_t c = f->get_8();
	if (c=='\n')
		lines++;
	return c;

}

ObjectFormatLoaderXML::~ObjectFormatLoaderXML() {

	if (f) {
		if (f->is_open())
			f->close();
		memdelete(f);
	}
}



ObjectFormatLoaderXML* ObjectFormatLoaderInstancerXML::instance(const String& p_file,const String& p_magic) {

	Error err;
	FileAccess *f=FileAccess::open(p_file,FileAccess::READ,&err);
	if (err!=OK) {

		ERR_FAIL_COND_V(err!=OK,NULL);
	}

	ObjectFormatLoaderXML *loader = memnew( ObjectFormatLoaderXML );

	loader->lines=1;
	loader->f=f;
	loader->local_path = Globals::get_singleton()->localize_path(p_file);

	ObjectFormatLoaderXML::Tag *tag = loader->parse_tag();
	if (!tag || tag->name!="?xml" || !tag->args.has("version") || !tag->args.has("encoding") || tag->args["encoding"]!="UTF-8") {

		f->close();
		memdelete(loader);
		ERR_EXPLAIN("Not a XML:UTF-8 File: "+p_file);
		ERR_FAIL_V(NULL);
	}
	
	loader->tag_stack.clear(); 
	
	tag = loader->parse_tag();
	
	if (!tag || tag->name!="object_file" || !tag->args.has("magic") || !tag->args.has("version") || tag->args["magic"]!=p_magic) {
		
		f->close();
		memdelete(loader);
		ERR_EXPLAIN("Unrecognized XML File: "+p_file);
		ERR_FAIL_V(NULL);
	}

	String version = tag->args["version"];
	if (version.get_slice_count(".")!=2) {

		f->close();
		memdelete(loader);
		ERR_EXPLAIN("Invalid Version String '"+version+"'' in file: "+p_file);
		ERR_FAIL_V(NULL);
	}

	int major = version.get_slice(".",0).to_int();
	int minor = version.get_slice(".",1).to_int();

	if (major>VERSION_MAJOR || (major==VERSION_MAJOR && minor>VERSION_MINOR)) {

		f->close();
		memdelete(loader);
		ERR_EXPLAIN("File Format '"+version+"' is too new! Please upgrade to a a new engine version: "+p_file);
		ERR_FAIL_V(NULL);

	}

	return loader;
}

void ObjectFormatLoaderInstancerXML::get_recognized_extensions(List<String> *p_extensions) const {
	
	p_extensions->push_back("xml");	
}


#endif
#endif
#endif

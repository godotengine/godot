/*************************************************************************/
/*  object_format_binary.cpp                                             */
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
#include "object_format_binary.h"
#include "resource.h"
#include "io/resource_loader.h"
#include "print_string.h"
#include "object_type_db.h"
#include "globals.h"
#include "os/os.h"
#include "version.h"


#define print_bl(m_what)
#ifdef OLD_SCENE_FORMAT_ENABLED


enum {

	SECTION_RESOURCE=0,
	SECTION_OBJECT=1,
	SECTION_META_OBJECT=2,
	SECTION_PROPERTY=3,
	SECTION_END=4,

	//numbering must be different from variant, in case new variant types are added (variant must be always contiguous for jumptable optimization)
	VARIANT_NIL=1,
	VARIANT_BOOL=2,
	VARIANT_INT=3,
	VARIANT_REAL=4,
	VARIANT_STRING=5,
	VARIANT_VECTOR2=10,
	VARIANT_RECT2=11,
	VARIANT_VECTOR3=12,
	VARIANT_PLANE=13,
	VARIANT_QUAT=14,
	VARIANT_AABB=15,
	VARIANT_MATRIX3=16,
	VARIANT_TRANSFORM=17,
	VARIANT_MATRIX32=18,
	VARIANT_COLOR=20,
	VARIANT_IMAGE=21,
	VARIANT_NODE_PATH=22,
	VARIANT_RID=23,
	VARIANT_OBJECT=24,
	VARIANT_INPUT_EVENT=25,
	VARIANT_DICTIONARY=26,
	VARIANT_ARRAY=30,
	VARIANT_RAW_ARRAY=31,
	VARIANT_INT_ARRAY=32,
	VARIANT_REAL_ARRAY=33,
	VARIANT_STRING_ARRAY=34,
	VARIANT_VECTOR3_ARRAY=35,
	VARIANT_COLOR_ARRAY=36,
	VARIANT_VECTOR2_ARRAY=37,

	IMAGE_ENCODING_EMPTY=0,
	IMAGE_ENCODING_RAW=1,
	IMAGE_ENCODING_PNG=2, //not yet
	IMAGE_ENCODING_JPG=3,

	IMAGE_FORMAT_GRAYSCALE=0,
	IMAGE_FORMAT_INTENSITY=1,
	IMAGE_FORMAT_GRAYSCALE_ALPHA=2,
	IMAGE_FORMAT_RGB=3,
	IMAGE_FORMAT_RGBA=4,
	IMAGE_FORMAT_INDEXED=5,
	IMAGE_FORMAT_INDEXED_ALPHA=6,
	IMAGE_FORMAT_BC1=7,
	IMAGE_FORMAT_BC2=8,
	IMAGE_FORMAT_BC3=9,
	IMAGE_FORMAT_BC4=10,
	IMAGE_FORMAT_BC5=11,
	IMAGE_FORMAT_CUSTOM=12,

	OBJECT_EMPTY=0,
	OBJECT_EXTERNAL_RESOURCE=1,
	OBJECT_INTERNAL_RESOURCE=2,


};


void ObjectFormatSaverBinary::_pad_buffer(int p_bytes) {

	int extra = 4-(p_bytes%4);
	if (extra<4) {
		for(int i=0;i<extra;i++)
			f->store_8(0); //pad to 32
	}

}


void ObjectFormatSaverBinary::write_property(int p_idx,const Variant& p_property) {

	f->store_32(SECTION_PROPERTY);
	f->store_32(p_idx);

	switch(p_property.get_type()) {

		case Variant::NIL: {

			f->store_32(VARIANT_NIL);
			// don't store anything
		} break;
		case Variant::BOOL: {

			f->store_32(VARIANT_BOOL);
			bool val=p_property;
			f->store_32(val);
		} break;
		case Variant::INT: {

			f->store_32(VARIANT_INT);
			int val=p_property;
			f->store_32(val);
		} break;
		case Variant::REAL: {

			f->store_32(VARIANT_REAL);
			real_t val=p_property;
			f->store_real(val);

		} break;
		case Variant::STRING: {

			f->store_32(VARIANT_STRING);
			String val=p_property;
			save_unicode_string(val);

		} break;
		case Variant::VECTOR2: {

			f->store_32(VARIANT_VECTOR2);
			Vector2 val=p_property;
			f->store_real(val.x);
			f->store_real(val.y);

		} break;
		case Variant::RECT2: {

			f->store_32(VARIANT_RECT2);
			Rect2 val=p_property;
			f->store_real(val.pos.x);
			f->store_real(val.pos.y);
			f->store_real(val.size.x);
			f->store_real(val.size.y);

		} break;
		case Variant::VECTOR3: {

			f->store_32(VARIANT_VECTOR3);
			Vector3 val=p_property;
			f->store_real(val.x);
			f->store_real(val.y);
			f->store_real(val.z);

		} break;
		case Variant::PLANE: {

			f->store_32(VARIANT_PLANE);
			Plane val=p_property;
			f->store_real(val.normal.x);
			f->store_real(val.normal.y);
			f->store_real(val.normal.z);
			f->store_real(val.d);

		} break;
		case Variant::QUAT: {

			f->store_32(VARIANT_QUAT);
			Quat val=p_property;
			f->store_real(val.x);
			f->store_real(val.y);
			f->store_real(val.z);
			f->store_real(val.w);

		} break;
		case Variant::_AABB: {

			f->store_32(VARIANT_AABB);
			AABB val=p_property;
			f->store_real(val.pos.x);
			f->store_real(val.pos.y);
			f->store_real(val.pos.z);
			f->store_real(val.size.x);
			f->store_real(val.size.y);
			f->store_real(val.size.z);

		} break;
		case Variant::MATRIX32: {

			f->store_32(VARIANT_MATRIX32);
			Matrix32 val=p_property;
			f->store_real(val.elements[0].x);
			f->store_real(val.elements[0].y);
			f->store_real(val.elements[1].x);
			f->store_real(val.elements[1].y);
			f->store_real(val.elements[2].x);
			f->store_real(val.elements[2].y);

		} break;
		case Variant::MATRIX3: {

			f->store_32(VARIANT_MATRIX3);
			Matrix3 val=p_property;
			f->store_real(val.elements[0].x);
			f->store_real(val.elements[0].y);
			f->store_real(val.elements[0].z);
			f->store_real(val.elements[1].x);
			f->store_real(val.elements[1].y);
			f->store_real(val.elements[1].z);
			f->store_real(val.elements[2].x);
			f->store_real(val.elements[2].y);
			f->store_real(val.elements[2].z);

		} break;
		case Variant::TRANSFORM: {

			f->store_32(VARIANT_TRANSFORM);
			Transform val=p_property;
			f->store_real(val.basis.elements[0].x);
			f->store_real(val.basis.elements[0].y);
			f->store_real(val.basis.elements[0].z);
			f->store_real(val.basis.elements[1].x);
			f->store_real(val.basis.elements[1].y);
			f->store_real(val.basis.elements[1].z);
			f->store_real(val.basis.elements[2].x);
			f->store_real(val.basis.elements[2].y);
			f->store_real(val.basis.elements[2].z);
			f->store_real(val.origin.x);
			f->store_real(val.origin.y);
			f->store_real(val.origin.z);

		} break;
		case Variant::COLOR: {

			f->store_32(VARIANT_COLOR);
			Color val=p_property;
			f->store_real(val.r);
			f->store_real(val.g);
			f->store_real(val.b);
			f->store_real(val.a);

		} break;
		case Variant::IMAGE: {

			f->store_32(VARIANT_IMAGE);
			Image val =p_property;
			if (val.empty()) {
				f->store_32(IMAGE_ENCODING_EMPTY);
				break;
			}
			f->store_32(IMAGE_ENCODING_RAW); //raw encoding
			f->store_32(val.get_width());
			f->store_32(val.get_height());
			f->store_32(val.get_mipmaps());
			switch(val.get_format()) {

				case Image::FORMAT_GRAYSCALE: f->store_32(IMAGE_FORMAT_GRAYSCALE ); break; ///< one byte per pixel: f->store_32(IMAGE_FORMAT_ ); break; 0-255
				case Image::FORMAT_INTENSITY: f->store_32(IMAGE_FORMAT_INTENSITY ); break; ///< one byte per pixel: f->store_32(IMAGE_FORMAT_ ); break; 0-255
				case Image::FORMAT_GRAYSCALE_ALPHA: f->store_32(IMAGE_FORMAT_GRAYSCALE_ALPHA ); break; ///< two bytes per pixel: f->store_32(IMAGE_FORMAT_ ); break; 0-255. alpha 0-255
				case Image::FORMAT_RGB: f->store_32(IMAGE_FORMAT_RGB ); break; ///< one byte R: f->store_32(IMAGE_FORMAT_ ); break; one byte G: f->store_32(IMAGE_FORMAT_ ); break; one byte B
				case Image::FORMAT_RGBA: f->store_32(IMAGE_FORMAT_RGBA ); break; ///< one byte R: f->store_32(IMAGE_FORMAT_ ); break; one byte G: f->store_32(IMAGE_FORMAT_ ); break; one byte B: f->store_32(IMAGE_FORMAT_ ); break; one byte A
				case Image::FORMAT_INDEXED: f->store_32(IMAGE_FORMAT_INDEXED ); break; ///< index byte 0-256: f->store_32(IMAGE_FORMAT_ ); break; and after image end: f->store_32(IMAGE_FORMAT_ ); break; 256*3 bytes of palette
				case Image::FORMAT_INDEXED_ALPHA: f->store_32(IMAGE_FORMAT_INDEXED_ALPHA ); break; ///< index byte 0-256: f->store_32(IMAGE_FORMAT_ ); break; and after image end: f->store_32(IMAGE_FORMAT_ ); break; 256*4 bytes of palette (alpha)
				case Image::FORMAT_BC1: f->store_32(IMAGE_FORMAT_BC1 ); break; // DXT1
				case Image::FORMAT_BC2: f->store_32(IMAGE_FORMAT_BC2 ); break; // DXT3
				case Image::FORMAT_BC3: f->store_32(IMAGE_FORMAT_BC3 ); break; // DXT5
				case Image::FORMAT_BC4: f->store_32(IMAGE_FORMAT_BC4 ); break; // ATI1
				case Image::FORMAT_BC5: f->store_32(IMAGE_FORMAT_BC5 ); break; // ATI2
				case Image::FORMAT_CUSTOM: f->store_32(IMAGE_FORMAT_CUSTOM ); break;
				default: {}

			}

			int dlen = val.get_data().size();
			f->store_32(dlen);
			DVector<uint8_t>::Read r = val.get_data().read();
			f->store_buffer(r.ptr(),dlen);
			_pad_buffer(dlen);

		} break;
		case Variant::NODE_PATH: {
			f->store_32(VARIANT_NODE_PATH);
			save_unicode_string(p_property);
		} break;
		case Variant::_RID: {

			f->store_32(VARIANT_RID);
			WARN_PRINT("Can't save RIDs");
			RID val = p_property;
			f->store_32(val.get_id());
		} break;
		case Variant::OBJECT: {

			f->store_32(VARIANT_OBJECT);
			RES res = p_property;
			if (res.is_null()) {
				f->store_32(OBJECT_EMPTY);
				return; // don't save it
			}

			if (res->get_path().length() && res->get_path().find("::")==-1) {
				f->store_32(OBJECT_EXTERNAL_RESOURCE);
				save_unicode_string(res->get_type());
				String path=relative_paths?local_path.path_to_file(res->get_path()):res->get_path();
				save_unicode_string(path);
			} else {

				if (!resource_map.has(res)) {
					f->store_32(OBJECT_EMPTY);
					ERR_EXPLAIN("Resource was not pre cached for the resource section, bug?");
					ERR_FAIL();
				}

				f->store_32(OBJECT_INTERNAL_RESOURCE);
				f->store_32(resource_map[res]);
				//internal resource
			}


		} break;
		case Variant::INPUT_EVENT: {

			f->store_32(VARIANT_INPUT_EVENT);
			WARN_PRINT("Can't save InputEvent (maybe it could..)");
		} break;
		case Variant::DICTIONARY: {

			f->store_32(VARIANT_DICTIONARY);
			Dictionary d = p_property;
			f->store_32(d.size());

			List<Variant> keys;
			d.get_key_list(&keys);

			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				//if (!_check_type(dict[E->get()]))
				//	continue;

				write_property(0,E->get());
				write_property(0,d[E->get()]);
			}


		} break;
		case Variant::ARRAY: {

			f->store_32(VARIANT_ARRAY);
			Array a=p_property;
			f->store_32(a.size());
			for(int i=0;i<a.size();i++) {

				write_property(i,a[i]);
			}

		} break;
		case Variant::RAW_ARRAY: {

			f->store_32(VARIANT_RAW_ARRAY);
			DVector<uint8_t> arr = p_property;
			int len=arr.size();
			f->store_32(len);
			DVector<uint8_t>::Read r = arr.read();
			f->store_buffer(r.ptr(),len);
			_pad_buffer(len);

		} break;
		case Variant::INT_ARRAY: {

			f->store_32(VARIANT_INT_ARRAY);
			DVector<int> arr = p_property;
			int len=arr.size();
			f->store_32(len);
			DVector<int>::Read r = arr.read();
			for(int i=0;i<len;i++)
				f->store_32(r[i]);

		} break;
		case Variant::REAL_ARRAY: {

			f->store_32(VARIANT_REAL_ARRAY);
			DVector<real_t> arr = p_property;
			int len=arr.size();
			f->store_32(len);
			DVector<real_t>::Read r = arr.read();
			for(int i=0;i<len;i++) {
				f->store_real(r[i]);
			}

		} break;
		case Variant::STRING_ARRAY: {

			f->store_32(VARIANT_STRING_ARRAY);
			DVector<String> arr = p_property;
			int len=arr.size();
			f->store_32(len);
			DVector<String>::Read r = arr.read();
			for(int i=0;i<len;i++) {
				save_unicode_string(r[i]);
			}

		} break;
		case Variant::VECTOR3_ARRAY: {

			f->store_32(VARIANT_VECTOR3_ARRAY);
			DVector<Vector3> arr = p_property;
			int len=arr.size();
			f->store_32(len);
			DVector<Vector3>::Read r = arr.read();
			for(int i=0;i<len;i++) {
				f->store_real(r[i].x);
				f->store_real(r[i].y);
				f->store_real(r[i].z);
			}

		} break;
		case Variant::VECTOR2_ARRAY: {

			f->store_32(VARIANT_VECTOR2_ARRAY);
			DVector<Vector2> arr = p_property;
			int len=arr.size();
			f->store_32(len);
			DVector<Vector2>::Read r = arr.read();
			for(int i=0;i<len;i++) {
				f->store_real(r[i].x);
				f->store_real(r[i].y);
			}

		} break;
		case Variant::COLOR_ARRAY: {

			f->store_32(VARIANT_COLOR_ARRAY);
			DVector<Color> arr = p_property;
			int len=arr.size();
			f->store_32(len);
			DVector<Color>::Read r = arr.read();
			for(int i=0;i<len;i++) {
				f->store_real(r[i].r);
				f->store_real(r[i].g);
				f->store_real(r[i].b);
				f->store_real(r[i].a);
			}

		} break;
		default: {

			ERR_EXPLAIN("Invalid variant");
			ERR_FAIL();
		}
	}
}


void ObjectFormatSaverBinary::_find_resources(const Variant& p_variant) {


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

			res->get_property_list(&property_list);

			for(List<PropertyInfo>::Element *E=property_list.front();E;E=E->next()) {

				if (E->get().usage&PROPERTY_USAGE_STORAGE || (bundle_resources && E->get().usage&PROPERTY_USAGE_BUNDLE)) {

					_find_resources(res->get(E->get().name));
				}
			}

			SavedObject *so = memnew( SavedObject );
			_save_obj(res.ptr(),so);
			so->meta=res.get_ref_ptr();

			resource_map[ res ] = saved_resources.size();
			saved_resources.push_back(so);

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
Error ObjectFormatSaverBinary::_save_obj(const Object *p_object,SavedObject *so) {

	if (optimizer.is_valid()) {
		//use optimizer

		List<OptimizedSaver::Property> props;
		optimizer->get_property_list(p_object,&props);

		for(List<OptimizedSaver::Property>::Element *E=props.front();E;E=E->next()) {

			if (skip_editor && String(E->get().name).begins_with("__editor"))
				continue;
			_find_resources(E->get().value);
			SavedObject::SavedProperty sp;

			sp.name_idx=get_string_index(E->get().name);
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
				sp.name_idx=get_string_index(E->get().name);
				sp.value = p_object->get(E->get().name);
				_find_resources(sp.value);
				so->properties.push_back(sp);
			}
		}
	}

	return OK;

}

Error ObjectFormatSaverBinary::save(const Object *p_object,const Variant &p_meta) {

	ERR_FAIL_COND_V(!f,ERR_UNCONFIGURED);
	ERR_EXPLAIN("write_object should supply either an object, a meta, or both");
	ERR_FAIL_COND_V(!p_object && p_meta.get_type()==Variant::NIL, ERR_INVALID_PARAMETER);

	SavedObject *so = memnew( SavedObject );

	if (p_object)
		so->type=p_object->get_type();

	_find_resources(p_meta);
	so->meta=p_meta;
	Error err = _save_obj(p_object,so);
	ERR_FAIL_COND_V( err, ERR_INVALID_DATA );

	saved_objects.push_back(so);

	return OK;
}

void ObjectFormatSaverBinary::save_unicode_string(const String& p_string) {


	CharString utf8 = p_string.utf8();
	f->store_32(utf8.length()+1);
	f->store_buffer((const uint8_t*)utf8.get_data(),utf8.length()+1);
}

ObjectFormatSaverBinary::ObjectFormatSaverBinary(FileAccess *p_file,const String& p_magic,const String& p_local_path,uint32_t p_flags,const Ref<OptimizedSaver>& p_optimizer) {

	optimizer=p_optimizer;
	relative_paths=p_flags&ObjectSaver::FLAG_RELATIVE_PATHS;
	skip_editor=p_flags&ObjectSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	bundle_resources=p_flags&ObjectSaver::FLAG_BUNDLE_RESOURCES;
	big_endian=p_flags&ObjectSaver::FLAG_SAVE_BIG_ENDIAN;
	f=p_file; // should be already opened
	local_path=p_local_path;
	magic=p_magic;

	bin_meta_idx = get_string_index("__bin_meta__"); //is often used, so create
}

int ObjectFormatSaverBinary::get_string_index(const String& p_string) {

	StringName s=p_string;
	if (string_map.has(s))
		return string_map[s];

	string_map[s]=strings.size();
	strings.push_back(s);
	return strings.size()-1;
}

ObjectFormatSaverBinary::~ObjectFormatSaverBinary() {


	static const uint8_t header[4]={'O','B','D','B'};
	f->store_buffer(header,4);
	if (big_endian) {
		f->store_32(1);
		f->set_endian_swap(true);
	} else
		f->store_32(0);

	f->store_32(0); //64 bits file, false for now
	f->store_32(VERSION_MAJOR);
	f->store_32(VERSION_MINOR);
	save_unicode_string(magic);
	for(int i=0;i<16;i++)
		f->store_32(0); // reserved

	f->store_32(strings.size()); //string table size
	for(int i=0;i<strings.size();i++) {
		print_bl("saving string: "+strings[i]);
		save_unicode_string(strings[i]);
	}

	// save resources

	for(int i=0;i<saved_resources.size();i++) {

		SavedObject *so = saved_resources[i];
		RES res = so->meta;
		ERR_CONTINUE(!resource_map.has(res));

		f->store_32(SECTION_RESOURCE);		
		size_t skip_pos = f->get_pos();
		f->store_64(0); // resource skip seek pos
		save_unicode_string(res->get_type());

		if (res->get_path().length() && res->get_path().find("::") == -1 )
			save_unicode_string(res->get_path());
		else
			save_unicode_string("local://"+itos(i));



		List<SavedObject::SavedProperty>::Element *SE = so->properties.front();

		while(SE) {

			write_property(SE->get().name_idx,SE->get().value);
			SE=SE->next();
		}

		f->store_32(SECTION_END);

		size_t end=f->get_pos();
		f->seek(skip_pos);
		f->store_64(end);
		f->seek_end();

		memdelete( so );
	}

	if (!saved_objects.empty()) {


		for(List<SavedObject*>::Element *E=saved_objects.front();E;E=E->next()) {

			SavedObject *so = E->get();


			size_t section_end;

			if (so->type!="") {
				f->store_32(SECTION_OBJECT);
				section_end=f->get_pos();
				f->store_64(0); //section end
				save_unicode_string(so->type);
			} else {
				f->store_32(SECTION_META_OBJECT);
				section_end=f->get_pos();
				f->store_64(0); //section end
			}


			if (so->meta.get_type()!=Variant::NIL)
				write_property(bin_meta_idx,so->meta);

			List<SavedObject::SavedProperty>::Element *SE = so->properties.front();

			while(SE) {

				write_property(SE->get().name_idx,SE->get().value);
				SE=SE->next();
			}

			f->store_32(SECTION_END);

			size_t end=f->get_pos();
			f->seek(section_end);
			f->store_64(end);
			f->seek_end();

			memdelete(so); //no longer needed
		}


	}

	f->store_32(SECTION_END);

	f->close();
	memdelete(f);
}


ObjectFormatSaver* ObjectFormatSaverInstancerBinary::instance(const String& p_file,const String& p_magic,uint32_t p_flags,const Ref<OptimizedSaver>& p_optimizer) {

	FileAccess *f = FileAccess::open(p_file, FileAccess::WRITE);

	ERR_FAIL_COND_V( !f, NULL );
	String local_path = Globals::get_singleton()->localize_path(p_file);

	return memnew( ObjectFormatSaverBinary( f, p_magic,local_path,p_flags,p_optimizer ) );
}

void ObjectFormatSaverInstancerBinary::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("bin");
}


ObjectFormatSaverInstancerBinary::~ObjectFormatSaverInstancerBinary() {


}



/************************************************/
/************************************************/
/************************************************/
/************************************************/
/************************************************/


void ObjectFormatLoaderBinary::_advance_padding(uint32_t p_len) {

	uint32_t extra = 4-(p_len%4);
	if (extra<4) {
		for(uint32_t i=0;i<extra;i++)
			f->get_8(); //pad to 32
	}

}

Error ObjectFormatLoaderBinary::parse_property(Variant& r_v, int &r_index)  {


	uint32_t prop = f->get_32();
	if (prop==SECTION_END)
		return ERR_FILE_EOF;
	ERR_FAIL_COND_V(prop!=SECTION_PROPERTY,ERR_FILE_CORRUPT);

	r_index = f->get_32();

	uint32_t type = f->get_32();
	print_bl("find property of type: "+itos(type));


	switch(type) {

		case VARIANT_NIL: {

			r_v=Variant();
		} break;
		case VARIANT_BOOL: {

			r_v=bool(f->get_32());
		} break;
		case VARIANT_INT: {

			r_v=int(f->get_32());
		} break;
		case VARIANT_REAL: {

			r_v=f->get_real();
		} break;
		case VARIANT_STRING: {

			r_v=get_unicode_string();
		} break;
		case VARIANT_VECTOR2: {

			Vector2 v;
			v.x=f->get_real();
			v.y=f->get_real();
			r_v=v;

		} break;
		case VARIANT_RECT2: {

			Rect2 v;
			v.pos.x=f->get_real();
			v.pos.y=f->get_real();
			v.size.x=f->get_real();
			v.size.y=f->get_real();
			r_v=v;

		} break;
		case VARIANT_VECTOR3: {

			Vector3 v;
			v.x=f->get_real();
			v.y=f->get_real();
			v.z=f->get_real();
			r_v=v;
		} break;
		case VARIANT_PLANE: {

			Plane v;
			v.normal.x=f->get_real();
			v.normal.y=f->get_real();
			v.normal.z=f->get_real();
			v.d=f->get_real();
			r_v=v;
		} break;
		case VARIANT_QUAT: {
			Quat v;
			v.x=f->get_real();
			v.y=f->get_real();
			v.z=f->get_real();
			v.w=f->get_real();
			r_v=v;

		} break;
		case VARIANT_AABB: {

			AABB v;
			v.pos.x=f->get_real();
			v.pos.y=f->get_real();
			v.pos.z=f->get_real();
			v.size.x=f->get_real();
			v.size.y=f->get_real();
			v.size.z=f->get_real();
			r_v=v;

		} break;
		case VARIANT_MATRIX32: {

			Matrix32 v;
			v.elements[0].x=f->get_real();
			v.elements[0].y=f->get_real();
			v.elements[1].x=f->get_real();
			v.elements[1].y=f->get_real();
			v.elements[2].x=f->get_real();
			v.elements[2].y=f->get_real();
			r_v=v;

		} break;
		case VARIANT_MATRIX3: {

			Matrix3 v;
			v.elements[0].x=f->get_real();
			v.elements[0].y=f->get_real();
			v.elements[0].z=f->get_real();
			v.elements[1].x=f->get_real();
			v.elements[1].y=f->get_real();
			v.elements[1].z=f->get_real();
			v.elements[2].x=f->get_real();
			v.elements[2].y=f->get_real();
			v.elements[2].z=f->get_real();
			r_v=v;

		} break;
		case VARIANT_TRANSFORM: {

			Transform v;
			v.basis.elements[0].x=f->get_real();
			v.basis.elements[0].y=f->get_real();
			v.basis.elements[0].z=f->get_real();
			v.basis.elements[1].x=f->get_real();
			v.basis.elements[1].y=f->get_real();
			v.basis.elements[1].z=f->get_real();
			v.basis.elements[2].x=f->get_real();
			v.basis.elements[2].y=f->get_real();
			v.basis.elements[2].z=f->get_real();
			v.origin.x=f->get_real();
			v.origin.y=f->get_real();
			v.origin.z=f->get_real();
			r_v=v;
		} break;
		case VARIANT_COLOR: {

			Color v;
			v.r=f->get_real();
			v.g=f->get_real();
			v.b=f->get_real();
			v.a=f->get_real();
			r_v=v;

		} break;
		case VARIANT_IMAGE: {


			uint32_t encoding = f->get_32();
			if (encoding==IMAGE_ENCODING_EMPTY) {
				r_v=Variant();
				break;
			}

			if (encoding==IMAGE_ENCODING_RAW) {
				uint32_t width = f->get_32();
				uint32_t height = f->get_32();
				uint32_t mipmaps = f->get_32();
				uint32_t format = f->get_32();
				Image::Format fmt;
				switch(format) {

					case IMAGE_FORMAT_GRAYSCALE: { fmt=Image::FORMAT_GRAYSCALE; } break;
					case IMAGE_FORMAT_INTENSITY: { fmt=Image::FORMAT_INTENSITY; } break;
					case IMAGE_FORMAT_GRAYSCALE_ALPHA: { fmt=Image::FORMAT_GRAYSCALE_ALPHA; } break;
					case IMAGE_FORMAT_RGB: { fmt=Image::FORMAT_RGB; } break;
					case IMAGE_FORMAT_RGBA: { fmt=Image::FORMAT_RGBA; } break;
					case IMAGE_FORMAT_INDEXED: { fmt=Image::FORMAT_INDEXED; } break;
					case IMAGE_FORMAT_INDEXED_ALPHA: { fmt=Image::FORMAT_INDEXED_ALPHA; } break;
					case IMAGE_FORMAT_BC1: { fmt=Image::FORMAT_BC1; } break;
					case IMAGE_FORMAT_BC2: { fmt=Image::FORMAT_BC2; } break;
					case IMAGE_FORMAT_BC3: { fmt=Image::FORMAT_BC3; } break;
					case IMAGE_FORMAT_BC4: { fmt=Image::FORMAT_BC4; } break;
					case IMAGE_FORMAT_BC5: { fmt=Image::FORMAT_BC5; } break;
					case IMAGE_FORMAT_CUSTOM: { fmt=Image::FORMAT_CUSTOM; } break;
					default: {

						ERR_FAIL_V(ERR_FILE_CORRUPT);
					}

				}


				uint32_t datalen = f->get_32();

				print_bl("width: "+itos(width));
				print_bl("height: "+itos(height));
				print_bl("mipmaps: "+itos(mipmaps));
				print_bl("format: "+itos(format));
				print_bl("datalen: "+itos(datalen));

				DVector<uint8_t> imgdata;
				imgdata.resize(datalen);
				DVector<uint8_t>::Write w = imgdata.write();
				f->get_buffer(w.ptr(),datalen);
				_advance_padding(datalen);
				w=DVector<uint8_t>::Write();

				r_v=Image(width,height,mipmaps,fmt,imgdata);
			}


		} break;
		case VARIANT_NODE_PATH: {

			r_v=NodePath(get_unicode_string());
		} break;
		case VARIANT_RID: {

			r_v=f->get_32();
		} break;
		case VARIANT_OBJECT: {

			uint32_t type=f->get_32();

			switch(type) {

				case OBJECT_EMPTY: {
					//do none

				} break;
				case OBJECT_INTERNAL_RESOURCE: {
					uint32_t index=f->get_32();
					String path = local_path+"::"+itos(index);
					RES res = ResourceLoader::load(path);
					if (res.is_null()) {
						WARN_PRINT(String("Couldn't load resource: "+path).utf8().get_data());
					}
					r_v=res;

				} break;
				case OBJECT_EXTERNAL_RESOURCE: {

					String type = get_unicode_string();
					String path = get_unicode_string();

					if (path.find("://")==-1 && path.is_rel_path()) {
						// path is relative to file being loaded, so convert to a resource path
						path=Globals::get_singleton()->localize_path(local_path.get_base_dir()+"/"+path);

					}

					RES res=ResourceLoader::load(path,type);

					if (res.is_null()) {
						WARN_PRINT(String("Couldn't load resource: "+path).utf8().get_data());
					}
					r_v=res;

				} break;
				default: {

					ERR_FAIL_V(ERR_FILE_CORRUPT);
				} break;
			}

		} break;
		case VARIANT_INPUT_EVENT: {

		} break;
		case VARIANT_DICTIONARY: {

			int len=f->get_32();
			Dictionary d;
			for(int i=0;i<len;i++) {
				int idx;
				Variant key;
				Error err = parse_property(key,idx);
				ERR_FAIL_COND_V(err,ERR_FILE_CORRUPT);
				Variant value;
				err = parse_property(value,idx);
				ERR_FAIL_COND_V(err,ERR_FILE_CORRUPT);
				d[key]=value;
			}
			r_v=d;
		} break;
		case VARIANT_ARRAY: {
			int len=f->get_32();
			Array a;
			a.resize(len);
			for(int i=0;i<len;i++) {
				int idx;
				Variant val;
				Error err = parse_property(val,idx);
				ERR_FAIL_COND_V(err,ERR_FILE_CORRUPT);
				a[i]=val;
			}
			r_v=a;

		} break;
		case VARIANT_RAW_ARRAY: {

			uint32_t len = f->get_32();

			DVector<uint8_t> array;
			array.resize(len);
			DVector<uint8_t>::Write w = array.write();
			f->get_buffer(w.ptr(),len);
			_advance_padding(len);
			w=DVector<uint8_t>::Write();
			r_v=array;

		} break;
		case VARIANT_INT_ARRAY: {

			uint32_t len = f->get_32();

			DVector<int> array;
			array.resize(len);
			DVector<int>::Write w = array.write();
			f->get_buffer((uint8_t*)w.ptr(),len*4);
			w=DVector<int>::Write();
			r_v=array;
		} break;
		case VARIANT_REAL_ARRAY: {

			uint32_t len = f->get_32();

			DVector<real_t> array;
			array.resize(len);
			DVector<real_t>::Write w = array.write();
			f->get_buffer((uint8_t*)w.ptr(),len*sizeof(real_t));
			w=DVector<real_t>::Write();
			r_v=array;
		} break;
		case VARIANT_STRING_ARRAY: {

			uint32_t len = f->get_32();
			DVector<String> array;
			array.resize(len);
			DVector<String>::Write w = array.write();
			for(int i=0;i<len;i++)
				w[i]=get_unicode_string();
			w=DVector<String>::Write();
			r_v=array;


		} break;
		case VARIANT_VECTOR2_ARRAY: {

			uint32_t len = f->get_32();

			DVector<Vector2> array;
			array.resize(len);
			DVector<Vector2>::Write w = array.write();
			if (sizeof(Vector2)==8) {
				f->get_buffer((uint8_t*)w.ptr(),len*sizeof(real_t)*2);
			} else {
				ERR_EXPLAIN("Vector2 size is NOT 8!");
				ERR_FAIL_V(ERR_UNAVAILABLE);
			}
			w=DVector<Vector2>::Write();
			r_v=array;

		} break;
		case VARIANT_VECTOR3_ARRAY: {

			uint32_t len = f->get_32();

			DVector<Vector3> array;
			array.resize(len);
			DVector<Vector3>::Write w = array.write();
			if (sizeof(Vector3)==12) {
				f->get_buffer((uint8_t*)w.ptr(),len*sizeof(real_t)*3);
			} else {
				ERR_EXPLAIN("Vector3 size is NOT 12!");
				ERR_FAIL_V(ERR_UNAVAILABLE);
			}
			w=DVector<Vector3>::Write();
			r_v=array;

		} break;
		case VARIANT_COLOR_ARRAY: {

			uint32_t len = f->get_32();

			DVector<Color> array;
			array.resize(len);
			DVector<Color>::Write w = array.write();
			if (sizeof(Color)==16) {
				f->get_buffer((uint8_t*)w.ptr(),len*sizeof(real_t)*4);
			} else {
				ERR_EXPLAIN("Color size is NOT 16!");
				ERR_FAIL_V(ERR_UNAVAILABLE);
			}
			w=DVector<Color>::Write();
			r_v=array;
		} break;

		default: {
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		} break;
	}



	return OK; //never reach anyway

}

Error ObjectFormatLoaderBinary::load(Object **p_object,Variant &p_meta) {



	while(true) {

		if (f->eof_reached()) {
			ERR_EXPLAIN("Premature end of file at: "+local_path);
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		}

		RES resource;
		Object *obj=NULL;
		bool meta=false;

		uint32_t section = f->get_32();

		switch(section) {


			case SECTION_RESOURCE: {

				print_bl("resource found");

				size_t section_end = f->get_64();
				print_bl("section end: "+itos(section_end));
				String type = get_unicode_string();
				String path = get_unicode_string();
				print_bl("path: "+path);

				if (path.begins_with("local://")) {
					//built-in resource (but really external)
					path=path.replace("local://",local_path+"::");
				}

				if (ResourceCache::has(path)) {
					f->seek(section_end);
					continue;
				}

				//load properties


				obj = ObjectTypeDB::instance(type);
				if (!obj) {
					ERR_EXPLAIN("Object of unrecognized type '"+type+"' in file: "+type);
				}

				ERR_FAIL_COND_V(!obj,ERR_FILE_CORRUPT);

				Resource *r = obj->cast_to<Resource>();
				if (!r) {
					memdelete(obj); //bye
					ERR_EXPLAIN("Object type in resource field not a resource, type is: "+obj->get_type());
					ERR_FAIL_COND_V(!obj->cast_to<Resource>(),ERR_FILE_CORRUPT);
				}

				resource = RES( r );
				r->set_path(path);
			} break;
			case SECTION_META_OBJECT:
				meta=true;
				print_bl("meta found");

			case SECTION_OBJECT: {

				uint64_t section_end = f->get_64();

				if (!meta) {
					print_bl("object");

					String type = get_unicode_string();
					if (ObjectTypeDB::can_instance(type)) {
						obj = ObjectTypeDB::instance(type);
						if (!obj) {
							ERR_EXPLAIN("Object of unrecognized type in file: "+type);
						}
						ERR_FAIL_COND_V(!obj,ERR_FILE_CORRUPT);
					} else {

						f->seek(section_end);
						return ERR_SKIP;
					};


				}


			} break;
			case SECTION_END: {


				return ERR_FILE_EOF;
			} break;

			default: {

				ERR_EXPLAIN("Invalid Section ID '"+itos(section)+"' in file: "+local_path);
				ERR_FAIL_V(ERR_FILE_CORRUPT);

			}

		}


		//load properties

		while(true) {

			int name_idx;
			Variant v;
			Error err;
			err = parse_property(v,name_idx);

			print_bl("prop idx "+itos(name_idx)+" value: "+String(v));

			if (err==ERR_FILE_EOF)
				break;

			if (err!=OK) {
				ERR_EXPLAIN("File Corrupted");
				ERR_FAIL_COND_V(err!=OK,ERR_FILE_CORRUPT);
			}


			if (resource.is_null() && name_idx==0) { //0 is __bin_meta__

				p_meta=v;
				continue;
			} else if (!obj) {

				ERR_EXPLAIN("Normal property found in meta object.");
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}

			Map<int,StringName>::Element *E=string_map.find(name_idx);
			if (!E) {
				ERR_EXPLAIN("Property ID has no matching name: "+itos(name_idx));
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}

			obj->set(E->get(),v);
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

	}
}


ObjectFormatLoaderBinary::~ObjectFormatLoaderBinary() {

	if (f) {
		if (f->is_open())
			f->close();
		memdelete(f);
	}
}


String ObjectFormatLoaderBinary::get_unicode_string() {

	uint32_t len = f->get_32();
	if (len>str_buf.size()) {
		str_buf.resize(len);
	}
	f->get_buffer((uint8_t*)&str_buf[0],len);
	String s;
	s.parse_utf8(&str_buf[0]);
	return s;
}

ObjectFormatLoaderBinary::ObjectFormatLoaderBinary(FileAccess *p_f,bool p_endian_swap,bool p_use64) {

	f=p_f;
	endian_swap=p_endian_swap;
	use_real64=p_use64;

	//load string table
	uint32_t string_table_size = f->get_32();
	print_bl("string table size: "+itos(string_table_size));
	for(int i=0;i<string_table_size;i++) {

		String str = get_unicode_string();
		print_bl("string "+itos(i)+" is: "+str);
		string_map[i]=str;
	}


}

ObjectFormatLoaderBinary* ObjectFormatLoaderInstancerBinary::instance(const String& p_file,const String& p_magic) {

	FileAccess *f=FileAccess::open(p_file,FileAccess::READ);
	ERR_FAIL_COND_V(!f,NULL);

	uint8_t header[4];
	f->get_buffer(header,4);
	if (header[0]!='O' || header[1]!='B' || header[2]!='D' || header[3]!='B') {

		ERR_EXPLAIN("File not in valid binary format: "+p_file);
		ERR_FAIL_V(NULL);
	}

	uint32_t big_endian = f->get_32();
#ifdef BIG_ENDIAN_ENABLED
	bool endian_swap = !big_endian;
#else
	bool endian_swap = big_endian;
#endif

	bool use_real64 = f->get_32();

	f->set_endian_swap(big_endian!=0); //read big endian if saved as big endian

	uint32_t ver_major=f->get_32();
	uint32_t ver_minor=f->get_32();

	print_bl("big endian: "+itos(big_endian));
	print_bl("endian swap: "+itos(endian_swap));
	print_bl("real64: "+itos(use_real64));
	print_bl("major: "+itos(ver_major));
	print_bl("minor: "+itos(ver_minor));

	if (ver_major>VERSION_MAJOR || (ver_major==VERSION_MAJOR && ver_minor>VERSION_MINOR)) {

		f->close();
		memdelete(f);
		ERR_EXPLAIN("File Format '"+itos(ver_major)+"."+itos(ver_minor)+"' is too new! Please upgrade to a a new engine version: "+p_file);
		ERR_FAIL_V(NULL);

	}

	uint32_t magic_len = f->get_32();
	Vector<char> magic;
	magic.resize(magic_len);
	f->get_buffer((uint8_t*)&magic[0],magic_len);
	String magic_str;
	magic_str.parse_utf8(&magic[0]);

	print_bl("magic: "+magic_str);
	if (magic_str!=p_magic) {

		f->close();
		memdelete(f);
		ERR_EXPLAIN("File magic mismatch, found '"+magic_str+"' in : "+p_file);
		ERR_FAIL_V(NULL);
	}

	print_bl("skipping 32");
	for(int i=0;i<16;i++)
		f->get_32(); //skip a few reserved fields

	if (f->eof_reached()) {

		f->close();
		memdelete(f);
		ERR_EXPLAIN("Premature End Of File: "+p_file);
		ERR_FAIL_V(NULL);

	}

	print_bl("creating loader");
	ObjectFormatLoaderBinary *loader = memnew( ObjectFormatLoaderBinary(f,endian_swap,use_real64) );
	loader->local_path=p_file;

	return loader;
}

void ObjectFormatLoaderInstancerBinary::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("bin");
}


#endif

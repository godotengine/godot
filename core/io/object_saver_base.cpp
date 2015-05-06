/*************************************************************************/
/*  object_saver_base.cpp                                                */
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
#include "object_saver_base.h"
#ifdef OLD_SCENE_FORMAT_ENABLED
void ObjectSaverBase::_find_resources(const Variant& p_variant) {

	switch(p_variant.get_type()) {
		case Variant::OBJECT: {


			RES res = p_variant.operator RefPtr();

			if (res.is_null() || (res->get_path().length() && res->get_path().find("::") == -1 ))
				return;

			if (resource_map.has(res))
				return;

			List<PropertyInfo> property_list;

			res->get_property_list( &property_list );

			List<PropertyInfo>::Element *I=property_list.front();

			while(I) {

				PropertyInfo pi=I->get();

				if (pi.usage&PROPERTY_USAGE_STORAGE) {

					if (pi.type==Variant::OBJECT) {

						Variant v=res->get(I->get().name);
						_find_resources(v);
					}
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


Error ObjectSaverBase::save(const Object *p_object,const Variant &p_meta) {

	ERR_EXPLAIN("write_object should supply either an object, a meta, or both");
	ERR_FAIL_COND_V(!p_object && p_meta.get_type()==Variant::NIL, ERR_INVALID_PARAMETER);

	SavedObject *so = memnew( SavedObject );

	if (p_object) {
		so->type=p_object->get_type();
	};

	_find_resources(p_meta);
	so->meta=p_meta;

	if (p_object) {

		List<PropertyInfo> property_list;
		p_object->get_property_list( &property_list );

		List<PropertyInfo>::Element *I=property_list.front();

		while(I) {

			if (I->get().usage&PROPERTY_USAGE_STORAGE) {

				SavedObject::SavedProperty sp;
				sp.name=I->get().name;
				sp.value = p_object->get(I->get().name);
				_find_resources(sp.value);
				so->properties.push_back(sp);
			}

			I=I->next();
		}

	}

	saved_objects.push_back(so);

	return OK;
}

ObjectSaverBase::ObjectSaverBase() {

};

ObjectSaverBase::~ObjectSaverBase() {
	
};
#endif

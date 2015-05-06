/*************************************************************************/
/*  object_saver_base.h                                                  */
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
#ifndef OBJECT_SAVER_BASE_H
#define OBJECT_SAVER_BASE_H


#ifdef OLD_SCENE_FORMAT_ENABLED
#include "object_saver.h"

#include "map.h"
#include "resource.h"

class ObjectSaverBase : public ObjectFormatSaver {

protected:

	Map<RES,int> resource_map;

	struct SavedObject {

		Variant meta;
		String type;


		struct SavedProperty {

			String name;
			Variant value;
		};

		List<SavedProperty> properties;
	};

	List<RES> saved_resources;

	List<SavedObject*> saved_objects;

	void _find_resources(const Variant& p_variant);

	virtual Error write()=0;
public:

	virtual Error save(const Object *p_object,const Variant &p_meta);

	ObjectSaverBase();
	~ObjectSaverBase();
};

#endif
#endif // OBJECT_SAVER_BASE_H

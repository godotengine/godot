/**************************************************************************/
/*  object_gdextension.h                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/extension/gdextension_interface.gen.h"
#include "core/object/gdtype.h"
#include "core/string/string_name.h"
#include "core/templates/list.h"

// API used to extend in GDExtension and other C compatible compiled languages.
class MethodBind;
class GDExtension;

struct ObjectGDExtension {
	GDExtension *library = nullptr;
	ObjectGDExtension *parent = nullptr;
	List<ObjectGDExtension *> children;
	StringName parent_class_name;
	StringName class_name;
	bool editor_class = false;
	bool reloadable = false;
	bool is_virtual = false;
	bool is_abstract = false;
	bool is_exposed = true;
#ifdef TOOLS_ENABLED
	bool is_runtime = false;
	bool is_placeholder = false;
#endif
#ifndef DISABLE_DEPRECATED
	bool legacy_unexposed_class = false;
#endif // DISABLE_DEPRECATED
	GDExtensionClassSet set;
	GDExtensionClassGet get;
	GDExtensionClassGetPropertyList get_property_list;
	GDExtensionClassFreePropertyList2 free_property_list2;
	GDExtensionClassPropertyCanRevert property_can_revert;
	GDExtensionClassPropertyGetRevert property_get_revert;
	GDExtensionClassValidateProperty validate_property;
#ifndef DISABLE_DEPRECATED
	GDExtensionClassNotification notification;
	GDExtensionClassFreePropertyList free_property_list;
#endif // DISABLE_DEPRECATED
	GDExtensionClassNotification2 notification2;
	GDExtensionClassToString to_string;
	GDExtensionClassReference reference;
	GDExtensionClassReference unreference;
	GDExtensionClassGetRID get_rid;

	void *class_userdata = nullptr;

#ifndef DISABLE_DEPRECATED
	GDExtensionClassCreateInstance create_instance;
#endif // DISABLE_DEPRECATED
	GDExtensionClassCreateInstance2 create_instance2;
	GDExtensionClassFreeInstance free_instance;
#ifndef DISABLE_DEPRECATED
	GDExtensionClassGetVirtual get_virtual;
	GDExtensionClassGetVirtualCallData get_virtual_call_data;
#endif // DISABLE_DEPRECATED
	GDExtensionClassGetVirtual2 get_virtual2;
	GDExtensionClassGetVirtualCallData2 get_virtual_call_data2;
	GDExtensionClassCallVirtualWithData call_virtual_with_data;
	GDExtensionClassRecreateInstance recreate_instance;

#ifdef TOOLS_ENABLED
	void *tracking_userdata = nullptr;
	void (*track_instance)(void *p_userdata, void *p_instance) = nullptr;
	void (*untrack_instance)(void *p_userdata, void *p_instance) = nullptr;
#endif

	/// A type for this Object extension.
	/// This is not exposed through the GDExtension API (yet) so it is inferred from above parameters.
	GDType *gdtype;
	void create_gdtype();
	void destroy_gdtype();

	~ObjectGDExtension();
};

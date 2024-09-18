/**************************************************************************/
/*  openxr_binding_modifier.h                                             */
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

#ifndef OPENXR_BINDING_MODIFIER_H
#define OPENXR_BINDING_MODIFIER_H

#include "../action_map/openxr_action.h"
#include "core/io/resource.h"

// Part of implementation for:
// https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_KHR_binding_modification

struct XrBindingModificationBaseHeaderKHR;

class OpenXRBindingModifier : public Resource {
	GDCLASS(OpenXRBindingModifier, Resource);

private:
protected:
	static void _bind_methods();

	Ref<OpenXRAction> action; // Action, only applicable for BINDING_MODIFIER_IO_ACTION
	String input_path; // Input path, only applicable for BINDING_MODIFIER_IO_PATH and BINDING_MODIFIER_IO_ACTION

public:
	enum BindingModifierType {
		BINDING_MODIFIER_GLOBAL, // Binding applies to entire suggested interaction binding
		BINDING_MODIFIER_IO_PATH, // Binding is linked to a specific input/output path
		BINDING_MODIFIER_IO_ACTION, // Binding is linked to an action bound to a specific input/output path
	};

	void set_action(const Ref<OpenXRAction> p_action);
	Ref<OpenXRAction> get_action() const;

	void set_input_path(const String &p_input_path);
	String get_input_path() const;

	virtual BindingModifierType get_binding_modifier_type() const = 0; // Return our binding modifier type
	virtual int get_binding_modification_struct_size() const = 0; // Return the size of the struct returned by get_binding_modification
	virtual const XrBindingModificationBaseHeaderKHR *get_binding_modification() = 0; // Return the binding modifier struct used when calling xrSuggestInteractionProfileBindings
};

VARIANT_ENUM_CAST(OpenXRBindingModifier::BindingModifierType)

#endif // OPENXR_BINDING_MODIFIER_H

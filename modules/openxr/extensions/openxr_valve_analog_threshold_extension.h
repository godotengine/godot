/**************************************************************************/
/*  openxr_valve_analog_threshold_extension.h                             */
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

#ifndef OPENXR_VALVE_ANALOG_THRESHOLD_EXTENSION_H
#define OPENXR_VALVE_ANALOG_THRESHOLD_EXTENSION_H

#include "../action_map/openxr_binding_modifier.h"
#include "../util.h"
#include "core/io/resource.h"
#include "openxr_extension_wrapper.h"

#ifdef TOOLS_ENABLED
#include "../editor/openxr_binding_modifier_editor.h"
#endif // TOOLS_ENABLED

class OpenXRValveAnalogThresholdExtension : public OpenXRExtensionWrapper {
public:
	static OpenXRValveAnalogThresholdExtension *get_singleton();

	OpenXRValveAnalogThresholdExtension();
	virtual ~OpenXRValveAnalogThresholdExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions() override;

	bool is_available();

private:
	static OpenXRValveAnalogThresholdExtension *singleton;

	bool binding_modifier_ext = false;
	bool threshold_ext = false;
};

class OpenXRAnalogThresholdModifier : public OpenXRBindingModifier {
	GDCLASS(OpenXRAnalogThresholdModifier, OpenXRBindingModifier);

private:
	XrInteractionProfileAnalogThresholdVALVE analog_threshold;

protected:
	static void _bind_methods();

public:
	OpenXRAnalogThresholdModifier();

	void set_on_threshold(float p_threshold);
	float get_on_threshold() const;

	void set_off_threshold(float p_threshold);
	float get_off_threshold() const;

	virtual String get_description() const override { return "Analog threshold modifier"; }
	virtual PackedByteArray get_ip_modification() override;
};

#ifdef TOOLS_ENABLED

class OpenXRAnalogThresholdEditor : public OpenXRBindingModifierEditor {
	GDCLASS(OpenXRAnalogThresholdEditor, OpenXRBindingModifierEditor);

private:
	EditorPropertyFloat *on_threshold_property = nullptr;
	EditorPropertyFloat *off_threshold_property = nullptr;

protected:
	static void _bind_methods();

public:
	virtual void set_binding_modifier(Ref<OpenXRActionMap> p_action_map, Ref<OpenXRBindingModifier> p_binding_modifier) override;

	OpenXRAnalogThresholdEditor();
};

#endif // TOOLS_ENABLED

#endif // OPENXR_VALVE_ANALOG_THRESHOLD_EXTENSION_H

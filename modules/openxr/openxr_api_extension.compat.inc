/**************************************************************************/
/*  openxr_api_extension.compat.inc                                       */
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

#ifndef DISABLE_DEPRECATED

#include "extensions/openxr_extension_wrapper_extension.h"

void OpenXRAPIExtension::_register_composition_layer_provider_bind_compat_104087(OpenXRExtensionWrapperExtension *p_extension) {
	register_composition_layer_provider(p_extension);
}

void OpenXRAPIExtension::_unregister_composition_layer_provider_bind_compat_104087(OpenXRExtensionWrapperExtension *p_extension) {
	unregister_composition_layer_provider(p_extension);
}

void OpenXRAPIExtension::_register_projection_views_extension_bind_compat_104087(OpenXRExtensionWrapperExtension *p_extension) {
	register_projection_views_extension(p_extension);
}

void OpenXRAPIExtension::_unregister_projection_views_extension_bind_compat_104087(OpenXRExtensionWrapperExtension *p_extension) {
	unregister_projection_views_extension(p_extension);
}

void OpenXRAPIExtension::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("register_composition_layer_provider", "extension"), &OpenXRAPIExtension::_register_composition_layer_provider_bind_compat_104087);
	ClassDB::bind_compatibility_method(D_METHOD("unregister_composition_layer_provider", "extension"), &OpenXRAPIExtension::_unregister_composition_layer_provider_bind_compat_104087);
	ClassDB::bind_compatibility_method(D_METHOD("register_projection_views_extension", "extension"), &OpenXRAPIExtension::_register_projection_views_extension_bind_compat_104087);
	ClassDB::bind_compatibility_method(D_METHOD("unregister_projection_views_extension", "extension"), &OpenXRAPIExtension::_unregister_projection_views_extension_bind_compat_104087);
}

#endif // DISABLE_DEPRECATED

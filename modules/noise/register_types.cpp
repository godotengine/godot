/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "fastnoise_lite.h"
#include "noise.h"
#include "noise_texture_2d.h"
#include "noise_texture_3d.h"
#include "noise_texture_generator.h"

#ifdef TOOLS_ENABLED
#include "editor/noise_editor_plugin.h"
#endif

#ifdef TOOLS_ENABLED
#include "editor/plugins/editor_plugin.h"
#endif

NoiseTextureGenerator *_noise_texture_generator = nullptr;

void initialize_noise_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		_noise_texture_generator = memnew(NoiseTextureGenerator);
		GDREGISTER_CLASS(NoiseTextureGenerator);
		Engine::get_singleton()->add_singleton(Engine::Singleton("NoiseTextureGenerator", NoiseTextureGenerator::get_singleton()));
	}

	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(NoiseTexture3D);
		GDREGISTER_CLASS(NoiseTexture2D);
		GDREGISTER_ABSTRACT_CLASS(Noise);
		GDREGISTER_CLASS(FastNoiseLite);
		ClassDB::add_compatibility_class("NoiseTexture", "NoiseTexture2D");

		NoiseTextureGenerator::get_singleton()->init();
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorPlugins::add_by_type<NoiseEditorPlugin>();
	}
#endif
}

void uninitialize_noise_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		if (_noise_texture_generator) {
			_noise_texture_generator->finish();
			memdelete(_noise_texture_generator);
		}
	}

	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}

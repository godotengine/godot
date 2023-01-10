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

#include "core/engine.h"
#include "godot_navigation_server.h"
#include "servers/navigation_server.h"

#ifndef _3D_DISABLED
#include "navigation_mesh_generator.h"
#endif

#ifdef TOOLS_ENABLED
#include "navigation_mesh_editor_plugin.h"
#endif

#ifndef _3D_DISABLED
NavigationMeshGenerator *_nav_mesh_generator = nullptr;
#endif

NavigationServer *new_server() {
	return memnew(GodotNavigationServer);
}

void register_navigation_types() {
	NavigationServerManager::set_default_server(new_server);

#ifndef _3D_DISABLED
	_nav_mesh_generator = memnew(NavigationMeshGenerator);
	ClassDB::register_class<NavigationMeshGenerator>();
	Engine::get_singleton()->add_singleton(Engine::Singleton("NavigationMeshGenerator", NavigationMeshGenerator::get_singleton()));
#endif

#ifdef TOOLS_ENABLED
	EditorPlugins::add_by_type<NavigationMeshEditorPlugin>();
#endif
}

void unregister_navigation_types() {
#ifndef _3D_DISABLED
	if (_nav_mesh_generator) {
		memdelete(_nav_mesh_generator);
	}
#endif
}

/*************************************************************************/
/*  editor_run_native.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef EDITOR_RUN_NATIVE_H
#define EDITOR_RUN_NATIVE_H

#include "scene/gui/box_container.h"
#include "scene/gui/menu_button.h"

class EditorRunNative : public HBoxContainer {

	GDCLASS(EditorRunNative, BoxContainer);

	Map<int, MenuButton *> menus;
	bool first;
	bool deploy_dumb;
	bool deploy_debug_remote;
	bool debug_collisions;
	bool debug_navigation;

	void _run_native(int p_idx, int p_platform);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_deploy_dumb(bool p_enabled);
	bool is_deploy_dumb_enabled() const;

	void set_deploy_debug_remote(bool p_enabled);
	bool is_deploy_debug_remote_enabled() const;

	void set_debug_collisions(bool p_debug);
	bool get_debug_collisions() const;

	void set_debug_navigation(bool p_debug);
	bool get_debug_navigation() const;

	EditorRunNative();
};

#endif // EDITOR_RUN_NATIVE_H

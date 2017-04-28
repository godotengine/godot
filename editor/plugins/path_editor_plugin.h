/*************************************************************************/
/*  path_editor_plugin.h                                                 */
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
#ifndef PATH_EDITOR_PLUGIN_H
#define PATH_EDITOR_PLUGIN_H

#include "editor/spatial_editor_gizmos.h"
#include "scene/3d/path.h"
#if 0
class PathSpatialGizmo  : public EditorSpatialGizmo {

	GDCLASS(PathSpatialGizmo,EditorSpatialGizmo);

	Path* path;
	mutable Vector3 original;

public:

	virtual String get_handle_name(int p_idx) const;
	virtual Variant get_handle_value(int p_idx) const;
	virtual void set_handle(int p_idx,Camera *p_camera, const Point2& p_point);
	virtual void commit_handle(int p_idx,const Variant& p_restore,bool p_cancel=false);

	void redraw();
	PathSpatialGizmo(Path* p_path=NULL);

};

class PathEditorPlugin : public EditorPlugin {

	GDCLASS( PathEditorPlugin, EditorPlugin );


	Separator *sep;
	ToolButton *curve_create;
	ToolButton *curve_edit;
	ToolButton *curve_del;
	ToolButton *curve_close;

	EditorNode *editor;


	Path *path;

	void _mode_changed(int p_idx);
    void _close_curve();
protected:
	void _notification(int p_what);
	static void _bind_methods();

public:

	Path *get_edited_path() { return path; }

	static PathEditorPlugin* singleton;
	Ref<SpatialMaterial> path_material;
	Ref<SpatialMaterial> path_thin_material;
	virtual bool forward_spatial_gui_input(Camera* p_camera,const InputEvent& p_event);

	//virtual bool forward_gui_input(const InputEvent& p_event) { return collision_polygon_editor->forward_gui_input(p_event); }
	virtual Ref<SpatialEditorGizmo> create_spatial_gizmo(Spatial* p_spatial);
	virtual String get_name() const { return "Path"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	PathEditorPlugin(EditorNode *p_node);
	~PathEditorPlugin();

};

#endif
#endif // PATH_EDITOR_PLUGIN_H

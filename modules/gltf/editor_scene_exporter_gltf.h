/*************************************************************************/
/*  editor_scene_exporter_gltf.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/object.h"
#include "core/project_settings.h"
#include "core/vector.h"
#include "editor/editor_plugin.h"
#include "editor/saver/editor_saver_plugin.h"
#include "editor/saver/resource_saver_scene.h"
#include "modules/csg/csg_shape.h"
#include "modules/gridmap/grid_map.h"
#include "scene/3d/mesh_instance.h"
#include "scene/gui/check_box.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/surface_tool.h"

#ifndef EDITOR_SCENE_EXPORTER_GLTF_H
#define EDITOR_SCENE_EXPORTER_GLTF_H
class EditorSceneExporter;
class EditorSceneExporterGLTF : public EditorSceneExporter {

	GDCLASS(EditorSceneExporterGLTF, EditorSceneExporter);

public:
	virtual void save_scene(Node *p_node, const String &p_path, const String &p_src_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = NULL);

	struct MeshInfo {
		Transform transform;
		Ref<Mesh> mesh;
		String name;
		Vector<Ref<Material> > materials;
		Node *original_node = NULL;
	};
	void _find_all_gridmaps(Vector<GridMap *> &r_items, Node *p_current_node, const Node *p_owner);
	void _find_all_csg_roots(Vector<CSGShape *> &r_items, Node *p_current_node, const Node *p_owner);
	void get_exporter_extensions(List<String> *r_extensions) const;
	EditorSceneExporterGLTF() {}
	~EditorSceneExporterGLTF() {}
};

#endif

#ifndef EDITOR_FOLDING_H
#define EDITOR_FOLDING_H

#include "scene/main/node.h"

class EditorFolding {

	PoolVector<String> _get_unfolds(const Object *p_object);
	void _set_unfolds(Object *p_object, const PoolVector<String> &p_unfolds);

	void _fill_folds(const Node *p_root, const Node *p_node, Array &p_folds, Array &resource_folds, Set<RES> &resources);

public:
	void save_resource_folding(const RES &p_resource, const String &p_path);
	void load_resource_folding(RES p_resource, const String &p_path);

	void save_scene_folding(const Node *p_scene, const String &p_path);
	void load_scene_folding(Node *p_scene, const String &p_path);

	bool has_folding_data(const String &p_path);

	EditorFolding();
};

#endif // EDITOR_FOLDING_H

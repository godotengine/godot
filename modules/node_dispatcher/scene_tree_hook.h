#ifndef SCENE_TREE_HOOK_H
#define SCENE_TREE_HOOK_H

#include "node_dispatcher.h"
#include "modules/instance_pool/instance_pool.h"

class SceneTreeHook {
public:
	static void dispatch_idle(const float& delta);
	static void dispatch_physics(const float& delta);

	friend class SceneTree;
};

#endif

#ifndef SCENE_TREE_HOOK_H
#define SCENE_TREE_HOOK_H

#include "node_dispatcher.h"
#include "modules/instance_pool/instance_pool.h"
#include "modules/rts_com/combat_server.h"
#include "modules/enhancer/basic_scheduler.h"

class SceneTreeHook {
public:
	static _ALWAYS_INLINE_ void dispatch_idle(const float& delta){
		NodeDispatcher::get_singleton()->dispatch_idle();
		InstancePool::get_singleton()->dispatch_idle(delta);
	}
	static _ALWAYS_INLINE_ void dispatch_physics(const float& delta){
		NodeDispatcher::get_singleton()->dispatch_physics();
		Sentrience::get_singleton()->poll(delta);
		SynchronizationPoint::get_singleton()->iter_sync(delta);
	}
	static _ALWAYS_INLINE_ void dispatch_close(){
		// Sentrience::get_singleton()->pre_close();
	}

	friend class SceneTree;
};

#endif

#include "scene_tree_hook.h"

void SceneTreeHook::dispatch_idle(const float& delta){
	auto singleton1 = NodeDispatcher::get_singleton();
	if (singleton1) singleton1->dispatch_idle();
	auto singleton2 = InstancePool::get_singleton();
	if (singleton2) singleton2->dispatch_idle(delta);
}

void SceneTreeHook::dispatch_physics(const float& delta){
	auto singleton1 = NodeDispatcher::get_singleton();
	if (singleton1) singleton1->dispatch_physics();
	auto singleton2 = Sentrience::get_singleton();
	if (singleton2) singleton2->poll(delta);
}

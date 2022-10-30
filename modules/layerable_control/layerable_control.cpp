#include "layerable_control.h"

#define SIGNAL_LAYER_PUSH	StringName("layer_pushed")
#define SIGNAL_LAYER_POP	StringName("layer_popped")
#define SIGNAL_LAYER_EMPTY	StringName("layer_emptied")

#define VMETHOD_LAYER_PUSH	StringName("_push")
#define VMETHOD_LAYER_POP	StringName("_pop")

LayerableControl::LayerableControl(){
	connect(SceneStringNames::get_singleton()->child_entered_tree, this, StringName("add_to_stack"));
	connect(SceneStringNames::get_singleton()->child_exiting_tree, this, StringName("remove_from_stack"));
}

LayerableControl::~LayerableControl(){
	disconnect(SceneStringNames::get_singleton()->child_entered_tree, this, StringName("add_to_stack"));
	disconnect(SceneStringNames::get_singleton()->child_exiting_tree, this, StringName("remove_from_stack"));
	layer_stack.resize(0);
	all_layers.resize(0);
}

void LayerableControl::_notification(int p_notification){
	switch (p_notification){
		case NOTIFICATION_ENTER_TREE:
			Hub::get_singleton()->print_custom("LayerableControl", "LayerableControl is ready");
			return;
		default: return;
	}
}

void LayerableControl::_bind_methods(){
	ClassDB::bind_method(D_METHOD("add_to_stack", "node"), &LayerableControl::add_to_stack);
	ClassDB::bind_method(D_METHOD("remove_from_stack", "node"), &LayerableControl::remove_from_stack);

	ClassDB::bind_method(D_METHOD("get_location_in_all", "object_id"), &LayerableControl::get_location_in_all);
	ClassDB::bind_method(D_METHOD("get_location_in_stack", "object_id"), &LayerableControl::get_location_in_stack);
	
	ClassDB::bind_method(D_METHOD("push_by_name", "node_name"), &LayerableControl::push_by_name);
	ClassDB::bind_method(D_METHOD("push_by_id", "object_id"), &LayerableControl::push_by_id);
	ClassDB::bind_method(D_METHOD("push_by_index", "index"), &LayerableControl::push_by_index);
	ClassDB::bind_method(D_METHOD("auto_push"), &LayerableControl::auto_push);

	ClassDB::bind_method(D_METHOD("pop_by_name", "node_name"), &LayerableControl::pop_by_name);
	ClassDB::bind_method(D_METHOD("pop_by_id", "object_id"), &LayerableControl::pop_by_id);
	ClassDB::bind_method(D_METHOD("pop_by_index", "index"), &LayerableControl::pop_by_index);
	ClassDB::bind_method(D_METHOD("auto_pop"), &LayerableControl::auto_pop);

	ClassDB::bind_method(D_METHOD("get_layer_count"), &LayerableControl::get_layer_count);
	ClassDB::bind_method(D_METHOD("get_stack_count"), &LayerableControl::get_stack_count);

	ADD_SIGNAL(MethodInfo(SIGNAL_LAYER_PUSH));
	ADD_SIGNAL(MethodInfo(SIGNAL_LAYER_POP));
	ADD_SIGNAL(MethodInfo(SIGNAL_LAYER_EMPTY));

	BIND_VMETHOD(MethodInfo(Variant::BOOL, VMETHOD_LAYER_PUSH,	PropertyInfo(Variant::INT, "item_index")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, VMETHOD_LAYER_POP,	PropertyInfo(Variant::INT, "item_index")));
}

int LayerableControl::get_location_in_all(const ObjectID& object_id) const{
	uint32_t index = -1;
	for (int i = 0; i < all_layers.size(); i++){

		auto this_layer = all_layers[i];
		if (this_layer && this_layer->get_instance_id() == object_id){
			index = i;
			break;
		}
	}
	return index;
}
int LayerableControl::get_location_in_stack(const ObjectID& object_id) const{
	uint32_t index = -1;
	for (int i = 0; i < layer_stack.size(); i++){

		auto this_layer = layer_stack[i];
		if (this_layer && this_layer->get_instance_id() == object_id){
			index = i;
			break;
		}
	}
	return index;
}

void LayerableControl::add_to_stack(Node* child){
	if (child->is_class("CanvasItem") && is_inside_tree()){
		all_layers.push_back((CanvasItem*)child);
	}
}
void LayerableControl::remove_from_stack(Node* child){
	if (child && child->is_class("CanvasItem")){
		((CanvasItem*)child)->set_visible(false);
		auto object_id = child->get_instance_id();
		bool is_exist = false;
		for (int i = 0; i < all_layers.size(); i++){
			auto layer = all_layers[i];
			if (layer && layer->get_instance_id() == object_id) {
				all_layers.remove(i);
				is_exist = true;
				break;
			}
		}
		if (is_exist) pop_by_id(object_id);
	}
}
void LayerableControl::push_layer(const uint32_t& index){
	auto layer = all_layers[index];
	layer_stack.push_back(layer);
	emit_signal(SIGNAL_LAYER_PUSH, Variant(layer));
}
void LayerableControl::pop_layer(const uint32_t& index){
	auto layer = layer_stack[index];
	layer_stack.remove(index);
	emit_signal(SIGNAL_LAYER_POP, Variant(layer));
}
bool LayerableControl::push_by_name(const StringName& node_name){
	uint32_t index = -1;
	for (int i = 0; i < all_layers.size(); i++){
		auto this_layer = all_layers[i];
		if (this_layer && this_layer->get_name() == node_name){
			index = i;
			break;
		}
	}
	return push_by_index(index);
}
bool LayerableControl::push_by_id(const ObjectID& object_id){
	uint32_t index = get_location_in_all(object_id);
	return push_by_index(index);
}
bool LayerableControl::push_by_index(const uint64_t& index){
	if (index >= all_layers.size() || index < 0) return false;
	push_layer(index);
	return true;
}
bool LayerableControl::auto_push(){
	auto stack_count = layer_stack.size();
	auto index = stack_count;
	if (index >= all_layers.size()) return false;
	auto script = get_script_instance();
	if (script && script->has_method(VMETHOD_LAYER_PUSH)){
		auto res = script->call(VMETHOD_LAYER_PUSH, Variant(index));
		if (res.get_type() != Variant::BOOL) return false;
		return (bool)res;
	}
	push_layer(index);
	return true;
}
bool LayerableControl::pop_by_name(const StringName& node_name){
	auto index = -1;
	for (int i = 0; i < layer_stack.size(); i++){
		auto layer = layer_stack[i];
		if (layer && layer->get_name() == node_name) {
			index = i; break;
		}
	}
	return pop_by_index(index);
}
bool LayerableControl::pop_by_id(const ObjectID& object_id){
	auto index = get_location_in_stack(object_id);
	return pop_by_index(index);
}
bool LayerableControl::pop_by_index(const uint64_t& index){
	if (index >= layer_stack.size() || index < 0) return false;
	pop_layer(index);
	return true;
}
bool LayerableControl::auto_pop(){
	auto stack_count = layer_stack.size();
	auto index = stack_count - 1;
	if (index < 0) return false;
	auto script = get_script_instance();
	auto res = true;
	if (script && script->has_method(VMETHOD_LAYER_POP)){
		auto local_res = script->call(VMETHOD_LAYER_POP, Variant(index));
		if (local_res.get_type() != Variant::BOOL) res = false;
		res = res && (bool)local_res;
	}
	else pop_layer(index);
	if (index == 0) {
		emit_signal(SIGNAL_LAYER_EMPTY);
	}
	return true;
}

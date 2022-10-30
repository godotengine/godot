#ifndef LAYERABLE_CONTROL_H
#define LAYERABLE_CONTROL_H

#include <string>

#include "scene/scene_string_names.h"
#include "scene/gui/control.h"
#include "core/string_name.h"
#include "modules/hub/hub.h"
#include "core/vector.h"

class LayerableControl : public Control {
	GDCLASS(LayerableControl, Control);
private:
	Vector<CanvasItem*> layer_stack;
	Vector<CanvasItem*> all_layers;
protected:
	static void _bind_methods();
	void _notification(int p_notification);
	
	void push_layer(const uint32_t& index);
	void pop_layer(const uint32_t& index);
public:
	LayerableControl();
	~LayerableControl();

	int get_location_in_all(const ObjectID& object_id) const;
	int get_location_in_stack(const ObjectID& object_id) const;

	void add_to_stack(Node* child);
	void remove_from_stack(Node* child);

	bool push_by_name(const StringName& node_name);
	bool push_by_id(const ObjectID& object_id);
	bool push_by_index(const uint64_t& index);
	bool auto_push();

	bool pop_by_name(const StringName& node_name);
	bool pop_by_id(const ObjectID& object_id);
	bool pop_by_index(const uint64_t& index);
	bool auto_pop();

	_FORCE_INLINE_ uint32_t get_layer_count() const { return  all_layers.size(); }
	_FORCE_INLINE_ uint32_t get_stack_count() const { return layer_stack.size(); }
};

#endif

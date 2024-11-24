#pragma once

#include "scene/gui/scroll_container.h"
#include "scene/gui/flow_container.h"
class ItemBoxItem : public Control {
    GDCLASS(ItemBoxItem, Control);

public:

    ItemBoxItem() {
    }
    void update_visible() {
        Node *parent_node = get_parent();
        if(parent_node) {
			parent_node = parent_node->get_parent();
        }
		Control* parent = Object::cast_to<Control>(parent_node);
        if(parent) {
            bool visible = parent->get_global_rect().intersects(get_global_rect());
            for(int i = 0; i < get_child_count(); i++) {
				Control* child = Object::cast_to<Control>(get_child(i));
				if (child != nullptr) {
					child->set_visible(visible);
				}
            }

        }
        
    }
};

class ItemBox : public ScrollContainer {
    GDCLASS(ItemBox, ScrollContainer);
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("add_item", "item"), &ItemBox::add_item);
        ClassDB::bind_method(D_METHOD("remove_item", "item"), &ItemBox::remove_item);

        ClassDB::bind_method(D_METHOD("set_item_size", "size"), &ItemBox::set_item_size);
        ClassDB::bind_method(D_METHOD("get_item_size"), &ItemBox::get_item_size);

        ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "item_size"), "set_item_size", "get_item_size");
    }
public:

    ItemBox() {
        view_root = memnew(HFlowContainer);
        view_root->set_h_size_flags(SIZE_EXPAND_FILL);
        view_root->set_v_size_flags(SIZE_EXPAND_FILL);
        add_child(view_root);

		connect(SceneStringName(sort_children), callable_mp(this, &ItemBox::scroll_changed));
    }

    void scroll_changed(float ) {
        for(auto it : items) {
            it->update_visible();
        }
        
    }

    void add_item(Control *item) {
        ItemBoxItem *it = memnew(ItemBoxItem);
        it->set_custom_minimum_size(item_size);
        view_root->add_child(it);
        it->add_child(item);
        item->set_h_size_flags(SIZE_EXPAND_FILL);
        item->set_v_size_flags(SIZE_EXPAND_FILL);
        items.push_back(it);
    }
    void remove_item(Control *item) {
        
        for(auto it : items) {
            if(it->get_child(0) == item) {
				view_root->remove_child(it);
                it->queue_free();
                items.erase(it);
                break;
            }
        }
    }
    void set_item_size(Vector2 size) {
        item_size = size;
        for(auto it : items) {
            it->set_custom_minimum_size(item_size);
        }
    }
    Vector2 get_item_size() {
        return item_size;
    }

    void clear() {
        for(auto it : items) {
            view_root->remove_child(it);
            it->queue_free();
        }
        items.clear();
    }
protected:
    HFlowContainer *view_root = nullptr;
    List<ItemBoxItem *> items;
    Vector2 item_size = Vector2(150, 150);
    
};

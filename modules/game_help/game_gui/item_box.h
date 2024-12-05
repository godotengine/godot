#pragma once

#include "scene/gui/scroll_container.h"
#include "scene/gui/flow_container.h"
class ItemBoxItem : public Control {
    GDCLASS(ItemBoxItem, Control);

public:

    ItemBoxItem() {
    }
    Ref<RefCounted> data;
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
    }
    void _notification(int what) {
        if(what == NOTIFICATION_EXIT_TREE) {
            clear();
        }
    }

    void set_item_visible_change_callback(Callable& p_item_visible_change_cb) {
        item_visible_change_cb = p_item_visible_change_cb;
    }

    void scroll_changed(float ) {
        Rect2 rect = get_global_rect();
        for(auto it : items) {
            bool _visible = rect.intersects(it->get_global_rect());
            item_visible_change_cb.call(it,_visible);
        }
        
    }

    void add_item(const Ref<RefCounted>& item) {
        ItemBoxItem *it = memnew(ItemBoxItem);
        it->set_custom_minimum_size(item_size);
        view_root->add_child(it);

        it->data = item;
        items.push_back(it);

        bool _visible = it->get_global_rect().intersects(get_global_rect());
        item_visible_change_cb.call(it,_visible);
    }
    void remove_item(const Ref<RefCounted>& item) {
        
        for(auto it : items) {
            if(it->data == item) {
                item_visible_change_cb.call(it,false);
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
            item_visible_change_cb.call(it,false);
            view_root->remove_child(it);
            it->queue_free();
        }
        items.clear();
    }
protected:
    HFlowContainer *view_root = nullptr;
    List<ItemBoxItem *> items;
    Vector2 item_size = Vector2(150, 150);
    Callable item_visible_change_cb;
    
};

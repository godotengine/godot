#pragma once

#include "scene/gui/scroll_container.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/box_container.h"
class ItemBoxItem : public HBoxContainer {
    GDCLASS(ItemBoxItem, HBoxContainer);

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
        view_root->set_scale(Vector2(0.7, 0.7));
        view_root->set_h_size_flags(SIZE_EXPAND_FILL);
        view_root->set_v_size_flags(SIZE_EXPAND_FILL);
        add_child(view_root);
    }
    void _notification(int what) {
        if(what == NOTIFICATION_EXIT_TREE) {
            clear();
        }
    }

    void set_item_visible_change_callback(const Callable& p_item_visible_change_cb) {
        item_visible_change_cb = p_item_visible_change_cb;
    }

    void _scroll_changed(float ) override{
        Rect2 rect = get_global_rect();
        rect.position = Vector2(0,0);
        Rect2 child_rect = Rect2(0,0,item_size.x,item_size.y);
		if (rect.size.x < 5) {
			for (auto it : items) {
				// 一行第一個能顯示，剩餘的都能顯示
				item_visible_change_cb.call(it, false);
			}
			return;
		}
        int h_count = rect.size.x / item_size.x;
		
        int h_diff = rect.size.x - (h_count * item_size.x);
		if (h_count == 0) {
			h_count = 1;
		}
        int v_move = get_v_scroll() * get_vertical_custom_step();
        int i = 0;
        for(auto it : items) {
            // 一行第一個能顯示，剩餘的都能顯示
            child_rect.position.y = i / h_count * item_size.y + v_move;
            bool _visible = rect.intersects(child_rect);
            item_visible_change_cb.call(it,_visible);
            ++i;
        }
        is_dirty = false;
    }

    void add_item(const Ref<RefCounted>& item) {
        ItemBoxItem *it = memnew(ItemBoxItem);
        it->set_custom_minimum_size(item_size);
        view_root->add_child(it);

        it->data = item;
        items.push_back(it);

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
    Vector2 item_size = Vector2(400, 400);
    Callable item_visible_change_cb;
    bool is_dirty = false;
    
};

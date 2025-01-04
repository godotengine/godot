#pragma once
#include "scene/gui/button.h"
#include "core/object/undo_redo.h"

#define TRACK_HEIGHT  30
#define LINE_HEIGHT  4
class TimeLineClipData : public Resource {
    GDCLASS(TimeLineClipData, Resource);
    static void _bind_methods() {}
public:
    TimeLineClipData() {}
    static int64_t time_to_frame(double a_time) {
        // 設置成一秒為120個frame
        return (int64_t)(a_time * 120);
    }
    int duration = 0;
    int start_frame = 0;
    int begin = 0;
    int max_left_resize = 0;
    int max_right_resize = 0;
};
class Timeline : public Resource {
    GDCLASS(Timeline, Resource);
    static void _bind_methods() {}
public:
    Timeline() {}
    Ref<TimeLineClipData> get_clip_data(int a_track,int a_frame) { 
        return tracks[a_track][a_frame]; 
    }
    void _add_clip(Ref<TimeLineClipData> a_clip_data,int a_track_id) {
        tracks[a_track_id][a_clip_data->start_frame] = a_clip_data;
    }
    void delete_clip(int a_track_id,int a_frame_nr) {
        tracks[a_track_id].erase(a_frame_nr);
    }

    void undelete_clip(Ref<TimeLineClipData> a_clip_data,int a_track_id) {
        _add_clip(a_clip_data,a_track_id);
    }    

    Callable on_delete_clip;
    Callable on_undelete_clip;

    HashMap<int,Ref<TimeLineClipData>> clips;
    LocalVector<RBMap<int64_t, Ref<TimeLineClipData>>> tracks;
    float timeline_scale = 1.0f;

    int frame_nr = 0;
};



class TimelineButton : public Button {
    GDCLASS(TimelineButton, Button);
    static void _bind_methods() {}
public:
    void _add_risize_button(LayoutPreset a_preset,bool a_left) {

        Button* l_button = memnew(Button);
        add_child(l_button);

        Ref<StyleBoxEmpty> l_style = memnew(StyleBoxEmpty);
        l_button->add_theme_style_override("normal", l_style);
        l_button->add_theme_style_override("pressed", l_style);
        l_button->add_theme_style_override("hover", l_style);
        l_button->add_theme_style_override("focus", l_style);
        
        l_button->set_default_cursor_shape(CURSOR_HSIZE);
        l_button->set_anchors_and_offsets_preset(a_preset);
        Size2 size = l_button->get_size();
        size.x = 3;
        l_button->set_size(size);
        if(!a_left) {
            Point2 pos = l_button->get_position();
            pos.x -= 3;
            l_button->set_position(pos);
        }
        l_button->set_mouse_filter(MOUSE_FILTER_PASS);
		TimelineButton* tb = (TimelineButton*)this;
        l_button->connect("button_down",callable_mp(tb, &TimelineButton::_on_risize_engaged).bind(a_left));
        l_button->connect("button_up",callable_mp(tb, &TimelineButton::_on_commit_resize));


    }
    void _process(float delta) {
        if(is_resizing_left || is_resizing_right) {
            int64_t l_new_frame = get_frame_nr(((Control*)get_parent())->get_local_mouse_position().x );
            l_new_frame = CLAMP(l_new_frame ,
            max_left_resize, max_right_resize != -1 ? max_right_resize : INT64_MAX);
            
            if(is_resizing_left) {
                Size2 size = get_size();
                size.x = (l_new_frame - start_frame) * timeline->timeline_scale;
                set_size(size);
            }
            else if(is_resizing_right) {
                Point2 pos = get_position();
                Size2 size = get_size();
                pos.x = l_new_frame * timeline->timeline_scale;
                size.x = (duration - (l_new_frame - start_frame)) * timeline->timeline_scale;
                set_position(pos);
                set_size(size);
            }
        }
    }
    
    void _on_button_down() {
        is_dragging = true;
    }
	virtual void node_input(const Ref<InputEvent> &p_gui_input)const override{
        if(is_pressed() && p_gui_input->is_action_pressed("ui_accept")) {
            undo_redo->create_action("Delete clip on timeline");

			TimelineButton* tb = (TimelineButton*)this;
            undo_redo->add_do_method(callable_mp(tb, &TimelineButton::_cut_clip).bind(timeline->frame_nr));
            undo_redo->add_undo_method(callable_mp(tb, &TimelineButton::_uncut_clip).bind(timeline->frame_nr));

            undo_redo->commit_action();
        }
    }

    void _on_gui_input(const Ref<InputEvent> &p_event) {

        Ref<InputEventMouse> mb = p_event;
        if(mb.is_valid()) {
            Ref<InputEventWithModifiers> m = p_event;
            if(!mb->is_alt_pressed() && mb->is_released()) {
                
            }
        }

        if(p_event->is_action_pressed("ui_graph_delete")) {
            undo_redo->create_action("Delete clip on timeline");

            Point2 pos = get_position();

            undo_redo->add_do_method(callable_mp(timeline.ptr(), &Timeline::delete_clip).bind(get_track_id(pos.y),get_frame_nr(pos.x)));
            undo_redo->add_undo_method(callable_mp(timeline.ptr(), &Timeline::undelete_clip).bind(clip_data, get_track_id(pos.y)));

            undo_redo->commit_action();
        }
        
    }
    void _notification(int p_what) {
        if(p_what == NOTIFICATION_DRAG_END) {
            if(is_dragging) {
                is_dragging = false;
                set_modulate(Color(1, 1, 1, 1));
            }
        }
    }
    virtual Variant get_drag_data(const Point2 &p_point) {
        if (is_resizing_left || is_resizing_right)
        {
            return Variant();
        }
        Dictionary d;

        Vector2i l_ignore = Vector2i(get_track_id(p_point.y), get_frame_nr(p_point.x));

        Array ids;
        ObjectID id = get_instance_id();
        ids.push_back(id);
        d["type"] = "timeline_clip";
        d["ids"] = ids;

        Array ignore;
        ignore.push_back(l_ignore);
        d["ignore"] = ignore;

        d["files"] = false;

        d["duration"] = clip_data->duration;

        d["mouse_offset"] = get_frame_nr(get_local_mouse_position().x);

        Array clip_buttons;
        clip_buttons.push_back(this);

        set_modulate(Color(1, 1, 1, 0.1));
        return d;
    }


    static int64_t floori(double x) {
        return int64_t(Math::floor(x));
    }
    static int64_t get_track_id(float a_pos_y) {

	    return floori(a_pos_y / (TRACK_HEIGHT + LINE_HEIGHT));
    }
    int64_t get_frame_nr(float a_pos_x ) {
        
		return floori(a_pos_x / timeline->timeline_scale);
    }
    void _on_risize_engaged(bool a_left) {
        Point2 pos = get_position();
        int64_t l_track = get_track_id(pos.y);
        int64_t l_frame = get_frame_nr(pos.x);

        int64_t l_previous = -1;
        start_frame = clip_data->start_frame;
        duration = clip_data->duration;

        Ref<TimeLineClipData> _clip;
        RBMap<int64_t, Ref<TimeLineClipData>>& track = timeline->tracks[l_track];
        if(a_left) {
            for(auto it : track) {
                if(it.key < l_frame) {
                    l_previous = it.key ;
                    _clip = it.value;
                }
                else {
                    break;
                }
            }
            if(l_previous != -1) {
                max_left_resize = 0;
            }
            else {
                max_left_resize = _clip->duration + l_previous;
            }
        }
        else {
            max_left_resize = clip_data->start_frame + 1;
        }

        l_previous = -1;

        if(!a_left) {
            
            for(auto it : track) {
                if(it.key > l_frame) {
                    l_previous = it.key;
                    _clip = it.value;
                    break;
                }
            }
            max_right_resize = l_previous;
        }
        else {
            max_right_resize = clip_data->duration  + clip_data->start_frame   ;
        }

        if(a_left) {
            is_resizing_left = true;
        }
        else {
            is_resizing_right = true;
        }

    }

    void _on_commit_resize() {
        is_resizing_left = false;
        is_resizing_right = false;

        undo_redo->create_action("Resize clip on timeline");
        Point2 pos = get_position();
        Size2 size = get_size();
        undo_redo->add_do_method(callable_mp(this, &TimelineButton::_set_resize_data).bind(get_frame_nr(pos.x),get_frame_nr(size.x)));

        undo_redo->add_undo_method(callable_mp(this, &TimelineButton::_set_resize_data).bind(clip_data->start_frame,clip_data->duration));
        undo_redo->commit_action();

    }

    void _set_resize_data(int a_new_start,int a_new_duration) {
        if(clip_data->start_frame != a_new_start) {
            clip_data->begin += a_new_start - clip_data->start_frame;
        }

        Point2 pos = get_position();
        Size2 size = get_size();
		pos.x = a_new_start * timeline->timeline_scale;
        size.x = a_new_duration * timeline->timeline_scale;

        set_position(pos);
        set_size(size);

        
        int64_t l_track = get_track_id(pos.y);
        RBMap<int64_t, Ref<TimeLineClipData>>& track = timeline->tracks[l_track];

        track.erase(clip_data->start_frame);
        track[a_new_start] = clip_data;

        clip_data->start_frame = a_new_start;
        clip_data->duration = a_new_duration;
    }

    void _cut_clip(int a_playhead) {


        if(a_playhead <= clip_data->start_frame) {
            return;
        }
        else if(a_playhead >= clip_data->start_frame + clip_data->duration) {
            return;
        }
        Ref<TimeLineClipData> l_new_clip = clip_data->duplicate();

        int l_frame = a_playhead - clip_data->start_frame;
        l_new_clip->start_frame = a_playhead;
        l_new_clip->duration = ABS(clip_data->duration - l_frame);
        l_new_clip->begin = clip_data->begin + l_frame;

        clip_data->duration -= l_new_clip->duration;

        Size2 size = get_size();
        size.x = l_new_clip->duration * timeline->timeline_scale;
        set_size(size);

        Point2 pos = get_position();

        timeline->_add_clip(l_new_clip , get_track_id(pos.y));
    }

    void _uncut_clip(int a_playhead) {
        Point2 pos = get_position();
        int l_track = get_track_id(pos.y);

        Ref<TimeLineClipData> l_clip_data = timeline->get_clip_data(l_track,a_playhead);

        clip_data->duration += l_clip_data->duration;

        Size2 size = get_size();
        size.x = clip_data->duration * timeline->timeline_scale;
        set_size(size);

        timeline->delete_clip(l_track, a_playhead);
    }
    void update_rect() {
        Size2 size = get_size();
        size.x = clip_data->duration * timeline->timeline_scale;
        size.y = LINE_HEIGHT;
        set_size(size);

        Point2 pos = get_position();
        pos.x = clip_data->start_frame * timeline->timeline_scale;
        pos.y = track_id * (TRACK_HEIGHT + LINE_HEIGHT);
        set_position(pos);
        
    }
public:

	TimelineButton(UndoRedo* p_undo_redo,Ref<Timeline> p_timeline,Ref<TimeLineClipData> p_clip_data,int a_track_id) {
        timeline = p_timeline;
        undo_redo = p_undo_redo  ;
        clip_data = p_clip_data;
        track_id = a_track_id;

        duration = clip_data->duration;
        start_frame = clip_data->start_frame;

        max_left_resize = clip_data->start_frame;
        max_right_resize = max_right_resize;
        
		_add_risize_button(PRESET_LEFT_WIDE, true);
		_add_risize_button(PRESET_RIGHT_WIDE, false);

        connect("button_down",callable_mp(this, &TimelineButton::_on_button_down));
        connect("gui_input",callable_mp(this, &TimelineButton::_on_gui_input));

        set_process_input(true);
        set_process(true);
        set_mouse_filter(MOUSE_FILTER_PASS);
        set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);

        update_rect();
    }
protected:
    UndoRedo* undo_redo = nullptr;
    Ref<Timeline> timeline;
    Ref<TimeLineClipData> clip_data;

    bool is_resizing_left = false;
    bool is_resizing_right = false;
    bool is_dragging = false;

    int track_id = 0;
    int64_t duration = 0;
    int64_t start_frame = 0;
    int64_t max_left_resize = 0;
    int64_t max_right_resize = 0;
};

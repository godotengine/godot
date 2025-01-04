#pragma once
#include "timeline_button.h"
#include "scene/gui/box_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/animation.h"
#include "scene/resources/audio_stream_wav.h"
#include "scene/resources/packed_scene.h"

class TimelineFileManager{
public:
    enum Type {
        TYPE_NONE,
        TYPE_AUDIO,
        TYPE_SCENE,
        TYPE_ANIMATION,
    };
    struct FileInfo {
    
      float duration = 0; 
      Type type = TYPE_NONE;
    };
    static TimelineFileManager & get_instance() {
        static TimelineFileManager instance;
        return instance;
    }

    const FileInfo& get_file_info(const String& p_file) { 
        if(files.has(p_file) ) {
            return files[p_file];             
        }

        Ref<Resource> res = ResourceLoader::load(p_file);
        if(res.is_null()) {
            files[p_file] = FileInfo();
            return files[p_file];
        }
        Ref<Animation> anim = res;
        if(anim.is_valid()) {
            files[p_file] = FileInfo();
            files[p_file].duration = anim->get_length();
            files[p_file].type = TYPE_ANIMATION;
            return files[p_file];
        }
        Ref<AudioStream> audio = res;
        if(audio.is_valid()) {
            files[p_file] = FileInfo();
            files[p_file].duration = audio->get_length();
            files[p_file].type = TYPE_AUDIO;
            return files[p_file];
        }
        Ref<PackedScene> scene = res;
        if(scene.is_valid()) {
            files[p_file] = FileInfo();
            files[p_file].duration = 10;
            files[p_file].type = TYPE_SCENE;
            return files[p_file];
        }
        files[p_file] = FileInfo();
        files[p_file].duration = 0;
        files[p_file].type = TYPE_NONE;
        return files[p_file];
    }
    HashMap<String, FileInfo> files;


};

class TimelineClipPanel : public Control {
    GDCLASS(TimelineClipPanel, Control)
    static void _bind_methods() {}
public:
	static int64_t floori(double x) {
		return int64_t(Math::floor(x));
	}
    int64_t get_frame_nr(float a_pos_x ) const{
        
		return floori(a_pos_x / timeline->timeline_scale);
    }
    bool can_drop_data(const Point2 &a_pos, const Variant &p_data) const {
        Dictionary d = p_data;
        float duration = 0;
        if(d["type"] == "files") {
            Vector<String> p_paths = d["files"];
            if(p_paths.size() != 1) {
                return false;
            }
            if(TimelineFileManager::get_instance().get_file_info(p_paths[0]).type  == TimelineFileManager::Type::TYPE_NONE) {
                return false;
            };
            duration = TimelineFileManager::get_instance().get_file_info(p_paths[0]).duration;
        }
        else if(d["type"] == "timeline_move_clip") {
            // 拖动timeline节点
            duration = d["duration"];
        }
        int64_t duration_fs = TimeLineClipData::time_to_frame(duration) ;
        if(duration_fs < 0) {
            return false;
        }
        int64_t l_track = TimelineButton::get_track_id(a_pos.y);
        int64_t mouse_offset = d["mouse_offset"];
        int64_t l_frame = MAX(get_frame_nr(a_pos.x) - mouse_offset,0);
        int64_t l_end = l_frame + duration_fs;

        TypedArray<Vector2i> l_ignore = d["ignore"];
        int64_t l_lowest = get_lowest_frame(l_track,l_frame,l_ignore);
        int64_t l_highest = get_highest_frame(l_track,l_end,l_ignore);
        if(l_highest == -1 && l_lowest < l_frame) {
            show_priview(l_track,l_frame,duration_fs);
            return true;
        }

        if(l_frame > l_lowest && l_end < l_highest) {
            show_priview(l_track,l_frame,duration_fs);
            return true;
        }


        if(l_frame <= l_lowest) {
            int64_t l_difference = l_lowest - l_frame;
            if((l_frame + duration_fs < l_highest) || l_highest == -1) {
                show_priview(l_track, l_frame + l_difference, duration_fs);
                return true;
            }
        }
        else if(l_end >= l_highest) {
            int64_t l_difference = l_end - l_highest;
            if(l_frame  - l_difference > l_lowest) {
                show_priview(l_track, l_frame - l_difference, duration_fs);
                return true;
            }
        }
        hide_priview();

        return false;
    }

    void drop_data(const Point2 &p_point, const Variant &p_data) {

        int64_t l_start_frame = get_frame_nr(private_position.x);
        int64_t l_track_id = TimelineButton::get_track_id(private_position.y);

        Dictionary l_data = p_data;

        if(l_data["type"] == "files") {
            Vector<String> p_paths = l_data["files"];
            
            Ref<TimeLineClipData> l_clip_data = Ref<TimeLineClipData>(memnew(TimeLineClipData));
            l_clip_data->file_path = p_paths[0];
            l_clip_data->duration = TimelineFileManager::get_instance().get_file_info(p_paths[0]).duration;
            l_clip_data->type = TimelineFileManager::get_instance().get_file_info(p_paths[0]).type;
            l_clip_data->start_frame = l_start_frame;


            undo_redo->create_action(TTR("Add Clip"));
            undo_redo->add_do_method(callable_mp(this,&TimelineClipPanel::_add_new_clips).bind(l_clip_data,l_track_id));
            undo_redo->add_undo_method(callable_mp(this,&TimelineClipPanel::_remove_new_clips).bind(l_clip_data,l_track_id));
            undo_redo->commit_action();
        }
        else if(l_data["type"] == "timeline_move_clip") {
            TypedArray<int64_t> l_clip_ids = l_data["clip_ids"];
            if(l_clip_ids.size() == 0) {
                return;
            }
			int64_t l_clip_id = (int64_t)l_clip_ids[0];
			ObjectID id;
			id = l_clip_id;
            TimelineButton* l_clip_button = Object::cast_to<TimelineButton>(ObjectDB::get_instance(id));
            if(!l_clip_button) {
                return;
            }
            undo_redo->create_action(TTR("Move Clip"));
            undo_redo->add_do_method(callable_mp(this,&TimelineClipPanel::_move_clip).bind(l_clip_button,private_position));
            undo_redo->add_undo_method(callable_mp(this,&TimelineClipPanel::_move_clip).bind(l_clip_button,l_clip_button->get_position()));
            undo_redo->commit_action();
        }

    }
    mutable Point2 private_position;
	mutable Size2 private_size;
    void show_priview(int64_t a_track_id,int64_t a_frame,int64_t a_duration) const{
        private_position.y = a_track_id * (TRACK_HEIGHT + LINE_HEIGHT);
        private_position.x = timeline->timeline_scale * a_frame;
	    private_size.x = timeline->timeline_scale * a_duration;

    }
    void hide_priview() const{

    }
    int64_t get_lowest_frame(int64_t a_track_id,int64_t a_frame_nr,TypedArray<Vector2i> a_ignore)const {
        int64_t l_lowest = -1;
        Ref<TimeLineClipData> l_clip_data;
        const RBMap<int64_t, Ref<TimeLineClipData>>& track = timeline->tracks[a_track_id];
        for(auto& it : track) {
            if(it.key < a_frame_nr) {
                if(a_ignore.size() > 0) {
					Vector2i rect = a_ignore[0];
                    if(it.key == rect.y && a_track_id == rect.x) {
                        continue;   
                    }                    
                }
                l_lowest = it.key;
                l_clip_data = it.value;
            }
            else if(it.key > a_frame_nr) {
                break;
            }
        }

        if(l_lowest == -1) {
            return -1;
        }
        return l_clip_data->duration + l_lowest;
    }

    int64_t get_highest_frame(int64_t a_track_id,int64_t a_frame_nr,TypedArray<Vector2i> a_ignore)const {

        RBMap<int64_t, Ref<TimeLineClipData>>* track = &timeline->tracks[a_track_id];
        for(auto&it : *track) {
            if(it.key > a_frame_nr) {
                if(a_ignore.size() > 0) {
					Vector2i rect = a_ignore[0];
					if (it.key == rect.y && a_track_id == rect.x) {
                        continue;   
                    }
                }
                return it.key;
            }
            else if(it.key < a_frame_nr) {
                break;
            }
        }
        return -1;
    }
    void _move_clip(TimelineButton* node,const Vector2& a_new_pos) {

        int64_t l_old_track_id = TimelineButton::get_track_id(node->get_position().y);
        int64_t l_new_track_id = TimelineButton::get_track_id(a_new_pos.y);

        int64_t l_old_frame = get_frame_nr(node->get_position().x);
        int64_t l_new_frame = get_frame_nr(a_new_pos.x);
        
        timeline->delete_clip(l_old_track_id,l_old_frame);
        timeline->_add_clip(node->get_clip_data(),l_new_track_id);
    }
    void _add_new_clips(Ref<TimeLineClipData> l_clip_data,int a_track_id) {
        timeline->_add_clip(l_clip_data,a_track_id);
    }
    void _remove_new_clips(Ref<TimeLineClipData> l_clip_data,int a_track_id) {
        timeline->delete_clip(a_track_id,l_clip_data->start_frame);
    }
public:
    void reload_clips() {
        if(timeline.is_null()) {
            clear_clips();
            return;
        }
		int i = 0;
        for(auto&it : timeline->tracks){
            for(auto& it2 : it) {
                ObjectID id = it2.value->get_instance_id();
                if(!clip_buttons.has(id)) {
                    TimelineButton* button = memnew(TimelineButton(undo_redo,timeline,it2.value, i));
                    clip_buttons.insert(id,button);
                }
            }
			++i;
        }
    }
    void remove_clip(ObjectID id) {
        if(clip_buttons.has(id)) {
            remove_child(clip_buttons[id]);
            clip_buttons.erase(id);
        }
    }
    void clear_clips() {
        for(auto&it : clip_buttons) {
            remove_child(it.value);
            it.value->queue_free();
        }
        clip_buttons.clear();
    }
    void set_timeline(Ref<Timeline> a_timeline) {
        timeline = a_timeline;
        reload_clips();
    }
    TimelineClipPanel() {
        undo_redo = memnew(UndoRedo);
        reload_clips();
    }

    Ref<Timeline> timeline;
    UndoRedo* undo_redo = nullptr;
    HashMap<ObjectID,TimelineButton*> clip_buttons;
};

class TimelinePlayhead : public Panel {
    GDCLASS(TimelinePlayhead, Panel)
    static void _bind_methods() {}
public:
    void on_update_position() {
        if(!timeline.is_valid()) {
            return;
        }
        Point2 pos = get_position();
        pos.x = timeline->frame_nr * timeline->timeline_scale;
        set_position(pos);
    }
    void _process(float delta) {
        on_update_position();
    }
    TimelinePlayhead() {
        set_process(true);
    }
    Ref<Timeline> timeline;
};

class TimelinePanel : public ScrollContainer {
    GDCLASS(TimelinePanel, ScrollContainer)
    static void _bind_methods() {}
public:
    TimelinePanel() {
        set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);

        main = memnew(Control);
        main->set_layout_mode(Control::LAYOUT_MODE_CONTAINER);
        main->set_custom_minimum_size(Size2(1080, 0));
        main->set_h_size_flags(SIZE_EXPAND_FILL);
        main->set_v_size_flags(SIZE_EXPAND_FILL);
        add_child(main);

        {

            clips = memnew(TimelineClipPanel);
            clips->set_layout_mode(Control::LAYOUT_MODE_ANCHORS);
            clips->set_anchors_preset(Control::PRESET_FULL_RECT);
            clips->set_anchor(SIDE_RIGHT,1);
            clips->set_anchor(SIDE_BOTTOM,1);
            clips->set_h_grow_direction(Control::GROW_DIRECTION_BOTH);
            clips->set_v_grow_direction(Control::GROW_DIRECTION_BOTH);
            clips->set_mouse_filter(MOUSE_FILTER_PASS);
            main->add_child(clips);

            clip_preview = memnew(PanelContainer);
            clip_preview->set_visible(false);
            clip_preview->set_layout_mode(Control::LAYOUT_MODE_POSITION);
            clip_preview->set_offset(SIDE_RIGHT,229);
            clip_preview->set_offset(SIDE_BOTTOM,35);
            clip_preview->set_mouse_filter(MOUSE_FILTER_PASS);
            main->add_child(clip_preview);

            playhead = memnew(TimelinePlayhead);
            playhead->set_custom_minimum_size(Size2(2, 0));
            playhead->set_layout_mode(Control::LAYOUT_MODE_ANCHORS);
            playhead->set_anchors_preset(Control::PRESET_LEFT_WIDE);
            playhead->set_anchor(SIDE_BOTTOM,1);
            playhead->set_offset(SIDE_RIGHT,2);
            playhead->set_v_grow_direction(Control::GROW_DIRECTION_BOTH);
            playhead->set_mouse_filter(MOUSE_FILTER_PASS);
            Ref<StyleBoxFlat> style = memnew(StyleBoxFlat);
            style->set_bg_color(Color(0.683393, 0, 1, 1));
            playhead->set("theme_override_styles/panel", style);
            main->add_child(playhead);
            
            checker = memnew(Panel);
            checker->set_visible(false);
            checker->set_layout_mode(Control::LAYOUT_MODE_POSITION);
            checker->set_offset(SIDE_LEFT, 300);
            checker->set_offset(SIDE_RIGHT, 309);
            checker->set_offset(SIDE_BOTTOM, 216);
            main->add_child(checker);

            planel2 = memnew(Panel);
            planel2->set_visible(false);
            planel2->set_layout_mode(Control::LAYOUT_MODE_POSITION);
            planel2->set_offset(SIDE_LEFT, 780);
            planel2->set_offset(SIDE_RIGHT, 789);
            planel2->set_offset(SIDE_BOTTOM, 216);
            main->add_child(planel2);


        }

    }
private:

    Control* main = nullptr;
    TimelineClipPanel* clips = nullptr;

    PanelContainer* clip_preview = nullptr;
    TimelinePlayhead* playhead = nullptr;
    Panel* checker = nullptr;
    Panel* planel2 = nullptr;
};

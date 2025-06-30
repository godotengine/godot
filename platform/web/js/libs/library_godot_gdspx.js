const GodotGdspx = {
	$GodotGdspx__deps: ['$GodotConfig', '$GodotRuntime'],
	$GodotGdspx: {
	},
	
	// godot gdspx extensions
	godot_js_spx_on_engine_start__proxy: 'sync',
	godot_js_spx_on_engine_start__sig: 'v',
	godot_js_spx_on_engine_start : function () {
		window.gdspx_on_engine_start();
	},

	godot_js_spx_on_engine_update__proxy: 'sync',
	godot_js_spx_on_engine_update__sig: 'vf', 
	godot_js_spx_on_engine_update : function (delta) {
		window.gdspx_on_engine_update(delta);
	},

	godot_js_spx_on_engine_fixed_update__proxy: 'sync',
	godot_js_spx_on_engine_fixed_update__sig: 'vf', 
	godot_js_spx_on_engine_fixed_update : function (delta) {
		window.gdspx_on_engine_fixed_update(delta);
	},

	godot_js_spx_on_engine_destroy__proxy: 'sync',
	godot_js_spx_on_engine_destroy__sig: 'v',
	godot_js_spx_on_engine_destroy : function () {
		window.gdspx_on_engine_destroy();
	},


	godot_js_spx_on_engine_update__proxy: 'sync',
	godot_js_spx_on_engine_update__sig: 'vf', 
	godot_js_spx_on_engine_update : function (delta) {
		window.gdspx_on_engine_update(delta);
	},

	godot_js_spx_on_engine_fixed_update__proxy: 'sync',
	godot_js_spx_on_engine_fixed_update__sig: 'vf', 
	godot_js_spx_on_engine_fixed_update : function (delta) {
		window.gdspx_on_engine_fixed_update(delta);
	},

	godot_js_spx_on_engine_destroy__proxy: 'sync',
	godot_js_spx_on_engine_destroy__sig: 'v',
	godot_js_spx_on_engine_destroy : function () {
		window.gdspx_on_engine_destroy();
	},

	godot_js_spx_on_scene_sprite_instantiated__proxy: 'sync',
	godot_js_spx_on_scene_sprite_instantiated__sig: 'vii',
	godot_js_spx_on_scene_sprite_instantiated: function (obj, type_name) {
		window.gdspx_on_scene_sprite_instantiated(GodotRuntime.ToJsObj(obj), GodotRuntime.parseString(type_name));
	},
	
	godot_js_spx_on_runtime_panic__proxy: 'sync',
	godot_js_spx_on_runtime_panic__sig: 'vi',
	godot_js_spx_on_runtime_panic: function (msg) {
		window.gdspx_on_runtime_panic(GodotRuntime.parseString(msg));
	},
	

	godot_js_spx_on_sprite_ready__proxy: 'sync',
	godot_js_spx_on_sprite_ready__sig: 'vi',
	godot_js_spx_on_sprite_ready: function (obj) {
		window.gdspx_on_sprite_ready(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_updated__proxy: 'sync',
	godot_js_spx_on_sprite_updated__sig: 'vf',
	godot_js_spx_on_sprite_updated: function (delta) {
		window.gdspx_on_sprite_updated(delta);
	},

	godot_js_spx_on_sprite_fixed_updated__proxy: 'sync',
	godot_js_spx_on_sprite_fixed_updated__sig: 'vf',
	godot_js_spx_on_sprite_fixed_updated: function (delta) {
		window.gdspx_on_sprite_fixed_updated(delta);
	},

	godot_js_spx_on_sprite_destroyed__proxy: 'sync',
	godot_js_spx_on_sprite_destroyed__sig: 'vi',
	godot_js_spx_on_sprite_destroyed: function (obj) {
		window.gdspx_on_sprite_destroyed(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_frames_set_changed__proxy: 'sync',
	godot_js_spx_on_sprite_frames_set_changed__sig: 'vi',
	godot_js_spx_on_sprite_frames_set_changed: function (obj) {
		window.gdspx_on_sprite_frames_set_changed(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_animation_changed__proxy: 'sync',
	godot_js_spx_on_sprite_animation_changed__sig: 'vi',
	godot_js_spx_on_sprite_animation_changed: function (obj) {
		window.gdspx_on_sprite_animation_changed(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_frame_changed__proxy: 'sync',
	godot_js_spx_on_sprite_frame_changed__sig: 'vi',
	godot_js_spx_on_sprite_frame_changed: function (obj) {
		window.gdspx_on_sprite_frame_changed(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_animation_looped__proxy: 'sync',
	godot_js_spx_on_sprite_animation_looped__sig: 'vi',
	godot_js_spx_on_sprite_animation_looped: function (obj) {
		window.gdspx_on_sprite_animation_looped(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_animation_finished__proxy: 'sync',
	godot_js_spx_on_sprite_animation_finished__sig: 'vi',
	godot_js_spx_on_sprite_animation_finished: function (obj) {
		window.gdspx_on_sprite_animation_finished(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_vfx_finished__proxy: 'sync',
	godot_js_spx_on_sprite_vfx_finished__sig: 'vi',
	godot_js_spx_on_sprite_vfx_finished: function (obj) {
		window.gdspx_on_sprite_vfx_finished(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_screen_exited__proxy: 'sync',
	godot_js_spx_on_sprite_screen_exited__sig: 'vi',
	godot_js_spx_on_sprite_screen_exited: function (obj) {
		window.gdspx_on_sprite_screen_exited(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_sprite_screen_entered__proxy: 'sync',
	godot_js_spx_on_sprite_screen_entered__sig: 'vi',
	godot_js_spx_on_sprite_screen_entered: function (obj) {
		window.gdspx_on_sprite_screen_entered(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_mouse_pressed__proxy: 'sync',
	godot_js_spx_on_mouse_pressed__sig: 'vi',
	godot_js_spx_on_mouse_pressed: function (keyid) {
		window.gdspx_on_mouse_pressed(GodotRuntime.ToJsInt(keyid));
	},

	godot_js_spx_on_mouse_released__proxy: 'sync',
	godot_js_spx_on_mouse_released__sig: 'vi',
	godot_js_spx_on_mouse_released: function (keyid) {
		window.gdspx_on_mouse_released(GodotRuntime.ToJsInt(keyid));
	},

	godot_js_spx_on_key_pressed__proxy: 'sync',
	godot_js_spx_on_key_pressed__sig: 'vi',
	godot_js_spx_on_key_pressed: function (keyid) {
		window.gdspx_on_key_pressed(GodotRuntime.ToJsInt(keyid));
	},

	godot_js_spx_on_key_released__proxy: 'sync',
	godot_js_spx_on_key_released__sig: 'vi',
	godot_js_spx_on_key_released: function (keyid) {
		window.gdspx_on_key_released(GodotRuntime.ToJsInt(keyid));
	},

	godot_js_spx_on_action_pressed__proxy: 'sync',
	godot_js_spx_on_action_pressed__sig: 'vi',
	godot_js_spx_on_action_pressed: function (action_name) {
		window.gdspx_on_action_pressed(GodotRuntime.parseString(action_name));
	},

	godot_js_spx_on_action_just_pressed__proxy: 'sync',
	godot_js_spx_on_action_just_pressed__sig: 'vi',
	godot_js_spx_on_action_just_pressed: function (action_name) {
		window.gdspx_on_action_just_pressed(GodotRuntime.parseString(action_name));
	},

	godot_js_spx_on_action_just_released__proxy: 'sync',
	godot_js_spx_on_action_just_released__sig: 'vi',
	godot_js_spx_on_action_just_released: function (action_name) {
		window.gdspx_on_action_just_released(GodotRuntime.parseString(action_name));
	},

	godot_js_spx_on_axis_changed__proxy: 'sync',
	godot_js_spx_on_axis_changed__sig: 'vif',
	godot_js_spx_on_axis_changed: function (action_name, value) {
		window.gdspx_on_axis_changed(GodotRuntime.parseString(action_name), value);
	},

	godot_js_spx_on_collision_enter__proxy: 'sync',
	godot_js_spx_on_collision_enter__sig: 'vii',
	godot_js_spx_on_collision_enter: function (self_id, other_id) {
		window.gdspx_on_collision_enter(GodotRuntime.ToJsInt(self_id), GodotRuntime.ToJsInt(other_id));
	},

	godot_js_spx_on_collision_stay__proxy: 'sync',
	godot_js_spx_on_collision_stay__sig: 'vii',
	godot_js_spx_on_collision_stay: function (self_id, other_id) {
		window.gdspx_on_collision_stay(GodotRuntime.ToJsInt(self_id), GodotRuntime.ToJsInt(other_id));
	},

	godot_js_spx_on_collision_exit__proxy: 'sync',
	godot_js_spx_on_collision_exit__sig: 'vii',
	godot_js_spx_on_collision_exit: function (self_id, other_id) {
		window.gdspx_on_collision_exit(GodotRuntime.ToJsInt(self_id), GodotRuntime.ToJsInt(other_id));
	},

	godot_js_spx_on_trigger_enter__proxy: 'sync',
	godot_js_spx_on_trigger_enter__sig: 'vii',
	godot_js_spx_on_trigger_enter: function (self_id, other_id) {
		window.gdspx_on_trigger_enter(GodotRuntime.ToJsInt(self_id), GodotRuntime.ToJsInt(other_id));
	},

	godot_js_spx_on_trigger_stay__proxy: 'sync',
	godot_js_spx_on_trigger_stay__sig: 'vii',
	godot_js_spx_on_trigger_stay: function (self_id, other_id) {
		window.gdspx_on_trigger_stay(GodotRuntime.ToJsInt(self_id), GodotRuntime.ToJsInt(other_id));
	},

	godot_js_spx_on_trigger_exit__proxy: 'sync',
	godot_js_spx_on_trigger_exit__sig: 'vii',
	godot_js_spx_on_trigger_exit: function (self_id, other_id) {
		window.gdspx_on_trigger_exit(GodotRuntime.ToJsInt(self_id), GodotRuntime.ToJsInt(other_id));
	},

	godot_js_spx_on_ui_ready__proxy: 'sync',
	godot_js_spx_on_ui_ready__sig: 'vi',
	godot_js_spx_on_ui_ready: function (obj) {
		window.gdspx_on_ui_ready(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_ui_updated__proxy: 'sync',
	godot_js_spx_on_ui_updated__sig: 'vi',
	godot_js_spx_on_ui_updated: function (obj) {
		window.gdspx_on_ui_updated(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_ui_destroyed__proxy: 'sync',
	godot_js_spx_on_ui_destroyed__sig: 'vi',
	godot_js_spx_on_ui_destroyed: function (obj) {
		window.gdspx_on_ui_destroyed(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_ui_pressed__proxy: 'sync',
	godot_js_spx_on_ui_pressed__sig: 'vi',
	godot_js_spx_on_ui_pressed: function (obj) {
		window.gdspx_on_ui_pressed(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_ui_released__proxy: 'sync',
	godot_js_spx_on_ui_released__sig: 'vi',
	godot_js_spx_on_ui_released: function (obj) {
		window.gdspx_on_ui_released(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_ui_hovered__proxy: 'sync',
	godot_js_spx_on_ui_hovered__sig: 'vi',
	godot_js_spx_on_ui_hovered: function (obj) {
		window.gdspx_on_ui_hovered(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_ui_clicked__proxy: 'sync',
	godot_js_spx_on_ui_clicked__sig: 'vi',
	godot_js_spx_on_ui_clicked: function (obj) {
		window.gdspx_on_ui_clicked(GodotRuntime.ToJsObj(obj));
	},

	godot_js_spx_on_ui_toggle__proxy: 'sync',
	godot_js_spx_on_ui_toggle__sig: 'vii',
	godot_js_spx_on_ui_toggle: function (obj, is_on) {
		window.gdspx_on_ui_toggle(GodotRuntime.ToJsObj(obj), is_on);
	},

	godot_js_spx_on_ui_text_changed__proxy: 'sync',
	godot_js_spx_on_ui_text_changed__sig: 'vii',
	godot_js_spx_on_ui_text_changed: function (obj, text) {
		window.gdspx_on_ui_text_changed(GodotRuntime.ToJsObj(obj), GodotRuntime.parseString(text));
	},
};

autoAddDeps(GodotGdspx, '$GodotGdspx');
mergeInto(LibraryManager.library, GodotGdspx);

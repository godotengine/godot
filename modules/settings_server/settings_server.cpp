#include "settings_server.h"

#define GP_CHANGED "graphics_preset_changed"
#define DP_CHANGED "display_preset_changed"
#define RS_CHANGED "resolution_preset_changed"
#define GS_CHANGED "graphics_setting_changed"

#define SSAA_LIMIT_MAX 2.0
#define SSAA_LIMIT_MIN 0.7

SettingsServer *SettingsServer::singleton = nullptr;

VARIANT_ENUM_CAST(SettingsServer::GraphicsPreset);
VARIANT_ENUM_CAST(SettingsServer::GraphicsSetting);
VARIANT_ENUM_CAST(SettingsServer::DisplaySettings);
VARIANT_ENUM_CAST(SettingsServer::ResolutionSettings);

SettingsServer::SettingsServer(){
	ERR_FAIL_COND(singleton);
	singleton = this;
	rendering_device = VisualServer::get_singleton()->get_video_adapter_name();

	// Setup resolutions
	acceptable_resolution[RES_CUSTOM]		= Vector2(0.0, 0.0);
	acceptable_resolution[RES_DEFAULT]		= Vector2(0.0, 0.0);
	acceptable_resolution[RES_FULLSCREEN]	= Vector2(0.0, 0.0);
	acceptable_resolution[RES_1024_600]		= Vector2(1024, 600);
	acceptable_resolution[RES_1280_720]		= Vector2(1280, 720);
	acceptable_resolution[RES_1366_768]		= Vector2(1366, 768);
	acceptable_resolution[RES_1400_1050]	= Vector2(1400, 1050);
	acceptable_resolution[RES_1300_900]		= Vector2(1300, 900);
	acceptable_resolution[RES_1600_900]		= Vector2(1600, 900);
	acceptable_resolution[RES_1680_1050]	= Vector2(1680, 1050);
	acceptable_resolution[RES_1920_1080]	= Vector2(1920, 1080);
	acceptable_resolution[RES_2048_1080]	= Vector2(2048, 1080);
	acceptable_resolution[RES_2560_1440]	= Vector2(2560, 1440);
	acceptable_resolution[RES_3840_2160]	= Vector2(3840, 2160);
	acceptable_resolution[RES_4096_2160]	= Vector2(4096, 2160);
	acceptable_resolution[RES_69420_69420]  = Vector2(69420, 69420);
	acceptable_resolution[RES_HIGHEST]		= acceptable_resolution[RES_4096_2160];

	reset_screen_info();
	last_window_pos = OS::get_singleton()->get_window_position();

	// if (!Engine::get_singleton()->is_editor_hint())
	// 	OS::get_singleton()->set_window_resizable(false);

	// auto main_loop = OS::get_singleton()->get_main_loop();
	// ERR_FAIL_COND(!main_loop->is_class("SceneTree"));
	// set_vp_internal(((SceneTree*)main_loop)->get_root());
}
SettingsServer::~SettingsServer(){}

void SettingsServer::_bind_methods(){
	ClassDB::bind_method(D_METHOD("get_rendering_device"), &SettingsServer::get_rendering_device);

	// ClassDB::bind_method(D_METHOD("set_main_viewport", "viewport"), &SettingsServer::set_main_viewport);
	ClassDB::bind_method(D_METHOD("get_main_viewport"), &SettingsServer::get_main_viewport);
	
	ClassDB::bind_method(D_METHOD("load_graphics_preset", "preset"), &SettingsServer::load_graphics_preset);
	ClassDB::bind_method(D_METHOD("get_graphics_preset"), &SettingsServer::get_graphics_preset);
	
	ClassDB::bind_method(D_METHOD("set_graphics_setting", "setting", "value"), &SettingsServer::set_graphics_setting);
	ClassDB::bind_method(D_METHOD("get_graphics_setting", "setting"), &SettingsServer::get_graphics_setting);

	ClassDB::bind_method(D_METHOD("get_default_window_size"), &SettingsServer::get_default_window_size);
	ClassDB::bind_method(D_METHOD("rs_to_vec2", "rs"), &SettingsServer::rs_to_vec2);

	ClassDB::bind_method(D_METHOD("set_window_size", "size"), &SettingsServer::set_window_size);
	ClassDB::bind_method(D_METHOD("get_window_size"), &SettingsServer::get_window_size);

	ClassDB::bind_method(D_METHOD("set_display_mode", "mode"), &SettingsServer::set_display_mode);
	ClassDB::bind_method(D_METHOD("get_display_mode"), &SettingsServer::get_display_mode);

	ClassDB::bind_method(D_METHOD("set_current_display", "display_no"), &SettingsServer::set_current_display);
	ClassDB::bind_method(D_METHOD("get_current_display"), &SettingsServer::get_current_display);

	auto gp_pi = PropertyInfo(Variant::INT, "current_graphics_preset", PROPERTY_HINT_ENUM, "GRAPHICS_CUSTOM, GRAPHICS_LOWEST,GRAPHICS_LOW,GRAPHICS_MEDIUM,GRAPHICS_HIGH, GRAPHICS_HIGHEST");
	auto dp_pi = PropertyInfo(Variant::INT, "current_display_preset", PROPERTY_HINT_ENUM, "WINDOWED,FULLSCREEN,BORDERLESS_FULLSCREEN");
	auto rs_pi = PropertyInfo(Variant::INT, "current_resolution_preset", PROPERTY_HINT_ENUM,
		"RES_CUSTOM,RES_DEFAULT,RES_FULLSCREEN,RES_1024_600,RES_1280_720,RES_1366_768,RES_1400_1050,RES_1300_900,RES_1600_900,RES_1680_1050,RES_1920_1080,RES_2048_1080,RES_2560_1440,RES_3840_2160,RES_4096_2160,RES_HIGHEST");
	auto gs_pi = PropertyInfo(Variant::INT, "what_setting");

	ADD_SIGNAL(MethodInfo(GP_CHANGED, gp_pi));
	ADD_SIGNAL(MethodInfo(DP_CHANGED, dp_pi));
	ADD_SIGNAL(MethodInfo(RS_CHANGED, rs_pi));
	ADD_SIGNAL(MethodInfo(GS_CHANGED, gs_pi));

	BIND_ENUM_CONSTANT(GRAPHICS_CUSTOM);
	BIND_ENUM_CONSTANT(GRAPHICS_LOWEST);
	BIND_ENUM_CONSTANT(GRAPHICS_LOW);
	BIND_ENUM_CONSTANT(GRAPHICS_MEDIUM);
	BIND_ENUM_CONSTANT(GRAPHICS_HIGH);
	BIND_ENUM_CONSTANT(GRAPHICS_HIGHEST);

	BIND_ENUM_CONSTANT(TARGET_FPS);
	BIND_ENUM_CONSTANT(TARGET_IPS);
	BIND_ENUM_CONSTANT(USE_HDR);
	BIND_ENUM_CONSTANT(USE_32_BPC_DEPTH);
	BIND_ENUM_CONSTANT(FXAA_ENABLED);
	BIND_ENUM_CONSTANT(SHARPEN_INTENSITY);
	BIND_ENUM_CONSTANT(MSAA_LEVEL);
	BIND_ENUM_CONSTANT(SSAA_LEVEL);
	BIND_ENUM_CONSTANT(DEBANDING);
	BIND_ENUM_CONSTANT(GS_END);

	BIND_ENUM_CONSTANT(WINDOWED);
	BIND_ENUM_CONSTANT(FULLSCREEN);
	BIND_ENUM_CONSTANT(BORDERLESS_FULLSCREEN);

	BIND_ENUM_CONSTANT(RES_CUSTOM);
	BIND_ENUM_CONSTANT(RES_DEFAULT);
	BIND_ENUM_CONSTANT(RES_FULLSCREEN);
	BIND_ENUM_CONSTANT(RES_1024_600);
	BIND_ENUM_CONSTANT(RES_1280_720);
	BIND_ENUM_CONSTANT(RES_1366_768);
	BIND_ENUM_CONSTANT(RES_1400_1050);
	BIND_ENUM_CONSTANT(RES_1300_900);
	BIND_ENUM_CONSTANT(RES_1600_900);
	BIND_ENUM_CONSTANT(RES_1680_1050);
	BIND_ENUM_CONSTANT(RES_1920_1080);
	BIND_ENUM_CONSTANT(RES_2048_1080);
	BIND_ENUM_CONSTANT(RES_2560_1440);
	BIND_ENUM_CONSTANT(RES_3840_2160);
	BIND_ENUM_CONSTANT(RES_4096_2160);
	BIND_ENUM_CONSTANT(RES_69420_69420);
	BIND_ENUM_CONSTANT(RES_HIGHEST);

	// ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "main_viewport", PROPERTY_HINT_NONE), "set_main_viewport", "get_main_viewport");
	ADD_PROPERTY(gp_pi, "load_graphics_preset", "get_graphics_preset");
	ADD_PROPERTY(dp_pi, "set_display_mode", "get_display_mode");
	ADD_PROPERTY(rs_pi, "set_window_size", "get_window_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_screen"), "set_current_display", "get_current_display");
}

void SettingsServer::set_vp_internal(Viewport* vp){
	main_viewport = vp;
	// current_resolution = main_viewport->get_size();
	current_resolution = default_window_size;
	set_res_internal();
	load_gpp_internal();
}

void SettingsServer::set_main_viewport(Node* vp){
	if (vp && vp->is_class("Viewport")) {
		set_vp_internal((Viewport*)vp);
	}
}

void SettingsServer::set_ssaa_internal(const float& val){
	if (val > SSAA_LIMIT_MAX || val < SSAA_LIMIT_MIN) return;
	ssaa = val;
	set_res_internal();
	// main_viewport->set_size(current_resolution * ssaa);
	// OS::get_singleton()->set_window_size(current_resolution);
}

void SettingsServer::emit_gs_changed(const uint16_t& setting){
	emit_signal(GS_CHANGED, setting);
}

void SettingsServer::reset_screen_info(){
	default_window_size.x = ProjectSettings::get_singleton()->get_setting("display/window/size/width");
	default_window_size.y = ProjectSettings::get_singleton()->get_setting("display/window/size/height");

	acceptable_resolution[RES_DEFAULT]		= default_window_size;
	acceptable_resolution[RES_FULLSCREEN]	= OS::get_singleton()->get_screen_size(current_screen);

	last_window_pos = Vector2();
}

void SettingsServer::set_res_internal(){
	switch (curr_ds){
		case DisplaySettings::WINDOWED:
			OS::get_singleton()->set_window_size(default_window_size);
			OS::get_singleton()->set_window_position(last_window_pos);
			OS::get_singleton()->set_window_fullscreen(false);
			OS::get_singleton()->set_borderless_window(false);
			OS::get_singleton()->set_window_maximized(false);
			break;
		case DisplaySettings::FULLSCREEN:
			last_window_pos = OS::get_singleton()->get_window_position();
			OS::get_singleton()->set_window_fullscreen(true);
			// OS::get_singleton()->set_window_size(acceptable_resolution[ResolutionSettings::RES_FULLSCREEN]);
			// OS::get_singleton()->set_borderless_window(false);
			// OS::get_singleton()->set_window_maximized(true);
			break;
		case DisplaySettings::BORDERLESS_FULLSCREEN:
			last_window_pos = OS::get_singleton()->get_window_position();
			OS::get_singleton()->set_window_fullscreen(false);
			// OS::get_singleton()->set_window_size(acceptable_resolution[ResolutionSettings::RES_FULLSCREEN]);
			OS::get_singleton()->set_borderless_window(true);
			OS::get_singleton()->set_window_maximized(true);

			break;
	}
	main_viewport->set_size(current_resolution * ssaa);
}

Vector2 SettingsServer::rs_to_vec2(ResolutionSettings rs){
	if (rs < 0 || rs > ResolutionSettings::RES_HIGHEST)
		return default_window_size;
	return acceptable_resolution[rs];
}

void SettingsServer::set_window_size(ResolutionSettings res){
	if (res < 0 || res > ResolutionSettings::RES_HIGHEST || res == curr_rs) return;
	curr_rs = res;
	curr_ds = DisplaySettings::WINDOWED;
	curr_rs = res;
	current_resolution = acceptable_resolution[res];
	set_res_internal();
	emit_signal(DP_CHANGED, curr_ds);
	emit_signal(RS_CHANGED, curr_rs);
}

void SettingsServer::set_display_mode(DisplaySettings mode) {
	if (mode < 0 || mode > DisplaySettings::BORDERLESS_FULLSCREEN || mode == curr_ds) return;
	curr_ds = mode;
	if (mode == DisplaySettings::WINDOWED){
		curr_rs = ResolutionSettings::RES_DEFAULT;
		last_window_pos = OS::get_singleton()->get_window_position();
	}
	else {
		curr_rs = ResolutionSettings::RES_FULLSCREEN;

	}
	current_resolution = acceptable_resolution[curr_rs];
	set_res_internal();
	emit_signal(DP_CHANGED, curr_ds);
	emit_signal(RS_CHANGED, curr_rs);
}

void SettingsServer::set_current_display(const uint16_t& display_no){
	auto max_dis = OS::get_singleton()->get_screen_count();
	if (display_no >= max_dis || display_no < 0 || display_no == current_screen) return;
	current_screen = display_no;
	reset_screen_info();
	set_res_internal();
	emit_signal(RS_CHANGED, curr_rs);
}

void SettingsServer::load_gpp_internal() const{
	switch (current_gp){
		case GraphicsPreset::GRAPHICS_LOWEST:
			Engine::get_singleton()->set_target_fps(60);
			// Engine::get_singleton()->set_iterations_per_second(30);
			main_viewport->set_hdr(false);
			main_viewport->set_use_debanding(false);
			main_viewport->set_use_32_bpc_depth(false);
			main_viewport->set_use_fxaa(false);
			main_viewport->set_sharpen_intensity(false);
			main_viewport->set_msaa(Viewport::MSAA_DISABLED);
			break;
		case GraphicsPreset::GRAPHICS_LOW:
			Engine::get_singleton()->set_target_fps(60);
			// Engine::get_singleton()->set_iterations_per_second(30);
			main_viewport->set_hdr(false);
			main_viewport->set_use_debanding(false);
			main_viewport->set_use_32_bpc_depth(false);
			main_viewport->set_use_fxaa(true);
			main_viewport->set_sharpen_intensity(0.5);
			main_viewport->set_msaa(Viewport::MSAA_DISABLED);
			break;
		case GraphicsPreset::GRAPHICS_MEDIUM:
			Engine::get_singleton()->set_target_fps(60);
			// Engine::get_singleton()->set_iterations_per_second(60);
			main_viewport->set_hdr(true);
			main_viewport->set_use_debanding(true);
			main_viewport->set_use_32_bpc_depth(true);
			main_viewport->set_use_fxaa(false);
			main_viewport->set_sharpen_intensity(0.0);
			main_viewport->set_msaa(Viewport::MSAA_4X);
			break;
		case GraphicsPreset::GRAPHICS_HIGH:
			Engine::get_singleton()->set_target_fps(0);
			// Engine::get_singleton()->set_iterations_per_second(120);
			main_viewport->set_hdr(true);
			main_viewport->set_use_debanding(true);
			main_viewport->set_use_32_bpc_depth(false);
			main_viewport->set_use_fxaa(false);
			main_viewport->set_sharpen_intensity(0.0);
			main_viewport->set_msaa(Viewport::MSAA_8X);
			break;
		case GraphicsPreset::GRAPHICS_HIGHEST:
			Engine::get_singleton()->set_target_fps(0);
			// Engine::get_singleton()->set_iterations_per_second(120);
			main_viewport->set_hdr(true);
			main_viewport->set_use_debanding(true);
			main_viewport->set_use_32_bpc_depth(true);
			main_viewport->set_use_fxaa(false);
			main_viewport->set_sharpen_intensity(0.0);
			main_viewport->set_msaa(Viewport::MSAA_16X);
			break;
		default: return;
	}
}

void SettingsServer::load_graphics_preset(GraphicsPreset preset){
	if (preset == GraphicsPreset::GRAPHICS_CUSTOM || preset == current_gp) return;
	current_gp = preset;
	if (main_viewport)
		load_gpp_internal();
	emit_signal(GP_CHANGED, current_gp);
}

bool SettingsServer::set_graphics_setting(GraphicsSetting setting, const Variant& value){
	if (!main_viewport) return false;
	auto msaa = 0;
	switch (setting){
		case GraphicsSetting::TARGET_FPS:
			if (!(value.get_type() == Variant::INT)) return false;
			Engine::get_singleton()->set_target_fps((int)value);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::TARGET_IPS:
			if (!(value.get_type() == Variant::INT)) return false;
			if ((int)value < 30 || (int)value > 240) return false;
			Engine::get_singleton()->set_iterations_per_second((int)value);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::USE_HDR:
			if (!(value.get_type() == Variant::BOOL)) return false;
			main_viewport->set_hdr((bool)value);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::USE_32_BPC_DEPTH:
			if (!(value.get_type() == Variant::BOOL)) return false;
			main_viewport->set_use_32_bpc_depth((bool)value);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::FXAA_ENABLED:
			if (!(value.get_type() == Variant::BOOL)) return false;
			main_viewport->set_use_fxaa((bool)value);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::SHARPEN_INTENSITY:
			if (!(value.get_type() == Variant::REAL)) return false;
			main_viewport->set_sharpen_intensity((float)value);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::MSAA_LEVEL:
			if (!(value.get_type() == Variant::INT)) return false;
			msaa = (int)value;
			if (msaa < 0 || msaa > Viewport::MSAA_16X) return false;
			main_viewport->set_msaa((Viewport::MSAA)msaa);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::SSAA_LEVEL:
			if (!(value.get_type() == Variant::REAL)) return false;
			set_ssaa_internal((float)value);
			emit_gs_changed(setting);
			break;
		case GraphicsSetting::DEBANDING:
			if (!(value.get_type() == Variant::BOOL)) return false;
			main_viewport->set_use_debanding((bool)value);
			emit_gs_changed(setting);
			break;
		default: return false;
	}
	current_gp = GraphicsPreset::GRAPHICS_CUSTOM;
	emit_signal(GP_CHANGED, current_gp);
	return true;
}

Variant SettingsServer::get_graphics_setting(GraphicsSetting setting) const{
	if (!main_viewport) return Variant();
	switch (setting){
		case GraphicsSetting::TARGET_FPS:
			return Variant(Engine::get_singleton()->get_target_fps());
		case GraphicsSetting::TARGET_IPS:
			return Variant(Engine::get_singleton()->get_iterations_per_second());
		case GraphicsSetting::USE_HDR:
			return main_viewport->get_hdr();
		case GraphicsSetting::USE_32_BPC_DEPTH:
			return main_viewport->is_using_32_bpc_depth();
		case GraphicsSetting::FXAA_ENABLED:
			return main_viewport->get_use_fxaa();
		case GraphicsSetting::SHARPEN_INTENSITY:
			return main_viewport->get_sharpen_intensity();
		case GraphicsSetting::MSAA_LEVEL:
			return main_viewport->get_msaa();
		case GraphicsSetting::SSAA_LEVEL:
			return ssaa;
		case GraphicsSetting::DEBANDING:
			return main_viewport->get_use_debanding();
		default: return Variant();
	}
}

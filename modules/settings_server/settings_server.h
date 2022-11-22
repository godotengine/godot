#ifndef SETTINGS_SERVER_H
#define SETTINGS_SERVER_H

#include "core/project_settings.h"
#include "scene/main/viewport.h"
#include "core/engine.h"
#include "core/os/os.h"
#include "core/math/vector2.h"
#include "core/ordered_hash_map.h"

// enum SettingsServer::ResolutionSettings;
typedef Map<int, Vector2> ResolutionList;


class SettingsServer : public Object {
	GDCLASS(SettingsServer, Object);
public:
	enum GraphicsPreset {
		GRAPHICS_CUSTOM,
		GRAPHICS_LOWEST,
		GRAPHICS_LOW,
		GRAPHICS_MEDIUM,
		GRAPHICS_HIGH,
		GRAPHICS_HIGHEST
	};
	enum GraphicsSetting {
		TARGET_FPS,
		TARGET_IPS,
		USE_HDR,
		USE_32_BPC_DEPTH,
		FXAA_ENABLED,
		SHARPEN_INTENSITY,
		MSAA_LEVEL,
		SSAA_LEVEL,
		DEBANDING,
		GS_END,
	};
	enum DisplaySettings {
		WINDOWED,
		FULLSCREEN,
		BORDERLESS_FULLSCREEN,
	};
	enum ResolutionSettings {
		RES_CUSTOM,
		RES_DEFAULT,
		RES_FULLSCREEN,
		RES_1024_600,
		RES_1280_720,
		RES_1366_768,
		RES_1400_1050,
		RES_1300_900,
		RES_1600_900,
		RES_1680_1050,
		RES_1920_1080,
		RES_2048_1080,
		RES_2560_1440,
		RES_3840_2160,
		RES_4096_2160,
		RES_HIGHEST,

	};
private:
	Viewport *main_viewport;
	GraphicsPreset current_gp = GraphicsPreset::GRAPHICS_CUSTOM;
	DisplaySettings curr_ds = DisplaySettings::WINDOWED;
	ResolutionSettings curr_rs = ResolutionSettings::RES_DEFAULT;

	ResolutionList acceptable_resolution;

	int current_screen = 0;
	Vector2 last_window_pos;
	Vector2 default_window_size;
	Vector2 current_resolution;
	float ssaa = 1.0;
protected:
	static SettingsServer* singleton;
	static void _bind_methods();

	void set_vp_internal(Viewport* vp);
	void reset_screen_info();
	void set_res_internal();
	void set_ssaa_internal(const float& val);
	void load_gpp_internal() const;
	void emit_gs_changed(const uint16_t& setting);
public:
	SettingsServer();
	~SettingsServer();

	friend class SceneTree;

	static SettingsServer* get_singleton() { return singleton; }

	void set_main_viewport(Node* vp);
	inline Node* get_main_viewport() const { return (Node*)main_viewport; }
	
	void load_graphics_preset(GraphicsPreset preset);
	inline GraphicsPreset get_graphics_preset() const { return current_gp; }

	bool set_graphics_setting(GraphicsSetting setting, const Variant& value);
	Variant get_graphics_setting(GraphicsSetting setting) const;
	
	inline Vector2 get_default_window_size() const { return default_window_size; }
	Vector2 rs_to_vec2(ResolutionSettings rs);

	void set_window_size(ResolutionSettings res);
	inline ResolutionSettings get_window_size() const { return curr_rs; }

	void set_display_mode(DisplaySettings mode);
	inline DisplaySettings get_display_mode() const { return curr_ds; }

	void set_current_display(const uint16_t& display_no);
	inline int get_current_display() { return current_screen; }
};

#endif
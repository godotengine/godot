#include "svg_mgr.h"

#include "core/io/image_loader.h"
#include "spx_res_mgr.h"
#include "spx_engine.h"

SvgManager *SvgManager::singleton = nullptr;

SvgManager *SvgManager::get_singleton() {
	if (!singleton) {
		singleton = memnew(SvgManager);
	}
	return singleton;
}

SvgManager::SvgManager() {
	singleton = this;
}

SvgManager::~SvgManager() {
	svg_image_cache.clear();
	svg_animation_cache.clear();
	singleton = nullptr;
}

bool SvgManager::is_svg_file(const String& path) const {
	return path.to_lower().ends_with(".svg");
}

String SvgManager::_make_image_key(const String& path, int scale) {
	return String::num(scale) + "@" + path;
}

String SvgManager::_make_animation_key(const String& name, int scale) {
	return String::num(scale) + "@" + name;
}

Ref<ImageTexture> SvgManager::get_svg_image(const String& image_path, int scale) {
	if (!is_svg_file(image_path)) {
		return Ref<ImageTexture>();
	}
	
	String path = SpxEngine::get_singleton()->get_res()->_to_engine_path(image_path);
	return _load_image(path, scale);
}

Ref<SpriteFrames> SvgManager::get_svg_animation(const String& base_anim_key, int scale) {
	String key = _make_animation_key(base_anim_key, scale);
	
	if (svg_animation_cache.has(key)) {
		return svg_animation_cache[key];
	}
	
	return _load_animation(base_anim_key, scale);
}

bool SvgManager::is_svg_animation(const String& base_anim_key) {
	return is_svg_animation_registry[base_anim_key];
}

void SvgManager::mark_svg_animation(const String& base_anim_key, bool is_svg_animation) {
	is_svg_animation_registry[base_anim_key] = is_svg_animation;
}

Ref<SpriteFrames> SvgManager::_load_animation(const String& anim_name, int scale) {
	String key = _make_animation_key(anim_name, scale);
	
	if (svg_animation_cache.has(key)) {
		return svg_animation_cache[key];
	}
	
	Ref<SpriteFrames> frames;
	
	if (is_svg_file(anim_name)) {
		print_line("_load_animation: end with .svg anim_name: " + anim_name, "scale: " + String::num(scale));
		return frames;
	} 

	// Get animation definition from SpxResMgr
	auto res_mgr = SpxEngine::get_singleton()->get_res();
	if (!res_mgr) {
		print_error("[SvgManager] Cannot access SpxResMgr");
		return Ref<SpriteFrames>();
	}

	// Get existing animation frame list
	auto existing_frames = res_mgr->get_anim_frames(anim_name);
	if (!existing_frames.is_valid()) {
		print_error("[SvgManager] Animation not found: " + anim_name);
		return Ref<SpriteFrames>();
	}

	// If scale is 1, it's already created during res_mgr initialization, so return the cached one directly.
	if(scale == 1){
		svg_animation_cache[key] = existing_frames;
		return existing_frames;
	}

	// Check if animation contains SVG frames
	if (!existing_frames->has_animation(anim_name)) {
		print_error("[SvgManager] Animation key not found: " + anim_name);
		return Ref<SpriteFrames>();
	}

	// Create new SpriteFrames, replacing SVG textures with scaled versions
	Ref<SpriteFrames> new_frames;
	new_frames.instantiate();
	new_frames->add_animation(anim_name);

	// Copy animation properties
	new_frames->set_animation_loop(anim_name, existing_frames->get_animation_loop(anim_name));
	new_frames->set_animation_speed(anim_name, existing_frames->get_animation_speed(anim_name));

	int frame_count = existing_frames->get_frame_count(anim_name);
	for (int i = 0; i < frame_count; i++) {
		auto original_texture = existing_frames->get_frame_texture(anim_name, i);
		float duration = existing_frames->get_frame_duration(anim_name, i);
		
		// Check if it's an SVG texture
		String texture_path = original_texture->get_path(); // engine path
		if (is_svg_file(texture_path)) {
			// Load scaled version of SVG
			Ref<ImageTexture> scaled_texture = _load_image(texture_path, scale);
			if (scaled_texture.is_valid()) {
				new_frames->add_frame(anim_name, scaled_texture, duration);
			} else {
				// If SVG loading fails, use original texture
				new_frames->add_frame(anim_name, original_texture, duration);
			}
		} else {
			// Non-SVG textures directly use original texture
			new_frames->add_frame(anim_name, original_texture, duration);
		}
	}

	svg_animation_cache[key] = new_frames;

	return new_frames;
}

Ref<ImageTexture> SvgManager::_load_image(const String& path/*engine path*/, int scale) {
	String key = _make_image_key(path, scale);
	
	if (svg_image_cache.has(key)) {
		return svg_image_cache[key];
	}
	// Load SVG image
	Ref<Image> image;
	image.instantiate();
	Error err = ImageLoader::load_image(path, image, nullptr, 
							  ImageFormatLoader::FLAG_NONE, (float)scale);
	if (err == OK) {
		Ref<ImageTexture> texture;
		texture.instantiate();
		texture->set_image(image);
		texture->set_path_cache(path); // cache raw path, not engine path
		
		if(!svg_image_raw_size_cache.has(path)){
			svg_image_raw_size_cache[path] = Vector2(image->get_width()/scale,image->get_height()/scale);
		}
		// Cache texture
		svg_image_cache[key] = texture;
		return texture;
	}
	
	print_error("[SvgManager] Failed to load SVG image: " + path + " at scale " + String::num(scale));
	return Ref<ImageTexture>();
}

void SvgManager::destroy() {
	svg_image_cache.clear();
	svg_animation_cache.clear();
	singleton = nullptr;
}

int SvgManager::calculate_svg_scale(Vector2 required_scale) {
	float scale = MAX(required_scale.x, required_scale.y);
	// Use powers of 2: 1, 2, 4, 8, 16...
	if (scale <= 1.5f) return 1;
	if (scale <= 3.0f) return 2;
	if (scale <= 6.0f) return 4;
	return 8;
}

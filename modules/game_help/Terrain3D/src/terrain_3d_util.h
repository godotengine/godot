// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef TERRAIN3D_UTIL_CLASS_H
#define TERRAIN3D_UTIL_CLASS_H

#include "core/io/image.h"

#include "constants.h"


using namespace godot;

class Terrain3DUtil : public Object {
	GDCLASS(Terrain3DUtil, Object);
	CLASS_NAME_STATIC("Terrain3DUtil");

public:
	// Print info to the console
	static void print_dict(String name, const Dictionary &p_dict, int p_level = 2); // Level 2: DEBUG
	static void dump_gen(GeneratedTexture p_gen, String name = "", int p_level = 2);
	static void dump_maps(const TypedArray<Image> p_maps, String p_name = "");

	// Image operations
	static Ref<Image> black_to_alpha(const Ref<Image> p_image);
	static Vector2 get_min_max(const Ref<Image> p_image);
	static Ref<Image> get_thumbnail(const Ref<Image> p_image, Vector2i p_size = Vector2i(256, 256));
	static Ref<Image> get_filled_image(Vector2i p_size,
			Color p_color = COLOR_BLACK,
			bool p_create_mipmaps = true,
			Image::Format p_format = Image::FORMAT_MAX);
	static Ref<Image> load_image(String p_file_name, int p_cache_mode = ResourceFormatLoader::CACHE_MODE_IGNORE,
			Vector2 p_r16_height_range = Vector2(0.f, 255.f), Vector2i p_r16_size = Vector2i(0, 0));
	static Ref<Image> pack_image(const Ref<Image> p_src_rgb, const Ref<Image> p_src_r, bool p_invert_green_channel = false);

protected:
	static void _bind_methods();
};

typedef Terrain3DUtil Util;

// Inline Functions

///////////////////////////
// Math
///////////////////////////

template <typename T>
T round_multiple(T p_value, T p_multiple) {
	if (p_multiple == 0) {
		return p_value;
	}
	return static_cast<T>(std::round(static_cast<double>(p_value) / static_cast<double>(p_multiple)) * static_cast<double>(p_multiple));
}

// Returns the bilinearly interpolated value derived from parameters:
// * 4 values to be interpolated
// * Positioned at the 4 corners of the p_pos00 - p_pos11 rectangle
// * Interpolated to the position p_pos, which is global, not a 0-1 percentage
inline real_t bilerp(real_t p_v00, real_t p_v01, real_t p_v10, real_t p_v11,
		Vector2 p_pos00, Vector2 p_pos11, Vector2 p_pos) {
	real_t x2x1 = p_pos11.x - p_pos00.x;
	real_t y2y1 = p_pos11.y - p_pos00.y;
	real_t x2x = p_pos11.x - p_pos.x;
	real_t y2y = p_pos11.y - p_pos.y;
	real_t xx1 = p_pos.x - p_pos00.x;
	real_t yy1 = p_pos.y - p_pos00.y;
	return (p_v00 * x2x * y2y +
				   p_v01 * x2x * yy1 +
				   p_v10 * xx1 * y2y +
				   p_v11 * xx1 * yy1) /
			(x2x1 * y2y1);
}

inline real_t bilerp(real_t p_v00, real_t p_v01, real_t p_v10, real_t p_v11,
		Vector3 p_pos00, Vector3 p_pos11, Vector3 p_pos) {
	Vector2 pos00 = Vector2(p_pos00.x, p_pos00.z);
	Vector2 pos11 = Vector2(p_pos11.x, p_pos11.z);
	Vector2 pos = Vector2(p_pos.x, p_pos.z);
	return bilerp(p_v00, p_v01, p_v10, p_v11, pos00, pos11, pos);
}

///////////////////////////
// Controlmap Handling
///////////////////////////

// Getters read the 32-bit float as a 32-bit uint, then mask bits to retreive value
// Encoders return a full 32-bit uint with bits in the proper place for ORing
inline float as_float(uint32_t value) { return *(float *)&value; }
inline uint32_t as_uint(float value) { return *(uint32_t *)&value; }

inline uint8_t get_base(uint32_t pixel) { return pixel >> 27 & 0x1F; }
inline uint8_t get_base(float pixel) { return get_base(as_uint(pixel)); }
inline uint32_t enc_base(uint8_t base) { return (base & 0x1F) << 27; }

inline uint8_t get_overlay(uint32_t pixel) { return pixel >> 22 & 0x1F; }
inline uint8_t get_overlay(float pixel) { return get_overlay(as_uint(pixel)); }
inline uint32_t enc_overlay(uint8_t over) { return (over & 0x1F) << 22; }

inline uint8_t get_blend(uint32_t pixel) { return pixel >> 14 & 0xFF; }
inline uint8_t get_blend(float pixel) { return get_blend(as_uint(pixel)); }
inline uint32_t enc_blend(uint8_t blend) { return (blend & 0xFF) << 14; }

inline uint8_t get_uv_rotation(uint32_t pixel) { return pixel >> 10 & 0xF; }
inline uint8_t get_uv_rotation(float pixel) { return get_uv_rotation(as_uint(pixel)); }
inline uint32_t enc_uv_rotation(uint8_t rotation) { return (rotation & 0xF) << 10; }

inline uint8_t get_uv_scale(uint32_t pixel) { return pixel >> 7 & 0x7; }
inline uint8_t get_uv_scale(float pixel) { return get_uv_scale(as_uint(pixel)); }
inline uint32_t enc_uv_scale(uint8_t scale) { return (scale & 0x7) << 7; }

inline bool is_hole(uint32_t pixel) { return (pixel >> 2 & 0x1) == 1; }
inline bool is_hole(float pixel) { return is_hole(as_uint(pixel)); }
inline uint32_t enc_hole(bool hole) { return (hole & 0x1) << 2; }

inline bool is_nav(uint32_t pixel) { return (pixel >> 1 & 0x1) == 1; }
inline bool is_nav(float pixel) { return is_nav(as_uint(pixel)); }
inline uint32_t enc_nav(bool nav) { return (nav & 0x1) << 1; }

inline bool is_auto(uint32_t pixel) { return (pixel & 0x1) == 1; }
inline bool is_auto(float pixel) { return is_auto(as_uint(pixel)); }
inline uint32_t enc_auto(bool autosh) { return autosh & 0x1; }

// Aliases for GDScript since it can't handle overridden functions
inline uint32_t gd_get_base(uint32_t pixel) { return get_base(pixel); }
inline uint32_t gd_enc_base(uint32_t base) { return enc_base(base); }
inline uint32_t gd_get_overlay(uint32_t pixel) { return get_overlay(pixel); }
inline uint32_t gd_enc_overlay(uint32_t over) { return enc_overlay(over); }
inline uint32_t gd_get_blend(uint32_t pixel) { return get_overlay(pixel); }
inline uint32_t gd_enc_blend(uint32_t blend) { return enc_blend(blend); }
inline bool gd_is_hole(uint32_t pixel) { return is_hole(pixel); }
inline bool gd_is_auto(uint32_t pixel) { return is_auto(pixel); }
inline bool gd_is_nav(uint32_t pixel) { return is_nav(pixel); }
inline uint32_t gd_get_uv_rotation(uint32_t pixel) { return get_uv_rotation(pixel); }
inline uint32_t gd_enc_uv_rotation(uint32_t rotation) { return enc_uv_rotation(rotation); }
inline uint32_t gd_get_uv_scale(uint32_t pixel) { return get_uv_rotation(pixel); }
inline uint32_t gd_enc_uv_scale(uint32_t scale) { return enc_uv_rotation(scale); }


///////////////////////////
// Memory
///////////////////////////

template <typename TType>
_FORCE_INLINE_ bool memdelete_safely(TType *&p_ptr) {
	if (p_ptr != nullptr) {
		memdelete(p_ptr);
		p_ptr = nullptr;
		return true;
	}
	return false;
}

_FORCE_INLINE_ bool remove_from_tree(Node *p_node) {
	// Note: is_in_tree() doesn't work in Godot-cpp 4.1.3
	if (p_node != nullptr) {
		Node *parent = p_node->get_parent();
		if (parent != nullptr) {
			parent->remove_child(p_node);
			return true;
		}
	}
	return false;
}

#endif // TERRAIN3D_UTIL_CLASS_H

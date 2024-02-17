// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef UTIL_CLASS_H
#define UTIL_CLASS_H

//#include <godot_cpp/classes/image.hpp>
#include "core/io/image.h"

#include "constants.h"
#include "generated_tex.h"

using namespace godot;

class Util {
public:
	static inline const char *__class__ = "Terrain3DUtil";

	// Controlmap handling
	// Getters read the 32-bit float as a 32-bit uint, then mask bits to retreive value
	// Encoders return a full 32-bit uint with bits in the proper place for ORing
	static inline float as_float(uint32_t value) { return *(float *)&value; }
	static inline uint32_t as_uint(float value) { return *(uint32_t *)&value; }
	static inline uint32_t get_mask(float pixel, uint32_t mask) { return as_uint(pixel) & mask; }

	static inline uint8_t get_base(float pixel) { return get_base(as_uint(pixel)); }
	static inline uint8_t get_base(uint32_t pixel) { return pixel >> 27 & 0x1F; }
	static inline uint32_t enc_base(uint8_t base) { return (base & 0x1F) << 27; }

	static inline uint8_t get_overlay(float pixel) { return get_overlay(as_uint(pixel)); }
	static inline uint8_t get_overlay(uint32_t pixel) { return pixel >> 22 & 0x1F; }
	static inline uint32_t enc_overlay(uint8_t over) { return (over & 0x1F) << 22; }

	static inline uint8_t get_blend(float pixel) { return get_blend(as_uint(pixel)); }
	static inline uint8_t get_blend(uint32_t pixel) { return pixel >> 14 & 0xFF; }
	static inline uint32_t enc_blend(uint8_t blend) { return (blend & 0xFF) << 14; }

	static inline bool is_hole(float pixel) { return is_hole(as_uint(pixel)); }
	static inline bool is_hole(uint32_t pixel) { return (pixel >> 2 & 0x1) == 1; }
	static inline uint32_t enc_hole(bool hole) { return (hole & 0x1) << 2; }

	static inline bool is_nav(float pixel) { return is_nav(as_uint(pixel)); }
	static inline bool is_nav(uint32_t pixel) { return (pixel >> 1 & 0x1) == 1; }
	static inline uint32_t enc_nav(bool nav) { return (nav & 0x1) << 1; }

	static inline bool is_auto(float pixel) { return is_auto(as_uint(pixel)); }
	static inline bool is_auto(uint32_t pixel) { return (pixel & 0x1) == 1; }
	static inline uint32_t enc_auto(bool autosh) { return autosh & 0x1; }

	// Print info to the console
	static void print_dict(String name, const Dictionary &p_dict, int p_level = 1); // Defaults to INFO
	static void dump_gen(GeneratedTex p_gen, String name = "");
	static void dump_maps(const TypedArray<Image> p_maps, String p_name = "");

	// Image operations
	static Vector2 get_min_max(const Ref<Image> p_image);
	static Ref<Image> get_thumbnail(const Ref<Image> p_image, Vector2i p_size = Vector2i(256, 256));
	static Ref<Image> get_filled_image(Vector2i p_size,
			Color p_color = COLOR_BLACK,
			bool p_create_mipmaps = true,
			Image::Format p_format = Image::FORMAT_MAX);
	static Ref<Image> pack_image(const Ref<Image> p_src_rgb, const Ref<Image> p_src_r, bool p_invert_green_channel = false);
};

template <typename TType>
_FORCE_INLINE_ void memdelete_safely(TType *&p_ptr) {
	if (p_ptr != nullptr) {
		memdelete(p_ptr);
		p_ptr = nullptr;
	}
}

#endif // UTIL_CLASS_H

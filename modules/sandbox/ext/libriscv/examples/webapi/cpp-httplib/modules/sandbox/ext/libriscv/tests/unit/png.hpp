#include "ext/lodepng/lodepng.h"
#include "ext/lodepng/lodepng.cpp"
#include <array>
#include <cstdint>
#include <vector>
#define PNG_8BIT

inline constexpr uint32_t bgr24(uint32_t r, uint32_t g, uint32_t b) {
	return r | (g << 8) | (b << 16) | (255 << 24);
}

static constexpr std::array<uint32_t, 16> color_mapping {
	bgr24(66, 30, 15),
	bgr24(25, 7, 26),
	bgr24(9, 1, 47),
	bgr24(4, 4, 73),
	bgr24(0, 7, 100),
	bgr24(12, 44, 138),
	bgr24(24, 82, 177),
	bgr24(57, 125, 209),
	bgr24(134, 181, 229),
	bgr24(211, 236, 248),
	bgr24(241, 233, 191),
	bgr24(248, 201, 95),
	bgr24(255, 170, 0),
	bgr24(204, 128, 0),
	bgr24(153, 87, 0),
	bgr24(106, 52, 3),
};

auto encode(size_t W, size_t H, const uint8_t* data)
{
	std::vector<uint8_t> png;

#ifdef PNG_8BIT
	lodepng::State state;
	lodepng_state_init(&state);
	for (const auto color : color_mapping) {
		lodepng_palette_add(&state.info_raw,
			(color >> 0)  & 0xFF,
			(color >> 8)  & 0xFF,
			(color >> 16) & 0xFF,
			(color >> 24) & 0xFF);
		lodepng_palette_add(&state.info_png.color,
			(color >> 0)  & 0xFF,
			(color >> 8)  & 0xFF,
			(color >> 16) & 0xFF,
			(color >> 24) & 0xFF);
	}
	state.info_png.color.colortype = LCT_PALETTE;
	state.info_png.color.bitdepth  = 8;
	state.info_raw.colortype = LCT_PALETTE;
	state.info_raw.bitdepth  = 8;
	state.encoder.auto_convert = 0;
	// 1: Disable LZ77 but still use Huffman
	state.encoder.zlibsettings.btype = 2;
	state.encoder.zlibsettings.use_lz77 = 0;
	// 2: Disable compression completely
	//state.encoder.zlibsettings.btype = 0;

	lodepng::encode(png, data, W, H, state);
#else
	lodepng::encode(png, data, W, H);
#endif
	return png;
}

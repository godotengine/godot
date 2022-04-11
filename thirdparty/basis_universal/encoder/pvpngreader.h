// pngreader.h - Public Domain - see unlicense at bottom of pvpngreader.cpp
#pragma once
#include <stdint.h>

namespace pv_png
{
	// PNG color types
	enum
	{
		PNG_COLOR_TYPE_GREYSCALE = 0,
		PNG_COLOR_TYPE_TRUECOLOR = 2,
		PNG_COLOR_TYPE_PALETTIZED = 3,
		PNG_COLOR_TYPE_GREYSCALE_ALPHA = 4,
		PNG_COLOR_TYPE_TRUECOLOR_ALPHA = 6
	};

	// PNG file description
	struct png_info
	{
		uint32_t m_width;
		uint32_t m_height;
				
		uint32_t m_num_chans;	// The number of channels, factoring in transparency. Ranges from [1-4].

		uint32_t m_bit_depth;	// PNG ihdr bit depth: 1, 2, 4, 8 or 16
		uint32_t m_color_type;	// PNG ihdr color type, PNG_COLOR_TYPE_GRAYSCALE etc.

		bool m_has_gamma;		// true if the PNG file had a GAMA chunk
		uint32_t m_gamma_value; // PNG GAMA chunk value, scaled by 100000

		bool m_has_trns;		// true if the PNG file used colorkey transparency
	};

	// Retrieved information about the PNG file.
	// Returns false on any errors.
	bool get_png_info(const void* pImage_buf, size_t buf_size, png_info& info);

	// Input parameters:
	// pImage_buf, buf_size - pointer to PNG image data
	// desired_chans - desired number of output channels. 0=auto, 1=grayscale, 2=grayscale alpha, 3=24bpp RGB, 4=32bpp RGBA
	//
	// Output parameters:
	// width, height - PNG image resolution
	// num_chans - actual number of channels in PNG, from [1,4] (factoring in transparency)
	//
	// Returns nullptr on any errors.
	void* load_png(const void* pImage_buf, size_t buf_size, uint32_t desired_chans, uint32_t &width, uint32_t &height, uint32_t& num_chans);
}

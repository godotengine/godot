// rgbcx.h v1.13
// High-performance scalar encoders and RDO (Rate Distortion Optimization) post processors for BC1-5.
// Public Domain or MIT license (you choose - see below), written by Richard Geldreich 2020 <richgel99@gmail.com>.
//
// Influential references:
// https://tinyurl.com/y3vxz457 (Ortego and Ramchandran, "Rate-distortion Methods for Image and Video Compression", 1998)
// http://sjbrown.co.uk/2006/01/19/dxt-compression-techniques/
// https://github.com/nothings/stb/blob/master/stb_dxt.h
// https://gist.github.com/castano/c92c7626f288f9e99e158520b14a61cf
// https://github.com/castano/icbc/blob/master/icbc.h
// http://www.humus.name/index.php?page=3D&ID=79
//
// This is a single header file library. Be sure to "#define RGBCX_IMPLEMENTATION" in one .cpp file somewhere.
//
// Instructions:
//
// The library MUST be initialized by calling this function at least once before using any encoder or decoder functions:
//
// void rgbcx::init(bc1_approx_mode mode = cBC1Ideal);
//
// This function manipulates global state, so it is not thread safe. 
// You can call it multiple times to change the global BC1 approximation mode.
// Important: BC1/3 textures encoded using non-ideal BC1 approximation modes should only be sampled on parts from that vendor.
// If you encode for AMD, average error on AMD parts will go down, but average error on NVidia parts will go up and vice versa.
// If in doubt, encode in ideal BC1 mode.
//
// Call these functions to encode BC1-5:
// void rgbcx::encode_bc1(uint32_t level, void* pDst, const uint8_t* pPixels, bool allow_3color, bool use_transparent_texels_for_black);
// void rgbcx::encode_bc3(uint32_t level, void* pDst, const uint8_t* pPixels);
// void rgbcx::encode_bc4(void* pDst, const uint8_t* pPixels, uint32_t stride = 4);
// void rgbcx::encode_bc5(void* pDst, const uint8_t* pPixels, uint32_t chan0 = 0, uint32_t chan1 = 1, uint32_t stride = 4);
//
// - level ranges from MIN_LEVEL to MAX_LEVEL. The higher the level, the slower the encoder goes, but the higher the average quality.
// levels [0,4] are fast and compete against stb_dxt (default and HIGHQUAL). The remaining levels compete against squish/NVTT/icbc and icbc HQ.
// If in doubt just use level 10, set allow_3color to true and use_transparent_texels_for_black to false, and adjust as needed.
//
// - pDst is a pointer to the 8-byte (BC1/4) or 16-byte (BC3/5) destination block. 
//
// - pPixels is a pointer to the 32-bpp pixels, in either RGBX or RGBA format (R is first in memory). 
// Alpha is always ignored by encode_bc1().
//
// - allow_3color: If true the encoder will use 3-color blocks. This flag is ignored unless level is >= 5 (because lower levels compete against stb_dxt and it doesn't support 3-color blocks).
// Do not enable on BC3-5 textures. 3-color block usage slows down encoding.
//
// - use_transparent_texels_for_black: If true the encoder will use 3-color block transparent black pixels to code very dark or black texels. Your engine/shader MUST ignore the sampled 
// alpha value for textures encoded in this mode. This is how NVidia's classic "nvdxt" encoder (used by many original Xbox titles) used to work by default on DXT1C textures. It increases 
// average quality substantially (because dark texels/black are very common) and is highly recommended.
// Do not enable on BC3-5 textures. 
//
// - stride is the source pixel stride, in bytes. It's typically 4.
//
// - chan0 and chan1 are the source channels. Typically they will be 0 and 1.
//
// All encoding and decoding functions are threade-safe.
//
// To reduce the compiled size of the encoder, set #define RGBCX_USE_SMALLER_TABLES to 1 before including this header.
//
#ifndef RGBCX_INCLUDE_H
#define RGBCX_INCLUDE_H

#ifdef _MSC_VER
#pragma warning (disable:4201) //nameless struct/union
#endif

#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <assert.h>
#include <limits.h>

// By default, the table used to accelerate cluster fit on 4 color blocks uses a 969x128 entry table. 
// To reduce the executable size, set RGBCX_USE_SMALLER_TABLES to 1, which selects the smaller 969x32 entry table. 
#ifndef RGBCX_USE_SMALLER_TABLES
#define RGBCX_USE_SMALLER_TABLES 0
#endif

namespace rgbcx
{
	enum class bc1_approx_mode
	{
		// The default mode. No rounding for 4-color colors 2,3. My older tools/compressors use this mode. 
		// This matches the D3D10 docs on BC1.
		cBC1Ideal = 0,

		// NVidia GPU mode.
		cBC1NVidia = 1,

		// AMD GPU mode.
		cBC1AMD = 2,
		
		// This mode matches AMD Compressonator's output. It rounds 4-color colors 2,3 (not 3-color color 2).
		// This matches the D3D9 docs on DXT1.
		cBC1IdealRound4 = 3
	};

	enum class eNoClamp { cNoClamp };
	static inline uint8_t clamp255(int32_t i) { return (uint8_t)((i & 0xFFFFFF00U) ? (~(i >> 31)) : i); }

	template <typename S> inline S maximum(S a, S b) { return (a > b) ? a : b; }
	template <typename S> inline S maximum(S a, S b, S c) { return maximum(maximum(a, b), c); }
	template <typename S> inline S maximum(S a, S b, S c, S d) { return maximum(maximum(maximum(a, b), c), d); }

	template <typename S> inline S minimum(S a, S b) { return (a < b) ? a : b; }
	template <typename S> inline S minimum(S a, S b, S c) { return minimum(minimum(a, b), c); }
	template <typename S> inline S minimum(S a, S b, S c, S d) { return minimum(minimum(minimum(a, b), c), d); }
		
	struct color32
	{
		union
		{
			struct
			{
				uint8_t r;
				uint8_t g;
				uint8_t b;
				uint8_t a;
			};

			uint8_t c[4];

			uint32_t m;
		};

		color32() { }

		color32(uint32_t vr, uint32_t vg, uint32_t vb, uint32_t va) { set(vr, vg, vb, va); }
		color32(eNoClamp unused, uint32_t vr, uint32_t vg, uint32_t vb, uint32_t va) { (void)unused; set_noclamp_rgba(vr, vg, vb, va); }

		void set(uint32_t vr, uint32_t vg, uint32_t vb, uint32_t va) { c[0] = static_cast<uint8_t>(vr); c[1] = static_cast<uint8_t>(vg); c[2] = static_cast<uint8_t>(vb); c[3] = static_cast<uint8_t>(va); }

		void set_noclamp_rgb(uint32_t vr, uint32_t vg, uint32_t vb) { c[0] = static_cast<uint8_t>(vr); c[1] = static_cast<uint8_t>(vg); c[2] = static_cast<uint8_t>(vb); }
		void set_noclamp_rgba(uint32_t vr, uint32_t vg, uint32_t vb, uint32_t va) { set(vr, vg, vb, va); }

		void set_clamped(int vr, int vg, int vb, int va) { c[0] = clamp255(vr); c[1] = clamp255(vg);	c[2] = clamp255(vb); c[3] = clamp255(va); }

		uint8_t operator[] (uint32_t idx) const { assert(idx < 4); return c[idx]; }
		uint8_t& operator[] (uint32_t idx) { assert(idx < 4); return c[idx]; }

		bool operator== (const color32& rhs) const { return m == rhs.m; }

		void set_rgb(const color32& other) { c[0] = static_cast<uint8_t>(other.c[0]); c[1] = static_cast<uint8_t>(other.c[1]); c[2] = static_cast<uint8_t>(other.c[2]); }

		static color32 comp_min(const color32& a, const color32& b) { return color32(eNoClamp::cNoClamp, std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]), std::min(a[3], b[3])); }
		static color32 comp_max(const color32& a, const color32& b) { return color32(eNoClamp::cNoClamp, std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]), std::max(a[3], b[3])); }
	};

	// init() MUST be called once before using the BC1 encoder.
	// This function may be called multiple times to change the BC1 approximation mode. 
	// This function initializes global state, so don't call it while other threads inside the encoder.
	// Important: If you encode textures for a specific vendor's GPU's, beware that using that texture data on other GPU's may result in ugly artifacts. 
	// Encode to cBC1Ideal unless you know the texture data will only be deployed or used on a specific vendor's GPU.
	void init(bc1_approx_mode mode = bc1_approx_mode::cBC1Ideal);

	// Optimally encodes a solid color block to BC1 format.
	void encode_bc1_solid_block(void* pDst, uint32_t fr, uint32_t fg, uint32_t fb, bool allow_3color);

	// BC1 low-level API encoder flags. You can ignore this if you use the simple level API.
	enum
	{
		// Try to improve quality using the most likely total orderings. 
		// The total_orderings_to_try parameter will then control the number of total orderings to try for 4 color blocks, and the 
		// total_orderings_to_try3 parameter will control the number of total orderings to try for 3 color blocks (if they are enabled).
		cEncodeBC1UseLikelyTotalOrderings = 2,
		
		// Use 2 least squares pass, instead of one (same as stb_dxt's HIGHQUAL option).
		// Recommended if you're enabling cEncodeBC1UseLikelyTotalOrderings.
		cEncodeBC1TwoLeastSquaresPasses = 4,
				
		// cEncodeBC1Use3ColorBlocksForBlackPixels allows the BC1 encoder to use 3-color blocks for blocks containing black or very dark pixels. 
		// You shader/engine MUST ignore the alpha channel on textures encoded with this flag.
		// Average quality goes up substantially for my 100 texture corpus (~.5 dB), so it's worth using if you can.
		// Note the BC1 encoder does not actually support transparency in 3-color mode.
		// Don't set when encoding to BC3.
		cEncodeBC1Use3ColorBlocksForBlackPixels = 8,

		// If cEncodeBC1Use3ColorBlocks is set, the encoder can use 3-color mode for a small but noticeable gain in average quality, but lower perf.
		// If you also specify the cEncodeBC1UseLikelyTotalOrderings flag, set the total_orderings_to_try3 paramter to the number of total orderings to try.
		// Don't set when encoding to BC3.
		cEncodeBC1Use3ColorBlocks = 16,

		// cEncodeBC1Iterative will greatly increase encode time, but is very slightly higher quality.
		// Same as squish's iterative cluster fit option. Not really worth the tiny boost in quality, unless you just don't care about perf. at all.
		cEncodeBC1Iterative = 32,

		// cEncodeBC1BoundingBox enables a fast all-integer PCA approximation on 4-color blocks. 
		// At level 0 options (no other flags), this is ~15% faster, and higher *average* quality.
		cEncodeBC1BoundingBox = 64,
				
		// Use a slightly lower quality, but ~30% faster MSE evaluation function for 4-color blocks.
		cEncodeBC1UseFasterMSEEval = 128,
		
		// Examine all colors to compute selectors/MSE (slower than default)
		cEncodeBC1UseFullMSEEval = 256,

		// Use 2D least squares+inset+optimal rounding (the method used in Humus's GPU texture encoding demo), instead of PCA. 
		// Around 18% faster, very slightly lower average quality to better (depends on the content).
		cEncodeBC1Use2DLS = 512,

		// Use 6 power iterations vs. 4 for PCA.
		cEncodeBC1Use6PowerIters = 2048,
		
		// Check all total orderings - *very* slow. The encoder is not designed to be used in this way.
		cEncodeBC1Exhaustive = 8192,

		// Try 2 different ways of choosing the initial endpoints.
		cEncodeBC1TryAllInitialEndponts = 16384,
		
		// Same as cEncodeBC1BoundingBox, but implemented using integer math (faster, slightly less quality)
		cEncodeBC1BoundingBoxInt = 32768,
		
		// Try refining the final endpoints by examining nearby colors.
		cEncodeBC1EndpointSearchRoundsShift = 22,
		cEncodeBC1EndpointSearchRoundsMask = 1023U << cEncodeBC1EndpointSearchRoundsShift,
	};
		
	const uint32_t MIN_TOTAL_ORDERINGS = 1;
	const uint32_t MAX_TOTAL_ORDERINGS3 = 32;

#if RGBCX_USE_SMALLER_TABLES
	const uint32_t MAX_TOTAL_ORDERINGS4 = 32;
#else
	const uint32_t MAX_TOTAL_ORDERINGS4 = 128;
#endif
	
	// DEFAULT_TOTAL_ORDERINGS_TO_TRY is around 3x faster than libsquish at slightly higher average quality. 10-16 is a good range to start to compete against libsquish.
	const uint32_t DEFAULT_TOTAL_ORDERINGS_TO_TRY = 10;

	const uint32_t DEFAULT_TOTAL_ORDERINGS_TO_TRY3 = 1;
	
	// Encodes a 4x4 block of RGBX (X=ignored) pixels to BC1 format. 
	// This is the simplified interface for BC1 encoding, which accepts a level parameter and converts that to the best overall flags.
	// The pixels are in RGBA format, where R is first in memory. The BC1 encoder completely ignores the alpha channel (i.e. there is no punchthrough alpha support).
	// This is the recommended function to use for BC1 encoding, becuase it configures the encoder for you in the best possible way (on average).
	// Note that the 3 color modes won't be used at all until level 5 or higher.
	// No transparency supported, however if you set use_transparent_texels_for_black to true the encocer will use transparent selectors on very dark/black texels to reduce MSE. 
	const uint32_t MIN_LEVEL = 0, MAX_LEVEL = 18;
	void encode_bc1(uint32_t level, void* pDst, const uint8_t* pPixels, bool allow_3color, bool use_transparent_texels_for_black, const uint8_t* pForce_selectors = nullptr);

	// Low-level interface for BC1 encoding.
	// Always returns a 4 color block, unless cEncodeBC1Use3ColorBlocksForBlackPixels or cEncodeBC1Use3ColorBlock flags are specified. 
	// total_orderings_to_try controls the perf. vs. quality tradeoff on 4-color blocks when the cEncodeBC1UseLikelyTotalOrderings flag is used. It must range between [MIN_TOTAL_ORDERINGS, MAX_TOTAL_ORDERINGS4].
	// total_orderings_to_try3 controls the perf. vs. quality tradeoff on 3-color bocks when the cEncodeBC1UseLikelyTotalOrderings and the cEncodeBC1Use3ColorBlocks flags are used. Valid range is [0,MAX_TOTAL_ORDERINGS3] (0=disabled).
	void encode_bc1(void* pDst, const uint8_t* pPixels, uint32_t flags = 0, uint32_t total_orderings_to_try = DEFAULT_TOTAL_ORDERINGS_TO_TRY, uint32_t total_orderings_to_try3 = DEFAULT_TOTAL_ORDERINGS_TO_TRY3, const uint8_t *pForce_selectors = nullptr);
	
	// Constants used for high quality BC4/BC5 encoding (and alpha of BC3)
	const uint32_t BC4_DEFAULT_SEARCH_RAD = 3;
	const uint32_t BC4_USE_MODE8_FLAG = 1;
	const uint32_t BC4_USE_MODE6_FLAG = 2;
	const uint32_t BC4_USE_ALL_MODES = 3;

	// Encodes a 4x4 block of RGBA pixels to BC3 format.
	// There are two encode_bc3() functions. 
	// The first is the recommended function, which accepts a level parameter.
	// The second is a low-level version that allows fine control over BC1 encoding. 
	void encode_bc3(uint32_t level, void* pDst, const uint8_t* pPixels);
	void encode_bc3(void* pDst, const uint8_t* pPixels, uint32_t flags = 0, uint32_t total_orderings_to_try = DEFAULT_TOTAL_ORDERINGS_TO_TRY);
	void encode_bc3_hq(uint32_t level, void* pDst, const uint8_t* pPixels, uint32_t alpha_search_rad = BC4_DEFAULT_SEARCH_RAD, uint32_t alpha_modes = BC4_USE_ALL_MODES);
			
	// Encodes a single channel to BC4.
	// stride is the source pixel stride in bytes.
	void encode_bc4(void* pDst, const uint8_t* pPixels, uint32_t stride = 4);
	uint32_t encode_bc4_hq(void* pDst, const uint8_t* pPixels, uint32_t stride = 4, uint32_t search_rad = BC4_DEFAULT_SEARCH_RAD, uint32_t mode_flag = BC4_USE_ALL_MODES, const uint8_t* pForce_selectors = nullptr);

	// Encodes two channels to BC5. 
	// chan0/chan1 control which channels, stride is the source pixel stride in bytes.
	void encode_bc5(void* pDst, const uint8_t* pPixels, uint32_t chan0 = 0, uint32_t chan1 = 1, uint32_t stride = 4);
	void encode_bc5_hq(void* pDst, const uint8_t* pPixels, uint32_t chan0 = 0, uint32_t chan1 = 1, uint32_t stride = 4, uint32_t alpha_search_rad = BC4_DEFAULT_SEARCH_RAD, uint32_t alpha_modes = BC4_USE_ALL_MODES);

	// Decompression functions. 

	bool unpack_bc1_block_colors(const void* pBlock_bits, color32* c, bc1_approx_mode mode = bc1_approx_mode::cBC1Ideal);
	
	// Returns true if the block uses 3 color punchthrough alpha mode.
	bool unpack_bc1(const void* pBlock_bits, void* pPixels, bool set_alpha = true, bc1_approx_mode mode = bc1_approx_mode::cBC1Ideal);
	
	void unpack_bc4(const void* pBlock_bits, uint8_t* pPixels, uint32_t stride = 4);
	
	// Returns true if the block uses 3 color punchthrough alpha mode.
	bool unpack_bc3(const void* pBlock_bits, void* pPixels, bc1_approx_mode mode = bc1_approx_mode::cBC1Ideal);

	void unpack_bc5(const void* pBlock_bits, void* pPixels, uint32_t chan0 = 0, uint32_t chan1 = 1, uint32_t stride = 4);

	// Rate Distortion Optimization (RDO)
	enum dxt_constants
	{
		cDXT1SelectorBits = 2U, cDXT1SelectorValues = 1U << cDXT1SelectorBits, cDXT1SelectorMask = cDXT1SelectorValues - 1U,
		cDXT5SelectorBits = 3U, cDXT5SelectorValues = 1U << cDXT5SelectorBits, cDXT5SelectorMask = cDXT5SelectorValues - 1U,
	};

	struct bc1_block
	{
		enum { cTotalEndpointBytes = 2, cTotalSelectorBytes = 4 };

		uint8_t m_low_color[cTotalEndpointBytes];
		uint8_t m_high_color[cTotalEndpointBytes];
		uint8_t m_selectors[cTotalSelectorBytes];

		inline uint32_t get_low_color() const { return m_low_color[0] | (m_low_color[1] << 8U); }
		inline uint32_t get_high_color() const { return m_high_color[0] | (m_high_color[1] << 8U); }
		inline bool is_3color() const { return get_low_color() <= get_high_color(); }
		inline void set_low_color(uint16_t c) { m_low_color[0] = static_cast<uint8_t>(c & 0xFF); m_low_color[1] = static_cast<uint8_t>((c >> 8) & 0xFF); }
		inline void set_high_color(uint16_t c) { m_high_color[0] = static_cast<uint8_t>(c & 0xFF); m_high_color[1] = static_cast<uint8_t>((c >> 8) & 0xFF); }
		inline uint32_t get_selector(uint32_t x, uint32_t y) const { assert((x < 4U) && (y < 4U)); return (m_selectors[y] >> (x * cDXT1SelectorBits)) & cDXT1SelectorMask; }
		inline void set_selector(uint32_t x, uint32_t y, uint32_t val) { assert((x < 4U) && (y < 4U) && (val < 4U)); m_selectors[y] &= (~(cDXT1SelectorMask << (x * cDXT1SelectorBits))); m_selectors[y] |= (val << (x * cDXT1SelectorBits)); }

		inline uint32_t get_endpoint_bits() const { return m_low_color[0] | (m_low_color[1] << 8) | (m_high_color[0] << 16) | (m_high_color[1] << 24); }
		inline void set_endpoint_bits(uint32_t s) { m_low_color[0] = (uint8_t)s; m_low_color[1] = (uint8_t)(s >> 8); m_high_color[0] = (uint8_t)(s >> 16); m_high_color[1] = (uint8_t)(s >> 24); }

		inline uint32_t get_selector_bits() const { return m_selectors[0] | (m_selectors[1] << 8) | (m_selectors[2] << 16) | (m_selectors[3] << 24); }
		inline void set_selector_bits(uint32_t s) { m_selectors[0] = (uint8_t)s; m_selectors[1] = (uint8_t)(s >> 8); m_selectors[2] = (uint8_t)(s >> 16); m_selectors[3] = (uint8_t)(s >> 24); }

		inline bool any_selectors_transparent() const
		{
			uint32_t sel_bits = get_selector_bits();
			for (uint32_t i = 0; i < 16; i++)
			{
				if ((sel_bits & 3) == 3)
					return true;

				sel_bits >>= 2;
			}
			return false;
		}

		static inline uint16_t pack_color(const color32& color, bool scaled, uint32_t bias = 127U)
		{
			uint32_t r = color.r, g = color.g, b = color.b;
			if (scaled)
			{
				r = (r * 31U + bias) / 255U;
				g = (g * 63U + bias) / 255U;
				b = (b * 31U + bias) / 255U;
			}
			return static_cast<uint16_t>(minimum(b, 31U) | (minimum(g, 63U) << 5U) | (minimum(r, 31U) << 11U));
		}

		static inline uint16_t pack_unscaled_color(uint32_t r, uint32_t g, uint32_t b) { return static_cast<uint16_t>(b | (g << 5U) | (r << 11U)); }

		static inline void unpack_color(uint32_t c, uint32_t& r, uint32_t& g, uint32_t& b)
		{
			r = (c >> 11) & 31;
			g = (c >> 5) & 63;
			b = c & 31;

			r = (r << 3) | (r >> 2);
			g = (g << 2) | (g >> 4);
			b = (b << 3) | (b >> 2);
		}

		static inline void unpack_color_unscaled(uint32_t c, uint32_t& r, uint32_t& g, uint32_t& b)
		{
			r = (c >> 11) & 31;
			g = (c >> 5) & 63;
			b = c & 31;
		}
	};

	struct bc4_block
	{
		enum { cBC4SelectorBits = 3, cTotalSelectorBytes = 6, cMaxSelectorValues = 8 };
		uint8_t m_endpoints[2];

		uint8_t m_selectors[cTotalSelectorBytes];

		inline uint32_t get_low_alpha() const { return m_endpoints[0]; }
		inline uint32_t get_high_alpha() const { return m_endpoints[1]; }
		inline bool is_alpha6_block() const { return get_low_alpha() <= get_high_alpha(); }

		inline uint64_t get_selector_bits() const
		{
			return ((uint64_t)((uint32_t)m_selectors[0] | ((uint32_t)m_selectors[1] << 8U) | ((uint32_t)m_selectors[2] << 16U) | ((uint32_t)m_selectors[3] << 24U))) |
				(((uint64_t)m_selectors[4]) << 32U) |
				(((uint64_t)m_selectors[5]) << 40U);
		}

		inline void set_selector_bits(uint64_t v)
		{
			for (uint32_t i = 0; i < 6; i++)
			{
				m_selectors[i] = (uint8_t)v;
				v >>= 8;
			}
		}

		inline uint32_t get_selector(uint32_t x, uint32_t y, uint64_t selector_bits) const
		{
			assert((x < 4U) && (y < 4U));
			return (selector_bits >> (((y * 4) + x) * cBC4SelectorBits)) & (cMaxSelectorValues - 1);
		}

		static inline uint32_t get_block_values6(uint8_t* pDst, uint32_t l, uint32_t h)
		{
			pDst[0] = static_cast<uint8_t>(l);
			pDst[1] = static_cast<uint8_t>(h);
			pDst[2] = static_cast<uint8_t>((l * 4 + h) / 5);
			pDst[3] = static_cast<uint8_t>((l * 3 + h * 2) / 5);
			pDst[4] = static_cast<uint8_t>((l * 2 + h * 3) / 5);
			pDst[5] = static_cast<uint8_t>((l + h * 4) / 5);
			pDst[6] = 0;
			pDst[7] = 255;
			return 6;
		}

		static inline uint32_t get_block_values8(uint8_t* pDst, uint32_t l, uint32_t h)
		{
			pDst[0] = static_cast<uint8_t>(l);
			pDst[1] = static_cast<uint8_t>(h);
			pDst[2] = static_cast<uint8_t>((l * 6 + h) / 7);
			pDst[3] = static_cast<uint8_t>((l * 5 + h * 2) / 7);
			pDst[4] = static_cast<uint8_t>((l * 4 + h * 3) / 7);
			pDst[5] = static_cast<uint8_t>((l * 3 + h * 4) / 7);
			pDst[6] = static_cast<uint8_t>((l * 2 + h * 5) / 7);
			pDst[7] = static_cast<uint8_t>((l + h * 6) / 7);
			return 8;
		}

		static inline uint32_t get_block_values(uint8_t* pDst, uint32_t l, uint32_t h)
		{
			if (l > h)
				return get_block_values8(pDst, l, h);
			else
				return get_block_values6(pDst, l, h);
		}
	};

}
#endif // #ifndef RGBCX_INCLUDE_H

#ifdef RGBCX_IMPLEMENTATION
#endif //#ifdef RGBCX_IMPLEMENTATION

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright(c) 2020 Richard Geldreich, Jr.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions :
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain(www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non - commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain.We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors.We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/


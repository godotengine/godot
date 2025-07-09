// File: basisu_astc_hdr_common.h
#pragma once
#include "basisu_enc.h"
#include "basisu_gpu_texture.h"
#include "../transcoder/basisu_astc_helpers.h"
#include "../transcoder/basisu_astc_hdr_core.h"

namespace basisu
{
	const uint32_t MAX_ASTC_HDR_BLOCK_W = 6, MAX_ASTC_HDR_BLOCK_H = 6;
	const uint32_t MAX_ASTC_HDR_ENC_BLOCK_PIXELS = 6 * 6;

	const uint32_t MODE11_TOTAL_SUBMODES = 8; // plus an extra hidden submode, directly encoded, for direct, so really 9 (see tables 99/100 of the ASTC spec)
	const uint32_t MODE7_TOTAL_SUBMODES = 6;
		
	// [ise_range][0] = # levels
	// [ise_range][1...] = lerp value [0,64]
	// in ASTC order
	// Supported ISE weight ranges: 0 to 11, 12 total
	const uint32_t MIN_SUPPORTED_ISE_WEIGHT_INDEX = astc_helpers::BISE_2_LEVELS; // ISE 0=2 levels
	const uint32_t MAX_SUPPORTED_ISE_WEIGHT_INDEX = astc_helpers::BISE_32_LEVELS; // ISE 11=16 levels
	const uint32_t MIN_SUPPORTED_WEIGHT_LEVELS = 2;
	const uint32_t MAX_SUPPORTED_WEIGHT_LEVELS = 32;

	extern const uint8_t g_ise_weight_lerps[MAX_SUPPORTED_ISE_WEIGHT_INDEX + 1][33];

	const float Q_LOG_BIAS_4x4 = .125f; // the original UASTC HDR 4x4 log bias
	const float Q_LOG_BIAS_6x6 = 1.0f; // the log bias both encoders use now

	const float LDR_TO_HDR_NITS = 100.0f;

	struct astc_hdr_codec_base_options
	{
		float m_r_err_scale, m_g_err_scale;
		float m_q_log_bias;
		
		bool m_ultra_quant;
		
		// If true, the ASTC HDR compressor is allowed to more aggressively vary weight indices for slightly higher compression in non-fastest mode. This will hurt BC6H quality, however.
		bool m_allow_uber_mode;

		bool m_mode7_full_s_optimization;

		bool m_take_first_non_clamping_mode11_submode;
		bool m_take_first_non_clamping_mode7_submode;

		bool m_disable_weight_plane_optimization;
		
		astc_hdr_codec_base_options() { init(); }

		void init();
	};

	inline int get_bit(
		int src_val, int src_bit)
	{
		assert(src_bit >= 0 && src_bit <= 31);
		int bit = (src_val >> src_bit) & 1;
		return bit;
	}

	inline void pack_bit(
		int& dst, int dst_bit,
		int src_val, int src_bit = 0)
	{
		assert(dst_bit >= 0 && dst_bit <= 31);
		int bit = get_bit(src_val, src_bit);
		dst |= (bit << dst_bit);
	}

	inline uint32_t get_max_qlog(uint32_t bits)
	{
		switch (bits)
		{
		case 7: return basist::MAX_QLOG7;
		case 8: return basist::MAX_QLOG8;
		case 9: return basist::MAX_QLOG9;
		case 10: return basist::MAX_QLOG10;
		case 11: return basist::MAX_QLOG11;
		case 12: return basist::MAX_QLOG12;
		case 16: return basist::MAX_QLOG16;
		default: assert(0); break;
		}
		return 0;
	}

#if 0
	inline float get_max_qlog_val(uint32_t bits)
	{
		switch (bits)
		{
		case 7: return MAX_QLOG7_VAL;
		case 8: return MAX_QLOG8_VAL;
		case 9: return MAX_QLOG9_VAL;
		case 10: return MAX_QLOG10_VAL;
		case 11: return MAX_QLOG11_VAL;
		case 12: return MAX_QLOG12_VAL;
		case 16: return MAX_QLOG16_VAL;
		default: assert(0); break;
		}
		return 0;
	}
#endif

#if 0
	// Input is the low 11 bits of the qlog
	// Returns the 10-bit mantissa of the half float value
	int qlog11_to_half_float_mantissa(int M)
	{
		assert(M <= 0x7FF);
		int Mt;
		if (M < 512)
			Mt = 3 * M;
		else if (M >= 1536)
			Mt = 5 * M - 2048;
		else
			Mt = 4 * M - 512;
		return (Mt >> 3);
	}
#endif

	// Input is the 10-bit mantissa of the half float value
	// Output is the 11-bit qlog value
	// Inverse of qlog11_to_half_float_mantissa()
	inline int half_float_mantissa_to_qlog11(int hf)
	{
		int q0 = (hf * 8 + 2) / 3;
		int q1 = (hf * 8 + 2048 + 4) / 5;

		if (q0 < 512)
			return q0;
		else if (q1 >= 1536)
			return q1;

		int q2 = (hf * 8 + 512 + 2) / 4;
		return q2;
	}

	inline int half_to_qlog16(int hf)
	{
		assert(!basist::half_is_signed((basist::half_float)hf) && !basist::is_half_inf_or_nan((basist::half_float)hf));

		// extract 5 bits exponent, which is carried through to qlog16 unchanged
		const int exp = (hf >> 10) & 0x1F;

		// extract and invert the 10 bit mantissa to nearest qlog11 (should be lossless)
		const int mantissa = half_float_mantissa_to_qlog11(hf & 0x3FF);
		assert(mantissa <= 0x7FF);

		// Now combine to qlog16, which is what ASTC HDR interpolates using the [0-64] weights.
		uint32_t qlog16 = (exp << 11) | mantissa;

		// should be a lossless operation
		assert(astc_helpers::qlog16_to_half(qlog16) == hf);

		return qlog16;
	}

	void interpolate_qlog12_colors(
		const int e[2][3],
		basist::half_float* pDecoded_half,
		vec3F* pDecoded_float,
		uint32_t n, uint32_t ise_weight_range);

	bool get_astc_hdr_mode_11_block_colors(
		const uint8_t* pEndpoints,
		basist::half_float* pDecoded_half,
		vec3F* pDecoded_float,
		uint32_t n, uint32_t ise_weight_range, uint32_t ise_endpoint_range);

	bool get_astc_hdr_mode_7_block_colors(
		const uint8_t* pEndpoints,
		basist::half_float* pDecoded_half,
		vec3F* pDecoded_float,
		uint32_t n, uint32_t ise_weight_range, uint32_t ise_endpoint_range);
			
	// Fast high precision piecewise linear approximation of log2(bias+x).
	// Half may be zero, positive or denormal. No NaN/Inf/negative.
	BASISU_FORCE_INLINE double q(basist::half_float x, float log_bias)
	{
		union { float f; int32_t i; uint32_t u; } fi;

		fi.f = fast_half_to_float_pos_not_inf_or_nan(x);

		assert(fi.f >= 0.0f);
						
		fi.f += log_bias;

		return (double)fi.u; // approx log2f(fi.f), need to return double for the precision
	}

	BASISU_FORCE_INLINE uint32_t q2(basist::half_float x, float log_bias)
	{
		union { float f; int32_t i; uint32_t u; } fi;

		fi.f = fast_half_to_float_pos_not_inf_or_nan(x);

		assert(fi.f >= 0.0f);
		
		fi.f += log_bias;

		return fi.u;
	}

	double eval_selectors(
		uint32_t num_pixels,
		uint8_t* pWeights,
		uint32_t ise_weight_range,
		const basist::half_float* pBlock_pixels_half,
		uint32_t num_weight_levels,
		const basist::half_float* pDecoded_half,
		const astc_hdr_codec_base_options& coptions,
		uint32_t usable_selector_bitmask = UINT32_MAX);

	double eval_selectors_dual_plane(
		uint32_t channel_index,
		uint32_t num_pixels,
		uint8_t* pWeights0, uint8_t* pWeights1,
		const basist::half_float* pBlock_pixels_half,
		uint32_t num_weight_levels,
		const basist::half_float* pDecoded_half,
		const astc_hdr_codec_base_options& coptions,
		uint32_t usable_selector_bitmask = UINT32_MAX);

	double compute_block_error(uint32_t num_pixels, const basist::half_float* pOrig_block, const basist::half_float* pPacked_block, const astc_hdr_codec_base_options& coptions);

	const uint32_t FIRST_MODE7_SUBMODE_INDEX = 0;
	const uint32_t MAX_MODE7_SUBMODE_INDEX = 5;

	bool pack_mode7(
		const vec3F& high_color_q16, const float s_q16,
		uint32_t ise_endpoint_range, uint8_t* pEndpoints,
		uint32_t ise_weight_range, // only used for determining biasing during CEM 7 packing
		const astc_hdr_codec_base_options& coptions,
		int32_t first_submode, int32_t last_submode, bool ignore_clamping, uint32_t& submode_used);

	bool try_mode7(
		uint32_t num_pixels,
		uint8_t* pEndpoints, uint8_t* pWeights, double& cur_block_error, uint32_t& submode_used,
		const vec3F& high_color_q16, const float s_q16,
		const basist::half_float block_pixels_half[][3],
		uint32_t num_weight_levels, uint32_t ise_weight_range, const astc_hdr_codec_base_options& coptions,
		uint32_t ise_endpoint_range,
		int32_t first_submode = 0, int32_t last_submode = MAX_MODE7_SUBMODE_INDEX);

	bool pack_mode11(
		const vec3F& low_color_q16, const vec3F& high_color_q16,
		uint32_t ise_endpoint_range, uint8_t* pEndpoints,
		const astc_hdr_codec_base_options& coptions,
		bool direct_only, int32_t first_submode, int32_t last_submode, bool ignore_clamping, uint32_t& submode_used);

	bool try_mode11(uint32_t num_pixels,
		uint8_t* pEndpoints, uint8_t* pWeights, double& cur_block_error, uint32_t& submode_used,
		const vec3F& low_color_q16, const vec3F& high_color_q16,
		const basist::half_float block_pixels_half[][3],
		uint32_t num_weight_levels, uint32_t ise_weight_range, const astc_hdr_codec_base_options& coptions, bool direct_only, uint32_t ise_endpoint_range,
		bool constrain_ise_weight_selectors,
		int32_t first_submode, int32_t last_submode, bool ignore_clamping);

	bool try_mode11_dual_plane(uint32_t channel_index, uint32_t num_pixels,
		uint8_t* pEndpoints, uint8_t* pWeights0, uint8_t* pWeights1, double& cur_block_error, uint32_t& submode_used,
		const vec3F& low_color_q16, const vec3F& high_color_q16,
		const basist::half_float block_pixels_half[][3],
		uint32_t num_weight_levels, uint32_t ise_weight_range, const astc_hdr_codec_base_options& coptions, bool direct_only, uint32_t ise_endpoint_range,
		bool constrain_ise_weight_selectors,
		int32_t first_submode, int32_t last_submode, bool ignore_clamping);

	const int FIRST_MODE11_SUBMODE_INDEX = -1;
	const int MAX_MODE11_SUBMODE_INDEX = 7;

	enum opt_mode_t
	{
		cNoOpt,
		cOrdinaryLeastSquares,
		cWeightedLeastSquares,
		cWeightedLeastSquaresHeavy,
		cWeightedAverage
	};

	struct encode_astc_block_stats
	{
		uint32_t m_num_pixels;
		vec3F m_mean_q16;
		vec3F m_axis_q16;

		void init(uint32_t num_pixels, const vec4F pBlock_pixels_q16[]);
	};

	double encode_astc_hdr_block_mode_11(
		uint32_t num_pixels,
		const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
		uint32_t ise_weight_range,
		uint32_t& best_submode,
		double cur_block_error,
		uint8_t* blk_endpoints, uint8_t* blk_weights,
		const astc_hdr_codec_base_options& coptions,
		bool direct_only,
		uint32_t ise_endpoint_range,
		bool uber_mode,
		bool constrain_ise_weight_selectors,
		int32_t first_submode, int32_t last_submode, bool ignore_clamping, 
		opt_mode_t opt_mode, 
		const encode_astc_block_stats *pBlock_stats = nullptr);

	double encode_astc_hdr_block_downsampled_mode_11(
		uint32_t block_x, uint32_t block_y, uint32_t grid_x, uint32_t grid_y,
		uint32_t ise_weight_range, uint32_t ise_endpoint_range,
		uint32_t num_pixels, const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
		double cur_block_error,
		int32_t first_submode, int32_t last_submode, bool ignore_clamping, opt_mode_t opt_mode,
		uint8_t* pBlk_endpoints, uint8_t* pBlk_weights, uint32_t& best_submode,
		const astc_hdr_codec_base_options& coptions,
		const encode_astc_block_stats* pBlock_stats = nullptr);

	double encode_astc_hdr_block_mode_11_dual_plane(
		uint32_t num_pixels,
		const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
		uint32_t channel_index,		// 0-2
		uint32_t ise_weight_range,
		uint32_t& best_submode,
		double cur_block_error,
		uint8_t* blk_endpoints, uint8_t* blk_weights0, uint8_t* blk_weights1,
		const astc_hdr_codec_base_options& coptions,
		bool direct_only,
		uint32_t ise_endpoint_range,
		bool uber_mode,
		bool constrain_ise_weight_selectors,
		int32_t first_submode, int32_t last_submode, 
		bool ignore_clamping);

	double encode_astc_hdr_block_mode_7(
		uint32_t num_pixels,
		const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
		uint32_t ise_weight_range,
		uint32_t& best_submode,
		double cur_block_error,
		uint8_t* blk_endpoints,  //[4]
		uint8_t* blk_weights, // [num_pixels]
		const astc_hdr_codec_base_options& coptions,
		uint32_t ise_endpoint_range, 
		int first_submode = 0, int last_submode = MAX_MODE7_SUBMODE_INDEX, 
		const encode_astc_block_stats *pBlock_stats = nullptr);

	//--------------------------------------------------------------------------------------------------------------------------

	struct mode11_log_desc
	{
		int32_t m_submode;
		int32_t m_maj_comp;

		// Or R0, G0, B0 if maj_comp==3 (direct)
		int32_t m_a;  // positive
		int32_t m_c;  // positive
		int32_t m_b0; // positive

		// Or R1, G1, B1 if maj_comp==3 (direct)
		int32_t m_b1; // positive
		int32_t m_d0; // if not direct, is signed
		int32_t m_d1; // if not direct, is signed

		// limits if not direct
		int32_t m_a_bits, m_c_bits, m_b_bits, m_d_bits;
		int32_t m_max_a_val, m_max_c_val, m_max_b_val, m_min_d_val, m_max_d_val;

		void clear() { clear_obj(*this); }

		bool is_direct() const { return m_maj_comp == 3; }
	};

	//--------------------------------------------------------------------------------------------------------------------------
	bool pack_astc_mode7_submode(uint32_t submode, uint8_t* pEndpoints, const vec3F& rgb_q16, float s_q16, int& max_clamp_mag, uint32_t ise_weight_range, bool early_out_if_clamped, int max_clamp_mag_accept_thresh);

	bool pack_astc_mode11_submode(uint32_t submode, uint8_t* pEndpoints, int val_q[2][3], int& max_clamp_mag, bool early_out_if_clamped = false, int max_clamp_mag_accept_thresh = 0);
	bool pack_astc_mode11_submode(uint32_t submode, uint8_t* pEndpoints, const vec3F& low_q16, const vec3F& high_q16, int& max_clamp_mag, bool early_out_if_clamped = false, int max_clamp_mag_accept_thresh = 0);
	void pack_astc_mode11_direct(uint8_t* pEndpoints, vec3F l_q16, vec3F h_q16);
	
	bool pack_mode11(mode11_log_desc& desc, uint8_t* pEndpoints);
	void unpack_mode11(const uint8_t* pEndpoints, mode11_log_desc& desc);

	void decode_cem_11_config(const uint8_t* pEndpoints, int& submode_index, int& maj_index);
	void decode_cem_7_config(const uint8_t* pEndpoints, int& submode_index, int& maj_index);
		
	void dequantize_astc_weights(uint32_t n, const uint8_t* pSrc_ise_vals, uint32_t from_ise_range, uint8_t* pDst_raw_weights);

	const float* get_6x6_downsample_matrix(uint32_t grid_width, uint32_t grid_height);
	
	void downsample_weight_grid(
		const float* pMatrix_weights,
		uint32_t bx, uint32_t by,		// source/from dimension (block size)
		uint32_t wx, uint32_t wy,		// dest/to dimension (grid size)
		const uint8_t* pSrc_weights,	// these are dequantized weights, NOT ISE symbols, [by][bx]
		uint8_t* pDst_weights);			// [wy][wx]

	void downsample_ise_weights(
		uint32_t weight_ise_range, uint32_t quant_weight_ise_range,
		uint32_t block_w, uint32_t block_h,
		uint32_t grid_w, uint32_t grid_h,
		const uint8_t* pSrc_weights, uint8_t* pDst_weights);

	void downsample_ise_weights_dual_plane(
		uint32_t dequant_weight_ise_range, uint32_t quant_weight_ise_range,
		uint32_t block_w, uint32_t block_h,
		uint32_t grid_w, uint32_t grid_h,
		const uint8_t* pSrc_weights0, const uint8_t* pSrc_weights1,
		uint8_t* pDst_weights);

	bool refine_endpoints(
		uint32_t cem,
		uint32_t endpoint_ise_range,
		uint8_t* pEndpoint_vals, // the endpoints to optimize
		uint32_t block_w, uint32_t block_h, // block dimensions
		uint32_t grid_w, uint32_t grid_h, const uint8_t* pWeights, uint32_t weight_ise_range, // weight grid
		uint32_t num_pixels, const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
		const uint8_t* pPixel_block_ofs, // maps this subset's pixels to block offsets
		astc_hdr_codec_base_options& coptions, opt_mode_t opt_mode);
	
	extern bool g_astc_hdr_enc_initialized;

	// This MUST be called before encoding any blocks.
	void astc_hdr_enc_init();

} // namespace basisu


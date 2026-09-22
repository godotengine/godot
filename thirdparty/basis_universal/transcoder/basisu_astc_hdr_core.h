// File: basisu_astc_hdr_core.h
#pragma once
#include "basisu_astc_helpers.h"

namespace basist
{
	struct astc_blk
	{
		uint8_t m_vals[16];
	};

	// ASTC_HDR_MAX_VAL is the maximum color component value that can be encoded.
	// If the input has values higher than this, they need to be linearly scaled so all values are between [0,ASTC_HDR_MAX_VAL], and the linear scaling inverted in the shader.
	const float ASTC_HDR_MAX_VAL = 65216.0f; // actually MAX_QLOG12_VAL

	// Maximum usable QLOG encodings, and their floating point equivalent values, that don't result in NaN/Inf's.
	const uint32_t MAX_QLOG7 = 123;
	//const float MAX_QLOG7_VAL = 55296.0f;

	const uint32_t MAX_QLOG8 = 247;
	//const float MAX_QLOG8_VAL = 60416.0f;

	const uint32_t MAX_QLOG9 = 495;
	//const float MAX_QLOG9_VAL = 62976.0f;

	const uint32_t MAX_QLOG10 = 991;
	//const float MAX_QLOG10_VAL = 64256.0f;

	const uint32_t MAX_QLOG11 = 1983;
	//const float MAX_QLOG11_VAL = 64896.0f;

	const uint32_t MAX_QLOG12 = 3967;
	//const float MAX_QLOG12_VAL = 65216.0f;

	const uint32_t MAX_QLOG16 = 63487;
	const float MAX_QLOG16_VAL = 65504.0f;

	// TODO: Should be called something like "NUM_MODE11_ENDPOINT_VALUES"
	const uint32_t NUM_MODE11_ENDPOINTS = 6, NUM_MODE7_ENDPOINTS = 4;

	// This is not lossless
	inline half_float qlog_to_half(uint32_t qlog, uint32_t bits)
	{
		assert((bits >= 7U) && (bits <= 16U));
		assert(qlog < (1U << bits));

		int C = qlog << (16 - bits);
		return astc_helpers::qlog16_to_half(C);
	}

	void astc_hdr_core_init();

	void decode_mode7_to_qlog12_ise20(
		const uint8_t* pEndpoints,
		int e[2][3],
		int* pScale);

	bool decode_mode7_to_qlog12(
		const uint8_t* pEndpoints,
		int e[2][3],
		int* pScale,
		uint32_t ise_endpoint_range);

	void decode_mode11_to_qlog12_ise20(
		const uint8_t* pEndpoints,
		int e[2][3]);

	bool decode_mode11_to_qlog12(
		const uint8_t* pEndpoints,
		int e[2][3],
		uint32_t ise_endpoint_range);

	bool transcode_bc6h_1subset(half_float h_e[3][2], const astc_helpers::log_astc_block& best_blk, bc6h_block& transcoded_bc6h_blk);
	bool transcode_bc6h_2subsets(uint32_t common_part_index, const astc_helpers::log_astc_block& best_blk, bc6h_block& transcoded_bc6h_blk);

	bool astc_hdr_transcode_to_bc6h(const astc_blk& src_blk, bc6h_block& dst_blk);
	bool astc_hdr_transcode_to_bc6h(const astc_helpers::log_astc_block& log_blk, bc6h_block& dst_blk);

	namespace astc_6x6_hdr
	{
		const uint32_t MAX_ASTC_HDR_6X6_DIM = 32768;
		const int32_t REUSE_MAX_BUFFER_ROWS = 5; // 1+-(-4), so we need to buffer 5 rows total

		struct block_mode_desc
		{
			bool m_dp;
			uint32_t m_cem;
			uint32_t m_num_partitions;
			uint32_t m_grid_x;
			uint32_t m_grid_y;

			// the coding ISE ranges (which may not be valid ASTC ranges for this configuration)
			uint32_t m_endpoint_ise_range;
			uint32_t m_weight_ise_range;

			// the physical/output ASTC decompression ISE ranges (i.e. what the decompressor must output)
			uint32_t m_transcode_endpoint_ise_range;
			uint32_t m_transcode_weight_ise_range;

			uint32_t m_flags;
			int m_dp_channel;
		};

		// Lack of level flag indicates level 3+
		const uint32_t BASIST_HDR_6X6_LEVEL0 = 1;
		const uint32_t BASIST_HDR_6X6_LEVEL1 = 2;
		const uint32_t BASIST_HDR_6X6_LEVEL2 = 4;

		const uint32_t TOTAL_BLOCK_MODE_DECS = 75;
		extern const block_mode_desc g_block_mode_descs[TOTAL_BLOCK_MODE_DECS];

		void copy_weight_grid(bool dual_plane, uint32_t grid_x, uint32_t grid_y, const uint8_t* transcode_weights, astc_helpers::log_astc_block& decomp_blk);

		enum class encoding_type
		{
			cInvalid = -1,
			cRun = 0,
			cSolid = 1,
			cReuse = 2,
			cBlock = 3,
			cTotal
		};

		const uint32_t REUSE_XY_DELTA_BITS = 5;
		const uint32_t NUM_REUSE_XY_DELTAS = 1 << REUSE_XY_DELTA_BITS;

		struct reuse_xy_delta
		{
			int8_t m_x, m_y;
		};

		extern const reuse_xy_delta g_reuse_xy_deltas[NUM_REUSE_XY_DELTAS];

		const uint32_t RUN_CODE = 0b000, RUN_CODE_LEN = 3;
		const uint32_t SOLID_CODE = 0b100, SOLID_CODE_LEN = 3;
		const uint32_t REUSE_CODE = 0b10, REUSE_CODE_LEN = 2;
		const uint32_t BLOCK_CODE = 0b1, BLOCK_CODE_LEN = 1;

		enum class endpoint_mode
		{
			cInvalid = -1,

			cRaw = 0,
			cUseLeft,
			cUseUpper,
			cUseLeftDelta,
			cUseUpperDelta,

			cTotal
		};

		enum class block_mode
		{
			cInvalid = -1,

			cBMTotalModes = TOTAL_BLOCK_MODE_DECS
		};

		const uint32_t NUM_ENDPOINT_DELTA_BITS = 5;

		const uint32_t NUM_UNIQUE_PARTITIONS2 = 521;
		extern const uint32_t g_part2_unique_index_to_seed[NUM_UNIQUE_PARTITIONS2];

		const uint32_t NUM_UNIQUE_PARTITIONS3 = 333;
		extern const uint32_t g_part3_unique_index_to_seed[NUM_UNIQUE_PARTITIONS3];

		bool decode_values(basist::bitwise_decoder& decoder, uint32_t total_values, uint32_t ise_range, uint8_t* pValues);

		void requantize_astc_weights(uint32_t n, const uint8_t* pSrc_ise_vals, uint32_t from_ise_range, uint8_t* pDst_ise_vals, uint32_t to_ise_range);

		void requantize_ise_endpoints(uint32_t cem, uint32_t src_ise_endpoint_range, const uint8_t* pSrc_endpoints, uint32_t dst_ise_endpoint_range, uint8_t* pDst_endpoints);

		const uint32_t BC6H_NUM_DIFF_ENDPOINT_MODES_TO_TRY_2 = 2;
		const uint32_t BC6H_NUM_DIFF_ENDPOINT_MODES_TO_TRY_4 = 4;
		const uint32_t BC6H_NUM_DIFF_ENDPOINT_MODES_TO_TRY_9 = 9;

		struct fast_bc6h_params
		{
			uint32_t m_num_diff_endpoint_modes_to_try;
			uint32_t m_max_2subset_pats_to_try;

			bool m_hq_ls;
			bool m_brute_force_weight4_assignment;
			
			fast_bc6h_params()
			{
				init();
			}

			void init()
			{
				m_hq_ls = true;
				m_num_diff_endpoint_modes_to_try = BC6H_NUM_DIFF_ENDPOINT_MODES_TO_TRY_2;
				m_max_2subset_pats_to_try = 1;
				m_brute_force_weight4_assignment = false;
			}
		};

		void fast_encode_bc6h(const basist::half_float* pPixels, basist::bc6h_block* pBlock, const fast_bc6h_params &params);

		bool decode_6x6_hdr(const uint8_t* pComp_data, uint32_t comp_data_size, basisu::vector2D<astc_helpers::astc_block>& decoded_blocks, uint32_t& width, uint32_t& height);

	} // namespace astc_6x6_hdr

} // namespace basist


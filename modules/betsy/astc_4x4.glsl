#[versions]

rgb = "";
rgba = "#define HAS_ALPHA";
normal = "#define IS_NORMALMAP";

#[compute]
#version 450

#extension GL_EXT_samplerless_texture_functions : require

#VERSION_DEFINES

// #define TEXEL_VEC4

#ifdef TEXEL_VEC4
#define TEXEL_TYPE vec4
#define TEXEL_READ(tex, idx) tex[idx]
#else
#define TEXEL_TYPE uint
#define TEXEL_READ(tex, idx) (unpackUnorm4x8(tex[idx]) * 255.0)
#endif

#ifndef THREAD_NUM_X
#define THREAD_NUM_X 8
#endif

#ifndef THREAD_NUM_Y
#define THREAD_NUM_Y 8
#endif

#ifdef BLOCK_6X6
#define DIM 6
#else
#define DIM 4
#endif

#define BLOCK_SIZE ((DIM) * (DIM))

#define BLOCK_BYTES 16

#define X_GRIDS 4
#define Y_GRIDS 4

#define SMALL_VALUE 0.00001

/*
 * supported color_endpoint_mode
 */
#define CEM_LDR_RGB_DIRECT 8
#define CEM_LDR_RGBA_DIRECT 12

/**
 * form [ARM:astc-encoder]
 * Define normalized (starting at zero) numeric ranges that can be represented
 * with 8 bits or less.
 */
#define QUANT_2 0
#define QUANT_3 1
#define QUANT_4 2
#define QUANT_5 3
#define QUANT_6 4
#define QUANT_8 5
#define QUANT_10 6
#define QUANT_12 7
#define QUANT_16 8
#define QUANT_20 9
#define QUANT_24 10
#define QUANT_32 11
#define QUANT_40 12
#define QUANT_48 13
#define QUANT_64 14
#define QUANT_80 15
#define QUANT_96 16
#define QUANT_128 17
#define QUANT_160 18
#define QUANT_192 19
#define QUANT_256 20
#define QUANT_MAX 21

layout(push_constant, std430) uniform Params {
	uint texel_height;
	uint texel_width;
	uint group_num_x;
	uint pad;
}
params;

layout(binding = 0) uniform sampler2D InTexture;

layout(binding = 1, std430) restrict writeonly buffer OutBuffer {
	uvec4 _data[];
}
OutBuffer_1;

// layout(binding = 1, rgba8ui) uniform restrict writeonly uimage2D dstTexture;

layout(local_size_x = THREAD_NUM_X, local_size_y = THREAD_NUM_Y, local_size_z = 1) in;

const int bits_trits_quints_table[QUANT_MAX * 3] = int[](
		1, 0, 0, // RANGE_2
		0, 1, 0, // RANGE_3
		2, 0, 0, // RANGE_4
		0, 0, 1, // RANGE_5
		1, 1, 0, // RANGE_6
		3, 0, 0, // RANGE_8
		1, 0, 1, // RANGE_10
		2, 1, 0, // RANGE_12
		4, 0, 0, // RANGE_16
		2, 0, 1, // RANGE_20
		3, 1, 0, // RANGE_24
		5, 0, 0, // RANGE_32
		3, 0, 1, // RANGE_40
		4, 1, 0, // RANGE_48
		6, 0, 0, // RANGE_64
		4, 0, 1, // RANGE_80
		5, 1, 0, // RANGE_96
		7, 0, 0, // RANGE_128
		5, 0, 1, // RANGE_160
		6, 1, 0, // RANGE_192
		8, 0, 0 // RANGE_256
);

// clang-format off
const int integer_from_trits[243] = int[](
		0,1,2,    4,5,6,    8,9,10,
		16,17,18, 20,21,22, 24,25,26,
		3,7,15,   19,23,27, 12,13,14,
		32,33,34, 36,37,38, 40,41,42,
		48,49,50, 52,53,54, 56,57,58,
		35,39,47, 51,55,59, 44,45,46,
		64,65,66, 68,69,70, 72,73,74,
		80,81,82, 84,85,86, 88,89,90,
		67,71,79, 83,87,91, 76,77,78,

		128,129,130, 132,133,134, 136,137,138,
		144,145,146, 148,149,150, 152,153,154,
		131,135,143, 147,151,155, 140,141,142,
		160,161,162, 164,165,166, 168,169,170,
		176,177,178, 180,181,182, 184,185,186,
		163,167,175, 179,183,187, 172,173,174,
		192,193,194, 196,197,198, 200,201,202,
		208,209,210, 212,213,214, 216,217,218,
		195,199,207, 211,215,219, 204,205,206,

		96,97,98,    100,101,102, 104,105,106,
		112,113,114, 116,117,118, 120,121,122,
		99,103,111,  115,119,123, 108,109,110,
		224,225,226, 228,229,230, 232,233,234,
		240,241,242, 244,245,246, 248,249,250,
		227,231,239, 243,247,251, 236,237,238,
		28,29,30,    60,61,62,    92,93,94,
		156,157,158, 188,189,190, 220,221,222,
		31,63,127,   159,191,255, 252,253,254
);

const int integer_from_quints[125] = int[](
		0,1,2,3,4,          8,9,10,11,12,           16,17,18,19,20,         24,25,26,27,28,         5,13,21,29,6,
		32,33,34,35,36,     40,41,42,43,44,         48,49,50,51,52,         56,57,58,59,60,         37,45,53,61,14,
		64,65,66,67,68,     72,73,74,75,76,         80,81,82,83,84,         88,89,90,91,92,         69,77,85,93,22,
		96,97,98,99,100,    104,105,106,107,108,    112,113,114,115,116,    120,121,122,123,124,    101,109,117,125,30,
		102,103,70,71,38,   110,111,78,79,46,       118,119,86,87,54,       126,127,94,95,62,       39,47,55,63,31
);

// clang-format on

//
// ASTC_Table.hlsl
//
// from [ARM:astc-encoder] quantization_and_transfer_table quant_and_xfer_tables
#define WEIGHT_QUANTIZE_NUM 32
const int scramble_table[12 * WEIGHT_QUANTIZE_NUM] = int[](
		// quantization method 0, range 0..1
		0, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 1, range 0..2
		0, 1, 2,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 2, range 0..3
		0, 1, 2, 3,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 3, range 0..4
		0, 1, 2, 3, 4,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 4, range 0..5
		0, 2, 4, 5, 3, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 5, range 0..7
		0, 1, 2, 3, 4, 5, 6, 7,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 6, range 0..9
		0, 2, 4, 6, 8, 9, 7, 5, 3, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 7, range 0..11
		0, 4, 8, 2, 6, 10, 11, 7, 3, 9, 5, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 8, range 0..15
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 9, range 0..19
		0, 4, 8, 12, 16, 2, 6, 10, 14, 18, 19, 15, 11, 7, 3, 17, 13, 9, 5, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 10, range 0..23
		0, 8, 16, 2, 10, 18, 4, 12, 20, 6, 14, 22, 23, 15, 7, 21, 13, 5, 19, 11, 3, 17, 9, 1,
		0, 0, 0, 0, 0, 0, 0, 0,
		// quantization method 11, range 0..31
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

// number must be <= 255; bitcount must be <= 8
void orbits8_ptr(inout uvec4 outputs, inout uint bitoffset, uint number, int bitcount) {
	uint newpos = bitoffset + uint(bitcount);
	uint nidx = newpos >> uint(5);
	uint uidx = bitoffset >> uint(5);
	uint bit_idx = bitoffset & 31u;
	uint bytes[4] = uint[](outputs.x, outputs.y, outputs.z, outputs.w);
	bytes[uidx] |= (number << bit_idx);
	bytes[uidx + 1u] |= ((nidx > uidx) ? (number >> (32u - bit_idx)) : 0u);
	outputs.x = bytes[0];
	outputs.y = bytes[1];
	outputs.z = bytes[2];
	outputs.w = bytes[3];
	bitoffset = newpos;
}

void split_high_low(uint n, int i, out int high, out uint low) {
	low = bitfieldExtract(n, 0, i);
	high = int(bitfieldExtract(n, i, 8));
}

/**
 * Encode a group of 5 numbers using trits and bits.
 */
void encode_trits(int bitcount, uint b0, uint b1, uint b2, uint b3, uint b4, inout uvec4 outputs, inout uint outpos) {
	int t0, t1, t2, t3, t4;
	uint m0, m1, m2, m3, m4;

	split_high_low(b0, bitcount, t0, m0);
	split_high_low(b1, bitcount, t1, m1);
	split_high_low(b2, bitcount, t2, m2);
	split_high_low(b3, bitcount, t3, m3);
	split_high_low(b4, bitcount, t4, m4);

	uint packhigh = uint(integer_from_trits[((((t4 * 81) + (t3 * 27)) + (t2 * 9)) + (t1 * 3)) + t0]);

	orbits8_ptr(outputs, outpos, m0, bitcount);
	orbits8_ptr(outputs, outpos, packhigh & 3u, 2);

	orbits8_ptr(outputs, outpos, m1, bitcount);
	orbits8_ptr(outputs, outpos, (packhigh >> 2u) & 3u, 2);

	orbits8_ptr(outputs, outpos, m2, bitcount);
	orbits8_ptr(outputs, outpos, (packhigh >> 4u) & 1u, 1);

	orbits8_ptr(outputs, outpos, m3, bitcount);
	orbits8_ptr(outputs, outpos, (packhigh >> 5u) & 3u, 2);

	orbits8_ptr(outputs, outpos, m4, bitcount);
	orbits8_ptr(outputs, outpos, (packhigh >> 7u) & 1u, 1);
}

/**
 * Encode a group of 3 numbers using quints and bits.
 */
void encode_quints(int bitcount, uint b0, uint b1, uint b2, inout uvec4 outputs, inout uint outpos) {
	int q0, q1, q2;
	uint m0, m1, m2;

	split_high_low(b0, bitcount, q0, m0);
	split_high_low(b1, bitcount, q1, m1);
	split_high_low(b2, bitcount, q2, m2);

	uint packhigh = uint(integer_from_quints[((q2 * 25) + (q1 * 5)) + q0]);

	orbits8_ptr(outputs, outpos, m0, bitcount);
	orbits8_ptr(outputs, outpos, packhigh & 7u, 3);

	orbits8_ptr(outputs, outpos, m1, bitcount);
	orbits8_ptr(outputs, outpos, (packhigh >> 3u) & 3u, 2);

	orbits8_ptr(outputs, outpos, m2, bitcount);
	orbits8_ptr(outputs, outpos, (packhigh >> 5u) & 3u, 2);
}

void bise_endpoints(uint numbers[8], int range, inout uvec4 outputs) {
	uint bitpos = 0u;
	int bits = bits_trits_quints_table[(range * 3) + 0];
	uint trits = uint(bits_trits_quints_table[(range * 3) + 1]);
	uint quints = uint(bits_trits_quints_table[(range * 3) + 2]);

#ifdef HAS_ALPHA
	int count = 8;
#else
	int count = 6;
#endif

	if (trits == 1u) {
		encode_trits(bits, numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], outputs, bitpos);
		encode_trits(bits, numbers[5], numbers[6], numbers[7], 0, 0, outputs, bitpos);
		bitpos = (((8u + (5u * bits)) * uint(count)) + 4u) / 5u;
	} else if (quints == 1u) {
		encode_quints(bits, numbers[0], numbers[1], numbers[2], outputs, bitpos);
		encode_quints(bits, numbers[3], numbers[4], numbers[5], outputs, bitpos);
		encode_quints(bits, numbers[6], numbers[7], 0, outputs, bitpos);
		bitpos = (((7u + (3u * bits)) * uint(count)) + 2u) / 3u;
	} else {
		for (int i = 0; i < count; ++i) {
			orbits8_ptr(outputs, bitpos, numbers[i], bits);
		}
	}
}

void bise_weights(uint numbers[X_GRIDS * Y_GRIDS], int range, inout uvec4 outputs) {
	uint bitpos = 0u;
	int bits = bits_trits_quints_table[(range * 3) + 0];
	uint trits = uint(bits_trits_quints_table[(range * 3) + 1]);
	uint quints = uint(bits_trits_quints_table[(range * 3) + 2]);
	if (trits == 1u) {
		encode_trits(bits, numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], outputs, bitpos);
		encode_trits(bits, numbers[5], numbers[6], numbers[7], numbers[8], numbers[9], outputs, bitpos);
		encode_trits(bits, numbers[10], numbers[11], numbers[12], numbers[13], numbers[14], outputs, bitpos);
		encode_trits(bits, numbers[15], 0, 0, 0, 0, outputs, bitpos);
		bitpos = (((8u + (5u * bits)) * 16u) + 4u) / 5u;
	} else if (quints == 1u) {
		encode_quints(bits, numbers[0], numbers[1], numbers[2], outputs, bitpos);
		encode_quints(bits, numbers[3], numbers[4], numbers[5], outputs, bitpos);
		encode_quints(bits, numbers[6], numbers[7], numbers[8], outputs, bitpos);
		encode_quints(bits, numbers[9], numbers[10], numbers[11], outputs, bitpos);
		encode_quints(bits, numbers[12], numbers[13], numbers[14], outputs, bitpos);
		encode_quints(bits, numbers[15], 0, 0, outputs, bitpos);
		bitpos = (((7u + (3u * bits)) * 16u) + 2u) / 3u;
	} else {
		for (int i = 0; i < X_GRIDS * Y_GRIDS; ++i) {
			orbits8_ptr(outputs, bitpos, numbers[i], bits);
		}
	}
}

// END OF ASTC_IntegerSequenceEncoding.hlsl

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// calc the dominant axis
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void swap(inout vec4 lhs, inout vec4 rhs) {
	vec4 tmp = lhs;
	lhs = rhs;
	rhs = tmp;
}

vec4 eigen_vector(mat4 m) {
	// calc the max eigen value by iteration
	vec4 v = vec4(0.26726f, 0.80178f, 0.53452f, 0.0f);

	for (int i = 0; i < 8; i++) {
		v = m * v;
		if (length(v) < SMALL_VALUE) {
			return v;
		}
		v = normalize(v * m);
	}
	return v;
}

void find_min_max(TEXEL_TYPE texels[BLOCK_SIZE], vec4 pt_mean, vec4 vec_k, out vec4 e0, out vec4 e1) {
	float a = 1e31f;
	float b = -1e31f;
	for (int i = 0; i < BLOCK_SIZE; i++) {
		vec4 texel = TEXEL_READ(texels, i) - pt_mean;
		float t = dot(texel, vec_k);
		a = min(a, t);
		b = max(b, t);
	}

	e0 = clamp((vec_k * a) + pt_mean, vec4(0.0), vec4(255.0));
	e1 = clamp((vec_k * b) + pt_mean, vec4(0.0), vec4(255.0));

	// if the direction-vector ends up pointing from light to dark, FLIP IT!
	// this will make the first endpoint the darkest one.
	vec4 e0u = roundEven(e0);
	vec4 e1u = roundEven(e1);
	if (((e0u.x + e0u.y) + e0u.z) > ((e1u.x + e1u.y) + e1u.z)) {
		swap(e0, e1);
	}

#ifndef HAS_ALPHA
	e0.w = 255.0;
	e1.w = 255.0;
#endif
}

void principal_component_analysis(TEXEL_TYPE texels[BLOCK_SIZE], out vec4 e0, out vec4 e1) {
	int i = 0;
	vec4 pt_mean = vec4(0.0);
	for (; i < BLOCK_SIZE; i++) {
#ifdef TEXEL_VEC4
		pt_mean += texels[i];
#else
		pt_mean += unpackUnorm4x8(texels[i]);
#endif
	}
	pt_mean /= vec4(BLOCK_SIZE);

	mat4 cov = mat4(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
	for (int k = 0; k < BLOCK_SIZE; k++) {
#ifdef TEXEL_VEC4
		vec4 texel = texels[k] - pt_mean;
#else
		vec4 texel = unpackUnorm4x8(texels[k]) - pt_mean;
#endif
		for (i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				cov[i][j] += (texel[i] * texel[j]);
			}
		}
	}
#ifndef TEXEL_VEC4
	pt_mean *= 255.0;
	cov *= 255.0 * 255.0;
#endif
	cov /= BLOCK_SIZE - 1;

	vec4 vec_k = eigen_vector(cov);

	find_min_max(texels, pt_mean, vec_k, e0, e1);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize & unquantize the endpoints
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// for QUANT_256 quantization
void encode_color(uint qm_index, vec4 e0, vec4 e1, inout uint endpoint_quantized[8]) {
	uvec4 e0q = uvec4(roundEven(e0));
	uvec4 e1q = uvec4(roundEven(e1));
	endpoint_quantized[0] = e0q.x;
	endpoint_quantized[1] = e1q.x;
	endpoint_quantized[2] = e0q.y;
	endpoint_quantized[3] = e1q.y;
	endpoint_quantized[4] = e0q.z;
	endpoint_quantized[5] = e1q.z;
	endpoint_quantized[6] = e0q.w;
	endpoint_quantized[7] = e1q.w;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// calculate quantized weights
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
uint quantize_weight(uint weight_range, float weight) {
	uint q = uint(roundEven(weight * float(weight_range)));
	return clamp(q, 0u, weight_range);
}

#ifdef BLOCK_6X6
const uvec4 idx_grids[16] = uvec4[](
		uvec4(0u, 1u, 6u, 7u),
		uvec4(1u, 2u, 7u, 8u),
		uvec4(3u, 4u, 9u, 10u),
		uvec4(4u, 5u, 10u, 11u),
		uvec4(6u, 7u, 12u, 13u),
		uvec4(7u, 8u, 13u, 14u),
		uvec4(9u, 10u, 15u, 16u),
		uvec4(10u, 11u, 16u, 17u),
		uvec4(18u, 19u, 24u, 25u),
		uvec4(19u, 20u, 25u, 26u),
		uvec4(21u, 22u, 27u, 28u),
		uvec4(22u, 23u, 28u, 29u),
		uvec4(24u, 25u, 30u, 31u),
		uvec4(25u, 26u, 31u, 32u),
		uvec4(27u, 28u, 33u, 34u),
		uvec4(28u, 29u, 34u, 35u));

const vec4 wt_grids[16] = vec4[](
		vec4(0.444, 0.222, 0.222, 0.111),
		vec4(0.222, 0.444, 0.111, 0.222),
		vec4(0.444, 0.222, 0.222, 0.111),
		vec4(0.222, 0.444, 0.111, 0.222),
		vec4(0.222, 0.111, 0.444, 0.222),
		vec4(0.111, 0.222, 0.222, 0.444),
		vec4(0.222, 0.111, 0.444, 0.222),
		vec4(0.111, 0.222, 0.222, 0.444),
		vec4(0.444, 0.222, 0.222, 0.111),
		vec4(0.222, 0.444, 0.111, 0.222),
		vec4(0.444, 0.222, 0.222, 0.111),
		vec4(0.222, 0.444, 0.111, 0.222),
		vec4(0.222, 0.111, 0.444, 0.222),
		vec4(0.111, 0.222, 0.222, 0.444),
		vec4(0.222, 0.111, 0.444, 0.222),
		vec4(0.111, 0.222, 0.222, 0.444));

vec4 sample_texel(TEXEL_TYPE texels[BLOCK_SIZE], uvec4 index, vec4 coff) {
	vec4 sum = TEXEL_READ(texels, index.x) * coff.x;
	sum += TEXEL_READ(texels, index.y) * coff.y;
	sum += TEXEL_READ(texels, index.z) * coff.z;
	sum += TEXEL_READ(texels, index.w) * coff.w;
	return sum;
}

#endif

void calculate_normal_weights(TEXEL_TYPE texels[BLOCK_SIZE], vec4 ep0, vec4 ep1, inout float projw[X_GRIDS * Y_GRIDS]) {
	int i = 0;
	vec4 vec_k = ep1 - ep0;
	if (length(vec_k) < SMALL_VALUE) {
		i = 0;
		for (; i < X_GRIDS * Y_GRIDS; i++) {
			projw[i] = 0.0;
		}
	} else {
		vec_k = normalize(vec_k);
		float minw = 1e31f;
		float maxw = -1e31f;
#ifdef BLOCK_6X6

		/* bilinear interpolation: GirdSize is 4,BlockSize is 6

	0     1     2     3     4     5
|-----|-----|-----|-----|-----|-----|
|--------|--------|--------|--------|
	0        1        2        3
		*/
		for (i = 0; i < X_GRIDS * Y_GRIDS; ++i) {
			vec4 sum = sample_texel(texels, idx_grids[i], wt_grids[i]);
			float w = dot(vec_k, sum - ep0);
			minw = min(w, minw);
			maxw = max(w, maxw);
			projw[i] = w;
		}
#else
		// ensure "X_GRIDS * Y_GRIDS == BLOCK_SIZE"
		for (i = 0; i < X_GRIDS * Y_GRIDS; i++) {
			vec4 texel = TEXEL_READ(texels, i);
			float w = dot(vec_k, texel - ep0);
			minw = min(w, minw);
			maxw = max(w, maxw);
			projw[i] = w;
		}
#endif
		float invlen = maxw - minw;
		invlen = max(SMALL_VALUE, invlen);
		invlen = 1.0 / invlen;
		i = 0;
		for (; i < X_GRIDS * Y_GRIDS; i++) {
			projw[i] = (projw[i] - minw) * invlen;
		}
	}
}

void quantize_weights(
		float projw[X_GRIDS * Y_GRIDS],
		uint weight_range,
		inout uint weights[X_GRIDS * Y_GRIDS]) {
	for (int i = 0; i < X_GRIDS * Y_GRIDS; i++) {
		weights[i] = quantize_weight(weight_range, projw[i]);
	}
}

void calculate_quantized_weights(TEXEL_TYPE texels[BLOCK_SIZE], uint weight_range, vec4 ep0, vec4 ep1, out uint weights[X_GRIDS * Y_GRIDS]) {
	float projw[X_GRIDS * Y_GRIDS];
	calculate_normal_weights(texels, ep0, ep1, projw);
	quantize_weights(projw, weight_range, weights);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// encode single partition
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// candidate blockmode uvec4(weights quantmethod, endpoints quantmethod, weights range, endpoints quantmethod index of table)

uvec4 assemble_block(uint blockmode, uint color_endpoint_mode, uint partition_count, uint partition_index, uvec4 ep_ise, uvec4 wt_ise) {
	uvec4 phy_blk = uvec4(0u);

	// weights ise
	phy_blk.w = bitfieldReverse(wt_ise.x);
	phy_blk.z = bitfieldReverse(wt_ise.y);
	phy_blk.y = bitfieldReverse(wt_ise.z);

	// blockmode & partition count
	phy_blk.x = blockmode; // blockmode is 11 bit

	// cem: color_endpoint_mode is 4 bit
	phy_blk.x = bitfieldInsert(phy_blk.x, color_endpoint_mode, 13, 4);

	// endpoints start from ( multi_part ? bits 29 : bits 17 )
	phy_blk.x = bitfieldInsert(phy_blk.x, ep_ise.x, 17, 15);
	phy_blk.y = bitfieldExtract(ep_ise.x, 15, 17);
	phy_blk.y = bitfieldInsert(phy_blk.y, ep_ise.y, 17, 15);
	phy_blk.z |= bitfieldExtract(ep_ise.y, 15, 17);
	return phy_blk;
}

uint assemble_blockmode(uint weight_quantmethod) {
	/*
		the first row of "Table C.2.8 - 2D Block Mode Layout".
		------------------------------------------------------------------------
		10  9   8   7   6   5   4   3   2   1   0   Width Height Notes
		------------------------------------------------------------------------
		D   H     B       A     R0  0   0   R2  R1  B + 4   A + 2
	*/
	uint a = (Y_GRIDS - 2) & 0x3;
	uint b = (X_GRIDS - 4) & 0x3;

	uint d = 0u; // dual plane

	// more details from "Table C.2.7 - Weight Range Encodings"
	uint h = (weight_quantmethod < 6u) ? 0u : 1u; // "a precision bit H"
	uint r = (weight_quantmethod % 6u) + 2u; // "The weight ranges are encoded using a 3 bit value R"

	// block mode
	uint blockmode = bitfieldExtract(r, 1, 2);
	blockmode = bitfieldInsert(blockmode, r, 4, 1);
	blockmode = bitfieldInsert(blockmode, a, 5, 2);
	blockmode = bitfieldInsert(blockmode, b, 7, 2);
	blockmode = bitfieldInsert(blockmode, h, 9, 1);
	blockmode |= d << 10u;
	return blockmode;
}

uvec4 endpoint_ise(uint colorquant_index, vec4 ep0, vec4 ep1, uint endpoint_quantmethod) {
	// encode endpoints
	uint ep_quantized[8];
	encode_color(colorquant_index, ep0, ep1, ep_quantized);
#ifndef HAS_ALPHA
	ep_quantized[6] = 0;
	ep_quantized[7] = 0;
#endif

	// endpoints quantized ise encode
	uvec4 ep_ise = uvec4(0u);
	bise_endpoints(ep_quantized, int(endpoint_quantmethod), ep_ise);
	return ep_ise;
}

uvec4 weight_ise(TEXEL_TYPE texels[BLOCK_SIZE], uint weight_range, vec4 ep0, vec4 ep1, uint weight_quantmethod) {
	// encode weights
	uint wt_quantized[X_GRIDS * Y_GRIDS];
	calculate_quantized_weights(texels, weight_range, ep0, ep1, wt_quantized);
	for (int i = 0; i < X_GRIDS * Y_GRIDS; i++) {
		int w = int(weight_quantmethod * WEIGHT_QUANTIZE_NUM + wt_quantized[i]);
		wt_quantized[i] = uint(scramble_table[w]);
	}

	// weights quantized ise encode
	uvec4 wt_ise = uvec4(0);
	bise_weights(wt_quantized, int(weight_quantmethod), wt_ise);
	return wt_ise;
}

uvec4 encode_block(TEXEL_TYPE texels[BLOCK_SIZE]) {
	vec4 ep0, ep1;
	principal_component_analysis(texels, ep0, ep1);

#ifdef HAS_ALPHA
	uvec4 best_blockmode = uvec4(QUANT_6, QUANT_256, 6, 7);
#else
	uvec4 best_blockmode = uvec4(QUANT_12, QUANT_256, 12, 7);
#endif

	uint weight_quantmethod = best_blockmode.x;
	uint endpoint_quantmethod = best_blockmode.y;
	uint weight_range = best_blockmode.z;
	uint colorquant_index = best_blockmode.w;

	uint blockmode = assemble_blockmode(weight_quantmethod);

	uvec4 ep_ise = endpoint_ise(colorquant_index, ep0, ep1, endpoint_quantmethod);

	uvec4 wt_ise = weight_ise(texels, weight_range - 1u, ep0, ep1, weight_quantmethod);

// assemble to astcblock
#ifdef HAS_ALPHA
	uint color_endpoint_mode = CEM_LDR_RGBA_DIRECT;
#else
	uint color_endpoint_mode = CEM_LDR_RGB_DIRECT;
#endif

	return assemble_block(blockmode, color_endpoint_mode, 1, 0, ep_ise, wt_ise);
}

void main() {
	uint blockID = gl_GlobalInvocationID.y * uint(params.group_num_x) * THREAD_NUM_X + gl_GlobalInvocationID.x;
	TEXEL_TYPE texels[BLOCK_SIZE];
	uvec2 blockPos;
	for (int k = 0; k < BLOCK_SIZE; k++) {
		blockPos.y = blockID / params.group_num_x;
		blockPos.x = blockID - blockPos.y * params.group_num_x;
		uint y = uint(k / DIM);
		uint x = uint(k) - y * DIM;
		uvec2 pixelPos = blockPos * uvec2(DIM) + uvec2(x, y);
		vec4 texel = texelFetch(InTexture, ivec2(pixelPos), 0);
#ifdef IS_NORMALMAP
		texel.b = 1.0f;
		texel.a = 1.0f;
#endif
#ifdef TEXEL_VEC4
		texels[k] = texel * 255.0;
#else
		texels[k] = packUnorm4x8(texel);
#endif
	}
	OutBuffer_1._data[blockID] = encode_block(texels);
}

#version 450

layout(constant_id = 200) const float sp_float = 3.1415926;
layout(constant_id = 201) const int sp_int = 10;
layout(constant_id = 202) const uint sp_uint = 100;
layout(constant_id = 203) const int sp_sint = -10;
layout(constant_id = 204) const double sp_double = 2.718281828459;

//
// Scalars
//

// float <-> double conversion
const float float_from_double = float(sp_double);
const double double_from_float = double(sp_float);

// uint/int <-> bool conversion
const bool bool_from_int = bool(sp_int);
const bool bool_from_uint = bool(sp_uint);
const int int_from_bool = int(bool_from_int);
const uint uint_from_bool = uint(bool_from_int);

// uint <-> int
const uint sp_uint_from_sint = uint(sp_sint);
const int sp_sint_from_uint = int(sp_uint);

// Negate and Not
const int negate_int = -sp_int;
const int not_int = ~sp_int;

// Add and Subtract
const int sp_int_add_two = sp_int + 2;
const int sp_int_add_two_sub_three = sp_int + 2 - 3;
const int sp_int_add_two_sub_four = sp_int_add_two - 4;

// Mul, Div and Rem
const int sp_sint_mul_two = sp_sint * 2;
const uint sp_uint_mul_two = sp_uint * 2;
const int sp_sint_mul_two_div_five = sp_sint_mul_two / 5;
const uint sp_uint_mul_two_div_five = sp_uint_mul_two / 5;
const int sp_sint_rem_four = sp_sint % 4;
const uint sp_uint_rem_four = sp_uint % 4;
const int sp_sint_mul_three_div_five = sp_sint * 3 / 5;

// Shift
const int sp_sint_shift_right_arithmetic = sp_sint >> 10;
const uint sp_uint_shift_right_arithmetic = sp_uint >> 20;
const int sp_sint_shift_left = sp_sint << 1;
const uint sp_uint_shift_left = sp_uint << 2;

// Bitwise And, Or, Xor
const int sp_sint_or_256 = sp_sint | 0x100;
const uint sp_uint_xor_512 = sp_uint ^ 0x200;

/* // Scalar comparison */
const bool sp_int_lt_sp_sint = sp_int < sp_sint;
const bool sp_uint_equal_sp_uint = sp_uint == sp_uint;
const bool sp_int_gt_sp_sint = sp_int > sp_sint;

//
// Vectors
//
const ivec4 iv = ivec4(20, 30, sp_int, sp_int);
const uvec4 uv = uvec4(sp_uint, sp_uint, -1, -2);
//const vec4 fv = vec4(sp_float, 1.25, sp_float, 1.25);

// uint/int <-> bool conversion
const bvec4 bv_from_iv = bvec4(iv);
const bvec4 bv_from_uv = bvec4(uv);
const ivec4 iv_from_bv = ivec4(bv_from_iv);
const uvec4 uv_from_bv = uvec4(bv_from_iv);

// uint <-> int
const uvec4 uv_from_iv = uvec4(iv);
const ivec4 iv_from_uv = ivec4(uv);

// Negate and Not
const ivec4 not_iv = ~iv;
const ivec4 negate_iv = -iv;

// Add and Subtract
const ivec4 iv_add_two = iv + 2;
const ivec4 iv_add_two_sub_three = iv + 2 - 3;
const ivec4 iv_add_two_sub_four = iv_add_two_sub_three - 4;

// Mul, Div and Rem
const ivec4 iv_mul_two = iv * 2;
const ivec4 iv_mul_two_div_five = iv_mul_two / 5;
const ivec4 iv_rem_four = iv % 4;

// Shift
const ivec4 iv_shift_right_arithmetic = iv >> 10;
const ivec4 iv_shift_left = iv << 2;

// Bitwise And, Or, Xor
const ivec4 iv_or_1024 = iv | 0x400;
const uvec4 uv_xor_2048 = uv ^ 0x800;

// Swizzles
const int iv_x = iv.x;
const ivec2 iv_yx = iv.yx;
const ivec3 iv_zyx = iv.zyx;
const ivec4 iv_yzxw = iv.yzxw;

int non_const_array_size_from_spec_const() {
    int array[sp_int + 2];
    for (int i = 0; i < sp_int + 2; i++) {
        array[i] = 1023;
    }
    return array[sp_int + 1];
}

// ternary
layout(constant_id = 210) const int a = 4;
layout(constant_id = 211) const int b = 6;
layout(constant_id = 212) const bool c = true;
int ternayArray1[a > b ? a : b];
const int t1 = c ? 13 : 17;
const int t2 = c ? a : 17;
const int t3 = true ? a : 17;
const int t4 = a > b ? 13 + a : 17 * b;
const vec2 v2 = !c ? vec2(1.0) : vec2(2.0);

void main() {}

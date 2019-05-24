#version 450

// constant_id specified scalar spec constants
layout(constant_id = 200) const int spec_int = 3;
layout(constant_id = 201) const float spec_float = 3.14;
layout(constant_id = 202) const
    double spec_double = 3.1415926535897932384626433832795;
layout(constant_id = 203) const bool spec_bool = true;

// const float cast_spec_float = float(spec_float);

// Flat struct
struct flat_struct {
    int i;
    float f;
    double d;
    bool b;
};

// Nesting struct
struct nesting_struct {
    flat_struct nested;
    vec4 v;
    int i;
};

// Expect OpSpecConstantComposite
// Flat struct initializer
//const flat_struct spec_flat_struct_all_spec = {spec_int, spec_float,
//                                               spec_double, spec_bool};
//const flat_struct spec_flat_struct_partial_spec = {30, 30.14, spec_double,
//                                                   spec_bool};

// Nesting struct initializer
//const nesting_struct nesting_struct_ctor = {
//    {spec_int, spec_float, spec_double, false},
//    vec4(0.1, 0.1, 0.1, 0.1),
//    spec_int};

// Vector constructor
//const vec4 spec_vec4_all_spec =
//    vec4(spec_float, spec_float, spec_float, spec_float);
//const vec4 spec_vec4_partial_spec =
//    vec4(spec_float, spec_float, 300.14, 300.14);
//const vec4 spec_vec4_from_one_scalar = vec4(spec_float);

// Matrix constructor
//const mat2x3 spec_mat2x3 = mat2x3(spec_float, spec_float, spec_float, 1.1, 2.2, 3.3);
//const mat2x3 spec_mat2x3_from_one_scalar = mat2x3(spec_float);

// Struct nesting constructor
//const nesting_struct spec_nesting_struct_all_spec = {
//    spec_flat_struct_all_spec, spec_vec4_all_spec, spec_int};
//const nesting_struct spec_nesting_struct_partial_spec = {
//    spec_flat_struct_partial_spec, spec_vec4_partial_spec, 3000};

//const float spec_float_array[5] = {spec_float, spec_float, 1.0, 2.0, 3.0};
//const int spec_int_array[5] = {spec_int, spec_int, 1, 2, 30};

// global_vec4_array_with_spec_length is not a spec constant, but its array
// size is. When calling global_vec4_array_with_spec_length.length(), A
// TIntermSymbol Node should be returned, instead of a TIntermConstantUnion
// node which represents a known constant value.
in vec4 global_vec4_array_with_spec_length[spec_int];

out vec4 color;

void refer_primary_spec_const() {
    if (spec_bool) color *= spec_int;
}

void refer_composite_spec_const() {
    //color += spec_vec4_all_spec;
    //color -= spec_vec4_partial_spec;
}

void refer_copmosite_dot_dereference() {
    //color *= spec_nesting_struct_all_spec.i;
    //color += spec_vec4_all_spec.x;
}

void refer_composite_bracket_dereference() {
    //color -= spec_float_array[1];
    //color /= spec_int_array[spec_int_array[spec_int]];
}

int refer_spec_const_array_length() {
    int len = global_vec4_array_with_spec_length.length();
    return len;
}

void declare_spec_const_in_func() {
    //const nesting_struct spec_const_declared_in_func = {
    //    spec_flat_struct_partial_spec, spec_vec4_partial_spec, 10};
    //color /= spec_const_declared_in_func.i;
}

void main() {}

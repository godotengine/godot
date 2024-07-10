// Copyright © 2024 Cory Petkovsek, Roope Palmroos, and Contributors.

R"(

//INSERT: TEXTURE_SAMPLERS_LINEAR
uniform sampler2DArray _color_maps : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform sampler2DArray _texture_array_albedo : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2DArray _texture_array_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2D noise_texture : source_color, filter_linear_mipmap_anisotropic, repeat_enable;

//INSERT: TEXTURE_SAMPLERS_NEAREST
uniform sampler2DArray _color_maps : source_color, filter_nearest_mipmap_anisotropic, repeat_disable;
uniform sampler2DArray _texture_array_albedo : source_color, filter_nearest_mipmap_anisotropic, repeat_enable;
uniform sampler2DArray _texture_array_normal : hint_normal, filter_nearest_mipmap_anisotropic, repeat_enable;
uniform sampler2D noise_texture : source_color, filter_nearest_mipmap_anisotropic, repeat_enable;

)"
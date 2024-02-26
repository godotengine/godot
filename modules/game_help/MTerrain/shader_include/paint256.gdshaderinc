//Include this before Vertex function
vec4 get_indices256(in sampler2D _tex,out vec2 _weight){
	_weight = vec2(textureSize(_tex,0))*(region_uv - region_b);
	ivec2 ipx_pos = ivec2(_weight);
	_weight = fract(_weight);
	vec4 o;
	o.x = texelFetch(_tex,ipx_pos,0).r*255.0;
	o.y = texelFetch(_tex,ipx_pos+ivec2(1,0),0).r*255.0;
	o.z = texelFetch(_tex,ipx_pos+ivec2(0,1),0).r*255.0;
	o.w = texelFetch(_tex,ipx_pos+ivec2(1,1),0).r*255.0;
	return o;
}
vec4 get_texture256(in sampler2DArray _tex_arr,in lowp vec4 _indices,in vec2 _weight,in vec2 _uv){
	vec4 a00 = texture(_tex_arr,vec3(_uv,_indices.x));
	vec4 a01 = texture(_tex_arr,vec3(_uv,_indices.y));
	vec4 a10 = texture(_tex_arr,vec3(_uv,_indices.z));
	vec4 a11 = texture(_tex_arr,vec3(_uv,_indices.w));
	vec4 t = mix(a00,a01,_weight.x);
	vec4 b = mix(a10,a11,_weight.x);
	return mix(t,b,_weight.y);
}
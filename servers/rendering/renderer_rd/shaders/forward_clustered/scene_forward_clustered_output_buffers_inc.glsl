#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MATERIAL

layout(location = 0) out vec4 albedo_output_buffer;
layout(location = 1) out vec4 normal_output_buffer;
layout(location = 2) out vec4 orm_output_buffer;
layout(location = 3) out vec4 emission_output_buffer;
layout(location = 4) out float depth_output_buffer;

#endif // MODE_RENDER_MATERIAL

#ifdef MODE_RENDER_NORMAL_ROUGHNESS
layout(location = 0) out vec4 normal_roughness_output_buffer;

#ifdef MODE_RENDER_VOXEL_GI
layout(location = 1) out uvec2 voxel_gi_buffer;
#endif

#endif //MODE_RENDER_NORMAL
#else // RENDER DEPTH

#ifdef MODE_SEPARATE_SPECULAR

layout(location = 0) out vec4 diffuse_buffer; //diffuse (rgb) and roughness
layout(location = 1) out vec4 specular_buffer; //specular and SSS (subsurface scatter)
#else

layout(location = 0) out vec4 frag_color;
#endif // MODE_SEPARATE_SPECULAR

#endif // RENDER DEPTH

#ifdef MOTION_VECTORS
layout(location = 2) out vec2 motion_vector;
#endif

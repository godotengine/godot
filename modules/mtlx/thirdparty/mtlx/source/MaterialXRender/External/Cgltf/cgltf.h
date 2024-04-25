/**
 * cgltf - a single-file glTF 2.0 parser written in C99.
 *
 * Version: 1.13
 *
 * Website: https://github.com/jkuhlmann/cgltf
 *
 * Distributed under the MIT License, see notice at the end of this file.
 *
 * Building:
 * Include this file where you need the struct and function
 * declarations. Have exactly one source file where you define
 * `CGLTF_IMPLEMENTATION` before including this file to get the
 * function definitions.
 *
 * Reference:
 * `cgltf_result cgltf_parse(const cgltf_options*, const void*,
 * cgltf_size, cgltf_data**)` parses both glTF and GLB data. If
 * this function returns `cgltf_result_success`, you have to call
 * `cgltf_free()` on the created `cgltf_data*` variable.
 * Note that contents of external files for buffers and images are not
 * automatically loaded. You'll need to read these files yourself using
 * URIs in the `cgltf_data` structure.
 *
 * `cgltf_options` is the struct passed to `cgltf_parse()` to control
 * parts of the parsing process. You can use it to force the file type
 * and provide memory allocation as well as file operation callbacks.
 * Should be zero-initialized to trigger default behavior.
 *
 * `cgltf_data` is the struct allocated and filled by `cgltf_parse()`.
 * It generally mirrors the glTF format as described by the spec (see
 * https://github.com/KhronosGroup/glTF/tree/master/specification/2.0).
 *
 * `void cgltf_free(cgltf_data*)` frees the allocated `cgltf_data`
 * variable.
 *
 * `cgltf_result cgltf_load_buffers(const cgltf_options*, cgltf_data*,
 * const char* gltf_path)` can be optionally called to open and read buffer
 * files using the `FILE*` APIs. The `gltf_path` argument is the path to
 * the original glTF file, which allows the parser to resolve the path to
 * buffer files.
 *
 * `cgltf_result cgltf_load_buffer_base64(const cgltf_options* options,
 * cgltf_size size, const char* base64, void** out_data)` decodes
 * base64-encoded data content. Used internally by `cgltf_load_buffers()`.
 * This is useful when decoding data URIs in images.
 *
 * `cgltf_result cgltf_parse_file(const cgltf_options* options, const
 * char* path, cgltf_data** out_data)` can be used to open the given
 * file using `FILE*` APIs and parse the data using `cgltf_parse()`.
 *
 * `cgltf_result cgltf_validate(cgltf_data*)` can be used to do additional
 * checks to make sure the parsed glTF data is valid.
 *
 * `cgltf_node_transform_local` converts the translation / rotation / scale properties of a node
 * into a mat4.
 *
 * `cgltf_node_transform_world` calls `cgltf_node_transform_local` on every ancestor in order
 * to compute the root-to-node transformation.
 *
 * `cgltf_accessor_unpack_floats` reads in the data from an accessor, applies sparse data (if any),
 * and converts them to floating point. Assumes that `cgltf_load_buffers` has already been called.
 * By passing null for the output pointer, users can find out how many floats are required in the
 * output buffer.
 *
 * `cgltf_num_components` is a tiny utility that tells you the dimensionality of
 * a certain accessor type. This can be used before `cgltf_accessor_unpack_floats` to help allocate
 * the necessary amount of memory.
 *
 * `cgltf_accessor_read_float` reads a certain element from a non-sparse accessor and converts it to
 * floating point, assuming that `cgltf_load_buffers` has already been called. The passed-in element
 * size is the number of floats in the output buffer, which should be in the range [1, 16]. Returns
 * false if the passed-in element_size is too small, or if the accessor is sparse.
 *
 * `cgltf_accessor_read_uint` is similar to its floating-point counterpart, but limited to reading
 * vector types and does not support matrix types. The passed-in element size is the number of uints
 * in the output buffer, which should be in the range [1, 4]. Returns false if the passed-in 
 * element_size is too small, or if the accessor is sparse.
 *
 * `cgltf_accessor_read_index` is similar to its floating-point counterpart, but it returns size_t
 * and only works with single-component data types.
 *
 * `cgltf_result cgltf_copy_extras_json(const cgltf_data*, const cgltf_extras*,
 * char* dest, cgltf_size* dest_size)` allows users to retrieve the "extras" data that
 * can be attached to many glTF objects (which can be arbitrary JSON data). The
 * `cgltf_extras` struct stores the offsets of the start and end of the extras JSON data
 * as it appears in the complete glTF JSON data. This function copies the extras data
 * into the provided buffer. If `dest` is NULL, the length of the data is written into
 * `dest_size`. You can then parse this data using your own JSON parser
 * or, if you've included the cgltf implementation using the integrated JSMN JSON parser.
 */
#ifndef CGLTF_H_INCLUDED__
#define CGLTF_H_INCLUDED__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t cgltf_size;
typedef long long int cgltf_ssize;
typedef float cgltf_float;
typedef int cgltf_int;
typedef unsigned int cgltf_uint;
typedef int cgltf_bool;

typedef enum cgltf_file_type
{
	cgltf_file_type_invalid,
	cgltf_file_type_gltf,
	cgltf_file_type_glb,
	cgltf_file_type_max_enum
} cgltf_file_type;

typedef enum cgltf_result
{
	cgltf_result_success,
	cgltf_result_data_too_short,
	cgltf_result_unknown_format,
	cgltf_result_invalid_json,
	cgltf_result_invalid_gltf,
	cgltf_result_invalid_options,
	cgltf_result_file_not_found,
	cgltf_result_io_error,
	cgltf_result_out_of_memory,
	cgltf_result_legacy_gltf,
    cgltf_result_max_enum
} cgltf_result;

typedef struct cgltf_memory_options
{
	void* (*alloc_func)(void* user, cgltf_size size);
	void (*free_func) (void* user, void* ptr);
	void* user_data;
} cgltf_memory_options;

typedef struct cgltf_file_options
{
	cgltf_result(*read)(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, const char* path, cgltf_size* size, void** data);
	void (*release)(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data);
	void* user_data;
} cgltf_file_options;

typedef struct cgltf_options
{
	cgltf_file_type type; /* invalid == auto detect */
	cgltf_size json_token_count; /* 0 == auto */
	cgltf_memory_options memory;
	cgltf_file_options file;
} cgltf_options;

typedef enum cgltf_buffer_view_type
{
	cgltf_buffer_view_type_invalid,
	cgltf_buffer_view_type_indices,
	cgltf_buffer_view_type_vertices,
	cgltf_buffer_view_type_max_enum
} cgltf_buffer_view_type;

typedef enum cgltf_attribute_type
{
	cgltf_attribute_type_invalid,
	cgltf_attribute_type_position,
	cgltf_attribute_type_normal,
	cgltf_attribute_type_tangent,
	cgltf_attribute_type_texcoord,
	cgltf_attribute_type_color,
	cgltf_attribute_type_joints,
	cgltf_attribute_type_weights,
	cgltf_attribute_type_custom,
	cgltf_attribute_type_max_enum
} cgltf_attribute_type;

typedef enum cgltf_component_type
{
	cgltf_component_type_invalid,
	cgltf_component_type_r_8, /* BYTE */
	cgltf_component_type_r_8u, /* UNSIGNED_BYTE */
	cgltf_component_type_r_16, /* SHORT */
	cgltf_component_type_r_16u, /* UNSIGNED_SHORT */
	cgltf_component_type_r_32u, /* UNSIGNED_INT */
	cgltf_component_type_r_32f, /* FLOAT */
    cgltf_component_type_max_enum
} cgltf_component_type;

typedef enum cgltf_type
{
	cgltf_type_invalid,
	cgltf_type_scalar,
	cgltf_type_vec2,
	cgltf_type_vec3,
	cgltf_type_vec4,
	cgltf_type_mat2,
	cgltf_type_mat3,
	cgltf_type_mat4,
	cgltf_type_max_enum
} cgltf_type;

typedef enum cgltf_primitive_type
{
	cgltf_primitive_type_points,
	cgltf_primitive_type_lines,
	cgltf_primitive_type_line_loop,
	cgltf_primitive_type_line_strip,
	cgltf_primitive_type_triangles,
	cgltf_primitive_type_triangle_strip,
	cgltf_primitive_type_triangle_fan,
	cgltf_primitive_type_max_enum
} cgltf_primitive_type;

typedef enum cgltf_alpha_mode
{
	cgltf_alpha_mode_opaque,
	cgltf_alpha_mode_mask,
	cgltf_alpha_mode_blend,
	cgltf_alpha_mode_max_enum
} cgltf_alpha_mode;

typedef enum cgltf_animation_path_type {
	cgltf_animation_path_type_invalid,
	cgltf_animation_path_type_translation,
	cgltf_animation_path_type_rotation,
	cgltf_animation_path_type_scale,
	cgltf_animation_path_type_weights,
	cgltf_animation_path_type_max_enum
} cgltf_animation_path_type;

typedef enum cgltf_interpolation_type {
	cgltf_interpolation_type_linear,
	cgltf_interpolation_type_step,
	cgltf_interpolation_type_cubic_spline,
	cgltf_interpolation_type_max_enum
} cgltf_interpolation_type;

typedef enum cgltf_camera_type {
	cgltf_camera_type_invalid,
	cgltf_camera_type_perspective,
	cgltf_camera_type_orthographic,
	cgltf_camera_type_max_enum
} cgltf_camera_type;

typedef enum cgltf_light_type {
	cgltf_light_type_invalid,
	cgltf_light_type_directional,
	cgltf_light_type_point,
	cgltf_light_type_spot,
	cgltf_light_type_max_enum
} cgltf_light_type;

typedef enum cgltf_data_free_method {
	cgltf_data_free_method_none,
	cgltf_data_free_method_file_release,
	cgltf_data_free_method_memory_free,
	cgltf_data_free_method_max_enum
} cgltf_data_free_method;

typedef struct cgltf_extras {
	cgltf_size start_offset;
	cgltf_size end_offset;
} cgltf_extras;

typedef struct cgltf_extension {
	char* name;
	char* data;
} cgltf_extension;

typedef struct cgltf_buffer
{
	char* name;
	cgltf_size size;
	char* uri;
	void* data; /* loaded by cgltf_load_buffers */
	cgltf_data_free_method data_free_method;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_buffer;

typedef enum cgltf_meshopt_compression_mode {
	cgltf_meshopt_compression_mode_invalid,
	cgltf_meshopt_compression_mode_attributes,
	cgltf_meshopt_compression_mode_triangles,
	cgltf_meshopt_compression_mode_indices,
	cgltf_meshopt_compression_mode_max_enum
} cgltf_meshopt_compression_mode;

typedef enum cgltf_meshopt_compression_filter {
	cgltf_meshopt_compression_filter_none,
	cgltf_meshopt_compression_filter_octahedral,
	cgltf_meshopt_compression_filter_quaternion,
	cgltf_meshopt_compression_filter_exponential,
	cgltf_meshopt_compression_filter_max_enum
} cgltf_meshopt_compression_filter;

typedef struct cgltf_meshopt_compression
{
	cgltf_buffer* buffer;
	cgltf_size offset;
	cgltf_size size;
	cgltf_size stride;
	cgltf_size count;
	cgltf_meshopt_compression_mode mode;
	cgltf_meshopt_compression_filter filter;
} cgltf_meshopt_compression;

typedef struct cgltf_buffer_view
{
	char *name;
	cgltf_buffer* buffer;
	cgltf_size offset;
	cgltf_size size;
	cgltf_size stride; /* 0 == automatically determined by accessor */
	cgltf_buffer_view_type type;
	void* data; /* overrides buffer->data if present, filled by extensions */
	cgltf_bool has_meshopt_compression;
	cgltf_meshopt_compression meshopt_compression;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_buffer_view;

typedef struct cgltf_accessor_sparse
{
	cgltf_size count;
	cgltf_buffer_view* indices_buffer_view;
	cgltf_size indices_byte_offset;
	cgltf_component_type indices_component_type;
	cgltf_buffer_view* values_buffer_view;
	cgltf_size values_byte_offset;
	cgltf_extras extras;
	cgltf_extras indices_extras;
	cgltf_extras values_extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
	cgltf_size indices_extensions_count;
	cgltf_extension* indices_extensions;
	cgltf_size values_extensions_count;
	cgltf_extension* values_extensions;
} cgltf_accessor_sparse;

typedef struct cgltf_accessor
{
	char* name;
	cgltf_component_type component_type;
	cgltf_bool normalized;
	cgltf_type type;
	cgltf_size offset;
	cgltf_size count;
	cgltf_size stride;
	cgltf_buffer_view* buffer_view;
	cgltf_bool has_min;
	cgltf_float min[16];
	cgltf_bool has_max;
	cgltf_float max[16];
	cgltf_bool is_sparse;
	cgltf_accessor_sparse sparse;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_accessor;

typedef struct cgltf_attribute
{
	char* name;
	cgltf_attribute_type type;
	cgltf_int index;
	cgltf_accessor* data;
} cgltf_attribute;

typedef struct cgltf_image
{
	char* name;
	char* uri;
	cgltf_buffer_view* buffer_view;
	char* mime_type;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_image;

typedef struct cgltf_sampler
{
	char* name;
	cgltf_int mag_filter;
	cgltf_int min_filter;
	cgltf_int wrap_s;
	cgltf_int wrap_t;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_sampler;

typedef struct cgltf_texture
{
	char* name;
	cgltf_image* image;
	cgltf_sampler* sampler;
	cgltf_bool has_basisu;
	cgltf_image* basisu_image;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_texture;

typedef struct cgltf_texture_transform
{
	cgltf_float offset[2];
	cgltf_float rotation;
	cgltf_float scale[2];
	cgltf_bool has_texcoord;
	cgltf_int texcoord;
} cgltf_texture_transform;

typedef struct cgltf_texture_view
{
	cgltf_texture* texture;
	cgltf_int texcoord;
	cgltf_float scale; /* equivalent to strength for occlusion_texture */
	cgltf_bool has_transform;
	cgltf_texture_transform transform;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_texture_view;

typedef struct cgltf_pbr_metallic_roughness
{
	cgltf_texture_view base_color_texture;
	cgltf_texture_view metallic_roughness_texture;

	cgltf_float base_color_factor[4];
	cgltf_float metallic_factor;
	cgltf_float roughness_factor;

	cgltf_extras extras;
} cgltf_pbr_metallic_roughness;

typedef struct cgltf_pbr_specular_glossiness
{
	cgltf_texture_view diffuse_texture;
	cgltf_texture_view specular_glossiness_texture;

	cgltf_float diffuse_factor[4];
	cgltf_float specular_factor[3];
	cgltf_float glossiness_factor;
} cgltf_pbr_specular_glossiness;

typedef struct cgltf_clearcoat
{
	cgltf_texture_view clearcoat_texture;
	cgltf_texture_view clearcoat_roughness_texture;
	cgltf_texture_view clearcoat_normal_texture;

	cgltf_float clearcoat_factor;
	cgltf_float clearcoat_roughness_factor;
} cgltf_clearcoat;

typedef struct cgltf_transmission
{
	cgltf_texture_view transmission_texture;
	cgltf_float transmission_factor;
} cgltf_transmission;

typedef struct cgltf_ior
{
	cgltf_float ior;
} cgltf_ior;

typedef struct cgltf_specular
{
	cgltf_texture_view specular_texture;
	cgltf_texture_view specular_color_texture;
	cgltf_float specular_color_factor[3];
	cgltf_float specular_factor;
} cgltf_specular;

typedef struct cgltf_volume
{
	cgltf_texture_view thickness_texture;
	cgltf_float thickness_factor;
	cgltf_float attenuation_color[3];
	cgltf_float attenuation_distance;
} cgltf_volume;

typedef struct cgltf_sheen
{
	cgltf_texture_view sheen_color_texture;
	cgltf_float sheen_color_factor[3];
	cgltf_texture_view sheen_roughness_texture;
	cgltf_float sheen_roughness_factor;
} cgltf_sheen;

typedef struct cgltf_emissive_strength
{
	cgltf_float emissive_strength;
} cgltf_emissive_strength;

typedef struct cgltf_iridescence
{
	cgltf_float iridescence_factor;
	cgltf_texture_view iridescence_texture;
	cgltf_float iridescence_ior;
	cgltf_float iridescence_thickness_min;
	cgltf_float iridescence_thickness_max;
	cgltf_texture_view iridescence_thickness_texture;
} cgltf_iridescence;

typedef struct cgltf_material
{
	char* name;
	cgltf_bool has_pbr_metallic_roughness;
	cgltf_bool has_pbr_specular_glossiness;
	cgltf_bool has_clearcoat;
	cgltf_bool has_transmission;
	cgltf_bool has_volume;
	cgltf_bool has_ior;
	cgltf_bool has_specular;
	cgltf_bool has_sheen;
	cgltf_bool has_emissive_strength;
	cgltf_bool has_iridescence;
	cgltf_pbr_metallic_roughness pbr_metallic_roughness;
	cgltf_pbr_specular_glossiness pbr_specular_glossiness;
	cgltf_clearcoat clearcoat;
	cgltf_ior ior;
	cgltf_specular specular;
	cgltf_sheen sheen;
	cgltf_transmission transmission;
	cgltf_volume volume;
	cgltf_emissive_strength emissive_strength;
	cgltf_iridescence iridescence;
	cgltf_texture_view normal_texture;
	cgltf_texture_view occlusion_texture;
	cgltf_texture_view emissive_texture;
	cgltf_float emissive_factor[3];
	cgltf_alpha_mode alpha_mode;
	cgltf_float alpha_cutoff;
	cgltf_bool double_sided;
	cgltf_bool unlit;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_material;

typedef struct cgltf_material_mapping
{
	cgltf_size variant;
	cgltf_material* material;
	cgltf_extras extras;
} cgltf_material_mapping;

typedef struct cgltf_morph_target {
	cgltf_attribute* attributes;
	cgltf_size attributes_count;
} cgltf_morph_target;

typedef struct cgltf_draco_mesh_compression {
	cgltf_buffer_view* buffer_view;
	cgltf_attribute* attributes;
	cgltf_size attributes_count;
} cgltf_draco_mesh_compression;

typedef struct cgltf_mesh_gpu_instancing {
	cgltf_buffer_view* buffer_view;
	cgltf_attribute* attributes;
	cgltf_size attributes_count;
} cgltf_mesh_gpu_instancing;

typedef struct cgltf_primitive {
	cgltf_primitive_type type;
	cgltf_accessor* indices;
	cgltf_material* material;
	cgltf_attribute* attributes;
	cgltf_size attributes_count;
	cgltf_morph_target* targets;
	cgltf_size targets_count;
	cgltf_extras extras;
	cgltf_bool has_draco_mesh_compression;
	cgltf_draco_mesh_compression draco_mesh_compression;
	cgltf_material_mapping* mappings;
	cgltf_size mappings_count;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_primitive;

typedef struct cgltf_mesh {
	char* name;
	cgltf_primitive* primitives;
	cgltf_size primitives_count;
	cgltf_float* weights;
	cgltf_size weights_count;
	char** target_names;
	cgltf_size target_names_count;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_mesh;

typedef struct cgltf_node cgltf_node;

typedef struct cgltf_skin {
	char* name;
	cgltf_node** joints;
	cgltf_size joints_count;
	cgltf_node* skeleton;
	cgltf_accessor* inverse_bind_matrices;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_skin;

typedef struct cgltf_camera_perspective {
	cgltf_bool has_aspect_ratio;
	cgltf_float aspect_ratio;
	cgltf_float yfov;
	cgltf_bool has_zfar;
	cgltf_float zfar;
	cgltf_float znear;
	cgltf_extras extras;
} cgltf_camera_perspective;

typedef struct cgltf_camera_orthographic {
	cgltf_float xmag;
	cgltf_float ymag;
	cgltf_float zfar;
	cgltf_float znear;
	cgltf_extras extras;
} cgltf_camera_orthographic;

typedef struct cgltf_camera {
	char* name;
	cgltf_camera_type type;
	union {
		cgltf_camera_perspective perspective;
		cgltf_camera_orthographic orthographic;
	} data;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_camera;

typedef struct cgltf_light {
	char* name;
	cgltf_float color[3];
	cgltf_float intensity;
	cgltf_light_type type;
	cgltf_float range;
	cgltf_float spot_inner_cone_angle;
	cgltf_float spot_outer_cone_angle;
	cgltf_extras extras;
} cgltf_light;

struct cgltf_node {
	char* name;
	cgltf_node* parent;
	cgltf_node** children;
	cgltf_size children_count;
	cgltf_skin* skin;
	cgltf_mesh* mesh;
	cgltf_camera* camera;
	cgltf_light* light;
	cgltf_float* weights;
	cgltf_size weights_count;
	cgltf_bool has_translation;
	cgltf_bool has_rotation;
	cgltf_bool has_scale;
	cgltf_bool has_matrix;
	cgltf_float translation[3];
	cgltf_float rotation[4];
	cgltf_float scale[3];
	cgltf_float matrix[16];
	cgltf_extras extras;
	cgltf_bool has_mesh_gpu_instancing;
	cgltf_mesh_gpu_instancing mesh_gpu_instancing;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
};

typedef struct cgltf_scene {
	char* name;
	cgltf_node** nodes;
	cgltf_size nodes_count;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_scene;

typedef struct cgltf_animation_sampler {
	cgltf_accessor* input;
	cgltf_accessor* output;
	cgltf_interpolation_type interpolation;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_animation_sampler;

typedef struct cgltf_animation_channel {
	cgltf_animation_sampler* sampler;
	cgltf_node* target_node;
	cgltf_animation_path_type target_path;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_animation_channel;

typedef struct cgltf_animation {
	char* name;
	cgltf_animation_sampler* samplers;
	cgltf_size samplers_count;
	cgltf_animation_channel* channels;
	cgltf_size channels_count;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_animation;

typedef struct cgltf_material_variant
{
	char* name;
	cgltf_extras extras;
} cgltf_material_variant;

typedef struct cgltf_asset {
	char* copyright;
	char* generator;
	char* version;
	char* min_version;
	cgltf_extras extras;
	cgltf_size extensions_count;
	cgltf_extension* extensions;
} cgltf_asset;

typedef struct cgltf_data
{
	cgltf_file_type file_type;
	void* file_data;

	cgltf_asset asset;

	cgltf_mesh* meshes;
	cgltf_size meshes_count;

	cgltf_material* materials;
	cgltf_size materials_count;

	cgltf_accessor* accessors;
	cgltf_size accessors_count;

	cgltf_buffer_view* buffer_views;
	cgltf_size buffer_views_count;

	cgltf_buffer* buffers;
	cgltf_size buffers_count;

	cgltf_image* images;
	cgltf_size images_count;

	cgltf_texture* textures;
	cgltf_size textures_count;

	cgltf_sampler* samplers;
	cgltf_size samplers_count;

	cgltf_skin* skins;
	cgltf_size skins_count;

	cgltf_camera* cameras;
	cgltf_size cameras_count;

	cgltf_light* lights;
	cgltf_size lights_count;

	cgltf_node* nodes;
	cgltf_size nodes_count;

	cgltf_scene* scenes;
	cgltf_size scenes_count;

	cgltf_scene* scene;

	cgltf_animation* animations;
	cgltf_size animations_count;

	cgltf_material_variant* variants;
	cgltf_size variants_count;

	cgltf_extras extras;

	cgltf_size data_extensions_count;
	cgltf_extension* data_extensions;

	char** extensions_used;
	cgltf_size extensions_used_count;

	char** extensions_required;
	cgltf_size extensions_required_count;

	const char* json;
	cgltf_size json_size;

	const void* bin;
	cgltf_size bin_size;

	cgltf_memory_options memory;
	cgltf_file_options file;
} cgltf_data;

cgltf_result cgltf_parse(
		const cgltf_options* options,
		const void* data,
		cgltf_size size,
		cgltf_data** out_data);

cgltf_result cgltf_parse_file(
		const cgltf_options* options,
		const char* path,
		cgltf_data** out_data);

cgltf_result cgltf_load_buffers(
		const cgltf_options* options,
		cgltf_data* data,
		const char* gltf_path);

cgltf_result cgltf_load_buffer_base64(const cgltf_options* options, cgltf_size size, const char* base64, void** out_data);

cgltf_size cgltf_decode_string(char* string);
cgltf_size cgltf_decode_uri(char* uri);

cgltf_result cgltf_validate(cgltf_data* data);

void cgltf_free(cgltf_data* data);

void cgltf_node_transform_local(const cgltf_node* node, cgltf_float* out_matrix);
void cgltf_node_transform_world(const cgltf_node* node, cgltf_float* out_matrix);

cgltf_bool cgltf_accessor_read_float(const cgltf_accessor* accessor, cgltf_size index, cgltf_float* out, cgltf_size element_size);
cgltf_bool cgltf_accessor_read_uint(const cgltf_accessor* accessor, cgltf_size index, cgltf_uint* out, cgltf_size element_size);
cgltf_size cgltf_accessor_read_index(const cgltf_accessor* accessor, cgltf_size index);

cgltf_size cgltf_num_components(cgltf_type type);

cgltf_size cgltf_accessor_unpack_floats(const cgltf_accessor* accessor, cgltf_float* out, cgltf_size float_count);

cgltf_result cgltf_copy_extras_json(const cgltf_data* data, const cgltf_extras* extras, char* dest, cgltf_size* dest_size);

#ifdef __cplusplus
}
#endif

#endif /* #ifndef CGLTF_H_INCLUDED__ */

/*
 *
 * Stop now, if you are only interested in the API.
 * Below, you find the implementation.
 *
 */

#if defined(__INTELLISENSE__) || defined(__JETBRAINS_IDE__)
/* This makes MSVC/CLion intellisense work. */
#define CGLTF_IMPLEMENTATION
#endif

#ifdef CGLTF_IMPLEMENTATION

#include <stdint.h> /* For uint8_t, uint32_t */
#include <string.h> /* For strncpy */
#include <stdio.h>  /* For fopen */
#include <limits.h> /* For UINT_MAX etc */
#include <float.h>  /* For FLT_MAX */

#if !defined(CGLTF_MALLOC) || !defined(CGLTF_FREE) || !defined(CGLTF_ATOI) || !defined(CGLTF_ATOF) || !defined(CGLTF_ATOLL)
#include <stdlib.h> /* For malloc, free, atoi, atof */
#endif

#if CGLTF_VALIDATE_ENABLE_ASSERTS
#include <assert.h>
#endif

/* JSMN_PARENT_LINKS is necessary to make parsing large structures linear in input size */
#define JSMN_PARENT_LINKS

/* JSMN_STRICT is necessary to reject invalid JSON documents */
#define JSMN_STRICT

/*
 * -- jsmn.h start --
 * Source: https://github.com/zserge/jsmn
 * License: MIT
 */
typedef enum {
	JSMN_UNDEFINED = 0,
	JSMN_OBJECT = 1,
	JSMN_ARRAY = 2,
	JSMN_STRING = 3,
	JSMN_PRIMITIVE = 4
} jsmntype_t;
enum jsmnerr {
	/* Not enough tokens were provided */
	JSMN_ERROR_NOMEM = -1,
	/* Invalid character inside JSON string */
	JSMN_ERROR_INVAL = -2,
	/* The string is not a full JSON packet, more bytes expected */
	JSMN_ERROR_PART = -3
};
typedef struct {
	jsmntype_t type;
	int start;
	int end;
	int size;
#ifdef JSMN_PARENT_LINKS
	int parent;
#endif
} jsmntok_t;
typedef struct {
	unsigned int pos; /* offset in the JSON string */
	unsigned int toknext; /* next token to allocate */
	int toksuper; /* superior token node, e.g parent object or array */
} jsmn_parser;
static void jsmn_init(jsmn_parser *parser);
static int jsmn_parse(jsmn_parser *parser, const char *js, size_t len, jsmntok_t *tokens, size_t num_tokens);
/*
 * -- jsmn.h end --
 */


static const cgltf_size GlbHeaderSize = 12;
static const cgltf_size GlbChunkHeaderSize = 8;
static const uint32_t GlbVersion = 2;
static const uint32_t GlbMagic = 0x46546C67;
static const uint32_t GlbMagicJsonChunk = 0x4E4F534A;
static const uint32_t GlbMagicBinChunk = 0x004E4942;

#ifndef CGLTF_MALLOC
#define CGLTF_MALLOC(size) malloc(size)
#endif
#ifndef CGLTF_FREE
#define CGLTF_FREE(ptr) free(ptr)
#endif
#ifndef CGLTF_ATOI
#define CGLTF_ATOI(str) atoi(str)
#endif
#ifndef CGLTF_ATOF
#define CGLTF_ATOF(str) atof(str)
#endif
#ifndef CGLTF_ATOLL
#define CGLTF_ATOLL(str) atoll(str)
#endif
#ifndef CGLTF_VALIDATE_ENABLE_ASSERTS
#define CGLTF_VALIDATE_ENABLE_ASSERTS 0
#endif

static void* cgltf_default_alloc(void* user, cgltf_size size)
{
	(void)user;
	return CGLTF_MALLOC(size);
}

static void cgltf_default_free(void* user, void* ptr)
{
	(void)user;
	CGLTF_FREE(ptr);
}

static void* cgltf_calloc(cgltf_options* options, size_t element_size, cgltf_size count)
{
	if (SIZE_MAX / element_size < count)
	{
		return NULL;
	}
	void* result = options->memory.alloc_func(options->memory.user_data, element_size * count);
	if (!result)
	{
		return NULL;
	}
	memset(result, 0, element_size * count);
	return result;
}

static cgltf_result cgltf_default_file_read(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, const char* path, cgltf_size* size, void** data)
{
	(void)file_options;
	void* (*memory_alloc)(void*, cgltf_size) = memory_options->alloc_func ? memory_options->alloc_func : &cgltf_default_alloc;
	void (*memory_free)(void*, void*) = memory_options->free_func ? memory_options->free_func : &cgltf_default_free;

	FILE* file = fopen(path, "rb");
	if (!file)
	{
		return cgltf_result_file_not_found;
	}

	cgltf_size file_size = size ? *size : 0;

	if (file_size == 0)
	{
		fseek(file, 0, SEEK_END);

#ifdef _WIN32
		__int64 length = _ftelli64(file);
#else
		long length = ftell(file);
#endif

		if (length < 0)
		{
			fclose(file);
			return cgltf_result_io_error;
		}

		fseek(file, 0, SEEK_SET);
		file_size = (cgltf_size)length;
	}

	char* file_data = (char*)memory_alloc(memory_options->user_data, file_size);
	if (!file_data)
	{
		fclose(file);
		return cgltf_result_out_of_memory;
	}
	
	cgltf_size read_size = fread(file_data, 1, file_size, file);

	fclose(file);

	if (read_size != file_size)
	{
		memory_free(memory_options->user_data, file_data);
		return cgltf_result_io_error;
	}

	if (size)
	{
		*size = file_size;
	}
	if (data)
	{
		*data = file_data;
	}

	return cgltf_result_success;
}

static void cgltf_default_file_release(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data)
{
	(void)file_options;
	void (*memfree)(void*, void*) = memory_options->free_func ? memory_options->free_func : &cgltf_default_free;
	memfree(memory_options->user_data, data);
}

static cgltf_result cgltf_parse_json(cgltf_options* options, const uint8_t* json_chunk, cgltf_size size, cgltf_data** out_data);

cgltf_result cgltf_parse(const cgltf_options* options, const void* data, cgltf_size size, cgltf_data** out_data)
{
	if (size < GlbHeaderSize)
	{
		return cgltf_result_data_too_short;
	}

	if (options == NULL)
	{
		return cgltf_result_invalid_options;
	}

	cgltf_options fixed_options = *options;
	if (fixed_options.memory.alloc_func == NULL)
	{
		fixed_options.memory.alloc_func = &cgltf_default_alloc;
	}
	if (fixed_options.memory.free_func == NULL)
	{
		fixed_options.memory.free_func = &cgltf_default_free;
	}

	uint32_t tmp;
	// Magic
	memcpy(&tmp, data, 4);
	if (tmp != GlbMagic)
	{
		if (fixed_options.type == cgltf_file_type_invalid)
		{
			fixed_options.type = cgltf_file_type_gltf;
		}
		else if (fixed_options.type == cgltf_file_type_glb)
		{
			return cgltf_result_unknown_format;
		}
	}

	if (fixed_options.type == cgltf_file_type_gltf)
	{
		cgltf_result json_result = cgltf_parse_json(&fixed_options, (const uint8_t*)data, size, out_data);
		if (json_result != cgltf_result_success)
		{
			return json_result;
		}

		(*out_data)->file_type = cgltf_file_type_gltf;

		return cgltf_result_success;
	}

	const uint8_t* ptr = (const uint8_t*)data;
	// Version
	memcpy(&tmp, ptr + 4, 4);
	uint32_t version = tmp;
	if (version != GlbVersion)
	{
		return version < GlbVersion ? cgltf_result_legacy_gltf : cgltf_result_unknown_format;
	}

	// Total length
	memcpy(&tmp, ptr + 8, 4);
	if (tmp > size)
	{
		return cgltf_result_data_too_short;
	}

	const uint8_t* json_chunk = ptr + GlbHeaderSize;

	if (GlbHeaderSize + GlbChunkHeaderSize > size)
	{
		return cgltf_result_data_too_short;
	}

	// JSON chunk: length
	uint32_t json_length;
	memcpy(&json_length, json_chunk, 4);
	if (GlbHeaderSize + GlbChunkHeaderSize + json_length > size)
	{
		return cgltf_result_data_too_short;
	}

	// JSON chunk: magic
	memcpy(&tmp, json_chunk + 4, 4);
	if (tmp != GlbMagicJsonChunk)
	{
		return cgltf_result_unknown_format;
	}

	json_chunk += GlbChunkHeaderSize;

	const void* bin = 0;
	cgltf_size bin_size = 0;

	if (GlbHeaderSize + GlbChunkHeaderSize + json_length + GlbChunkHeaderSize <= size)
	{
		// We can read another chunk
		const uint8_t* bin_chunk = json_chunk + json_length;

		// Bin chunk: length
		uint32_t bin_length;
		memcpy(&bin_length, bin_chunk, 4);
		if (GlbHeaderSize + GlbChunkHeaderSize + json_length + GlbChunkHeaderSize + bin_length > size)
		{
			return cgltf_result_data_too_short;
		}

		// Bin chunk: magic
		memcpy(&tmp, bin_chunk + 4, 4);
		if (tmp != GlbMagicBinChunk)
		{
			return cgltf_result_unknown_format;
		}

		bin_chunk += GlbChunkHeaderSize;

		bin = bin_chunk;
		bin_size = bin_length;
	}

	cgltf_result json_result = cgltf_parse_json(&fixed_options, json_chunk, json_length, out_data);
	if (json_result != cgltf_result_success)
	{
		return json_result;
	}

	(*out_data)->file_type = cgltf_file_type_glb;
	(*out_data)->bin = bin;
	(*out_data)->bin_size = bin_size;

	return cgltf_result_success;
}

cgltf_result cgltf_parse_file(const cgltf_options* options, const char* path, cgltf_data** out_data)
{
	if (options == NULL)
	{
		return cgltf_result_invalid_options;
	}

	cgltf_result (*file_read)(const struct cgltf_memory_options*, const struct cgltf_file_options*, const char*, cgltf_size*, void**) = options->file.read ? options->file.read : &cgltf_default_file_read;
	void (*file_release)(const struct cgltf_memory_options*, const struct cgltf_file_options*, void* data) = options->file.release ? options->file.release : cgltf_default_file_release;

	void* file_data = NULL;
	cgltf_size file_size = 0;
	cgltf_result result = file_read(&options->memory, &options->file, path, &file_size, &file_data);
	if (result != cgltf_result_success)
	{
		return result;
	}

	result = cgltf_parse(options, file_data, file_size, out_data);

	if (result != cgltf_result_success)
	{
		file_release(&options->memory, &options->file, file_data);
		return result;
	}

	(*out_data)->file_data = file_data;

	return cgltf_result_success;
}

static void cgltf_combine_paths(char* path, const char* base, const char* uri)
{
	const char* s0 = strrchr(base, '/');
	const char* s1 = strrchr(base, '\\');
	const char* slash = s0 ? (s1 && s1 > s0 ? s1 : s0) : s1;

	if (slash)
	{
		size_t prefix = slash - base + 1;

		strncpy(path, base, prefix);
		strcpy(path + prefix, uri);
	}
	else
	{
		strcpy(path, uri);
	}
}

static cgltf_result cgltf_load_buffer_file(const cgltf_options* options, cgltf_size size, const char* uri, const char* gltf_path, void** out_data)
{
	void* (*memory_alloc)(void*, cgltf_size) = options->memory.alloc_func ? options->memory.alloc_func : &cgltf_default_alloc;
	void (*memory_free)(void*, void*) = options->memory.free_func ? options->memory.free_func : &cgltf_default_free;
	cgltf_result (*file_read)(const struct cgltf_memory_options*, const struct cgltf_file_options*, const char*, cgltf_size*, void**) = options->file.read ? options->file.read : &cgltf_default_file_read;

	char* path = (char*)memory_alloc(options->memory.user_data, strlen(uri) + strlen(gltf_path) + 1);
	if (!path)
	{
		return cgltf_result_out_of_memory;
	}

	cgltf_combine_paths(path, gltf_path, uri);

	// after combining, the tail of the resulting path is a uri; decode_uri converts it into path
	cgltf_decode_uri(path + strlen(path) - strlen(uri));

	void* file_data = NULL;
	cgltf_result result = file_read(&options->memory, &options->file, path, &size, &file_data);

	memory_free(options->memory.user_data, path);

	*out_data = (result == cgltf_result_success) ? file_data : NULL;

	return result;
}

cgltf_result cgltf_load_buffer_base64(const cgltf_options* options, cgltf_size size, const char* base64, void** out_data)
{
	void* (*memory_alloc)(void*, cgltf_size) = options->memory.alloc_func ? options->memory.alloc_func : &cgltf_default_alloc;
	void (*memory_free)(void*, void*) = options->memory.free_func ? options->memory.free_func : &cgltf_default_free;

	unsigned char* data = (unsigned char*)memory_alloc(options->memory.user_data, size);
	if (!data)
	{
		return cgltf_result_out_of_memory;
	}

	unsigned int buffer = 0;
	unsigned int buffer_bits = 0;

	for (cgltf_size i = 0; i < size; ++i)
	{
		while (buffer_bits < 8)
		{
			char ch = *base64++;

			int index =
				(unsigned)(ch - 'A') < 26 ? (ch - 'A') :
				(unsigned)(ch - 'a') < 26 ? (ch - 'a') + 26 :
				(unsigned)(ch - '0') < 10 ? (ch - '0') + 52 :
				ch == '+' ? 62 :
				ch == '/' ? 63 :
				-1;

			if (index < 0)
			{
				memory_free(options->memory.user_data, data);
				return cgltf_result_io_error;
			}

			buffer = (buffer << 6) | index;
			buffer_bits += 6;
		}

		data[i] = (unsigned char)(buffer >> (buffer_bits - 8));
		buffer_bits -= 8;
	}

	*out_data = data;

	return cgltf_result_success;
}

static int cgltf_unhex(char ch)
{
	return
		(unsigned)(ch - '0') < 10 ? (ch - '0') :
		(unsigned)(ch - 'A') < 6 ? (ch - 'A') + 10 :
		(unsigned)(ch - 'a') < 6 ? (ch - 'a') + 10 :
		-1;
}

cgltf_size cgltf_decode_string(char* string)
{
	char* read = string + strcspn(string, "\\");
	if (*read == 0)
	{
		return read - string;
	}
	char* write = string;
	char* last = string;

	for (;;)
	{
		// Copy characters since last escaped sequence
		cgltf_size written = read - last;
		memmove(write, last, written);
		write += written;

		if (*read++ == 0)
		{
			break;
		}

		// jsmn already checked that all escape sequences are valid
		switch (*read++)
		{
		case '\"': *write++ = '\"'; break;
		case '/':  *write++ = '/';  break;
		case '\\': *write++ = '\\'; break;
		case 'b':  *write++ = '\b'; break;
		case 'f':  *write++ = '\f'; break;
		case 'r':  *write++ = '\r'; break;
		case 'n':  *write++ = '\n'; break;
		case 't':  *write++ = '\t'; break;
		case 'u':
		{
			// UCS-2 codepoint \uXXXX to UTF-8
			int character = 0;
			for (cgltf_size i = 0; i < 4; ++i)
			{
				character = (character << 4) + cgltf_unhex(*read++);
			}

			if (character <= 0x7F)
			{
				*write++ = character & 0xFF;
			}
			else if (character <= 0x7FF)
			{
				*write++ = 0xC0 | ((character >> 6) & 0xFF);
				*write++ = 0x80 | (character & 0x3F);
			}
			else
			{
				*write++ = 0xE0 | ((character >> 12) & 0xFF);
				*write++ = 0x80 | ((character >> 6) & 0x3F);
				*write++ = 0x80 | (character & 0x3F);
			}
			break;
		}
		default:
			break;
		}

		last = read;
		read += strcspn(read, "\\");
	}

	*write = 0;
	return write - string;
}

cgltf_size cgltf_decode_uri(char* uri)
{
	char* write = uri;
	char* i = uri;

	while (*i)
	{
		if (*i == '%')
		{
			int ch1 = cgltf_unhex(i[1]);

			if (ch1 >= 0)
			{
				int ch2 = cgltf_unhex(i[2]);

				if (ch2 >= 0)
				{
					*write++ = (char)(ch1 * 16 + ch2);
					i += 3;
					continue;
				}
			}
		}

		*write++ = *i++;
	}

	*write = 0;
	return write - uri;
}

cgltf_result cgltf_load_buffers(const cgltf_options* options, cgltf_data* data, const char* gltf_path)
{
	if (options == NULL)
	{
		return cgltf_result_invalid_options;
	}

	if (data->buffers_count && data->buffers[0].data == NULL && data->buffers[0].uri == NULL && data->bin)
	{
		if (data->bin_size < data->buffers[0].size)
		{
			return cgltf_result_data_too_short;
		}

		data->buffers[0].data = (void*)data->bin;
		data->buffers[0].data_free_method = cgltf_data_free_method_none;
	}

	for (cgltf_size i = 0; i < data->buffers_count; ++i)
	{
		if (data->buffers[i].data)
		{
			continue;
		}

		const char* uri = data->buffers[i].uri;

		if (uri == NULL)
		{
			continue;
		}

		if (strncmp(uri, "data:", 5) == 0)
		{
			const char* comma = strchr(uri, ',');

			if (comma && comma - uri >= 7 && strncmp(comma - 7, ";base64", 7) == 0)
			{
				cgltf_result res = cgltf_load_buffer_base64(options, data->buffers[i].size, comma + 1, &data->buffers[i].data);
				data->buffers[i].data_free_method = cgltf_data_free_method_memory_free;

				if (res != cgltf_result_success)
				{
					return res;
				}
			}
			else
			{
				return cgltf_result_unknown_format;
			}
		}
		else if (strstr(uri, "://") == NULL && gltf_path)
		{
			cgltf_result res = cgltf_load_buffer_file(options, data->buffers[i].size, uri, gltf_path, &data->buffers[i].data);
			data->buffers[i].data_free_method = cgltf_data_free_method_file_release;

			if (res != cgltf_result_success)
			{
				return res;
			}
		}
		else
		{
			return cgltf_result_unknown_format;
		}
	}

	return cgltf_result_success;
}

static cgltf_size cgltf_calc_size(cgltf_type type, cgltf_component_type component_type);

static cgltf_size cgltf_calc_index_bound(cgltf_buffer_view* buffer_view, cgltf_size offset, cgltf_component_type component_type, cgltf_size count)
{
	char* data = (char*)buffer_view->buffer->data + offset + buffer_view->offset;
	cgltf_size bound = 0;

	switch (component_type)
	{
	case cgltf_component_type_r_8u:
		for (size_t i = 0; i < count; ++i)
		{
			cgltf_size v = ((unsigned char*)data)[i];
			bound = bound > v ? bound : v;
		}
		break;

	case cgltf_component_type_r_16u:
		for (size_t i = 0; i < count; ++i)
		{
			cgltf_size v = ((unsigned short*)data)[i];
			bound = bound > v ? bound : v;
		}
		break;

	case cgltf_component_type_r_32u:
		for (size_t i = 0; i < count; ++i)
		{
			cgltf_size v = ((unsigned int*)data)[i];
			bound = bound > v ? bound : v;
		}
		break;

	default:
		;
	}

	return bound;
}

#if CGLTF_VALIDATE_ENABLE_ASSERTS
#define CGLTF_ASSERT_IF(cond, result) assert(!(cond)); if (cond) return result;
#else
#define CGLTF_ASSERT_IF(cond, result) if (cond) return result;
#endif

cgltf_result cgltf_validate(cgltf_data* data)
{
	for (cgltf_size i = 0; i < data->accessors_count; ++i)
	{
		cgltf_accessor* accessor = &data->accessors[i];

		cgltf_size element_size = cgltf_calc_size(accessor->type, accessor->component_type);

		if (accessor->buffer_view)
		{
			cgltf_size req_size = accessor->offset + accessor->stride * (accessor->count - 1) + element_size;

			CGLTF_ASSERT_IF(accessor->buffer_view->size < req_size, cgltf_result_data_too_short);
		}

		if (accessor->is_sparse)
		{
			cgltf_accessor_sparse* sparse = &accessor->sparse;

			cgltf_size indices_component_size = cgltf_calc_size(cgltf_type_scalar, sparse->indices_component_type);
			cgltf_size indices_req_size = sparse->indices_byte_offset + indices_component_size * sparse->count;
			cgltf_size values_req_size = sparse->values_byte_offset + element_size * sparse->count;

			CGLTF_ASSERT_IF(sparse->indices_buffer_view->size < indices_req_size ||
							sparse->values_buffer_view->size < values_req_size, cgltf_result_data_too_short);

			CGLTF_ASSERT_IF(sparse->indices_component_type != cgltf_component_type_r_8u &&
							sparse->indices_component_type != cgltf_component_type_r_16u &&
							sparse->indices_component_type != cgltf_component_type_r_32u, cgltf_result_invalid_gltf);

			if (sparse->indices_buffer_view->buffer->data)
			{
				cgltf_size index_bound = cgltf_calc_index_bound(sparse->indices_buffer_view, sparse->indices_byte_offset, sparse->indices_component_type, sparse->count);

				CGLTF_ASSERT_IF(index_bound >= accessor->count, cgltf_result_data_too_short);
			}
		}
	}

	for (cgltf_size i = 0; i < data->buffer_views_count; ++i)
	{
		cgltf_size req_size = data->buffer_views[i].offset + data->buffer_views[i].size;

		CGLTF_ASSERT_IF(data->buffer_views[i].buffer && data->buffer_views[i].buffer->size < req_size, cgltf_result_data_too_short);

		if (data->buffer_views[i].has_meshopt_compression)
		{
			cgltf_meshopt_compression* mc = &data->buffer_views[i].meshopt_compression;

			CGLTF_ASSERT_IF(mc->buffer == NULL || mc->buffer->size < mc->offset + mc->size, cgltf_result_data_too_short);

			CGLTF_ASSERT_IF(data->buffer_views[i].stride && mc->stride != data->buffer_views[i].stride, cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF(data->buffer_views[i].size != mc->stride * mc->count, cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF(mc->mode == cgltf_meshopt_compression_mode_invalid, cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF(mc->mode == cgltf_meshopt_compression_mode_attributes && !(mc->stride % 4 == 0 && mc->stride <= 256), cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF(mc->mode == cgltf_meshopt_compression_mode_triangles && mc->count % 3 != 0, cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF((mc->mode == cgltf_meshopt_compression_mode_triangles || mc->mode == cgltf_meshopt_compression_mode_indices) && mc->stride != 2 && mc->stride != 4, cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF((mc->mode == cgltf_meshopt_compression_mode_triangles || mc->mode == cgltf_meshopt_compression_mode_indices) && mc->filter != cgltf_meshopt_compression_filter_none, cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF(mc->filter == cgltf_meshopt_compression_filter_octahedral && mc->stride != 4 && mc->stride != 8, cgltf_result_invalid_gltf);

			CGLTF_ASSERT_IF(mc->filter == cgltf_meshopt_compression_filter_quaternion && mc->stride != 8, cgltf_result_invalid_gltf);
		}
	}

	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		if (data->meshes[i].weights)
		{
			CGLTF_ASSERT_IF(data->meshes[i].primitives_count && data->meshes[i].primitives[0].targets_count != data->meshes[i].weights_count, cgltf_result_invalid_gltf);
		}

		if (data->meshes[i].target_names)
		{
			CGLTF_ASSERT_IF(data->meshes[i].primitives_count && data->meshes[i].primitives[0].targets_count != data->meshes[i].target_names_count, cgltf_result_invalid_gltf);
		}

		for (cgltf_size j = 0; j < data->meshes[i].primitives_count; ++j)
		{
			CGLTF_ASSERT_IF(data->meshes[i].primitives[j].targets_count != data->meshes[i].primitives[0].targets_count, cgltf_result_invalid_gltf);

			if (data->meshes[i].primitives[j].attributes_count)
			{
				cgltf_accessor* first = data->meshes[i].primitives[j].attributes[0].data;

				for (cgltf_size k = 0; k < data->meshes[i].primitives[j].attributes_count; ++k)
				{
					CGLTF_ASSERT_IF(data->meshes[i].primitives[j].attributes[k].data->count != first->count, cgltf_result_invalid_gltf);
				}

				for (cgltf_size k = 0; k < data->meshes[i].primitives[j].targets_count; ++k)
				{
					for (cgltf_size m = 0; m < data->meshes[i].primitives[j].targets[k].attributes_count; ++m)
					{
						CGLTF_ASSERT_IF(data->meshes[i].primitives[j].targets[k].attributes[m].data->count != first->count, cgltf_result_invalid_gltf);
					}
				}

				cgltf_accessor* indices = data->meshes[i].primitives[j].indices;

				CGLTF_ASSERT_IF(indices &&
					indices->component_type != cgltf_component_type_r_8u &&
					indices->component_type != cgltf_component_type_r_16u &&
					indices->component_type != cgltf_component_type_r_32u, cgltf_result_invalid_gltf);

				if (indices && indices->buffer_view && indices->buffer_view->buffer->data)
				{
					cgltf_size index_bound = cgltf_calc_index_bound(indices->buffer_view, indices->offset, indices->component_type, indices->count);

					CGLTF_ASSERT_IF(index_bound >= first->count, cgltf_result_data_too_short);
				}

				for (cgltf_size k = 0; k < data->meshes[i].primitives[j].mappings_count; ++k)
				{
					CGLTF_ASSERT_IF(data->meshes[i].primitives[j].mappings[k].variant >= data->variants_count, cgltf_result_invalid_gltf);
				}
			}
		}
	}

	for (cgltf_size i = 0; i < data->nodes_count; ++i)
	{
		if (data->nodes[i].weights && data->nodes[i].mesh)
		{
			CGLTF_ASSERT_IF (data->nodes[i].mesh->primitives_count && data->nodes[i].mesh->primitives[0].targets_count != data->nodes[i].weights_count, cgltf_result_invalid_gltf);
		}
	}

	for (cgltf_size i = 0; i < data->nodes_count; ++i)
	{
		cgltf_node* p1 = data->nodes[i].parent;
		cgltf_node* p2 = p1 ? p1->parent : NULL;

		while (p1 && p2)
		{
			CGLTF_ASSERT_IF(p1 == p2, cgltf_result_invalid_gltf);

			p1 = p1->parent;
			p2 = p2->parent ? p2->parent->parent : NULL;
		}
	}

	for (cgltf_size i = 0; i < data->scenes_count; ++i)
	{
		for (cgltf_size j = 0; j < data->scenes[i].nodes_count; ++j)
		{
			CGLTF_ASSERT_IF(data->scenes[i].nodes[j]->parent, cgltf_result_invalid_gltf);
		}
	}

	for (cgltf_size i = 0; i < data->animations_count; ++i)
	{
		for (cgltf_size j = 0; j < data->animations[i].channels_count; ++j)
		{
			cgltf_animation_channel* channel = &data->animations[i].channels[j];

			if (!channel->target_node)
			{
				continue;
			}

			cgltf_size components = 1;

			if (channel->target_path == cgltf_animation_path_type_weights)
			{
				CGLTF_ASSERT_IF(!channel->target_node->mesh || !channel->target_node->mesh->primitives_count, cgltf_result_invalid_gltf);

				components = channel->target_node->mesh->primitives[0].targets_count;
			}

			cgltf_size values = channel->sampler->interpolation == cgltf_interpolation_type_cubic_spline ? 3 : 1;

			CGLTF_ASSERT_IF(channel->sampler->input->count * components * values != channel->sampler->output->count, cgltf_result_data_too_short);
		}
	}

	return cgltf_result_success;
}

cgltf_result cgltf_copy_extras_json(const cgltf_data* data, const cgltf_extras* extras, char* dest, cgltf_size* dest_size)
{
	cgltf_size json_size = extras->end_offset - extras->start_offset;

	if (!dest)
	{
		if (dest_size)
		{
			*dest_size = json_size + 1;
			return cgltf_result_success;
		}
		return cgltf_result_invalid_options;
	}

	if (*dest_size + 1 < json_size)
	{
		strncpy(dest, data->json + extras->start_offset, *dest_size - 1);
		dest[*dest_size - 1] = 0;
	}
	else
	{
		strncpy(dest, data->json + extras->start_offset, json_size);
		dest[json_size] = 0;
	}

	return cgltf_result_success;
}

void cgltf_free_extensions(cgltf_data* data, cgltf_extension* extensions, cgltf_size extensions_count)
{
	for (cgltf_size i = 0; i < extensions_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, extensions[i].name);
		data->memory.free_func(data->memory.user_data, extensions[i].data);
	}
	data->memory.free_func(data->memory.user_data, extensions);
}

void cgltf_free(cgltf_data* data)
{
	if (!data)
	{
		return;
	}

	void (*file_release)(const struct cgltf_memory_options*, const struct cgltf_file_options*, void* data) = data->file.release ? data->file.release : cgltf_default_file_release;

	data->memory.free_func(data->memory.user_data, data->asset.copyright);
	data->memory.free_func(data->memory.user_data, data->asset.generator);
	data->memory.free_func(data->memory.user_data, data->asset.version);
	data->memory.free_func(data->memory.user_data, data->asset.min_version);

	cgltf_free_extensions(data, data->asset.extensions, data->asset.extensions_count);

	for (cgltf_size i = 0; i < data->accessors_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->accessors[i].name);

		if(data->accessors[i].is_sparse)
		{
			cgltf_free_extensions(data, data->accessors[i].sparse.extensions, data->accessors[i].sparse.extensions_count);
			cgltf_free_extensions(data, data->accessors[i].sparse.indices_extensions, data->accessors[i].sparse.indices_extensions_count);
			cgltf_free_extensions(data, data->accessors[i].sparse.values_extensions, data->accessors[i].sparse.values_extensions_count);
		}
		cgltf_free_extensions(data, data->accessors[i].extensions, data->accessors[i].extensions_count);
	}
	data->memory.free_func(data->memory.user_data, data->accessors);

	for (cgltf_size i = 0; i < data->buffer_views_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->buffer_views[i].name);
		data->memory.free_func(data->memory.user_data, data->buffer_views[i].data);

		cgltf_free_extensions(data, data->buffer_views[i].extensions, data->buffer_views[i].extensions_count);
	}
	data->memory.free_func(data->memory.user_data, data->buffer_views);

	for (cgltf_size i = 0; i < data->buffers_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->buffers[i].name);

		if (data->buffers[i].data_free_method == cgltf_data_free_method_file_release)
		{
			file_release(&data->memory, &data->file, data->buffers[i].data);
		}
		else if (data->buffers[i].data_free_method == cgltf_data_free_method_memory_free)
		{
			data->memory.free_func(data->memory.user_data, data->buffers[i].data);
		}

		data->memory.free_func(data->memory.user_data, data->buffers[i].uri);

		cgltf_free_extensions(data, data->buffers[i].extensions, data->buffers[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->buffers);

	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->meshes[i].name);

		for (cgltf_size j = 0; j < data->meshes[i].primitives_count; ++j)
		{
			for (cgltf_size k = 0; k < data->meshes[i].primitives[j].attributes_count; ++k)
			{
				data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].attributes[k].name);
			}

			data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].attributes);

			for (cgltf_size k = 0; k < data->meshes[i].primitives[j].targets_count; ++k)
			{
				for (cgltf_size m = 0; m < data->meshes[i].primitives[j].targets[k].attributes_count; ++m)
				{
					data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].targets[k].attributes[m].name);
				}

				data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].targets[k].attributes);
			}

			data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].targets);

			if (data->meshes[i].primitives[j].has_draco_mesh_compression)
			{
				for (cgltf_size k = 0; k < data->meshes[i].primitives[j].draco_mesh_compression.attributes_count; ++k)
				{
					data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].draco_mesh_compression.attributes[k].name);
				}

				data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].draco_mesh_compression.attributes);
			}

			data->memory.free_func(data->memory.user_data, data->meshes[i].primitives[j].mappings);

			cgltf_free_extensions(data, data->meshes[i].primitives[j].extensions, data->meshes[i].primitives[j].extensions_count);
		}

		data->memory.free_func(data->memory.user_data, data->meshes[i].primitives);
		data->memory.free_func(data->memory.user_data, data->meshes[i].weights);

		for (cgltf_size j = 0; j < data->meshes[i].target_names_count; ++j)
		{
			data->memory.free_func(data->memory.user_data, data->meshes[i].target_names[j]);
		}

		cgltf_free_extensions(data, data->meshes[i].extensions, data->meshes[i].extensions_count);

		data->memory.free_func(data->memory.user_data, data->meshes[i].target_names);
	}

	data->memory.free_func(data->memory.user_data, data->meshes);

	for (cgltf_size i = 0; i < data->materials_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->materials[i].name);

		if(data->materials[i].has_pbr_metallic_roughness)
		{
			cgltf_free_extensions(data, data->materials[i].pbr_metallic_roughness.metallic_roughness_texture.extensions, data->materials[i].pbr_metallic_roughness.metallic_roughness_texture.extensions_count);
			cgltf_free_extensions(data, data->materials[i].pbr_metallic_roughness.base_color_texture.extensions, data->materials[i].pbr_metallic_roughness.base_color_texture.extensions_count);
		}
		if(data->materials[i].has_pbr_specular_glossiness)
		{
			cgltf_free_extensions(data, data->materials[i].pbr_specular_glossiness.diffuse_texture.extensions, data->materials[i].pbr_specular_glossiness.diffuse_texture.extensions_count);
			cgltf_free_extensions(data, data->materials[i].pbr_specular_glossiness.specular_glossiness_texture.extensions, data->materials[i].pbr_specular_glossiness.specular_glossiness_texture.extensions_count);
		}
		if(data->materials[i].has_clearcoat)
		{
			cgltf_free_extensions(data, data->materials[i].clearcoat.clearcoat_texture.extensions, data->materials[i].clearcoat.clearcoat_texture.extensions_count);
			cgltf_free_extensions(data, data->materials[i].clearcoat.clearcoat_roughness_texture.extensions, data->materials[i].clearcoat.clearcoat_roughness_texture.extensions_count);
			cgltf_free_extensions(data, data->materials[i].clearcoat.clearcoat_normal_texture.extensions, data->materials[i].clearcoat.clearcoat_normal_texture.extensions_count);
		}
		if(data->materials[i].has_specular)
		{
			cgltf_free_extensions(data, data->materials[i].specular.specular_texture.extensions, data->materials[i].specular.specular_texture.extensions_count);
			cgltf_free_extensions(data, data->materials[i].specular.specular_color_texture.extensions, data->materials[i].specular.specular_color_texture.extensions_count);
		}
		if(data->materials[i].has_transmission)
		{
			cgltf_free_extensions(data, data->materials[i].transmission.transmission_texture.extensions, data->materials[i].transmission.transmission_texture.extensions_count);
		}
		if (data->materials[i].has_volume)
		{
			cgltf_free_extensions(data, data->materials[i].volume.thickness_texture.extensions, data->materials[i].volume.thickness_texture.extensions_count);
		}
		if(data->materials[i].has_sheen)
		{
			cgltf_free_extensions(data, data->materials[i].sheen.sheen_color_texture.extensions, data->materials[i].sheen.sheen_color_texture.extensions_count);
			cgltf_free_extensions(data, data->materials[i].sheen.sheen_roughness_texture.extensions, data->materials[i].sheen.sheen_roughness_texture.extensions_count);
		}
		if(data->materials[i].has_iridescence)
		{
			cgltf_free_extensions(data, data->materials[i].iridescence.iridescence_texture.extensions, data->materials[i].iridescence.iridescence_texture.extensions_count);
			cgltf_free_extensions(data, data->materials[i].iridescence.iridescence_thickness_texture.extensions, data->materials[i].iridescence.iridescence_thickness_texture.extensions_count);
		}

		cgltf_free_extensions(data, data->materials[i].normal_texture.extensions, data->materials[i].normal_texture.extensions_count);
		cgltf_free_extensions(data, data->materials[i].occlusion_texture.extensions, data->materials[i].occlusion_texture.extensions_count);
		cgltf_free_extensions(data, data->materials[i].emissive_texture.extensions, data->materials[i].emissive_texture.extensions_count);

		cgltf_free_extensions(data, data->materials[i].extensions, data->materials[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->materials);

	for (cgltf_size i = 0; i < data->images_count; ++i) 
	{
		data->memory.free_func(data->memory.user_data, data->images[i].name);
		data->memory.free_func(data->memory.user_data, data->images[i].uri);
		data->memory.free_func(data->memory.user_data, data->images[i].mime_type);

		cgltf_free_extensions(data, data->images[i].extensions, data->images[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->images);

	for (cgltf_size i = 0; i < data->textures_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->textures[i].name);
		cgltf_free_extensions(data, data->textures[i].extensions, data->textures[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->textures);

	for (cgltf_size i = 0; i < data->samplers_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->samplers[i].name);
		cgltf_free_extensions(data, data->samplers[i].extensions, data->samplers[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->samplers);

	for (cgltf_size i = 0; i < data->skins_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->skins[i].name);
		data->memory.free_func(data->memory.user_data, data->skins[i].joints);

		cgltf_free_extensions(data, data->skins[i].extensions, data->skins[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->skins);

	for (cgltf_size i = 0; i < data->cameras_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->cameras[i].name);
		cgltf_free_extensions(data, data->cameras[i].extensions, data->cameras[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->cameras);

	for (cgltf_size i = 0; i < data->lights_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->lights[i].name);
	}

	data->memory.free_func(data->memory.user_data, data->lights);

	for (cgltf_size i = 0; i < data->nodes_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->nodes[i].name);
		data->memory.free_func(data->memory.user_data, data->nodes[i].children);
		data->memory.free_func(data->memory.user_data, data->nodes[i].weights);
		cgltf_free_extensions(data, data->nodes[i].extensions, data->nodes[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->nodes);

	for (cgltf_size i = 0; i < data->scenes_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->scenes[i].name);
		data->memory.free_func(data->memory.user_data, data->scenes[i].nodes);

		cgltf_free_extensions(data, data->scenes[i].extensions, data->scenes[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->scenes);

	for (cgltf_size i = 0; i < data->animations_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->animations[i].name);
		for (cgltf_size j = 0; j <  data->animations[i].samplers_count; ++j)
		{
			cgltf_free_extensions(data, data->animations[i].samplers[j].extensions, data->animations[i].samplers[j].extensions_count);
		}
		data->memory.free_func(data->memory.user_data, data->animations[i].samplers);

		for (cgltf_size j = 0; j <  data->animations[i].channels_count; ++j)
		{
			cgltf_free_extensions(data, data->animations[i].channels[j].extensions, data->animations[i].channels[j].extensions_count);
		}
		data->memory.free_func(data->memory.user_data, data->animations[i].channels);

		cgltf_free_extensions(data, data->animations[i].extensions, data->animations[i].extensions_count);
	}

	data->memory.free_func(data->memory.user_data, data->animations);

	for (cgltf_size i = 0; i < data->variants_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->variants[i].name);
	}

	data->memory.free_func(data->memory.user_data, data->variants);

	cgltf_free_extensions(data, data->data_extensions, data->data_extensions_count);

	for (cgltf_size i = 0; i < data->extensions_used_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->extensions_used[i]);
	}

	data->memory.free_func(data->memory.user_data, data->extensions_used);

	for (cgltf_size i = 0; i < data->extensions_required_count; ++i)
	{
		data->memory.free_func(data->memory.user_data, data->extensions_required[i]);
	}

	data->memory.free_func(data->memory.user_data, data->extensions_required);

	file_release(&data->memory, &data->file, data->file_data);

	data->memory.free_func(data->memory.user_data, data);
}

void cgltf_node_transform_local(const cgltf_node* node, cgltf_float* out_matrix)
{
	cgltf_float* lm = out_matrix;

	if (node->has_matrix)
	{
		memcpy(lm, node->matrix, sizeof(float) * 16);
	}
	else
	{
		float tx = node->translation[0];
		float ty = node->translation[1];
		float tz = node->translation[2];

		float qx = node->rotation[0];
		float qy = node->rotation[1];
		float qz = node->rotation[2];
		float qw = node->rotation[3];

		float sx = node->scale[0];
		float sy = node->scale[1];
		float sz = node->scale[2];

		lm[0] = (1 - 2 * qy*qy - 2 * qz*qz) * sx;
		lm[1] = (2 * qx*qy + 2 * qz*qw) * sx;
		lm[2] = (2 * qx*qz - 2 * qy*qw) * sx;
		lm[3] = 0.f;

		lm[4] = (2 * qx*qy - 2 * qz*qw) * sy;
		lm[5] = (1 - 2 * qx*qx - 2 * qz*qz) * sy;
		lm[6] = (2 * qy*qz + 2 * qx*qw) * sy;
		lm[7] = 0.f;

		lm[8] = (2 * qx*qz + 2 * qy*qw) * sz;
		lm[9] = (2 * qy*qz - 2 * qx*qw) * sz;
		lm[10] = (1 - 2 * qx*qx - 2 * qy*qy) * sz;
		lm[11] = 0.f;

		lm[12] = tx;
		lm[13] = ty;
		lm[14] = tz;
		lm[15] = 1.f;
	}
}

void cgltf_node_transform_world(const cgltf_node* node, cgltf_float* out_matrix)
{
	cgltf_float* lm = out_matrix;
	cgltf_node_transform_local(node, lm);

	const cgltf_node* parent = node->parent;

	while (parent)
	{
		float pm[16];
		cgltf_node_transform_local(parent, pm);

		for (int i = 0; i < 4; ++i)
		{
			float l0 = lm[i * 4 + 0];
			float l1 = lm[i * 4 + 1];
			float l2 = lm[i * 4 + 2];

			float r0 = l0 * pm[0] + l1 * pm[4] + l2 * pm[8];
			float r1 = l0 * pm[1] + l1 * pm[5] + l2 * pm[9];
			float r2 = l0 * pm[2] + l1 * pm[6] + l2 * pm[10];

			lm[i * 4 + 0] = r0;
			lm[i * 4 + 1] = r1;
			lm[i * 4 + 2] = r2;
		}

		lm[12] += pm[12];
		lm[13] += pm[13];
		lm[14] += pm[14];

		parent = parent->parent;
	}
}

static cgltf_ssize cgltf_component_read_integer(const void* in, cgltf_component_type component_type)
{
	switch (component_type)
	{
		case cgltf_component_type_r_16:
			return *((const int16_t*) in);
		case cgltf_component_type_r_16u:
			return *((const uint16_t*) in);
		case cgltf_component_type_r_32u:
			return *((const uint32_t*) in);
		case cgltf_component_type_r_32f:
			return (cgltf_ssize)*((const float*) in);
		case cgltf_component_type_r_8:
			return *((const int8_t*) in);
		case cgltf_component_type_r_8u:
			return *((const uint8_t*) in);
		default:
			return 0;
	}
}

static cgltf_size cgltf_component_read_index(const void* in, cgltf_component_type component_type)
{
	switch (component_type)
	{
		case cgltf_component_type_r_16u:
			return *((const uint16_t*) in);
		case cgltf_component_type_r_32u:
			return *((const uint32_t*) in);
		case cgltf_component_type_r_32f:
			return (cgltf_size)*((const float*) in);
		case cgltf_component_type_r_8u:
			return *((const uint8_t*) in);
		default:
			return 0;
	}
}

static cgltf_float cgltf_component_read_float(const void* in, cgltf_component_type component_type, cgltf_bool normalized)
{
	if (component_type == cgltf_component_type_r_32f)
	{
		return *((const float*) in);
	}

	if (normalized)
	{
		switch (component_type)
		{
			// note: glTF spec doesn't currently define normalized conversions for 32-bit integers
			case cgltf_component_type_r_16:
				return *((const int16_t*) in) / (cgltf_float)32767;
			case cgltf_component_type_r_16u:
				return *((const uint16_t*) in) / (cgltf_float)65535;
			case cgltf_component_type_r_8:
				return *((const int8_t*) in) / (cgltf_float)127;
			case cgltf_component_type_r_8u:
				return *((const uint8_t*) in) / (cgltf_float)255;
			default:
				return 0;
		}
	}

	return (cgltf_float)cgltf_component_read_integer(in, component_type);
}

static cgltf_size cgltf_component_size(cgltf_component_type component_type);

static cgltf_bool cgltf_element_read_float(const uint8_t* element, cgltf_type type, cgltf_component_type component_type, cgltf_bool normalized, cgltf_float* out, cgltf_size element_size)
{
	cgltf_size num_components = cgltf_num_components(type);

	if (element_size < num_components) {
		return 0;
	}

	// There are three special cases for component extraction, see #data-alignment in the 2.0 spec.

	cgltf_size component_size = cgltf_component_size(component_type);

	if (type == cgltf_type_mat2 && component_size == 1)
	{
		out[0] = cgltf_component_read_float(element, component_type, normalized);
		out[1] = cgltf_component_read_float(element + 1, component_type, normalized);
		out[2] = cgltf_component_read_float(element + 4, component_type, normalized);
		out[3] = cgltf_component_read_float(element + 5, component_type, normalized);
		return 1;
	}

	if (type == cgltf_type_mat3 && component_size == 1)
	{
		out[0] = cgltf_component_read_float(element, component_type, normalized);
		out[1] = cgltf_component_read_float(element + 1, component_type, normalized);
		out[2] = cgltf_component_read_float(element + 2, component_type, normalized);
		out[3] = cgltf_component_read_float(element + 4, component_type, normalized);
		out[4] = cgltf_component_read_float(element + 5, component_type, normalized);
		out[5] = cgltf_component_read_float(element + 6, component_type, normalized);
		out[6] = cgltf_component_read_float(element + 8, component_type, normalized);
		out[7] = cgltf_component_read_float(element + 9, component_type, normalized);
		out[8] = cgltf_component_read_float(element + 10, component_type, normalized);
		return 1;
	}

	if (type == cgltf_type_mat3 && component_size == 2)
	{
		out[0] = cgltf_component_read_float(element, component_type, normalized);
		out[1] = cgltf_component_read_float(element + 2, component_type, normalized);
		out[2] = cgltf_component_read_float(element + 4, component_type, normalized);
		out[3] = cgltf_component_read_float(element + 8, component_type, normalized);
		out[4] = cgltf_component_read_float(element + 10, component_type, normalized);
		out[5] = cgltf_component_read_float(element + 12, component_type, normalized);
		out[6] = cgltf_component_read_float(element + 16, component_type, normalized);
		out[7] = cgltf_component_read_float(element + 18, component_type, normalized);
		out[8] = cgltf_component_read_float(element + 20, component_type, normalized);
		return 1;
	}

	for (cgltf_size i = 0; i < num_components; ++i)
	{
		out[i] = cgltf_component_read_float(element + component_size * i, component_type, normalized);
	}
	return 1;
}

const uint8_t* cgltf_buffer_view_data(const cgltf_buffer_view* view)
{
	if (view->data)
		return (const uint8_t*)view->data;

	if (!view->buffer->data)
		return NULL;

	const uint8_t* result = (const uint8_t*)view->buffer->data;
	result += view->offset;
	return result;
}

cgltf_bool cgltf_accessor_read_float(const cgltf_accessor* accessor, cgltf_size index, cgltf_float* out, cgltf_size element_size)
{
	if (accessor->is_sparse)
	{
		return 0;
	}
	if (accessor->buffer_view == NULL)
	{
		memset(out, 0, element_size * sizeof(cgltf_float));
		return 1;
	}
	const uint8_t* element = cgltf_buffer_view_data(accessor->buffer_view);
	if (element == NULL)
	{
		return 0;
	}
	element += accessor->offset + accessor->stride * index;
	return cgltf_element_read_float(element, accessor->type, accessor->component_type, accessor->normalized, out, element_size);
}

cgltf_size cgltf_accessor_unpack_floats(const cgltf_accessor* accessor, cgltf_float* out, cgltf_size float_count)
{
	cgltf_size floats_per_element = cgltf_num_components(accessor->type);
	cgltf_size available_floats = accessor->count * floats_per_element;
	if (out == NULL)
	{
		return available_floats;
	}

	float_count = available_floats < float_count ? available_floats : float_count;
	cgltf_size element_count = float_count / floats_per_element;

	// First pass: convert each element in the base accessor.
	cgltf_float* dest = out;
	cgltf_accessor dense = *accessor;
	dense.is_sparse = 0;
	for (cgltf_size index = 0; index < element_count; index++, dest += floats_per_element)
	{
		if (!cgltf_accessor_read_float(&dense, index, dest, floats_per_element))
		{
			return 0;
		}
	}

	// Second pass: write out each element in the sparse accessor.
	if (accessor->is_sparse)
	{
		const cgltf_accessor_sparse* sparse = &dense.sparse;

		const uint8_t* index_data = cgltf_buffer_view_data(sparse->indices_buffer_view);
		const uint8_t* reader_head = cgltf_buffer_view_data(sparse->values_buffer_view);

		if (index_data == NULL || reader_head == NULL)
		{
			return 0;
		}

		index_data += sparse->indices_byte_offset;
		reader_head += sparse->values_byte_offset;

		cgltf_size index_stride = cgltf_component_size(sparse->indices_component_type);
		for (cgltf_size reader_index = 0; reader_index < sparse->count; reader_index++, index_data += index_stride)
		{
			size_t writer_index = cgltf_component_read_index(index_data, sparse->indices_component_type);
			float* writer_head = out + writer_index * floats_per_element;

			if (!cgltf_element_read_float(reader_head, dense.type, dense.component_type, dense.normalized, writer_head, floats_per_element))
			{
				return 0;
			}

			reader_head += dense.stride;
		}
	}

	return element_count * floats_per_element;
}

static cgltf_uint cgltf_component_read_uint(const void* in, cgltf_component_type component_type)
{
	switch (component_type)
	{
		case cgltf_component_type_r_8:
			return *((const int8_t*) in);

		case cgltf_component_type_r_8u:
			return *((const uint8_t*) in);

		case cgltf_component_type_r_16:
			return *((const int16_t*) in);

		case cgltf_component_type_r_16u:
			return *((const uint16_t*) in);

		case cgltf_component_type_r_32u:
			return *((const uint32_t*) in);

		default:
			return 0;
	}
}

static cgltf_bool cgltf_element_read_uint(const uint8_t* element, cgltf_type type, cgltf_component_type component_type, cgltf_uint* out, cgltf_size element_size)
{
	cgltf_size num_components = cgltf_num_components(type);

	if (element_size < num_components)
	{
		return 0;
	}

	// Reading integer matrices is not a valid use case
	if (type == cgltf_type_mat2 || type == cgltf_type_mat3 || type == cgltf_type_mat4)
	{
		return 0;
	}

	cgltf_size component_size = cgltf_component_size(component_type);

	for (cgltf_size i = 0; i < num_components; ++i)
	{
		out[i] = cgltf_component_read_uint(element + component_size * i, component_type);
	}
	return 1;
}

cgltf_bool cgltf_accessor_read_uint(const cgltf_accessor* accessor, cgltf_size index, cgltf_uint* out, cgltf_size element_size)
{
	if (accessor->is_sparse)
	{
		return 0;
	}
	if (accessor->buffer_view == NULL)
	{
		memset(out, 0, element_size * sizeof( cgltf_uint ));
		return 1;
	}
	const uint8_t* element = cgltf_buffer_view_data(accessor->buffer_view);
	if (element == NULL)
	{
		return 0;
	}
	element += accessor->offset + accessor->stride * index;
	return cgltf_element_read_uint(element, accessor->type, accessor->component_type, out, element_size);
}

cgltf_size cgltf_accessor_read_index(const cgltf_accessor* accessor, cgltf_size index)
{
	if (accessor->is_sparse)
	{
		return 0; // This is an error case, but we can't communicate the error with existing interface.
	}
	if (accessor->buffer_view == NULL)
	{
		return 0;
	}
	const uint8_t* element = cgltf_buffer_view_data(accessor->buffer_view);
	if (element == NULL)
	{
		return 0; // This is an error case, but we can't communicate the error with existing interface.
	}
	element += accessor->offset + accessor->stride * index;
	return cgltf_component_read_index(element, accessor->component_type);
}

#define CGLTF_ERROR_JSON -1
#define CGLTF_ERROR_NOMEM -2
#define CGLTF_ERROR_LEGACY -3

#define CGLTF_CHECK_TOKTYPE(tok_, type_) if ((tok_).type != (type_)) { return CGLTF_ERROR_JSON; }
#define CGLTF_CHECK_TOKTYPE_RETTYPE(tok_, type_, ret_) if ((tok_).type != (type_)) { return (ret_)CGLTF_ERROR_JSON; }
#define CGLTF_CHECK_KEY(tok_) if ((tok_).type != JSMN_STRING || (tok_).size == 0) { return CGLTF_ERROR_JSON; } /* checking size for 0 verifies that a value follows the key */

#define CGLTF_PTRINDEX(type, idx) (type*)((cgltf_size)idx + 1)
#define CGLTF_PTRFIXUP(var, data, size) if (var) { if ((cgltf_size)var > size) { return CGLTF_ERROR_JSON; } var = &data[(cgltf_size)var-1]; }
#define CGLTF_PTRFIXUP_REQ(var, data, size) if (!var || (cgltf_size)var > size) { return CGLTF_ERROR_JSON; } var = &data[(cgltf_size)var-1];

static int cgltf_json_strcmp(jsmntok_t const* tok, const uint8_t* json_chunk, const char* str)
{
	CGLTF_CHECK_TOKTYPE(*tok, JSMN_STRING);
	size_t const str_len = strlen(str);
	size_t const name_length = tok->end - tok->start;
	return (str_len == name_length) ? strncmp((const char*)json_chunk + tok->start, str, str_len) : 128;
}

static int cgltf_json_to_int(jsmntok_t const* tok, const uint8_t* json_chunk)
{
	CGLTF_CHECK_TOKTYPE(*tok, JSMN_PRIMITIVE);
	char tmp[128];
	int size = (cgltf_size)(tok->end - tok->start) < sizeof(tmp) ? tok->end - tok->start : (int)(sizeof(tmp) - 1);
	strncpy(tmp, (const char*)json_chunk + tok->start, size);
	tmp[size] = 0;
	return CGLTF_ATOI(tmp);
}

static cgltf_size cgltf_json_to_size(jsmntok_t const* tok, const uint8_t* json_chunk)
{
	CGLTF_CHECK_TOKTYPE_RETTYPE(*tok, JSMN_PRIMITIVE, cgltf_size);
	char tmp[128];
	int size = (cgltf_size)(tok->end - tok->start) < sizeof(tmp) ? tok->end - tok->start : (int)(sizeof(tmp) - 1);
	strncpy(tmp, (const char*)json_chunk + tok->start, size);
	tmp[size] = 0;
	return (cgltf_size)CGLTF_ATOLL(tmp);
}

static cgltf_float cgltf_json_to_float(jsmntok_t const* tok, const uint8_t* json_chunk)
{
	CGLTF_CHECK_TOKTYPE(*tok, JSMN_PRIMITIVE);
	char tmp[128];
	int size = (cgltf_size)(tok->end - tok->start) < sizeof(tmp) ? tok->end - tok->start : (int)(sizeof(tmp) - 1);
	strncpy(tmp, (const char*)json_chunk + tok->start, size);
	tmp[size] = 0;
	return (cgltf_float)CGLTF_ATOF(tmp);
}

static cgltf_bool cgltf_json_to_bool(jsmntok_t const* tok, const uint8_t* json_chunk)
{
	int size = tok->end - tok->start;
	return size == 4 && memcmp(json_chunk + tok->start, "true", 4) == 0;
}

static int cgltf_skip_json(jsmntok_t const* tokens, int i)
{
	int end = i + 1;

	while (i < end)
	{
		switch (tokens[i].type)
		{
		case JSMN_OBJECT:
			end += tokens[i].size * 2;
			break;

		case JSMN_ARRAY:
			end += tokens[i].size;
			break;

		case JSMN_PRIMITIVE:
		case JSMN_STRING:
			break;

		default:
			return -1;
		}

		i++;
	}

	return i;
}

static void cgltf_fill_float_array(float* out_array, int size, float value)
{
	for (int j = 0; j < size; ++j)
	{
		out_array[j] = value;
	}
}

static int cgltf_parse_json_float_array(jsmntok_t const* tokens, int i, const uint8_t* json_chunk, float* out_array, int size)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);
	if (tokens[i].size != size)
	{
		return CGLTF_ERROR_JSON;
	}
	++i;
	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
		out_array[j] = cgltf_json_to_float(tokens + i, json_chunk);
		++i;
	}
	return i;
}

static int cgltf_parse_json_string(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, char** out_string)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_STRING);
	if (*out_string)
	{
		return CGLTF_ERROR_JSON;
	}
	int size = tokens[i].end - tokens[i].start;
	char* result = (char*)options->memory.alloc_func(options->memory.user_data, size + 1);
	if (!result)
	{
		return CGLTF_ERROR_NOMEM;
	}
	strncpy(result, (const char*)json_chunk + tokens[i].start, size);
	result[size] = 0;
	*out_string = result;
	return i + 1;
}

static int cgltf_parse_json_array(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, size_t element_size, void** out_array, cgltf_size* out_size)
{
	(void)json_chunk;
	if (tokens[i].type != JSMN_ARRAY)
	{
		return tokens[i].type == JSMN_OBJECT ? CGLTF_ERROR_LEGACY : CGLTF_ERROR_JSON;
	}
	if (*out_array)
	{
		return CGLTF_ERROR_JSON;
	}
	int size = tokens[i].size;
	void* result = cgltf_calloc(options, element_size, size);
	if (!result)
	{
		return CGLTF_ERROR_NOMEM;
	}
	*out_array = result;
	*out_size = size;
	return i + 1;
}

static int cgltf_parse_json_string_array(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, char*** out_array, cgltf_size* out_size)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(char*), (void**)out_array, out_size);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < *out_size; ++j)
	{
		i = cgltf_parse_json_string(options, tokens, i, json_chunk, j + (*out_array));
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static void cgltf_parse_attribute_type(const char* name, cgltf_attribute_type* out_type, int* out_index)
{
	if (*name == '_')
	{
		*out_type = cgltf_attribute_type_custom;
		return;
	}

	const char* us = strchr(name, '_');
	size_t len = us ? (size_t)(us - name) : strlen(name);

	if (len == 8 && strncmp(name, "POSITION", 8) == 0)
	{
		*out_type = cgltf_attribute_type_position;
	}
	else if (len == 6 && strncmp(name, "NORMAL", 6) == 0)
	{
		*out_type = cgltf_attribute_type_normal;
	}
	else if (len == 7 && strncmp(name, "TANGENT", 7) == 0)
	{
		*out_type = cgltf_attribute_type_tangent;
	}
	else if (len == 8 && strncmp(name, "TEXCOORD", 8) == 0)
	{
		*out_type = cgltf_attribute_type_texcoord;
	}
	else if (len == 5 && strncmp(name, "COLOR", 5) == 0)
	{
		*out_type = cgltf_attribute_type_color;
	}
	else if (len == 6 && strncmp(name, "JOINTS", 6) == 0)
	{
		*out_type = cgltf_attribute_type_joints;
	}
	else if (len == 7 && strncmp(name, "WEIGHTS", 7) == 0)
	{
		*out_type = cgltf_attribute_type_weights;
	}
	else
	{
		*out_type = cgltf_attribute_type_invalid;
	}

	if (us && *out_type != cgltf_attribute_type_invalid)
	{
		*out_index = CGLTF_ATOI(us + 1);
	}
}

static int cgltf_parse_json_attribute_list(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_attribute** out_attributes, cgltf_size* out_attributes_count)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	if (*out_attributes)
	{
		return CGLTF_ERROR_JSON;
	}

	*out_attributes_count = tokens[i].size;
	*out_attributes = (cgltf_attribute*)cgltf_calloc(options, sizeof(cgltf_attribute), *out_attributes_count);
	++i;

	if (!*out_attributes)
	{
		return CGLTF_ERROR_NOMEM;
	}

	for (cgltf_size j = 0; j < *out_attributes_count; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		i = cgltf_parse_json_string(options, tokens, i, json_chunk, &(*out_attributes)[j].name);
		if (i < 0)
		{
			return CGLTF_ERROR_JSON;
		}

		cgltf_parse_attribute_type((*out_attributes)[j].name, &(*out_attributes)[j].type, &(*out_attributes)[j].index);

		(*out_attributes)[j].data = CGLTF_PTRINDEX(cgltf_accessor, cgltf_json_to_int(tokens + i, json_chunk));
		++i;
	}

	return i;
}

static int cgltf_parse_json_extras(jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_extras* out_extras)
{
	(void)json_chunk;
	out_extras->start_offset = tokens[i].start;
	out_extras->end_offset = tokens[i].end;
	i = cgltf_skip_json(tokens, i);
	return i;
}

static int cgltf_parse_json_unprocessed_extension(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_extension* out_extension)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_STRING);
	CGLTF_CHECK_TOKTYPE(tokens[i+1], JSMN_OBJECT);
	if (out_extension->name)
	{
		return CGLTF_ERROR_JSON;
	}

	cgltf_size name_length = tokens[i].end - tokens[i].start;
	out_extension->name = (char*)options->memory.alloc_func(options->memory.user_data, name_length + 1);
	if (!out_extension->name)
	{
		return CGLTF_ERROR_NOMEM;
	}
	strncpy(out_extension->name, (const char*)json_chunk + tokens[i].start, name_length);
	out_extension->name[name_length] = 0;
	i++;

	size_t start = tokens[i].start;
	size_t size = tokens[i].end - start;
	out_extension->data = (char*)options->memory.alloc_func(options->memory.user_data, size + 1);
	if (!out_extension->data)
	{
		return CGLTF_ERROR_NOMEM;
	}
	strncpy(out_extension->data, (const char*)json_chunk + start, size);
	out_extension->data[size] = '\0';

	i = cgltf_skip_json(tokens, i);

	return i;
}

static int cgltf_parse_json_unprocessed_extensions(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_size* out_extensions_count, cgltf_extension** out_extensions)
{
	++i;

	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	if(*out_extensions)
	{
		return CGLTF_ERROR_JSON;
	}

	int extensions_size = tokens[i].size;
	*out_extensions_count = 0;
	*out_extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);

	if (!*out_extensions)
	{
		return CGLTF_ERROR_NOMEM;
	}

	++i;

	for (int j = 0; j < extensions_size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		cgltf_size extension_index = (*out_extensions_count)++;
		cgltf_extension* extension = &((*out_extensions)[extension_index]);
		i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, extension);

		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_draco_mesh_compression(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_draco_mesh_compression* out_draco_mesh_compression)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "attributes") == 0)
		{
			i = cgltf_parse_json_attribute_list(options, tokens, i + 1, json_chunk, &out_draco_mesh_compression->attributes, &out_draco_mesh_compression->attributes_count);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "bufferView") == 0)
		{
			++i;
			out_draco_mesh_compression->buffer_view = CGLTF_PTRINDEX(cgltf_buffer_view, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_mesh_gpu_instancing(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_mesh_gpu_instancing* out_mesh_gpu_instancing)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "attributes") == 0)
		{
			i = cgltf_parse_json_attribute_list(options, tokens, i + 1, json_chunk, &out_mesh_gpu_instancing->attributes, &out_mesh_gpu_instancing->attributes_count);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "bufferView") == 0)
		{
			++i;
			out_mesh_gpu_instancing->buffer_view = CGLTF_PTRINDEX(cgltf_buffer_view, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_material_mapping_data(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_material_mapping* out_mappings, cgltf_size* offset)
{
	(void)options;
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

		int obj_size = tokens[i].size;
		++i;

		int material = -1;
		int variants_tok = -1;
		cgltf_extras extras = {0, 0};

		for (int k = 0; k < obj_size; ++k)
		{
			CGLTF_CHECK_KEY(tokens[i]);

			if (cgltf_json_strcmp(tokens + i, json_chunk, "material") == 0)
			{
				++i;
				material = cgltf_json_to_int(tokens + i, json_chunk);
				++i;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "variants") == 0)
			{
				variants_tok = i+1;
				CGLTF_CHECK_TOKTYPE(tokens[variants_tok], JSMN_ARRAY);

				i = cgltf_skip_json(tokens, i+1);
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
			{
				i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &extras);
			}
			else
			{
				i = cgltf_skip_json(tokens, i+1);
			}

			if (i < 0)
			{
				return i;
			}
		}

		if (material < 0 || variants_tok < 0)
		{
			return CGLTF_ERROR_JSON;
		}

		if (out_mappings)
		{
			for (int k = 0; k < tokens[variants_tok].size; ++k)
			{
				int variant = cgltf_json_to_int(&tokens[variants_tok + 1 + k], json_chunk);
				if (variant < 0)
					return variant;

				out_mappings[*offset].material = CGLTF_PTRINDEX(cgltf_material, material);
				out_mappings[*offset].variant = variant;
				out_mappings[*offset].extras = extras;

				(*offset)++;
			}
		}
		else
		{
			(*offset) += tokens[variants_tok].size;
		}
	}

	return i;
}

static int cgltf_parse_json_material_mappings(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_primitive* out_prim)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "mappings") == 0)
		{
			if (out_prim->mappings)
			{
				return CGLTF_ERROR_JSON;
			}

			cgltf_size mappings_offset = 0;
			int k = cgltf_parse_json_material_mapping_data(options, tokens, i + 1, json_chunk, NULL, &mappings_offset);
			if (k < 0)
			{
				return k;
			}

			out_prim->mappings_count = mappings_offset;
			out_prim->mappings = (cgltf_material_mapping*)cgltf_calloc(options, sizeof(cgltf_material_mapping), out_prim->mappings_count);

			mappings_offset = 0;
			i = cgltf_parse_json_material_mapping_data(options, tokens, i + 1, json_chunk, out_prim->mappings, &mappings_offset);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_primitive(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_primitive* out_prim)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	out_prim->type = cgltf_primitive_type_triangles;

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "mode") == 0)
		{
			++i;
			out_prim->type
					= (cgltf_primitive_type)
					cgltf_json_to_int(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "indices") == 0)
		{
			++i;
			out_prim->indices = CGLTF_PTRINDEX(cgltf_accessor, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "material") == 0)
		{
			++i;
			out_prim->material = CGLTF_PTRINDEX(cgltf_material, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "attributes") == 0)
		{
			i = cgltf_parse_json_attribute_list(options, tokens, i + 1, json_chunk, &out_prim->attributes, &out_prim->attributes_count);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "targets") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_morph_target), (void**)&out_prim->targets, &out_prim->targets_count);
			if (i < 0)
			{
				return i;
			}

			for (cgltf_size k = 0; k < out_prim->targets_count; ++k)
			{
				i = cgltf_parse_json_attribute_list(options, tokens, i, json_chunk, &out_prim->targets[k].attributes, &out_prim->targets[k].attributes_count);
				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_prim->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
			if(out_prim->extensions)
			{
				return CGLTF_ERROR_JSON;
			}

			int extensions_size = tokens[i].size;
			out_prim->extensions_count = 0;
			out_prim->extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);

			if (!out_prim->extensions)
			{
				return CGLTF_ERROR_NOMEM;
			}

			++i;
			for (int k = 0; k < extensions_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_draco_mesh_compression") == 0)
				{
					out_prim->has_draco_mesh_compression = 1;
					i = cgltf_parse_json_draco_mesh_compression(options, tokens, i + 1, json_chunk, &out_prim->draco_mesh_compression);
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_variants") == 0)
				{
					i = cgltf_parse_json_material_mappings(options, tokens, i + 1, json_chunk, out_prim);
				}
				else
				{
					i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, &(out_prim->extensions[out_prim->extensions_count++]));
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_mesh(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_mesh* out_mesh)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_mesh->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "primitives") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_primitive), (void**)&out_mesh->primitives, &out_mesh->primitives_count);
			if (i < 0)
			{
				return i;
			}

			for (cgltf_size prim_index = 0; prim_index < out_mesh->primitives_count; ++prim_index)
			{
				i = cgltf_parse_json_primitive(options, tokens, i, json_chunk, &out_mesh->primitives[prim_index]);
				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "weights") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_float), (void**)&out_mesh->weights, &out_mesh->weights_count);
			if (i < 0)
			{
				return i;
			}

			i = cgltf_parse_json_float_array(tokens, i - 1, json_chunk, out_mesh->weights, (int)out_mesh->weights_count);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			++i;

			out_mesh->extras.start_offset = tokens[i].start;
			out_mesh->extras.end_offset = tokens[i].end;

			if (tokens[i].type == JSMN_OBJECT)
			{
				int extras_size = tokens[i].size;
				++i;

				for (int k = 0; k < extras_size; ++k)
				{
					CGLTF_CHECK_KEY(tokens[i]);

					if (cgltf_json_strcmp(tokens+i, json_chunk, "targetNames") == 0 && tokens[i+1].type == JSMN_ARRAY)
					{
						i = cgltf_parse_json_string_array(options, tokens, i + 1, json_chunk, &out_mesh->target_names, &out_mesh->target_names_count);
					}
					else
					{
						i = cgltf_skip_json(tokens, i+1);
					}

					if (i < 0)
					{
						return i;
					}
				}
			}
			else
			{
				i = cgltf_skip_json(tokens, i);
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_mesh->extensions_count, &out_mesh->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_meshes(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_mesh), (void**)&out_data->meshes, &out_data->meshes_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->meshes_count; ++j)
	{
		i = cgltf_parse_json_mesh(options, tokens, i, json_chunk, &out_data->meshes[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static cgltf_component_type cgltf_json_to_component_type(jsmntok_t const* tok, const uint8_t* json_chunk)
{
	int type = cgltf_json_to_int(tok, json_chunk);

	switch (type)
	{
	case 5120:
		return cgltf_component_type_r_8;
	case 5121:
		return cgltf_component_type_r_8u;
	case 5122:
		return cgltf_component_type_r_16;
	case 5123:
		return cgltf_component_type_r_16u;
	case 5125:
		return cgltf_component_type_r_32u;
	case 5126:
		return cgltf_component_type_r_32f;
	default:
		return cgltf_component_type_invalid;
	}
}

static int cgltf_parse_json_accessor_sparse(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_accessor_sparse* out_sparse)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "count") == 0)
		{
			++i;
			out_sparse->count = cgltf_json_to_int(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "indices") == 0)
		{
			++i;
			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

			int indices_size = tokens[i].size;
			++i;

			for (int k = 0; k < indices_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "bufferView") == 0)
				{
					++i;
					out_sparse->indices_buffer_view = CGLTF_PTRINDEX(cgltf_buffer_view, cgltf_json_to_int(tokens + i, json_chunk));
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteOffset") == 0)
				{
					++i;
					out_sparse->indices_byte_offset = cgltf_json_to_size(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "componentType") == 0)
				{
					++i;
					out_sparse->indices_component_type = cgltf_json_to_component_type(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
				{
					i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_sparse->indices_extras);
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
				{
					i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_sparse->indices_extensions_count, &out_sparse->indices_extensions);
				}
				else
				{
					i = cgltf_skip_json(tokens, i+1);
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "values") == 0)
		{
			++i;
			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

			int values_size = tokens[i].size;
			++i;

			for (int k = 0; k < values_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "bufferView") == 0)
				{
					++i;
					out_sparse->values_buffer_view = CGLTF_PTRINDEX(cgltf_buffer_view, cgltf_json_to_int(tokens + i, json_chunk));
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteOffset") == 0)
				{
					++i;
					out_sparse->values_byte_offset = cgltf_json_to_size(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
				{
					i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_sparse->values_extras);
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
				{
					i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_sparse->values_extensions_count, &out_sparse->values_extensions);
				}
				else
				{
					i = cgltf_skip_json(tokens, i+1);
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_sparse->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_sparse->extensions_count, &out_sparse->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_accessor(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_accessor* out_accessor)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_accessor->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "bufferView") == 0)
		{
			++i;
			out_accessor->buffer_view = CGLTF_PTRINDEX(cgltf_buffer_view, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteOffset") == 0)
		{
			++i;
			out_accessor->offset =
					cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "componentType") == 0)
		{
			++i;
			out_accessor->component_type = cgltf_json_to_component_type(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "normalized") == 0)
		{
			++i;
			out_accessor->normalized = cgltf_json_to_bool(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "count") == 0)
		{
			++i;
			out_accessor->count =
					cgltf_json_to_int(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "type") == 0)
		{
			++i;
			if (cgltf_json_strcmp(tokens+i, json_chunk, "SCALAR") == 0)
			{
				out_accessor->type = cgltf_type_scalar;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "VEC2") == 0)
			{
				out_accessor->type = cgltf_type_vec2;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "VEC3") == 0)
			{
				out_accessor->type = cgltf_type_vec3;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "VEC4") == 0)
			{
				out_accessor->type = cgltf_type_vec4;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "MAT2") == 0)
			{
				out_accessor->type = cgltf_type_mat2;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "MAT3") == 0)
			{
				out_accessor->type = cgltf_type_mat3;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "MAT4") == 0)
			{
				out_accessor->type = cgltf_type_mat4;
			}
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "min") == 0)
		{
			++i;
			out_accessor->has_min = 1;
			// note: we can't parse the precise number of elements since type may not have been computed yet
			int min_size = tokens[i].size > 16 ? 16 : tokens[i].size;
			i = cgltf_parse_json_float_array(tokens, i, json_chunk, out_accessor->min, min_size);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "max") == 0)
		{
			++i;
			out_accessor->has_max = 1;
			// note: we can't parse the precise number of elements since type may not have been computed yet
			int max_size = tokens[i].size > 16 ? 16 : tokens[i].size;
			i = cgltf_parse_json_float_array(tokens, i, json_chunk, out_accessor->max, max_size);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "sparse") == 0)
		{
			out_accessor->is_sparse = 1;
			i = cgltf_parse_json_accessor_sparse(options, tokens, i + 1, json_chunk, &out_accessor->sparse);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_accessor->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_accessor->extensions_count, &out_accessor->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_texture_transform(jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_texture_transform* out_texture_transform)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "offset") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_texture_transform->offset, 2);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "rotation") == 0)
		{
			++i;
			out_texture_transform->rotation = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "scale") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_texture_transform->scale, 2);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "texCoord") == 0)
		{
			++i;
			out_texture_transform->has_texcoord = 1;
			out_texture_transform->texcoord = cgltf_json_to_int(tokens + i, json_chunk);
			++i;
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_texture_view(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_texture_view* out_texture_view)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	out_texture_view->scale = 1.0f;
	cgltf_fill_float_array(out_texture_view->transform.scale, 2, 1.0f);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "index") == 0)
		{
			++i;
			out_texture_view->texture = CGLTF_PTRINDEX(cgltf_texture, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "texCoord") == 0)
		{
			++i;
			out_texture_view->texcoord = cgltf_json_to_int(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "scale") == 0) 
		{
			++i;
			out_texture_view->scale = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "strength") == 0)
		{
			++i;
			out_texture_view->scale = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_texture_view->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
			if(out_texture_view->extensions)
			{
				return CGLTF_ERROR_JSON;
			}

			int extensions_size = tokens[i].size;
			out_texture_view->extensions_count = 0;
			out_texture_view->extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);

			if (!out_texture_view->extensions)
			{
				return CGLTF_ERROR_NOMEM;
			}

			++i;

			for (int k = 0; k < extensions_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_texture_transform") == 0)
				{
					out_texture_view->has_transform = 1;
					i = cgltf_parse_json_texture_transform(tokens, i + 1, json_chunk, &out_texture_view->transform);
				}
				else
				{
					i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, &(out_texture_view->extensions[out_texture_view->extensions_count++]));
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_pbr_metallic_roughness(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_pbr_metallic_roughness* out_pbr)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "metallicFactor") == 0)
		{
			++i;
			out_pbr->metallic_factor = 
				cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "roughnessFactor") == 0) 
		{
			++i;
			out_pbr->roughness_factor =
				cgltf_json_to_float(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "baseColorFactor") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_pbr->base_color_factor, 4);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "baseColorTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk,
				&out_pbr->base_color_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "metallicRoughnessTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk,
				&out_pbr->metallic_roughness_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_pbr->extras);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_pbr_specular_glossiness(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_pbr_specular_glossiness* out_pbr)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "diffuseFactor") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_pbr->diffuse_factor, 4);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "specularFactor") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_pbr->specular_factor, 3);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "glossinessFactor") == 0)
		{
			++i;
			out_pbr->glossiness_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "diffuseTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_pbr->diffuse_texture);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "specularGlossinessTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_pbr->specular_glossiness_texture);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_clearcoat(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_clearcoat* out_clearcoat)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "clearcoatFactor") == 0)
		{
			++i;
			out_clearcoat->clearcoat_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "clearcoatRoughnessFactor") == 0)
		{
			++i;
			out_clearcoat->clearcoat_roughness_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "clearcoatTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_clearcoat->clearcoat_texture);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "clearcoatRoughnessTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_clearcoat->clearcoat_roughness_texture);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "clearcoatNormalTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_clearcoat->clearcoat_normal_texture);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_ior(jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_ior* out_ior)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	// Default values
	out_ior->ior = 1.5f;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "ior") == 0)
		{
			++i;
			out_ior->ior = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_specular(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_specular* out_specular)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	// Default values
	out_specular->specular_factor = 1.0f;
	cgltf_fill_float_array(out_specular->specular_color_factor, 3, 1.0f);

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "specularFactor") == 0)
		{
			++i;
			out_specular->specular_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "specularColorFactor") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_specular->specular_color_factor, 3);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "specularTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_specular->specular_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "specularColorTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_specular->specular_color_texture);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_transmission(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_transmission* out_transmission)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "transmissionFactor") == 0)
		{
			++i;
			out_transmission->transmission_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "transmissionTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_transmission->transmission_texture);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_volume(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_volume* out_volume)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "thicknessFactor") == 0)
		{
			++i;
			out_volume->thickness_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "thicknessTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_volume->thickness_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "attenuationColor") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_volume->attenuation_color, 3);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "attenuationDistance") == 0)
		{
			++i;
			out_volume->attenuation_distance = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_sheen(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_sheen* out_sheen)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "sheenColorFactor") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_sheen->sheen_color_factor, 3);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "sheenColorTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_sheen->sheen_color_texture);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "sheenRoughnessFactor") == 0)
		{
			++i;
			out_sheen->sheen_roughness_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "sheenRoughnessTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_sheen->sheen_roughness_texture);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_emissive_strength(jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_emissive_strength* out_emissive_strength)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	// Default
	out_emissive_strength->emissive_strength = 1.f;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "emissiveStrength") == 0)
		{
			++i;
			out_emissive_strength->emissive_strength = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_iridescence(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_iridescence* out_iridescence)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
	int size = tokens[i].size;
	++i;

	// Default
	out_iridescence->iridescence_ior = 1.3f;
	out_iridescence->iridescence_thickness_min = 100.f;
	out_iridescence->iridescence_thickness_max = 400.f;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "iridescenceFactor") == 0)
		{
			++i;
			out_iridescence->iridescence_factor = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "iridescenceTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_iridescence->iridescence_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "iridescenceIor") == 0)
		{
			++i;
			out_iridescence->iridescence_ior = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "iridescenceThicknessMinimum") == 0)
		{
			++i;
			out_iridescence->iridescence_thickness_min = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "iridescenceThicknessMaximum") == 0)
		{
			++i;
			out_iridescence->iridescence_thickness_max = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "iridescenceThicknessTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk, &out_iridescence->iridescence_thickness_texture);
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_image(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_image* out_image)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j) 
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "uri") == 0) 
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_image->uri);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "bufferView") == 0)
		{
			++i;
			out_image->buffer_view = CGLTF_PTRINDEX(cgltf_buffer_view, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "mimeType") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_image->mime_type);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_image->name);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_image->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_image->extensions_count, &out_image->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_sampler(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_sampler* out_sampler)
{
	(void)options;
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	out_sampler->wrap_s = 10497;
	out_sampler->wrap_t = 10497;

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_sampler->name);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "magFilter") == 0)
		{
			++i;
			out_sampler->mag_filter
				= cgltf_json_to_int(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "minFilter") == 0)
		{
			++i;
			out_sampler->min_filter
				= cgltf_json_to_int(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "wrapS") == 0)
		{
			++i;
			out_sampler->wrap_s
				= cgltf_json_to_int(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "wrapT") == 0) 
		{
			++i;
			out_sampler->wrap_t
				= cgltf_json_to_int(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_sampler->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_sampler->extensions_count, &out_sampler->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_texture(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_texture* out_texture)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_texture->name);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "sampler") == 0)
		{
			++i;
			out_texture->sampler = CGLTF_PTRINDEX(cgltf_sampler, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "source") == 0) 
		{
			++i;
			out_texture->image = CGLTF_PTRINDEX(cgltf_image, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_texture->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
			if (out_texture->extensions)
			{
				return CGLTF_ERROR_JSON;
			}

			int extensions_size = tokens[i].size;
			++i;
			out_texture->extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);
			out_texture->extensions_count = 0;

			if (!out_texture->extensions)
			{
				return CGLTF_ERROR_NOMEM;
			}

			for (int k = 0; k < extensions_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens + i, json_chunk, "KHR_texture_basisu") == 0)
				{
					out_texture->has_basisu = 1;
					++i;
					CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
					int num_properties = tokens[i].size;
					++i;

					for (int t = 0; t < num_properties; ++t)
					{
						CGLTF_CHECK_KEY(tokens[i]);

						if (cgltf_json_strcmp(tokens + i, json_chunk, "source") == 0)
						{
							++i;
							out_texture->basisu_image = CGLTF_PTRINDEX(cgltf_image, cgltf_json_to_int(tokens + i, json_chunk));
							++i;
						}
						else
						{
							i = cgltf_skip_json(tokens, i + 1);
						}
						if (i < 0)
						{
							return i;
						}
					}
				}
				else
				{
					i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, &(out_texture->extensions[out_texture->extensions_count++]));
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_material(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_material* out_material)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	cgltf_fill_float_array(out_material->pbr_metallic_roughness.base_color_factor, 4, 1.0f);
	out_material->pbr_metallic_roughness.metallic_factor = 1.0f;
	out_material->pbr_metallic_roughness.roughness_factor = 1.0f;

	cgltf_fill_float_array(out_material->pbr_specular_glossiness.diffuse_factor, 4, 1.0f);
	cgltf_fill_float_array(out_material->pbr_specular_glossiness.specular_factor, 3, 1.0f);
	out_material->pbr_specular_glossiness.glossiness_factor = 1.0f;

	cgltf_fill_float_array(out_material->volume.attenuation_color, 3, 1.0f);
	out_material->volume.attenuation_distance = FLT_MAX;

	out_material->alpha_cutoff = 0.5f;

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_material->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "pbrMetallicRoughness") == 0)
		{
			out_material->has_pbr_metallic_roughness = 1;
			i = cgltf_parse_json_pbr_metallic_roughness(options, tokens, i + 1, json_chunk, &out_material->pbr_metallic_roughness);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "emissiveFactor") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_material->emissive_factor, 3);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "normalTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk,
				&out_material->normal_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "occlusionTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk,
				&out_material->occlusion_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "emissiveTexture") == 0)
		{
			i = cgltf_parse_json_texture_view(options, tokens, i + 1, json_chunk,
				&out_material->emissive_texture);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "alphaMode") == 0)
		{
			++i;
			if (cgltf_json_strcmp(tokens + i, json_chunk, "OPAQUE") == 0)
			{
				out_material->alpha_mode = cgltf_alpha_mode_opaque;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "MASK") == 0)
			{
				out_material->alpha_mode = cgltf_alpha_mode_mask;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "BLEND") == 0)
			{
				out_material->alpha_mode = cgltf_alpha_mode_blend;
			}
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "alphaCutoff") == 0)
		{
			++i;
			out_material->alpha_cutoff = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "doubleSided") == 0)
		{
			++i;
			out_material->double_sided =
				cgltf_json_to_bool(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_material->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
			if(out_material->extensions)
			{
				return CGLTF_ERROR_JSON;
			}

			int extensions_size = tokens[i].size;
			++i;
			out_material->extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);
			out_material->extensions_count= 0;

			if (!out_material->extensions)
			{
				return CGLTF_ERROR_NOMEM;
			}

			for (int k = 0; k < extensions_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_pbrSpecularGlossiness") == 0)
				{
					out_material->has_pbr_specular_glossiness = 1;
					i = cgltf_parse_json_pbr_specular_glossiness(options, tokens, i + 1, json_chunk, &out_material->pbr_specular_glossiness);
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_unlit") == 0)
				{
					out_material->unlit = 1;
					i = cgltf_skip_json(tokens, i+1);
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_clearcoat") == 0)
				{
					out_material->has_clearcoat = 1;
					i = cgltf_parse_json_clearcoat(options, tokens, i + 1, json_chunk, &out_material->clearcoat);
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_ior") == 0)
				{
					out_material->has_ior = 1;
					i = cgltf_parse_json_ior(tokens, i + 1, json_chunk, &out_material->ior);
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_specular") == 0)
				{
					out_material->has_specular = 1;
					i = cgltf_parse_json_specular(options, tokens, i + 1, json_chunk, &out_material->specular);
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_transmission") == 0)
				{
					out_material->has_transmission = 1;
					i = cgltf_parse_json_transmission(options, tokens, i + 1, json_chunk, &out_material->transmission);
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "KHR_materials_volume") == 0)
				{
					out_material->has_volume = 1;
					i = cgltf_parse_json_volume(options, tokens, i + 1, json_chunk, &out_material->volume);
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_sheen") == 0)
				{
					out_material->has_sheen = 1;
					i = cgltf_parse_json_sheen(options, tokens, i + 1, json_chunk, &out_material->sheen);
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "KHR_materials_emissive_strength") == 0)
				{
					out_material->has_emissive_strength = 1;
					i = cgltf_parse_json_emissive_strength(tokens, i + 1, json_chunk, &out_material->emissive_strength);
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "KHR_materials_iridescence") == 0)
				{
					out_material->has_iridescence = 1;
					i = cgltf_parse_json_iridescence(options, tokens, i + 1, json_chunk, &out_material->iridescence);
				}
				else
				{
					i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, &(out_material->extensions[out_material->extensions_count++]));
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_accessors(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_accessor), (void**)&out_data->accessors, &out_data->accessors_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->accessors_count; ++j)
	{
		i = cgltf_parse_json_accessor(options, tokens, i, json_chunk, &out_data->accessors[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_materials(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_material), (void**)&out_data->materials, &out_data->materials_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->materials_count; ++j)
	{
		i = cgltf_parse_json_material(options, tokens, i, json_chunk, &out_data->materials[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_images(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_image), (void**)&out_data->images, &out_data->images_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->images_count; ++j)
	{
		i = cgltf_parse_json_image(options, tokens, i, json_chunk, &out_data->images[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_textures(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_texture), (void**)&out_data->textures, &out_data->textures_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->textures_count; ++j)
	{
		i = cgltf_parse_json_texture(options, tokens, i, json_chunk, &out_data->textures[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_samplers(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_sampler), (void**)&out_data->samplers, &out_data->samplers_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->samplers_count; ++j)
	{
		i = cgltf_parse_json_sampler(options, tokens, i, json_chunk, &out_data->samplers[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_meshopt_compression(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_meshopt_compression* out_meshopt_compression)
{
	(void)options;
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "buffer") == 0)
		{
			++i;
			out_meshopt_compression->buffer = CGLTF_PTRINDEX(cgltf_buffer, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteOffset") == 0)
		{
			++i;
			out_meshopt_compression->offset = cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteLength") == 0)
		{
			++i;
			out_meshopt_compression->size = cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteStride") == 0)
		{
			++i;
			out_meshopt_compression->stride = cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "count") == 0)
		{
			++i;
			out_meshopt_compression->count = cgltf_json_to_int(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "mode") == 0)
		{
			++i;
			if (cgltf_json_strcmp(tokens+i, json_chunk, "ATTRIBUTES") == 0)
			{
				out_meshopt_compression->mode = cgltf_meshopt_compression_mode_attributes;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "TRIANGLES") == 0)
			{
				out_meshopt_compression->mode = cgltf_meshopt_compression_mode_triangles;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "INDICES") == 0)
			{
				out_meshopt_compression->mode = cgltf_meshopt_compression_mode_indices;
			}
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "filter") == 0)
		{
			++i;
			if (cgltf_json_strcmp(tokens+i, json_chunk, "NONE") == 0)
			{
				out_meshopt_compression->filter = cgltf_meshopt_compression_filter_none;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "OCTAHEDRAL") == 0)
			{
				out_meshopt_compression->filter = cgltf_meshopt_compression_filter_octahedral;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "QUATERNION") == 0)
			{
				out_meshopt_compression->filter = cgltf_meshopt_compression_filter_quaternion;
			}
			else if (cgltf_json_strcmp(tokens+i, json_chunk, "EXPONENTIAL") == 0)
			{
				out_meshopt_compression->filter = cgltf_meshopt_compression_filter_exponential;
			}
			++i;
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_buffer_view(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_buffer_view* out_buffer_view)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_buffer_view->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "buffer") == 0)
		{
			++i;
			out_buffer_view->buffer = CGLTF_PTRINDEX(cgltf_buffer, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteOffset") == 0)
		{
			++i;
			out_buffer_view->offset =
					cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteLength") == 0)
		{
			++i;
			out_buffer_view->size =
					cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteStride") == 0)
		{
			++i;
			out_buffer_view->stride =
					cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "target") == 0)
		{
			++i;
			int type = cgltf_json_to_int(tokens+i, json_chunk);
			switch (type)
			{
			case 34962:
				type = cgltf_buffer_view_type_vertices;
				break;
			case 34963:
				type = cgltf_buffer_view_type_indices;
				break;
			default:
				type = cgltf_buffer_view_type_invalid;
				break;
			}
			out_buffer_view->type = (cgltf_buffer_view_type)type;
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_buffer_view->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
			if(out_buffer_view->extensions)
			{
				return CGLTF_ERROR_JSON;
			}

			int extensions_size = tokens[i].size;
			out_buffer_view->extensions_count = 0;
			out_buffer_view->extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);

			if (!out_buffer_view->extensions)
			{
				return CGLTF_ERROR_NOMEM;
			}

			++i;
			for (int k = 0; k < extensions_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "EXT_meshopt_compression") == 0)
				{
					out_buffer_view->has_meshopt_compression = 1;
					i = cgltf_parse_json_meshopt_compression(options, tokens, i + 1, json_chunk, &out_buffer_view->meshopt_compression);
				}
				else
				{
					i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, &(out_buffer_view->extensions[out_buffer_view->extensions_count++]));
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_buffer_views(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_buffer_view), (void**)&out_data->buffer_views, &out_data->buffer_views_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->buffer_views_count; ++j)
	{
		i = cgltf_parse_json_buffer_view(options, tokens, i, json_chunk, &out_data->buffer_views[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_buffer(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_buffer* out_buffer)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_buffer->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "byteLength") == 0)
		{
			++i;
			out_buffer->size =
					cgltf_json_to_size(tokens+i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "uri") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_buffer->uri);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_buffer->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_buffer->extensions_count, &out_buffer->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_buffers(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_buffer), (void**)&out_data->buffers, &out_data->buffers_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->buffers_count; ++j)
	{
		i = cgltf_parse_json_buffer(options, tokens, i, json_chunk, &out_data->buffers[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_skin(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_skin* out_skin)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_skin->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "joints") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_node*), (void**)&out_skin->joints, &out_skin->joints_count);
			if (i < 0)
			{
				return i;
			}

			for (cgltf_size k = 0; k < out_skin->joints_count; ++k)
			{
				out_skin->joints[k] = CGLTF_PTRINDEX(cgltf_node, cgltf_json_to_int(tokens + i, json_chunk));
				++i;
			}
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "skeleton") == 0)
		{
			++i;
			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
			out_skin->skeleton = CGLTF_PTRINDEX(cgltf_node, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "inverseBindMatrices") == 0)
		{
			++i;
			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
			out_skin->inverse_bind_matrices = CGLTF_PTRINDEX(cgltf_accessor, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_skin->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_skin->extensions_count, &out_skin->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_skins(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_skin), (void**)&out_data->skins, &out_data->skins_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->skins_count; ++j)
	{
		i = cgltf_parse_json_skin(options, tokens, i, json_chunk, &out_data->skins[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_camera(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_camera* out_camera)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_camera->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "type") == 0)
		{
			++i;
			if (cgltf_json_strcmp(tokens + i, json_chunk, "perspective") == 0)
			{
				out_camera->type = cgltf_camera_type_perspective;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "orthographic") == 0)
			{
				out_camera->type = cgltf_camera_type_orthographic;
			}
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "perspective") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

			int data_size = tokens[i].size;
			++i;

			out_camera->type = cgltf_camera_type_perspective;

			for (int k = 0; k < data_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "aspectRatio") == 0)
				{
					++i;
					out_camera->data.perspective.has_aspect_ratio = 1;
					out_camera->data.perspective.aspect_ratio = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "yfov") == 0)
				{
					++i;
					out_camera->data.perspective.yfov = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "zfar") == 0)
				{
					++i;
					out_camera->data.perspective.has_zfar = 1;
					out_camera->data.perspective.zfar = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "znear") == 0)
				{
					++i;
					out_camera->data.perspective.znear = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
				{
					i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_camera->data.perspective.extras);
				}
				else
				{
					i = cgltf_skip_json(tokens, i+1);
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "orthographic") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

			int data_size = tokens[i].size;
			++i;

			out_camera->type = cgltf_camera_type_orthographic;

			for (int k = 0; k < data_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "xmag") == 0)
				{
					++i;
					out_camera->data.orthographic.xmag = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "ymag") == 0)
				{
					++i;
					out_camera->data.orthographic.ymag = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "zfar") == 0)
				{
					++i;
					out_camera->data.orthographic.zfar = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "znear") == 0)
				{
					++i;
					out_camera->data.orthographic.znear = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
				{
					i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_camera->data.orthographic.extras);
				}
				else
				{
					i = cgltf_skip_json(tokens, i+1);
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_camera->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_camera->extensions_count, &out_camera->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_cameras(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_camera), (void**)&out_data->cameras, &out_data->cameras_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->cameras_count; ++j)
	{
		i = cgltf_parse_json_camera(options, tokens, i, json_chunk, &out_data->cameras[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_light(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_light* out_light)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	out_light->color[0] = 1.f;
	out_light->color[1] = 1.f;
	out_light->color[2] = 1.f;
	out_light->intensity = 1.f;

	out_light->spot_inner_cone_angle = 0.f;
	out_light->spot_outer_cone_angle = 3.1415926535f / 4.0f;

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_light->name);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "color") == 0)
		{
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_light->color, 3);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "intensity") == 0)
		{
			++i;
			out_light->intensity = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "type") == 0)
		{
			++i;
			if (cgltf_json_strcmp(tokens + i, json_chunk, "directional") == 0)
			{
				out_light->type = cgltf_light_type_directional;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "point") == 0)
			{
				out_light->type = cgltf_light_type_point;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "spot") == 0)
			{
				out_light->type = cgltf_light_type_spot;
			}
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "range") == 0)
		{
			++i;
			out_light->range = cgltf_json_to_float(tokens + i, json_chunk);
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "spot") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

			int data_size = tokens[i].size;
			++i;

			for (int k = 0; k < data_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "innerConeAngle") == 0)
				{
					++i;
					out_light->spot_inner_cone_angle = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "outerConeAngle") == 0)
				{
					++i;
					out_light->spot_outer_cone_angle = cgltf_json_to_float(tokens + i, json_chunk);
					++i;
				}
				else
				{
					i = cgltf_skip_json(tokens, i+1);
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_light->extras);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_lights(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_light), (void**)&out_data->lights, &out_data->lights_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->lights_count; ++j)
	{
		i = cgltf_parse_json_light(options, tokens, i, json_chunk, &out_data->lights[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_node(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_node* out_node)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	out_node->rotation[3] = 1.0f;
	out_node->scale[0] = 1.0f;
	out_node->scale[1] = 1.0f;
	out_node->scale[2] = 1.0f;
	out_node->matrix[0] = 1.0f;
	out_node->matrix[5] = 1.0f;
	out_node->matrix[10] = 1.0f;
	out_node->matrix[15] = 1.0f;

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_node->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "children") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_node*), (void**)&out_node->children, &out_node->children_count);
			if (i < 0)
			{
				return i;
			}

			for (cgltf_size k = 0; k < out_node->children_count; ++k)
			{
				out_node->children[k] = CGLTF_PTRINDEX(cgltf_node, cgltf_json_to_int(tokens + i, json_chunk));
				++i;
			}
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "mesh") == 0)
		{
			++i;
			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
			out_node->mesh = CGLTF_PTRINDEX(cgltf_mesh, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "skin") == 0)
		{
			++i;
			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
			out_node->skin = CGLTF_PTRINDEX(cgltf_skin, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "camera") == 0)
		{
			++i;
			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
			out_node->camera = CGLTF_PTRINDEX(cgltf_camera, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "translation") == 0)
		{
			out_node->has_translation = 1;
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_node->translation, 3);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "rotation") == 0)
		{
			out_node->has_rotation = 1;
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_node->rotation, 4);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "scale") == 0)
		{
			out_node->has_scale = 1;
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_node->scale, 3);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "matrix") == 0)
		{
			out_node->has_matrix = 1;
			i = cgltf_parse_json_float_array(tokens, i + 1, json_chunk, out_node->matrix, 16);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "weights") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_float), (void**)&out_node->weights, &out_node->weights_count);
			if (i < 0)
			{
				return i;
			}

			i = cgltf_parse_json_float_array(tokens, i - 1, json_chunk, out_node->weights, (int)out_node->weights_count);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_node->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
			if(out_node->extensions)
			{
				return CGLTF_ERROR_JSON;
			}

			int extensions_size = tokens[i].size;
			out_node->extensions_count= 0;
			out_node->extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);

			if (!out_node->extensions)
			{
				return CGLTF_ERROR_NOMEM;
			}

			++i;

			for (int k = 0; k < extensions_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_lights_punctual") == 0)
				{
					++i;

					CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

					int data_size = tokens[i].size;
					++i;

					for (int m = 0; m < data_size; ++m)
					{
						CGLTF_CHECK_KEY(tokens[i]);

						if (cgltf_json_strcmp(tokens + i, json_chunk, "light") == 0)
						{
							++i;
							CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
							out_node->light = CGLTF_PTRINDEX(cgltf_light, cgltf_json_to_int(tokens + i, json_chunk));
							++i;
						}
						else
						{
							i = cgltf_skip_json(tokens, i + 1);
						}

						if (i < 0)
						{
							return i;
						}
					}
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "EXT_mesh_gpu_instancing") == 0)
				{
					out_node->has_mesh_gpu_instancing = 1;
					i = cgltf_parse_json_mesh_gpu_instancing(options, tokens, i + 1, json_chunk, &out_node->mesh_gpu_instancing);
				}
				else
				{
					i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, &(out_node->extensions[out_node->extensions_count++]));
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_nodes(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_node), (void**)&out_data->nodes, &out_data->nodes_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->nodes_count; ++j)
	{
		i = cgltf_parse_json_node(options, tokens, i, json_chunk, &out_data->nodes[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_scene(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_scene* out_scene)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_scene->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "nodes") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_node*), (void**)&out_scene->nodes, &out_scene->nodes_count);
			if (i < 0)
			{
				return i;
			}

			for (cgltf_size k = 0; k < out_scene->nodes_count; ++k)
			{
				out_scene->nodes[k] = CGLTF_PTRINDEX(cgltf_node, cgltf_json_to_int(tokens + i, json_chunk));
				++i;
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_scene->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_scene->extensions_count, &out_scene->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_scenes(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_scene), (void**)&out_data->scenes, &out_data->scenes_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->scenes_count; ++j)
	{
		i = cgltf_parse_json_scene(options, tokens, i, json_chunk, &out_data->scenes[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_animation_sampler(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_animation_sampler* out_sampler)
{
	(void)options;
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "input") == 0)
		{
			++i;
			out_sampler->input = CGLTF_PTRINDEX(cgltf_accessor, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "output") == 0)
		{
			++i;
			out_sampler->output = CGLTF_PTRINDEX(cgltf_accessor, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "interpolation") == 0)
		{
			++i;
			if (cgltf_json_strcmp(tokens + i, json_chunk, "LINEAR") == 0)
			{
				out_sampler->interpolation = cgltf_interpolation_type_linear;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "STEP") == 0)
			{
				out_sampler->interpolation = cgltf_interpolation_type_step;
			}
			else if (cgltf_json_strcmp(tokens + i, json_chunk, "CUBICSPLINE") == 0)
			{
				out_sampler->interpolation = cgltf_interpolation_type_cubic_spline;
			}
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_sampler->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_sampler->extensions_count, &out_sampler->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_animation_channel(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_animation_channel* out_channel)
{
	(void)options;
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "sampler") == 0)
		{
			++i;
			out_channel->sampler = CGLTF_PTRINDEX(cgltf_animation_sampler, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "target") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

			int target_size = tokens[i].size;
			++i;

			for (int k = 0; k < target_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "node") == 0)
				{
					++i;
					out_channel->target_node = CGLTF_PTRINDEX(cgltf_node, cgltf_json_to_int(tokens + i, json_chunk));
					++i;
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "path") == 0)
				{
					++i;
					if (cgltf_json_strcmp(tokens+i, json_chunk, "translation") == 0)
					{
						out_channel->target_path = cgltf_animation_path_type_translation;
					}
					else if (cgltf_json_strcmp(tokens+i, json_chunk, "rotation") == 0)
					{
						out_channel->target_path = cgltf_animation_path_type_rotation;
					}
					else if (cgltf_json_strcmp(tokens+i, json_chunk, "scale") == 0)
					{
						out_channel->target_path = cgltf_animation_path_type_scale;
					}
					else if (cgltf_json_strcmp(tokens+i, json_chunk, "weights") == 0)
					{
						out_channel->target_path = cgltf_animation_path_type_weights;
					}
					++i;
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
				{
					i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_channel->extras);
				}
				else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
				{
					i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_channel->extensions_count, &out_channel->extensions);
				}
				else
				{
					i = cgltf_skip_json(tokens, i+1);
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_animation(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_animation* out_animation)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_animation->name);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "samplers") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_animation_sampler), (void**)&out_animation->samplers, &out_animation->samplers_count);
			if (i < 0)
			{
				return i;
			}

			for (cgltf_size k = 0; k < out_animation->samplers_count; ++k)
			{
				i = cgltf_parse_json_animation_sampler(options, tokens, i, json_chunk, &out_animation->samplers[k]);
				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "channels") == 0)
		{
			i = cgltf_parse_json_array(options, tokens, i + 1, json_chunk, sizeof(cgltf_animation_channel), (void**)&out_animation->channels, &out_animation->channels_count);
			if (i < 0)
			{
				return i;
			}

			for (cgltf_size k = 0; k < out_animation->channels_count; ++k)
			{
				i = cgltf_parse_json_animation_channel(options, tokens, i, json_chunk, &out_animation->channels[k]);
				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_animation->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_animation->extensions_count, &out_animation->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_animations(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_animation), (void**)&out_data->animations, &out_data->animations_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->animations_count; ++j)
	{
		i = cgltf_parse_json_animation(options, tokens, i, json_chunk, &out_data->animations[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_variant(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_material_variant* out_variant)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "name") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_variant->name);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_variant->extras);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

static int cgltf_parse_json_variants(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	i = cgltf_parse_json_array(options, tokens, i, json_chunk, sizeof(cgltf_material_variant), (void**)&out_data->variants, &out_data->variants_count);
	if (i < 0)
	{
		return i;
	}

	for (cgltf_size j = 0; j < out_data->variants_count; ++j)
	{
		i = cgltf_parse_json_variant(options, tokens, i, json_chunk, &out_data->variants[j]);
		if (i < 0)
		{
			return i;
		}
	}
	return i;
}

static int cgltf_parse_json_asset(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_asset* out_asset)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens+i, json_chunk, "copyright") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_asset->copyright);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "generator") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_asset->generator);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "version") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_asset->version);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "minVersion") == 0)
		{
			i = cgltf_parse_json_string(options, tokens, i + 1, json_chunk, &out_asset->min_version);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_asset->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			i = cgltf_parse_json_unprocessed_extensions(options, tokens, i, json_chunk, &out_asset->extensions_count, &out_asset->extensions);
		}
		else
		{
			i = cgltf_skip_json(tokens, i+1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	if (out_asset->version && CGLTF_ATOF(out_asset->version) < 2)
	{
		return CGLTF_ERROR_LEGACY;
	}

	return i;
}

cgltf_size cgltf_num_components(cgltf_type type) {
	switch (type)
	{
	case cgltf_type_vec2:
		return 2;
	case cgltf_type_vec3:
		return 3;
	case cgltf_type_vec4:
		return 4;
	case cgltf_type_mat2:
		return 4;
	case cgltf_type_mat3:
		return 9;
	case cgltf_type_mat4:
		return 16;
	case cgltf_type_invalid:
	case cgltf_type_scalar:
	default:
		return 1;
	}
}

static cgltf_size cgltf_component_size(cgltf_component_type component_type) {
	switch (component_type)
	{
	case cgltf_component_type_r_8:
	case cgltf_component_type_r_8u:
		return 1;
	case cgltf_component_type_r_16:
	case cgltf_component_type_r_16u:
		return 2;
	case cgltf_component_type_r_32u:
	case cgltf_component_type_r_32f:
		return 4;
	case cgltf_component_type_invalid:
	default:
		return 0;
	}
}

static cgltf_size cgltf_calc_size(cgltf_type type, cgltf_component_type component_type)
{
	cgltf_size component_size = cgltf_component_size(component_type);
	if (type == cgltf_type_mat2 && component_size == 1)
	{
		return 8 * component_size;
	}
	else if (type == cgltf_type_mat3 && (component_size == 1 || component_size == 2))
	{
		return 12 * component_size;
	}
	return component_size * cgltf_num_components(type);
}

static int cgltf_fixup_pointers(cgltf_data* out_data);

static int cgltf_parse_json_root(cgltf_options* options, jsmntok_t const* tokens, int i, const uint8_t* json_chunk, cgltf_data* out_data)
{
	CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

	int size = tokens[i].size;
	++i;

	for (int j = 0; j < size; ++j)
	{
		CGLTF_CHECK_KEY(tokens[i]);

		if (cgltf_json_strcmp(tokens + i, json_chunk, "asset") == 0)
		{
			i = cgltf_parse_json_asset(options, tokens, i + 1, json_chunk, &out_data->asset);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "meshes") == 0)
		{
			i = cgltf_parse_json_meshes(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "accessors") == 0)
		{
			i = cgltf_parse_json_accessors(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "bufferViews") == 0)
		{
			i = cgltf_parse_json_buffer_views(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "buffers") == 0)
		{
			i = cgltf_parse_json_buffers(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "materials") == 0)
		{
			i = cgltf_parse_json_materials(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "images") == 0)
		{
			i = cgltf_parse_json_images(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "textures") == 0)
		{
			i = cgltf_parse_json_textures(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "samplers") == 0)
		{
			i = cgltf_parse_json_samplers(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "skins") == 0)
		{
			i = cgltf_parse_json_skins(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "cameras") == 0)
		{
			i = cgltf_parse_json_cameras(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "nodes") == 0)
		{
			i = cgltf_parse_json_nodes(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "scenes") == 0)
		{
			i = cgltf_parse_json_scenes(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "scene") == 0)
		{
			++i;
			out_data->scene = CGLTF_PTRINDEX(cgltf_scene, cgltf_json_to_int(tokens + i, json_chunk));
			++i;
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "animations") == 0)
		{
			i = cgltf_parse_json_animations(options, tokens, i + 1, json_chunk, out_data);
		}
		else if (cgltf_json_strcmp(tokens+i, json_chunk, "extras") == 0)
		{
			i = cgltf_parse_json_extras(tokens, i + 1, json_chunk, &out_data->extras);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensions") == 0)
		{
			++i;

			CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);
			if(out_data->data_extensions)
			{
				return CGLTF_ERROR_JSON;
			}

			int extensions_size = tokens[i].size;
			out_data->data_extensions_count = 0;
			out_data->data_extensions = (cgltf_extension*)cgltf_calloc(options, sizeof(cgltf_extension), extensions_size);

			if (!out_data->data_extensions)
			{
				return CGLTF_ERROR_NOMEM;
			}

			++i;

			for (int k = 0; k < extensions_size; ++k)
			{
				CGLTF_CHECK_KEY(tokens[i]);

				if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_lights_punctual") == 0)
				{
					++i;

					CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

					int data_size = tokens[i].size;
					++i;

					for (int m = 0; m < data_size; ++m)
					{
						CGLTF_CHECK_KEY(tokens[i]);

						if (cgltf_json_strcmp(tokens + i, json_chunk, "lights") == 0)
						{
							i = cgltf_parse_json_lights(options, tokens, i + 1, json_chunk, out_data);
						}
						else
						{
							i = cgltf_skip_json(tokens, i + 1);
						}

						if (i < 0)
						{
							return i;
						}
					}
				}
				else if (cgltf_json_strcmp(tokens+i, json_chunk, "KHR_materials_variants") == 0)
				{
					++i;

					CGLTF_CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

					int data_size = tokens[i].size;
					++i;

					for (int m = 0; m < data_size; ++m)
					{
						CGLTF_CHECK_KEY(tokens[i]);

						if (cgltf_json_strcmp(tokens + i, json_chunk, "variants") == 0)
						{
							i = cgltf_parse_json_variants(options, tokens, i + 1, json_chunk, out_data);
						}
						else
						{
							i = cgltf_skip_json(tokens, i + 1);
						}

						if (i < 0)
						{
							return i;
						}
					}
				}
				else
				{
					i = cgltf_parse_json_unprocessed_extension(options, tokens, i, json_chunk, &(out_data->data_extensions[out_data->data_extensions_count++]));
				}

				if (i < 0)
				{
					return i;
				}
			}
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensionsUsed") == 0)
		{
			i = cgltf_parse_json_string_array(options, tokens, i + 1, json_chunk, &out_data->extensions_used, &out_data->extensions_used_count);
		}
		else if (cgltf_json_strcmp(tokens + i, json_chunk, "extensionsRequired") == 0)
		{
			i = cgltf_parse_json_string_array(options, tokens, i + 1, json_chunk, &out_data->extensions_required, &out_data->extensions_required_count);
		}
		else
		{
			i = cgltf_skip_json(tokens, i + 1);
		}

		if (i < 0)
		{
			return i;
		}
	}

	return i;
}

cgltf_result cgltf_parse_json(cgltf_options* options, const uint8_t* json_chunk, cgltf_size size, cgltf_data** out_data)
{
	jsmn_parser parser = { 0, 0, 0 };

	if (options->json_token_count == 0)
	{
		int token_count = jsmn_parse(&parser, (const char*)json_chunk, size, NULL, 0);

		if (token_count <= 0)
		{
			return cgltf_result_invalid_json;
		}

		options->json_token_count = token_count;
	}

	jsmntok_t* tokens = (jsmntok_t*)options->memory.alloc_func(options->memory.user_data, sizeof(jsmntok_t) * (options->json_token_count + 1));

	if (!tokens)
	{
		return cgltf_result_out_of_memory;
	}

	jsmn_init(&parser);

	int token_count = jsmn_parse(&parser, (const char*)json_chunk, size, tokens, options->json_token_count);

	if (token_count <= 0)
	{
		options->memory.free_func(options->memory.user_data, tokens);
		return cgltf_result_invalid_json;
	}

	// this makes sure that we always have an UNDEFINED token at the end of the stream
	// for invalid JSON inputs this makes sure we don't perform out of bound reads of token data
	tokens[token_count].type = JSMN_UNDEFINED;

	cgltf_data* data = (cgltf_data*)options->memory.alloc_func(options->memory.user_data, sizeof(cgltf_data));

	if (!data)
	{
		options->memory.free_func(options->memory.user_data, tokens);
		return cgltf_result_out_of_memory;
	}

	memset(data, 0, sizeof(cgltf_data));
	data->memory = options->memory;
	data->file = options->file;

	int i = cgltf_parse_json_root(options, tokens, 0, json_chunk, data);

	options->memory.free_func(options->memory.user_data, tokens);

	if (i < 0)
	{
		cgltf_free(data);

		switch (i)
		{
		case CGLTF_ERROR_NOMEM: return cgltf_result_out_of_memory;
		case CGLTF_ERROR_LEGACY: return cgltf_result_legacy_gltf;
		default: return cgltf_result_invalid_gltf;
		}
	}

	if (cgltf_fixup_pointers(data) < 0)
	{
		cgltf_free(data);
		return cgltf_result_invalid_gltf;
	}

	data->json = (const char*)json_chunk;
	data->json_size = size;

	*out_data = data;

	return cgltf_result_success;
}

static int cgltf_fixup_pointers(cgltf_data* data)
{
	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		for (cgltf_size j = 0; j < data->meshes[i].primitives_count; ++j)
		{
			CGLTF_PTRFIXUP(data->meshes[i].primitives[j].indices, data->accessors, data->accessors_count);
			CGLTF_PTRFIXUP(data->meshes[i].primitives[j].material, data->materials, data->materials_count);

			for (cgltf_size k = 0; k < data->meshes[i].primitives[j].attributes_count; ++k)
			{
				CGLTF_PTRFIXUP_REQ(data->meshes[i].primitives[j].attributes[k].data, data->accessors, data->accessors_count);
			}

			for (cgltf_size k = 0; k < data->meshes[i].primitives[j].targets_count; ++k)
			{
				for (cgltf_size m = 0; m < data->meshes[i].primitives[j].targets[k].attributes_count; ++m)
				{
					CGLTF_PTRFIXUP_REQ(data->meshes[i].primitives[j].targets[k].attributes[m].data, data->accessors, data->accessors_count);
				}
			}

			if (data->meshes[i].primitives[j].has_draco_mesh_compression)
			{
				CGLTF_PTRFIXUP_REQ(data->meshes[i].primitives[j].draco_mesh_compression.buffer_view, data->buffer_views, data->buffer_views_count);
				for (cgltf_size m = 0; m < data->meshes[i].primitives[j].draco_mesh_compression.attributes_count; ++m)
				{
					CGLTF_PTRFIXUP_REQ(data->meshes[i].primitives[j].draco_mesh_compression.attributes[m].data, data->accessors, data->accessors_count);
				}
			}

			for (cgltf_size k = 0; k < data->meshes[i].primitives[j].mappings_count; ++k)
			{
				CGLTF_PTRFIXUP_REQ(data->meshes[i].primitives[j].mappings[k].material, data->materials, data->materials_count);
			}
		}
	}

	for (cgltf_size i = 0; i < data->accessors_count; ++i)
	{
		CGLTF_PTRFIXUP(data->accessors[i].buffer_view, data->buffer_views, data->buffer_views_count);

		if (data->accessors[i].is_sparse)
		{
			CGLTF_PTRFIXUP_REQ(data->accessors[i].sparse.indices_buffer_view, data->buffer_views, data->buffer_views_count);
			CGLTF_PTRFIXUP_REQ(data->accessors[i].sparse.values_buffer_view, data->buffer_views, data->buffer_views_count);
		}

		if (data->accessors[i].buffer_view)
		{
			data->accessors[i].stride = data->accessors[i].buffer_view->stride;
		}

		if (data->accessors[i].stride == 0)
		{
			data->accessors[i].stride = cgltf_calc_size(data->accessors[i].type, data->accessors[i].component_type);
		}
	}

	for (cgltf_size i = 0; i < data->textures_count; ++i)
	{
		CGLTF_PTRFIXUP(data->textures[i].image, data->images, data->images_count);
		CGLTF_PTRFIXUP(data->textures[i].basisu_image, data->images, data->images_count);
		CGLTF_PTRFIXUP(data->textures[i].sampler, data->samplers, data->samplers_count);
	}

	for (cgltf_size i = 0; i < data->images_count; ++i)
	{
		CGLTF_PTRFIXUP(data->images[i].buffer_view, data->buffer_views, data->buffer_views_count);
	}

	for (cgltf_size i = 0; i < data->materials_count; ++i)
	{
		CGLTF_PTRFIXUP(data->materials[i].normal_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].emissive_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].occlusion_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].pbr_metallic_roughness.base_color_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].pbr_metallic_roughness.metallic_roughness_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].pbr_specular_glossiness.diffuse_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].pbr_specular_glossiness.specular_glossiness_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].clearcoat.clearcoat_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].clearcoat.clearcoat_roughness_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].clearcoat.clearcoat_normal_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].specular.specular_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].specular.specular_color_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].transmission.transmission_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].volume.thickness_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].sheen.sheen_color_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].sheen.sheen_roughness_texture.texture, data->textures, data->textures_count);

		CGLTF_PTRFIXUP(data->materials[i].iridescence.iridescence_texture.texture, data->textures, data->textures_count);
		CGLTF_PTRFIXUP(data->materials[i].iridescence.iridescence_thickness_texture.texture, data->textures, data->textures_count);
	}

	for (cgltf_size i = 0; i < data->buffer_views_count; ++i)
	{
		CGLTF_PTRFIXUP_REQ(data->buffer_views[i].buffer, data->buffers, data->buffers_count);

		if (data->buffer_views[i].has_meshopt_compression)
		{
			CGLTF_PTRFIXUP_REQ(data->buffer_views[i].meshopt_compression.buffer, data->buffers, data->buffers_count);
		}
	}

	for (cgltf_size i = 0; i < data->skins_count; ++i)
	{
		for (cgltf_size j = 0; j < data->skins[i].joints_count; ++j)
		{
			CGLTF_PTRFIXUP_REQ(data->skins[i].joints[j], data->nodes, data->nodes_count);
		}

		CGLTF_PTRFIXUP(data->skins[i].skeleton, data->nodes, data->nodes_count);
		CGLTF_PTRFIXUP(data->skins[i].inverse_bind_matrices, data->accessors, data->accessors_count);
	}

	for (cgltf_size i = 0; i < data->nodes_count; ++i)
	{
		for (cgltf_size j = 0; j < data->nodes[i].children_count; ++j)
		{
			CGLTF_PTRFIXUP_REQ(data->nodes[i].children[j], data->nodes, data->nodes_count);

			if (data->nodes[i].children[j]->parent)
			{
				return CGLTF_ERROR_JSON;
			}

			data->nodes[i].children[j]->parent = &data->nodes[i];
		}

		CGLTF_PTRFIXUP(data->nodes[i].mesh, data->meshes, data->meshes_count);
		CGLTF_PTRFIXUP(data->nodes[i].skin, data->skins, data->skins_count);
		CGLTF_PTRFIXUP(data->nodes[i].camera, data->cameras, data->cameras_count);
		CGLTF_PTRFIXUP(data->nodes[i].light, data->lights, data->lights_count);

		if (data->nodes[i].has_mesh_gpu_instancing)
		{
			CGLTF_PTRFIXUP_REQ(data->nodes[i].mesh_gpu_instancing.buffer_view, data->buffer_views, data->buffer_views_count);
			for (cgltf_size m = 0; m < data->nodes[i].mesh_gpu_instancing.attributes_count; ++m)
			{
				CGLTF_PTRFIXUP_REQ(data->nodes[i].mesh_gpu_instancing.attributes[m].data, data->accessors, data->accessors_count);
			}
		}
	}

	for (cgltf_size i = 0; i < data->scenes_count; ++i)
	{
		for (cgltf_size j = 0; j < data->scenes[i].nodes_count; ++j)
		{
			CGLTF_PTRFIXUP_REQ(data->scenes[i].nodes[j], data->nodes, data->nodes_count);

			if (data->scenes[i].nodes[j]->parent)
			{
				return CGLTF_ERROR_JSON;
			}
		}
	}

	CGLTF_PTRFIXUP(data->scene, data->scenes, data->scenes_count);

	for (cgltf_size i = 0; i < data->animations_count; ++i)
	{
		for (cgltf_size j = 0; j < data->animations[i].samplers_count; ++j)
		{
			CGLTF_PTRFIXUP_REQ(data->animations[i].samplers[j].input, data->accessors, data->accessors_count);
			CGLTF_PTRFIXUP_REQ(data->animations[i].samplers[j].output, data->accessors, data->accessors_count);
		}

		for (cgltf_size j = 0; j < data->animations[i].channels_count; ++j)
		{
			CGLTF_PTRFIXUP_REQ(data->animations[i].channels[j].sampler, data->animations[i].samplers, data->animations[i].samplers_count);
			CGLTF_PTRFIXUP(data->animations[i].channels[j].target_node, data->nodes, data->nodes_count);
		}
	}

	return 0;
}

/*
 * -- jsmn.c start --
 * Source: https://github.com/zserge/jsmn
 * License: MIT
 *
 * Copyright (c) 2010 Serge A. Zaitsev

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * Allocates a fresh unused token from the token pull.
 */
static jsmntok_t *jsmn_alloc_token(jsmn_parser *parser,
				   jsmntok_t *tokens, size_t num_tokens) {
	jsmntok_t *tok;
	if (parser->toknext >= num_tokens) {
		return NULL;
	}
	tok = &tokens[parser->toknext++];
	tok->start = tok->end = -1;
	tok->size = 0;
#ifdef JSMN_PARENT_LINKS
	tok->parent = -1;
#endif
	return tok;
}

/**
 * Fills token type and boundaries.
 */
static void jsmn_fill_token(jsmntok_t *token, jsmntype_t type,
				int start, int end) {
	token->type = type;
	token->start = start;
	token->end = end;
	token->size = 0;
}

/**
 * Fills next available token with JSON primitive.
 */
static int jsmn_parse_primitive(jsmn_parser *parser, const char *js,
				size_t len, jsmntok_t *tokens, size_t num_tokens) {
	jsmntok_t *token;
	int start;

	start = parser->pos;

	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		switch (js[parser->pos]) {
#ifndef JSMN_STRICT
		/* In strict mode primitive must be followed by "," or "}" or "]" */
		case ':':
#endif
		case '\t' : case '\r' : case '\n' : case ' ' :
		case ','  : case ']'  : case '}' :
			goto found;
		}
		if (js[parser->pos] < 32 || js[parser->pos] >= 127) {
			parser->pos = start;
			return JSMN_ERROR_INVAL;
		}
	}
#ifdef JSMN_STRICT
	/* In strict mode primitive must be followed by a comma/object/array */
	parser->pos = start;
	return JSMN_ERROR_PART;
#endif

found:
	if (tokens == NULL) {
		parser->pos--;
		return 0;
	}
	token = jsmn_alloc_token(parser, tokens, num_tokens);
	if (token == NULL) {
		parser->pos = start;
		return JSMN_ERROR_NOMEM;
	}
	jsmn_fill_token(token, JSMN_PRIMITIVE, start, parser->pos);
#ifdef JSMN_PARENT_LINKS
	token->parent = parser->toksuper;
#endif
	parser->pos--;
	return 0;
}

/**
 * Fills next token with JSON string.
 */
static int jsmn_parse_string(jsmn_parser *parser, const char *js,
				 size_t len, jsmntok_t *tokens, size_t num_tokens) {
	jsmntok_t *token;

	int start = parser->pos;

	parser->pos++;

	/* Skip starting quote */
	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		char c = js[parser->pos];

		/* Quote: end of string */
		if (c == '\"') {
			if (tokens == NULL) {
				return 0;
			}
			token = jsmn_alloc_token(parser, tokens, num_tokens);
			if (token == NULL) {
				parser->pos = start;
				return JSMN_ERROR_NOMEM;
			}
			jsmn_fill_token(token, JSMN_STRING, start+1, parser->pos);
#ifdef JSMN_PARENT_LINKS
			token->parent = parser->toksuper;
#endif
			return 0;
		}

		/* Backslash: Quoted symbol expected */
		if (c == '\\' && parser->pos + 1 < len) {
			int i;
			parser->pos++;
			switch (js[parser->pos]) {
			/* Allowed escaped symbols */
			case '\"': case '/' : case '\\' : case 'b' :
			case 'f' : case 'r' : case 'n'  : case 't' :
				break;
				/* Allows escaped symbol \uXXXX */
			case 'u':
				parser->pos++;
				for(i = 0; i < 4 && parser->pos < len && js[parser->pos] != '\0'; i++) {
					/* If it isn't a hex character we have an error */
					if(!((js[parser->pos] >= 48 && js[parser->pos] <= 57) || /* 0-9 */
						 (js[parser->pos] >= 65 && js[parser->pos] <= 70) || /* A-F */
						 (js[parser->pos] >= 97 && js[parser->pos] <= 102))) { /* a-f */
						parser->pos = start;
						return JSMN_ERROR_INVAL;
					}
					parser->pos++;
				}
				parser->pos--;
				break;
				/* Unexpected symbol */
			default:
				parser->pos = start;
				return JSMN_ERROR_INVAL;
			}
		}
	}
	parser->pos = start;
	return JSMN_ERROR_PART;
}

/**
 * Parse JSON string and fill tokens.
 */
static int jsmn_parse(jsmn_parser *parser, const char *js, size_t len,
		   jsmntok_t *tokens, size_t num_tokens) {
	int r;
	int i;
	jsmntok_t *token;
	int count = parser->toknext;

	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		char c;
		jsmntype_t type;

		c = js[parser->pos];
		switch (c) {
		case '{': case '[':
			count++;
			if (tokens == NULL) {
				break;
			}
			token = jsmn_alloc_token(parser, tokens, num_tokens);
			if (token == NULL)
				return JSMN_ERROR_NOMEM;
			if (parser->toksuper != -1) {
				tokens[parser->toksuper].size++;
#ifdef JSMN_PARENT_LINKS
				token->parent = parser->toksuper;
#endif
			}
			token->type = (c == '{' ? JSMN_OBJECT : JSMN_ARRAY);
			token->start = parser->pos;
			parser->toksuper = parser->toknext - 1;
			break;
		case '}': case ']':
			if (tokens == NULL)
				break;
			type = (c == '}' ? JSMN_OBJECT : JSMN_ARRAY);
#ifdef JSMN_PARENT_LINKS
			if (parser->toknext < 1) {
				return JSMN_ERROR_INVAL;
			}
			token = &tokens[parser->toknext - 1];
			for (;;) {
				if (token->start != -1 && token->end == -1) {
					if (token->type != type) {
						return JSMN_ERROR_INVAL;
					}
					token->end = parser->pos + 1;
					parser->toksuper = token->parent;
					break;
				}
				if (token->parent == -1) {
					if(token->type != type || parser->toksuper == -1) {
						return JSMN_ERROR_INVAL;
					}
					break;
				}
				token = &tokens[token->parent];
			}
#else
			for (i = parser->toknext - 1; i >= 0; i--) {
				token = &tokens[i];
				if (token->start != -1 && token->end == -1) {
					if (token->type != type) {
						return JSMN_ERROR_INVAL;
					}
					parser->toksuper = -1;
					token->end = parser->pos + 1;
					break;
				}
			}
			/* Error if unmatched closing bracket */
			if (i == -1) return JSMN_ERROR_INVAL;
			for (; i >= 0; i--) {
				token = &tokens[i];
				if (token->start != -1 && token->end == -1) {
					parser->toksuper = i;
					break;
				}
			}
#endif
			break;
		case '\"':
			r = jsmn_parse_string(parser, js, len, tokens, num_tokens);
			if (r < 0) return r;
			count++;
			if (parser->toksuper != -1 && tokens != NULL)
				tokens[parser->toksuper].size++;
			break;
		case '\t' : case '\r' : case '\n' : case ' ':
			break;
		case ':':
			parser->toksuper = parser->toknext - 1;
			break;
		case ',':
			if (tokens != NULL && parser->toksuper != -1 &&
					tokens[parser->toksuper].type != JSMN_ARRAY &&
					tokens[parser->toksuper].type != JSMN_OBJECT) {
#ifdef JSMN_PARENT_LINKS
				parser->toksuper = tokens[parser->toksuper].parent;
#else
				for (i = parser->toknext - 1; i >= 0; i--) {
					if (tokens[i].type == JSMN_ARRAY || tokens[i].type == JSMN_OBJECT) {
						if (tokens[i].start != -1 && tokens[i].end == -1) {
							parser->toksuper = i;
							break;
						}
					}
				}
#endif
			}
			break;
#ifdef JSMN_STRICT
			/* In strict mode primitives are: numbers and booleans */
		case '-': case '0': case '1' : case '2': case '3' : case '4':
		case '5': case '6': case '7' : case '8': case '9':
		case 't': case 'f': case 'n' :
			/* And they must not be keys of the object */
			if (tokens != NULL && parser->toksuper != -1) {
				jsmntok_t *t = &tokens[parser->toksuper];
				if (t->type == JSMN_OBJECT ||
						(t->type == JSMN_STRING && t->size != 0)) {
					return JSMN_ERROR_INVAL;
				}
			}
#else
			/* In non-strict mode every unquoted value is a primitive */
		default:
#endif
			r = jsmn_parse_primitive(parser, js, len, tokens, num_tokens);
			if (r < 0) return r;
			count++;
			if (parser->toksuper != -1 && tokens != NULL)
				tokens[parser->toksuper].size++;
			break;

#ifdef JSMN_STRICT
			/* Unexpected char in strict mode */
		default:
			return JSMN_ERROR_INVAL;
#endif
		}
	}

	if (tokens != NULL) {
		for (i = parser->toknext - 1; i >= 0; i--) {
			/* Unmatched opened object or array */
			if (tokens[i].start != -1 && tokens[i].end == -1) {
				return JSMN_ERROR_PART;
			}
		}
	}

	return count;
}

/**
 * Creates a new parser based over a given  buffer with an array of tokens
 * available.
 */
static void jsmn_init(jsmn_parser *parser) {
	parser->pos = 0;
	parser->toknext = 0;
	parser->toksuper = -1;
}
/*
 * -- jsmn.c end --
 */

#endif /* #ifdef CGLTF_IMPLEMENTATION */

/* cgltf is distributed under MIT license:
 *
 * Copyright (c) 2018-2021 Johannes Kuhlmann

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

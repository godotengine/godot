#ifndef UFBXW_UFBX_WRITE_H_INCLUDED
#define UFBXW_UFBX_WRITE_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef ufbxw_abi
#define ufbxw_abi
#endif

#ifndef ufbxw_abi
	#if defined(UFBXW_STATIC)
		#define ufbxw_abi static
	#else
		#define ufbxw_abi
	#endif
#endif

#ifndef ufbxw_abi_data
	#if defined(UFBXW_STATIC)
		#define ufbxw_abi_data static
	#else
		#define ufbxw_abi_data extern
	#endif
#endif

// Linkage of the main ufbx data fields in the source.
// Defaults to nothing, or `static` if `UFBX_STATIC` is defined.
#ifndef ufbxw_abi_data_def
	#if defined(UFBX_STATIC)
		#define ufbxw_abi_data_def static
	#else
		#define ufbxw_abi_data_def
	#endif
#endif


#ifndef ufbxw_assert
	#if defined(UFBXW_NO_ASSERT) || defined(UFBXW_NO_LIBC)
		#define ufbxw_assert(cond) (void)0
	#else
		#include <assert.h>
		#define ufbxw_assert(cond) assert(cond)
	#endif
#endif

#if defined(__cplusplus)
	#define UFBXW_LIST_TYPE(p_name, p_type) struct p_name { p_type *data; size_t count; \
		p_type &operator[](size_t index) const { ufbxw_assert(index < count); return data[index]; } \
		p_type *begin() const { return data; } \
		p_type *end() const { return data + count; } \
	}
#else
	#define UFBXW_LIST_TYPE(p_name, p_type) typedef struct p_name { p_type *data; size_t count; } p_name
#endif

#define ufbxw_unsafe

#define UFBXW_ERROR_DESCRIPTION_LENGTH 256

// -- Version

// Packing/unpacking for `UFBX_HEADER_VERSION` and `ufbx_source_version`.
#define ufbxw_pack_version(major, minor, patch) ((major)*1000000u + (minor)*1000u + (patch))
#define ufbxw_version_major(version) ((uint32_t)(version)/1000000u%1000u)
#define ufbxw_version_minor(version) ((uint32_t)(version)/1000u%1000u)
#define ufbxw_version_patch(version) ((uint32_t)(version)%1000u)

// Version of the ufbx_write header.
// `UFBXW_VERSION` is simply an alias of `UFBXW_HEADER_VERSION`.
// `ufbxw_source_version` contains the version of the corresponding source file.
// HINT: The version can be compared numerically to the result of `ufbxw_pack_version()`,
// for example `#if UFBXW_VERSION >= ufbxw_pack_version(0, 12, 0)`.
#define UFBXW_HEADER_VERSION ufbxw_pack_version(0, 1, 0)
#define UFBXW_VERSION UFBXW_HEADER_VERSION

// -- Configuration

// UTF-8
typedef struct ufbxw_string {
	const char *data;
	size_t length;
} ufbxw_string;

typedef struct ufbxw_blob {
	const void *data;
	size_t size;
} ufbxw_blob;

typedef double ufbxw_real;

typedef struct ufbxw_vec2 {
	ufbxw_real x, y;
} ufbxw_vec2;

typedef struct ufbxw_vec3 {
	ufbxw_real x, y, z;
} ufbxw_vec3;

typedef struct ufbxw_vec4 {
	ufbxw_real x, y, z, w;
} ufbxw_vec4;

typedef struct ufbxw_bool3 {
	bool x, y, z;
} ufbxw_bool3;

typedef struct ufbxw_quat {
	ufbxw_real x, y, z, w;
} ufbxw_quat;

typedef struct ufbxw_matrix {
	union {
		ufbxw_real m[16];
		struct {
			ufbxw_real m00, m10, m20, m30;
			ufbxw_real m01, m11, m21, m31;
			ufbxw_real m02, m12, m22, m32;
			ufbxw_real m03, m13, m23, m33;
		};
	};
} ufbxw_matrix;

typedef struct ufbxw_transform {
	ufbxw_vec3 translation;
	ufbxw_quat rotation;
	ufbxw_vec3 scale;
} ufbxw_transform;

ufbxw_abi_data const ufbxw_string ufbxw_empty_string;
ufbxw_abi_data const ufbxw_vec2 ufbxw_zero_vec2;
ufbxw_abi_data const ufbxw_vec3 ufbxw_zero_vec3;
ufbxw_abi_data const ufbxw_vec4 ufbxw_zero_vec4;
ufbxw_abi_data const ufbxw_matrix ufbxw_identity_matrix;

typedef enum ufbxw_rotation_order {
	UFBXW_ROTATION_ORDER_XYZ,
	UFBXW_ROTATION_ORDER_XZY,
	UFBXW_ROTATION_ORDER_YZX,
	UFBXW_ROTATION_ORDER_YXZ,
	UFBXW_ROTATION_ORDER_ZXY,
	UFBXW_ROTATION_ORDER_ZYX,
	UFBXW_ROTATION_ORDER_SPHERIC,
} ufbxw_rotation_order;

typedef enum ufbxw_inherit_mode {
	UFBXW_INHERIT_TYPE_NORMAL,
	UFBXW_INHERIT_TYPE_IGNORE_PARENT_SCALE,
	UFBXW_INHERIT_TYPE_COMPONENTWISE_SCALE,
} ufbxw_inherit_type;

typedef struct ufbxw_scene ufbxw_scene;

typedef enum ufbxw_element_type {
	UFBXW_ELEMENT_CUSTOM = 1,
	UFBXW_ELEMENT_NODE,

	UFBXW_ELEMENT_NODE_ATTRIBUTE,
	UFBXW_ELEMENT_MESH,
	UFBXW_ELEMENT_SKIN_DEFORMER,
	UFBXW_ELEMENT_SKIN_CLUSTER,
	UFBXW_ELEMENT_BLEND_DEFORMER,
	UFBXW_ELEMENT_BLEND_CHANNEL,
	UFBXW_ELEMENT_BLEND_SHAPE,
	UFBXW_ELEMENT_LIGHT,
	UFBXW_ELEMENT_CAMERA,
	UFBXW_ELEMENT_SKELETON,
	UFBXW_ELEMENT_BIND_POSE,

	UFBXW_ELEMENT_MATERIAL,

	UFBXW_ELEMENT_TEXTURE,

	UFBXW_ELEMENT_ANIM_CURVE,
	UFBXW_ELEMENT_ANIM_PROP,
	UFBXW_ELEMENT_ANIM_LAYER,
	UFBXW_ELEMENT_ANIM_STACK,

	UFBXW_ELEMENT_TEMPLATE,
	UFBXW_ELEMENT_SCENE_INFO,
	UFBXW_ELEMENT_GLOBAL_SETTINGS,
	UFBXW_ELEMENT_DOCUMENT,

	UFBXW_ELEMENT_TYPE_COUNT,
} ufbxw_element_type;

typedef uint64_t ufbxw_id;
typedef struct ufbxw_node { ufbxw_id id; } ufbxw_node;
typedef struct ufbxw_mesh { ufbxw_id id; } ufbxw_mesh;
typedef struct ufbxw_skin_deformer { ufbxw_id id; } ufbxw_skin_deformer;
typedef struct ufbxw_skin_cluster { ufbxw_id id; } ufbxw_skin_cluster;
typedef struct ufbxw_blend_deformer { ufbxw_id id; } ufbxw_blend_deformer;
typedef struct ufbxw_blend_channel { ufbxw_id id; } ufbxw_blend_channel;
typedef struct ufbxw_blend_shape { ufbxw_id id; } ufbxw_blend_shape;
typedef struct ufbxw_light { ufbxw_id id; } ufbxw_light;
typedef struct ufbxw_camera { ufbxw_id id; } ufbxw_camera;
typedef struct ufbxw_bind_pose { ufbxw_id id; } ufbxw_bind_pose;
typedef struct ufbxw_material { ufbxw_id id; } ufbxw_material;
typedef struct ufbxw_anim_prop { ufbxw_id id; } ufbxw_anim_prop;
typedef struct ufbxw_anim_curve { ufbxw_id id; } ufbxw_anim_curve;
typedef struct ufbxw_anim_layer { ufbxw_id id; } ufbxw_anim_layer;
typedef struct ufbxw_anim_stack { ufbxw_id id; } ufbxw_anim_stack ;

typedef uint64_t ufbxw_buffer_id;
typedef struct ufbxw_int_buffer { ufbxw_buffer_id id; } ufbxw_int_buffer;
typedef struct ufbxw_long_buffer { ufbxw_buffer_id id; } ufbxw_long_buffer;
typedef struct ufbxw_real_buffer { ufbxw_buffer_id id; } ufbxw_real_buffer;
typedef struct ufbxw_vec2_buffer { ufbxw_buffer_id id; } ufbxw_vec2_buffer;
typedef struct ufbxw_vec3_buffer { ufbxw_buffer_id id; } ufbxw_vec3_buffer;
typedef struct ufbxw_vec4_buffer { ufbxw_buffer_id id; } ufbxw_vec4_buffer;
typedef struct ufbxw_float_buffer { ufbxw_buffer_id id; } ufbxw_float_buffer;

UFBXW_LIST_TYPE(ufbxw_int_list, int32_t);
UFBXW_LIST_TYPE(ufbxw_long_list, int64_t);
UFBXW_LIST_TYPE(ufbxw_real_list, ufbxw_real);
UFBXW_LIST_TYPE(ufbxw_vec2_list, ufbxw_vec2);
UFBXW_LIST_TYPE(ufbxw_vec3_list, ufbxw_vec3);
UFBXW_LIST_TYPE(ufbxw_vec4_list, ufbxw_vec4);
UFBXW_LIST_TYPE(ufbxw_float_list, float);

typedef int64_t ufbxw_ktime;

typedef struct ufbxw_ktime_range {
	ufbxw_ktime begin;
	ufbxw_ktime end;
} ufbxw_ktime_range;

#if defined(__cplusplus)
	#define ufbxw_new(m_type) m_type
#else
	#define ufbxw_new(m_type) (m_type)
#endif

#define ufbxw_null_id ((ufbxw_id)0)
#define ufbxw_null_node (ufbxw_new(ufbxw_node){0})
#define ufbxw_null_mesh (ufbxw_new(ufbxw_mesh){0})
#define ufbxw_null_skin_deformer (ufbxw_new(ufbxw_skin_deformer){0})
#define ufbxw_null_skin_cluster (ufbxw_new(ufbxw_skin_cluster){0})
#define ufbxw_null_blend_deformer (ufbxw_new(ufbxw_blend_deformer){0})
#define ufbxw_null_blend_channel (ufbxw_new(ufbxw_blend_channel){0})
#define ufbxw_null_blend_shape (ufbxw_new(ufbxw_blend_shape){0})
#define ufbxw_null_light (ufbxw_new(ufbxw_light){0})
#define ufbxw_null_camera (ufbxw_new(ufbxw_camera){0})
#define ufbxw_null_bind_pose (ufbxw_new(ufbxw_bind_pose){0})
#define ufbxw_null_material (ufbxw_new(ufbxw_material){0})
#define ufbxw_null_anim_prop (ufbxw_new(ufbxw_anim_prop){0})
#define ufbxw_null_anim_curve (ufbxw_new(ufbxw_anim_curve){0})
#define ufbxw_null_anim_layer (ufbxw_new(ufbxw_anim_layer){0})
#define ufbxw_null_anim_stack (ufbxw_new(ufbxw_anim_stack){0})

typedef enum ufbxw_connection_type {
	UFBXW_CONNECTION_NODE_PARENT = 1,   // NODE* -> NODE
	UFBXW_CONNECTION_NODE_ATTRIBUTE,    // NODE_ATTRIBUTE -> NODE*
	UFBXW_CONNECTION_MATERIAL,          // MATERIAL* -> NODE*
	UFBXW_CONNECTION_TEXTURE,           // TEXTURE* -> MATERIAL(property)*
	UFBXW_CONNECTION_MESH_DEFORMER,     // DEFORMER -> MESH
	UFBXW_CONNECTION_SKIN_CLUSTER,      // SKIN_CLUSTER -> SKIN_DEFORMER
	UFBXW_CONNECTION_SKIN_CLUSTER_NODE, // NODE -> SKIN_CLUSTER
	UFBXW_CONNECTION_BLEND_CHANNEL,     // BLEND_CHANNEL -> BLEND_DEFORMER
	UFBXW_CONNECTION_BLEND_SHAPE,       // BLEND_SHAPE -> BLEND_CHANNEL
	UFBXW_CONNECTION_ANIM_PROPERTY,     // ANIM_PROP* -> ANY(property)*
	UFBXW_CONNECTION_ANIM_CURVE_PROP,   // ANIM_CURVE* -> ANIM_PROP
	UFBXW_CONNECTION_ANIM_PROP_LAYER,   // ANIM_PROP* -> ANIM_LAYER
	UFBXW_CONNECTION_ANIM_LAYER_STACK,  // ANIM_LAYER* -> ANIM_STACK
	UFBXW_CONNECTION_CUSTOM,            // ANY* -> ANY*

	UFBXW_CONNECTION_TYPE_COUNT,
} ufbxw_connection_type;

// -- Memory callbacks

typedef void *ufbxw_alloc_fn(void *user, size_t size);
typedef void ufbxw_free_fn(void *user, void *ptr, size_t size);
typedef void ufbxw_free_allocator_fn(void *user);

typedef struct ufbxw_allocator {
	ufbxw_alloc_fn *alloc_fn;
	ufbxw_free_fn *free_fn;
	ufbxw_free_allocator_fn *free_allocator_fn;
	void *user;
} ufbxw_allocator;

// -- Error

typedef enum ufbxw_error_type {
	UFBXW_ERROR_NONE,

	UFBXW_ERROR_ELEMENT_NOT_FOUND,
	UFBXW_ERROR_ELEMENT_WRONG_TYPE,
	UFBXW_ERROR_ELEMENT_TYPE_NOT_FOUND,
	UFBXW_ERROR_PROP_DATA_TYPE,
	UFBXW_ERROR_PROP_NOT_FOUND,
	UFBXW_ERROR_WRONG_DATA_TYPE,
	UFBXW_ERROR_INDEX_OUT_OF_BOUNDS,
	UFBXW_ERROR_CYCLICAL_PARENT,

	// Fatal errors
	UFBXW_ERROR_FATAL,
	UFBXW_ERROR_STRING_TOO_LONG,
	UFBXW_ERROR_MEMORY_LIMIT,
	UFBXW_ERROR_ALLOCATION_LIMIT,
	UFBXW_ERROR_ALLOCATION_FAILURE,
	UFBXW_ERROR_BUFFER_STREAM,
	UFBXW_ERROR_WRITE_FAILED,
	UFBXW_ERROR_DEFLATE_FAILED,
	UFBXW_ERROR_ARRAY_TOO_BIG,
	UFBXW_ERROR_PATH_TOO_LONG,
	UFBXW_ERROR_FILE_OPEN_FAILED,
	UFBXW_ERROR_ASCII_FORMAT,
	UFBXW_ERROR_THREAD_SYNC_INIT,

} ufbxw_error_type;

typedef struct ufbxw_error {
	ufbxw_error_type type;

	ufbxw_string function;

	size_t description_length;
	char description[UFBXW_ERROR_DESCRIPTION_LENGTH];

} ufbxw_error;

// -- Memory

typedef struct ufbxw_memory_stats {
	size_t allocated_bytes;
	size_t allocation_count;
	size_t block_allocation_count;
} ufbxw_memory_stats;

// -- Scene API

typedef struct ufbxw_scene_opts {
	uint32_t _begin_zero;

	ufbxw_allocator allocator;

	bool no_default_elements;
	bool no_default_scene_info;
	bool no_default_global_settings;
	bool no_default_document;
	bool no_default_anim_stack;
	bool no_default_anim_layer;

	// Maximum amount of allocations to do
	size_t max_allocations;

	uint32_t _end_zero;
} ufbxw_scene_opts;

typedef struct ufbxw_real_string {
	ufbxw_real value;
	ufbxw_string string;
} ufbxw_real_string;

typedef struct ufbxw_user_int {
	int32_t value;
	int32_t min_value;
	int32_t max_value;
} ufbxw_user_int;

typedef struct ufbxw_user_real {
	ufbxw_real value;
	ufbxw_real min_value;
	ufbxw_real max_value;
} ufbxw_user_real;

typedef struct ufbxw_user_enum {
	int32_t value;
	ufbxw_string options;
} ufbxw_user_enum;

typedef enum ufbxw_prop_data_type {
	UFBXW_PROP_DATA_NONE,         // < `void`
	UFBXW_PROP_DATA_BOOL,         // < `bool`
	UFBXW_PROP_DATA_INT32,        // < `int32_t`
	UFBXW_PROP_DATA_INT64,        // < `int64_t`
	UFBXW_PROP_DATA_REAL,         // < `ufbxw_real`
	UFBXW_PROP_DATA_VEC2,         // < `ufbxw_vec2`
	UFBXW_PROP_DATA_VEC3,         // < `ufbxw_vec3`
	UFBXW_PROP_DATA_VEC4,         // < `ufbxw_vec4`
	UFBXW_PROP_DATA_STRING,       // < `ufbxw_string`
	UFBXW_PROP_DATA_ID,           // < `ufbxw_id`
	UFBXW_PROP_DATA_REAL_STRING,  // < `ufbxw_real_string`
	UFBXW_PROP_DATA_BLOB,         // < `ufbxw_blob`
	UFBXW_PROP_DATA_USER_INT,     // < `ufbxw_user_int`
	UFBXW_PROP_DATA_USER_REAL,    // < `ufbxw_user_real`
	UFBXW_PROP_DATA_USER_ENUM,    // < `ufbxw_user_enum`

	UFBXW_PROP_DATA_TYPE_COUNT,
} ufbxw_prop_data_type;

typedef enum ufbxw_prop_flag {
	UFBXW_PROP_FLAG_ANIMATABLE = 0x1,
	UFBXW_PROP_FLAG_USER = 0x2,
} ufbxw_prop_flag;

typedef enum ufbxw_prop_type {
	UFBXW_PROP_TYPE_NONE,

	UFBXW_PROP_TYPE_COMPOUND,   // < "Compound", "", `UFBXW_PROP_DATA_NONE`
	UFBXW_PROP_TYPE_BOOL,       // < "bool", "", `UFBXW_PROP_DATA_BOOL`
	UFBXW_PROP_TYPE_ENUM,       // < "enum", "", `UFBXW_PROP_DATA_INT32`
	UFBXW_PROP_TYPE_INT,        // < "int", "Integer", `UFBXW_PROP_DATA_INT32`
	UFBXW_PROP_TYPE_TIME,       // < "KTime", "Time", `UFBXW_PROP_DATA_INT64`
	UFBXW_PROP_TYPE_UBYTE,      // < "UByte", "", `UFBXW_PROP_DATA_INT64`
	UFBXW_PROP_TYPE_ULONGLONG,  // < "ULongLong", "", `UFBXW_PROP_DATA_INT64`
	UFBXW_PROP_TYPE_FLOAT,      // < "Float", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_DOUBLE,     // < "double", "Number", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_NUMBER,     // < "Number", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_VECTOR2D,   // < "Vector2D", "Vector2", `UFBXW_PROP_DATA_VEC2`
	UFBXW_PROP_TYPE_VECTOR,     // < "Vector", "", `UFBXW_PROP_DATA_VEC3`
	UFBXW_PROP_TYPE_VECTOR3D,   // < "Vector3D", "Vector", `UFBXW_PROP_DATA_VEC3`
	UFBXW_PROP_TYPE_COLOR,      // < "Color", "", `UFBXW_PROP_DATA_VEC3`
	UFBXW_PROP_TYPE_COLOR_RGB,  // < "ColorRGB", "Color", `UFBXW_PROP_DATA_VEC3`
	UFBXW_PROP_TYPE_COLOR_RGBA, // < "ColorAndAlpha", "", `UFBXW_PROP_DATA_VEC4`
	UFBXW_PROP_TYPE_STRING,     // < "KString", "", `UFBXW_PROP_DATA_STRING`
	UFBXW_PROP_TYPE_URL,        // < "KString", "Url", `UFBXW_PROP_DATA_STRING`
	UFBXW_PROP_TYPE_XREFURL,    // < "KString", "XRefUrl", `UFBXW_PROP_DATA_STRING`
	UFBXW_PROP_TYPE_DATE_TIME,  // < "DateTime", "", `UFBXW_PROP_DATA_STRING`
	UFBXW_PROP_TYPE_OBJECT,     // < "object", "", `UFBXW_PROP_DATA_ID`
	UFBXW_PROP_TYPE_DISTANCE,   // < "Distance", "", `UFBXW_PROP_DATA_REAL_STRING`
	UFBXW_PROP_TYPE_BLOB,       // < "Blob", "", `UFBXW_PROP_DATA_BLOB`

	// User property types
	UFBXW_PROP_TYPE_USER_BOOL,    // < "Bool", "", `UFBXW_PROP_DATA_BOOL`
	UFBXW_PROP_TYPE_USER_VECTOR,  // < "Vector", "", `UFBXW_PROP_DATA_VEC3`
	UFBXW_PROP_TYPE_USER_INTEGER, // < "Integer", "", `UFBXW_PROP_DATA_USER_INT`
	UFBXW_PROP_TYPE_USER_NUMBER,  // < "Number", "", `UFBXW_PROP_DATA_USER_REAL`
	UFBXW_PROP_TYPE_USER_STRING,  // < "KString", "", `UFBXW_PROP_DATA_STRING`
	UFBXW_PROP_TYPE_USER_ENUM,    // < "Enum", "", `UFBXW_PROP_DATA_USER_ENUM`

	// Decicated types
	UFBXW_PROP_TYPE_VISIBILITY,             // < "Visibility", "", `UFBXW_PROP_DATA_BOOL`
	UFBXW_PROP_TYPE_VISIBILITY_INHERITANCE, // < "Visibility Inheritance", "", `UFBXW_PROP_DATA_BOOL`
	UFBXW_PROP_TYPE_ROLL,                   // < "Roll", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_OPTICAL_CENTER_X,       // < "OpticalCenterX", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_OPTICAL_CENTER_Y,       // < "OpticalCenterY", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_FIELD_OF_VIEW,          // < "FieldOfView", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_FIELD_OF_VIEW_X,        // < "FieldOfViewX", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_FIELD_OF_VIEW_Y,        // < "FieldOfViewY", "", `UFBXW_PROP_DATA_REAL`
	UFBXW_PROP_TYPE_LCL_TRANSLATION,        // < "Lcl Translation", "", `UFBXW_PROP_DATA_VEC3`
	UFBXW_PROP_TYPE_LCL_ROTATION,           // < "Lcl Rotation", "", `UFBXW_PROP_DATA_VEC3`
	UFBXW_PROP_TYPE_LCL_SCALING,            // < "Lcl Scaling", "", `UFBXW_PROP_DATA_VEC3`

	// Custom
	UFBXW_PROP_TYPE_FIRST_CUSTOM,
	UFBXW_PROP_TYPE_FORCE_32BIT = 0x7fffffff,
} ufbxw_prop_type;

typedef enum ufbxw_prop_type_flags {
	UFBXW_PROP_TYPE_FLAG_ENCODE_AS_BYTE = 0x1,
	UFBXW_PROP_TYPE_FLAG_ENCODE_AS_FLOAT = 0x2,
} ufbxw_prop_type_flags;

typedef struct ufbxw_prop_type_desc {
	ufbxw_prop_data_type data_type;
	uint32_t flags;
	ufbxw_string type;
	ufbxw_string sub_type;
} ufbxw_prop_type_desc;

typedef enum ufbxw_keyframe_flags {
	UFBXW_KEYFRAME_INTERPOLATION_CONSTANT = 0x1,
	UFBXW_KEYFRAME_INTERPOLATION_CONSTANT_NEXT = 0x2,
	UFBXW_KEYFRAME_INTERPOLATION_LINEAR = 0x4,
	UFBXW_KEYFRAME_INTERPOLATION_CUBIC = 0x8,

	UFBXW_KEYFRAME_TANGENT_AUTO = 0x10,
	UFBXW_KEYFRAME_TANGENT_AUTO_UNCLAMPED = 0x20,
	UFBXW_KEYFRAME_TANGENT_AUTO_LEGACY = 0x40,
	UFBXW_KEYFRAME_TANGENT_AUTO_LEGACY_CLAMPED = 0x80,
	UFBXW_KEYFRAME_TANGENT_USER = 0x100,
	UFBXW_KEYFRAME_TANGENT_BROKEN = 0x200,
	UFBXW_KEYFRAME_TANGENT_TCB = 0x400,

	UFBXW_KEYFRAME_WEIGHTED_LEFT = 0x1000,
	UFBXW_KEYFRAME_WEIGHTED_RIGHT = 0x2000,
	UFBXW_KEYFRAME_WEIGHTED_BOTH = 0x4000,

	UFBXW_KEYFRAME_TANGENT_SHOW_LEFT = 0x10000,
	UFBXW_KEYFRAME_TANGENT_SHOW_RIGHT = 0x20000,
	UFBXW_KEYFRAME_TANGENT_SHOW_BOTH = 0x40000,
} ufbxw_keyframe_flags;

// Simple keyframe types
typedef enum ufbxw_keyframe_type {
	UFBXW_KEYFRAME_CONSTANT = UFBXW_KEYFRAME_INTERPOLATION_CONSTANT,
	UFBXW_KEYFRAME_CONSTANT_NEXT = UFBXW_KEYFRAME_INTERPOLATION_CONSTANT_NEXT,
	UFBXW_KEYFRAME_LINEAR = UFBXW_KEYFRAME_INTERPOLATION_LINEAR,
	UFBXW_KEYFRAME_CUBIC_AUTO = UFBXW_KEYFRAME_INTERPOLATION_CUBIC | UFBXW_KEYFRAME_TANGENT_AUTO,
	UFBXW_KEYFRAME_CUBIC_AUTO_BROKEN = UFBXW_KEYFRAME_INTERPOLATION_CUBIC | UFBXW_KEYFRAME_TANGENT_AUTO | UFBXW_KEYFRAME_TANGENT_BROKEN,
	UFBXW_KEYFRAME_CUBIC_USER_UNIFIED = UFBXW_KEYFRAME_INTERPOLATION_CUBIC | UFBXW_KEYFRAME_TANGENT_USER,
	UFBXW_KEYFRAME_CUBIC_USER_BROKEN = UFBXW_KEYFRAME_INTERPOLATION_CUBIC | UFBXW_KEYFRAME_TANGENT_USER | UFBXW_KEYFRAME_TANGENT_BROKEN,
} ufbxw_keyframe_type;

typedef enum ufbxw_time_mode {
	UFBXW_TIME_MODE_DEFAULT,
	UFBXW_TIME_MODE_120_FPS,
	UFBXW_TIME_MODE_100_FPS,
	UFBXW_TIME_MODE_60_FPS,
	UFBXW_TIME_MODE_50_FPS,
	UFBXW_TIME_MODE_48_FPS,
	UFBXW_TIME_MODE_30_FPS,
	UFBXW_TIME_MODE_30_FPS_DROP,
	UFBXW_TIME_MODE_NTSC_DROP_FRAME,
	UFBXW_TIME_MODE_NTSC_FULL_FRAME,
	UFBXW_TIME_MODE_PAL,
	UFBXW_TIME_MODE_24_FPS,
	UFBXW_TIME_MODE_1000_FPS,
	UFBXW_TIME_MODE_FILM_FULL_FRAME,
	UFBXW_TIME_MODE_CUSTOM,
	UFBXW_TIME_MODE_96_FPS,
	UFBXW_TIME_MODE_72_FPS,
	UFBXW_TIME_MODE_59_94_FPS,
} ufbxw_time_mode;

typedef enum ufbxw_time_protocol {
	UFBXW_TIME_PROTOCOL_SMPTE,
	UFBXW_TIME_PROTOCOL_FRAME_COUNT,
	UFBXW_TIME_PROTOCOL_DEFAULT,
} ufbxw_time_protocol;

typedef enum ufbxw_snap_mode {
	UFBXW_SNAP_MODE_NONE,
	UFBXW_SNAP_MODE_SNAP,
	UFBXW_SNAP_MODE_PLAY,
	UFBXW_SNAP_MODE_SNAP_AND_PLAY,
} ufbxw_snap_mode;

typedef struct ufbxw_datetime {
	int32_t year;        // < Year
	int32_t month;       // < Month of year 1-12
	int32_t day;         // < Day of month 1-31
	int32_t hour;        // < Hour 0-23
	int32_t minute;      // < Minute 0-59
	int32_t second;      // < Second 0-60
	int32_t millisecond; // < Millisecond 0-999
} ufbxw_datetime;

typedef struct ufbxw_keyframe_real {
	ufbxw_ktime time;
	ufbxw_real value;
	uint32_t flags;

	ufbxw_real weight_left;
	ufbxw_real weight_right;
	ufbxw_real slope_left;
	ufbxw_real slope_right;
} ufbxw_keyframe_real;

typedef struct ufbxw_keyframe_vec2 {
	ufbxw_ktime time;
	ufbxw_vec2 value;
	uint32_t flags;

	ufbxw_real weight_left;
	ufbxw_real weight_right;
	ufbxw_vec2 slope_left;
	ufbxw_vec2 slope_right;
} ufbxw_keyframe_vec2;

typedef struct ufbxw_keyframe_vec3 {
	ufbxw_ktime time;
	ufbxw_vec3 value;
	uint32_t flags;

	ufbxw_real weight_left;
	ufbxw_real weight_right;
	ufbxw_vec3 slope_left;
	ufbxw_vec3 slope_right;
} ufbxw_keyframe_vec3;

#define UFBXW_KTIME_SECOND INT64_C(46186158000)

typedef enum ufbxw_mesh_attribute {
	UFBXW_MESH_ATTRIBUTE_NORMAL = 1,
	UFBXW_MESH_ATTRIBUTE_UV,
	UFBXW_MESH_ATTRIBUTE_TANGENT,
	UFBXW_MESH_ATTRIBUTE_BINORMAL,
	UFBXW_MESH_ATTRIBUTE_COLOR,
	UFBXW_MESH_ATTRIBUTE_SMOOTHING,
	UFBXW_MESH_ATTRIBUTE_MATERIAL,

	UFBXW_MESH_ATTRIBUTE_FIRST_CUSTOM,
} ufbxw_mesh_attribute;

typedef enum ufbxw_attribute_mapping {
	UFBXW_ATTRIBUTE_MAPPING_NONE,
	UFBXW_ATTRIBUTE_MAPPING_POLYGON_VERTEX,
	UFBXW_ATTRIBUTE_MAPPING_VERTEX,
	UFBXW_ATTRIBUTE_MAPPING_EDGE,
	UFBXW_ATTRIBUTE_MAPPING_POLYGON,
	UFBXW_ATTRIBUTE_MAPPING_ALL_SAME,
} ufbxw_attribute_mapping;

typedef struct ufbxw_mesh_attribute_desc {
	uint32_t _begin_zero;

	ufbxw_attribute_mapping mapping;

	ufbxw_buffer_id values;
	ufbxw_buffer_id indices;

	ufbxw_string name;
	ufbxw_buffer_id values_w;

	bool generate_indices;

	uint32_t _end_zero;
} ufbxw_mesh_attribute_desc;

typedef enum ufbxw_skinning_type {
	UFBXW_SKINNING_TYPE_RIGID,
	UFBXW_SKINNING_TYPE_LINEAR,
	UFBXW_SKINNING_TYPE_DUAL_QUATERNION,
	UFBXW_SKINNING_TYPE_BLEND,
} ufbxw_skinning_type;

typedef void ufbxw_error_fn(void *user, const ufbxw_error *error);

// -- API

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t ufbxw_int_stream_fn(void *user, int32_t *dst, size_t dst_size, size_t offset);
typedef size_t ufbxw_long_stream_fn(void *user, int64_t *dst, size_t dst_size, size_t offset);
typedef size_t ufbxw_real_stream_fn(void *user, ufbxw_real *dst, size_t dst_size, size_t offset);
typedef size_t ufbxw_vec2_stream_fn(void *user, ufbxw_vec2 *dst, size_t dst_size, size_t offset);
typedef size_t ufbxw_vec3_stream_fn(void *user, ufbxw_vec3 *dst, size_t dst_size, size_t offset);
typedef size_t ufbxw_vec4_stream_fn(void *user, ufbxw_vec4 *dst, size_t dst_size, size_t offset);
typedef size_t ufbxw_float_stream_fn(void *user, float *dst, size_t dst_size, size_t offset);
typedef void ufbxw_buffer_deleter_fn(void *user, void *data);

ufbxw_abi void ufbxw_retain_buffer(ufbxw_scene *scene, ufbxw_buffer_id buffer);
ufbxw_abi void ufbxw_free_buffer(ufbxw_scene *scene, ufbxw_buffer_id buffer);

ufbxw_abi void ufbxw_buffer_set_deleter(ufbxw_scene *scene, ufbxw_buffer_id buffer, ufbxw_buffer_deleter_fn *fn, void *user);

ufbxw_abi ufbxw_int_buffer ufbxw_create_int_buffer(ufbxw_scene *scene, size_t count);
ufbxw_abi ufbxw_int_buffer ufbxw_copy_int_array(ufbxw_scene *scene, const int32_t *data, size_t count);
ufbxw_abi ufbxw_int_buffer ufbxw_view_int_array(ufbxw_scene *scene, const int32_t *data, size_t count);
ufbxw_abi ufbxw_int_buffer ufbxw_external_int_array(ufbxw_scene *scene, const int32_t *data, size_t count);
ufbxw_abi ufbxw_int_buffer ufbxw_external_int_stream(ufbxw_scene *scene, ufbxw_int_stream_fn *fn, void *user, size_t count);

ufbxw_abi ufbxw_long_buffer ufbxw_create_long_buffer(ufbxw_scene *scene, size_t count);
ufbxw_abi ufbxw_long_buffer ufbxw_copy_long_array(ufbxw_scene *scene, const int64_t *data, size_t count);
ufbxw_abi ufbxw_long_buffer ufbxw_view_long_array(ufbxw_scene *scene, const int64_t *data, size_t count);
ufbxw_abi ufbxw_long_buffer ufbxw_external_long_array(ufbxw_scene *scene, const int64_t *data, size_t count);
ufbxw_abi ufbxw_long_buffer ufbxw_external_long_stream(ufbxw_scene *scene, ufbxw_long_stream_fn *fn, void *user, size_t count);

ufbxw_abi ufbxw_real_buffer ufbxw_create_real_buffer(ufbxw_scene *scene, size_t count);
ufbxw_abi ufbxw_real_buffer ufbxw_copy_real_array(ufbxw_scene *scene, const ufbxw_real *data, size_t count);
ufbxw_abi ufbxw_real_buffer ufbxw_view_real_array(ufbxw_scene *scene, const ufbxw_real *data, size_t count);
ufbxw_abi ufbxw_real_buffer ufbxw_external_real_array(ufbxw_scene *scene, const ufbxw_real *data, size_t count);
ufbxw_abi ufbxw_real_buffer ufbxw_external_real_stream(ufbxw_scene *scene, ufbxw_real_stream_fn *fn, void *user, size_t count);

ufbxw_abi ufbxw_vec2_buffer ufbxw_create_vec2_buffer(ufbxw_scene *scene, size_t count);
ufbxw_abi ufbxw_vec2_buffer ufbxw_copy_vec2_array(ufbxw_scene *scene, const ufbxw_vec2 *data, size_t count);
ufbxw_abi ufbxw_vec2_buffer ufbxw_view_vec2_array(ufbxw_scene *scene, const ufbxw_vec2 *data, size_t count);
ufbxw_abi ufbxw_vec2_buffer ufbxw_external_vec2_array(ufbxw_scene *scene, const ufbxw_vec2 *data, size_t count);
ufbxw_abi ufbxw_vec2_buffer ufbxw_external_vec2_stream(ufbxw_scene *scene, ufbxw_vec2_stream_fn *fn, void *user, size_t count);

ufbxw_abi ufbxw_vec3_buffer ufbxw_create_vec3_buffer(ufbxw_scene *scene, size_t count);
ufbxw_abi ufbxw_vec3_buffer ufbxw_copy_vec3_array(ufbxw_scene *scene, const ufbxw_vec3 *data, size_t count);
ufbxw_abi ufbxw_vec3_buffer ufbxw_view_vec3_array(ufbxw_scene *scene, const ufbxw_vec3 *data, size_t count);
ufbxw_abi ufbxw_vec3_buffer ufbxw_external_vec3_array(ufbxw_scene *scene, const ufbxw_vec3 *data, size_t count);
ufbxw_abi ufbxw_vec3_buffer ufbxw_external_vec3_stream(ufbxw_scene *scene, ufbxw_vec3_stream_fn *fn, void *user, size_t count);

ufbxw_abi ufbxw_vec4_buffer ufbxw_create_vec4_buffer(ufbxw_scene* scene, size_t count);
ufbxw_abi ufbxw_vec4_buffer ufbxw_copy_vec4_array(ufbxw_scene *scene, const ufbxw_vec4 *data, size_t count);
ufbxw_abi ufbxw_vec4_buffer ufbxw_view_vec4_array(ufbxw_scene *scene, const ufbxw_vec4 *data, size_t count);
ufbxw_abi ufbxw_vec4_buffer ufbxw_external_vec4_array(ufbxw_scene *scene, const ufbxw_vec4 *data, size_t count);
ufbxw_abi ufbxw_vec4_buffer ufbxw_external_vec4_stream(ufbxw_scene *scene, ufbxw_vec4_stream_fn *fn, void *user, size_t count);

ufbxw_abi ufbxw_float_buffer ufbxw_create_float_buffer(ufbxw_scene* scene, size_t count);
ufbxw_abi ufbxw_float_buffer ufbxw_copy_float_array(ufbxw_scene *scene, const float *data, size_t count);
ufbxw_abi ufbxw_float_buffer ufbxw_view_float_array(ufbxw_scene *scene, const float *data, size_t count);
ufbxw_abi ufbxw_float_buffer ufbxw_external_float_array(ufbxw_scene *scene, const float *data, size_t count);
ufbxw_abi ufbxw_float_buffer ufbxw_external_float_stream(ufbxw_scene *scene, ufbxw_float_stream_fn *fn, void *user, size_t count);

// TODO: Lock/unlock version for Rust
ufbxw_abi ufbxw_int_list ufbxw_edit_int_buffer(ufbxw_scene *scene, ufbxw_int_buffer buffer);
ufbxw_abi ufbxw_long_list ufbxw_edit_long_buffer(ufbxw_scene *scene, ufbxw_long_buffer buffer);
ufbxw_abi ufbxw_real_list ufbxw_edit_real_buffer(ufbxw_scene *scene, ufbxw_real_buffer buffer);
ufbxw_abi ufbxw_vec2_list ufbxw_edit_vec2_buffer(ufbxw_scene *scene, ufbxw_vec2_buffer buffer);
ufbxw_abi ufbxw_vec3_list ufbxw_edit_vec3_buffer(ufbxw_scene *scene, ufbxw_vec3_buffer buffer);
ufbxw_abi ufbxw_vec4_list ufbxw_edit_vec4_buffer(ufbxw_scene* scene, ufbxw_vec4_buffer buffer);
ufbxw_abi ufbxw_float_list ufbxw_edit_float_buffer(ufbxw_scene* scene, ufbxw_float_buffer buffer);

// --

ufbxw_abi ufbxw_scene *ufbxw_create_scene(const ufbxw_scene_opts *opts);
ufbxw_abi void ufbxw_free_scene(ufbxw_scene *scene);

ufbxw_abi void ufbxw_set_error_callback(ufbxw_scene *scene, ufbxw_error_fn *fn, void *user);

ufbxw_abi bool ufbxw_get_error(ufbxw_scene *scene, ufbxw_error *error);
ufbxw_abi ufbxw_memory_stats ufbxw_get_memory_stats(ufbxw_scene *scene);

ufbxw_abi ufbxw_id ufbxw_create_element(ufbxw_scene *scene, ufbxw_element_type type);
ufbxw_abi ufbxw_id ufbxw_create_element_ex(ufbxw_scene *scene, ufbxw_element_type type, const char *class_type);
ufbxw_abi ufbxw_id ufbxw_create_element_ex_len(ufbxw_scene *scene, ufbxw_element_type type, const char *class_type, size_t class_type_len);
ufbxw_abi void ufbxw_delete_element(ufbxw_scene *scene, ufbxw_id id);

ufbxw_abi size_t ufbxw_get_num_elements(ufbxw_scene *scene);
ufbxw_abi size_t ufbxw_get_elements(const ufbxw_scene *scene, ufbxw_id *elements, size_t num_elements);

ufbxw_abi void ufbxw_set_name(ufbxw_scene *scene, ufbxw_id id, const char *name);
ufbxw_abi void ufbxw_set_name_len(ufbxw_scene *scene, ufbxw_id id, const char *name, size_t name_len);
ufbxw_abi ufbxw_string ufbxw_get_name(ufbxw_scene *scene, ufbxw_id id);

// TODO: Connect function taht takes connection type as an argument
ufbxw_abi void ufbxw_connect(ufbxw_scene *scene, ufbxw_id src, ufbxw_id dst);
ufbxw_abi void ufbxw_connect_prop(ufbxw_scene *scene, ufbxw_id src, const char *src_prop, ufbxw_id dst, const char *dst_prop);
ufbxw_abi void ufbxw_connect_prop_len(ufbxw_scene *scene, ufbxw_id src, const char *src_prop, size_t src_prop_len, ufbxw_id dst, const char *dst_prop, size_t dst_prop_len);

ufbxw_abi void ufbxw_set_bool(ufbxw_scene *scene, ufbxw_id id, const char *prop, bool value);
ufbxw_abi void ufbxw_set_int(ufbxw_scene *scene, ufbxw_id id, const char *prop, int32_t value);
ufbxw_abi void ufbxw_set_int64(ufbxw_scene *scene, ufbxw_id id, const char *prop, int64_t value);
ufbxw_abi void ufbxw_set_real(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_real value);
ufbxw_abi void ufbxw_set_vec2(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_vec2 value);
ufbxw_abi void ufbxw_set_vec3(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_vec3 value);
ufbxw_abi void ufbxw_set_vec4(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_vec4 value);
ufbxw_abi void ufbxw_set_string(ufbxw_scene *scene, ufbxw_id id, const char *prop, const char *value);

ufbxw_abi void ufbxw_add_bool(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, bool value);
ufbxw_abi void ufbxw_add_int(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, int32_t value);
ufbxw_abi void ufbxw_add_int64(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, int64_t value);
ufbxw_abi void ufbxw_add_real(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_real value);
ufbxw_abi void ufbxw_add_vec2(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_vec2 value);
ufbxw_abi void ufbxw_add_vec3(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_vec3 value);
ufbxw_abi void ufbxw_add_vec4(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_vec4 value);
ufbxw_abi void ufbxw_add_string(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, const char *value);

ufbxw_abi bool ufbxw_get_bool(ufbxw_scene *scene, ufbxw_id id, const char *prop);
ufbxw_abi int32_t ufbxw_get_int(ufbxw_scene *scene, ufbxw_id id, const char *prop);
ufbxw_abi int64_t ufbxw_get_int64(ufbxw_scene *scene, ufbxw_id id, const char *prop);
ufbxw_abi ufbxw_real ufbxw_get_real(ufbxw_scene *scene, ufbxw_id id, const char *prop);
ufbxw_abi ufbxw_vec2 ufbxw_get_vec2(ufbxw_scene *scene, ufbxw_id id, const char *prop);
ufbxw_abi ufbxw_vec3 ufbxw_get_vec3(ufbxw_scene *scene, ufbxw_id id, const char *prop);
ufbxw_abi ufbxw_vec4 ufbxw_get_vec4(ufbxw_scene *scene, ufbxw_id id, const char *prop);
ufbxw_abi ufbxw_string ufbxw_get_string(ufbxw_scene *scene, ufbxw_id id, const char *prop);

ufbxw_abi ufbxw_anim_prop ufbxw_animate_prop(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_anim_layer layer);
ufbxw_abi ufbxw_anim_prop ufbxw_animate_prop_len(ufbxw_scene *scene, ufbxw_id id, const char *prop, size_t prop_len, ufbxw_anim_layer layer);

// -- Node

ufbxw_abi ufbxw_node ufbxw_create_node(ufbxw_scene *scene);
ufbxw_abi ufbxw_node ufbxw_as_node(ufbxw_id id);

// -- Children

ufbxw_abi void ufbxw_node_set_parent(ufbxw_scene *scene, ufbxw_node node, ufbxw_node parent);
ufbxw_abi ufbxw_node ufbxw_node_get_parent(ufbxw_scene *scene, ufbxw_node node);
ufbxw_abi size_t ufbxw_node_get_num_children(ufbxw_scene *scene, ufbxw_node node);
ufbxw_abi ufbxw_node ufbxw_node_get_child(ufbxw_scene *scene, ufbxw_node node, size_t index);

// -- Local transformation of the node

// Local translation.
// FBX property: `Lcl Translation`.
ufbxw_abi void ufbxw_node_set_translation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 translation);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_translation(ufbxw_scene *scene, ufbxw_node node);

// Local rotation in Euler angles (degrees).
// Rotation order is controlled by `rotation_order`.
// FBX property: `Lcl Rotation`.
ufbxw_abi void ufbxw_node_set_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_rotation(ufbxw_scene *scene, ufbxw_node node);

// Local scaling of the node.
// FBX property: `Lcl Scaling`.
ufbxw_abi void ufbxw_node_set_scaling(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 scaling);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_scaling(ufbxw_scene *scene, ufbxw_node node);

// Order which `rotation` is applied in.
// FBX property: `RotationOrder`.
ufbxw_abi void ufbxw_node_set_rotation_order(ufbxw_scene *scene, ufbxw_node node, ufbxw_rotation_order order);
ufbxw_abi ufbxw_rotation_order ufbxw_node_get_rotation_order(ufbxw_scene *scene, ufbxw_node node);

// Set the rotation as a quaternion.
ufbxw_abi void ufbxw_node_set_rotation_quat(ufbxw_scene *scene, ufbxw_node node, ufbxw_quat rotation, ufbxw_rotation_order order);

// -- Advanced transform

// Controls how transformations are inherited to from the parent.
// See `ufbxw_inherit_type` for more information.
// FBX property: `InheritType`.
ufbxw_abi void ufbxw_node_set_inherit_type(ufbxw_scene *scene, ufbxw_node node, ufbxw_inherit_type order);
ufbxw_abi ufbxw_inherit_type ufbxw_node_get_inherit_type(ufbxw_scene *scene, ufbxw_node node);

// Rotation applied before the main `rotation`.
// FBX property: `PreRotation`.
ufbxw_abi void ufbxw_node_set_pre_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_pre_rotation(ufbxw_scene *scene, ufbxw_node node);

// Rotation applied after the main `rotation`.
// FBX property: `PostRotation`.
ufbxw_abi void ufbxw_node_set_post_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_post_rotation(ufbxw_scene *scene, ufbxw_node node);

// Offset applied before `rotation`.
// FBX property: `RotationOffset`.
ufbxw_abi void ufbxw_node_set_rotation_offset(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 offset);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_rotation_offset(ufbxw_scene *scene, ufbxw_node node);

// Pivot around which `rotation` is applied.
// FBX property: `RotationOffset`.
ufbxw_abi void ufbxw_node_set_rotation_pivot(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 pivot);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_rotation_pivot(ufbxw_scene *scene, ufbxw_node node);

// Offset applied before `scaling`.
// FBX property: `ScalingOffset`.
ufbxw_abi void ufbxw_node_set_scaling_offset(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 offset);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_scaling_offset(ufbxw_scene *scene, ufbxw_node node);

// Pivot relative to which `scaling` is applied.
// FBX property: `ScalingOffset`.
ufbxw_abi void ufbxw_node_set_scaling_pivot(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 pivot);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_scaling_pivot(ufbxw_scene *scene, ufbxw_node node);

// -- Geometric transform
//
// This transform is only applied to the geometry or other attributes, and is not inherited to children.

// Geometric translation only applied to the geometry, and not inherited to children.
// FBX property: `GeometricTranslation`.
ufbxw_abi void ufbxw_node_set_geometric_translation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 geometric_translation);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_geometric_translation(ufbxw_scene *scene, ufbxw_node node);

// Geometric rotation only applied to the geometry, and not inherited to children.
// Rotation order is always `UFBXW_ROTATION_ORDER_XYZ`.
// FBX property: `GeometricRotation`.
ufbxw_abi void ufbxw_node_set_geometric_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 geometric_rotation);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_geometric_rotation(ufbxw_scene *scene, ufbxw_node node);

// Geometric scaling only applied to the geometry, and not inherited to children.
// FBX property: `GeometricScaling`.
ufbxw_abi void ufbxw_node_set_geometric_scaling(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 geometric_scaling);
ufbxw_abi ufbxw_vec3 ufbxw_node_get_geometric_scaling(ufbxw_scene *scene, ufbxw_node node);

// -- Visibility

// Controls whether the node should be visible.
// FBX property: `Visibility`.
ufbxw_abi void ufbxw_node_set_visibility(ufbxw_scene *scene, ufbxw_node node, bool visible);
ufbxw_abi bool ufbxw_node_get_visibility(ufbxw_scene *scene, ufbxw_node node);

// Controls whether the node should inherit the visibility of its parent.
// FBX property: `Visibility Inheritance`.
ufbxw_abi void ufbxw_node_set_visibility_inheritance(ufbxw_scene *scene, ufbxw_node node, bool inherit);
ufbxw_abi bool ufbxw_node_get_visibility_inheritance(ufbxw_scene *scene, ufbxw_node node);

// -- Animation

ufbxw_abi ufbxw_anim_prop ufbxw_node_animate_translation(ufbxw_scene *scene, ufbxw_node node, ufbxw_anim_layer layer);
ufbxw_abi ufbxw_anim_prop ufbxw_node_animate_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_anim_layer layer);
ufbxw_abi ufbxw_anim_prop ufbxw_node_animate_scaling(ufbxw_scene *scene, ufbxw_node node, ufbxw_anim_layer layer);

// -- Transform helpers

ufbxw_abi ufbxw_transform ufbxw_node_get_local_transform(ufbxw_scene *scene, ufbxw_node node);
ufbxw_abi ufbxw_matrix ufbxw_node_get_global_transform(ufbxw_scene *scene, ufbxw_node node);

// -- Attributes

ufbxw_abi void ufbxw_node_set_attribute(ufbxw_scene *scene, ufbxw_node node, ufbxw_id attrib);

// -- Mesh

ufbxw_abi ufbxw_mesh ufbxw_create_mesh(ufbxw_scene *scene);
ufbxw_abi ufbxw_mesh ufbxw_as_mesh(ufbxw_id id);

ufbxw_abi void ufbxw_mesh_set_vertices(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_vec3_buffer vertices);

ufbxw_abi void ufbxw_mesh_set_triangles(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer indices);
ufbxw_abi void ufbxw_mesh_set_polygons(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer indices, ufbxw_int_buffer face_offsets);
ufbxw_abi void ufbxw_mesh_set_fbx_polygon_vertex_index(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer polygon_vertex_index);

ufbxw_abi void ufbxw_mesh_set_normals(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_vec3_buffer normals, ufbxw_attribute_mapping mapping);

ufbxw_abi void ufbxw_mesh_set_uvs(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec2_buffer uvs, ufbxw_attribute_mapping mapping);
ufbxw_abi void ufbxw_mesh_set_uvs_indexed(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec2_buffer uvs, ufbxw_int_buffer indices, ufbxw_attribute_mapping mapping);

ufbxw_abi void ufbxw_mesh_set_tangents(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec3_buffer tangents, ufbxw_attribute_mapping mapping);
ufbxw_abi void ufbxw_mesh_set_binormals(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec3_buffer binormals, ufbxw_attribute_mapping mapping);

ufbxw_abi void ufbxw_mesh_set_colors(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec4_buffer colors, ufbxw_attribute_mapping mapping);
ufbxw_abi void ufbxw_mesh_set_colors_indexed(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec4_buffer colors, ufbxw_int_buffer indices, ufbxw_attribute_mapping mapping);

ufbxw_abi void ufbxw_mesh_set_single_material(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t material_index);
ufbxw_abi void ufbxw_mesh_set_face_material(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer material_indices);

ufbxw_abi void ufbxw_mesh_set_fbx_edges(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer edges);

ufbxw_abi void ufbxw_mesh_set_attribute(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_mesh_attribute attribute, int32_t set, const ufbxw_mesh_attribute_desc *desc);

ufbxw_abi void ufbxw_mesh_set_attribute_name(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_mesh_attribute attribute, int32_t set, const char *name);
ufbxw_abi void ufbxw_mesh_set_attribute_name_len(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_mesh_attribute attribute, int32_t set, const char *name, size_t name_len);

ufbxw_vec3_buffer ufbxw_mesh_get_vertices(ufbxw_scene *scene, ufbxw_mesh mesh);

ufbxw_abi void ufbxw_mesh_add_instance(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_node node);

// -- Skin deformer

ufbxw_abi ufbxw_skin_deformer ufbxw_create_skin_deformer(ufbxw_scene *scene, ufbxw_mesh mesh);

ufbxw_abi void ufbxw_skin_deformer_add_mesh(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_mesh mesh);

ufbxw_abi void ufbxw_skin_deformer_set_skinning_type(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_skinning_type type);
ufbxw_abi ufbxw_skinning_type ufbxw_skin_deformer_get_skinning_type(ufbxw_scene *scene, ufbxw_skin_deformer skin);

ufbxw_abi void ufbxw_skin_deformer_set_mesh_bind_transform(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_matrix matrix);

ufbxw_abi void ufbxw_skin_deformer_set_bind_pose(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_bind_pose pose);

// -- Skin cluster

ufbxw_abi ufbxw_skin_cluster ufbxw_create_skin_cluster(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_node node);

ufbxw_abi void ufbxw_skin_cluster_set_deformer(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_skin_deformer skin);
ufbxw_abi void ufbxw_skin_cluster_set_node(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_node node);

ufbxw_abi void ufbxw_skin_cluster_set_weights(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_int_buffer indices, ufbxw_real_buffer weights);

// TODO: Better naming?
ufbxw_abi void ufbxw_skin_cluster_set_transform(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_matrix matrix);
ufbxw_abi void ufbxw_skin_cluster_set_link_transform(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_matrix matrix);

// -- Blend deformer

ufbxw_abi ufbxw_blend_deformer ufbxw_create_blend_deformer(ufbxw_scene *scene, ufbxw_mesh mesh);

ufbxw_abi void ufbxw_blend_deformer_add_mesh(ufbxw_scene *scene, ufbxw_blend_deformer deformer, ufbxw_mesh mesh);

// -- Blend channel

ufbxw_abi ufbxw_blend_channel ufbxw_create_blend_channel(ufbxw_scene *scene, ufbxw_blend_deformer deformer);

ufbxw_abi void ufbxw_blend_channel_set_deformer(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_blend_deformer deformer);

ufbxw_abi void ufbxw_blend_channel_set_shape(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_blend_shape shape);

ufbxw_abi void ufbxw_blend_channel_add_shape(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_blend_shape shape, ufbxw_real target_weight);

ufbxw_abi void ufbxw_blend_channel_set_weight(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_real weight);
ufbxw_abi ufbxw_real ufbxw_blend_channel_get_weight(ufbxw_scene *scene, ufbxw_blend_channel channel);

// -- Blend shape

ufbxw_abi ufbxw_blend_shape ufbxw_create_blend_shape(ufbxw_scene *scene);

ufbxw_abi void ufbxw_blend_shape_set_offsets(ufbxw_scene *scene, ufbxw_blend_shape shape, ufbxw_int_buffer indices, ufbxw_vec3_buffer offsets);
ufbxw_abi void ufbxw_blend_shape_set_normals(ufbxw_scene *scene, ufbxw_blend_shape shape, ufbxw_vec3_buffer normals);

// -- Light

typedef enum ufbxw_light_type {
	UFBXW_LIGHT_POINT,
	UFBXW_LIGHT_DIRECTIONAL,
	UFBXW_LIGHT_SPOT,
	UFBXW_LIGHT_AREA,
	UFBXW_LIGHT_VOLUME,
} ufbxw_light_type;

typedef enum ufbxw_light_decay {
	UFBXW_LIGHT_DECAY_NONE,      // < 1 (no decay)
	UFBXW_LIGHT_DECAY_LINEAR,    // < 1 / d
	UFBXW_LIGHT_DECAY_QUADRATIC, // < 1 / d^2 (physically accurate)
	UFBXW_LIGHT_DECAY_CUBIC,     // < 1 / d^3
} ufbxw_light_decay;

ufbxw_abi ufbxw_light ufbxw_create_light(ufbxw_scene *scene, ufbxw_node node);

// Color of the light.
// FBX property: `Color`.
ufbxw_abi void ufbxw_light_set_color(ufbxw_scene *scene, ufbxw_light light, ufbxw_vec3 color);
ufbxw_abi ufbxw_vec3 ufbxw_light_get_color(ufbxw_scene *scene, ufbxw_light light);

// Intensity of the light.
// Usually multiplied by `100.0` to conventional values.
// FBX property: `Intensity`.
ufbxw_abi void ufbxw_light_set_intensity(ufbxw_scene *scene, ufbxw_light light, ufbxw_real intensity);
ufbxw_abi ufbxw_real ufbxw_light_get_intensity(ufbxw_scene *scene, ufbxw_light light);

// Decay type of the light.
// FBX property: `DecayType`.
ufbxw_abi void ufbxw_light_set_decay(ufbxw_scene *scene, ufbxw_light light, ufbxw_light_decay decay);
ufbxw_abi ufbxw_light_decay ufbxw_light_get_decay(ufbxw_scene *scene, ufbxw_light light);

// Type of the light.
// FBX property: `LightType`.
ufbxw_abi void ufbxw_light_set_type(ufbxw_scene *scene, ufbxw_light light, ufbxw_light_type type);
ufbxw_abi ufbxw_light_type ufbxw_light_get_type(ufbxw_scene *scene, ufbxw_light light);

// Spotlight inner angle.
// FBX property: `InnerAngle`.
ufbxw_abi void ufbxw_light_set_inner_angle(ufbxw_scene *scene, ufbxw_light light, ufbxw_real value);
ufbxw_abi ufbxw_real ufbxw_light_get_inner_angle(ufbxw_scene *scene, ufbxw_light light);

// Spotlight outer angle.
// FBX property: `OuterAngle`.
ufbxw_abi void ufbxw_light_set_outer_angle(ufbxw_scene *scene, ufbxw_light light, ufbxw_real value);
ufbxw_abi ufbxw_real ufbxw_light_get_outer_angle(ufbxw_scene *scene, ufbxw_light light);

// TODO: Spot lights etc.

// -- Camera

ufbxw_abi ufbxw_camera ufbxw_create_camera(ufbxw_scene *scene, ufbxw_node node);

// -- Bind pose

ufbxw_abi ufbxw_bind_pose ufbxw_create_bind_pose(ufbxw_scene *scene);

ufbxw_abi void ufbxw_bind_pose_add_node(ufbxw_scene *scene, ufbxw_bind_pose pose, ufbxw_node node, ufbxw_matrix matrix);

// -- Animation stack

ufbxw_abi ufbxw_anim_stack ufbxw_get_default_anim_stack(ufbxw_scene *scene);
ufbxw_abi ufbxw_anim_stack ufbxw_create_anim_stack(ufbxw_scene *scene);

ufbxw_abi ufbxw_anim_layer ufbxw_anim_stack_get_layer(ufbxw_scene *scene, ufbxw_anim_stack stack, size_t index);

ufbxw_abi void ufbxw_anim_stack_set_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack, ufbxw_ktime time_begin, ufbxw_ktime time_end);
ufbxw_abi void ufbxw_anim_stack_set_reference_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack, ufbxw_ktime time_begin, ufbxw_ktime time_end);

ufbxw_abi ufbxw_ktime_range ufbxw_anim_stack_get_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack);
ufbxw_abi ufbxw_ktime_range ufbxw_anim_stack_get_reference_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack);

ufbxw_abi void ufbxw_set_active_anim_stack(ufbxw_scene *scene, ufbxw_anim_stack stack);
ufbxw_abi ufbxw_anim_stack ufbxw_get_active_anim_stack(const ufbxw_scene *scene);

// -- Animation layer

ufbxw_abi ufbxw_anim_layer ufbxw_get_default_anim_layer(ufbxw_scene *scene);
ufbxw_abi ufbxw_anim_layer ufbxw_create_anim_layer(ufbxw_scene *scene, ufbxw_anim_stack stack);

ufbxw_abi void ufbxw_anim_layer_set_weight(ufbxw_scene *scene, ufbxw_anim_layer layer, ufbxw_real weight);

ufbxw_abi void ufbxw_anim_layer_set_stack(ufbxw_scene *scene, ufbxw_anim_layer layer, ufbxw_anim_stack stack);

// -- Animation property

ufbxw_abi ufbxw_anim_curve ufbxw_anim_get_curve(ufbxw_scene *scene, ufbxw_anim_prop anim, size_t index);
ufbxw_abi void ufbxw_anim_set_default_value(ufbxw_scene *scene, ufbxw_anim_prop anim, size_t index, ufbxw_real value);

ufbxw_abi void ufbxw_anim_add_keyframe_real(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_ktime time, ufbxw_real value, uint32_t type);
ufbxw_abi void ufbxw_anim_add_keyframe_vec2(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_ktime time, ufbxw_vec2 value, uint32_t type);
ufbxw_abi void ufbxw_anim_add_keyframe_vec3(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_ktime time, ufbxw_vec3 value, uint32_t type);

ufbxw_abi void ufbxw_anim_add_keyframe_real_key(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_keyframe_real key);
ufbxw_abi void ufbxw_anim_add_keyframe_vec2_key(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_keyframe_vec2 key);
ufbxw_abi void ufbxw_anim_add_keyframe_vec3_key(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_keyframe_vec3 key);

ufbxw_abi void ufbxw_anim_finish_keyframes(ufbxw_scene *scene, ufbxw_anim_prop anim);

ufbxw_abi void ufbxw_anim_set_layer(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_anim_layer layer);

// -- Animation curve

ufbxw_abi void ufbxw_anim_curve_add_keyframe(ufbxw_scene *scene, ufbxw_anim_curve curve, ufbxw_ktime time, ufbxw_real value, uint32_t type);
ufbxw_abi void ufbxw_anim_curve_add_keyframe_key(ufbxw_scene *scene, ufbxw_anim_curve curve, ufbxw_keyframe_real key);
ufbxw_abi void ufbxw_anim_curve_finish_keyframes(ufbxw_scene *scene, ufbxw_anim_curve curve);

typedef struct ufbxw_anim_curve_data_desc {
	uint32_t _begin_zero;

	ufbxw_long_buffer key_times;
	ufbxw_float_buffer key_values;
	uint32_t key_flags;

	// Optional: FBX internal values
	ufbxw_int_buffer fbx_attr_refcounts;
	ufbxw_int_buffer fbx_attr_flags;
	ufbxw_float_buffer fbx_attr_data;

	uint32_t _end_zero;
} ufbxw_anim_curve_data_desc;

// Advanced API for manually specifying the animation.
ufbxw_abi void ufbxw_anim_curve_set_data(ufbxw_scene *scene, ufbxw_anim_curve curve, const ufbxw_anim_curve_data_desc *data);

// -- Scene info

ufbxw_abi ufbxw_id ufbxw_get_scene_info_id(ufbxw_scene *scene);
ufbxw_abi ufbxw_id ufbxw_get_global_settings_id(ufbxw_scene *scene);

// -- Templates

ufbxw_abi ufbxw_id ufbxw_get_template_id(ufbxw_scene *scene, ufbxw_element_type type);

// -- Pre-saving

typedef struct ufbxw_save_info {
	uint32_t _begin_zero;

	// Usually aboslute path of the exported FBX
	ufbxw_string document_url;

	// Application info
	ufbxw_string application_vendor;
	ufbxw_string application_name;
	ufbxw_string application_version;

	// Defaults to `document_url`
	ufbxw_string src_document_url;
	ufbxw_string original_filename;

	// Defaults to application info
	ufbxw_string original_application_vendor;
	ufbxw_string original_application_name;
	ufbxw_string original_application_version;

	// Timestamps, defaults to current time UTC
	bool no_default_date_time;
	ufbxw_datetime date_time_utc;

	// Defaults to `date_time_utc`.
	ufbxw_datetime original_date_time_utc;

	uint32_t _end_zero;
} ufbxw_save_info;

// Sets a bunch of SceneInfo properties conveniently
ufbxw_abi void ufbxw_set_save_info(ufbxw_scene *scene, const ufbxw_save_info *info);

typedef struct ufbxw_prepare_opts {
	bool finish_keyframes;
	bool patch_anim_stack_times;
	bool patch_anim_stack_reference_times;
	bool patch_global_settings_times;
	bool add_missing_skeletons;
	bool add_missing_bind_poses;
} ufbxw_prepare_opts;

ufbxw_abi_data const ufbxw_prepare_opts ufbxw_default_prepare_opts;

ufbxw_abi void ufbxw_prepare_scene(ufbxw_scene *scene, const ufbxw_prepare_opts *opts);

ufbxw_abi void ufbxw_validate_scene(const ufbxw_scene *scene);

// -- IO callbacks

// Write at a specified offset in the file
typedef bool ufbxw_write_fn(void *user, uint64_t offset, const void *data, size_t size);

// Close the file
typedef void ufbxw_close_fn(void *user);

typedef struct ufbxw_write_stream {
	ufbxw_write_fn *write_fn; // < Required
	ufbxw_close_fn *close_fn; // < Optional

	// Context passed to other functions
	void *user;
} ufbxw_write_stream;

ufbxw_abi bool ufbxw_open_file_write(ufbxw_write_stream *stream, const char *path, size_t path_len, ufbxw_error *error);

// TODO: Unify all these kind of APIs, they're all a bit different now...
typedef void *ufbxw_thread_sync_init_fn(void *user);
typedef void ufbxw_thread_sync_wait_fn(void *user, void *ctx, uint32_t *p_value, uint32_t ref_value);
typedef void ufbxw_thread_sync_notify_fn(void *user, void *ctx, uint32_t *p_value, uint32_t wake_count);
typedef void ufbxw_thread_sync_free_fn(void *user, void *ctx);

//  support interface.
typedef struct ufbxw_thread_sync {
	ufbxw_thread_sync_init_fn *init_fn;     // < Optional
	ufbxw_thread_sync_wait_fn *wait_fn;     // < Required
	ufbxw_thread_sync_notify_fn *notify_fn; // < Required
	ufbxw_thread_sync_free_fn *free_fn;     // < Optional
	void *user;
} ufbxw_thread_sync;

// Internal thread pool handle.
// Passed to `ufbxw_thread_pool_try/blocking_run_tasks()` from an user thread to run ufbxw tasks.
// HINT: This context can store a user pointer via `ufbxw_thread_pool_set_user_ptr()`.
typedef uintptr_t ufbxw_thread_pool_context;

typedef bool ufbxw_thread_pool_init_fn(void *user, ufbxw_thread_pool_context ctx, size_t num_threads);
typedef void ufbxw_thread_pool_run_fn(void *user, ufbxw_thread_pool_context ctx, uint32_t count);
typedef void ufbxw_thread_pool_free_fn(void *user, ufbxw_thread_pool_context ctx);

// Thread pool interface.
// See functions above for more information.
typedef struct ufbxw_thread_pool {
	ufbxw_thread_pool_init_fn *init_fn; // < Optional
	ufbxw_thread_pool_run_fn *run_fn;   // < Optional
	ufbxw_thread_pool_free_fn *free_fn; // < Optional
	void *user;
} ufbxw_thread_pool;

// -- Deflate

typedef struct ufbxw_deflate_advance_status {
	size_t bytes_written;
	size_t bytes_read;
} ufbxw_deflate_advance_status;

typedef enum ufbxw_deflate_advance_flag {
	UFBXW_DEFLATE_ADVANCE_FLAG_FLUSH = 0x1,
	UFBXW_DEFLATE_ADVANCE_FLAG_FINISH = 0x2,

	UFBXW_DEFLAT_ADVANCE_FLAG_FORCE_32BIT = 0x7fffffff,
} ufbxw_deflate_advance_flag;

typedef enum ufbxw_deflate_advance_result {
	// The compressor is finished with the current stream.
	UFBXW_DEFLATE_ADVANCE_RESULT_COMPLETED = 1,

	// There is still buffered output left in the compressor.
	// If you return this, ufbxw will call `advance_fn()` again with more output space.
	UFBXW_DEFLATE_ADVANCE_RESULT_INCOMPLETE = 2,

	// Error in the compressor.
	UFBXW_DEFLATE_ADVANCE_RESULT_ERROR = 3,

	UFBXW_DEFLATE_ADVANCE_RESULT_FORCE_32BIT = 0x7fffffff,
} ufbxw_deflate_advance_result;

// Begin DEFLATE compressing a new stream.
// You must return the buffer size you require for decompression.
// Returning `0` is allowed to indicate that the decompressor supports streaming.
typedef size_t ufbxw_deflate_begin_fn(void *user, size_t input_size);

// Advance the compression stream.
typedef ufbxw_deflate_advance_result ufbxw_deflate_advance_fn(void *user, ufbxw_deflate_advance_status *status, void *dst, size_t dst_size, const void *src, size_t src_size, uint32_t flags);

// Finish compressing one stream.
typedef void ufbxw_deflate_end_fn(void *user);

// Free the deflate compressor.
typedef void ufbxw_deflate_free_fn(void *user);

// DEFLATE compressor interface.
typedef struct ufbxw_deflate_compressor {
	ufbxw_deflate_begin_fn *begin_fn;
	ufbxw_deflate_advance_fn *advance_fn;
	ufbxw_deflate_end_fn *end_fn;
	ufbxw_deflate_free_fn *free_fn;
	void *user;
} ufbxw_deflate_compressor;

// Function for instantiating deflate compressors

typedef bool ufbxw_deflate_create_fn(void *user, ufbxw_deflate_compressor *compressor, int32_t compression_level);

typedef struct ufbxw_deflate_create_cb {
	ufbxw_deflate_create_fn *fn;
	void *user;
} ufbxw_deflate_create_cb;

typedef struct ufbxw_deflate {
	ufbxw_deflate_create_cb create_cb;
	bool streaming_input;
	bool streaming_output;
} ufbxw_deflate;

// -- ASCII formatting

// TODO: Should round trip be default?
typedef enum ufbxw_ascii_float_format {
	// Fixed-precision formatting, matching what the FBX SDK outputs.
	// Equivalent to printing the floats as `%.15g` (double) and `%.7g` (float)
	UFBXW_ASCII_FLOAT_FORMAT_FIXED_PRECISION,

	// Print the floats using an implementation-defined method that guarantees round trip.
	UFBXW_ASCII_FLOAT_FORMAT_ROUND_TRIP,

} ufbxw_ascii_float_format;

typedef size_t ufbxw_ascii_format_int_fn(void *user, char *dst, size_t dst_size, const int32_t *src, size_t src_count);
typedef size_t ufbxw_ascii_format_long_fn(void *user, char *dst, size_t dst_size, const int64_t *src, size_t src_count);
typedef size_t ufbxw_ascii_format_float_fn(void *user, char *dst, size_t dst_size, const float *src, size_t src_count, ufbxw_ascii_float_format format);
typedef size_t ufbxw_ascii_format_double_fn(void *user, char *dst, size_t dst_size, const double *src, size_t src_count, ufbxw_ascii_float_format format);

typedef struct ufbxw_ascii_formatter {
	ufbxw_ascii_format_int_fn *format_int_fn;
	ufbxw_ascii_format_long_fn *format_long_fn;
	ufbxw_ascii_format_float_fn *format_float_fn;
	ufbxw_ascii_format_double_fn *format_double_fn;
	void *user;
} ufbxw_ascii_formatter;

// -- Writing API

typedef enum ufbxw_save_format {
	UFBXW_SAVE_FORMAT_BINARY,
	UFBXW_SAVE_FORMAT_ASCII,
} ufbxw_save_format;

typedef struct ufbxw_save_opts {
	uint32_t _begin_zero;

	ufbxw_save_format format;
	uint32_t version;

	ufbxw_deflate deflate;

	ufbxw_ascii_formatter ascii_formatter;

	ufbxw_thread_sync thread_sync;

	ufbxw_thread_pool thread_pool;

	ufbxw_allocator allocator;

	// Compression level.
	// Defaults to `6`.
	int32_t compression_level;

	// Window size to use when doing streaming deflate compression.
	// Defaults to `64kB`.
	size_t deflate_window_size;

	// How to format floating point numbers in ASCII files.
	ufbxw_ascii_float_format ascii_float_format;

	// TODO: Do not save animation
	bool ignore_animation;

	bool debug_comments;

	bool no_default_timestamp;

	// Local date time when the file was saved.
	ufbxw_datetime local_timestamp;

	// Override the ufbx_write creator.
	// Generally, you should prefer using `ufbxw_save_info.application_name` to indicate your application,
	// as Creator is used for the actual file writer (in most cases, FBX SDK).
	bool enable_override_creator;
	ufbxw_string override_creator;

	// Buffer size to use for writing streaming output.
	size_t buffer_size;

	// Limits for threading
	size_t threaded_min_deflate_bytes;
	size_t threaded_min_ascii_floats;
	size_t threaded_min_ascii_ints;

	uint32_t _end_zero;
} ufbxw_save_opts;

ufbxw_abi bool ufbxw_save_file(ufbxw_scene *scene, const char *path, const ufbxw_save_opts *opts, ufbxw_error *error);
ufbxw_abi bool ufbxw_save_file_len(ufbxw_scene *scene, const char *path, size_t path_len, const ufbxw_save_opts *opts, ufbxw_error *error);

ufbxw_abi bool ufbxw_save_stream(ufbxw_scene *scene, ufbxw_write_stream *stream, const ufbxw_save_opts *opts, ufbxw_error *error);

// -- Thread pool

typedef enum ufbxw_task_run_result {
	UFBXW_TASK_RUN_RESULT_NO_TASKS,
	UFBXW_TASK_RUN_RESULT_COMPLETED,
	UFBXW_TASK_RUN_RESULT_FAILED,
	UFBXW_TASK_RUN_RESULT_ALL_FINISHED,
} ufbxw_task_run_result;

ufbxw_unsafe ufbxw_abi ufbxw_task_run_result ufbxw_thread_pool_try_run_tasks(ufbxw_thread_pool_context ctx, uint32_t thread_id_hint, size_t max_count);
ufbxw_unsafe ufbxw_abi ufbxw_task_run_result ufbxw_thread_pool_blocking_run_tasks(ufbxw_thread_pool_context ctx, uint32_t thread_id_hint, size_t max_count);

// Get or set an arbitrary user pointer for the thread pool context.
// `ufbxw_thread_pool_get_user_ptr()` returns `NULL` if unset.
ufbxw_unsafe ufbxw_abi void ufbxw_thread_pool_set_user_ptr(ufbxw_thread_pool_context ctx, void *user_ptr);
ufbxw_unsafe ufbxw_abi void *ufbxw_thread_pool_get_user_ptr(ufbxw_thread_pool_context ctx);

// -- Utility

ufbxw_abi ufbxw_matrix ufbxw_transform_to_matrix(const ufbxw_transform *t);

ufbxw_abi ufbxw_vec3 ufbxw_quat_rotate_vec3(ufbxw_quat q, ufbxw_vec3 v);
ufbxw_abi ufbxw_vec3 ufbxw_quat_to_euler(ufbxw_quat q, ufbxw_rotation_order order);
ufbxw_abi ufbxw_quat ufbxw_euler_to_quat(ufbxw_vec3 v, ufbxw_rotation_order order);

ufbxw_abi ufbxw_string ufbxw_str(const char *str);

#ifdef __cplusplus
}
#endif

#endif


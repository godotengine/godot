#include "../ufbx_write.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define array_count(arr) (sizeof(arr) / sizeof(*(arr)))

static const double double_values[] = {
	0.0,
	DBL_TRUE_MIN,
	DBL_TRUE_MIN * 2.0,
	DBL_TRUE_MIN * 5.0,
	DBL_MIN * 2.0,
	DBL_MIN * 5.0,
	DBL_EPSILON,
	DBL_EPSILON * 2.0,
	DBL_EPSILON * 5.0,
	1e-10,
	1e-9,
	1e-8,
	1e-7,
	1e-6,
	1e-5,
	1e-4,
	1e-3,
	1e-2,
	1e-1,
	0.5,
	1.0,
	2.0,
	3.14159265359,
	0.123456789,
	123456789.0,
	1e2,
	1e3,
	1e4,
	1e5,
	1e6,
	1e7,
	1e8,
	1e9,
	1e10,
	1e11,
	1e12,
	1e13,
	1e14,
	1e15,
	DBL_MAX / 5.0,
	DBL_MAX / 2.0,
	DBL_MAX,
	1.0/3.0,
	1.0/5.0,
	1.0/7.0,
	1.0/23.0,
	3.0/7.0,
};

static const float float_values[] = {
	0.0f,
	FLT_TRUE_MIN,
	FLT_TRUE_MIN * 2.0f,
	FLT_TRUE_MIN * 5.0f,
	FLT_MIN * 2.0f,
	FLT_MIN * 5.0f,
	FLT_EPSILON,
	FLT_EPSILON * 2.0f,
	FLT_EPSILON * 5.0f,
	1e-10f,
	1e-9f,
	1e-8f,
	1e-7f,
	1e-6f,
	1e-5f,
	1e-4f,
	1e-3f,
	1e-2f,
	1e-1f,
	0.5f,
	1.0f,
	2.0f,
	3.14159265359f,
	0.123456789f,
	123456789.0f,
	1e2f,
	1e3f,
	1e4f,
	1e5f,
	1e6f,
	1e7f,
	1e8f,
	1e9f,
	1e10f,
	1e11f,
	1e12f,
	1e13f,
	1e14f,
	1e15f,
	FLT_MAX / 5.0f,
	FLT_MAX / 2.0f,
	FLT_MAX,
	1.0f/3.0f,
	1.0f/5.0f,
	1.0f/7.0f,
	1.0f/23.0f,
	3.0f/7.0f,
};

int main(int argc, char **argv)
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);

	// Encode some tricky floats to mesh position (FP64) and animation values (FP32)

	size_t double_count = array_count(double_values);
	ufbxw_vec3_buffer vertex_buffer = ufbxw_create_vec3_buffer(scene, double_count);
	ufbxw_vec3_list vertex_list = ufbxw_edit_vec3_buffer(scene, vertex_buffer);

	for (size_t i = 0; i < double_count; i++) {
		double value = double_values[i];
		ufbxw_vec3 v;
		v.x = nextafter(value, -INFINITY);
		v.y = value;
		v.z = nextafter(value, +INFINITY);
		vertex_list.data[i] = v;
	}

	size_t float_count = array_count(float_values);
	ufbxw_float_buffer anim_value_buffer = ufbxw_create_float_buffer(scene, float_count * 3);
	ufbxw_float_list anim_value_list = ufbxw_edit_float_buffer(scene, anim_value_buffer);

	for (size_t i = 0; i < float_count; i++) {
		float value = float_values[i];
		float *v = anim_value_list.data + i * 3;
		v[0] = nextafterf(value, -INFINITY);
		v[1] = value;
		v[2] = nextafterf(value, +INFINITY);
	}

	// Create some dummy indices and animation times so that he FBX is valid

	ufbxw_int_buffer index_buffer = ufbxw_create_int_buffer(scene, double_count);
	ufbxw_int_list index_list = ufbxw_edit_int_buffer(scene, index_buffer);
	for (size_t i = 0; i < double_count; i++) {
		int32_t value = (int32_t)i;
		if (i + 1 == double_count) {
			value = ~value;
		}
		index_list.data[i] = value;
	}

	ufbxw_long_buffer anim_time_buffer = ufbxw_create_long_buffer(scene, float_count * 3);
	ufbxw_long_list anim_time_list = ufbxw_edit_long_buffer(scene, anim_time_buffer);
	for (size_t i = 0; i < float_count * 3; i++) {
		anim_time_list.data[i] = i * (UFBXW_KTIME_SECOND / 30);
	}

	// Plug the buffers into the scene

	ufbxw_node node = ufbxw_create_node(scene);
	ufbxw_set_name(scene, node.id, "Node");

	ufbxw_mesh mesh = ufbxw_create_mesh(scene);
	ufbxw_set_name(scene, mesh.id, "Mesh");

	ufbxw_mesh_set_vertices(scene, mesh, vertex_buffer);
	ufbxw_mesh_set_fbx_polygon_vertex_index(scene, mesh, index_buffer);

	ufbxw_anim_layer layer = ufbxw_get_default_anim_layer(scene);
	ufbxw_anim_prop anim_prop = ufbxw_node_animate_translation(scene, node, layer);
	ufbxw_anim_curve anim_curve = ufbxw_anim_get_curve(scene, anim_prop, 0);

	ufbxw_anim_curve_data_desc anim_data = { 0 };
	anim_data.key_times = anim_time_buffer;
	anim_data.key_values = anim_value_buffer;
	ufbxw_anim_curve_set_data(scene, anim_curve, &anim_data);

	{
		ufbxw_save_opts save_opts = { 0 };
		save_opts.version = 7400;
		save_opts.format = UFBXW_SAVE_FORMAT_BINARY;

		ufbxw_error error;
		bool ok = ufbxw_save_file(scene, "floats_binary.fbx", &save_opts, &error);
		if (!ok) {
			fprintf(stderr, "failed to save scene: %s\n", error.description);
			exit(1);
		}
	}

	{
		ufbxw_save_opts save_opts = { 0 };
		save_opts.version = 7400;
		save_opts.format = UFBXW_SAVE_FORMAT_ASCII;

		ufbxw_error error;
		bool ok = ufbxw_save_file(scene, "floats_ascii.fbx", &save_opts, &error);
		if (!ok) {
			fprintf(stderr, "failed to save scene: %s\n", error.description);
			exit(1);
		}
	}

	ufbxw_free_scene(scene);

	return 0;
}

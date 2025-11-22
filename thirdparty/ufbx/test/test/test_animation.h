#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "animation"

#if UFBXWT_IMPL
typedef struct {
	double time;
	ufbxw_vec3 value;
	uint32_t type;
} ufbxwt_keyframe_vec3;

static void ufbxwt_add_keyframes_vec3(ufbxw_scene *scene, ufbxw_anim_prop anim, const ufbxwt_keyframe_vec3 *keys, size_t count)
{
	for (size_t i = 0; i < count; i++) {
		ufbxw_ktime ktime = (ufbxw_ktime)(keys[i].time * UFBXW_KTIME_SECOND);
		ufbxw_anim_add_keyframe_vec3(scene, anim, ktime, keys[i].value, keys[i].type);
	}
}

static const ufbxwt_keyframe_vec3 ufbxwt_simple_keys[] = {
	{ 0.0, { 0.0f, 0.0f, 0.0f }, UFBXW_KEYFRAME_CUBIC_AUTO },
	{ 0.5, { 1.0f, 0.0f, 0.0f }, UFBXW_KEYFRAME_CUBIC_AUTO },
	{ 1.0, { -1.0f, 0.0f, 1.0f }, UFBXW_KEYFRAME_CUBIC_AUTO },
};

#endif

UFBXWT_SCENE_TEST(anim_simple)
#if UFBXWT_IMPL
{
	ufbxw_node node = ufbxwt_create_node(scene, "Node");

	ufbxw_anim_layer anim_layer = ufbxw_get_default_anim_layer(scene);
	ufbxw_anim_prop anim = ufbxw_node_animate_translation(scene, node, anim_layer);

	ufbxwt_add_keyframes_vec3(scene, anim, ufbxwt_simple_keys, ufbxwt_arraycount(ufbxwt_simple_keys));
}
#endif

UFBXWT_SCENE_CHECK(anim_simple)
#if UFBXWT_IMPL
{
	ufbx_node *node = ufbx_find_node(scene, "Node");
	ufbxwt_assert(node);

	for (size_t i = 0; i < ufbxwt_arraycount(ufbxwt_simple_keys); i++) {
		ufbxwt_keyframe_vec3 key = ufbxwt_simple_keys[i];
		ufbx_transform transform = ufbx_evaluate_transform(scene->anim, node, key.time);

		ufbx_vec3 value = { key.value.x, key.value.y, key.value.z };
		ufbxwt_assert_close_uvec3(err, transform.translation, value);
	}
}
#endif

#if UFBXWT_IMPL
static float ufbxwt_sin_anim_value(double time)
{
	double v = 0.0;
	v += sin(time * 1.0) / 1.0;
	v += sin(time * 2.0) / 2.0;
	v += sin(time * 3.0) / 3.0;
	v += sin(time * 4.0) / 2.0;
	v += sin(time * 10.0) / 7.0;
	return (float)v;
}
#endif

UFBXWT_SCENE_TEST(anim_buffers)
#if UFBXWT_IMPL
{
	ufbxw_node node = ufbxwt_create_node(scene, "Node");

	ufbxw_anim_layer anim_layer = ufbxw_get_default_anim_layer(scene);
	ufbxw_anim_prop anim = ufbxw_node_animate_translation(scene, node, anim_layer);

	ufbxw_anim_curve curve = ufbxw_anim_get_curve(scene, anim, 0);

	const size_t frame_count = 120;
	const size_t frame_rate = 30;

	ufbxw_long_buffer time_buffer = ufbxw_create_long_buffer(scene, frame_count);
	ufbxw_float_buffer value_buffer = ufbxw_create_float_buffer(scene, frame_count);

	ufbxw_long_list times = ufbxw_edit_long_buffer(scene, time_buffer);
	ufbxw_float_list values = ufbxw_edit_float_buffer(scene, value_buffer);

	ufbxwt_assert(times.count == frame_count);
	ufbxwt_assert(values.count == frame_count);

	for (size_t i = 0; i < frame_count; i++) {
		double time = (double)i / (double)frame_rate;
		int64_t ktime = (int64_t)(time * (double)UFBXW_KTIME_SECOND);
		float value = ufbxwt_sin_anim_value(time);

		times.data[i] = ktime;
		values.data[i] = value;
	}

	ufbxw_anim_curve_data_desc data = { 0 };
	data.key_times = time_buffer;
	data.key_values = value_buffer;
	data.key_flags = UFBXW_KEYFRAME_LINEAR;
	ufbxw_anim_curve_set_data(scene, curve, &data);

	// TODO: Proper API
	ufbxw_id global_settings = ufbxw_get_global_settings_id(scene);
	ufbxwt_assert(global_settings != 0);

	ufbxw_set_int(scene, global_settings, "TimeMode", (int32_t)UFBXW_TIME_MODE_30_FPS);
}
#endif

UFBXWT_SCENE_CHECK(anim_buffers)
#if UFBXWT_IMPL
{
	ufbx_node *node = ufbx_find_node(scene, "Node");
	ufbxwt_assert(node);

	ufbxwt_assert(scene->anim_layers.count == 1);
	ufbx_anim_layer *layer = scene->anim_layers.data[0];
	
	ufbx_anim_prop *anim_prop = ufbx_find_anim_prop(layer, &node->element, UFBX_Lcl_Translation);
	ufbxwt_assert(anim_prop);

	ufbx_anim_value *anim_value = anim_prop->anim_value;
	ufbxwt_assert(anim_value);

	ufbx_anim_curve *anim_curve = anim_value->curves[0];
	ufbxwt_assert(anim_curve);

	const size_t frame_count = 120;
	const size_t frame_rate = 30;

	ufbxwt_assert(anim_curve->keyframes.count == frame_count);
	for (size_t i = 0; i < anim_curve->keyframes.count; i++) {
		ufbx_keyframe key = anim_curve->keyframes.data[i];
		ufbxwt_assert(key.interpolation == UFBX_INTERPOLATION_LINEAR);

		ufbxwt_assert_close_real(err, (ufbxw_real)key.time, (ufbxw_real)(i / (double)frame_rate));

		float ref_value = ufbxwt_sin_anim_value(key.time);
		ufbxwt_assert_close_real(err, key.value, ref_value);
	}
}
#endif


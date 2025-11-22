#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "light"

UFBXWT_SCENE_TEST(light_simple)
#if UFBXWT_IMPL
{
	ufbxw_node node = ufbxwt_create_node(scene, "Node");

	ufbxw_light light = ufbxw_create_light(scene, node);

	ufbxw_vec3 color = { 0.25f, 0.5f, 0.75f };
	ufbxw_light_set_color(scene, light, color);
	ufbxw_light_set_intensity(scene, light, 50.0f);
}
#endif

UFBXWT_SCENE_CHECK(light_simple)
#if UFBXWT_IMPL
{
	ufbx_node *node = ufbx_find_node(scene, "Node");
	ufbxwt_assert(node);

	ufbx_light *light = node->light;
	ufbxwt_assert(light);

	ufbx_vec3 color = { 0.25f, 0.5f, 0.75f };
	ufbxwt_assert_close_uvec3(err, light->color, color);

	// Converted by ufbx
	ufbxwt_assert_close_real(err, light->intensity, 0.5f);
}
#endif


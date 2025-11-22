#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "skin"

UFBXWT_SCENE_TEST(skin_plane)
#if UFBXWT_IMPL
{
	ufbxw_node node = ufbxwt_create_node(scene, "Node");
	ufbxw_mesh mesh = ufbxw_create_mesh(scene);

	ufbxw_mesh_add_instance(scene, mesh, node);

	ufbxw_vec3 vertices[] = {
		{ -1.0f, 0.0f, -1.0f },
		{ +1.0f, 0.0f, -1.0f },
		{ -1.0f, 0.0f, +1.0f },
		{ +1.0f, 0.0f, +1.0f },
	};
	int32_t indices[] = {
		0, 2, 3, 1,
	};
	int32_t face_offsets[] = {
		0, 4,
	};

	ufbxw_vec3_buffer vertex_buffer = ufbxw_view_vec3_array(scene, vertices, ufbxwt_arraycount(vertices));
	ufbxw_int_buffer index_buffer = ufbxw_view_int_array(scene, indices, ufbxwt_arraycount(indices));
	ufbxw_int_buffer face_buffer = ufbxw_view_int_array(scene, face_offsets, ufbxwt_arraycount(face_offsets));

	ufbxw_mesh_set_vertices(scene, mesh, vertex_buffer);
	ufbxw_mesh_set_polygons(scene, mesh, index_buffer, face_buffer);

	ufbxw_node left = ufbxwt_create_node(scene, "Left");
	ufbxw_node right = ufbxwt_create_node(scene, "Right");

	ufbxw_skin_deformer skin = ufbxw_create_skin_deformer(scene, mesh);
	ufbxw_skin_cluster left_cluster = ufbxw_create_skin_cluster(scene, skin, left);
	ufbxw_skin_cluster right_cluster = ufbxw_create_skin_cluster(scene, skin, right);

	ufbxw_real full_weights[] = { 1.0f, 1.0f };
	int32_t left_indices[] = { 0, 2 };
	int32_t right_indices[] = { 1, 3 };

	ufbxw_real_buffer weight_buffer = ufbxw_view_real_array(scene, full_weights, ufbxwt_arraycount(full_weights));
	ufbxw_int_buffer left_buffer = ufbxw_view_int_array(scene, left_indices, ufbxwt_arraycount(left_indices));
	ufbxw_int_buffer right_buffer = ufbxw_view_int_array(scene, right_indices, ufbxwt_arraycount(right_indices));

	ufbxw_retain_buffer(scene, weight_buffer.id);
	ufbxw_skin_cluster_set_weights(scene, left_cluster, left_buffer, weight_buffer);

	ufbxw_retain_buffer(scene, weight_buffer.id);
	ufbxw_skin_cluster_set_weights(scene, right_cluster, right_buffer, weight_buffer);

	ufbxw_free_buffer(scene, weight_buffer.id);
}
#endif

UFBXWT_SCENE_CHECK(skin_plane)
#if UFBXWT_IMPL
{
	ufbx_node *node = ufbx_find_node(scene, "Node");
	ufbxwt_assert(node);

	ufbx_mesh *mesh = node->mesh;
	ufbxwt_assert(mesh);

	ufbxwt_assert(mesh->faces.count == 1);

	ufbx_face face = mesh->faces.data[0];
	ufbxwt_assert(face.index_begin == 0);
	ufbxwt_assert(face.num_indices == 4);

	ufbx_vec3 vertices[] = {
		{ -1.0f, 0.0f, -1.0f },
		{ -1.0f, 0.0f, +1.0f },
		{ +1.0f, 0.0f, +1.0f },
		{ +1.0f, 0.0f, -1.0f },
	};

	for (size_t i = 0; i < 4; i++) {
		ufbx_vec3 v = ufbx_get_vertex_vec3(&mesh->vertex_position, i);
		ufbxwt_assert_close_uvec3(err, v, vertices[i]);
	}
}
#endif


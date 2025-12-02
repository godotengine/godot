#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "mesh"

#if UFBXWT_IMPL
typedef ufbxw_real ufbxwt_grid_height_fn(void *user, ufbxw_real x, ufbxw_real y);

typedef struct {
	uint32_t resolution;
	ufbxwt_grid_height_fn *height_fn;
	void *height_user;
} ufbxwt_grid_mesh;

static size_t ufbxwt_grid_vertex_stream(void *user, ufbxw_vec3 *dst, size_t dst_size, size_t offset)
{
	ufbxwt_grid_mesh grid = *(ufbxwt_grid_mesh*)user;

	for (size_t i = 0; i < dst_size; i++) {
		uint32_t index = (uint32_t)(offset + i);
		uint32_t ix = index % grid.resolution;
		uint32_t iy = index / grid.resolution;

		ufbxw_real x = (ufbxw_real)ix / (ufbxw_real)(grid.resolution - 1);
		ufbxw_real y = (ufbxw_real)iy / (ufbxw_real)(grid.resolution - 1);

		dst[i].x = (ufbxw_real)x - 0.5f;
		dst[i].y = grid.height_fn(grid.height_user, x, y);
		dst[i].z = (ufbxw_real)y - 0.5f;
	}

	return dst_size;
}

static size_t ufbxwt_grid_poly_stream(void *user, int32_t *dst, size_t dst_size, size_t offset)
{
	ufbxwt_grid_mesh grid = *(ufbxwt_grid_mesh*)user;

	for (size_t i = 0; i < dst_size; i++) {
		uint32_t buffer_ix = (uint32_t)(offset + i);
		uint32_t quad_ix = buffer_ix / 4;
		uint32_t corner_ix = buffer_ix % 4;

		uint32_t x = quad_ix % (grid.resolution - 1);
		uint32_t y = quad_ix / (grid.resolution - 1);

		int32_t ix0 = (int32_t)(y * grid.resolution + x);
		int32_t ix1 = (int32_t)((y + 1) * grid.resolution + x);

		switch (corner_ix) {
		case 0: dst[i] = ix0; break;
		case 1: dst[i] = ix1; break;
		case 2: dst[i] = ix1 + 1; break;
		case 3: dst[i] = ~(ix0 + 1); break;
		}
	}

	return dst_size;
}

static ufbxw_real ufbxwt_grid_wave_height(void *user, ufbxw_real x, ufbxw_real y)
{
	enum { wave_count = 8 };
	static const double freq_x[wave_count] = { 1.0, 0.1, 2.5, 4.3, 5.2, 3.2, 0.8, 7.0 };
	static const double freq_y[wave_count] = { 0.2, 0.7, 1.3, 3.5, 1.3, 7.2, 9.0, 7.0 };
	static const double amp[wave_count] = { 0.8, 0.6, 0.3, 0.15, 0.08, 0.09, 0.03, 0.04 };
	static const double phase[wave_count] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

	double res = 0.0;
	for (int i = 0; i < wave_count; i++) {
		res += sin((freq_x[i] * x + freq_y[i] * y) * 5.0 + phase[i]) * amp[i] * 0.1;
	}
	return (ufbxw_real)res;
}

static void ufbxwt_create_grid_mesh(ufbxw_scene *scene, const ufbxwt_grid_mesh *grid)
{
	ufbxw_node node = ufbxwt_create_node(scene, "Grid");

	ufbxw_mesh mesh = ufbxw_create_mesh(scene);
	ufbxw_set_name(scene, mesh.id, "Grid_Mesh");
	ufbxw_mesh_add_instance(scene, mesh, node);

	uint32_t resolution = grid->resolution;
	uint32_t vertex_count = resolution * resolution;
	uint32_t index_count = (resolution - 1) * (resolution - 1) * 4;

	ufbxw_vec3_buffer vertex_buffer = ufbxw_external_vec3_stream(scene, &ufbxwt_grid_vertex_stream, (void*)grid, vertex_count);
	ufbxw_int_buffer poly_buffer = ufbxw_external_int_stream(scene, &ufbxwt_grid_poly_stream, (void*)grid, index_count);

	ufbxw_mesh_set_vertices(scene, mesh, vertex_buffer);
	ufbxw_mesh_set_fbx_polygon_vertex_index(scene, mesh, poly_buffer);
}

static void ufbxwt_check_grid_mesh(ufbxwt_diff_error *err, ufbx_scene *scene, const ufbxwt_grid_mesh *grid)
{
	ufbx_node *node = ufbx_find_node(scene, "Grid");
	ufbxwt_assert(node);

	ufbx_mesh *mesh = node->mesh;
	ufbxwt_assert(mesh);

	uint32_t resolution = grid->resolution;
	uint32_t vertex_count = resolution * resolution;
	uint32_t index_count = (resolution - 1) * (resolution - 1) * 4;

	ufbxwt_assert(mesh->vertices.count == vertex_count);
	ufbxwt_assert(mesh->vertex_indices.count == index_count);

	for (uint32_t iy = 0; iy < resolution; iy++) {
		for (uint32_t ix = 0; ix < resolution; ix++) {
			ufbx_real x = (ufbx_real)ix / (ufbx_real)(resolution - 1);
			ufbx_real y = (ufbx_real)iy / (ufbx_real)(resolution - 1);

			ufbx_vec3 v = mesh->vertices.data[iy * resolution + ix];
			ufbxwt_assert_close_real(err, v.x, x - 0.5f);
			ufbxwt_assert_close_real(err, v.z, y - 0.5f);

			ufbx_real height = grid->height_fn(grid->height_user, x, y);
			ufbxwt_assert_close_real(err, v.y, height);
		}
	}

	size_t face_ix = 0;
	for (uint32_t iy = 0; iy < resolution - 1; iy++) {
		for (uint32_t ix = 0; ix < resolution - 1; ix++) {
			int32_t ix0 = (int32_t)(iy * resolution + ix);
			int32_t ix1 = (int32_t)((iy + 1) * resolution + ix);

			ufbx_face face = mesh->faces.data[face_ix];
			ufbxwt_assert(face.index_begin == face_ix * 4);
			ufbxwt_assert(face.num_indices == 4);

			const int32_t *indices = mesh->vertex_indices.data + face.index_begin;
			ufbxwt_assert(indices[0] == ix0);
			ufbxwt_assert(indices[1] == ix1);
			ufbxwt_assert(indices[2] == ix1 + 1);
			ufbxwt_assert(indices[3] == ix0 + 1);

			face_ix++;
		}
	}
}
#endif

UFBXWT_SCENE_TEST(mesh_plane)
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
}
#endif

UFBXWT_SCENE_CHECK(mesh_plane)
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

#if UFBXWT_IMPL
static const ufbxwt_grid_mesh ufbxwt_grid_wave_64 = { 64, &ufbxwt_grid_wave_height };
static const ufbxwt_grid_mesh ufbxwt_grid_wave_128 = { 128, &ufbxwt_grid_wave_height };
#endif

UFBXWT_SCENE_TEST(mesh_wave)
#if UFBXWT_IMPL
{
	ufbxwt_create_grid_mesh(scene, &ufbxwt_grid_wave_64);
}
#endif

UFBXWT_SCENE_CHECK(mesh_wave)
#if UFBXWT_IMPL
{
	ufbxwt_check_grid_mesh(err, scene, &ufbxwt_grid_wave_64);
}
#endif


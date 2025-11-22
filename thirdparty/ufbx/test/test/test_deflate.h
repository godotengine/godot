#undef UFBXWT_TEST_GROUP
#define UFBXWT_TEST_GROUP "deflate"

#if UFBXWT_IMPL

typedef struct {
	void *data;
	size_t data_size;
} ufbxwt_memory_stream;

static bool ufbxwt_memory_stream_write(void *user, uint64_t offset, const void *data, size_t size)
{
	ufbxwt_memory_stream *s = (ufbxwt_memory_stream*)user;
	if (offset >= s->data_size || s->data_size - offset < size) {
		return false;
	}
	memcpy((char*)s->data + offset, data, size);
	return true;
}

static ufbxw_scene *ufbxwt_deflate_scene_simple()
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);
	ufbxwt_assert(scene);

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

	return scene;
}

static ufbxw_scene *ufbxwt_deflate_scene_wave()
{
	ufbxw_scene *scene = ufbxw_create_scene(NULL);
	ufbxwt_assert(scene);

	ufbxwt_create_grid_mesh(scene, &ufbxwt_grid_wave_128);

	return scene;
}

static void ufbxwt_deflate_test(const char *name, ufbxw_scene *scene, const ufbxw_save_opts *opts, size_t result_size)
{
	static const uint32_t versions[] = { 7400, 7500 };

	void *result = malloc(result_size);
	ufbxwt_assert(result);

	ufbxwt_memory_stream stream = { result, result_size };

	ufbxw_write_stream ws = { 0 };
	ws.write_fn = &ufbxwt_memory_stream_write;
	ws.user = &stream;

	static const ufbxw_save_format formats[] = { UFBXW_SAVE_FORMAT_ASCII, UFBXW_SAVE_FORMAT_BINARY };

	for (int thread_ix = 0; thread_ix < UFBXWT_THREAD_IMPL_COUNT; thread_ix++) {
		for (int format_ix = 0; format_ix < ufbxwt_arraycount(formats); format_ix++) {
			for (int deflate_ix = 0; deflate_ix < UFBXWT_DEFLATE_IMPL_COUNT; deflate_ix++) {
				for (int version_ix = 0; version_ix < ufbxwt_arraycount(versions); version_ix++) {

					ufbxw_save_opts save_opts = { 0 };
					if (opts) {
						save_opts = *opts;
					}

					save_opts.version = versions[version_ix];
					save_opts.format = formats[format_ix];

					save_opts.threaded_min_deflate_bytes = 1;

					ufbxwt_deflate_impl deflate_impl = (ufbxwt_deflate_impl)deflate_ix;
					ufbxwt_thread_impl thread_impl = (ufbxwt_thread_impl)thread_ix;
					const char *format = save_opts.format == UFBXW_SAVE_FORMAT_ASCII ? "ascii" : "binary";

					if (g_file_version && save_opts.version != g_file_version) continue;
					if (g_file_format && strcmp(format, g_file_format) != 0) continue;

					if (save_opts.format == UFBXW_SAVE_FORMAT_ASCII && deflate_impl != UFBXWT_DEFLATE_IMPL_NONE) {
						continue;
					}

					if (thread_impl != UFBXWT_THREAD_IMPL_NONE && deflate_impl == UFBXWT_DEFLATE_IMPL_NONE) {
						continue;
					}

					if (!ufbxwt_deflate_setup(&save_opts.deflate, deflate_impl)) {
						continue;
					}

					if (!ufbxwt_thread_setup(&save_opts.thread_sync, &save_opts.thread_pool, thread_impl)) {
						continue;
					}

					const char *deflate = ufbxwt_deflate_impl_name(deflate_impl);
					const char *thread = ufbxwt_thread_impl_name(thread_impl);
					ufbxwt_logf("format: %s, version: %u, deflate: %s, thread: %s", format, save_opts.version, deflate, thread);

					memset(result, 0, result_size);

					ufbxw_error save_error;
					bool save_ok = ufbxw_save_stream(scene, &ws, &save_opts, &save_error);
					if (save_error.type != UFBXW_ERROR_NONE) {
						ufbxwt_log_error(&save_error);
					}
					ufbxwt_assert(save_ok);

					ufbx_error load_error;
					ufbx_scene *loaded_scene = ufbx_load_memory(result, result_size, NULL, &load_error);
					if (load_error.type != UFBX_ERROR_NONE) {
						ufbxwt_log_uerror(&load_error);
					}
					ufbxwt_assert(loaded_scene);
					ufbx_free_scene(loaded_scene);
				}
			}
		}
	}

	free(result);
	ufbxw_free_scene(scene);
}

#endif

UFBXWT_TEST(deflate_simple)
#if UFBXWT_IMPL
{
	ufbxwt_deflate_test("simple", ufbxwt_deflate_scene_simple(), NULL, 64 * 1024);
}
#endif

UFBXWT_TEST(deflate_streaming)
#if UFBXWT_IMPL
{
	ufbxw_save_opts opts = { 0 };
	opts.buffer_size = 128;
	opts.deflate_window_size = 128;
	ufbxwt_deflate_test("simple", ufbxwt_deflate_scene_simple(), &opts, 64 * 1024);
}
#endif

UFBXWT_TEST(deflate_streaming_wave)
#if UFBXWT_IMPL
{
	ufbxw_save_opts opts = { 0 };
	opts.buffer_size = 128;
	opts.deflate_window_size = 128;
	ufbxwt_deflate_test("simple", ufbxwt_deflate_scene_wave(), &opts, 2 * 1024 * 1024);
}
#endif

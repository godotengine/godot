extends SceneTree


class SampleProbe extends CompositorEffect:
	var samples := -1

	func _init() -> void:
		effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
		if RenderingServer.get_current_rendering_method() == "forward_plus":
			access_resolved_depth = true
			needs_normal_roughness = true
			needs_separate_specular = true

	func _render_callback(_type: int, data: RenderData) -> void:
		var buffers: RenderSceneBuffersRD = data.get_render_scene_buffers()
		samples = buffers.get_texture_samples()


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	_check(RenderingServer.get_current_rendering_driver_name() == "webgpu", "Rendering driver fell back")
	_check_direct_requests()
	var viewport := SubViewport.new()
	viewport.size = Vector2i(96, 96)
	viewport.own_world_3d = true
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)
	var probes: Array[SampleProbe] = []
	for target in [root, viewport]:
		var box := MeshInstance3D.new()
		box.mesh = BoxMesh.new()
		var material := StandardMaterial3D.new()
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		material.albedo_color = Color.RED
		box.material_override = material
		target.add_child(box)
		var camera := Camera3D.new()
		camera.position = Vector3(0, 0, 3)
		camera.current = true
		target.add_child(camera)
		var probe := SampleProbe.new()
		var compositor := Compositor.new()
		compositor.compositor_effects = [probe]
		camera.compositor = compositor
		probes.append(probe)

	# Switch both directions so fallback never replaces the user's requested setting.
	for requested in [Viewport.MSAA_DISABLED, Viewport.MSAA_2X, Viewport.MSAA_4X, Viewport.MSAA_8X, Viewport.MSAA_2X, Viewport.MSAA_4X]:
		ProjectSettings.set_setting("rendering/anti_aliasing/quality/msaa_3d", requested)
		for probe in probes:
			probe.samples = -1
		viewport.msaa_3d = requested
		root.msaa_3d = requested
		for frame in 6:
			await process_frame
			await RenderingServer.frame_post_draw
		_check(viewport.msaa_3d == requested and root.msaa_3d == requested, "Fallback replaced requested 3D MSAA")
		var expected := RenderingDevice.TEXTURE_SAMPLES_1 if requested < Viewport.MSAA_4X else RenderingDevice.TEXTURE_SAMPLES_4
		for probe in probes:
			_check(probe.samples == expected, "Effective sample count disagrees with the viewport buffers")
		for target in [root, viewport]:
			var image: Image = target.get_texture().get_image()
			var pixel := image.get_pixel(image.get_width() / 2, image.get_height() / 2)
			_check(pixel.r > 0.8, "MSAA lost the red 3D target: requested=%d size=%s pixel=%s" % [requested, image.get_size(), pixel])

	var canvas := SubViewport.new()
	canvas.size = Vector2i(64, 64)
	canvas.disable_3d = true
	canvas.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(canvas)
	var rectangle := ColorRect.new()
	rectangle.size = Vector2(64, 64)
	rectangle.color = Color.BLUE
	canvas.add_child(rectangle)
	for hdr in [false, true]:
		canvas.use_hdr_2d = hdr
		for requested in [Viewport.MSAA_DISABLED, Viewport.MSAA_2X, Viewport.MSAA_4X, Viewport.MSAA_8X, Viewport.MSAA_2X, Viewport.MSAA_4X]:
			ProjectSettings.set_setting("rendering/anti_aliasing/quality/msaa_2d", requested)
			canvas.msaa_2d = requested
			for frame in 6:
				await process_frame
				await RenderingServer.frame_post_draw
			_check(canvas.msaa_2d == requested, "Fallback replaced requested 2D MSAA")
			var image := canvas.get_texture().get_image()
			_check(image.get_pixel(32, 32).b > 0.8, "MSAA lost the blue 2D target")
	canvas.free()
	viewport.free()
	print("NATIVE_WEBGPU_MSAA_PASSED")
	quit()


func _check_direct_requests() -> void:
	var rd := RenderingServer.create_local_rendering_device()
	var format := RDTextureFormat.new()
	format.width = 16
	format.height = 16
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT
	var view := RDTextureView.new()
	for data_format in [RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM, RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT, RenderingDevice.DATA_FORMAT_R32G32_SFLOAT]:
		format.format = data_format
		for samples in [RenderingDevice.TEXTURE_SAMPLES_1, RenderingDevice.TEXTURE_SAMPLES_2, RenderingDevice.TEXTURE_SAMPLES_4, RenderingDevice.TEXTURE_SAMPLES_8]:
			format.samples = samples
			var texture := rd.texture_create(format, view)
			var supported: bool = samples == RenderingDevice.TEXTURE_SAMPLES_1 or (samples == RenderingDevice.TEXTURE_SAMPLES_4 and data_format != RenderingDevice.DATA_FORMAT_R32G32_SFLOAT)
			_check(texture.is_valid() == supported, "Direct texture request violated the format's sample-count support")
			if texture.is_valid():
				rd.free_rid(texture)
	_check_integer_storage(rd)
	_check_pipeline_requests(rd)
	rd.free()


func _check_integer_storage(rd: RenderingDevice) -> void:
	var usage := RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
	var textures: Array[RID] = []
	for data_format in [RenderingDevice.DATA_FORMAT_R8_UINT, RenderingDevice.DATA_FORMAT_R8_SINT, RenderingDevice.DATA_FORMAT_R8G8_UINT, RenderingDevice.DATA_FORMAT_R8G8_SINT]:
		_check(rd.texture_is_format_supported_for_usage(data_format, usage), "Integer storage capability is missing")
		var format := RDTextureFormat.new()
		format.width = 16
		format.height = 16
		format.format = data_format
		format.usage_bits = usage
		var texture := rd.texture_create(format, RDTextureView.new())
		_check(texture.is_valid(), "Integer storage attachment could not be created")
		textures.append(texture)

	var normal_format := RDTextureFormat.new()
	normal_format.width = 16
	normal_format.height = 16
	normal_format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	normal_format.usage_bits = usage
	var normal := rd.texture_create(normal_format, RDTextureView.new())
	var framebuffer := rd.framebuffer_create([normal, textures[2]])
	_check(framebuffer.is_valid(), "VoxelGI framebuffer lost its integer attachment")
	var source := RDShaderSource.new()
	source.source_vertex = "#version 450\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }"
	source.source_fragment = "#version 450\nlayout(location=0) out vec4 normal; layout(location=1) out uvec4 voxel; void main() { normal = vec4(1.0); voxel = uvec4(1, 2, 0, 0); }"
	var shader := rd.shader_create_from_spirv(rd.shader_compile_spirv_from_source(source))
	var blend := RDPipelineColorBlendState.new()
	blend.attachments = [RDPipelineColorBlendStateAttachment.new(), RDPipelineColorBlendStateAttachment.new()]
	var pipeline := rd.render_pipeline_create(shader, rd.framebuffer_get_format(framebuffer), -1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES,
		RDPipelineRasterizationState.new(), RDPipelineMultisampleState.new(), RDPipelineDepthStencilState.new(), blend)
	_check(pipeline.is_valid(), "VoxelGI fragment outputs do not match the framebuffer")
	for resource in [pipeline, shader, framebuffer, normal] + textures:
		rd.free_rid(resource)


func _check_pipeline_requests(rd: RenderingDevice) -> void:
	var source := RDShaderSource.new()
	source.source_vertex = "#version 450\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }"
	source.source_fragment = "#version 450\nlayout(location=0) out vec4 color; void main() { color = vec4(1.0); }"
	var shader := rd.shader_create_from_spirv(rd.shader_compile_spirv_from_source(source))
	_check(shader.is_valid(), "Pipeline rejection probe failed to compile")
	for data_format in [RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM, RenderingDevice.DATA_FORMAT_R32G32_SFLOAT]:
		for samples in [RenderingDevice.TEXTURE_SAMPLES_2, RenderingDevice.TEXTURE_SAMPLES_4, RenderingDevice.TEXTURE_SAMPLES_8]:
			var attachment := RDAttachmentFormat.new()
			attachment.format = data_format
			attachment.samples = samples
			attachment.usage_flags = RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT
			var framebuffer := rd.framebuffer_format_create([attachment])
			var multisample := RDPipelineMultisampleState.new()
			multisample.sample_count = samples
			var blend := RDPipelineColorBlendState.new()
			blend.attachments = [RDPipelineColorBlendStateAttachment.new()]
			var pipeline := rd.render_pipeline_create(shader, framebuffer, -1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES,
				RDPipelineRasterizationState.new(), multisample, RDPipelineDepthStencilState.new(), blend)
			_check(pipeline.is_valid() == (samples == RenderingDevice.TEXTURE_SAMPLES_4 and data_format == RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM),
				"Direct pipeline request violated the format's sample-count support")
			if pipeline.is_valid():
				rd.free_rid(pipeline)
	rd.free_rid(shader)


func _check(condition: bool, message: String) -> void:
	if not condition:
		push_error(message)
		quit(1)

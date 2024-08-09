# meta-description: Base template for CompositorEffect with support for custom inline code insertion

@tool
# Having a class name is handy for picking the effect in the Inspector.
class_name CompositorEffect_CLASS_
extends _BASE_


const SHADER_CODE: String = "#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform image2D color_image;

layout(push_constant, std430) uniform Params {
	vec2 raster_size;
	vec2 reserved;
} params;

void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = ivec2(params.raster_size);

	if (uv.x >= size.x || uv.y >= size.y) {
		return;
	}

	vec4 color = imageLoad(color_image, uv);

	#COMPUTE_CODE

	imageStore(color_image, uv, color);
}"

@export_multiline var shader_code: String = "":
	set(value):
		mutex.lock()
		shader_code = value
		shader_is_dirty = true
		mutex.unlock()

var rd: RenderingDevice
var shader := RID()
var pipeline := RID()

var mutex := Mutex.new()
var shader_is_dirty := true


func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	rd = RenderingServer.get_rendering_device()


func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		if shader.is_valid():
			RenderingServer.free_rid(shader)


func _check_shader() -> bool:
	if not rd:
		return false

	var new_shader_code: String = ""

	mutex.lock()
	if shader_is_dirty:
		new_shader_code = shader_code
		shader_is_dirty = false
	mutex.unlock()

	if new_shader_code.is_empty():
		return pipeline.is_valid()

	new_shader_code = SHADER_CODE.replace("#COMPUTE_CODE", new_shader_code);

	if shader.is_valid():
		rd.free_rid(shader)
		shader = RID()
		pipeline = RID()

	var shader_source := RDShaderSource.new()
	shader_source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	shader_source.source_compute = new_shader_code
	var shader_spirv := rd.shader_compile_spirv_from_source(shader_source)

	if not shader_spirv.compile_error_compute.is_empty():
		push_error(shader_spirv.compile_error_compute)
		return false

	shader = rd.shader_create_from_spirv(shader_spirv)
	if not shader.is_valid():
		return false

	pipeline = rd.compute_pipeline_create(shader)
	return pipeline.is_valid()


func _render_callback(effect_callback_type: int, render_data: RenderData) -> void:
	if rd and effect_callback_type == EFFECT_CALLBACK_TYPE_POST_TRANSPARENT and _check_shader():
		var render_scene_buffers: RenderSceneBuffersRD = render_data.get_render_scene_buffers()
		if render_scene_buffers:
			var size := render_scene_buffers.get_internal_size()
			if size.x == 0 and size.y == 0:
				return

			var x_groups := (size.x - 1) / 8 + 1
			var y_groups := (size.y - 1) / 8 + 1
			var z_groups := 1

			var push_constant := PackedFloat32Array()
			push_constant.push_back(size.x)
			push_constant.push_back(size.y)
			push_constant.push_back(0.0)
			push_constant.push_back(0.0)

			var view_count := render_scene_buffers.get_view_count()
			for view in range(view_count):
				var input_image := render_scene_buffers.get_color_layer(view)

				var uniform := RDUniform.new()
				uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
				uniform.binding = 0
				uniform.add_id(input_image)
				var uniform_set := UniformSetCacheRD.get_cache(shader, 0, [uniform])

				var compute_list := rd.compute_list_begin()
				rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
				rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
				rd.compute_list_set_push_constant(compute_list, push_constant.to_byte_array(),
						push_constant.size() * 4)
				rd.compute_list_dispatch(compute_list, x_groups, y_groups, z_groups)
				rd.compute_list_end()

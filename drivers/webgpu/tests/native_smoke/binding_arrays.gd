# Run with --rendering-driver webgpu --path drivers/webgpu/tests/native_smoke --script res://binding_arrays.gd
extends SceneTree


func _initialize():
	_run.call_deferred()


func _run():
	var rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		push_error("No local rendering device")
		quit(1)
		return
	var source = RDShaderSource.new()
	source.source_compute = """#version 450
layout(local_size_x=1) in;
layout(set=0,binding=0) uniform sampler2D layers[4];
layout(set=0,binding=1,std430) readonly buffer Indices { uint indices[4]; } params;
layout(set=0,binding=2,std430) writeonly buffer Output { vec4 colors[4]; } result;
void main() {
    uint i = gl_GlobalInvocationID.x;
    result.colors[i] = textureLod(layers[params.indices[i]], vec2(0.5), 0.0);
}
"""
	var spirv = rd.shader_compile_spirv_from_source(source)
	if not spirv.compile_error_compute.is_empty():
		push_error(spirv.compile_error_compute)
		quit(1)
		return
	var shader = rd.shader_create_from_spirv(spirv)
	var sampler = rd.sampler_create(RDSamplerState.new())
	var textures: Array[RID] = []
	var pixels = [
		PackedByteArray([255, 0, 0, 255]),
		PackedByteArray([0, 255, 0, 255]),
		PackedByteArray([0, 0, 255, 255]),
		PackedByteArray([255, 255, 0, 255])
	]
	for pixel in pixels:
		var format = RDTextureFormat.new()
		format.width = 1
		format.height = 1
		format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
		format.usage_bits = (
			RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
			| RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
		)
		textures.append(rd.texture_create(format, RDTextureView.new(), [pixel]))
	var indices = rd.storage_buffer_create(16, PackedInt32Array([2, 0, 3, 1]).to_byte_array())
	var output = rd.storage_buffer_create(64)
	var layers = RDUniform.new()
	layers.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	layers.binding = 0
	for texture in textures:
		layers.add_id(sampler)
		layers.add_id(texture)
	var input_uniform = RDUniform.new()
	input_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	input_uniform.binding = 1
	input_uniform.add_id(indices)
	var output_uniform = RDUniform.new()
	output_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	output_uniform.binding = 2
	output_uniform.add_id(output)
	var uniforms = rd.uniform_set_create([layers, input_uniform, output_uniform], shader, 0)
	var pipeline = rd.compute_pipeline_create(shader)
	var commands = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(commands, pipeline)
	rd.compute_list_bind_uniform_set(commands, uniforms, 0)
	rd.compute_list_dispatch(commands, 4, 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	var bytes = rd.buffer_get_data(output)
	var values = bytes.to_float32_array()
	print("WEBGPU_ARRAY_VALUES: ", values)
	var expected = PackedFloat32Array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1])
	if values != expected:
		push_error("WebGPU bound the wrong array texture")
		quit(1)
		return
	for rid in [pipeline, uniforms, shader, sampler, indices, output]:
		rd.free_rid(rid)
	for texture in textures:
		rd.free_rid(texture)
	rd.free()
	print("WEBGPU_DYNAMIC_ARRAY_BINDINGS_PASSED")
	quit()

#version 310 es
#extension GL_EXT_tessellation_shader : require
#extension GL_EXT_gpu_shader5 : require

layout(triangles, equal_spacing) in;

layout(location = 0) in highp vec2 in_te_position[];

layout(location = 0) out mediump vec4 in_f_color;

precise gl_Position;

void main(void) {
  highp vec2 pos = gl_TessCoord.x * in_te_position[0] +
                   gl_TessCoord.y * in_te_position[1] +
                   gl_TessCoord.z * in_te_position[2];

  highp float f =
      sqrt(3.0 * min(gl_TessCoord.x, min(gl_TessCoord.y, gl_TessCoord.z))) *
          0.5 +
      0.5;
  in_f_color = vec4(gl_TessCoord * f, 1.0);

  // Offset the position slightly, based on the parity of the bits in the float
  // representation.
  // This is done to detect possible small differences in edge vertex positions
  // between patches.
  uvec2 bits = floatBitsToUint(pos);
  uint numBits = 0u;
  for (uint i = 0u; i < 32u; i++)
    numBits +=
        ((bits[0] << i) & 1u) + ((bits[1] << i) & 1u);
  pos += float(numBits & 1u) * 0.04;

  gl_Position = vec4(pos, 0.0, 1.0);
}

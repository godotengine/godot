void mx_transformmatrix_vector2M3(vec2 val, mat3 transform, out vec2 result)
{
  vec3 res = transform * vec3(val, 1.0);
  result = res.xy;
}

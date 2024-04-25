void mx_creatematrix_vector3_matrix33(vec3 in1, vec3 in2, vec3 in3, out mat3 result)
{
    result = mat3(in1.x, in1.y, in1.z,
                  in2.x, in2.y, in2.z,
                  in3.x, in3.y, in3.z);
}

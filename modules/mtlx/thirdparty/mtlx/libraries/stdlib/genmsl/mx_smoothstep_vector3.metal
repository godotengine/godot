#include "mx_smoothstep_float.metal"

void mx_smoothstep_vector3(vec3 val, vec3 low, vec3 high, thread vec3& result)
    {
        float f;
        mx_smoothstep_float(val.x, low.x, high.x, f); result.x = f;
        mx_smoothstep_float(val.y, low.y, high.y, f); result.y = f;
        mx_smoothstep_float(val.z, low.z, high.z, f); result.z = f;
    }
// Restrict to 7x7 kernel size for performance reasons
#define MX_MAX_SAMPLE_COUNT 49
// Size of all weights for all levels (including level 1)
#define MX_WEIGHT_ARRAY_SIZE 84

//
// Function to compute the sample size relative to a texture coordinate
//
vec2 mx_compute_sample_size_uv(vec2 uv, float filterSize, float filterOffset)
{
   vec2 derivUVx = dFdx(uv) * 0.5f;
   vec2 derivUVy = dFdy(uv) * 0.5f;
   float derivX = abs(derivUVx.x) + abs(derivUVy.x);
   float derivY = abs(derivUVx.y) + abs(derivUVy.y);
   float sampleSizeU = 2.0f * filterSize * derivX + filterOffset;
   if (sampleSizeU < 1.0E-05f)
       sampleSizeU = 1.0E-05f;
   float sampleSizeV = 2.0f * filterSize * derivY + filterOffset;
   if (sampleSizeV < 1.0E-05f)
       sampleSizeV = 1.0E-05f;
   return vec2(sampleSizeU, sampleSizeV);
}

//
// Compute a normal mapped to 0..1 space based on a set of input
// samples using a Sobel filter.
//
vec3 mx_normal_from_samples_sobel(float S[9], float _scale)
{
    float nx = S[0] - S[2] + (2.0*S[3]) - (2.0*S[5]) + S[6] - S[8];
    float ny = S[0] + (2.0*S[1]) + S[2] - S[6] - (2.0*S[7]) - S[8];
    float nz = max(_scale, M_FLOAT_EPS) * sqrt(max(1.0 - nx * nx - ny * ny, M_FLOAT_EPS));
    vec3 norm = normalize(vec3(nx, ny, nz));
    return (norm + 1.0) * 0.5;
}

//
// Apply filter for float samples S, using weights W.
// sampleCount should be a square of a odd number in the range { 1, 3, 5, 7 }
//
float mx_convolution_float(float S[MX_MAX_SAMPLE_COUNT], float W[MX_WEIGHT_ARRAY_SIZE], int offset, int sampleCount)
{
    float result = 0.0;
    for (int i = 0;  i < sampleCount; i++)
    {
        result += S[i]*W[i+offset];
    }
    return result;
}

//
// Apply filter for vec2 samples S, using weights W.
// sampleCount should be a square of a odd number in the range { 1, 3, 5, 7 }
//
vec2 mx_convolution_vec2(vec2 S[MX_MAX_SAMPLE_COUNT], float W[MX_WEIGHT_ARRAY_SIZE], int offset, int sampleCount)
{
    vec2 result = vec2(0.0);
    for (int i=0;  i<sampleCount; i++)
    {
        result += S[i]*W[i+offset];
    }
    return result;
}

//
// Apply filter for vec3 samples S, using weights W.
// sampleCount should be a square of a odd number in the range { 1, 3, 5, 7 }
//
vec3 mx_convolution_vec3(vec3 S[MX_MAX_SAMPLE_COUNT], float W[MX_WEIGHT_ARRAY_SIZE], int offset, int sampleCount)
{
    vec3 result = vec3(0.0);
    for (int i=0;  i<sampleCount; i++)
    {
        result += S[i]*W[i+offset];
    }
    return result;
}

//
// Apply filter for vec4 samples S, using weights W.
// sampleCount should be a square of a odd number { 1, 3, 5, 7 }
//
vec4 mx_convolution_vec4(vec4 S[MX_MAX_SAMPLE_COUNT], float W[MX_WEIGHT_ARRAY_SIZE], int offset, int sampleCount)
{
    vec4 result = vec4(0.0);
    for (int i=0;  i<sampleCount; i++)
    {
        result += S[i]*W[i+offset];
    }
    return result;
}

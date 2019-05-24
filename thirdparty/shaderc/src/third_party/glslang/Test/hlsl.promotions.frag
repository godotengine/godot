
struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

uniform int3    i3;
uniform bool3   b3;
uniform float3  f3;
uniform uint3   u3;
uniform double3 d3;

uniform int    is;
uniform bool   bs;
uniform float  fs;
uniform uint   us;
uniform double ds;

void Fn_F3(float3 x) { }
void Fn_I3(int3 x) { }
void Fn_U3(uint3 x) { }
void Fn_B3(bool3 x) { }
void Fn_D3(double3 x) { }

// ----------- Test implicit conversions on function returns -----------
float3  Fn_R_F3I(out float3 p) { p = i3; return i3; }
float3  Fn_R_F3U(out float3 p) { p = u3; return u3; }
float3  Fn_R_F3B(out float3 p) { p = b3; return b3; }
float3  Fn_R_F3D(out float3 p) { p = d3; return d3; }  // valid, but loss of precision on downconversion.

int3    Fn_R_I3U(out int3 p) { p = u3; return u3; }
int3    Fn_R_I3B(out int3 p) { p = b3; return b3; }
int3    Fn_R_I3F(out int3 p) { p = f3; return f3; }
int3    Fn_R_I3D(out int3 p) { p = d3; return d3; }  // valid, but loss of precision on downconversion.

uint3   Fn_R_U3I(out uint3 p) { p = i3; return i3; }
uint3   Fn_R_U3F(out uint3 p) { p = f3; return f3; }
uint3   Fn_R_U3B(out uint3 p) { p = b3; return b3; }
uint3   Fn_R_U3D(out uint3 p) { p = d3; return d3; }  // valid, but loss of precision on downconversion.

bool3   Fn_R_B3I(out bool3 p) { p = i3; return i3; }
bool3   Fn_R_B3U(out bool3 p) { p = u3; return u3; }
bool3   Fn_R_B3F(out bool3 p) { p = f3; return f3; }
bool3   Fn_R_B3D(out bool3 p) { p = d3; return d3; }

double3 Fn_R_D3I(out double3 p) { p = i3; return i3; }
double3 Fn_R_D3U(out double3 p) { p = u3; return u3; }
double3 Fn_R_D3B(out double3 p) { p = b3; return b3; }
double3 Fn_R_D3F(out double3 p) { p = f3; return f3; }

PS_OUTPUT main()
{
    // ----------- assignment conversions -----------
    float3 r00 = i3;
    float3 r01 = b3;
    float3 r02 = u3;
    float3 r03 = d3;  // valid, but loss of precision on downconversion.

    int3   r10 = b3;
    int3   r11 = u3;
    int3   r12 = f3;
    int3   r13 = d3;  // valid, but loss of precision on downconversion.

    uint3  r20 = b3;
    uint3  r21 = i3;
    uint3  r22 = f3;
    uint3  r23 = d3;  // valid, but loss of precision on downconversion.

    bool3  r30 = i3;
    bool3  r31 = u3;
    bool3  r32 = f3;
    bool3  r33 = d3;

    double3 r40 = i3;
    double3 r41 = u3;
    double3 r42 = f3;
    double3 r43 = b3;

    // ----------- assign ops: vector times vector ----------- 
    r00 *= i3;
    r01 *= b3;
    r02 *= u3;
    r03 *= d3;  // valid, but loss of precision on downconversion.
    
    r10 *= b3;
    r11 *= u3;
    r12 *= f3;
    r13 *= d3;  // valid, but loss of precision on downconversion.
    
    r20 *= b3;
    r21 *= i3;
    r22 *= f3;
    r23 *= d3;  // valid, but loss of precision on downconversion.

    // No mul operator for bools
    
    r40 *= i3;
    r41 *= u3;
    r42 *= f3;
    r43 *= b3;

    // ----------- assign ops: vector times scalar ----------- 
    r00 *= is;
    r01 *= bs;
    r02 *= us;
    r03 *= ds;  // valid, but loss of precision on downconversion.
    
    r10 *= bs;
    r11 *= us;
    r12 *= fs;
    r13 *= ds;  // valid, but loss of precision on downconversion.
    
    r20 *= bs;
    r21 *= is;
    r22 *= fs;
    r23 *= ds;  // valid, but loss of precision on downconversion.

    // No mul operator for bools
    
    r40 *= is;
    r41 *= us;
    r42 *= fs;
    r43 *= bs;


#define FN_OVERLOADS 0 // change to 1 when overloads under promotions are in place

#if FN_OVERLOADS
    Fn_F3(i3);
    Fn_F3(u3);
    Fn_F3(f3);
    Fn_F3(b3);
    Fn_F3(d3);  // valid, but loss of precision on downconversion.

    Fn_I3(i3);
    Fn_I3(u3);
    Fn_I3(f3);
    Fn_I3(b3);
    Fn_I3(d3);  // valid, but loss of precision on downconversion.

    Fn_U3(i3);
    Fn_U3(u3);
    Fn_U3(f3);
    Fn_U3(b3);
    Fn_U3(d3);  // valid, but loss of precision on downconversion.

    Fn_B3(i3);
    Fn_B3(u3);
    Fn_B3(f3);
    Fn_B3(b3);
    Fn_B3(d3);

    Fn_D3(i3);
    Fn_D3(u3);
    Fn_D3(f3);
    Fn_D3(b3);
    Fn_D3(d3);

    Fn_F3(i3.x);
    Fn_F3(u3.x);
    Fn_F3(f3.x);
    Fn_F3(b3.x);
    Fn_F3(d3.x);  // valid, but loss of precision on downconversion.

    Fn_I3(i3.x);
    Fn_I3(u3.x);
    Fn_I3(f3.x);
    Fn_I3(b3.x);
    Fn_I3(d3.x);  // valid, but loss of precision on downconversion.

    Fn_U3(i3.x);
    Fn_U3(u3.x);
    Fn_U3(f3.x);
    Fn_U3(b3.x);
    Fn_U3(d3.x);  // valid, but loss of precision on downconversion.

    Fn_B3(i3.x);
    Fn_B3(u3.x);
    Fn_B3(f3.x);
    Fn_B3(b3.x);
    Fn_B3(d3.x);

    Fn_D3(i3.x);
    Fn_D3(u3.x);
    Fn_D3(f3.x);
    Fn_D3(b3.x);
    Fn_D3(d3.x);
#endif

    const int   si = 3;
    const float sf = 1.2;

    int   c1 = si * sf;  // 3.6 (not 3!)
    int   c2 = sf * si;  // 3.6 (not 3!)

    float4 outval = float4(si * sf, sf*si, c1, c2);

    PS_OUTPUT psout;
    psout.Color = outval;
    return psout;
}

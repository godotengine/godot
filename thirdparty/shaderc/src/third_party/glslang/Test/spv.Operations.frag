#version 450

flat in ivec4 uiv4;
in vec4 uv4;
bool ub;
bvec4 ub41, ub42;
in float uf;
flat in int ui;
flat in uvec4 uuv4;
flat in uint uui;

out vec4 FragColor;

void main()
{
    vec4 v;
	float f;
	bool b;
	bvec4 bv4;
	int i;
	uint u;

	// floating point
    v = radians(uv4);
    v += degrees(v);
    v += (i = ui*ui, sin(v));
    v += cos(v);
    v += tan(v);
    v += asin(v);
    v += acos(v);

    v += atan(v);
    v += sinh(v);
    v += cosh(v);
    v += tanh(v);
    v += asinh(v);
    v += acosh(v);
    v += atanh(v);

    v += pow(v, v);
    v += exp(v);
    v += log(v);
    v += exp2(v);
    v += log2(v);
    v += sqrt(v);
    v += inversesqrt(v);
    v += abs(v);
    v += sign(v);
    v += floor(v);

    v += trunc(v);
    v += round(v);
    v += roundEven(v);

    v += ceil(v);
    v += fract(v);
    v += mod(v, v);
	v += mod(v, v.x);

    v += modf(v, v);

    v += min(v, uv4);
    v += max(v, uv4);
    v += clamp(v, uv4, uv4);
    v += mix(v,v,v);

    v += mix(v,v,ub41);
    v += mix(v,v,f);
//spv    v += intBitsToFloat(ui);
//    v += uintBitsToFloat(uui);
//    i += floatBitsToInt(f);
//    u += floatBitsToUint(f);
    v += fma(v, uv4, v);

    v += step(v,v);
    v += smoothstep(v,v,v);
    v += step(uf,v);
    v += smoothstep(uf,uf,v);
    v += normalize(v);
    v += faceforward(v, v, v);
    v += reflect(v, v);
    v += refract(v, v, uf);
    v += dFdx(v);
    v += dFdy(v);
    v += fwidth(v);

	// signed integer
	i += abs(ui);
	i += sign(i);
	i += min(i, ui);
	i += max(i, ui);
	i += clamp(i, ui, ui);

	// unsigned integer
    u += min(u, uui);
    u += max(u, uui);
    u += clamp(u, uui, uui);

	//// bool
	b = isnan(uf);
    b = isinf(f);
	b = any(lessThan(v, uv4));
	b = (b && any(lessThanEqual(v, uv4)));
    b = (b && any(greaterThan(v, uv4)));
    b = (b && any(greaterThanEqual(v, uv4)));
    b = (b && any(equal(ub41, ub42)));
    b = (b && any(notEqual(ub41, ub42)));
    b = (b && any(ub41));
    b = (b && all(ub41));
    b = (b && any(not(ub41)));
	
	i = ((i + ui) * i - ui) / i;
	i = i % ui;
	if (i == ui || i != ui && i == ui ^^ i != 2)
	    ++i;
	
	f = ((uf + uf) * uf - uf) / uf;

	f += length(v);
    f += distance(v, v);
    f += dot(v, v);
    f += dot(f, uf);
	f += cross(v.xyz, v.xyz).x;

	if (f == uf || f != uf && f != 2.0)
	    ++f;

    i &= ui;
    i |= 0x42;
    i ^= ui;
    i %= 17;
    i >>= 2;
    i <<= ui;
    i = ~i;
    b = !b;

    FragColor = b ? vec4(i) + vec4(f) + v : v;

    mat4 m1 = mat4(1.0), m2 = mat4(0.0);
    FragColor += (b ? m1 : m2)[1];
}

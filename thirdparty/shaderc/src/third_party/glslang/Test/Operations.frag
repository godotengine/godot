#version 130

uniform ivec4 uiv4;
uniform vec4 uv4;
uniform bool ub;
uniform bvec4 ub41, ub42;
uniform float uf;
uniform int ui;


uniform uvec4 uuv4;
uniform uint uui;


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


    v += mix(v,v,bv4);
    v += intBitsToFloat(ivec4(i));
    v += uintBitsToFloat(uv4);
    v += fma(v,v,v);
    v += frexp(v);
    v += ldexp(v);
    v += unpackUnorm2x16(v);
    v += unpackUnorm4x8(v);
    v += unpackSnorm4x8(v);


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
	//noise*(v);


	// signed integer
	i += abs(ui);
	i += sign(i);
	i += min(i, ui);
	i += max(i, ui);
	i += clamp(i, ui, ui);

	floatsBitsToInt(v);
	packUnorm2x16(v);
	packUnorm4x8(v);
	packSnorm4x8(v);

	// unsigned integer
    u = abs(uui);
    u += sign(u);
    u += min(u, uui);
    u += max(u, uui);
    u += clamp(u, uui, uui);
    u += floatsBitToInt(v);
    u += packUnorm2x16(v);
    u += packUnorm4x8(v);
    i += uui & i;          // ERRORs, no int/uint conversions before 400
    i += uui ^ i;
    i += i | uui;

	// bool

	b = isnan(uf);
    b = isinf(v.y);

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

    gl_FragColor = b ? vec4(i) + vec4(f) + v : v;
}

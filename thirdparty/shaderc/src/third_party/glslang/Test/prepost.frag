#version 140

void main()
{
    struct s {
        float y[5];
    } str;

    float t;
    int index = 5;  // all indexing is 4

    str.y[4] = 2.0;             // 2.0
    t = ++str.y[--index];       // 3.0
    str.y[4] += t;              // 6.0
    t = str.y[4]--;             // 5.0 (t = 6.0)
    str.y[index++] += t;        // 11.0
    --str.y[--index];           // 10.0

    float x = str.y[4];
	++x;
	--x;
	x++;
	x--;

	// x is 10.0

	float y = x * ++x;  // 10 * 11
	float z = y * x--;  // 110 * 11

    // x is 10.0
    // z is 1210.0

    vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
    v.y = v.z--;  // (1,3,2,4)
    v.x = --v.w;  // (3,3,2,3)

    gl_FragColor = z * v;// (3630.0, 3630.0, 2420.0, 3630.0)
}

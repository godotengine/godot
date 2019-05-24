#version 310 es

lowp float foo();

in lowp float low, high;

lowp float face1 = 11.0;

out lowp vec4 Color;

void main()
{
    int z = 3;

    if (2.0 * low + 1.0 < high)
        ++z;

    Color = face1 * vec4(z) + foo();
}

lowp float face2 = -2.0;

lowp float foo()
{
    // testing if face2 initializer insert logic is correct in main
    return face2;
}

#version 450

out vec4  outf4;
in flat ivec4  ini4;
in float  inf;

float Test1(int bound)
{
    float r = 0;
    for (int x=0; x<bound; ++x)
        r += 0.5;
    r += 0.2;
    return r;
}

float Test2(int bound)
{
    if (bound > 2) {
        return Test1(bound * 2);
    } else
        return float(bound * 4 +
                     ini4.y * ini4.z +
                     ini4.x);
}

void main()
{
    outf4 = vec4(Test1(int(inf)) + 
                 Test2(int(inf)));
}

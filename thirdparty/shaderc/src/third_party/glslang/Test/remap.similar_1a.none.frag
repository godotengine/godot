#version 450

in float  inf;
in flat ivec4  ini4;
out vec4  outf4;

float Test1(int bound)
{
    float r = 0;
    for (int x=0; x<bound; ++x)
        r += 0.5;
    return r;
}

float Test2(int bound)
{
    if (bound > 2)
        return Test1(bound);
    else
        return float(bound * 2 +
                     ini4.y * ini4.z +
                     ini4.x);
}

void main()
{
    outf4 = vec4(Test1(int(inf)) + 
                 Test2(int(inf)));
}

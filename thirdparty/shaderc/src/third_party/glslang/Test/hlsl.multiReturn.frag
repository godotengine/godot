struct S {
    float f;
    float3 v;
    float3x3 m;
};

cbuffer bufName {
    S s;
};

S foo()
{
    return s;
}

void main()
{
    foo();
}

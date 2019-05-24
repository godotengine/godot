struct S {
    float clip0 : SV_ClipDistance0;
    float clip1 : SV_ClipDistance1;
    float cull0 : SV_CullDistance0;
    float cull1 : SV_CullDistance1;
    int ii      : SV_InstanceID;
};

S main(S ins)
{
    S s;
    return s;
}

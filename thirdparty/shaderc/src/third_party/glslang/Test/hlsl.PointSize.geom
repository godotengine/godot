struct S {
    [[vk::builtin("PointSize")]] float ps : PSIZE;
};

[maxvertexcount(4)]
void main([[vk::builtin("PointSize")]] triangle in uint ps[3],
       inout LineStream<S> OutputStream)
{
    S s;
    OutputStream.Append(s);
}

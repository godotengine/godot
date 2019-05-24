struct N1 {
    int a;
    float b;
};

struct N2 {
    N1 s1;
    N1 s2;
};

struct N3 {
    N2 t1;
    N1 t2;
    N2 t3;
};

typedef N3 T3;

T3 foo;

float main()
{
    return foo.t3.s2.b;
}

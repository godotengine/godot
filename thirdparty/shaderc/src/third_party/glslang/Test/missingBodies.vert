#version 450

void bar();
void foo() { bar(); }

void B();
void C(int);
void C(int, int) { }
void C(bool);
void A() { B(); C(1); C(true); C(1, 2); }

void main()
{
    foo();
    C(true);
}

int ret1();

int f1 = ret1();

int ret2() { return 3; }

int f2 = ret2();

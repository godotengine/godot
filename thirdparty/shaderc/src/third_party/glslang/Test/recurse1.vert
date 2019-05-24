#version 330 core

void main() {}

float bar(int);

// direct recursion

void self()
{
    self();
}

// two-level recursion

void foo(float)
{
	bar(2);
}

float bar(int)
{
	foo(4.2);

	return 3.2;
}

// four-level, out of order

void B();
void D();
void A() { B(); }
void C() { D(); }
void B() { C(); }
void D() { A(); }

// high degree

void BT();
void DT();
void AT() { BT(); BT(); BT(); }
void CT() { DT(); AT(); DT(); BT(); }
void BT() { CT(); CT(); CT(); }
void DT() { AT(); }

#version 330 core

// cross-unit recursion

void main() {}

// two-level recursion

float cbar(int);

void cfoo(float)
{
	cbar(2);
}

// four-level, out of order

void CB();
void CD();
void CA() { CB(); }
void CC() { CD(); }

// high degree

void CBT();
void CDT();
void CAT() { CBT(); CBT(); CBT(); }
void CCT() { CDT(); CDT(); CBT(); }

// not recursive

void norA() {}
void norB() { norA(); }
void norC() { norA(); }
void norD() { norA(); }
void norE() { norB(); }
void norF() { norB(); }
void norG() { norE(); }
void norH() { norE(); }
void norI() { norE(); }

// not recursive, but with a call leading into a cycle if ignoring direction

void norcA() { }
void norcB() { norcA(); }
void norcC() { norcB(); }
void norcD() { norcC(); norcB(); } // head of cycle
void norcE() { norcD(); } // lead into cycle

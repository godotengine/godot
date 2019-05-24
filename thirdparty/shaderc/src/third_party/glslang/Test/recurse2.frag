#version 330 core

// cross-unit recursion

// two-level recursion

void cfoo(float);

float cbar(int)
{
	cfoo(4.2);

	return 3.2;
}

// four-level, out of order

void CA();
void CC();
void CB() { CC(); }
void CD() { CA(); }

// high degree

void CAT();
void CCT();
void CBT() { CCT(); CCT(); CCT(); }
void CDT() { CAT(); }

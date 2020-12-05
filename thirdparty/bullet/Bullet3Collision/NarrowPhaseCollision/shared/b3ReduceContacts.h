#ifndef B3_REDUCE_CONTACTS_H
#define B3_REDUCE_CONTACTS_H

inline int b3ReduceContacts(const b3Float4* p, int nPoints, const b3Float4& nearNormal, b3Int4* contactIdx)
{
	if (nPoints == 0)
		return 0;

	if (nPoints <= 4)
		return nPoints;

	if (nPoints > 64)
		nPoints = 64;

	b3Float4 center = b3MakeFloat4(0, 0, 0, 0);
	{
		for (int i = 0; i < nPoints; i++)
			center += p[i];
		center /= (float)nPoints;
	}

	//	sample 4 directions

	b3Float4 aVector = p[0] - center;
	b3Float4 u = b3Cross3(nearNormal, aVector);
	b3Float4 v = b3Cross3(nearNormal, u);
	u = b3FastNormalized3(u);
	v = b3FastNormalized3(v);

	//keep point with deepest penetration
	float minW = FLT_MAX;

	int minIndex = -1;

	b3Float4 maxDots;
	maxDots.x = FLT_MIN;
	maxDots.y = FLT_MIN;
	maxDots.z = FLT_MIN;
	maxDots.w = FLT_MIN;

	//	idx, distance
	for (int ie = 0; ie < nPoints; ie++)
	{
		if (p[ie].w < minW)
		{
			minW = p[ie].w;
			minIndex = ie;
		}
		float f;
		b3Float4 r = p[ie] - center;
		f = b3Dot3F4(u, r);
		if (f < maxDots.x)
		{
			maxDots.x = f;
			contactIdx[0].x = ie;
		}

		f = b3Dot3F4(-u, r);
		if (f < maxDots.y)
		{
			maxDots.y = f;
			contactIdx[0].y = ie;
		}

		f = b3Dot3F4(v, r);
		if (f < maxDots.z)
		{
			maxDots.z = f;
			contactIdx[0].z = ie;
		}

		f = b3Dot3F4(-v, r);
		if (f < maxDots.w)
		{
			maxDots.w = f;
			contactIdx[0].w = ie;
		}
	}

	if (contactIdx[0].x != minIndex && contactIdx[0].y != minIndex && contactIdx[0].z != minIndex && contactIdx[0].w != minIndex)
	{
		//replace the first contact with minimum (todo: replace contact with least penetration)
		contactIdx[0].x = minIndex;
	}

	return 4;
}

#endif  //B3_REDUCE_CONTACTS_H

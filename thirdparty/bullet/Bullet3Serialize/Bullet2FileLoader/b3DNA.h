/*
bParse
Copyright (c) 2006-2009 Charlie C & Erwin Coumans  http://gamekit.googlecode.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef __BDNA_H__
#define __BDNA_H__

#include "b3Common.h"

namespace bParse
{
struct bNameInfo
{
	char *m_name;
	bool m_isPointer;
	int m_dim0;
	int m_dim1;
};

class bDNA
{
public:
	bDNA();
	~bDNA();

	void init(char *data, int len, bool swap = false);

	int getArraySize(char *str);
	int getArraySizeNew(short name)
	{
		const bNameInfo &nameInfo = m_Names[name];
		return nameInfo.m_dim0 * nameInfo.m_dim1;
	}
	int getElementSize(short type, short name)
	{
		const bNameInfo &nameInfo = m_Names[name];
		int size = nameInfo.m_isPointer ? mPtrLen * nameInfo.m_dim0 * nameInfo.m_dim1 : mTlens[type] * nameInfo.m_dim0 * nameInfo.m_dim1;
		return size;
	}

	int getNumNames() const
	{
		return m_Names.size();
	}

	char *getName(int ind);
	char *getType(int ind);
	short *getStruct(int ind);
	short getLength(int ind);
	int getReverseType(short type);
	int getReverseType(const char *type);

	int getNumStructs();

	//
	bool lessThan(bDNA *other);

	void initCmpFlags(bDNA *memDNA);
	bool flagNotEqual(int dna_nr);
	bool flagEqual(int dna_nr);
	bool flagNone(int dna_nr);

	int getPointerSize();

	void dumpTypeDefinitions();

private:
	enum FileDNAFlags
	{
		FDF_NONE = 0,
		FDF_STRUCT_NEQU,
		FDF_STRUCT_EQU
	};

	void initRecurseCmpFlags(int i);

	b3AlignedObjectArray<int> mCMPFlags;

	b3AlignedObjectArray<bNameInfo> m_Names;
	b3AlignedObjectArray<char *> mTypes;
	b3AlignedObjectArray<short *> mStructs;
	b3AlignedObjectArray<short> mTlens;
	b3HashMap<b3HashInt, int> mStructReverse;
	b3HashMap<b3HashString, int> mTypeLookup;

	int mPtrLen;
};
}  // namespace bParse

#endif  //__BDNA_H__

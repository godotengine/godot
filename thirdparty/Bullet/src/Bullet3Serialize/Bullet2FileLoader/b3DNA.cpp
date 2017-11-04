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
#include <assert.h>

#include "b3DNA.h"
#include "b3Chunk.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

//this define will force traversal of structures, to check backward (and forward) compatibility
//#define TEST_BACKWARD_FORWARD_COMPATIBILITY


using namespace bParse;


// ----------------------------------------------------- //
bDNA::bDNA()
	:	mPtrLen(0)
{
	// --
}

// ----------------------------------------------------- //
bDNA::~bDNA()
{
	// --
}

// ----------------------------------------------------- //
bool bDNA::lessThan(bDNA *file)
{
	return ( m_Names.size() < file->m_Names.size());
}

// ----------------------------------------------------- //
char *bDNA::getName(int ind)
{
	assert(ind <= (int)m_Names.size());
	return m_Names[ind].m_name;
}


// ----------------------------------------------------- //
char *bDNA::getType(int ind)
{
	assert(ind<=  (int)mTypes.size());
	return mTypes[ind];
}


// ----------------------------------------------------- //
short *bDNA::getStruct(int ind)
{
	assert(ind <=  (int)mStructs.size());
	return mStructs[ind];
}


// ----------------------------------------------------- //
short bDNA::getLength(int ind)
{
	assert(ind <=  (int)mTlens.size());
	return mTlens[ind];
}


// ----------------------------------------------------- //
int bDNA::getReverseType(short type)
{

	int* intPtr = mStructReverse.find(type);
	if (intPtr)
		return *intPtr;

	return -1;
}

// ----------------------------------------------------- //
int bDNA::getReverseType(const char *type)
{

	b3HashString key(type);
	int* valuePtr = mTypeLookup.find(key);
	if (valuePtr)
		return *valuePtr;
	
	return -1;
}

// ----------------------------------------------------- //
int bDNA::getNumStructs()
{
	return (int)mStructs.size();
}

// ----------------------------------------------------- //
bool bDNA::flagNotEqual(int dna_nr)
{
	assert(dna_nr <=	(int)mCMPFlags.size());
	return mCMPFlags[dna_nr] == FDF_STRUCT_NEQU;
}

// ----------------------------------------------------- //
bool bDNA::flagEqual(int dna_nr)
{
	assert(dna_nr <=	(int)mCMPFlags.size());
	int flag = mCMPFlags[dna_nr];
	return  flag == FDF_STRUCT_EQU;
}

// ----------------------------------------------------- //
bool bDNA::flagNone(int dna_nr)
{
	assert(dna_nr <=	(int)mCMPFlags.size());
	return mCMPFlags[dna_nr] == FDF_NONE;
}

// ----------------------------------------------------- //
int bDNA::getPointerSize()
{
	return mPtrLen;
}

// ----------------------------------------------------- //
void bDNA::initRecurseCmpFlags(int iter)
{
	// iter is FDF_STRUCT_NEQU

	short *oldStrc = mStructs[iter];
	short type = oldStrc[0];

	for (int i=0; i<(int)mStructs.size(); i++)
	{
		if (i != iter && mCMPFlags[i] == FDF_STRUCT_EQU )
		{
			short *curStruct = mStructs[i];
			int eleLen = curStruct[1];
			curStruct+=2;

			for (int j=0; j<eleLen; j++, curStruct+=2)
			{
				if (curStruct[0] == type)
				{
					//char *name = m_Names[curStruct[1]].m_name;
					//if (name[0] != '*')
					if (m_Names[curStruct[1]].m_isPointer)
					{
						mCMPFlags[i] = FDF_STRUCT_NEQU;
						initRecurseCmpFlags(i);
					}
				}
			}
		}
	}
}

// ----------------------------------------------------- //
void bDNA::initCmpFlags(bDNA *memDNA)
{

    // compare the file to memory
	// this ptr should be the file data


	assert(!(m_Names.size() == 0));// && "SDNA empty!");
	mCMPFlags.resize(mStructs.size(), FDF_NONE);



	int i;
	for ( i=0; i<(int)mStructs.size(); i++)
	{
		short *oldStruct = mStructs[i];

		int oldLookup = getReverseType(oldStruct[0]);
		if (oldLookup == -1)
		{
			mCMPFlags[i] = FDF_NONE;
			continue;
		}
		//char* typeName = mTypes[oldStruct[0]];

//#define SLOW_FORWARD_COMPATIBLE 1
#ifdef SLOW_FORWARD_COMPATIBLE
		char* typeName = mTypes[oldLookup];
		int newLookup = memDNA->getReverseType(typeName);
		if (newLookup == -1)
		{
			mCMPFlags[i] = FDF_NONE;
			continue;
		}
		short *curStruct = memDNA->mStructs[newLookup];
#else
		// memory for file

		if (oldLookup < memDNA->mStructs.size())
		{
			short *curStruct = memDNA->mStructs[oldLookup];
#endif	
	
		

			// rebuild...
			mCMPFlags[i] = FDF_STRUCT_NEQU;

#ifndef TEST_BACKWARD_FORWARD_COMPATIBILITY

			if (curStruct[1] == oldStruct[1])
			{
				// type len same ...
				if (mTlens[oldStruct[0]] == memDNA->mTlens[curStruct[0]])
				{
					bool isSame = true;
					int elementLength = oldStruct[1];


					curStruct+=2;
					oldStruct+=2;


					for (int j=0; j<elementLength; j++, curStruct+=2, oldStruct+=2)
					{
						// type the same
						//const char* typeFileDNA = mTypes[oldStruct[0]];
						//const char* typeMemDNA = mTypes[curStruct[0]];
						if (strcmp(mTypes[oldStruct[0]], memDNA->mTypes[curStruct[0]])!=0)
						{
							isSame=false;
							break;
						}

						// name the same
						if (strcmp(m_Names[oldStruct[1]].m_name, memDNA->m_Names[curStruct[1]].m_name)!=0)
						{
							isSame=false;
							break;
						}
					}
					// flag valid ==
					if (isSame)
						mCMPFlags[i] = FDF_STRUCT_EQU;
				}
			}
#endif
		}
	}





	// recurse in
	for ( i=0; i<(int)mStructs.size(); i++)
	{
		if (mCMPFlags[i] == FDF_STRUCT_NEQU)
			initRecurseCmpFlags(i);
	}
}




static int name_is_array(char* name, int* dim1, int* dim2) {
	int len = strlen(name);
	/*fprintf(stderr,"[%s]",name);*/
	/*if (len >= 1) {
	if (name[len-1] != ']')
	return 1;
	}
	return 0;*/
	char *bp;
	int num;
	if (dim1) {
		*dim1 = 1;
	}
	if (dim2) {
		*dim2 = 1;
	}
	bp = strchr(name, '[');
	if (!bp) {
		return 0;
	}
	num = 0;
	while (++bp < name+len-1) {
		const char c = *bp;
		if (c == ']') {
			break;
		}
		if (c <= '9' && c >= '0') {
			num *= 10;
			num += (c - '0');
		} else {
			printf("array parse error.\n");
			return 0;
		}
	}
	if (dim2) {
		*dim2 = num;
	}

	/* find second dim, if any. */
	bp = strchr(bp, '[');
	if (!bp) {
		return 1; /* at least we got the first dim. */
	}
	num = 0;
	while (++bp < name+len-1) {
		const char c = *bp;
		if (c == ']') {
			break;
		}
		if (c <= '9' && c >= '0') {
			num *= 10;
			num += (c - '0');
		} else {
			printf("array2 parse error.\n");
			return 1;
		}
	}
	if (dim1) {
		if (dim2) {
			*dim1 = *dim2;
			*dim2 = num;
		} else {
			*dim1 = num;
		}
	}

	return 1;
}


// ----------------------------------------------------- //
void bDNA::init(char *data, int len, bool swap)
{
	int *intPtr=0;short *shtPtr=0;
	char *cp = 0;int dataLen =0;
	//long nr=0;
	intPtr = (int*)data;

	/*
		SDNA (4 bytes) (magic number)
		NAME (4 bytes)
		<nr> (4 bytes) amount of names (int)
		<string>
		<string>
	*/

	if (strncmp(data, "SDNA", 4)==0)
	{
		// skip ++ NAME
		intPtr++; intPtr++;
	}



	// Parse names
	if (swap) 
	{
		*intPtr = ChunkUtils::swapInt(*intPtr);
	}
	dataLen = *intPtr;
	intPtr++;

	cp = (char*)intPtr;
	int i;
	for ( i=0; i<dataLen; i++)
	{
		bNameInfo info;
		info.m_name = cp;
		info.m_isPointer = (info.m_name[0] == '*') || (info.m_name[1] == '*');
		name_is_array(info.m_name,&info.m_dim0,&info.m_dim1);
		m_Names.push_back(info);
		while (*cp)cp++;
		cp++;
	}

	
	cp = b3AlignPointer(cp,4);
	

	/*
		TYPE (4 bytes)
		<nr> amount of types (int)
		<string>
		<string>
	*/

	intPtr = (int*)cp;
	assert(strncmp(cp, "TYPE", 4)==0); intPtr++;

	if (swap) 
	{
		*intPtr = ChunkUtils::swapInt(*intPtr);
	}
	dataLen = *intPtr;
	intPtr++;

	cp = (char*)intPtr;
	for ( i=0; i<dataLen; i++)
	{
		mTypes.push_back(cp);
		while (*cp)cp++;
		cp++;
	}


	cp = b3AlignPointer(cp,4);

	/*
		TLEN (4 bytes)
		<len> (short) the lengths of types
		<len>
	*/

	// Parse type lens
	intPtr = (int*)cp;
	assert(strncmp(cp, "TLEN", 4)==0); intPtr++;

	dataLen = (int)mTypes.size();

	shtPtr = (short*)intPtr;
	for ( i=0; i<dataLen; i++, shtPtr++)
	{
		if (swap)
			shtPtr[0] = ChunkUtils::swapShort(shtPtr[0]);
		mTlens.push_back(shtPtr[0]);
	}

	if (dataLen & 1) shtPtr++;

	/*
		STRC (4 bytes)
		<nr> amount of structs (int)
		<typenr>
		<nr_of_elems>
		<typenr>
		<namenr>
		<typenr>
		<namenr>
	*/

	intPtr = (int*)shtPtr;
	cp = (char*)intPtr;
	assert(strncmp(cp, "STRC", 4)==0); intPtr++;

	if (swap) 
	{
		*intPtr = ChunkUtils::swapInt(*intPtr);
	}
	dataLen = *intPtr;
	intPtr++;


	shtPtr = (short*)intPtr;
	for ( i=0; i<dataLen; i++)
	{
		mStructs.push_back (shtPtr);
		if (swap)
		{
			shtPtr[0]= ChunkUtils::swapShort(shtPtr[0]);
			shtPtr[1]= ChunkUtils::swapShort(shtPtr[1]);

			int len = shtPtr[1];
			shtPtr+= 2;

			for (int a=0; a<len; a++, shtPtr+=2)
			{
				shtPtr[0]= ChunkUtils::swapShort(shtPtr[0]);
				shtPtr[1]= ChunkUtils::swapShort(shtPtr[1]);
			}
		}
		else
			shtPtr+= (2*shtPtr[1])+2;
	}


	// build reverse lookups
	for ( i=0; i<(int)mStructs.size(); i++)
	{
		short *strc = mStructs.at(i);
		if (!mPtrLen && strcmp(mTypes[strc[0]],"ListBase")==0)
		{
			mPtrLen = mTlens[strc[0]]/2;
		}

		mStructReverse.insert(strc[0], i);
		mTypeLookup.insert(b3HashString(mTypes[strc[0]]),i);
	}
}


// ----------------------------------------------------- //
int bDNA::getArraySize(char* string)
{
	int ret = 1;
	int len = strlen(string);

	
	char* next = 0;
	for (int i=0; i<len; i++)
	{
		char c = string[i];

		if (c == '[')
			next = &string[i+1];
		else if (c==']')
			if (next)
				ret *= atoi(next);
	}

//	print (string << ' ' << ret);
	return ret;
}


void bDNA::dumpTypeDefinitions()
{
	int i;

	int numTypes = mTypes.size();
	
	for (i=0;i<numTypes;i++)
	{

	}

	for ( i=0; i<(int)mStructs.size(); i++)
	{
		int totalBytes=0;
		short *oldStruct = mStructs[i];

		int oldLookup = getReverseType(oldStruct[0]);
		if (oldLookup == -1)
		{
			mCMPFlags[i] = FDF_NONE;
			continue;
		}

		short* newStruct = mStructs[oldLookup];
		char* typeName = mTypes[newStruct[0]];
		printf("%3d: %s ",i,typeName);
		
		//char *name = mNames[oldStruct[1]];
		int len = oldStruct[1];
		printf(" (%d fields) ",len);
		oldStruct+=2;

		printf("{");
		int j;
		for (j=0; j<len; ++j,oldStruct+=2) {
			const char* name = m_Names[oldStruct[1]].m_name;
			printf("%s %s",	mTypes[oldStruct[0]],name);
			int elemNumBytes= 0;
			int arrayDimensions = getArraySizeNew(oldStruct[1]);

			if (m_Names[oldStruct[1]].m_isPointer)
			{
				elemNumBytes = VOID_IS_8 ? 8 : 4;
			} else
			{
				elemNumBytes = getLength(oldStruct[0]);
			}
			printf(" /* %d bytes */",elemNumBytes*arrayDimensions);
			
			if (j == len-1) {
				printf(";}");
			} else {
				printf("; ");
			}
			totalBytes+=elemNumBytes*arrayDimensions;
		}
		printf("\ntotalBytes=%d\n\n",totalBytes);

	}
	


#if 0
	/* dump out display of types and their sizes */
	for (i=0; i<bf->types_count; ++i) {
		/* if (!bf->types[i].is_struct)*/
		{
			printf("%3d: sizeof(%s%s)=%d",
				i,
				bf->types[i].is_struct ? "struct " : "atomic ",
				bf->types[i].name, bf->types[i].size);
			if (bf->types[i].is_struct) {
				int j;
				printf(", %d fields: { ", bf->types[i].fieldtypes_count);
				for (j=0; j<bf->types[i].fieldtypes_count; ++j) {
					printf("%s %s",
						bf->types[bf->types[i].fieldtypes[j]].name,
						bf->names[bf->types[i].fieldnames[j]]);
					if (j == bf->types[i].fieldtypes_count-1) {
						printf(";}");
					} else {
						printf("; ");
					}
				}
			}
			printf("\n\n");

		}
	}
#endif

}




//eof



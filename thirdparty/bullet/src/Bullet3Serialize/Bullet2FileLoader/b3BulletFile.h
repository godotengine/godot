/*
bParse
Copyright (c) 2006-2010 Charlie C & Erwin Coumans  http://gamekit.googlecode.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_BULLET_FILE_H
#define B3_BULLET_FILE_H


#include "b3File.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "b3Defines.h"

#include "Bullet3Serialize/Bullet2FileLoader/b3Serializer.h"



namespace bParse {

	// ----------------------------------------------------- //
	class b3BulletFile : public bFile
	{
		

	protected:
	
		char*	m_DnaCopy;
				
	public:

		b3AlignedObjectArray<bStructHandle*>	m_softBodies;

		b3AlignedObjectArray<bStructHandle*>	m_rigidBodies;

		b3AlignedObjectArray<bStructHandle*>	m_collisionObjects;

		b3AlignedObjectArray<bStructHandle*>	m_collisionShapes;

		b3AlignedObjectArray<bStructHandle*>	m_constraints;

		b3AlignedObjectArray<bStructHandle*>	m_bvhs;

		b3AlignedObjectArray<bStructHandle*>	m_triangleInfoMaps;

		b3AlignedObjectArray<bStructHandle*>	m_dynamicsWorldInfo;

		b3AlignedObjectArray<char*>				m_dataBlocks;
		b3BulletFile();

		b3BulletFile(const char* fileName);

		b3BulletFile(char *memoryBuffer, int len);

		virtual ~b3BulletFile();

		virtual	void	addDataBlock(char* dataBlock);
	

		// experimental
		virtual int		write(const char* fileName, bool fixupPointers=false);

		virtual	void	parse(int verboseMode);

		virtual	void parseData();

		virtual	void	writeDNA(FILE* fp);

		void	addStruct(const char* structType,void* data, int len, void* oldPtr, int code);

	};
};

#endif //B3_BULLET_FILE_H

/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_SOFT_BODY_SOLVER_VERTEX_BUFFER_H
#define BT_SOFT_BODY_SOLVER_VERTEX_BUFFER_H

class btVertexBufferDescriptor
{
public:
	enum BufferTypes
	{
		CPU_BUFFER,
		DX11_BUFFER,
		OPENGL_BUFFER
	};

protected:
	bool m_hasVertexPositions;
	bool m_hasNormals;

	int m_vertexOffset;
	int m_vertexStride;

	int m_normalOffset;
	int m_normalStride;

public:
	btVertexBufferDescriptor()
	{
		m_hasVertexPositions = false;
		m_hasNormals = false;
		m_vertexOffset = 0;
		m_vertexStride = 0;
		m_normalOffset = 0;
		m_normalStride = 0;
	}

	virtual ~btVertexBufferDescriptor()
	{
	}

	virtual bool hasVertexPositions() const
	{
		return m_hasVertexPositions;
	}

	virtual bool hasNormals() const
	{
		return m_hasNormals;
	}

	/**
	 * Return the type of the vertex buffer descriptor.
	 */
	virtual BufferTypes getBufferType() const = 0;

	/**
	 * Return the vertex offset in floats from the base pointer.
	 */
	virtual int getVertexOffset() const
	{
		return m_vertexOffset;
	}

	/**
	 * Return the vertex stride in number of floats between vertices.
	 */
	virtual int getVertexStride() const
	{
		return m_vertexStride;
	}

	/**
	 * Return the vertex offset in floats from the base pointer.
	 */
	virtual int getNormalOffset() const
	{
		return m_normalOffset;
	}

	/**
	 * Return the vertex stride in number of floats between vertices.
	 */
	virtual int getNormalStride() const
	{
		return m_normalStride;
	}
};

class btCPUVertexBufferDescriptor : public btVertexBufferDescriptor
{
protected:
	float *m_basePointer;

public:
	/**
	 * vertexBasePointer is pointer to beginning of the buffer.
	 * vertexOffset is the offset in floats to the first vertex.
	 * vertexStride is the stride in floats between vertices.
	 */
	btCPUVertexBufferDescriptor(float *basePointer, int vertexOffset, int vertexStride)
	{
		m_basePointer = basePointer;
		m_vertexOffset = vertexOffset;
		m_vertexStride = vertexStride;
		m_hasVertexPositions = true;
	}

	/**
	 * vertexBasePointer is pointer to beginning of the buffer.
	 * vertexOffset is the offset in floats to the first vertex.
	 * vertexStride is the stride in floats between vertices.
	 */
	btCPUVertexBufferDescriptor(float *basePointer, int vertexOffset, int vertexStride, int normalOffset, int normalStride)
	{
		m_basePointer = basePointer;

		m_vertexOffset = vertexOffset;
		m_vertexStride = vertexStride;
		m_hasVertexPositions = true;

		m_normalOffset = normalOffset;
		m_normalStride = normalStride;
		m_hasNormals = true;
	}

	virtual ~btCPUVertexBufferDescriptor()
	{
	}

	/**
	 * Return the type of the vertex buffer descriptor.
	 */
	virtual BufferTypes getBufferType() const
	{
		return CPU_BUFFER;
	}

	/**
	 * Return the base pointer in memory to the first vertex.
	 */
	virtual float *getBasePointer() const
	{
		return m_basePointer;
	}
};

#endif  // #ifndef BT_SOFT_BODY_SOLVER_VERTEX_BUFFER_H

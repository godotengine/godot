#ifndef GIM_BITSET_H_INCLUDED
#define GIM_BITSET_H_INCLUDED
/*! \file gim_bitset.h
\author Francisco Leon Najera
*/
/*
-----------------------------------------------------------------------------
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2006 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com

 This library is free software; you can redistribute it and/or
 modify it under the terms of EITHER:
   (1) The GNU Lesser General Public License as published by the Free
       Software Foundation; either version 2.1 of the License, or (at
       your option) any later version. The text of the GNU Lesser
       General Public License is included with this library in the
       file GIMPACT-LICENSE-LGPL.TXT.
   (2) The BSD-style license that is included with this library in
       the file GIMPACT-LICENSE-BSD.TXT.
   (3) The zlib/libpng license that is included with this library in
       the file GIMPACT-LICENSE-ZLIB.TXT.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
 GIMPACT-LICENSE-LGPL.TXT, GIMPACT-LICENSE-ZLIB.TXT and GIMPACT-LICENSE-BSD.TXT for more details.

-----------------------------------------------------------------------------
*/

#include "gim_array.h"


#define GUINT_BIT_COUNT 32
#define GUINT_EXPONENT 5

class gim_bitset
{
public:
    gim_array<GUINT> m_container;

    gim_bitset()
    {

    }

    gim_bitset(GUINT bits_count)
    {
        resize(bits_count);
    }

    ~gim_bitset()
    {
    }

	inline bool resize(GUINT newsize)
	{
		GUINT oldsize = m_container.size();
		m_container.resize(newsize/GUINT_BIT_COUNT + 1,false);
		while(oldsize<m_container.size())
		{
			m_container[oldsize] = 0;
		}
		return true;
	}

	inline GUINT size()
	{
		return m_container.size()*GUINT_BIT_COUNT;
	}

	inline void set_all()
	{
		for(GUINT i = 0;i<m_container.size();++i)
		{
			m_container[i] = 0xffffffff;
		}
	}

	inline void clear_all()
	{
	    for(GUINT i = 0;i<m_container.size();++i)
		{
			m_container[i] = 0;
		}
	}

	inline void set(GUINT bit_index)
	{
		if(bit_index>=size())
		{
			resize(bit_index);
		}
		m_container[bit_index >> GUINT_EXPONENT] |= (1 << (bit_index & (GUINT_BIT_COUNT-1)));
	}

	///Return 0 or 1
	inline char get(GUINT bit_index)
	{
		if(bit_index>=size())
		{
			return 0;
		}
		char value = m_container[bit_index >> GUINT_EXPONENT] &
					 (1 << (bit_index & (GUINT_BIT_COUNT-1)));
		return value;
	}

	inline void clear(GUINT bit_index)
	{
	    m_container[bit_index >> GUINT_EXPONENT] &= ~(1 << (bit_index & (GUINT_BIT_COUNT-1)));
	}
};





#endif // GIM_CONTAINERS_H_INCLUDED

/*
 * Musepack audio compression
 * Copyright (C) 1999-2004 Buschmann/Klemm/Piecha/Wolf
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#pragma once

# define clip(x,min,max) ( (x) < (min) ? (min) : (x) > (max) ? (max) : (x) )

#ifdef __cplusplus

# define maxi(A,B)  ( (A) >? (B) )
# define mini(A,B)  ( (A) <? (B) )
# define maxd(A,B)  ( (A) >? (B) )
# define mind(A,B)  ( (A) <? (B) )
# define maxf(A,B)  ( (A) >? (B) )
# define minf(A,B)  ( (A) <? (B) )

#else

# define maxi(A,B)  ( (A) > (B)  ?  (A)  :  (B) )
# define mini(A,B)  ( (A) < (B)  ?  (A)  :  (B) )
# define maxd(A,B)  ( (A) > (B)  ?  (A)  :  (B) )
# define mind(A,B)  ( (A) < (B)  ?  (A)  :  (B) )
# define maxf(A,B)  ( (A) > (B)  ?  (A)  :  (B) )
# define minf(A,B)  ( (A) < (B)  ?  (A)  :  (B) )

#endif

#ifdef __GNUC__

# define absi(A)    abs   (A)
# define absf(A)    fabsf (A)
# define absd(A)    fabs  (A)

#else

# define absi(A)    ( (A) >= 0    ?  (A)  : -(A) )
# define absf(A)    ( (A) >= 0.f  ?  (A)  : -(A) )
# define absd(A)    ( (A) >= 0.   ?  (A)  : -(A) )

#endif


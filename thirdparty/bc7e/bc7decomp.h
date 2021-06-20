#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4201) //  nonstandard extension used: nameless struct/union
#endif

#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <math.h>
#include <assert.h>

namespace bc7decomp 
{

enum eNoClamp { cNoClamp };

template <typename S> inline S clamp(S value, S low, S high) { return (value < low) ? low : ((value > high) ? high : value); }

class color_rgba
{
public:
	union
	{
		uint8_t m_comps[4];

		struct
		{
			uint8_t r;
			uint8_t g;
			uint8_t b;
			uint8_t a;
		};
	};

	inline color_rgba()
	{
		static_assert(sizeof(*this) == 4, "sizeof(*this) != 4");
	}

	inline color_rgba(int y)
	{
		set(y);
	}

	inline color_rgba(int y, int na)
	{
		set(y, na);
	}

	inline color_rgba(int sr, int sg, int sb, int sa)
	{
		set(sr, sg, sb, sa);
	}

	inline color_rgba(eNoClamp, int sr, int sg, int sb, int sa)
	{
		set_noclamp_rgba((uint8_t)sr, (uint8_t)sg, (uint8_t)sb, (uint8_t)sa);
	}

	inline color_rgba& set_noclamp_y(int y)
	{
		m_comps[0] = (uint8_t)y;
		m_comps[1] = (uint8_t)y;
		m_comps[2] = (uint8_t)y;
		m_comps[3] = (uint8_t)255;
		return *this;
	}

	inline color_rgba &set_noclamp_rgba(int sr, int sg, int sb, int sa)
	{
		m_comps[0] = (uint8_t)sr;
		m_comps[1] = (uint8_t)sg;
		m_comps[2] = (uint8_t)sb;
		m_comps[3] = (uint8_t)sa;
		return *this;
	}

	inline color_rgba &set(int y)
	{
		m_comps[0] = static_cast<uint8_t>(clamp<int>(y, 0, 255));
		m_comps[1] = m_comps[0];
		m_comps[2] = m_comps[0];
		m_comps[3] = 255;
		return *this;
	}

	inline color_rgba &set(int y, int na)
	{
		m_comps[0] = static_cast<uint8_t>(clamp<int>(y, 0, 255));
		m_comps[1] = m_comps[0];
		m_comps[2] = m_comps[0];
		m_comps[3] = static_cast<uint8_t>(clamp<int>(na, 0, 255));
		return *this;
	}

	inline color_rgba &set(int sr, int sg, int sb, int sa)
	{
		m_comps[0] = static_cast<uint8_t>(clamp<int>(sr, 0, 255));
		m_comps[1] = static_cast<uint8_t>(clamp<int>(sg, 0, 255));
		m_comps[2] = static_cast<uint8_t>(clamp<int>(sb, 0, 255));
		m_comps[3] = static_cast<uint8_t>(clamp<int>(sa, 0, 255));
		return *this;
	}

	inline color_rgba &set_rgb(int sr, int sg, int sb)
	{
		m_comps[0] = static_cast<uint8_t>(clamp<int>(sr, 0, 255));
		m_comps[1] = static_cast<uint8_t>(clamp<int>(sg, 0, 255));
		m_comps[2] = static_cast<uint8_t>(clamp<int>(sb, 0, 255));
		return *this;
	}

	inline color_rgba &set_rgb(const color_rgba &other)
	{
		r = other.r;
		g = other.g;
		b = other.b;
		return *this;
	}

	inline const uint8_t &operator[] (uint32_t index) const { assert(index < 4); return m_comps[index]; }
	inline uint8_t &operator[] (uint32_t index) { assert(index < 4); return m_comps[index]; }
		
	inline void clear()
	{
		m_comps[0] = 0;
		m_comps[1] = 0;
		m_comps[2] = 0;
		m_comps[3] = 0;
	}

	inline bool operator== (const color_rgba &rhs) const
	{
		if (m_comps[0] != rhs.m_comps[0]) return false;
		if (m_comps[1] != rhs.m_comps[1]) return false;
		if (m_comps[2] != rhs.m_comps[2]) return false;
		if (m_comps[3] != rhs.m_comps[3]) return false;
		return true;
	}

	inline bool operator!= (const color_rgba &rhs) const
	{
		return !(*this == rhs);
	}

	inline bool operator<(const color_rgba &rhs) const
	{
		for (int i = 0; i < 4; i++)
		{
			if (m_comps[i] < rhs.m_comps[i])
				return true;
			else if (m_comps[i] != rhs.m_comps[i])
				return false;
		}
		return false;
	}

	inline int get_601_luma() const { return (19595U * m_comps[0] + 38470U * m_comps[1] + 7471U * m_comps[2] + 32768U) >> 16U; }
	inline int get_709_luma() const { return (13938U * m_comps[0] + 46869U * m_comps[1] + 4729U * m_comps[2] + 32768U) >> 16U; } 
	inline int get_luma(bool luma_601) const { return luma_601 ? get_601_luma() : get_709_luma(); }

	static color_rgba comp_min(const color_rgba& a, const color_rgba& b) { return color_rgba(std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]), std::min(a[3], b[3])); }
	static color_rgba comp_max(const color_rgba& a, const color_rgba& b) { return color_rgba(std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]), std::max(a[3], b[3])); }
};

bool unpack_bc7(const void *pBlock, color_rgba *pPixels);

} // namespace bc7decomp

namespace bc7decomp_ref
{
	bool unpack_bc7(const void* pBlock, bc7decomp::color_rgba* pPixels);
} // namespace bc7decomp_ref

#ifdef _MSC_VER
#pragma warning(pop)
#endif
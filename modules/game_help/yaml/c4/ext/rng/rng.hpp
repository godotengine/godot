/* Copyright (c) 2018 Arvid Gerstmann.
 *
 * https://arvid.io/2018/07/02/better-cxx-prng/
 *
 * This code is licensed under MIT license. */
#ifndef AG_RANDOM_H
#define AG_RANDOM_H

#include <stdint.h>
#include <random>


#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace c4 {
namespace rng {

class splitmix
{
public:
    using result_type = uint32_t;
    static constexpr result_type (min)() { return 0; }
    static constexpr result_type (max)() { return UINT32_MAX; }
    friend bool operator==(splitmix const &, splitmix const &);
    friend bool operator!=(splitmix const &, splitmix const &);

    splitmix() : m_seed(1) {}
    explicit splitmix(uint64_t s) : m_seed(s) {}
    explicit splitmix(std::random_device &rd)
    {
        seed(rd);
    }

    void seed(uint64_t s) { m_seed = s; }
    void seed(std::random_device &rd)
    {
        m_seed = uint64_t(rd()) << 31 | uint64_t(rd());
    }

    result_type operator()()
    {
        uint64_t z = (m_seed += UINT64_C(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        return result_type((z ^ (z >> 31)) >> 31);
    }

    void discard(unsigned long long n)
    {
        for (unsigned long long i = 0; i < n; ++i)
            operator()();
    }

private:
    uint64_t m_seed;
};

inline bool operator==(splitmix const &lhs, splitmix const &rhs)
{
    return lhs.m_seed == rhs.m_seed;
}
inline bool operator!=(splitmix const &lhs, splitmix const &rhs)
{
    return lhs.m_seed != rhs.m_seed;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class xorshift
{
public:
    using result_type = uint32_t;
    static constexpr result_type (min)() { return 0; }
    static constexpr result_type (max)() { return UINT32_MAX; }
    friend bool operator==(xorshift const &, xorshift const &);
    friend bool operator!=(xorshift const &, xorshift const &);

    xorshift() : m_seed(0xc1f651c67c62c6e0ull) {}
    explicit xorshift(std::random_device &rd)
    {
        seed(rd);
    }

    void seed(uint64_t s) { m_seed = s; }
    void seed(std::random_device &rd)
    {
        m_seed = uint64_t(rd()) << 31 | uint64_t(rd());
    }

    result_type operator()()
    {
        uint64_t result = m_seed * 0xd989bcacc137dcd5ull;
        m_seed ^= m_seed >> 11;
        m_seed ^= m_seed << 31;
        m_seed ^= m_seed >> 18;
        return uint32_t(result >> 32ull);
    }

    void discard(unsigned long long n)
    {
        for (unsigned long long i = 0; i < n; ++i)
            operator()();
    }

private:
    uint64_t m_seed;
};

inline bool operator==(xorshift const &lhs, xorshift const &rhs)
{
    return lhs.m_seed == rhs.m_seed;
}
inline bool operator!=(xorshift const &lhs, xorshift const &rhs)
{
    return lhs.m_seed != rhs.m_seed;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class pcg
{
public:
    using result_type = uint32_t;
    static constexpr result_type (min)() { return 0; }
    static constexpr result_type (max)() { return UINT32_MAX; }
    friend bool operator==(pcg const &, pcg const &);
    friend bool operator!=(pcg const &, pcg const &);

    pcg()
        : m_state(0x853c49e6748fea9bULL)
        , m_inc(0xda3e39cb94b95bdbULL)
    {}
    explicit pcg(uint64_t s) { m_state = s; m_inc = m_state << 1; }
    explicit pcg(std::random_device &rd)
    {
        seed(rd);
    }

    void seed(uint64_t s) { m_state = s; }
    void seed(std::random_device &rd)
    {
        uint64_t s0 = uint64_t(rd()) << 31 | uint64_t(rd());
        uint64_t s1 = uint64_t(rd()) << 31 | uint64_t(rd());

        m_state = 0;
        m_inc = (s1 << 1) | 1;
        (void)operator()();
        m_state += s0;
        (void)operator()();
    }

    result_type operator()()
    {
        uint64_t oldstate = m_state;
        m_state = oldstate * 6364136223846793005ULL + m_inc;
        uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
        //int rot = oldstate >> 59u; // the original. error?
        int64_t rot = (int64_t)oldstate >> 59u; // error?
        return (xorshifted >> rot) | (xorshifted << ((uint64_t)(-rot) & 31));
    }

    void discard(unsigned long long n)
    {
        for (unsigned long long i = 0; i < n; ++i)
            operator()();
    }

private:
    uint64_t m_state;
    uint64_t m_inc;
};

inline bool operator==(pcg const &lhs, pcg const &rhs)
{
    return lhs.m_state == rhs.m_state
        && lhs.m_inc == rhs.m_inc;
}
inline bool operator!=(pcg const &lhs, pcg const &rhs)
{
    return lhs.m_state != rhs.m_state
        || lhs.m_inc != rhs.m_inc;
}

} // namespace rng
} // namespace c4

#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif

#endif /* AG_RANDOM_H */

#ifndef _C4_HASH_HPP_
#define _C4_HASH_HPP_

#include "config.hpp"
#include <climits>

/** @file hash.hpp */

/** @defgroup hash Hash utils
 * @see http://aras-p.info/blog/2016/08/02/Hash-Functions-all-the-way-down/ */

namespace c4 {

namespace detail {

/** @internal
 * @ingroup hash
 * @see this was taken a great answer in stackoverflow:
 * https://stackoverflow.com/a/34597785/5875572
 * @see http://aras-p.info/blog/2016/08/02/Hash-Functions-all-the-way-down/ */
template<typename ResultT, ResultT OffsetBasis, ResultT Prime>
class basic_fnv1a final
{

  static_assert(std::is_unsigned<ResultT>::value, "need unsigned integer");

public:

    using result_type = ResultT;

private:

    result_type state_ {};

public:

    C4_CONSTEXPR14 basic_fnv1a() noexcept : state_ {OffsetBasis} {}

    C4_CONSTEXPR14 void update(const void *const data, const size_t size) noexcept
    {
        auto cdata = static_cast<const unsigned char *>(data);
        auto acc = this->state_;
        for(size_t i = 0; i < size; ++i)
        {
            const auto next = size_t(cdata[i]);
            acc = (acc ^ next) * Prime;
        }
        this->state_ = acc;
    }

    C4_CONSTEXPR14 result_type digest() const noexcept
    {
        return this->state_;
    }

};

using fnv1a_32 = basic_fnv1a<uint32_t, UINT32_C(          2166136261), UINT32_C(     16777619)>;
using fnv1a_64 = basic_fnv1a<uint64_t, UINT64_C(14695981039346656037), UINT64_C(1099511628211)>;

template<size_t Bits> struct fnv1a;
template<> struct fnv1a<32> { using type = fnv1a_32; };
template<> struct fnv1a<64> { using type = fnv1a_64; };

} // namespace detail


/** @ingroup hash */
template<size_t Bits>
using fnv1a_t = typename detail::fnv1a<Bits>::type;


/** @ingroup hash */
C4_CONSTEXPR14 inline size_t hash_bytes(const void *const data, const size_t size) noexcept
{
    fnv1a_t<CHAR_BIT * sizeof(size_t)> fn{};
    fn.update(data, size);
    return fn.digest();
}

/**
 * @overload hash_bytes
 * @ingroup hash */
template<size_t N>
C4_CONSTEXPR14 inline size_t hash_bytes(const char (&str)[N]) noexcept
{
    fnv1a_t<CHAR_BIT * sizeof(size_t)> fn{};
    fn.update(str, N);
    return fn.digest();
}

} // namespace c4


#endif // _C4_HASH_HPP_

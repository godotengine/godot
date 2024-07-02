// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#if PSD_USE_MSVC
	#pragma intrinsic(_byteswap_ushort)
	#pragma intrinsic(_byteswap_ulong)
	#pragma intrinsic(_byteswap_uint64)

	#define ByteSwap16		_byteswap_ushort
	#define ByteSwap32		_byteswap_ulong
	#define ByteSwap64		_byteswap_uint64
#else
	#define ByteSwap16		__builtin_bswap16
	#define ByteSwap32		__builtin_bswap32
	#define ByteSwap64		__builtin_bswap64
#endif


namespace endianUtil
{
	namespace internal
	{
		/// \brief Base template for types of size N.
		/// \details The individual template specializations take care of working with the correct type
		/// based on the size of the generic type T.
		template <size_t N>
		struct Implementation {};


		/// Template specialization for 1-byte types.
		template <>
		struct Implementation<1u>
		{
			/// Internal function used by util::BigEndianToNative.
			template <typename T>
			static PSD_INLINE T BigEndianToNative(T value)
			{
				static_assert(sizeof(T) == 1, "sizeof(T) is not 1 byte.");

				return value;
			}
		};


		/// Template specialization for 2-byte types.
		template <>
		struct Implementation<2u>
		{
			/// Internal function used by util::BigEndianToNative.
			template <typename T>
			static inline T BigEndianToNative(T value)
			{
				static_assert(sizeof(T) == 2, "sizeof(T) is not 2 byte.");

				union
				{
					T asT;
					uint16_t as_uint16_t;
				};

				asT = value;
				as_uint16_t = ByteSwap16(as_uint16_t);

				return asT;
			}
		};


		/// Template specialization for 4-byte types.
		template <>
		struct Implementation<4u>
		{
			/// Internal function used by util::BigEndianToNative.
			template <typename T>
			static inline T BigEndianToNative(T value)
			{
				static_assert(sizeof(T) == 4, "sizeof(T) is not 4 byte.");

				union
				{
					T asT;
					uint32_t as_uint32_t;
				};

				asT = value;
				as_uint32_t = ByteSwap32(as_uint32_t);

				return asT;
			}
		};


		/// Template specialization for 8-byte types.
		template <>
		struct Implementation<8u>
		{
			/// Internal function used by util::BigEndianToNative.
			template <typename T>
			static inline T BigEndianToNative(T value)
			{
				static_assert(sizeof(T) == 8, "sizeof(T) is not 8 byte.");

				union
				{
					T asT;
					uint64_t as_uint64_t;
				};

				asT = value;
				as_uint64_t = ByteSwap64(as_uint64_t);

				return asT;
			}
		};
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	PSD_INLINE T BigEndianToNative(T value)
	{
		// defer the implementation to the correct helper template, based on the size of the type
		return internal::Implementation<sizeof(T)>::BigEndianToNative(value);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	PSD_INLINE T LittleEndianToNative(T value)
	{
		return value;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	PSD_INLINE T NativeToBigEndian(T value)
	{
		return BigEndianToNative(value);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	PSD_INLINE T NativeToLittleEndian(T value)
	{
		return LittleEndianToNative(value);
	}
}


#undef ByteSwap16
#undef ByteSwap32
#undef ByteSwap64

// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

namespace util
{
	/// \ingroup Util
	/// \class Key
	/// \brief Helper template used to produce a 32-bit value out of 4 characters for comparison purposes.
	/// \details Many parts of a .PSD file store a 4-character string for identification purposes, e.g. a valid .PSD file
	/// always starts with the character sequence "8BPS". This helper template can be used to build a 32-bit value that
	/// can be compared against the big-endian signature read from the .PSD file.
	/// \code
	///   // read the signature from a file, and convert the big-endian value to native endianness
	///   const uint32_t signature = util::ReadFromFileBE<uint32_t>(reader);
	///
	///   // check if the signature matches "8BPS" by using the util::Key helper template
	///   if (signature == util::Key<'8', 'B', 'P', 'S'>::VALUE)
	///     ...
	/// \endcode
	/// \remark Note that util::Key::VALUE is a compile-time constant, and can therefore be used in other template expressions,
	/// switch cases, etc.
	template <char a, char b, char c, char d>
	struct Key
	{
		static const uint32_t VALUE =
			((static_cast<uint32_t>(a) << 24u) |
			(static_cast<uint32_t>(b) << 16u) |
			(static_cast<uint32_t>(c) << 8u) |
			(static_cast<uint32_t>(d)));
	};

	template <char a, char b, char c, char d> const uint32_t Key<a, b, c, d>::VALUE;
}

PSD_NAMESPACE_END

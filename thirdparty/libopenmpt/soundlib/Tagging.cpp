/*
 * tagging.cpp
 * -----------
 * Purpose: Structure holding a superset of tags for all supported output sample or stream files or types.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Tagging.h"
#include "../common/version.h"

OPENMPT_NAMESPACE_BEGIN


FileTags::FileTags()
{
	encoder = mpt::ToUnicode(mpt::CharsetASCII, MptVersion::GetOpenMPTVersionStr());
}


mpt::ustring GetSampleNameFromTags(const FileTags &tags)
{
	mpt::ustring result;
	if(tags.artist.empty())
	{
		result = tags.title;
	} else
	{
		result = mpt::format(MPT_USTRING("%1 (by %2)"))(tags.title, tags.artist);
	}
	return result;
}


OPENMPT_NAMESPACE_END

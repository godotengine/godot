/*
 * Tagging.h
 * ---------
 * Purpose: Structure holding a superset of tags for all supported output sample or stream files or types.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include <string>

OPENMPT_NAMESPACE_BEGIN


struct FileTags
{

	mpt::ustring encoder;

	mpt::ustring title;
	mpt::ustring comments;

	mpt::ustring bpm;

	mpt::ustring artist;
	mpt::ustring album;
	mpt::ustring trackno;
	mpt::ustring year;
	mpt::ustring url;

	mpt::ustring genre;

	FileTags();

};


mpt::ustring GetSampleNameFromTags(const FileTags &tags);


OPENMPT_NAMESPACE_END

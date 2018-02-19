/*
 * version.h
 * ---------
 * Purpose: OpenMPT version handling.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "FlagSet.h"

#include <string>


OPENMPT_NAMESPACE_BEGIN


//Creates version number from version parts that appears in version string.
//For example MAKE_VERSION_NUMERIC(1,17,02,28) gives version number of 
//version 1.17.02.28. 
#define MAKE_VERSION_NUMERIC_HELPER(prefix,v0,v1,v2,v3) ((prefix##v0 << 24) | (prefix##v1<<16) | (prefix##v2<<8) | (prefix##v3))
#define MAKE_VERSION_NUMERIC(v0,v1,v2,v3) (MptVersion::VersionNum(MAKE_VERSION_NUMERIC_HELPER(0x,v0,v1,v2,v3)))


namespace MptVersion
{

	typedef uint32 VersionNum;

	extern const VersionNum num; // e.g. 0x01170208
	extern const char * const str; // e.g "1.17.02.08"

	// Return a OpenMPT version string suitable for file format tags 
	std::string GetOpenMPTVersionStr(); // e.g. "OpenMPT 1.17.02.08"

	// Returns numerical version value from given version string.
	VersionNum ToNum(const std::string &s);

	// Returns version string from given numerical version value.
	std::string ToStr(const VersionNum v);
	mpt::ustring ToUString(const VersionNum v);

	// Return a version without build number (the last number in the version).
	// The current versioning scheme uses this number only for test builds, and it should be 00 for official builds,
	// So sometimes it might be wanted to do comparisons without the build number.
	VersionNum RemoveBuildNumber(const VersionNum num_);

	// Returns true if a given version number is from a test build, false if it's a release build.
	bool IsTestBuild(const VersionNum num_ = MptVersion::num);

	// Return true if this is a debug build with no optimizations
	bool IsDebugBuild();

	struct SourceInfo
	{
		std::string Url; // svn repository url (or empty string)
		int Revision; // svn revision (or 0)
		bool IsDirty; // svn working copy is dirty (or false)
		bool HasMixedRevisions; // svn working copy has mixed revisions (or false)
		bool IsPackage; // source code originates from a packaged version of the source code
		std::string Date; // svn date (or empty string)
		SourceInfo() : Url(std::string()), Revision(0), IsDirty(false), HasMixedRevisions(false), IsPackage(false) { }
	public:
		std::string GetUrlWithRevision() const; // i.e. "https://source.openmpt.org/svn/openmpt/trunk/OpenMPT@1234" or empty string
		std::string GetStateString() const; // i.e. "+dirty" or "clean"
	};
	SourceInfo GetSourceInfo();

	// Returns either the URL to download release builds or the URL to download test builds, depending on the current build.
	mpt::ustring GetDownloadURL();

	// Return a string decribing the time of the build process (if built from a svn working copy and tsvn was available during build, otherwise it returns the time version.cpp was last rebuild which could be unreliable as it does not get rebuild every time without tsvn)
	std::string GetBuildDateString();

	// Return a string decribing some of the build features
	std::string GetBuildFeaturesString(); // e.g. " NO_VST NO_DSOUND"

	// Return a string describing the compiler version used for building.
	std::string GetBuildCompilerString(); // e.g. "Microsoft Compiler 15.00.20706.01"

	enum Strings
	{
		StringsNone         = 0,
		StringVersion       = 1<<0, // "1.23.35.45"
		StringRevision      = 1<<2, // "-r1234+"
		StringBitness       = 1<<3, // "32 bit"
		StringSourceInfo    = 1<<4, // "https://source.openmpt.org/svn/openmpt/trunk/OpenMPT@1234 (2016-01-02) +dirty"
		StringBuildFlags    = 1<<5, // "TEST DEBUG"
		StringBuildFeatures = 1<<6, // "NO_VST NO_DSOUND"
	};
	MPT_DECLARE_ENUM(Strings)

	// Returns a versions string with the fields selected via @strings.
	std::string GetVersionString(FlagSet<MptVersion::Strings> strings);

	// Returns a pure version string
	std::string GetVersionStringPure(); // e.g. "1.17.02.08-r1234+ 32 bit"

	// Returns a simple version string
	std::string GetVersionStringSimple(); // e.g. "1.17.02.08-r1234+ TEST"

	// Returns MptVersion::str if the build is a clean release build straight from the repository or an extended string otherwise (if built from a svn working copy and tsvn was available during build)
	std::string GetVersionStringExtended(); // e.g. "1.17.02.08-r1234+ 32 bit DEBUG"

	// Returns a URL for the respective keys. Supported keys: "website", "forum", "bugtracker", "updates", "top_picks"
	mpt::ustring GetURL(std::string key);

	// Returns a multi-line string containing the full credits for the code base
	mpt::ustring GetFullCreditsString();

	// Returns the OpenMPT license text
	mpt::ustring GetLicenseString();

} //namespace MptVersion


OPENMPT_NAMESPACE_END

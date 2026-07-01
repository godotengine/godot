#ifndef EFSW_DIRECTORYSNAPSHOTDIFF_HPP
#define EFSW_DIRECTORYSNAPSHOTDIFF_HPP

#include <efsw/FileInfo.hpp>

namespace efsw {

class DirectorySnapshotDiff {
  public:
	FileInfoList FilesDeleted;
	FileInfoList FilesCreated;
	FileInfoList FilesModified;
	MovedList FilesMoved;
	FileInfoList DirsDeleted;
	FileInfoList DirsCreated;
	FileInfoList DirsModified;
	MovedList DirsMoved;
	bool DirChanged;

	void clear();

	bool changed();
};

#define DiffIterator( FileInfoListName ) \
	it = Diff.FileInfoListName.begin();  \
	for ( ; it != Diff.FileInfoListName.end(); it++ )

#define DiffMovedIterator( MovedListName ) \
	mit = Diff.MovedListName.begin();      \
	for ( ; mit != Diff.MovedListName.end(); mit++ )

} // namespace efsw

#endif

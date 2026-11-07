#include <efsw/DirectorySnapshotDiff.hpp>

namespace efsw {

void DirectorySnapshotDiff::clear() {
	FilesCreated.clear();
	FilesModified.clear();
	FilesMoved.clear();
	FilesDeleted.clear();
	DirsCreated.clear();
	DirsModified.clear();
	DirsMoved.clear();
	DirsDeleted.clear();
}

bool DirectorySnapshotDiff::changed() {
	return !FilesCreated.empty() || !FilesModified.empty() || !FilesMoved.empty() ||
		   !FilesDeleted.empty() || !DirsCreated.empty() || !DirsModified.empty() ||
		   !DirsMoved.empty() || !DirsDeleted.empty();
}

} // namespace efsw

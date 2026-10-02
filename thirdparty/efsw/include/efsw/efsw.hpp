/**
	@author Martín Lucas Golini

	Copyright (c) 2024 Martín Lucas Golini

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.

	This software is a fork of the "simplefilewatcher" by James Wynn (james@jameswynn.com)
	http://code.google.com/p/simplefilewatcher/ also MIT licensed.
*/

#ifndef ESFW_HPP
#define ESFW_HPP

#include <string>
#include <vector>

#if defined( _WIN32 )
#ifdef EFSW_DYNAMIC
// Windows platforms
#ifdef EFSW_EXPORTS
// From DLL side, we must export
#define EFSW_API __declspec( dllexport )
#else
// From client application side, we must import
#define EFSW_API __declspec( dllimport )
#endif
#else
// No specific directive needed for static build
#ifndef EFSW_API
#define EFSW_API
#endif
#endif
#else
#if ( __GNUC__ >= 4 ) && defined( EFSW_EXPORTS )
#ifndef EFSW_API
#define EFSW_API __attribute__( ( visibility( "default" ) ) )
#endif
#endif

// Other platforms don't need to define anything
#ifndef EFSW_API
#define EFSW_API
#endif
#endif

namespace efsw {

/// Type for a watch id
typedef long WatchID;

// forward declarations
class FileWatcherImpl;
class FileWatchListener;
class WatcherOption;

/// Actions to listen for. Rename will send two events, one for
/// the deletion of the old file, and one for the creation of the
/// new file.
namespace Actions {
enum Action {
	/// Sent when a file is created or renamed
	Add = 1,
	/// Sent when a file is deleted or renamed
	Delete = 2,
	/// Sent when a file is modified
	Modified = 3,
	/// Sent when a file is moved
	Moved = 4
};
}
typedef Actions::Action Action;

/// Errors log namespace
namespace Errors {

enum Error {
	NoError = 0,
	FileNotFound = -1,
	FileRepeated = -2,
	FileOutOfScope = -3,
	FileNotReadable = -4,
	/// Directory in remote file system
	/// ( create a generic FileWatcher instance to watch this directory ).
	FileRemote = -5,
	/// File system watcher failed to watch for changes.
	WatcherFailed = -6,
	Unspecified = -7
};

class EFSW_API Log {
  public:
	/// @return The last error logged
	static std::string getLastErrorLog();

	/// @return The code of the last error logged
	static Error getLastErrorCode();

	/// Reset last error
	static void clearLastError();

	/// Creates an error of the type specified
	static Error createLastError( Error err, std::string log );
};

} // namespace Errors
typedef Errors::Error Error;

/// Optional file watcher settings.
namespace Options {
enum Option {
	/// For Windows, the default buffer size of 63*1024 bytes sometimes is not enough and
	/// file system events may be dropped. For that, using a different (bigger) buffer size
	/// can be defined here, but note that this does not work for network drives,
	/// because a buffer larger than 64K will fail the folder being watched, see
	/// http://msdn.microsoft.com/en-us/library/windows/desktop/aa365465(v=vs.85).aspx)
	WinBufferSize = 1,
	/// For Windows, per default all events are captured but we might only be interested
	/// in a subset; the value of the option should be set to a bitwise or'ed set of
	/// FILE_NOTIFY_CHANGE_* flags.
	WinNotifyFilter = 2,
	/// For Windows, prevents the watched directory from being deleted while being monitored.
	/// When set to 1, FILE_SHARE_DELETE is NOT used, preventing directory deletion.
	/// When set to 0 (default), FILE_SHARE_DELETE is used, allowing directory deletion
	/// (original behavior).
	WinPreventDirectoryDeletion = 3,
	/// For macOS (FSEvents backend), per default all modified event types are capture but we might
	/// only be interested in a subset; the value of the option should be set to a set of bitwise
	/// from:
	/// kFSEventStreamEventFlagItemFinderInfoMod
	/// kFSEventStreamEventFlagItemModified
	/// kFSEventStreamEventFlagItemInodeMetaMod
	/// Default configuration will set the 3 flags
	MacModifiedFilter = 3,
	/// macOS sometimes informs incorrect or old file states that may confuse the consumer
	/// The events sanitizer will try to sanitize incorrectly reported events in favor of reducing
	/// the number of events reported. This will have an small performance and memory impact as a
	/// consequence.
	MacSanitizeEvents = 4,
	/// Linux does not support natively recursive watchers. This means that when using recursive
	/// watches efsw registers new watchers for each directory. If new file are created between
	/// the time efsw takes to register the new directory those events might be missed. To avoid
	/// missing new file notifications efsw will trigger synthetic created file events for existing
	/// files in the new directroy watched. This might have the unintended consequence of sending
	/// duplicated created events due to the system also emitting this event.
	LinuxProduceSyntheticEvents = 5,
};
}
typedef Options::Option Option;

/// Listens to files and directories and dispatches events
/// to notify the listener of files and directories changes.
/// @class FileWatcher
class EFSW_API FileWatcher {
  public:
	/// Default constructor, will use the default platform file watcher
	FileWatcher();

	/// Constructor that lets you force the use of the Generic File Watcher
	explicit FileWatcher( bool useGenericFileWatcher );

	virtual ~FileWatcher();

	/// Add a directory watch. Same as the other addWatch, but doesn't have recursive option.
	/// For backwards compatibility.
	/// On error returns WatchID with Error type.
	WatchID addWatch( const std::string& directory, FileWatchListener* watcher );

	/// Add a directory watch
	/// On error returns WatchID with Error type.
	WatchID addWatch( const std::string& directory, FileWatchListener* watcher, bool recursive );

	/// Add a directory watch, allowing customization with options
	/// @param directory The folder to be watched
	/// @param watcher The listener to receive events
	/// @param recursive Set this to true to include subdirectories
	/// @param options Allows customization of a watcher
	/// @return Returns the watch id for the directory or, on error, a WatchID with Error type.
	WatchID addWatch( const std::string& directory, FileWatchListener* watcher, bool recursive,
					  const std::vector<WatcherOption>& options );

	/// Remove a directory watch. This is a brute force search O(nlogn).
	void removeWatch( const std::string& directory );

	/// Remove a directory watch. This is a map lookup O(logn).
	void removeWatch( WatchID watchid );

	/// Starts watching ( in other thread )
	void watch();

	/// @return Returns a list of the directories that are being watched
	std::vector<std::string> directories();

	/** Allow recursive watchers to follow symbolic links to other directories
	 * followSymlinks is disabled by default
	 */
	void followSymlinks( bool follow );

	/** @return If can follow symbolic links to directorioes */
	const bool& followSymlinks() const;

	/** When enable this it will allow symlinks to watch recursively out of the pointed directory.
	 * follorSymlinks must be enabled to this work.
	 * For example, added symlink to /home/folder, and the symlink points to /, this by default is
	 * not allowed, it's only allowed to symlink anything from /home/ and deeper. This is to avoid
	 * great levels of recursion. Enabling this could lead in infinite recursion, and crash the
	 * watcher ( it will try not to avoid this ). Buy enabling out of scope links, it will allow
	 * this behavior. allowOutOfScopeLinks are disabled by default.
	 */
	void allowOutOfScopeLinks( bool allow );

	/// @return Returns if out of scope links are allowed
	const bool& allowOutOfScopeLinks() const;

  private:
	/// The implementation
	FileWatcherImpl* mImpl;
	bool mFollowSymlinks;
	bool mOutOfScopeLinks;
};

/// Basic interface for listening for file events.
/// @class FileWatchListener
class FileWatchListener {
  public:
	virtual ~FileWatchListener() {}

	/// Handles the action file action
	/// @param watchid The watch id for the directory
	/// @param dir The directory
	/// @param filename The filename that was accessed (not full path)
	/// @param action Action that was performed
	/// @param oldFilename The name of the file or directory moved
	virtual void handleFileAction( WatchID watchid, const std::string& dir,
								   const std::string& filename, bool isDir, Action action,
								   std::string oldFilename = "" ) = 0;

	/// Handles that have missed file actions
	/// @param watchid The watch id for the directory
	/// @param dir The directory
	virtual void handleMissedFileActions( WatchID /*watchid*/,
										  const std::string& /*dir*/ ) {}
};

/// Optional, typically platform specific parameter for customization of a watcher.
/// @class WatcherOption
class WatcherOption {
  public:
	WatcherOption( Option option, int value ) : mOption( option ), mValue( value ){};
	Option mOption;
	int mValue;
};

} // namespace efsw

#endif

#include <efsw/Debug.hpp>
#include <efsw/FileSystem.hpp>
#include <efsw/FileWatcherWin32.hpp>
#include <efsw/String.hpp>
#include <efsw/WatcherWin32.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32

#include <algorithm>

namespace efsw {

struct EFSW_FILE_NOTIFY_EXTENDED_INFORMATION_EX {
	DWORD NextEntryOffset;
	DWORD Action;
	LARGE_INTEGER CreationTime;
	LARGE_INTEGER LastModificationTime;
	LARGE_INTEGER LastChangeTime;
	LARGE_INTEGER LastAccessTime;
	LARGE_INTEGER AllocatedLength;
	LARGE_INTEGER FileSize;
	DWORD FileAttributes;
	DWORD ReparsePointTag;
	LARGE_INTEGER FileId;
	LARGE_INTEGER ParentFileId;
	DWORD FileNameLength;
	WCHAR FileName[1];
};

typedef EFSW_FILE_NOTIFY_EXTENDED_INFORMATION_EX* EFSW_PFILE_NOTIFY_EXTENDED_INFORMATION_EX;

typedef BOOL( WINAPI* EFSW_LPREADDIRECTORYCHANGESEXW )( HANDLE hDirectory, LPVOID lpBuffer,
												   DWORD nBufferLength, BOOL bWatchSubtree,
												   DWORD dwNotifyFilter, LPDWORD lpBytesReturned,
												   LPOVERLAPPED lpOverlapped, LPOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine,
												   DWORD ReadDirectoryNotifyInformationClass );

static EFSW_LPREADDIRECTORYCHANGESEXW pReadDirectoryChangesExW = NULL;

#define EFSW_ReadDirectoryNotifyExtendedInformation 2

static void initReadDirectoryChangesEx() {
	static bool hasInit = false;
	if ( !hasInit ) {
		hasInit = true;

		HMODULE hModule = GetModuleHandleW( L"Kernel32.dll" );
		if ( !hModule )
			return;

		pReadDirectoryChangesExW =
			(EFSW_LPREADDIRECTORYCHANGESEXW)GetProcAddress( hModule, "ReadDirectoryChangesExW" );
	}
}

void WatchCallbackOld( WatcherWin32* pWatch ) {
	PFILE_NOTIFY_INFORMATION pNotify;
	size_t offset = 0;
	do {
		bool skip = false;

		pNotify = (PFILE_NOTIFY_INFORMATION)&pWatch->Buffer[offset];
		offset += pNotify->NextEntryOffset;
		int count =
			WideCharToMultiByte( CP_UTF8, 0, pNotify->FileName,
								 pNotify->FileNameLength / sizeof( WCHAR ), NULL, 0, NULL, NULL );
		if ( count == 0 )
			continue;

		std::string nfile( count, '\0' );

		count = WideCharToMultiByte( CP_UTF8, 0, pNotify->FileName,
									 pNotify->FileNameLength / sizeof( WCHAR ), &nfile[0], count,
									 NULL, NULL );

		if ( FILE_ACTION_MODIFIED == pNotify->Action ) {
			FileInfo fifile( std::string( pWatch->DirName ) + nfile );

			if ( pWatch->LastModifiedEvent.file.ModificationTime == fifile.ModificationTime &&
				 pWatch->LastModifiedEvent.file.Size == fifile.Size &&
				 pWatch->LastModifiedEvent.fileName == nfile ) {
				skip = true;
			}

			pWatch->LastModifiedEvent.fileName = nfile;
			pWatch->LastModifiedEvent.file = fifile;
		} else if ( FILE_ACTION_REMOVED == pNotify->Action ) {
			FileInfo fifile( std::string( pWatch->DirName ) + nfile );
			PendingDelete pd;
			pd.isExtended = false;
			pd.fileID.Inode = fifile.Inode;
			pd.fileName = nfile;
			pd.dirName = std::string( pWatch->DirName );
			pd.isDir = fifile.isDirectory();
			pd.timestamp = std::chrono::steady_clock::now();

			FileWatcherWin32::registerPendingDelete( pd );
			skip = true;
		} else if ( FILE_ACTION_ADDED == pNotify->Action ) {
			FileInfo fifile( std::string( pWatch->DirName ) + nfile );

			std::string folderPath( pWatch->DirName );
			std::string realFilename = nfile;
			std::size_t sepPos = nfile.find_last_of( "/\\" );

			if ( sepPos != std::string::npos ) {
				folderPath += nfile.substr( 0, sepPos + 1 < nfile.size() ? sepPos + 1 : sepPos );
				realFilename = nfile.substr( sepPos + 1 );
			}

			FileSystem::dirAddSlashAtEnd( folderPath );

			std::string oldDir, oldFileName;

			FileID fileID;
			fileID.Inode = fifile.Inode;

			if ( FileWatcherWin32::tryMatchMove( fileID, oldDir, oldFileName, false ) ) {
				pWatch->Listener->handleFileAction( pWatch->ID, folderPath, realFilename,
													fifile.isDirectory(), Actions::Moved,
													std::string( oldDir + oldFileName ) );
			} else {
				pWatch->Listener->handleFileAction( pWatch->ID, folderPath, realFilename,
													fifile.isDirectory(), Actions::Add );
			}
			skip = true;
		}

		if ( !skip ) {
			pWatch->Watch->handleAction(
				pWatch, nfile, FileSystem::isDirectory( std::string( pWatch->DirName ) + nfile ),
				pNotify->Action );
		}
	} while ( pNotify->NextEntryOffset != 0 );
}

void WatchCallbackEx( WatcherWin32* pWatch ) {
	EFSW_PFILE_NOTIFY_EXTENDED_INFORMATION_EX pNotify;
	size_t offset = 0;
	do {
		bool skip = false;

		pNotify = (EFSW_PFILE_NOTIFY_EXTENDED_INFORMATION_EX)&pWatch->Buffer[offset];
		offset += pNotify->NextEntryOffset;
		int count =
			WideCharToMultiByte( CP_UTF8, 0, pNotify->FileName,
								 pNotify->FileNameLength / sizeof( WCHAR ), NULL, 0, NULL, NULL );
		if ( count == 0 )
			continue;

		std::string nfile( count, '\0' );

		count = WideCharToMultiByte( CP_UTF8, 0, pNotify->FileName,
									 pNotify->FileNameLength / sizeof( WCHAR ), &nfile[0], count,
									 NULL, NULL );

		if ( FILE_ACTION_MODIFIED == pNotify->Action ) {
			FileInfo fifile( std::string( pWatch->DirName ) + nfile );

			if ( pWatch->LastModifiedEvent.file.ModificationTime == fifile.ModificationTime &&
				 pWatch->LastModifiedEvent.file.Size == fifile.Size &&
				 pWatch->LastModifiedEvent.fileName == nfile ) {
				skip = true;
			}

			pWatch->LastModifiedEvent.fileName = nfile;
			pWatch->LastModifiedEvent.file = fifile;
		} else if ( FILE_ACTION_RENAMED_OLD_NAME == pNotify->Action ) {
			pWatch->OldFiles.emplace_back( nfile, pNotify->FileId );
			skip = true;
		} else if ( FILE_ACTION_RENAMED_NEW_NAME == pNotify->Action ) {
			std::string oldFile;
			LARGE_INTEGER oldFileId{};

			for ( auto it = pWatch->OldFiles.begin(); it != pWatch->OldFiles.end(); ++it ) {
				if ( it->second.QuadPart == pNotify->FileId.QuadPart ) {
					oldFile = it->first;
					oldFileId = it->second;
					it = pWatch->OldFiles.erase( it );
					break;
				}
			}

			if ( oldFile.empty() ) {
				pWatch->Watch->handleAction( pWatch, nfile,
											 pNotify->FileAttributes & FILE_ATTRIBUTE_DIRECTORY,
											 FILE_ACTION_ADDED );
				skip = true;
			} else {
				pWatch->Watch->handleAction( pWatch, oldFile,
											 pNotify->FileAttributes & FILE_ATTRIBUTE_DIRECTORY,
											 FILE_ACTION_RENAMED_OLD_NAME );
			}
		} else if ( FILE_ACTION_REMOVED == pNotify->Action ) {
			PendingDelete pd;
			pd.isExtended = true;
			pd.fileID.ID = pNotify->FileId;
			pd.fileName = nfile;
			pd.dirName = std::string( pWatch->DirName );
			pd.isDir = pNotify->FileAttributes & FILE_ATTRIBUTE_DIRECTORY;
			pd.timestamp = std::chrono::steady_clock::now();

			FileWatcherWin32::registerPendingDelete( pd );
			skip = true;
		} else if ( FILE_ACTION_ADDED == pNotify->Action ) {
			std::string folderPath( pWatch->DirName );
			std::string realFilename = nfile;
			std::size_t sepPos = nfile.find_last_of( "/\\" );

			if ( sepPos != std::string::npos ) {
				folderPath += nfile.substr( 0, sepPos + 1 < nfile.size() ? sepPos + 1 : sepPos );
				realFilename = nfile.substr( sepPos + 1 );
			}

			FileSystem::dirAddSlashAtEnd( folderPath );

			std::string oldDir, oldFileName;

			FileID fileID;
			fileID.ID = pNotify->FileId;

			if ( FileWatcherWin32::tryMatchMove( fileID, oldDir, oldFileName, true ) ) {
				pWatch->Listener->handleFileAction(
					pWatch->ID, folderPath, realFilename,
					pNotify->FileAttributes & FILE_ATTRIBUTE_DIRECTORY, Actions::Moved,
					std::string( oldDir + oldFileName ) );
			} else {
				pWatch->Listener->handleFileAction(
					pWatch->ID, folderPath, realFilename,
					pNotify->FileAttributes & FILE_ATTRIBUTE_DIRECTORY, Actions::Add );
			}
			skip = true;
		}

		if ( !skip ) {
			pWatch->Watch->handleAction( pWatch, nfile,
										 pNotify->FileAttributes & FILE_ATTRIBUTE_DIRECTORY,
										 pNotify->Action );
		}
	} while ( pNotify->NextEntryOffset != 0 );
}

/// Unpacks events and passes them to a user defined callback.
void CALLBACK WatchCallback( DWORD dwNumberOfBytesTransfered, LPOVERLAPPED lpOverlapped ) {
	if ( NULL == lpOverlapped ) {
		return;
	}

	WatcherStructWin32* tWatch = (WatcherStructWin32*)lpOverlapped;
	WatcherWin32* pWatch = tWatch->Watch;

	if ( dwNumberOfBytesTransfered == 0 ) {
		if ( nullptr != pWatch && !pWatch->StopNow ) {
			/// Missed file actions due to buffer overflowed
			std::string dir = pWatch->DirName;
			FileSystem::dirRemoveSlashAtEnd( dir );
			pWatch->Listener->handleMissedFileActions( pWatch->ID, dir );
			RefreshWatch( tWatch );
		} else {
			return;
		}
	}

	// Fork watch depending on the Windows API supported
	if ( pWatch->Extended ) {
		WatchCallbackEx( pWatch );
	} else {
		WatchCallbackOld( pWatch );
	}

	if ( !pWatch->StopNow ) {
		RefreshWatch( tWatch );
	}
}

/// Refreshes the directory monitoring.
RefreshResult RefreshWatch( WatcherStructWin32* pWatch ) {
	initReadDirectoryChangesEx();

	bool bRet = false;
	RefreshResult ret = RefreshResult::Failed;
	pWatch->Watch->Extended = false;

	if ( pReadDirectoryChangesExW ) {
		bRet = pReadDirectoryChangesExW( pWatch->Watch->DirHandle, pWatch->Watch->Buffer.data(),
									  (DWORD)pWatch->Watch->Buffer.size(), pWatch->Watch->Recursive,
									  pWatch->Watch->NotifyFilter, NULL, &pWatch->Overlapped,
										 NULL, EFSW_ReadDirectoryNotifyExtendedInformation ) != 0;
		if ( bRet ) {
			ret = RefreshResult::SucessEx;
			pWatch->Watch->Extended = true;
		}
	}

	if ( !bRet ) {
		bRet = ReadDirectoryChangesW( pWatch->Watch->DirHandle, pWatch->Watch->Buffer.data(),
									  (DWORD)pWatch->Watch->Buffer.size(), pWatch->Watch->Recursive,
									  pWatch->Watch->NotifyFilter, NULL, &pWatch->Overlapped,
									  NULL ) != 0;

		if ( bRet )
			ret = RefreshResult::Success;
	}

	if ( !bRet ) {
		std::string error = std::to_string( GetLastError() );
		Errors::Log::createLastError( Errors::WatcherFailed, error );
	}

	return ret;
}

/// Stops monitoring a directory.
void DestroyWatch( WatcherStructWin32* pWatch ) {
	if ( pWatch ) {
		WatcherWin32* tWatch = pWatch->Watch;
		tWatch->StopNow = true;
		CancelIoEx( pWatch->Watch->DirHandle, &pWatch->Overlapped );
		CloseHandle( pWatch->Watch->DirHandle );
		efSAFE_DELETE_ARRAY( pWatch->Watch->DirName );
		efSAFE_DELETE( pWatch->Watch );
		efSAFE_DELETE( pWatch );
	}
}

/// Starts monitoring a directory.
WatcherStructWin32* CreateWatch( LPCWSTR szDirectory, bool recursive, DWORD bufferSize,
								 DWORD notifyFilter, HANDLE iocp, bool preventDeletion ) {
	WatcherStructWin32* tWatch = new WatcherStructWin32();
	WatcherWin32* pWatch = new WatcherWin32(bufferSize);
	if (tWatch)
		tWatch->Watch = pWatch;

	DWORD shareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
	if ( !preventDeletion ) {
		shareMode |= FILE_SHARE_DELETE;
	}

	pWatch->DirHandle = CreateFileW( szDirectory, GENERIC_READ, shareMode, NULL, OPEN_EXISTING,
									 FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, NULL );

	if ( pWatch->DirHandle != INVALID_HANDLE_VALUE &&
		 CreateIoCompletionPort( pWatch->DirHandle, iocp, 0, 1 ) ) {
		pWatch->NotifyFilter = notifyFilter;
		pWatch->Recursive = recursive;

		if ( RefreshResult::Failed != RefreshWatch( tWatch ) ) {
			return tWatch;
		}
	}

	CloseHandle( pWatch->DirHandle );
	efSAFE_DELETE( pWatch->Watch );
	efSAFE_DELETE( tWatch );
	return NULL;
}

} // namespace efsw

#endif

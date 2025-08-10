/**************************************************************************/
/*  display_server_ios.mm                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#import "display_server_ios.h"

#import "device_metrics.h"

#import <UIKit/UIKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <sys/utsname.h>

// MARK: - GodotDocumentPickerDelegate

@interface GodotDocumentPickerDelegate : NSObject <UIDocumentPickerDelegate>

@property(nonatomic, copy) void (^completion_handler)(bool success, NSArray<NSURL *> *urls);

- (instancetype)initWithCompletionHandler:(void (^)(bool success, NSArray<NSURL *> *urls))handler;

@end

@implementation GodotDocumentPickerDelegate

// Static set to keep strong references to active delegates
static NSMutableSet<GodotDocumentPickerDelegate *> *activeDelegates = nil;

+ (void)initialize {
	if (self == [GodotDocumentPickerDelegate class]) {
		activeDelegates = [[NSMutableSet alloc] init];
	}
}

- (instancetype)initWithCompletionHandler:(void (^)(bool success, NSArray<NSURL *> *urls))handler {
	self = [super init];
	if (self) {
		self.completion_handler = handler;
		// Add to active delegates to keep strong reference
		@synchronized(activeDelegates) {
			[activeDelegates addObject:self];
		}
	}
	return self;
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
	if (self.completion_handler) {
		self.completion_handler(true, urls);
	}
	[self cleanup];
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller {
	if (self.completion_handler) {
		self.completion_handler(false, @[]);
	}
	[self cleanup];
}

- (void)cleanup {
	// Remove from active delegates to allow deallocation
	@synchronized(activeDelegates) {
		[activeDelegates removeObject:self];
	}
	self.completion_handler = nil;
}

@end

// MARK: - Security-Scoped URL Management

@interface GodotSecurityScopedURLManager : NSObject

+ (instancetype)sharedManager;
- (void)addSecurityScopedURL:(NSURL *)url forPath:(NSString *)path;
- (void)removeSecurityScopedURLForPath:(NSString *)path;
- (void)removeAllSecurityScopedURLs;

@end

@implementation GodotSecurityScopedURLManager

static GodotSecurityScopedURLManager *sharedInstance = nil;
static NSMutableDictionary<NSString *, NSURL *> *securityScopedURLs = nil;

+ (instancetype)sharedManager {
	static dispatch_once_t onceToken;
	dispatch_once(&onceToken, ^{
		sharedInstance = [[GodotSecurityScopedURLManager alloc] init];
		securityScopedURLs = [[NSMutableDictionary alloc] init];
	});
	return sharedInstance;
}

- (void)addSecurityScopedURL:(NSURL *)url forPath:(NSString *)path {
	@synchronized(securityScopedURLs) {
		// Remove any existing URL for this path first
		NSURL *existingURL = securityScopedURLs[path];
		if (existingURL) {
			[existingURL stopAccessingSecurityScopedResource];
		}

		// Store the new URL
		securityScopedURLs[path] = url;
		NSLog(@"Added security-scoped URL for path: %@", path);
	}
}

- (void)removeSecurityScopedURLForPath:(NSString *)path {
	@synchronized(securityScopedURLs) {
		NSURL *url = securityScopedURLs[path];
		if (url) {
			[url stopAccessingSecurityScopedResource];
			[securityScopedURLs removeObjectForKey:path];
			NSLog(@"Removed security-scoped URL for path: %@", path);
		}
	}
}

- (void)removeAllSecurityScopedURLs {
	@synchronized(securityScopedURLs) {
		for (NSURL *url in securityScopedURLs.allValues) {
			[url stopAccessingSecurityScopedResource];
		}
		[securityScopedURLs removeAllObjects];
		NSLog(@"Removed all security-scoped URLs");
	}
}

@end

DisplayServerIOS *DisplayServerIOS::get_singleton() {
	return (DisplayServerIOS *)DisplayServerAppleEmbedded::get_singleton();
}

DisplayServerIOS::DisplayServerIOS(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) :
		DisplayServerAppleEmbedded(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error) {
}

DisplayServerIOS::~DisplayServerIOS() {
}

DisplayServer *DisplayServerIOS::create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	return memnew(DisplayServerIOS(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error));
}

void DisplayServerIOS::register_ios_driver() {
	register_create_function("iOS", create_func, get_rendering_drivers_func);
}

String DisplayServerIOS::get_name() const {
	return "iOS";
}

int DisplayServerIOS::screen_get_dpi(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 72);

	struct utsname systemInfo;
	uname(&systemInfo);

	NSString *string = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];

	NSDictionary *iOSModelToDPI = [GDTDeviceMetrics dpiList];

	for (NSArray *keyArray in iOSModelToDPI) {
		if ([keyArray containsObject:string]) {
			NSNumber *value = iOSModelToDPI[keyArray];
			return [value intValue];
		}
	}

	// If device wasn't found in dictionary
	// make a best guess from device metrics.
	CGFloat scale = [UIScreen mainScreen].scale;

	UIUserInterfaceIdiom idiom = [UIDevice currentDevice].userInterfaceIdiom;

	switch (idiom) {
		case UIUserInterfaceIdiomPad:
			return scale == 2 ? 264 : 132;
		case UIUserInterfaceIdiomPhone: {
			if (scale == 3) {
				CGFloat nativeScale = [UIScreen mainScreen].nativeScale;
				return nativeScale == 3 ? 458 : 401;
			}

			return 326;
		}
		default:
			return 72;
	}
}

float DisplayServerIOS::screen_get_refresh_rate(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, SCREEN_REFRESH_RATE_FALLBACK);

	float fps = [UIScreen mainScreen].maximumFramesPerSecond;
	if ([NSProcessInfo processInfo].lowPowerModeEnabled) {
		fps = 60;
	}
	return fps;
}

float DisplayServerIOS::screen_get_scale(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 1.0f);

	return [UIScreen mainScreen].scale;
}

bool DisplayServerIOS::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_NATIVE_DIALOG:
		case FEATURE_NATIVE_DIALOG_FILE:
		case FEATURE_NATIVE_DIALOG_FILE_EXTRA:
			return true;
		default:
			return DisplayServerAppleEmbedded::has_feature(p_feature);
	}
}

Error DisplayServerIOS::file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback, WindowID p_window_id) {
	return file_dialog_with_options_show(p_title, p_current_directory, String(), p_filename, p_show_hidden, p_mode, p_filters, TypedArray<Dictionary>(), p_callback, p_window_id);
}

Error DisplayServerIOS::file_dialog_with_options_show(const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, bool p_show_hidden, FileDialogMode p_mode, const Vector<String> &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback, WindowID p_window_id) {
	ERR_FAIL_INDEX_V(int(p_mode), FILE_DIALOG_MODE_SAVE_MAX, FAILED);

	// Get the root view controller
	UIViewController *rootViewController = nil;
	UIWindow *keyWindow = nil;

	// Find the key window
	for (UIWindow *window in [UIApplication sharedApplication].windows) {
		if (window.isKeyWindow) {
			keyWindow = window;
			break;
		}
	}

	if (!keyWindow) {
		// Fallback for iOS 13+
		if (@available(iOS 13.0, *)) {
			NSSet<UIScene *> *connectedScenes = [UIApplication sharedApplication].connectedScenes;
			for (UIScene *scene in connectedScenes) {
				if ([scene isKindOfClass:[UIWindowScene class]]) {
					UIWindowScene *windowScene = (UIWindowScene *)scene;
					for (UIWindow *window in windowScene.windows) {
						if (window.isKeyWindow) {
							keyWindow = window;
							break;
						}
					}
					if (keyWindow) break;
				}
			}
		}
	}

	rootViewController = keyWindow.rootViewController;
	ERR_FAIL_NULL_V(rootViewController, FAILED);

	// Convert filters to UTTypes (iOS 14+) or legacy type strings
	NSMutableArray *allowedTypes = [NSMutableArray array];
	NSMutableArray<NSString *> *legacyTypes = [NSMutableArray array];

	if (p_filters.size() > 0) {
		for (int i = 0; i < p_filters.size(); i++) {
			String filter = p_filters[i];

			// Parse filter format: "*.ext1,*.ext2;Description;mime/type"
			Vector<String> parts = filter.split(";");
			if (parts.size() > 0) {
				String extensions_part = parts[0];
				Vector<String> extensions = extensions_part.split(",");

				for (int j = 0; j < extensions.size(); j++) {
					String ext = extensions[j].strip_edges();
					if (ext.begins_with("*.")) {
						ext = ext.substr(2); // Remove "*."
					}

					if (ext.length() > 0) {
						NSString *nsExt = [NSString stringWithUTF8String:ext.utf8().get_data()];
						if (@available(iOS 14.0, *)) {
							UTType *type = [UTType typeWithFilenameExtension:nsExt];
							if (type) {
								[allowedTypes addObject:type];
							}
						} else {
							// Fallback for older iOS versions - use common UTI strings
							NSString *utiString = nil;
							if ([nsExt isEqualToString:@"txt"]) {
								utiString = @"public.plain-text";
							} else if ([nsExt isEqualToString:@"pdf"]) {
								utiString = @"com.adobe.pdf";
							} else if ([nsExt isEqualToString:@"png"]) {
								utiString = @"public.png";
							} else if ([nsExt isEqualToString:@"jpg"] || [nsExt isEqualToString:@"jpeg"]) {
								utiString = @"public.jpeg";
							} else {
								utiString = @"public.data";
							}
							[legacyTypes addObject:utiString];
						}
					}
				}
			}
		}
	}

	// If no specific types, allow all documents
	if (allowedTypes.count == 0 && legacyTypes.count == 0) {
		if (@available(iOS 14.0, *)) {
			if (p_mode == FILE_DIALOG_MODE_OPEN_DIR) {
				[allowedTypes addObject:UTTypeFolder];
			} else {
				[allowedTypes addObject:UTTypeData];
				[allowedTypes addObject:UTTypeContent];
			}
		} else {
			if (p_mode == FILE_DIALOG_MODE_OPEN_DIR) {
				[legacyTypes addObject:@"public.folder"];
			} else {
				[legacyTypes addObject:@"public.data"];
			}
		}
	}

	// Create document picker based on mode
	UIDocumentPickerViewController *documentPicker = nil;
	NSURL *tempFileURL = nil;

	if (p_mode == FILE_DIALOG_MODE_SAVE_FILE) {
		// For save mode, we need to return a path where the app can write
		// iOS doesn't have a traditional "save dialog" - we use document picker for location selection
		// The app will need to handle the actual file writing after getting the path

		// Create a placeholder filename if none provided
		NSString *filename = nil;
		NSCharacterSet *invalidChars = [NSCharacterSet characterSetWithCharactersInString:@"/:?<>\\*|\""];

		if (p_filename.length() > 0) {
			filename = [NSString stringWithUTF8String:p_filename.utf8().get_data()];
			filename = [[filename componentsSeparatedByCharactersInSet:invalidChars] componentsJoinedByString:@"_"];
			if (filename.length == 0) {
				filename = @"untitled";
			}
		} else {
			filename = @"untitled";
			// Add extension from filters if available
			if (p_filters.size() > 0) {
				String filter = p_filters[0];
				Vector<String> parts = filter.split(";");
				if (parts.size() > 0) {
					String extensions_part = parts[0];
					Vector<String> extensions = extensions_part.split(",");
					if (extensions.size() > 0) {
						String ext = extensions[0].strip_edges();
						if (ext.begins_with("*.")) {
							ext = ext.substr(2);
							NSString *nsExt = [NSString stringWithUTF8String:ext.utf8().get_data()];
							nsExt = [[nsExt componentsSeparatedByCharactersInSet:invalidChars] componentsJoinedByString:@""];
							if (nsExt.length > 0) {
								filename = [filename stringByAppendingFormat:@".%@", nsExt];
							}
						}
					}
				}
			}
		}

		// Create temporary file that will be used for export
		NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
		if (paths.count == 0) {
			ERR_PRINT("Failed to get documents directory");
			return FAILED;
		}

		NSString *documentsDirectory = [paths objectAtIndex:0];
		NSFileManager *fileManager = [NSFileManager defaultManager];

		// Ensure the documents directory exists and is writable
		if (![fileManager fileExistsAtPath:documentsDirectory]) {
			NSError *createError = nil;
			BOOL created = [fileManager createDirectoryAtPath:documentsDirectory
											withIntermediateDirectories:YES
													attributes:nil
														 error:&createError];
			if (!created) {
				ERR_PRINT(vformat("Failed to create documents directory: %s", [createError.localizedDescription UTF8String]));
				return FAILED;
			}
		}

		// Check if documents directory is writable
		if (![fileManager isWritableFileAtPath:documentsDirectory]) {
			ERR_PRINT(vformat("Documents directory is not writable: %s", [documentsDirectory UTF8String]));
			return FAILED;
		}

		// Create unique filename to avoid conflicts
		NSString *baseFilename = [filename stringByDeletingPathExtension];
		NSString *extension = [filename pathExtension];
		NSString *uniqueFilename = filename;
		int counter = 1;

		NSString *tempFilePath = [documentsDirectory stringByAppendingPathComponent:uniqueFilename];
		while ([fileManager fileExistsAtPath:tempFilePath]) {
			if (extension.length > 0) {
				uniqueFilename = [NSString stringWithFormat:@"%@_%d.%@", baseFilename, counter, extension];
			} else {
				uniqueFilename = [NSString stringWithFormat:@"%@_%d", baseFilename, counter];
			}
			tempFilePath = [documentsDirectory stringByAppendingPathComponent:uniqueFilename];
			counter++;

			if (counter > 1000) {
				ERR_PRINT("Failed to create unique temporary filename");
				return FAILED;
			}
		}

		tempFileURL = [NSURL fileURLWithPath:tempFilePath];

		// Create the temporary file with minimal placeholder content
		// The app should write actual content to this file before the dialog is shown
		NSData *placeholderData = [@"" dataUsingEncoding:NSUTF8StringEncoding];
		NSError *writeError = nil;
		BOOL writeSuccess = [placeholderData writeToURL:tempFileURL options:NSDataWritingAtomic error:&writeError];
		if (!writeSuccess) {
			ERR_PRINT(vformat("Failed to create temporary file at %s: %s", [tempFileURL.path UTF8String], [writeError.localizedDescription UTF8String]));
			return FAILED;
		}

		// Verify the file was created successfully
		if (![fileManager fileExistsAtPath:tempFilePath]) {
			ERR_PRINT("Temporary file creation verification failed");
			return FAILED;
		}

		// Log the temporary file path for debugging
		print_line(vformat("iOS Save Dialog: Temporary file created at %s", [tempFileURL.path UTF8String]));
		print_line("iOS Save Dialog: Write your content to this file before the dialog appears");

		// Give the app a brief moment to write content to the temporary file
		// This is a workaround for the iOS save dialog workflow
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.1 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
			// Check if the file has content now
			NSError *readError = nil;
			NSData *fileData = [NSData dataWithContentsOfURL:tempFileURL options:0 error:&readError];
			if (fileData && fileData.length > 0) {
				NSLog(@"iOS Save Dialog: File has content (%lu bytes), proceeding with save dialog", (unsigned long)fileData.length);
			} else {
				NSLog(@"iOS Save Dialog: Warning - File appears to be empty, save dialog may not work as expected");
			}
		});

		// Use document picker for exporting the file
		if (@available(iOS 14.0, *)) {
			documentPicker = [[UIDocumentPickerViewController alloc] initForExportingURLs:@[tempFileURL] asCopy:YES];
		} else {
			NSArray<NSString *> *saveTypes = legacyTypes.count > 0 ? legacyTypes : @[@"public.data"];
			documentPicker = [[UIDocumentPickerViewController alloc] initWithDocumentTypes:saveTypes inMode:UIDocumentPickerModeExportToService];
		}
	} else {
		// For open modes
		if (@available(iOS 14.0, *)) {
			documentPicker = [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:allowedTypes];
			documentPicker.allowsMultipleSelection = (p_mode == FILE_DIALOG_MODE_OPEN_FILES);
		} else {
			// Fallback for older iOS versions
			NSArray<NSString *> *openTypes = legacyTypes.count > 0 ? legacyTypes : @[@"public.data"];
			documentPicker = [[UIDocumentPickerViewController alloc] initWithDocumentTypes:openTypes inMode:UIDocumentPickerModeOpen];
			documentPicker.allowsMultipleSelection = (p_mode == FILE_DIALOG_MODE_OPEN_FILES);
		}
	}

	ERR_FAIL_NULL_V(documentPicker, FAILED);

	// Create delegate with completion handler
	Callable callback = p_callback; // Make a copy for the block
	FileDialogMode mode = p_mode; // Copy mode for the block
	GodotDocumentPickerDelegate *delegate = [[GodotDocumentPickerDelegate alloc] initWithCompletionHandler:^(bool success, NSArray<NSURL *> *urls) {
		Vector<String> selected_files;

		if (success && urls.count > 0) {
			for (NSURL *url in urls) {
				// Validate URL before processing
				if (!url || ![url isFileURL]) {
					NSLog(@"Warning: Invalid or non-file URL received from document picker");
					continue;
				}

				// Start accessing security-scoped resource
				BOOL hasAccess = [url startAccessingSecurityScopedResource];
				if (!hasAccess) {
					NSLog(@"Warning: Failed to gain access to security-scoped resource: %@", url.path);
					continue;
				}

				// Validate file access and existence
				NSError *accessError = nil;
				BOOL isAccessible = [url checkResourceIsReachableAndReturnError:&accessError];
				if (!isAccessible) {
					NSLog(@"Warning: File is not accessible: %@ (Error: %@)", url.path, accessError.localizedDescription);
					[url stopAccessingSecurityScopedResource];
					continue;
				}

				// Check file permissions based on dialog mode
				NSFileManager *fileManager = [NSFileManager defaultManager];
				NSString *filePath = [url path];

				if (mode == FILE_DIALOG_MODE_SAVE_FILE) {
					// For save mode, the URL represents the final save location
					// With security-scoped URLs, we need to check access differently

					// First, try to access the URL directly (this should work with security-scoped URLs)
					NSError *coordinatedError = nil;
					__block BOOL canWrite = NO;

					NSFileCoordinator *coordinator = [[NSFileCoordinator alloc] init];
					[coordinator coordinateWritingItemAtURL:url
													options:NSFileCoordinatorWritingForReplacing
													  error:&coordinatedError
												 byAccessor:^(NSURL *writingURL) {
						// If we can coordinate writing, we have permission
						canWrite = YES;
					}];

					if (!canWrite || coordinatedError) {
						// Fallback: Check parent directory permissions
						NSString *parentDirectory = [filePath stringByDeletingLastPathComponent];
						if (![fileManager isWritableFileAtPath:parentDirectory]) {
							NSLog(@"Warning: No write permission for directory: %@ (Error: %@)", parentDirectory, coordinatedError.localizedDescription);
							[url stopAccessingSecurityScopedResource];
							continue;
						}
					}
				} else {
					// For open modes, check if file is readable
					if (![fileManager isReadableFileAtPath:filePath]) {
						NSLog(@"Warning: No read permission for file: %@", filePath);
						[url stopAccessingSecurityScopedResource];
						continue;
					}

					// Additional validation for directory mode
					if (mode == FILE_DIALOG_MODE_OPEN_DIR) {
						BOOL isDirectory = NO;
						BOOL exists = [fileManager fileExistsAtPath:filePath isDirectory:&isDirectory];
						if (!exists || !isDirectory) {
							NSLog(@"Warning: Selected path is not a valid directory: %@", filePath);
							[url stopAccessingSecurityScopedResource];
							continue;
						}
					}
				}

				// Validate file size for reasonable limits (optional safety check)
				if (mode != FILE_DIALOG_MODE_OPEN_DIR && mode != FILE_DIALOG_MODE_SAVE_FILE) {
					NSDictionary *fileAttributes = [fileManager attributesOfItemAtPath:filePath error:nil];
					if (fileAttributes) {
						NSNumber *fileSize = fileAttributes[NSFileSize];
						// Check for extremely large files (>2GB) as a safety measure
						if (fileSize && [fileSize longLongValue] > 2147483648LL) {
							NSLog(@"Warning: File size exceeds reasonable limit: %@ (%@ bytes)", filePath, fileSize);
							// Still allow but log the warning
						}
					}
				}

				// Validate file path encoding
				String file_path;
				const char *utf8_path = [filePath UTF8String];
				if (!utf8_path) {
					NSLog(@"Warning: Failed to convert file path to UTF-8: %@", filePath);
					[url stopAccessingSecurityScopedResource];
					continue;
				}

				file_path.append_utf8(utf8_path);

				// Additional validation: ensure the path is not empty and is absolute
				if (file_path.is_empty() || !file_path.is_absolute_path()) {
					NSLog(@"Warning: Invalid file path format: %s", utf8_path);
					[url stopAccessingSecurityScopedResource];
					continue;
				}

				selected_files.push_back(file_path);

				// Store the security-scoped URL for later cleanup
				// This maintains access until the app is done using the file
				[[GodotSecurityScopedURLManager sharedManager] addSecurityScopedURL:url forPath:filePath];
			}
		}

		// Clean up temporary file if it was created for save mode
		if (tempFileURL && mode == FILE_DIALOG_MODE_SAVE_FILE) {
			// Validate the temporary file URL before deletion
			if ([tempFileURL isFileURL]) {
				NSFileManager *fileManager = [NSFileManager defaultManager];
				NSString *tempFilePath = [tempFileURL path];

				// Verify the file is within our documents directory for security
				NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
				if (paths.count > 0) {
					NSString *documentsDirectory = [paths objectAtIndex:0];
					if ([tempFilePath hasPrefix:documentsDirectory]) {
						// Check if file exists before attempting deletion
						if ([fileManager fileExistsAtPath:tempFilePath]) {
							NSError *error = nil;
							BOOL deleteSuccess = [fileManager removeItemAtURL:tempFileURL error:&error];
							if (!deleteSuccess) {
								NSLog(@"Warning: Failed to clean up temporary file: %@ (Error: %@)", tempFilePath, error.localizedDescription);
							}
						}
					} else {
						NSLog(@"Warning: Temporary file path is outside documents directory, skipping deletion for security: %@", tempFilePath);
					}
				}
			}
		}

		// Execute callback
		if (callback.is_valid()) {
			Variant v_result = success;
			Variant v_files = selected_files;
			Variant v_index = 0; // iOS doesn't provide filter index
			Variant ret;
			Callable::CallError ce;
			const Variant *args[3] = { &v_result, &v_files, &v_index };

			callback.callp(args, 3, ret, ce);
			if (ce.error != Callable::CallError::CALL_OK) {
				ERR_PRINT(vformat("Failed to execute file dialog callback: %s.", Variant::get_callable_error_text(callback, args, 3, ce)));
			}
		}
	}];

	documentPicker.delegate = delegate;

	// Dismiss any active text input to prevent RTI warnings
	[rootViewController.view endEditing:YES];

	// Present the document picker
	[rootViewController presentViewController:documentPicker animated:YES completion:^{
		// Add a safety timeout to clean up the delegate if something goes wrong
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(30.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
			[delegate cleanup];
		});
	}];

	return OK;
}

// iOS-specific methods for security-scoped URL management
void DisplayServerIOS::release_file_access(const String &p_file_path) {
	NSString *nsPath = [NSString stringWithUTF8String:p_file_path.utf8().get_data()];
	[[GodotSecurityScopedURLManager sharedManager] removeSecurityScopedURLForPath:nsPath];
}

void DisplayServerIOS::release_all_file_access() {
	[[GodotSecurityScopedURLManager sharedManager] removeAllSecurityScopedURLs];
}

// iOS-specific method to get temporary file path for save operations
String DisplayServerIOS::get_temp_file_path_for_save(const String &p_filename) {
	// Get the app's documents directory
	NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
	if (paths.count == 0) {
		ERR_PRINT("Failed to get documents directory for temporary file");
		return String();
	}

	NSString *documentsDirectory = [paths objectAtIndex:0];

	// Sanitize the filename
	NSString *nsFilename = [NSString stringWithUTF8String:p_filename.utf8().get_data()];
	NSCharacterSet *invalidChars = [NSCharacterSet characterSetWithCharactersInString:@"/:?<>\\*|\""];
	NSString *sanitizedFilename = [[nsFilename componentsSeparatedByCharactersInSet:invalidChars] componentsJoinedByString:@"_"];

	// Create a unique filename to avoid conflicts
	NSString *baseFilename = [sanitizedFilename stringByDeletingPathExtension];
	NSString *extension = [sanitizedFilename pathExtension];
	NSString *uniqueFilename = sanitizedFilename;

	NSFileManager *fileManager = [NSFileManager defaultManager];
	NSString *tempFilePath = [documentsDirectory stringByAppendingPathComponent:uniqueFilename];

	// Find a unique filename if the file already exists
	int counter = 1;
	while ([fileManager fileExistsAtPath:tempFilePath] && counter <= 1000) {
		if (extension.length > 0) {
			uniqueFilename = [NSString stringWithFormat:@"%@_%d.%@", baseFilename, counter, extension];
		} else {
			uniqueFilename = [NSString stringWithFormat:@"%@_%d", baseFilename, counter];
		}
		tempFilePath = [documentsDirectory stringByAppendingPathComponent:uniqueFilename];
		counter++;
	}

	if (counter > 1000) {
		ERR_PRINT("Failed to create unique temporary filename");
		return String();
	}

	// Convert to Godot String
	String result;
	const char *utf8_path = [tempFilePath UTF8String];
	if (utf8_path) {
		result.append_utf8(utf8_path);
	}

	return result;
}

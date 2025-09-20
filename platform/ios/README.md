# iOS platform port

This folder contains the C++, Objective-C and Objective-C++ code for the iOS
platform port.

This platform derives from the Apple embedded abstract platform ([`drivers/apple_embedded`](/drivers/apple_embedded)).

This platform uses shared Apple code ([`drivers/apple`](/drivers/apple)).

See also [`misc/dist/apple_embedded_xcode`](/misc/dist/apple_embedded_xcode) folder for the Xcode
project template used for packaging the iOS export templates.

## Native File Dialog Implementation

The iOS platform port includes a comprehensive native file dialog implementation using `UIDocumentPickerViewController`, providing seamless integration with iOS's document management system.

### Features

#### Core Functionality
- **Native iOS Integration**: Uses `UIDocumentPickerViewController` for authentic iOS user experience
- **Multiple Dialog Modes**: Supports open file, open multiple files, open directory, and save file operations
- **File Type Filtering**: Advanced filter support with file extensions, MIME types, and UTI (Uniform Type Identifiers)
- **iOS Version Compatibility**: Supports iOS 11+ with optimized features for iOS 14+ using modern `UTType` APIs

#### Security & Permissions
- **Security-Scoped URLs**: Automatic management of iOS security-scoped resource access
- **Sandboxing Compliance**: Full compliance with iOS app sandboxing requirements
- **Permission Validation**: Comprehensive read/write permission checking with fallback mechanisms
- **Safe File Access**: Coordinated file operations using `NSFileCoordinator` for thread-safe access

#### Save Dialog Workflow
- **Temporary File Management**: Intelligent temporary file creation in app's documents directory
- **Unique Filename Generation**: Automatic conflict resolution with incremental naming
- **Export-Based Saving**: Uses iOS document export pattern for user-controlled file placement
- **Content Validation**: Pre-export content verification and size limit checking

#### Advanced Features
- **Multi-File Selection**: Support for selecting multiple files in open modes
- **Directory Selection**: Native folder picker integration for directory-based operations
- **File Size Validation**: Safety checks for large files (>2GB warning system)
- **Path Sanitization**: Automatic filename sanitization for iOS filesystem compatibility
- **UTF-8 Encoding**: Robust Unicode filename support with encoding validation

#### Memory Management
- **Automatic Cleanup**: Smart delegate lifecycle management with timeout protection
- **Resource Deallocation**: Proper cleanup of security-scoped URLs and temporary files
- **Memory Safety**: Strong reference management for async operations

### Technical Implementation

#### Key Components
- **GodotDocumentPickerDelegate**: Custom delegate handling document picker callbacks
- **GodotSecurityScopedURLManager**: Centralized management of security-scoped URL access
- **Temporary File System**: Secure temporary file creation and cleanup in documents directory

#### API Methods
- `file_dialog_show()`: Standard file dialog interface
- `file_dialog_with_options_show()`: Extended dialog with additional options
- `get_temp_file_path_for_save()`: iOS-specific temporary file path generation
- `release_file_access()`: Manual security-scoped URL cleanup
- `release_all_file_access()`: Bulk cleanup of all active file access permissions

#### iOS-Specific Considerations
- **Document Picker Modes**: Automatic mode selection based on dialog type (open/export)
- **UTType Integration**: Modern iOS 14+ type system with legacy fallback support
- **View Controller Management**: Proper presentation from root view controller
- **Text Input Dismissal**: Automatic keyboard dismissal to prevent RTI warnings

### Usage Examples

#### Basic File Opening
```gdscript
DisplayServer.file_dialog_show(
    "Open File",
    "",
    "document.txt",
    false,
    DisplayServer.FILE_DIALOG_MODE_OPEN_FILE,
    ["*.txt", "*.pdf"],
    _on_file_selected
)
```

#### Save File with Temporary Workflow
```gdscript
# Step 1: Get temporary file path
var temp_path = DisplayServer.get_temp_file_path_for_save("my_file.txt")

# Step 2: Write content to temporary file
var file = FileAccess.open(temp_path, FileAccess.WRITE)
file.store_string("File content")
file.close()

# Step 3: Show save dialog
DisplayServer.file_dialog_show(
    "Save File",
    "",
    "my_file.txt",
    false,
    DisplayServer.FILE_DIALOG_MODE_SAVE_FILE,
    ["*.txt"],
    _on_file_saved
)
```

### Compatibility & Requirements

- **Minimum iOS Version**: iOS 11.0+
- **Optimized for**: iOS 14.0+ (UTType support)
- **Required Capabilities**: File system access, document picker framework
- **Export Settings**: "Accessible From Files App" recommended for enhanced functionality

This implementation provides a robust, secure, and user-friendly file dialog system that seamlessly integrates with iOS's native document management while maintaining compatibility with Godot's cross-platform file dialog API.

## Documentation

- [Compiling for iOS](https://docs.godotengine.org/en/latest/engine_details/development/compiling/compiling_for_ios.html)
  - Instructions on building this platform port from source.
- [Exporting for iOS](https://docs.godotengine.org/en/latest/tutorials/export/exporting_for_ios.html)
  - Instructions on using the compiled export templates to export a project.

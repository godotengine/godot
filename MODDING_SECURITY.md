# Modding Security Implementation

This document describes the security restrictions implemented in this Godot fork for safe modding.

## Overview

This fork has been hardened to prevent modders from performing dangerous operations while still allowing them to create mods using GDScript. The security model is designed to prevent malicious or untrusted mod code from compromising the host system.

## Security Requirements

**What Modders CAN Do:**
- ✅ Read files from `user://mods/` directory only (for mod dependencies)
- ✅ Execute GDScript code with access to safe game APIs
- ✅ Access game logic, scenes, and resources
- ✅ Use math, string manipulation, and basic Godot features

**What Modders CANNOT Do:**
- ❌ Write or create ANY files
- ❌ Delete files
- ❌ List or read directories
- ❌ Read files outside `user://mods/`
- ❌ Execute OS commands or spawn processes
- ❌ Access environment variables
- ❌ Make network requests (HTTP, TCP, UDP, WebSocket)
- ❌ Load native code (GDExtension/DLLs)
- ❌ Modify file attributes or permissions

## Implementation Details

### 1. File System Restrictions

**Modified Files:**
- `core/io/file_access.cpp` - Added path validation and write blocking
- `core/io/modding_security.h` - Security validation module
- `core/io/dir_access.cpp` - Disabled ALL directory operations

**Restrictions:**
- `FileAccess.open()` only allows READ mode for paths starting with `user://mods/`
- All write methods (`store_*`, `resize`, `flush`, etc.) are disabled in GDScript bindings
- File attribute modification methods are disabled
- Static helper methods (`get_file_as_bytes`, etc.) validate paths
- `DirAccess` class is completely disabled - no directory listing or manipulation

### 2. OS Command Restrictions

**Modified Files:**
- `core/core_bind.cpp` - Disabled OS command execution methods

**Disabled Methods:**
- `OS.execute()`
- `OS.execute_with_pipe()`
- `OS.create_process()`
- `OS.create_instance()`
- `OS.open_with_program()`
- `OS.kill()`
- `OS.shell_open()`
- `OS.shell_show_in_file_manager()`
- `OS.has_environment()` / `OS.get_environment()` / `OS.set_environment()`

### 3. Network Restrictions

**Modified Files:**
- `core/io/http_client.cpp` - Disabled HTTPClient
- `scene/main/http_request.cpp` - Disabled HTTPRequest
- `core/io/tcp_server.cpp` - Disabled TCPServer
- `core/io/udp_server.cpp` - Disabled UDPServer
- `core/io/stream_peer_tcp.cpp` - Disabled StreamPeerTCP
- `modules/websocket/websocket_peer.cpp` - Disabled WebSocketPeer

**Restrictions:**
- NO network access of any kind
- All HTTP, TCP, UDP, and WebSocket functionality is disabled

### 4. Native Code Loading Restrictions

**Modified Files:**
- `core/extension/gdextension_manager.cpp` - Disabled GDExtension loading

**Restrictions:**
- `GDExtensionManager.load_extension()` and all related methods are disabled
- This is the **MOST CRITICAL** restriction as native code can bypass all other security

### 5. C# Scripting

C# scripting can be disabled at compile time by not including the mono module. GDScript is the only supported modding language.

## Building the Secure Fork

To build this fork:

```bash
# Standard Godot build (adjust for your platform)
scons platform=linux target=template_release

# Or for Windows:
scons platform=windows target=template_release

# The resulting binary will have all security restrictions enabled
```

## Testing Security

To verify the security restrictions are working:

1. **Test File Access:**
   ```gdscript
   # Should FAIL - writing is blocked
   var file = FileAccess.open("user://test.txt", FileAccess.WRITE)

   # Should FAIL - reading outside user://mods/
   var file = FileAccess.open("res://project.godot", FileAccess.READ)

   # Should SUCCEED - reading from user://mods/
   var file = FileAccess.open("user://mods/dependency.txt", FileAccess.READ)
   ```

2. **Test Directory Access:**
   ```gdscript
   # Should FAIL - DirAccess is completely disabled
   var dir = DirAccess.open("user://")
   ```

3. **Test OS Commands:**
   ```gdscript
   # Should FAIL - method doesn't exist in bindings
   OS.execute("ls", [])
   ```

4. **Test Network Access:**
   ```gdscript
   # Should FAIL - HTTPClient is disabled
   var http = HTTPClient.new()
   ```

5. **Test Native Code:**
   ```gdscript
   # Should FAIL - GDExtension loading is disabled
   GDExtensionManager.load_extension("path/to/extension.so")
   ```

## Security Model

This implementation uses a **whitelist-based security model**:
- By default, everything dangerous is blocked
- Only explicitly safe operations are allowed
- Path validation occurs at multiple layers
- Critical restrictions cannot be bypassed from GDScript

## Attack Surface Reduction

The following attack vectors have been eliminated:
- ✅ File system manipulation
- ✅ Command injection / OS command execution
- ✅ Data exfiltration via network
- ✅ Malware download and execution
- ✅ Native code injection
- ✅ Environment variable exploitation
- ✅ Process spawning and manipulation

## Distribution

When distributing your modding SDK:
1. Build the Godot binary with these modifications
2. Distribute only the compiled executable
3. Provide clear modding guidelines to modders
4. Only allow mods to place files in `user://mods/` directory
5. The engine will enforce all security restrictions automatically

## Important Notes

- **Source Code:** Keep this fork's source code private. Do not distribute the modified source.
- **Binary Only:** Only distribute the compiled executable to modders.
- **No Recompilation:** Modders should not be able to recompile or modify the engine.
- **GDScript Only:** Only GDScript mods are supported. No C# or GDNative/GDExtension.
- **Testing:** Thoroughly test your game with these restrictions to ensure legitimate gameplay features still work.

## License Compliance

This is a fork of Godot Engine, which is licensed under the MIT license. You are free to use this modified version in proprietary software, but you must:
1. Include the original Godot MIT license with your distribution
2. Credit the Godot Engine contributors

## Maintenance

When updating to newer Godot versions:
1. Re-apply all security modifications
2. Review new APIs added in the Godot update for security implications
3. Test all security restrictions still function correctly
4. Update this document with any new changes

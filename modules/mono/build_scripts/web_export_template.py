#!/usr/bin/env python

import os
import shutil
import subprocess
import sys

def find_emscripten_sdk():
    """Find the Emscripten SDK directory using environment variables."""
    emsdk = os.environ.get("EMSDK")
    if emsdk:
        return emsdk
    
    # Try to find it in PATH
    path = os.environ.get("PATH", "").split(os.pathsep)
    for p in path:
        em_config = os.path.join(p, "em-config")
        if os.path.exists(em_config):
            # This is in bin, go up one level
            return os.path.dirname(p)
    
    return None

def build_mono_wasm(godot_dir, output_dir, debug=False):
    """Build the Mono WebAssembly runtime for Godot."""
    print("Building Mono WebAssembly runtime...")
    
    # Find the Emscripten SDK
    emsdk = find_emscripten_sdk()
    if not emsdk:
        print("Error: Emscripten SDK not found. Please set the EMSDK environment variable.")
        return False
    
    # Create output directory
    mono_wasm_dir = os.path.join(output_dir, "wasm")
    os.makedirs(mono_wasm_dir, exist_ok=True)

    # Set build configuration
    config = "Debug" if debug else "Release"
    
    # Download and build mono-wasm
    try:
        # Clone mono-wasm repository if needed
        mono_wasm_repo = os.path.join(godot_dir, "thirdparty", "mono-wasm")
        if not os.path.exists(mono_wasm_repo):
            os.makedirs(os.path.dirname(mono_wasm_repo), exist_ok=True)
            subprocess.check_call(["git", "clone", "https://github.com/mono/mono-wasm.git", mono_wasm_repo])
        
        # Build mono-wasm
        env = os.environ.copy()
        env["EMSDK"] = emsdk
        
        # Configure build options
        build_cmd = [
            "emcmake", "cmake", 
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DENABLE_WASM=1",
            "-DENABLE_WASM_DYNAMIC_RUNTIME=1",
            "-DENABLE_WASM_THREADS=1",
            "-DUSE_STATIC_MONO=1",
            "."
        ]
        
        subprocess.check_call(build_cmd, cwd=mono_wasm_repo, env=env)
        subprocess.check_call(["emmake", "make", "-j4"], cwd=mono_wasm_repo, env=env)
        
        # Copy the built files
        wasm_files = [
            "mono.wasm",
            "mono.js",
            "mono.worker.js"
        ]
        
        for file in wasm_files:
            src_file = os.path.join(mono_wasm_repo, file)
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(mono_wasm_dir, file))
        
        # Create wrapper JS file for initialization
        with open(os.path.join(mono_wasm_dir, "godot-mono-support.js"), "w") as f:
            f.write("""
// Godot Mono WebAssembly Support
var GodotMonoSupport = {
    // Initialize the Mono runtime
    init: function() {
        // This will be called when Godot starts
        console.log("Initializing Mono runtime for WebAssembly...");
        
        // TODO: Add initialization code here
        return true;
    },
    
    // Load assemblies
    loadAssemblies: function(assemblies) {
        // Called to load .NET assemblies
        console.log("Loading .NET assemblies: " + assemblies);
    }
};
""")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building Mono WASM: {e}")
        return False

def prepare_export_template(godot_dir, output_dir, debug=False):
    """Prepare the C# web export template."""
    template_type = "debug" if debug else "release"
    print(f"Preparing {template_type} export template...")
    
    # Copy the Godot web export template
    godot_template_dir = os.path.join(godot_dir, "bin", f"godot.web.{template_type}.zip")
    if not os.path.exists(godot_template_dir):
        print(f"Error: Godot web template not found at {godot_template_dir}")
        print("Did you build Godot with 'scons platform=web target=template_release module_mono_enabled=yes'?")
        return False
    
    # Copy to output
    output_template = os.path.join(output_dir, f"godot.web.{template_type}.zip")
    shutil.copy(godot_template_dir, output_template)
    print(f"Template copied to {output_template}")

    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python web_export_template.py <godot_dir> <output_dir> [debug]")
        return 1
    
    godot_dir = sys.argv[1]
    output_dir = sys.argv[2]
    debug = len(sys.argv) > 3 and sys.argv[3].lower() == "debug"
    
    if not os.path.exists(godot_dir):
        print(f"Error: Godot directory '{godot_dir}' does not exist.")
        return 1
    
    if not build_mono_wasm(godot_dir, output_dir, debug):
        return 1
    
    if not prepare_export_template(godot_dir, output_dir, debug):
        return 1
    
    print("Web export template created successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
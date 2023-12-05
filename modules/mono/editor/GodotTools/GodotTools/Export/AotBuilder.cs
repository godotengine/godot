using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using GodotTools.Internals;
using Directory = GodotTools.Utils.Directory;
using File = GodotTools.Utils.File;
using OS = GodotTools.Utils.OS;
using Path = System.IO.Path;

namespace GodotTools.Export
{
    public struct AotOptions
    {
        public bool EnableLLVM;
        public bool LLVMOnly;
        public string LLVMPath;
        public string LLVMOutputPath;

        public bool FullAot;

        private bool _useInterpreter;
        public bool UseInterpreter { readonly get => _useInterpreter && !LLVMOnly; set => _useInterpreter = value; }

        public string[] ExtraAotOptions;
        public string[] ExtraOptimizerOptions;

        public string ToolchainPath;
    }

    public static class AotBuilder
    {
        public static void CompileAssemblies(ExportPlugin exporter, AotOptions aotOpts, string[] features, string platform, bool isDebug, string bclDir, string outputDir, string outputDataDir, IDictionary<string, string> assemblies)
        {
            // TODO: WASM

            string aotTempDir = Path.Combine(Path.GetTempPath(), $"godot-aot-{Process.GetCurrentProcess().Id}");

            if (!Directory.Exists(aotTempDir))
                Directory.CreateDirectory(aotTempDir);

            var assembliesPrepared = new Dictionary<string, string>();

            foreach (var dependency in assemblies)
            {
                string assemblyName = dependency.Key;
                string assemblyPath = dependency.Value;

                string assemblyPathInBcl = Path.Combine(bclDir, assemblyName + ".dll");

                if (File.Exists(assemblyPathInBcl))
                {
                    // Don't create teporaries for assemblies from the BCL
                    assembliesPrepared.Add(assemblyName, assemblyPathInBcl);
                }
                else
                {
                    string tempAssemblyPath = Path.Combine(aotTempDir, assemblyName + ".dll");
                    File.Copy(assemblyPath, tempAssemblyPath);
                    assembliesPrepared.Add(assemblyName, tempAssemblyPath);
                }
            }

            if (platform == OS.Platforms.iOS)
            {
                string[] architectures = GetEnablediOSArchs(features).ToArray();
                CompileAssembliesForiOS(exporter, isDebug, architectures, aotOpts, aotTempDir, assembliesPrepared, bclDir);
            }
            else if (platform == OS.Platforms.Android)
            {
                string[] abis = GetEnabledAndroidAbis(features).ToArray();
                CompileAssembliesForAndroid(exporter, isDebug, abis, aotOpts, aotTempDir, assembliesPrepared, bclDir);
            }
            else
            {
                string arch = "";
                if (features.Contains("x86_64"))
                {
                    arch = "x86_64";
                }
                else if (features.Contains("x86_32"))
                {
                    arch = "x86_32";
                }
                else if (features.Contains("arm64"))
                {
                    arch = "arm64";
                }
                else if (features.Contains("arm32"))
                {
                    arch = "arm32";
                }
                CompileAssembliesForDesktop(exporter, platform, isDebug, arch, aotOpts, aotTempDir, outputDataDir, assembliesPrepared, bclDir);
            }
        }

        public static void CompileAssembliesForAndroid(ExportPlugin exporter, bool isDebug, string[] abis, AotOptions aotOpts, string aotTempDir, IDictionary<string, string> assemblies, string bclDir)
        {

            foreach (var assembly in assemblies)
            {
                string assemblyName = assembly.Key;
                string assemblyPath = assembly.Value;

                // Not sure if the 'lib' prefix is an Android thing or just Godot being picky,
                // but we use '-aot-' as well just in case to avoid conflicts with other libs.
                string outputFileName = "lib-aot-" + assemblyName + ".dll.so";

                foreach (string abi in abis)
                {
                    string aotAbiTempDir = Path.Combine(aotTempDir, abi);
                    string soFilePath = Path.Combine(aotAbiTempDir, outputFileName);

                    var compilerArgs = GetAotCompilerArgs(OS.Platforms.Android, isDebug, abi, aotOpts, assemblyPath, soFilePath);

                    // Make sure the output directory exists
                    Directory.CreateDirectory(aotAbiTempDir);

                    string compilerDirPath = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "aot-compilers", $"{OS.Platforms.Android}-{abi}");

                    ExecuteCompiler(FindCrossCompiler(compilerDirPath), compilerArgs, bclDir);

                    // The Godot exporter expects us to pass the abi in the tags parameter
                    exporter.AddSharedObject(soFilePath, tags: new[] { abi }, "");
                }
            }
        }

        public static void CompileAssembliesForDesktop(ExportPlugin exporter, string platform, bool isDebug, string arch, AotOptions aotOpts, string aotTempDir, string outputDataDir, IDictionary<string, string> assemblies, string bclDir)
        {
            foreach (var assembly in assemblies)
            {
                string assemblyName = assembly.Key;
                string assemblyPath = assembly.Value;

                string outputFileExtension = platform == OS.Platforms.Windows ? ".dll" :
                    platform == OS.Platforms.MacOS ? ".dylib" :
                    ".so";

                string outputFileName = assemblyName + ".dll" + outputFileExtension;
                string tempOutputFilePath = Path.Combine(aotTempDir, outputFileName);

                var compilerArgs = GetAotCompilerArgs(platform, isDebug, arch, aotOpts, assemblyPath, tempOutputFilePath);

                string compilerDirPath = GetMonoCrossDesktopDirName(platform, arch);

                ExecuteCompiler(FindCrossCompiler(compilerDirPath), compilerArgs, bclDir);

                if (platform == OS.Platforms.MacOS)
                {
                    exporter.AddSharedObject(tempOutputFilePath, tags: null, "");
                }
                else
                {
                    string libDir = platform == OS.Platforms.Windows ? "bin" : "lib";
                    string outputDataLibDir = Path.Combine(outputDataDir, "Mono", libDir);
                    File.Copy(tempOutputFilePath, Path.Combine(outputDataLibDir, outputFileName));
                }
            }
        }

        public static void CompileAssembliesForiOS(ExportPlugin exporter, bool isDebug, string[] architectures, AotOptions aotOpts, string aotTempDir, IDictionary<string, string> assemblies, string bclDir)
        {
            var cppCode = new StringBuilder();
            var aotModuleInfoSymbols = new List<string>(assemblies.Count);

            // {arch: paths}
            var objFilePathsForiOSArch = architectures.ToDictionary(arch => arch, arch => new List<string>(assemblies.Count));

            foreach (var assembly in assemblies)
            {
                string assemblyName = assembly.Key;
                string assemblyPath = assembly.Value;

                string asmFileName = assemblyName + ".dll.S";
                string objFileName = assemblyName + ".dll.o";

                foreach (string arch in architectures)
                {
                    string aotArchTempDir = Path.Combine(aotTempDir, arch);
                    string asmFilePath = Path.Combine(aotArchTempDir, asmFileName);

                    var compilerArgs = GetAotCompilerArgs(OS.Platforms.iOS, isDebug, arch, aotOpts, assemblyPath, asmFilePath);

                    // Make sure the output directory exists
                    Directory.CreateDirectory(aotArchTempDir);

                    string compilerDirPath = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "aot-compilers", $"{OS.Platforms.iOS}-{arch}");

                    ExecuteCompiler(FindCrossCompiler(compilerDirPath), compilerArgs, bclDir);

                    // Assembling
                    bool isSim = arch == "i386" || arch == "x86_64"; // Shouldn't really happen as we don't do AOT for the simulator
                    string versionMinName = isSim ? "iphonesimulator" : "iphoneos";
                    string iOSPlatformName = isSim ? "iPhoneSimulator" : "iPhoneOS";
                    const string versionMin = "12.0"; // TODO: Turn this hard-coded version into an exporter setting
                    string iOSSdkPath = Path.Combine(XcodeHelper.XcodePath,
                            $"Contents/Developer/Platforms/{iOSPlatformName}.platform/Developer/SDKs/{iOSPlatformName}.sdk");

                    string objFilePath = Path.Combine(aotArchTempDir, objFileName);

                    var clangArgs = new List<string>()
                    {
                        "-isysroot", iOSSdkPath,
                        "-Qunused-arguments",
                        $"-m{versionMinName}-version-min={versionMin}",
                        "-arch", arch,
                        "-c",
                        "-o", objFilePath,
                        "-x", "assembler"
                    };

                    if (isDebug)
                        clangArgs.Add("-DDEBUG");

                    clangArgs.Add(asmFilePath);

                    int clangExitCode = OS.ExecuteCommand(XcodeHelper.FindXcodeTool("clang"), clangArgs);
                    if (clangExitCode != 0)
                        throw new InvalidOperationException($"Command 'clang' exited with code: {clangExitCode}.");

                    objFilePathsForiOSArch[arch].Add(objFilePath);
                }

                aotModuleInfoSymbols.Add($"mono_aot_module_{AssemblyNameToAotSymbol(assemblyName)}_info");
            }

            // Generate driver code
            cppCode.AppendLine("#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)");
            cppCode.AppendLine("#define IOS_DEVICE");
            cppCode.AppendLine("#endif");

            cppCode.AppendLine("#ifdef IOS_DEVICE");
            cppCode.AppendLine("extern \"C\" {");
            cppCode.AppendLine("// Mono API");
            cppCode.AppendLine(@"
typedef enum {
MONO_AOT_MODE_NONE,
MONO_AOT_MODE_NORMAL,
MONO_AOT_MODE_HYBRID,
MONO_AOT_MODE_FULL,
MONO_AOT_MODE_LLVMONLY,
MONO_AOT_MODE_INTERP,
MONO_AOT_MODE_INTERP_LLVMONLY,
MONO_AOT_MODE_LLVMONLY_INTERP,
MONO_AOT_MODE_LAST = 1000,
} MonoAotMode;");
            cppCode.AppendLine("void mono_jit_set_aot_mode(MonoAotMode);");
            cppCode.AppendLine("void mono_aot_register_module(void *);");

            if (aotOpts.UseInterpreter)
            {
                cppCode.AppendLine("void mono_ee_interp_init(const char *);");
                cppCode.AppendLine("void mono_icall_table_init();");
                cppCode.AppendLine("void mono_marshal_ilgen_init();");
                cppCode.AppendLine("void mono_method_builder_ilgen_init();");
                cppCode.AppendLine("void mono_sgen_mono_ilgen_init();");
            }

            foreach (string symbol in aotModuleInfoSymbols)
                cppCode.AppendLine($"extern void *{symbol};");

            cppCode.AppendLine("void gd_mono_setup_aot() {");

            foreach (string symbol in aotModuleInfoSymbols)
                cppCode.AppendLine($"\tmono_aot_register_module({symbol});");

            if (aotOpts.UseInterpreter)
            {
                cppCode.AppendLine("\tmono_icall_table_init();");
                cppCode.AppendLine("\tmono_marshal_ilgen_init();");
                cppCode.AppendLine("\tmono_method_builder_ilgen_init();");
                cppCode.AppendLine("\tmono_sgen_mono_ilgen_init();");
                cppCode.AppendLine("\tmono_ee_interp_init(0);");
            }

            string aotModeStr = null;

            if (aotOpts.LLVMOnly)
            {
                aotModeStr = "MONO_AOT_MODE_LLVMONLY"; // --aot=llvmonly
            }
            else
            {
                if (aotOpts.UseInterpreter)
                    aotModeStr = "MONO_AOT_MODE_INTERP"; // --aot=interp or --aot=interp,full
                else if (aotOpts.FullAot)
                    aotModeStr = "MONO_AOT_MODE_FULL"; // --aot=full
            }

            // One of the options above is always set for iOS
            Debug.Assert(aotModeStr != null);

            cppCode.AppendLine($"\tmono_jit_set_aot_mode({aotModeStr});");

            cppCode.AppendLine("} // gd_mono_setup_aot");
            cppCode.AppendLine("} // extern \"C\"");
            cppCode.AppendLine("#endif // IOS_DEVICE");

            // Add the driver code to the Xcode project
            exporter.AddIosCppCode(cppCode.ToString());

            // Archive the AOT object files into a static library

            var arFilePathsForAllArchs = new List<string>();
            string projectAssemblyName = GodotSharpDirs.ProjectAssemblyName;

            foreach (var archPathsPair in objFilePathsForiOSArch)
            {
                string arch = archPathsPair.Key;
                var objFilePaths = archPathsPair.Value;

                string arOutputFilePath = Path.Combine(aotTempDir, $"lib-aot-{projectAssemblyName}.{arch}.a");

                var arArgs = new List<string>()
                {
                    "cr",
                    arOutputFilePath
                };

                foreach (string objFilePath in objFilePaths)
                    arArgs.Add(objFilePath);

                int arExitCode = OS.ExecuteCommand(XcodeHelper.FindXcodeTool("ar"), arArgs);
                if (arExitCode != 0)
                    throw new InvalidOperationException($"Command 'ar' exited with code: {arExitCode}.");

                arFilePathsForAllArchs.Add(arOutputFilePath);
            }

            // It's lipo time

            string fatOutputFileName = $"lib-aot-{projectAssemblyName}.fat.a";
            string fatOutputFilePath = Path.Combine(aotTempDir, fatOutputFileName);

            var lipoArgs = new List<string>();
            lipoArgs.Add("-create");
            lipoArgs.AddRange(arFilePathsForAllArchs);
            lipoArgs.Add("-output");
            lipoArgs.Add(fatOutputFilePath);

            int lipoExitCode = OS.ExecuteCommand(XcodeHelper.FindXcodeTool("lipo"), lipoArgs);
            if (lipoExitCode != 0)
                throw new InvalidOperationException($"Command 'lipo' exited with code: {lipoExitCode}.");

            // TODO: Add the AOT lib and interpreter libs as device only to suppress warnings when targeting the simulator

            // Add the fat AOT static library to the Xcode project
            exporter.AddIosProjectStaticLib(fatOutputFilePath);

            // Add the required Mono libraries to the Xcode project

            string MonoLibFile(string libFileName) => libFileName + ".ios.fat.a";

            string MonoLibFromTemplate(string libFileName) =>
                Path.Combine(Internal.FullExportTemplatesDir, "ios-mono-libs", MonoLibFile(libFileName));

            exporter.AddIosProjectStaticLib(MonoLibFromTemplate("libmonosgen-2.0"));

            exporter.AddIosProjectStaticLib(MonoLibFromTemplate("libmono-native"));

            if (aotOpts.UseInterpreter)
            {
                exporter.AddIosProjectStaticLib(MonoLibFromTemplate("libmono-ee-interp"));
                exporter.AddIosProjectStaticLib(MonoLibFromTemplate("libmono-icall-table"));
                exporter.AddIosProjectStaticLib(MonoLibFromTemplate("libmono-ilgen"));
            }

            // TODO: Turn into an exporter option
            bool enableProfiling = false;
            if (enableProfiling)
                exporter.AddIosProjectStaticLib(MonoLibFromTemplate("libmono-profiler-log"));

            // Add frameworks required by Mono to the Xcode project
            exporter.AddIosFramework("libiconv.tbd");
            exporter.AddIosFramework("GSS.framework");
            exporter.AddIosFramework("CFNetwork.framework");

            // Force load and export dynamic are needed for the linker to not strip required symbols.
            // In theory we shouldn't be relying on this for P/Invoked functions (as is the case with
            // functions in System.Native/libmono-native). Instead, we should use cecil to search for
            // DllImports in assemblies and pass them to 'ld' as '-u/--undefined {pinvoke_symbol}'.
            exporter.AddIosLinkerFlags("-rdynamic");
            exporter.AddIosLinkerFlags($"-force_load \"$(SRCROOT)/{MonoLibFile("libmono-native")}\"");
        }

        /// Converts an assembly name to a valid symbol name in the same way the AOT compiler does
        private static string AssemblyNameToAotSymbol(string assemblyName)
        {
            var builder = new StringBuilder();

            foreach (var charByte in Encoding.UTF8.GetBytes(assemblyName))
            {
                char @char = (char)charByte;
                builder.Append(Char.IsLetterOrDigit(@char) || @char == '_' ? @char : '_');
            }

            return builder.ToString();
        }

        private static IEnumerable<string> GetAotCompilerArgs(string platform, bool isDebug, string target, AotOptions aotOpts, string assemblyPath, string outputFilePath)
        {
            // TODO: LLVM

            bool aotSoftDebug = isDebug && !aotOpts.EnableLLVM;
            bool aotDwarfDebug = platform == OS.Platforms.iOS;

            var aotOptions = new List<string>();
            var optimizerOptions = new List<string>();

            if (aotOpts.LLVMOnly)
            {
                aotOptions.Add("llvmonly");
            }
            else
            {
                // Can be both 'interp' and 'full'
                if (aotOpts.UseInterpreter)
                    aotOptions.Add("interp");
                if (aotOpts.FullAot)
                    aotOptions.Add("full");
            }

            aotOptions.Add(aotSoftDebug ? "soft-debug" : "nodebug");

            if (aotDwarfDebug)
                aotOptions.Add("dwarfdebug");

            if (platform == OS.Platforms.Android)
            {
                string abi = target;

                string androidToolchain = aotOpts.ToolchainPath;

                if (string.IsNullOrEmpty(androidToolchain))
                {
                    androidToolchain = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "android-toolchains", $"{abi}"); // TODO: $"{abi}-{apiLevel}{(clang?"clang":"")}"

                    if (!Directory.Exists(androidToolchain))
                        throw new FileNotFoundException("Missing android toolchain. Specify one in the AOT export settings.");
                }
                else if (!Directory.Exists(androidToolchain))
                {
                    throw new FileNotFoundException($"Android toolchain not found: '{androidToolchain}'.");
                }

                var androidToolPrefixes = new Dictionary<string, string>
                {
                    ["arm32"] = "arm-linux-androideabi-",
                    ["arm64"] = "aarch64-linux-android-",
                    ["x86_32"] = "i686-linux-android-",
                    ["x86_64"] = "x86_64-linux-android-"
                };

                aotOptions.Add("tool-prefix=" + Path.Combine(androidToolchain, "bin", androidToolPrefixes[abi]));

                string triple = GetAndroidTriple(abi);
                aotOptions.Add($"mtriple={triple}");
            }
            else if (platform == OS.Platforms.iOS)
            {
                if (!aotOpts.LLVMOnly && !aotOpts.UseInterpreter)
                    optimizerOptions.Add("gsharedvt");

                aotOptions.Add("static");

                // I couldn't get the Mono cross-compiler to do assembling, so we'll have to do it ourselves
                aotOptions.Add("asmonly");

                aotOptions.Add("direct-icalls");

                if (aotSoftDebug)
                    aotOptions.Add("no-direct-calls");

                if (aotOpts.LLVMOnly || !aotOpts.UseInterpreter)
                    aotOptions.Add("direct-pinvoke");

                string arch = target;
                aotOptions.Add($"mtriple={arch}-ios");
            }

            aotOptions.Add($"outfile={outputFilePath}");

            if (aotOpts.EnableLLVM)
            {
                aotOptions.Add($"llvm-path={aotOpts.LLVMPath}");
                aotOptions.Add($"llvm-outfile={aotOpts.LLVMOutputPath}");
            }

            if (aotOpts.ExtraAotOptions.Length > 0)
                aotOptions.AddRange(aotOpts.ExtraAotOptions);

            if (aotOpts.ExtraOptimizerOptions.Length > 0)
                optimizerOptions.AddRange(aotOpts.ExtraOptimizerOptions);

            string EscapeOption(string option) => option.Contains(',') ? $"\"{option}\"" : option;
            string OptionsToString(IEnumerable<string> options) => string.Join(",", options.Select(EscapeOption));

            var runtimeArgs = new List<string>();

            // The '--debug' runtime option is required when using the 'soft-debug' and 'dwarfdebug' AOT options
            if (aotSoftDebug || aotDwarfDebug)
                runtimeArgs.Add("--debug");

            if (aotOpts.EnableLLVM)
                runtimeArgs.Add("--llvm");

            runtimeArgs.Add(aotOptions.Count > 0 ? $"--aot={OptionsToString(aotOptions)}" : "--aot");

            if (optimizerOptions.Count > 0)
                runtimeArgs.Add($"-O={OptionsToString(optimizerOptions)}");

            runtimeArgs.Add(assemblyPath);

            return runtimeArgs;
        }

        private static void ExecuteCompiler(string compiler, IEnumerable<string> compilerArgs, string bclDir)
        {
            // TODO: Once we move to .NET Standard 2.1 we can use ProcessStartInfo.ArgumentList instead
            string CmdLineArgsToString(IEnumerable<string> args)
            {
                // Not perfect, but as long as we are careful...
                return string.Join(" ", args.Select(arg => arg.Contains(" ") ? $@"""{arg}""" : arg));
            }

            using (var process = new Process())
            {
                process.StartInfo = new ProcessStartInfo(compiler, CmdLineArgsToString(compilerArgs))
                {
                    UseShellExecute = false
                };

                process.StartInfo.EnvironmentVariables.Remove("MONO_ENV_OPTIONS");
                process.StartInfo.EnvironmentVariables.Remove("MONO_THREADS_SUSPEND");
                process.StartInfo.EnvironmentVariables.Add("MONO_PATH", bclDir);

                Console.WriteLine($"Running: \"{process.StartInfo.FileName}\" {process.StartInfo.Arguments}");

                if (!process.Start())
                    throw new InvalidOperationException("Failed to start process for Mono AOT compiler.");

                process.WaitForExit();

                if (process.ExitCode != 0)
                    throw new InvalidOperationException($"Mono AOT compiler exited with code: {process.ExitCode}.");
            }
        }

        private static IEnumerable<string> GetEnablediOSArchs(string[] features)
        {
            var iosArchs = new[]
            {
                "arm64"
            };

            return iosArchs.Where(features.Contains);
        }

        private static IEnumerable<string> GetEnabledAndroidAbis(string[] features)
        {
            var androidAbis = new[]
            {
                "arm32",
                "arm64",
                "x86_32",
                "x86_64"
            };

            return androidAbis.Where(features.Contains);
        }

        private static string GetAndroidTriple(string abi)
        {
            var abiArchs = new Dictionary<string, string>
            {
                ["arm32"] = "armv7",
                ["arm64"] = "aarch64-v8a",
                ["x86_32"] = "i686",
                ["x86_64"] = "x86_64"
            };

            string arch = abiArchs[abi];

            return $"{arch}-linux-android";
        }

        private static string GetMonoCrossDesktopDirName(string platform, string arch)
        {
            switch (platform)
            {
                case OS.Platforms.Windows:
                {
                    return $"windows-{arch}";
                }
                case OS.Platforms.MacOS:
                {
                    return $"{platform}-{arch}";
                }
                case OS.Platforms.LinuxBSD:
                {
                    return $"linux-{arch}";
                }
                default:
                    throw new NotSupportedException($"Platform not supported: {platform}");
            }
        }

        // TODO: Replace this for a specific path for each platform
        private static string FindCrossCompiler(string monoCrossBin)
        {
            string exeExt = OS.IsWindows ? ".exe" : string.Empty;

            var files = new DirectoryInfo(monoCrossBin).GetFiles($"*mono-sgen{exeExt}", SearchOption.TopDirectoryOnly);
            if (files.Length > 0)
                return Path.Combine(monoCrossBin, files[0].Name);

            throw new FileNotFoundException($"Cannot find the mono runtime executable in {monoCrossBin}");
        }
    }
}

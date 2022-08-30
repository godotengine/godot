using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Godot;
using GodotTools.Core;
using GodotTools.Internals;
using Mono.Cecil;
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

        public bool UseInterpreter
        {
            get => _useInterpreter && !LLVMOnly;
            set => _useInterpreter = value;
        }

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
                string bits = features.Contains("64") ? "64" : features.Contains("32") ? "32" : null;
                CompileAssembliesForDesktop(exporter, platform, isDebug, bits, aotOpts, aotTempDir, outputDataDir, assembliesPrepared, bclDir);
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
                    exporter.AddSharedObject(soFilePath, tags: new[] { abi });
                }
            }
        }

        public static void CompileAssembliesForDesktop(ExportPlugin exporter, string platform, bool isDebug, string bits, AotOptions aotOpts, string aotTempDir, string outputDataDir, IDictionary<string, string> assemblies, string bclDir)
        {
            foreach (var assembly in assemblies)
            {
                string assemblyName = assembly.Key;
                string assemblyPath = assembly.Value;

                string outputFileExtension = platform == OS.Platforms.Windows ? ".dll" :
                    platform == OS.Platforms.OSX ? ".dylib" :
                    ".so";

                string outputFileName = assemblyName + ".dll" + outputFileExtension;
                string tempOutputFilePath = Path.Combine(aotTempDir, outputFileName);

                var compilerArgs = GetAotCompilerArgs(platform, isDebug, bits, aotOpts, assemblyPath, tempOutputFilePath);

                string compilerDirPath = GetMonoCrossDesktopDirName(platform, bits);

                ExecuteCompiler(FindCrossCompiler(compilerDirPath), compilerArgs, bclDir);

                if (platform == OS.Platforms.OSX)
                {
                    exporter.AddSharedObject(tempOutputFilePath, tags: null);
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
            void RunAr(IEnumerable<string> objFilePaths, string outputFilePath)
            {
                var arArgs = new List<string>()
                {
                    "cr",
                    outputFilePath
                };

                foreach (string objFilePath in objFilePaths)
                    arArgs.Add(objFilePath);

                int arExitCode = OS.ExecuteCommand(XcodeHelper.FindXcodeTool("ar"), arArgs);
                if (arExitCode != 0)
                    throw new Exception($"Command 'ar' exited with code: {arExitCode}");
            }

            void RunLipo(IEnumerable<string> libFilePaths, string outputFilePath)
            {
                var lipoArgs = new List<string>();
                lipoArgs.Add("-create");
                lipoArgs.AddRange(libFilePaths);
                lipoArgs.Add("-output");
                lipoArgs.Add(outputFilePath);

                int lipoExitCode = OS.ExecuteCommand(XcodeHelper.FindXcodeTool("lipo"), lipoArgs);
                if (lipoExitCode != 0)
                    throw new Exception($"Command 'lipo' exited with code: {lipoExitCode}");
            }

            void CreateDummyLibForSimulator(string name, string xcFrameworkPath = null)
            {
                xcFrameworkPath = xcFrameworkPath ?? MonoFrameworkFromTemplate(name);
                string simulatorSubDir = Path.Combine(xcFrameworkPath, "ios-arm64_x86_64-simulator");

                string libFilePath = Path.Combine(simulatorSubDir, name + ".a");

                if (File.Exists(libFilePath))
                    return;

                string CompileForArch(string arch)
                {
                    string baseFilePath = Path.Combine(aotTempDir, $"{name}.{arch}");
                    string sourceFilePath = baseFilePath + ".c";

                    string source = $"int _{AssemblyNameToAotSymbol(name)}() {{ return 0; }}\n";
                    File.WriteAllText(sourceFilePath, source);

                    const string iOSPlatformName = "iPhoneSimulator";
                    const string versionMin = "10.0";
                    string iOSSdkPath = Path.Combine(XcodeHelper.XcodePath,
                        $"Contents/Developer/Platforms/{iOSPlatformName}.platform/Developer/SDKs/{iOSPlatformName}.sdk");

                    string objFilePath = baseFilePath + ".o";

                    var clangArgs = new[]
                    {
                        "-isysroot", iOSSdkPath,
                        $"-miphonesimulator-version-min={versionMin}",
                        "-arch", arch,
                        "-c",
                        "-o", objFilePath,
                        sourceFilePath
                    };

                    int clangExitCode = OS.ExecuteCommand(XcodeHelper.FindXcodeTool("clang"), clangArgs);
                    if (clangExitCode != 0)
                        throw new Exception($"Command 'clang' exited with code: {clangExitCode}");

                    string arOutputFilePath = Path.Combine(aotTempDir, baseFilePath + ".a");
                    RunAr(new[] {objFilePath}, arOutputFilePath);

                    return arOutputFilePath;
                }

                RunLipo(new[] {CompileForArch("arm64"), CompileForArch("x86_64")}, libFilePath);
            }

            string projectAssemblyName = GodotSharpDirs.ProjectAssemblyName;
            string libAotName = $"lib-aot-{projectAssemblyName}";

            string libAotXcFrameworkPath = Path.Combine(aotTempDir, $"{libAotName}.xcframework");
            string libAotXcFrameworkDevicePath = Path.Combine(libAotXcFrameworkPath, "ios-arm64");
            string libAotXcFrameworkSimPath = Path.Combine(libAotXcFrameworkPath, "ios-arm64_x86_64-simulator");

            Directory.CreateDirectory(libAotXcFrameworkPath);
            Directory.CreateDirectory(libAotXcFrameworkDevicePath);
            Directory.CreateDirectory(libAotXcFrameworkSimPath);

            string libAotFileName = $"{libAotName}.a";
            string libAotFilePath = Path.Combine(libAotXcFrameworkDevicePath, libAotFileName);

            var cppCode = new StringBuilder();
            var aotModuleInfoSymbols = new List<string>(assemblies.Count);

            var aotObjFilePaths = new List<string>(assemblies.Count);

            string compilerDirPath = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "aot-compilers",
                $"{OS.Platforms.iOS}-arm64");
            string crossCompiler = FindCrossCompiler(compilerDirPath);

            string aotCacheDir = Path.Combine(ProjectSettings.GlobalizePath(GodotSharpDirs.ResTempDir),
                "obj", isDebug ? "ExportDebug" : "ExportRelease", "godot-aot-cache");

            if (!Directory.Exists(aotCacheDir))
                Directory.CreateDirectory(aotCacheDir);

            var aotCache = new AotCache(Path.Combine(aotCacheDir, "cache.json"));

            try
            {
                foreach (var assembly in assemblies)
                {
                    string assemblyName = assembly.Key;
                    string assemblyPath = assembly.Value;

                    string asmFilePath = Path.Combine(aotCacheDir, assemblyName + ".dll.S");
                    string objFilePath = Path.Combine(aotCacheDir, assemblyName + ".dll.o");

                    aotCache.RunCached(name: assemblyName, input: assemblyPath, output: objFilePath, () =>
                    {
                        Console.WriteLine($"AOT compiler: Compiling '{assemblyName}'...");

                        var compilerArgs = GetAotCompilerArgs(OS.Platforms.iOS, isDebug,
                            "arm64", aotOpts, assemblyPath, asmFilePath);

                        ExecuteCompiler(crossCompiler, compilerArgs, bclDir);

                        // Assembling
                        const string iOSPlatformName = "iPhoneOS";
                        const string versionMin = "10.0"; // TODO: Turn this hard-coded version into an exporter setting
                        string iOSSdkPath = Path.Combine(XcodeHelper.XcodePath,
                            $"Contents/Developer/Platforms/{iOSPlatformName}.platform/Developer/SDKs/{iOSPlatformName}.sdk");

                        var clangArgs = new List<string>()
                        {
                            "-isysroot", iOSSdkPath,
                            "-Qunused-arguments",
                            $"-miphoneos-version-min={versionMin}",
                            "-arch", "arm64",
                            "-c",
                            "-o", objFilePath,
                            "-x", "assembler"
                        };

                        if (isDebug)
                            clangArgs.Add("-DDEBUG");

                        clangArgs.Add(asmFilePath);

                        int clangExitCode = OS.ExecuteCommand(XcodeHelper.FindXcodeTool("clang"), clangArgs);
                        if (clangExitCode != 0)
                            throw new Exception($"Command 'clang' exited with code: {clangExitCode}");
                    });

                    aotObjFilePaths.Add(objFilePath);

                    aotModuleInfoSymbols.Add($"mono_aot_module_{AssemblyNameToAotSymbol(assemblyName)}_info");
                }
            }
            finally
            {
                aotCache.SaveCache();
            }

            RunAr(aotObjFilePaths, libAotFilePath);

            // Archive the AOT object files into a static library

            File.WriteAllText(Path.Combine(libAotXcFrameworkPath, "Info.plist"),
                $@"<?xml version=""1.0"" encoding=""UTF-8""?>
<!DOCTYPE plist PUBLIC ""-//Apple//DTD PLIST 1.0//EN"" ""http://www.apple.com/DTDs/PropertyList-1.0.dtd"">
<plist version=""1.0"">
<dict>
	<key>AvailableLibraries</key>
	<array>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64</string>
			<key>LibraryPath</key>
			<string>{libAotFileName}</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
		</dict>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64_x86_64-simulator</string>
			<key>LibraryPath</key>
			<string>{libAotFileName}</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
				<string>x86_64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
			<key>SupportedPlatformVariant</key>
			<string>simulator</string>
		</dict>
	</array>
	<key>CFBundlePackageType</key>
	<string>XFWK</string>
	<key>XCFrameworkFormatVersion</key>
	<string>1.0</string>
</dict>
</plist>
");

            // Add the fat AOT static library to the Xcode project
            CreateDummyLibForSimulator(libAotName, libAotXcFrameworkPath);
            exporter.AddIosProjectStaticLib(libAotXcFrameworkPath);

            // Generate driver code
            cppCode.AppendLine("#include <TargetConditionals.h>");

            cppCode.AppendLine("#if !TARGET_OS_SIMULATOR");
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

            // Prevent symbols from being stripped

            var symbols = CollectSymbols(assemblies);

            foreach (string symbol in symbols)
            {
                cppCode.Append("extern void *");
                cppCode.Append(symbol);
                cppCode.AppendLine(";");
            }

            cppCode.AppendLine("__attribute__((used)) __attribute__((optnone)) static void __godot_symbol_referencer() {");
            cppCode.AppendLine("\tvoid *aux;");

            foreach (string symbol in symbols)
            {
                cppCode.Append("\taux = ");
                cppCode.Append(symbol);
                cppCode.AppendLine(";");
            }

            cppCode.AppendLine("} // __godot_symbol_referencer");

            cppCode.AppendLine("} // extern \"C\"");
            cppCode.AppendLine("#endif // !TARGET_OS_SIMULATOR");

            // Add the driver code to the Xcode project
            exporter.AddIosCppCode(cppCode.ToString());

            // Add the required Mono libraries to the Xcode project

            string MonoLibFile(string libFileName) => libFileName + ".iphone.fat.a";

            string MonoLibFromTemplate(string libFileName) =>
                Path.Combine(Internal.FullTemplatesDir, "iphone-mono-libs", MonoLibFile(libFileName));

            string MonoFrameworkFile(string frameworkFileName) => frameworkFileName + ".xcframework";

            string MonoFrameworkFromTemplate(string frameworkFileName) =>
                Path.Combine(Internal.FullTemplatesDir, "iphone-mono-libs", MonoFrameworkFile(frameworkFileName));

            exporter.AddIosProjectStaticLib(MonoFrameworkFromTemplate("libmonosgen-2.0"));

            exporter.AddIosProjectStaticLib(MonoFrameworkFromTemplate("libmono-native"));

            if (aotOpts.UseInterpreter)
            {
                CreateDummyLibForSimulator("libmono-ee-interp");
                exporter.AddIosProjectStaticLib(MonoFrameworkFromTemplate("libmono-ee-interp"));
                CreateDummyLibForSimulator("libmono-icall-table");
                exporter.AddIosProjectStaticLib(MonoFrameworkFromTemplate("libmono-icall-table"));
                CreateDummyLibForSimulator("libmono-ilgen");
                exporter.AddIosProjectStaticLib(MonoFrameworkFromTemplate("libmono-ilgen"));
            }

            // TODO: Turn into an exporter option
            bool enableProfiling = false;
            if (enableProfiling)
                exporter.AddIosProjectStaticLib(MonoLibFromTemplate("libmono-profiler-log"));

            // Add frameworks required by Mono to the Xcode project
            exporter.AddIosFramework("libiconv.tbd");
            exporter.AddIosFramework("GSS.framework");
            exporter.AddIosFramework("CFNetwork.framework");
            if (!aotOpts.UseInterpreter)
                exporter.AddIosFramework("SystemConfiguration.framework");
        }

        private static List<string> CollectSymbols(IDictionary<string, string> assemblies)
        {
            var symbols = new List<string>();

            var resolver = new DefaultAssemblyResolver();
            foreach (var searchDir in resolver.GetSearchDirectories())
                resolver.RemoveSearchDirectory(searchDir);
            foreach (var searchDir in assemblies
                .Select(a => a.Value.GetBaseDir().NormalizePath()).Distinct())
            {
                resolver.AddSearchDirectory(searchDir);
            }

            AssemblyDefinition ReadAssembly(string fileName)
                => AssemblyDefinition.ReadAssembly(fileName,
                    new ReaderParameters {AssemblyResolver = resolver});

            foreach (var assembly in assemblies)
            {
                using (var assemblyDef = ReadAssembly(assembly.Value))
                    CollectSymbolsFromAssembly(assemblyDef, symbols);
            }

            return symbols;
        }

        private static void CollectSymbolsFromAssembly(AssemblyDefinition assembly, ICollection<string> symbols)
        {
            if (!assembly.MainModule.HasTypes)
                return;

            foreach (var type in assembly.MainModule.Types)
            {
                CollectSymbolsFromType(type, symbols);
            }
        }

        private static void CollectSymbolsFromType(TypeDefinition type, ICollection<string> symbols)
        {
            if (type.HasNestedTypes)
            {
                foreach (var nestedType in type.NestedTypes)
                    CollectSymbolsFromType(nestedType, symbols);
            }

            if (type.Module.HasModuleReferences)
                CollectPInvokeSymbols(type, symbols);
        }

        private static void CollectPInvokeSymbols(TypeDefinition type, ICollection<string> symbols)
        {
            if (!type.HasMethods)
                return;

            foreach (var method in type.Methods)
            {
                if (!method.IsPInvokeImpl || !method.HasPInvokeInfo)
                    continue;

                var pInvokeInfo = method.PInvokeInfo;

                if (pInvokeInfo == null)
                    continue;

                switch (pInvokeInfo.Module.Name)
                {
                    case "__Internal":
                    case "libSystem.Net.Security.Native":
                    case "System.Net.Security.Native":
                    case "libSystem.Security.Cryptography.Native.Apple":
                    case "System.Security.Cryptography.Native.Apple":
                    case "libSystem.Native":
                    case "System.Native":
                    case "libSystem.Globalization.Native":
                    case "System.Globalization.Native":
                    {
                        symbols.Add(pInvokeInfo.EntryPoint);
                        break;
                    }
                }
            }
        }

        /// Converts an assembly name to a valid symbol name in the same way the AOT compiler does
        private static string AssemblyNameToAotSymbol(string assemblyName)
        {
            var builder = new StringBuilder();

            foreach (char @char in assemblyName)
                builder.Append(char.IsLetterOrDigit(@char) || @char == '_' ? @char : '_');

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
                    throw new FileNotFoundException("Android toolchain not found: " + androidToolchain);
                }

                var androidToolPrefixes = new Dictionary<string, string>
                {
                    ["armeabi-v7a"] = "arm-linux-androideabi-",
                    ["arm64-v8a"] = "aarch64-linux-android-",
                    ["x86"] = "i686-linux-android-",
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
                    throw new Exception("Failed to start process for Mono AOT compiler");

                process.WaitForExit();

                if (process.ExitCode != 0)
                    throw new Exception($"Mono AOT compiler exited with code: {process.ExitCode}");
            }
        }

        private static IEnumerable<string> GetEnablediOSArchs(string[] features)
        {
            var iosArchs = new[]
            {
                "armv7",
                "arm64"
            };

            return iosArchs.Where(features.Contains);
        }

        private static IEnumerable<string> GetEnabledAndroidAbis(string[] features)
        {
            var androidAbis = new[]
            {
                "armeabi-v7a",
                "arm64-v8a",
                "x86",
                "x86_64"
            };

            return androidAbis.Where(features.Contains);
        }

        private static string GetAndroidTriple(string abi)
        {
            var abiArchs = new Dictionary<string, string>
            {
                ["armeabi-v7a"] = "armv7",
                ["arm64-v8a"] = "aarch64-v8a",
                ["x86"] = "i686",
                ["x86_64"] = "x86_64"
            };

            string arch = abiArchs[abi];

            return $"{arch}-linux-android";
        }

        private static string GetMonoCrossDesktopDirName(string platform, string bits)
        {
            switch (platform)
            {
                case OS.Platforms.Windows:
                case OS.Platforms.UWP:
                {
                    string arch = bits == "64" ? "x86_64" : "i686";
                    return $"windows-{arch}";
                }
                case OS.Platforms.OSX:
                {
                    Debug.Assert(bits == null || bits == "64");
                    string arch = "x86_64";
                    return $"{platform}-{arch}";
                }
                case OS.Platforms.X11:
                case OS.Platforms.Server:
                {
                    string arch = bits == "64" ? "x86_64" : "i686";
                    return $"linux-{arch}";
                }
                case OS.Platforms.Haiku:
                {
                    string arch = bits == "64" ? "x86_64" : "i686";
                    return $"{platform}-{arch}";
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

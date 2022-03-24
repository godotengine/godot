using System.Text;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using System.Collections.Generic;
using Microsoft.CodeAnalysis.CSharp;

namespace Godot.SourceGenerators
{
    [Generator]
    public class UnmanagedCallbacksGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotToolsProject())
                return;

            var nativeFuncsSymbol = context.Compilation.GetTypeByMetadataName("Godot.NativeInterop.NativeFuncs");
            if (nativeFuncsSymbol == null)
                return;

            IMethodSymbol[] callbacks = nativeFuncsSymbol.GetMembers()
                .Where(symbol => symbol is IMethodSymbol mds && mds.IsPartialDefinition)
                .Cast<IMethodSymbol>().ToArray();

            GenerateNativeFuncsImplementation(context, callbacks);
            GenerateUnmanagedCallbacks(context, callbacks);
        }

        private void GenerateNativeFuncsImplementation(GeneratorExecutionContext context, IEnumerable<IMethodSymbol> callbacks)
        {
            var source = new StringBuilder();
            var methodSource = new StringBuilder();
            var methodCallArguments = new StringBuilder();
            var methodSourceAfterCall = new StringBuilder();

            source.Append(
                @"using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.Bridge;

#pragma warning disable CA1707 // Disable warning: Identifiers should not contain underscores

namespace Godot.NativeInterop
{
    unsafe partial class NativeFuncs
    {
        internal static UnmanagedCallbacks _unmanagedCallbacks;
");

            foreach (var callback in callbacks)
            {
                methodSource.Clear();
                methodCallArguments.Clear();
                methodSourceAfterCall.Clear();

                source.Append(@$"
        {SyntaxFacts.GetText(callback.DeclaredAccessibility)} ");
                if (callback.IsStatic)
                {
                    source.Append("static ");
                }
                source.Append("partial ");
                source.Append(callback.ReturnType.FullQualifiedName());
                source.Append(' ');
                source.Append(callback.Name);
                source.Append('(');
                for (int i = 0; i < callback.Parameters.Length; i++)
                {
                    var parameter = callback.Parameters[i];
                    source.Append(parameter.ToDisplayString());
                    source.Append(' ');
                    source.Append(parameter.Name);

                    if (parameter.RefKind == RefKind.Out)
                    {
                        // Only assign default if the parameter won't be passed by-ref or copied later.
                        if (IsGodotInteropStruct(parameter.Type))
                        {
                            methodSource.Append(@$"
            {parameter.Name} = default;");
                        }
                    }

                    if (IsByRefParameter(parameter))
                    {
                        if (IsGodotInteropStruct(parameter.Type))
                        {
                            methodSource.Append(@$"
            ");
                            AppendCustomUnsafeAsPointer(methodSource, parameter, out string varName);
                            methodCallArguments.Append(varName);
                        }
                        else if (parameter.Type.IsValueType)
                        {
                            methodSource.Append(@$"
            ");
                            AppendCopyToStackAndGetPointer(methodSource, parameter, out string varName);
                            methodCallArguments.Append($"&{varName}");
                            if (parameter.RefKind is RefKind.Out or RefKind.Ref)
                            {
                                methodSourceAfterCall.Append(@$"
            {parameter.Name} = {varName};");
                            }
                        }
                        else
                        {
                            // If it's a by-ref param and we can't get the pointer
                            // just pass it by-ref and let it be pinned.
                            AppendRefKind(methodCallArguments, parameter.RefKind);
                            methodCallArguments.Append(' ');
                            methodCallArguments.Append(parameter.Name);
                        }
                    }
                    else
                    {
                        methodCallArguments.Append(parameter.Name);
                    }

                    if (i < callback.Parameters.Length - 1)
                    {
                        source.Append(", ");
                        methodCallArguments.Append(", ");
                    }
                }
                source.Append(')');
                source.Append(@"
        {");

                source.Append(methodSource);
                source.Append(@"
            ");
                if (!callback.ReturnsVoid)
                {
                    if (methodSourceAfterCall.Length != 0)
                    {
                        source.Append($"{callback.ReturnType.FullQualifiedName()} ret = ");
                    }
                    else
                    {
                        source.Append("return ");
                    }
                }
                source.Append($"_unmanagedCallbacks.{callback.Name}(");
                source.Append(methodCallArguments);
                source.Append(");");
                if (methodSourceAfterCall.Length != 0)
                {
                    source.Append(methodSourceAfterCall);
                    if (!callback.ReturnsVoid)
                    {
                        source.Append(@"
            return ret;");
                    }
                }
                source.Append(@"
        }");
            }

            source.Append(@"
    }
}

#pragma warning restore CA1707
");

            context.AddSource("NativeFuncs.generated",
                SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private void GenerateUnmanagedCallbacks(GeneratorExecutionContext context, IEnumerable<IMethodSymbol> callbacks)
        {
            var source = new StringBuilder();

            source.Append(
                @"using System.Runtime.InteropServices;
using Godot.NativeInterop;

#pragma warning disable CA1707 // Disable warning: Identifiers should not contain underscores

namespace Godot.Bridge
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct UnmanagedCallbacks
    {");

            foreach (var callback in callbacks)
            {
                source.Append(@$"
        {SyntaxFacts.GetText(callback.DeclaredAccessibility)} ");
                source.Append("delegate* unmanaged<");
                foreach (var parameter in callback.Parameters)
                {
                    if (IsByRefParameter(parameter))
                    {
                        if (IsGodotInteropStruct(parameter.Type) || parameter.Type.IsValueType)
                        {
                            AppendPointerType(source, parameter.Type);
                        }
                        else
                        {
                            // If it's a by-ref param and we can't get the pointer
                            // just pass it by-ref and let it be pinned.
                            AppendRefKind(source, parameter.RefKind);
                            source.Append(' ');
                            source.Append(parameter.Type.FullQualifiedName());
                        }
                    }
                    else
                    {
                        source.Append(parameter.Type.FullQualifiedName());
                    }
                    source.Append(", ");
                }
                source.Append(callback.ReturnType.FullQualifiedName());
                source.Append($"> {callback.Name};");
            }

            source.Append(@"
    }
}

#pragma warning restore CA1707
");

            context.AddSource("UnmanagedCallbacks.generated",
                SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private static bool IsGodotInteropStruct(ITypeSymbol type) =>
            GodotInteropStructs.Contains(type.FullQualifiedName());

        private static bool IsByRefParameter(IParameterSymbol parameter) =>
            parameter.RefKind is RefKind.In or RefKind.Out or RefKind.Ref;

        private static StringBuilder AppendRefKind(StringBuilder source, RefKind refKind) =>
            refKind switch
            {
                RefKind.In => source.Append("in"),
                RefKind.Out => source.Append("out"),
                RefKind.Ref => source.Append("ref"),
                _ => source,
            };

        private static void AppendPointerType(StringBuilder source, ITypeSymbol type)
        {
            source.Append(type.FullQualifiedName());
            source.Append('*');
        }

        private static void AppendCustomUnsafeAsPointer(StringBuilder source, IParameterSymbol parameter, out string varName)
        {
            varName = $"{parameter.Name}_ptr";

            AppendPointerType(source, parameter.Type);
            source.Append(' ');
            source.Append(varName);
            source.Append(" = ");

            source.Append('(');
            AppendPointerType(source, parameter.Type);
            source.Append(')');

            source.Append("CustomUnsafe.AsPointer(ref ");
            if (parameter.RefKind == RefKind.In)
            {
                source.Append("CustomUnsafe.AsRef(in ");
            }
            source.Append(parameter.Name);
            if (parameter.RefKind == RefKind.In)
            {
                source.Append(')');
            }
            source.Append(");");
        }

        private static void AppendCopyToStackAndGetPointer(StringBuilder source, IParameterSymbol parameter, out string varName)
        {
            varName = $"{parameter.Name}_copy";

            source.Append(parameter.Type.FullQualifiedName());
            source.Append(' ');
            source.Append(varName);
            if (parameter.RefKind is RefKind.In or RefKind.Ref)
            {
                source.Append(" = ");
                source.Append(parameter.Name);
            }
            source.Append(';');
        }

        private static readonly string[] GodotInteropStructs = {
            "Godot.NativeInterop.godot_ref",
            "Godot.NativeInterop.godot_variant_call_error",
            "Godot.NativeInterop.godot_variant",
            "Godot.NativeInterop.godot_string",
            "Godot.NativeInterop.godot_string_name",
            "Godot.NativeInterop.godot_node_path",
            "Godot.NativeInterop.godot_signal",
            "Godot.NativeInterop.godot_callable",
            "Godot.NativeInterop.godot_array",
            "Godot.NativeInterop.godot_dictionary",
            "Godot.NativeInterop.godot_packed_byte_array",
            "Godot.NativeInterop.godot_packed_int32_array",
            "Godot.NativeInterop.godot_packed_int64_array",
            "Godot.NativeInterop.godot_packed_float32_array",
            "Godot.NativeInterop.godot_packed_float64_array",
            "Godot.NativeInterop.godot_packed_string_array",
            "Godot.NativeInterop.godot_packed_vector2_array",
            "Godot.NativeInterop.godot_packed_vector3_array",
            "Godot.NativeInterop.godot_packed_color_array",
        };
    }
}

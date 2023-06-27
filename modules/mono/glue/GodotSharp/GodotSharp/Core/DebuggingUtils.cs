using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Godot.NativeInterop;

#nullable enable

namespace Godot
{
    internal static class DebuggingUtils
    {
        private static void AppendTypeName(this StringBuilder sb, Type type)
        {
            if (type.IsPrimitive)
                sb.Append(type.Name);
            else if (type == typeof(void))
                sb.Append("void");
            else
                sb.Append(type);

            sb.Append(' ');
        }

        internal static void InstallTraceListener()
        {
            Trace.Listeners.Clear();
            Trace.Listeners.Add(new GodotTraceListener());
        }

        [StructLayout(LayoutKind.Sequential)]
        // ReSharper disable once InconsistentNaming
        internal ref struct godot_stack_info
        {
            public godot_string File;
            public godot_string Func;
            public int Line;
        }

        [StructLayout(LayoutKind.Sequential)]
        // ReSharper disable once InconsistentNaming
        internal ref struct godot_stack_info_vector
        {
            private IntPtr _writeProxy;
            private unsafe godot_stack_info* _ptr;

            public readonly unsafe godot_stack_info* Elements
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => _ptr;
            }

            public void Resize(int size)
            {
                if (size < 0)
                    throw new ArgumentOutOfRangeException(nameof(size));
                var err = NativeFuncs.godotsharp_stack_info_vector_resize(ref this, size);
                if (err != Error.Ok)
                    throw new InvalidOperationException("Failed to resize vector. Error code is: " + err.ToString());
            }

            public unsafe void Dispose()
            {
                if (_ptr == null)
                    return;
                NativeFuncs.godotsharp_stack_info_vector_destroy(ref this);
                _ptr = null;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetCurrentStackInfo(void* destVector)
        {
            try
            {
                var vector = (godot_stack_info_vector*)destVector;
                var stackTrace = new StackTrace(skipFrames: 1, fNeedFileInfo: true);
                int frameCount = stackTrace.FrameCount;

                if (frameCount == 0)
                    return;

                vector->Resize(frameCount);

                int i = 0;
                foreach (StackFrame frame in stackTrace.GetFrames())
                {
                    string? fileName = frame.GetFileName();
                    int fileLineNumber = frame.GetFileLineNumber();

                    GetStackFrameMethodDecl(frame, out string methodDecl);

                    godot_stack_info* stackInfo = &vector->Elements[i];

                    // Assign directly to element in Vector. This way we don't need to worry
                    // about disposal if an exception is thrown. The Vector takes care of it.
                    stackInfo->File = Marshaling.ConvertStringToNative(fileName);
                    stackInfo->Func = Marshaling.ConvertStringToNative(methodDecl);
                    stackInfo->Line = fileLineNumber;

                    i++;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        internal static void GetStackFrameMethodDecl(StackFrame frame, out string methodDecl)
        {
            MethodBase? methodBase = frame.GetMethod();

            if (methodBase == null)
            {
                methodDecl = string.Empty;
                return;
            }

            var sb = new StringBuilder();

            if (methodBase is MethodInfo methodInfo)
                sb.AppendTypeName(methodInfo.ReturnType);

            sb.Append(methodBase.DeclaringType?.FullName ?? "<unknown>");
            sb.Append('.');
            sb.Append(methodBase.Name);

            if (methodBase.IsGenericMethod)
            {
                Type[] genericParams = methodBase.GetGenericArguments();

                sb.Append('<');

                for (int j = 0; j < genericParams.Length; j++)
                {
                    if (j > 0)
                        sb.Append(", ");

                    sb.AppendTypeName(genericParams[j]);
                }

                sb.Append('>');
            }

            sb.Append('(');

            bool varArgs = (methodBase.CallingConvention & CallingConventions.VarArgs) != 0;

            ParameterInfo[] parameter = methodBase.GetParameters();

            for (int i = 0; i < parameter.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");

                if (i == parameter.Length - 1 && varArgs)
                    sb.Append("params ");

                sb.AppendTypeName(parameter[i].ParameterType);
            }

            sb.Append(')');

            methodDecl = sb.ToString();
        }
    }
}

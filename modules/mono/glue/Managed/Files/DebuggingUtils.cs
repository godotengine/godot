using System;
using System.Diagnostics;
using System.Reflection;
using System.Text;

namespace Godot
{
    internal static class DebuggingUtils
    {
        internal static void AppendTypeName(this StringBuilder sb, Type type)
        {
            if (type.IsPrimitive)
                sb.Append(type.Name);
            else if (type == typeof(void))
                sb.Append("void");
            else
                sb.Append(type);

            sb.Append(" ");
        }

        public static void InstallTraceListener()
        {
            Trace.Listeners.Clear();
            Trace.Listeners.Add(new GodotTraceListener());
        }

        public static void GetStackFrameInfo(StackFrame frame, out string fileName, out int fileLineNumber, out string methodDecl)
        {
            fileName = frame.GetFileName();
            fileLineNumber = frame.GetFileLineNumber();

            MethodBase methodBase = frame.GetMethod();

            if (methodBase == null)
            {
                methodDecl = string.Empty;
                return;
            }

            var sb = new StringBuilder();

            if (methodBase is MethodInfo)
                sb.AppendTypeName(((MethodInfo)methodBase).ReturnType);

            sb.Append(methodBase.DeclaringType.FullName);
            sb.Append(".");
            sb.Append(methodBase.Name);

            if (methodBase.IsGenericMethod)
            {
                Type[] genericParams = methodBase.GetGenericArguments();

                sb.Append("<");

                for (int j = 0; j < genericParams.Length; j++)
                {
                    if (j > 0)
                        sb.Append(", ");

                    sb.AppendTypeName(genericParams[j]);
                }

                sb.Append(">");
            }

            sb.Append("(");

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

            sb.Append(")");

            methodDecl = sb.ToString();
        }
    }
}

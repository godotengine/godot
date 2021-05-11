using System;

#nullable enable

namespace Godot
{
    public partial class Object
    {
        public class NativeMemberNotFoundException : Exception
        {
            public NativeMemberNotFoundException()
            {
            }

            public NativeMemberNotFoundException(string? message) : base(message)
            {
            }

            public NativeMemberNotFoundException(string? message, Exception? innerException)
                : base(message, innerException)
            {
            }
        }

        public class NativeConstructorNotFoundException : NativeMemberNotFoundException
        {
            private readonly string? _nativeClassName;

            // ReSharper disable once InconsistentNaming
            private const string Arg_NativeConstructorNotFoundException = "Unable to find the native constructor.";

            public NativeConstructorNotFoundException()
                : base(Arg_NativeConstructorNotFoundException)
            {
            }

            public NativeConstructorNotFoundException(string? nativeClassName)
                : this(Arg_NativeConstructorNotFoundException, nativeClassName)
            {
            }

            public NativeConstructorNotFoundException(string? message, Exception? innerException)
                : base(message, innerException)
            {
            }

            public NativeConstructorNotFoundException(string? message, string? nativeClassName)
                : base(message)
            {
                _nativeClassName = nativeClassName;
            }

            public NativeConstructorNotFoundException(string? message, string? nativeClassName, Exception? innerException)
                : base(message, innerException)
            {
                _nativeClassName = nativeClassName;
            }

            public override string Message
            {
                get
                {
                    string s = base.Message;

                    if (string.IsNullOrEmpty(s))
                    {
                        s = Arg_NativeConstructorNotFoundException;
                    }

                    if (!string.IsNullOrEmpty(_nativeClassName))
                    {
                        s += " " + string.Format("(Class '{0}')", _nativeClassName);
                    }

                    return s;
                }
            }
        }

        public class NativeMethodBindNotFoundException : NativeMemberNotFoundException
        {
            private readonly string? _nativeMethodName;

            // ReSharper disable once InconsistentNaming
            private const string Arg_NativeMethodBindNotFoundException = "Unable to find the native method bind.";

            public NativeMethodBindNotFoundException()
                : base(Arg_NativeMethodBindNotFoundException)
            {
            }

            public NativeMethodBindNotFoundException(string? nativeMethodName)
                : this(Arg_NativeMethodBindNotFoundException, nativeMethodName)
            {
            }

            public NativeMethodBindNotFoundException(string? message, Exception? innerException)
                : base(message, innerException)
            {
            }

            public NativeMethodBindNotFoundException(string? message, string? nativeMethodName)
                : base(message)
            {
                _nativeMethodName = nativeMethodName;
            }

            public NativeMethodBindNotFoundException(string? message, string? nativeMethodName, Exception? innerException)
                : base(message, innerException)
            {
                _nativeMethodName = nativeMethodName;
            }

            public override string Message
            {
                get
                {
                    string s = base.Message;

                    if (string.IsNullOrEmpty(s))
                    {
                        s = Arg_NativeMethodBindNotFoundException;
                    }

                    if (!string.IsNullOrEmpty(_nativeMethodName))
                    {
                        s += " " + string.Format("(Method '{0}')", _nativeMethodName);
                    }

                    return s;
                }
            }
        }
    }
}

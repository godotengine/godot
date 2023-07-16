using System;
using System.Text;

#nullable enable

namespace Godot
{
    public partial class GodotObject
    {
        /// <summary>
        /// The general exception that is thrown when a native member cannot be found.
        /// </summary>
        public class NativeMemberNotFoundException : Exception
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMemberNotFoundException"/> class with default properties.
            /// </summary>
            public NativeMemberNotFoundException()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMemberNotFoundException"/> class with a specified error message.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            public NativeMemberNotFoundException(string? message) : base(message)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMemberNotFoundException"/> class with a specified error message
            /// and a reference to the inner exception that is the cause of this exception.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            /// <param name="innerException">The exception that is the cause of the current exception. If the inner parameter
            /// is not null, the current exception is raised in a catch block that handles the inner exception.</param>
            public NativeMemberNotFoundException(string? message, Exception? innerException)
                : base(message, innerException)
            {
            }
        }

        /// <summary>
        /// The exception that is thrown when a native constructor cannot be found.
        /// </summary>
        public class NativeConstructorNotFoundException : NativeMemberNotFoundException
        {
            private readonly string? _nativeClassName;

            // ReSharper disable once InconsistentNaming
            private const string Arg_NativeConstructorNotFoundException = "Unable to find the native constructor.";

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeConstructorNotFoundException"/> class with default properties.
            /// </summary>
            public NativeConstructorNotFoundException()
                : base(Arg_NativeConstructorNotFoundException)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeConstructorNotFoundException"/> class with the native class name.
            /// </summary>
            /// <param name="nativeClassName">The native class name that failed to retrieve a constructor.</param>
            public NativeConstructorNotFoundException(string? nativeClassName)
                : this(Arg_NativeConstructorNotFoundException, nativeClassName)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeConstructorNotFoundException"/> class with a specified error message
            /// and a reference to the inner exception that is the cause of this exception.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            /// <param name="innerException">The exception that is the cause of the current exception. If the inner parameter
            /// is not null, the current exception is raised in a catch block that handles the inner exception.</param>
            public NativeConstructorNotFoundException(string? message, Exception? innerException)
                : base(message, innerException)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeConstructorNotFoundException"/> class with a specified error message
            /// and the native class name.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            /// <param name="nativeClassName">The native class name that failed to retrieve a constructor.</param>
            public NativeConstructorNotFoundException(string? message, string? nativeClassName)
                : base(message)
            {
                _nativeClassName = nativeClassName;
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeConstructorNotFoundException"/> class with a specified error message,
            /// the native class name, and a reference to the inner exception that is the cause of this exception.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            /// <param name="nativeClassName">The native class name that failed to retrieve a constructor.</param>
            /// <param name="innerException">The exception that is the cause of the current exception. If the inner parameter
            /// is not null, the current exception is raised in a catch block that handles the inner exception.</param>
            public NativeConstructorNotFoundException(string? message, string? nativeClassName, Exception? innerException)
                : base(message, innerException)
            {
                _nativeClassName = nativeClassName;
            }

            /// <summary>
            /// Gets the error message for this exception.
            /// </summary>
            public override string Message
            {
                get
                {
                    StringBuilder sb;
                    if (string.IsNullOrEmpty(base.Message))
                    {
                        sb = new(Arg_NativeConstructorNotFoundException);
                    }
                    else
                    {
                        sb = new(base.Message);
                    }

                    if (!string.IsNullOrEmpty(_nativeClassName))
                    {
                        sb.Append($" (Method '{_nativeClassName}')");
                    }

                    return sb.ToString();
                }
            }
        }

        /// <summary>
        /// The exception that is thrown when a native method binding cannot be found.
        /// </summary>
        public class NativeMethodBindNotFoundException : NativeMemberNotFoundException
        {
            private readonly string? _nativeMethodName;

            // ReSharper disable once InconsistentNaming
            private const string Arg_NativeMethodBindNotFoundException = "Unable to find the native method bind.";

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMethodBindNotFoundException"/> class with default properties.
            /// </summary>
            public NativeMethodBindNotFoundException()
                : base(Arg_NativeMethodBindNotFoundException)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMethodBindNotFoundException"/> class with the native method name.
            /// </summary>
            /// <param name="nativeMethodName">The native method name that failed to bind.</param>
            public NativeMethodBindNotFoundException(string? nativeMethodName)
                : this(Arg_NativeMethodBindNotFoundException, nativeMethodName)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMethodBindNotFoundException"/> class with a specified error message
            /// and a reference to the inner exception that is the cause of this exception.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            /// <param name="innerException">The exception that is the cause of the current exception. If the inner parameter
            /// is not null, the current exception is raised in a catch block that handles the inner exception.</param>
            public NativeMethodBindNotFoundException(string? message, Exception? innerException)
                : base(message, innerException)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMethodBindNotFoundException"/> class with a specified error message
            /// and the native method name.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            /// <param name="nativeMethodName">The native method name that failed to bind.</param>
            public NativeMethodBindNotFoundException(string? message, string? nativeMethodName)
                : base(message)
            {
                _nativeMethodName = nativeMethodName;
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeMethodBindNotFoundException"/> class with a specified error message,
            /// the native method name, and a reference to the inner exception that is the cause of this exception.
            /// </summary>
            /// <param name="message">The error message that explains the reason for the exception.</param>
            /// <param name="nativeMethodName">The native method name that failed to bind.</param>
            /// <param name="innerException">The exception that is the cause of the current exception. If the inner parameter
            /// is not null, the current exception is raised in a catch block that handles the inner exception.</param>
            public NativeMethodBindNotFoundException(string? message, string? nativeMethodName, Exception? innerException)
                : base(message, innerException)
            {
                _nativeMethodName = nativeMethodName;
            }

            /// <summary>
            /// Gets the error message for this exception.
            /// </summary>
            public override string Message
            {
                get
                {
                    StringBuilder sb;
                    if (string.IsNullOrEmpty(base.Message))
                    {
                        sb = new(Arg_NativeMethodBindNotFoundException);
                    }
                    else
                    {
                        sb = new(base.Message);
                    }

                    if (!string.IsNullOrEmpty(_nativeMethodName))
                    {
                        sb.Append($" (Method '{_nativeMethodName}')");
                    }

                    return sb.ToString();
                }
            }
        }
    }
}

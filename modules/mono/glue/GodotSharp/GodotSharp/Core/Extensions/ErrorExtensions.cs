using System;
using System.IO;

namespace Godot
{
    public static class ErrorExtensions
    {
        public static void ThrowOnError(this Error error)
        {
            if (error.IsException(out var e))
                throw e;
        }

        internal static void DisposeAndThrowOnError(this Error error, IDisposable disposable)
        {
            if (error.IsException(out var e))
            {
                disposable.Dispose();
                throw e;
            }
        }

        public static bool IsException(this Error error, out Exception e)
        {
            if (error == Error.Ok)
            {
                e = null;
                return false;
            }

            var msg = $"Method returned with Godot error: '{Enum.GetName(typeof(Error), error)}'";

            switch (error)
            {
                case Error.InvalidParameter:
                    e = new ArgumentException(message: msg, paramName: null);
                    return true;
                case Error.ParameterRangeError:
                    e = new ArgumentOutOfRangeException(paramName: null, message: msg);
                    return true;
                case Error.FileBadDrive:
                case Error.FileBadPath:
                case Error.FileNotFound:
                    e = new FileNotFoundException(msg);
                    return true;
                case Error.AlreadyInUse:
                case Error.CantAcquireResource:
                case Error.CantOpen:
                case Error.CantCreate:
                case Error.FileAlreadyInUse:
                case Error.FileCantOpen:
                case Error.FileCantRead:
                case Error.FileCantWrite:
                case Error.FileEof:
                case Error.Locked:
                    e = new IOException(msg);
                    return true;
                case Error.InvalidData:
                    e = new InvalidDataException(msg);
                    return true;
                case Error.OutOfMemory:
                    e = new OutOfMemoryException(msg);
                    return true;
                case Error.Timeout:
                    e = new TimeoutException(msg);
                    return true;
                case Error.FileNoPermission:
                case Error.Unauthorized:
                    e = new UnauthorizedAccessException(msg);
                    return true;
                default:
                    e = new Exception(msg);
                    return true;
            }
        }
    }
}

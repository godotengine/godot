using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Godot
{
    public partial class File
    {
        private static void ThrowIfParamPathIsNullOrEmpty(string path, string paramName)
        {
            if (path == null)
                throw new ArgumentNullException(paramName, "File name cannot be null.");
            if (path.Length == 0)
                throw new ArgumentException("Empty file name is not legal.", paramName);
        }

        public static StreamReader OpenText(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            var stream = Open(path, FileAccess.Read);

            return new StreamReader(stream);
        }

        public static StreamWriter CreateText(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            var stream = Open(path, FileAccess.Write);

            return new StreamWriter(stream);
        }

        public static void Copy(string sourceFileName, string destFileName)
            => Copy(sourceFileName, destFileName, overwrite: false);

        public static void Copy(string sourceFileName, string destFileName, bool overwrite)
        {
            ThrowIfParamPathIsNullOrEmpty(sourceFileName, nameof(sourceFileName));
            ThrowIfParamPathIsNullOrEmpty(destFileName, nameof(destFileName));

            if (Directory.Exists(destFileName))
                throw new ArgumentException($"The target file '{destFileName}' is a directory, not a file.", nameof(destFileName));

            using var sourceFileStream = Open(sourceFileName, FileAccess.Read);

            if (!overwrite && Exists(destFileName))
                throw new IOException($"The file '{destFileName}' already exists.");

            using var destFileStream = Open(destFileName, FileAccess.Write);
            sourceFileStream.CopyTo(destFileStream);
        }

        public static FileStream Create(string path) => Open(path, FileAccess.ReadWrite);

        public static void Delete(string path)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            using var dir = new Directory();
            dir.Remove(path).ThrowOnError();
        }

        public static bool Exists(string path)
        {
            try
            {
                if (string.IsNullOrEmpty(path))
                    return false;

                using var dir = new Directory();
                return dir.FileExists(path);
            }
            catch (ArgumentException)
            {
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }

            return false;
        }

        private static ModeFlags FileAccessToModeFlags(FileAccess access)
        {
            switch (access)
            {
                case FileAccess.Read:
                    return ModeFlags.Read;
                case FileAccess.Write:
                    return ModeFlags.Write;
                case FileAccess.ReadWrite:
                    return ModeFlags.ReadWrite; // Assume FileMode.Create
                default:
                    throw new ArgumentOutOfRangeException(nameof(access), access, message: null);
            }
        }

        public static FileStream Open(string path) => Open(path, FileAccess.ReadWrite);

        public static FileStream Open(string path, FileAccess access)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            var file = new File();

            file.Open(path, FileAccessToModeFlags(access))
                .DisposeAndThrowOnError(file);

            return new FileStream(file, access);
        }

        public static FileStream OpenCompressed(string path, FileAccess access, CompressionMode mode = CompressionMode.Fastlz)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            var file = new File();

            file.OpenCompressed(path, FileAccessToModeFlags(access), mode)
                .DisposeAndThrowOnError(file);

            return new FileStream(file, access);
        }

        public static FileStream OpenEncrypted(string path, FileAccess access, byte[] key)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (key.Length == 0)
                throw new ArgumentException("Key cannot be empty.", nameof(key));

            var file = new File();

            file.OpenEncrypted(path, FileAccessToModeFlags(access), key)
                .DisposeAndThrowOnError(file);

            return new FileStream(file, access);
        }

        public static FileStream OpenEncryptedWithPass(string path, FileAccess access, string pass)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (pass == null)
                throw new ArgumentNullException(nameof(pass), "Password cannot be null.");
            if (pass.Length == 0)
                throw new ArgumentException("Password cannot be empty.", nameof(pass));

            var file = new File();

            file.OpenEncryptedWithPass(path, FileAccessToModeFlags(access), pass)
                .DisposeAndThrowOnError(file);

            return new FileStream(file, access);
        }

        public static DateTime GetCreationTime(string path)
        {
            using var file = new File();
            var dateTime = DateTimeOffset.FromUnixTimeSeconds((long)file.GetModifiedTime(path));
            return dateTime.LocalDateTime;
        }

        public static DateTime GetCreationTimeUtc(string path)
        {
            using var file = new File();
            var dateTime = DateTimeOffset.FromUnixTimeSeconds((long)file.GetModifiedTime(path));
            return dateTime.UtcDateTime;
        }

        public static FileStream OpenRead(string path) => Open(path, FileAccess.Read);

        public static FileStream OpenWrite(string path) => Open(path, FileAccess.Write);

        public static string ReadAllText(string path) => ReadAllText(path, Encoding.UTF8);

        public static string ReadAllText(string path, Encoding encoding)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            using var stream = new StreamReader(Open(path, FileAccess.Read), encoding, detectEncodingFromByteOrderMarks: true);
            return stream.ReadToEnd();
        }

        public static void WriteAllText(string path, string contents)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            using var stream = new StreamWriter(Open(path, FileAccess.Write));
            stream.Write(contents);
        }

        public static void WriteAllText(string path, string contents, Encoding encoding)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            using var stream = new StreamWriter(Open(path, FileAccess.Write), encoding);
            stream.Write(contents);
        }

        public static byte[] ReadAllBytes(string path)
        {
            using var stream = Open(path, FileAccess.Read);

            long length = stream.Length;

            if (length > int.MaxValue)
                throw new IOException("The file is too long. Only files less than 2 GBs in size are supported.");

            int offset = 0;
            var bytes = new byte[length];
            int count = (int)length;
            while (count > 0)
            {
                int bytesRead = stream.Read(bytes, offset, count);

                if (bytesRead == 0)
                    throw new EndOfStreamException();

                offset += bytesRead;
                count -= bytesRead;
            }

            return bytes;
        }

        public static void WriteAllBytes(string path, byte[] bytes)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (bytes == null)
                throw new ArgumentNullException(nameof(bytes));

            using var stream = Open(path, FileAccess.Write);
            stream.Write(bytes, offset: 0, count: bytes.Length);
        }

        public static string[] ReadAllLines(string path) => ReadAllLines(path, Encoding.UTF8);

        public static string[] ReadAllLines(string path, Encoding encoding)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            string line;
            List<string> lines = new List<string>();

            using var stream = new StreamReader(Open(path, FileAccess.Read), encoding);

            while ((line = stream.ReadLine()) != null)
                lines.Add(line);

            return lines.ToArray();
        }

        public static IEnumerable<string> ReadLines(string path) => ReadLines(path, Encoding.UTF8);

        public static IEnumerable<string> ReadLines(string path, Encoding encoding)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            using var stream = new StreamReader(Open(path, FileAccess.Read), encoding);

            string line;
            while ((line = stream.ReadLine()) != null)
                yield return line;
        }

        public static void WriteAllLines(string path, string[] contents) =>
            WriteAllLines(path, (IEnumerable<string>)contents);

        public static void WriteAllLines(string path, IEnumerable<string> contents)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (contents == null)
                throw new ArgumentNullException(nameof(contents));

            using var stream = new StreamWriter(Open(path, FileAccess.Write));

            foreach (string line in contents)
                stream.WriteLine(line);
        }

        public static void WriteAllLines(string path, string[] contents, Encoding encoding) =>
            WriteAllLines(path, (IEnumerable<string>)contents, encoding);

        public static void WriteAllLines(string path, IEnumerable<string> contents, Encoding encoding)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (contents == null)
                throw new ArgumentNullException(nameof(contents));
            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            using var stream = new StreamWriter(Open(path, FileAccess.Write), encoding);

            foreach (string line in contents)
                stream.WriteLine(line);
        }

        public static void Move(string sourceFileName, string destFileName) =>
            Move(sourceFileName, destFileName, overwrite: false);

        public static void Move(string sourceFileName, string destFileName, bool overwrite)
        {
            ThrowIfParamPathIsNullOrEmpty(sourceFileName, nameof(sourceFileName));
            ThrowIfParamPathIsNullOrEmpty(destFileName, nameof(destFileName));

            if (!Exists(sourceFileName))
                throw new FileNotFoundException($"The source file '{sourceFileName}' could not be found", sourceFileName);

            Copy(sourceFileName, destFileName, overwrite);
            Delete(sourceFileName);
        }

        // TODO: Try to improve async implementations. Currently blocked due to limitations with the Godot IO API.

        private const int BufferSize = 4096;

        public static Task<string> ReadAllTextAsync(string path, CancellationToken cancellationToken = default)
            => ReadAllTextAsync(path, Encoding.UTF8, cancellationToken);

        public static Task<string> ReadAllTextAsync(string path, Encoding encoding, CancellationToken cancellationToken = default)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled<string>(cancellationToken);

            return _ReadAllTextAsync(path, encoding, cancellationToken);
        }

        private static async Task<string> _ReadAllTextAsync(string path, Encoding encoding, CancellationToken cancellationToken)
        {
            char[] buffer = null;

            using var stream = new StreamReader(Open(path, FileAccess.Read), encoding);

            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                buffer = ArrayPool<char>.Shared.Rent(stream.CurrentEncoding.GetMaxCharCount(BufferSize));

                var builder = new StringBuilder();

                while (true)
                {
                    int read = await stream.ReadAsync(new Memory<char>(buffer), cancellationToken).ConfigureAwait(false);

                    if (read == 0)
                        return builder.ToString();

                    builder.Append(buffer, 0, read);
                }
            }
            finally
            {
                if (buffer != null)
                {
                    ArrayPool<char>.Shared.Return(buffer);
                }
            }
        }

        public static Task WriteAllTextAsync(string path, string contents, CancellationToken cancellationToken = default)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled(cancellationToken);

            if (string.IsNullOrEmpty(contents))
                return Task.CompletedTask;

            return _WriteAllTextAsync(new StreamWriter(Open(path, FileAccess.Write)), contents, cancellationToken);
        }

        public static Task WriteAllTextAsync(string path, string contents, Encoding encoding, CancellationToken cancellationToken = default)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled(cancellationToken);

            if (string.IsNullOrEmpty(contents))
                return Task.CompletedTask;

            return _WriteAllTextAsync(new StreamWriter(Open(path, FileAccess.Write), encoding), contents, cancellationToken);
        }

        public static async Task<byte[]> ReadAllBytesAsync(string path, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var stream = Open(path, FileAccess.Read);

            long lengthLong = stream.Length;
            if (lengthLong > int.MaxValue)
                throw new IOException("The file is too long. Only files less than 2 GBs in size are supported.");

            int length = (int)lengthLong;

            int index = 0;
            var bytes = new byte[length];
            do
            {
                var buffer = new Memory<byte>(bytes, index, length - index);
                int bytesRead = await stream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);

                if (bytesRead == 0)
                    throw new EndOfStreamException();

                index += bytesRead;
            } while (index < length);

            return bytes;
        }

        public static Task WriteAllBytesAsync(string path, byte[] bytes, CancellationToken cancellationToken = default)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (bytes == null)
                throw new ArgumentNullException(nameof(bytes));

            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled(cancellationToken);

            return _WriteAllBytesAsync(path, bytes, cancellationToken);
        }

        private static async Task _WriteAllBytesAsync(string path, byte[] bytes, CancellationToken cancellationToken)
        {
            using var stream = Open(path, FileAccess.Write);

            await stream.WriteAsync(new ReadOnlyMemory<byte>(bytes), cancellationToken).ConfigureAwait(false);
            await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
        }

        public static Task<string[]> ReadAllLinesAsync(string path, CancellationToken cancellationToken = default)
            => ReadAllLinesAsync(path, Encoding.UTF8, cancellationToken);

        public static Task<string[]> ReadAllLinesAsync(string path, Encoding encoding, CancellationToken cancellationToken = default)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled<string[]>(cancellationToken);

            return _ReadAllLinesAsync(path, encoding, cancellationToken);
        }

        private static async Task<string[]> _ReadAllLinesAsync(string path, Encoding encoding, CancellationToken cancellationToken)
        {
            using var stream = new StreamReader(Open(path, FileAccess.Read), encoding);

            cancellationToken.ThrowIfCancellationRequested();

            var lines = new List<string>();
            string line;
            while ((line = await stream.ReadLineAsync().ConfigureAwait(false)) != null)
            {
                lines.Add(line);
                cancellationToken.ThrowIfCancellationRequested();
            }

            return lines.ToArray();
        }

        public static Task WriteAllLinesAsync(string path, IEnumerable<string> contents, CancellationToken cancellationToken = default)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (contents == null)
                throw new ArgumentNullException(nameof(contents));

            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled(cancellationToken);

            return _WriteAllLinesAsync(new StreamWriter(Open(path, FileAccess.Write)), contents, cancellationToken);
        }

        public static Task WriteAllLinesAsync(string path, IEnumerable<string> contents, Encoding encoding, CancellationToken cancellationToken = default)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (contents == null)
                throw new ArgumentNullException(nameof(contents));
            if (encoding == null)
                throw new ArgumentNullException(nameof(encoding));

            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled(cancellationToken);

            return _WriteAllLinesAsync(new StreamWriter(Open(path, FileAccess.Write), encoding), contents, cancellationToken);
        }

        private static async Task _WriteAllLinesAsync(TextWriter writer, IEnumerable<string> contents, CancellationToken cancellationToken)
        {
            using (writer)
            {
                foreach (string line in contents)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    await writer.WriteLineAsync(line).ConfigureAwait(false);
                }

                cancellationToken.ThrowIfCancellationRequested();
                await writer.FlushAsync().ConfigureAwait(false);
            }
        }

        private static async Task _WriteAllTextAsync(StreamWriter stream, string contents, CancellationToken cancellationToken)
        {
            char[] buffer = null;

            try
            {
                buffer = ArrayPool<char>.Shared.Rent(BufferSize);
                int count = contents.Length;
                int index = 0;
                while (index < count)
                {
                    int length = Math.Min(BufferSize, count - index);
                    contents.CopyTo(index, buffer, 0, length);
                    await stream.WriteAsync(new ReadOnlyMemory<char>(buffer, 0, length), cancellationToken).ConfigureAwait(false);
                    index += length;
                }

                cancellationToken.ThrowIfCancellationRequested();
                await stream.FlushAsync().ConfigureAwait(false);
            }
            finally
            {
                stream.Dispose();

                if (buffer != null)
                    ArrayPool<char>.Shared.Return(buffer);
            }
        }
    }
}

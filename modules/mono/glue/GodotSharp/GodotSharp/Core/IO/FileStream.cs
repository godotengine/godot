using System;
using System.IO;

namespace Godot
{
    public class FileStream : Stream
    {
        private readonly File _file; // NotNull

        private readonly FileAccess _access;

        private bool _disposed;

        public override bool CanSeek => _file.IsOpen();

        public override bool CanRead => (_access & FileAccess.Read) != 0 && _file.IsOpen() && !_file.EofReached();

        public override bool CanWrite => (_access & FileAccess.Write) != 0 && _file.IsOpen();

        public override long Length
        {
            get
            {
                ThrowIfClosed();
                return _file.GetLen();
            }
        }

        public override long Position
        {
            get
            {
                ThrowIfClosed();
                return _file.GetPosition();
            }
            set
            {
                ThrowIfClosed();
                _file.Seek((int)value);
            }
        }

        public FileStream(File file, FileAccess access)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file), "File cannot be null.");

            if (!file.IsOpen())
                throw new ArgumentException("Cannot access a closed file.", nameof(file));

            _file = file;
            _access = access;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (Object.IsInstanceValid(_file) && _file.IsOpen())
                {
                    _file.Close();
                }

                _file.Dispose();

                _disposed = true;
            }

            base.Dispose(disposing);
        }

        private void ThrowIfClosed()
        {
            if (_disposed || !_file.IsOpen())
                throw new ObjectDisposedException("Cannot access a closed file.");
        }

        public override void Flush() => _file.Flush();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin)
        {
            ThrowIfClosed();

            switch (origin)
            {
                case SeekOrigin.Begin:
                    _file.Seek((int)offset);
                    break;
                case SeekOrigin.Current:
                    _file.Seek((int)(Position + offset));
                    break;
                case SeekOrigin.End:
                    _file.SeekEnd((int)offset);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(origin), origin, null);
            }

            _file.GetError().ThrowOnError();

            return Position;
        }

        private void ValidateReadWriteArgs(byte[] buffer, int offset, int count)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer), "Buffer cannot be null.");
            if (offset < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), "Non-negative number required.");
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Non-negative number required.");

            if (buffer.Length < offset)
                throw new ArgumentException("Offset and length are out of bounds.");
            if (buffer.Length < offset + count)
                throw new ArgumentException("Count is greater than the number of elements from offset to the end of the source collection.");

            ThrowIfClosed();
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            ValidateReadWriteArgs(buffer, offset, count);

            int remaining = (int)(Length - Position);

            int size = Math.Min(Math.Min(buffer.Length - offset, count), remaining);
            byte[] data = _file.GetBuffer(size);

            _file.GetError().ThrowOnError();

            Array.Copy(sourceArray: data, sourceIndex: 0, destinationArray: buffer, destinationIndex: offset, length: data.Length);

            return data.Length;
        }

        public override void Write(byte[] buffer, int offset, int count)
        {
            ValidateReadWriteArgs(buffer, offset, count);

            int size = Math.Min(buffer.Length - offset, count);

            if (offset == 0 && buffer.Length <= count)
            {
                _file.StoreBuffer(buffer);
            }
            else
            {
                var data = new byte[size];

                Array.Copy(sourceArray: buffer, sourceIndex: offset, destinationArray: data, destinationIndex: 0, length: size);

                _file.StoreBuffer(data);
            }

            _file.GetError().ThrowOnError();
        }
    }
}

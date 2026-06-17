using System;
using System.Runtime.InteropServices;
using System.Diagnostics.CodeAnalysis;

#nullable enable
namespace Godot
{
	/// <summary>
	/// Stores the target instance id and method name.
	/// </summary>
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public readonly partial struct Callable : IEquatable<Callable>
	{
		private readonly ulong _targetId;
		private readonly string _method;

		/// <summary>
		/// Object id that contains the method.
		/// </summary>
		public ulong TargetId => _targetId;

		/// <summary>
		/// Name of the method that will be called.
		/// </summary>
		public string Method => _method;

		public Callable(ulong id, string method)
		{
			_targetId = id;
			_method = method ?? string.Empty;
		}

		public static Callable Create(ulong id, string method) => new Callable(id, method);

		public override readonly string ToString()
		{
			return $"{_targetId}::{_method}";
		}

		public readonly bool Equals(Callable other)
		{
			return _targetId == other._targetId &&
				   string.Equals(_method, other._method, StringComparison.Ordinal);
		}

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Callable other && Equals(other);
		}

		public override readonly int GetHashCode()
		{
			return HashCode.Combine(_targetId, _method);
		}

		public static bool operator ==(Callable left, Callable right) => left.Equals(right);
		public static bool operator !=(Callable left, Callable right) => !left.Equals(right);
	}
}

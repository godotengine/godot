using System;
using System.Diagnostics.CodeAnalysis;

#nullable enable
namespace Godot
{
	/// <summary>
	/// A pre-parsed relative or absolute path in a scene tree.
	/// </summary>
	public readonly struct NodePath : IEquatable<NodePath>
	{
		private readonly string _path;

		public NodePath()
		{
			_path = string.Empty;
		}

		public NodePath(string path)
		{
			_path = path ?? string.Empty;
		}

		/// <summary>
		/// Converts a string to a <see cref="NodePath"/>.
		/// </summary>
		/// <param name="from">The string to convert.</param>
		public static implicit operator NodePath(string from) => new NodePath(from);

		/// <summary>
		/// Converts this <see cref="NodePath"/> to a string.
		/// </summary>
		/// <param name="from">The <see cref="NodePath"/> to convert.</param>
		[return: NotNullIfNotNull("from")]
		public static implicit operator string?(NodePath? from) => from?.ToString();

		public override readonly string ToString()
		{
			return _path ?? string.Empty;
		}

		// Implementation of IEquatable<NodePath>
		public bool Equals(NodePath other) => _path == other._path;

		// get property as path not implemented
		//concatednames not implemented
		//concated subnames not implemented
		//get name not implemented
		// getnamecount not implemented
		// get subname not implemented
		//getsubname count not implemented
		// isabsolute not implemented
		//is empty not implemented


		public override bool Equals(object? obj) => obj is NodePath other && Equals(other);

		public override int GetHashCode() => _path.GetHashCode();

		public static bool operator ==(NodePath? left, NodePath? right)
		{
			if (left is null)
				return right is null;
			return left.Equals(right);
		}

		public static bool operator !=(NodePath? left, NodePath? right)
		{
			return !(left == right);
		}
	}
}

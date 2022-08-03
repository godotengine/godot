namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> rotate(qua<T, Q> const& q, T const& angle, vec<3, T, Q> const& v)
	{
		vec<3, T, Q> Tmp = v;

		// Axis of rotation must be normalised
		T len = glm::length(Tmp);
		if(abs(len - static_cast<T>(1)) > static_cast<T>(0.001))
		{
			T oneOverLen = static_cast<T>(1) / len;
			Tmp.x *= oneOverLen;
			Tmp.y *= oneOverLen;
			Tmp.z *= oneOverLen;
		}

		T const AngleRad(angle);
		T const Sin = sin(AngleRad * static_cast<T>(0.5));

		return q * qua<T, Q>(cos(AngleRad * static_cast<T>(0.5)), Tmp.x * Sin, Tmp.y * Sin, Tmp.z * Sin);
	}
}//namespace glm


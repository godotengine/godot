namespace glm
{
	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> ortho(T left, T right, T bottom, T top)
	{
		mat<4, 4, T, defaultp> Result(static_cast<T>(1));
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = - static_cast<T>(1);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoLH_ZO(T left, T right, T bottom, T top, T zNear, T zFar)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = static_cast<T>(1) / (zFar - zNear);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - zNear / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoLH_NO(T left, T right, T bottom, T top, T zNear, T zFar)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = static_cast<T>(2) / (zFar - zNear);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - (zFar + zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_ZO(T left, T right, T bottom, T top, T zNear, T zFar)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = - static_cast<T>(1) / (zFar - zNear);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - zNear / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH_NO(T left, T right, T bottom, T top, T zNear, T zFar)
	{
		mat<4, 4, T, defaultp> Result(1);
		Result[0][0] = static_cast<T>(2) / (right - left);
		Result[1][1] = static_cast<T>(2) / (top - bottom);
		Result[2][2] = - static_cast<T>(2) / (zFar - zNear);
		Result[3][0] = - (right + left) / (right - left);
		Result[3][1] = - (top + bottom) / (top - bottom);
		Result[3][2] = - (zFar + zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoZO(T left, T right, T bottom, T top, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return orthoLH_ZO(left, right, bottom, top, zNear, zFar);
#		else
			return orthoRH_ZO(left, right, bottom, top, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoNO(T left, T right, T bottom, T top, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return orthoLH_NO(left, right, bottom, top, zNear, zFar);
#		else
			return orthoRH_NO(left, right, bottom, top, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoLH(T left, T right, T bottom, T top, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return orthoLH_ZO(left, right, bottom, top, zNear, zFar);
#		else
			return orthoLH_NO(left, right, bottom, top, zNear, zFar);
#		endif

	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> orthoRH(T left, T right, T bottom, T top, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return orthoRH_ZO(left, right, bottom, top, zNear, zFar);
#		else
			return orthoRH_NO(left, right, bottom, top, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> ortho(T left, T right, T bottom, T top, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return orthoLH_ZO(left, right, bottom, top, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return orthoLH_NO(left, right, bottom, top, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return orthoRH_ZO(left, right, bottom, top, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return orthoRH_NO(left, right, bottom, top, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumLH_ZO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = -(right + left) / (right - left);
		Result[2][1] = -(top + bottom) / (top - bottom);
		Result[2][2] = farVal / (farVal - nearVal);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = -(farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumLH_NO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = -(right + left) / (right - left);
		Result[2][1] = -(top + bottom) / (top - bottom);
		Result[2][2] = (farVal + nearVal) / (farVal - nearVal);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH_ZO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = (right + left) / (right - left);
		Result[2][1] = (top + bottom) / (top - bottom);
		Result[2][2] = farVal / (nearVal - farVal);
		Result[2][3] = static_cast<T>(-1);
		Result[3][2] = -(farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH_NO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
		mat<4, 4, T, defaultp> Result(0);
		Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
		Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
		Result[2][0] = (right + left) / (right - left);
		Result[2][1] = (top + bottom) / (top - bottom);
		Result[2][2] = - (farVal + nearVal) / (farVal - nearVal);
		Result[2][3] = static_cast<T>(-1);
		Result[3][2] = - (static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumZO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return frustumLH_ZO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumRH_ZO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumNO(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return frustumLH_NO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumRH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumLH(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return frustumLH_ZO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumLH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return frustumRH_ZO(left, right, bottom, top, nearVal, farVal);
#		else
			return frustumRH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustum(T left, T right, T bottom, T top, T nearVal, T farVal)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return frustumLH_ZO(left, right, bottom, top, nearVal, farVal);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return frustumLH_NO(left, right, bottom, top, nearVal, farVal);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return frustumRH_ZO(left, right, bottom, top, nearVal, farVal);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return frustumRH_NO(left, right, bottom, top, nearVal, farVal);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveRH_ZO(T fovy, T aspect, T zNear, T zFar)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fovy / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = zFar / (zNear - zFar);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveRH_NO(T fovy, T aspect, T zNear, T zFar)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fovy / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = - (zFar + zNear) / (zFar - zNear);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveLH_ZO(T fovy, T aspect, T zNear, T zFar)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fovy / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = zFar / (zFar - zNear);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveLH_NO(T fovy, T aspect, T zNear, T zFar)
	{
		assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

		T const tanHalfFovy = tan(fovy / static_cast<T>(2));

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
		Result[2][2] = (zFar + zNear) / (zFar - zNear);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveZO(T fovy, T aspect, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveLH_ZO(fovy, aspect, zNear, zFar);
#		else
			return perspectiveRH_ZO(fovy, aspect, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveNO(T fovy, T aspect, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveLH_NO(fovy, aspect, zNear, zFar);
#		else
			return perspectiveRH_NO(fovy, aspect, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveLH(T fovy, T aspect, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveLH_ZO(fovy, aspect, zNear, zFar);
#		else
			return perspectiveLH_NO(fovy, aspect, zNear, zFar);
#		endif

	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveRH(T fovy, T aspect, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveRH_ZO(fovy, aspect, zNear, zFar);
#		else
			return perspectiveRH_NO(fovy, aspect, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspective(T fovy, T aspect, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return perspectiveLH_ZO(fovy, aspect, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return perspectiveLH_NO(fovy, aspect, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return perspectiveRH_ZO(fovy, aspect, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return perspectiveRH_NO(fovy, aspect, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovRH_ZO(T fov, T width, T height, T zNear, T zFar)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = zFar / (zNear - zFar);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovRH_NO(T fov, T width, T height, T zNear, T zFar)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = - (zFar + zNear) / (zFar - zNear);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovLH_ZO(T fov, T width, T height, T zNear, T zFar)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = zFar / (zFar - zNear);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovLH_NO(T fov, T width, T height, T zNear, T zFar)
	{
		assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));

		T const rad = fov;
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = (zFar + zNear) / (zFar - zNear);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - (static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovZO(T fov, T width, T height, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveFovLH_ZO(fov, width, height, zNear, zFar);
#		else
			return perspectiveFovRH_ZO(fov, width, height, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovNO(T fov, T width, T height, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return perspectiveFovLH_NO(fov, width, height, zNear, zFar);
#		else
			return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovLH(T fov, T width, T height, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveFovLH_ZO(fov, width, height, zNear, zFar);
#		else
			return perspectiveFovLH_NO(fov, width, height, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFovRH(T fov, T width, T height, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
			return perspectiveFovRH_ZO(fov, width, height, zNear, zFar);
#		else
			return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> perspectiveFov(T fov, T width, T height, T zNear, T zFar)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return perspectiveFovLH_ZO(fov, width, height, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return perspectiveFovLH_NO(fov, width, height, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return perspectiveFovRH_ZO(fov, width, height, zNear, zFar);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return perspectiveFovRH_NO(fov, width, height, zNear, zFar);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspectiveRH_NO(T fovy, T aspect, T zNear)
	{
		T const range = tan(fovy / static_cast<T>(2)) * zNear;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = (static_cast<T>(2) * zNear) / (right - left);
		Result[1][1] = (static_cast<T>(2) * zNear) / (top - bottom);
		Result[2][2] = - static_cast<T>(1);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = - static_cast<T>(2) * zNear;
		return Result;
	}
	
	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspectiveRH_ZO(T fovy, T aspect, T zNear)
	{
		T const range = tan(fovy / static_cast<T>(2)) * zNear;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = (static_cast<T>(2) * zNear) / (right - left);
		Result[1][1] = (static_cast<T>(2) * zNear) / (top - bottom);
		Result[2][2] = - static_cast<T>(1);
		Result[2][3] = - static_cast<T>(1);
		Result[3][2] = - zNear;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspectiveLH_NO(T fovy, T aspect, T zNear)
	{
		T const range = tan(fovy / static_cast<T>(2)) * zNear;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(T(0));
		Result[0][0] = (static_cast<T>(2) * zNear) / (right - left);
		Result[1][1] = (static_cast<T>(2) * zNear) / (top - bottom);
		Result[2][2] = static_cast<T>(1);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - static_cast<T>(2) * zNear;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspectiveLH_ZO(T fovy, T aspect, T zNear)
	{
		T const range = tan(fovy / static_cast<T>(2)) * zNear;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(T(0));
		Result[0][0] = (static_cast<T>(2) * zNear) / (right - left);
		Result[1][1] = (static_cast<T>(2) * zNear) / (top - bottom);
		Result[2][2] = static_cast<T>(1);
		Result[2][3] = static_cast<T>(1);
		Result[3][2] = - zNear;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> infinitePerspective(T fovy, T aspect, T zNear)
	{
#		if GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO
			return infinitePerspectiveLH_ZO(fovy, aspect, zNear);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO
			return infinitePerspectiveLH_NO(fovy, aspect, zNear);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO
			return infinitePerspectiveRH_ZO(fovy, aspect, zNear);
#		elif GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO
			return infinitePerspectiveRH_NO(fovy, aspect, zNear);
#		endif
	}

	// Infinite projection matrix: http://www.terathon.com/gdc07_lengyel.pdf
	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> tweakedInfinitePerspective(T fovy, T aspect, T zNear, T ep)
	{
		T const range = tan(fovy / static_cast<T>(2)) * zNear;
		T const left = -range * aspect;
		T const right = range * aspect;
		T const bottom = -range;
		T const top = range;

		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
		Result[0][0] = (static_cast<T>(2) * zNear) / (right - left);
		Result[1][1] = (static_cast<T>(2) * zNear) / (top - bottom);
		Result[2][2] = ep - static_cast<T>(1);
		Result[2][3] = static_cast<T>(-1);
		Result[3][2] = (ep - static_cast<T>(2)) * zNear;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> tweakedInfinitePerspective(T fovy, T aspect, T zNear)
	{
		return tweakedInfinitePerspective(fovy, aspect, zNear, epsilon<T>());
	}
}//namespace glm

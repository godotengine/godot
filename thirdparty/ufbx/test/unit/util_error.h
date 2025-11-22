#pragma once

struct ufbxwt_error
{
	ufbxwi_error error = { };

	ufbxwt_error()
	{
	}

	#if UFBXWI_FEATURE_THREAD_POOL
		ufbxwt_error(ufbxwi_thread_pool &tp)
			: ufbxwt_error()
		{
			error.thread_pool = &tp;
		}
	#endif
};

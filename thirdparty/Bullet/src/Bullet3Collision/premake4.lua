	project "Bullet3Collision"

	language "C++"
				
	kind "StaticLib"
		
	includedirs {".."}


	files {
		"**.cpp",
		"**.h"
	}
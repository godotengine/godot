#include <iostream>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

glm::mat4 camera(float Translate, glm::vec2 const& Rotate)
{
	glm::mat4 Projection = glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, 0.1f, 100.f);
	glm::mat4 View = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -Translate));
	View = glm::rotate(View, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
	View = glm::rotate(View, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
	return Projection * View * Model;
}

int main()
{
    const glm::mat4 m = camera(1.f, glm::vec2(1.f, 0.5f));
    std::cout << "matrix diagonal: " << m[0][0] << ", "
              << m[1][1] << ", " << m[2][2] << ", " << m[3][3] << "\n";
    return 0;
}


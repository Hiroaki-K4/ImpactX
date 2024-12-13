#ifndef PARTICLECOLOR_HPP
#define PARTICLECOLOR_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class ParticleColor {
    private:
        glm::vec3 core_color;
        glm::vec3 middle_color;
        glm::vec3 outer_color;

    public:
        ParticleColor();
        ~ParticleColor();

        void initialize(glm::vec3 core_color, glm::vec3 middle_color, glm::vec3 outer_color);
        glm::vec3 mix_color(glm::vec3 c1, glm::vec3 c2, float ratio);
        void calculate_gradient_color(glm::vec3 center_pos, glm::vec3 particle_pos,
                                        float radius, glm::vec3 &gradient_color);
};

#endif

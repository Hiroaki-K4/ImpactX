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

        void initialize(const glm::vec3 &core_color, const glm::vec3 &middle_color, const glm::vec3 &outer_color);
        glm::vec3 mix_color(const glm::vec3 &c1, const glm::vec3 &c2, const float ratio);
        void calculate_gradient_color(const glm::vec3 &center_pos, const glm::vec3 &particle_pos,
                                        const float radius, glm::vec3 &gradient_color);
};

#endif

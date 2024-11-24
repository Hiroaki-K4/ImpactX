#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <random>
#include <iostream>

class Particle {
    private:
        std::vector<glm::vec3> position;

    public:
        Particle(glm::vec3 center_pos, float planet_radius, int particle_num);
        ~Particle();

        std::vector<glm::vec3> get_particle_position();

        void initialize_position(glm::vec3 center_pos, float planet_radius, int particle_num);
};

#endif

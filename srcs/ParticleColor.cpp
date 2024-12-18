#include "ParticleColor.hpp"

ParticleColor::ParticleColor() {
}

ParticleColor::~ParticleColor() {
}

void ParticleColor::initialize(glm::vec3 core_color, glm::vec3 middle_color, glm::vec3 outer_color) {
    this->core_color = core_color;
    this->middle_color = middle_color;
    this->outer_color = outer_color;
}

glm::vec3 ParticleColor::mix_color(glm::vec3 c1, glm::vec3 c2, float ratio) {
    glm::vec3 mixed_color(
        c1.x + ratio * (c2.x - c1.x),
        c1.y + ratio * (c2.y - c1.y),
        c1.z + ratio * (c2.z - c1.z)
    );

    return mixed_color;
}

void ParticleColor::calculate_gradient_color(
    glm::vec3 center_pos, glm::vec3 particle_pos, float radius, glm::vec3 &gradient_color) {
    float norm_dist = glm::distance(center_pos, particle_pos) / radius;

    if (norm_dist < 0.5) {
        gradient_color = mix_color(this->core_color, this->middle_color, norm_dist * 2.0);
    } else {
        gradient_color = mix_color(this->middle_color, this->outer_color, (norm_dist - 0.5) * 2.0);
    }
}


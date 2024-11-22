#include "Particle.hpp"

Particle::Particle(int particle_num) {
    initialize_position(particle_num);
}

Particle::~Particle() {

}

std::vector<glm::vec3> Particle::get_particle_position() {
    return this->position;
}

void Particle::initialize_position(int particle_num) {
    this->position.push_back(glm::vec3(-10.0f, 1.5f, -1.0f));
    this->position.push_back(glm::vec3(1.0f, 1.5f, -10.0f));
    this->position.push_back(glm::vec3(-15.0f, -1.5f, -1.0f));
    this->position.push_back(glm::vec3(1.0f, -1.5f, -15.0f));
}
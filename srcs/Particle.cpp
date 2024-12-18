#include "Particle.hpp"


Particle::Particle(const glm::vec3 &center_pos_1, const glm::vec3 &center_pos_2, const float planet_radius,
                    const int particle_num_1, const int particle_num_2, const glm::vec3 &initial_velocity_1,
                    const glm::vec3 &initial_velocity_2, const float mass, const float particle_radius, const int threads) {
    this->mass = mass;
    this->collision_distance = particle_radius * 2;
    this->particle_color.initialize(glm::vec3(1.0f, 1.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.79f, 0.29f, 0.21f));
    initialize(center_pos_1, planet_radius, particle_num_1);
    for (int i = 0; i < particle_num_1; i++) {
        this->velocity.push_back(initial_velocity_1);
    }
    this->particle_color.initialize(glm::vec3(0.1f, 0.1f, 0.1f),
        glm::vec3(0.3f, 0.6f, 0.8f), glm::vec3(0.12f, 0.38f, 0.93f));
    initialize(center_pos_2, planet_radius, particle_num_2);
    for (int i = 0; i < particle_num_2; i++) {
        this->velocity.push_back(initial_velocity_2);
    }
    this->particle_cuda.initialize(this->position, this->velocity, particle_num_1 + particle_num_2,
                                    threads, this->collision_distance);
}

Particle::~Particle() {
}

std::vector<glm::vec3> Particle::get_particle_position() {
    return this->position;
}

std::vector<glm::vec3> Particle::get_particle_color() {
    return this->color;
}

void Particle::initialize(
    const glm::vec3 &center_pos, const float planet_radius, const int particle_num) {
    std::random_device rd;   // Seed for the random number engine
    std::mt19937 gen(rd());  // Mersenne Twister engine

    std::uniform_real_distribution<float> angle_phi_dis(-M_PI / 2.0f, M_PI / 2.0f);
    std::uniform_real_distribution<float> angle_theta_dis(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dis(0.0f, planet_radius);
    for (int i = 0; i < particle_num; i++) {
        glm::vec3 pos;
        float angle_phi = angle_phi_dis(gen);
        float angle_theta = angle_theta_dis(gen);
        float radius = radius_dis(gen);

        pos.x = center_pos.x + radius * cos(angle_phi) * cos(angle_theta);
        pos.y = center_pos.y + radius * sin(angle_phi);
        pos.z = center_pos.z + radius * cos(angle_phi) * sin(angle_theta);
        this->position.push_back(pos);

        glm::vec3 gradient_color;
        this->particle_color.calculate_gradient_color(center_pos, pos, planet_radius, gradient_color);
        this->color.push_back(gradient_color);
    }
}

void Particle::update_particle(float delta_time) {
    this->particle_cuda.update_position_velocity(this->position, this->mass, delta_time);
}

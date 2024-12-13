#include "kernel.cuh"


__global__ void update_particle_kernel(glm::vec3 *cu_position, glm::vec3 *cu_velocity, float mass,
                                float delta_time, int num_particles, float collision_distance) {
    // Get current index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) {
        return;
    }

    for (int j = 0; j < num_particles; j++) {
        if (i == j) {
            continue;
        }
        float dist = glm::distance(cu_position[i], cu_position[j]);
        if (dist <= collision_distance) {
            // Calculate collision
            glm::vec3 new_velocity_vec = cu_position[i] - cu_position[j];
            cu_velocity[i] = (mass - mass) / (mass + mass) * cu_velocity[i] +
                                2 * mass / (mass + mass) * cu_velocity[j];
        } else {
            // Calculate gravity
            // TODO: Make impact more real
            float G = 6.6743 * std::pow(10, -30);
            float accel_power = mass * mass / std::pow(dist, 2) * G;
            glm::vec3 accel = (cu_position[j] - cu_position[i]) / dist;
            accel *= accel_power;
            cu_velocity[i] += accel * delta_time;
        }
    }
    cu_position[i] += cu_velocity[i] * delta_time;
}

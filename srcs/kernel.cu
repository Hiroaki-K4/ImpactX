#include "kernel.cuh"

// __global__ void update_particle_kernel(glm::vec2 *cu_position, glm::vec2 *cu_velocity,
//                                        glm::vec3 *cu_color, glm::vec2 gravity_pos, float delta_time,
//                                        float aspect_ratio, int num_particles, float max_distance, float gravity_strength) {
//     // Get current index
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_particles) {
//         return;
//     }

//     // Update velocity and position
//     glm::vec2 rescaled_pos = cu_position[idx];
//     rescaled_pos.x /= aspect_ratio;
//     glm::vec2 accel = gravity_pos - rescaled_pos;
//     glm::vec2 upscale_accel = accel * glm::length(accel) * gravity_strength;
//     cu_velocity[idx].x += upscale_accel.x * delta_time;
//     cu_velocity[idx].y += upscale_accel.y * delta_time;
//     cu_position[idx].x += cu_velocity[idx].x * delta_time * aspect_ratio;
//     cu_position[idx].y += cu_velocity[idx].y * delta_time;

//     // Update color
//     float new_color_val = fminf(glm::length(accel) / max_distance, 1.0f);
//     cu_color[idx] = glm::vec3(1.0f - new_color_val, 0.0f, new_color_val);
// }

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
            float accel_power = mass * mass / std::pow(dist, 2);
            glm::vec3 accel = (cu_position[j] - cu_position[i]) / dist;
            accel *= accel_power;
            cu_velocity[i] += accel * delta_time;
        }
    }
    cu_position[i] += cu_velocity[i] * delta_time;
}

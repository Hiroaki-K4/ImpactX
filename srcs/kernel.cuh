#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


__global__ void update_particle_kernel(glm::vec3 *cu_position, glm::vec3 *cu_velocity, float mass,
                                        float delta_time, int num_particles, float collision_distance);

#endif

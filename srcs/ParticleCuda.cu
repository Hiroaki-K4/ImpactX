#include "ParticleCuda.cuh"


ParticleCuda::ParticleCuda() {}

ParticleCuda::~ParticleCuda() {
    cudaFree(this->cu_position);
    cudaFree(this->cu_velocity);
}

void ParticleCuda::initialize(const std::vector<glm::vec3> &position, const std::vector<glm::vec3> &velocity,
    const int particle_num, const int threads, const float collision_distance) {
    this->threads = threads;
    this->blocks = (particle_num + threads - 1) / threads;
    this->collision_distance = collision_distance;

    // Allocate device memory
    cudaMalloc(&this->cu_position, particle_num * sizeof(glm::vec3));
    cudaMalloc(&this->cu_velocity, particle_num * sizeof(glm::vec3));

    cudaMemcpy(this->cu_position, position.data(), particle_num * sizeof(glm::vec3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->cu_velocity, velocity.data(), particle_num * sizeof(glm::vec3),
               cudaMemcpyHostToDevice);
}

void ParticleCuda::update_position_velocity(std::vector<glm::vec3> &position,
    const float mass, const float delta_time) {
    update_particle_kernel<<<this->blocks, this->threads>>>(
        this->cu_position, this->cu_velocity, mass, delta_time,
        position.size(), this->collision_distance);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        ParticleCuda::~ParticleCuda();
        exit(1);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(position.data(), this->cu_position, position.size() * sizeof(glm::vec3),
               cudaMemcpyDeviceToHost);
}

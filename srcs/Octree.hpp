#ifndef OCTREE_HPP
#define OCTREE_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <utility>
#include <vector>
#include <iostream>


struct Node {
    float mass;
    glm::vec3 position;
    glm::vec3 min_bound;
    glm::vec3 max_bound;
    std::vector<Node> children;
    bool is_leaf = true;
    Node *prev_node;
};


class Octree {
    private:
        Node root;

    public:
        Octree(glm::vec3 min_3d_coord, glm::vec3 max_3d_coord);
        ~Octree();

        void insert(std::vector<glm::vec3> position, std::vector<float> mass);
        void insert_point(Node node, std::vector<glm::vec3> position, std::vector<float> mass);
        void subdivide(Node &node);
        Node create_child_node(glm::vec3 min_bound, glm::vec3 max_bound);
};

#endif

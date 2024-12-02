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
    bool is_empty = true;
    Node *prev_node;
};


class Octree {
    private:
        Node root;

    public:
        Octree(glm::vec3 min_3d_coord, glm::vec3 max_3d_coord);
        ~Octree();

        bool is_point_inside(Node &node, const glm::vec3 &position);
        void insert(std::vector<glm::vec3> &position, std::vector<float> &mass);
        void insert_point(Node &node, const glm::vec3 &position, float mass);
        void subdivide(Node &node);
        void create_child_node(Node &new_node, const glm::vec3 &min_bound, const glm::vec3 &max_bound);
};

#endif

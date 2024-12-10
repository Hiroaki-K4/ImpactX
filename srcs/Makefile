PARENT_DIR := /home/h-kubo/mypro/
SRCS := main.cpp Shader.cpp Particle.cpp Octree.cpp kernel.cu glad.c
INCLUDE := -I../glfw/include -I../glad/include -I../glm
LDFLAGS := -L$(PARENT_DIR)ImpactX/glfw/build/src `pkg-config --libs glfw3` -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi -ldl
NAME := ImpactX
CXX := nvcc

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDE) $(LDFLAGS) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re

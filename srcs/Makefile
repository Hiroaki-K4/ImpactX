PARENT_DIR := /home/h-kubo/mypro/
SRCS := main.cpp Shader.cpp glad.c
INCLUDE := -I../glfw/include -I../glad/include -I../glm
LDFLAGS := -L$(PARENT_DIR)NewtonX/glfw/build/src `pkg-config --libs glfw3` -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi -ldl
NAME := NewtonX
CXX := g++

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDE) $(LDFLAGS) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re

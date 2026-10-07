CXX = g++

GTEST_DIR = lib/googletest/googletest
GTEST_BUILD_DIR = lib/googletest/build/lib
GTEST_ARCHIVE = $(GTEST_BUILD_DIR)/libgtest.a

SRC = $(wildcard test/*.cpp)
OBJ = $(patsubst %.cpp, build/obj/%.o, $(SRC))
TARGET = build/mlinalg_gtest

CPPFLAGS = -I$(GTEST_DIR)/include
CXXFLAGS = -g -O0 -std=c++23 -Wall -Wextra -pthread

ifeq ($(OS),Windows_NT)
    ASAN_FLAGS =
else
    ASAN_FLAGS = -fsanitize=address -fno-omit-frame-pointer
endif

CXXFLAGS += $(ASAN_FLAGS)

ifeq ($(shell uname -s),Darwin)
    ifeq ($(shell uname -m),arm64)
        CXXFLAGS += -mcpu=native
    endif
else
    CXXFLAGS += -mavx2 -mfma
endif

vpath %.cpp src test

all: test_build

test_build: $(TARGET)

$(GTEST_ARCHIVE):
	mkdir -p lib/googletest/build; \
	cd lib/googletest/build && cmake .. && make -j4;

$(TARGET): $(OBJ) $(GTEST_ARCHIVE)
	mkdir -p build
	$(CXX) $(OBJ) -o $(TARGET) $(CXXFLAGS) $(CPPFLAGS) $(GTEST_ARCHIVE) $(GTEST_BUILD_DIR)/libgtest_main.a

build/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(CPPFLAGS)

clean:
	rm -rf build $(TARGET)

clean_all:
	rm -rf build $(TARGET)
	rm -rf lib/googletest/build

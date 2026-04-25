CXX      := g++
CC       := gcc
CXXFLAGS := -std=c++17 -O3 -march=native -mavx2 -mfma -Wall -Wextra -Iinclude -fopenmp
CFLAGS   := -O3 -march=native -Iinclude
TARGET   := adapTQ_demo

SRCS := \
    core/fwht.cpp \
    core/codebook.cpp \
    core/quantizer.cpp \
    core/adaptq_c_api.cpp \
    core/adaptq_backend_vtable.cpp \
    cache/ring_buffer.cpp \
    attention/attention.cpp \
    utils/timer.cpp \
    main.cpp

SRCS_LIB := \
    core/fwht.cpp \
    core/codebook.cpp \
    core/quantizer.cpp \
    core/adaptq_c_api.cpp \
    core/adaptq_backend_vtable.cpp \
    cache/ring_buffer.cpp \
    attention/attention.cpp \
    utils/timer.cpp

OBJS     := $(SRCS:.cpp=.o)
OBJS_LIB := $(SRCS_LIB:.cpp=.o)
LIB      := libadaptq.so

# --- Adapter targets ---
# Plain-C standalone adapter test (no C++, no pybind11)
STANDALONE := adapters/standalone_test

# llama.cpp thin adapter shared library (no GGML headers needed)
ADAPTER_LLAMA := adapters/libadaptq_llamacpp.so

all: $(TARGET) $(LIB)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

$(LIB): $(OBJS_LIB)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ -lm

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c -o $@ $<

# Build standalone C adapter test
$(STANDALONE): adapters/adapter_standalone.c $(LIB)
	$(CC) $(CFLAGS) -fPIC adapters/adapter_standalone.c -L. -ladaptq -Wl,-rpath,. -lm -o $@

# Build llama.cpp adapter shared library (link against core libadaptq.so)
$(ADAPTER_LLAMA): adapters/adapter_llamacpp.cpp $(LIB)
	$(CXX) $(CXXFLAGS) -shared -fPIC adapters/adapter_llamacpp.cpp \
	    -L. -ladaptq -Wl,-rpath,. -lm -o $@

adapters: $(STANDALONE) $(ADAPTER_LLAMA)

test: $(STANDALONE)
	LD_LIBRARY_PATH=. ./$(STANDALONE)

install: $(LIB)
	cp $(LIB) /usr/local/lib/
	cp include/adaptq.h /usr/local/include/adaptq.h
	cp include/adaptq_backend.h /usr/local/include/adaptq_backend.h
	ldconfig

clean:
	rm -f $(OBJS) $(OBJS_LIB) $(TARGET) $(LIB) \
	      $(STANDALONE) $(ADAPTER_LLAMA) \
	      adapters/*.o adapters/*.so

.PHONY: all adapters test install clean


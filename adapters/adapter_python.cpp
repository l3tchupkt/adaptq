/* adapter_python.cpp
 *
 * Pluggable adapter: Python (pybind11)  →  AdapTQ core
 *
 * Build with:
 *   pip install pybind11
 *   g++ -O3 -march=native -mavx2 -shared -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       -Iinclude \
 *       adapters/adapter_python.cpp \
 *       -L. -ladaptq -Wl,-rpath,. \
 *       -o adaptq_py$(python3-config --extension-suffix)
 *
 * Python usage:
 *   import adaptq_py as aq
 *   ctx = aq.MHAContext(n_heads=32, head_dim=128, bits=4, capacity=8192)
 *   ctx.append(head=0, key=k_np, val=v_np, pos=0)
 *   out = ctx.compute(head=0, query=q_np)
 *   batch_out = ctx.compute_batch(head=0, queries=q_batch_np)
 */

#include "../include/adaptq_mha_backend.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>


namespace py = pybind11;

class PyMHAContext {
public:
  PyMHAContext(int n_heads, int head_dim, int bits, int capacity,
               uint64_t seed = 0, float v_mass = 0.95f, int hybrid_thresh = 512)
      : _backend(n_heads, head_dim, bits, capacity, seed, v_mass,
                 hybrid_thresh),
        _head_dim(head_dim), _n_heads(n_heads) {}

  void append(int head, py::array_t<float> key, py::array_t<float> val,
              int pos) {
    auto k = key.unchecked<1>();
    auto v = val.unchecked<1>();
    if (k.shape(0) != _head_dim || v.shape(0) != _head_dim)
      throw std::invalid_argument("key/val dim mismatch");
    _backend.append_kv(head, k.data(0), v.data(0), pos);
  }

  py::array_t<float> compute(int head, py::array_t<float> query) {
    auto q = query.unchecked<1>();
    if (q.shape(0) != _head_dim)
      throw std::invalid_argument("query dim mismatch");
    auto out = py::array_t<float>(_head_dim);
    _backend.compute(head, q.data(0), out.mutable_data(0));
    return out;
  }

  py::array_t<float> compute_batch(int head, py::array_t<float> queries) {
    auto q = queries.unchecked<2>();
    int nq = (int)q.shape(0);
    if (q.shape(1) != _head_dim)
      throw std::invalid_argument("queries dim mismatch");
    auto out = py::array_t<float>({nq, _head_dim});
    _backend.compute_batch(head, q.data(0, 0), nq, out.mutable_data(0, 0));
    return out;
  }

  void reset() { _backend.reset(); }
  size_t kv_bytes() const { return _backend.kv_bytes(); }
  int n_heads() const { return _n_heads; }
  int head_dim() const { return _head_dim; }

private:
  AdaptQMHABackend _backend;
  int _head_dim, _n_heads;
};

PYBIND11_MODULE(adaptq_py, m) {
  m.doc() = "AdapTQ — Quantized KV-cache attention, Python bindings";

  py::class_<PyMHAContext>(m, "MHAContext")
      .def(py::init<int, int, int, int, uint64_t, float, int>(),
           py::arg("n_heads"), py::arg("head_dim"), py::arg("bits"),
           py::arg("capacity"), py::arg("seed") = 0, py::arg("v_mass") = 0.95f,
           py::arg("hybrid_thresh") = 512)
      .def("append", &PyMHAContext::append, py::arg("head"), py::arg("key"),
           py::arg("val"), py::arg("pos"))
      .def("compute", &PyMHAContext::compute, py::arg("head"), py::arg("query"))
      .def("compute_batch", &PyMHAContext::compute_batch, py::arg("head"),
           py::arg("queries"))
      .def("reset", &PyMHAContext::reset)
      .def("kv_bytes", &PyMHAContext::kv_bytes)
      .def_property_readonly("n_heads", &PyMHAContext::n_heads)
      .def_property_readonly("head_dim", &PyMHAContext::head_dim);
}

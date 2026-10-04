#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * adaptq/error.h — Error codes and thread-local error reporting
 *
 * All AdapTQ C++ functions that can fail set a thread-local error string
 * readable via adaptq_last_error(). The adaptq_error_t enum is also
 * defined in adaptq.h (the stable C ABI); this header provides the
 * canonical C++ definition for internal use.
 * ----------------------------------------------------------------------- */

/* Included via adaptq.h for the C ABI. Standalone C++ callers can include
 * this header directly. */

#ifndef ADAPTQ_ERROR_DEFINED
#define ADAPTQ_ERROR_DEFINED

typedef enum {
  ADAPTQ_OK                = 0,
  ADAPTQ_ERR_ALLOC         = 1, /* memory allocation failed              */
  ADAPTQ_ERR_INVALID_ARG   = 2, /* NULL handle, out-of-range bits/dim    */
  ADAPTQ_ERR_OUT_OF_BOUNDS = 3, /* head_idx >= n_heads, dim > buffer cap */
  ADAPTQ_ERR_UNSUPPORTED   = 4  /* requested feature not compiled in     */
} adaptq_error_t;

/**
 * Returns a human-readable description of the last error on this thread.
 * Returns an empty string if no error has occurred.
 * Valid until the next AdapTQ call on this thread.
 */
const char *adaptq_last_error(void);

#endif /* ADAPTQ_ERROR_DEFINED */

#ifdef __cplusplus
} /* extern "C" */

#include <string>
#include <stdexcept>

namespace adaptq {

/**
 * C++ exception wrapper. Thrown by C++ API functions on error.
 * The C ABI never throws; it sets the thread-local error string instead.
 */
class AdaptQError : public std::runtime_error {
public:
    explicit AdaptQError(adaptq_error_t code, const std::string &msg)
        : std::runtime_error(msg), code_(code) {}

    adaptq_error_t code() const noexcept { return code_; }

private:
    adaptq_error_t code_;
};

} /* namespace adaptq */
#endif /* __cplusplus */

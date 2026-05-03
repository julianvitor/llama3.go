//go:build cgo

package llama

/*
#cgo CFLAGS: -O3
#cgo amd64 CFLAGS: -mavx2
#cgo arm64 CFLAGS: -march=armv8-a
#include "simd.h"
#cgo amd64 CFLAGS: -O3 -march=native

float dot_product(const float* a, const float* b, int n);
*/
import "C"
import "unsafe"

func dot(a, b []float32) float32 {
    return float32(C.dot_product(
        (*C.float)(unsafe.Pointer(&a[0])),
        (*C.float)(unsafe.Pointer(&b[0])),
        C.int(len(a)),
    ))
}
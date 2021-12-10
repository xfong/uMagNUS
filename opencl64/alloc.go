package opencl64

import (
	"log"
	//	"unsafe"

	data "github.com/seeder-research/uMagNUS/data64"
	"github.com/seeder-research/uMagNUS/cl"
)

// Wrapper for cu.MemAlloc, fatal exit on out of memory.
func MemAlloc(bytes int) *cl.MemObject {
	memObj, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, bytes)
	if err == cl.ErrMemObjectAllocationFailure || err == cl.ErrOutOfResources {
		log.Fatal(err)
	}
	if err != nil {
		panic(err)
	}
	return memObj
}

// Returns a copy of in, allocated on GPU.
func GPUCopy(in *data.Slice) *data.Slice {
	s := NewSlice(in.NComp(), in.Size())
	data.Copy(s, in)
	return s
}

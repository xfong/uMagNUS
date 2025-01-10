package opencl

import (
	"log"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
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
	var err error

	s := NewSlice(in.NComp(), in.Size())
	data.Copy(s, in)

	if err = ClCmdQueue.Flush(); err != nil {
		log.Printf("failed to flush queue in gpucopy: %+v \n, err")
	}

	if Synchronous {
		if err := cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
			log.Printf("failed to wait for copy to finish in gpucopy: %+v \n", err)
		}
	}
	return s
}

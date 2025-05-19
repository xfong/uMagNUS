package opencl

import (
	"log"
	"unsafe"

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

	// TODO: zero before returning (??)
	return memObj
}

func MemAllocFloat32(N int) *cl.MemObject {
	var event *cl.Event
	var queue *cl.CommandQueue

	initVal := float32(0.0)

	memObj, err := ClCtx.CreateEmptyBufferFloat32(cl.MemReadWrite, N)
	if err == cl.ErrMemObjectAllocationFailure || err == cl.ErrOutOfResources {
		log.Fatal(err)
	}
	if err != nil {
		panic(err)
	}
	bytes := N * SIZEOF_FLOAT32

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in memallocfloat32: %+v \n", err)
	}
	// execute
	if event, err = queue.EnqueueFillBuffer(memObj, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, bytes, evtWL); err != nil {
		log.Panicf("enqueuefillbuffer failed in memallocfloat32: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in memallocfloat32: %+v \n", err)
	}

	// set event markers
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	return memObj
}

func MemAllocFloat64(N int) *cl.MemObject {
	var event *cl.Event
	var queue *cl.CommandQueue

	initVal := float32(0.0)

	memObj, err := ClCtx.CreateEmptyBufferFloat64(cl.MemReadWrite, N)
	if err == cl.ErrMemObjectAllocationFailure || err == cl.ErrOutOfResources {
		log.Fatal(err)
	}
	if err != nil {
		panic(err)
	}
	bytes := N * SIZEOF_FLOAT64

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in memallocfloat64: %+v \n", err)
	}
	// execute
	if event, err = queue.EnqueueFillBuffer(memObj, unsafe.Pointer(&initVal), SIZEOF_FLOAT64, 0, bytes, evtWL); err != nil {
		log.Panicf("enqueuefillbuffer failed in memallocfloat64: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in memallocfloat64: %+v \n", err)
	}

	// set event markers
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	return memObj
}

// Returns a copy of in, allocated on GPU.
func GPUCopy(in *data.Slice) *data.Slice {
	s := NewSlice(in.NComp(), in.Size())
	data.Copy(s, in)
	return s
}

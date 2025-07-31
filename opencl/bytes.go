package opencl

// This file provides GPU byte slices, used to store regions.

import (
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	timer "github.com/seeder-research/uMagNUS/timer"
	util "github.com/seeder-research/uMagNUS/util"
)

// 3D byte slice, used for region lookup.
type Bytes struct {
	Ptr unsafe.Pointer
	Len int
}

// Construct new byte slice with given length,
// initialised to zeros.
func NewBytes(Len int) *Bytes {
	var err error
	var event cl.Event
	var queue *cl.CommandQueue

	ptr, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, Len)
	if err != nil {
		panic(err)
	}
	zeroPattern := uint8(0)

	if Synchronous { // debug
		WaitCommandSequence()
		timer.Start("newbytes")
	}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in newbytes: %+v \n", err)
	}
	// execute
	if event, err = queue.EnqueueFillBuffer(ptr, unsafe.Pointer(&zeroPattern), 1, 0, Len, evtWL); err != nil {
		log.Panicf("failed to fill buffer in newbytes: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in newbytes: %+v \n", err)
	}

	// set event markers
	log.Printf("in newbytes")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	if Synchronous { // debug
		WaitCommandSequence()
		timer.Stop("newbytes")
	}

	return &Bytes{unsafe.Pointer(ptr), Len}
}

// Upload src (host) to dst (gpu).
func (dst *Bytes) Upload(src []byte) {
	util.Argument(dst.Len == len(src))
	MemCpyHtoD(dst.Ptr, unsafe.Pointer(&src[0]), dst.Len)
}

// Copy on device: dst = src.
func (dst *Bytes) Copy(src *Bytes) {
	util.Argument(dst.Len == src.Len)
	MemCpy(dst.Ptr, src.Ptr, dst.Len)
}

// Copy to host: dst = src.
func (src *Bytes) Download(dst []byte) {
	util.Argument(src.Len == len(dst))
	MemCpyDtoH(unsafe.Pointer(&dst[0]), src.Ptr, src.Len)
}

// Set one element to value.
// data.Index can be used to find the index for x,y,z.
func (dst *Bytes) Set(index int, value byte) {
	var err error
	var event cl.Event
	var queue *cl.CommandQueue

	if index < 0 || index >= dst.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	src := value

	if Synchronous { // debug
		WaitCommandSequence()
		timer.Start("bytesSet")
	}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in bytes.set: %+v \n", err)
	}
	// execute
	if event, err = queue.EnqueueWriteBuffer((*cl.MemObject)(dst.Ptr), false, index, 1, unsafe.Pointer(&src), evtWL); err != nil {
		log.Panicf("failed to fill buffer in bytes.set: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in bytes.set: %+v \n", err)
	}

	// set event markers
	log.Printf("in bytes.set")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	if Synchronous { // debug
		WaitCommandSequence()
		timer.Stop("bytesSet")
	}

}

// Get one element.
// data.Index can be used to find the index for x,y,z.
// TODO: return as pointer and use events for synchronizing, which will allow us to wait on a
//
//	list of events rather than an individual event
func (src *Bytes) Get(index int) byte {
	var err error
	var event cl.Event
	var queue *cl.CommandQueue

	if index < 0 || index >= src.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	dst := make([]byte, 1)

	if Synchronous { // debug
		WaitCommandSequence()
		timer.Start("bytesGet")
	}

	// sequence command ccording to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in bytes.get: %+v \n", err)
	}
	// execute
	if event, err = queue.EnqueueReadBufferByte((*cl.MemObject)(src.Ptr), false, index, dst, evtWL); err != nil {
		log.Panicf("failed to read buffer in bytes.get: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in bytes.get: %+v \n", err)
	}

	// set event markers
	log.Printf("in bytes.get")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	// Must synchronize (needed??)
	WaitCommandSequence()

	if Synchronous {
		timer.Stop("bytesGet")
	}

	return dst[0]
}

// Frees the GPU memory and disables the slice.
func (b *Bytes) Free() {
	// Must synchronize
	WaitCommandSequence()

	if b.Ptr != nil {
		tmpObj := (*cl.MemObject)(b.Ptr)
		tmpObj.Release()
	}
	b.Ptr = nil
	b.Len = 0
}

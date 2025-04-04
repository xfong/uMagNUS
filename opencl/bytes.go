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
	var event *cl.Event

	ptr, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, Len)
	if err != nil {
		panic(err)
	}
	zeroPattern := uint8(0)

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in newbytes: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			log.Printf("failed to wait for last event in newbytes: %+v \n", err)
		}
		timer.Start("newbytes")
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueFillBuffer(ptr, unsafe.Pointer(&zeroPattern), 1, 0, Len, evtWL); err != nil {
		panic(err)
	}

	if err = ClCmdQueue.Flush(); err != nil {
		log.Panic("failed to flush queue in newbytes:", err)
	}

	// set event marker
	ClLastEvent = []*cl.Event{event}

	if Synchronous { // debug
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			log.Panic("WaitForEvents failed in newbytes:", err)
		}
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
	var event *cl.Event

	if index < 0 || index >= dst.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	src := value

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in bytes.set: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			log.Printf("failed to wait for last event in bytes.set: %+v \n", err)
		}
		timer.Start("bytesSet")
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(dst.Ptr), false, index, 1, unsafe.Pointer(&src), evtWL); err != nil {
		panic(err)
	}

	if err = ClCmdQueue.Flush(); err != nil {
		log.Panic("failed fllush queue in bytes.set:", err)
	}

	// set event marker
	ClLastEvent = []*cl.Event{event}

	if Synchronous { // debug
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			log.Panic("WaitForEvents failed in bytes.set:", err)
		}
		timer.Stop("bytesSet")
	}

}

// Get one element.
// data.Index can be used to find the index for x,y,z.
func (src *Bytes) Get(index int) byte {
	var err error
	var event *cl.Event

	if index < 0 || index >= src.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	dst := make([]byte, 1)

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in bytes.get: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			log.Printf("failed to wait for last event in bytes.get: %+v \n", err)
		}
		timer.Start("bytesGet")
	}

	// sequence command ccording to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueReadBufferByte((*cl.MemObject)(src.Ptr), false, index, dst, evtWL); err != nil {
		panic(err)
	}

	// set event marker
	ClLastEvent = []*cl.Event{event}

	// Must synchronize (needed??)
	if err = cl.WaitForEvents([](*cl.Event){event}); err != nil {
		log.Panic("WaitForEvents failed in Bytes.Get():", err)
	}

	if Synchronous {
		timer.Stop("bytesGet")
	}

	return dst[0]
}

// Frees the GPU memory and disables the slice.
func (b *Bytes) Free() {
	var err error

	// Must synchronize
	if err = ClCmdQueue.Finish(); err != nil {
		log.Printf("failed to wait for queue to finish in bytes.free: %+v \n", err)
	}
	if err = cl.WaitForEvents(ClLastEvent); err != nil {
		log.Printf("wait for last event in bytes.free failed: %+v \n", err)
	}

	ClLastEvent = make([]*cl.Event, 1)
	if ClLastEvent[0], err = ClCmdQueue.EnqueueMarkerWithWaitList(nil); err != nil {
		log.Printf("failed to enqueue marker in bytes.free(): %+v \n", err)
	}

	if b.Ptr != nil {
		tmpObj := (*cl.MemObject)(b.Ptr)
		tmpObj.Release()
	}
	b.Ptr = nil
	b.Len = 0
}

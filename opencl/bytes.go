package opencl

// This file provides GPU byte slices, used to store regions.

import (
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
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
	ptr, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, Len)
	if err != nil {
		panic(err)
	}
	zeroPattern := uint8(0)

	if Synchronous {
		if err := ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in newbytes: %+v \n", err)
		}
	}

	var event *cl.Event
	event, err = ClCmdQueue.EnqueueFillBuffer(ptr, unsafe.Pointer(&zeroPattern), 1, 0, Len, nil)
	if err != nil {
		panic(err)
	}
	if Synchronous {
		if err = cl.WaitForEvents([](*cl.Event){event}); err != nil {
			log.Panic("WaitForEvents failed in NewBytes:", err)
		}
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
	if index < 0 || index >= dst.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	src := value

	if Synchronous {
		if err := ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in bytes.set: %+v \n", err)
		}
	}

	event, err := ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(dst.Ptr), false, index, 1, unsafe.Pointer(&src), nil)
	if err != nil {
		panic(err)
	}

	if Synchronous {
		if err = cl.WaitForEvents([](*cl.Event){event}); err != nil {
			log.Panic("WaitForEvents failed in Bytes.Set():", err)
		}
	}
}

// Get one element.
// data.Index can be used to find the index for x,y,z.
func (src *Bytes) Get(index int) byte {
	if index < 0 || index >= src.Len {
		log.Panic("Bytes.Set: index out of range:", index)
	}
	dst := make([]byte, 1)

	var err error
	var event *cl.Event
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in bytes.get: %+v \n", err)
		}
	}

	event, err = ClCmdQueue.EnqueueReadBufferByte((*cl.MemObject)(src.Ptr), false, index, dst, nil)
	if err != nil {
		panic(err)
	}
	// Must synchronize
	if err = cl.WaitForEvents([](*cl.Event){event}); err != nil {
		log.Panic("WaitForEvents failed in Bytes.Get():", err)
	}
	return dst[0]
}

// Frees the GPU memory and disables the slice.
func (b *Bytes) Free() {
	// Must synchronize
	if err := ClCmdQueue.Finish(); err != nil {
		log.Printf("failed to wait for queue to finish in bytes.free: %+v \n", err)
	}

	if b.Ptr != nil {
		tmpObj := (*cl.MemObject)(b.Ptr)
		tmpObj.Release()
	}
	b.Ptr = nil
	b.Len = 0
}

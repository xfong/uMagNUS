package opencl

// Pool of re-usable GPU buffers.
// Synchronization subtlety:
// async kernel launches mean a buffer may already be recycled when still in use.
// That should be fine since the next launch run in the same stream (0), and will
// effectively wait for the previous operation on the buffer.

import (
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
)

var (
	buf_pool  = make(map[int][]unsafe.Pointer)    // pool of GPU buffers indexed by size
	buf_check = make(map[unsafe.Pointer]struct{}) // checks if pointer originates here to avoid unintended recycle
)

const buf_max = 100 // maximum number of buffers to allocate (detect memory leak early)

// Returns a GPU slice for temporary use. To be returned to the pool with Recycle
func Buffer(nComp int, size [3]int) *data.Slice {
	var err error
	var event *cl.Event
	var queue *cl.CommandQueue

	if Synchronous {
		WaitCommandSequence()
	}

	ptrs := make([]unsafe.Pointer, nComp)

	// re-use as many buffers as possible form our stack
	N := prod(size)
	bytes := N * SIZEOF_FLOAT32
	initVal := float32(0.0)
	pool := buf_pool[N]
	nFromPool := iMin(nComp, len(pool))
	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	buf_pool[N] = pool[:len(pool)-nFromPool]

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// allocate as much new memory as needed
	evtList := []*cl.Event{}
	j := int(0)
	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic("too many buffers in use, possible memory leak")
		}
		tmpPtr, err := ClCtx.CreateEmptyBufferFloat32(cl.MemReadWrite, N)
		if err != nil {
			panic(err)
		}
		ptrs[i] = unsafe.Pointer(tmpPtr)

		// create command queue and execute
		if queue, err = CreateCommandQueue(); err != nil {
			log.Panicf("failed to create command queue in buffer: %+v \n", err)
		}
		// execute
		if event, err = queue.EnqueueFillBuffer(tmpPtr, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, bytes, evtWL); err != nil {
			log.Panicf("CreateEmptyBuffer failed in buffer: %+v \n", err)
		}
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as mine

		if err = queue.Release(); err != nil { // implicit flush
			log.Panicf("failed to release queue in buffer: %+v \n", err)
		}
		InsertEventToCmdSeqTail(event)
		evtList = append(evtList, event)

		// Synchronize
		if Synchronous {
			WaitCommandSequence()
		}
	}

	UpdateLatestCmdList(evtList)

	outBuffer := data.SliceFromPtrs(size, data.GPUMemory, ptrs)
	return outBuffer
}

// Returns a buffer obtained from GetBuffer to the pool.
func Recycle(s *data.Slice) {
	N := s.Len()
	pool := buf_pool[N]
	// put each component buffer back on the stack
	for i := 0; i < s.NComp(); i++ {
		ptr := s.DevPtr(i)
		if ptr == unsafe.Pointer(uintptr(0)) {
			continue
		}
		if _, ok := buf_check[ptr]; !ok {
			log.Panic("recycle: was not obtained with getbuffer")
		}
		pool = append(pool, ptr)
	}
	s.Disable() // make it unusable, protect against accidental use after recycle
	buf_pool[N] = pool
}

// Frees all buffers. Called after mesh resize.
func FreeBuffers() {
	var err error

	// synchronize to all events
	WaitCommandSequence()
	for _, size := range buf_pool {
		for i := range size {
			tmpObj := (*cl.MemObject)(size[i])
			tmpObj.Release()
			size[i] = nil
		}
	}
	buf_pool = make(map[int][]unsafe.Pointer)
	buf_check = make(map[unsafe.Pointer]struct{})

}

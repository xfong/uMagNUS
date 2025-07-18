package opencl

import (
	"fmt"
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
	timer "github.com/seeder-research/uMagNUS/timer"
	util "github.com/seeder-research/uMagNUS/util"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, size [3]int) *data.Slice {
	return newSlice(nComp, size, data.GPUMemory)
}

func newSlice(nComp int, size [3]int, memType int8) *data.Slice {
	var err error
	var fillWait []cl.Event
	var tmp_buf *cl.MemObject
	var queue cl.CommandQueue
	var event cl.Event

	length := prod(size)
	bytes := length * SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	initVal := float32(0.0)
	fillWait = []cl.Event{}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	for c := range ptrs {
		tmp_buf = new(cl.MemObject)
		*tmp_buf, err = ClCtx.CreateEmptyBuffer(cl.MemReadWrite, bytes)
		if err != nil {
			fmt.Printf("CreateEmptyBuffer failed in newSlice: %+v \n", err)
		}
		ptrs[c] = unsafe.Pointer(tmp_buf)

		if Synchronous { // debug
			WaitCommandSequence()
		}

		// create command queue and zero the buffer (needed??)
		if queue, err = CreateCommandQueue(); err != nil {
			log.Fatalf("failed to create command queue in newslice: %+v \n", err)
		}

		// execute
		if event, err = queue.EnqueueFillBuffer(*tmp_buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, bytes, evtWL); err != nil {
			log.Fatalf("EnqueueFillBuffer failed in newslice: %+v \n", err)
		}

		if err = queue.Release(); err != nil { // implicit flush
			log.Fatalf("failed to release queue in newSlice: %+v \n", err)
		}

		// set event marker
		log.Println("in newslice")
		InsertEventToCmdSeqTail(event)
		fillWait = append(fillWait, event)

		if Synchronous { // debug
			WaitCommandSequence()
		}

	}

	// set event marker
	UpdateLatestCmdList(fillWait)

	dataPtr := data.SliceFromPtrs(size, memType, ptrs)
	return dataPtr
}

// wrappers for data.EnableGPU arguments

func memFree(ptr unsafe.Pointer) {
	if ptr != nil {
		buf := (*cl.MemObject)(ptr)
		buf.Release()
	}
}

func MemCpyDtoH(dst, src unsafe.Pointer, bytes int) {
	var err error
	var event cl.Event
	var queue cl.CommandQueue

	// debug
	if Synchronous {
		WaitCommandSequence()
		timer.Start("memcpyDtoH")
	}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in memcpyDtoH: %+v \n", err)
	}
	// execute
	log.Printf("d2h event: %+v \n", event)
	src_ := (*cl.MemObject)(src)
	if event, err = queue.EnqueueReadBuffer(*src_, false, 0, bytes, dst, evtWL); err != nil {
		log.Panicf("EnqueueReadBuffer in memcpyDtoH failed: %+v \n", err)
	}
	log.Printf("d2h event (after): %+v \n", event)

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in memcpyDtoH: %+v \n", err)
	}

	// set event markers
	log.Println("in memcpyd2h")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	// sync copy (needed??)
	if err = WaitLatestCmd(); err != nil {
		fmt.Printf("wait for last event in memcpyDtoH failed: %+v \n", err)
	}

	// debug
	if Synchronous {
		timer.Stop("memcpyDtoH")
	}
}

func MemCpyHtoD(dst, src unsafe.Pointer, bytes int) {
	var err error
	var event cl.Event
	var queue cl.CommandQueue

	// debug
	if Synchronous {
		WaitCommandSequence()
		timer.Start("memcpyHtoD")
	}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in memcpyHtoD: %+v \n", err)
	}
	dst_ := (*cl.MemObject)(dst)
	if event, err = queue.EnqueueWriteBuffer(*dst_, false, 0, bytes, src, evtWL); err != nil {
		log.Panicf("EnqueueWriteBuffer in memcpyHtoD failed: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in memcpyHtoD: %+v \n", err)
	}

	// set event marker
	log.Println("in memcpyh2d")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	if Synchronous {
		// sync copy
		if err = WaitLatestCmd(); err != nil {
			fmt.Printf("wait for last event in memcpyHtoD failed: %+v \n", err)
		}
		timer.Stop("memcpyHtoD")
	}
}

func MemCpy(dst, src unsafe.Pointer, bytes int) {
	var err error
	var event cl.Event
	var queue cl.CommandQueue

	// debug
	if Synchronous {
		WaitCommandSequence()
		timer.Start("memcpy")
	}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in memcpy: %+v \n", err)
	}
	// execute
	dst_, src_ := (*cl.MemObject)(dst), (*cl.MemObject)(src)
	if event, err = queue.EnqueueCopyBuffer(*src_, *dst_, 0, 0, bytes, evtWL); err != nil {
		log.Panicf("EnqueueCopyBuffer in memcpy failed: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in memcpy: %+v \n", err)
	}

	// set event markers
	log.Println("in memcpy")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	if Synchronous {
		// sync copy
		if err = WaitLatestCmd(); err != nil {
			fmt.Printf("wait for last event in memcpy failed: %+v \n", err)
		}
		timer.Stop("memcpy")
	}
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
	var err error
	var event cl.Event
	var queue cl.CommandQueue

	// debug
	if Synchronous {
		WaitCommandSequence()
		timer.Start("memset")
	}

	util.Argument(len(val) == s.NComp())

	// sequence command according to queue
	evtWL := GetLatestCmd()

	evtList := []cl.Event{}
	for c, v := range val {

		// create command queue and execute
		if queue, err = CreateCommandQueue(); err != nil {
			log.Panicf("failed to create command queue in memset: %+v \n", err)
		}
		// execute
		buf_ptr := (*cl.MemObject)(s.DevPtr(c))
		if event, err = queue.EnqueueFillBuffer(*buf_ptr, unsafe.Pointer(&v), SIZEOF_FLOAT32, 0, s.Len()*SIZEOF_FLOAT32, evtWL); err != nil {
			log.Panicf("EnqueueFillBuffer in memset failed: %+v \n", err)
		}

		if err = queue.Release(); err != nil { // implicit flush
			log.Panicf("failed to release queue in memset: %+v \n", err)
		}

		// set event markers
		evtList = append(evtList, event)
		log.Println("in memset")
		InsertEventToCmdSeqTail(event)

		if Synchronous { // debug
			WaitCommandSequence()
		}
	}

	// set event markers
	UpdateLatestCmdList(evtList)

	// debug
	if Synchronous {
		timer.Stop("memset")
	}
}

// Set all elements of all components to zero.
func Zero(s *data.Slice) {
	Memset(s, make([]float32, s.NComp())...)
}

func SetCell(s *data.Slice, comp int, ix, iy, iz int, value float32) {
	SetElem(s, comp, s.Index(ix, iy, iz), value)
}

func SetElem(s *data.Slice, comp int, index int, value float32) {
	var err error
	var event cl.Event
	var queue cl.CommandQueue

	f := value

	if Synchronous { // debug
		WaitCommandSequence()
		timer.Start("setelem")
	}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in setelem: %+v \n", err)
	}
	// execute
	buf_ptr := (*cl.MemObject)(s.DevPtr(comp))
	if event, err = queue.EnqueueWriteBuffer(*buf_ptr, false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), evtWL); err != nil {
		log.Panicf("setelem failed: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in setelem: %+v \n", err)
	}

	// set event markers
	log.Println("in setelem")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	if Synchronous { // debug
		if err = WaitLatestCmd(); err != nil {
			fmt.Printf("wait for last marker in setelem failed: %+v \n", err)
		}
		timer.Stop("setelem")
	}
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var err error
	var event cl.Event
	var queue cl.CommandQueue
	var f float32

	if Synchronous { // debug
		WaitCommandSequence()
		timer.Start("getelem")
	}

	// sequence command according to queue
	evtWL := GetLatestCmd()

	// create command queue and execute
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in getelem: %+v \n", err)
	}
	// execute
	buf_ptr := (*cl.MemObject)(s.DevPtr(comp))
	if event, err = queue.EnqueueReadBuffer(*buf_ptr, false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), evtWL); err != nil {
		log.Panicf("EnqueueReadBuffer failed: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in getelem: %+v \n", err)
	}

	// set event markers
	log.Println("in getelem")
	InsertEventToCmdSeqTail(event)
	UpdateLatestCmdSingle(event)

	// Must sync (needed??)
	if err = WaitLatestCmd(); err != nil {
		fmt.Printf("failed to wait for last marker in getelem: %+v \n", err)
	}

	if Synchronous { // debug
		timer.Stop("getelem")
	}
	return f
}

func GetCell(s *data.Slice, comp, ix, iy, iz int) float32 {
	return GetElem(s, comp, s.Index(ix, iy, iz))
}

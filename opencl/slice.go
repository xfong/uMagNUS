package opencl

import (
	"fmt"
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
	var fillWait []*cl.Event
	var tmp_buf *cl.MemObject

	length := prod(size)
	bytes := length * SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	initVal := float32(0.0)
	fillWait = make([]*cl.Event, nComp)
	for c := range ptrs {
		tmp_buf, err = ClCtx.CreateEmptyBuffer(cl.MemReadWrite, bytes)
		if err != nil {
			fmt.Printf("CreateEmptyBuffer failed in newSlice: %+v \n", err)
		}
		ptrs[c] = unsafe.Pointer(tmp_buf)

		if Synchronous { // debug
			if err = ClCmdQueue.Finish(); err != nil {
				fmt.Printf("failed to wait for queue to finish in newslice: %+v \n", err)
			}
			if err = cl.WaitForEvents(ClLastEvent); err != nil {
				fmt.Printf("failed to wait for last event in newslice: %+v \n", err)
			}
		}

		// sequence command according to queue
		evtWL := ClLastEvent

		// execute (needed??)
		if fillWait[c], err = ClCmdQueue.EnqueueFillBuffer(tmp_buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, bytes, evtWL); err != nil {
			fmt.Printf("EnqueueFillBuffer failed in newSlice: %+v \n", err)
		}

		if err = ClCmdQueue.Flush(); err != nil {
			fmt.Printf("flush queue in newSlice failed: %+v \n", err)
		}

		if Synchronous { // debug
			if err = cl.WaitForEvents([]*cl.Event{fillWait[c]}); err != nil {
				fmt.Printf("Wait for event in newSlice failed: %+v \n", err)
			}
		}

	}

	// set event marker
	ClLastEvent = fillWait

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
	var event *cl.Event

	// debug
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in MemCpyDtoH: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("failed to wait for last event in MemCpyDtoH: %+v \n", err)
		}
		timer.Start("memcpyDtoH")
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueReadBuffer((*cl.MemObject)(src), false, 0, bytes, dst, evtWL); err != nil {
		fmt.Printf("EnqueueReadBuffer in memcpyDtoH failed: %+v \n", err)
	}

	// sync copy (needed??)
	if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
		fmt.Printf("WaitForEvents in memcpyDtoH failed: %+v \n", err)
	}

	// debug
	if Synchronous {
		timer.Stop("memcpyDtoH")
	}

	// reset event marker
	ClLastEvent = make([]*cl.Event, 1)
	if ClLastEvent[0], err = ClCmdQueue.EnqueueMarkerWithWaitList(nil); err != nil {
		fmt.Printf("failed to enqueue marker in memcpyDtoH: %+v \n", err)
	}
}

func MemCpyHtoD(dst, src unsafe.Pointer, bytes int) {
	var err error
	var event *cl.Event

	// debug
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in memcpyHtoD: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("failed to wait for last event in memcpyHtoD: %+v \n", err)
		}
		timer.Start("memcpyHtoD")
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(dst), false, 0, bytes, src, evtWL); err != nil {
		fmt.Printf("EnqueueWriteBuffer in memcpyHtoD failed: %+v \n", err)
	}

	if err = ClCmdQueue.Flush(); err != nil {
		fmt.Printf("flush queue in memcpyHtoD failed: %+v \n", err)
	}

	// set event marker
	ClLastEvent = []*cl.Event{event}

	if Synchronous {
		// sync copy
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("WaitForEvents in memcpyHtoD failed: %+v \n", err)
		}
		timer.Stop("memcpyHtoD")
	}
}

func MemCpy(dst, src unsafe.Pointer, bytes int) {
	var err error
	var event *cl.Event

	// debug
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in memcpy: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("failed to wait for last event in memcpy: %+v \n", err)
		}
		timer.Start("memcpy")
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(src), (*cl.MemObject)(dst), 0, 0, bytes, evtWL); err != nil {
		fmt.Printf("EnqueueCopyBuffer in memcpy failed: %+v \n", err)
	}

	if err = ClCmdQueue.Flush(); err != nil {
		fmt.Printf("flush queue in memcpy failed: %+v \n", err)
	}
	ClLastEvent = make([]*cl.Event, 1)
	ClLastEvent[0] = event

	if Synchronous {
		// sync copy
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("WaitForEvents in memcpy failed: %+v \n", err)
		}
		timer.Stop("memcpy")
	}
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
	var err error
	var event *cl.Event

	// debug
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in beginning of memset: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("failed to wait for last event in memset: %+v \n", err)
		}
		timer.Start("memset")
	}

	util.Argument(len(val) == s.NComp())

	evtList := []*cl.Event{}
	for c, v := range val {

		// sequence command according to queue
		evtWL := ClLastEvent

		// execute
		if event, err = ClCmdQueue.EnqueueFillBuffer((*cl.MemObject)(s.DevPtr(c)), unsafe.Pointer(&v), SIZEOF_FLOAT32, 0, s.Len()*SIZEOF_FLOAT32, evtWL); err != nil {
			fmt.Printf("EnqueueFillBuffer in memset failed: %+v \n", err)
		}

		if err = ClCmdQueue.Flush(); err != nil {
			fmt.Printf("flush queue in memset failed: %+v \n", err)
		}

		// set event marker
		evtList = append(evtList, event)

		if Synchronous { // debug
			if err = cl.WaitForEvents([]*cl.Event{event}); err != nil {
				fmt.Printf("WaitForEvents in memset failed: %+v \n", err)
			}
		}
	}

	// set event marker
	ClLastEvent = evtList

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
	var event *cl.Event

	f := value

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in setelem: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("failed to wait for last event in setelem: %+v \n", err)
		}
		timer.Start("setelem")
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(s.DevPtr(comp)), false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), evtWL); err != nil {
		fmt.Printf("setelem failed: %+v \n", err)
	}

	if err = ClCmdQueue.Flush(); err != nil {
		fmt.Printf("flush queue in setelem failed: %+v \n", err)
	}

	// set event marker
	ClLastEvent = []*cl.Event{event}

	if Synchronous { // debug
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("WaitForEvents in setelem failed: %+v \n", err)
		}
		timer.Stop("setelem")
	}
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var err error
	var event *cl.Event
	var f float32

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("wait for queue to finish in getelem failed: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("wait for last event in getelem failed: %+v \n", err)
		}
		timer.Start("getelem")
	}

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if event, err = ClCmdQueue.EnqueueReadBuffer((*cl.MemObject)(s.DevPtr(comp)), false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), evtWL); err != nil {
		fmt.Printf("EnqueueReadBuffer failed: %+v \n", err)
	}

	// set event marker
	ClLastEvent = []*cl.Event{event}

	// Must sync
	if err = cl.WaitForEvents(ClLastEvent); err != nil {
		fmt.Printf("WaitForEvents in GetElem failed: %+v \n", err)
	}

	if Synchronous { // debug
		timer.Stop("getelem")
	}
	return f
}

func GetCell(s *data.Slice, comp, ix, iy, iz int) float32 {
	return GetElem(s, comp, s.Index(ix, iy, iz))
}

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

// GPU slice is initialized to zero.
// TODO: create own CommandQueue (??)
// TODO: Can be completely async since there is no data dependency
func newSlice(nComp int, size [3]int, memType int8) *data.Slice {
	var tmpEvent *cl.Event

	length := prod(size)
	bytes := length * SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	initVal := float32(0.0)

	for c := range ptrs {
		tmp_buf, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, bytes)
		if err != nil {
			fmt.Printf("CreateEmptyBuffer failed in newslice: %+v \n", err)
		}
		ptrs[c] = unsafe.Pointer(tmp_buf)
		tmpEvent, err = ClCmdQueue[c].EnqueueFillBuffer(tmp_buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, bytes, nil)
		if err != nil {
			fmt.Printf("CreateEmptyBuffer failed in newslice: %+v \n", err)
		}
		if err = ClCmdQueue[c].Flush(); err != nil {
			fmt.Printf("failed to flush queue in newslice: %+v \n", err)
		}
		if Synchronous {
			if err = cl.WaitForEvents([]*cl.Event{tmpEvent}); err != nil {
				fmt.Printf("Wait for EnqueueFillBuffer failed in newslice: %+v \n", err)
			}
		} else {
			ClLastEvent = append(ClLastEvent, tmpEvent)
		}
	}

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

// TODO: create own CommandQueue (??)
func MemCpyDtoH(dst, src unsafe.Pointer, bytes int) {
	var err error
	var tmpEvent *cl.Event

	tmpEvents := LastEventToWaitList()

	// debug
	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			fmt.Printf("failed to wait for last event to finish in memcpyDtoH: %+v \n", err)
		}
		timer.Start("memcpyDtoH")
	}

	// execute
	if tmpEvent, err = ClCmdQueue[0].EnqueueReadBuffer((*cl.MemObject)(src), false, 0, bytes, dst, tmpEvents); err != nil {
		fmt.Printf("EnqueueReadBuffer failed in memcpyDtoH: %+v \n", err)
	}

	// sync copy
	if err = cl.WaitForEvents([]*cl.Event{tmpEvent}); err != nil {
		fmt.Printf("WaitForEvents in memcpyDtoH failed: %+v \n", err)
	}

	// debug
	if Synchronous {
		timer.Stop("memcpyDtoH")
	}

	ClLastEvent = []*cl.Event{}
}

// TODO: create own CommandQueue (??)
func MemCpyHtoD(dst, src unsafe.Pointer, bytes int) {
	var err error
	var tmpEvent *cl.Event

	tmpEvents := LastEventToWaitList()

	// debug
	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			fmt.Printf("failed to wait for last event in memcpyHtoD: %+v \n", err)
		}
		timer.Start("memcpyHtoD")
	}

	// execute
	tmpEvent, err = ClCmdQueue[0].EnqueueWriteBuffer((*cl.MemObject)(dst), false, 0, bytes, src, tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueWriteBuffer failed in memcpyHtoD: %+v \n", err)
	}

	if err = ClCmdQueue[0].Flush(); err != nil {
		fmt.Printf("failed to flush queue in memcpyHtoD: %+v \n", err)
	}

	if Synchronous {
		// sync copy
		if err = WaitLastEvent(); err != nil {
			fmt.Printf("WaitForEvents in memcpyHtoD failed: %+v \n", err)
		}
		ClLastEvent = []*cl.Event{}
		timer.Stop("memcpyHtoD")
	} else {
		ClLastEvent = []*cl.Event{tmpEvent}
	}
}

// TODO: create own CommandQueue (??)
func MemCpy(dst, src unsafe.Pointer, bytes int) {
	var err error
	var tmpEvent *cl.Event

	tmpEvents := LastEventToWaitList()

	// debug
	if Synchronous {
		if tmpEvents != nil {
			if err = cl.WaitForEvents(tmpEvents); err != nil {
				fmt.Printf("failed to wait for last event to finish in memcpy: %+v \n", err)
			}
		}
		timer.Start("memcpy")
	}

	// execute
	tmpEvent, err = ClCmdQueue[0].EnqueueCopyBuffer((*cl.MemObject)(src), (*cl.MemObject)(dst), 0, 0, bytes, tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueCopyBuffer failed in memcpy: %+v \n", err)
	}

	if Synchronous {
		// sync copy
		if err = cl.WaitForEvents([]*cl.Event{tmpEvent}); err != nil {
			fmt.Printf("WaitForEvents in memcpy failed: %+v \n", err)
		}
		timer.Stop("memcpy")
		ClLastEvent = []*cl.Event{}
	} else {
		ClLastEvent = append(ClLastEvent, tmpEvent)
	}
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
	var err error
	var tmpEvent *cl.Event

	// debug
	if Synchronous {
		if err = WaitLastEvent(); err != nil {
			fmt.Printf("failed to wait for last event to finish in beginning of memset: %+v \n", err)
		}
		timer.Start("memset")
	}

	util.Argument(len(val) == s.NComp())

	tmpEvents := LastEventToWaitList()
	ClLastEvent = []*cl.Event{}
	for c, v := range val {
		tmpEvent, err := ClCmdQueue[c].EnqueueFillBuffer((*cl.MemObject)(s.DevPtr(c)), unsafe.Pointer(&v), SIZEOF_FLOAT32, 0, s.Len()*SIZEOF_FLOAT32, tmpEvents)
		if err != nil {
			fmt.Printf("EnqueueFillBuffer failed in memset: %+v \n", err)
		}
		if err = ClCmdQueue[c].Flush(); err != nil {
			fmt.Printf("failed to flush queue in memset: %+v \n", err)
		}

		if Synchronous { // debug
			if err = cl.WaitForEvents([]*cl.Event{tmpEvent}); err != nil {
				fmt.Printf("WaitForEvents failed in memset: %+v \n", err)
			}
		} else {
			ClLastEvent = append(ClLastEvent, tmpEvent)
		}
	}

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
	var tmpEvent *cl.Event

	f := value

	tmpEvents := LastEventToWaitList()
	ClLastEvent = []*cl.Event{}

	tmpEvent, err = ClCmdQueue[0].EnqueueWriteBuffer((*cl.MemObject)(s.DevPtr(comp)), false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueWriteBuffer failed in setelem: %+v \n", err)
		return
	}
	if err = ClCmdQueue[0].Flush(); err != nil {
		fmt.Printf("failed to flush queue in setelem: %+v \n", err)
	}

	// debug
	if Synchronous {
		if err = cl.WaitForEvents([]*cl.Event{tmpEvent}); err != nil {
			fmt.Printf("WaitForEvents in setelem failed: %+v \n", err)
		}
	} else {
		ClLastEvent = []*cl.Event{tmpEvent}
	}
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var err error
	var f float32
	var tmpEvent *cl.Event

	tmpEvents := LastEventToWaitList()

	tmpEvent, err = ClCmdQueue[0].EnqueueReadBuffer((*cl.MemObject)(s.DevPtr(comp)), false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueReadBuffer failed in getelem: %+v \n", err)
	}
	if err = ClCmdQueue[0].Flush(); err != nil {
		fmt.Printf("failed to flush queue in getelem: %+v \n", err)
	}

	ClLastEvent = []*cl.Event{}
	// Must sync
	if err = cl.WaitForEvents([](*cl.Event){tmpEvent}); err != nil {
		fmt.Printf("WaitForEvents in getelem failed: %+v \n", err)
	}

	return f
}

func GetCell(s *data.Slice, comp, ix, iy, iz int) float32 {
	return GetElem(s, comp, s.Index(ix, iy, iz))
}

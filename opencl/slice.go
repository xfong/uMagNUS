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
	var tmpEvents []*cl.Event

	length := prod(size)
	bytes := length * SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	initVal := float32(0.0)
	tmpEvents = nil

	for c := range ptrs {
		tmp_buf, err := ClCtx.CreateEmptyBuffer(cl.MemReadWrite, bytes)
		if err != nil {
			fmt.Printf("CreateEmptyBuffer failed: %+v \n", err)
		}
		ptrs[c] = unsafe.Pointer(tmp_buf)
		if ClLastEvent != nil {
			tmpEvents = []*cl.Event{ClLastEvent}
		}
		ClLastEvent, err = ClCmdQueue.EnqueueFillBuffer(tmp_buf, unsafe.Pointer(&initVal), SIZEOF_FLOAT32, 0, bytes, tmpEvents)
		if err != nil {
			fmt.Printf("CreateEmptyBuffer failed: %+v \n", err)
		}
		if err = ClCmdQueue.Flush(); err != nil {
			fmt.Printf("failed to flush queue in newSlice: %+v \n", err)
		}
		if Synchronous {
			if err = cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
				fmt.Printf("Wait for EnqueueFillBuffer failed: %+v \n", err)
			}
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

func MemCpyDtoH(dst, src unsafe.Pointer, bytes int) {
	var err error
	var tmpEvents []*cl.Event

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}

	// debug
	if Synchronous {
		if tmpEvents != nil {
			if err = cl.WaitForEvents(tmpEvents); err != nil {
				fmt.Printf("failed to wait for last event to finish: %+v \n", err)
			}
		}
		timer.Start("memcpyDtoH")
	}

	// execute
	if ClLastEvent, err = ClCmdQueue.EnqueueReadBuffer((*cl.MemObject)(src), false, 0, bytes, dst, tmpEvents); err != nil {
		fmt.Printf("EnqueueReadBuffer failed: %+v \n", err)
	}

	// sync copy
	if err = cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
		fmt.Printf("WaitForEvents in memcpyDtoH failed: %+v \n", err)
	}

	// debug
	if Synchronous {
		timer.Stop("memcpyDtoH")
	}
}

func MemCpyHtoD(dst, src unsafe.Pointer, bytes int) {
	var err error
	var tmpEvents []*cl.Event

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}

	// debug
	if Synchronous {
		if tmpEvents != nil {
			if err = cl.WaitForEvents(tmpEvents); err != nil {
				fmt.Printf("failed to wait for last event to finish: %+v \n", err)
			}
		}
		timer.Start("memcpyHtoD")
	}

	// execute
	ClLastEvent, err = ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(dst), false, 0, bytes, src, tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueWriteBuffer failed: %+v \n", err)
	}

	if Synchronous {
		// sync copy
		if err = cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
			fmt.Printf("WaitForEvents in memcpyHtoD failed: %+v \n", err)
		}
		timer.Stop("memcpyHtoD")
	}
}

func MemCpy(dst, src unsafe.Pointer, bytes int) {
	var err error
	var tmpEvents []*cl.Event

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}

	// debug
	if Synchronous {
		if tmpEvents != nil {
			if err = cl.WaitForEvents(tmpEvents); err != nil {
				fmt.Printf("failed to wait for last event to finish: %+v \n", err)
			}
		}
		timer.Start("memcpy")
	}

	// execute
	ClLastEvent, err = ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(src), (*cl.MemObject)(dst), 0, 0, bytes, tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueCopyBuffer failed: %+v \n", err)
	}

	if Synchronous {
		// sync copy
		if err = cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
			fmt.Printf("First WaitForEvents in memcpy failed: %+v \n", err)
		}
		timer.Stop("memcpy")
	}
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
	var err error
	var tmpEvents []*cl.Event

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}

	// debug
	if Synchronous {
		if tmpEvents != nil {
			if err = cl.WaitForEvents(tmpEvents); err != nil {
				fmt.Printf("failed to wait for last event to finish in beginning of memset: %+v \n", err)
			}
		}
		timer.Start("memset")
	}

	util.Argument(len(val) == s.NComp())

	for c, v := range val {
		ClLastEvent, err := ClCmdQueue.EnqueueFillBuffer((*cl.MemObject)(s.DevPtr(c)), unsafe.Pointer(&v), SIZEOF_FLOAT32, 0, s.Len()*SIZEOF_FLOAT32, tmpEvents)
		if err != nil {
			fmt.Printf("EnqueueFillBuffer failed: %+v \n", err)
		}
		if err = ClCmdQueue.Flush(); err != nil {
			fmt.Printf("failed to flush queue in memset: %+v \n", err)
		}
		tmpEvents = []*cl.Event{ClLastEvent}
		if Synchronous { // debug
			if err = cl.WaitForEvents(tmpEvents); err != nil {
				fmt.Printf("WaitForEvents failed in memset: %+v \n", err)
			}
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
	var tmpEvents []*cl.Event

	f := value

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}
	ClLastEvent, err = ClCmdQueue.EnqueueWriteBuffer((*cl.MemObject)(s.DevPtr(comp)), false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueWriteBuffer failed: %+v \n", err)
		return
	}
	if err = ClCmdQueue.Flush(); err != nil {
		fmt.Printf("failed to flush queue in enqueuewritebuffer: %+v \n", err)
	}
	if Synchronous {
		if err = cl.WaitForEvents([]*cl.Event{ClLastEvent}); err != nil {
			fmt.Printf("WaitForEvents in SetElem failed: %+v \n", err)
		}
	}
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var err error
	var f float32
	var tmpEvents []*cl.Event

	tmpEvents = nil
	if ClLastEvent != nil {
		tmpEvents = []*cl.Event{ClLastEvent}
	}
	ClLastEvent, err = ClCmdQueue.EnqueueReadBuffer((*cl.MemObject)(s.DevPtr(comp)), false, index*SIZEOF_FLOAT32, SIZEOF_FLOAT32, unsafe.Pointer(&f), tmpEvents)
	if err != nil {
		fmt.Printf("EnqueueReadBuffer failed: %+v \n", err)
	}
	if err = ClCmdQueue.Flush(); err != nil {
		fmt.Printf("failed to flush queue in enqueuewritebuffer: %+v \n", err)
	}

	// Must sync
	if err = cl.WaitForEvents([](*cl.Event){ClLastEvent}); err != nil {
		fmt.Printf("WaitForEvents in GetElem failed: %+v \n", err)
	}
	return f
}

func GetCell(s *data.Slice, comp, ix, iy, iz int) float32 {
	return GetElem(s, comp, s.Index(ix, iy, iz))
}

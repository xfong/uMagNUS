package opencl

import (
	"fmt"
	"log"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	util "github.com/seeder-research/uMagNUS/util"
)

// Type size in bytes
const (
	SIZEOF_FLOAT32    = 4
	SIZEOF_FLOAT64    = 8
	SIZEOF_COMPLEX64  = 8
	SIZEOF_COMPLEX128 = 16
)

// Assumes kernel arguments set prior to launch
func LaunchKernel(kernname string, gridDim, workDim []int, events []*cl.Event) *cl.Event {
	var err error
	var queue *cl.CommandQueue
	var KernEvent *cl.Event

	if KernList[kernname] == nil { // get kernel object
		util.Fatal("Kernel " + kernname + " does not exist!")
		return nil
	}

	if Debug { // debug
		fmt.Printf("Launching kernel: %+v with Grid = %+v and Block = %+v \n", kernname, gridDim, workDim)
	}

	if queue, err = CreateCommandQueue(); err != nil { // get command queue
		util.Fatal(err)
		return nil
	}

	// execute
	if KernEvent, err = queue.EnqueueNDRangeKernel(KernList[kernname], nil, gridDim, workDim, events); err != nil {
		util.Fatal(err)
		return nil
	}

	if err = queue.Release(); err != nil { // implicit flush to device
		log.Panicf("failed to release queue: %+v \n", err)
	}

	return KernEvent
}

func SetKernelArgWrapper(kernname string, index int, arg interface{}) {
	if KernList[kernname] == nil {
		util.Fatal("Kernel " + kernname + " does not exist!")
	}
	switch val := arg.(type) {
	default:
		if err := KernList[kernname].SetArg(index, val); err != nil {
			util.Fatal(err)
		}
	case unsafe.Pointer:
		memBufHandle, flag := arg.(unsafe.Pointer)
		if memBufHandle == unsafe.Pointer(uintptr(0)) {
			if err := KernList[kernname].SetArgUnsafe(index, 8, memBufHandle); err != nil {
				util.Fatal(err)
			}
		} else {
			if flag {
				if err := KernList[kernname].SetArg(index, (*cl.MemObject)(memBufHandle)); err != nil {
					util.Fatal(err)
				}
			} else {
				util.Fatal("Unable to change argument type to *cl.MemObject")
			}
		}
	case int:
		if err := KernList[kernname].SetArg(index, (int32)(val)); err != nil {
			util.Fatal(err)
		}
	}
}

func CreateCommandQueue() (*cl.CommandQueue, error) {
	if ClCtx == nil {
		log.Panicf("ClCtx (context) cannot be nil! \n")
		return nil, nil
	}
	if ClDevice == nil {
		log.Panicf("ClDevice (device) cannot be nil! \n")
		return nil, nil
	}
	return ClCtx.CreateCommandQueue(ClDevice, 0)
}

func WaitCommandSequence() {
	var err error
	if err = WaitLastMarker(); err != nil {
		fmt.Printf("failed to wait for event in WaitCommandSequence: %+v \n", err)
	}
}

func InitMarkers() {
	var err error
	var queue *cl.CommandQueue
	var marker *cl.Event

	if queue, err = CreateCommandQueue(); err != nil { // get queue
		log.Panicf("failed to create command queue in InitMarkers: %+v \n", err)
		return
	}

	// enqueue a marker with no dependencies that completes when executed
	if marker, err = queue.EnqueueMarkerWithWaitList(nil); err != nil {
		log.Panicf("failed to enqueue marker in InitMarkers: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in InitMarkers: %+v \n", err)
	}

	// update
	ClInitMarker, ClLastMarker = marker, marker
	ClLastEvent = []*cl.Event{marker}

}

func AddEventToSequence(ev *cl.Event) {
	var err error
	var queue *cl.CommandQueue
	var marker *cl.Event

	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in addeventtosequence: %+v \n", err)
		return
	}

	if marker, err = queue.EnqueueMarkerWithWaitList([]*cl.Event{ClLastMarker, ev}); err != nil {
		log.Panicf("failed to enqueue marker in addeventtosequence: %+v \n", err)
		return
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in AddEventToSequence: %+v \n", err)
	}

	ClLastMarker = marker
}

func WaitLastMarker() error {
	if ClLastMarker == nil {
		fmt.Printf("ClLastMarker cannot be nil in waitlastmarker! \n")
	}
	return cl.WaitForEvents([]*cl.Event{ClLastMarker})
}

func UpdateLastEventSingle(ev *cl.Event) {
	if ev == nil {
		fmt.Printf("ev cannot be nil in updatelasteventsingle! \n")
		return
	}

	ClLastEvent = []*cl.Event{ev}
}

func UpdateLastEventList(evList []*cl.Event) {
	if evList == nil {
		fmt.Printf("ev cannot be nil in updatelasteventlist! \n")
		return
	}

	if len(evList) > 0 {
		ClLastEvent = evList
	} else {
		if ClInitMarker != nil {
			ClLastEvent = []*cl.Event{ClInitMarker}
		} else {
			fmt.Printf("failed to update ClLastEvent in UpdateLastEventList! \n")
		}
	}
}

func WaitLastEvent() error {
	if ClLastEvent == nil {
		fmt.Printf("ClLastEvent cannot be nil in waitlastevent! \n")
	}

	if len(ClLastEvent) > 0 {
		return cl.WaitForEvents(ClLastEvent)
	}

	return nil
}

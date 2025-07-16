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

// sets arguments for kernel functions
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

// return a command queue for the initialized context nd device
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

// initialize cl.Event markers
func InitMarkers() {
	var err error
	var queue *cl.CommandQueue
	var marker *cl.Event

	if queue, err = CreateCommandQueue(); err != nil { // get queue
		log.Panicf("failed to create command queue in InitMarkers: %+v \n", err)
		return
	}

	// enqueue a marker with no dependencies that completes when executed
	if ClInitMarker, err = queue.EnqueueMarkerWithWaitList(nil); err != nil {
		log.Panicf("failed to enqueue marker (clinitmarker) in InitMarkers: %+v \n", err)
	}

	// enqueue a marker with no dependencies that completes when executed
	if ClCmdSeqTail, err = queue.EnqueueMarkerWithWaitList(nil); err != nil {
		log.Panicf("failed to enqueue marker (clcmdseqtail) in InitMarkers: %+v \n", err)
	}

	// enqueue a marker with no dependencies that completes when executed
	if marker, err = queue.EnqueueMarkerWithWaitList(nil); err != nil {
		log.Panicf("failed to enqueue marker in InitMarkers: %+v \n", err)
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in InitMarkers: %+v \n", err)
	}

	// update
	ClLatestCmd = []*cl.Event{marker}

}

// update ClCmdSeqTail to track event for an enqueued command that
// has been enqueued
func InsertEventToCmdSeqTail(ev *cl.Event) {
	var err error
	var queue *cl.CommandQueue
	var marker *cl.Event

	log.Println("attempting to update event to cmd seq...")
	marker = nil
	queue = nil
	if queue, err = CreateCommandQueue(); err != nil { // create queue
		log.Fatalf("failed to create command queue in addeventtosequence: %+v \n", err)
		return
	}

	// generate the new event marker
	log.Printf("input: %+v \n", ev)
	marker = ClCmdSeqTail
	tmpMarker, err2 := queue.EnqueueMarkerWithWaitList([]*cl.Event{marker, ev})
	log.Printf("previous tail: %+v \n", marker)
	log.Printf("current tail: %+v \n", tmpMarker)
	if err2 != nil {
		log.Fatalf("failed to enqueue marker in inserteventtocmdseqtail: %+v \n", err2)
		return
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Fatalf("failed to release queue in inserteventtocmdseqtail: %+v \n", err)
		return
	}

	// release the old marker to the memory will be deallocated before updating the tracker
	ClCmdSeqTail = tmpMarker
	log.Printf("new tail: %+v \n", ClCmdSeqTail)
	log.Printf("releasing: %+v \n", marker)
	if err = marker.Release(); err != nil {
		log.Fatalf("failed to release marker event in inserteventtocmdseqtail: %+v \n", err)
		return
	}

}

// wait on all enqueued commands to finish. Similar to waiting for command queue to finish
// except that commands have all been flushed to the device and so an event based queue
// is used to create a marker that can only successfully complete execution if all
// previously enqueued commands have completed successfully
func WaitCommandSequence() {
	if err := WaitCmdSeqTail(); err != nil {
		fmt.Printf("failed to wait for event in WaitCommandSequence: %+v \n", err)
	}
}

// wait for event in ClCmdSeqTail
func WaitCmdSeqTail() error {
	if ClCmdSeqTail == nil {
		fmt.Printf("ClCmdSeqTail cannot be nil in waitlastmarker! \n")
	}
	return cl.WaitForEvents([]*cl.Event{ClCmdSeqTail})
}

// update latest device command that was enqueued by a host function
func UpdateLatestCmdSingle(ev *cl.Event) {
	var err error

	if ev == nil {
		fmt.Printf("ev cannot be nil in updatelatestcmdsingle! \n")
		return
	}

	// release all previous events for commands that were enqueued
	// to deallocate memory (no longer need to track them within
	// the program)
	if err = ReleasePreviousDeviceCommandEvents(); err != nil {
		log.Printf("failed to release event in updatelatestcmdsingle: %+v \n", err)
	}

	// upate tracker
	ClLatestCmd = []*cl.Event{ev}
}

// update latest list of device commands that was enqueued by a host function
func UpdateLatestCmdList(evList []*cl.Event) {
	var err error

	// error check input to ensure it is neither a nil pointer nor
	// an empty list
	if evList == nil {
		fmt.Printf("evList cannot be nil in updatelatestcmdlist! \n")
		return
	} else {
		if len(evList) == 0 {
			fmt.Printf("evList cannot be empty in updatelatestcmdlist! \n")
			return
		}
	}

	// release all previous events for commands that were enqueued
	// to deallocate memory (no longer need to track them within
	// the program)
	if err = ReleasePreviousDeviceCommandEvents(); err != nil {
		log.Printf("failed to release event in updatelatesteventlist: %+v \n", err)
	}

	// update tracker
	ClLatestCmd = evList
}

// release all previous events for commands that were enqueued
// to deallocate memory (no longer need to track them within
// the program)
func ReleasePreviousDeviceCommandEvents() error {
	var err error
	for _, v := range ClLatestCmd {
		if v != ClInitMarker {
			if err = v.Release(); err != nil {
				log.Printf("failed to release event in releasepreviousdevicecommandEvents: %+v \n", err)
				return err
			}
		}
	}

	return nil
}

// wait for latest device commands enqueued by a host function
func WaitLatestCmd() error {
	// error check
	if ClLatestCmd == nil {
		fmt.Printf("ClLatestCmd cannot be nil in waitlastevent! \n")
	}

	// if ClLatestCmd has events, wait for them to complete
	if len(ClLatestCmd) > 0 {
		return cl.WaitForEvents(ClLatestCmd)
	}

	return nil
}

// get the latest device commands enqueued by a host function
func GetLatestCmd() []*cl.Event {
	return ClLatestCmd
}

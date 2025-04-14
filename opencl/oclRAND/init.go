package oclRAND

import (
	"github.com/seeder-research/uMagNUS/cl"
	"log"
	"unsafe"
)

type config struct {
	Grid, Block []int
}

var (
	Synchronous bool
	clCtx       = (*cl.Context)(nil)
	clDevice    = (*cl.Device)(nil)
	KernList    = map[string]*cl.Kernel{} // Store pointers to all compiled kernels
)

func Init(ctx *cl.Context, dev *cl.Device, synch bool, kList map[string]*cl.Kernel) {
	Synchronous = synch
	clCtx = ctx
	clDevice = dev
	KernList = kList
}

func LaunchKernel(kernname string, gridDim, workDim []int, events []*cl.Event) *cl.Event {
	var err error
	var event *cl.Event
	var queue *cl.CommandQueue

	if KernList[kernname] == nil {
		log.Panic("Kernel " + kernname + " does not exist!")
		return nil
	}

	// get command queue
	if queue, err = CreateCommandQueue(); err != nil {
		log.Panicf("failed to create command queue in oclrand.launchkernel: %+v \n", err)
	}

	// execute
	if event, err = queue.EnqueueNDRangeKernel(KernList[kernname], nil, gridDim, workDim, events); err != nil {
		log.Fatal(err)
		return nil
	}

	if err = queue.Release(); err != nil { // implicit flush
		log.Panicf("failed to release queue in oclrand.launchkernel: %+v \n", err)
	}

	return event
}

func SetKernelArgWrapper(kernname string, index int, arg interface{}) {
	if KernList[kernname] == nil {
		log.Panic("Kernel " + kernname + " does not exist!")
	}
	switch val := arg.(type) {
	default:
		if err := KernList[kernname].SetArg(index, val); err != nil {
			log.Fatal(err)
		}
	case unsafe.Pointer:
		memBufHandle, flag := arg.(unsafe.Pointer)
		if memBufHandle == unsafe.Pointer(uintptr(0)) {
			if err := KernList[kernname].SetArgUnsafe(index, 8, memBufHandle); err != nil {
				log.Fatal(err)
			}
		} else {
			if flag {
				if err := KernList[kernname].SetArg(index, (*cl.MemObject)(memBufHandle)); err != nil {
					log.Fatal(err)
				}
			} else {
				log.Fatal("Unable to change argument type to *cl.MemObject")
			}
		}
	case int:
		if err := KernList[kernname].SetArg(index, (int32)(val)); err != nil {
			log.Fatal(err)
		}
	}
}

func CreateCommandQueue() (*cl.CommandQueue, error) {
	if clCtx == nil {
		log.Panicf("clCtx (context) cannot be nil! \n")
		return nil, nil
	}
	if clDevice == nil {
		log.Panicf("clDevice (device) cannot be nil! \n")
		return nil, nil
	}
	return clCtx.CreateCommandQueue(clDevice, 0)
}

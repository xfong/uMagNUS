package opencl

import (
	"fmt"
	"log"
	"math/rand"
	"time"
	"unsafe"

	cl "github.com/seeder-research/uMagNUS/cl"
	data "github.com/seeder-research/uMagNUS/data"
	oclRAND "github.com/seeder-research/uMagNUS/opencl/oclRAND"
)

type Prng_ interface {
	Init(uint64, *cl.CommandQueue, []*cl.Event) *cl.Event
	GenerateUniform(unsafe.Pointer, int, *cl.CommandQueue, []*cl.Event) *cl.Event
	GenerateNormal(unsafe.Pointer, int, *cl.CommandQueue, []*cl.Event) *cl.Event
	GetGroupSize() int
	GetGroupCount() int
	RecommendSize() int
}

type Generator struct {
	Name       string
	PRNG       Prng_
	r_seed     *uint32
	buf_size   int
	buf        *data.Slice
	supply     int
	sup_offset int
}

func NewGenerator(name string) *Generator {
	switch name {
	case "threefry":
		oclRAND.Init(ClCmdQueue, Synchronous, KernList)
		var prng_ptr Prng_
		prng_ptr = NewTHREEFRYRNGParams()
		return &Generator{Name: "threefry", PRNG: prng_ptr}
	case "xorwow":
		oclRAND.Init(ClCmdQueue, Synchronous, KernList)
		var prng_ptr Prng_
		prng_ptr = NewXORWOWRNGParams()
		return &Generator{Name: "xorwow", PRNG: prng_ptr}
	default:
		fmt.Println("RNG not implemented: ", name)
		return nil
	}
}

func (g *Generator) CreatePNG() {
	switch g.Name {
	case "threefry":
		oclRAND.Init(ClCmdQueue, Synchronous, KernList)
		var prng_ptr Prng_
		prng_ptr = NewTHREEFRYRNGParams()
		g.PRNG = prng_ptr
	case "xorwow":
		oclRAND.Init(ClCmdQueue, Synchronous, KernList)
		var prng_ptr Prng_
		prng_ptr = NewXORWOWRNGParams()
		g.PRNG = prng_ptr
	default:
		fmt.Println("RNG not implemented: ", g.Name)
	}
}

func (g *Generator) Init(seed *uint64) {
	var event *cl.Event

	g.buf_size = g.PRNG.RecommendSize()

	// sequence command according to queue
	evtWL := ClLastEvent

	// execute
	if seed == nil {
		event = g.PRNG.Init(initRNG(), ClCmdQueue, evtWL)
	} else {
		event = g.PRNG.Init(*seed, ClCmdQueue, evtWL)
	}

	// set event marker
	ClLastEvent = []*cl.Event{event}

	if g.buf == nil {
		g.buf = Buffer(1, [3]int{g.buf_size, 1, 1})
	} else {
		if g.buf.NComp() != 1 {
			log.Fatalln("Bad buffer for RNG \n")
		} else {
			bufferSize := g.buf.Size()
			if bufferSize[0] != g.buf_size {
				g.buf.Free()
				g.buf = Buffer(1, [3]int{g.buf_size, 1, 1})
			}
		}
	}
	g.supply = 0
}

func (g *Generator) Uniform(data unsafe.Pointer, d_size int) {
	var err error
	var event *cl.Event

	demand, demand_offset := d_size, 0

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to empty prior to uniform rng: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("wait for last event in uniform rng failed: %+v \n", err)
		}
	}

	for demand > 0 {
		// sequence command according to queue
		evtWL := ClLastEvent

		if g.supply <= 0 {
			// execute
			event = g.PRNG.GenerateUniform(g.buf.DevPtr(0), g.buf_size, ClCmdQueue, evtWL)

			bufferSize := g.buf.Size()
			if bufferSize[0] != g.buf_size {
				fmt.Printf("error in buffer size variables! \n")
			}
			g.supply = bufferSize[0]
			g.sup_offset = 0
		}
		if g.supply >= demand {
			// execute
			if event, err = ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(g.buf.DevPtr(0)), (*cl.MemObject)(data), SIZEOF_FLOAT32*g.sup_offset, SIZEOF_FLOAT32*demand_offset, SIZEOF_FLOAT32*demand, evtWL); err != nil {
				fmt.Printf("enqueuecopybuffer failed in copying uniform random numbers: %+v \n", err)
			}

			g.sup_offset += demand
			g.supply -= demand
			demand = 0
		} else {
			// execute
			if event, err = ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(g.buf.DevPtr(0)), (*cl.MemObject)(data), SIZEOF_FLOAT32*g.sup_offset, SIZEOF_FLOAT32*demand_offset, SIZEOF_FLOAT32*g.supply, evtWL); err != nil {
				fmt.Printf("enqueuecopybuffer in copying uniform random numbers failed: %+v \n", err)
			}

			demand -= g.supply
			demand_offset += g.supply
			g.supply = 0
		}

		// set event marker
		ClLastEvent = []*cl.Event{event}
	}

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in uniform rng: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("failed to wait for last event in uniform rng: %+v \n", err)
		}
	}
}

func (g *Generator) Normal(data unsafe.Pointer, d_size int) {
	var event *cl.Event
	var err error

	demand, demand_offset := d_size, 0

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to empty prior to normal rng: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("wait for last event in normal rng failed: %+v \n", err)
		}
	}

	for demand > 0 {
		// sequence command according to queue
		evtWL := ClLastEvent

		if g.supply <= 0 {
			// execute
			event = g.PRNG.GenerateNormal(g.buf.DevPtr(0), g.buf_size, ClCmdQueue, evtWL)

			bufferSize := g.buf.Size()
			if bufferSize[0] != g.buf_size {
				fmt.Printf("Error in buffer size variables! \n")
			}
			g.supply = bufferSize[0]
			g.sup_offset = 0
		}
		if g.supply >= demand {
			// execute
			if event, err = ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(g.buf.DevPtr(0)), (*cl.MemObject)(data), SIZEOF_FLOAT32*g.sup_offset, SIZEOF_FLOAT32*demand_offset, SIZEOF_FLOAT32*demand, evtWL); err != nil {
				fmt.Printf("enqueuecopybuffer failed in copying normal random numbers: %+v \n", err)
			}

			g.sup_offset += demand
			g.supply -= demand
			demand = 0
		} else {
			// execute
			if event, err = ClCmdQueue.EnqueueCopyBuffer((*cl.MemObject)(g.buf.DevPtr(0)), (*cl.MemObject)(data), SIZEOF_FLOAT32*g.sup_offset, SIZEOF_FLOAT32*demand_offset, SIZEOF_FLOAT32*g.supply, evtWL); err != nil {
				fmt.Printf("enqueuecopybuffer in copying normal random numbers failed: %+v \n", err)
			}

			demand -= g.supply
			demand_offset += g.supply
			g.supply = 0
		}

		// set event marker
		ClLastEvent = []*cl.Event{event}
	}

	if Synchronous { // debug
		if err = ClCmdQueue.Finish(); err != nil {
			fmt.Printf("failed to wait for queue to finish in normal rng: %+v \n", err)
		}
		if err = cl.WaitForEvents(ClLastEvent); err != nil {
			fmt.Printf("failed to wait for last event in normal rng: %+v \n", err)
		}
	}
}

func NewTHREEFRYRNGParams() *oclRAND.THREEFRY_status_array_ptr {
	tmp := oclRAND.NewTHREEFRYStatus()
	tmp.SetGroupCount(ClCUnits)
	tmp.SetGroupSize(ClPrefWGSz)
	tmp.SetStatusSize(ClCUnits * ClPrefWGSz)
	tmp.CreateStatusBuffer(ClCtx)

	return tmp
}

func NewXORWOWRNGParams() *oclRAND.XORWOW_status_array_ptr {
	tmp := oclRAND.NewXORWOWStatus()
	tmp.SetGroupCount(ClCUnits)
	tmp.SetGroupSize(ClPrefWGSz)
	tmp.SetStatusSize(ClCUnits * ClPrefWGSz)
	tmp.CreateStatusBuffer(ClCtx)

	return tmp
}

func initRNG() uint64 {
	rand.Seed(time.Now().UTC().UnixNano())
	return rand.Uint64()
}

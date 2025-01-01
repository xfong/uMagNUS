package opencl

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	util "github.com/seeder-research/uMagNUS/util"
)

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge or clampR at right edge.
func ShiftX(dst, src *data.Slice, shiftX int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)

	var err error
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftx: %+v \n", err)
		}
	}

	k_shiftx_async(dst.DevPtr(0), src.DevPtr(0),
		N[X], N[Y], N[Z],
		shiftX, clampL, clampR,
		cfg, ClCmdQueue, nil)

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftx end: %+v \n", err)
		}
	}
}

func ShiftY(dst, src *data.Slice, shiftY int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)

	var err error
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shifty: %+v \n", err)
		}
	}

	k_shifty_async(dst.DevPtr(0), src.DevPtr(0),
		N[X], N[Y], N[Z],
		shiftY, clampL, clampR,
		cfg, ClCmdQueue, nil)

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shifty end: %+v \n", err)
		}
	}
}

func ShiftZ(dst, src *data.Slice, shiftZ int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)

	var err error
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftz: %+v \n", err)
		}
	}

	k_shiftz_async(dst.DevPtr(0), src.DevPtr(0),
		N[X], N[Y], N[Z],
		shiftZ, clampL, clampR,
		cfg, ClCmdQueue, nil)

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftz end: %+v \n", err)
		}
	}
}

// Like Shift, but for bytes
func ShiftBytes(dst, src *Bytes, m *data.Mesh, shiftX int, clamp byte) {
	N := m.Size()
	cfg := make3DConf(N)

	var err error
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftbytes: %+v \n", err)
		}
	}

	k_shiftbytes_async(dst.Ptr, src.Ptr,
		N[X], N[Y], N[Z],
		shiftX, clamp,
		cfg, ClCmdQueue, nil)

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftbytes end: %+v \n", err)
		}
	}
}

func ShiftBytesY(dst, src *Bytes, m *data.Mesh, shiftY int, clamp byte) {
	N := m.Size()
	cfg := make3DConf(N)

	var err error
	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftbytesy: %+v \n", err)
		}
	}

	k_shiftbytesy_async(dst.Ptr, src.Ptr,
		N[X], N[Y], N[Z],
		shiftY, clamp,
		cfg, ClCmdQueue, nil)

	if Synchronous {
		if err = ClCmdQueue.Finish(); err != nil {
			log.Printf("failed to wait for queue to finish in shiftbytesy end: %+v \n", err)
		}
	}
}

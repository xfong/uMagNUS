package engine

// Calculation of magnetostatic field

import (
	"log"

	data "github.com/seeder-research/uMagNUS/data"
	mag "github.com/seeder-research/uMagNUS/mag"
	opencl "github.com/seeder-research/uMagNUS/opencl"
)

// Demag variables
var (
	Msat        = NewScalarParam("Msat", "A/m", "Saturation magnetization")
	M_full      = NewVectorField("m_full", "A/m", "Unnormalized magnetization", SetMFull)
	B_demag     = NewVectorField("B_demag", "T", "Magnetostatic field", SetDemagField)
	Edens_demag = NewScalarField("Edens_demag", "J/m3", "Magnetostatic energy density", AddEdens_demag)
	E_demag     = NewScalarValue("E_demag", "J", "Magnetostatic energy", GetDemagEnergy)

	EnableDemag  = true // enable/disable global demag field
	NoDemagSpins = NewScalarParam("NoDemagSpins", "", "Disable magnetostatic interaction per region (default=0, set to 1 to disable). "+
		"E.g.: NoDemagSpins.SetRegion(5, 1) disables the magnetostatic interaction in region 5.")
	conv_             *opencl.DemagConvolution // does the heavy lifting
	DemagAccuracy     = 6.0                    // Demag accuracy (divide cubes in at most N^3 points)
	EnableNewellDemag = false                  // enable/disable global demag field calculated using Newell formulation
	asymptotic_radius = 32                     // Radius (in number of cells) beyond which demag calculations fall back to far-field approximation
	zero_self_demag   = 0                      // Include/exclude self-demag
)

var AddEdens_demag = makeEdensAdder(&B_demag, -0.5)

func init() {

	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")
	DeclVar("EnableNewellDemag", &EnableNewellDemag, "Enables/disables demag using Newell's formulation (default=false)")
	DeclVar("DemagAccuracy", &DemagAccuracy, "Controls accuracy of demag kernel")
	registerEnergy(GetDemagEnergy, AddEdens_demag)
}

// Sets dst to the current demag field
func SetDemagField(dst *data.Slice) {
	if EnableDemag || EnableNewellDemag {
		msat := Msat.MSlice()
		defer msat.Recycle()
		if NoDemagSpins.isZero() {
			if EnableNewellDemag && EnableDemag {
				log.Fatal("Cannot enable both Newell and brute force demag! \n")
			} else {
				if EnableNewellDemag {
					// Normal demag (Newell formulation), everywhere
					newellDemagConv().Exec(dst, M.Buffer(), geometry.Gpu(), msat)
				}
				if EnableDemag {
					// Normal demag, everywhere
					demagConv().Exec(dst, M.Buffer(), geometry.Gpu(), msat)
				}
			}
		} else {
			setMaskedDemagField(dst, msat)
		}
	} else {
		opencl.Zero(dst) // will ADD other terms to it
	}
}

// Sets dst to the demag field, but cells where NoDemagSpins != 0 do not generate nor recieve field.
func setMaskedDemagField(dst *data.Slice, msat opencl.MSlice) {
	// No-demag spins: mask-out geometry with zeros where NoDemagSpins is set,
	// so these spins do not generate a field

	buf := opencl.Buffer(SCALAR, geometry.Gpu().Size()) // masked-out geometry
	defer opencl.Recycle(buf)

	// obtain a copy of the geometry mask, which we can overwrite
	geom, r := geometry.Slice()
	if r {
		defer opencl.Recycle(geom)
	}
	data.Copy(buf, geom)

	// mask-out
	opencl.ZeroMask(buf, NoDemagSpins.gpuLUT1(), regions.Gpu())

	// convolution with masked-out cells.
	if EnableDemag {
		// Normal demag, everywhere
		demagConv().Exec(dst, M.Buffer(), buf, msat)
	}
	if EnableNewellDemag {
		// Normal demag (Newell formulation), everywhere
		newellDemagConv().Exec(dst, M.Buffer(), buf, msat)
	}

	// After convolution, mask-out the field in the NoDemagSpins cells
	// so they don't feel the field generated by others.
	opencl.ZeroMask(dst, NoDemagSpins.gpuLUT1(), regions.Gpu())
}

// Sets dst to the full (unnormalized) magnetization in A/m
func SetMFull(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat.Slice()
	if rM {
		defer opencl.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		opencl.Mul(dst.Comp(c), M.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer opencl.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			opencl.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

// returns demag convolution, making sure it's initialized
func demagConv() *opencl.DemagConvolution {
	if conv_ == nil {
		SetBusy(true)
		defer SetBusy(false)
		kernel := mag.DemagKernel(Mesh().Size(), Mesh().PBC(), Mesh().CellSize(), DemagAccuracy, *Flag_cachedir)
		conv_ = opencl.NewDemag(Mesh().Size(), Mesh().PBC(), kernel, *Flag_selftest)
	}
	return conv_
}

// returns demag convolution, making sure it's initialized
func newellDemagConv() *opencl.DemagConvolution {
	if conv_ == nil {
		SetBusy(true)
		defer SetBusy(false)
		kernel := mag.NewellDemagKernel(Mesh().Size(), Mesh().PBC(), Mesh().CellSize(), asymptotic_radius, zero_self_demag, *Flag_cachedir)
		conv_ = opencl.NewDemag(Mesh().Size(), Mesh().PBC(), kernel, *Flag_selftest)
	}
	return conv_
}

// Returns the current demag energy in Joules.
func GetDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}

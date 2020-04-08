# CUDA Stereo Disparity
Creates "simplyfied" disparity maps from stereo images using AD-Census cost initialization and cost aggregation, for use as ground-truth labels for CCNN ([M. POGGI, S. MATTOCCIA: LEARNING FROM SCRATCH A CONFIDENCE MEASURE](http://www.bmva.org/bmvc/2016/papers/paper046/paper046.pdf)).

## To-Do
- [x] Census Transform
- [ ] AD-Census Cost
- [ ] Cost Aggregation
- [ ] Making it work with CCNN xD
- [ ] Half-decent CLI

## Building

### Dependencies
- CUDA Toolkit 10

### Build
- `git clone https://github.com/Check2016/cuda-stereo-disparity.git`
- `cd cuda-stereo-disparity`
- `make`

## Running it
`./disparity-cuda`

Currently only displays samples for debugging.

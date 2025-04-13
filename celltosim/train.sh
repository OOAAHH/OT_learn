python -m celltosim.examples.train_cellot \
    --data /Users/ooaahh/docs/99_archieve/cellot/celltosim/test_data.h5ad \
    --source source \
    --target target \
    --output cellot_output \
    --hidden-units 64 \
    --n-layers 4 \
    --batch-size 128 \
    --n-iters 500000

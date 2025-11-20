import sys
import h5py

if len(sys.argv) < 2:
    print('Usage: python inspect_weights.py /path/to/weights.hdf5')
    sys.exit(1)

path = sys.argv[1]
print('Inspecting', path)
with h5py.File(path, 'r') as f:
    # try common locations for top model dense kernel
    candidates = [
        'sequential/dense/kernel:0',
        'sequential/dense/kernel',
        'top_model/dense/kernel:0',
        'top_model/dense/kernel',
        'dense/kernel:0',
        'dense/kernel'
    ]
    found = False
    def print_group(name, obj):
        print(name, type(obj))
    # list top-level keys
    print('Top-level keys:', list(f.keys()))
    for key in candidates:
        if key in f:
            d = f[key]
            print('Found', key, 'shape=', d.shape)
            found = True
    # If not found, search for datasets named 'kernel' under groups
    if not found:
        print('Searching for datasets named kernel...')
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.name.endswith('kernel'):
                print('Dataset', obj.name, 'shape=', obj.shape)
        f.visititems(visitor)

    # Also print layer names if available
    if 'layer_names' in f.attrs:
        print('layer_names attr present, count=', len(f.attrs['layer_names']))

print('Done')

# Đọc checkpoint PyTorch mà không cần torch
import zipfile, pickle, numpy as np, sys, collections
z = zipfile.ZipFile(sys.argv[1]); pre = z.namelist()[0].split('/')[0]
DT = {'FloatStorage': np.float32, 'LongStorage': np.int64}
class St:
    def __init__(s, name): s.name = name
def rebuild(storage, off, size, stride, *a):
    dt, key = storage
    arr = np.frombuffer(z.read(f'{pre}/data/{key}'), dtype=dt)
    n = int(np.prod(size)) if size else 1
    return arr[off:off+n].reshape(size) if size else arr[off]
class U(pickle.Unpickler):
    def find_class(s, mod, name):
        if name == '_rebuild_tensor_v2': return rebuild
        if name in DT: return DT[name]
        if name == 'OrderedDict': return collections.OrderedDict
        return super().find_class(mod, name)
    def persistent_load(s, pid):  # ('storage', type, key, loc, n)
        return (pid[1], pid[2])
sd = U(__import__('io').BytesIO(z.read(f'{pre}/data.pkl'))).load()
np.save(sys.argv[2], sd, allow_pickle=True)
print(len(sd), 'tensor; tham số =', sum(v.size for k,v in sd.items() if 'running' not in k and 'num_batches' not in k))
for k, v in sd.items():
    if 'num_batches' in k: print(k, int(v))

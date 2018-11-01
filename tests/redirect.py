import ctypes
import os
import io

# from utils import stdout_redirector
#
# libc = ctypes.CDLL(None)
# f = io.BytesIO()
#
# with stdout_redirector(f):
#     print('foobar')
#     print(12)
#     libc.puts(b'this comes from C')
#     os.system('echo and this is from echo')
#
#
# print('Got stdout: "{0}"'.format(f.getvalue().decode('utf-8')))
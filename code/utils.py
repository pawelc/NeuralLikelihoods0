import sys
import tempfile
from contextlib import contextmanager
import ctypes
import os
import io
import tensorflow as tf
from IPython.display import clear_output, Image, display, HTML
import numpy as np

from flags import FLAGS

# libc = ctypes.CDLL(None)
# c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
# c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

# @contextmanager
# def stdout_redirector(stream):
#     # The original fd stdout points to. Usually 1 on POSIX systems.
#     original_stdout_fd = sys.stdout.fileno()
#
#     def _redirect_stdout(to_fd):
#         """Redirect stdout to the given file descriptor."""
#         # Flush the C-level buffer stdout
#         libc.fflush(c_stdout)
#         # Flush and close sys.stdout - also closes the file descriptor (fd)
#         sys.stdout.close()
#         # Make original_stdout_fd point to the same file as to_fd
#         os.dup2(to_fd, original_stdout_fd)
#         # Create a new sys.stdout that points to the redirected fd
#         sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))
#
#     # Save a copy of the original stdout fd in saved_stdout_fd
#     saved_stdout_fd = os.dup(original_stdout_fd)
#     try:
#         # Create a temporary file and redirect stdout to it
#         tfile = tempfile.TemporaryFile(mode='w+b')
#         _redirect_stdout(tfile.fileno())
#         # Yield to caller, then redirect stdout back to the saved fd
#         yield
#         _redirect_stdout(saved_stdout_fd)
#         # Copy contents of temporary file to the given stream
#         tfile.flush()
#         tfile.seek(0, io.SEEK_SET)
#         stream.write(tfile.read())
#     finally:
#         tfile.close()
#         os.close(saved_stdout_fd)

# @contextmanager
# def stderr_redirector(stream):
#     # The original fd stderr points to. Usually 1 on POSIX systems.
#     original_stderr_fd = sys.stderr.fileno()
#
#     def _redirect_stderr(to_fd):
#         """Redirect stderr to the given file descriptor."""
#         # Flush the C-level buffer stderr
#         libc.fflush(c_stderr)
#         # Flush and close sys.stderr - also closes the file descriptor (fd)
#         sys.stderr.close()
#         # Make original_stderr_fd point to the same file as to_fd
#         os.dup2(to_fd, original_stderr_fd)
#         # Create a new sys.stderr that points to the redirected fd
#         sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))
#
#     # Save a copy of the original stderr fd in saved_stderr_fd
#     saved_stderr_fd = os.dup(original_stderr_fd)
#     try:
#         # Create a temporary file and redirect stderr to it
#         tfile = tempfile.TemporaryFile(mode='w+b')
#         _redirect_stderr(tfile.fileno())
#         # Yield to caller, then redirect stderr back to the saved fd
#         yield
#         _redirect_stderr(saved_stderr_fd)
#         # Copy contents of temporary file to the given stream
#         tfile.flush()
#         tfile.seek(0, io.SEEK_SET)
#         stream.write(tfile.read())
#     finally:
#         tfile.close()
#         os.close(saved_stderr_fd)

def create_session_config():
    return tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=float(FLAGS.per_process_gpu_memory_fraction)))

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))